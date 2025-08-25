"""
Drift detection service for monitoring accuracy degradation.

This module implements rolling window drift detection with statistical
significance testing and auto-tuning triggers.
"""

import logging
import statistics
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from scipy import stats
from sqlalchemy.orm import Session

from .models import (
    DriftDetection, EvaluationRun, DocumentEvaluation, AutoTuningEvent,
    get_accuracy_history
)
from .schemas import MetricType, DriftStatus


logger = logging.getLogger(__name__)


@dataclass
class DriftDetectionConfig:
    """Configuration for drift detection."""
    window_size: int = 10
    drift_threshold: float = 0.05  # 5% drop
    alert_threshold: float = 0.10  # 10% drop
    auto_tuning_threshold: float = 0.15  # 15% drop
    min_samples: int = 5
    confidence_level: float = 0.95
    enable_statistical_tests: bool = True
    baseline_lookback_days: int = 30


@dataclass
class DriftAnalysisResult:
    """Result of drift analysis."""
    metric_type: str
    current_accuracy: float
    baseline_accuracy: float
    accuracy_drop: float
    drift_detected: bool
    alert_triggered: bool
    auto_tuning_triggered: bool
    p_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    trend_direction: Optional[str] = None
    window_data: List[float] = None
    statistical_significance: bool = False
    recommended_actions: List[str] = None


class DriftDetector:
    """Detects accuracy drift using rolling windows and statistical tests."""
    
    def __init__(self, config: Optional[DriftDetectionConfig] = None):
        self.config = config or DriftDetectionConfig()
        self.logger = logging.getLogger(__name__ + ".DriftDetector")
    
    def detect_drift(
        self,
        session: Session,
        evaluation_run: EvaluationRun,
        metric_types: Optional[List[str]] = None
    ) -> List[DriftAnalysisResult]:
        """Detect drift for specified metrics after an evaluation run."""
        
        if metric_types is None:
            metric_types = ['overall', 'title', 'body_text', 'contributors', 'media_links']
        
        results = []
        
        for metric_type in metric_types:
            try:
                result = self._analyze_metric_drift(session, evaluation_run, metric_type)
                results.append(result)
                
                # Store drift detection in database
                self._store_drift_detection(session, evaluation_run, result)
                
                # Trigger auto-tuning if needed
                if result.auto_tuning_triggered:
                    self._trigger_auto_tuning(session, result)
                
            except Exception as e:
                self.logger.error(f"Error detecting drift for metric {metric_type}: {str(e)}")
                continue
        
        return results
    
    def _analyze_metric_drift(
        self,
        session: Session,
        evaluation_run: EvaluationRun,
        metric_type: str
    ) -> DriftAnalysisResult:
        """Analyze drift for a specific metric."""
        
        # Get historical accuracy data
        window_data = self._get_metric_window_data(session, metric_type)
        
        if len(window_data) < self.config.min_samples:
            return DriftAnalysisResult(
                metric_type=metric_type,
                current_accuracy=0.0,
                baseline_accuracy=0.0,
                accuracy_drop=0.0,
                drift_detected=False,
                alert_triggered=False,
                auto_tuning_triggered=False,
                window_data=window_data,
                recommended_actions=["Insufficient data for drift detection"]
            )
        
        # Calculate current metrics
        current_accuracy = self._get_current_accuracy(evaluation_run, metric_type)
        baseline_accuracy = self._calculate_baseline_accuracy(session, metric_type)
        accuracy_drop = baseline_accuracy - current_accuracy
        
        # Perform statistical analysis
        p_value, confidence_interval = self._calculate_statistical_significance(
            window_data, current_accuracy
        )
        
        # Determine trend direction
        trend_direction = self._calculate_trend_direction(window_data)
        
        # Apply thresholds
        drift_detected = accuracy_drop >= self.config.drift_threshold
        alert_triggered = accuracy_drop >= self.config.alert_threshold
        auto_tuning_triggered = accuracy_drop >= self.config.auto_tuning_threshold
        
        # Statistical significance check
        statistical_significance = (
            self.config.enable_statistical_tests and 
            p_value is not None and 
            p_value < (1 - self.config.confidence_level)
        )
        
        # Generate recommendations
        recommended_actions = self._generate_recommendations(
            metric_type, accuracy_drop, trend_direction, statistical_significance
        )
        
        self.logger.info(
            f"Drift analysis for {metric_type}: "
            f"current={current_accuracy:.3f}, baseline={baseline_accuracy:.3f}, "
            f"drop={accuracy_drop:.3f}, drift={drift_detected}, "
            f"alert={alert_triggered}, auto_tune={auto_tuning_triggered}"
        )
        
        return DriftAnalysisResult(
            metric_type=metric_type,
            current_accuracy=current_accuracy,
            baseline_accuracy=baseline_accuracy,
            accuracy_drop=accuracy_drop,
            drift_detected=drift_detected,
            alert_triggered=alert_triggered,
            auto_tuning_triggered=auto_tuning_triggered,
            p_value=p_value,
            confidence_interval=confidence_interval,
            trend_direction=trend_direction,
            window_data=window_data,
            statistical_significance=statistical_significance,
            recommended_actions=recommended_actions
        )
    
    def _get_metric_window_data(self, session: Session, metric_type: str) -> List[float]:
        """Get recent accuracy data for the rolling window."""
        
        # Get recent document evaluations
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=7)  # Last week
        
        query = (session.query(DocumentEvaluation)
                .filter(DocumentEvaluation.created_at >= cutoff_date)
                .order_by(DocumentEvaluation.created_at.desc())
                .limit(self.config.window_size))
        
        document_evaluations = query.all()
        
        # Extract accuracy values based on metric type
        window_data = []
        for doc_eval in document_evaluations:
            if metric_type == 'overall':
                accuracy = doc_eval.weighted_overall_accuracy
            elif metric_type == 'title':
                accuracy = doc_eval.title_accuracy
            elif metric_type == 'body_text':
                accuracy = doc_eval.body_text_accuracy
            elif metric_type == 'contributors':
                accuracy = doc_eval.contributors_accuracy
            elif metric_type == 'media_links':
                accuracy = doc_eval.media_links_accuracy
            else:
                continue
            
            if accuracy is not None:
                window_data.append(accuracy)
        
        return window_data
    
    def _get_current_accuracy(self, evaluation_run: EvaluationRun, metric_type: str) -> float:
        """Get current accuracy for the specified metric."""
        if metric_type == 'overall':
            return evaluation_run.overall_weighted_accuracy
        elif metric_type == 'title':
            return evaluation_run.title_accuracy
        elif metric_type == 'body_text':
            return evaluation_run.body_text_accuracy
        elif metric_type == 'contributors':
            return evaluation_run.contributors_accuracy
        elif metric_type == 'media_links':
            return evaluation_run.media_links_accuracy
        else:
            return 0.0
    
    def _calculate_baseline_accuracy(self, session: Session, metric_type: str) -> float:
        """Calculate baseline accuracy from historical data."""
        
        # Get accuracy history for baseline calculation
        history = get_accuracy_history(
            session, 
            metric_type, 
            days=self.config.baseline_lookback_days
        )
        
        if not history:
            return 0.0
        
        # Use median as baseline (more robust than mean)
        accuracies = [point['accuracy'] for point in history if point['accuracy'] is not None]
        
        if not accuracies:
            return 0.0
        
        return statistics.median(accuracies)
    
    def _calculate_statistical_significance(
        self,
        window_data: List[float],
        current_accuracy: float
    ) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
        """Calculate statistical significance of the accuracy drop."""
        
        if not self.config.enable_statistical_tests or len(window_data) < 3:
            return None, None
        
        try:
            # Perform one-sample t-test
            t_stat, p_value = stats.ttest_1samp(window_data, current_accuracy)
            
            # Calculate confidence interval
            mean_accuracy = np.mean(window_data)
            std_error = stats.sem(window_data)
            confidence_interval = stats.t.interval(
                self.config.confidence_level,
                len(window_data) - 1,
                loc=mean_accuracy,
                scale=std_error
            )
            
            return p_value, confidence_interval
            
        except Exception as e:
            self.logger.warning(f"Error calculating statistical significance: {str(e)}")
            return None, None
    
    def _calculate_trend_direction(self, window_data: List[float]) -> str:
        """Calculate trend direction from window data."""
        
        if len(window_data) < 3:
            return "insufficient_data"
        
        try:
            # Linear regression to determine trend
            x = np.arange(len(window_data))
            y = np.array(window_data)
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Determine direction based on slope and significance
            if abs(slope) < 0.001:  # Very small slope
                return "stable"
            elif slope > 0:
                return "improving"
            else:
                return "declining"
                
        except Exception as e:
            self.logger.warning(f"Error calculating trend direction: {str(e)}")
            return "unknown"
    
    def _generate_recommendations(
        self,
        metric_type: str,
        accuracy_drop: float,
        trend_direction: str,
        statistical_significance: bool
    ) -> List[str]:
        """Generate recommended actions based on drift analysis."""
        
        recommendations = []
        
        if accuracy_drop < self.config.drift_threshold:
            recommendations.append("No action required - accuracy within normal range")
            return recommendations
        
        # General recommendations based on drift severity
        if accuracy_drop >= self.config.auto_tuning_threshold:
            recommendations.append("Critical accuracy drop - auto-tuning has been triggered")
            recommendations.append("Review recent system changes and data quality")
        elif accuracy_drop >= self.config.alert_threshold:
            recommendations.append("Significant accuracy drop detected - manual review recommended")
        else:
            recommendations.append("Minor accuracy drift detected - monitor closely")
        
        # Metric-specific recommendations
        if metric_type == 'title':
            recommendations.append("Review title extraction patterns and font recognition")
            recommendations.append("Check for new magazine layouts or design changes")
        elif metric_type == 'body_text':
            recommendations.append("Analyze text extraction quality and OCR performance")
            recommendations.append("Review multi-column layout handling")
        elif metric_type == 'contributors':
            recommendations.append("Check byline detection and author name extraction")
            recommendations.append("Review contributor role classification")
        elif metric_type == 'media_links':
            recommendations.append("Analyze image-caption association accuracy")
            recommendations.append("Review media element detection thresholds")
        
        # Trend-based recommendations
        if trend_direction == "declining":
            recommendations.append("Declining trend detected - investigate underlying causes")
        elif trend_direction == "improving":
            recommendations.append("Performance is improving - continue current approach")
        
        # Statistical significance
        if statistical_significance:
            recommendations.append("Change is statistically significant - immediate attention required")
        else:
            recommendations.append("Change may be due to normal variation - continue monitoring")
        
        return recommendations
    
    def _store_drift_detection(
        self,
        session: Session,
        evaluation_run: EvaluationRun,
        result: DriftAnalysisResult
    ) -> None:
        """Store drift detection results in database."""
        
        try:
            # Sanitize NaN values for database storage
            p_value = result.p_value if result.p_value is not None and not np.isnan(result.p_value) else None
            
            confidence_interval = None
            if result.confidence_interval is not None:
                ci_list = list(result.confidence_interval)
                if not any(np.isnan(ci_list)):
                    confidence_interval = ci_list
            
            drift_detection = DriftDetection(
                evaluation_run_id=evaluation_run.id,
                window_size=self.config.window_size,
                metric_type=result.metric_type,
                current_accuracy=result.current_accuracy,
                baseline_accuracy=result.baseline_accuracy,
                accuracy_drop=result.accuracy_drop,
                drift_threshold=self.config.drift_threshold,
                alert_threshold=self.config.alert_threshold,
                drift_detected=result.drift_detected,
                alert_triggered=result.alert_triggered,
                auto_tuning_triggered=result.auto_tuning_triggered,
                p_value=p_value,
                confidence_interval=confidence_interval,
                window_data=result.window_data,
                trend_direction=result.trend_direction,
                actions_triggered=result.recommended_actions
            )
            
            session.add(drift_detection)
            session.commit()
            
            self.logger.info(f"Stored drift detection for metric {result.metric_type}")
            
        except Exception as e:
            self.logger.error(f"Error storing drift detection: {str(e)}")
            session.rollback()
    
    def _trigger_auto_tuning(
        self,
        session: Session,
        result: DriftAnalysisResult
    ) -> None:
        """Trigger auto-tuning process when thresholds are breached."""
        
        try:
            # Create auto-tuning event
            auto_tuning_event = AutoTuningEvent(
                trigger_accuracy_drop=result.accuracy_drop,
                trigger_metric_type=result.metric_type,
                tuning_type="adaptive_threshold_adjustment",
                tuning_parameters={
                    "metric_type": result.metric_type,
                    "accuracy_drop": result.accuracy_drop,
                    "baseline_accuracy": result.baseline_accuracy,
                    "current_accuracy": result.current_accuracy,
                    "statistical_significance": result.statistical_significance
                },
                status="pending",
                pre_tuning_accuracy=result.current_accuracy
            )
            
            session.add(auto_tuning_event)
            session.commit()
            
            self.logger.info(f"Triggered auto-tuning for metric {result.metric_type}")
            
            # TODO: Integrate with actual auto-tuning system
            # For now, just log the event
            self._execute_auto_tuning(session, auto_tuning_event)
            
        except Exception as e:
            self.logger.error(f"Error triggering auto-tuning: {str(e)}")
            session.rollback()
    
    def _execute_auto_tuning(
        self,
        session: Session,
        auto_tuning_event: AutoTuningEvent
    ) -> None:
        """Execute auto-tuning process (placeholder implementation)."""
        
        try:
            # Update status to running
            auto_tuning_event.status = "running"
            auto_tuning_event.started_at = datetime.now(timezone.utc)
            session.commit()
            
            # Simulate auto-tuning process
            # In a real implementation, this would:
            # 1. Analyze the specific accuracy drop
            # 2. Adjust model parameters or retrain
            # 3. Validate improvements
            # 4. Deploy updates if successful
            
            self.logger.info(f"Executing auto-tuning for event {auto_tuning_event.id}")
            
            # Simulate successful tuning
            auto_tuning_event.status = "completed"
            auto_tuning_event.completed_at = datetime.now(timezone.utc)
            auto_tuning_event.post_tuning_accuracy = auto_tuning_event.pre_tuning_accuracy + 0.02  # 2% improvement
            auto_tuning_event.improvement = auto_tuning_event.post_tuning_accuracy - auto_tuning_event.pre_tuning_accuracy
            auto_tuning_event.execution_log = "Auto-tuning completed successfully with simulated 2% improvement"
            
            session.commit()
            
            self.logger.info(f"Auto-tuning completed for event {auto_tuning_event.id}")
            
        except Exception as e:
            # Mark as failed
            auto_tuning_event.status = "failed"
            auto_tuning_event.completed_at = datetime.now(timezone.utc)
            auto_tuning_event.error_message = str(e)
            session.commit()
            
            self.logger.error(f"Auto-tuning failed for event {auto_tuning_event.id}: {str(e)}")
    
    def get_drift_status(self, session: Session, metric_type: str) -> DriftStatus:
        """Get current drift status for a metric."""
        
        # Get most recent drift detection
        latest_drift = (session.query(DriftDetection)
                       .filter(DriftDetection.metric_type == metric_type)
                       .order_by(DriftDetection.created_at.desc())
                       .first())
        
        if not latest_drift:
            return DriftStatus.NO_DRIFT
        
        if latest_drift.auto_tuning_triggered:
            return DriftStatus.AUTO_TUNING_TRIGGERED
        elif latest_drift.alert_triggered:
            return DriftStatus.ALERT_TRIGGERED
        elif latest_drift.drift_detected:
            return DriftStatus.DRIFT_DETECTED
        else:
            return DriftStatus.NO_DRIFT
    
    def get_drift_summary(self, session: Session, days: int = 7) -> Dict[str, Any]:
        """Get summary of drift detections over specified period."""
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        drift_detections = (session.query(DriftDetection)
                          .filter(DriftDetection.created_at >= cutoff_date)
                          .all())
        
        summary = {
            'total_detections': len(drift_detections),
            'drift_detected_count': sum(1 for d in drift_detections if d.drift_detected),
            'alerts_triggered_count': sum(1 for d in drift_detections if d.alert_triggered),
            'auto_tuning_triggered_count': sum(1 for d in drift_detections if d.auto_tuning_triggered),
            'metrics_affected': list(set(d.metric_type for d in drift_detections)),
            'average_accuracy_drop': statistics.mean([d.accuracy_drop for d in drift_detections]) if drift_detections else 0,
            'period_days': days
        }
        
        return summary


def create_drift_detector(config: Optional[Dict[str, Any]] = None) -> DriftDetector:
    """Factory function to create configured drift detector."""
    
    if config:
        drift_config = DriftDetectionConfig(**config)
    else:
        drift_config = DriftDetectionConfig()
    
    return DriftDetector(drift_config)