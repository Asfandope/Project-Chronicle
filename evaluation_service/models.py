"""
Database models for the evaluation service.

This module defines SQLAlchemy models for storing evaluation results,
accuracy metrics, and drift detection data in PostgreSQL.
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, JSON, Boolean, 
    ForeignKey, Index, UniqueConstraint, Text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()


class EvaluationRun(Base):
    """Represents a single evaluation run of the extraction system."""
    
    __tablename__ = "evaluation_runs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Evaluation metadata
    evaluation_type = Column(String(50), nullable=False)  # 'automatic', 'manual', 'batch'
    trigger_source = Column(String(100))  # 'drift_detection', 'manual_upload', 'scheduled'
    
    # System version info
    extractor_version = Column(String(50))
    model_version = Column(String(50))
    
    # Document information
    document_count = Column(Integer, nullable=False, default=0)
    total_articles = Column(Integer, nullable=False, default=0)
    
    # Overall metrics
    overall_weighted_accuracy = Column(Float, nullable=False, default=0.0)
    title_accuracy = Column(Float, nullable=False, default=0.0)
    body_text_accuracy = Column(Float, nullable=False, default=0.0)
    contributors_accuracy = Column(Float, nullable=False, default=0.0)
    media_links_accuracy = Column(Float, nullable=False, default=0.0)
    
    # Processing statistics
    processing_time_seconds = Column(Float)
    successful_extractions = Column(Integer, default=0)
    failed_extractions = Column(Integer, default=0)
    
    # Metadata
    evaluation_metadata = Column(JSON)
    notes = Column(Text)
    
    # Relationships
    document_evaluations = relationship("DocumentEvaluation", back_populates="evaluation_run")
    drift_detections = relationship("DriftDetection", back_populates="evaluation_run")
    
    __table_args__ = (
        Index('idx_evaluation_runs_created_at', 'created_at'),
        Index('idx_evaluation_runs_type', 'evaluation_type'),
    )


class DocumentEvaluation(Base):
    """Evaluation results for a single document."""
    
    __tablename__ = "document_evaluations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    evaluation_run_id = Column(UUID(as_uuid=True), ForeignKey("evaluation_runs.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Document identification
    document_id = Column(String(255), nullable=False)
    document_path = Column(String(500))
    ground_truth_path = Column(String(500))
    extracted_output_path = Column(String(500))
    
    # Document metadata
    brand_name = Column(String(100))
    page_count = Column(Integer)
    complexity_level = Column(String(50))
    edge_cases = Column(JSON)  # List of edge case types
    
    # Accuracy metrics
    weighted_overall_accuracy = Column(Float, nullable=False, default=0.0)
    title_accuracy = Column(Float, nullable=False, default=0.0)
    body_text_accuracy = Column(Float, nullable=False, default=0.0)
    contributors_accuracy = Column(Float, nullable=False, default=0.0)
    media_links_accuracy = Column(Float, nullable=False, default=0.0)
    
    # Field-level statistics
    title_correct = Column(Integer, default=0)
    title_total = Column(Integer, default=0)
    body_text_correct = Column(Integer, default=0)
    body_text_total = Column(Integer, default=0)
    contributors_correct = Column(Integer, default=0)
    contributors_total = Column(Integer, default=0)
    media_links_correct = Column(Integer, default=0)
    media_links_total = Column(Integer, default=0)
    
    # Processing info
    extraction_time_seconds = Column(Float)
    extraction_successful = Column(Boolean, default=True)
    extraction_error = Column(Text)
    
    # Detailed results
    detailed_results = Column(JSON)  # Full accuracy calculation results
    
    # Relationships
    evaluation_run = relationship("EvaluationRun", back_populates="document_evaluations")
    article_evaluations = relationship("ArticleEvaluation", back_populates="document_evaluation")
    
    __table_args__ = (
        Index('idx_document_evaluations_run_id', 'evaluation_run_id'),
        Index('idx_document_evaluations_document_id', 'document_id'),
        Index('idx_document_evaluations_brand', 'brand_name'),
        Index('idx_document_evaluations_created_at', 'created_at'),
    )


class ArticleEvaluation(Base):
    """Evaluation results for a single article within a document."""
    
    __tablename__ = "article_evaluations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_evaluation_id = Column(UUID(as_uuid=True), ForeignKey("document_evaluations.id"), nullable=False)
    
    # Article identification
    article_id = Column(String(255), nullable=False)
    article_title = Column(String(500))
    article_type = Column(String(50))
    page_range_start = Column(Integer)
    page_range_end = Column(Integer)
    
    # Article-level accuracy metrics
    weighted_accuracy = Column(Float, nullable=False, default=0.0)
    title_accuracy = Column(Float, nullable=False, default=0.0)
    body_text_accuracy = Column(Float, nullable=False, default=0.0)
    contributors_accuracy = Column(Float, nullable=False, default=0.0)
    media_links_accuracy = Column(Float, nullable=False, default=0.0)
    
    # Word Error Rate details
    body_text_wer = Column(Float)
    body_text_wer_threshold = Column(Float, default=0.001)
    body_text_meets_threshold = Column(Boolean)
    
    # Contributors details
    contributors_found = Column(Integer, default=0)
    contributors_expected = Column(Integer, default=0)
    contributors_matched = Column(Integer, default=0)
    
    # Media details
    media_elements_found = Column(Integer, default=0)
    media_elements_expected = Column(Integer, default=0)
    media_pairs_matched = Column(Integer, default=0)
    
    # Detailed results
    field_details = Column(JSON)  # Detailed breakdown of each field
    
    # Relationships
    document_evaluation = relationship("DocumentEvaluation", back_populates="article_evaluations")
    
    __table_args__ = (
        Index('idx_article_evaluations_doc_id', 'document_evaluation_id'),
        Index('idx_article_evaluations_article_id', 'article_id'),
    )


class DriftDetection(Base):
    """Records drift detection events and metrics."""
    
    __tablename__ = "drift_detections"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    evaluation_run_id = Column(UUID(as_uuid=True), ForeignKey("evaluation_runs.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Drift detection parameters
    window_size = Column(Integer, nullable=False, default=10)  # Rolling window size
    metric_type = Column(String(50), nullable=False)  # 'overall', 'title', 'body_text', etc.
    
    # Current metrics
    current_accuracy = Column(Float, nullable=False)
    baseline_accuracy = Column(Float, nullable=False)
    accuracy_drop = Column(Float, nullable=False)  # Difference from baseline
    
    # Threshold settings
    drift_threshold = Column(Float, nullable=False, default=0.05)  # 5% drop threshold
    alert_threshold = Column(Float, nullable=False, default=0.10)  # 10% drop threshold
    
    # Drift status
    drift_detected = Column(Boolean, nullable=False, default=False)
    alert_triggered = Column(Boolean, nullable=False, default=False)
    auto_tuning_triggered = Column(Boolean, nullable=False, default=False)
    
    # Statistical significance
    p_value = Column(Float)  # Statistical significance of drift
    confidence_interval = Column(JSON)  # [lower, upper] bounds
    
    # Window data
    window_data = Column(JSON)  # Historical accuracy values in window
    trend_direction = Column(String(20))  # 'declining', 'stable', 'improving'
    
    # Actions taken
    actions_triggered = Column(JSON)  # List of actions taken (alerts, auto-tuning, etc.)
    notification_sent = Column(Boolean, default=False)
    
    # Relationships
    evaluation_run = relationship("EvaluationRun", back_populates="drift_detections")
    
    __table_args__ = (
        Index('idx_drift_detections_run_id', 'evaluation_run_id'),
        Index('idx_drift_detections_created_at', 'created_at'),
        Index('idx_drift_detections_metric_type', 'metric_type'),
        Index('idx_drift_detections_drift_detected', 'drift_detected'),
    )


class AutoTuningEvent(Base):
    """Records auto-tuning events triggered by drift detection."""
    
    __tablename__ = "auto_tuning_events"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Trigger information
    trigger_drift_detection_id = Column(UUID(as_uuid=True), ForeignKey("drift_detections.id"))
    trigger_accuracy_drop = Column(Float, nullable=False)
    trigger_metric_type = Column(String(50), nullable=False)
    
    # Tuning configuration
    tuning_type = Column(String(50), nullable=False)  # 'model_retrain', 'parameter_adjust', etc.
    tuning_parameters = Column(JSON)  # Parameters to adjust
    
    # Execution status
    status = Column(String(50), nullable=False, default='pending')  # pending, running, completed, failed
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    
    # Results
    pre_tuning_accuracy = Column(Float)
    post_tuning_accuracy = Column(Float)
    improvement = Column(Float)
    
    # Execution details
    execution_log = Column(Text)
    error_message = Column(Text)
    
    # Configuration used
    model_config_before = Column(JSON)
    model_config_after = Column(JSON)
    
    __table_args__ = (
        Index('idx_auto_tuning_events_created_at', 'created_at'),
        Index('idx_auto_tuning_events_status', 'status'),
        Index('idx_auto_tuning_events_trigger_type', 'trigger_metric_type'),
    )


class SystemHealth(Base):
    """Tracks overall system health metrics over time."""
    
    __tablename__ = "system_health"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    recorded_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Time period
    period_start = Column(DateTime(timezone=True), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=False)
    
    # Volume metrics
    documents_processed = Column(Integer, default=0)
    articles_processed = Column(Integer, default=0)
    average_processing_time = Column(Float)
    
    # Accuracy metrics (rolling averages)
    average_overall_accuracy = Column(Float)
    average_title_accuracy = Column(Float)
    average_body_text_accuracy = Column(Float)
    average_contributors_accuracy = Column(Float)
    average_media_links_accuracy = Column(Float)
    
    # Quality metrics
    extraction_success_rate = Column(Float)
    low_confidence_extractions = Column(Integer, default=0)
    
    # Drift indicators
    drift_alerts_count = Column(Integer, default=0)
    auto_tuning_events_count = Column(Integer, default=0)
    
    # System performance
    average_response_time = Column(Float)
    error_rate = Column(Float)
    uptime_percentage = Column(Float)
    
    # Alerts and notifications
    critical_alerts_count = Column(Integer, default=0)
    warning_alerts_count = Column(Integer, default=0)
    
    __table_args__ = (
        Index('idx_system_health_recorded_at', 'recorded_at'),
        Index('idx_system_health_period', 'period_start', 'period_end'),
    )


class EvaluationMetrics(Base):
    """Stores aggregated evaluation metrics for reporting and analysis."""
    
    __tablename__ = "evaluation_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Aggregation parameters
    metric_name = Column(String(100), nullable=False)
    aggregation_level = Column(String(50), nullable=False)  # 'daily', 'weekly', 'monthly'
    aggregation_period = Column(DateTime(timezone=True), nullable=False)
    
    # Filters applied
    brand_filter = Column(String(100))
    complexity_filter = Column(String(50))
    document_type_filter = Column(String(50))
    
    # Statistical metrics
    sample_size = Column(Integer, nullable=False)
    mean_value = Column(Float, nullable=False)
    median_value = Column(Float)
    std_deviation = Column(Float)
    min_value = Column(Float)
    max_value = Column(Float)
    
    # Percentiles
    p25_value = Column(Float)
    p75_value = Column(Float)
    p90_value = Column(Float)
    p95_value = Column(Float)
    p99_value = Column(Float)
    
    # Trend information
    trend_slope = Column(Float)  # Linear trend slope
    trend_r_squared = Column(Float)  # Trend fit quality
    
    # Comparison to previous period
    previous_period_value = Column(Float)
    period_over_period_change = Column(Float)
    period_over_period_percent_change = Column(Float)
    
    __table_args__ = (
        Index('idx_evaluation_metrics_name_period', 'metric_name', 'aggregation_period'),
        Index('idx_evaluation_metrics_level', 'aggregation_level'),
        Index('idx_evaluation_metrics_created_at', 'created_at'),
        UniqueConstraint('metric_name', 'aggregation_level', 'aggregation_period', 
                        'brand_filter', 'complexity_filter', 'document_type_filter',
                        name='uq_evaluation_metrics_unique_combination')
    )


# Database utility functions
def create_tables(engine):
    """Create all tables in the database."""
    Base.metadata.create_all(engine)


def get_latest_evaluation_runs(session: Session, limit: int = 10) -> List[EvaluationRun]:
    """Get the most recent evaluation runs."""
    return session.query(EvaluationRun).order_by(EvaluationRun.created_at.desc()).limit(limit).all()


def get_drift_detections_in_window(
    session: Session, 
    metric_type: str, 
    window_size: int = 10
) -> List[DriftDetection]:
    """Get recent drift detections for a specific metric."""
    return (session.query(DriftDetection)
            .filter(DriftDetection.metric_type == metric_type)
            .order_by(DriftDetection.created_at.desc())
            .limit(window_size)
            .all())


def get_accuracy_history(
    session: Session,
    metric_type: str = 'overall',
    days: int = 30,
    brand_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get historical accuracy data for trend analysis."""
    query = session.query(DocumentEvaluation)
    
    if brand_filter:
        query = query.filter(DocumentEvaluation.brand_name == brand_filter)
    
    # Filter by date range
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    query = query.filter(DocumentEvaluation.created_at >= cutoff_date)
    
    query = query.order_by(DocumentEvaluation.created_at.asc())
    
    results = []
    for doc_eval in query.all():
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
            
        results.append({
            'timestamp': doc_eval.created_at,
            'accuracy': accuracy,
            'document_id': doc_eval.document_id,
            'brand_name': doc_eval.brand_name
        })
    
    return results