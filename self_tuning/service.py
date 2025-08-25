"""
Self-tuning system service implementation.

This module implements the core self-tuning logic from PRD section 7.3:
1. Identify failure patterns from quarantined issues
2. Generate targeted synthetic examples
3. Grid search over parameter space
4. Validate on holdout set
5. Deploy if improvement, rollback if not
"""

import logging
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sqlalchemy.orm import Session
from sqlalchemy import and_, func, desc

from .models import (
    TuningRun, FailurePattern, SyntheticDataset, ParameterExperiment,
    ValidationResult, TuningRunRateLimit,
    TuningStatus, OptimizationStrategy, FailurePatternType, 
    ExperimentStatus, ValidationStatus
)
from evaluation_service.models import EvaluationRun, DocumentEvaluation
from parameter_management.service import ParameterService, ParameterUpdateRequest
from parameter_management.models import ParameterScope
from synthetic_data.generator import SyntheticDataGenerator
from synthetic_data.types import BrandConfiguration, GenerationConfig


logger = logging.getLogger(__name__)


@dataclass
class FailureAnalysis:
    """Analysis of failure patterns in quarantined issues."""
    failure_type: FailurePatternType
    affected_parameters: List[str]
    severity_score: float
    frequency: int
    examples: List[Dict[str, Any]]
    suggested_adjustments: Dict[str, Any]


@dataclass
class GridSearchConfig:
    """Configuration for grid search optimization."""
    parameters: Dict[str, List[Any]]
    max_experiments: int = 50
    early_stopping_threshold: float = 0.01
    validation_split: float = 0.2


@dataclass
class TuningResult:
    """Result of a complete tuning run."""
    tuning_run_id: str
    status: TuningStatus
    improvement_achieved: bool
    baseline_accuracy: float
    final_accuracy: float
    accuracy_improvement: float
    deployed_parameters: Optional[Dict[str, Any]]
    rollback_reason: Optional[str]


class SelfTuningService:
    """Core service for automated parameter optimization."""
    
    def __init__(self):
        self.parameter_service = ParameterService()
        # Create a default generation config for synthetic data generation
        from synthetic_data.types import GenerationConfig
        from pathlib import Path
        default_config = GenerationConfig(
            output_directory=Path("/tmp/synthetic_data"),
            documents_per_brand=10,
            generate_pdfs=True,
            generate_ground_truth=True
        )
        self.synthetic_generator = SyntheticDataGenerator(default_config)
        self.logger = logging.getLogger(__name__ + ".SelfTuningService")
        
        # Configuration
        self.min_failure_frequency = 3
        self.min_improvement_threshold = 0.02  # 2% minimum improvement
        self.confidence_level = 0.95
        self.max_tuning_time_hours = 12
    
    def start_tuning_run(
        self,
        session: Session,
        brand_name: str,
        triggered_by: str = "drift_detection",
        strategy: OptimizationStrategy = OptimizationStrategy.GRID_SEARCH,
        force: bool = False
    ) -> TuningRun:
        """
        Start a new tuning run for a brand.
        
        Args:
            session: Database session
            brand_name: Target brand name
            triggered_by: What triggered this tuning run
            strategy: Optimization strategy to use
            force: Skip rate limiting check
            
        Returns:
            Created TuningRun instance
            
        Raises:
            ValueError: If rate limit exceeded or insufficient data
        """
        self.logger.info(f"Starting tuning run for brand: {brand_name}")
        
        # Check rate limiting unless forced
        if not force and not self._check_rate_limit(session, brand_name):
            raise ValueError(f"Rate limit exceeded: max one tuning run per brand per day")
        
        # Check if we have sufficient quarantined data
        quarantined_count = self._get_quarantined_issues_count(session, brand_name)
        if quarantined_count < 10:
            raise ValueError(f"Insufficient quarantined data: {quarantined_count} issues (need ≥10)")
        
        # Create tuning run
        tuning_run = TuningRun(
            brand_name=brand_name,
            status=TuningStatus.ANALYZING_FAILURES,
            triggered_by=triggered_by,
            optimization_strategy=strategy,
            created_at=datetime.now(timezone.utc)
        )
        
        session.add(tuning_run)
        session.flush()
        
        # Record rate limit entry
        rate_limit = TuningRunRateLimit(
            brand_name=brand_name,
            tuning_run_id=tuning_run.id,
            created_at=datetime.now(timezone.utc)
        )
        session.add(rate_limit)
        session.commit()
        
        self.logger.info(f"Created tuning run {tuning_run.id} for brand {brand_name}")
        return tuning_run
    
    def identify_failure_patterns(
        self,
        session: Session,
        tuning_run: TuningRun
    ) -> List[FailureAnalysis]:
        """
        Analyze quarantined issues to identify failure patterns.
        
        Args:
            session: Database session
            tuning_run: Current tuning run
            
        Returns:
            List of identified failure patterns
        """
        self.logger.info(f"Analyzing failure patterns for tuning run {tuning_run.id}")
        
        # Get low-accuracy evaluation results for this brand (below 80% accuracy)
        quarantined_results = session.query(DocumentEvaluation).join(
            EvaluationRun
        ).filter(
            and_(
                EvaluationRun.brand_name == tuning_run.brand_name,
                DocumentEvaluation.weighted_overall_accuracy < 0.8,
                EvaluationRun.created_at >= datetime.now(timezone.utc) - timedelta(days=30)
            )
        ).all()
        
        failure_analyses = []
        
        # Analyze title extraction failures
        title_failures = [r for r in quarantined_results if r.title_accuracy < 0.5]
        if len(title_failures) >= self.min_failure_frequency:
            analysis = self._analyze_title_failures(title_failures)
            failure_analyses.append(analysis)
            self._create_failure_pattern(session, tuning_run, analysis)
        
        # Analyze body text extraction failures
        body_failures = [r for r in quarantined_results if r.body_text_accuracy < 0.8]
        if len(body_failures) >= self.min_failure_frequency:
            analysis = self._analyze_body_text_failures(body_failures)
            failure_analyses.append(analysis)
            self._create_failure_pattern(session, tuning_run, analysis)
        
        # Analyze contributor extraction failures
        contributor_failures = [r for r in quarantined_results if r.contributors_accuracy < 0.7]
        if len(contributor_failures) >= self.min_failure_frequency:
            analysis = self._analyze_contributor_failures(contributor_failures)
            failure_analyses.append(analysis)
            self._create_failure_pattern(session, tuning_run, analysis)
        
        # Analyze media link failures
        media_failures = [r for r in quarantined_results if r.media_links_accuracy < 0.6]
        if len(media_failures) >= self.min_failure_frequency:
            analysis = self._analyze_media_failures(media_failures)
            failure_analyses.append(analysis)
            self._create_failure_pattern(session, tuning_run, analysis)
        
        session.commit()
        
        self.logger.info(f"Identified {len(failure_analyses)} failure patterns")
        return failure_analyses
    
    def generate_targeted_synthetic_data(
        self,
        session: Session,
        tuning_run: TuningRun,
        failure_patterns: List[FailurePattern],
        dataset_size: int = 100
    ) -> SyntheticDataset:
        """
        Generate synthetic test data targeting specific failure patterns.
        
        Args:
            session: Database session
            tuning_run: Current tuning run
            failure_patterns: Identified failure patterns to target
            dataset_size: Number of synthetic examples to generate
            
        Returns:
            Created synthetic dataset
        """
        self.logger.info(f"Generating targeted synthetic data for tuning run {tuning_run.id}")
        
        # Create dataset record
        dataset = SyntheticDataset(
            tuning_run_id=tuning_run.id,
            dataset_size=dataset_size,
            generation_config=self._build_generation_config(failure_patterns),
            created_at=datetime.now(timezone.utc)
        )
        
        session.add(dataset)
        session.flush()
        
        # Generate synthetic examples targeting each failure pattern
        synthetic_examples = []
        examples_per_pattern = dataset_size // len(failure_patterns)
        
        for pattern in failure_patterns:
            pattern_examples = self._generate_examples_for_pattern(
                pattern, examples_per_pattern, tuning_run.brand_name
            )
            synthetic_examples.extend(pattern_examples)
        
        # Store examples
        dataset.file_paths = [ex['pdf_path'] for ex in synthetic_examples]
        dataset.ground_truth_paths = [ex['xml_path'] for ex in synthetic_examples]
        dataset.generation_metadata = {
            'patterns_targeted': [p.failure_type.value for p in failure_patterns],
            'examples_per_pattern': examples_per_pattern,
            'total_examples': len(synthetic_examples)
        }
        
        session.commit()
        
        self.logger.info(f"Generated {len(synthetic_examples)} targeted synthetic examples")
        return dataset
    
    def optimize_parameters(
        self,
        session: Session,
        tuning_run: TuningRun,
        failure_patterns: List[FailurePattern],
        synthetic_dataset: SyntheticDataset
    ) -> List[ParameterExperiment]:
        """
        Perform grid search optimization over parameter space.
        
        Args:
            session: Database session
            tuning_run: Current tuning run
            failure_patterns: Identified failure patterns
            synthetic_dataset: Generated synthetic dataset
            
        Returns:
            List of parameter experiments conducted
        """
        self.logger.info(f"Starting parameter optimization for tuning run {tuning_run.id}")
        
        # Build grid search configuration
        grid_config = self._build_grid_search_config(failure_patterns)
        
        # Get baseline accuracy
        baseline_accuracy = self._calculate_baseline_accuracy(
            session, tuning_run.brand_name, synthetic_dataset
        )
        
        # Conduct grid search experiments
        experiments = []
        experiment_count = 0
        
        for param_combination in self._generate_parameter_combinations(grid_config):
            if experiment_count >= grid_config.max_experiments:
                break
            
            # Create experiment
            experiment = ParameterExperiment(
                tuning_run_id=tuning_run.id,
                parameter_values=param_combination,
                status=ExperimentStatus.RUNNING,
                created_at=datetime.now(timezone.utc)
            )
            
            session.add(experiment)
            session.flush()
            
            try:
                # Apply parameters temporarily
                self._apply_experimental_parameters(session, param_combination, tuning_run.brand_name)
                
                # Evaluate on synthetic dataset
                accuracy_score = self._evaluate_parameter_combination(
                    session, synthetic_dataset, tuning_run.brand_name
                )
                
                # Update experiment results
                experiment.accuracy_score = accuracy_score
                experiment.improvement_over_baseline = accuracy_score - baseline_accuracy
                experiment.status = ExperimentStatus.COMPLETED
                experiment.completed_at = datetime.now(timezone.utc)
                
                experiments.append(experiment)
                experiment_count += 1
                
                self.logger.debug(f"Experiment {experiment.id}: accuracy={accuracy_score:.4f}, improvement={experiment.improvement_over_baseline:.4f}")
                
                # Early stopping if we found a good improvement
                if experiment.improvement_over_baseline > grid_config.early_stopping_threshold:
                    self.logger.info(f"Early stopping triggered: improvement={experiment.improvement_over_baseline:.4f}")
                    break
                
            except Exception as e:
                experiment.status = ExperimentStatus.FAILED
                experiment.error_message = str(e)
                self.logger.error(f"Experiment {experiment.id} failed: {e}")
            
            finally:
                # Rollback experimental parameters
                self._rollback_experimental_parameters(session, tuning_run.brand_name)
        
        session.commit()
        
        self.logger.info(f"Completed {len(experiments)} parameter experiments")
        return experiments
    
    def validate_best_parameters(
        self,
        session: Session,
        tuning_run: TuningRun,
        experiments: List[ParameterExperiment]
    ) -> ValidationResult:
        """
        Validate the best parameter combination on holdout set.
        
        Args:
            session: Database session
            tuning_run: Current tuning run
            experiments: Completed parameter experiments
            
        Returns:
            Validation result
        """
        self.logger.info(f"Validating best parameters for tuning run {tuning_run.id}")
        
        # Find best experiment
        successful_experiments = [e for e in experiments if e.status == ExperimentStatus.COMPLETED]
        if not successful_experiments:
            raise ValueError("No successful experiments to validate")
        
        best_experiment = max(successful_experiments, key=lambda e: e.accuracy_score)
        
        # Create validation result
        validation = ValidationResult(
            tuning_run_id=tuning_run.id,
            parameter_experiment_id=best_experiment.id,
            status=ValidationStatus.RUNNING,
            created_at=datetime.now(timezone.utc)
        )
        
        session.add(validation)
        session.flush()
        
        try:
            # Apply best parameters
            self._apply_experimental_parameters(
                session, best_experiment.parameter_values, tuning_run.brand_name
            )
            
            # Get holdout validation set (recent real evaluation data)
            holdout_results = self._get_holdout_validation_set(session, tuning_run.brand_name)
            
            if len(holdout_results) < 10:
                raise ValueError(f"Insufficient holdout data: {len(holdout_results)} samples")
            
            # Calculate validation metrics
            holdout_accuracy = self._calculate_holdout_accuracy(holdout_results)
            baseline_accuracy = self._get_baseline_accuracy_for_validation(session, tuning_run.brand_name)
            
            # Statistical significance testing
            confidence_interval, p_value = self._calculate_statistical_significance(
                holdout_accuracy, baseline_accuracy, len(holdout_results)
            )
            
            # Update validation result
            validation.holdout_accuracy = holdout_accuracy
            validation.baseline_accuracy = baseline_accuracy
            validation.accuracy_improvement = holdout_accuracy - baseline_accuracy
            validation.confidence_interval_lower = confidence_interval[0]
            validation.confidence_interval_upper = confidence_interval[1]
            validation.p_value = p_value
            validation.is_statistically_significant = p_value < (1 - self.confidence_level)
            validation.meets_improvement_threshold = (
                validation.accuracy_improvement > self.min_improvement_threshold
            )
            validation.status = ValidationStatus.COMPLETED
            validation.completed_at = datetime.now(timezone.utc)
            
        except Exception as e:
            validation.status = ValidationStatus.FAILED
            validation.error_message = str(e)
            self.logger.error(f"Validation failed: {e}")
            raise
        
        finally:
            # Rollback experimental parameters
            self._rollback_experimental_parameters(session, tuning_run.brand_name)
        
        session.commit()
        
        self.logger.info(
            f"Validation completed: accuracy={validation.holdout_accuracy:.4f}, "
            f"improvement={validation.accuracy_improvement:.4f}, "
            f"significant={validation.is_statistically_significant}"
        )
        
        return validation
    
    def deploy_or_rollback(
        self,
        session: Session,
        tuning_run: TuningRun,
        validation_result: ValidationResult
    ) -> TuningResult:
        """
        Deploy improved parameters or rollback if validation failed.
        
        Args:
            session: Database session
            tuning_run: Current tuning run
            validation_result: Validation results
            
        Returns:
            Final tuning result
        """
        self.logger.info(f"Making deployment decision for tuning run {tuning_run.id}")
        
        should_deploy = (
            validation_result.status == ValidationStatus.COMPLETED and
            validation_result.meets_improvement_threshold and
            validation_result.is_statistically_significant
        )
        
        if should_deploy:
            # Deploy the improved parameters
            best_experiment = session.query(ParameterExperiment).filter(
                ParameterExperiment.id == validation_result.parameter_experiment_id
            ).first()
            
            self._deploy_parameters(session, best_experiment.parameter_values, tuning_run.brand_name)
            
            tuning_run.status = TuningStatus.DEPLOYED
            tuning_run.deployed_parameters = best_experiment.parameter_values
            tuning_run.baseline_accuracy = validation_result.baseline_accuracy
            tuning_run.final_accuracy = validation_result.holdout_accuracy
            tuning_run.accuracy_improvement = validation_result.accuracy_improvement
            
            result = TuningResult(
                tuning_run_id=str(tuning_run.id),
                status=TuningStatus.DEPLOYED,
                improvement_achieved=True,
                baseline_accuracy=validation_result.baseline_accuracy,
                final_accuracy=validation_result.holdout_accuracy,
                accuracy_improvement=validation_result.accuracy_improvement,
                deployed_parameters=best_experiment.parameter_values,
                rollback_reason=None
            )
            
            self.logger.info(f"Parameters deployed successfully: improvement={validation_result.accuracy_improvement:.4f}")
            
        else:
            # Rollback - keep existing parameters
            rollback_reasons = []
            
            if validation_result.status == ValidationStatus.FAILED:
                rollback_reasons.append(f"Validation failed: {validation_result.error_message}")
            if not validation_result.meets_improvement_threshold:
                rollback_reasons.append(f"Insufficient improvement: {validation_result.accuracy_improvement:.4f} < {self.min_improvement_threshold}")
            if not validation_result.is_statistically_significant:
                rollback_reasons.append(f"Not statistically significant: p={validation_result.p_value:.4f}")
            
            rollback_reason = "; ".join(rollback_reasons)
            
            tuning_run.status = TuningStatus.ROLLED_BACK
            tuning_run.rollback_reason = rollback_reason
            tuning_run.baseline_accuracy = validation_result.baseline_accuracy
            tuning_run.final_accuracy = validation_result.baseline_accuracy  # No change
            tuning_run.accuracy_improvement = 0.0
            
            result = TuningResult(
                tuning_run_id=str(tuning_run.id),
                status=TuningStatus.ROLLED_BACK,
                improvement_achieved=False,
                baseline_accuracy=validation_result.baseline_accuracy,
                final_accuracy=validation_result.baseline_accuracy,
                accuracy_improvement=0.0,
                deployed_parameters=None,
                rollback_reason=rollback_reason
            )
            
            self.logger.info(f"Parameters rolled back: {rollback_reason}")
        
        tuning_run.completed_at = datetime.now(timezone.utc)
        session.commit()
        
        return result
    
    def run_complete_tuning_cycle(
        self,
        session: Session,
        brand_name: str,
        triggered_by: str = "manual",
        force: bool = False
    ) -> TuningResult:
        """
        Execute a complete tuning cycle from start to finish.
        
        Args:
            session: Database session
            brand_name: Target brand name
            triggered_by: What triggered this tuning run
            force: Skip rate limiting check
            
        Returns:
            Final tuning result
        """
        self.logger.info(f"Starting complete tuning cycle for brand: {brand_name}")
        
        try:
            # Step 1: Start tuning run
            tuning_run = self.start_tuning_run(session, brand_name, triggered_by, force=force)
            
            # Step 2: Identify failure patterns
            failure_analyses = self.identify_failure_patterns(session, tuning_run)
            failure_patterns = session.query(FailurePattern).filter(
                FailurePattern.tuning_run_id == tuning_run.id
            ).all()
            
            if not failure_patterns:
                tuning_run.status = TuningStatus.NO_PATTERNS_FOUND
                tuning_run.completed_at = datetime.now(timezone.utc)
                session.commit()
                
                return TuningResult(
                    tuning_run_id=str(tuning_run.id),
                    status=TuningStatus.NO_PATTERNS_FOUND,
                    improvement_achieved=False,
                    baseline_accuracy=0.0,
                    final_accuracy=0.0,
                    accuracy_improvement=0.0,
                    deployed_parameters=None,
                    rollback_reason="No failure patterns found"
                )
            
            # Step 3: Generate synthetic data
            tuning_run.status = TuningStatus.GENERATING_DATA
            session.commit()
            
            synthetic_dataset = self.generate_targeted_synthetic_data(
                session, tuning_run, failure_patterns
            )
            
            # Step 4: Optimize parameters
            tuning_run.status = TuningStatus.OPTIMIZING_PARAMETERS
            session.commit()
            
            experiments = self.optimize_parameters(
                session, tuning_run, failure_patterns, synthetic_dataset
            )
            
            # Step 5: Validate best parameters
            tuning_run.status = TuningStatus.VALIDATING
            session.commit()
            
            validation_result = self.validate_best_parameters(session, tuning_run, experiments)
            
            # Step 6: Deploy or rollback
            final_result = self.deploy_or_rollback(session, tuning_run, validation_result)
            
            return final_result
            
        except Exception as e:
            # Mark tuning run as failed
            if 'tuning_run' in locals():
                tuning_run.status = TuningStatus.FAILED
                tuning_run.error_message = str(e)
                tuning_run.completed_at = datetime.now(timezone.utc)
                session.commit()
            
            self.logger.error(f"Tuning cycle failed: {e}")
            raise
    
    # Private helper methods
    
    def _check_rate_limit(self, session: Session, brand_name: str) -> bool:
        """Check if brand is within rate limit (max 1 tuning run per day)."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=1)
        
        recent_runs = session.query(TuningRunRateLimit).filter(
            and_(
                TuningRunRateLimit.brand_name == brand_name,
                TuningRunRateLimit.created_at > cutoff
            )
        ).count()
        
        return recent_runs == 0
    
    def _get_quarantined_issues_count(self, session: Session, brand_name: str) -> int:
        """Get count of quarantined issues for a brand."""
        return session.query(DocumentEvaluation).join(
            EvaluationRun
        ).filter(
            and_(
                EvaluationRun.brand_name == brand_name,
                DocumentEvaluation.weighted_overall_accuracy < 0.8
            )
        ).count()
    
    def _analyze_title_failures(self, failures: List[DocumentEvaluation]) -> FailureAnalysis:
        """Analyze title extraction failure patterns."""
        severity = sum(1 - r.title_accuracy for r in failures) / len(failures)
        
        return FailureAnalysis(
            failure_type=FailurePatternType.TITLE_EXTRACTION,
            affected_parameters=['accuracy.title_weight', 'model.extraction_confidence_threshold'],
            severity_score=severity,
            frequency=len(failures),
            examples=[{'evaluation_id': str(r.id), 'accuracy': r.title_accuracy} for r in failures[:5]],
            suggested_adjustments={
                'accuracy.title_weight': {'increase': 0.05},
                'model.extraction_confidence_threshold': {'decrease': 0.1}
            }
        )
    
    def _analyze_body_text_failures(self, failures: List[DocumentEvaluation]) -> FailureAnalysis:
        """Analyze body text extraction failure patterns."""
        severity = sum(1 - r.body_text_accuracy for r in failures) / len(failures)
        
        return FailureAnalysis(
            failure_type=FailurePatternType.BODY_TEXT_EXTRACTION,
            affected_parameters=['accuracy.body_text_weight', 'processing.batch_size'],
            severity_score=severity,
            frequency=len(failures),
            examples=[{'evaluation_id': str(r.id), 'accuracy': r.body_text_accuracy} for r in failures[:5]],
            suggested_adjustments={
                'accuracy.body_text_weight': {'increase': 0.1},
                'processing.batch_size': {'decrease': 8}
            }
        )
    
    def _analyze_contributor_failures(self, failures: List[DocumentEvaluation]) -> FailureAnalysis:
        """Analyze contributor extraction failure patterns."""
        severity = sum(1 - r.contributors_accuracy for r in failures) / len(failures)
        
        return FailureAnalysis(
            failure_type=FailurePatternType.CONTRIBUTOR_EXTRACTION,
            affected_parameters=['accuracy.contributors_weight'],
            severity_score=severity,
            frequency=len(failures),
            examples=[{'evaluation_id': str(r.id), 'accuracy': r.contributors_accuracy} for r in failures[:5]],
            suggested_adjustments={
                'accuracy.contributors_weight': {'increase': 0.05}
            }
        )
    
    def _analyze_media_failures(self, failures: List[DocumentEvaluation]) -> FailureAnalysis:
        """Analyze media link extraction failure patterns."""
        severity = sum(1 - r.media_links_accuracy for r in failures) / len(failures)
        
        return FailureAnalysis(
            failure_type=FailurePatternType.MEDIA_LINKS_EXTRACTION,
            affected_parameters=['accuracy.media_links_weight'],
            severity_score=severity,
            frequency=len(failures),
            examples=[{'evaluation_id': str(r.id), 'accuracy': r.media_links_accuracy} for r in failures[:5]],
            suggested_adjustments={
                'accuracy.media_links_weight': {'increase': 0.05}
            }
        )
    
    def _create_failure_pattern(
        self,
        session: Session,
        tuning_run: TuningRun,
        analysis: FailureAnalysis
    ) -> FailurePattern:
        """Create database record for identified failure pattern."""
        pattern = FailurePattern(
            tuning_run_id=tuning_run.id,
            failure_type=analysis.failure_type,
            affected_parameters=analysis.affected_parameters,
            severity_score=analysis.severity_score,
            frequency=analysis.frequency,
            examples=analysis.examples,
            suggested_adjustments=analysis.suggested_adjustments,
            created_at=datetime.now(timezone.utc)
        )
        
        session.add(pattern)
        return pattern
    
    def _build_generation_config(self, failure_patterns: List[FailurePattern]) -> Dict[str, Any]:
        """Build synthetic data generation config targeting failure patterns."""
        config = {
            'focus_areas': [],
            'edge_cases': [],
            'variations': {}
        }
        
        for pattern in failure_patterns:
            if pattern.failure_type == FailurePatternType.TITLE_EXTRACTION:
                config['focus_areas'].append('complex_titles')
                config['edge_cases'].extend(['decorative_fonts', 'multi_line_titles'])
            elif pattern.failure_type == FailurePatternType.BODY_TEXT_EXTRACTION:
                config['focus_areas'].append('dense_text_layouts')
                config['edge_cases'].extend(['multi_column', 'text_over_images'])
            elif pattern.failure_type == FailurePatternType.CONTRIBUTOR_EXTRACTION:
                config['focus_areas'].append('author_bylines')
                config['edge_cases'].extend(['multiple_authors', 'abbreviated_names'])
            elif pattern.failure_type == FailurePatternType.MEDIA_LINKS_EXTRACTION:
                config['focus_areas'].append('image_captions')
                config['edge_cases'].extend(['inline_images', 'image_galleries'])
        
        return config
    
    def _generate_examples_for_pattern(
        self,
        pattern: FailurePattern,
        count: int,
        brand_name: str
    ) -> List[Dict[str, str]]:
        """Generate synthetic examples targeting a specific failure pattern."""
        # This would use the synthetic data generator to create targeted examples
        # For now, return mock paths
        examples = []
        
        for i in range(count):
            examples.append({
                'pdf_path': f'/tmp/synthetic_{pattern.failure_type.value}_{i}.pdf',
                'xml_path': f'/tmp/synthetic_{pattern.failure_type.value}_{i}.xml'
            })
        
        return examples
    
    def _build_grid_search_config(self, failure_patterns: List[FailurePattern]) -> GridSearchConfig:
        """Build grid search configuration based on failure patterns."""
        parameters = {}
        
        for pattern in failure_patterns:
            for param_key, adjustment in pattern.suggested_adjustments.items():
                if param_key not in parameters:
                    # Get current parameter value
                    current_value = 0.5  # Mock - would get from parameter service
                    
                    # Generate search space around current value
                    if 'increase' in adjustment:
                        delta = adjustment['increase']
                        parameters[param_key] = [
                            current_value,
                            current_value + delta,
                            current_value + delta * 2
                        ]
                    elif 'decrease' in adjustment:
                        delta = adjustment['decrease']
                        parameters[param_key] = [
                            current_value,
                            current_value - delta,
                            current_value - delta * 2
                        ]
        
        return GridSearchConfig(
            parameters=parameters,
            max_experiments=50,
            early_stopping_threshold=0.02
        )
    
    def _generate_parameter_combinations(self, config: GridSearchConfig):
        """Generate all parameter combinations for grid search."""
        import itertools
        
        param_names = list(config.parameters.keys())
        param_values = list(config.parameters.values())
        
        for combination in itertools.product(*param_values):
            yield dict(zip(param_names, combination))
    
    def _calculate_baseline_accuracy(
        self,
        session: Session,
        brand_name: str,
        synthetic_dataset: SyntheticDataset
    ) -> float:
        """Calculate baseline accuracy before optimization."""
        # Mock implementation - would evaluate current parameters on synthetic data
        return 0.75
    
    def _apply_experimental_parameters(
        self,
        session: Session,
        parameters: Dict[str, Any],
        brand_name: str
    ):
        """Temporarily apply experimental parameters."""
        # Would use parameter service to create temporary overrides
        pass
    
    def _rollback_experimental_parameters(self, session: Session, brand_name: str):
        """Rollback experimental parameters to original values."""
        # Would remove temporary overrides
        pass
    
    def _evaluate_parameter_combination(
        self,
        session: Session,
        synthetic_dataset: SyntheticDataset,
        brand_name: str
    ) -> float:
        """Evaluate a parameter combination on synthetic data."""
        # Mock implementation - would run extraction on synthetic dataset
        import random
        return 0.7 + random.random() * 0.2
    
    def _get_holdout_validation_set(
        self,
        session: Session,
        brand_name: str
    ) -> List[DocumentEvaluation]:
        """Get recent real evaluation data for holdout validation."""
        return session.query(DocumentEvaluation).join(
            EvaluationRun
        ).filter(
            and_(
                EvaluationRun.brand_name == brand_name,
                DocumentEvaluation.weighted_overall_accuracy >= 0.8,
                EvaluationRun.created_at >= datetime.now(timezone.utc) - timedelta(days=7)
            )
        ).order_by(desc(EvaluationRun.created_at)).limit(50).all()
    
    def _calculate_holdout_accuracy(self, holdout_results: List[DocumentEvaluation]) -> float:
        """Calculate weighted accuracy on holdout set."""
        if not holdout_results:
            return 0.0
        
        total_weighted = 0.0
        for result in holdout_results:
            weighted = (
                result.title_accuracy * 0.30 +
                result.body_text_accuracy * 0.40 +
                result.contributors_accuracy * 0.20 +
                result.media_links_accuracy * 0.10
            )
            total_weighted += weighted
        
        return total_weighted / len(holdout_results)
    
    def _get_baseline_accuracy_for_validation(self, session: Session, brand_name: str) -> float:
        """Get baseline accuracy for comparison during validation."""
        # Get historical average accuracy for this brand
        recent_runs = session.query(EvaluationRun).filter(
            and_(
                EvaluationRun.brand_name == brand_name,
                EvaluationRun.created_at >= datetime.now(timezone.utc) - timedelta(days=30)
            )
        ).all()
        
        if not recent_runs:
            return 0.5  # Default baseline
        
        total_accuracy = sum(run.overall_weighted_accuracy for run in recent_runs)
        return total_accuracy / len(recent_runs)
    
    def _calculate_statistical_significance(
        self,
        new_accuracy: float,
        baseline_accuracy: float,
        sample_size: int
    ) -> Tuple[Tuple[float, float], float]:
        """Calculate confidence interval and p-value for accuracy improvement."""
        import scipy.stats as stats
        import numpy as np
        
        # Mock implementation - would use proper statistical testing
        improvement = new_accuracy - baseline_accuracy
        std_error = 0.05  # Mock standard error
        
        # 95% confidence interval
        margin = stats.norm.ppf(0.975) * std_error
        ci_lower = improvement - margin
        ci_upper = improvement + margin
        
        # t-test for significance
        t_stat = improvement / std_error
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        
        return (ci_lower, ci_upper), p_value
    
    def _deploy_parameters(
        self,
        session: Session,
        parameters: Dict[str, Any],
        brand_name: str
    ):
        """Deploy improved parameters as brand-specific overrides."""
        for param_key, param_value in parameters.items():
            # Create parameter override using parameter service
            self.parameter_service.create_parameter_override(
                session=session,
                override_request=ParameterUpdateRequest(
                    parameter_key=param_key,
                    new_value=param_value,
                    change_reason=f"Self-tuning deployment for {brand_name}",
                    created_by="self_tuning_system",
                    scope=ParameterScope.BRAND_SPECIFIC,
                    scope_identifier=brand_name
                )
            )


# Global service instance
self_tuning_service = SelfTuningService()