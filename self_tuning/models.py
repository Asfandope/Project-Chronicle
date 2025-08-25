"""
Database models for the self-tuning system.

This module defines the schema for tracking tuning runs, failure patterns,
generated synthetic data, and optimization results.
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, JSON, Boolean, Text,
    ForeignKey, Index, UniqueConstraint, CheckConstraint, Enum
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
import enum

Base = declarative_base()


class TuningStatus(enum.Enum):
    """Status of tuning runs."""
    PENDING = "pending"
    ANALYZING = "analyzing"
    GENERATING = "generating"
    OPTIMIZING = "optimizing"
    VALIDATING = "validating"
    DEPLOYING = "deploying"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class FailurePatternType(enum.Enum):
    """Types of failure patterns identified."""
    TITLE_EXTRACTION = "title_extraction"
    BODY_TEXT_OCR = "body_text_ocr"
    CONTRIBUTOR_PARSING = "contributor_parsing"
    MEDIA_ASSOCIATION = "media_association"
    LAYOUT_COMPLEXITY = "layout_complexity"
    FONT_RECOGNITION = "font_recognition"
    COLUMN_DETECTION = "column_detection"
    LANGUAGE_DETECTION = "language_detection"


class OptimizationStrategy(enum.Enum):
    """Optimization strategies for parameter tuning."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    GRADIENT_DESCENT = "gradient_descent"


class ExperimentStatus(enum.Enum):
    """Status of parameter experiments."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ValidationStatus(enum.Enum):
    """Status of validation runs."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TuningRun(Base):
    """Represents a complete self-tuning execution."""
    
    __tablename__ = "tuning_runs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Run identification
    brand_name = Column(String(100), nullable=False, index=True)
    run_name = Column(String(255), nullable=False)
    
    # Trigger information
    trigger_source = Column(String(100), nullable=False)  # drift_detection, manual, scheduled
    trigger_accuracy_drop = Column(Float)
    trigger_metric_type = Column(String(50))
    
    # Status tracking
    status = Column(Enum(TuningStatus), nullable=False, default=TuningStatus.PENDING, index=True)
    
    # Timing
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    total_duration_seconds = Column(Float)
    
    # Analysis phase
    analysis_duration_seconds = Column(Float)
    quarantined_issues_analyzed = Column(Integer, default=0)
    failure_patterns_identified = Column(Integer, default=0)
    
    # Generation phase
    generation_duration_seconds = Column(Float)
    synthetic_examples_generated = Column(Integer, default=0)
    targeted_examples_generated = Column(Integer, default=0)
    
    # Optimization phase
    optimization_duration_seconds = Column(Float)
    optimization_strategy = Column(Enum(OptimizationStrategy), default=OptimizationStrategy.GRID_SEARCH)
    parameter_combinations_tested = Column(Integer, default=0)
    best_parameter_set_found = Column(Boolean, default=False)
    
    # Validation phase
    validation_duration_seconds = Column(Float)
    holdout_set_size = Column(Integer, default=0)
    validation_accuracy_improvement = Column(Float)
    validation_passed = Column(Boolean, default=False)
    
    # Deployment results
    deployment_successful = Column(Boolean, default=False)
    rollback_performed = Column(Boolean, default=False)
    rollback_reason = Column(Text)
    
    # Performance metrics
    baseline_accuracy = Column(Float)
    optimized_accuracy = Column(Float)
    accuracy_improvement = Column(Float)
    
    # Configuration
    tuning_config = Column(JSONB)  # Complete tuning configuration
    
    # Results and logs
    execution_log = Column(Text)
    error_message = Column(Text)
    results_summary = Column(JSONB)
    
    # Rate limiting
    daily_run_number = Column(Integer, default=1)  # Track runs per day per brand
    
    # Relationships
    failure_patterns = relationship("FailurePattern", back_populates="tuning_run", cascade="all, delete-orphan")
    synthetic_datasets = relationship("SyntheticDataset", back_populates="tuning_run", cascade="all, delete-orphan")
    parameter_experiments = relationship("ParameterExperiment", back_populates="tuning_run", cascade="all, delete-orphan")
    validation_results = relationship("ValidationResult", back_populates="tuning_run", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_tuning_runs_brand_created', 'brand_name', 'created_at'),
        Index('idx_tuning_runs_status', 'status'),
        Index('idx_tuning_runs_trigger', 'trigger_source'),
    )


class FailurePattern(Base):
    """Identified failure patterns from quarantined issues."""
    
    __tablename__ = "failure_patterns"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tuning_run_id = Column(UUID(as_uuid=True), ForeignKey("tuning_runs.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Pattern identification
    pattern_type = Column(Enum(FailurePatternType), nullable=False, index=True)
    pattern_name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Pattern characteristics
    frequency = Column(Integer, nullable=False, default=1)
    severity_score = Column(Float, nullable=False, default=0.5)  # 0-1 scale
    confidence = Column(Float, nullable=False, default=0.5)  # 0-1 confidence in pattern
    
    # Affected parameters
    affected_parameters = Column(JSONB)  # List of parameter keys that might fix this
    suggested_parameter_changes = Column(JSONB)  # Specific parameter adjustments
    
    # Pattern details
    common_characteristics = Column(JSONB)  # Common features of failing cases
    error_signatures = Column(JSONB)  # Error patterns or signatures
    
    # Examples
    sample_document_ids = Column(JSONB)  # Sample documents showing this pattern
    extraction_errors = Column(JSONB)  # Specific extraction errors
    
    # Impact metrics
    accuracy_impact = Column(Float)  # How much this pattern affects accuracy
    frequency_trend = Column(String(20))  # increasing, stable, decreasing
    
    # Analysis metadata
    analysis_method = Column(String(100))  # How this pattern was identified
    analysis_confidence = Column(Float, default=0.5)
    
    # Relationships
    tuning_run = relationship("TuningRun", back_populates="failure_patterns")
    targeted_examples = relationship("TargetedSyntheticExample", back_populates="failure_pattern")
    
    __table_args__ = (
        Index('idx_failure_patterns_type', 'pattern_type'),
        Index('idx_failure_patterns_severity', 'severity_score'),
        Index('idx_failure_patterns_tuning_run', 'tuning_run_id'),
    )


class SyntheticDataset(Base):
    """Generated synthetic datasets for tuning."""
    
    __tablename__ = "synthetic_datasets"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tuning_run_id = Column(UUID(as_uuid=True), ForeignKey("tuning_runs.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Dataset metadata
    dataset_name = Column(String(255), nullable=False)
    dataset_type = Column(String(50), nullable=False)  # targeted, baseline, holdout
    brand_name = Column(String(100), nullable=False)
    
    # Generation parameters
    generation_config = Column(JSONB)  # Configuration used for generation
    targeted_patterns = Column(JSONB)  # Patterns this dataset targets
    
    # Dataset statistics
    document_count = Column(Integer, nullable=False, default=0)
    article_count = Column(Integer, default=0)
    
    # Complexity distribution
    complexity_simple = Column(Integer, default=0)
    complexity_moderate = Column(Integer, default=0)
    complexity_complex = Column(Integer, default=0)
    complexity_chaotic = Column(Integer, default=0)
    
    # Edge case coverage
    edge_cases_covered = Column(JSONB)  # List of edge cases included
    edge_case_frequency = Column(Float, default=0.0)  # Fraction with edge cases
    
    # Quality metrics
    generation_success_rate = Column(Float, default=1.0)
    validation_accuracy = Column(Float)  # Self-validation accuracy
    
    # File paths
    dataset_directory = Column(String(500))  # Path to generated files
    ground_truth_path = Column(String(500))  # Path to ground truth data
    
    # Generation timing
    generation_duration_seconds = Column(Float)
    generation_start_time = Column(DateTime(timezone=True))
    generation_end_time = Column(DateTime(timezone=True))
    
    # Relationships
    tuning_run = relationship("TuningRun", back_populates="synthetic_datasets")
    targeted_examples = relationship("TargetedSyntheticExample", back_populates="synthetic_dataset")
    
    __table_args__ = (
        Index('idx_synthetic_datasets_tuning_run', 'tuning_run_id'),
        Index('idx_synthetic_datasets_type', 'dataset_type'),
        Index('idx_synthetic_datasets_brand', 'brand_name'),
    )


class TargetedSyntheticExample(Base):
    """Individual synthetic examples targeting specific failure patterns."""
    
    __tablename__ = "targeted_synthetic_examples"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    synthetic_dataset_id = Column(UUID(as_uuid=True), ForeignKey("synthetic_datasets.id"), nullable=False)
    failure_pattern_id = Column(UUID(as_uuid=True), ForeignKey("failure_patterns.id"), nullable=False)
    
    # Example identification
    document_id = Column(String(255), nullable=False)
    example_name = Column(String(255))
    
    # Target characteristics
    target_pattern_type = Column(Enum(FailurePatternType), nullable=False)
    difficulty_level = Column(Float, nullable=False, default=0.5)  # 0-1 scale
    
    # Generation parameters
    generation_parameters = Column(JSONB)  # Specific parameters for this example
    edge_cases_applied = Column(JSONB)  # Edge cases intentionally included
    
    # Expected results
    expected_accuracy = Column(Float)  # Expected extraction accuracy
    expected_challenges = Column(JSONB)  # Expected extraction challenges
    
    # File references
    pdf_path = Column(String(500))
    ground_truth_path = Column(String(500))
    
    # Validation
    generation_successful = Column(Boolean, default=True)
    validation_notes = Column(Text)
    
    # Relationships
    synthetic_dataset = relationship("SyntheticDataset", back_populates="targeted_examples")
    failure_pattern = relationship("FailurePattern", back_populates="targeted_examples")
    
    __table_args__ = (
        Index('idx_targeted_examples_pattern', 'failure_pattern_id'),
        Index('idx_targeted_examples_dataset', 'synthetic_dataset_id'),
        Index('idx_targeted_examples_difficulty', 'difficulty_level'),
    )


class ParameterExperiment(Base):
    """Individual parameter combination experiments."""
    
    __tablename__ = "parameter_experiments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tuning_run_id = Column(UUID(as_uuid=True), ForeignKey("tuning_runs.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Experiment identification
    experiment_name = Column(String(255), nullable=False)
    experiment_number = Column(Integer, nullable=False)
    
    # Parameter configuration
    parameter_set = Column(JSONB, nullable=False)  # The parameter values tested
    parameter_hash = Column(String(64), index=True)  # Hash of parameter set for deduplication
    
    # Optimization context
    optimization_step = Column(Integer, default=0)  # Step in optimization process
    grid_coordinates = Column(JSONB)  # Coordinates in parameter grid
    parent_experiment_id = Column(UUID(as_uuid=True), ForeignKey("parameter_experiments.id"))
    
    # Execution
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    execution_duration_seconds = Column(Float)
    
    # Results
    training_accuracy = Column(Float)  # Accuracy on training/tuning set
    validation_accuracy = Column(Float)  # Accuracy on validation set
    overall_accuracy = Column(Float)  # Overall weighted accuracy
    
    # Detailed metrics
    title_accuracy = Column(Float)
    body_text_accuracy = Column(Float)
    contributors_accuracy = Column(Float)
    media_links_accuracy = Column(Float)
    
    # Performance metrics
    processing_time_seconds = Column(Float)
    memory_usage_mb = Column(Float)
    success_rate = Column(Float)  # Fraction of documents processed successfully
    
    # Comparison metrics
    improvement_over_baseline = Column(Float)  # Improvement vs baseline
    rank = Column(Integer)  # Rank among all experiments in this run
    
    # Execution details
    execution_successful = Column(Boolean, default=False)
    error_message = Column(Text)
    execution_log = Column(Text)
    
    # Statistical significance
    confidence_interval = Column(JSONB)  # [lower, upper] confidence bounds
    p_value = Column(Float)  # Statistical significance vs baseline
    
    # Relationships
    tuning_run = relationship("TuningRun", back_populates="parameter_experiments")
    child_experiments = relationship("ParameterExperiment")
    
    __table_args__ = (
        Index('idx_parameter_experiments_tuning_run', 'tuning_run_id'),
        Index('idx_parameter_experiments_hash', 'parameter_hash'),
        Index('idx_parameter_experiments_accuracy', 'overall_accuracy'),
        Index('idx_parameter_experiments_rank', 'rank'),
    )


class ValidationResult(Base):
    """Results from holdout set validation."""
    
    __tablename__ = "validation_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tuning_run_id = Column(UUID(as_uuid=True), ForeignKey("tuning_runs.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Validation configuration
    holdout_set_size = Column(Integer, nullable=False)
    holdout_set_composition = Column(JSONB)  # Breakdown by complexity, edge cases
    validation_strategy = Column(String(50), default="random_holdout")
    
    # Parameter set being validated
    parameter_set = Column(JSONB, nullable=False)
    parameter_experiment_id = Column(UUID(as_uuid=True), ForeignKey("parameter_experiments.id"))
    
    # Baseline comparison
    baseline_parameter_set = Column(JSONB)
    baseline_accuracy = Column(Float)
    
    # Validation results
    validation_accuracy = Column(Float, nullable=False)
    accuracy_improvement = Column(Float)  # Improvement over baseline
    
    # Detailed metrics
    title_accuracy = Column(Float)
    body_text_accuracy = Column(Float)
    contributors_accuracy = Column(Float)
    media_links_accuracy = Column(Float)
    
    # Statistical analysis
    confidence_interval = Column(JSONB)
    p_value = Column(Float)
    statistical_significance = Column(Boolean, default=False)
    
    # Performance analysis
    processing_time_improvement = Column(Float)
    memory_usage_change = Column(Float)
    stability_score = Column(Float)  # Consistency across different documents
    
    # Validation decision
    validation_passed = Column(Boolean, nullable=False)
    improvement_threshold_met = Column(Boolean, default=False)
    significance_threshold_met = Column(Boolean, default=False)
    
    # Detailed results
    document_level_results = Column(JSONB)  # Results per document
    pattern_specific_improvements = Column(JSONB)  # Improvement per failure pattern
    
    # Validation execution
    validation_duration_seconds = Column(Float)
    validation_successful = Column(Boolean, default=True)
    validation_error = Column(Text)
    
    # Relationships
    tuning_run = relationship("TuningRun", back_populates="validation_results")
    parameter_experiment = relationship("ParameterExperiment")
    
    __table_args__ = (
        Index('idx_validation_results_tuning_run', 'tuning_run_id'),
        Index('idx_validation_results_passed', 'validation_passed'),
        Index('idx_validation_results_improvement', 'accuracy_improvement'),
    )


class TuningRunRateLimit(Base):
    """Rate limiting for tuning runs (max one per brand per day)."""
    
    __tablename__ = "tuning_run_rate_limits"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Rate limiting key
    brand_name = Column(String(100), nullable=False)
    date = Column(DateTime(timezone=True), nullable=False)  # Date (truncated to day)
    
    # Run tracking
    tuning_run_count = Column(Integer, nullable=False, default=0)
    last_run_id = Column(UUID(as_uuid=True), ForeignKey("tuning_runs.id"))
    last_run_at = Column(DateTime(timezone=True))
    
    # Rate limit status
    limit_exceeded = Column(Boolean, default=False)
    next_allowed_run = Column(DateTime(timezone=True))
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Relationships
    last_run = relationship("TuningRun")
    
    __table_args__ = (
        UniqueConstraint('brand_name', 'date', name='uq_rate_limit_brand_date'),
        Index('idx_rate_limits_brand', 'brand_name'),
        Index('idx_rate_limits_date', 'date'),
    )


# Database utility functions
def create_tuning_tables(engine):
    """Create all tuning-related tables."""
    Base.metadata.create_all(engine)


def check_tuning_rate_limit(session: Session, brand_name: str) -> tuple[bool, Optional[datetime]]:
    """
    Check if a brand can start a new tuning run today.
    
    Returns:
        (can_run, next_allowed_time)
    """
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    
    rate_limit = (session.query(TuningRunRateLimit)
                 .filter(
                     TuningRunRateLimit.brand_name == brand_name,
                     TuningRunRateLimit.date == today
                 )
                 .first())
    
    if not rate_limit:
        # No runs today, allowed to run
        return True, None
    
    if rate_limit.tuning_run_count >= 1:
        # Already ran today, not allowed
        next_allowed = today + timedelta(days=1)
        return False, next_allowed
    
    return True, None


def record_tuning_run_start(session: Session, brand_name: str, tuning_run_id: str) -> None:
    """Record that a tuning run has started for rate limiting."""
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    now = datetime.now(timezone.utc)
    
    rate_limit = (session.query(TuningRunRateLimit)
                 .filter(
                     TuningRunRateLimit.brand_name == brand_name,
                     TuningRunRateLimit.date == today
                 )
                 .first())
    
    if rate_limit:
        rate_limit.tuning_run_count += 1
        rate_limit.last_run_id = tuning_run_id
        rate_limit.last_run_at = now
        rate_limit.limit_exceeded = rate_limit.tuning_run_count >= 1
        rate_limit.updated_at = now
    else:
        rate_limit = TuningRunRateLimit(
            brand_name=brand_name,
            date=today,
            tuning_run_count=1,
            last_run_id=tuning_run_id,
            last_run_at=now,
            limit_exceeded=False
        )
        session.add(rate_limit)
    
    session.commit()


def get_recent_tuning_runs(session: Session, brand_name: str = None, limit: int = 10) -> List[TuningRun]:
    """Get recent tuning runs, optionally filtered by brand."""
    query = session.query(TuningRun)
    
    if brand_name:
        query = query.filter(TuningRun.brand_name == brand_name)
    
    return query.order_by(TuningRun.created_at.desc()).limit(limit).all()


def get_tuning_run_statistics(session: Session, days: int = 30) -> Dict[str, Any]:
    """Get statistics about tuning runs over the specified period."""
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    
    runs = (session.query(TuningRun)
           .filter(TuningRun.created_at >= cutoff_date)
           .all())
    
    if not runs:
        return {
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'average_improvement': 0.0,
            'brands_tuned': 0
        }
    
    successful_runs = [r for r in runs if r.status == TuningStatus.COMPLETED]
    failed_runs = [r for r in runs if r.status == TuningStatus.FAILED]
    
    improvements = [r.accuracy_improvement for r in successful_runs if r.accuracy_improvement is not None]
    
    return {
        'total_runs': len(runs),
        'successful_runs': len(successful_runs),
        'failed_runs': len(failed_runs),
        'rollback_count': sum(1 for r in runs if r.rollback_performed),
        'average_improvement': sum(improvements) / len(improvements) if improvements else 0.0,
        'brands_tuned': len(set(r.brand_name for r in runs)),
        'average_duration_minutes': sum(r.total_duration_seconds or 0 for r in successful_runs) / len(successful_runs) / 60 if successful_runs else 0.0
    }