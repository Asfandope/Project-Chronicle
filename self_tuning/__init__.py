"""
Self-tuning system for automated parameter optimization.

This package implements the self-tuning system from PRD section 7.3:
1. Identify failure patterns from quarantined issues
2. Generate targeted synthetic examples
3. Grid search over parameter space
4. Validate on holdout set
5. Deploy if improvement, rollback if not

Key features:
- Rate limiting: Max one tuning run per brand per day
- Statistical validation with confidence intervals
- Automatic rollback if improvements don't meet thresholds
- Comprehensive audit trails and monitoring
- Integration with parameter management and evaluation systems
"""

from .api import create_self_tuning_api, mount_self_tuning_api
from .models import (
    ExperimentStatus,
    FailurePattern,
    FailurePatternType,
    OptimizationStrategy,
    ParameterExperiment,
    SyntheticDataset,
    TuningRun,
    TuningRunRateLimit,
    TuningStatus,
    ValidationResult,
    ValidationStatus,
)
from .service import (
    FailureAnalysis,
    GridSearchConfig,
    SelfTuningService,
    TuningResult,
    self_tuning_service,
)


# Main interface functions
def start_tuning_for_brand(brand_name: str, session, force: bool = False) -> str:
    """
    Start a tuning run for a specific brand.

    Args:
        brand_name: Target brand name
        session: Database session
        force: Skip rate limiting check

    Returns:
        Tuning run ID

    Example:
        session = get_db_session()
        tuning_run_id = start_tuning_for_brand("TechWeekly", session)
    """
    service = SelfTuningService()
    tuning_run = service.start_tuning_run(
        session=session, brand_name=brand_name, triggered_by="manual", force=force
    )
    return str(tuning_run.id)


def run_complete_tuning_cycle(
    brand_name: str, session, force: bool = False
) -> TuningResult:
    """
    Execute a complete tuning cycle from start to finish.

    Args:
        brand_name: Target brand name
        session: Database session
        force: Skip rate limiting check

    Returns:
        Final tuning result

    Example:
        session = get_db_session()
        result = run_complete_tuning_cycle("TechWeekly", session)
        if result.improvement_achieved:
            print(f"Improvement: {result.accuracy_improvement:.2%}")
    """
    service = SelfTuningService()
    return service.run_complete_tuning_cycle(
        session=session, brand_name=brand_name, triggered_by="manual", force=force
    )


def check_brand_tuning_eligibility(brand_name: str, session) -> dict:
    """
    Check if a brand is eligible for tuning.

    Args:
        brand_name: Target brand name
        session: Database session

    Returns:
        Dictionary with eligibility information

    Example:
        eligibility = check_brand_tuning_eligibility("TechWeekly", session)
        if eligibility["can_tune"]:
            start_tuning_for_brand("TechWeekly", session)
    """
    service = SelfTuningService()

    within_rate_limit = service._check_rate_limit(session, brand_name)
    quarantined_count = service._get_quarantined_issues_count(session, brand_name)

    can_tune = within_rate_limit and quarantined_count >= 10

    return {
        "can_tune": can_tune,
        "within_rate_limit": within_rate_limit,
        "quarantined_issues_count": quarantined_count,
        "minimum_required": 10,
        "message": (
            "Ready for tuning"
            if can_tune
            else "Rate limit exceeded"
            if not within_rate_limit
            else f"Insufficient quarantined data ({quarantined_count}/10)"
        ),
    }


def get_tuning_system_status(session) -> dict:
    """
    Get overall status of the self-tuning system.

    Args:
        session: Database session

    Returns:
        Dictionary with system status information
    """
    from datetime import datetime, timedelta, timezone

    from sqlalchemy import and_

    # Basic counts
    total_runs = session.query(TuningRun).count()
    active_runs = (
        session.query(TuningRun)
        .filter(
            TuningRun.status.in_(
                [
                    TuningStatus.ANALYZING_FAILURES,
                    TuningStatus.GENERATING_DATA,
                    TuningStatus.OPTIMIZING_PARAMETERS,
                    TuningStatus.VALIDATING,
                ]
            )
        )
        .count()
    )

    successful_deployments = (
        session.query(TuningRun)
        .filter(TuningRun.status == TuningStatus.DEPLOYED)
        .count()
    )

    failed_runs = (
        session.query(TuningRun)
        .filter(TuningRun.status.in_([TuningStatus.FAILED, TuningStatus.ROLLED_BACK]))
        .count()
    )

    # Recent activity
    last_24h_cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    last_24h_runs = (
        session.query(TuningRun).filter(TuningRun.created_at > last_24h_cutoff).count()
    )

    # Average improvement
    successful_runs = (
        session.query(TuningRun)
        .filter(
            and_(
                TuningRun.status == TuningStatus.DEPLOYED,
                TuningRun.accuracy_improvement.isnot(None),
            )
        )
        .all()
    )

    average_improvement = 0.0
    if successful_runs:
        average_improvement = sum(
            run.accuracy_improvement for run in successful_runs
        ) / len(successful_runs)

    return {
        "total_tuning_runs": total_runs,
        "active_tuning_runs": active_runs,
        "successful_deployments": successful_deployments,
        "failed_runs": failed_runs,
        "average_improvement": average_improvement,
        "last_24h_runs": last_24h_runs,
        "success_rate": successful_deployments / total_runs if total_runs > 0 else 0.0,
    }


# Convenience functions for common workflows
def trigger_tuning_from_drift(brand_name: str, session) -> str:
    """
    Trigger tuning run from drift detection.

    This is typically called automatically by the drift detection system
    when accuracy thresholds are breached.
    """
    service = SelfTuningService()
    tuning_run = service.start_tuning_run(
        session=session, brand_name=brand_name, triggered_by="drift_detection"
    )
    return str(tuning_run.id)


def get_recent_tuning_runs(session, brand_name: str = None, limit: int = 10):
    """
    Get recent tuning runs with optional brand filtering.

    Args:
        session: Database session
        brand_name: Optional brand filter
        limit: Maximum number of runs to return

    Returns:
        List of tuning run records
    """
    query = session.query(TuningRun)

    if brand_name:
        query = query.filter(TuningRun.brand_name == brand_name)

    return query.order_by(TuningRun.created_at.desc()).limit(limit).all()


def get_tuning_run_summary(tuning_run_id: str, session) -> dict:
    """
    Get comprehensive summary of a tuning run.

    Args:
        tuning_run_id: Tuning run ID
        session: Database session

    Returns:
        Dictionary with complete tuning run information
    """
    # Get tuning run
    tuning_run = session.query(TuningRun).filter(TuningRun.id == tuning_run_id).first()

    if not tuning_run:
        raise ValueError(f"Tuning run not found: {tuning_run_id}")

    # Get related data
    failure_patterns = (
        session.query(FailurePattern)
        .filter(FailurePattern.tuning_run_id == tuning_run_id)
        .all()
    )

    experiments = (
        session.query(ParameterExperiment)
        .filter(ParameterExperiment.tuning_run_id == tuning_run_id)
        .all()
    )

    validation = (
        session.query(ValidationResult)
        .filter(ValidationResult.tuning_run_id == tuning_run_id)
        .first()
    )

    # Build summary
    summary = {
        "tuning_run": {
            "id": str(tuning_run.id),
            "brand_name": tuning_run.brand_name,
            "status": tuning_run.status.value,
            "triggered_by": tuning_run.triggered_by,
            "optimization_strategy": tuning_run.optimization_strategy.value,
            "created_at": tuning_run.created_at,
            "completed_at": tuning_run.completed_at,
            "baseline_accuracy": tuning_run.baseline_accuracy,
            "final_accuracy": tuning_run.final_accuracy,
            "accuracy_improvement": tuning_run.accuracy_improvement,
            "deployed_parameters": tuning_run.deployed_parameters,
            "rollback_reason": tuning_run.rollback_reason,
            "error_message": tuning_run.error_message,
        },
        "failure_patterns": [
            {
                "failure_type": fp.failure_type.value,
                "affected_parameters": fp.affected_parameters,
                "severity_score": fp.severity_score,
                "frequency": fp.frequency,
            }
            for fp in failure_patterns
        ],
        "experiments": [
            {
                "parameter_values": exp.parameter_values,
                "accuracy_score": exp.accuracy_score,
                "improvement_over_baseline": exp.improvement_over_baseline,
                "status": exp.status.value,
            }
            for exp in experiments
        ],
        "validation": {
            "holdout_accuracy": validation.holdout_accuracy,
            "baseline_accuracy": validation.baseline_accuracy,
            "accuracy_improvement": validation.accuracy_improvement,
            "is_statistically_significant": validation.is_statistically_significant,
            "meets_improvement_threshold": validation.meets_improvement_threshold,
            "status": validation.status.value,
        }
        if validation
        else None,
    }

    return summary


__version__ = "1.0.0"

__all__ = [
    # Core models
    "TuningRun",
    "FailurePattern",
    "SyntheticDataset",
    "ParameterExperiment",
    "ValidationResult",
    "TuningRunRateLimit",
    # Enums
    "TuningStatus",
    "FailurePatternType",
    "OptimizationStrategy",
    "ExperimentStatus",
    "ValidationStatus",
    # Service layer
    "SelfTuningService",
    "TuningResult",
    "FailureAnalysis",
    "GridSearchConfig",
    "self_tuning_service",
    # API
    "create_self_tuning_api",
    "mount_self_tuning_api",
    # Main interface functions
    "start_tuning_for_brand",
    "run_complete_tuning_cycle",
    "check_brand_tuning_eligibility",
    "get_tuning_system_status",
    # Convenience functions
    "trigger_tuning_from_drift",
    "get_recent_tuning_runs",
    "get_tuning_run_summary",
]
