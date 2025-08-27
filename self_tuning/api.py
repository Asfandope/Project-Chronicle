"""
Self-tuning system REST API.

Provides endpoints for managing and monitoring the self-tuning system.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from db_deps import get_db_session_dependency as get_db_session

from .models import (
    FailurePattern,
    FailurePatternType,
    OptimizationStrategy,
    ParameterExperiment,
    TuningRun,
    TuningStatus,
    ValidationResult,
)
from .service import SelfTuningService

logger = logging.getLogger(__name__)


# Pydantic models for API requests/responses


class TuningRunRequest(BaseModel):
    """Request to start a new tuning run."""

    brand_name: str = Field(..., description="Target brand name")
    triggered_by: str = Field(
        default="manual", description="What triggered this tuning run"
    )
    strategy: OptimizationStrategy = Field(
        default=OptimizationStrategy.GRID_SEARCH, description="Optimization strategy"
    )
    force: bool = Field(default=False, description="Skip rate limiting check")


class TuningRunResponse(BaseModel):
    """Response containing tuning run information."""

    id: str
    brand_name: str
    status: TuningStatus
    triggered_by: str
    optimization_strategy: OptimizationStrategy
    created_at: datetime
    completed_at: Optional[datetime]
    baseline_accuracy: Optional[float]
    final_accuracy: Optional[float]
    accuracy_improvement: Optional[float]
    deployed_parameters: Optional[Dict[str, Any]]
    rollback_reason: Optional[str]
    error_message: Optional[str]

    class Config:
        from_attributes = True


class FailurePatternResponse(BaseModel):
    """Response containing failure pattern information."""

    id: str
    failure_type: FailurePatternType
    affected_parameters: List[str]
    severity_score: float
    frequency: int
    examples: List[Dict[str, Any]]
    suggested_adjustments: Dict[str, Any]
    created_at: datetime

    class Config:
        from_attributes = True


class ParameterExperimentResponse(BaseModel):
    """Response containing parameter experiment results."""

    id: str
    parameter_values: Dict[str, Any]
    accuracy_score: Optional[float]
    improvement_over_baseline: Optional[float]
    status: str
    error_message: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


class ValidationResultResponse(BaseModel):
    """Response containing validation results."""

    id: str
    holdout_accuracy: Optional[float]
    baseline_accuracy: Optional[float]
    accuracy_improvement: Optional[float]
    confidence_interval_lower: Optional[float]
    confidence_interval_upper: Optional[float]
    p_value: Optional[float]
    is_statistically_significant: Optional[bool]
    meets_improvement_threshold: Optional[bool]
    status: str
    error_message: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


class TuningRunSummary(BaseModel):
    """Summary of tuning run with related data."""

    tuning_run: TuningRunResponse
    failure_patterns: List[FailurePatternResponse]
    experiments: List[ParameterExperimentResponse]
    validation: Optional[ValidationResultResponse]


class TuningSystemStatus(BaseModel):
    """Overall status of the self-tuning system."""

    total_tuning_runs: int
    active_tuning_runs: int
    successful_deployments: int
    failed_runs: int
    brands_with_recent_tuning: List[str]
    average_improvement: float
    last_24h_runs: int


# Create router
def create_self_tuning_api() -> APIRouter:
    """Create the self-tuning API router."""
    router = APIRouter(prefix="/self-tuning", tags=["self-tuning"])
    service = SelfTuningService()

    @router.post("/start", response_model=TuningRunResponse)
    async def start_tuning_run(
        request: TuningRunRequest,
        background_tasks: BackgroundTasks,
        session: Session = Depends(get_db_session),
    ):
        """
        Start a new tuning run for a brand.

        The tuning run will execute in the background and can be monitored
        using the status endpoints.
        """
        try:
            # Start tuning run
            tuning_run = service.start_tuning_run(
                session=session,
                brand_name=request.brand_name,
                triggered_by=request.triggered_by,
                strategy=request.strategy,
                force=request.force,
            )

            # Execute full tuning cycle in background
            background_tasks.add_task(
                _execute_tuning_cycle_background,
                request.brand_name,
                request.triggered_by,
                request.force,
            )

            return TuningRunResponse.from_orm(tuning_run)

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Failed to start tuning run: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @router.post("/run-complete/{brand_name}", response_model=dict)
    async def run_complete_tuning_cycle(
        brand_name: str,
        triggered_by: str = "manual",
        force: bool = False,
        session: Session = Depends(get_db_session),
    ):
        """
        Execute a complete tuning cycle synchronously.

        This endpoint will block until the tuning cycle is complete.
        Use the /start endpoint for asynchronous execution.
        """
        try:
            result = service.run_complete_tuning_cycle(
                session=session,
                brand_name=brand_name,
                triggered_by=triggered_by,
                force=force,
            )

            return {
                "tuning_run_id": result.tuning_run_id,
                "status": result.status.value,
                "improvement_achieved": result.improvement_achieved,
                "baseline_accuracy": result.baseline_accuracy,
                "final_accuracy": result.final_accuracy,
                "accuracy_improvement": result.accuracy_improvement,
                "deployed_parameters": result.deployed_parameters,
                "rollback_reason": result.rollback_reason,
            }

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Tuning cycle failed: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @router.get("/runs", response_model=List[TuningRunResponse])
    async def list_tuning_runs(
        brand_name: Optional[str] = None,
        status: Optional[TuningStatus] = None,
        limit: int = 50,
        session: Session = Depends(get_db_session),
    ):
        """List tuning runs with optional filtering."""
        try:
            query = session.query(TuningRun)

            if brand_name:
                query = query.filter(TuningRun.brand_name == brand_name)
            if status:
                query = query.filter(TuningRun.status == status)

            runs = query.order_by(TuningRun.created_at.desc()).limit(limit).all()
            return [TuningRunResponse.from_orm(run) for run in runs]

        except Exception as e:
            logger.error(f"Failed to list tuning runs: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @router.get("/runs/{tuning_run_id}", response_model=TuningRunSummary)
    async def get_tuning_run(
        tuning_run_id: str, session: Session = Depends(get_db_session)
    ):
        """Get detailed information about a specific tuning run."""
        try:
            # Get tuning run
            tuning_run = (
                session.query(TuningRun).filter(TuningRun.id == tuning_run_id).first()
            )

            if not tuning_run:
                raise HTTPException(status_code=404, detail="Tuning run not found")

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

            return TuningRunSummary(
                tuning_run=TuningRunResponse.from_orm(tuning_run),
                failure_patterns=[
                    FailurePatternResponse.from_orm(fp) for fp in failure_patterns
                ],
                experiments=[
                    ParameterExperimentResponse.from_orm(exp) for exp in experiments
                ],
                validation=ValidationResultResponse.from_orm(validation)
                if validation
                else None,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get tuning run: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @router.post("/runs/{tuning_run_id}/cancel")
    async def cancel_tuning_run(
        tuning_run_id: str, session: Session = Depends(get_db_session)
    ):
        """Cancel a running tuning run."""
        try:
            tuning_run = (
                session.query(TuningRun).filter(TuningRun.id == tuning_run_id).first()
            )

            if not tuning_run:
                raise HTTPException(status_code=404, detail="Tuning run not found")

            if tuning_run.status in [
                TuningStatus.COMPLETED,
                TuningStatus.DEPLOYED,
                TuningStatus.ROLLED_BACK,
                TuningStatus.FAILED,
            ]:
                raise HTTPException(
                    status_code=400, detail="Tuning run already completed"
                )

            tuning_run.status = TuningStatus.CANCELLED
            tuning_run.completed_at = datetime.now(timezone.utc)
            session.commit()

            return {"message": "Tuning run cancelled successfully"}

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to cancel tuning run: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @router.get("/status", response_model=TuningSystemStatus)
    async def get_system_status(session: Session = Depends(get_db_session)):
        """Get overall status of the self-tuning system."""
        try:
            from datetime import timedelta

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
                .filter(
                    TuningRun.status.in_(
                        [TuningStatus.FAILED, TuningStatus.ROLLED_BACK]
                    )
                )
                .count()
            )

            # Recent activity
            last_24h_cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
            last_24h_runs = (
                session.query(TuningRun)
                .filter(TuningRun.created_at > last_24h_cutoff)
                .count()
            )

            # Brands with recent tuning
            recent_brands = (
                session.query(TuningRun.brand_name)
                .filter(
                    TuningRun.created_at
                    > datetime.now(timezone.utc) - timedelta(days=7)
                )
                .distinct()
                .all()
            )
            brands_with_recent_tuning = [brand[0] for brand in recent_brands]

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

            return TuningSystemStatus(
                total_tuning_runs=total_runs,
                active_tuning_runs=active_runs,
                successful_deployments=successful_deployments,
                failed_runs=failed_runs,
                brands_with_recent_tuning=brands_with_recent_tuning,
                average_improvement=average_improvement,
                last_24h_runs=last_24h_runs,
            )

        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @router.get("/brands/{brand_name}/can-tune")
    async def check_brand_can_tune(
        brand_name: str, session: Session = Depends(get_db_session)
    ):
        """Check if a brand is eligible for tuning (within rate limits)."""
        try:
            can_tune = service._check_rate_limit(session, brand_name)
            quarantined_count = service._get_quarantined_issues_count(
                session, brand_name
            )

            return {
                "can_tune": can_tune and quarantined_count >= 10,
                "within_rate_limit": can_tune,
                "quarantined_issues_count": quarantined_count,
                "minimum_required": 10,
                "message": (
                    "Ready for tuning"
                    if can_tune and quarantined_count >= 10
                    else "Rate limit exceeded"
                    if not can_tune
                    else f"Insufficient quarantined data ({quarantined_count}/10)"
                ),
            }

        except Exception as e:
            logger.error(f"Failed to check brand eligibility: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    return router


async def _execute_tuning_cycle_background(
    brand_name: str, triggered_by: str, force: bool
):
    """Execute tuning cycle in background task."""
    service = SelfTuningService()

    try:
        with get_db_session() as session:
            result = service.run_complete_tuning_cycle(
                session=session,
                brand_name=brand_name,
                triggered_by=triggered_by,
                force=force,
            )
            logger.info(f"Background tuning cycle completed: {result.tuning_run_id}")

    except Exception as e:
        logger.error(f"Background tuning cycle failed: {e}")


def mount_self_tuning_api(app, prefix: str = "/api/v1"):
    """Mount the self-tuning API on a FastAPI app."""
    router = create_self_tuning_api()
    app.include_router(router, prefix=prefix)
