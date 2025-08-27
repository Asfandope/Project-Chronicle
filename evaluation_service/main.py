"""
FastAPI application for the evaluation service.

This module provides REST API endpoints for:
- Manual evaluation uploads
- Batch evaluations
- Drift detection monitoring
- Auto-tuning triggers
- System health monitoring
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, desc, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from .drift_detector import create_drift_detector
from .evaluation_service import EvaluationService
from .models import (
    AutoTuningEvent,
    DocumentEvaluation,
    DriftDetection,
    EvaluationRun,
)
from .schemas import (
    AccuracyTrendResponse,
    AutoTuningEventResponse,
    BatchEvaluationRequest,
    DocumentEvaluationResponse,
    DriftDetectionResponse,
    ErrorResponse,
    EvaluationRunResponse,
    EvaluationType,
    HealthCheckResponse,
    ManualEvaluationRequest,
    MetricType,
    PaginatedResponse,
    PaginationParams,
    SystemHealthResponse,
    TriggerSource,
    XMLValidationRequest,
    XMLValidationResponse,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/magazine_evaluation")

# Create engine and session factory
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Tables are created by quick_start.py during initialization
# create_tables(engine)

# Initialize services
evaluation_service = EvaluationService()
drift_detector = create_drift_detector()

# FastAPI app
app = FastAPI(
    title="Magazine Extraction Evaluation Service",
    description="Service for evaluating magazine extraction accuracy against ground truth",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency to get database session
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=exc.detail, timestamp=datetime.utcnow()).dict(),
    )


@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_exception_handler(request, exc: SQLAlchemyError):
    logger.error(f"Database error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Database error occurred", detail="Please try again later"
        ).dict(),
    )


# Health check endpoint
@app.get("/health", response_model=HealthCheckResponse)
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint."""

    database_connected = True
    try:
        # Test database connection
        db.execute(text("SELECT 1"))
    except Exception:
        database_connected = False

    return HealthCheckResponse(
        status="healthy" if database_connected else "unhealthy",
        version="1.0.0",
        database_connected=database_connected,
    )


# Evaluation endpoints
@app.post("/evaluate/single", response_model=DocumentEvaluationResponse)
async def evaluate_single_document(
    request: ManualEvaluationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Evaluate a single document against its ground truth."""

    try:
        # Validate XML formats
        gt_valid, gt_errors = evaluation_service.validate_xml_format(
            request.ground_truth_content, "ground_truth"
        )
        if not gt_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid ground truth XML: {'; '.join(gt_errors)}",
            )

        ex_valid, ex_errors = evaluation_service.validate_xml_format(
            request.extracted_content, "extracted"
        )
        if not ex_valid:
            raise HTTPException(
                status_code=400, detail=f"Invalid extracted XML: {'; '.join(ex_errors)}"
            )

        # Create evaluation run for single document
        evaluation_run = EvaluationRun(
            evaluation_type=EvaluationType.MANUAL.value,
            trigger_source=TriggerSource.MANUAL_UPLOAD.value,
            document_count=1,
            extractor_version=request.extractor_version,
            model_version=request.model_version,
        )

        db.add(evaluation_run)
        db.flush()

        # Perform evaluation
        doc_evaluation = evaluation_service.evaluate_single_document(
            db, request, evaluation_run.id
        )

        # Update evaluation run metrics
        evaluation_run.total_articles = len(doc_evaluation.article_evaluations)
        evaluation_run.successful_extractions = (
            1 if doc_evaluation.extraction_successful else 0
        )
        evaluation_run.failed_extractions = (
            0 if doc_evaluation.extraction_successful else 1
        )
        evaluation_run.overall_weighted_accuracy = (
            doc_evaluation.weighted_overall_accuracy
        )
        evaluation_run.title_accuracy = doc_evaluation.title_accuracy
        evaluation_run.body_text_accuracy = doc_evaluation.body_text_accuracy
        evaluation_run.contributors_accuracy = doc_evaluation.contributors_accuracy
        evaluation_run.media_links_accuracy = doc_evaluation.media_links_accuracy

        db.commit()

        # Run drift detection in background
        background_tasks.add_task(run_drift_detection, evaluation_run.id)

        return DocumentEvaluationResponse.from_orm(doc_evaluation)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error evaluating single document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.post("/evaluate/batch", response_model=EvaluationRunResponse)
async def evaluate_batch(
    request: BatchEvaluationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Evaluate a batch of documents."""

    try:
        # Validate all documents first
        for i, doc_request in enumerate(request.documents):
            gt_valid, gt_errors = evaluation_service.validate_xml_format(
                doc_request.ground_truth_content, "ground_truth"
            )
            if not gt_valid:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid ground truth XML for document {i+1}: {'; '.join(gt_errors)}",
                )

            ex_valid, ex_errors = evaluation_service.validate_xml_format(
                doc_request.extracted_content, "extracted"
            )
            if not ex_valid:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid extracted XML for document {i+1}: {'; '.join(ex_errors)}",
                )

        # Perform batch evaluation
        evaluation_run = evaluation_service.evaluate_batch(
            db, request.documents, EvaluationType.BATCH, TriggerSource.API_REQUEST
        )

        # Run drift detection in background if enabled
        if request.enable_drift_detection:
            background_tasks.add_task(run_drift_detection, evaluation_run.id)

        return EvaluationRunResponse.from_orm(evaluation_run)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error evaluating batch: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Batch evaluation failed: {str(e)}"
        )


# Drift detection endpoints
@app.post("/drift/detect", response_model=List[DriftDetectionResponse])
async def trigger_drift_detection(
    evaluation_run_id: str,
    metric_types: Optional[List[MetricType]] = None,
    db: Session = Depends(get_db),
):
    """Manually trigger drift detection for an evaluation run."""

    try:
        # Get evaluation run
        evaluation_run = (
            db.query(EvaluationRun)
            .filter(EvaluationRun.id == evaluation_run_id)
            .first()
        )

        if not evaluation_run:
            raise HTTPException(
                status_code=404, detail=f"Evaluation run {evaluation_run_id} not found"
            )

        # Convert metric types to strings
        metric_type_strings = None
        if metric_types:
            metric_type_strings = [mt.value for mt in metric_types]

        # Run drift detection
        drift_results = drift_detector.detect_drift(
            db, evaluation_run, metric_type_strings
        )

        # Convert to response format
        responses = []
        for result in drift_results:
            # Find corresponding drift detection record
            drift_detection = (
                db.query(DriftDetection)
                .filter(
                    DriftDetection.evaluation_run_id == evaluation_run.id,
                    DriftDetection.metric_type == result.metric_type,
                )
                .order_by(DriftDetection.created_at.desc())
                .first()
            )

            if drift_detection:
                responses.append(DriftDetectionResponse.from_orm(drift_detection))

        return responses

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering drift detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Drift detection failed: {str(e)}")


@app.get("/drift/status/{metric_type}", response_model=Dict[str, Any])
async def get_drift_status(metric_type: MetricType, db: Session = Depends(get_db)):
    """Get current drift status for a specific metric."""

    try:
        status = drift_detector.get_drift_status(db, metric_type.value)

        # Get latest drift detection
        latest_drift = (
            db.query(DriftDetection)
            .filter(DriftDetection.metric_type == metric_type.value)
            .order_by(DriftDetection.created_at.desc())
            .first()
        )

        response = {
            "metric_type": metric_type.value,
            "status": status.value,
            "last_checked": latest_drift.created_at if latest_drift else None,
            "current_accuracy": latest_drift.current_accuracy if latest_drift else None,
            "baseline_accuracy": latest_drift.baseline_accuracy
            if latest_drift
            else None,
            "accuracy_drop": latest_drift.accuracy_drop if latest_drift else None,
        }

        return response

    except Exception as e:
        logger.error(f"Error getting drift status: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get drift status: {str(e)}"
        )


@app.get("/drift/summary", response_model=Dict[str, Any])
async def get_drift_summary(
    days: int = Query(7, ge=1, le=90), db: Session = Depends(get_db)
):
    """Get summary of drift detections over specified period."""

    try:
        summary = drift_detector.get_drift_summary(db, days)
        return summary

    except Exception as e:
        logger.error(f"Error getting drift summary: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get drift summary: {str(e)}"
        )


# Auto-tuning endpoints
@app.get("/autotuning/events", response_model=PaginatedResponse)
async def get_autotuning_events(
    pagination: PaginationParams = Depends(),
    status_filter: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Get auto-tuning events with pagination."""

    try:
        query = db.query(AutoTuningEvent)

        if status_filter:
            query = query.filter(AutoTuningEvent.status == status_filter)

        # Get total count
        total_count = query.count()

        # Apply pagination
        offset = (pagination.page - 1) * pagination.page_size
        events = (
            query.order_by(desc(AutoTuningEvent.created_at))
            .offset(offset)
            .limit(pagination.page_size)
            .all()
        )

        # Convert to response format
        event_responses = [AutoTuningEventResponse.from_orm(event) for event in events]

        return PaginatedResponse(
            items=event_responses,
            total_count=total_count,
            page=pagination.page,
            page_size=pagination.page_size,
        )

    except Exception as e:
        logger.error(f"Error getting auto-tuning events: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get auto-tuning events: {str(e)}"
        )


@app.get("/autotuning/events/{event_id}", response_model=AutoTuningEventResponse)
async def get_autotuning_event(event_id: str, db: Session = Depends(get_db)):
    """Get specific auto-tuning event details."""

    try:
        event = db.query(AutoTuningEvent).filter(AutoTuningEvent.id == event_id).first()

        if not event:
            raise HTTPException(
                status_code=404, detail=f"Auto-tuning event {event_id} not found"
            )

        return AutoTuningEventResponse.from_orm(event)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting auto-tuning event: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get auto-tuning event: {str(e)}"
        )


# Evaluation history endpoints
@app.get("/evaluations", response_model=PaginatedResponse)
async def get_evaluations(
    pagination: PaginationParams = Depends(),
    evaluation_type: Optional[EvaluationType] = None,
    brand_filter: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Get evaluation runs with pagination and filtering."""

    try:
        query = db.query(EvaluationRun)

        if evaluation_type:
            query = query.filter(EvaluationRun.evaluation_type == evaluation_type.value)

        # Get total count
        total_count = query.count()

        # Apply pagination
        offset = (pagination.page - 1) * pagination.page_size

        if pagination.sort_by == "created_at":
            if pagination.sort_order == "desc":
                query = query.order_by(desc(EvaluationRun.created_at))
            else:
                query = query.order_by(EvaluationRun.created_at)

        evaluations = query.offset(offset).limit(pagination.page_size).all()

        # Convert to response format
        evaluation_responses = [
            EvaluationRunResponse.from_orm(eval_run) for eval_run in evaluations
        ]

        return PaginatedResponse(
            items=evaluation_responses,
            total_count=total_count,
            page=pagination.page,
            page_size=pagination.page_size,
        )

    except Exception as e:
        logger.error(f"Error getting evaluations: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get evaluations: {str(e)}"
        )


@app.get("/evaluations/{evaluation_id}", response_model=EvaluationRunResponse)
async def get_evaluation(evaluation_id: str, db: Session = Depends(get_db)):
    """Get specific evaluation run details."""

    try:
        evaluation = (
            db.query(EvaluationRun).filter(EvaluationRun.id == evaluation_id).first()
        )

        if not evaluation:
            raise HTTPException(
                status_code=404, detail=f"Evaluation run {evaluation_id} not found"
            )

        return EvaluationRunResponse.from_orm(evaluation)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting evaluation: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get evaluation: {str(e)}"
        )


@app.get("/evaluations/{evaluation_id}/documents", response_model=PaginatedResponse)
async def get_evaluation_documents(
    evaluation_id: str,
    pagination: PaginationParams = Depends(),
    db: Session = Depends(get_db),
):
    """Get document evaluations for a specific evaluation run."""

    try:
        query = db.query(DocumentEvaluation).filter(
            DocumentEvaluation.evaluation_run_id == evaluation_id
        )

        total_count = query.count()

        # Apply pagination
        offset = (pagination.page - 1) * pagination.page_size
        documents = (
            query.order_by(desc(DocumentEvaluation.created_at))
            .offset(offset)
            .limit(pagination.page_size)
            .all()
        )

        # Convert to response format
        document_responses = [
            DocumentEvaluationResponse.from_orm(doc) for doc in documents
        ]

        return PaginatedResponse(
            items=document_responses,
            total_count=total_count,
            page=pagination.page,
            page_size=pagination.page_size,
        )

    except Exception as e:
        logger.error(f"Error getting evaluation documents: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get evaluation documents: {str(e)}"
        )


# Analytics and reporting endpoints
@app.get("/analytics/trends", response_model=AccuracyTrendResponse)
async def get_accuracy_trends(
    metric_type: MetricType = MetricType.OVERALL,
    days: int = Query(30, ge=7, le=90),
    brand_filter: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Get accuracy trends over time."""

    try:
        # Get historical data
        from .models import get_accuracy_history

        history = get_accuracy_history(db, metric_type.value, days, brand_filter)

        if not history:
            return AccuracyTrendResponse(
                metric_type=metric_type,
                time_period_days=days,
                data_points=[],
                current_accuracy=0.0,
                average_accuracy=0.0,
            )

        # Calculate statistics
        accuracies = [point["accuracy"] for point in history]
        current_accuracy = accuracies[-1] if accuracies else 0.0
        average_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0

        # Calculate trend
        trend_direction = "stable"
        if len(accuracies) >= 2:
            recent_avg = sum(accuracies[-7:]) / min(7, len(accuracies))
            earlier_avg = (
                sum(accuracies[:-7]) / max(1, len(accuracies) - 7)
                if len(accuracies) > 7
                else average_accuracy
            )

            if recent_avg > earlier_avg + 0.01:
                trend_direction = "improving"
            elif recent_avg < earlier_avg - 0.01:
                trend_direction = "declining"

        return AccuracyTrendResponse(
            metric_type=metric_type,
            time_period_days=days,
            data_points=history,
            current_accuracy=current_accuracy,
            average_accuracy=average_accuracy,
            trend_direction=trend_direction,
            min_accuracy=min(accuracies) if accuracies else 0.0,
            max_accuracy=max(accuracies) if accuracies else 0.0,
        )

    except Exception as e:
        logger.error(f"Error getting accuracy trends: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get accuracy trends: {str(e)}"
        )


# System health endpoint
@app.get("/system/health", response_model=SystemHealthResponse)
async def get_system_health(
    period_hours: int = Query(24, ge=1, le=168), db: Session = Depends(get_db)
):
    """Get system health metrics for specified period."""

    try:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=period_hours)

        # Query document evaluations in period
        doc_evaluations = (
            db.query(DocumentEvaluation)
            .filter(
                DocumentEvaluation.created_at >= start_time,
                DocumentEvaluation.created_at <= end_time,
            )
            .all()
        )

        if not doc_evaluations:
            return SystemHealthResponse(
                recorded_at=datetime.now(timezone.utc),
                period_start=start_time,
                period_end=end_time,
                documents_processed=0,
                articles_processed=0,
            )

        # Calculate metrics
        documents_processed = len(doc_evaluations)
        successful_extractions = sum(
            1 for de in doc_evaluations if de.extraction_successful
        )
        articles_processed = sum(len(de.article_evaluations) for de in doc_evaluations)

        success_rate = (
            successful_extractions / documents_processed
            if documents_processed > 0
            else 0.0
        )

        # Calculate average accuracies
        successful_docs = [de for de in doc_evaluations if de.extraction_successful]
        if successful_docs:
            avg_overall = sum(
                de.weighted_overall_accuracy for de in successful_docs
            ) / len(successful_docs)
            avg_title = sum(de.title_accuracy for de in successful_docs) / len(
                successful_docs
            )
            avg_body = sum(de.body_text_accuracy for de in successful_docs) / len(
                successful_docs
            )
            avg_contributors = sum(
                de.contributors_accuracy for de in successful_docs
            ) / len(successful_docs)
            avg_media = sum(de.media_links_accuracy for de in successful_docs) / len(
                successful_docs
            )
        else:
            avg_overall = avg_title = avg_body = avg_contributors = avg_media = 0.0

        # Get drift alerts count
        drift_alerts = (
            db.query(DriftDetection)
            .filter(
                DriftDetection.created_at >= start_time,
                DriftDetection.created_at <= end_time,
                DriftDetection.drift_detected == True,
            )
            .count()
        )

        # Get auto-tuning events count
        auto_tuning_events = (
            db.query(AutoTuningEvent)
            .filter(
                AutoTuningEvent.created_at >= start_time,
                AutoTuningEvent.created_at <= end_time,
            )
            .count()
        )

        return SystemHealthResponse(
            recorded_at=datetime.now(timezone.utc),
            period_start=start_time,
            period_end=end_time,
            documents_processed=documents_processed,
            articles_processed=articles_processed,
            average_overall_accuracy=avg_overall,
            average_title_accuracy=avg_title,
            average_body_text_accuracy=avg_body,
            average_contributors_accuracy=avg_contributors,
            average_media_links_accuracy=avg_media,
            extraction_success_rate=success_rate,
            drift_alerts_count=drift_alerts,
            auto_tuning_events_count=auto_tuning_events,
        )

    except Exception as e:
        logger.error(f"Error getting system health: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get system health: {str(e)}"
        )


# Utility endpoints
@app.post("/validate/xml", response_model=XMLValidationResponse)
async def validate_xml(request: XMLValidationRequest):
    """Validate XML format and structure."""

    try:
        is_valid, errors = evaluation_service.validate_xml_format(
            request.xml_content, request.xml_type
        )

        # Count elements if valid
        element_count = None
        article_count = None

        if is_valid:
            try:
                import xml.etree.ElementTree as ET

                root = ET.fromstring(request.xml_content)

                if request.xml_type == "ground_truth":
                    elements = root.findall(".//*")
                    element_count = len(elements)
                    article_count = len(root.findall(".//article"))
                else:
                    articles = root.findall(".//article")
                    article_count = len(articles)
                    element_count = len(root.findall(".//*"))
            except Exception:
                pass

        return XMLValidationResponse(
            is_valid=is_valid,
            validation_errors=errors,
            element_count=element_count,
            article_count=article_count,
        )

    except Exception as e:
        logger.error(f"Error validating XML: {str(e)}")
        raise HTTPException(status_code=500, detail=f"XML validation failed: {str(e)}")


# Background task for drift detection
async def run_drift_detection(evaluation_run_id: str):
    """Background task to run drift detection after evaluation."""

    try:
        db = SessionLocal()

        evaluation_run = (
            db.query(EvaluationRun)
            .filter(EvaluationRun.id == evaluation_run_id)
            .first()
        )

        if evaluation_run:
            drift_detector.detect_drift(db, evaluation_run)
            logger.info(
                f"Completed drift detection for evaluation run {evaluation_run_id}"
            )

        db.close()

    except Exception as e:
        logger.error(f"Error in background drift detection: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "evaluation_service.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
