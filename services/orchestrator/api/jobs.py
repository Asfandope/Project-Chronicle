from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, Request, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List, Optional
from uuid import UUID
import structlog
import os
import tempfile
from pathlib import Path
from datetime import datetime

from orchestrator.core.database import get_db
from orchestrator.models.job import Job
from orchestrator.core.workflow import WorkflowStatus
from orchestrator.core.config import get_settings
from shared.schemas.job import JobResponse, JobCreate, JobListResponse
from orchestrator.utils.correlation import get_correlation_id
from orchestrator.utils.file_utils import extract_brand_from_filename, validate_pdf_file
from orchestrator.core.logging import log_job_event

logger = structlog.get_logger()
router = APIRouter()

@router.post("/process", response_model=JobResponse)
async def process_pdf(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    brand: Optional[str] = Query(None, description="Brand identifier for processing configuration"),
    priority: int = Query(0, description="Job priority (higher = more priority)", ge=0, le=10),
    db: AsyncSession = Depends(get_db)
):
    """
    Queue a PDF for processing.
    
    This endpoint accepts PDF file uploads and queues them for extraction processing.
    Files are validated, stored securely, and processed asynchronously.
    
    Parameters:
    - file: PDF file to process (required)
    - brand: Brand identifier for processing configuration (optional, auto-detected if not provided)
    - priority: Job priority from 0-10, higher numbers get processed first (default: 0)
    
    Returns:
    - Job information with tracking ID and status
    """
    settings = get_settings()
    correlation_id = get_correlation_id(request)
    
    # Validate file type
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400, 
            detail="Only PDF files are supported"
        )
    
    # Read file content
    try:
        content = await file.read()
        file_size = len(content)
    except Exception as e:
        logger.error("Failed to read uploaded file", error=str(e), filename=file.filename)
        raise HTTPException(status_code=400, detail="Failed to read uploaded file")
    
    # Validate file size
    max_size = settings.max_file_size_mb * 1024 * 1024
    if file_size == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if file_size > max_size:
        raise HTTPException(
            status_code=413, 
            detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB"
        )
    
    # Create secure temporary file for validation
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
        temp_file.write(content)
        temp_path = temp_file.name
    
    try:
        # Validate PDF file
        validation_result = await validate_pdf_file(temp_path)
        if not validation_result["valid"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid PDF file: {validation_result['reason']}"
            )
        
        # Extract brand if not provided
        if not brand:
            brand = extract_brand_from_filename(file.filename)
        
        # Create final storage path
        input_dir = Path(settings.input_directory)
        input_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename}"
        final_path = input_dir / safe_filename
        
        # Move file to final location
        Path(temp_path).rename(final_path)
        
        # Get job queue manager from app state
        job_queue_manager = request.app.state.services.get("job_queue")
        if not job_queue_manager:
            raise HTTPException(status_code=503, detail="Job queue service unavailable")
        
        # Enqueue job
        job_id = await job_queue_manager.enqueue_job(
            file_path=str(final_path),
            filename=file.filename,
            file_size=file_size,
            brand=brand,
            priority=priority,
            correlation_id=correlation_id
        )
        
        # Get job details for response
        job_status = await job_queue_manager.get_job_status(UUID(job_id))
        
        log_job_event(
            logger,
            "Job created via API",
            job_id=job_id,
            filename=file.filename,
            brand=brand,
            file_size=file_size,
            priority=priority,
            correlation_id=correlation_id
        )
        
        return {
            "job_id": job_id,
            "filename": file.filename,
            "brand": brand,
            "status": "pending",
            "file_size": file_size,
            "priority": priority,
            "created_at": job_status["created_at"],
            "message": "PDF queued for processing"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to process PDF upload",
            error=str(e),
            filename=file.filename,
            correlation_id=correlation_id,
            exc_info=True
        )
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        # Clean up temp file if it still exists
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@router.get("/", response_model=JobListResponse)
async def list_jobs(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[WorkflowStatus] = None,
    brand: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """List jobs with filtering and pagination"""
    query = select(Job)
    
    if status:
        query = query.where(Job.overall_status == status)
    if brand:
        query = query.where(Job.brand == brand)
    
    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query)
    
    # Get paginated results
    query = query.offset(skip).limit(limit).order_by(Job.created_at.desc())
    result = await db.execute(query)
    jobs = result.scalars().all()
    
    return JobListResponse(
        jobs=[JobResponse.from_orm(job) for job in jobs],
        total=total,
        skip=skip,
        limit=limit
    )

@router.get("/status/{job_id}")
async def get_job_status(
    request: Request,
    job_id: UUID
):
    """
    Check job status and processing progress.
    
    Returns detailed status information including:
    - Overall job status and current processing stage
    - Processing timestamps and duration
    - Accuracy scores and confidence metrics
    - Error information if applicable
    - Detailed processing state history
    
    Parameters:
    - job_id: UUID of the job to check
    
    Returns:
    - Comprehensive job status information
    """
    correlation_id = get_correlation_id(request)
    
    # Get job queue manager from app state
    job_queue_manager = request.app.state.services.get("job_queue")
    if not job_queue_manager:
        raise HTTPException(status_code=503, detail="Job queue service unavailable")
    
    # Get detailed job status
    job_status = await job_queue_manager.get_job_status(job_id)
    if not job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Calculate additional metrics
    if job_status["started_at"] and job_status["completed_at"]:
        start_time = datetime.fromisoformat(job_status["started_at"].replace('Z', '+00:00'))
        end_time = datetime.fromisoformat(job_status["completed_at"].replace('Z', '+00:00'))
        total_duration = (end_time - start_time).total_seconds()
        job_status["total_duration_seconds"] = total_duration
    
    # Add progress percentage
    stage_progress = {
        "INGESTION": 10,
        "LAYOUT_ANALYSIS": 30,
        "OCR": 60,
        "ARTICLE_RECONSTRUCTION": 80,
        "VALIDATION": 95,
        "EXPORT": 100
    }
    
    current_stage = job_status["current_stage"]
    if job_status["overall_status"] == "COMPLETED":
        job_status["progress_percentage"] = 100
    elif job_status["overall_status"] == "FAILED":
        job_status["progress_percentage"] = stage_progress.get(current_stage, 0)
    else:
        job_status["progress_percentage"] = stage_progress.get(current_stage, 0)
    
    logger.info(
        "Job status requested",
        job_id=str(job_id),
        status=job_status["overall_status"],
        correlation_id=correlation_id
    )
    
    return job_status

@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get job details by ID (legacy endpoint)"""
    query = select(Job).where(Job.id == job_id)
    result = await db.execute(query)
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobResponse.from_orm(job)

@router.post("/{job_id}/retry")
async def retry_job(
    job_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Retry a failed job"""
    query = select(Job).where(Job.id == job_id)
    result = await db.execute(query)
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.overall_status not in [WorkflowStatus.FAILED, WorkflowStatus.COMPLETED]:
        raise HTTPException(
            status_code=400, 
            detail="Can only retry failed or quarantined jobs"
        )
    
    # Reset job state
    job.overall_status = WorkflowStatus.PENDING
    job.retry_count += 1
    job.error_message = None
    job.workflow_steps = {}
    
    # Start new processing task
    task = process_pdf_task.delay(str(job.id))
    job.celery_task_id = task.id
    
    await db.commit()
    
    logger.info("Retrying job", job_id=str(job_id), retry_count=job.retry_count)
    
    return {"message": "Job retry initiated", "task_id": task.id}

@router.delete("/{job_id}")
async def delete_job(
    job_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Delete a job and its associated data"""
    query = select(Job).where(Job.id == job_id)
    result = await db.execute(query)
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # TODO: Clean up associated files and data
    
    await db.delete(job)
    await db.commit()
    
    logger.info("Deleted job", job_id=str(job_id))
    
    return {"message": "Job deleted successfully"}