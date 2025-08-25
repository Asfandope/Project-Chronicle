"""
Job Queue Manager for handling PDF processing job lifecycle.
Manages job queuing, priority scheduling, and resource allocation.
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set
from uuid import UUID
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, func
from sqlalchemy.orm import selectinload

from orchestrator.core.database import AsyncSessionLocal
from orchestrator.core.config import get_settings
from orchestrator.models.job import Job
from orchestrator.models.processing_state import ProcessingState
from orchestrator.core.workflow import WorkflowStatus, WorkflowStage
from orchestrator.tasks.ingestion import process_pdf_task
from orchestrator.utils.correlation import propagate_correlation_id

logger = structlog.get_logger(__name__)

class JobQueueManager:
    """
    Manages the job processing queue with priority scheduling and resource management.
    
    Features:
    - Priority-based job scheduling
    - Concurrent job processing limits
    - Failed job retry logic with exponential backoff
    - Job status monitoring and health checks
    - Resource allocation and load balancing
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._active_jobs: Set[str] = set()  # Track currently processing jobs
        self._retry_queue: List[Dict] = []  # Jobs waiting for retry
    
    async def start(self) -> None:
        """Start the job queue manager."""
        if self._running:
            logger.warning("Job queue manager already running")
            return
        
        self._running = True
        logger.info("Starting job queue manager")
        
        # Start the job scheduler
        self._scheduler_task = asyncio.create_task(self._run_scheduler())
        
        logger.info(
            "Job queue manager started",
            max_concurrent_jobs=self.settings.max_concurrent_jobs
        )
    
    async def stop(self) -> None:
        """Stop the job queue manager."""
        if not self._running:
            return
        
        logger.info("Stopping job queue manager")
        self._running = False
        
        # Cancel scheduler task
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        # Wait for active jobs to complete (with timeout)
        if self._active_jobs:
            logger.info(f"Waiting for {len(self._active_jobs)} active jobs to complete")
            timeout = 30  # 30 seconds timeout
            
            for _ in range(timeout):
                if not self._active_jobs:
                    break
                await asyncio.sleep(1)
            
            if self._active_jobs:
                logger.warning(
                    f"Timeout waiting for jobs to complete",
                    remaining_jobs=len(self._active_jobs)
                )
        
        logger.info("Job queue manager stopped")
    
    async def enqueue_job(
        self, 
        file_path: str, 
        filename: str,
        file_size: int,
        brand: Optional[str] = None,
        priority: int = 0,
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Enqueue a new PDF processing job.
        
        Args:
            file_path: Path to the PDF file
            filename: Original filename
            file_size: File size in bytes
            brand: Brand identifier for processing
            priority: Job priority (higher = more priority)
            correlation_id: Request correlation ID
            
        Returns:
            Job ID as string
        """
        async with AsyncSessionLocal() as session:
            # Create new job
            job = Job(
                filename=filename,
                file_path=file_path,
                file_size=file_size,
                brand=brand,
                priority=priority,
                overall_status=WorkflowStatus.PENDING,
                current_stage=WorkflowStage.INGESTION,
                max_retries=self.settings.max_retries
            )
            
            session.add(job)
            await session.commit()
            await session.refresh(job)
            
            job_id = str(job.id)
            
            logger.info(
                "Job enqueued",
                job_id=job_id,
                filename=filename,
                brand=brand,
                priority=priority,
                file_size=file_size,
                correlation_id=correlation_id
            )
            
            return job_id
    
    async def get_job_status(self, job_id: UUID) -> Optional[Dict]:
        """Get detailed job status information."""
        async with AsyncSessionLocal() as session:
            query = select(Job).options(
                selectinload(Job.processing_states)
            ).where(Job.id == job_id)
            
            result = await session.execute(query)
            job = result.scalar_one_or_none()
            
            if not job:
                return None
            
            # Build status response
            status = {
                "job_id": str(job.id),
                "filename": job.filename,
                "brand": job.brand,
                "overall_status": job.overall_status.value,
                "current_stage": job.current_stage.value,
                "accuracy_score": job.accuracy_score,
                "created_at": job.created_at.isoformat(),
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "processing_time_seconds": job.processing_time_seconds,
                "retry_count": job.retry_count,
                "error_message": job.error_message,
                "is_quarantined": job.is_quarantined,
                "processing_states": []
            }
            
            # Add processing state details
            for state in job.processing_states:
                status["processing_states"].append({
                    "stage": state.stage.value,
                    "started_at": state.started_at.isoformat(),
                    "completed_at": state.completed_at.isoformat() if state.completed_at else None,
                    "success": state.success,
                    "overall_confidence": state.overall_confidence,
                    "processing_time_ms": state.processing_time_ms,
                    "error_message": state.error_message
                })
            
            return status
    
    async def cancel_job(self, job_id: UUID) -> bool:
        """Cancel a pending or in-progress job."""
        async with AsyncSessionLocal() as session:
            query = select(Job).where(Job.id == job_id)
            result = await session.execute(query)
            job = result.scalar_one_or_none()
            
            if not job:
                return False
            
            # Can only cancel pending or in-progress jobs
            if job.overall_status not in [WorkflowStatus.PENDING, WorkflowStatus.IN_PROGRESS]:
                return False
            
            job.overall_status = WorkflowStatus.FAILED
            job.error_message = "Job cancelled by user"
            job.completed_at = datetime.now(timezone.utc)
            
            # Cancel Celery task if exists
            if job.celery_task_id:
                from orchestrator.celery_app import app as celery_app
                celery_app.control.revoke(job.celery_task_id, terminate=True)
            
            await session.commit()
            
            # Remove from active jobs
            job_id_str = str(job_id)
            if job_id_str in self._active_jobs:
                self._active_jobs.remove(job_id_str)
            
            logger.info("Job cancelled", job_id=job_id_str)
            return True
    
    async def retry_job(self, job_id: UUID) -> bool:
        """Retry a failed job if retries are available."""
        async with AsyncSessionLocal() as session:
            query = select(Job).where(Job.id == job_id)
            result = await session.execute(query)
            job = result.scalar_one_or_none()
            
            if not job or not job.can_retry:
                return False
            
            # Reset job state
            job.overall_status = WorkflowStatus.PENDING
            job.current_stage = WorkflowStage.INGESTION
            job.retry_count += 1
            job.error_message = None
            job.error_details = {}
            job.workflow_steps = {}
            job.started_at = None
            job.completed_at = None
            job.processing_time_seconds = None
            
            await session.commit()
            
            logger.info(
                "Job queued for retry",
                job_id=str(job_id),
                retry_count=job.retry_count
            )
            
            return True
    
    async def _run_scheduler(self) -> None:
        """Main scheduler loop that processes the job queue."""
        logger.info("Job scheduler started")
        
        while self._running:
            try:
                await self._process_job_queue()
                await self._handle_retry_queue()
                await self._cleanup_stale_jobs()
                
                # Wait before next scheduling cycle
                await asyncio.sleep(5)  # 5 second scheduling interval
                
            except Exception as e:
                logger.error(
                    "Error in job scheduler",
                    error=str(e),
                    exc_info=True
                )
                await asyncio.sleep(10)  # Wait longer on error
    
    async def _process_job_queue(self) -> None:
        """Process pending jobs from the queue."""
        if len(self._active_jobs) >= self.settings.max_concurrent_jobs:
            return  # At capacity
        
        available_slots = self.settings.max_concurrent_jobs - len(self._active_jobs)
        
        async with AsyncSessionLocal() as session:
            # Get pending jobs ordered by priority and creation time
            query = (
                select(Job)
                .where(Job.overall_status == WorkflowStatus.PENDING)
                .where(Job.scheduled_at.is_(None) | (Job.scheduled_at <= datetime.now(timezone.utc)))
                .order_by(Job.priority.desc(), Job.created_at.asc())
                .limit(available_slots)
            )
            
            result = await session.execute(query)
            jobs = result.scalars().all()
            
            for job in jobs:
                await self._start_job_processing(job, session)
    
    async def _start_job_processing(self, job: Job, session: AsyncSession) -> None:
        """Start processing a job."""
        job_id = str(job.id)
        
        # Mark job as in progress
        job.overall_status = WorkflowStatus.IN_PROGRESS
        job.started_at = datetime.now(timezone.utc)
        
        # Add to active jobs
        self._active_jobs.add(job_id)
        
        try:
            # Start Celery task
            correlation_id = f"job-{job_id}"
            headers = propagate_correlation_id(correlation_id)
            
            task = process_pdf_task.apply_async(
                args=[job_id],
                headers=headers
            )
            job.celery_task_id = task.id
            
            await session.commit()
            
            logger.info(
                "Started job processing",
                job_id=job_id,
                celery_task_id=task.id,
                filename=job.filename,
                brand=job.brand
            )
            
        except Exception as e:
            # Remove from active jobs on error
            self._active_jobs.discard(job_id)
            
            # Mark job as failed
            job.overall_status = WorkflowStatus.FAILED
            job.error_message = f"Failed to start processing: {str(e)}"
            job.completed_at = datetime.now(timezone.utc)
            
            await session.commit()
            
            logger.error(
                "Failed to start job processing",
                job_id=job_id,
                error=str(e),
                exc_info=True
            )
    
    async def _handle_retry_queue(self) -> None:
        """Handle jobs in the retry queue."""
        if not self._retry_queue:
            return
        
        now = datetime.now(timezone.utc)
        ready_jobs = []
        
        # Find jobs ready for retry
        for retry_info in self._retry_queue[:]:
            if retry_info["retry_at"] <= now:
                ready_jobs.append(retry_info)
                self._retry_queue.remove(retry_info)
        
        # Process ready retry jobs
        for retry_info in ready_jobs:
            await self.retry_job(UUID(retry_info["job_id"]))
    
    async def _cleanup_stale_jobs(self) -> None:
        """Clean up stale job states and remove completed jobs from active set."""
        if not self._active_jobs:
            return
        
        async with AsyncSessionLocal() as session:
            # Get status of active jobs
            job_ids = [UUID(job_id) for job_id in self._active_jobs]
            query = select(Job).where(Job.id.in_(job_ids))
            result = await session.execute(query)
            jobs = {str(job.id): job for job in result.scalars().all()}
            
            # Remove completed jobs from active set
            completed_jobs = []
            for job_id in list(self._active_jobs):
                job = jobs.get(job_id)
                if job and job.is_completed:
                    completed_jobs.append(job_id)
                    self._active_jobs.remove(job_id)
            
            if completed_jobs:
                logger.info(
                    "Cleaned up completed jobs",
                    completed_count=len(completed_jobs)
                )
    
    async def get_queue_stats(self) -> Dict:
        """Get queue statistics."""
        async with AsyncSessionLocal() as session:
            # Count jobs by status
            stats_query = select(
                Job.overall_status,
                func.count(Job.id).label('count')
            ).group_by(Job.overall_status)
            
            result = await session.execute(stats_query)
            status_counts = {row.overall_status.value: row.count for row in result}
            
            return {
                "active_jobs": len(self._active_jobs),
                "max_concurrent_jobs": self.settings.max_concurrent_jobs,
                "retry_queue_size": len(self._retry_queue),
                "status_counts": status_counts,
                "queue_utilization": len(self._active_jobs) / self.settings.max_concurrent_jobs
            }