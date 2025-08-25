"""
Celery application configuration for async task processing.
Handles PDF processing workflow with proper error handling and retry logic.
"""

from celery import Celery
from celery.signals import task_prerun, task_postrun, task_failure, task_retry, worker_ready
import structlog

from orchestrator.core.config import get_settings
from orchestrator.core.logging import configure_logging

# Configure logging
configure_logging()
logger = structlog.get_logger(__name__)

settings = get_settings()

# Create Celery application
celery_app = Celery(
    "orchestrator",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["orchestrator.tasks.ingestion", "orchestrator.tasks.workflow_executor"]
)

# Configure Celery
celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    result_accept_content=["json"],
    
    # Timezone settings
    timezone="UTC",
    enable_utc=True,
    
    # Task execution settings
    task_track_started=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    
    # Timeout settings (from configuration)
    task_time_limit=settings.processing_timeout_minutes * 60,
    task_soft_time_limit=(settings.processing_timeout_minutes - 2) * 60,
    
    # Worker settings
    worker_prefetch_multiplier=1,  # Process one task at a time for better resource management
    worker_max_tasks_per_child=50,  # Restart workers after 50 tasks to prevent memory leaks
    worker_concurrency=1,  # Single process worker for consistent resource usage
    
    # Result settings
    result_expires=3600 * 24,  # Keep results for 24 hours
    result_persistent=True,
    
    # Retry settings
    task_default_retry_delay=60,  # 1 minute default retry delay
    task_max_retries=settings.max_retries,
    
    # Route settings for different task types
    task_routes={
        'orchestrator.tasks.ingestion.*': {'queue': 'ingestion'},
        'orchestrator.tasks.workflow_executor.*': {'queue': 'processing'},
        'orchestrator.tasks.monitoring.*': {'queue': 'monitoring'},
    },
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # Security
    task_always_eager=False,  # Process tasks asynchronously
    task_store_eager_result=False,
)

# Configure task error handling
@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds):
    """Handler called before task execution."""
    logger.info(
        "Task started",
        task_id=task_id,
        task_name=task.name,
        args=args[:2] if args else [],  # Log first 2 args only for brevity
    )

@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, 
                        retval=None, state=None, **kwds):
    """Handler called after task execution."""
    logger.info(
        "Task completed",
        task_id=task_id,
        task_name=task.name,
        state=state,
        success=state == "SUCCESS"
    )

@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwds):
    """Handler called when task fails."""
    logger.error(
        "Task failed",
        task_id=task_id,
        task_name=sender.name if sender else "unknown",
        error=str(exception),
        exc_info=True
    )

@task_retry.connect
def task_retry_handler(sender=None, task_id=None, reason=None, einfo=None, **kwds):
    """Handler called when task is retried."""
    logger.warning(
        "Task retry",
        task_id=task_id,
        task_name=sender.name if sender else "unknown",
        reason=str(reason)
    )

@worker_ready.connect
def worker_ready_handler(sender=None, **kwds):
    """Handler called when worker is ready."""
    logger.info(
        "Celery worker ready",
        worker_name=sender.hostname if sender else "unknown"
    )

# Alias for import convenience
app = celery_app