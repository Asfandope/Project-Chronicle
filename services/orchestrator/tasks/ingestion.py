from celery import shared_task
import structlog
from uuid import UUID

logger = structlog.get_logger()

@shared_task(bind=True, max_retries=3)
def process_pdf_task(self, job_id: str):
    """Main PDF processing task"""
    from orchestrator.core.workflow import WorkflowEngine
    from orchestrator.tasks.workflow_executor import WorkflowExecutor
    
    logger.info("Starting PDF processing", job_id=job_id)
    
    try:
        executor = WorkflowExecutor(UUID(job_id))
        result = executor.execute_workflow()
        
        logger.info("PDF processing completed", job_id=job_id, result=result)
        return result
        
    except Exception as exc:
        logger.error("PDF processing failed", job_id=job_id, error=str(exc))
        raise self.retry(exc=exc, countdown=60, max_retries=3)