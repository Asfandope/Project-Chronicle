from datetime import datetime
from typing import Any, Dict
from uuid import UUID

import httpx
import structlog
from orchestrator.core.config import get_settings
from orchestrator.core.database import AsyncSessionLocal
from orchestrator.core.workflow import (
    WorkflowEngine,
    WorkflowStage,
    WorkflowStatus,
    WorkflowStep,
)
from orchestrator.models.job import Job

logger = structlog.get_logger()


class WorkflowExecutor:
    """Executes the complete PDF processing workflow"""

    def __init__(self, job_id: UUID):
        self.job_id = job_id
        self.settings = get_settings()
        self.workflow_engine = WorkflowEngine()
        self.logger = logger.bind(job_id=str(job_id))

    async def execute_workflow(self) -> Dict[str, Any]:
        """Execute the complete workflow for a job"""
        async with AsyncSessionLocal() as db:
            # Load job
            job = await db.get(Job, self.job_id)
            if not job:
                raise ValueError(f"Job {self.job_id} not found")

            job.started_at = datetime.utcnow()
            job.overall_status = WorkflowStatus.IN_PROGRESS
            await db.commit()

            try:
                # Execute workflow stages
                current_steps = self._load_workflow_steps(job.workflow_steps)

                while not self.workflow_engine.is_workflow_complete(current_steps):
                    if self.workflow_engine.is_workflow_failed(current_steps):
                        job.overall_status = WorkflowStatus.FAILED
                        job.error_message = "Workflow failed - too many retry attempts"
                        await db.commit()
                        return {"status": "failed", "reason": "max_retries_exceeded"}

                    # Get next stages to execute
                    ready_stages = self.workflow_engine.get_next_stages(current_steps)

                    if not ready_stages:
                        self.logger.warning(
                            "No ready stages found, workflow may be stuck"
                        )
                        break

                    # Execute ready stages
                    for stage in ready_stages:
                        await self._execute_stage(stage, current_steps, job, db)

                # Check final status
                if self.workflow_engine.is_workflow_complete(current_steps):
                    job.overall_status = WorkflowStatus.COMPLETED
                    job.completed_at = datetime.utcnow()

                    # Calculate processing time
                    if job.started_at:
                        processing_time = (
                            job.completed_at - job.started_at
                        ).total_seconds()
                        job.processing_time_seconds = int(processing_time)

                # Update job with final workflow steps
                job.workflow_steps = self._serialize_workflow_steps(current_steps)
                await db.commit()

                return {
                    "status": job.overall_status,
                    "workflow_steps": current_steps,
                    "accuracy": job.accuracy_score,
                }

            except Exception as e:
                self.logger.error("Workflow execution failed", error=str(e))
                job.overall_status = WorkflowStatus.FAILED
                job.error_message = str(e)
                await db.commit()
                raise

    def _load_workflow_steps(
        self, workflow_steps_data: Dict
    ) -> Dict[WorkflowStage, WorkflowStep]:
        """Load workflow steps from database JSON"""
        steps = {}
        for stage_name, step_data in workflow_steps_data.items():
            stage = WorkflowStage(stage_name)
            steps[stage] = WorkflowStep(
                stage=stage,
                status=WorkflowStatus(step_data.get("status", WorkflowStatus.PENDING)),
                task_id=step_data.get("task_id"),
                error_message=step_data.get("error_message"),
                started_at=step_data.get("started_at"),
                completed_at=step_data.get("completed_at"),
                attempts=step_data.get("attempts", 0),
                max_attempts=step_data.get("max_attempts", 3),
            )
        return steps

    def _serialize_workflow_steps(
        self, steps: Dict[WorkflowStage, WorkflowStep]
    ) -> Dict:
        """Serialize workflow steps for database storage"""
        result = {}
        for stage, step in steps.items():
            result[stage.value] = {
                "status": step.status.value,
                "task_id": step.task_id,
                "error_message": step.error_message,
                "started_at": step.started_at,
                "completed_at": step.completed_at,
                "attempts": step.attempts,
                "max_attempts": step.max_attempts,
            }
        return result

    async def _execute_stage(
        self,
        stage: WorkflowStage,
        current_steps: Dict[WorkflowStage, WorkflowStep],
        job: Job,
        db,
    ):
        """Execute a specific workflow stage"""
        # Initialize step if not exists
        if stage not in current_steps:
            current_steps[stage] = WorkflowStep(
                stage=stage, status=WorkflowStatus.PENDING
            )

        step = current_steps[stage]
        step.status = WorkflowStatus.IN_PROGRESS
        step.started_at = datetime.utcnow().isoformat()
        step.attempts += 1

        try:
            # Execute stage-specific logic
            if stage == WorkflowStage.INGESTION:
                await self._execute_ingestion(job)
            elif stage == WorkflowStage.PREPROCESSING:
                await self._execute_preprocessing(job)
            elif stage == WorkflowStage.LAYOUT_ANALYSIS:
                await self._execute_layout_analysis(job)
            elif stage == WorkflowStage.OCR:
                await self._execute_ocr(job)
            elif stage == WorkflowStage.ARTICLE_RECONSTRUCTION:
                await self._execute_article_reconstruction(job)
            elif stage == WorkflowStage.CONTRIBUTOR_PARSING:
                await self._execute_contributor_parsing(job)
            elif stage == WorkflowStage.IMAGE_EXTRACTION:
                await self._execute_image_extraction(job)
            elif stage == WorkflowStage.EXPORT:
                await self._execute_export(job)
            elif stage == WorkflowStage.EVALUATION:
                await self._execute_evaluation(job)

            step.status = WorkflowStatus.COMPLETED
            step.completed_at = datetime.utcnow().isoformat()

        except Exception as e:
            self.logger.error("Stage execution failed", stage=stage.value, error=str(e))
            step.status = WorkflowStatus.FAILED
            step.error_message = str(e)

            if step.attempts >= step.max_attempts:
                self.logger.error("Stage max retries exceeded", stage=stage.value)

        # Update job in database
        job.workflow_steps = self._serialize_workflow_steps(current_steps)
        await db.commit()

    async def _execute_ingestion(self, job: Job):
        """Execute ingestion stage - validate and prepare PDF"""
        self.logger.info("Executing ingestion stage")
        # TODO: Implement PDF validation and metadata extraction

    async def _execute_preprocessing(self, job: Job):
        """Execute preprocessing stage - split PDF into pages"""
        self.logger.info("Executing preprocessing stage")
        # TODO: Implement PDF page splitting

    async def _execute_layout_analysis(self, job: Job):
        """Execute layout analysis stage"""
        self.logger.info("Executing layout analysis stage")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.settings.model_service_url}/api/v1/layout/analyze",
                json={"job_id": str(job.id), "file_path": job.file_path},
            )
            response.raise_for_status()

    async def _execute_ocr(self, job: Job):
        """Execute OCR stage"""
        self.logger.info("Executing OCR stage")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.settings.model_service_url}/api/v1/ocr/process",
                json={"job_id": str(job.id)},
            )
            response.raise_for_status()

    async def _execute_article_reconstruction(self, job: Job):
        """Execute article reconstruction stage"""
        self.logger.info("Executing article reconstruction stage")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.settings.model_service_url}/api/v1/articles/reconstruct",
                json={"job_id": str(job.id)},
            )
            response.raise_for_status()

    async def _execute_contributor_parsing(self, job: Job):
        """Execute contributor parsing stage"""
        self.logger.info("Executing contributor parsing stage")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.settings.model_service_url}/api/v1/contributors/extract",
                json={"job_id": str(job.id)},
            )
            response.raise_for_status()

    async def _execute_image_extraction(self, job: Job):
        """Execute image extraction stage"""
        self.logger.info("Executing image extraction stage")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.settings.model_service_url}/api/v1/images/extract",
                json={"job_id": str(job.id)},
            )
            response.raise_for_status()

    async def _execute_export(self, job: Job):
        """Execute export stage - generate XML and CSV"""
        self.logger.info("Executing export stage")
        # TODO: Implement XML/CSV generation

    async def _execute_evaluation(self, job: Job):
        """Execute evaluation stage"""
        self.logger.info("Executing evaluation stage")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.settings.evaluation_service_url}/api/v1/evaluate",
                json={"job_id": str(job.id)},
            )
            response.raise_for_status()

            evaluation_result = response.json()
            job.accuracy_score = evaluation_result.get("accuracy_score")
            job.confidence_scores = evaluation_result.get("confidence_scores", {})
