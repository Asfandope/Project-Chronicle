from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import structlog

logger = structlog.get_logger()


class WorkflowStage(str, Enum):
    INGESTION = "ingestion"
    PREPROCESSING = "preprocessing"
    LAYOUT_ANALYSIS = "layout_analysis"
    OCR = "ocr"
    ARTICLE_RECONSTRUCTION = "article_reconstruction"
    CONTRIBUTOR_PARSING = "contributor_parsing"
    IMAGE_EXTRACTION = "image_extraction"
    EXPORT = "export"
    EVALUATION = "evaluation"
    COMPLETED = "completed"
    FAILED = "failed"
    QUARANTINED = "quarantined"


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class WorkflowStep:
    stage: WorkflowStage
    status: WorkflowStatus
    task_id: Optional[str] = None
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    attempts: int = 0
    max_attempts: int = 3


class WorkflowEngine:
    """Central workflow orchestration engine"""

    STAGE_DEPENDENCIES = {
        WorkflowStage.PREPROCESSING: [WorkflowStage.INGESTION],
        WorkflowStage.LAYOUT_ANALYSIS: [WorkflowStage.PREPROCESSING],
        WorkflowStage.OCR: [WorkflowStage.LAYOUT_ANALYSIS],
        WorkflowStage.ARTICLE_RECONSTRUCTION: [
            WorkflowStage.OCR,
            WorkflowStage.LAYOUT_ANALYSIS,
        ],
        WorkflowStage.CONTRIBUTOR_PARSING: [WorkflowStage.ARTICLE_RECONSTRUCTION],
        WorkflowStage.IMAGE_EXTRACTION: [WorkflowStage.LAYOUT_ANALYSIS],
        WorkflowStage.EXPORT: [
            WorkflowStage.ARTICLE_RECONSTRUCTION,
            WorkflowStage.CONTRIBUTOR_PARSING,
            WorkflowStage.IMAGE_EXTRACTION,
        ],
        WorkflowStage.EVALUATION: [WorkflowStage.EXPORT],
        WorkflowStage.COMPLETED: [WorkflowStage.EVALUATION],
    }

    def __init__(self):
        self.logger = logger.bind(component="workflow_engine")

    def get_next_stages(
        self, current_steps: Dict[WorkflowStage, WorkflowStep]
    ) -> List[WorkflowStage]:
        """Get next stages ready for execution"""
        ready_stages = []

        for stage, dependencies in self.STAGE_DEPENDENCIES.items():
            # Skip if this stage is already done or in progress
            if stage in current_steps:
                current_status = current_steps[stage].status
                if current_status in [
                    WorkflowStatus.COMPLETED,
                    WorkflowStatus.IN_PROGRESS,
                ]:
                    continue
                # Allow retry for failed stages
                if (
                    current_status == WorkflowStatus.FAILED
                    and current_steps[stage].attempts
                    >= current_steps[stage].max_attempts
                ):
                    continue

            # Check if all dependencies are completed
            if all(
                dep in current_steps
                and current_steps[dep].status == WorkflowStatus.COMPLETED
                for dep in dependencies
            ):
                ready_stages.append(stage)

        return ready_stages

    def is_workflow_complete(
        self, current_steps: Dict[WorkflowStage, WorkflowStep]
    ) -> bool:
        """Check if workflow is complete"""
        return (
            WorkflowStage.COMPLETED in current_steps
            and current_steps[WorkflowStage.COMPLETED].status
            == WorkflowStatus.COMPLETED
        )

    def is_workflow_failed(
        self, current_steps: Dict[WorkflowStage, WorkflowStep]
    ) -> bool:
        """Check if workflow has failed permanently"""
        # Check for any stages that have exceeded max attempts
        for step in current_steps.values():
            if (
                step.status == WorkflowStatus.FAILED
                and step.attempts >= step.max_attempts
            ):
                return True
        return False

    def should_quarantine(
        self,
        current_steps: Dict[WorkflowStage, WorkflowStep],
        accuracy: Optional[float] = None,
    ) -> bool:
        """Determine if job should be quarantined based on accuracy or failures"""
        if accuracy is not None and accuracy < 0.999:  # Below accuracy threshold
            return True

        # Check for critical stage failures
        critical_stages = [
            WorkflowStage.LAYOUT_ANALYSIS,
            WorkflowStage.ARTICLE_RECONSTRUCTION,
        ]
        for stage in critical_stages:
            if (
                stage in current_steps
                and current_steps[stage].status == WorkflowStatus.FAILED
            ):
                return True

        return False
