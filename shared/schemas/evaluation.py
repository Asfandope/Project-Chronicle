from typing import Any, Dict
from uuid import UUID

from pydantic import BaseModel


class EvaluationRequest(BaseModel):
    job_id: UUID
    brand: str


class EvaluationResponse(BaseModel):
    job_id: UUID
    overall_accuracy: float
    field_accuracies: Dict[str, float]  # title, body, contributors, media
    confidence_scores: Dict[str, float]
    passed_threshold: bool
    quarantine_recommended: bool
    evaluation_details: Dict[str, Any]
