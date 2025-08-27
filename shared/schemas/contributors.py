from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel


class ContributorExtractionRequest(BaseModel):
    job_id: UUID
    brand_config: Optional[Dict[str, Any]] = None


class Contributor(BaseModel):
    name: str
    normalized_name: str
    role: str  # "author", "photographer", "illustrator"
    confidence: float
    source_text: str


class ContributorExtractionResponse(BaseModel):
    job_id: UUID
    contributors: Dict[str, List[Contributor]]  # article_id -> contributors
    confidence_scores: Dict[str, float]
