from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel


class JobStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    QUARANTINED = "quarantined"


class JobCreate(BaseModel):
    filename: str
    brand: Optional[str] = None


class JobResponse(BaseModel):
    id: UUID
    filename: str
    file_path: str
    file_size: int
    brand: Optional[str] = None
    issue_date: Optional[str] = None
    overall_status: JobStatus
    accuracy_score: Optional[float] = None
    confidence_scores: Dict[str, float] = {}
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    processing_time_seconds: Optional[int] = None
    xml_output_path: Optional[str] = None
    csv_output_path: Optional[str] = None
    images_output_directory: Optional[str] = None
    error_message: Optional[str] = None
    quarantine_reason: Optional[str] = None

    class Config:
        from_attributes = True


class JobListResponse(BaseModel):
    jobs: List[JobResponse]
    total: int
    skip: int
    limit: int
