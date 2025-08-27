from typing import Dict, List, Optional, Tuple
from uuid import UUID

from pydantic import BaseModel


class ImageExtractionRequest(BaseModel):
    job_id: UUID
    min_size: Optional[Tuple[int, int]] = None


class ExtractedImage(BaseModel):
    id: str
    filename: str
    original_bbox: List[int]
    page: int
    width: int
    height: int
    format: str
    file_size: int
    hash: str


class ImageCaptionLink(BaseModel):
    caption_block_id: str
    caption_text: str
    confidence: float
    spatial_distance: float


class ImageExtractionResponse(BaseModel):
    job_id: UUID
    images: Dict[str, ExtractedImage]
    image_caption_links: Dict[str, ImageCaptionLink]  # image_id -> caption_link
    confidence_scores: Dict[str, float]
