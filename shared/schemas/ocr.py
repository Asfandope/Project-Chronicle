from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from uuid import UUID

class OCRRequest(BaseModel):
    job_id: UUID
    brand_config: Optional[Dict[str, Any]] = None

class OCRBlockResult(BaseModel):
    block_id: str
    text: str
    confidence: float
    method: str  # "direct_extraction" or "tesseract_ocr"
    word_confidences: Optional[List[int]] = None

class PageOCRResult(BaseModel):
    blocks: List[OCRBlockResult]
    page_confidence: float

class OCRResponse(BaseModel):
    job_id: UUID
    ocr_results: Dict[str, PageOCRResult]
    confidence_scores: Dict[str, float]