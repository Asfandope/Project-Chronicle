from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from uuid import UUID

class LayoutAnalysisRequest(BaseModel):
    job_id: UUID
    file_path: str
    brand_config: Optional[Dict[str, Any]] = None

class PageBlock(BaseModel):
    id: str
    type: str
    bbox: List[int]  # [x1, y1, x2, y2]
    text: str
    confidence: float

class PageLayout(BaseModel):
    blocks: List[PageBlock]
    page_dimensions: List[int]  # [width, height]

class SemanticGraph(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]

class LayoutAnalysisResponse(BaseModel):
    job_id: UUID
    pages: Dict[str, PageLayout]
    semantic_graph: SemanticGraph
    confidence_scores: Dict[str, float]