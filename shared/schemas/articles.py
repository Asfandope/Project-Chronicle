from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel


class ArticleReconstructionRequest(BaseModel):
    job_id: UUID
    brand_config: Optional[Dict[str, Any]] = None


class Article(BaseModel):
    id: str
    title: str
    body_blocks: List[str]
    pages: List[int]
    contributors: List[str]
    images: List[str]
    confidence: float


class ArticleBoundary(BaseModel):
    start_block: str
    end_block: str
    pages: List[int]
    boundary_confidence: float


class ArticleReconstructionResponse(BaseModel):
    job_id: UUID
    articles: Dict[str, Article]
    article_boundaries: Dict[str, ArticleBoundary]
    confidence_scores: Dict[str, float]
