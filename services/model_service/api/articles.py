from typing import Any, Dict

import structlog
from fastapi import APIRouter, HTTPException, Request
from model_service.models.article_reconstructor import ArticleReconstructor

from shared.schemas.articles import (
    ArticleReconstructionRequest,
    ArticleReconstructionResponse,
)

logger = structlog.get_logger()
router = APIRouter()


@router.post("/reconstruct", response_model=ArticleReconstructionResponse)
async def reconstruct_articles(
    request_data: ArticleReconstructionRequest, request: Request
) -> ArticleReconstructionResponse:
    """Reconstruct complete articles from semantic graph"""
    model_manager = request.app.state.model_manager

    try:
        reconstructor = ArticleReconstructor(model_manager)
        result = await reconstructor.reconstruct_articles(
            job_id=request_data.job_id, brand_config=request_data.brand_config
        )

        return ArticleReconstructionResponse(
            job_id=request_data.job_id,
            articles=result["articles"],
            article_boundaries=result["article_boundaries"],
            confidence_scores=result["confidence_scores"],
        )

    except Exception as e:
        logger.error(
            "Article reconstruction failed", job_id=request_data.job_id, error=str(e)
        )
        raise HTTPException(
            status_code=500, detail=f"Article reconstruction failed: {str(e)}"
        )


@router.post("/identify-boundaries")
async def identify_article_boundaries(
    semantic_graph: Dict[str, Any], request: Request
) -> Dict[str, Any]:
    """Identify article boundaries from semantic graph"""
    model_manager = request.app.state.model_manager

    try:
        reconstructor = ArticleReconstructor(model_manager)
        result = await reconstructor.identify_boundaries(semantic_graph)

        return {
            "article_boundaries": result["boundaries"],
            "confidence_scores": result["confidence_scores"],
        }

    except Exception as e:
        logger.error("Article boundary identification failed", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Boundary identification failed: {str(e)}"
        )


@router.post("/stitch-split-articles")
async def stitch_split_articles(
    articles_data: Dict[str, Any], request: Request
) -> Dict[str, Any]:
    """Stitch together articles split across multiple pages"""
    model_manager = request.app.state.model_manager

    try:
        reconstructor = ArticleReconstructor(model_manager)
        result = await reconstructor.stitch_split_articles(articles_data)

        return {
            "stitched_articles": result["articles"],
            "stitch_operations": result["operations"],
        }

    except Exception as e:
        logger.error("Article stitching failed", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Article stitching failed: {str(e)}"
        )
