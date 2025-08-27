from typing import Any, Dict

import structlog
from fastapi import APIRouter, HTTPException, Request
from model_service.models.contributor_extractor import ContributorExtractor

from shared.schemas.contributors import (
    ContributorExtractionRequest,
    ContributorExtractionResponse,
)

logger = structlog.get_logger()
router = APIRouter()


@router.post("/extract", response_model=ContributorExtractionResponse)
async def extract_contributors(
    request_data: ContributorExtractionRequest, request: Request
) -> ContributorExtractionResponse:
    """Extract contributor names and roles from articles"""
    model_manager = request.app.state.model_manager

    if not model_manager or not model_manager.is_model_loaded("ner"):
        raise HTTPException(status_code=503, detail="NER model not available")

    try:
        extractor = ContributorExtractor(model_manager)
        result = await extractor.extract_contributors(
            job_id=request_data.job_id, brand_config=request_data.brand_config
        )

        return ContributorExtractionResponse(
            job_id=request_data.job_id,
            contributors=result["contributors"],
            confidence_scores=result["confidence_scores"],
        )

    except Exception as e:
        logger.error(
            "Contributor extraction failed", job_id=request_data.job_id, error=str(e)
        )
        raise HTTPException(
            status_code=500, detail=f"Contributor extraction failed: {str(e)}"
        )


@router.post("/parse-bylines")
async def parse_bylines(
    bylines_data: Dict[str, Any], request: Request
) -> Dict[str, Any]:
    """Parse bylines to extract contributor names and roles"""
    model_manager = request.app.state.model_manager

    if not model_manager or not model_manager.is_model_loaded("ner"):
        raise HTTPException(status_code=503, detail="NER model not available")

    try:
        extractor = ContributorExtractor(model_manager)
        result = await extractor.parse_bylines(bylines_data)

        return {
            "parsed_contributors": result["contributors"],
            "confidence_scores": result["confidence_scores"],
        }

    except Exception as e:
        logger.error("Byline parsing failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Byline parsing failed: {str(e)}")


@router.post("/normalize-names")
async def normalize_contributor_names(
    names_data: Dict[str, Any], request: Request
) -> Dict[str, Any]:
    """Normalize contributor names to canonical format"""
    try:
        extractor = ContributorExtractor(None)  # Name normalization doesn't need models
        result = await extractor.normalize_names(names_data)

        return {
            "normalized_names": result["names"],
            "normalization_rules": result["rules_applied"],
        }

    except Exception as e:
        logger.error("Name normalization failed", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Name normalization failed: {str(e)}"
        )


@router.post("/classify-roles")
async def classify_contributor_roles(
    contributors_data: Dict[str, Any], request: Request
) -> Dict[str, Any]:
    """Classify contributor roles (author, photographer, etc.)"""
    model_manager = request.app.state.model_manager

    try:
        extractor = ContributorExtractor(model_manager)
        result = await extractor.classify_roles(contributors_data)

        return {
            "classified_roles": result["roles"],
            "confidence_scores": result["confidence_scores"],
        }

    except Exception as e:
        logger.error("Role classification failed", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Role classification failed: {str(e)}"
        )
