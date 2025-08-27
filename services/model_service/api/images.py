from typing import Any, Dict

import structlog
from fastapi import APIRouter, HTTPException, Request
from model_service.models.image_extractor import ImageExtractor

from shared.schemas.images import ImageExtractionRequest, ImageExtractionResponse

logger = structlog.get_logger()
router = APIRouter()


@router.post("/extract", response_model=ImageExtractionResponse)
async def extract_images(
    request_data: ImageExtractionRequest, request: Request
) -> ImageExtractionResponse:
    """Extract images and link them to captions"""
    try:
        extractor = ImageExtractor()
        result = await extractor.extract_images_and_captions(
            job_id=request_data.job_id, min_size=request_data.min_size or (100, 100)
        )

        return ImageExtractionResponse(
            job_id=request_data.job_id,
            images=result["images"],
            image_caption_links=result["image_caption_links"],
            confidence_scores=result["confidence_scores"],
        )

    except Exception as e:
        logger.error(
            "Image extraction failed", job_id=request_data.job_id, error=str(e)
        )
        raise HTTPException(
            status_code=500, detail=f"Image extraction failed: {str(e)}"
        )


@router.post("/link-captions")
async def link_images_to_captions(
    linking_data: Dict[str, Any], request: Request
) -> Dict[str, Any]:
    """Link extracted images to their captions using spatial proximity"""
    try:
        extractor = ImageExtractor()
        result = await extractor.link_images_to_captions(linking_data)

        return {
            "image_caption_links": result["links"],
            "confidence_scores": result["confidence_scores"],
            "unlinked_images": result["unlinked_images"],
            "unlinked_captions": result["unlinked_captions"],
        }

    except Exception as e:
        logger.error("Image-caption linking failed", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Image-caption linking failed: {str(e)}"
        )


@router.post("/filter-size")
async def filter_images_by_size(
    images_data: Dict[str, Any],
    min_width: int = 100,
    min_height: int = 100,
    request: Request = None,
) -> Dict[str, Any]:
    """Filter images by minimum size requirements"""
    try:
        extractor = ImageExtractor()
        result = await extractor.filter_images_by_size(
            images_data, min_width=min_width, min_height=min_height
        )

        return {
            "filtered_images": result["images"],
            "removed_count": result["removed_count"],
            "filter_criteria": {"min_width": min_width, "min_height": min_height},
        }

    except Exception as e:
        logger.error("Image filtering failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Image filtering failed: {str(e)}")


@router.get("/formats")
async def get_supported_image_formats() -> Dict[str, Any]:
    """Get list of supported image formats"""
    return {
        "supported_formats": ["JPEG", "JPG", "PNG", "TIFF", "TIF", "BMP", "GIF"],
        "recommended_formats": ["JPEG", "PNG"],
        "output_format": "JPEG",
    }
