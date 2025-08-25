from fastapi import APIRouter, Request, HTTPException
from typing import Dict, Any
import structlog

from shared.schemas.ocr import OCRRequest, OCRResponse
from model_service.models.ocr_processor import OCRProcessor

logger = structlog.get_logger()
router = APIRouter()

@router.post("/process", response_model=OCRResponse)
async def process_ocr(
    request_data: OCRRequest,
    request: Request
) -> OCRResponse:
    """Process OCR for PDF pages"""
    try:
        processor = OCRProcessor()
        result = await processor.process_pdf_ocr(
            job_id=request_data.job_id,
            brand_config=request_data.brand_config
        )
        
        return OCRResponse(
            job_id=request_data.job_id,
            ocr_results=result["ocr_results"],
            confidence_scores=result["confidence_scores"]
        )
        
    except Exception as e:
        logger.error("OCR processing failed", 
                    job_id=request_data.job_id, 
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

@router.post("/extract-text")
async def extract_text_from_blocks(
    blocks_data: Dict[str, Any],
    request: Request
) -> Dict[str, Any]:
    """Extract text from specific image blocks"""
    try:
        processor = OCRProcessor()
        result = await processor.extract_text_from_blocks(blocks_data)
        
        return {
            "extracted_text": result["extracted_text"],
            "confidence_scores": result["confidence_scores"]
        }
        
    except Exception as e:
        logger.error("Text extraction failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")

@router.get("/supported-languages")
async def get_supported_languages() -> Dict[str, Any]:
    """Get list of supported OCR languages"""
    try:
        processor = OCRProcessor()
        languages = processor.get_supported_languages()
        
        return {"supported_languages": languages}
        
    except Exception as e:
        logger.error("Failed to get supported languages", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get supported languages")