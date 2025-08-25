from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any
import structlog

from shared.schemas.layout import LayoutAnalysisRequest, LayoutAnalysisResponse
from model_service.models.layout_analyzer import LayoutAnalyzer

logger = structlog.get_logger()
router = APIRouter()

@router.post("/analyze", response_model=LayoutAnalysisResponse)
async def analyze_layout(
    request_data: LayoutAnalysisRequest,
    request: Request
) -> LayoutAnalysisResponse:
    """Analyze PDF layout and extract blocks with bounding boxes"""
    model_manager = request.app.state.model_manager
    
    if not model_manager or not model_manager.is_model_loaded('layout'):
        raise HTTPException(status_code=503, detail="Layout model not available")
    
    try:
        analyzer = LayoutAnalyzer(model_manager)
        result = await analyzer.analyze_pdf_layout(
            job_id=request_data.job_id,
            file_path=request_data.file_path,
            brand_config=request_data.brand_config
        )
        
        return LayoutAnalysisResponse(
            job_id=request_data.job_id,
            pages=result["pages"],
            semantic_graph=result["semantic_graph"],
            confidence_scores=result["confidence_scores"]
        )
        
    except Exception as e:
        logger.error("Layout analysis failed", 
                    job_id=request_data.job_id, 
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Layout analysis failed: {str(e)}")

@router.post("/classify-blocks")
async def classify_blocks(
    blocks_data: Dict[str, Any],
    request: Request
) -> Dict[str, Any]:
    """Classify extracted blocks into semantic categories"""
    model_manager = request.app.state.model_manager
    
    if not model_manager or not model_manager.is_model_loaded('layout'):
        raise HTTPException(status_code=503, detail="Layout model not available")
    
    try:
        analyzer = LayoutAnalyzer(model_manager)
        result = await analyzer.classify_blocks(blocks_data)
        
        return {
            "classified_blocks": result["classified_blocks"],
            "confidence_scores": result["confidence_scores"]
        }
        
    except Exception as e:
        logger.error("Block classification failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Block classification failed: {str(e)}")