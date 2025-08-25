from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any
from uuid import UUID
import structlog

from evaluation.core.database import get_db
from shared.schemas.evaluation import EvaluationRequest, EvaluationResponse
from evaluation.models.accuracy_evaluator import AccuracyEvaluator

logger = structlog.get_logger()
router = APIRouter()

@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_job_accuracy(
    request_data: EvaluationRequest,
    db: AsyncSession = Depends(get_db)
) -> EvaluationResponse:
    """Evaluate accuracy of a completed job against gold standard"""
    try:
        evaluator = AccuracyEvaluator(db)
        result = await evaluator.evaluate_job(
            job_id=request_data.job_id,
            brand=request_data.brand
        )
        
        return EvaluationResponse(
            job_id=request_data.job_id,
            overall_accuracy=result["overall_accuracy"],
            field_accuracies=result["field_accuracies"],
            confidence_scores=result["confidence_scores"],
            passed_threshold=result["passed_threshold"],
            quarantine_recommended=result["quarantine_recommended"],
            evaluation_details=result["details"]
        )
        
    except Exception as e:
        logger.error("Job evaluation failed", 
                    job_id=request_data.job_id, 
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@router.get("/brand/{brand_name}/metrics")
async def get_brand_accuracy_metrics(
    brand_name: str,
    days: int = 30,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Get accuracy metrics for a specific brand over time"""
    try:
        evaluator = AccuracyEvaluator(db)
        metrics = await evaluator.get_brand_metrics(brand_name, days)
        
        return {
            "brand": brand_name,
            "period_days": days,
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error("Failed to get brand metrics", brand=brand_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@router.get("/brand/{brand_name}/pass-rate")
async def get_brand_pass_rate(
    brand_name: str,
    recent_issues: int = 10,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Get pass rate for a brand's recent issues"""
    try:
        evaluator = AccuracyEvaluator(db)
        pass_rate = await evaluator.calculate_brand_pass_rate(brand_name, recent_issues)
        
        return {
            "brand": brand_name,
            "recent_issues_count": recent_issues,
            "pass_rate": pass_rate["pass_rate"],
            "passing_issues": pass_rate["passing_count"],
            "total_issues": pass_rate["total_count"],
            "threshold": pass_rate["threshold"]
        }
        
    except Exception as e:
        logger.error("Failed to get brand pass rate", brand=brand_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get pass rate: {str(e)}")

@router.post("/compare-with-gold")
async def compare_with_gold_standard(
    comparison_data: Dict[str, Any],
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Compare extraction results with gold standard"""
    try:
        evaluator = AccuracyEvaluator(db)
        result = await evaluator.compare_with_gold_standard(comparison_data)
        
        return {
            "comparison_results": result["results"],
            "field_scores": result["field_scores"], 
            "overall_score": result["overall_score"],
            "detailed_diff": result["diff"]
        }
        
    except Exception as e:
        logger.error("Gold standard comparison failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

@router.get("/thresholds")
async def get_accuracy_thresholds() -> Dict[str, Any]:
    """Get current accuracy thresholds and weights"""
    from evaluation.core.config import get_settings
    settings = get_settings()
    
    return {
        "accuracy_threshold": settings.accuracy_threshold,
        "brand_pass_rate_threshold": settings.brand_pass_rate_threshold,
        "quarantine_threshold": settings.quarantine_threshold,
        "field_weights": settings.field_weights
    }