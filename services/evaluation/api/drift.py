from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List
import structlog

from evaluation.core.database import get_db
from evaluation.models.drift_detector import DriftDetector

logger = structlog.get_logger()
router = APIRouter()

@router.post("/detect")
async def detect_accuracy_drift(
    drift_data: Dict[str, Any],
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Detect accuracy drift for a brand"""
    try:
        detector = DriftDetector(db)
        result = await detector.detect_drift(
            brand=drift_data["brand"],
            recent_jobs=drift_data.get("recent_jobs", [])
        )
        
        return {
            "brand": drift_data["brand"],
            "drift_detected": result["drift_detected"],
            "drift_score": result["drift_score"],
            "trend_analysis": result["trend"],
            "trigger_reason": result["reason"],
            "recommendation": result["recommendation"]
        }
        
    except Exception as e:
        logger.error("Drift detection failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Drift detection failed: {str(e)}")

@router.get("/brand/{brand_name}/status")
async def get_brand_drift_status(
    brand_name: str,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Get current drift status for a brand"""
    try:
        detector = DriftDetector(db)
        status = await detector.get_brand_drift_status(brand_name)
        
        return {
            "brand": brand_name,
            "current_status": status["status"],
            "last_drift_event": status["last_drift_event"],
            "recent_accuracy_trend": status["trend"],
            "drift_risk_score": status["risk_score"]
        }
        
    except Exception as e:
        logger.error("Failed to get drift status", brand=brand_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get drift status: {str(e)}")

@router.get("/brand/{brand_name}/history")
async def get_brand_drift_history(
    brand_name: str,
    days: int = 90,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Get drift detection history for a brand"""
    try:
        detector = DriftDetector(db)
        history = await detector.get_drift_history(brand_name, days)
        
        return {
            "brand": brand_name,
            "period_days": days,
            "drift_events": history["events"],
            "total_drift_events": len(history["events"]),
            "accuracy_timeline": history["timeline"]
        }
        
    except Exception as e:
        logger.error("Failed to get drift history", brand=brand_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get drift history: {str(e)}")

@router.post("/analyze-trend")
async def analyze_accuracy_trend(
    trend_data: Dict[str, Any],
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Analyze accuracy trend for drift patterns"""
    try:
        detector = DriftDetector(db)
        analysis = await detector.analyze_accuracy_trend(trend_data)
        
        return {
            "trend_direction": analysis["direction"],
            "trend_strength": analysis["strength"],
            "statistical_significance": analysis["significance"],
            "predicted_future_accuracy": analysis["prediction"],
            "risk_factors": analysis["risk_factors"]
        }
        
    except Exception as e:
        logger.error("Trend analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Trend analysis failed: {str(e)}")

@router.get("/alerts")
async def get_drift_alerts(
    active_only: bool = True,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Get current drift alerts across all brands"""
    try:
        detector = DriftDetector(db)
        alerts = await detector.get_drift_alerts(active_only)
        
        return {
            "alerts": alerts["alerts"],
            "total_alerts": len(alerts["alerts"]),
            "brands_at_risk": alerts["brands_at_risk"],
            "severity_breakdown": alerts["severity_breakdown"]
        }
        
    except Exception as e:
        logger.error("Failed to get drift alerts", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get drift alerts: {str(e)}")