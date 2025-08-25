from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any
import structlog

from evaluation.core.database import get_db
from evaluation.models.auto_tuner import AutoTuner

logger = structlog.get_logger()
router = APIRouter()

@router.post("/trigger")
async def trigger_auto_tuning(
    tuning_request: Dict[str, Any],
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Trigger auto-tuning for a brand"""
    try:
        tuner = AutoTuner(db)
        
        # Check if tuning is needed and allowed
        can_tune = await tuner.can_trigger_tuning(
            brand=tuning_request["brand"],
            reason=tuning_request.get("reason", "manual_trigger")
        )
        
        if not can_tune["allowed"]:
            return {
                "triggered": False,
                "reason": can_tune["reason"],
                "next_allowed_time": can_tune.get("next_allowed_time")
            }
        
        # Start tuning in background
        background_tasks.add_task(
            tuner.run_tuning_cycle,
            brand=tuning_request["brand"],
            trigger_reason=tuning_request.get("reason", "manual_trigger")
        )
        
        return {
            "triggered": True,
            "brand": tuning_request["brand"],
            "estimated_duration_minutes": 30,
            "message": "Auto-tuning started in background"
        }
        
    except Exception as e:
        logger.error("Failed to trigger auto-tuning", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to trigger tuning: {str(e)}")

@router.get("/brand/{brand_name}/status")
async def get_tuning_status(
    brand_name: str,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Get current tuning status for a brand"""
    try:
        tuner = AutoTuner(db)
        status = await tuner.get_tuning_status(brand_name)
        
        return {
            "brand": brand_name,
            "current_status": status["status"],
            "last_tuning_event": status["last_event"],
            "next_allowed_tuning": status["next_allowed"],
            "tuning_history_summary": status["history_summary"]
        }
        
    except Exception as e:
        logger.error("Failed to get tuning status", brand=brand_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get tuning status: {str(e)}")

@router.get("/brand/{brand_name}/history")
async def get_tuning_history(
    brand_name: str,
    limit: int = 50,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Get tuning history for a brand"""
    try:
        tuner = AutoTuner(db)
        history = await tuner.get_tuning_history(brand_name, limit)
        
        return {
            "brand": brand_name,
            "tuning_events": history["events"],
            "total_events": len(history["events"]),
            "success_rate": history["success_rate"],
            "average_improvement": history["average_improvement"]
        }
        
    except Exception as e:
        logger.error("Failed to get tuning history", brand=brand_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get tuning history: {str(e)}")

@router.post("/validate-parameters")
async def validate_tuning_parameters(
    parameters: Dict[str, Any],
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Validate proposed tuning parameters"""
    try:
        tuner = AutoTuner(db)
        validation = await tuner.validate_parameters(parameters)
        
        return {
            "valid": validation["valid"],
            "validation_errors": validation["errors"],
            "parameter_analysis": validation["analysis"],
            "estimated_impact": validation["impact"]
        }
        
    except Exception as e:
        logger.error("Parameter validation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Parameter validation failed: {str(e)}")

@router.post("/rollback")
async def rollback_tuning(
    rollback_request: Dict[str, Any],
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Rollback to previous tuning configuration"""
    try:
        tuner = AutoTuner(db)
        result = await tuner.rollback_tuning(
            brand=rollback_request["brand"],
            target_version=rollback_request.get("target_version")
        )
        
        return {
            "rolled_back": result["success"],
            "previous_version": result["previous_version"],
            "current_version": result["current_version"],
            "rollback_reason": rollback_request.get("reason", "manual_rollback")
        }
        
    except Exception as e:
        logger.error("Tuning rollback failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Rollback failed: {str(e)}")

@router.get("/parameters/{brand_name}")
async def get_current_parameters(
    brand_name: str,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Get current tuning parameters for a brand"""
    try:
        tuner = AutoTuner(db)
        parameters = await tuner.get_current_parameters(brand_name)
        
        return {
            "brand": brand_name,
            "current_parameters": parameters["parameters"],
            "parameter_version": parameters["version"],
            "last_updated": parameters["last_updated"],
            "parameter_sources": parameters["sources"]
        }
        
    except Exception as e:
        logger.error("Failed to get current parameters", brand=brand_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get parameters: {str(e)}")