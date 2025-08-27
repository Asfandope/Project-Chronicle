import structlog
import torch
from fastapi import APIRouter, Request

logger = structlog.get_logger()
router = APIRouter()


@router.get("/")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy", "service": "model_service"}


@router.get("/detailed")
async def detailed_health_check(request: Request):
    """Detailed health check including model status"""
    model_manager = request.app.state.model_manager

    health_status = {
        "status": "healthy",
        "service": "model_service",
        "device": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count()
            if torch.cuda.is_available()
            else 0,
            "current_device": "cuda" if torch.cuda.is_available() else "cpu",
        },
        "models": model_manager.get_model_info() if model_manager else {},
    }

    # Check if critical models are loaded
    if model_manager:
        required_models = ["layout", "ner"]
        missing_models = [
            model
            for model in required_models
            if not model_manager.is_model_loaded(model)
        ]

        if missing_models:
            health_status["status"] = "degraded"
            health_status["missing_models"] = missing_models
    else:
        health_status["status"] = "unhealthy"
        health_status["error"] = "Model manager not initialized"

    return health_status


@router.get("/models")
async def get_model_info(request: Request):
    """Get detailed information about loaded models"""
    model_manager = request.app.state.model_manager

    if not model_manager:
        return {"error": "Model manager not initialized"}

    return model_manager.get_model_info()
