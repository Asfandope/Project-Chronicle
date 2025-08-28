from datetime import datetime, timezone

from fastapi import APIRouter

router = APIRouter()


@router.get("/health/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "evaluation-service",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "service": "evaluation-service",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": {
            "drift_detector": {"status": "available"},
            "evaluation_metrics": {"status": "available"},
        },
    }
