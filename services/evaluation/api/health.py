from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import structlog

from evaluation.core.database import get_db

logger = structlog.get_logger()
router = APIRouter()

@router.get("/")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy", "service": "evaluation"}

@router.get("/detailed")
async def detailed_health_check(db: AsyncSession = Depends(get_db)):
    """Detailed health check including database connectivity"""
    health_status = {
        "status": "healthy",
        "service": "evaluation",
        "components": {}
    }
    
    # Check database
    try:
        await db.execute(text("SELECT 1"))
        health_status["components"]["database"] = "healthy"
    except Exception as e:
        logger.error("Database health check failed", error=str(e))
        health_status["components"]["database"] = "unhealthy"
        health_status["status"] = "degraded"
    
    # Check gold sets availability
    try:
        # TODO: Check if required gold sets are available
        health_status["components"]["gold_sets"] = "healthy"
    except Exception as e:
        logger.error("Gold sets health check failed", error=str(e))
        health_status["components"]["gold_sets"] = "degraded"
        health_status["status"] = "degraded"
    
    if health_status["status"] != "healthy":
        raise HTTPException(status_code=503, detail=health_status)
    
    return health_status