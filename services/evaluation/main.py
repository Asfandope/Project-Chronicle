"""
Evaluation Service - Model evaluation and drift detection.
"""

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.health import router as health_router
from .core.config import get_settings

logger = structlog.get_logger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Magazine PDF Extractor - Evaluation Service",
        description="Model evaluation and drift detection service",
        version="1.0.0",
        debug=settings.debug,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health_router, tags=["health"])

    @app.on_event("startup")
    async def startup_event():
        logger.info("Evaluation Service starting up")

    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Evaluation Service shutting down")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
