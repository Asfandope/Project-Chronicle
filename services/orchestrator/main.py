from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import structlog
import uuid
import time
from typing import Dict, Any
import asyncio

from .api import jobs, health, config
from .core.config import get_settings
from .core.database import init_db
from .core.logging import configure_logging
from .utils.correlation import CorrelationIdMiddleware
from .services.file_watcher import FileWatcherService
from .services.job_queue_manager import JobQueueManager

# Configure structured logging
configure_logging()
logger = structlog.get_logger()

# Global services
services: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan with proper startup and shutdown."""
    settings = get_settings()
    
    try:
        # Startup
        logger.info("Starting Orchestrator Service", version="1.0.0")
        
        # Initialize database
        await init_db()
        logger.info("Database initialized")
        
        # Initialize job queue manager
        services["job_queue"] = JobQueueManager()
        await services["job_queue"].start()
        logger.info("Job queue manager started")
        
        # Initialize file watcher service if hot folder is configured
        if settings.enable_file_watcher:
            services["file_watcher"] = FileWatcherService(
                watch_directory=settings.input_directory,
                job_queue_manager=services["job_queue"]
            )
            await services["file_watcher"].start()
            logger.info("File watcher service started", directory=settings.input_directory)
        
        # Health check services
        services["health_checks"] = {
            "database": True,
            "redis": True,
            "file_system": True
        }
        
        logger.info("All services started successfully")
        
        yield
        
    except Exception as e:
        logger.error("Failed to start services", error=str(e), exc_info=True)
        raise
    finally:
        # Shutdown
        logger.info("Shutting down Orchestrator Service")
        
        # Stop file watcher
        if "file_watcher" in services:
            await services["file_watcher"].stop()
            logger.info("File watcher service stopped")
        
        # Stop job queue manager
        if "job_queue" in services:
            await services["job_queue"].stop()
            logger.info("Job queue manager stopped")
        
        logger.info("Orchestrator Service shutdown complete")

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="Magazine PDF Extractor - Orchestrator",
        description="Central orchestration service for PDF extraction workflow",
        version="1.0.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan,
        root_path=settings.root_path,
    )
    
    # Add security middleware
    if not settings.debug:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.allowed_hosts
        )
    
    # Add correlation ID middleware for request tracking
    app.add_middleware(CorrelationIdMiddleware)
    
    # Add request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all HTTP requests with timing and correlation IDs."""
        start_time = time.time()
        correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))
        
        # Bind correlation ID to logger context
        logger_ctx = structlog.contextvars.bind_contextvars(
            correlation_id=correlation_id,
            method=request.method,
            url=str(request.url),
            user_agent=request.headers.get("user-agent"),
            client_ip=request.client.host if request.client else None,
        )
        
        logger_ctx.info("Request started")
        
        try:
            response: Response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Correlation-ID"] = correlation_id
            
            logger_ctx.info(
                "Request completed",
                status_code=response.status_code,
                process_time=process_time
            )
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger_ctx.error(
                "Request failed",
                error=str(e),
                process_time=process_time,
                exc_info=True
            )
            raise
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Correlation-ID", "X-Process-Time"]
    )
    
    # Include API routers
    app.include_router(health.router, tags=["health"])
    app.include_router(jobs.router, tags=["jobs"])
    app.include_router(config.router, prefix="/api/v1/config", tags=["config"])
    
    # Global exception handler
    @app.exception_handler(500)
    async def internal_server_error(request: Request, exc: Exception):
        """Handle internal server errors with proper logging."""
        correlation_id = getattr(request.state, "correlation_id", "unknown")
        logger.error(
            "Internal server error",
            correlation_id=correlation_id,
            error=str(exc),
            exc_info=True
        )
        return {
            "detail": "Internal server error",
            "correlation_id": correlation_id
        }
    
    # Make services available to routes
    app.state.services = services
    
    return app

app = create_app()