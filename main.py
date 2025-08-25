"""
Main application entry point for Project Chronicle.

Combines all services: evaluation, parameter management, self-tuning, and quarantine
into a single FastAPI application for local testing and development.
"""

import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from database import init_database, get_db_session
from db_deps import SessionLocal
from evaluation_service.main import app as evaluation_app
from parameter_management.api import mount_parameter_management
from self_tuning.api import mount_self_tuning_api
from quarantine.api import mount_quarantine_api
from parameter_management.initialization import initialize_parameter_management_system

# Try to import model service and orchestrator (with error handling)
try:
    from services.model_service.main import create_app as create_model_app
    model_service_available = True
except ImportError as e:
    print(f"Model service not available: {e}")
    model_service_available = False

try:
    from services.orchestrator.main import create_app as create_orchestrator_app
    orchestrator_service_available = True
except ImportError as e:
    print(f"Orchestrator service not available: {e}")
    orchestrator_service_available = False


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('project_chronicle.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    logger.info("Starting Project Chronicle application...")
    
    try:
        # Initialize database
        logger.info("Initializing database...")
        init_database()
        
        # Initialize parameter management with default parameters
        logger.info("Initializing parameter management...")
        with get_db_session() as session:
            try:
                results = initialize_parameter_management_system(
                    session=session,
                    force_recreate=False,
                    skip_existing=True
                )
                logger.info(f"Parameter initialization: {results}")
            except Exception as e:
                logger.warning(f"Parameter initialization failed (continuing anyway): {e}")
        
        logger.info("Application startup completed successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise
    finally:
        logger.info("Application shutdown completed")


def create_app() -> FastAPI:
    """Create the main FastAPI application."""
    
    app = FastAPI(
        title="Project Chronicle",
        description="Magazine extraction testing and optimization system",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        try:
            # Test database connection
            with get_db_session() as session:
                session.execute(text("SELECT 1"))
            
            services = [
                "evaluation_service",
                "parameter_management", 
                "self_tuning",
                "quarantine"
            ]
            
            if model_service_available:
                services.append("model_service")
            if orchestrator_service_available:
                services.append("orchestrator_service")
                
            return {
                "status": "healthy",
                "services": services,
                "database": "connected"
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")
    
    # System status endpoint
    @app.get("/status")
    async def system_status():
        """Get overall system status."""
        try:
            with get_db_session() as session:
                from evaluation_service.models import EvaluationRun
                from parameter_management.models import Parameter
                from self_tuning.models import TuningRun
                from quarantine.models import QuarantineItem
                
                # Get counts from each service
                evaluation_runs = session.query(EvaluationRun).count()
                parameters = session.query(Parameter).count()
                tuning_runs = session.query(TuningRun).count()
                quarantined_items = session.query(QuarantineItem).count()
                
                return {
                    "system": "Project Chronicle",
                    "version": "1.0.0",
                    "statistics": {
                        "evaluation_runs": evaluation_runs,
                        "parameters": parameters,
                        "tuning_runs": tuning_runs,
                        "quarantined_items": quarantined_items
                    }
                }
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Status check failed: {str(e)}"}
            )
    
    # Mount all service APIs
    mount_parameter_management(app, SessionLocal, prefix="/api/v1")
    mount_self_tuning_api(app, prefix="/api/v1")
    mount_quarantine_api(app, prefix="/api/v1")
    
    # Mount evaluation service (it's a separate app)
    app.mount("/api/v1/evaluation", evaluation_app)
    
    # Mount model service if available
    if model_service_available:
        try:
            model_app = create_model_app()
            app.mount("/api/v1/model", model_app)
            logger.info("Model service mounted at /api/v1/model")
        except Exception as e:
            logger.error(f"Failed to mount model service: {e}")
    
    # Mount orchestrator service if available  
    if orchestrator_service_available:
        try:
            orchestrator_app = create_orchestrator_app()
            app.mount("/api/v1/orchestrator", orchestrator_app)
            logger.info("Orchestrator service mounted at /api/v1/orchestrator")
        except Exception as e:
            logger.error(f"Failed to mount orchestrator service: {e}")
    
    logger.info("FastAPI application created with all services mounted")
    
    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    log_level = os.getenv("LOG_LEVEL", "info")
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=True,  # Enable auto-reload for development
        reload_dirs=[".", "evaluation_service", "parameter_management", "self_tuning", "quarantine", "synthetic_data"]
    )