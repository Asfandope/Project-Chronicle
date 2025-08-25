"""
Model Service - Stub implementation for development.
Provides mock ML model endpoints for PDF processing pipeline.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import structlog
import asyncio
from typing import Dict, Any
from datetime import datetime, timezone
import random
import uuid
import os

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(colors=True) if os.getenv("LOG_FORMAT") == "console" else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

def create_app() -> FastAPI:
    app = FastAPI(
        title="Magazine PDF Extractor - Model Service",
        description="ML model service for PDF layout analysis, OCR, and article reconstruction (Development Stub)",
        version="1.0.0-stub",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup_event():
        logger.info("Model Service (Stub) starting up")

    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Model Service (Stub) shutting down")

    # Health check endpoints
    @app.get("/health/")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "model-service",
            "version": "1.0.0-stub",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "stub"
        }

    @app.get("/health/detailed")
    async def detailed_health_check():
        """Detailed health check with mock model status."""
        return {
            "status": "healthy",
            "service": "model-service",
            "version": "1.0.0-stub",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "stub",
            "models": {
                "layout_analyzer": {"status": "loaded", "version": "mock-v1.0"},
                "ocr_processor": {"status": "loaded", "version": "mock-v1.0"},
                "article_reconstructor": {"status": "loaded", "version": "mock-v1.0"},
            },
            "device": "cpu",
            "memory_usage": "mock"
        }

    # Layout Analysis endpoint
    @app.post("/layout/analyze")
    async def analyze_layout(request: Request):
        """Analyze PDF layout and return mock results."""
        request_data = await request.json()
        
        # Simulate processing time
        await asyncio.sleep(random.uniform(1, 3))
        
        logger.info("Layout analysis requested", 
                    correlation_id=request.headers.get("x-correlation-id"))
        
        # Generate mock layout analysis results
        mock_result = {
            "job_id": request_data.get("job_id"),
            "page_count": random.randint(20, 150),
            "layout_confidence": random.uniform(0.85, 0.98),
            "blocks": [
                {
                    "block_id": f"block_{i}",
                    "type": random.choice(["text", "image", "title", "caption"]),
                    "bbox": [
                        random.randint(0, 500),
                        random.randint(0, 700),
                        random.randint(500, 800),
                        random.randint(700, 900)
                    ],
                    "confidence": random.uniform(0.8, 0.99)
                }
                for i in range(random.randint(15, 40))
            ],
            "processing_time_ms": random.randint(800, 3000)
        }
        
        return mock_result

    # OCR Processing endpoint
    @app.post("/ocr/process")
    async def process_ocr(request: Request):
        """Process OCR on PDF blocks and return mock results."""
        request_data = await request.json()
        
        # Simulate processing time
        await asyncio.sleep(random.uniform(2, 5))
        
        logger.info("OCR processing requested",
                    correlation_id=request.headers.get("x-correlation-id"))
        
        # Generate mock OCR results
        mock_result = {
            "job_id": request_data.get("job_id"),
            "ocr_confidence": random.uniform(0.88, 0.96),
            "text_blocks": [
                {
                    "block_id": f"text_block_{i}",
                    "text": f"This is mock extracted text from block {i}. " * random.randint(5, 20),
                    "confidence": random.uniform(0.85, 0.98),
                    "language": "en",
                    "word_count": random.randint(50, 200)
                }
                for i in range(random.randint(10, 25))
            ],
            "processing_time_ms": random.randint(1500, 5000)
        }
        
        return mock_result

    # Article Reconstruction endpoint
    @app.post("/articles/reconstruct")
    async def reconstruct_articles(request: Request):
        """Reconstruct articles from processed content and return mock results."""
        request_data = await request.json()
        
        # Simulate processing time
        await asyncio.sleep(random.uniform(1, 2))
        
        logger.info("Article reconstruction requested",
                    correlation_id=request.headers.get("x-correlation-id"))
        
        # Generate mock article reconstruction results
        mock_articles = []
        article_count = random.randint(8, 15)
        
        for i in range(article_count):
            mock_articles.append({
                "article_id": f"article_{i}",
                "title": {
                    "content": f"Mock Article Title {i + 1}",
                    "confidence": random.uniform(0.90, 0.98)
                },
                "body": {
                    "content": [
                        {
                            "content": f"This is mock article content paragraph {j}. " * random.randint(10, 30),
                            "confidence": random.uniform(0.85, 0.95)
                        }
                        for j in range(random.randint(3, 8))
                    ]
                },
                "contributors": [
                    {
                        "name": f"Mock Author {i}",
                        "role": "author",
                        "confidence": random.uniform(0.80, 0.95)
                    }
                ],
                "page_range": [random.randint(1, 50), random.randint(51, 100)],
                "accuracy_score": random.uniform(0.85, 0.97)
            })
        
        mock_result = {
            "job_id": request_data.get("job_id"),
            "articles": mock_articles,
            "overall_confidence": random.uniform(0.88, 0.95),
            "processing_time_ms": random.randint(800, 2000)
        }
        
        return mock_result

    # Metrics endpoint for Prometheus
    @app.get("/metrics")
    async def get_metrics():
        """Prometheus metrics endpoint (stub)."""
        # Return mock metrics in Prometheus format
        return f"""# HELP model_requests_total Total number of model requests
# TYPE model_requests_total counter
model_requests_total{{service="model-service",endpoint="layout"}} {random.randint(100, 1000)}
model_requests_total{{service="model-service",endpoint="ocr"}} {random.randint(100, 1000)}
model_requests_total{{service="model-service",endpoint="articles"}} {random.randint(100, 1000)}

# HELP model_processing_duration_seconds Time spent processing requests
# TYPE model_processing_duration_seconds histogram
model_processing_duration_seconds_bucket{{service="model-service",le="1.0"}} {random.randint(50, 200)}
model_processing_duration_seconds_bucket{{service="model-service",le="5.0"}} {random.randint(200, 500)}
model_processing_duration_seconds_bucket{{service="model-service",le="+Inf"}} {random.randint(500, 1000)}

# HELP model_accuracy_score Current model accuracy score
# TYPE model_accuracy_score gauge
model_accuracy_score{{service="model-service"}} {random.uniform(0.85, 0.95)}
"""

    return app

app = create_app()