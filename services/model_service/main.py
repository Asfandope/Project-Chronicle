"""
Model Service - Stub implementation for development.
Provides mock ML model endpoints for PDF processing pipeline.
"""

import asyncio
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

# Import real business logic from shared modules
try:
    from shared.layout.analyzer import LayoutAnalyzer
    from shared.layout.types import LayoutConfig
    from shared.ocr.engine import OCREngine
    from shared.ocr.types import OCRConfig
    from shared.reconstruction.reconstructor import ArticleReconstructor
    from shared.reconstruction.types import ReconstructionConfig

    layout_available = True
except ImportError as e:
    print(f"Warning: Could not import shared modules: {e}")
    layout_available = False

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(colors=True)
        if os.getenv("LOG_FORMAT") == "console"
        else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Magazine PDF Extractor - Model Service",
        description="ML model service for PDF layout analysis, OCR, and article reconstruction with real business logic",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
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
        logger.info("Model Service starting up with real business logic integration")

    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Model Service shutting down")

    # Health check endpoints
    @app.get("/health/")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "model-service",
            "version": "1.0.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "production" if layout_available else "fallback",
            "business_logic_available": layout_available,
        }

    @app.get("/health/detailed")
    async def detailed_health_check():
        """Detailed health check with mock model status."""
        return {
            "status": "healthy",
            "service": "model-service",
            "version": "1.0.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "production" if layout_available else "fallback",
            "models": {
                "layout_analyzer": {
                    "status": "available" if layout_available else "unavailable",
                    "version": "LayoutAnalyzer-v1.0",
                },
                "ocr_processor": {
                    "status": "available" if layout_available else "unavailable",
                    "version": "OCREngine-v1.0",
                },
                "article_reconstructor": {
                    "status": "available" if layout_available else "unavailable",
                    "version": "ArticleReconstructor-v1.0",
                },
            },
            "device": "cpu",
            "business_logic_integration": "enabled" if layout_available else "disabled",
            "shared_modules_loaded": layout_available,
        }

    # Layout Analysis endpoint
    @app.post("/layout/analyze")
    async def analyze_layout(request: Request):
        """Analyze PDF layout using real LayoutAnalyzer from shared modules."""
        request_data = await request.json()

        logger.info(
            "Layout analysis requested",
            correlation_id=request.headers.get("x-correlation-id"),
        )

        if not layout_available:
            # Fallback to mock if shared modules not available
            await asyncio.sleep(random.uniform(1, 3))
            return {
                "job_id": request_data.get("job_id"),
                "error": "Layout analysis modules not available",
                "page_count": 0,
                "layout_confidence": 0.0,
                "blocks": [],
                "processing_time_ms": 0,
            }

        try:
            # Get PDF path from request
            pdf_path_str = request_data.get("pdf_path")
            if not pdf_path_str:
                raise HTTPException(status_code=400, detail="pdf_path is required")

            pdf_path = Path(pdf_path_str)
            if not pdf_path.exists():
                raise HTTPException(
                    status_code=404, detail=f"PDF file not found: {pdf_path}"
                )

            # Initialize layout analyzer with configuration
            config = LayoutConfig.create_default()
            analyzer = LayoutAnalyzer(config)

            # Get page range if specified
            page_range = request_data.get("page_range")
            if page_range:
                page_range = (page_range.get("start", 1), page_range.get("end", None))

            # Perform real layout analysis
            layout_result = analyzer.analyze_pdf(pdf_path, page_range)

            # Transform result to API format
            blocks = []
            block_id = 0

            for page in layout_result.pages:
                for text_block in page.text_blocks:
                    block_id += 1
                    blocks.append(
                        {
                            "block_id": f"block_{block_id}",
                            "type": text_block.classification.value
                            if text_block.classification
                            else "text",
                            "text": text_block.text,
                            "bbox": [
                                text_block.bbox.x0,
                                text_block.bbox.y0,
                                text_block.bbox.x1,
                                text_block.bbox.y1,
                            ],
                            "confidence": text_block.classification_confidence or 0.9,
                            "page_num": text_block.page_num,
                            "font_size": text_block.font_size,
                            "font_family": text_block.font_family,
                            "is_bold": text_block.is_bold,
                            "is_italic": text_block.is_italic,
                        }
                    )

            # Calculate overall confidence from blocks
            if blocks:
                layout_confidence = sum(block["confidence"] for block in blocks) / len(
                    blocks
                )
            else:
                layout_confidence = 0.0

            result = {
                "job_id": request_data.get("job_id"),
                "page_count": layout_result.page_count,
                "layout_confidence": layout_confidence,
                "blocks": blocks,
                "processing_time_ms": int(layout_result.total_processing_time * 1000),
                "analysis_config": layout_result.analysis_config,
            }

            logger.info(
                "Layout analysis completed",
                pdf_path=str(pdf_path),
                page_count=layout_result.page_count,
                total_blocks=len(blocks),
                processing_time=layout_result.total_processing_time,
            )

            return result

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Error in layout analysis", error=str(e), exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Layout analysis failed: {str(e)}"
            )

    # OCR Processing endpoint
    @app.post("/ocr/process")
    async def process_ocr(request: Request):
        """Process OCR on PDF using real OCREngine from shared modules."""
        request_data = await request.json()

        logger.info(
            "OCR processing requested",
            correlation_id=request.headers.get("x-correlation-id"),
        )

        if not layout_available:
            # Fallback to mock if shared modules not available
            await asyncio.sleep(random.uniform(2, 5))
            return {
                "job_id": request_data.get("job_id"),
                "error": "OCR processing modules not available",
                "ocr_confidence": 0.0,
                "text_blocks": [],
                "processing_time_ms": 0,
            }

        try:
            # Get PDF path from request
            pdf_path_str = request_data.get("pdf_path")
            if not pdf_path_str:
                raise HTTPException(status_code=400, detail="pdf_path is required")

            pdf_path = Path(pdf_path_str)
            if not pdf_path.exists():
                raise HTTPException(
                    status_code=404, detail=f"PDF file not found: {pdf_path}"
                )

            # Initialize OCR engine with configuration
            config = OCRConfig()
            brand = request_data.get("brand")  # Optional brand-specific config

            ocr_engine = OCREngine(config)

            # Get page range if specified
            page_range = request_data.get("page_range")
            if page_range:
                page_range = (page_range.get("start", 1), page_range.get("end", None))

            # Perform real OCR processing
            ocr_result = ocr_engine.process_pdf(pdf_path, brand, page_range)

            # Transform result to API format
            text_blocks = []
            block_id = 0

            for page in ocr_result.pages:
                for text_block in page.text_blocks:
                    block_id += 1

                    # Count words in the text
                    word_count = len(text_block.text.split()) if text_block.text else 0

                    text_blocks.append(
                        {
                            "block_id": f"text_block_{block_id}",
                            "text": text_block.text,
                            "confidence": text_block.confidence,
                            "language": "en",  # Could be extracted from OCR config
                            "word_count": word_count,
                            "bbox": text_block.bbox,
                            "page_num": page.page_num,
                            "block_type": text_block.block_type,
                            "lines": len(text_block.lines) if text_block.lines else 0,
                        }
                    )

            result = {
                "job_id": request_data.get("job_id"),
                "ocr_confidence": ocr_result.average_confidence,
                "text_blocks": text_blocks,
                "processing_time_ms": int(ocr_result.total_processing_time * 1000),
                "document_type": ocr_result.document_type.value,
                "total_words": ocr_result.total_words,
                "quality_metrics": {
                    "wer": ocr_result.quality_metrics.wer
                    if hasattr(ocr_result.quality_metrics, "wer")
                    else None,
                    "meets_target": ocr_result.quality_metrics.meets_wer_target
                    if hasattr(ocr_result.quality_metrics, "meets_wer_target")
                    else None,
                },
            }

            logger.info(
                "OCR processing completed",
                pdf_path=str(pdf_path),
                document_type=ocr_result.document_type.value,
                total_blocks=len(text_blocks),
                total_words=ocr_result.total_words,
                avg_confidence=ocr_result.average_confidence,
                processing_time=ocr_result.total_processing_time,
            )

            return result

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Error in OCR processing", error=str(e), exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"OCR processing failed: {str(e)}"
            )

    # Article Reconstruction endpoint
    @app.post("/articles/reconstruct")
    async def reconstruct_articles(request: Request):
        """Reconstruct articles using real ArticleReconstructor from shared modules."""
        request_data = await request.json()

        logger.info(
            "Article reconstruction requested",
            correlation_id=request.headers.get("x-correlation-id"),
        )

        if not layout_available:
            # Fallback to mock if shared modules not available
            await asyncio.sleep(random.uniform(1, 2))
            return {
                "job_id": request_data.get("job_id"),
                "error": "Article reconstruction modules not available",
                "articles": [],
                "overall_confidence": 0.0,
                "processing_time_ms": 0,
            }

        try:
            # For now, since we need a SemanticGraph as input and the graph construction
            # is complex, we'll implement a simplified version that processes PDF directly
            # In a full implementation, this would receive a pre-constructed graph

            pdf_path_str = request_data.get("pdf_path")
            semantic_graph_data = request_data.get(
                "semantic_graph"
            )  # Optional pre-built graph

            if not pdf_path_str and not semantic_graph_data:
                raise HTTPException(
                    status_code=400,
                    detail="Either pdf_path or semantic_graph data is required",
                )

            # Initialize reconstruction configuration
            config = ReconstructionConfig()
            max_articles = request_data.get("max_articles")

            reconstructor = ArticleReconstructor(config)

            if semantic_graph_data:
                # Use pre-built semantic graph (would need to deserialize)
                # For now, return error since graph deserialization is complex
                raise HTTPException(
                    status_code=501,
                    detail="Semantic graph input not yet implemented. Use pdf_path for direct processing.",
                )

            elif pdf_path_str:
                # For direct PDF processing, we need to build a semantic graph first
                # This is a complex process involving layout analysis + OCR + graph construction
                # For now, we'll create a simplified demonstration

                pdf_path = Path(pdf_path_str)
                if not pdf_path.exists():
                    raise HTTPException(
                        status_code=404, detail=f"PDF file not found: {pdf_path}"
                    )

                # Step 1: Perform layout analysis
                layout_config = LayoutConfig.create_default()
                layout_analyzer = LayoutAnalyzer(layout_config)
                layout_result = layout_analyzer.analyze_pdf(pdf_path)

                # Step 2: Perform OCR if needed
                ocr_config = OCRConfig()
                ocr_engine = OCREngine(ocr_config)
                ocr_result = ocr_engine.process_pdf(pdf_path)

                # Step 3: Build semantic graph (simplified - would use SemanticGraph factory)
                # For this implementation, we'll create a mock graph structure
                from shared.graph.factory import SemanticGraphFactory

                # Create graph from layout and OCR results
                graph_factory = SemanticGraphFactory()
                semantic_graph = graph_factory.create_from_analysis(
                    layout_result, ocr_result
                )

                # Step 4: Reconstruct articles
                reconstructed_articles = reconstructor.reconstruct_articles(
                    semantic_graph, max_articles
                )

                # Transform results to API format
                articles = []
                for article in reconstructed_articles:
                    # Extract contributors/byline from components
                    contributors = []
                    body_content = []

                    for component in article.components:
                        if component.get("block_type") == "byline":
                            contributors.append(
                                {
                                    "name": component.get("text", "Unknown Author"),
                                    "role": "author",
                                    "confidence": component.get("confidence", 0.8),
                                }
                            )
                        elif component.get("block_type") in ["body", "paragraph"]:
                            body_content.append(
                                {
                                    "content": component.get("text", ""),
                                    "confidence": component.get("confidence", 0.8),
                                }
                            )

                    articles.append(
                        {
                            "article_id": article.article_id,
                            "title": {
                                "content": article.title,
                                "confidence": article.reconstruction_confidence,
                            },
                            "body": {"content": body_content},
                            "contributors": contributors,
                            "page_range": [
                                article.boundary.start_page,
                                article.boundary.end_page,
                            ],
                            "accuracy_score": article.reconstruction_confidence,
                            "completeness_score": article.completeness_score,
                            "word_count": article.boundary.word_count,
                            "quality_issues": article.quality_issues,
                            "processing_time": article.processing_time,
                        }
                    )

                # Calculate overall confidence
                if articles:
                    overall_confidence = sum(
                        a["accuracy_score"] for a in articles
                    ) / len(articles)
                else:
                    overall_confidence = 0.0

                # Get total processing time
                processing_time_ms = int(
                    sum(a["processing_time"] for a in articles) * 1000
                )

                result = {
                    "job_id": request_data.get("job_id"),
                    "articles": articles,
                    "overall_confidence": overall_confidence,
                    "processing_time_ms": processing_time_ms,
                    "reconstruction_method": "graph_traversal",
                    "articles_found": len(articles),
                    "graph_stats": {
                        "nodes": semantic_graph.node_count,
                        "edges": semantic_graph.edge_count,
                    },
                }

                logger.info(
                    "Article reconstruction completed",
                    pdf_path=str(pdf_path),
                    articles_found=len(articles),
                    overall_confidence=overall_confidence,
                    processing_time_ms=processing_time_ms,
                )

                return result

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Error in article reconstruction", error=str(e), exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Article reconstruction failed: {str(e)}"
            )

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
