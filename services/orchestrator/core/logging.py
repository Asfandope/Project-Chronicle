"""
Structured logging configuration for the Orchestrator service.
Provides consistent, JSON-structured logging with correlation ID support.
"""

import logging
import sys
from typing import Any, Dict
import structlog
from structlog.typing import FilteringBoundLogger

from orchestrator.core.config import get_settings

def configure_logging() -> None:
    """Configure structured logging for the application."""
    settings = get_settings()
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )
    
    # Set log levels for noisy libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # Configure processors based on format
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        filter_by_level,
    ]
    
    if settings.log_format == "console":
        # Human-readable console output for development
        processors.extend([
            structlog.dev.ConsoleRenderer(colors=True),
        ])
    else:
        # JSON output for production
        processors.extend([
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ])
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper())
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

def filter_by_level(
    logger: FilteringBoundLogger, method_name: str, event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Filter log entries by level."""
    return event_dict

def get_logger(name: str = None) -> FilteringBoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)

# Correlation ID context management
def bind_correlation_id(correlation_id: str) -> None:
    """Bind correlation ID to the logging context."""
    structlog.contextvars.bind_contextvars(correlation_id=correlation_id)

def clear_correlation_id() -> None:
    """Clear correlation ID from logging context."""
    structlog.contextvars.clear_contextvars()

# Logging utilities for common patterns
def log_job_event(
    logger: FilteringBoundLogger,
    event: str,
    job_id: str,
    **kwargs
) -> None:
    """Log a job-related event with consistent formatting."""
    logger.info(
        event,
        job_id=job_id,
        event_type="job",
        **kwargs
    )

def log_processing_stage(
    logger: FilteringBoundLogger,
    stage: str,
    job_id: str,
    status: str,
    **kwargs
) -> None:
    """Log processing stage information."""
    logger.info(
        f"Processing stage {status}",
        job_id=job_id,
        stage=stage,
        status=status,
        event_type="processing",
        **kwargs
    )

def log_error_with_context(
    logger: FilteringBoundLogger,
    error: Exception,
    context: Dict[str, Any]
) -> None:
    """Log an error with additional context."""
    logger.error(
        "Error occurred",
        error=str(error),
        error_type=type(error).__name__,
        **context,
        exc_info=True
    )