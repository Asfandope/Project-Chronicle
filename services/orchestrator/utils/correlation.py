"""
Correlation ID middleware for request tracking across services.
Provides unique request identifiers for distributed tracing.
"""

import uuid
from typing import Callable

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger(__name__)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """
    Middleware to generate and propagate correlation IDs for request tracking.

    Correlation IDs help track requests across multiple services and components,
    making it easier to debug issues and analyze request flows.
    """

    def __init__(
        self,
        app: Callable,
        header_name: str = "X-Correlation-ID",
        generate_if_missing: bool = True,
    ):
        super().__init__(app)
        self.header_name = header_name
        self.generate_if_missing = generate_if_missing

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and inject correlation ID."""

        # Try to get correlation ID from request headers
        correlation_id = request.headers.get(self.header_name)

        # Generate new correlation ID if missing and configured to do so
        if not correlation_id and self.generate_if_missing:
            correlation_id = str(uuid.uuid4())

        # Store correlation ID in request state for access in route handlers
        if correlation_id:
            request.state.correlation_id = correlation_id

            # Bind correlation ID to structlog context
            structlog.contextvars.bind_contextvars(correlation_id=correlation_id)

        try:
            # Process the request
            response = await call_next(request)

            # Add correlation ID to response headers
            if correlation_id:
                response.headers[self.header_name] = correlation_id

            return response

        finally:
            # Clear the correlation ID from context after request
            if correlation_id:
                structlog.contextvars.clear_contextvars()


def get_correlation_id(request: Request) -> str:
    """
    Extract correlation ID from request state.

    Args:
        request: FastAPI request object

    Returns:
        Correlation ID string, or generates a new one if missing
    """
    return getattr(request.state, "correlation_id", str(uuid.uuid4()))


def propagate_correlation_id(correlation_id: str = None) -> dict:
    """
    Get headers to propagate correlation ID to downstream services.

    Args:
        correlation_id: Optional correlation ID, generates new one if None

    Returns:
        Dictionary with correlation ID header
    """
    if not correlation_id:
        correlation_id = str(uuid.uuid4())

    return {"X-Correlation-ID": correlation_id}
