FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.7.1

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

# Copy Poetry configuration
COPY pyproject.toml poetry.lock* ./

# Development stage
FROM base as development

# Install all dependencies including dev dependencies
RUN poetry install --no-root && rm -rf $POETRY_CACHE_DIR

# Create directories
RUN mkdir -p /app/{data,logs,gold_sets}

# Set Python path
ENV PYTHONPATH=/app

# Production stage
FROM base as production

# Install only production dependencies
RUN poetry install --only=main --no-root && rm -rf $POETRY_CACHE_DIR

# Copy application code
COPY services/evaluation/ /app/evaluation/
COPY shared/ /app/shared/
COPY schemas/ /app/schemas/

# Create directories
RUN mkdir -p /app/{data,logs,gold_sets}

# Set Python path
ENV PYTHONPATH=/app

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8002

CMD ["uvicorn", "evaluation.main:app", "--host", "0.0.0.0", "--port", "8002"]