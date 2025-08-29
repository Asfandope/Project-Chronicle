FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry with reduced memory footprint
RUN pip install --no-cache-dir poetry==1.7.1

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    POETRY_INSTALLER_MAX_WORKERS=1

WORKDIR /app

# Copy Poetry configuration
COPY pyproject.toml poetry.lock* ./

# Development stage
FROM base as development

# Install all dependencies including dev dependencies
RUN poetry config installer.max-workers 1 && \
    poetry install --no-root && \
    rm -rf $POETRY_CACHE_DIR

# Create directories
RUN mkdir -p /app/{data,logs,outputs,quarantine,temp}

# Set Python path
ENV PYTHONPATH=/app

# Production stage
FROM base as production

# Ensure Poetry creates venv in project
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Install only production dependencies
RUN poetry config installer.max-workers 1 && \
    poetry install --only=main --no-root && \
    rm -rf $POETRY_CACHE_DIR

# Copy application code
COPY services/orchestrator/ /app/orchestrator/
COPY shared/ /app/shared/

# Create directories
RUN mkdir -p /app/{data,logs,outputs,quarantine,temp}

# Set Python path and add Poetry venv to PATH
ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "orchestrator.main:app", "--host", "0.0.0.0", "--port", "8000"]