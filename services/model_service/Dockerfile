FROM python:3.11-slim as base

# Install system dependencies in stages for better error handling
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install ML/OCR dependencies in separate stage
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Install computer vision dependencies with correct package names for Debian Trixie
RUN apt-get update && apt-get install -y --fix-missing \
    poppler-utils \
    libgl1-mesa-dri \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
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
RUN mkdir -p /app/{models,temp,data}

# Set Python path
ENV PYTHONPATH=/app

# Production CPU stage
FROM base as production-cpu

# Ensure Poetry creates venv in project
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Install only production dependencies  
RUN poetry config virtualenvs.create true && \
    poetry config virtualenvs.in-project true && \
    poetry install --only=main --no-root && \
    ls -la /app/.venv/bin/ && \
    rm -rf $POETRY_CACHE_DIR

# Copy application code
COPY services/model_service/ /app/model_service/
COPY shared/ /app/shared/

# Create directories
RUN mkdir -p /app/{models,temp,data}

# Set Python path and add Poetry venv to PATH
ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8001

CMD ["uvicorn", "model_service.main:app", "--host", "0.0.0.0", "--port", "8001"]

# Production GPU stage
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 as gpu

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    curl \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Install Poetry
RUN pip install poetry==1.7.1

WORKDIR /app

# Copy Poetry configuration
COPY pyproject.toml poetry.lock* ./

# Configure Poetry - MUST be set before install
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Install production dependencies with GPU support
RUN poetry config virtualenvs.create true && \
    poetry config virtualenvs.in-project true && \
    poetry install --only=main,gpu --no-root && \
    ls -la /app/.venv/bin/ && \
    rm -rf $POETRY_CACHE_DIR

# Copy application code
COPY services/model_service/ /app/model_service/
COPY shared/ /app/shared/

# Create directories
RUN mkdir -p /app/{models,temp,data}

# Set Python path and add Poetry venv to PATH
ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8001

CMD ["uvicorn", "model_service.main:app", "--host", "0.0.0.0", "--port", "8001"]