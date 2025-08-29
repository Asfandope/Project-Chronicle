FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY evaluation_service/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and required modules
COPY evaluation_service/ ./evaluation_service/
COPY synthetic_data/ ./synthetic_data/
COPY shared/ ./shared/
COPY database.py ./

# Create logs directory
RUN mkdir -p logs

# Create non-root user
RUN useradd -m -u 1000 evaluation && chown -R evaluation:evaluation /app
USER evaluation

# Expose port (changed to match ECS task definition)
EXPOSE 8002

# Health check (updated port)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Default command (updated port)
CMD ["uvicorn", "evaluation_service.main:app", "--host", "0.0.0.0", "--port", "8002"]