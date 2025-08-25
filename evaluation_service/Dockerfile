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
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy synthetic_data module (assuming it's in parent directory)
COPY ../synthetic_data ./synthetic_data

# Create logs directory
RUN mkdir -p logs

# Create non-root user
RUN useradd -m -u 1000 evaluation && chown -R evaluation:evaluation /app
USER evaluation

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "evaluation_service.main:app", "--host", "0.0.0.0", "--port", "8000"]