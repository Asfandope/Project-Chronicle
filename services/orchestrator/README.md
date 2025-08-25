# Orchestrator Service

The Orchestrator Service is the central coordination hub for the Magazine PDF Extractor system. It manages job queues, workflow state, and inter-service communication.

## 🎯 Purpose

- **Job Management**: Create, track, and manage PDF processing jobs
- **Workflow Orchestration**: Coordinate the execution of processing stages across services
- **API Gateway**: Primary entry point for external clients
- **Export Pipeline**: Generate final XML and CSV outputs
- **File Management**: Handle PDF ingestion and output organization

## 🏗️ Architecture

### Core Components

- **FastAPI Application** (`main.py`): HTTP API server
- **Celery Worker** (`celery_app.py`): Asynchronous task processing
- **Workflow Engine** (`core/workflow.py`): Manages processing stages and dependencies
- **Database Models** (`models/`): Job state and processing data
- **API Endpoints** (`api/`): REST API for job management

### Database Schema

#### Jobs Table
- `id`: Unique job identifier (UUID)
- `filename`: Original PDF filename
- `file_path`: Path to uploaded PDF
- `brand`: Magazine/newspaper brand
- `overall_status`: Current job status
- `workflow_steps`: JSON containing stage-by-stage progress
- `accuracy_score`: Final accuracy evaluation
- `output_paths`: Generated XML, CSV, and image paths

#### Processing States Table
- `job_id`: Reference to jobs table
- `stage`: Current processing stage
- `stage_data`: Stage-specific processing results
- `semantic_graph`: Layout analysis results
- `reconstructed_articles`: Final article data

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Poetry

### Setup

1. **Install dependencies**
   ```bash
   poetry install
   ```

2. **Environment configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your database and service URLs
   ```

3. **Database setup**
   ```bash
   poetry run alembic upgrade head
   ```

4. **Start the service**
   ```bash
   # API server
   poetry run uvicorn orchestrator.main:app --reload --host 0.0.0.0 --port 8000
   
   # Celery worker (separate terminal)
   poetry run celery -A orchestrator.celery_app worker --loglevel=info
   ```

### Docker Development

```bash
docker-compose up orchestrator orchestrator-worker
```

## 📚 API Reference

### Job Management

#### Create Job
```http
POST /api/v1/jobs/
Content-Type: multipart/form-data

{
  "file": <PDF file>,
  "brand": "economist"
}
```

#### Get Job Status
```http
GET /api/v1/jobs/{job_id}
```

#### List Jobs
```http
GET /api/v1/jobs/?status=completed&brand=economist&skip=0&limit=100
```

#### Retry Job
```http
POST /api/v1/jobs/{job_id}/retry
```

### Configuration Management

#### Get Brand Configuration
```http
GET /api/v1/config/brands/economist
```

#### Update Brand Configuration
```http
PUT /api/v1/config/brands/economist
Content-Type: application/json

{
  "layout_hints": {...},
  "ocr_preprocessing": {...}
}
```

### Health Checks

#### Basic Health
```http
GET /health/
```

#### Detailed Health
```http
GET /health/detailed
```

## ⚙️ Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/magazine_extractor

# Redis/Celery
REDIS_URL=redis://localhost:6379

# External Services
MODEL_SERVICE_URL=http://localhost:8001
EVALUATION_SERVICE_URL=http://localhost:8002

# File Processing
INPUT_DIRECTORY=data/input
OUTPUT_DIRECTORY=data/output
QUARANTINE_DIRECTORY=data/quarantine

# Processing Limits
MAX_FILE_SIZE_MB=100
MAX_PAGES_PER_ISSUE=500
PROCESSING_TIMEOUT_MINUTES=30

# Quality Thresholds
ACCURACY_THRESHOLD=0.999
QUARANTINE_THRESHOLD=0.95
```

### Workflow Configuration

The workflow engine manages these processing stages:

1. **INGESTION**: PDF validation and metadata extraction
2. **PREPROCESSING**: Page splitting and preparation
3. **LAYOUT_ANALYSIS**: Layout model inference
4. **OCR**: Text extraction
5. **ARTICLE_RECONSTRUCTION**: Article assembly
6. **CONTRIBUTOR_PARSING**: Name and role extraction
7. **IMAGE_EXTRACTION**: Image processing
8. **EXPORT**: XML/CSV generation
9. **EVALUATION**: Quality assessment

## 🔄 Workflow Engine

### Stage Dependencies

```python
STAGE_DEPENDENCIES = {
    WorkflowStage.PREPROCESSING: [WorkflowStage.INGESTION],
    WorkflowStage.LAYOUT_ANALYSIS: [WorkflowStage.PREPROCESSING],
    WorkflowStage.OCR: [WorkflowStage.LAYOUT_ANALYSIS], 
    WorkflowStage.ARTICLE_RECONSTRUCTION: [WorkflowStage.OCR, WorkflowStage.LAYOUT_ANALYSIS],
    WorkflowStage.CONTRIBUTOR_PARSING: [WorkflowStage.ARTICLE_RECONSTRUCTION],
    WorkflowStage.IMAGE_EXTRACTION: [WorkflowStage.LAYOUT_ANALYSIS],
    WorkflowStage.EXPORT: [WorkflowStage.ARTICLE_RECONSTRUCTION, WorkflowStage.CONTRIBUTOR_PARSING, WorkflowStage.IMAGE_EXTRACTION],
    WorkflowStage.EVALUATION: [WorkflowStage.EXPORT],
    WorkflowStage.COMPLETED: [WorkflowStage.EVALUATION]
}
```

### Error Handling

- **Retry Logic**: Failed stages retry up to 3 times with exponential backoff
- **Quarantine**: Jobs with accuracy below threshold are quarantined
- **Rollback**: Ability to rollback to previous processing state

## 🧪 Testing

### Unit Tests
```bash
poetry run pytest tests/orchestrator/unit/ -v
```

### Integration Tests
```bash
# Requires running PostgreSQL and Redis
poetry run pytest tests/orchestrator/integration/ -v
```

### API Tests
```bash
poetry run pytest tests/orchestrator/api/ -v
```

## 📊 Monitoring

### Metrics

The service exposes Prometheus metrics:
- `jobs_created_total`: Total jobs created
- `jobs_completed_total`: Total jobs completed
- `jobs_failed_total`: Total jobs failed
- `processing_time_seconds`: Job processing duration
- `accuracy_score`: Job accuracy distribution

### Logging

Structured logging with correlation IDs:
```python
logger.info("Job created", job_id=job.id, filename=job.filename)
logger.error("Processing failed", job_id=job.id, stage="layout_analysis", error=str(e))
```

### Health Monitoring

Health endpoints check:
- Database connectivity
- Redis availability
- External service health
- Celery worker status
- File system access

## 🚀 Deployment

### Production Configuration

```yaml
# docker-compose.prod.yml
services:
  orchestrator:
    build:
      context: .
      dockerfile: services/orchestrator/Dockerfile
      target: production
    environment:
      - DEBUG=false
      - LOG_LEVEL=INFO
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Scaling Considerations

- **Horizontal Scaling**: Multiple orchestrator instances can run concurrently
- **Celery Workers**: Scale workers based on job volume
- **Database**: Use connection pooling and read replicas for high load
- **File Storage**: Use shared storage (NFS/S3) for multi-instance deployments

## 🔧 Troubleshooting

### Common Issues

#### Jobs Stuck in Processing
```bash
# Check Celery worker status
poetry run celery -A orchestrator.celery_app inspect active

# Check database for stuck jobs
psql -c "SELECT id, filename, overall_status, created_at FROM jobs WHERE overall_status = 'in_progress' AND created_at < NOW() - INTERVAL '1 hour';"
```

#### High Memory Usage
```bash
# Monitor worker memory
poetry run celery -A orchestrator.celery_app inspect stats

# Restart workers if needed
poetry run celery -A orchestrator.celery_app control shutdown
```

#### Database Connection Issues
```bash
# Test database connectivity
python -c "from orchestrator.core.database import engine; engine.execute('SELECT 1')"

# Check connection pool
SELECT * FROM pg_stat_activity WHERE datname = 'magazine_extractor';
```

### Debug Mode

Enable debug mode for detailed logging:
```bash
export DEBUG=true
export LOG_LEVEL=DEBUG
poetry run uvicorn orchestrator.main:app --reload
```

## 📝 Development

### Adding New Workflow Stages

1. **Define stage enum** in `core/workflow.py`
2. **Add stage dependencies** in `STAGE_DEPENDENCIES`
3. **Implement stage logic** in `WorkflowExecutor._execute_stage()`
4. **Add tests** for the new stage
5. **Update API documentation**

### Adding New API Endpoints

1. **Create endpoint** in appropriate `api/` module
2. **Add request/response schemas** in `shared/schemas/`
3. **Include router** in `main.py`
4. **Add tests** in `tests/orchestrator/api/`
5. **Update OpenAPI documentation**

### Database Migrations

```bash
# Create new migration
poetry run alembic revision --autogenerate -m "Add new table"

# Apply migrations
poetry run alembic upgrade head

# Rollback migration
poetry run alembic downgrade -1
```

---

**For system-wide documentation, see the main [README.md](../../README.md)**