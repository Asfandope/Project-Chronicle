# Magazine PDF Extractor

A fully automated system that extracts articles from heterogeneous magazine/newspaper PDFs into canonical XML with 99.9% field-level accuracy, featuring self-healing capabilities and zero human intervention.

## 🎯 Key Features

- **Dual-Pass Architecture**: Layout-aware language models create semantic graphs, then traverse them to reconstruct articles
- **Generalist-First Model Strategy**: Master generalist model handles unknown publications, with specialist models for enhanced accuracy on critical brands
- **99.9% Accuracy**: Field-level accuracy with automatic quality assurance and three-tier model fallback
- **Self-Healing**: Continuous evaluation with synthetic gold sets and automatic fine-tuning when drift is detected
- **Zero Human Touch**: No manual QA or approval steps; outputs are automatically quarantined if accuracy drops below threshold
- **Universal Processing**: Handles both known and unknown publications through intelligent model selection and YAML configuration
- **CPU/GPU Flexible**: Designed for both CPU and GPU deployment

## 🏗️ Architecture

The system consists of three core services:

### 1. Orchestrator Service (`services/orchestrator/`)
- **Purpose**: Central workflow coordination and job management
- **Tech Stack**: FastAPI, SQLAlchemy, Celery, PostgreSQL, Redis
- **Responsibilities**:
  - Job queue management and workflow state
  - PDF ingestion and validation
  - Inter-service communication
  - Export pipeline (XML/CSV generation)

### 2. Model Service (`services/model_service/`)
- **Purpose**: AI/ML models for content extraction
- **Tech Stack**: FastAPI, PyTorch, Transformers, Tesseract, OpenCV
- **Responsibilities**:
  - **Intelligent model selection**: Three-tier hierarchy (brand-specific → generalist → base LayoutLM)
  - Layout analysis using LayoutLMv3-Large with brand-optimized fine-tuning
  - OCR processing (born-digital + scanned)
  - Article reconstruction via graph traversal
  - Contributor extraction with NER
  - Image extraction and caption linking
- **Model Architecture**:
  - **Specialist models**: Brand-specific fine-tuned LayoutLM for critical publications
  - **Generalist model**: Master model trained on diverse publications for unknown brands
  - **Automatic fallback**: Ensures robust processing regardless of publication type

### 3. Evaluation Service (`services/evaluation/`)
- **Purpose**: Accuracy evaluation, drift detection, and auto-tuning
- **Tech Stack**: FastAPI, SQLAlchemy, PostgreSQL, NumPy, Pandas
- **Responsibilities**:
  - Accuracy evaluation against gold standards
  - Drift detection and trend analysis
  - Auto-tuning parameter optimization
  - Gold set management and synthetic augmentation

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Poetry 1.7+
- Docker & Docker Compose
- PostgreSQL 15+
- Redis 7+
- Tesseract OCR

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd magazine-pdf-extractor
   ```

2. **Install dependencies**
   ```bash
   poetry install
   ```

3. **Set up pre-commit hooks**
   ```bash
   poetry run pre-commit install
   ```

4. **Start services with Docker Compose**
   ```bash
   docker-compose up -d
   ```

5. **Initialize the database**
   ```bash
   poetry run alembic upgrade head
   ```

6. **Verify setup**
   ```bash
   curl http://localhost:8000/health/detailed
   curl http://localhost:8001/health/detailed
   curl http://localhost:8002/health/detailed
   ```

### Using the API

Upload and process a PDF:

```bash
curl -X POST "http://localhost:8000/api/v1/jobs/" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample_magazine.pdf" \
     -F "brand=economist"
```

Monitor job progress:

```bash
curl "http://localhost:8000/api/v1/jobs/{job_id}"
```

## 🚀 Model Training

The system provides comprehensive model training capabilities for both specialist and generalist models:

### Generalist Model Training
Train a master model on all brand data for handling unknown publications:

```bash
# Train generalist model on combined dataset
make train-generalist

# This aggregates data from all brands and trains for 15 epochs
# Model saved to: models/fine_tuned/generalist/
```

### Brand-Specific Training
Train specialist models for enhanced accuracy on critical publications:

```bash
# Train individual brand models
make train-brand BRAND=economist
make train-brand BRAND=time
make train-brand BRAND=newsweek
make train-brand BRAND=vogue

# View all training experiments
make training-summary
```

### Model Management
The system automatically selects the best available model for each publication:

1. **Brand Identified** → Load specialist model (highest accuracy)
2. **Unknown Brand** → Load generalist model (robust cross-publication)  
3. **Fallback** → Base LayoutLMv3-Large model

## 📋 Processing Pipeline

1. **Ingestion**: PDF validation and metadata extraction
2. **Brand Detection**: Filename analysis and content-based identification
3. **Model Selection**: Intelligent three-tier model hierarchy
4. **Preprocessing**: Page splitting and preparation
5. **Layout Analysis**: LayoutLM-based block classification and semantic graph creation
4. **OCR**: Text extraction (direct from born-digital, Tesseract for scanned)
5. **Article Reconstruction**: Graph traversal to rebuild complete articles
6. **Contributor Parsing**: NER-based name and role extraction
7. **Image Extraction**: Image extraction with spatial caption linking
8. **Export**: XML/CSV generation following canonical schema
9. **Evaluation**: Accuracy assessment and drift detection

## 🎯 Accuracy & Quality

### Accuracy Definition
- **Title match**: 30% weight (exact match after normalization)
- **Body text**: 40% weight (WER < 0.1%)
- **Contributors**: 20% weight (name + role correct)
- **Media links**: 10% weight (correct image-caption pairs)

### Quality Thresholds
- **Issue Pass**: ≥ 99.9% weighted accuracy
- **Brand Pass**: 95% of last 10 issues pass
- **Auto-tuning Trigger**: 2 consecutive issues below 99.9% OR brand pass rate < 95%

## 🛠️ Configuration

### Brand Configuration Example (`configs/brands/economist.yaml`)

```yaml
brand: economist
layout_hints:
  column_count: [2, 3]
  title_patterns: ["^[A-Z][a-z]+.*", "^The.*"]
  jump_indicators: ["continued on page", "from page"]
ocr_preprocessing:
  deskew: true
  denoise_level: 2
confidence_overrides:
  title: 0.95
  body: 0.92
```

### Processing Configuration (`configs/processing.yaml`)

Global settings for OCR, layout analysis, article reconstruction, and quality thresholds.

## 🔄 Self-Healing System

### Drift Detection
- Evaluates every processed issue against synthetic gold set
- Rolling 10-issue window tracks accuracy trends
- Triggers auto-tuning on accuracy degradation

### Auto-Tuning Process
1. **Isolate failing patterns** from quarantined issues
2. **Generate targeted synthetic examples**
3. **Tune parameters** on synthetic set
4. **Validate** on holdout set
5. **Deploy** if accuracy improves, rollback if not

### Tunable Parameters
- Graph traversal rules (YAML configs)
- Confidence thresholds per field
- Block classifier prompts
- OCR preprocessing parameters

## 📊 Monitoring

### Metrics Available
- **Accuracy per brand/issue**
- **Processing time and throughput**
- **Queue depth and health status**
- **Drift detection alerts**
- **Auto-tuning events**

### Health Endpoints
- `/health` - Basic health check
- `/health/detailed` - Full dependency status

## 🧪 Testing

```bash
# Run all tests
poetry run pytest

# Run specific service tests
poetry run pytest tests/orchestrator/
poetry run pytest tests/model_service/
poetry run pytest tests/evaluation/

# Run integration tests
poetry run pytest tests/integration/

# Generate coverage report
poetry run pytest --cov=services --cov-report=html
```

## 🚢 Deployment

### Docker Deployment

```bash
# Build production images
docker build --target production-cpu -t magazine-extractor-orchestrator .
docker build --target production-gpu -t magazine-extractor-model-service .
docker build --target production-cpu -t magazine-extractor-evaluation .

# Deploy with docker-compose
docker-compose -f docker-compose.prod.yml up -d
```

### Kubernetes Deployment

See `k8s/` directory for Kubernetes manifests and Helm charts.

### Environment Configuration

Key environment variables:

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/magazine_extractor

# Redis/Celery
REDIS_URL=redis://localhost:6379

# Service URLs
MODEL_SERVICE_URL=http://model-service:8001
EVALUATION_SERVICE_URL=http://evaluation:8002

# Processing
DEVICE=cuda  # or cpu
MAX_CONCURRENT_JOBS=4
```

## 📈 Performance

### Throughput
- **50 pages/minute** on single GPU
- **<5 minutes** for 100-page issue
- **Linear scaling** with horizontal deployment

### Resource Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: Optional but provides 10x speedup
- **Storage**: ~50MB per issue (images + XML)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`poetry run pytest`)
5. Run pre-commit hooks (`poetry run pre-commit run --all-files`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Style

- **Python**: Black formatting, isort imports, flake8 linting
- **Type Hints**: mypy type checking required
- **Documentation**: Docstrings for all public functions
- **Testing**: Pytest with >90% coverage requirement

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: See individual service READMEs in `services/*/README.md`
- **Issues**: Report bugs and feature requests via GitHub Issues
- **Security**: Report security vulnerabilities via GitHub Security Advisories

## 🗺️ Roadmap

### v1.1 (Next Release)
- [ ] Multi-language support
- [ ] Advanced layout detection for artistic layouts
- [ ] Real-time processing mode
- [ ] Web UI for job monitoring

### v1.2 (Future)
- [ ] Video/audio content extraction
- [ ] Historical archive migration tools
- [ ] Advanced analytics dashboard
- [ ] Cloud deployment templates

---

**Built with ❤️ for accurate, automated content extraction**