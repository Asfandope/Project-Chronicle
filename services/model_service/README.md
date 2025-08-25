# Model Service

The Model Service hosts all AI/ML models for PDF content extraction, including layout analysis, OCR, article reconstruction, contributor extraction, and image processing.

## 🎯 Purpose

- **Layout Analysis**: LayoutLM-based document understanding and semantic graph creation
- **OCR Processing**: Text extraction from both born-digital and scanned PDFs
- **Article Reconstruction**: Graph traversal algorithms to rebuild complete articles
- **Contributor Extraction**: NER-based name and role identification
- **Image Processing**: Image extraction and caption linking using spatial proximity

## 🏗️ Architecture

### Core Components

- **Model Manager** (`core/model_manager.py`): Manages loading and caching of ML models
- **Layout Analyzer** (`models/layout_analyzer.py`): Document layout understanding
- **OCR Processor** (`models/ocr_processor.py`): Text extraction with Tesseract integration
- **Article Reconstructor** (`models/article_reconstructor.py`): Graph-based article assembly
- **Contributor Extractor** (`models/contributor_extractor.py`): NER and name normalization
- **Image Extractor** (`models/image_extractor.py`): Image processing and caption linking

### Model Pipeline

1. **Layout Analysis** → Semantic graph of page elements
2. **OCR Processing** → Text extraction with confidence scores
3. **Article Reconstruction** → Complete article assembly via graph traversal
4. **Contributor Parsing** → Author, photographer, illustrator identification
5. **Image Extraction** → Image-caption pairing using spatial algorithms

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- PyTorch 2.1+
- Tesseract OCR 5.0+
- CUDA (optional, for GPU acceleration)

### Setup

1. **Install dependencies**
   ```bash
   poetry install
   
   # For GPU support
   poetry install --extras gpu
   ```

2. **Install Tesseract**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr tesseract-ocr-eng libtesseract-dev
   
   # macOS
   brew install tesseract
   ```

3. **Environment configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Start the service**
   ```bash
   poetry run uvicorn model_service.main:app --reload --host 0.0.0.0 --port 8001
   ```

### Docker Development

```bash
# CPU version
docker-compose up model-service

# GPU version (requires nvidia-docker)
docker-compose up model-service-gpu
```

## 📚 API Reference

### Layout Analysis

#### Analyze PDF Layout
```http
POST /api/v1/layout/analyze
Content-Type: application/json

{
  "job_id": "uuid",
  "file_path": "/path/to/pdf",
  "brand_config": {...}
}
```

#### Classify Blocks
```http
POST /api/v1/layout/classify-blocks
Content-Type: application/json

{
  "blocks": {...}
}
```

### OCR Processing

#### Process OCR
```http
POST /api/v1/ocr/process
Content-Type: application/json

{
  "job_id": "uuid",
  "brand_config": {...}
}
```

#### Extract Text from Blocks
```http
POST /api/v1/ocr/extract-text
Content-Type: application/json

{
  "blocks": {...}
}
```

### Article Reconstruction

#### Reconstruct Articles
```http
POST /api/v1/articles/reconstruct
Content-Type: application/json

{
  "job_id": "uuid",
  "brand_config": {...}
}
```

#### Identify Boundaries
```http
POST /api/v1/articles/identify-boundaries
Content-Type: application/json

{
  "semantic_graph": {...}
}
```

### Contributor Extraction

#### Extract Contributors
```http
POST /api/v1/contributors/extract
Content-Type: application/json

{
  "job_id": "uuid",
  "brand_config": {...}
}
```

#### Normalize Names
```http
POST /api/v1/contributors/normalize-names
Content-Type: application/json

{
  "names": {...}
}
```

### Image Processing

#### Extract Images
```http
POST /api/v1/images/extract
Content-Type: application/json

{
  "job_id": "uuid",
  "min_size": [100, 100]
}
```

#### Link Captions
```http
POST /api/v1/images/link-captions
Content-Type: application/json

{
  "images": {...},
  "captions": {...}
}
```

## 🤖 Models

### Layout Analysis Model

- **Model**: Microsoft LayoutLM-v3
- **Purpose**: Document layout understanding and block classification
- **Input**: PDF page images + OCR text
- **Output**: Semantic blocks with bounding boxes and classifications
- **GPU Memory**: ~2GB

### NER Model

- **Model**: BERT Large (CoNLL-03 fine-tuned)
- **Purpose**: Person name extraction from bylines and credits
- **Input**: Text segments
- **Output**: Named entities with confidence scores
- **GPU Memory**: ~1.5GB

### OCR Engine

- **Engine**: Tesseract 5.0+ with LSTM
- **Languages**: English (configurable)
- **Modes**: Born-digital extraction + scanned image OCR
- **Preprocessing**: Deskewing, denoising, contrast enhancement

## ⚙️ Configuration

### Environment Variables

```bash
# Device configuration
DEVICE=cuda  # or cpu
MODEL_CACHE_DIR=models/

# Model settings
LAYOUT_MODEL_NAME=microsoft/layoutlm-v3-base
NER_MODEL_NAME=dbmdz/bert-large-cased-finetuned-conll03-english
BATCH_SIZE=8

# Processing parameters
MAX_IMAGE_SIZE=2048
OCR_CONFIDENCE_THRESHOLD=0.7
LAYOUT_CONFIDENCE_THRESHOLD=0.8

# Tesseract settings
TESSERACT_CONFIG="--oem 3 --psm 6"

# Performance
MODEL_LOADING_TIMEOUT=300
INFERENCE_TIMEOUT=120
```

### Model Configuration

Models are automatically downloaded on first use and cached locally:

```python
# Model cache structure
models/
├── layoutlm-v3/
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer/
├── bert-ner/
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer/
└── cache_info.json
```

## 🧠 Model Details

### Layout Analyzer

The layout analyzer uses LayoutLM-v3 to understand document structure:

```python
class LayoutAnalyzer:
    def analyze_pdf_layout(self, job_id, file_path, brand_config):
        # 1. Extract text and visual features
        # 2. Run LayoutLM inference
        # 3. Classify blocks (title, body, caption, etc.)
        # 4. Build semantic graph with spatial relationships
        # 5. Return structured results with confidence scores
```

**Supported Block Types**:
- `title`: Article titles
- `body`: Body text paragraphs
- `caption`: Image/figure captions
- `pullquote`: Highlighted quotes
- `header`: Page headers
- `footer`: Page footers
- `ad`: Advertisement blocks

### OCR Processor

Handles both direct text extraction and OCR processing:

```python
class OCRProcessor:
    def process_pdf_ocr(self, job_id, brand_config):
        # Born-digital PDFs: Direct text extraction
        # Scanned PDFs: Tesseract with preprocessing
        # Brand-specific preprocessing pipeline
        # Confidence scoring per block
```

**Preprocessing Pipeline**:
1. **Deskewing**: Correct image rotation
2. **Denoising**: Remove image artifacts
3. **Contrast Enhancement**: Improve text clarity
4. **Tesseract OCR**: LSTM-based text recognition

### Article Reconstructor

Uses graph traversal to rebuild complete articles:

```python
class ArticleReconstructor:
    def reconstruct_articles(self, job_id, brand_config):
        # 1. Load semantic graph
        # 2. Find article start points (titles)
        # 3. Traverse graph following content relationships
        # 4. Handle cross-page articles and jump references
        # 5. Stitch split articles together
```

**Graph Relationships**:
- `title_to_body`: Title connects to first paragraph
- `body_continues`: Paragraph flow
- `jump_reference`: "Continued on page X"
- `spatial_proximity`: Physical layout relationships

### Contributor Extractor

Combines NER with domain-specific patterns:

```python
class ContributorExtractor:
    def extract_contributors(self, job_id, brand_config):
        # 1. Find bylines and photo credits
        # 2. Use NER to extract person names
        # 3. Classify roles (author, photographer, illustrator)
        # 4. Normalize names to "Last, First" format
```

**Role Patterns**:
- Author: "By John Smith", "John Smith reports"
- Photographer: "Photo by Jane Doe", "Photography: Jane Doe"
- Illustrator: "Illustration by Bob Wilson"

### Image Extractor

Spatial algorithms for image-caption linking:

```python
class ImageExtractor:
    def extract_images_and_captions(self, job_id, min_size):
        # 1. Extract all images above size threshold
        # 2. Find caption blocks from layout analysis
        # 3. Use spatial proximity to link images to captions
        # 4. Apply confidence scoring based on distance and positioning
```

**Spatial Heuristics**:
- Captions typically appear below images
- Horizontal alignment indicates strong relationship
- Distance-based confidence scoring
- Handle edge cases (side captions, overlapping layouts)

## 🧪 Testing

### Unit Tests
```bash
poetry run pytest tests/model_service/unit/ -v
```

### Model Tests
```bash
# Test model loading
poetry run pytest tests/model_service/models/ -v

# Test with sample data
poetry run pytest tests/model_service/integration/ -v --sample-data
```

### Performance Tests
```bash
# Benchmark inference speed
poetry run pytest tests/model_service/performance/ -v --benchmark-only
```

## 📊 Performance

### Throughput

| Component | CPU (per core) | GPU (RTX 4090) |
|-----------|----------------|----------------|
| Layout Analysis | 2 pages/min | 20 pages/min |
| OCR Processing | 5 pages/min | 15 pages/min |
| Article Reconstruction | 10 pages/min | 30 pages/min |
| Contributor Extraction | 50 articles/min | 200 articles/min |
| Image Processing | 20 images/min | 100 images/min |

### Memory Requirements

- **CPU Mode**: 4GB RAM minimum, 8GB recommended
- **GPU Mode**: 8GB VRAM (models + batch processing)
- **Model Cache**: ~3GB disk space for all models

### Optimization Tips

1. **Batch Processing**: Group requests for better GPU utilization
2. **Model Caching**: Keep models loaded in memory
3. **Image Preprocessing**: Resize large images before processing
4. **Parallel Processing**: Use multiprocessing for CPU-bound tasks

## 🐛 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size
export BATCH_SIZE=4

# Use gradient checkpointing
export GRADIENT_CHECKPOINTING=true
```

#### Tesseract Not Found
```bash
# Install Tesseract
sudo apt-get install tesseract-ocr

# Set custom path
export TESSERACT_CMD=/usr/local/bin/tesseract
```

#### Model Download Failures
```bash
# Check internet connectivity
curl -I https://huggingface.co/microsoft/layoutlm-v3-base

# Clear model cache
rm -rf models/
```

#### Low OCR Accuracy
```bash
# Check image quality
# Adjust preprocessing parameters in brand config
# Try different Tesseract PSM modes
export TESSERACT_CONFIG="--oem 3 --psm 3"
```

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
poetry run uvicorn model_service.main:app --reload
```

## 🚀 Deployment

### Production Deployment

```dockerfile
# GPU production build
FROM nvidia/cuda:11.8-runtime-ubuntu20.04
COPY . /app
RUN pip install -r requirements.txt
CMD ["uvicorn", "model_service.main:app", "--host", "0.0.0.0"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-service
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: model-service
        image: magazine-extractor-model-service:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 16Gi
          requests:
            memory: 8Gi
```

### Scaling Considerations

- **GPU Instances**: One model per GPU for optimal performance
- **CPU Instances**: Multiple workers per instance
- **Load Balancing**: Sticky sessions for model caching
- **Auto-scaling**: Based on GPU/CPU utilization

---

**For system-wide documentation, see the main [README.md](../../README.md)**