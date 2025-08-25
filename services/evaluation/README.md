# Evaluation Service

The Evaluation Service provides accuracy assessment, drift detection, and auto-tuning capabilities for the Magazine PDF Extractor system. It ensures 99.9% accuracy through continuous monitoring and automated parameter optimization.

## 🎯 Purpose

- **Accuracy Evaluation**: Compare extraction results against gold standard datasets
- **Drift Detection**: Monitor accuracy trends and detect degradation patterns
- **Auto-Tuning**: Automatically optimize system parameters when drift is detected
- **Gold Set Management**: Manage and augment gold standard datasets
- **Quality Assurance**: Quarantine outputs that don't meet accuracy thresholds

## 🏗️ Architecture

### Core Components

- **Accuracy Evaluator** (`models/accuracy_evaluator.py`): Field-level accuracy calculation
- **Drift Detector** (`models/drift_detector.py`): Trend analysis and drift identification
- **Auto Tuner** (`models/auto_tuner.py`): Parameter optimization and deployment
- **Gold Set Manager** (`models/gold_set_manager.py`): Gold dataset management and augmentation
- **Evaluation Database** (`models/`): Store evaluation results and historical data

### Evaluation Pipeline

1. **Accuracy Assessment** → Compare against gold standard
2. **Drift Analysis** → Monitor rolling accuracy trends
3. **Auto-Tuning** → Optimize parameters on detected drift
4. **Quality Gates** → Pass/fail decisions for outputs

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL 15+
- NumPy, Pandas for statistical analysis
- Gold standard datasets per brand

### Setup

1. **Install dependencies**
   ```bash
   poetry install
   ```

2. **Database setup**
   ```bash
   poetry run alembic upgrade head
   ```

3. **Initialize gold sets**
   ```bash
   poetry run python -m evaluation.scripts.init_gold_sets --brand economist --data-dir data/gold_sets/economist/
   ```

4. **Start the service**
   ```bash
   poetry run uvicorn evaluation.main:app --reload --host 0.0.0.0 --port 8002
   ```

### Docker Development

```bash
docker-compose up evaluation
```

## 📚 API Reference

### Accuracy Evaluation

#### Evaluate Job Accuracy
```http
POST /api/v1/accuracy/evaluate
Content-Type: application/json

{
  "job_id": "uuid",
  "brand": "economist"
}
```

#### Get Brand Metrics
```http
GET /api/v1/accuracy/brand/economist/metrics?days=30
```

#### Brand Pass Rate
```http
GET /api/v1/accuracy/brand/economist/pass-rate?recent_issues=10
```

#### Compare with Gold Standard
```http
POST /api/v1/accuracy/compare-with-gold
Content-Type: application/json

{
  "extracted_data": {...},
  "gold_standard": {...}
}
```

### Drift Detection

#### Detect Drift
```http
POST /api/v1/drift/detect
Content-Type: application/json

{
  "brand": "economist",
  "recent_jobs": [...]
}
```

#### Get Drift Status
```http
GET /api/v1/drift/brand/economist/status
```

#### Drift History
```http
GET /api/v1/drift/brand/economist/history?days=90
```

#### Drift Alerts
```http
GET /api/v1/drift/alerts?active_only=true
```

### Auto-Tuning

#### Trigger Tuning
```http
POST /api/v1/tuning/trigger
Content-Type: application/json

{
  "brand": "economist",
  "reason": "drift_detected"
}
```

#### Tuning Status
```http
GET /api/v1/tuning/brand/economist/status
```

#### Tuning History
```http
GET /api/v1/tuning/brand/economist/history?limit=50
```

#### Rollback Tuning
```http
POST /api/v1/tuning/rollback
Content-Type: application/json

{
  "brand": "economist",
  "target_version": "v1.2.3"
}
```

### Gold Set Management

#### List Gold Sets
```http
GET /api/v1/gold-sets/brands
```

#### Upload Gold Issue
```http
POST /api/v1/gold-sets/brand/economist/upload
Content-Type: multipart/form-data

{
  "file": <annotated_pdf>,
  "metadata": {...}
}
```

#### Generate Synthetic Variants
```http
POST /api/v1/gold-sets/brand/economist/generate-synthetic
Content-Type: application/json

{
  "augmentation_factor": 10,
  "methods": ["font_variation", "noise_injection", "layout_shift"]
}
```

## 📊 Accuracy Metrics

### Field-Level Accuracy Calculation

Weighted accuracy across four key fields:

```python
field_weights = {
    "title": 0.30,      # 30% - Title text accuracy
    "body": 0.40,       # 40% - Body text WER < 0.1%
    "contributors": 0.20, # 20% - Name + role accuracy
    "media": 0.10       # 10% - Image-caption linking
}

overall_accuracy = sum(field_accuracy[field] * weight 
                      for field, weight in field_weights.items())
```

### Quality Thresholds

- **Issue Pass**: ≥ 99.9% weighted accuracy
- **Brand Pass**: 95% of last 10 issues pass
- **Quarantine**: < 99.9% accuracy
- **Auto-tuning Trigger**: 2 consecutive failures OR brand pass rate < 95%

### Accuracy Comparison Methods

#### Title Accuracy
- Exact string match after normalization
- Case-insensitive comparison
- Strip whitespace and punctuation
- Handle common variations ("The" prefix, etc.)

#### Body Text Accuracy
- Word Error Rate (WER) calculation
- Character-level similarity (Levenshtein distance)
- Semantic similarity using embeddings
- Handle formatting differences (line breaks, spacing)

#### Contributor Accuracy
- Name matching with fuzzy logic
- Role classification accuracy
- Handle name variations and nicknames
- Normalized format matching ("Last, First")

#### Media Accuracy
- Image-caption pair correctness
- Spatial relationship validation
- Caption text similarity
- Missing image detection

## 🔍 Drift Detection

### Statistical Methods

The drift detector uses multiple statistical approaches:

```python
class DriftDetector:
    def detect_drift(self, brand, recent_jobs):
        # 1. Rolling window analysis (10 issues)
        # 2. Trend detection (Mann-Kendall test)
        # 3. Change point detection
        # 4. Statistical significance testing
        # 5. Confidence interval analysis
```

### Drift Triggers

1. **Consecutive Failures**: 2+ issues below 99.9%
2. **Trend Analysis**: Significant downward trend
3. **Threshold Breach**: Brand pass rate < 95%
4. **Statistical Significance**: p-value < 0.05 for degradation

### Risk Scoring

```python
def calculate_drift_risk_score(accuracy_history, recent_failures):
    trend_score = calculate_trend_severity(accuracy_history)
    failure_score = recent_failures / total_recent_jobs
    volatility_score = std_dev(accuracy_history)
    
    risk_score = (trend_score * 0.5 + 
                  failure_score * 0.3 + 
                  volatility_score * 0.2)
    return min(risk_score, 1.0)
```

## ⚙️ Auto-Tuning System

### Tunable Parameters

The auto-tuner optimizes these parameter categories:

#### Graph Traversal Rules
- Article boundary detection thresholds
- Cross-page article stitching rules
- Jump reference pattern confidence
- Spatial relationship weights

#### Confidence Thresholds
- Per-field minimum confidence scores
- Block classification thresholds
- OCR confidence requirements
- Image-caption linking confidence

#### Preprocessing Parameters
- OCR preprocessing settings (deskew, denoise, contrast)
- Image resize parameters
- Text normalization rules
- Brand-specific pattern adjustments

### Tuning Process

```python
class AutoTuner:
    async def run_tuning_cycle(self, brand, trigger_reason):
        # 1. Analyze failing patterns in quarantined issues
        # 2. Generate targeted synthetic test cases
        # 3. Search parameter space using Bayesian optimization
        # 4. Validate improvements on holdout set
        # 5. Deploy if accuracy improves, rollback if not
        # 6. Monitor post-deployment performance
```

### Optimization Algorithm

1. **Problem Analysis**: Identify specific failure patterns
2. **Synthetic Generation**: Create test cases targeting failures
3. **Parameter Search**: Bayesian optimization with Gaussian processes
4. **Validation**: Test on holdout gold set (20%)
5. **Deployment**: Gradual rollout with monitoring
6. **Rollback**: Automatic revert if performance degrades

### Safety Mechanisms

- **Rate Limiting**: Max once per day per brand
- **Validation Gate**: Must improve holdout accuracy by >1%
- **Monitoring**: Continuous post-deployment accuracy tracking
- **Automatic Rollback**: Revert if accuracy drops within 24 hours
- **Human Override**: Manual tuning controls for edge cases

## 📈 Gold Set Management

### Dataset Requirements

Per brand minimum requirements:
- **10+ annotated issues** covering:
  - Standard layouts (70%)
  - Edge cases (20%): spreads, artistic layouts
  - Extreme cases (10%): overlapping text, complex graphics

### Synthetic Augmentation

Generate 10x synthetic variants using:

```python
augmentation_methods = {
    "font_variation": vary_font_sizes_and_families,
    "noise_injection": add_scanning_artifacts,
    "layout_shift": adjust_column_positions,
    "compression_variation": change_pdf_compression,
    "resolution_scaling": modify_image_resolution,
    "color_space_conversion": grayscale_conversion,
    "ocr_simulation": simulate_scan_ocr_errors
}
```

### Quality Validation

Automated gold set quality checks:
- **Completeness**: All required fields annotated
- **Consistency**: Annotation format compliance
- **Coverage**: Layout variety and complexity distribution
- **Accuracy**: Cross-validator agreement scores

## 🧪 Testing

### Unit Tests
```bash
poetry run pytest tests/evaluation/unit/ -v
```

### Accuracy Tests
```bash
# Test evaluation algorithms
poetry run pytest tests/evaluation/accuracy/ -v

# Test with sample gold sets
poetry run pytest tests/evaluation/integration/ -v --gold-data
```

### Drift Detection Tests
```bash
# Test drift algorithms with synthetic data
poetry run pytest tests/evaluation/drift/ -v --generate-synthetic-trends
```

### Auto-Tuning Tests
```bash
# Test parameter optimization
poetry run pytest tests/evaluation/tuning/ -v --mock-parameter-search
```

## 📊 Monitoring

### Evaluation Metrics

Key metrics tracked:
- `evaluation_requests_total`: Total evaluation requests
- `accuracy_scores`: Distribution of accuracy scores
- `drift_events_total`: Number of drift detections
- `tuning_cycles_total`: Auto-tuning executions
- `quarantine_rate`: Percentage of quarantined outputs

### Dashboards

Grafana dashboards showing:
- Real-time accuracy by brand
- Drift detection timeline
- Auto-tuning success rate
- Gold set coverage status
- Performance trends

### Alerting

Automated alerts for:
- Accuracy below threshold (critical)
- Drift detection (warning)
- Auto-tuning failures (warning)
- Gold set coverage gaps (info)
- System performance degradation (critical)

## 🚀 Deployment

### Production Configuration

```yaml
# docker-compose.prod.yml
services:
  evaluation:
    build:
      target: production
    environment:
      - ACCURACY_THRESHOLD=0.999
      - DRIFT_WINDOW_SIZE=10
      - MAX_TUNING_FREQUENCY_HOURS=24
    volumes:
      - gold_sets:/app/gold_sets:ro
      - evaluation_results:/app/evaluation_results
```

### Scaling Considerations

- **Database Performance**: Index evaluation_results table
- **Statistical Computing**: CPU-intensive drift analysis
- **Gold Set Storage**: Read-only shared storage for gold datasets
- **Tuning Coordination**: Single tuner per brand to avoid conflicts

## 🔧 Troubleshooting

### Common Issues

#### Inaccurate Drift Detection
```python
# Check data quality
SELECT brand, COUNT(*), AVG(accuracy_score) 
FROM evaluation_results 
WHERE created_at > NOW() - INTERVAL '30 days' 
GROUP BY brand;

# Adjust sensitivity
export DRIFT_THRESHOLD=0.01  # Lower = more sensitive
```

#### Auto-Tuning Not Improving Accuracy
```python
# Check parameter search space
# Review failing patterns in logs
# Validate gold set quality
# Increase search iterations
```

#### High False Positive Rate
```python
# Review gold set annotations
# Check evaluation algorithm alignment
# Adjust field weights based on business importance
```

### Debug Tools

Enable detailed logging:
```bash
export LOG_LEVEL=DEBUG
export TRACE_EVALUATIONS=true
poetry run uvicorn evaluation.main:app --reload
```

Statistical analysis tools:
```python
# Interactive drift analysis
poetry run python -m evaluation.tools.drift_analyzer --brand economist

# Gold set validator
poetry run python -m evaluation.tools.validate_gold_sets --all-brands

# Parameter impact analysis
poetry run python -m evaluation.tools.parameter_analyzer --brand economist
```

---

**For system-wide documentation, see the main [README.md](../../README.md)**