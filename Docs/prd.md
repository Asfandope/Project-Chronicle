# Product Requirements Document: Automated Magazine/Newspaper PDF Extraction System

## 1. Summary

• **Delivers**: A fully automated system that extracts articles from heterogeneous magazine/newspaper PDFs into canonical XML with 99.9% field-level accuracy, including linked images and issue summaries.

• **Distinct Approach**: Uses a dual-pass architecture with layout-aware language models - first pass creates a semantic graph of page elements, second pass traverses the graph to reconstruct articles using constrained decoding directly to XML.

• **Self-Healing**: Continuously evaluates accuracy using synthetic gold sets and automatically fine-tunes graph traversal rules and confidence thresholds when drift is detected.

• **Zero Human Touch**: No manual QA or approval steps; outputs are automatically quarantined if accuracy drops below threshold until the system self-recovers.

• **Brand Agnostic**: All brand-specific quirks are expressed as YAML configurations and graph traversal rules, not code.

## 2. Goals & Non-Goals

**Goals:**
- Achieve 99.9% field-level accuracy across all supported brands without human intervention
- Process both born-digital and scanned PDFs with complex layouts
- Self-detect accuracy degradation and auto-tune to recover
- Maintain single canonical XML schema across all brands
- Run entirely on-prem or customer-controlled infrastructure

**Non-Goals (v1):**
- Real-time processing (batch mode is acceptable)
- Supporting non-Latin scripts or RTL languages
- Video/audio content extraction
- Advertisement content extraction (ads are filtered out)
- Historical archive migration
- Multi-language article detection within single issues

## 3. Users & Primary Flows

**Primary Users:**
- **System Operators**: DevOps teams who deploy and monitor the system
- **Content Consumers**: Downstream systems consuming XML/CSV outputs
- **Brand Onboarding Teams**: Initial configuration creators (one-time per brand)

**Primary Flow:**
1. PDFs deposited in watched directory or S3 bucket
2. System detects new files and queues for processing
3. Dual-pass extraction creates semantic graph and traverses to XML
4. Accuracy evaluation against synthetic gold standard
5. Outputs written to XML/images/CSV or quarantined if below threshold
6. Drift detection triggers auto-tuning if needed

## 4. Canonical XML Contract

**Location**: Single source of truth at `schemas/article-v1.0.xsd`

**Required Entities:**
```xml
<article id="uuid" brand="string" issue="date" page_start="int" page_end="int">
  <title confidence="float">Extracted Title</title>
  <contributors>
    <contributor role="author|photographer|illustrator" confidence="float">
      <name>Full Name</name>
      <normalized_name>Last, First</normalized_name>
    </contributor>
  </contributors>
  <body>
    <paragraph confidence="float">Text content...</paragraph>
    <pullquote confidence="float">Highlighted quote...</pullquote>
  </body>
  <media>
    <image src="images/uuid.jpg" confidence="float">
      <caption>Image caption text</caption>
      <credit>Photographer credit</credit>
    </image>
  </media>
  <provenance>
    <extracted_at>ISO-8601</extracted_at>
    <model_version>string</model_version>
    <confidence_overall>float</confidence_overall>
  </provenance>
</article>
```

**Versioning**: Schema follows semantic versioning. Minor versions add optional fields only. Major versions require migration pipelines.

## 5. Functional Requirements

### 5.1 Ingestion & Preprocessing
- Watch directories/buckets for new PDFs
- Validate PDF structure and quarantine corrupted files
- Split into individual pages maintaining order
- **Acceptance**: 100% of valid PDFs processed, <0.1% false quarantine rate

### 5.2 Layout Understanding (First Pass)
- Extract text blocks with bounding boxes using PDF libraries
- Run layout-aware LLM to classify each block (title, body, caption, pullquote, header/footer, ad)
- Build semantic graph with spatial relationships between blocks
- **Acceptance**: 99.5% block classification accuracy on hold-out set

### 5.3 OCR Strategy
- For born-digital: Direct text extraction from PDF
- For scanned: Tesseract with brand-specific preprocessing pipelines defined in YAML
- Confidence scoring per block based on character-level certainty
- **Acceptance**: <2% WER on scanned content, <0.1% on born-digital

### 5.4 Article Reconstruction (Second Pass)
- Graph traversal algorithm uses learned rules to connect related blocks across pages
- Handles split articles, jump references ("continued on page X"), and interleaved content
- **Acceptance**: 99.9% correct article boundaries, 100% of split articles properly stitched

### 5.5 Contributor Parsing
- NER model extracts names from bylines and photo credits
- Role classifier assigns author/photographer/illustrator based on context
- Name normalizer creates canonical "Last, First" format
- **Acceptance**: 99% name extraction recall, 99.5% role classification accuracy

### 5.6 Ad Filtering
- Graph nodes classified as ads are excluded from article reconstruction
- Ad detection uses visual features + text patterns defined in brand configs
- **Acceptance**: 99% ad filtering precision, <0.5% false positive rate

### 5.7 Image Extraction & Linking
- Extract all images above 100x100px threshold
- Match images to closest caption blocks using spatial proximity
- Generate deterministic filenames: `{issue_date}_{article_id}_{sequence}.jpg`
- **Acceptance**: 99% correct image-caption pairing, 100% of images extracted

### 5.8 Export Pipeline
- XML generation using constrained decoding (guarantees schema compliance)
- CSV summary with article count, avg confidence, contributor list per issue
- **Acceptance**: 100% schema-valid XML, deterministic output given same input

## 6. Accuracy Definition & Metrics

**Field-Level Accuracy**: Weighted average across fields:
- Title match: 30% weight (exact match after normalization)
- Body text: 40% weight (WER < 0.1%)  
- Contributors: 20% weight (name + role correct)
- Media links: 10% weight (correct image-caption pairs)

**Issue Pass/Fail**: An issue passes if weighted accuracy ≥ 99.9%

**Brand Pass/Fail**: A brand passes if 95% of last 10 issues pass

**Quarantine Rules**: 
- Any issue below 99.9% is quarantined
- Any brand below 95% pass rate triggers auto-tuning
- Outputs include confidence scores for downstream filtering

## 7. Self-Evaluation & Auto-Fine-Tuning

### 7.1 Gold Set Generation
- Each brand requires 10 manually annotated issues as gold standard
- Synthetic augmentation creates 100+ variants (font changes, scan quality, layout shifts)
- 20% holdout for final validation

### 7.2 Drift Detection
- Every processed issue is evaluated against synthetic gold set
- Rolling 10-issue window tracks accuracy trends
- Drift signal: 2 consecutive issues below 99.9% OR brand pass rate < 95%

### 7.3 Auto-Tuning Loop
**What Updates:**
- Graph traversal rules (YAML configs)
- Confidence thresholds per field
- Block classifier prompts
- OCR preprocessing parameters

**Cadence**: Triggered by drift detection, max once per day per brand

**Process**:
1. Isolate failing patterns from quarantined issues
2. Generate targeted synthetic examples
3. Tune parameters on synthetic set
4. Validate on holdout set
5. Deploy if accuracy improves, rollback if not

**DRY/KISS Preservation**: All tunable parameters in central registry, single tuning pipeline for all brands

## 8. System Architecture

### 8.1 Components

**Orchestrator Service**: 
- Manages job queues and workflow state
- Single source of workflow definitions (DRY)
- *Justification*: Central coordination prevents duplicate processing

**Model Service**:
- Hosts layout classifier and NER models
- Provides unified inference API
- *Justification*: Shared GPU resources, consistent preprocessing

**Evaluation Service**:
- Computes accuracy metrics
- Manages gold sets and drift detection  
- *Justification*: Centralized accuracy tracking enables auto-tuning

**Configuration Registry**:
- Stores brand configs, tunable parameters, model versions
- *Justification*: Single source of truth (DRY) for all configuration

### 8.2 Data Flow
1. PDF → Orchestrator → Page Splitter
2. Pages → Model Service → Semantic Graph
3. Graph → Traversal Engine → Article Boundaries  
4. Articles → Model Service → Field Extraction
5. Extracted Data → Evaluation Service → Accuracy Check
6. If Pass → Export Pipeline → XML/CSV/Images
7. If Fail → Quarantine + Tuning Trigger

### 8.3 Dependencies
- PyPDF2/pdfplumber (PDF manipulation) - MIT License
- Tesseract 5.0+ (OCR) - Apache 2.0
- Transformers/LayoutLM (layout understanding) - Apache 2.0
- NetworkX (graph algorithms) - BSD
- PostgreSQL (job queue/state) - PostgreSQL License

## 9. Data Strategy

### 9.1 Golden Dataset Policy
- Minimum 10 issues per brand covering:
  - Standard layouts (70%)
  - Edge cases: spreads, jump articles, heavy graphics (20%)
  - Extreme cases: artistic layouts, overlapping text (10%)

### 9.2 Brand Configuration as Data
Each brand has a YAML configuration:
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
  title: 0.95  # Higher threshold for this brand
```

No code changes needed for new brands (DRY).

## 10. Non-Functional Requirements

### 10.1 Performance
- Throughput: 50 pages/minute on single GPU
- Latency: <5 minutes for 100-page issue
- Deterministic: Identical outputs for identical inputs

### 10.2 Scalability  
- Horizontal scaling via job queue
- GPU optional (10x slower on CPU)

### 10.3 Complexity Budget
- Maximum 3 services for v1 (orchestrator, model, evaluation)
- Maximum 2 model types (layout + NER)
- Configuration changes preferred over new code

### 10.4 Observability
- Structured logs with correlation IDs
- Metrics: accuracy per brand/issue, processing time, queue depth
- Alerts: accuracy degradation, processing failures

## 11. Deployment & Operations

### 11.1 Environments
- **Dev**: Docker Compose on workstation (CPU only)
- **Prod**: Kubernetes on-prem or cloud VMs with GPU

### 11.2 CI/CD Pipeline
1. Code changes → Run test suite on synthetic data
2. Model changes → Validate on all brand holdout sets  
3. Config changes → Canary on 1 issue before full deployment
4. Automated rollback if accuracy drops

### 11.3 Monitoring
- Grafana dashboards showing:
  - Real-time accuracy by brand
  - Processing throughput
  - Drift detection alerts
  - Auto-tuning events

## 12. Timeline & Resourcing

### 12.1 MVP (Month 1-2)
- Single brand support
- Basic dual-pass extraction
- Manual accuracy evaluation
- Exit: 95% accuracy on test brand

### 12.2 Beta (Month 3-4)  
- 5 brands supported
- Auto-evaluation pipeline
- Basic auto-tuning
- Exit: 99% accuracy on all brands

### 12.3 v1.0 (Month 5-6)
- 10+ brands
- Full auto-tuning
- Production monitoring
- Exit: 99.9% accuracy with self-healing

### 12.4 Team
- 1 ML Engineer (models, tuning)
- 1 Backend Engineer (orchestration, exports)
- 0.5 DevOps (deployment, monitoring)
- Minimal team preserves KISS principle

## 13. Risks & Mitigations

**Risk**: Typography variations break OCR
**Mitigation**: Brand-specific preprocessing configs in YAML

**Risk**: Low-resolution scans  
**Mitigation**: Confidence thresholds auto-adjust per brand

**Risk**: Ambiguous figure-caption relationships
**Mitigation**: Spatial proximity rules in config, not code

**Risk**: Schema changes requested
**Mitigation**: Versioned schema with migration tools

All mitigations use configuration, preserving DRY/KISS.

## 14. Test Plan & Acceptance

### 14.1 Pre-Production Gates
- Each brand must pass 99.9% on 20-issue test set
- Regression suite runs on all brands for any change
- Performance benchmarks must stay within 10% of baseline

### 14.2 Auto-Quarantine Rules
- Issue quarantined if accuracy < 99.9%
- Brand quarantined if 3 consecutive issues fail
- Auto-unquarantine after successful tuning + validation

### 14.3 Integration Tests
- End-to-end tests with synthetic PDFs
- Accuracy computation validation
- Auto-tuning trigger validation

## 15. Cost & Capacity

**GPU Hours**: ~0.1 GPU-hour per 100-page issue
**Storage**: ~50MB per issue (images + XML)
**Monthly (1000 issues)**: 
- Compute: 100 GPU-hours ≈ $100-300
- Storage: 50GB ≈ $5
- Linear scaling with brands/issues

## 16. Why This PRD is Different

This approach differs from a vanilla "layout parser + regex" baseline by using a graph-based semantic understanding of the page that can handle complex, multi-page articles and decorative layouts. Instead of brittle position-based rules, we learn traversal patterns that adapt to each brand's style through configuration. The dual-pass architecture separates concerns cleanly: first understanding what's on each page, then understanding how pages connect. This beats simple approaches on complex layouts while maintaining simplicity - just three services, two model types, and zero code duplication across brands. By expressing brand differences purely as data (YAML configs and traversal rules), we achieve DRY without sacrificing accuracy. The self-tuning loop ensures we maintain 99.9% accuracy without human intervention, something impossible with static regex patterns.