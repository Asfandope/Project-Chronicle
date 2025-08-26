# Dataset Curation Tools for Gold Standard Magazine Extraction

This directory contains a comprehensive toolkit for creating, managing, and validating gold standard datasets for the magazine PDF extraction pipeline.

## Tools Overview

### 1. Dataset Curator (`dataset_curator.py`)
**Purpose**: Analyze PDF quality and generate metadata for gold standard datasets.

**Key Features**:
- PDF quality analysis (OCR accuracy estimation, layout complexity)
- Metadata generation with quality scores
- File integrity validation
- Comprehensive quality reporting

**Usage Examples**:
```bash
# Analyze a single PDF
python3 dataset_curator.py analyze --pdf /path/to/magazine.pdf --brand economist

# Generate quality report for all brands
python3 dataset_curator.py report --base-path ../data/gold_sets --output report.json

# Validate file integrity
python3 dataset_curator.py validate --pdf /path/to/magazine.pdf
```

### 2. Ground Truth Generator (`ground_truth_generator.py`)
**Purpose**: Generate XML ground truth files following the magazine extraction schema v1.0.

**Key Features**:
- Template generation for manual annotation
- Automated ground truth from extraction results
- XML structure validation
- Schema-compliant output

**Usage Examples**:
```bash
# Create annotation template
python3 ground_truth_generator.py template --brand economist --issue-id sample_20250825 --pages 1 --output template.xml

# Generate from extraction results
python3 ground_truth_generator.py generate --brand economist --extraction-results results.json --output ground_truth.xml

# Validate existing XML
python3 ground_truth_generator.py validate --xml ground_truth.xml
```

### 3. Validation Pipeline (`validation_pipeline.py`)
**Purpose**: Comprehensive validation and quality control for gold standard datasets.

**Key Features**:
- Schema compliance validation
- Data integrity checks
- Content quality assessment
- Cross-reference validation
- Performance benchmarking against quality thresholds

**Usage Examples**:
```bash
# Validate all datasets
python3 validation_pipeline.py --base-path ../data/gold_sets --threshold-check

# Validate specific brand
python3 validation_pipeline.py --brand economist --base-path ../data/gold_sets

# Generate detailed report
python3 validation_pipeline.py --output validation_report.json --threshold-check
```

### 4. Annotation Workflow (`annotation_workflow.py`)
**Purpose**: Complete workflow management for annotation tasks and quality control.

**Key Features**:
- Task creation and assignment
- Workspace management for annotators
- Quality control validation
- Batch processing support
- Progress tracking and reporting

**Usage Examples**:
```bash
# Create annotation task
python3 annotation_workflow.py create --brand economist --pdf /path/to/file.pdf --annotator alice

# Batch create tasks
python3 annotation_workflow.py batch --brand economist --pdf-dir /path/to/pdfs --annotator alice

# Validate completed annotations
python3 annotation_workflow.py validate --batch-size 10

# Generate workflow report
python3 annotation_workflow.py report --output workflow_report.json
```

## Quality Requirements

The tools enforce the following quality standards from the project requirements:

### OCR Accuracy
- **Born-digital PDFs**: WER < 0.0005 (>99.95% accuracy)
- **Scanned PDFs**: WER < 0.015 (>98.5% accuracy)

### Layout Classification
- **Accuracy**: >99.5% for block type identification
- **Coverage**: All major block types (title, body, byline, caption, ad, etc.)

### Article Reconstruction
- **Completeness**: >98% article boundary accuracy
- **Cross-page handling**: Support for split articles with jump references

### Manual Validation
- All gold standard files require manual validation
- Inter-annotator agreement checking for quality datasets
- Expert review process with confidence scoring

## Makefile Integration

The tools are fully integrated with the project Makefile for easy use:

```bash
# Setup complete directory structure
make curate-datasets

# Analyze single PDF
make curate-pdf PDF=/path/to/file.pdf BRAND=economist

# Create annotation tasks
make create-annotation-task PDF=/path/to/file.pdf BRAND=economist ANNOTATOR=alice

# Batch create tasks
make batch-create-tasks PDF_DIR=/path/to/pdfs BRAND=economist ANNOTATOR=alice

# Validate datasets
make validate-gold-sets BRAND=economist

# Generate comprehensive reports
make gold-sets-report
make annotation-report
```

## Workflow Process

### 1. Initial Setup
```bash
make curate-datasets
```
This creates the complete directory structure for all brands and workspaces.

### 2. PDF Analysis and Quality Assessment
```bash
python3 tools/dataset_curator.py analyze --pdf input.pdf --brand economist
```
Analyzes PDF for quality metrics, layout complexity, and suitability for annotation.

### 3. Annotation Task Creation
```bash
python3 tools/annotation_workflow.py create --brand economist --pdf input.pdf --annotator alice
```
Creates structured annotation task with workspace, templates, and guides.

### 4. Ground Truth Generation
```bash
python3 tools/ground_truth_generator.py template --brand economist --output template.xml
```
Generates XML template following schema v1.0 for manual annotation.

### 5. Validation and Quality Control
```bash
python3 tools/validation_pipeline.py --brand economist --threshold-check
```
Validates completed annotations against quality thresholds and requirements.

### 6. Dataset Finalization
Validated datasets are automatically moved to the final gold standard locations with complete metadata.

## Directory Structure

```
Project-Chronicle/
├── tools/                          # Curation tools (this directory)
│   ├── dataset_curator.py         # PDF analysis and quality assessment
│   ├── ground_truth_generator.py  # XML ground truth generation
│   ├── validation_pipeline.py     # Quality validation pipeline
│   ├── annotation_workflow.py     # Annotation workflow management
│   └── README.md                  # This documentation
├── data/gold_sets/                # Gold standard datasets
│   ├── {brand}/                   # Per-brand datasets
│   │   ├── pdfs/                  # Original PDF files
│   │   ├── ground_truth/          # XML ground truth files
│   │   ├── annotations/           # Human annotations
│   │   └── metadata/              # Quality metadata
│   └── staging/                   # Validation staging area
├── workspaces/                    # Annotation workspaces
│   ├── tasks/                     # Task management
│   ├── {annotator}/               # Per-annotator workspaces
│   │   ├── active/                # Active annotation tasks
│   │   └── completed/             # Completed tasks
│   └── reports/                   # Workflow reports
```

## XML Schema v1.0

The tools generate and validate XML files following this structure:

```xml
<magazine_extraction schema_version="v1.0" brand="economist" issue_id="issue_001">
  <metadata>
    <brand>economist</brand>
    <issue_id>issue_001</issue_id>
    <schema_version>v1.0</schema_version>
    <total_pages>1</total_pages>
    <extraction_date>2025-08-25T12:00:00</extraction_date>
    <annotator>manual</annotator>
    <validation_status>pending</validation_status>
  </metadata>
  
  <pages>
    <page number="1" width="612" height="792">
      <blocks>
        <block id="block_1_1" type="title" confidence="0.95">
          <text>Article Title</text>
          <bbox x="100" y="100" width="400" height="50"/>
          <font_info family="Arial" size="12" style="normal"/>
        </block>
      </blocks>
    </page>
  </pages>
  
  <articles>
    <article id="article_1" confidence="0.90">
      <title>Article Title</title>
      <title_blocks><block_id>block_1_1</block_id></title_blocks>
      <body_blocks><block_id>block_1_2</block_id></body_blocks>
      <byline_blocks></byline_blocks>
      <caption_blocks></caption_blocks>
      <contributors>
        <contributor id="contributor_1">
          <name>Author Name</name>
          <role>author</role>
          <confidence>0.95</confidence>
        </contributor>
      </contributors>
      <images></images>
      <pages><page_number>1</page_number></pages>
    </article>
  </articles>
</magazine_extraction>
```

## Error Handling and Fallbacks

The tools include comprehensive error handling:

- **Missing Dependencies**: Graceful fallbacks when optional libraries unavailable
- **Malformed Files**: Detailed error reporting with recovery suggestions
- **Validation Failures**: Clear issue identification with remediation steps
- **Processing Errors**: Robust exception handling with logging

## Quality Metrics

The validation pipeline tracks these key metrics:

- **Schema Compliance**: XML structure and attribute validation
- **Data Integrity**: Cross-reference validation and consistency
- **Content Quality**: Text completeness and bounding box accuracy
- **Annotation Consistency**: Block type and contributor role consistency
- **Cross References**: Article-block relationship validation

## Next Steps

For creating the 40 required gold standard issues (10 per brand):

1. **Collect representative PDF samples** covering various layouts and content types
2. **Use annotation workflow** to create systematic annotation tasks
3. **Apply quality validation** to ensure all files meet the >98% accuracy requirements
4. **Generate comprehensive reports** to track progress and quality metrics

This toolset provides a complete foundation for creating high-quality gold standard datasets that meet all project requirements for training and evaluating the magazine extraction pipeline.