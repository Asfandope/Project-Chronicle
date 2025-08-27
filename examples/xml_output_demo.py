#!/usr/bin/env python3
"""
Demonstration of XML output system with constrained generation.

This example shows how to:
1. Convert article data to canonical XML
2. Validate against schema
3. Generate deterministic output with confidence scores
4. Use pretty-printing for debugging
"""


# Add project root to path
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.xml_output import (
    ArticleXMLConverter,
    OutputFormat,
    XMLConfig,
)
from shared.xml_output.types import ArticleData


def create_sample_article_data() -> ArticleData:
    """Create sample article data for demonstration."""

    return ArticleData(
        article_id="demo_article_001",
        title="Advanced Magazine Processing with AI",
        title_confidence=0.957,
        brand="Tech Weekly",
        issue_date=datetime(2024, 3, 15),
        page_start=42,
        page_end=47,
        contributors=[
            {
                "name": "Dr. Sarah Johnson",
                "normalized_name": "Johnson, Sarah",
                "role": "author",
                "confidence": 0.923,
            },
            {
                "name": "Mike Chen",
                "normalized_name": "Chen, Mike",
                "role": "photographer",
                "confidence": 0.876,
            },
        ],
        text_blocks=[
            {
                "type": "paragraph",
                "text": "The field of automated document processing has seen remarkable advances in recent years.",
                "confidence": 0.891,
                "id": "block_001",
                "page": 42,
                "position": 1,
            },
            {
                "type": "paragraph",
                "text": "Machine learning models can now extract text, images, and metadata with high accuracy.",
                "confidence": 0.934,
                "id": "block_002",
                "page": 42,
                "position": 2,
            },
            {
                "type": "pullquote",
                "text": "The combination of computer vision and NLP creates powerful extraction capabilities.",
                "confidence": 0.812,
                "id": "block_003",
                "page": 43,
                "position": 1,
            },
        ],
        images=[
            {
                "filename": "img_001_tech_demo.jpg",
                "caption": "Demonstration of AI-powered document analysis workflow",
                "credit": "Photo by Mike Chen",
                "confidence": 0.887,
            },
            {
                "filename": "img_002_diagram.jpg",
                "caption": "System architecture diagram showing processing pipeline",
                "confidence": 0.923,
            },
        ],
        extraction_confidence=0.902,
        processing_pipeline_version="2.1.0",
        overall_quality="high",
        text_extraction_quality="high",
        media_matching_quality="high",
        contributor_extraction_quality="medium",
    )


def demo_production_output():
    """Demonstrate production-ready XML output."""
    print("=== Production XML Output Demo ===")

    # Create production configuration
    config = XMLConfig.for_production()
    converter = ArticleXMLConverter(config)

    # Convert article data
    article_data = create_sample_article_data()
    result = converter.convert_article(article_data)

    print(f"Conversion successful: {result.is_successful}")
    print(f"Validation passed: {result.validation_result.is_valid}")
    print(f"Elements created: {result.elements_created}")
    print(f"Processing time: {result.conversion_time:.3f}s")

    if result.validation_result.errors:
        print(f"Validation errors: {result.validation_result.errors}")

    # Save to file
    output_path = Path("output") / "article_production.xml"
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result.xml_content)

    print(f"Production XML saved to: {output_path}")
    print(f"File size: {len(result.xml_content)} characters")
    print()


def demo_debug_output():
    """Demonstrate debug XML output with pretty printing."""
    print("=== Debug XML Output Demo ===")

    # Create debug configuration
    config = XMLConfig.for_debugging()
    converter = ArticleXMLConverter(config)

    # Convert article data
    article_data = create_sample_article_data()
    result = converter.convert_article(article_data)

    print(f"Conversion successful: {result.is_successful}")
    print(f"Validation passed: {result.validation_result.is_valid}")

    # Save debug version
    output_path = Path("output") / "article_debug.xml"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result.xml_content)

    print(f"Debug XML saved to: {output_path}")

    # Show first few lines
    lines = result.xml_content.split("\n")[:15]
    print("First 15 lines of debug XML:")
    for i, line in enumerate(lines, 1):
        print(f"{i:2d}: {line}")

    if len(lines) < len(result.xml_content.split("\n")):
        print("    ... (truncated)")
    print()


def demo_canonical_output():
    """Demonstrate canonical XML output."""
    print("=== Canonical XML Output Demo ===")

    # Create canonical configuration
    config = XMLConfig(
        output_format=OutputFormat.CANONICAL,
        sort_attributes=True,
        sort_elements=True,
        validate_output=True,
    )
    converter = ArticleXMLConverter(config)

    # Convert same article data multiple times to show deterministic output
    article_data = create_sample_article_data()

    outputs = []
    for i in range(3):
        result = converter.convert_article(article_data)
        outputs.append(result.xml_content)

    # Verify all outputs are identical (deterministic)
    all_identical = all(output == outputs[0] for output in outputs)
    print(f"Deterministic output test: {'PASSED' if all_identical else 'FAILED'}")
    print(f"All {len(outputs)} conversions produced identical XML")

    # Save canonical version
    output_path = Path("output") / "article_canonical.xml"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(outputs[0])

    print(f"Canonical XML saved to: {output_path}")
    print()


def demo_validation_errors():
    """Demonstrate validation error handling."""
    print("=== Validation Error Demo ===")

    config = XMLConfig(strict_validation=True)
    converter = ArticleXMLConverter(config)

    # Create article with invalid data
    invalid_article = ArticleData(
        article_id="",  # Invalid: empty ID
        title="Test Article",
        title_confidence=1.5,  # Invalid: confidence > 1.0
        brand="Test Brand",
        issue_date=datetime.now(),
        page_start=0,  # Invalid: page < 1
        page_end=-1,  # Invalid: negative page
        contributors=[
            {
                "name": "Test Author",
                "role": "invalid_role",  # Invalid role
                "confidence": -0.5,  # Invalid: negative confidence
            }
        ],
    )

    result = converter.convert_article(invalid_article)

    print(f"Conversion attempted: {result is not None}")
    print(f"Validation passed: {result.validation_result.is_valid}")
    print(f"Number of errors: {len(result.validation_result.errors)}")

    if result.validation_result.errors:
        print("Validation errors found:")
        for i, error in enumerate(result.validation_result.errors, 1):
            print(f"  {i}. {error}")

    print()


def demo_confidence_filtering():
    """Demonstrate confidence-based filtering."""
    print("=== Confidence Filtering Demo ===")

    # Create configuration with confidence threshold
    config = XMLConfig(confidence_threshold=0.8, include_low_confidence=False)
    converter = ArticleXMLConverter(config)

    # Create article with mixed confidence scores
    article_data = create_sample_article_data()

    # Add some low-confidence data
    article_data.contributors.append(
        {
            "name": "Low Confidence Author",
            "role": "author",
            "confidence": 0.3,  # Below threshold
        }
    )

    article_data.text_blocks.append(
        {
            "type": "paragraph",
            "text": "This text has very low confidence.",
            "confidence": 0.2,  # Below threshold
            "id": "block_low",
        }
    )

    result = converter.convert_article(article_data)

    print(f"Original contributors: {len(article_data.contributors)}")
    print(f"Original text blocks: {len(article_data.text_blocks)}")
    print(f"Elements in XML: {result.elements_created}")
    print(
        f"Low-confidence elements filtered: {result.low_confidence_elements_filtered}"
    )

    # Save filtered version
    output_path = Path("output") / "article_filtered.xml"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result.xml_content)

    print(f"Filtered XML saved to: {output_path}")
    print()


def demo_conversion_statistics():
    """Demonstrate conversion statistics."""
    print("=== Conversion Statistics Demo ===")

    config = XMLConfig.for_production()
    converter = ArticleXMLConverter(config)

    # Convert multiple articles
    sample_articles = [
        create_sample_article_data(),
        create_sample_article_data(),
        create_sample_article_data(),
    ]

    # Modify article IDs to make them unique
    for i, article in enumerate(sample_articles):
        article.article_id = f"demo_article_{i+1:03d}"

    # Process all articles
    results = []
    for article in sample_articles:
        result = converter.convert_article(article)
        results.append(result)

    # Get conversion statistics
    stats = converter.get_conversion_statistics()

    print("Conversion Statistics:")
    print(f"  Total conversions: {stats['total_conversions']}")
    print(f"  Successful: {stats['successful_conversions']}")
    print(f"  Failed: {stats['failed_conversions']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"  Average processing time: {stats['average_processing_time']:.3f}s")
    print(f"  Elements created: {stats['total_elements_created']}")
    print(f"  Confidence scores added: {stats['total_confidence_scores']}")
    print(f"  Output format: {stats['output_format']}")
    print()


def main():
    """Run all demonstrations."""
    print("XML Output System Demonstration")
    print("=" * 50)
    print()

    # Ensure output directory exists
    Path("output").mkdir(exist_ok=True)

    try:
        # Run all demos
        demo_production_output()
        demo_debug_output()
        demo_canonical_output()
        demo_validation_errors()
        demo_confidence_filtering()
        demo_conversion_statistics()

        print("All demonstrations completed successfully!")
        print(f"Output files saved in: {Path('output').absolute()}")

    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure lxml is installed: pip install lxml")

    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
