#!/usr/bin/env python3
"""
Quick test of the XML output system.
"""

import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from shared.xml_output import ArticleXMLConverter, XMLConfig
    from shared.xml_output.types import ArticleData

    print("✓ Successfully imported XML output system")

    # Test basic conversion
    article_data = ArticleData(
        article_id="test_001",
        title="Test Article",
        title_confidence=0.95,
        brand="Test Magazine",
        issue_date=datetime(2024, 1, 1),
        page_start=1,
        page_end=2,
        contributors=[
            {
                "name": "Test Author",
                "normalized_name": "Author, Test",
                "role": "author",
                "confidence": 0.9,
            }
        ],
        text_blocks=[
            {
                "type": "paragraph",
                "text": "This is a test paragraph.",
                "confidence": 0.85,
                "id": "block_001",
            }
        ],
        images=[
            {
                "filename": "test_image.jpg",
                "caption": "Test image caption",
                "confidence": 0.8,
            }
        ],
    )

    # Test conversion
    config = XMLConfig(validate_output=False)  # Skip validation for now
    converter = ArticleXMLConverter(config)

    result = converter.convert_article(article_data)

    if result.is_successful:
        print("✓ XML conversion successful")
        print(f"  Elements created: {result.elements_created}")
        print(f"  Processing time: {result.conversion_time:.3f}s")

        # Show first few lines
        lines = result.xml_content.split("\n")[:10]
        print("\nFirst 10 lines of generated XML:")
        for i, line in enumerate(lines, 1):
            print(f"  {i:2d}: {line}")

    else:
        print("✗ XML conversion failed")
        if result.validation_result.errors:
            print(f"  Errors: {result.validation_result.errors}")

except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("  Make sure lxml is installed: pip install lxml")

except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback

    traceback.print_exc()

print("\nXML system test completed.")
