#!/usr/bin/env python3
"""
Simple test script to process TOTAL.pdf magazine
"""
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from shared.ocr.engine import OCREngine
    from shared.preprocessing.pdf import PDFPreprocessor

    print("✅ Successfully imported core modules")

    # Initialize components
    print("Initializing OCR engine...")
    ocr_engine = OCREngine()

    print("Initializing PDF preprocessor...")
    pdf_processor = PDFPreprocessor()

    # Process the PDF
    pdf_path = "magazine/TOTAL.pdf"
    print(f"Processing {pdf_path}...")

    # Extract pages
    pages = pdf_processor.extract_pages(pdf_path)
    print(f"Extracted {len(pages)} pages")

    # Process first page as test
    if pages:
        first_page = pages[0]
        print(f"First page dimensions: {first_page.width}x{first_page.height}")

        # Run OCR on first page
        print("Running OCR on first page...")
        ocr_result = ocr_engine.extract_text(first_page.image)

        print(f"OCR extracted {len(ocr_result.text)} characters")
        print(
            "Sample text:",
            ocr_result.text[:200] + "..."
            if len(ocr_result.text) > 200
            else ocr_result.text,
        )

        # Save results
        output_path = "outputs/total_test_simple.json"
        Path(output_path).parent.mkdir(exist_ok=True)

        result = {
            "pdf_path": pdf_path,
            "total_pages": len(pages),
            "first_page": {
                "dimensions": {"width": first_page.width, "height": first_page.height},
                "ocr_text": ocr_result.text,
                "confidence": getattr(ocr_result, "confidence", 0.0),
            },
        }

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"✅ Results saved to {output_path}")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Missing dependencies. Using basic PDF processing instead...")

    try:
        import fitz  # PyMuPDF

        print("Using PyMuPDF for basic text extraction...")
        doc = fitz.open("magazine/TOTAL.pdf")

        total_text = ""
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            total_text += f"\n--- Page {page_num + 1} ---\n" + text

        output_path = "outputs/total_basic_extraction.txt"
        Path(output_path).parent.mkdir(exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(total_text)

        print(f"✅ Basic text extraction saved to {output_path}")
        print(f"Extracted text from {len(doc)} pages ({len(total_text)} characters)")

    except ImportError:
        print("❌ No PDF processing libraries available")

except Exception as e:
    print(f"❌ Error processing PDF: {e}")
    import traceback

    traceback.print_exc()
