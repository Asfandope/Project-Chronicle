#!/usr/bin/env python3
"""
Simple script to create test gold standard data without external dependencies.

This creates basic XML ground truth files and metadata for testing the
dataset validation and ingestion systems.
"""

import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from datetime import datetime
import random


def create_sample_xml(brand: str, document_id: str, num_articles: int = 3) -> str:
    """Create a sample XML ground truth file."""
    
    # Create root element
    root = ET.Element("magazine")
    root.set("brand", brand)
    root.set("issue_date", "2024-01-15")
    root.set("total_pages", str(random.randint(20, 40)))
    root.set("document_id", document_id)
    
    # Add articles
    current_page = 1
    
    for i in range(num_articles):
        article = ET.SubElement(root, "article")
        article.set("id", f"article_{i:03d}")
        article.set("start_page", str(current_page))
        
        # Article spans 1-3 pages
        pages = random.randint(1, 3)
        end_page = current_page + pages - 1
        article.set("end_page", str(end_page))
        
        # Add title
        title = ET.SubElement(article, "title")
        title.text = f"Sample Article {i+1}: Analysis of Current Trends"
        title.set("confidence", str(round(random.uniform(0.90, 0.98), 3)))
        
        # Add body paragraphs
        num_paragraphs = random.randint(3, 6)
        for j in range(num_paragraphs):
            body = ET.SubElement(article, "body")
            body.text = f"This is paragraph {j+1} of article {i+1}. It contains detailed analysis and insights about the topic at hand. The content is designed to test the extraction and validation systems."
            body.set("confidence", str(round(random.uniform(0.88, 0.95), 3)))
            body.set("paragraph_index", str(j))
        
        # Add contributors (sometimes)
        if random.random() < 0.8:
            contributors = ET.SubElement(article, "contributors")
            
            # 1-2 contributors
            for k in range(random.randint(1, 2)):
                contributor = ET.SubElement(contributors, "contributor")
                contributor.set("name", f"Author {chr(65 + i + k)} Smith")
                contributor.set("role", random.choice(["author", "correspondent", "editor"]))
                contributor.set("confidence", str(round(random.uniform(0.85, 0.92), 3)))
        
        # Add images (sometimes)
        if random.random() < 0.6:
            images = ET.SubElement(article, "images")
            
            for k in range(random.randint(1, 2)):
                image = ET.SubElement(images, "image")
                image.set("id", f"img_{i}_{k}")
                image.set("page", str(random.randint(current_page, end_page)))
                image.set("bbox", f"100,{200 + k * 150},400,{350 + k * 150}")
                
                # Add caption
                caption = ET.SubElement(image, "caption")
                caption.text = f"Caption for image {k+1} in article {i+1}"
                caption.set("confidence", str(round(random.uniform(0.80, 0.90), 3)))
        
        current_page = end_page + 1
    
    # Format XML with pretty printing
    rough_string = ET.tostring(root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def create_sample_metadata(brand: str, document_id: str, xml_filename: str) -> dict:
    """Create sample metadata file."""
    
    return {
        "dataset_info": {
            "brand": brand,
            "filename": xml_filename,
            "creation_date": datetime.now().isoformat(),
            "file_type": "test_ground_truth"
        },
        "quality_metrics": {
            "manual_validation": True,
            "annotation_quality": round(random.uniform(0.90, 0.98), 3),
            "completeness_score": round(random.uniform(0.92, 0.99), 3)
        },
        "content_info": {
            "page_count": random.randint(20, 40),
            "article_count": 3,
            "layout_complexity": random.choice(["simple", "standard", "complex"])
        },
        "test_metadata": {
            "generated_timestamp": datetime.now().isoformat(),
            "generation_method": "test_script",
            "document_id": document_id,
            "purpose": "validation_testing"
        }
    }


def create_brand_test_data(brand: str, num_documents: int = 3) -> None:
    """Create test data for a specific brand."""
    
    print(f"Creating test data for {brand}...")
    
    # Ensure directories exist
    brand_path = Path(f"data/gold_sets/{brand}")
    (brand_path / "ground_truth").mkdir(parents=True, exist_ok=True)
    (brand_path / "metadata").mkdir(parents=True, exist_ok=True)
    
    files_created = []
    
    for i in range(num_documents):
        document_id = f"{brand}_test_{datetime.now().strftime('%Y%m%d')}_{i:03d}"
        xml_filename = f"{document_id}.xml"
        
        # Create XML ground truth
        xml_content = create_sample_xml(brand, document_id)
        xml_path = brand_path / "ground_truth" / xml_filename
        
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(xml_content)
        
        files_created.append(str(xml_path))
        
        # Create metadata
        metadata = create_sample_metadata(brand, document_id, xml_filename)
        metadata_path = brand_path / "metadata" / f"{document_id}_metadata.json"
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        files_created.append(str(metadata_path))
        
        print(f"  Created: {xml_filename}")
    
    print(f"✅ Created {len(files_created)} files for {brand}")
    return files_created


def main():
    """Main function to create test data for all brands."""
    brands = ["economist", "time", "newsweek", "vogue"]
    
    print("Creating test gold standard datasets...")
    print("=" * 50)
    
    all_files = []
    
    for brand in brands:
        files = create_brand_test_data(brand, 3)
        all_files.extend(files)
        print()
    
    print(f"🎉 Test data creation completed!")
    print(f"📊 Total files created: {len(all_files)}")
    print(f"📁 Files available in: data/gold_sets/{{brand}}/")
    print()
    print("Next steps:")
    print("1. Run validation: python scripts/validate_datasets.py")
    print("2. Test specific brand: python scripts/validate_datasets.py economist")
    print("3. Generate report: python -c \"from data_management.schema_validator import *; DatasetValidator().validate_brand_dataset('economist')\"")


if __name__ == "__main__":
    main()