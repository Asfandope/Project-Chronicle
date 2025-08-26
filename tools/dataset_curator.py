#!/usr/bin/env python3
"""
Dataset Curator for Gold Standard Magazine Extraction Data

This tool provides functionality for:
1. PDF quality analysis and assessment
2. Ground truth generation and validation
3. Dataset curation and management
4. Quality control and reporting
"""

import os
import json
import logging
import argparse
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from enum import Enum

# Optional imports with fallbacks
try:
    import pymupdf as fitz
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    print("Warning: PyMuPDF not available. PDF analysis features will be limited.")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL not available. Image analysis features will be limited.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QualityLevel(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"  
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILED = "failed"

@dataclass
class PDFAnalysis:
    """Results from PDF analysis"""
    file_path: str
    page_count: int
    file_size: int
    is_born_digital: bool
    estimated_ocr_quality: float
    layout_complexity: str
    has_images: bool
    text_density: float
    quality_level: QualityLevel
    issues: List[str]
    analysis_timestamp: str

@dataclass  
class DatasetMetadata:
    """Metadata for gold standard dataset entries"""
    file_id: str
    brand: str
    publication_date: str
    issue_number: str
    page_range: str
    annotator: str
    creation_date: str
    validation_status: str
    quality_scores: Dict[str, float]
    content_types: List[str]
    layout_features: List[str]
    file_hash: str
    schema_version: str

class DatasetCurator:
    """Main class for dataset curation operations"""
    
    def __init__(self, base_path: str = "data/gold_sets"):
        self.base_path = Path(base_path)
        self.brands = ["economist", "time", "newsweek", "vogue"]
        self.schema_version = "v1.0"
        
        # Quality thresholds from requirements
        self.quality_thresholds = {
            "ocr_accuracy_born_digital": 0.9995,  # WER < 0.0005
            "ocr_accuracy_scanned": 0.985,        # WER < 0.015
            "layout_classification": 0.995,       # >99.5%
            "article_reconstruction": 0.98        # >98%
        }
        
    def analyze_pdf(self, pdf_path: str) -> PDFAnalysis:
        """Analyze PDF quality and characteristics"""
        logger.info(f"Analyzing PDF: {pdf_path}")
        
        file_path = Path(pdf_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Basic file info
        file_size = file_path.stat().st_size
        issues = []
        
        if not HAS_PYMUPDF:
            # Fallback analysis without PyMuPDF
            return PDFAnalysis(
                file_path=str(file_path),
                page_count=1,  # Assume single page
                file_size=file_size,
                is_born_digital=True,  # Assume born digital
                estimated_ocr_quality=0.95,
                layout_complexity="unknown",
                has_images=False,
                text_density=0.5,
                quality_level=QualityLevel.ACCEPTABLE,
                issues=["PyMuPDF unavailable - limited analysis"],
                analysis_timestamp=datetime.now().isoformat()
            )
        
        try:
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            
            # Analyze first few pages for characteristics
            total_text_length = 0
            total_images = 0
            total_area = 0
            vector_content = 0
            
            for page_num in range(min(3, page_count)):  # Sample first 3 pages
                page = doc[page_num]
                
                # Text analysis
                text = page.get_text()
                total_text_length += len(text)
                
                # Image analysis
                images = page.get_images()
                total_images += len(images)
                
                # Check for vector vs raster content
                drawings = page.get_drawings()
                if drawings:
                    vector_content += len(drawings)
                
                # Page area
                rect = page.rect
                total_area += rect.width * rect.height
            
            doc.close()
            
            # Determine if born digital vs scanned
            # Born digital typically has more vector content and cleaner text
            is_born_digital = vector_content > 0 or (total_text_length / max(1, page_count) > 500)
            
            # Estimate OCR quality (higher for born digital)
            if is_born_digital:
                estimated_ocr_quality = 0.999  # Very high for born digital
            else:
                # Estimate based on text density and other factors
                text_per_page = total_text_length / max(1, page_count)
                if text_per_page > 1000:
                    estimated_ocr_quality = 0.99
                elif text_per_page > 500:
                    estimated_ocr_quality = 0.98
                else:
                    estimated_ocr_quality = 0.95
                    issues.append("Low text density may indicate poor OCR")
            
            # Layout complexity assessment
            images_per_page = total_images / max(1, page_count)
            if images_per_page > 3:
                layout_complexity = "high"
            elif images_per_page > 1:
                layout_complexity = "medium"  
            else:
                layout_complexity = "low"
            
            # Text density
            if total_area > 0:
                text_density = total_text_length / (total_area / 1000000)  # chars per sq inch approx
            else:
                text_density = 0.0
            
            # Quality assessment
            quality_level = self._assess_quality(
                is_born_digital, estimated_ocr_quality, layout_complexity, issues
            )
            
            return PDFAnalysis(
                file_path=str(file_path),
                page_count=page_count,
                file_size=file_size,
                is_born_digital=is_born_digital,
                estimated_ocr_quality=estimated_ocr_quality,
                layout_complexity=layout_complexity,
                has_images=total_images > 0,
                text_density=text_density,
                quality_level=quality_level,
                issues=issues,
                analysis_timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error analyzing PDF {pdf_path}: {str(e)}")
            return PDFAnalysis(
                file_path=str(file_path),
                page_count=0,
                file_size=file_size,
                is_born_digital=False,
                estimated_ocr_quality=0.0,
                layout_complexity="unknown",
                has_images=False,
                text_density=0.0,
                quality_level=QualityLevel.FAILED,
                issues=[f"Analysis failed: {str(e)}"],
                analysis_timestamp=datetime.now().isoformat()
            )
    
    def _assess_quality(self, is_born_digital: bool, ocr_quality: float, 
                       layout_complexity: str, issues: List[str]) -> QualityLevel:
        """Assess overall quality level"""
        
        # Check OCR quality against thresholds
        threshold = (self.quality_thresholds["ocr_accuracy_born_digital"] 
                    if is_born_digital 
                    else self.quality_thresholds["ocr_accuracy_scanned"])
        
        if len(issues) > 3:
            return QualityLevel.FAILED
        elif ocr_quality < threshold - 0.02:
            return QualityLevel.POOR
        elif ocr_quality < threshold:
            return QualityLevel.ACCEPTABLE
        elif layout_complexity == "high" and len(issues) == 0:
            return QualityLevel.EXCELLENT
        else:
            return QualityLevel.GOOD
    
    def generate_metadata(self, brand: str, pdf_path: str, 
                         analysis: PDFAnalysis, annotator: str = "system") -> DatasetMetadata:
        """Generate metadata for a dataset entry"""
        
        file_path = Path(pdf_path)
        file_hash = self._calculate_file_hash(pdf_path)
        
        # Extract info from filename if following convention
        filename = file_path.stem
        parts = filename.split('_')
        
        if len(parts) >= 4 and parts[0] == brand:
            publication_date = f"{parts[1]}_{parts[2]}_{parts[3]}"  
            issue_number = parts[4] if len(parts) > 4 else "unknown"
            page_range = "_".join(parts[5:]) if len(parts) > 5 else "unknown"
        else:
            publication_date = "unknown"
            issue_number = "unknown"
            page_range = "unknown"
        
        # Generate quality scores
        quality_scores = {
            "estimated_ocr_accuracy": analysis.estimated_ocr_quality,
            "layout_complexity_score": self._complexity_to_score(analysis.layout_complexity),
            "text_density_score": min(1.0, analysis.text_density / 100.0),
            "overall_quality": self._quality_to_score(analysis.quality_level)
        }
        
        # Determine content types based on analysis
        content_types = ["text"]
        if analysis.has_images:
            content_types.append("images")
        if analysis.layout_complexity == "high":
            content_types.append("complex_layout")
            
        # Layout features
        layout_features = [f"complexity_{analysis.layout_complexity}"]
        if analysis.has_images:
            layout_features.append("image_rich")
        if analysis.text_density > 50:
            layout_features.append("text_dense")
        
        return DatasetMetadata(
            file_id=filename,
            brand=brand,
            publication_date=publication_date,
            issue_number=issue_number,
            page_range=page_range,
            annotator=annotator,
            creation_date=datetime.now().isoformat(),
            validation_status="pending",
            quality_scores=quality_scores,
            content_types=content_types,
            layout_features=layout_features,
            file_hash=file_hash,
            schema_version=self.schema_version
        )
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _complexity_to_score(self, complexity: str) -> float:
        """Convert complexity level to numeric score"""
        mapping = {"low": 0.3, "medium": 0.6, "high": 1.0, "unknown": 0.5}
        return mapping.get(complexity, 0.5)
    
    def _quality_to_score(self, quality: QualityLevel) -> float:
        """Convert quality level to numeric score"""
        mapping = {
            QualityLevel.EXCELLENT: 1.0,
            QualityLevel.GOOD: 0.8,
            QualityLevel.ACCEPTABLE: 0.6,
            QualityLevel.POOR: 0.4,
            QualityLevel.FAILED: 0.0
        }
        return mapping.get(quality, 0.0)
    
    def validate_xml_schema(self, xml_path: str) -> Tuple[bool, List[str]]:
        """Validate ground truth XML against schema"""
        issues = []
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Check root element
            if root.tag != "magazine_extraction":
                issues.append(f"Invalid root element: {root.tag}")
            
            # Check required elements
            required_elements = ["metadata", "pages", "articles"]
            for element in required_elements:
                if root.find(element) is None:
                    issues.append(f"Missing required element: {element}")
            
            # Validate articles structure
            articles = root.find("articles")
            if articles is not None:
                for article in articles.findall("article"):
                    if not article.get("id"):
                        issues.append("Article missing id attribute")
                    
                    if article.find("title") is None:
                        issues.append(f"Article {article.get('id', 'unknown')} missing title")
                    
                    if article.find("content") is None:
                        issues.append(f"Article {article.get('id', 'unknown')} missing content")
            
            # Validate pages structure
            pages = root.find("pages")
            if pages is not None:
                for page in pages.findall("page"):
                    if not page.get("number"):
                        issues.append("Page missing number attribute")
                    
                    blocks = page.find("blocks")
                    if blocks is None:
                        issues.append(f"Page {page.get('number', 'unknown')} missing blocks")
            
        except ET.ParseError as e:
            issues.append(f"XML parsing error: {str(e)}")
        except Exception as e:
            issues.append(f"Validation error: {str(e)}")
        
        return len(issues) == 0, issues
    
    def save_metadata(self, metadata: DatasetMetadata, output_path: str):
        """Save metadata to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        logger.info(f"Metadata saved to {output_path}")
    
    def generate_quality_report(self, brand: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive quality report for datasets"""
        logger.info(f"Generating quality report for {brand or 'all brands'}")
        
        brands_to_check = [brand] if brand else self.brands
        report = {
            "generation_time": datetime.now().isoformat(),
            "brands": {}
        }
        
        for brand_name in brands_to_check:
            brand_path = self.base_path / brand_name
            if not brand_path.exists():
                continue
                
            brand_report = {
                "total_files": 0,
                "quality_distribution": {},
                "validation_status": {},
                "average_scores": {},
                "issues": []
            }
            
            # Check metadata files
            metadata_path = brand_path / "metadata"
            if metadata_path.exists():
                quality_levels = []
                validation_statuses = []
                all_quality_scores = []
                
                for metadata_file in metadata_path.glob("*.json"):
                    try:
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                        
                        brand_report["total_files"] += 1
                        
                        # Track quality scores
                        if "quality_scores" in metadata:
                            all_quality_scores.append(metadata["quality_scores"])
                        
                        # Track validation status
                        validation_status = metadata.get("validation_status", "unknown")
                        validation_statuses.append(validation_status)
                        
                    except Exception as e:
                        brand_report["issues"].append(f"Error reading {metadata_file}: {str(e)}")
                
                # Calculate distributions
                if validation_statuses:
                    from collections import Counter
                    brand_report["validation_status"] = dict(Counter(validation_statuses))
                
                # Calculate average scores
                if all_quality_scores:
                    avg_scores = {}
                    for score_key in all_quality_scores[0].keys():
                        scores = [qs[score_key] for qs in all_quality_scores if score_key in qs]
                        avg_scores[score_key] = sum(scores) / len(scores) if scores else 0.0
                    brand_report["average_scores"] = avg_scores
            
            report["brands"][brand_name] = brand_report
        
        return report

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Dataset Curator for Gold Standard Data")
    parser.add_argument("command", choices=["analyze", "validate", "report", "curate"], 
                       help="Command to execute")
    parser.add_argument("--pdf", help="Path to PDF file for analysis")
    parser.add_argument("--xml", help="Path to XML file for validation") 
    parser.add_argument("--brand", help="Brand name for operations")
    parser.add_argument("--output", help="Output path for results")
    parser.add_argument("--base-path", default="data/gold_sets", 
                       help="Base path for gold standard data")
    
    args = parser.parse_args()
    
    curator = DatasetCurator(args.base_path)
    
    if args.command == "analyze" and args.pdf:
        analysis = curator.analyze_pdf(args.pdf)
        print(json.dumps(asdict(analysis), indent=2))
        
        if args.brand:
            metadata = curator.generate_metadata(args.brand, args.pdf, analysis)
            if args.output:
                curator.save_metadata(metadata, args.output)
            else:
                print(json.dumps(asdict(metadata), indent=2))
    
    elif args.command == "validate" and args.xml:
        is_valid, issues = curator.validate_xml_schema(args.xml)
        result = {"valid": is_valid, "issues": issues}
        print(json.dumps(result, indent=2))
    
    elif args.command == "report":
        report = curator.generate_quality_report(args.brand)
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
        else:
            print(json.dumps(report, indent=2))
    
    elif args.command == "curate":
        print("Dataset curation workflow not yet implemented")
        # TODO: Implement full curation workflow
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()