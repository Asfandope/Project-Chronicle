"""
Document type detector for OCR strategy selection.
Auto-detects if PDF is born-digital or scanned using multiple heuristics.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

import fitz  # PyMuPDF
import structlog
from PIL import Image

from .types import DocumentType, OCRError


logger = structlog.get_logger(__name__)


class DocumentTypeDetector:
    """
    Detects document type (born-digital vs scanned) using multiple heuristics.
    
    Implements PRD Section 5.3 auto-detection strategy with high accuracy.
    """
    
    def __init__(
        self,
        text_density_threshold: float = 0.05,
        image_coverage_threshold: float = 0.8,
        font_analysis_enabled: bool = True,
        image_analysis_enabled: bool = True,
        confidence_threshold: float = 0.8
    ):
        """
        Initialize document type detector.
        
        Args:
            text_density_threshold: Minimum text density for born-digital classification
            image_coverage_threshold: Minimum image coverage for scanned classification
            font_analysis_enabled: Whether to analyze font information
            image_analysis_enabled: Whether to analyze image properties
            confidence_threshold: Minimum confidence for definitive classification
        """
        self.text_density_threshold = text_density_threshold
        self.image_coverage_threshold = image_coverage_threshold
        self.font_analysis_enabled = font_analysis_enabled
        self.image_analysis_enabled = image_analysis_enabled
        self.confidence_threshold = confidence_threshold
        self.logger = logger.bind(component="DocumentTypeDetector")
    
    def detect(self, pdf_path: Path, max_pages_to_analyze: int = 5) -> Tuple[DocumentType, float, Dict[str, Any]]:
        """
        Detect document type with confidence score.
        
        Args:
            pdf_path: Path to PDF file
            max_pages_to_analyze: Maximum number of pages to analyze
            
        Returns:
            Tuple of (document_type, confidence, analysis_details)
            
        Raises:
            OCRError: If detection fails
        """
        start_time = time.time()
        
        try:
            self.logger.info("Starting document type detection", pdf_path=str(pdf_path))
            
            doc = fitz.open(str(pdf_path))
            
            try:
                pages_to_check = min(max_pages_to_analyze, doc.page_count)
                
                # Collect evidence from multiple heuristics
                evidence = {
                    "text_analysis": [],
                    "font_analysis": [],
                    "image_analysis": [],
                    "layout_analysis": [],
                    "metadata_analysis": {}
                }
                
                # Analyze pages
                for page_num in range(pages_to_check):
                    try:
                        page = doc[page_num]
                        
                        # Text density analysis
                        text_evidence = self._analyze_text_density(page, page_num)
                        evidence["text_analysis"].append(text_evidence)
                        
                        # Font analysis
                        if self.font_analysis_enabled:
                            font_evidence = self._analyze_fonts(page, page_num)
                            evidence["font_analysis"].append(font_evidence)
                        
                        # Image coverage analysis
                        if self.image_analysis_enabled:
                            image_evidence = self._analyze_image_coverage(page, page_num)
                            evidence["image_analysis"].append(image_evidence)
                        
                        # Layout analysis
                        layout_evidence = self._analyze_layout_structure(page, page_num)
                        evidence["layout_analysis"].append(layout_evidence)
                        
                    except Exception as e:
                        self.logger.warning("Error analyzing page for type detection",
                                          page_num=page_num, error=str(e))
                        continue
                
                # Metadata analysis
                evidence["metadata_analysis"] = self._analyze_metadata(doc)
                
                # Make classification decision
                document_type, confidence = self._classify_document(evidence)
                
                processing_time = time.time() - start_time
                
                # Compile analysis details
                analysis_details = {
                    "pages_analyzed": pages_to_check,
                    "processing_time": processing_time,
                    "evidence_summary": self._summarize_evidence(evidence),
                    "classification_factors": self._get_classification_factors(evidence),
                    "confidence_breakdown": self._get_confidence_breakdown(evidence)
                }
                
                self.logger.info("Document type detection completed",
                               pdf_path=str(pdf_path),
                               document_type=document_type.value,
                               confidence=confidence,
                               processing_time=processing_time)
                
                return document_type, confidence, analysis_details
                
            finally:
                doc.close()
                
        except Exception as e:
            self.logger.error("Error in document type detection",
                            pdf_path=str(pdf_path), error=str(e), exc_info=True)
            raise OCRError(f"Document type detection failed: {str(e)}")
    
    def _analyze_text_density(self, page: fitz.Page, page_num: int) -> Dict[str, Any]:
        """Analyze text density and extractability."""
        try:
            # Get page dimensions
            page_area = page.rect.width * page.rect.height
            
            # Extract text
            text = page.get_text()
            text_length = len(text.strip())
            
            # Get text blocks with positioning
            text_dict = page.get_text("dict")
            text_blocks = [block for block in text_dict.get("blocks", []) if "lines" in block]
            
            # Calculate text coverage area
            text_area = 0
            for block in text_blocks:
                if "bbox" in block:
                    x0, y0, x1, y1 = block["bbox"]
                    text_area += (x1 - x0) * (y1 - y0)
            
            text_density = text_area / page_area if page_area > 0 else 0
            char_density = text_length / page_area if page_area > 0 else 0
            
            # Analyze text characteristics
            has_selectable_text = text_length > 10
            has_font_info = len(text_blocks) > 0
            
            # Calculate extractability score
            extractability_score = min(1.0, (text_length / 100) * 0.1)  # Normalized by expected text
            
            return {
                "page_num": page_num,
                "text_length": text_length,
                "text_density": text_density,
                "char_density": char_density,
                "text_area": text_area,
                "page_area": page_area,
                "has_selectable_text": has_selectable_text,
                "has_font_info": has_font_info,
                "extractability_score": extractability_score,
                "text_block_count": len(text_blocks)
            }
            
        except Exception as e:
            self.logger.warning("Error in text density analysis", 
                              page_num=page_num, error=str(e))
            return {"page_num": page_num, "error": str(e)}
    
    def _analyze_fonts(self, page: fitz.Page, page_num: int) -> Dict[str, Any]:
        """Analyze font information to detect born-digital characteristics."""
        try:
            fonts_detected = set()
            font_sizes = []
            
            text_dict = page.get_text("dict")
            
            for block in text_dict.get("blocks", []):
                if "lines" not in block:
                    continue
                
                for line in block["lines"]:
                    for span in line.get("spans", []):
                        font_name = span.get("font", "")
                        font_size = span.get("size", 0)
                        
                        if font_name:
                            fonts_detected.add(font_name)
                        if font_size > 0:
                            font_sizes.append(font_size)
            
            # Analyze font characteristics
            unique_fonts = len(fonts_detected)
            has_standard_fonts = any(font in fonts_detected for font in [
                "Times", "Arial", "Helvetica", "Calibri", "Georgia"
            ])
            
            font_size_variation = np.std(font_sizes) if font_sizes else 0
            avg_font_size = np.mean(font_sizes) if font_sizes else 0
            
            # Born-digital PDFs typically have:
            # - Multiple distinct fonts
            # - Standard system fonts
            # - Consistent font sizing
            born_digital_indicators = sum([
                unique_fonts >= 2,
                has_standard_fonts,
                font_size_variation < 3.0,  # Consistent sizing
                8 <= avg_font_size <= 18    # Readable font sizes
            ])
            
            font_analysis_score = born_digital_indicators / 4.0
            
            return {
                "page_num": page_num,
                "unique_fonts": unique_fonts,
                "fonts_detected": list(fonts_detected),
                "has_standard_fonts": has_standard_fonts,
                "font_size_variation": font_size_variation,
                "avg_font_size": avg_font_size,
                "born_digital_indicators": born_digital_indicators,
                "font_analysis_score": font_analysis_score
            }
            
        except Exception as e:
            self.logger.warning("Error in font analysis", 
                              page_num=page_num, error=str(e))
            return {"page_num": page_num, "error": str(e)}
    
    def _analyze_image_coverage(self, page: fitz.Page, page_num: int) -> Dict[str, Any]:
        """Analyze image coverage to detect scanned documents."""
        try:
            page_area = page.rect.width * page.rect.height
            
            # Get images on the page
            image_list = page.get_images()
            
            total_image_area = 0
            large_images = 0
            
            for img_index, img_ref in enumerate(image_list):
                try:
                    # Get image dimensions
                    xref = img_ref[0]
                    pix = fitz.Pixmap(page.parent, xref)
                    
                    # Estimate image area on page (this is approximate)
                    # In reality, images might be scaled/cropped
                    img_area = pix.width * pix.height * 0.5  # Conservative estimate
                    total_image_area += img_area
                    
                    # Large images suggest scanned content
                    if pix.width > 1000 or pix.height > 1000:
                        large_images += 1
                    
                    pix = None  # Clean up
                    
                except Exception:
                    continue
            
            image_coverage = min(1.0, total_image_area / page_area) if page_area > 0 else 0
            
            # Scanned documents typically have:
            # - High image coverage (>80%)
            # - Large images (full page scans)
            # - Few but large images
            scanned_indicators = sum([
                image_coverage > 0.7,
                large_images > 0,
                len(image_list) <= 3,  # Usually one main image per page
                len(image_list) > 0    # Must have images
            ])
            
            scanned_score = scanned_indicators / 4.0
            
            return {
                "page_num": page_num,
                "image_count": len(image_list),
                "large_images": large_images,
                "total_image_area": total_image_area,
                "page_area": page_area,
                "image_coverage": image_coverage,
                "scanned_indicators": scanned_indicators,
                "scanned_score": scanned_score
            }
            
        except Exception as e:
            self.logger.warning("Error in image coverage analysis",
                              page_num=page_num, error=str(e))
            return {"page_num": page_num, "error": str(e)}
    
    def _analyze_layout_structure(self, page: fitz.Page, page_num: int) -> Dict[str, Any]:
        """Analyze layout structure and organization."""
        try:
            text_dict = page.get_text("dict")
            blocks = text_dict.get("blocks", [])
            
            # Count different types of content
            text_blocks = [b for b in blocks if "lines" in b]
            image_blocks = [b for b in blocks if "lines" not in b]
            
            # Analyze text block organization
            block_sizes = []
            block_positions = []
            
            for block in text_blocks:
                if "bbox" in block:
                    x0, y0, x1, y1 = block["bbox"]
                    width = x1 - x0
                    height = y1 - y0
                    block_sizes.append(width * height)
                    block_positions.append((x0, y0))
            
            # Born-digital documents typically have:
            # - Multiple organized text blocks
            # - Consistent block sizing
            # - Structured layout
            layout_organization_score = 0.0
            
            if text_blocks:
                size_consistency = 1.0 - (np.std(block_sizes) / np.mean(block_sizes)) if block_sizes else 0
                layout_organization_score = min(1.0, len(text_blocks) / 10.0 + size_consistency * 0.5)
            
            return {
                "page_num": page_num,
                "text_blocks": len(text_blocks),
                "image_blocks": len(image_blocks),
                "block_sizes": block_sizes,
                "layout_organization_score": layout_organization_score,
                "has_structured_layout": len(text_blocks) > 2
            }
            
        except Exception as e:
            self.logger.warning("Error in layout structure analysis",
                              page_num=page_num, error=str(e))
            return {"page_num": page_num, "error": str(e)}
    
    def _analyze_metadata(self, doc: fitz.Document) -> Dict[str, Any]:
        """Analyze document metadata for type hints."""
        try:
            metadata = doc.metadata
            
            # Check creator/producer for scanning software
            creator = metadata.get("creator", "").lower()
            producer = metadata.get("producer", "").lower()
            
            scanning_keywords = [
                "scan", "scanner", "ocr", "abbyy", "readiris", 
                "finereader", "omnipage", "tesseract", "adobe acrobat"
            ]
            
            born_digital_keywords = [
                "microsoft", "word", "excel", "powerpoint", "latex",
                "indesign", "illustrator", "photoshop", "quark"
            ]
            
            has_scanning_indicators = any(keyword in creator + producer for keyword in scanning_keywords)
            has_born_digital_indicators = any(keyword in creator + producer for keyword in born_digital_keywords)
            
            return {
                "creator": creator,
                "producer": producer,
                "has_scanning_indicators": has_scanning_indicators,
                "has_born_digital_indicators": has_born_digital_indicators,
                "creation_date": metadata.get("creationDate"),
                "modification_date": metadata.get("modDate")
            }
            
        except Exception as e:
            self.logger.warning("Error in metadata analysis", error=str(e))
            return {"error": str(e)}
    
    def _classify_document(self, evidence: Dict[str, Any]) -> Tuple[DocumentType, float]:
        """Make final classification decision based on collected evidence."""
        try:
            # Aggregate scores from different analyses
            text_scores = []
            font_scores = []
            image_scores = []
            layout_scores = []
            
            # Text analysis scores
            for text_analysis in evidence["text_analysis"]:
                if "error" not in text_analysis:
                    # Higher extractability indicates born-digital
                    text_scores.append(text_analysis["extractability_score"])
            
            # Font analysis scores
            for font_analysis in evidence["font_analysis"]:
                if "error" not in font_analysis:
                    # Higher font analysis score indicates born-digital
                    font_scores.append(font_analysis["font_analysis_score"])
            
            # Image analysis scores
            for image_analysis in evidence["image_analysis"]:
                if "error" not in image_analysis:
                    # Higher scanned score indicates scanned document
                    image_scores.append(image_analysis["scanned_score"])
            
            # Layout analysis scores
            for layout_analysis in evidence["layout_analysis"]:
                if "error" not in layout_analysis:
                    # Higher organization score indicates born-digital
                    layout_scores.append(layout_analysis["layout_organization_score"])
            
            # Calculate average scores
            avg_text_score = np.mean(text_scores) if text_scores else 0
            avg_font_score = np.mean(font_scores) if font_scores else 0
            avg_image_score = np.mean(image_scores) if image_scores else 0
            avg_layout_score = np.mean(layout_scores) if layout_scores else 0
            
            # Born-digital evidence (higher is more born-digital)
            born_digital_evidence = (
                avg_text_score * 0.3 +      # 30% weight on text extractability
                avg_font_score * 0.25 +     # 25% weight on font analysis
                avg_layout_score * 0.2 +    # 20% weight on layout structure
                (1 - avg_image_score) * 0.25  # 25% weight on image analysis (inverted)
            )
            
            # Metadata influence
            metadata = evidence["metadata_analysis"]
            if metadata.get("has_born_digital_indicators"):
                born_digital_evidence += 0.1
            elif metadata.get("has_scanning_indicators"):
                born_digital_evidence -= 0.1
            
            # Determine document type and confidence
            if born_digital_evidence >= 0.7:
                document_type = DocumentType.BORN_DIGITAL
                confidence = min(0.95, born_digital_evidence)
            elif born_digital_evidence <= 0.3:
                document_type = DocumentType.SCANNED
                confidence = min(0.95, 1.0 - born_digital_evidence)
            elif 0.4 <= born_digital_evidence <= 0.6:
                document_type = DocumentType.HYBRID
                confidence = 0.8  # Lower confidence for hybrid classification
            else:
                document_type = DocumentType.UNKNOWN
                confidence = 0.5
            
            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                document_type = DocumentType.UNKNOWN
                confidence = 0.5
            
            return document_type, confidence
            
        except Exception as e:
            self.logger.error("Error in document classification", error=str(e))
            return DocumentType.UNKNOWN, 0.5
    
    def _summarize_evidence(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize collected evidence for reporting."""
        summary = {
            "pages_with_text": 0,
            "pages_with_fonts": 0,
            "pages_with_images": 0,
            "avg_text_density": 0.0,
            "avg_image_coverage": 0.0,
            "total_unique_fonts": set(),
            "has_metadata_indicators": False
        }
        
        try:
            # Summarize text analysis
            text_densities = []
            for analysis in evidence["text_analysis"]:
                if "error" not in analysis and analysis["has_selectable_text"]:
                    summary["pages_with_text"] += 1
                    text_densities.append(analysis["text_density"])
            
            if text_densities:
                summary["avg_text_density"] = np.mean(text_densities)
            
            # Summarize font analysis
            for analysis in evidence["font_analysis"]:
                if "error" not in analysis and analysis["unique_fonts"] > 0:
                    summary["pages_with_fonts"] += 1
                    summary["total_unique_fonts"].update(analysis["fonts_detected"])
            
            # Summarize image analysis
            image_coverages = []
            for analysis in evidence["image_analysis"]:
                if "error" not in analysis and analysis["image_count"] > 0:
                    summary["pages_with_images"] += 1
                    image_coverages.append(analysis["image_coverage"])
            
            if image_coverages:
                summary["avg_image_coverage"] = np.mean(image_coverages)
            
            # Metadata summary
            metadata = evidence["metadata_analysis"]
            summary["has_metadata_indicators"] = (
                metadata.get("has_born_digital_indicators", False) or
                metadata.get("has_scanning_indicators", False)
            )
            
            # Convert set to list for JSON serialization
            summary["total_unique_fonts"] = list(summary["total_unique_fonts"])
            
        except Exception as e:
            self.logger.warning("Error summarizing evidence", error=str(e))
        
        return summary
    
    def _get_classification_factors(self, evidence: Dict[str, Any]) -> List[str]:
        """Get human-readable classification factors."""
        factors = []
        
        try:
            # Check text extractability
            text_analyses = [a for a in evidence["text_analysis"] if "error" not in a]
            if text_analyses:
                avg_extractability = np.mean([a["extractability_score"] for a in text_analyses])
                if avg_extractability > 0.5:
                    factors.append("High text extractability")
                else:
                    factors.append("Low text extractability")
            
            # Check font diversity
            font_analyses = [a for a in evidence["font_analysis"] if "error" not in a]
            if font_analyses:
                total_fonts = sum(a["unique_fonts"] for a in font_analyses)
                if total_fonts > 5:
                    factors.append("Multiple fonts detected")
                elif total_fonts < 2:
                    factors.append("Few fonts detected")
            
            # Check image coverage
            image_analyses = [a for a in evidence["image_analysis"] if "error" not in a]
            if image_analyses:
                avg_coverage = np.mean([a["image_coverage"] for a in image_analyses])
                if avg_coverage > 0.8:
                    factors.append("High image coverage")
                elif avg_coverage < 0.2:
                    factors.append("Low image coverage")
            
            # Check metadata
            metadata = evidence["metadata_analysis"]
            if metadata.get("has_born_digital_indicators"):
                factors.append("Born-digital software detected")
            elif metadata.get("has_scanning_indicators"):
                factors.append("Scanning software detected")
            
        except Exception as e:
            self.logger.warning("Error getting classification factors", error=str(e))
        
        return factors
    
    def _get_confidence_breakdown(self, evidence: Dict[str, Any]) -> Dict[str, float]:
        """Get confidence breakdown by analysis type."""
        breakdown = {
            "text_analysis": 0.0,
            "font_analysis": 0.0,
            "image_analysis": 0.0,
            "layout_analysis": 0.0,
            "metadata_analysis": 0.0
        }
        
        try:
            # Text analysis confidence
            text_scores = [a["extractability_score"] for a in evidence["text_analysis"] if "error" not in a]
            if text_scores:
                breakdown["text_analysis"] = np.mean(text_scores)
            
            # Font analysis confidence
            font_scores = [a["font_analysis_score"] for a in evidence["font_analysis"] if "error" not in a]
            if font_scores:
                breakdown["font_analysis"] = np.mean(font_scores)
            
            # Image analysis confidence
            image_scores = [a["scanned_score"] for a in evidence["image_analysis"] if "error" not in a]
            if image_scores:
                breakdown["image_analysis"] = np.mean(image_scores)
            
            # Layout analysis confidence
            layout_scores = [a["layout_organization_score"] for a in evidence["layout_analysis"] if "error" not in a]
            if layout_scores:
                breakdown["layout_analysis"] = np.mean(layout_scores)
            
            # Metadata analysis confidence
            metadata = evidence["metadata_analysis"]
            if metadata.get("has_born_digital_indicators") or metadata.get("has_scanning_indicators"):
                breakdown["metadata_analysis"] = 0.8
            else:
                breakdown["metadata_analysis"] = 0.5
                
        except Exception as e:
            self.logger.warning("Error calculating confidence breakdown", error=str(e))
        
        return breakdown