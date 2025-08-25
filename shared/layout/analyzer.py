"""
Main layout analyzer - extracts text blocks with coordinates and basic classification.
"""

import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import fitz  # PyMuPDF
import structlog

from .types import (
    LayoutError, LayoutResult, PageLayout, TextBlock, BoundingBox, 
    BlockType, LayoutConfig
)
from .classifier import BlockClassifier


logger = structlog.get_logger(__name__)


class LayoutAnalyzer:
    """
    Simple rule-based layout analyzer for MVP.
    
    Extracts text blocks with coordinates and classifies them using
    rule-based heuristics. Will be replaced with LayoutLM in future iterations.
    """
    
    def __init__(self, config: Optional[LayoutConfig] = None):
        """
        Initialize layout analyzer.
        
        Args:
            config: Layout analysis configuration
        """
        self.config = config or LayoutConfig.create_default()
        self.classifier = BlockClassifier(self.config)
        self.logger = logger.bind(component="LayoutAnalyzer")
    
    def analyze_pdf(
        self, 
        pdf_path: Path, 
        page_range: Optional[Tuple[int, int]] = None
    ) -> LayoutResult:
        """
        Analyze layout of entire PDF document.
        
        Args:
            pdf_path: Path to PDF file
            page_range: Optional page range (start, end) 1-indexed
            
        Returns:
            Complete layout analysis result
            
        Raises:
            LayoutError: If analysis fails
        """
        start_time = time.time()
        
        try:
            self.logger.info("Starting PDF layout analysis", pdf_path=str(pdf_path))
            
            if not pdf_path.exists():
                raise LayoutError(f"PDF file does not exist: {pdf_path}", pdf_path=pdf_path)
            
            # Open PDF document
            try:
                doc = fitz.open(str(pdf_path))
            except Exception as e:
                raise LayoutError(f"Cannot open PDF: {str(e)}", pdf_path=pdf_path)
            
            try:
                total_pages = doc.page_count
                
                # Determine page range
                if page_range:
                    start_page, end_page = page_range
                    start_page = max(1, start_page)
                    end_page = min(total_pages, end_page)
                else:
                    start_page, end_page = 1, total_pages
                
                pages = []
                
                # Analyze each page
                for page_num in range(start_page, end_page + 1):
                    try:
                        page_layout = self.analyze_page(doc, page_num - 1)  # Convert to 0-indexed
                        pages.append(page_layout)
                        
                    except Exception as e:
                        self.logger.error("Error analyzing page layout",
                                        page_num=page_num, error=str(e))
                        # Create empty page layout for failed pages
                        pages.append(PageLayout(
                            page_num=page_num,
                            page_width=0,
                            page_height=0,
                            text_blocks=[],
                            processing_time=0
                        ))
                
                total_processing_time = time.time() - start_time
                
                # Create result
                result = LayoutResult(
                    pdf_path=pdf_path,
                    pages=pages,
                    total_processing_time=total_processing_time,
                    analysis_config=self._get_config_summary()
                )
                
                self.logger.info("PDF layout analysis completed",
                               pdf_path=str(pdf_path),
                               pages_analyzed=len(pages),
                               total_blocks=result.total_blocks,
                               processing_time=total_processing_time)
                
                return result
                
            finally:
                doc.close()
                
        except LayoutError:
            raise
        except Exception as e:
            self.logger.error("Unexpected error in PDF layout analysis",
                            pdf_path=str(pdf_path), error=str(e), exc_info=True)
            raise LayoutError(f"PDF layout analysis failed: {str(e)}", pdf_path=pdf_path)
    
    def analyze_page(self, doc: fitz.Document, page_index: int) -> PageLayout:
        """
        Analyze layout of a single page.
        
        Args:
            doc: PyMuPDF document
            page_index: Page index (0-based)
            
        Returns:
            Page layout analysis result
        """
        start_time = time.time()
        page_num = page_index + 1
        
        try:
            self.logger.debug("Analyzing page layout", page_num=page_num)
            
            page = doc[page_index]
            page_rect = page.rect
            
            # Extract text blocks with coordinates
            text_blocks = self._extract_text_blocks(page, page_num)
            
            # Classify text blocks
            classified_blocks = self.classifier.classify_blocks(text_blocks, page_rect)
            
            # Determine reading order
            if self.config.enable_reading_order:
                classified_blocks = self._determine_reading_order(classified_blocks, page_rect)
            
            # Detect columns
            if self.config.enable_column_detection:
                classified_blocks = self._detect_columns(classified_blocks, page_rect)
            
            processing_time = time.time() - start_time
            
            page_layout = PageLayout(
                page_num=page_num,
                page_width=page_rect.width,
                page_height=page_rect.height,
                text_blocks=classified_blocks,
                processing_time=processing_time
            )
            
            self.logger.debug("Page layout analysis completed",
                            page_num=page_num,
                            blocks_found=len(classified_blocks),
                            processing_time=processing_time)
            
            return page_layout
            
        except Exception as e:
            self.logger.error("Error analyzing page layout",
                            page_num=page_num, error=str(e))
            raise LayoutError(f"Page {page_num} layout analysis failed: {str(e)}", page_num)
    
    def _extract_text_blocks(self, page: fitz.Page, page_num: int) -> List[TextBlock]:
        """Extract text blocks with coordinates and font information."""
        try:
            text_blocks = []
            
            # Get text with detailed formatting information
            text_dict = page.get_text("dict")
            
            for block_idx, block in enumerate(text_dict.get("blocks", [])):
                if "lines" not in block:
                    continue  # Skip image blocks
                
                # Extract block-level information
                block_bbox_raw = block.get("bbox", (0, 0, 0, 0))
                block_bbox = BoundingBox(*block_bbox_raw)
                
                # Process lines within the block
                block_text_parts = []
                font_info = self._analyze_block_fonts(block)
                
                for line in block["lines"]:
                    line_text_parts = []
                    
                    for span in line.get("spans", []):
                        span_text = span.get("text", "").strip()
                        if span_text:
                            line_text_parts.append(span_text)
                    
                    if line_text_parts:
                        line_text = " ".join(line_text_parts)
                        block_text_parts.append(line_text)
                
                # Create text block if we have content
                if block_text_parts:
                    block_text = "\n".join(block_text_parts)
                    
                    # Filter out very short text blocks
                    if len(block_text.strip()) >= self.config.min_text_length:
                        text_block = TextBlock(
                            text=block_text,
                            bbox=block_bbox,
                            page_num=page_num,
                            font_size=font_info.get("avg_size"),
                            font_family=font_info.get("primary_family"),
                            is_bold=font_info.get("is_bold", False),
                            is_italic=font_info.get("is_italic", False),
                            classification_features={
                                "block_index": block_idx,
                                "font_info": font_info
                            }
                        )
                        text_blocks.append(text_block)
            
            # Merge nearby blocks if enabled
            if self.config.merge_nearby_blocks:
                text_blocks = self._merge_nearby_blocks(text_blocks)
            
            return text_blocks
            
        except Exception as e:
            self.logger.error("Error extracting text blocks",
                            page_num=page_num, error=str(e))
            return []
    
    def _analyze_block_fonts(self, block: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze font properties within a text block."""
        try:
            font_sizes = []
            font_families = []
            is_bold = False
            is_italic = False
            
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("text", "").strip():  # Only consider non-empty spans
                        # Font size
                        size = span.get("size", 0)
                        if size > 0:
                            font_sizes.append(size)
                        
                        # Font family
                        font = span.get("font", "")
                        if font:
                            # Remove subset prefix if present
                            clean_font = font.split("+")[-1]
                            font_families.append(clean_font)
                        
                        # Font flags
                        flags = span.get("flags", 0)
                        if flags & 2**4:  # Bold flag
                            is_bold = True
                        if flags & 2**1:  # Italic flag
                            is_italic = True
            
            # Calculate average font size
            avg_size = sum(font_sizes) / len(font_sizes) if font_sizes else None
            
            # Find most common font family
            if font_families:
                from collections import Counter
                primary_family = Counter(font_families).most_common(1)[0][0]
            else:
                primary_family = None
            
            return {
                "avg_size": avg_size,
                "primary_family": primary_family,
                "is_bold": is_bold,
                "is_italic": is_italic,
                "size_variation": max(font_sizes) - min(font_sizes) if len(font_sizes) > 1 else 0,
                "unique_fonts": len(set(font_families))
            }
            
        except Exception as e:
            self.logger.warning("Error analyzing block fonts", error=str(e))
            return {}
    
    def _merge_nearby_blocks(self, text_blocks: List[TextBlock]) -> List[TextBlock]:
        """Merge text blocks that are close to each other."""
        try:
            if len(text_blocks) <= 1:
                return text_blocks
            
            # Sort blocks by position (top to bottom, left to right)
            sorted_blocks = sorted(text_blocks, key=lambda b: (b.bbox.y0, b.bbox.x0))
            
            merged_blocks = []
            current_block = sorted_blocks[0]
            
            for next_block in sorted_blocks[1:]:
                # Check if blocks should be merged
                if self._should_merge_blocks(current_block, next_block):
                    current_block = self._merge_two_blocks(current_block, next_block)
                else:
                    merged_blocks.append(current_block)
                    current_block = next_block
            
            merged_blocks.append(current_block)
            return merged_blocks
            
        except Exception as e:
            self.logger.warning("Error merging nearby blocks", error=str(e))
            return text_blocks
    
    def _should_merge_blocks(self, block1: TextBlock, block2: TextBlock) -> bool:
        """Determine if two text blocks should be merged."""
        try:
            # Don't merge if font properties are very different
            if (block1.font_size and block2.font_size and 
                abs(block1.font_size - block2.font_size) > 2):
                return False
            
            if block1.is_bold != block2.is_bold or block1.is_italic != block2.is_italic:
                return False
            
            # Check distance between blocks
            distance = block1.bbox.distance_to(block2.bbox)
            
            if distance <= self.config.merge_distance_threshold:
                # Check alignment (horizontal or vertical)
                
                # Horizontal alignment (same line)
                h_overlap = min(block1.bbox.y1, block2.bbox.y1) - max(block1.bbox.y0, block2.bbox.y0)
                if h_overlap > min(block1.bbox.height, block2.bbox.height) * 0.5:
                    return True
                
                # Vertical alignment (same column)
                v_overlap = min(block1.bbox.x1, block2.bbox.x1) - max(block1.bbox.x0, block2.bbox.x0)
                if v_overlap > min(block1.bbox.width, block2.bbox.width) * 0.5:
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _merge_two_blocks(self, block1: TextBlock, block2: TextBlock) -> TextBlock:
        """Merge two text blocks into one."""
        try:
            # Combine text with appropriate spacing
            combined_text = block1.text
            if not combined_text.endswith((" ", "\n")):
                # Determine if blocks are on same line or different lines
                y_distance = abs(block1.bbox.center_y - block2.bbox.center_y)
                if y_distance < min(block1.bbox.height, block2.bbox.height) * 0.5:
                    combined_text += " "  # Same line
                else:
                    combined_text += "\n"  # Different lines
            combined_text += block2.text
            
            # Calculate combined bounding box
            combined_bbox = BoundingBox(
                x0=min(block1.bbox.x0, block2.bbox.x0),
                y0=min(block1.bbox.y0, block2.bbox.y0),
                x1=max(block1.bbox.x1, block2.bbox.x1),
                y1=max(block1.bbox.y1, block2.bbox.y1)
            )
            
            # Use properties from the first block (arbitrary choice)
            merged_features = block1.classification_features.copy()
            merged_features.update({"merged_from": [
                block1.classification_features.get("block_index", 0),
                block2.classification_features.get("block_index", 0)
            ]})
            
            return TextBlock(
                text=combined_text,
                bbox=combined_bbox,
                page_num=block1.page_num,
                font_size=block1.font_size or block2.font_size,
                font_family=block1.font_family or block2.font_family,
                is_bold=block1.is_bold or block2.is_bold,
                is_italic=block1.is_italic or block2.is_italic,
                classification_features=merged_features
            )
            
        except Exception as e:
            self.logger.warning("Error merging two blocks", error=str(e))
            return block1  # Return first block if merging fails
    
    def _determine_reading_order(self, text_blocks: List[TextBlock], page_rect: fitz.Rect) -> List[TextBlock]:
        """Determine reading order for text blocks."""
        try:
            # Simple reading order: top to bottom, left to right
            # For more complex layouts, this would need to be more sophisticated
            
            # Sort by vertical position first, then horizontal
            sorted_blocks = sorted(text_blocks, key=lambda b: (b.bbox.y0, b.bbox.x0))
            
            # Assign reading order
            for i, block in enumerate(sorted_blocks):
                block.reading_order = i
            
            return sorted_blocks
            
        except Exception as e:
            self.logger.warning("Error determining reading order", error=str(e))
            return text_blocks
    
    def _detect_columns(self, text_blocks: List[TextBlock], page_rect: fitz.Rect) -> List[TextBlock]:
        """Detect column layout and assign column numbers."""
        try:
            if not text_blocks:
                return text_blocks
            
            # Simple column detection based on x-coordinates
            # Group blocks by approximate x-position
            x_positions = [block.bbox.center_x for block in text_blocks]
            
            # Use simple clustering to find columns
            sorted_x = sorted(set(x_positions))
            
            if len(sorted_x) <= 1:
                # Single column
                for block in text_blocks:
                    block.column = 1
            else:
                # Multiple columns - simple threshold-based clustering
                column_boundaries = []
                for i in range(len(sorted_x) - 1):
                    gap = sorted_x[i + 1] - sorted_x[i]
                    if gap > page_rect.width * 0.1:  # 10% of page width
                        column_boundaries.append((sorted_x[i] + sorted_x[i + 1]) / 2)
                
                # Assign column numbers
                for block in text_blocks:
                    column = 1
                    for boundary in column_boundaries:
                        if block.bbox.center_x > boundary:
                            column += 1
                    block.column = column
            
            return text_blocks
            
        except Exception as e:
            self.logger.warning("Error detecting columns", error=str(e))
            return text_blocks
    
    def _get_config_summary(self) -> Dict[str, Any]:
        """Get summary of analysis configuration."""
        return {
            "min_text_length": self.config.min_text_length,
            "merge_nearby_blocks": self.config.merge_nearby_blocks,
            "merge_distance_threshold": self.config.merge_distance_threshold,
            "enable_reading_order": self.config.enable_reading_order,
            "enable_column_detection": self.config.enable_column_detection,
            "classification_rules_count": len(self.config.classification_rules),
            "analyzer_version": "MVP-1.0"
        }
    
    def get_analysis_summary(self, result: LayoutResult) -> Dict[str, Any]:
        """Get human-readable analysis summary."""
        try:
            total_blocks = result.total_blocks
            blocks_by_type = result.blocks_by_type_total
            
            summary = {
                "document_info": {
                    "pdf_path": str(result.pdf_path),
                    "page_count": result.page_count,
                    "total_blocks": total_blocks,
                    "processing_time": result.total_processing_time
                },
                "content_analysis": {
                    "titles": blocks_by_type.get(BlockType.TITLE, 0),
                    "headings": blocks_by_type.get(BlockType.HEADING, 0),
                    "body_paragraphs": blocks_by_type.get(BlockType.BODY, 0),
                    "captions": blocks_by_type.get(BlockType.CAPTION, 0),
                    "bylines": blocks_by_type.get(BlockType.BYLINE, 0)
                },
                "layout_features": {
                    "has_headers": blocks_by_type.get(BlockType.HEADER, 0) > 0,
                    "has_footers": blocks_by_type.get(BlockType.FOOTER, 0) > 0,
                    "has_page_numbers": blocks_by_type.get(BlockType.PAGE_NUMBER, 0) > 0,
                    "avg_blocks_per_page": total_blocks / max(result.page_count, 1)
                },
                "quality_indicators": {
                    "unclassified_blocks": blocks_by_type.get(BlockType.UNKNOWN, 0),
                    "classification_rate": (total_blocks - blocks_by_type.get(BlockType.UNKNOWN, 0)) / max(total_blocks, 1),
                    "processing_speed": total_blocks / max(result.total_processing_time, 1)  # blocks per second
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error("Error creating analysis summary", error=str(e))
            return {"error": str(e)}