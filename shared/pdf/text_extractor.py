"""
Text block extractor with bounding boxes for born-digital and scanned PDFs.
"""

import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import fitz  # PyMuPDF
import structlog

from .types import PDFProcessingError, TextBlock, BoundingBox, PageInfo


logger = structlog.get_logger(__name__)


class TextExtractionError(PDFProcessingError):
    """Exception raised when text extraction fails."""
    pass


class TextBlockExtractor:
    """
    Extracts text blocks with bounding boxes from PDF pages.
    
    Handles both born-digital and scanned PDFs with OCR fallback.
    """
    
    def __init__(
        self, 
        min_confidence: float = 0.7,
        min_text_length: int = 3,
        merge_nearby_blocks: bool = True,
        merge_distance_threshold: float = 10.0
    ):
        """
        Initialize text block extractor.
        
        Args:
            min_confidence: Minimum confidence threshold for text blocks
            min_text_length: Minimum text length to consider a valid block
            merge_nearby_blocks: Whether to merge nearby text blocks
            merge_distance_threshold: Distance threshold for merging blocks (points)
        """
        self.min_confidence = min_confidence
        self.min_text_length = min_text_length
        self.merge_nearby_blocks = merge_nearby_blocks
        self.merge_distance_threshold = merge_distance_threshold
        self.logger = logger.bind(component="TextBlockExtractor")
    
    def extract_from_page(
        self, 
        pdf_path: Path, 
        page_num: int,
        use_ocr_fallback: bool = True
    ) -> List[TextBlock]:
        """
        Extract text blocks from a specific page.
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number (1-indexed)
            use_ocr_fallback: Whether to use OCR if no text is found
            
        Returns:
            List of TextBlock objects
            
        Raises:
            TextExtractionError: If extraction fails
        """
        try:
            self.logger.info("Extracting text blocks from page",
                           pdf_path=str(pdf_path), page_num=page_num)
            
            doc = fitz.open(str(pdf_path))
            
            try:
                if page_num < 1 or page_num > doc.page_count:
                    raise TextExtractionError(
                        f"Invalid page number: {page_num} (PDF has {doc.page_count} pages)",
                        pdf_path, page_num
                    )
                
                page = doc[page_num - 1]  # Convert to 0-indexed
                
                # Try to extract text using native PDF text
                text_blocks = self._extract_native_text(page, page_num)
                
                # If no meaningful text found and OCR fallback is enabled
                if (not text_blocks or self._has_insufficient_text(text_blocks)) and use_ocr_fallback:
                    self.logger.info("No native text found, attempting OCR fallback",
                                   page_num=page_num)
                    ocr_blocks = self._extract_with_ocr(page, page_num)
                    if ocr_blocks:
                        text_blocks = ocr_blocks
                
                # Post-process blocks
                if self.merge_nearby_blocks and text_blocks:
                    text_blocks = self._merge_nearby_blocks(text_blocks)
                
                # Filter by confidence and length
                text_blocks = self._filter_blocks(text_blocks)
                
                self.logger.info("Text extraction completed",
                               page_num=page_num, blocks_found=len(text_blocks))
                
                return text_blocks
                
            finally:
                doc.close()
                
        except TextExtractionError:
            raise
        except Exception as e:
            self.logger.error("Unexpected error during text extraction",
                            pdf_path=str(pdf_path), page_num=page_num, 
                            error=str(e), exc_info=True)
            raise TextExtractionError(
                f"Unexpected text extraction error: {str(e)}",
                pdf_path, page_num
            )
    
    def extract_from_pdf(
        self, 
        pdf_path: Path,
        page_range: Optional[Tuple[int, int]] = None,
        use_ocr_fallback: bool = True
    ) -> Dict[int, List[TextBlock]]:
        """
        Extract text blocks from all pages or a range of pages.
        
        Args:
            pdf_path: Path to PDF file
            page_range: Optional tuple of (start_page, end_page) 1-indexed
            use_ocr_fallback: Whether to use OCR if no text is found
            
        Returns:
            Dictionary mapping page numbers to lists of TextBlock objects
            
        Raises:
            TextExtractionError: If extraction fails
        """
        start_time = time.time()
        
        try:
            self.logger.info("Starting text extraction from PDF",
                           pdf_path=str(pdf_path), page_range=page_range)
            
            doc = fitz.open(str(pdf_path))
            
            try:
                total_pages = doc.page_count
                
                # Determine page range
                if page_range:
                    start_page, end_page = page_range
                    start_page = max(1, start_page)
                    end_page = min(total_pages, end_page)
                else:
                    start_page, end_page = 1, total_pages
                
                results = {}
                
                for page_num in range(start_page, end_page + 1):
                    try:
                        text_blocks = self.extract_from_page(
                            pdf_path, page_num, use_ocr_fallback
                        )
                        results[page_num] = text_blocks
                        
                    except Exception as e:
                        self.logger.error("Error extracting text from page",
                                        page_num=page_num, error=str(e))
                        results[page_num] = []  # Continue with other pages
                
                processing_time = time.time() - start_time
                total_blocks = sum(len(blocks) for blocks in results.values())
                
                self.logger.info("PDF text extraction completed",
                               pdf_path=str(pdf_path),
                               pages_processed=len(results),
                               total_blocks=total_blocks,
                               processing_time=processing_time)
                
                return results
                
            finally:
                doc.close()
                
        except Exception as e:
            self.logger.error("Unexpected error during PDF text extraction",
                            pdf_path=str(pdf_path), error=str(e), exc_info=True)
            raise TextExtractionError(
                f"Unexpected PDF text extraction error: {str(e)}",
                pdf_path
            )
    
    def _extract_native_text(self, page: fitz.Page, page_num: int) -> List[TextBlock]:
        """Extract text using native PDF text extraction."""
        try:
            text_blocks = []
            
            # Get text blocks with detailed information
            blocks = page.get_text("dict")
            
            for block_idx, block in enumerate(blocks.get("blocks", [])):
                if "lines" not in block:
                    continue  # Skip image blocks
                
                # Process text lines in this block
                block_text = ""
                block_bbox = None
                font_info = {"size": None, "family": None, "bold": False, "italic": False}
                
                for line in block["lines"]:
                    line_text = ""
                    for span in line.get("spans", []):
                        span_text = span.get("text", "").strip()
                        if span_text:
                            line_text += span_text + " "
                            
                            # Extract font information from first span
                            if font_info["size"] is None:
                                font_info["size"] = span.get("size")
                                font_info["family"] = span.get("font", "").split("+")[-1]  # Remove subset prefix
                                flags = span.get("flags", 0)
                                font_info["bold"] = bool(flags & 2**4)  # Bold flag
                                font_info["italic"] = bool(flags & 2**1)  # Italic flag
                    
                    if line_text.strip():
                        block_text += line_text.strip() + "\\n"
                
                # Get block bounding box
                if "bbox" in block:
                    x0, y0, x1, y1 = block["bbox"]
                    block_bbox = BoundingBox(x0, y0, x1, y1)
                
                # Create text block if we have meaningful content
                if block_text.strip() and block_bbox and len(block_text.strip()) >= self.min_text_length:
                    text_block = TextBlock(
                        text=block_text.strip(),
                        bbox=block_bbox,
                        confidence=1.0,  # Native text has full confidence
                        font_size=font_info["size"],
                        font_family=font_info["family"],
                        is_bold=font_info["bold"],
                        is_italic=font_info["italic"],
                        text_type=self._classify_text_type(block_text.strip(), font_info),
                        page_num=page_num
                    )
                    text_blocks.append(text_block)
            
            return text_blocks
            
        except Exception as e:
            self.logger.warning("Error in native text extraction",
                              page_num=page_num, error=str(e))
            return []
    
    def _extract_with_ocr(self, page: fitz.Page, page_num: int) -> List[TextBlock]:
        """
        Extract text using OCR as fallback.
        
        Note: This is a placeholder for OCR functionality.
        In a real implementation, you would integrate with Tesseract or similar.
        """
        try:
            self.logger.info("OCR text extraction not implemented - returning empty blocks",
                           page_num=page_num)
            
            # Placeholder: In real implementation, you would:
            # 1. Render page to image
            # 2. Run OCR (Tesseract/PaddleOCR/etc.)
            # 3. Extract text blocks with bounding boxes
            # 4. Convert coordinates back to PDF space
            
            # For now, return empty list
            return []
            
            # Example OCR integration (commented out):
            # import pytesseract
            # from PIL import Image
            # 
            # # Render page to image
            # pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Higher resolution
            # img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            # 
            # # Run OCR with bounding box data
            # data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            # 
            # # Process OCR results into TextBlock objects
            # text_blocks = []
            # for i in range(len(data['text'])):
            #     if int(data['conf'][i]) > 0:  # Filter by confidence
            #         x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            #         # Convert from image coordinates to PDF coordinates
            #         bbox = BoundingBox(x/2, y/2, (x+w)/2, (y+h)/2)  # Adjust for 2x scaling
            #         
            #         text_blocks.append(TextBlock(
            #             text=data['text'][i],
            #             bbox=bbox,
            #             confidence=data['conf'][i] / 100.0,
            #             page_num=page_num
            #         ))
            # 
            # return text_blocks
            
        except Exception as e:
            self.logger.warning("Error in OCR text extraction",
                              page_num=page_num, error=str(e))
            return []
    
    def _classify_text_type(self, text: str, font_info: Dict[str, Any]) -> Optional[str]:
        """Classify text type based on content and formatting."""
        try:
            text_lower = text.lower().strip()
            
            # Simple heuristics for text classification
            if font_info.get("bold") and len(text) < 100:
                if len(text.split()) <= 10:  # Short bold text likely a title
                    return "title"
                else:
                    return "heading"
            
            if font_info.get("size", 0) > 14:  # Large text
                return "heading"
            
            if len(text) < 50:  # Short text
                return "caption"
            
            # Check for common magazine elements
            if any(keyword in text_lower for keyword in ["by ", "photo by", "courtesy of"]):
                return "attribution"
            
            if text.endswith((".", "!", "?")) and len(text) > 50:
                return "paragraph"
            
            return "text"
            
        except Exception:
            return "text"
    
    def _has_insufficient_text(self, text_blocks: List[TextBlock]) -> bool:
        """Check if extracted text blocks are insufficient."""
        if not text_blocks:
            return True
        
        total_chars = sum(len(block.text) for block in text_blocks)
        return total_chars < 50  # Less than 50 characters indicates likely scanned page
    
    def _merge_nearby_blocks(self, text_blocks: List[TextBlock]) -> List[TextBlock]:
        """Merge nearby text blocks that likely belong together."""
        if len(text_blocks) <= 1:
            return text_blocks
        
        try:
            # Sort blocks by vertical position, then horizontal
            sorted_blocks = sorted(
                text_blocks,
                key=lambda b: (b.bbox.y0, b.bbox.x0)
            )
            
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
            self.logger.warning("Error merging text blocks", error=str(e))
            return text_blocks  # Return original blocks if merging fails
    
    def _should_merge_blocks(self, block1: TextBlock, block2: TextBlock) -> bool:
        """Determine if two text blocks should be merged."""
        try:
            # Don't merge blocks with different formatting if both have formatting info
            if (block1.font_size and block2.font_size and 
                abs(block1.font_size - block2.font_size) > 2):
                return False
            
            if block1.is_bold != block2.is_bold or block1.is_italic != block2.is_italic:
                return False
            
            # Check distance between blocks
            vertical_distance = abs(block1.bbox.y0 - block2.bbox.y1)
            horizontal_distance = abs(block1.bbox.x1 - block2.bbox.x0)
            
            # Merge if blocks are vertically close (same column/paragraph)
            if vertical_distance < self.merge_distance_threshold:
                # Check if blocks are horizontally aligned
                overlap_threshold = 0.5
                horizontal_overlap = min(block1.bbox.x1, block2.bbox.x1) - max(block1.bbox.x0, block2.bbox.x0)
                min_width = min(block1.bbox.width, block2.bbox.width)
                
                if horizontal_overlap > min_width * overlap_threshold:
                    return True
            
            # Merge if blocks are horizontally close (same line)
            if horizontal_distance < self.merge_distance_threshold:
                # Check if blocks are vertically aligned
                vertical_overlap = min(block1.bbox.y1, block2.bbox.y1) - max(block1.bbox.y0, block2.bbox.y0)
                min_height = min(block1.bbox.height, block2.bbox.height)
                
                if vertical_overlap > min_height * 0.5:
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _merge_two_blocks(self, block1: TextBlock, block2: TextBlock) -> TextBlock:
        """Merge two text blocks into one."""
        try:
            # Combine text with appropriate spacing
            combined_text = block1.text
            if not combined_text.endswith((" ", "\\n", "-")):
                combined_text += " "
            combined_text += block2.text
            
            # Calculate combined bounding box
            combined_bbox = BoundingBox(
                x0=min(block1.bbox.x0, block2.bbox.x0),
                y0=min(block1.bbox.y0, block2.bbox.y0),
                x1=max(block1.bbox.x1, block2.bbox.x1),
                y1=max(block1.bbox.y1, block2.bbox.y1)
            )
            
            # Use properties from the first block, average confidence
            return TextBlock(
                text=combined_text,
                bbox=combined_bbox,
                confidence=(block1.confidence + block2.confidence) / 2,
                font_size=block1.font_size or block2.font_size,
                font_family=block1.font_family or block2.font_family,
                is_bold=block1.is_bold or block2.is_bold,
                is_italic=block1.is_italic or block2.is_italic,
                text_type=block1.text_type or block2.text_type,
                page_num=block1.page_num
            )
            
        except Exception as e:
            self.logger.warning("Error merging two blocks", error=str(e))
            return block1  # Return first block if merging fails
    
    def _filter_blocks(self, text_blocks: List[TextBlock]) -> List[TextBlock]:
        """Filter text blocks by confidence and length."""
        filtered_blocks = []
        
        for block in text_blocks:
            # Filter by confidence
            if block.confidence < self.min_confidence:
                continue
            
            # Filter by text length
            if len(block.text.strip()) < self.min_text_length:
                continue
            
            # Filter out blocks that are just whitespace or special characters
            if not any(c.isalnum() for c in block.text):
                continue
            
            filtered_blocks.append(block)
        
        return filtered_blocks
    
    def get_page_text_summary(self, text_blocks: List[TextBlock]) -> Dict[str, Any]:
        """Get summary statistics for text blocks on a page."""
        if not text_blocks:
            return {
                "total_blocks": 0,
                "total_characters": 0,
                "total_words": 0,
                "avg_confidence": 0.0,
                "text_types": {},
                "font_sizes": [],
                "has_bold": False,
                "has_italic": False
            }
        
        total_chars = sum(len(block.text) for block in text_blocks)
        total_words = sum(block.word_count for block in text_blocks)
        avg_confidence = sum(block.confidence for block in text_blocks) / len(text_blocks)
        
        text_types = {}
        font_sizes = []
        has_bold = False
        has_italic = False
        
        for block in text_blocks:
            # Count text types
            text_type = block.text_type or "unknown"
            text_types[text_type] = text_types.get(text_type, 0) + 1
            
            # Collect font sizes
            if block.font_size:
                font_sizes.append(block.font_size)
            
            # Check for formatting
            if block.is_bold:
                has_bold = True
            if block.is_italic:
                has_italic = True
        
        return {
            "total_blocks": len(text_blocks),
            "total_characters": total_chars,
            "total_words": total_words,
            "avg_confidence": avg_confidence,
            "text_types": text_types,
            "font_sizes": font_sizes,
            "has_bold": has_bold,
            "has_italic": has_italic
        }