"""
PDF page splitter maintaining order and handling both born-digital and scanned PDFs.
"""

import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterator

import fitz  # PyMuPDF
import structlog

from .types import PDFProcessingError
from .validator import PDFValidator


logger = structlog.get_logger(__name__)


class PageSplitError(PDFProcessingError):
    """Exception raised when page splitting fails."""
    pass


class PageSplitter:
    """
    Splits PDF pages while maintaining order and handling edge cases.
    
    Supports both born-digital and scanned PDFs with robust error handling.
    """
    
    def __init__(self, output_dir: Optional[Path] = None, preserve_metadata: bool = True):
        """
        Initialize page splitter.
        
        Args:
            output_dir: Directory to save split pages (if None, uses temp directory)
            preserve_metadata: Whether to preserve original PDF metadata in split pages
        """
        self.output_dir = output_dir
        self.preserve_metadata = preserve_metadata
        self.logger = logger.bind(component="PageSplitter")
        self.validator = PDFValidator()
    
    def split_to_files(
        self, 
        pdf_path: Path, 
        output_pattern: Optional[str] = None,
        page_range: Optional[tuple] = None,
        validate_input: bool = True
    ) -> List[Path]:
        """
        Split PDF into individual page files.
        
        Args:
            pdf_path: Path to source PDF
            output_pattern: Pattern for output filenames (e.g., "page_{:03d}.pdf")
            page_range: Tuple of (start_page, end_page) 1-indexed, None for all pages
            validate_input: Whether to validate input PDF first
            
        Returns:
            List of paths to created page files
            
        Raises:
            PageSplitError: If splitting fails
        """
        start_time = time.time()
        
        try:
            self.logger.info("Starting PDF page splitting", 
                           pdf_path=str(pdf_path), page_range=page_range)
            
            # Validate input PDF if requested
            if validate_input:
                pdf_info = self.validator.validate(pdf_path, quick_check=True)
                if not pdf_info.is_valid:
                    raise PageSplitError(
                        f"Input PDF validation failed: {pdf_info.validation_errors}",
                        pdf_path
                    )
            
            # Set up output directory
            if self.output_dir is None:
                import tempfile
                output_dir = Path(tempfile.mkdtemp(prefix="pdf_split_"))
            else:
                output_dir = self.output_dir
                output_dir.mkdir(parents=True, exist_ok=True)
            
            # Open source PDF
            try:
                doc = fitz.open(str(pdf_path))
            except Exception as e:
                raise PageSplitError(f"Cannot open source PDF: {str(e)}", pdf_path)
            
            try:
                total_pages = doc.page_count
                
                # Determine page range
                if page_range:
                    start_page, end_page = page_range
                    start_page = max(1, start_page)
                    end_page = min(total_pages, end_page)
                else:
                    start_page, end_page = 1, total_pages
                
                if start_page > end_page:
                    raise PageSplitError(
                        f"Invalid page range: {start_page} > {end_page}",
                        pdf_path
                    )
                
                # Set up output filename pattern
                if output_pattern is None:
                    base_name = pdf_path.stem
                    output_pattern = f"{base_name}_page_{{:03d}}.pdf"
                
                created_files = []
                
                # Split pages
                for page_num in range(start_page - 1, end_page):  # Convert to 0-indexed
                    try:
                        page_file = self._split_single_page(
                            doc, page_num, output_dir, output_pattern, pdf_path
                        )
                        created_files.append(page_file)
                        
                    except Exception as e:
                        self.logger.error("Error splitting page", 
                                        page_num=page_num + 1, error=str(e))
                        # Continue with other pages instead of failing completely
                        continue
                
                processing_time = time.time() - start_time
                
                self.logger.info("PDF page splitting completed",
                               pdf_path=str(pdf_path),
                               pages_processed=len(created_files),
                               total_pages=end_page - start_page + 1,
                               processing_time=processing_time)
                
                return created_files
                
            finally:
                doc.close()
                
        except PageSplitError:
            raise
        except Exception as e:
            self.logger.error("Unexpected error during page splitting",
                            pdf_path=str(pdf_path), error=str(e), exc_info=True)
            raise PageSplitError(f"Unexpected splitting error: {str(e)}", pdf_path)
    
    def _split_single_page(
        self, 
        doc: fitz.Document, 
        page_num: int, 
        output_dir: Path, 
        output_pattern: str,
        source_path: Path
    ) -> Path:
        """Split a single page from the document."""
        try:
            # Create new single-page document
            new_doc = fitz.open()
            
            # Copy the page
            new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            
            # Preserve metadata if requested
            if self.preserve_metadata:
                original_metadata = doc.metadata
                if original_metadata:
                    # Update metadata for single page
                    new_metadata = original_metadata.copy()
                    new_metadata["title"] = f"{original_metadata.get('title', '')} - Page {page_num + 1}"
                    new_doc.set_metadata(new_metadata)
            
            # Generate output filename
            output_filename = output_pattern.format(page_num + 1)
            output_path = output_dir / output_filename
            
            # Save the page
            new_doc.save(str(output_path))
            new_doc.close()
            
            # Verify the created file
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise PageSplitError(
                    f"Failed to create page file: {output_path}",
                    source_path, page_num + 1
                )
            
            return output_path
            
        except Exception as e:
            raise PageSplitError(
                f"Error splitting page {page_num + 1}: {str(e)}",
                source_path, page_num + 1
            )
    
    def split_to_memory(
        self, 
        pdf_path: Path, 
        page_range: Optional[tuple] = None,
        validate_input: bool = True
    ) -> Iterator[tuple]:
        """
        Split PDF pages and yield them as in-memory documents.
        
        Args:
            pdf_path: Path to source PDF
            page_range: Tuple of (start_page, end_page) 1-indexed
            validate_input: Whether to validate input PDF first
            
        Yields:
            Tuples of (page_number, fitz.Document) for each page
            
        Raises:
            PageSplitError: If splitting fails
        """
        try:
            self.logger.info("Starting in-memory PDF page splitting", 
                           pdf_path=str(pdf_path), page_range=page_range)
            
            # Validate input PDF if requested
            if validate_input:
                pdf_info = self.validator.validate(pdf_path, quick_check=True)
                if not pdf_info.is_valid:
                    raise PageSplitError(
                        f"Input PDF validation failed: {pdf_info.validation_errors}",
                        pdf_path
                    )
            
            # Open source PDF
            try:
                doc = fitz.open(str(pdf_path))
            except Exception as e:
                raise PageSplitError(f"Cannot open source PDF: {str(e)}", pdf_path)
            
            try:
                total_pages = doc.page_count
                
                # Determine page range
                if page_range:
                    start_page, end_page = page_range
                    start_page = max(1, start_page)
                    end_page = min(total_pages, end_page)
                else:
                    start_page, end_page = 1, total_pages
                
                # Yield pages one by one
                for page_num in range(start_page - 1, end_page):  # Convert to 0-indexed
                    try:
                        # Create new single-page document
                        new_doc = fitz.open()
                        new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
                        
                        # Preserve metadata if requested
                        if self.preserve_metadata:
                            original_metadata = doc.metadata
                            if original_metadata:
                                new_metadata = original_metadata.copy()
                                new_metadata["title"] = f"{original_metadata.get('title', '')} - Page {page_num + 1}"
                                new_doc.set_metadata(new_metadata)
                        
                        yield (page_num + 1, new_doc)
                        
                    except Exception as e:
                        self.logger.error("Error processing page in memory",
                                        page_num=page_num + 1, error=str(e))
                        # Continue with other pages
                        continue
                        
            finally:
                doc.close()
                
        except PageSplitError:
            raise
        except Exception as e:
            self.logger.error("Unexpected error during in-memory page splitting",
                            pdf_path=str(pdf_path), error=str(e), exc_info=True)
            raise PageSplitError(f"Unexpected splitting error: {str(e)}", pdf_path)
    
    def extract_page_ranges(
        self, 
        pdf_path: Path, 
        ranges: List[tuple],
        output_paths: Optional[List[Path]] = None,
        validate_input: bool = True
    ) -> List[Path]:
        """
        Extract specific page ranges into separate PDF files.
        
        Args:
            pdf_path: Path to source PDF
            ranges: List of (start_page, end_page) tuples (1-indexed)
            output_paths: Optional list of output paths (auto-generated if None)
            validate_input: Whether to validate input PDF first
            
        Returns:
            List of paths to created range files
            
        Raises:
            PageSplitError: If extraction fails
        """
        try:
            self.logger.info("Starting PDF page range extraction",
                           pdf_path=str(pdf_path), ranges=ranges)
            
            if not ranges:
                raise PageSplitError("No page ranges specified", pdf_path)
            
            # Validate input PDF if requested
            if validate_input:
                pdf_info = self.validator.validate(pdf_path, quick_check=True)
                if not pdf_info.is_valid:
                    raise PageSplitError(
                        f"Input PDF validation failed: {pdf_info.validation_errors}",
                        pdf_path
                    )
            
            # Set up output directory
            if self.output_dir is None:
                import tempfile
                output_dir = Path(tempfile.mkdtemp(prefix="pdf_ranges_"))
            else:
                output_dir = self.output_dir
                output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate output paths if not provided
            if output_paths is None:
                base_name = pdf_path.stem
                output_paths = []
                for i, (start, end) in enumerate(ranges):
                    if start == end:
                        filename = f"{base_name}_page_{start:03d}.pdf"
                    else:
                        filename = f"{base_name}_pages_{start:03d}-{end:03d}.pdf"
                    output_paths.append(output_dir / filename)
            
            if len(output_paths) != len(ranges):
                raise PageSplitError(
                    f"Number of output paths ({len(output_paths)}) must match number of ranges ({len(ranges)})",
                    pdf_path
                )
            
            # Open source PDF
            try:
                doc = fitz.open(str(pdf_path))
            except Exception as e:
                raise PageSplitError(f"Cannot open source PDF: {str(e)}", pdf_path)
            
            try:
                total_pages = doc.page_count
                created_files = []
                
                for (start_page, end_page), output_path in zip(ranges, output_paths):
                    try:
                        # Validate range
                        start_page = max(1, start_page)
                        end_page = min(total_pages, end_page)
                        
                        if start_page > end_page:
                            self.logger.warning("Invalid page range, skipping",
                                              range=(start_page, end_page))
                            continue
                        
                        # Create new document for this range
                        new_doc = fitz.open()
                        new_doc.insert_pdf(doc, 
                                         from_page=start_page - 1,  # Convert to 0-indexed
                                         to_page=end_page - 1)
                        
                        # Preserve metadata if requested
                        if self.preserve_metadata:
                            original_metadata = doc.metadata
                            if original_metadata:
                                new_metadata = original_metadata.copy()
                                if start_page == end_page:
                                    new_metadata["title"] = f"{original_metadata.get('title', '')} - Page {start_page}"
                                else:
                                    new_metadata["title"] = f"{original_metadata.get('title', '')} - Pages {start_page}-{end_page}"
                                new_doc.set_metadata(new_metadata)
                        
                        # Save the range
                        new_doc.save(str(output_path))
                        new_doc.close()
                        
                        # Verify the created file
                        if not output_path.exists() or output_path.stat().st_size == 0:
                            self.logger.error("Failed to create range file", 
                                            output_path=str(output_path))
                            continue
                        
                        created_files.append(output_path)
                        
                    except Exception as e:
                        self.logger.error("Error extracting page range",
                                        range=(start_page, end_page), error=str(e))
                        continue
                
                self.logger.info("PDF page range extraction completed",
                               pdf_path=str(pdf_path),
                               ranges_processed=len(created_files),
                               total_ranges=len(ranges))
                
                return created_files
                
            finally:
                doc.close()
                
        except PageSplitError:
            raise
        except Exception as e:
            self.logger.error("Unexpected error during page range extraction",
                            pdf_path=str(pdf_path), error=str(e), exc_info=True)
            raise PageSplitError(f"Unexpected range extraction error: {str(e)}", pdf_path)
    
    def get_page_info(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Get basic information about each page without extracting content.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of dictionaries with page information
            
        Raises:
            PageSplitError: If cannot read PDF
        """
        try:
            doc = fitz.open(str(pdf_path))
            
            try:
                page_info = []
                
                for page_num in range(doc.page_count):
                    try:
                        page = doc[page_num]
                        rect = page.rect
                        
                        info = {
                            "page_number": page_num + 1,
                            "width": rect.width,
                            "height": rect.height,
                            "rotation": page.rotation,
                            "has_text": len(page.get_text().strip()) > 0,
                            "image_count": len(page.get_images()),
                            "annotation_count": len(page.annots()),
                        }
                        page_info.append(info)
                        
                    except Exception as e:
                        self.logger.warning("Error getting info for page",
                                          page_num=page_num + 1, error=str(e))
                        # Add minimal info for corrupted pages
                        page_info.append({
                            "page_number": page_num + 1,
                            "width": 0,
                            "height": 0,
                            "rotation": 0,
                            "has_text": False,
                            "image_count": 0,
                            "annotation_count": 0,
                            "error": str(e)
                        })
                
                return page_info
                
            finally:
                doc.close()
                
        except Exception as e:
            raise PageSplitError(f"Cannot read PDF page information: {str(e)}", pdf_path)