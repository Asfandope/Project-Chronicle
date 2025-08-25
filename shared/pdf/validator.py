"""
PDF validator for checking corruption and ensuring processability.
"""

import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

import fitz  # PyMuPDF
import structlog

from .types import PDFInfo, PDFMetadata, PDFType, PDFProcessingError


logger = structlog.get_logger(__name__)


class PDFValidationError(PDFProcessingError):
    """Exception raised when PDF validation fails."""
    pass


class PDFValidator:
    """
    Validates PDF files for corruption and processability.
    
    Handles both born-digital and scanned PDFs with comprehensive checks.
    """
    
    def __init__(self, max_file_size_mb: int = 500, min_pages: int = 1, max_pages: int = 1000):
        """
        Initialize PDF validator.
        
        Args:
            max_file_size_mb: Maximum allowed file size in MB
            min_pages: Minimum number of pages required
            max_pages: Maximum number of pages allowed
        """
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.min_pages = min_pages
        self.max_pages = max_pages
        self.logger = logger.bind(component="PDFValidator")
    
    def validate(self, pdf_path: Path, quick_check: bool = False) -> PDFInfo:
        """
        Validate a PDF file and return comprehensive information.
        
        Args:
            pdf_path: Path to the PDF file
            quick_check: If True, perform only basic validation checks
            
        Returns:
            PDFInfo object with validation results and metadata
            
        Raises:
            PDFValidationError: If PDF is invalid or cannot be processed
        """
        start_time = time.time()
        validation_errors = []
        
        try:
            self.logger.info("Starting PDF validation", pdf_path=str(pdf_path))
            
            # Basic file checks
            if not pdf_path.exists():
                raise PDFValidationError(f"PDF file does not exist: {pdf_path}", pdf_path)
            
            if not pdf_path.is_file():
                raise PDFValidationError(f"Path is not a file: {pdf_path}", pdf_path)
            
            file_size = pdf_path.stat().st_size
            if file_size == 0:
                raise PDFValidationError(f"PDF file is empty: {pdf_path}", pdf_path)
            
            if file_size > self.max_file_size_bytes:
                raise PDFValidationError(
                    f"PDF file too large: {file_size / 1024 / 1024:.1f}MB > {self.max_file_size_bytes / 1024 / 1024}MB",
                    pdf_path
                )
            
            # Try to open PDF with PyMuPDF
            try:
                doc = fitz.open(str(pdf_path))
            except Exception as e:
                raise PDFValidationError(f"Cannot open PDF file: {str(e)}", pdf_path)
            
            try:
                # Basic document checks
                if doc.is_closed:
                    raise PDFValidationError("PDF document is closed", pdf_path)
                
                page_count = doc.page_count
                if page_count < self.min_pages:
                    validation_errors.append(f"Too few pages: {page_count} < {self.min_pages}")
                
                if page_count > self.max_pages:
                    validation_errors.append(f"Too many pages: {page_count} > {self.max_pages}")
                
                # Check if document is encrypted and we can't access it
                if doc.needs_pass:
                    validation_errors.append("PDF is password protected and cannot be processed")
                
                # Extract metadata
                metadata = self._extract_metadata(doc, file_size)
                
                # Determine PDF type
                pdf_type = self._determine_pdf_type(doc) if not quick_check else PDFType.UNKNOWN
                
                # Detailed validation checks
                if not quick_check:
                    validation_errors.extend(self._perform_detailed_validation(doc))
                
                # Create minimal page info for validation
                pages = []
                if not validation_errors:  # Only process pages if no critical errors
                    pages = self._create_basic_page_info(doc, quick_check)
                
                processing_time = time.time() - start_time
                
                pdf_info = PDFInfo(
                    file_path=pdf_path,
                    metadata=metadata,
                    pdf_type=pdf_type,
                    pages=pages,
                    is_valid=len(validation_errors) == 0,
                    validation_errors=validation_errors,
                    processing_time=processing_time
                )
                
                self.logger.info(
                    "PDF validation completed",
                    pdf_path=str(pdf_path),
                    is_valid=pdf_info.is_valid,
                    page_count=page_count,
                    pdf_type=pdf_type.value,
                    processing_time=processing_time,
                    error_count=len(validation_errors)
                )
                
                return pdf_info
                
            finally:
                doc.close()
                
        except PDFValidationError:
            raise
        except Exception as e:
            self.logger.error("Unexpected error during PDF validation", 
                            pdf_path=str(pdf_path), error=str(e), exc_info=True)
            raise PDFValidationError(f"Unexpected validation error: {str(e)}", pdf_path)
    
    def _extract_metadata(self, doc: fitz.Document, file_size: int) -> PDFMetadata:
        """Extract metadata from PDF document."""
        try:
            meta = doc.metadata
            
            return PDFMetadata(
                title=meta.get("title") or None,
                author=meta.get("author") or None,
                subject=meta.get("subject") or None,
                creator=meta.get("creator") or None,
                producer=meta.get("producer") or None,
                creation_date=self._parse_pdf_date(meta.get("creationDate")),
                modification_date=self._parse_pdf_date(meta.get("modDate")),
                keywords=self._parse_keywords(meta.get("keywords")),
                page_count=doc.page_count,
                file_size=file_size,
                pdf_version=f"1.{doc.pdf_version()[1]}" if doc.pdf_version() else None,
                is_encrypted=doc.needs_pass,
                is_linearized=doc.is_fast_web_view,
                permissions=self._extract_permissions(doc)
            )
        except Exception as e:
            self.logger.warning("Error extracting PDF metadata", error=str(e))
            return PDFMetadata(
                page_count=doc.page_count,
                file_size=file_size,
                is_encrypted=doc.needs_pass
            )
    
    def _parse_pdf_date(self, date_str: Optional[str]) -> Optional[Any]:
        """Parse PDF date string to datetime object."""
        if not date_str:
            return None
        
        try:
            # PDF dates are in format: D:YYYYMMDDHHmmSSOHH'mm'
            from datetime import datetime
            if date_str.startswith("D:"):
                date_str = date_str[2:]
            
            # Extract basic date part (YYYYMMDDHHMMSS)
            if len(date_str) >= 14:
                return datetime.strptime(date_str[:14], "%Y%m%d%H%M%S")
            elif len(date_str) >= 8:
                return datetime.strptime(date_str[:8], "%Y%m%d")
        except Exception:
            pass
        
        return None
    
    def _parse_keywords(self, keywords_str: Optional[str]) -> List[str]:
        """Parse keywords string into list."""
        if not keywords_str:
            return []
        
        # Common separators in PDF keywords
        for sep in [";", ",", "\n", "\r"]:
            if sep in keywords_str:
                return [kw.strip() for kw in keywords_str.split(sep) if kw.strip()]
        
        return [keywords_str.strip()] if keywords_str.strip() else []
    
    def _extract_permissions(self, doc: fitz.Document) -> Dict[str, bool]:
        """Extract document permissions."""
        try:
            perms = doc.permissions
            return {
                "print": bool(perms & fitz.PDF_PERM_PRINT),
                "modify": bool(perms & fitz.PDF_PERM_MODIFY),
                "copy": bool(perms & fitz.PDF_PERM_COPY),
                "annotate": bool(perms & fitz.PDF_PERM_ANNOTATE),
            }
        except Exception:
            return {}
    
    def _determine_pdf_type(self, doc: fitz.Document) -> PDFType:
        """
        Determine if PDF is born-digital, scanned, or hybrid.
        
        This is a heuristic-based approach that analyzes the first few pages.
        """
        try:
            pages_to_check = min(5, doc.page_count)  # Check first 5 pages
            text_pages = 0
            image_heavy_pages = 0
            
            for page_num in range(pages_to_check):
                page = doc[page_num]
                
                # Check for extractable text
                text = page.get_text().strip()
                has_meaningful_text = len(text) > 50  # At least 50 chars of text
                
                # Check for images
                image_list = page.get_images()
                has_large_images = len(image_list) > 0
                
                if has_meaningful_text:
                    text_pages += 1
                
                if has_large_images and not has_meaningful_text:
                    image_heavy_pages += 1
            
            if text_pages == pages_to_check:
                return PDFType.BORN_DIGITAL
            elif image_heavy_pages == pages_to_check:
                return PDFType.SCANNED
            elif text_pages > 0 and image_heavy_pages > 0:
                return PDFType.HYBRID
            else:
                return PDFType.UNKNOWN
                
        except Exception as e:
            self.logger.warning("Error determining PDF type", error=str(e))
            return PDFType.UNKNOWN
    
    def _perform_detailed_validation(self, doc: fitz.Document) -> List[str]:
        """Perform detailed validation checks on the PDF."""
        errors = []
        
        try:
            # Check each page for corruption
            corrupted_pages = []
            for page_num in range(doc.page_count):
                try:
                    page = doc[page_num]
                    
                    # Try to get page dimensions
                    rect = page.rect
                    if rect.width <= 0 or rect.height <= 0:
                        corrupted_pages.append(page_num + 1)
                        continue
                    
                    # Try to render page (this will catch many corruption issues)
                    try:
                        # Render at low resolution to check for corruption
                        pix = page.get_pixmap(matrix=fitz.Matrix(0.1, 0.1))
                        pix = None  # Clean up
                    except Exception:
                        corrupted_pages.append(page_num + 1)
                        continue
                        
                except Exception:
                    corrupted_pages.append(page_num + 1)
            
            if corrupted_pages:
                if len(corrupted_pages) > doc.page_count * 0.5:  # More than 50% corrupted
                    errors.append(f"More than half of pages are corrupted: {len(corrupted_pages)}/{doc.page_count}")
                else:
                    errors.append(f"Corrupted pages detected: {corrupted_pages[:10]}{'...' if len(corrupted_pages) > 10 else ''}")
            
            # Check for extremely large pages (potential memory issues)
            large_pages = []
            for page_num in range(min(10, doc.page_count)):  # Check first 10 pages
                try:
                    page = doc[page_num]
                    rect = page.rect
                    # Pages larger than 50 inches might cause issues
                    if rect.width > 3600 or rect.height > 3600:  # 50 inches at 72 DPI
                        large_pages.append(page_num + 1)
                except Exception:
                    pass
            
            if large_pages:
                errors.append(f"Extremely large pages detected: {large_pages}")
            
        except Exception as e:
            errors.append(f"Error during detailed validation: {str(e)}")
        
        return errors
    
    def _create_basic_page_info(self, doc: fitz.Document, quick_check: bool) -> List[Any]:
        """Create basic page information for validation purposes."""
        from .types import PageInfo  # Import here to avoid circular imports
        
        pages = []
        
        # For validation, we only need basic page info
        for page_num in range(doc.page_count):
            try:
                page = doc[page_num]
                rect = page.rect
                
                # Basic page info without extracting text/images (for performance)
                page_info = PageInfo(
                    page_num=page_num + 1,
                    width=rect.width,
                    height=rect.height,
                    rotation=page.rotation,
                    text_blocks=[],  # Will be populated by text extractor
                    images=[],       # Will be populated by image extractor
                    has_text=len(page.get_text().strip()) > 10 if not quick_check else True,
                    is_scanned=False  # Will be determined by other extractors
                )
                pages.append(page_info)
                
            except Exception as e:
                self.logger.warning("Error processing page for validation", 
                                  page_num=page_num + 1, error=str(e))
                # Skip corrupted pages
                continue
        
        return pages
    
    def quick_validate(self, pdf_path: Path) -> bool:
        """
        Perform quick validation check.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            True if PDF passes basic validation
        """
        try:
            pdf_info = self.validate(pdf_path, quick_check=True)
            return pdf_info.is_valid
        except PDFValidationError:
            return False
    
    def batch_validate(self, pdf_paths: List[Path], quick_check: bool = True) -> Dict[Path, bool]:
        """
        Validate multiple PDF files.
        
        Args:
            pdf_paths: List of PDF file paths
            quick_check: If True, perform only basic validation
            
        Returns:
            Dictionary mapping file paths to validation results
        """
        results = {}
        
        for pdf_path in pdf_paths:
            try:
                if quick_check:
                    results[pdf_path] = self.quick_validate(pdf_path)
                else:
                    pdf_info = self.validate(pdf_path, quick_check=False)
                    results[pdf_path] = pdf_info.is_valid
            except Exception as e:
                self.logger.error("Error validating PDF in batch", 
                                pdf_path=str(pdf_path), error=str(e))
                results[pdf_path] = False
        
        return results