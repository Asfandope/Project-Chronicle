"""
Metadata extractor for comprehensive PDF information.
Extracts page count, creation date, and other metadata for both born-digital and scanned PDFs.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
import structlog

from .types import PDFInfo, PDFMetadata, PDFProcessingError, PDFType

logger = structlog.get_logger(__name__)


class MetadataExtractionError(PDFProcessingError):
    """Exception raised when metadata extraction fails."""


class MetadataExtractor:
    """
    Extracts comprehensive metadata from PDF documents.

    Handles both born-digital and scanned PDFs with robust error handling.
    """

    def __init__(self, extract_xmp: bool = True, analyze_structure: bool = True):
        """
        Initialize metadata extractor.

        Args:
            extract_xmp: Whether to extract XMP metadata if available
            analyze_structure: Whether to analyze document structure
        """
        self.extract_xmp = extract_xmp
        self.analyze_structure = analyze_structure
        self.logger = logger.bind(component="MetadataExtractor")

    def extract(self, pdf_path: Path) -> PDFMetadata:
        """
        Extract comprehensive metadata from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            PDFMetadata object with all available metadata

        Raises:
            MetadataExtractionError: If extraction fails
        """
        try:
            self.logger.info("Extracting PDF metadata", pdf_path=str(pdf_path))

            if not pdf_path.exists():
                raise MetadataExtractionError(
                    f"PDF file does not exist: {pdf_path}", pdf_path
                )

            file_size = pdf_path.stat().st_size

            # Open PDF document
            try:
                doc = fitz.open(str(pdf_path))
            except Exception as e:
                raise MetadataExtractionError(f"Cannot open PDF: {str(e)}", pdf_path)

            try:
                # Extract basic metadata
                metadata = self._extract_basic_metadata(doc, file_size)

                # Extract XMP metadata if requested and available
                if self.extract_xmp:
                    xmp_data = self._extract_xmp_metadata(doc)
                    metadata = self._merge_xmp_metadata(metadata, xmp_data)

                # Analyze document structure if requested
                if self.analyze_structure:
                    structure_info = self._analyze_document_structure(doc)
                    metadata = self._merge_structure_info(metadata, structure_info)

                self.logger.info(
                    "Metadata extraction completed",
                    pdf_path=str(pdf_path),
                    page_count=metadata.page_count,
                    file_size=metadata.file_size,
                )

                return metadata

            finally:
                doc.close()

        except MetadataExtractionError:
            raise
        except Exception as e:
            self.logger.error(
                "Unexpected error during metadata extraction",
                pdf_path=str(pdf_path),
                error=str(e),
                exc_info=True,
            )
            raise MetadataExtractionError(
                f"Unexpected metadata extraction error: {str(e)}", pdf_path
            )

    def extract_enhanced(self, pdf_path: Path) -> PDFInfo:
        """
        Extract enhanced metadata including document analysis.

        Args:
            pdf_path: Path to PDF file

        Returns:
            PDFInfo object with comprehensive information

        Raises:
            MetadataExtractionError: If extraction fails
        """
        try:
            # Extract basic metadata
            metadata = self.extract(pdf_path)

            # Open document for additional analysis
            doc = fitz.open(str(pdf_path))

            try:
                # Determine PDF type
                pdf_type = self._determine_pdf_type(doc)

                # Create basic page info
                pages = self._create_basic_page_info(doc)

                # Validate document
                validation_errors = self._validate_document_structure(doc)

                pdf_info = PDFInfo(
                    file_path=pdf_path,
                    metadata=metadata,
                    pdf_type=pdf_type,
                    pages=pages,
                    is_valid=len(validation_errors) == 0,
                    validation_errors=validation_errors,
                )

                return pdf_info

            finally:
                doc.close()

        except Exception as e:
            self.logger.error(
                "Error in enhanced metadata extraction",
                pdf_path=str(pdf_path),
                error=str(e),
                exc_info=True,
            )
            raise MetadataExtractionError(
                f"Enhanced metadata extraction error: {str(e)}", pdf_path
            )

    def _extract_basic_metadata(
        self, doc: fitz.Document, file_size: int
    ) -> PDFMetadata:
        """Extract basic PDF metadata."""
        try:
            meta = doc.metadata

            # Parse dates
            creation_date = self._parse_pdf_date(meta.get("creationDate"))
            modification_date = self._parse_pdf_date(meta.get("modDate"))

            # Parse keywords
            keywords = self._parse_keywords(meta.get("keywords"))

            # Extract permissions
            permissions = self._extract_permissions(doc)

            # Get PDF version
            pdf_version = None
            try:
                version_info = doc.pdf_version()
                if version_info:
                    pdf_version = f"{version_info[0]}.{version_info[1]}"
            except Exception:
                pass

            metadata = PDFMetadata(
                title=self._clean_string(meta.get("title")),
                author=self._clean_string(meta.get("author")),
                subject=self._clean_string(meta.get("subject")),
                creator=self._clean_string(meta.get("creator")),
                producer=self._clean_string(meta.get("producer")),
                creation_date=creation_date,
                modification_date=modification_date,
                keywords=keywords,
                page_count=doc.page_count,
                file_size=file_size,
                pdf_version=pdf_version,
                is_encrypted=doc.needs_pass,
                is_linearized=doc.is_fast_web_view,
                permissions=permissions,
            )

            return metadata

        except Exception as e:
            self.logger.warning("Error extracting basic metadata", error=str(e))
            # Return minimal metadata
            return PDFMetadata(
                page_count=doc.page_count,
                file_size=file_size,
                is_encrypted=doc.needs_pass,
            )

    def _extract_xmp_metadata(self, doc: fitz.Document) -> Dict[str, Any]:
        """Extract XMP metadata if available."""
        try:
            xmp_metadata = doc.xref_xml_metadata()
            if not xmp_metadata:
                return {}

            # Parse XMP XML (basic parsing)
            xmp_data = {}

            # Extract common XMP fields
            import xml.etree.ElementTree as ET

            try:
                root = ET.fromstring(xmp_metadata)

                # Define common XMP namespaces
                namespaces = {
                    "dc": "http://purl.org/dc/elements/1.1/",
                    "xmp": "http://ns.adobe.com/xap/1.0/",
                    "pdf": "http://ns.adobe.com/pdf/1.3/",
                    "xmpMM": "http://ns.adobe.com/xap/1.0/mm/",
                }

                # Extract Dublin Core metadata
                for field in ["title", "creator", "description", "subject"]:
                    elements = root.findall(f".//dc:{field}", namespaces)
                    if elements:
                        values = []
                        for elem in elements:
                            if elem.text:
                                values.append(elem.text.strip())
                        if values:
                            xmp_data[f"dc_{field}"] = values

                # Extract XMP metadata
                for field in ["CreateDate", "ModifyDate", "CreatorTool"]:
                    elements = root.findall(f".//xmp:{field}", namespaces)
                    if elements and elements[0].text:
                        xmp_data[f"xmp_{field}"] = elements[0].text.strip()

                # Extract PDF-specific metadata
                for field in ["Producer", "Keywords"]:
                    elements = root.findall(f".//pdf:{field}", namespaces)
                    if elements and elements[0].text:
                        xmp_data[f"pdf_{field}"] = elements[0].text.strip()

            except ET.ParseError as e:
                self.logger.warning("Error parsing XMP XML", error=str(e))

            return xmp_data

        except Exception as e:
            self.logger.warning("Error extracting XMP metadata", error=str(e))
            return {}

    def _analyze_document_structure(self, doc: fitz.Document) -> Dict[str, Any]:
        """Analyze document structure and content."""
        try:
            structure = {
                "page_sizes": [],
                "page_orientations": [],
                "has_bookmarks": False,
                "has_annotations": False,
                "has_forms": False,
                "has_javascript": False,
                "font_info": {},
                "color_spaces": set(),
                "embedded_files": 0,
            }

            # Analyze pages
            for page_num in range(min(10, doc.page_count)):  # Analyze first 10 pages
                try:
                    page = doc[page_num]
                    rect = page.rect

                    # Page size and orientation
                    structure["page_sizes"].append((rect.width, rect.height))
                    structure["page_orientations"].append(
                        "portrait" if rect.height > rect.width else "landscape"
                    )

                    # Check for annotations
                    if page.annots() and not structure["has_annotations"]:
                        structure["has_annotations"] = True

                    # Analyze fonts (basic)
                    try:
                        blocks = page.get_text("dict")
                        for block in blocks.get("blocks", []):
                            if "lines" in block:
                                for line in block["lines"]:
                                    for span in line.get("spans", []):
                                        font = span.get("font", "Unknown")
                                        if font not in structure["font_info"]:
                                            structure["font_info"][font] = 0
                                        structure["font_info"][font] += 1
                    except Exception:
                        pass

                except Exception as e:
                    self.logger.warning(
                        "Error analyzing page structure",
                        page_num=page_num,
                        error=str(e),
                    )
                    continue

            # Check for bookmarks/outline
            try:
                toc = doc.get_toc()
                structure["has_bookmarks"] = len(toc) > 0
            except Exception:
                pass

            # Check for forms
            try:
                # This is a simplified check - would need more detailed analysis
                for page_num in range(min(3, doc.page_count)):
                    page = doc[page_num]
                    widgets = page.widgets()
                    if widgets:
                        structure["has_forms"] = True
                        break
            except Exception:
                pass

            # Check for embedded files
            try:
                structure["embedded_files"] = len(doc.embfile_names())
            except Exception:
                pass

            # Convert sets to lists for JSON serialization
            structure["color_spaces"] = list(structure["color_spaces"])

            return structure

        except Exception as e:
            self.logger.warning("Error analyzing document structure", error=str(e))
            return {}

    def _determine_pdf_type(self, doc: fitz.Document) -> PDFType:
        """Determine PDF type (born-digital, scanned, hybrid)."""
        try:
            pages_to_check = min(5, doc.page_count)
            text_pages = 0
            image_heavy_pages = 0

            for page_num in range(pages_to_check):
                try:
                    page = doc[page_num]

                    # Check for extractable text
                    text = page.get_text().strip()
                    has_meaningful_text = len(text) > 50

                    # Check for images
                    images = page.get_images()
                    has_large_images = len(images) > 0

                    # Analyze text-to-image ratio
                    if has_large_images:
                        # Check if page is mostly image
                        total_image_area = 0
                        for img_ref in images:
                            try:
                                # This is a rough estimation
                                total_image_area += (
                                    page.rect.width * page.rect.height * 0.3
                                )  # Assume 30% coverage per image
                            except Exception:
                                pass

                        page_area = page.rect.width * page.rect.height
                        image_coverage = (
                            min(total_image_area / page_area, 1.0)
                            if page_area > 0
                            else 0
                        )

                        if image_coverage > 0.7 and not has_meaningful_text:
                            image_heavy_pages += 1
                        elif has_meaningful_text:
                            text_pages += 1
                    elif has_meaningful_text:
                        text_pages += 1

                except Exception as e:
                    self.logger.warning(
                        "Error analyzing page for PDF type determination",
                        page_num=page_num,
                        error=str(e),
                    )
                    continue

            # Determine type based on analysis
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

    def _create_basic_page_info(self, doc: fitz.Document) -> List[Any]:
        """Create basic page information."""
        from .types import PageInfo  # Import here to avoid circular imports

        pages = []

        for page_num in range(doc.page_count):
            try:
                page = doc[page_num]
                rect = page.rect

                # Basic page info
                page_info = PageInfo(
                    page_num=page_num + 1,
                    width=rect.width,
                    height=rect.height,
                    rotation=page.rotation,
                    text_blocks=[],  # Will be populated by text extractor
                    images=[],  # Will be populated by image extractor
                    has_text=len(page.get_text().strip()) > 10,
                    is_scanned=False,  # Will be determined by other analysis
                )
                pages.append(page_info)

            except Exception as e:
                self.logger.warning(
                    "Error creating page info", page_num=page_num + 1, error=str(e)
                )
                continue

        return pages

    def _validate_document_structure(self, doc: fitz.Document) -> List[str]:
        """Validate document structure and return any issues."""
        errors = []

        try:
            # Check for basic issues
            if doc.page_count == 0:
                errors.append("Document has no pages")

            # Check a few pages for corruption
            corrupted_pages = 0
            pages_to_check = min(5, doc.page_count)

            for page_num in range(pages_to_check):
                try:
                    page = doc[page_num]
                    rect = page.rect

                    if rect.width <= 0 or rect.height <= 0:
                        corrupted_pages += 1
                        continue

                    # Try to access page content
                    page.get_text()

                except Exception:
                    corrupted_pages += 1

            if corrupted_pages > 0:
                errors.append(f"Found {corrupted_pages} corrupted pages in sample")

            # Check for password protection
            if doc.needs_pass:
                errors.append("Document is password protected")

        except Exception as e:
            errors.append(f"Error validating document structure: {str(e)}")

        return errors

    def _parse_pdf_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse PDF date string to datetime object."""
        if not date_str:
            return None

        try:
            # PDF dates are in format: D:YYYYMMDDHHmmSSOHH'mm'
            if date_str.startswith("D:"):
                date_str = date_str[2:]

            # Try different date formats
            date_formats = [
                "%Y%m%d%H%M%S%z",
                "%Y%m%d%H%M%S",
                "%Y%m%d%H%M",
                "%Y%m%d",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
            ]

            for fmt in date_formats:
                try:
                    # Clean up the date string for parsing
                    clean_date = (
                        date_str.replace("'", "").replace("+", "+").replace("-", "-")
                    )
                    if len(clean_date) > 14:
                        clean_date = clean_date[
                            :14
                        ]  # Take first 14 characters for basic format

                    return datetime.strptime(clean_date, fmt)
                except ValueError:
                    continue

            # If standard parsing fails, try manual parsing
            if len(date_str) >= 8:
                year = int(date_str[:4])
                month = int(date_str[4:6]) if len(date_str) >= 6 else 1
                day = int(date_str[6:8]) if len(date_str) >= 8 else 1
                hour = int(date_str[8:10]) if len(date_str) >= 10 else 0
                minute = int(date_str[10:12]) if len(date_str) >= 12 else 0
                second = int(date_str[12:14]) if len(date_str) >= 14 else 0

                return datetime(year, month, day, hour, minute, second)

        except Exception as e:
            self.logger.warning(
                "Error parsing PDF date", date_str=date_str, error=str(e)
            )

        return None

    def _parse_keywords(self, keywords_str: Optional[str]) -> List[str]:
        """Parse keywords string into list."""
        if not keywords_str:
            return []

        # Try different separators
        for sep in [";", ",", "\\n", "\\r", "|"]:
            if sep in keywords_str:
                keywords = [kw.strip() for kw in keywords_str.split(sep) if kw.strip()]
                if keywords:
                    return keywords

        # Single keyword
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
                "form": bool(perms & fitz.PDF_PERM_FORM),
                "accessibility": bool(perms & fitz.PDF_PERM_ACCESSIBILITY),
                "assemble": bool(perms & fitz.PDF_PERM_ASSEMBLE),
                "print_high_quality": bool(perms & fitz.PDF_PERM_PRINT_HQ),
            }
        except Exception as e:
            self.logger.warning("Error extracting permissions", error=str(e))
            return {}

    def _clean_string(self, value: Optional[str]) -> Optional[str]:
        """Clean and validate string value."""
        if not value:
            return None

        cleaned = value.strip()
        if not cleaned:
            return None

        # Remove null bytes and other problematic characters
        cleaned = cleaned.replace("\\x00", "").replace("\\0", "")

        return cleaned if cleaned else None

    def _merge_xmp_metadata(
        self, metadata: PDFMetadata, xmp_data: Dict[str, Any]
    ) -> PDFMetadata:
        """Merge XMP metadata with basic metadata."""
        try:
            # Update fields if XMP has better data
            if "dc_title" in xmp_data and not metadata.title:
                titles = xmp_data["dc_title"]
                metadata.title = (
                    titles[0] if isinstance(titles, list) and titles else None
                )

            if "dc_creator" in xmp_data and not metadata.author:
                creators = xmp_data["dc_creator"]
                metadata.author = (
                    creators[0] if isinstance(creators, list) and creators else None
                )

            if "dc_subject" in xmp_data and not metadata.keywords:
                subjects = xmp_data["dc_subject"]
                if isinstance(subjects, list):
                    metadata.keywords = subjects

            # Parse XMP dates
            if "xmp_CreateDate" in xmp_data and not metadata.creation_date:
                metadata.creation_date = self._parse_xmp_date(
                    xmp_data["xmp_CreateDate"]
                )

            if "xmp_ModifyDate" in xmp_data and not metadata.modification_date:
                metadata.modification_date = self._parse_xmp_date(
                    xmp_data["xmp_ModifyDate"]
                )

        except Exception as e:
            self.logger.warning("Error merging XMP metadata", error=str(e))

        return metadata

    def _merge_structure_info(
        self, metadata: PDFMetadata, structure: Dict[str, Any]
    ) -> PDFMetadata:
        """Merge document structure information with metadata."""
        try:
            # Add structure information as additional metadata
            # This could be stored in a separate field if needed
            pass
        except Exception as e:
            self.logger.warning("Error merging structure info", error=str(e))

        return metadata

    def _parse_xmp_date(self, date_str: str) -> Optional[datetime]:
        """Parse XMP date string."""
        try:
            # XMP dates are typically in ISO format
            # Remove timezone info for simple parsing
            if "+" in date_str:
                date_str = date_str.split("+")[0]
            if "T" in date_str:
                return datetime.fromisoformat(date_str.replace("Z", ""))

            return datetime.fromisoformat(date_str)

        except Exception as e:
            self.logger.warning(
                "Error parsing XMP date", date_str=date_str, error=str(e)
            )
            return None

    def get_metadata_summary(self, metadata: PDFMetadata) -> Dict[str, Any]:
        """Get a summary of extracted metadata."""
        return {
            "has_title": bool(metadata.title),
            "has_author": bool(metadata.author),
            "has_creation_date": bool(metadata.creation_date),
            "has_keywords": bool(metadata.keywords),
            "page_count": metadata.page_count,
            "file_size_mb": round(metadata.file_size / 1024 / 1024, 2),
            "pdf_version": metadata.pdf_version,
            "is_encrypted": metadata.is_encrypted,
            "is_linearized": metadata.is_linearized,
            "permissions_count": len([p for p in metadata.permissions.values() if p])
            if metadata.permissions
            else 0,
        }
