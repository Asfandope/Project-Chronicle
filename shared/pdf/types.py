"""
Type definitions for PDF processing utilities.
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class PDFType(Enum):
    """PDF document type classification."""

    BORN_DIGITAL = "born_digital"
    SCANNED = "scanned"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


class PDFProcessingError(Exception):
    """Base exception for PDF processing errors."""

    def __init__(
        self,
        message: str,
        pdf_path: Optional[Path] = None,
        page_num: Optional[int] = None,
    ):
        self.pdf_path = pdf_path
        self.page_num = page_num
        super().__init__(message)


@dataclass
class BoundingBox:
    """Represents a bounding box with coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    def overlaps(self, other: "BoundingBox") -> bool:
        """Check if this bounding box overlaps with another."""
        return not (
            self.x1 < other.x0
            or other.x1 < self.x0
            or self.y1 < other.y0
            or other.y1 < self.y0
        )

    def intersection_area(self, other: "BoundingBox") -> float:
        """Calculate intersection area with another bounding box."""
        if not self.overlaps(other):
            return 0.0

        x_overlap = min(self.x1, other.x1) - max(self.x0, other.x0)
        y_overlap = min(self.y1, other.y1) - max(self.y0, other.y0)
        return x_overlap * y_overlap


@dataclass
class TextBlock:
    """Represents an extracted text block with positioning information."""

    text: str
    bbox: BoundingBox
    confidence: float
    font_size: Optional[float] = None
    font_family: Optional[str] = None
    is_bold: bool = False
    is_italic: bool = False
    text_type: Optional[str] = None  # title, paragraph, caption, etc.
    page_num: int = 0

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def char_count(self) -> int:
        return len(self.text.strip())

    def get_hash(self) -> str:
        """Generate deterministic hash for the text block."""
        content = f"{self.text}_{self.bbox.x0}_{self.bbox.y0}_{self.page_num}"
        return hashlib.md5(content.encode()).hexdigest()


@dataclass
class ImageInfo:
    """Represents an extracted image with metadata."""

    image_id: str
    bbox: BoundingBox
    width: int
    height: int
    format: str  # PNG, JPEG, etc.
    file_path: Path
    file_size: int
    page_num: int
    confidence: float = 1.0
    is_photo: bool = False
    is_chart: bool = False
    is_diagram: bool = False

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 0.0

    @property
    def area_pixels(self) -> int:
        return self.width * self.height

    def get_deterministic_filename(self, pdf_name: str) -> str:
        """Generate deterministic filename for the image."""
        # Use page number, position, and size for deterministic naming
        position_hash = hashlib.md5(
            f"{self.bbox.x0}_{self.bbox.y0}_{self.width}_{self.height}".encode()
        ).hexdigest()[:8]
        return (
            f"{pdf_name}_page{self.page_num:03d}_{position_hash}.{self.format.lower()}"
        )


@dataclass
class PageInfo:
    """Information about a PDF page."""

    page_num: int
    width: float
    height: float
    rotation: int
    text_blocks: List[TextBlock]
    images: List[ImageInfo]
    has_text: bool = True
    is_scanned: bool = False

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 0.0

    @property
    def total_text_area(self) -> float:
        return sum(block.bbox.area for block in self.text_blocks)

    @property
    def total_image_area(self) -> float:
        return sum(img.bbox.area for img in self.images)


@dataclass
class PDFMetadata:
    """Comprehensive PDF metadata."""

    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    keywords: List[str] = None
    page_count: int = 0
    file_size: int = 0
    pdf_version: Optional[str] = None
    is_encrypted: bool = False
    is_linearized: bool = False
    permissions: Dict[str, bool] = None

    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.permissions is None:
            self.permissions = {}


@dataclass
class PDFInfo:
    """Complete PDF document information."""

    file_path: Path
    metadata: PDFMetadata
    pdf_type: PDFType
    pages: List[PageInfo]
    is_valid: bool = True
    validation_errors: List[str] = None
    processing_time: Optional[float] = None

    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = []

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    @property
    def total_text_blocks(self) -> int:
        return sum(len(page.text_blocks) for page in self.pages)

    @property
    def total_images(self) -> int:
        return sum(len(page.images) for page in self.pages)

    @property
    def has_text_content(self) -> bool:
        return any(page.has_text for page in self.pages)

    @property
    def scanned_pages_count(self) -> int:
        return sum(1 for page in self.pages if page.is_scanned)

    def get_page(self, page_num: int) -> Optional[PageInfo]:
        """Get page info by page number (1-indexed)."""
        if 1 <= page_num <= len(self.pages):
            return self.pages[page_num - 1]
        return None

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the PDF information."""
        return {
            "file_path": str(self.file_path),
            "pdf_type": self.pdf_type.value,
            "page_count": self.total_pages,
            "text_blocks": self.total_text_blocks,
            "images": self.total_images,
            "has_text": self.has_text_content,
            "scanned_pages": self.scanned_pages_count,
            "file_size": self.metadata.file_size,
            "is_valid": self.is_valid,
            "validation_errors_count": len(self.validation_errors),
        }
