"""
Shared PDF utilities for Magazine PDF Extractor.

This module provides comprehensive PDF processing utilities following DRY principles,
supporting both born-digital and scanned PDFs with robust error handling.
"""

from .image_extractor import ImageExtractionError, ImageExtractor
from .metadata_extractor import MetadataExtractionError, MetadataExtractor
from .splitter import PageSplitError, PageSplitter
from .text_extractor import TextBlockExtractor, TextExtractionError
from .types import (
    BoundingBox,
    ImageInfo,
    PageInfo,
    PDFInfo,
    PDFMetadata,
    PDFProcessingError,
    TextBlock,
)
from .validator import PDFValidationError, PDFValidator

__all__ = [
    # Core classes
    "PDFValidator",
    "PageSplitter",
    "TextBlockExtractor",
    "ImageExtractor",
    "MetadataExtractor",
    # Exception types
    "PDFValidationError",
    "PageSplitError",
    "TextExtractionError",
    "ImageExtractionError",
    "MetadataExtractionError",
    "PDFProcessingError",
    # Data types
    "PDFInfo",
    "PageInfo",
    "TextBlock",
    "ImageInfo",
    "PDFMetadata",
    "BoundingBox",
]
