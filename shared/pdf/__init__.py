"""
Shared PDF utilities for Magazine PDF Extractor.

This module provides comprehensive PDF processing utilities following DRY principles,
supporting both born-digital and scanned PDFs with robust error handling.
"""

from .validator import PDFValidator, PDFValidationError
from .splitter import PageSplitter, PageSplitError
from .text_extractor import TextBlockExtractor, TextExtractionError
from .image_extractor import ImageExtractor, ImageExtractionError
from .metadata_extractor import MetadataExtractor, MetadataExtractionError
from .types import (
    PDFInfo,
    PageInfo,
    TextBlock,
    ImageInfo,
    PDFMetadata,
    BoundingBox,
    PDFProcessingError
)

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