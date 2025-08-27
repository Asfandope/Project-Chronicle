"""
XML output generation with constrained generation and schema validation.

This module provides high-quality XML output generation with:
- Canonical XML schema compliance
- lxml-based validation
- Confidence scores as attributes
- Deterministic output with sorted attributes
- Pretty-printing for debugging
"""

from .converter import ArticleXMLConverter, XMLConfig
from .formatter import FormattingOptions, XMLFormatter
from .types import FormattingError, ValidationError, XMLError
from .validator import SchemaValidator, ValidationResult

__all__ = [
    # Core classes
    "ArticleXMLConverter",
    "SchemaValidator",
    "XMLFormatter",
    # Configuration
    "XMLConfig",
    "FormattingOptions",
    "ValidationResult",
    # Exceptions
    "XMLError",
    "ValidationError",
    "FormattingError",
]
