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
from .validator import SchemaValidator, ValidationResult
from .formatter import XMLFormatter, FormattingOptions
from .types import XMLError, ValidationError, FormattingError

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
    "FormattingError"
]