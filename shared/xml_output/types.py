"""
Type definitions for XML output module.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


class XMLError(Exception):
    """Base exception for XML output errors."""
    pass


class ValidationError(XMLError):
    """Schema validation error."""
    
    def __init__(self, message: str, line_number: Optional[int] = None, column: Optional[int] = None):
        self.line_number = line_number
        self.column = column
        super().__init__(message)


class FormattingError(XMLError):
    """XML formatting error."""
    pass


class SchemaVersion(Enum):
    """Supported schema versions."""
    V1_0 = "1.0"


class OutputFormat(Enum):
    """XML output formatting options."""
    COMPACT = "compact"           # No whitespace, minimal size
    PRETTY = "pretty"            # Human-readable with indentation
    CANONICAL = "canonical"      # C14N canonical form
    DEBUG = "debug"              # Pretty with extra debug attributes


@dataclass
class XMLConfig:
    """Configuration for XML output generation."""
    
    # Schema settings
    schema_version: SchemaVersion = SchemaVersion.V1_0
    schema_location: Optional[Path] = None
    validate_output: bool = True
    
    # Output formatting
    output_format: OutputFormat = OutputFormat.CANONICAL
    encoding: str = "utf-8"
    xml_declaration: bool = True
    
    # Namespace settings
    use_namespaces: bool = True
    namespace_prefix: str = "art"
    target_namespace: str = "https://magazine-extractor.com/schemas/article/v1.0"
    
    # Deterministic output settings
    sort_attributes: bool = True
    sort_elements: bool = True
    normalize_whitespace: bool = True
    
    # Confidence score settings
    confidence_precision: int = 6  # Decimal places for confidence scores
    include_low_confidence: bool = False  # Include elements with confidence < 0.5
    confidence_threshold: float = 0.0
    
    # Debug and validation settings
    include_debug_info: bool = False
    include_processing_metadata: bool = True
    strict_validation: bool = True
    
    @classmethod
    def for_production(cls) -> "XMLConfig":
        """Create configuration optimized for production output."""
        return cls(
            output_format=OutputFormat.COMPACT,
            validate_output=True,
            include_debug_info=False,
            strict_validation=True,
            confidence_threshold=0.7
        )
    
    @classmethod
    def for_debugging(cls) -> "XMLConfig":
        """Create configuration optimized for debugging."""
        return cls(
            output_format=OutputFormat.DEBUG,
            validate_output=True,
            include_debug_info=True,
            strict_validation=False,
            confidence_threshold=0.0
        )


@dataclass
class FormattingOptions:
    """Options for XML formatting."""
    
    # Indentation settings
    indent_size: int = 2
    indent_char: str = " "
    max_line_length: int = 120
    
    # Element formatting
    self_closing_tags: bool = True
    quote_char: str = '"'
    attribute_quote_escape: bool = True
    
    # Content formatting
    preserve_whitespace: bool = False
    normalize_line_endings: bool = True
    trim_text_content: bool = True
    
    # Attribute ordering
    attribute_sort_order: List[str] = field(default_factory=lambda: [
        'id', 'type', 'role', 'page', 'confidence', 'extraction_confidence',
        'pairing_confidence', 'normalization_confidence'
    ])
    
    # Namespace handling
    namespace_declarations_first: bool = True
    sort_namespace_declarations: bool = True


@dataclass 
class ValidationResult:
    """Result of XML schema validation."""
    
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validation_time: float = 0.0
    
    # Detailed error information
    error_details: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_error(self, message: str, line: Optional[int] = None, column: Optional[int] = None):
        """Add a validation error."""
        self.errors.append(message)
        self.error_details.append({
            'type': 'error',
            'message': message,
            'line': line,
            'column': column
        })
        self.is_valid = False
    
    def add_warning(self, message: str, line: Optional[int] = None, column: Optional[int] = None):
        """Add a validation warning."""
        self.warnings.append(message)
        self.error_details.append({
            'type': 'warning',
            'message': message,
            'line': line,
            'column': column
        })
    
    @property
    def has_errors(self) -> bool:
        """Check if there are validation errors."""
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are validation warnings."""
        return len(self.warnings) > 0
    
    def summary(self) -> str:
        """Get validation summary."""
        if self.is_valid:
            warning_text = f" ({len(self.warnings)} warnings)" if self.warnings else ""
            return f"Validation passed{warning_text}"
        else:
            return f"Validation failed: {len(self.errors)} errors, {len(self.warnings)} warnings"


@dataclass
class ArticleData:
    """Structured article data for XML conversion."""
    
    # Core article information
    article_id: str
    title: str
    title_confidence: float
    
    # Publication metadata
    brand: str
    issue_date: datetime
    page_start: int
    page_end: int
    
    # Content
    contributors: List[Dict[str, Any]] = field(default_factory=list)
    text_blocks: List[Dict[str, Any]] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)
    
    # Processing metadata
    extraction_timestamp: datetime = field(default_factory=datetime.now)
    extraction_confidence: float = 1.0
    processing_pipeline_version: str = "1.0"
    
    # Quality metrics
    overall_quality: str = "unknown"
    text_extraction_quality: str = "unknown"
    media_matching_quality: str = "unknown"
    contributor_extraction_quality: str = "unknown"
    
    def validate_data(self) -> List[str]:
        """Validate article data completeness."""
        errors = []
        
        if not self.article_id:
            errors.append("Missing article ID")
        
        if not self.title:
            errors.append("Missing article title")
            
        if self.title_confidence < 0.0 or self.title_confidence > 1.0:
            errors.append("Invalid title confidence score")
            
        if not self.brand:
            errors.append("Missing brand information")
            
        if self.page_start < 1:
            errors.append("Invalid page start")
            
        if self.page_end < self.page_start:
            errors.append("Invalid page range")
        
        return errors


@dataclass
class ConversionResult:
    """Result of article to XML conversion."""
    
    xml_content: str
    validation_result: ValidationResult
    conversion_time: float
    
    # Statistics
    elements_created: int = 0
    attributes_added: int = 0
    namespace_declarations: int = 0
    
    # Quality metrics
    confidence_scores_included: int = 0
    low_confidence_elements_filtered: int = 0
    
    @property
    def is_successful(self) -> bool:
        """Check if conversion was successful."""
        return bool(self.xml_content) and self.validation_result.is_valid
    
    def summary(self) -> str:
        """Get conversion summary."""
        status = "successful" if self.is_successful else "failed"
        return (
            f"XML conversion {status}: "
            f"{self.elements_created} elements, "
            f"{self.attributes_added} attributes, "
            f"validation: {self.validation_result.summary()}"
        )