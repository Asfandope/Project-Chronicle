"""
Type definitions for contributor extraction module.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import re


class ExtractionError(Exception):
    """Base exception for contributor extraction errors."""
    
    def __init__(self, message: str, text: Optional[str] = None, pattern: Optional[str] = None):
        self.text = text
        self.pattern = pattern
        super().__init__(message)


class ContributorRole(Enum):
    """Types of contributor roles."""
    AUTHOR = "author"
    PHOTOGRAPHER = "photographer" 
    ILLUSTRATOR = "illustrator"
    EDITOR = "editor"
    CORRESPONDENT = "correspondent"
    REPORTER = "reporter"
    COLUMNIST = "columnist"
    REVIEWER = "reviewer"
    GRAPHIC_DESIGNER = "graphic_designer"
    UNKNOWN = "unknown"


class NamePart(Enum):
    """Parts of a person's name."""
    PREFIX = "prefix"          # Mr., Dr., Prof.
    FIRST = "first"           # Given name
    MIDDLE = "middle"         # Middle names/initials
    LAST = "last"            # Family name
    SUFFIX = "suffix"        # Jr., Sr., III, Ph.D.
    UNKNOWN = "unknown"


@dataclass
class NormalizedName:
    """Normalized name structure."""
    
    # Core name components
    first_name: str = ""
    middle_names: List[str] = field(default_factory=list)
    last_name: str = ""
    
    # Optional components
    prefixes: List[str] = field(default_factory=list)  # Dr., Prof., Mr.
    suffixes: List[str] = field(default_factory=list)  # Jr., Ph.D., III
    
    # Metadata
    original_text: str = ""
    confidence: float = 1.0
    normalization_method: str = ""
    
    @property
    def full_name(self) -> str:
        """Get full name in natural order."""
        parts = []
        
        if self.prefixes:
            parts.extend(self.prefixes)
        
        if self.first_name:
            parts.append(self.first_name)
        
        if self.middle_names:
            parts.extend(self.middle_names)
        
        if self.last_name:
            parts.append(self.last_name)
        
        if self.suffixes:
            parts.extend(self.suffixes)
        
        return " ".join(parts)
    
    @property
    def last_first_format(self) -> str:
        """Get name in 'Last, First' format."""
        if not self.last_name:
            return self.full_name
        
        first_parts = []
        
        if self.prefixes:
            first_parts.extend(self.prefixes)
        
        if self.first_name:
            first_parts.append(self.first_name)
        
        if self.middle_names:
            first_parts.extend(self.middle_names)
        
        if self.suffixes:
            first_parts.extend(self.suffixes)
        
        if first_parts:
            return f"{self.last_name}, {' '.join(first_parts)}"
        else:
            return self.last_name
    
    @property
    def is_complete(self) -> bool:
        """Check if name has both first and last components."""
        return bool(self.first_name and self.last_name)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "first_name": self.first_name,
            "middle_names": self.middle_names,
            "last_name": self.last_name,
            "prefixes": self.prefixes,
            "suffixes": self.suffixes,
            "full_name": self.full_name,
            "last_first_format": self.last_first_format,
            "original_text": self.original_text,
            "confidence": self.confidence,
            "normalization_method": self.normalization_method,
            "is_complete": self.is_complete
        }


@dataclass
class ContributorMatch:
    """A potential contributor match from text."""
    
    # Match details
    text: str
    start_pos: int
    end_pos: int
    
    # Classification
    role: ContributorRole
    role_confidence: float
    
    # Name extraction
    extracted_names: List[str] = field(default_factory=list)
    normalized_names: List[NormalizedName] = field(default_factory=list)
    
    # Context information
    context_before: str = ""
    context_after: str = ""
    
    # Pattern information
    pattern_used: str = ""
    extraction_method: str = ""
    
    # Quality metrics
    extraction_confidence: float = 1.0
    quality_score: float = 1.0
    
    def get_primary_name(self) -> Optional[NormalizedName]:
        """Get the primary (first/most confident) normalized name."""
        if self.normalized_names:
            return max(self.normalized_names, key=lambda n: n.confidence)
        return None
    
    def get_all_names_formatted(self, format_type: str = "last_first") -> List[str]:
        """Get all names in specified format."""
        if format_type == "last_first":
            return [name.last_first_format for name in self.normalized_names]
        elif format_type == "full":
            return [name.full_name for name in self.normalized_names]
        else:
            return [name.original_text for name in self.normalized_names]


@dataclass
class ExtractedContributor:
    """Complete extracted contributor information."""
    
    # Core contributor data
    name: NormalizedName
    role: ContributorRole
    
    # Source information
    source_text: str
    source_match: ContributorMatch
    
    # Quality metrics
    extraction_confidence: float
    role_confidence: float
    name_confidence: float
    
    # Metadata
    extraction_timestamp: datetime = field(default_factory=datetime.now)
    extraction_method: str = ""
    
    @property
    def overall_confidence(self) -> float:
        """Calculate overall confidence score."""
        return (
            self.extraction_confidence * 0.4 +
            self.role_confidence * 0.3 +
            self.name_confidence * 0.3
        )
    
    @property
    def is_high_quality(self) -> bool:
        """Check if this is a high-quality extraction."""
        return (
            self.overall_confidence >= 0.9 and
            self.name.is_complete and
            self.role != ContributorRole.UNKNOWN
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name.to_dict(),
            "role": self.role.value,
            "source_text": self.source_text,
            "extraction_confidence": self.extraction_confidence,
            "role_confidence": self.role_confidence,
            "name_confidence": self.name_confidence,
            "overall_confidence": self.overall_confidence,
            "is_high_quality": self.is_high_quality,
            "extraction_timestamp": self.extraction_timestamp.isoformat(),
            "extraction_method": self.extraction_method
        }


@dataclass
class ExtractionResult:
    """Complete extraction result for a document/text."""
    
    # Extracted contributors
    contributors: List[ExtractedContributor] = field(default_factory=list)
    
    # All matches (including low-confidence)
    all_matches: List[ContributorMatch] = field(default_factory=list)
    
    # Processing metadata
    processing_time: float = 0.0
    text_length: int = 0
    
    # Quality metrics
    extraction_quality: str = "unknown"  # high, medium, low
    
    @property
    def author_count(self) -> int:
        """Get number of extracted authors."""
        return len([c for c in self.contributors if c.role == ContributorRole.AUTHOR])
    
    @property
    def photographer_count(self) -> int:
        """Get number of extracted photographers."""
        return len([c for c in self.contributors if c.role == ContributorRole.PHOTOGRAPHER])
    
    @property
    def total_contributors(self) -> int:
        """Get total number of contributors."""
        return len(self.contributors)
    
    @property
    def average_confidence(self) -> float:
        """Get average confidence across all contributors."""
        if not self.contributors:
            return 0.0
        return sum(c.overall_confidence for c in self.contributors) / len(self.contributors)
    
    def get_contributors_by_role(self, role: ContributorRole) -> List[ExtractedContributor]:
        """Get contributors filtered by role."""
        return [c for c in self.contributors if c.role == role]
    
    def get_high_quality_contributors(self) -> List[ExtractedContributor]:
        """Get only high-quality contributor extractions."""
        return [c for c in self.contributors if c.is_high_quality]


@dataclass
class ExtractionConfig:
    """Configuration for contributor extraction."""
    
    # NER model settings
    ner_model: str = "en_core_web_sm"  # spaCy model
    use_transformers: bool = True
    transformer_model: str = "dbmdz/bert-large-cased-finetuned-conll03-english"
    
    # Confidence thresholds
    min_extraction_confidence: float = 0.7
    min_role_confidence: float = 0.8
    min_name_confidence: float = 0.75
    
    # Pattern matching settings
    enable_pattern_matching: bool = True
    enable_contextual_analysis: bool = True
    context_window_size: int = 50  # characters before/after
    
    # Name normalization settings
    normalize_names: bool = True
    require_complete_names: bool = False  # Require both first and last
    handle_initials: bool = True
    expand_nicknames: bool = True
    
    # Quality filtering
    filter_low_quality: bool = True
    deduplicate_contributors: bool = True
    similarity_threshold: float = 0.85  # For deduplication
    
    # Role classification settings
    role_classification_method: str = "hybrid"  # pattern, ml, hybrid
    custom_role_patterns: Dict[str, List[str]] = field(default_factory=dict)
    
    # Processing options
    max_names_per_match: int = 5
    max_matches_per_text: int = 20
    
    @classmethod
    def create_high_precision(cls) -> "ExtractionConfig":
        """Create configuration optimized for high precision."""
        return cls(
            min_extraction_confidence=0.9,
            min_role_confidence=0.95,
            min_name_confidence=0.9,
            require_complete_names=True,
            filter_low_quality=True
        )
    
    @classmethod
    def create_high_recall(cls) -> "ExtractionConfig":
        """Create configuration optimized for high recall."""
        return cls(
            min_extraction_confidence=0.5,
            min_role_confidence=0.6,
            min_name_confidence=0.6,
            require_complete_names=False,
            filter_low_quality=False
        )


@dataclass
class ExtractionMetrics:
    """Metrics for evaluating extraction performance."""
    
    # Processing metrics
    total_texts_processed: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    
    # Extraction counts
    total_matches_found: int = 0
    total_contributors_extracted: int = 0
    high_quality_extractions: int = 0
    
    # Role distribution
    authors_extracted: int = 0
    photographers_extracted: int = 0
    illustrators_extracted: int = 0
    other_roles_extracted: int = 0
    
    # Quality metrics
    average_extraction_confidence: float = 0.0
    average_role_confidence: float = 0.0
    average_name_confidence: float = 0.0
    
    # Target achievement
    name_extraction_rate: float = 0.0  # Target: 99%
    role_classification_accuracy: float = 0.0  # Target: 99.5%
    
    # Error analysis
    failed_extractions: int = 0
    ambiguous_cases: int = 0
    edge_case_handling: int = 0
    
    def calculate_performance_score(self) -> float:
        """Calculate overall performance score."""
        return (
            self.name_extraction_rate * 0.5 +
            self.role_classification_accuracy * 0.5
        )
    
    @property
    def meets_targets(self) -> bool:
        """Check if extraction meets target performance."""
        return (
            self.name_extraction_rate >= 0.99 and
            self.role_classification_accuracy >= 0.995
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "processing": {
                "total_texts": self.total_texts_processed,
                "total_time": self.total_processing_time,
                "avg_time": self.average_processing_time
            },
            "extraction": {
                "total_matches": self.total_matches_found,
                "total_contributors": self.total_contributors_extracted,
                "high_quality": self.high_quality_extractions
            },
            "roles": {
                "authors": self.authors_extracted,
                "photographers": self.photographers_extracted,
                "illustrators": self.illustrators_extracted,
                "other": self.other_roles_extracted
            },
            "quality": {
                "avg_extraction_confidence": self.average_extraction_confidence,
                "avg_role_confidence": self.average_role_confidence,
                "avg_name_confidence": self.average_name_confidence
            },
            "performance": {
                "name_extraction_rate": self.name_extraction_rate,
                "role_classification_accuracy": self.role_classification_accuracy,
                "performance_score": self.calculate_performance_score(),
                "meets_targets": self.meets_targets
            },
            "errors": {
                "failed_extractions": self.failed_extractions,
                "ambiguous_cases": self.ambiguous_cases,
                "edge_cases": self.edge_case_handling
            }
        }