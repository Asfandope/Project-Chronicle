"""
NER-based Contributor Extraction Module.

This module provides Named Entity Recognition (NER) based extraction
of contributors (authors, photographers, illustrators) from document text
with high-accuracy name extraction and role classification.
"""

from .extractor import ContributorExtractor, ExtractedContributor
from .classifier import RoleClassifier, ContributorRole
from .normalizer import NameNormalizer, NormalizedName
from .patterns import BylinePatterns, CreditPatterns
from .edge_cases import EdgeCaseHandler
from .optimizer import PerformanceOptimizer, OptimizationStrategy
from .types import (
    ExtractionResult,
    ContributorMatch,
    ExtractionConfig,
    ExtractionError,
    ExtractionMetrics
)

__all__ = [
    # Core classes
    "ContributorExtractor",
    "RoleClassifier",
    "NameNormalizer",
    
    # Pattern matching
    "BylinePatterns",
    "CreditPatterns",
    
    # Edge case handling and optimization
    "EdgeCaseHandler", 
    "PerformanceOptimizer",
    "OptimizationStrategy",
    
    # Data types
    "ExtractedContributor",
    "ContributorRole",
    "NormalizedName",
    "ExtractionResult",
    "ContributorMatch",
    "ExtractionConfig",
    "ExtractionMetrics",
    
    # Exceptions
    "ExtractionError",
]