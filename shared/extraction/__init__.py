"""
NER-based Contributor Extraction Module.

This module provides Named Entity Recognition (NER) based extraction
of contributors (authors, photographers, illustrators) from document text
with high-accuracy name extraction and role classification.
"""

from .classifier import ContributorRole, RoleClassifier
from .edge_cases import EdgeCaseHandler
from .extractor import ContributorExtractor, ExtractedContributor
from .normalizer import NameNormalizer, NormalizedName
from .optimizer import OptimizationStrategy, PerformanceOptimizer
from .patterns import BylinePatterns, CreditPatterns
from .types import (
    ContributorMatch,
    ExtractionConfig,
    ExtractionError,
    ExtractionMetrics,
    ExtractionResult,
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
