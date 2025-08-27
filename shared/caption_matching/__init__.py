"""
Spatial proximity-based caption matching module.

This module provides high-accuracy image-caption pairing using spatial analysis,
semantic graph traversal, and keyword matching to achieve 99% correct pairing.
"""

from .analyzer import ProximityScore, SpatialAnalyzer
from .filename_generator import FilenameGenerator, FilenameStrategy
from .matcher import CaptionMatcher, SpatialMatch
from .orchestrator import CaptionMatchingOrchestrator
from .resolver import AmbiguityResolver, MatchConfidence
from .types import (
    ImageCaptionPair,
    MatchingError,
    MatchingMetrics,
    MatchingResult,
    SpatialConfig,
)

__all__ = [
    # Main orchestrator
    "CaptionMatchingOrchestrator",
    # Core classes
    "CaptionMatcher",
    "SpatialAnalyzer",
    "AmbiguityResolver",
    "FilenameGenerator",
    # Data types
    "SpatialMatch",
    "ProximityScore",
    "MatchConfidence",
    "FilenameStrategy",
    "ImageCaptionPair",
    "SpatialConfig",
    "MatchingResult",
    "MatchingMetrics",
    # Exceptions
    "MatchingError",
]
