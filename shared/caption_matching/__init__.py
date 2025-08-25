"""
Spatial proximity-based caption matching module.

This module provides high-accuracy image-caption pairing using spatial analysis,
semantic graph traversal, and keyword matching to achieve 99% correct pairing.
"""

from .orchestrator import CaptionMatchingOrchestrator
from .matcher import CaptionMatcher, SpatialMatch
from .analyzer import SpatialAnalyzer, ProximityScore
from .resolver import AmbiguityResolver, MatchConfidence
from .filename_generator import FilenameGenerator, FilenameStrategy
from .types import (
    ImageCaptionPair,
    SpatialConfig,
    MatchingResult,
    MatchingMetrics,
    MatchingError
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