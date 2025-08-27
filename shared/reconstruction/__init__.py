"""
Article Reconstruction Module.

This module provides graph traversal algorithms for reconstructing complete
articles from semantic graphs, handling split articles, continuation markers,
and ambiguous connections.
"""

from .reconstructor import ArticleReconstructor, ReconstructedArticle
from .resolver import AmbiguityResolver, ConnectionScore
from .traversal import GraphTraversal, TraversalPath
from .types import (
    ArticleBoundary,
    ContinuationMarker,
    ReconstructionConfig,
    ReconstructionError,
)

__all__ = [
    # Core classes
    "ArticleReconstructor",
    "GraphTraversal",
    "AmbiguityResolver",
    # Data types
    "ReconstructedArticle",
    "TraversalPath",
    "ConnectionScore",
    "ArticleBoundary",
    "ContinuationMarker",
    "ReconstructionConfig",
    # Exceptions
    "ReconstructionError",
]
