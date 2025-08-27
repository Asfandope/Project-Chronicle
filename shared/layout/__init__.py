"""
Advanced Layout Understanding System with LayoutLM and Semantic Graphs.

Provides high-accuracy document layout analysis using LayoutLM for block
classification and semantic graphs for document structure understanding.
Targets 99.5%+ classification accuracy with brand-specific optimization.
"""

# Core layout analysis
from .analyzer import LayoutAnalyzer, LayoutResult, PageLayout
from .classifier import BlockClassifier, BlockType, TextBlock

# Advanced LayoutLM integration
from .layoutlm import LayoutLMClassifier
from .optimizer import AccuracyMetrics, AccuracyOptimizer

# Type definitions
from .types import (
    BoundingBox,
    ClassificationRule,
    LayoutConfig,
    LayoutError,
    VisualizationConfig,
)
from .understanding import LayoutUnderstandingSystem
from .visualizer import LayoutVisualizer

__all__ = [
    # Core classes
    "LayoutAnalyzer",
    "BlockClassifier",
    "LayoutVisualizer",
    # Advanced understanding
    "LayoutLMClassifier",
    "LayoutUnderstandingSystem",
    "AccuracyOptimizer",
    # Data types
    "LayoutResult",
    "PageLayout",
    "TextBlock",
    "BlockType",
    "BoundingBox",
    "ClassificationRule",
    "LayoutConfig",
    "VisualizationConfig",
    "AccuracyMetrics",
    # Exceptions
    "LayoutError",
]
