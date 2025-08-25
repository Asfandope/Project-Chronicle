"""
Advanced Layout Understanding System with LayoutLM and Semantic Graphs.

Provides high-accuracy document layout analysis using LayoutLM for block
classification and semantic graphs for document structure understanding.
Targets 99.5%+ classification accuracy with brand-specific optimization.
"""

# Core layout analysis
from .analyzer import LayoutAnalyzer, LayoutResult, PageLayout
from .classifier import BlockClassifier, BlockType, TextBlock
from .visualizer import LayoutVisualizer

# Advanced LayoutLM integration
from .layoutlm import LayoutLMClassifier
from .understanding import LayoutUnderstandingSystem
from .optimizer import AccuracyOptimizer, AccuracyMetrics

# Type definitions
from .types import (
    LayoutError,
    BoundingBox,
    ClassificationRule,
    LayoutConfig,
    VisualizationConfig
)

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