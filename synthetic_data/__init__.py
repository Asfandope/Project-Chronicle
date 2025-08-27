"""
Synthetic test data generation for magazine layouts.

This module provides comprehensive synthetic data generation for testing
the magazine extraction pipeline with realistic layouts and known ground truth.
"""

from .accuracy_calculator import (
    AccuracyCalculator,
    ArticleAccuracy,
    DocumentAccuracy,
    FieldAccuracy,
)
from .content_factory import ArticleTemplate, ContentFactory, MediaTemplate
from .generator import (
    SyntheticDataGenerator,
    create_comprehensive_test_suite,
    create_edge_case_test_suite,
)
from .ground_truth import GroundTruthData, GroundTruthGenerator
from .layout_engine import LayoutEngine, LayoutTemplate, MagazineStyle
from .pdf_renderer import PDFRenderer, RenderingOptions
from .types import (
    ArticleData,
    BrandConfiguration,
    EdgeCaseType,
    GeneratedDocument,
    GenerationConfig,
    ImageElement,
    SyntheticDataError,
    TestSuite,
    TextElement,
)
from .variations import LayoutVariation, VariationEngine

__all__ = [
    # Main generator
    "SyntheticDataGenerator",
    "create_comprehensive_test_suite",
    "create_edge_case_test_suite",
    # Core engines
    "LayoutEngine",
    "ContentFactory",
    "PDFRenderer",
    "GroundTruthGenerator",
    "VariationEngine",
    "AccuracyCalculator",
    # Templates and styles
    "LayoutTemplate",
    "MagazineStyle",
    "ArticleTemplate",
    "MediaTemplate",
    # Configuration
    "GenerationConfig",
    "RenderingOptions",
    "BrandConfiguration",
    "LayoutVariation",
    # Data types
    "GeneratedDocument",
    "GroundTruthData",
    "TestSuite",
    "EdgeCaseType",
    "DocumentAccuracy",
    "ArticleAccuracy",
    "FieldAccuracy",
    "ArticleData",
    "TextElement",
    "ImageElement",
    # Exceptions
    "SyntheticDataError",
]
