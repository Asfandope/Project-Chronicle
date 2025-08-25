"""
Synthetic test data generation for magazine layouts.

This module provides comprehensive synthetic data generation for testing
the magazine extraction pipeline with realistic layouts and known ground truth.
"""

from .generator import SyntheticDataGenerator, create_comprehensive_test_suite, create_edge_case_test_suite
from .layout_engine import LayoutEngine, LayoutTemplate, MagazineStyle
from .content_factory import ContentFactory, ArticleTemplate, MediaTemplate
from .pdf_renderer import PDFRenderer, RenderingOptions
from .ground_truth import GroundTruthGenerator, GroundTruthData
from .variations import VariationEngine, LayoutVariation
from .accuracy_calculator import AccuracyCalculator, DocumentAccuracy, ArticleAccuracy, FieldAccuracy
from .types import (
    GeneratedDocument,
    TestSuite,
    EdgeCaseType,
    BrandConfiguration,
    GenerationConfig,
    SyntheticDataError,
    ArticleData,
    TextElement,
    ImageElement
)

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
    "SyntheticDataError"
]