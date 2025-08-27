"""
OCR Strategy Module - PRD Section 5.3 Implementation

Implements auto-detection, brand-specific preprocessing, and confidence scoring
to achieve <2% WER on scanned PDFs and <0.1% WER on born-digital PDFs.
"""

from .confidence import ConfidenceAnalyzer, ConfidenceMetrics
from .detector import DocumentType, DocumentTypeDetector
from .engine import OCRConfig, OCREngine, OCRResult
from .preprocessing import ImagePreprocessor, PreprocessingConfig
from .strategy import OCRStrategy, OCRStrategyConfig
from .types import (
    CharacterConfidence,
    LineConfidence,
    OCRError,
    PageOCRResult,
    QualityMetrics,
    TextBlock,
    WordConfidence,
)
from .wer import WERCalculator, WERMetrics

__all__ = [
    # Core strategy
    "OCRStrategy",
    "OCRStrategyConfig",
    # Components
    "DocumentTypeDetector",
    "OCREngine",
    "ImagePreprocessor",
    "ConfidenceAnalyzer",
    "WERCalculator",
    # Data types
    "DocumentType",
    "OCRResult",
    "OCRConfig",
    "PreprocessingConfig",
    "ConfidenceMetrics",
    "WERMetrics",
    "TextBlock",
    "CharacterConfidence",
    "WordConfidence",
    "LineConfidence",
    "PageOCRResult",
    "QualityMetrics",
    # Exceptions
    "OCRError",
]
