"""
OCR Strategy Module - PRD Section 5.3 Implementation

Implements auto-detection, brand-specific preprocessing, and confidence scoring
to achieve <2% WER on scanned PDFs and <0.1% WER on born-digital PDFs.
"""

from .detector import DocumentTypeDetector, DocumentType
from .engine import OCREngine, OCRResult, OCRConfig
from .preprocessing import ImagePreprocessor, PreprocessingConfig
from .confidence import ConfidenceAnalyzer, ConfidenceMetrics
from .wer import WERCalculator, WERMetrics
from .strategy import OCRStrategy, OCRStrategyConfig
from .types import (
    OCRError,
    TextBlock,
    CharacterConfidence,
    WordConfidence,
    LineConfidence,
    PageOCRResult,
    QualityMetrics
)

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