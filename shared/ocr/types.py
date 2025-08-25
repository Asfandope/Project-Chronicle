"""
Type definitions for OCR processing.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import numpy as np


class DocumentType(Enum):
    """Document type classification for OCR strategy selection."""
    BORN_DIGITAL = "born_digital"
    SCANNED = "scanned"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


class OCRError(Exception):
    """Base exception for OCR processing errors."""
    
    def __init__(self, message: str, page_num: Optional[int] = None, brand: Optional[str] = None):
        self.page_num = page_num
        self.brand = brand
        super().__init__(message)


@dataclass
class CharacterConfidence:
    """Character-level confidence information."""
    char: str
    confidence: float
    bbox: tuple  # (x0, y0, x1, y1)
    alternatives: List[tuple] = field(default_factory=list)  # [(char, confidence), ...]
    
    @property
    def is_reliable(self) -> bool:
        """Check if character confidence is reliable (>80%)."""
        return self.confidence > 0.8
    
    @property
    def is_uncertain(self) -> bool:
        """Check if character confidence is uncertain (<60%)."""
        return self.confidence < 0.6


@dataclass
class WordConfidence:
    """Word-level confidence aggregation."""
    text: str
    confidence: float
    characters: List[CharacterConfidence]
    bbox: tuple  # (x0, y0, x1, y1)
    
    @property
    def char_count(self) -> int:
        return len(self.characters)
    
    @property
    def reliable_char_count(self) -> int:
        return sum(1 for char in self.characters if char.is_reliable)
    
    @property
    def uncertain_char_count(self) -> int:
        return sum(1 for char in self.characters if char.is_uncertain)
    
    @property
    def reliability_ratio(self) -> float:
        """Ratio of reliable characters to total characters."""
        return self.reliable_char_count / max(self.char_count, 1)


@dataclass
class LineConfidence:
    """Line-level confidence aggregation."""
    text: str
    confidence: float
    words: List[WordConfidence]
    bbox: tuple  # (x0, y0, x1, y1)
    baseline: Optional[float] = None
    
    @property
    def word_count(self) -> int:
        return len(self.words)
    
    @property
    def reliable_word_count(self) -> int:
        return sum(1 for word in self.words if word.confidence > 0.8)
    
    @property
    def avg_word_confidence(self) -> float:
        if not self.words:
            return 0.0
        return sum(word.confidence for word in self.words) / len(self.words)


@dataclass
class TextBlock:
    """OCR text block with hierarchical confidence."""
    text: str
    confidence: float
    lines: List[LineConfidence]
    bbox: tuple  # (x0, y0, x1, y1)
    block_type: str = "paragraph"  # paragraph, title, caption, etc.
    
    @property
    def line_count(self) -> int:
        return len(self.lines)
    
    @property
    def total_words(self) -> int:
        return sum(line.word_count for line in self.lines)
    
    @property
    def total_characters(self) -> int:
        return sum(word.char_count for word in line.words for line in self.lines)
    
    @property
    def avg_line_confidence(self) -> float:
        if not self.lines:
            return 0.0
        return sum(line.confidence for line in self.lines) / len(self.lines)


@dataclass
class PageOCRResult:
    """Complete OCR result for a single page."""
    page_num: int
    text_blocks: List[TextBlock]
    document_type: DocumentType
    processing_time: float
    image_preprocessing_applied: List[str] = field(default_factory=list)
    ocr_engine_version: str = ""
    
    @property
    def full_text(self) -> str:
        """Get all text content concatenated."""
        return "\n\n".join(block.text for block in self.text_blocks)
    
    @property
    def total_confidence(self) -> float:
        """Average confidence across all text blocks."""
        if not self.text_blocks:
            return 0.0
        total_chars = sum(block.total_characters for block in self.text_blocks)
        if total_chars == 0:
            return 0.0
        
        weighted_sum = sum(
            block.confidence * block.total_characters 
            for block in self.text_blocks
        )
        return weighted_sum / total_chars
    
    @property
    def word_count(self) -> int:
        return sum(block.total_words for block in self.text_blocks)
    
    @property
    def character_count(self) -> int:
        return sum(block.total_characters for block in self.text_blocks)


@dataclass
class QualityMetrics:
    """OCR quality metrics for monitoring and evaluation."""
    
    # WER metrics
    wer: float = 0.0
    substitutions: int = 0
    insertions: int = 0
    deletions: int = 0
    total_words: int = 0
    
    # Confidence metrics
    avg_confidence: float = 0.0
    min_confidence: float = 1.0
    max_confidence: float = 0.0
    confidence_std: float = 0.0
    
    # Character-level metrics
    char_accuracy: float = 0.0
    reliable_char_ratio: float = 0.0
    uncertain_char_ratio: float = 0.0
    
    # Processing metrics
    processing_time: float = 0.0
    preprocessing_time: float = 0.0
    ocr_time: float = 0.0
    
    # Quality flags
    meets_wer_target: bool = False
    high_confidence_text: bool = False
    requires_review: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging/storage."""
        return {
            "wer": self.wer,
            "substitutions": self.substitutions,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "total_words": self.total_words,
            "avg_confidence": self.avg_confidence,
            "min_confidence": self.min_confidence,
            "max_confidence": self.max_confidence,
            "confidence_std": self.confidence_std,
            "char_accuracy": self.char_accuracy,
            "reliable_char_ratio": self.reliable_char_ratio,
            "uncertain_char_ratio": self.uncertain_char_ratio,
            "processing_time": self.processing_time,
            "preprocessing_time": self.preprocessing_time,
            "ocr_time": self.ocr_time,
            "meets_wer_target": self.meets_wer_target,
            "high_confidence_text": self.high_confidence_text,
            "requires_review": self.requires_review
        }


@dataclass
class OCRResult:
    """Complete OCR result for a document."""
    pages: List[PageOCRResult]
    document_type: DocumentType
    total_processing_time: float
    quality_metrics: QualityMetrics
    brand: Optional[str] = None
    config_used: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def page_count(self) -> int:
        return len(self.pages)
    
    @property
    def total_text(self) -> str:
        """Get all text content from all pages."""
        return "\n\n".join(f"=== Page {page.page_num} ===\n{page.full_text}" 
                          for page in self.pages)
    
    @property
    def total_words(self) -> int:
        return sum(page.word_count for page in self.pages)
    
    @property
    def total_characters(self) -> int:
        return sum(page.character_count for page in self.pages)
    
    @property
    def average_confidence(self) -> float:
        """Calculate document-wide average confidence."""
        if not self.pages:
            return 0.0
        
        total_chars = sum(page.character_count for page in self.pages)
        if total_chars == 0:
            return 0.0
        
        weighted_sum = sum(
            page.total_confidence * page.character_count 
            for page in self.pages
        )
        return weighted_sum / total_chars
    
    def get_page(self, page_num: int) -> Optional[PageOCRResult]:
        """Get OCR result for specific page."""
        for page in self.pages:
            if page.page_num == page_num:
                return page
        return None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the OCR result."""
        return {
            "document_type": self.document_type.value,
            "page_count": self.page_count,
            "total_words": self.total_words,
            "total_characters": self.total_characters,
            "average_confidence": round(self.average_confidence, 3),
            "wer": round(self.quality_metrics.wer, 4),
            "processing_time": round(self.total_processing_time, 2),
            "meets_targets": self.quality_metrics.meets_wer_target,
            "brand": self.brand,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class OCRConfig:
    """Configuration for OCR engine."""
    
    # Tesseract configuration
    tesseract_config: Dict[str, Any] = field(default_factory=lambda: {
        "oem": 3,  # LSTM + Legacy
        "psm": 6,  # Uniform block of text
        "lang": "eng",
        "dpi": 300,
        "timeout": 30
    })
    
    # Quality thresholds
    min_confidence: float = 0.6
    min_word_confidence: float = 0.7
    min_char_confidence: float = 0.5
    
    # WER targets
    born_digital_wer_target: float = 0.001  # <0.1%
    scanned_wer_target: float = 0.02        # <2%
    
    # Processing options
    enable_preprocessing: bool = True
    enable_postprocessing: bool = True
    enable_confidence_filtering: bool = True
    enable_spell_correction: bool = False
    
    # Performance settings
    max_image_size: tuple = (4000, 4000)
    parallel_processing: bool = False
    cache_preprocessed_images: bool = True
    
    # Brand-specific overrides
    brand_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def get_brand_config(self, brand: str) -> "OCRConfig":
        """Get brand-specific configuration."""
        if brand not in self.brand_overrides:
            return self
        
        # Create a copy with brand-specific overrides
        config = OCRConfig(
            tesseract_config=self.tesseract_config.copy(),
            min_confidence=self.min_confidence,
            min_word_confidence=self.min_word_confidence,
            min_char_confidence=self.min_char_confidence,
            born_digital_wer_target=self.born_digital_wer_target,
            scanned_wer_target=self.scanned_wer_target,
            enable_preprocessing=self.enable_preprocessing,
            enable_postprocessing=self.enable_postprocessing,
            enable_confidence_filtering=self.enable_confidence_filtering,
            enable_spell_correction=self.enable_spell_correction,
            max_image_size=self.max_image_size,
            parallel_processing=self.parallel_processing,
            cache_preprocessed_images=self.cache_preprocessed_images,
            brand_overrides=self.brand_overrides
        )
        
        # Apply brand-specific overrides
        overrides = self.brand_overrides[brand]
        for key, value in overrides.items():
            if hasattr(config, key):
                if key == "tesseract_config" and isinstance(value, dict):
                    config.tesseract_config.update(value)
                else:
                    setattr(config, key, value)
        
        return config


@dataclass
class PreprocessingConfig:
    """Configuration for image preprocessing."""
    
    # Noise reduction
    denoise_enabled: bool = True
    denoise_strength: float = 3.0
    gaussian_blur_kernel: int = 1
    
    # Deskewing
    deskew_enabled: bool = True
    deskew_angle_threshold: float = 0.5
    deskew_max_angle: float = 10.0
    
    # Contrast enhancement
    contrast_enhancement: bool = True
    adaptive_threshold: bool = True
    threshold_block_size: int = 11
    threshold_constant: float = 2.0
    
    # Morphological operations
    morphology_enabled: bool = True
    kernel_size: int = 2
    closing_iterations: int = 1
    opening_iterations: int = 1
    
    # Scale and resolution
    target_dpi: int = 300
    min_dpi: int = 150
    upscale_factor: float = 2.0
    
    # Border removal
    border_removal: bool = True
    border_threshold: float = 0.05
    
    # Quality-based selection
    auto_select_best: bool = True
    quality_metrics: List[str] = field(default_factory=lambda: [
        "sharpness", "contrast", "noise_level"
    ])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "denoise_enabled": self.denoise_enabled,
            "denoise_strength": self.denoise_strength,
            "gaussian_blur_kernel": self.gaussian_blur_kernel,
            "deskew_enabled": self.deskew_enabled,
            "deskew_angle_threshold": self.deskew_angle_threshold,
            "deskew_max_angle": self.deskew_max_angle,
            "contrast_enhancement": self.contrast_enhancement,
            "adaptive_threshold": self.adaptive_threshold,
            "threshold_block_size": self.threshold_block_size,
            "threshold_constant": self.threshold_constant,
            "morphology_enabled": self.morphology_enabled,
            "kernel_size": self.kernel_size,
            "closing_iterations": self.closing_iterations,
            "opening_iterations": self.opening_iterations,
            "target_dpi": self.target_dpi,
            "min_dpi": self.min_dpi,
            "upscale_factor": self.upscale_factor,
            "border_removal": self.border_removal,
            "border_threshold": self.border_threshold,
            "auto_select_best": self.auto_select_best,
            "quality_metrics": self.quality_metrics
        }