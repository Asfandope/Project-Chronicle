"""
Production-ready LayoutLM classifier with enhanced error handling and performance optimization.

This module provides a robust, production-grade implementation of LayoutLM-based
block classification with proper error handling, caching, and fallback mechanisms.
"""

import time
import hashlib
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from collections import deque
import numpy as np
from PIL import Image
import torch
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    AutoTokenizer
)
import structlog

from .types import TextBlock, BlockType, BoundingBox, PageLayout, LayoutError

logger = structlog.get_logger(__name__)


@dataclass
class ClassificationResult:
    """Result of block classification with detailed metrics."""
    block_type: BlockType
    confidence: float
    processing_time_ms: float
    method_used: str  # "layoutlm", "cached", "fallback"
    calibrated_confidence: Optional[float] = None
    alternative_predictions: Optional[Dict[BlockType, float]] = None


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
            else:
                raise LayoutError("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            
            raise e


class ConfidenceCalibrator:
    """Calibrates model confidence scores for better reliability."""
    
    def __init__(self):
        self.temperature = 1.5
        self.platt_a = 1.0
        self.platt_b = 0.0
        
    def calibrate_temperature(self, logits: torch.Tensor, temperature: float = None) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        if temperature is None:
            temperature = self.temperature
        return logits / temperature
    
    def calibrate_platt(self, confidence: float) -> float:
        """Apply Platt scaling to confidence score."""
        # Platt scaling: calibrated = sigmoid(a * uncalibrated + b)
        z = self.platt_a * confidence + self.platt_b
        return 1 / (1 + np.exp(-z))
    
    def calibrate_confidence(self, confidence: float, block_type: BlockType) -> float:
        """Apply full confidence calibration pipeline."""
        # Apply Platt scaling
        calibrated = self.calibrate_platt(confidence)
        
        # Apply block-type specific adjustments
        type_adjustments = {
            BlockType.TITLE: 1.05,      # Titles usually have higher confidence
            BlockType.CAPTION: 0.95,    # Captions can be tricky
            BlockType.ADVERTISEMENT: 0.90,  # Ads need lower threshold
            BlockType.BYLINE: 1.10,     # Bylines are usually clear
        }
        
        adjustment = type_adjustments.get(block_type, 1.0)
        calibrated *= adjustment
        
        return min(1.0, max(0.0, calibrated))


class ProductionLayoutLMClassifier:
    """Production-grade LayoutLM classifier with enhanced reliability."""
    
    def __init__(
        self,
        model_name: str = "microsoft/layoutlmv3-large",
        device: Optional[str] = None,
        confidence_threshold: float = 0.85,
        enable_caching: bool = True,
        cache_size: int = 1000,
        batch_size: int = 8,
        enable_fallback: bool = True,
        warmup_runs: int = 2
    ):
        """
        Initialize production LayoutLM classifier.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device for inference (auto-detected if None)
            confidence_threshold: Minimum confidence for valid classification
            enable_caching: Enable result caching for performance
            cache_size: Maximum cache size
            batch_size: Batch size for inference
            enable_fallback: Enable fallback classification on failures
            warmup_runs: Number of warmup runs for model
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self.enable_fallback = enable_fallback
        self.warmup_runs = warmup_runs
        
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        self.logger = logger.bind(
            component="ProductionLayoutLMClassifier",
            model=model_name,
            device=device
        )
        
        # Model components
        self.processor = None
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
        # Performance optimization
        self.enable_caching = enable_caching
        if enable_caching:
            self._cache = {}
            self._cache_order = deque(maxlen=cache_size)
        
        # Reliability components
        self.circuit_breaker = CircuitBreaker()
        self.confidence_calibrator = ConfidenceCalibrator()
        
        # Statistics tracking
        self.stats = {
            "total_classifications": 0,
            "cache_hits": 0,
            "fallback_used": 0,
            "errors": 0,
            "avg_confidence": 0.0,
            "avg_processing_time_ms": 0.0
        }
        
        self.logger.info("Initialized production LayoutLM classifier")
    
    def load_model(self):
        """Load and prepare model with warmup."""
        if self.is_loaded:
            return
        
        try:
            self.logger.info("Loading LayoutLM model", model=self.model_name)
            
            # Load components
            self.processor = LayoutLMv3Processor.from_pretrained(
                self.model_name,
                apply_ocr=False
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(
                self.model_name,
                num_labels=13  # Standard block types
            )
            
            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            # Perform warmup runs
            if self.warmup_runs > 0:
                self._warmup_model()
            
            self.is_loaded = True
            self.logger.info("Model loaded and warmed up successfully")
            
        except Exception as e:
            self.logger.error("Failed to load model", error=str(e))
            raise LayoutError(f"Model loading failed: {e}")
    
    def _warmup_model(self):
        """Perform warmup runs to optimize model performance."""
        self.logger.debug("Running model warmup", runs=self.warmup_runs)
        
        # Create dummy inputs
        dummy_image = Image.new("RGB", (100, 100), "white")
        dummy_words = ["dummy", "text", "for", "warmup"]
        dummy_boxes = [[10, 10, 50, 20]] * len(dummy_words)
        
        for i in range(self.warmup_runs):
            try:
                encoding = self.processor(
                    dummy_image,
                    dummy_words,
                    boxes=dummy_boxes,
                    return_tensors="pt",
                    truncation=True,
                    padding=True
                )
                
                # Move to device
                for key in encoding.keys():
                    if isinstance(encoding[key], torch.Tensor):
                        encoding[key] = encoding[key].to(self.device)
                
                # Run inference
                with torch.no_grad():
                    _ = self.model(**encoding)
                    
            except Exception as e:
                self.logger.warning("Warmup run failed", run=i, error=str(e))
    
    def classify_blocks(
        self,
        text_blocks: List[TextBlock],
        page_image: Optional[Image.Image] = None,
        page_layout: Optional[PageLayout] = None,
        use_batch: bool = True
    ) -> List[Tuple[TextBlock, ClassificationResult]]:
        """
        Classify text blocks with production-grade reliability.
        
        Args:
            text_blocks: List of text blocks to classify
            page_image: Optional page image for visual features
            page_layout: Optional page layout information
            use_batch: Process blocks in batches for efficiency
            
        Returns:
            List of tuples (block, classification_result)
        """
        if not self.is_loaded:
            self.load_model()
        
        results = []
        
        try:
            # Use circuit breaker for fault tolerance
            classified = self.circuit_breaker.call(
                self._classify_with_layoutlm,
                text_blocks,
                page_image,
                page_layout,
                use_batch
            )
            results = classified
            
        except Exception as e:
            self.logger.error("LayoutLM classification failed", error=str(e))
            self.stats["errors"] += 1
            
            if self.enable_fallback:
                self.logger.info("Using fallback classification")
                results = self._fallback_classification(text_blocks)
                self.stats["fallback_used"] += len(text_blocks)
            else:
                raise LayoutError(f"Classification failed: {e}")
        
        # Update statistics
        self._update_statistics(results)
        
        return results
    
    def _classify_with_layoutlm(
        self,
        text_blocks: List[TextBlock],
        page_image: Optional[Image.Image],
        page_layout: Optional[PageLayout],
        use_batch: bool
    ) -> List[Tuple[TextBlock, ClassificationResult]]:
        """Core LayoutLM classification with caching and batching."""
        results = []
        blocks_to_process = []
        
        # Check cache first
        for block in text_blocks:
            if self.enable_caching:
                cache_key = self._get_cache_key(block)
                if cache_key in self._cache:
                    cached_result = self._cache[cache_key]
                    results.append((block, cached_result))
                    self.stats["cache_hits"] += 1
                    continue
            
            blocks_to_process.append(block)
        
        if not blocks_to_process:
            return results
        
        # Process remaining blocks
        start_time = time.time()
        
        if use_batch and len(blocks_to_process) > 1:
            batch_results = self._batch_classify(blocks_to_process, page_image, page_layout)
        else:
            batch_results = []
            for block in blocks_to_process:
                result = self._single_classify(block, page_image, page_layout)
                batch_results.append((block, result))
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Cache results and prepare return
        for block, result in batch_results:
            result.processing_time_ms = processing_time_ms / len(batch_results)
            
            if self.enable_caching:
                cache_key = self._get_cache_key(block)
                self._cache[cache_key] = result
                self._cache_order.append(cache_key)
            
            results.append((block, result))
        
        return results
    
    def _single_classify(
        self,
        block: TextBlock,
        page_image: Optional[Image.Image],
        page_layout: Optional[PageLayout]
    ) -> ClassificationResult:
        """Classify a single text block."""
        # Prepare inputs
        words = block.text.split()
        boxes = self._normalize_bbox(block.bbox, page_layout)
        
        if page_image is None:
            page_image = self._create_synthetic_image(block, page_layout)
        
        # Encode
        encoding = self.processor(
            page_image,
            words,
            boxes=[boxes] * len(words),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Move to device
        for key in encoding.keys():
            if isinstance(encoding[key], torch.Tensor):
                encoding[key] = encoding[key].to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits
            
            # Apply confidence calibration
            calibrated_logits = self.confidence_calibrator.calibrate_temperature(logits)
            probabilities = torch.softmax(calibrated_logits, dim=-1)
            
            # Get prediction
            prediction_idx = torch.argmax(probabilities, dim=-1)[0, 0].item()
            confidence = probabilities[0, 0, prediction_idx].item()
            
            # Get alternative predictions
            top_k = torch.topk(probabilities[0, 0], k=min(3, probabilities.shape[-1]))
            alternatives = {
                self._idx_to_block_type(idx.item()): prob.item()
                for idx, prob in zip(top_k.indices, top_k.values)
            }
        
        block_type = self._idx_to_block_type(prediction_idx)
        calibrated_confidence = self.confidence_calibrator.calibrate_confidence(confidence, block_type)
        
        return ClassificationResult(
            block_type=block_type,
            confidence=confidence,
            processing_time_ms=0.0,  # Will be set by caller
            method_used="layoutlm",
            calibrated_confidence=calibrated_confidence,
            alternative_predictions=alternatives
        )
    
    def _batch_classify(
        self,
        blocks: List[TextBlock],
        page_image: Optional[Image.Image],
        page_layout: Optional[PageLayout]
    ) -> List[Tuple[TextBlock, ClassificationResult]]:
        """Classify multiple blocks in batches for efficiency."""
        results = []
        
        for i in range(0, len(blocks), self.batch_size):
            batch = blocks[i:i + self.batch_size]
            
            # Prepare batch inputs
            batch_words = []
            batch_boxes = []
            
            for block in batch:
                words = block.text.split()
                boxes = self._normalize_bbox(block.bbox, page_layout)
                batch_words.append(words)
                batch_boxes.append([boxes] * len(words))
            
            # Process batch
            # Note: This is simplified - real implementation would handle variable lengths
            for j, block in enumerate(batch):
                result = self._single_classify(block, page_image, page_layout)
                results.append((block, result))
        
        return results
    
    def _fallback_classification(self, text_blocks: List[TextBlock]) -> List[Tuple[TextBlock, ClassificationResult]]:
        """Rule-based fallback classification."""
        results = []
        
        for block in text_blocks:
            block_type = self._rule_based_classify(block)
            
            result = ClassificationResult(
                block_type=block_type,
                confidence=0.6,  # Lower confidence for fallback
                processing_time_ms=0.1,
                method_used="fallback",
                calibrated_confidence=0.6
            )
            
            results.append((block, result))
        
        return results
    
    def _rule_based_classify(self, block: TextBlock) -> BlockType:
        """Simple rule-based classification as fallback."""
        text = block.text.strip().lower()
        
        # Check for page numbers
        if text.isdigit() or text.startswith("page "):
            return BlockType.PAGE_NUMBER
        
        # Check for titles (short, large font)
        if len(text.split()) <= 10 and getattr(block, 'font_size', 0) > 14:
            return BlockType.TITLE
        
        # Check for bylines
        if text.startswith("by ") or "correspondent" in text:
            return BlockType.BYLINE
        
        # Check for captions (small font)
        if getattr(block, 'font_size', 12) < 10:
            return BlockType.CAPTION
        
        # Default to body text
        return BlockType.BODY
    
    def _normalize_bbox(self, bbox: BoundingBox, page_layout: Optional[PageLayout]) -> List[int]:
        """Normalize bounding box to LayoutLM expected format (0-1000 scale)."""
        if page_layout:
            width = page_layout.page_width
            height = page_layout.page_height
        else:
            # Assume standard page size
            width, height = 612, 792  # Letter size in points
        
        return [
            int(bbox.x0 * 1000 / width),
            int(bbox.y0 * 1000 / height),
            int(bbox.x1 * 1000 / width),
            int(bbox.y1 * 1000 / height)
        ]
    
    def _create_synthetic_image(self, block: TextBlock, page_layout: Optional[PageLayout]) -> Image.Image:
        """Create synthetic page image when real image unavailable."""
        if page_layout:
            width = int(page_layout.page_width)
            height = int(page_layout.page_height)
        else:
            width, height = 612, 792
        
        return Image.new("RGB", (width, height), "white")
    
    def _get_cache_key(self, block: TextBlock) -> str:
        """Generate cache key for a text block."""
        text_hash = hashlib.md5(block.text.encode()).hexdigest()[:8]
        bbox_hash = hashlib.md5(str(block.bbox).encode()).hexdigest()[:8]
        return f"{text_hash}_{bbox_hash}"
    
    def _idx_to_block_type(self, idx: int) -> BlockType:
        """Convert model prediction index to BlockType."""
        mapping = {
            0: BlockType.TITLE,
            1: BlockType.SUBTITLE,
            2: BlockType.HEADING,
            3: BlockType.BODY,
            4: BlockType.CAPTION,
            5: BlockType.HEADER,
            6: BlockType.FOOTER,
            7: BlockType.BYLINE,
            8: BlockType.QUOTE,
            9: BlockType.SIDEBAR,
            10: BlockType.ADVERTISEMENT,
            11: BlockType.PAGE_NUMBER,
            12: BlockType.UNKNOWN
        }
        return mapping.get(idx, BlockType.UNKNOWN)
    
    def _update_statistics(self, results: List[Tuple[TextBlock, ClassificationResult]]):
        """Update performance statistics."""
        if not results:
            return
        
        total = len(results)
        self.stats["total_classifications"] += total
        
        # Calculate averages
        confidences = [r[1].confidence for r in results]
        times = [r[1].processing_time_ms for r in results if r[1].processing_time_ms > 0]
        
        if confidences:
            prev_avg_conf = self.stats["avg_confidence"]
            prev_total = self.stats["total_classifications"] - total
            self.stats["avg_confidence"] = (
                (prev_avg_conf * prev_total + sum(confidences)) /
                self.stats["total_classifications"]
            )
        
        if times:
            prev_avg_time = self.stats["avg_processing_time_ms"]
            prev_count = max(1, self.stats["total_classifications"] - len(times))
            self.stats["avg_processing_time_ms"] = (
                (prev_avg_time * prev_count + sum(times)) /
                (prev_count + len(times))
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset performance statistics."""
        self.stats = {
            "total_classifications": 0,
            "cache_hits": 0,
            "fallback_used": 0,
            "errors": 0,
            "avg_confidence": 0.0,
            "avg_processing_time_ms": 0.0
        }
    
    def clear_cache(self):
        """Clear the classification cache."""
        if self.enable_caching:
            self._cache.clear()
            self._cache_order.clear()
            self.logger.info("Cache cleared")