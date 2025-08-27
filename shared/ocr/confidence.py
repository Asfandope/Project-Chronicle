"""
Confidence analysis for OCR results.
Implements character-level confidence scoring and analysis.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import structlog

from .types import (
    OCRResult,
    PageOCRResult,
)

logger = structlog.get_logger(__name__)


@dataclass
class ConfidenceMetrics:
    """Comprehensive confidence metrics for OCR results."""

    # Overall metrics
    average_confidence: float = 0.0
    min_confidence: float = 1.0
    max_confidence: float = 0.0
    confidence_std: float = 0.0

    # Character-level metrics
    total_characters: int = 0
    reliable_chars: int = 0  # >80% confidence
    uncertain_chars: int = 0  # <60% confidence
    reliable_char_ratio: float = 0.0
    uncertain_char_ratio: float = 0.0

    # Word-level metrics
    total_words: int = 0
    reliable_words: int = 0  # >80% confidence
    uncertain_words: int = 0  # <60% confidence
    reliable_word_ratio: float = 0.0
    uncertain_word_ratio: float = 0.0

    # Line-level metrics
    total_lines: int = 0
    reliable_lines: int = 0
    uncertain_lines: int = 0

    # Block-level metrics
    total_blocks: int = 0
    reliable_blocks: int = 0
    uncertain_blocks: int = 0

    # Distribution analysis
    confidence_histogram: Dict[str, int] = field(default_factory=dict)
    confidence_by_position: List[float] = field(default_factory=list)
    confidence_by_length: Dict[int, float] = field(default_factory=dict)

    # Quality indicators
    has_low_confidence_regions: bool = False
    confidence_uniformity: float = 0.0  # How uniform confidence is across document

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "average_confidence": self.average_confidence,
            "min_confidence": self.min_confidence,
            "max_confidence": self.max_confidence,
            "confidence_std": self.confidence_std,
            "total_characters": self.total_characters,
            "reliable_chars": self.reliable_chars,
            "uncertain_chars": self.uncertain_chars,
            "reliable_char_ratio": self.reliable_char_ratio,
            "uncertain_char_ratio": self.uncertain_char_ratio,
            "total_words": self.total_words,
            "reliable_words": self.reliable_words,
            "uncertain_words": self.uncertain_words,
            "reliable_word_ratio": self.reliable_word_ratio,
            "uncertain_word_ratio": self.uncertain_word_ratio,
            "total_lines": self.total_lines,
            "reliable_lines": self.reliable_lines,
            "uncertain_lines": self.uncertain_lines,
            "total_blocks": self.total_blocks,
            "reliable_blocks": self.reliable_blocks,
            "uncertain_blocks": self.uncertain_blocks,
            "confidence_histogram": self.confidence_histogram,
            "has_low_confidence_regions": self.has_low_confidence_regions,
            "confidence_uniformity": self.confidence_uniformity,
        }


class ConfidenceAnalyzer:
    """
    Analyzes confidence scores at character, word, line, and block levels.

    Provides detailed confidence metrics for quality assessment and improvement.
    """

    def __init__(
        self,
        reliable_threshold: float = 0.8,
        uncertain_threshold: float = 0.6,
        histogram_bins: int = 10,
    ):
        """
        Initialize confidence analyzer.

        Args:
            reliable_threshold: Threshold for reliable text (>= this value)
            uncertain_threshold: Threshold for uncertain text (< this value)
            histogram_bins: Number of bins for confidence histogram
        """
        self.reliable_threshold = reliable_threshold
        self.uncertain_threshold = uncertain_threshold
        self.histogram_bins = histogram_bins
        self.logger = logger.bind(component="ConfidenceAnalyzer")

    def analyze_result(self, ocr_result: OCRResult) -> ConfidenceMetrics:
        """
        Analyze confidence for complete OCR result.

        Args:
            ocr_result: Complete OCR result

        Returns:
            Comprehensive confidence metrics
        """
        try:
            self.logger.debug(
                "Starting confidence analysis", pages=len(ocr_result.pages)
            )

            # Collect confidence data from all pages
            all_char_confidences = []
            all_word_confidences = []
            all_line_confidences = []
            all_block_confidences = []

            total_chars = 0
            total_words = 0
            total_lines = 0
            total_blocks = 0

            reliable_chars = 0
            uncertain_chars = 0
            reliable_words = 0
            uncertain_words = 0
            reliable_lines = 0
            uncertain_lines = 0
            reliable_blocks = 0
            uncertain_blocks = 0

            for page_result in ocr_result.pages:
                self._analyze_page(page_result)

                # Aggregate character data
                for block in page_result.text_blocks:
                    all_block_confidences.append(block.confidence)
                    total_blocks += 1

                    if block.confidence >= self.reliable_threshold:
                        reliable_blocks += 1
                    elif block.confidence < self.uncertain_threshold:
                        uncertain_blocks += 1

                    for line in block.lines:
                        all_line_confidences.append(line.confidence)
                        total_lines += 1

                        if line.confidence >= self.reliable_threshold:
                            reliable_lines += 1
                        elif line.confidence < self.uncertain_threshold:
                            uncertain_lines += 1

                        for word in line.words:
                            all_word_confidences.append(word.confidence)
                            total_words += 1

                            if word.confidence >= self.reliable_threshold:
                                reliable_words += 1
                            elif word.confidence < self.uncertain_threshold:
                                uncertain_words += 1

                            for char in word.characters:
                                all_char_confidences.append(char.confidence)
                                total_chars += 1

                                if char.confidence >= self.reliable_threshold:
                                    reliable_chars += 1
                                elif char.confidence < self.uncertain_threshold:
                                    uncertain_chars += 1

            # Calculate overall statistics
            if all_char_confidences:
                avg_confidence = np.mean(all_char_confidences)
                min_confidence = np.min(all_char_confidences)
                max_confidence = np.max(all_char_confidences)
                confidence_std = np.std(all_char_confidences)
            else:
                avg_confidence = min_confidence = max_confidence = confidence_std = 0.0

            # Calculate ratios
            reliable_char_ratio = reliable_chars / max(total_chars, 1)
            uncertain_char_ratio = uncertain_chars / max(total_chars, 1)
            reliable_word_ratio = reliable_words / max(total_words, 1)
            uncertain_word_ratio = uncertain_words / max(total_words, 1)

            # Create confidence histogram
            confidence_histogram = self._create_confidence_histogram(
                all_char_confidences
            )

            # Analyze confidence distribution
            confidence_uniformity = self._calculate_confidence_uniformity(
                all_char_confidences
            )
            has_low_confidence_regions = (
                uncertain_char_ratio > 0.1
            )  # >10% uncertain characters

            # Position analysis
            confidence_by_position = self._analyze_confidence_by_position(ocr_result)

            # Length analysis
            confidence_by_length = self._analyze_confidence_by_length(ocr_result)

            metrics = ConfidenceMetrics(
                average_confidence=avg_confidence,
                min_confidence=min_confidence,
                max_confidence=max_confidence,
                confidence_std=confidence_std,
                total_characters=total_chars,
                reliable_chars=reliable_chars,
                uncertain_chars=uncertain_chars,
                reliable_char_ratio=reliable_char_ratio,
                uncertain_char_ratio=uncertain_char_ratio,
                total_words=total_words,
                reliable_words=reliable_words,
                uncertain_words=uncertain_words,
                reliable_word_ratio=reliable_word_ratio,
                uncertain_word_ratio=uncertain_word_ratio,
                total_lines=total_lines,
                reliable_lines=reliable_lines,
                uncertain_lines=uncertain_lines,
                total_blocks=total_blocks,
                reliable_blocks=reliable_blocks,
                uncertain_blocks=uncertain_blocks,
                confidence_histogram=confidence_histogram,
                confidence_by_position=confidence_by_position,
                confidence_by_length=confidence_by_length,
                has_low_confidence_regions=has_low_confidence_regions,
                confidence_uniformity=confidence_uniformity,
            )

            self.logger.debug(
                "Confidence analysis completed",
                avg_confidence=avg_confidence,
                reliable_char_ratio=reliable_char_ratio,
                uncertain_char_ratio=uncertain_char_ratio,
            )

            return metrics

        except Exception as e:
            self.logger.error(
                "Error in confidence analysis", error=str(e), exc_info=True
            )
            return ConfidenceMetrics()

    def _analyze_page(self, page_result: PageOCRResult) -> Dict[str, Any]:
        """Analyze confidence for a single page."""
        try:
            char_confidences = []

            for block in page_result.text_blocks:
                for line in block.lines:
                    for word in line.words:
                        for char in word.characters:
                            char_confidences.append(char.confidence)

            if char_confidences:
                return {
                    "page_num": page_result.page_num,
                    "avg_confidence": np.mean(char_confidences),
                    "min_confidence": np.min(char_confidences),
                    "max_confidence": np.max(char_confidences),
                    "char_count": len(char_confidences),
                    "reliable_chars": sum(
                        1 for c in char_confidences if c >= self.reliable_threshold
                    ),
                    "uncertain_chars": sum(
                        1 for c in char_confidences if c < self.uncertain_threshold
                    ),
                }
            else:
                return {
                    "page_num": page_result.page_num,
                    "avg_confidence": 0.0,
                    "min_confidence": 0.0,
                    "max_confidence": 0.0,
                    "char_count": 0,
                    "reliable_chars": 0,
                    "uncertain_chars": 0,
                }

        except Exception as e:
            self.logger.warning(
                "Error analyzing page confidence",
                page_num=page_result.page_num,
                error=str(e),
            )
            return {}

    def _create_confidence_histogram(self, confidences: List[float]) -> Dict[str, int]:
        """Create confidence histogram."""
        try:
            if not confidences:
                return {}

            # Create bins
            bins = np.linspace(0, 1, self.histogram_bins + 1)
            hist, _ = np.histogram(confidences, bins=bins)

            # Convert to dictionary with range labels
            histogram = {}
            for i, count in enumerate(hist):
                bin_start = bins[i]
                bin_end = bins[i + 1]
                label = f"{bin_start:.1f}-{bin_end:.1f}"
                histogram[label] = int(count)

            return histogram

        except Exception as e:
            self.logger.warning("Error creating confidence histogram", error=str(e))
            return {}

    def _calculate_confidence_uniformity(self, confidences: List[float]) -> float:
        """Calculate how uniform confidence is across the document."""
        try:
            if not confidences or len(confidences) < 2:
                return 1.0

            # Use coefficient of variation (inverted for uniformity)
            mean_conf = np.mean(confidences)
            std_conf = np.std(confidences)

            if mean_conf == 0:
                return 0.0

            cv = std_conf / mean_conf
            uniformity = max(0.0, 1.0 - cv)  # Higher uniformity = lower variation

            return uniformity

        except Exception as e:
            self.logger.warning("Error calculating confidence uniformity", error=str(e))
            return 0.5

    def _analyze_confidence_by_position(self, ocr_result: OCRResult) -> List[float]:
        """Analyze confidence distribution by position in document."""
        try:
            position_confidences = []

            for page_result in ocr_result.pages:
                page_confidences = []

                for block in page_result.text_blocks:
                    for line in block.lines:
                        for word in line.words:
                            page_confidences.extend(
                                [char.confidence for char in word.characters]
                            )

                if page_confidences:
                    position_confidences.append(np.mean(page_confidences))
                else:
                    position_confidences.append(0.0)

            return position_confidences

        except Exception as e:
            self.logger.warning("Error analyzing confidence by position", error=str(e))
            return []

    def _analyze_confidence_by_length(self, ocr_result: OCRResult) -> Dict[int, float]:
        """Analyze confidence by word length."""
        try:
            length_confidences = {}

            for page_result in ocr_result.pages:
                for block in page_result.text_blocks:
                    for line in block.lines:
                        for word in line.words:
                            word_length = len(word.characters)

                            if word_length not in length_confidences:
                                length_confidences[word_length] = []

                            length_confidences[word_length].append(word.confidence)

            # Calculate average confidence for each length
            avg_by_length = {}
            for length, confidences in length_confidences.items():
                avg_by_length[length] = np.mean(confidences)

            return avg_by_length

        except Exception as e:
            self.logger.warning("Error analyzing confidence by length", error=str(e))
            return {}

    def identify_low_confidence_regions(
        self, ocr_result: OCRResult
    ) -> List[Dict[str, Any]]:
        """Identify regions of low confidence text."""
        try:
            low_confidence_regions = []

            for page_result in ocr_result.pages:
                for block_idx, block in enumerate(page_result.text_blocks):
                    if block.confidence < self.uncertain_threshold:
                        region = {
                            "page_num": page_result.page_num,
                            "block_index": block_idx,
                            "text_preview": block.text[:100] + "..."
                            if len(block.text) > 100
                            else block.text,
                            "confidence": block.confidence,
                            "bbox": block.bbox,
                            "line_count": len(block.lines),
                            "word_count": sum(len(line.words) for line in block.lines),
                            "char_count": sum(
                                len(word.characters)
                                for line in block.lines
                                for word in line.words
                            ),
                        }
                        low_confidence_regions.append(region)

            # Sort by confidence (lowest first)
            low_confidence_regions.sort(key=lambda x: x["confidence"])

            return low_confidence_regions

        except Exception as e:
            self.logger.error("Error identifying low confidence regions", error=str(e))
            return []

    def analyze_confidence_patterns(self, ocr_result: OCRResult) -> Dict[str, Any]:
        """Analyze patterns in confidence distribution."""
        try:
            patterns = {
                "confidence_trends": {},
                "problematic_characters": {},
                "confidence_by_font_size": {},
                "spatial_patterns": {},
            }

            # Character-level analysis
            char_confidence_map = {}

            for page_result in ocr_result.pages:
                for block in page_result.text_blocks:
                    for line in block.lines:
                        for word in line.words:
                            for char in word.characters:
                                char_text = char.char
                                if char_text not in char_confidence_map:
                                    char_confidence_map[char_text] = []
                                char_confidence_map[char_text].append(char.confidence)

            # Find problematic characters
            for char, confidences in char_confidence_map.items():
                avg_conf = np.mean(confidences)
                if (
                    avg_conf < self.uncertain_threshold and len(confidences) >= 5
                ):  # At least 5 occurrences
                    patterns["problematic_characters"][char] = {
                        "avg_confidence": avg_conf,
                        "occurrences": len(confidences),
                        "std": np.std(confidences),
                    }

            # Spatial pattern analysis
            patterns["spatial_patterns"] = self._analyze_spatial_confidence_patterns(
                ocr_result
            )

            return patterns

        except Exception as e:
            self.logger.error("Error analyzing confidence patterns", error=str(e))
            return {}

    def _analyze_spatial_confidence_patterns(
        self, ocr_result: OCRResult
    ) -> Dict[str, Any]:
        """Analyze spatial patterns in confidence."""
        try:
            spatial_patterns = {
                "top_region_avg": 0.0,
                "middle_region_avg": 0.0,
                "bottom_region_avg": 0.0,
                "left_region_avg": 0.0,
                "right_region_avg": 0.0,
                "center_region_avg": 0.0,
            }

            all_blocks = []
            page_heights = []
            page_widths = []

            # Collect all blocks with positions
            for page_result in ocr_result.pages:
                # Estimate page dimensions from block positions
                if page_result.text_blocks:
                    max_y = max(block.bbox[3] for block in page_result.text_blocks)
                    max_x = max(block.bbox[2] for block in page_result.text_blocks)
                    page_heights.append(max_y)
                    page_widths.append(max_x)

                    for block in page_result.text_blocks:
                        all_blocks.append(
                            {
                                "confidence": block.confidence,
                                "bbox": block.bbox,
                                "page_height": max_y,
                                "page_width": max_x,
                            }
                        )

            if not all_blocks:
                return spatial_patterns

            # Analyze by vertical regions
            top_blocks = [
                b for b in all_blocks if b["bbox"][1] < b["page_height"] * 0.33
            ]
            middle_blocks = [
                b
                for b in all_blocks
                if 0.33 * b["page_height"] <= b["bbox"][1] <= 0.67 * b["page_height"]
            ]
            bottom_blocks = [
                b for b in all_blocks if b["bbox"][1] > b["page_height"] * 0.67
            ]

            if top_blocks:
                spatial_patterns["top_region_avg"] = np.mean(
                    [b["confidence"] for b in top_blocks]
                )
            if middle_blocks:
                spatial_patterns["middle_region_avg"] = np.mean(
                    [b["confidence"] for b in middle_blocks]
                )
            if bottom_blocks:
                spatial_patterns["bottom_region_avg"] = np.mean(
                    [b["confidence"] for b in bottom_blocks]
                )

            # Analyze by horizontal regions
            left_blocks = [
                b for b in all_blocks if b["bbox"][0] < b["page_width"] * 0.33
            ]
            right_blocks = [
                b for b in all_blocks if b["bbox"][0] > b["page_width"] * 0.67
            ]
            center_blocks = [
                b
                for b in all_blocks
                if 0.33 * b["page_width"] <= b["bbox"][0] <= 0.67 * b["page_width"]
            ]

            if left_blocks:
                spatial_patterns["left_region_avg"] = np.mean(
                    [b["confidence"] for b in left_blocks]
                )
            if right_blocks:
                spatial_patterns["right_region_avg"] = np.mean(
                    [b["confidence"] for b in right_blocks]
                )
            if center_blocks:
                spatial_patterns["center_region_avg"] = np.mean(
                    [b["confidence"] for b in center_blocks]
                )

            return spatial_patterns

        except Exception as e:
            self.logger.warning(
                "Error analyzing spatial confidence patterns", error=str(e)
            )
            return {}

    def get_confidence_report(self, metrics: ConfidenceMetrics) -> Dict[str, Any]:
        """Generate human-readable confidence report."""
        try:
            report = {
                "overall_assessment": "good",
                "key_findings": [],
                "recommendations": [],
                "detailed_metrics": metrics.to_dict(),
            }

            # Overall assessment
            if metrics.average_confidence >= 0.9:
                report["overall_assessment"] = "excellent"
            elif metrics.average_confidence >= 0.8:
                report["overall_assessment"] = "good"
            elif metrics.average_confidence >= 0.7:
                report["overall_assessment"] = "acceptable"
            else:
                report["overall_assessment"] = "poor"

            # Key findings
            if metrics.reliable_char_ratio > 0.9:
                report["key_findings"].append(
                    f"High reliability: {metrics.reliable_char_ratio:.1%} of characters are reliable"
                )

            if metrics.uncertain_char_ratio > 0.1:
                report["key_findings"].append(
                    f"Uncertainty concern: {metrics.uncertain_char_ratio:.1%} of characters are uncertain"
                )

            if metrics.confidence_std > 0.2:
                report["key_findings"].append(
                    "High confidence variation across document"
                )

            if metrics.has_low_confidence_regions:
                report["key_findings"].append(
                    "Document contains low-confidence regions"
                )

            # Recommendations
            if metrics.uncertain_char_ratio > 0.15:
                report["recommendations"].append(
                    "Consider preprocessing improvements to reduce uncertainty"
                )

            if metrics.confidence_uniformity < 0.7:
                report["recommendations"].append(
                    "Inconsistent confidence suggests varying image quality"
                )

            if metrics.average_confidence < 0.8:
                report["recommendations"].append(
                    "Overall confidence is low - review OCR settings"
                )

            return report

        except Exception as e:
            self.logger.error("Error generating confidence report", error=str(e))
            return {"error": str(e)}
