"""
Word Error Rate (WER) calculation and validation for OCR quality assessment.
Implements PRD Section 5.3 accuracy targets: <2% WER on scanned, <0.1% on born-digital.
"""

import logging
import re
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from difflib import SequenceMatcher

import structlog
import numpy as np
from jiwer import wer, mer, wil, compute_measures

from .types import OCRError, PageOCRResult, OCRResult, DocumentType


logger = structlog.get_logger(__name__)


@dataclass
class WERMetrics:
    """Word Error Rate metrics and detailed breakdown."""
    
    # Core WER metrics
    wer: float = 0.0
    mer: float = 0.0  # Match Error Rate
    wil: float = 0.0  # Word Information Lost
    
    # Detailed error counts
    substitutions: int = 0
    insertions: int = 0
    deletions: int = 0
    total_words: int = 0
    correct_words: int = 0
    
    # Character-level metrics
    cer: float = 0.0  # Character Error Rate
    char_substitutions: int = 0
    char_insertions: int = 0
    char_deletions: int = 0
    total_chars: int = 0
    
    # Quality indicators
    meets_target: bool = False
    target_wer: float = 0.02
    confidence_correlation: float = 0.0
    
    # Detailed analysis
    error_patterns: Dict[str, int] = None
    problematic_words: List[str] = None
    
    def __post_init__(self):
        if self.error_patterns is None:
            self.error_patterns = {}
        if self.problematic_words is None:
            self.problematic_words = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging/storage."""
        return {
            "wer": self.wer,
            "mer": self.mer, 
            "wil": self.wil,
            "cer": self.cer,
            "substitutions": self.substitutions,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "char_substitutions": self.char_substitutions,
            "char_insertions": self.char_insertions,
            "char_deletions": self.char_deletions,
            "total_words": self.total_words,
            "total_chars": self.total_chars,
            "correct_words": self.correct_words,
            "meets_target": self.meets_target,
            "target_wer": self.target_wer,
            "confidence_correlation": self.confidence_correlation,
            "error_patterns": self.error_patterns,
            "problematic_words": self.problematic_words[:10]  # Limit for storage
        }


class WERCalculator:
    """
    Word Error Rate calculator with comprehensive error analysis.
    
    Provides detailed WER metrics to meet PRD accuracy targets.
    """
    
    def __init__(
        self,
        born_digital_target: float = 0.001,  # <0.1%
        scanned_target: float = 0.02,        # <2%
        case_sensitive: bool = False,
        punctuation_handling: str = "ignore"  # ignore, strict, normalize
    ):
        """
        Initialize WER calculator.
        
        Args:
            born_digital_target: WER target for born-digital documents
            scanned_target: WER target for scanned documents
            case_sensitive: Whether to consider case in comparisons
            punctuation_handling: How to handle punctuation
        """
        self.born_digital_target = born_digital_target
        self.scanned_target = scanned_target
        self.case_sensitive = case_sensitive
        self.punctuation_handling = punctuation_handling
        self.logger = logger.bind(component="WERCalculator")
    
    def calculate_wer(
        self, 
        reference_text: str, 
        hypothesis_text: str,
        document_type: DocumentType = DocumentType.UNKNOWN
    ) -> WERMetrics:
        """
        Calculate comprehensive WER metrics.
        
        Args:
            reference_text: Ground truth text
            hypothesis_text: OCR output text
            document_type: Type of document for target selection
            
        Returns:
            WERMetrics with detailed analysis
            
        Raises:
            OCRError: If calculation fails
        """
        try:
            self.logger.debug("Starting WER calculation",
                            ref_length=len(reference_text),
                            hyp_length=len(hypothesis_text),
                            document_type=document_type.value)
            
            # Normalize texts
            ref_normalized = self._normalize_text(reference_text)
            hyp_normalized = self._normalize_text(hypothesis_text)
            
            # Calculate word-level metrics
            word_metrics = self._calculate_word_metrics(ref_normalized, hyp_normalized)
            
            # Calculate character-level metrics
            char_metrics = self._calculate_character_metrics(ref_normalized, hyp_normalized)
            
            # Determine target WER
            target_wer = self._get_target_wer(document_type)
            
            # Analyze error patterns
            error_patterns = self._analyze_error_patterns(ref_normalized, hyp_normalized)
            problematic_words = self._find_problematic_words(ref_normalized, hyp_normalized)
            
            # Create comprehensive metrics
            wer_metrics = WERMetrics(
                wer=word_metrics["wer"],
                mer=word_metrics["mer"],
                wil=word_metrics["wil"],
                substitutions=word_metrics["substitutions"],
                insertions=word_metrics["insertions"],
                deletions=word_metrics["deletions"],
                total_words=word_metrics["total_words"],
                correct_words=word_metrics["correct_words"],
                cer=char_metrics["cer"],
                char_substitutions=char_metrics["substitutions"],
                char_insertions=char_metrics["insertions"],
                char_deletions=char_metrics["deletions"],
                total_chars=char_metrics["total_chars"],
                meets_target=word_metrics["wer"] <= target_wer,
                target_wer=target_wer,
                error_patterns=error_patterns,
                problematic_words=problematic_words
            )
            
            self.logger.debug("WER calculation completed",
                            wer=wer_metrics.wer,
                            cer=wer_metrics.cer,
                            meets_target=wer_metrics.meets_target)
            
            return wer_metrics
            
        except Exception as e:
            self.logger.error("Error calculating WER", error=str(e), exc_info=True)
            raise OCRError(f"WER calculation failed: {str(e)}")
    
    def calculate_ocr_result_wer(
        self, 
        ocr_result: OCRResult, 
        reference_texts: Dict[int, str]
    ) -> WERMetrics:
        """
        Calculate WER for complete OCR result against reference texts.
        
        Args:
            ocr_result: Complete OCR result
            reference_texts: Dictionary mapping page numbers to reference texts
            
        Returns:
            Aggregated WER metrics
        """
        try:
            all_ref_text = []
            all_hyp_text = []
            
            for page_result in ocr_result.pages:
                page_num = page_result.page_num
                
                if page_num in reference_texts:
                    ref_text = reference_texts[page_num]
                    hyp_text = page_result.full_text
                    
                    all_ref_text.append(ref_text)
                    all_hyp_text.append(hyp_text)
            
            if not all_ref_text:
                raise OCRError("No reference texts provided for WER calculation")
            
            # Combine all text for overall WER
            combined_ref = " ".join(all_ref_text)
            combined_hyp = " ".join(all_hyp_text)
            
            return self.calculate_wer(combined_ref, combined_hyp, ocr_result.document_type)
            
        except Exception as e:
            self.logger.error("Error calculating OCR result WER", error=str(e))
            raise OCRError(f"OCR result WER calculation failed: {str(e)}")
    
    def calculate_page_wer(
        self, 
        page_result: PageOCRResult, 
        reference_text: str
    ) -> WERMetrics:
        """
        Calculate WER for a single page.
        
        Args:
            page_result: Page OCR result
            reference_text: Reference text for the page
            
        Returns:
            WER metrics for the page
        """
        return self.calculate_wer(
            reference_text, 
            page_result.full_text, 
            page_result.document_type
        )
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent comparison."""
        try:
            normalized = text
            
            # Case normalization
            if not self.case_sensitive:
                normalized = normalized.lower()
            
            # Punctuation handling
            if self.punctuation_handling == "ignore":
                # Remove all punctuation
                normalized = re.sub(r'[^\w\s]', ' ', normalized)
            elif self.punctuation_handling == "normalize":
                # Normalize common punctuation
                normalized = re.sub(r'["""]', '"', normalized)
                normalized = re.sub(r'[''']', "'", normalized)
                normalized = re.sub(r'[–—]', '-', normalized)
            
            # Normalize whitespace
            normalized = re.sub(r'\s+', ' ', normalized)
            normalized = normalized.strip()
            
            return normalized
            
        except Exception as e:
            self.logger.warning("Error normalizing text", error=str(e))
            return text
    
    def _calculate_word_metrics(self, reference: str, hypothesis: str) -> Dict[str, Any]:
        """Calculate word-level error metrics."""
        try:
            # Use jiwer library for accurate WER calculation
            measures = compute_measures(reference, hypothesis)
            
            return {
                "wer": measures["wer"],
                "mer": measures["mer"],
                "wil": measures["wil"],
                "substitutions": measures["substitutions"],
                "insertions": measures["insertions"],
                "deletions": measures["deletions"],
                "total_words": len(reference.split()),
                "correct_words": len(reference.split()) - measures["substitutions"] - measures["deletions"]
            }
            
        except Exception as e:
            self.logger.warning("Error calculating word metrics", error=str(e))
            # Fallback calculation
            return self._fallback_word_calculation(reference, hypothesis)
    
    def _fallback_word_calculation(self, reference: str, hypothesis: str) -> Dict[str, Any]:
        """Fallback word-level calculation using basic edit distance."""
        try:
            ref_words = reference.split()
            hyp_words = hypothesis.split()
            
            # Calculate edit distance
            edit_ops = self._calculate_edit_operations(ref_words, hyp_words)
            
            substitutions = sum(1 for op in edit_ops if op[0] == "substitute")
            insertions = sum(1 for op in edit_ops if op[0] == "insert")
            deletions = sum(1 for op in edit_ops if op[0] == "delete")
            
            total_words = len(ref_words)
            total_errors = substitutions + insertions + deletions
            wer = total_errors / max(total_words, 1)
            
            return {
                "wer": wer,
                "mer": wer,  # Simplified
                "wil": wer,  # Simplified
                "substitutions": substitutions,
                "insertions": insertions,
                "deletions": deletions,
                "total_words": total_words,
                "correct_words": total_words - substitutions - deletions
            }
            
        except Exception as e:
            self.logger.error("Error in fallback word calculation", error=str(e))
            return {
                "wer": 1.0,
                "mer": 1.0,
                "wil": 1.0,
                "substitutions": 0,
                "insertions": 0,
                "deletions": 0,
                "total_words": len(reference.split()),
                "correct_words": 0
            }
    
    def _calculate_character_metrics(self, reference: str, hypothesis: str) -> Dict[str, Any]:
        """Calculate character-level error metrics."""
        try:
            ref_chars = list(reference)
            hyp_chars = list(hypothesis)
            
            # Calculate character-level edit distance
            edit_ops = self._calculate_edit_operations(ref_chars, hyp_chars)
            
            substitutions = sum(1 for op in edit_ops if op[0] == "substitute")
            insertions = sum(1 for op in edit_ops if op[0] == "insert")
            deletions = sum(1 for op in edit_ops if op[0] == "delete")
            
            total_chars = len(ref_chars)
            total_errors = substitutions + insertions + deletions
            cer = total_errors / max(total_chars, 1)
            
            return {
                "cer": cer,
                "substitutions": substitutions,
                "insertions": insertions,
                "deletions": deletions,
                "total_chars": total_chars
            }
            
        except Exception as e:
            self.logger.warning("Error calculating character metrics", error=str(e))
            return {
                "cer": 1.0,
                "substitutions": 0,
                "insertions": 0,
                "deletions": 0,
                "total_chars": len(reference)
            }
    
    def _calculate_edit_operations(self, seq1: List[str], seq2: List[str]) -> List[Tuple[str, Any, Any]]:
        """Calculate edit operations between two sequences."""
        try:
            matcher = SequenceMatcher(None, seq1, seq2)
            operations = []
            
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == "replace":
                    # Substitutions
                    for k in range(max(i2 - i1, j2 - j1)):
                        if k < i2 - i1 and k < j2 - j1:
                            operations.append(("substitute", seq1[i1 + k], seq2[j1 + k]))
                        elif k < i2 - i1:
                            operations.append(("delete", seq1[i1 + k], None))
                        else:
                            operations.append(("insert", None, seq2[j1 + k]))
                elif tag == "delete":
                    for k in range(i1, i2):
                        operations.append(("delete", seq1[k], None))
                elif tag == "insert":
                    for k in range(j1, j2):
                        operations.append(("insert", None, seq2[k]))
                # "equal" operations don't count as errors
            
            return operations
            
        except Exception as e:
            self.logger.warning("Error calculating edit operations", error=str(e))
            return []
    
    def _analyze_error_patterns(self, reference: str, hypothesis: str) -> Dict[str, int]:
        """Analyze common error patterns in the OCR output."""
        try:
            ref_words = reference.split()
            hyp_words = hypothesis.split()
            
            error_patterns = {}
            
            # Character-level substitution patterns
            for ref_word, hyp_word in zip(ref_words, hyp_words):
                if ref_word != hyp_word:
                    # Find character substitutions
                    for i, (ref_char, hyp_char) in enumerate(zip(ref_word, hyp_word)):
                        if ref_char != hyp_char:
                            pattern = f"{ref_char}->{hyp_char}"
                            error_patterns[pattern] = error_patterns.get(pattern, 0) + 1
            
            # Common OCR error patterns
            ocr_patterns = {
                "rn->m": 0, "m->rn": 0,  # rn/m confusion
                "cl->d": 0, "d->cl": 0,  # cl/d confusion
                "vv->w": 0, "w->vv": 0,  # vv/w confusion
                "l->I": 0, "I->l": 0,    # l/I confusion
                "0->O": 0, "O->0": 0,    # 0/O confusion
                "1->l": 0, "l->1": 0,    # 1/l confusion
            }
            
            # Check for these patterns in the text
            for pattern in ocr_patterns:
                source, target = pattern.split("->")
                if source in reference and target in hypothesis:
                    ocr_patterns[pattern] += hypothesis.count(target)
            
            error_patterns.update(ocr_patterns)
            
            # Filter out patterns with zero occurrences
            return {k: v for k, v in error_patterns.items() if v > 0}
            
        except Exception as e:
            self.logger.warning("Error analyzing error patterns", error=str(e))
            return {}
    
    def _find_problematic_words(self, reference: str, hypothesis: str) -> List[str]:
        """Find words that are consistently problematic for OCR."""
        try:
            ref_words = reference.split()
            hyp_words = hypothesis.split()
            
            problematic = []
            
            # Align words and find mismatches
            for i, (ref_word, hyp_word) in enumerate(zip(ref_words, hyp_words)):
                if ref_word != hyp_word:
                    # Calculate word similarity
                    similarity = SequenceMatcher(None, ref_word, hyp_word).ratio()
                    if similarity < 0.7:  # Significant difference
                        problematic.append(f"{ref_word}->{hyp_word}")
            
            # Find words that were completely missed (deletions)
            if len(ref_words) > len(hyp_words):
                for i in range(len(hyp_words), len(ref_words)):
                    problematic.append(f"{ref_words[i]}->DELETED")
            
            # Find words that were incorrectly inserted
            if len(hyp_words) > len(ref_words):
                for i in range(len(ref_words), len(hyp_words)):
                    problematic.append(f"INSERTED->{hyp_words[i]}")
            
            # Return most frequent problematic words
            from collections import Counter
            counter = Counter(problematic)
            return [word for word, count in counter.most_common(20)]
            
        except Exception as e:
            self.logger.warning("Error finding problematic words", error=str(e))
            return []
    
    def _get_target_wer(self, document_type: DocumentType) -> float:
        """Get WER target based on document type."""
        if document_type == DocumentType.BORN_DIGITAL:
            return self.born_digital_target
        elif document_type == DocumentType.SCANNED:
            return self.scanned_target
        else:
            # For hybrid or unknown, use scanned target (more conservative)
            return self.scanned_target
    
    def validate_against_target(self, wer_metrics: WERMetrics, document_type: DocumentType) -> bool:
        """Validate WER metrics against target for document type."""
        target = self._get_target_wer(document_type)
        return wer_metrics.wer <= target
    
    def get_quality_assessment(self, wer_metrics: WERMetrics) -> Dict[str, Any]:
        """Get human-readable quality assessment."""
        try:
            assessment = {
                "overall_quality": "excellent",
                "meets_target": wer_metrics.meets_target,
                "recommendations": []
            }
            
            # Determine overall quality
            if wer_metrics.wer <= 0.01:  # ≤1%
                assessment["overall_quality"] = "excellent"
            elif wer_metrics.wer <= 0.05:  # ≤5%
                assessment["overall_quality"] = "good"
            elif wer_metrics.wer <= 0.1:   # ≤10%
                assessment["overall_quality"] = "acceptable"
            else:
                assessment["overall_quality"] = "poor"
            
            # Generate recommendations
            if wer_metrics.wer > wer_metrics.target_wer:
                assessment["recommendations"].append("WER exceeds target - consider preprocessing improvements")
            
            if wer_metrics.cer > 0.05:  # >5% character error rate
                assessment["recommendations"].append("High character error rate - check image quality")
            
            if len(wer_metrics.problematic_words) > 10:
                assessment["recommendations"].append("Many problematic words - consider vocabulary tuning")
            
            # Check for specific error patterns
            if any("rn->m" in pattern or "m->rn" in pattern for pattern in wer_metrics.error_patterns):
                assessment["recommendations"].append("rn/m confusion detected - improve image resolution")
            
            if any("0->O" in pattern or "O->0" in pattern for pattern in wer_metrics.error_patterns):
                assessment["recommendations"].append("Number/letter confusion - consider font-specific training")
            
            return assessment
            
        except Exception as e:
            self.logger.warning("Error creating quality assessment", error=str(e))
            return {
                "overall_quality": "unknown",
                "meets_target": False,
                "recommendations": ["Error in assessment generation"]
            }
    
    def compare_multiple_results(self, results: List[Tuple[str, WERMetrics]]) -> Dict[str, Any]:
        """Compare WER metrics from multiple OCR approaches."""
        try:
            if not results:
                return {"error": "No results to compare"}
            
            comparison = {
                "best_method": "",
                "best_wer": float('inf'),
                "detailed_comparison": [],
                "recommendations": []
            }
            
            for method_name, metrics in results:
                comparison["detailed_comparison"].append({
                    "method": method_name,
                    "wer": metrics.wer,
                    "cer": metrics.cer,
                    "meets_target": metrics.meets_target
                })
                
                if metrics.wer < comparison["best_wer"]:
                    comparison["best_wer"] = metrics.wer
                    comparison["best_method"] = method_name
            
            # Generate recommendations
            if comparison["best_wer"] > 0.05:
                comparison["recommendations"].append("All methods have high WER - consider different preprocessing")
            
            wer_variance = np.var([metrics.wer for _, metrics in results])
            if wer_variance > 0.01:
                comparison["recommendations"].append("High variance between methods - ensemble approach may help")
            
            return comparison
            
        except Exception as e:
            self.logger.error("Error comparing multiple results", error=str(e))
            return {"error": str(e)}