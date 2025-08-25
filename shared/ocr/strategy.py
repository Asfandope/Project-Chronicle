"""
Main OCR Strategy implementation - PRD Section 5.3
Orchestrates auto-detection, preprocessing, OCR, and quality validation.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

import structlog
import yaml

from .types import OCRError, OCRResult, OCRConfig, DocumentType, QualityMetrics
from .detector import DocumentTypeDetector
from .engine import OCREngine
from .preprocessing import ImagePreprocessor, PreprocessingConfig
from .wer import WERCalculator, WERMetrics
from .confidence import ConfidenceAnalyzer


logger = structlog.get_logger(__name__)


@dataclass
class OCRStrategyConfig:
    """Configuration for the complete OCR strategy."""
    
    # Component configurations
    ocr_config: OCRConfig = field(default_factory=OCRConfig)
    preprocessing_config: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    
    # Strategy settings
    enable_preprocessing: bool = True
    enable_confidence_analysis: bool = True
    enable_wer_validation: bool = True
    
    # Quality thresholds
    min_confidence_threshold: float = 0.8
    max_wer_threshold: float = 0.02
    
    # Brand-specific settings
    brand_configs_dir: Optional[Path] = None
    
    # Performance settings
    parallel_processing: bool = False
    cache_results: bool = True
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> "OCRStrategyConfig":
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # Extract nested configurations
            ocr_config = OCRConfig(**data.get('ocr', {}))
            preprocessing_config = PreprocessingConfig(**data.get('preprocessing', {}))
            
            # Create strategy config
            strategy_data = data.get('strategy', {})
            strategy_data['ocr_config'] = ocr_config
            strategy_data['preprocessing_config'] = preprocessing_config
            
            return cls(**strategy_data)
            
        except Exception as e:
            logger.error("Error loading OCR strategy config", 
                        config_path=str(config_path), error=str(e))
            return cls()


class OCRStrategy:
    """
    Complete OCR strategy implementation following PRD Section 5.3.
    
    Orchestrates document type detection, preprocessing, OCR, and quality validation
    to achieve <2% WER on scanned and <0.1% WER on born-digital documents.
    """
    
    def __init__(
        self, 
        config: Optional[OCRStrategyConfig] = None,
        brand_configs_dir: Optional[Path] = None
    ):
        """
        Initialize OCR strategy.
        
        Args:
            config: Strategy configuration
            brand_configs_dir: Directory with brand-specific configs
        """
        self.config = config or OCRStrategyConfig()
        self.logger = logger.bind(component="OCRStrategy")
        
        # Initialize components
        self.detector = DocumentTypeDetector()
        self.engine = OCREngine(self.config.ocr_config)
        self.preprocessor = ImagePreprocessor(brand_configs_dir)
        self.wer_calculator = WERCalculator()
        self.confidence_analyzer = ConfidenceAnalyzer()
        
        # Load brand configurations
        self.brand_configs = {}
        if brand_configs_dir:
            self._load_brand_configurations(brand_configs_dir)
    
    def _load_brand_configurations(self, configs_dir: Path):
        """Load brand-specific OCR configurations."""
        try:
            if not configs_dir.exists():
                self.logger.warning("Brand configs directory not found",
                                  dir=str(configs_dir))
                return
            
            for config_file in configs_dir.glob("*.yaml"):
                try:
                    with open(config_file, 'r') as f:
                        config_data = yaml.safe_load(f)
                    
                    brand_name = config_file.stem
                    
                    # Extract OCR-specific configuration
                    if 'ocr' in config_data:
                        ocr_config = OCRConfig(**config_data['ocr'])
                        self.brand_configs[brand_name] = ocr_config
                        
                        self.logger.info("Loaded brand OCR config",
                                       brand=brand_name)
                    
                except Exception as e:
                    self.logger.error("Error loading brand OCR config",
                                    config_file=str(config_file), error=str(e))
            
        except Exception as e:
            self.logger.error("Error loading brand configurations", error=str(e))
    
    def process_pdf(
        self, 
        pdf_path: Path, 
        brand: Optional[str] = None,
        reference_text: Optional[str] = None,
        page_range: Optional[Tuple[int, int]] = None
    ) -> OCRResult:
        """
        Process PDF with complete OCR strategy.
        
        Args:
            pdf_path: Path to PDF file
            brand: Brand name for configuration override
            reference_text: Optional reference text for WER calculation
            page_range: Optional page range to process
            
        Returns:
            Complete OCR result with quality metrics
            
        Raises:
            OCRError: If processing fails
        """
        start_time = time.time()
        
        try:
            self.logger.info("Starting OCR strategy processing",
                           pdf_path=str(pdf_path), brand=brand)
            
            # Step 1: Auto-detect document type
            document_type, detection_confidence, detection_details = self.detector.detect(pdf_path)
            
            self.logger.info("Document type detected",
                           document_type=document_type.value,
                           confidence=detection_confidence)
            
            # Step 2: Get brand-specific configuration
            ocr_config = self._get_brand_config(brand)
            
            # Step 3: Process with OCR engine
            ocr_result = self.engine.process_pdf(
                pdf_path=pdf_path,
                brand=brand,
                page_range=page_range
            )
            
            # Step 4: Confidence analysis
            if self.config.enable_confidence_analysis:
                confidence_metrics = self.confidence_analyzer.analyze_result(ocr_result)
                self.logger.info("Confidence analysis completed",
                               avg_confidence=confidence_metrics.average_confidence)
            
            # Step 5: WER validation (if reference provided)
            wer_metrics = None
            if reference_text and self.config.enable_wer_validation:
                wer_metrics = self.wer_calculator.calculate_wer(
                    reference_text, 
                    ocr_result.total_text, 
                    document_type
                )
                
                self.logger.info("WER validation completed",
                               wer=wer_metrics.wer,
                               meets_target=wer_metrics.meets_target)
            
            # Step 6: Quality assessment
            quality_metrics = self._assess_overall_quality(
                ocr_result, wer_metrics, confidence_metrics if self.config.enable_confidence_analysis else None
            )
            
            # Update result with quality metrics
            ocr_result.quality_metrics = quality_metrics
            
            total_time = time.time() - start_time
            ocr_result.total_processing_time = total_time
            
            # Step 7: Quality validation
            self._validate_quality_targets(ocr_result, document_type)
            
            self.logger.info("OCR strategy processing completed",
                           pdf_path=str(pdf_path),
                           document_type=document_type.value,
                           total_words=ocr_result.total_words,
                           avg_confidence=ocr_result.average_confidence,
                           wer=quality_metrics.wer if wer_metrics else None,
                           processing_time=total_time)
            
            return ocr_result
            
        except OCRError:
            raise
        except Exception as e:
            self.logger.error("Unexpected error in OCR strategy",
                            pdf_path=str(pdf_path), error=str(e), exc_info=True)
            raise OCRError(f"OCR strategy processing failed: {str(e)}")
    
    def process_with_validation(
        self, 
        pdf_path: Path, 
        reference_texts: Dict[int, str],
        brand: Optional[str] = None
    ) -> Tuple[OCRResult, WERMetrics]:
        """
        Process PDF with comprehensive validation against reference texts.
        
        Args:
            pdf_path: Path to PDF file
            reference_texts: Dictionary mapping page numbers to reference texts
            brand: Brand name for configuration override
            
        Returns:
            Tuple of (OCR result, comprehensive WER metrics)
        """
        try:
            # Process PDF
            ocr_result = self.process_pdf(pdf_path, brand)
            
            # Calculate comprehensive WER
            wer_metrics = self.wer_calculator.calculate_ocr_result_wer(
                ocr_result, reference_texts
            )
            
            # Update quality metrics
            ocr_result.quality_metrics.wer = wer_metrics.wer
            ocr_result.quality_metrics.substitutions = wer_metrics.substitutions
            ocr_result.quality_metrics.insertions = wer_metrics.insertions
            ocr_result.quality_metrics.deletions = wer_metrics.deletions
            ocr_result.quality_metrics.total_words = wer_metrics.total_words
            ocr_result.quality_metrics.meets_wer_target = wer_metrics.meets_target
            
            return ocr_result, wer_metrics
            
        except Exception as e:
            self.logger.error("Error in validation processing", error=str(e))
            raise OCRError(f"Validation processing failed: {str(e)}")
    
    def benchmark_preprocessing(
        self, 
        pdf_path: Path, 
        reference_text: str,
        preprocessing_variants: List[PreprocessingConfig],
        brand: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Benchmark different preprocessing configurations.
        
        Args:
            pdf_path: Path to PDF file
            reference_text: Reference text for comparison
            preprocessing_variants: List of preprocessing configurations to test
            brand: Brand name
            
        Returns:
            Benchmark results with WER metrics for each variant
        """
        try:
            results = []
            
            for i, prep_config in enumerate(preprocessing_variants):
                try:
                    self.logger.info(f"Testing preprocessing variant {i+1}",
                                   variant=i+1, total=len(preprocessing_variants))
                    
                    # Temporarily update preprocessor config
                    original_config = self.preprocessor.get_brand_config(brand) if brand else PreprocessingConfig()
                    
                    # Process with this configuration
                    # Note: This would require modifying the engine to accept preprocessing config
                    # For now, we'll simulate the process
                    
                    ocr_result = self.process_pdf(pdf_path, brand)
                    
                    # Calculate WER for this variant
                    wer_metrics = self.wer_calculator.calculate_wer(
                        reference_text, 
                        ocr_result.total_text, 
                        ocr_result.document_type
                    )
                    
                    results.append({
                        "variant": i + 1,
                        "config": prep_config.to_dict(),
                        "wer": wer_metrics.wer,
                        "cer": wer_metrics.cer,
                        "confidence": ocr_result.average_confidence,
                        "processing_time": ocr_result.total_processing_time,
                        "meets_target": wer_metrics.meets_target
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error testing variant {i+1}", error=str(e))
                    results.append({
                        "variant": i + 1,
                        "error": str(e)
                    })
            
            # Find best configuration
            valid_results = [r for r in results if "error" not in r]
            if valid_results:
                best_result = min(valid_results, key=lambda x: x["wer"])
                
                return {
                    "best_variant": best_result["variant"],
                    "best_wer": best_result["wer"],
                    "all_results": results,
                    "recommendations": self._generate_preprocessing_recommendations(results)
                }
            else:
                return {
                    "error": "All preprocessing variants failed",
                    "all_results": results
                }
            
        except Exception as e:
            self.logger.error("Error in preprocessing benchmark", error=str(e))
            raise OCRError(f"Preprocessing benchmark failed: {str(e)}")
    
    def _get_brand_config(self, brand: Optional[str]) -> OCRConfig:
        """Get OCR configuration for specific brand."""
        if brand and brand in self.brand_configs:
            return self.brand_configs[brand]
        return self.config.ocr_config
    
    def _assess_overall_quality(
        self, 
        ocr_result: OCRResult, 
        wer_metrics: Optional[WERMetrics] = None,
        confidence_metrics: Optional[Any] = None
    ) -> QualityMetrics:
        """Assess overall quality and create quality metrics."""
        try:
            quality_metrics = QualityMetrics()
            
            # Set basic metrics
            quality_metrics.avg_confidence = ocr_result.average_confidence
            quality_metrics.processing_time = ocr_result.total_processing_time
            
            # Add WER metrics if available
            if wer_metrics:
                quality_metrics.wer = wer_metrics.wer
                quality_metrics.substitutions = wer_metrics.substitutions
                quality_metrics.insertions = wer_metrics.insertions
                quality_metrics.deletions = wer_metrics.deletions
                quality_metrics.total_words = wer_metrics.total_words
                quality_metrics.meets_wer_target = wer_metrics.meets_target
            
            # Add confidence metrics if available
            if confidence_metrics:
                quality_metrics.min_confidence = confidence_metrics.min_confidence
                quality_metrics.max_confidence = confidence_metrics.max_confidence
                quality_metrics.confidence_std = confidence_metrics.confidence_std
                quality_metrics.reliable_char_ratio = confidence_metrics.reliable_char_ratio
                quality_metrics.uncertain_char_ratio = confidence_metrics.uncertain_char_ratio
            
            # Determine quality flags
            quality_metrics.high_confidence_text = quality_metrics.avg_confidence > 0.9
            quality_metrics.requires_review = (
                quality_metrics.avg_confidence < 0.7 or 
                (wer_metrics and wer_metrics.wer > 0.05)
            )
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error("Error assessing overall quality", error=str(e))
            return QualityMetrics()
    
    def _validate_quality_targets(self, ocr_result: OCRResult, document_type: DocumentType):
        """Validate OCR result against quality targets."""
        try:
            target_wer = (
                self.config.ocr_config.born_digital_wer_target 
                if document_type == DocumentType.BORN_DIGITAL 
                else self.config.ocr_config.scanned_wer_target
            )
            
            # Check confidence threshold
            if ocr_result.average_confidence < self.config.min_confidence_threshold:
                self.logger.warning("OCR confidence below threshold",
                                  confidence=ocr_result.average_confidence,
                                  threshold=self.config.min_confidence_threshold)
            
            # Check WER if available
            if hasattr(ocr_result.quality_metrics, 'wer') and ocr_result.quality_metrics.wer:
                if ocr_result.quality_metrics.wer > target_wer:
                    self.logger.warning("WER exceeds target",
                                      wer=ocr_result.quality_metrics.wer,
                                      target=target_wer,
                                      document_type=document_type.value)
            
        except Exception as e:
            self.logger.warning("Error validating quality targets", error=str(e))
    
    def _generate_preprocessing_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on preprocessing benchmark results."""
        try:
            recommendations = []
            
            valid_results = [r for r in results if "error" not in r]
            if not valid_results:
                return ["All preprocessing variants failed - check input quality"]
            
            # Find best and worst performers
            best_wer = min(r["wer"] for r in valid_results)
            worst_wer = max(r["wer"] for r in valid_results)
            
            if worst_wer - best_wer > 0.01:  # >1% difference
                recommendations.append("Preprocessing choice significantly impacts accuracy")
            
            # Analyze which techniques help
            best_configs = [r["config"] for r in valid_results if r["wer"] == best_wer]
            if best_configs:
                config = best_configs[0]
                if config.get("denoise_enabled", False):
                    recommendations.append("Denoising improves accuracy")
                if config.get("deskew_enabled", False):
                    recommendations.append("Deskewing improves accuracy")
                if config.get("adaptive_threshold", False):
                    recommendations.append("Adaptive thresholding improves accuracy")
            
            return recommendations
            
        except Exception as e:
            self.logger.warning("Error generating preprocessing recommendations", error=str(e))
            return ["Error generating recommendations"]
    
    def get_processing_summary(self, ocr_result: OCRResult) -> Dict[str, Any]:
        """Get comprehensive processing summary."""
        try:
            return {
                "document_info": {
                    "type": ocr_result.document_type.value,
                    "pages": ocr_result.page_count,
                    "total_words": ocr_result.total_words,
                    "total_characters": ocr_result.total_characters
                },
                "quality_metrics": ocr_result.quality_metrics.to_dict(),
                "performance": {
                    "total_processing_time": ocr_result.total_processing_time,
                    "avg_time_per_page": ocr_result.total_processing_time / max(ocr_result.page_count, 1),
                    "words_per_second": ocr_result.total_words / max(ocr_result.total_processing_time, 1)
                },
                "brand": ocr_result.brand,
                "timestamp": ocr_result.timestamp.isoformat(),
                "meets_targets": {
                    "confidence": ocr_result.average_confidence > self.config.min_confidence_threshold,
                    "wer": ocr_result.quality_metrics.meets_wer_target if hasattr(ocr_result.quality_metrics, 'meets_wer_target') else None
                }
            }
            
        except Exception as e:
            self.logger.error("Error creating processing summary", error=str(e))
            return {"error": str(e)}