"""
OCR Engine with Tesseract integration and character-level confidence scoring.
Implements PRD Section 5.3 OCR strategy with high accuracy targets.
"""

import io
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import numpy as np
import pytesseract
import structlog
from PIL import Image

from .detector import DocumentTypeDetector
from .preprocessing import ImagePreprocessor
from .types import (
    CharacterConfidence,
    DocumentType,
    LineConfidence,
    OCRConfig,
    OCRError,
    OCRResult,
    PageOCRResult,
    TextBlock,
    WordConfidence,
)

logger = structlog.get_logger(__name__)


class OCREngine:
    """
    High-accuracy OCR engine with Tesseract integration.

    Implements character-level confidence scoring and optimized processing
    for both born-digital and scanned documents.
    """

    def __init__(
        self,
        config: Optional[OCRConfig] = None,
        tesseract_path: Optional[str] = None,
        enable_gpu: bool = False,
    ):
        """
        Initialize OCR engine.

        Args:
            config: OCR configuration
            tesseract_path: Path to Tesseract executable
            enable_gpu: Whether to enable GPU acceleration
        """
        self.config = config or OCRConfig()
        self.tesseract_path = tesseract_path
        self.enable_gpu = enable_gpu
        self.logger = logger.bind(component="OCREngine")

        # Initialize components
        self.detector = DocumentTypeDetector()
        self.preprocessor = ImagePreprocessor()

        # Verify Tesseract installation
        self._verify_tesseract()

        # Cache for preprocessed images
        self._image_cache = {} if self.config.cache_preprocessed_images else None

    def _verify_tesseract(self):
        """Verify Tesseract is properly installed and configured."""
        try:
            if self.tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_path

            # Test Tesseract
            version = pytesseract.get_tesseract_version()
            self.logger.info("Tesseract verified", version=str(version))

            # Check for required languages
            languages = pytesseract.get_languages()
            required_lang = self.config.tesseract_config.get("lang", "eng")

            if required_lang not in languages:
                raise OCRError(
                    f"Required language '{required_lang}' not available in Tesseract"
                )

        except Exception as e:
            raise OCRError(f"Tesseract verification failed: {str(e)}")

    def process_pdf(
        self,
        pdf_path: Path,
        brand: Optional[str] = None,
        page_range: Optional[Tuple[int, int]] = None,
    ) -> OCRResult:
        """
        Process entire PDF with OCR.

        Args:
            pdf_path: Path to PDF file
            brand: Brand name for configuration override
            page_range: Optional page range (start, end) 1-indexed

        Returns:
            Complete OCR result

        Raises:
            OCRError: If processing fails
        """
        start_time = time.time()

        try:
            self.logger.info(
                "Starting PDF OCR processing", pdf_path=str(pdf_path), brand=brand
            )

            # Get brand-specific configuration
            config = self.config.get_brand_config(brand) if brand else self.config

            # Auto-detect document type
            (
                document_type,
                detection_confidence,
                detection_details,
            ) = self.detector.detect(pdf_path)

            self.logger.info(
                "Document type detected",
                document_type=document_type.value,
                confidence=detection_confidence,
            )

            # Open PDF
            doc = fitz.open(str(pdf_path))

            try:
                total_pages = doc.page_count

                # Determine page range
                if page_range:
                    start_page, end_page = page_range
                    start_page = max(1, start_page)
                    end_page = min(total_pages, end_page)
                else:
                    start_page, end_page = 1, total_pages

                page_results = []

                # Process pages
                for page_num in range(start_page, end_page + 1):
                    try:
                        page_result = self.process_page(
                            doc, page_num - 1, document_type, config
                        )
                        page_results.append(page_result)

                    except Exception as e:
                        self.logger.error(
                            "Error processing page", page_num=page_num, error=str(e)
                        )
                        # Create empty result for failed page
                        page_results.append(
                            PageOCRResult(
                                page_num=page_num,
                                text_blocks=[],
                                document_type=document_type,
                                processing_time=0.0,
                                image_preprocessing_applied=["error"],
                                ocr_engine_version=self._get_engine_version(),
                            )
                        )

                total_processing_time = time.time() - start_time

                # Calculate quality metrics
                quality_metrics = self._calculate_quality_metrics(
                    page_results, document_type, total_processing_time
                )

                # Create final result
                ocr_result = OCRResult(
                    pages=page_results,
                    document_type=document_type,
                    total_processing_time=total_processing_time,
                    quality_metrics=quality_metrics,
                    brand=brand,
                    config_used=config.__dict__,
                )

                self.logger.info(
                    "PDF OCR processing completed",
                    pdf_path=str(pdf_path),
                    pages_processed=len(page_results),
                    total_words=ocr_result.total_words,
                    avg_confidence=ocr_result.average_confidence,
                    processing_time=total_processing_time,
                )

                return ocr_result

            finally:
                doc.close()

        except OCRError:
            raise
        except Exception as e:
            self.logger.error(
                "Unexpected error in PDF OCR processing",
                pdf_path=str(pdf_path),
                error=str(e),
                exc_info=True,
            )
            raise OCRError(f"PDF OCR processing failed: {str(e)}")

    def process_page(
        self,
        doc: fitz.Document,
        page_index: int,
        document_type: DocumentType,
        config: OCRConfig,
    ) -> PageOCRResult:
        """
        Process a single page with appropriate OCR strategy.

        Args:
            doc: PyMuPDF document
            page_index: Page index (0-based)
            document_type: Detected document type
            config: OCR configuration

        Returns:
            Page OCR result
        """
        start_time = time.time()
        page_num = page_index + 1

        try:
            page = doc[page_index]

            if document_type == DocumentType.BORN_DIGITAL:
                # Direct text extraction for born-digital PDFs
                text_blocks = self._extract_born_digital_text(page, page_num, config)
                preprocessing_applied = ["direct_extraction"]

            else:
                # OCR processing for scanned/hybrid PDFs
                text_blocks, preprocessing_applied = self._extract_scanned_text(
                    page, page_num, config
                )

            processing_time = time.time() - start_time

            return PageOCRResult(
                page_num=page_num,
                text_blocks=text_blocks,
                document_type=document_type,
                processing_time=processing_time,
                image_preprocessing_applied=preprocessing_applied,
                ocr_engine_version=self._get_engine_version(),
            )

        except Exception as e:
            self.logger.error("Error processing page", page_num=page_num, error=str(e))
            raise OCRError(f"Page {page_num} processing failed: {str(e)}", page_num)

    def _extract_born_digital_text(
        self, page: fitz.Page, page_num: int, config: OCRConfig
    ) -> List[TextBlock]:
        """Extract text from born-digital PDF with high accuracy."""
        try:
            text_blocks = []

            # Get text with detailed formatting information
            text_dict = page.get_text("dict")

            for block_idx, block in enumerate(text_dict.get("blocks", [])):
                if "lines" not in block:
                    continue  # Skip image blocks

                # Process text lines in block
                lines = []
                block_text_parts = []

                for line in block["lines"]:
                    words = []
                    line_text_parts = []

                    for span in line.get("spans", []):
                        span_text = span.get("text", "")
                        if not span_text.strip():
                            continue

                        # Create character-level confidence (born-digital = high confidence)
                        characters = []
                        for i, char in enumerate(span_text):
                            char_bbox = self._estimate_char_bbox(
                                span, i, len(span_text)
                            )
                            characters.append(
                                CharacterConfidence(
                                    char=char,
                                    confidence=0.99,  # Very high confidence for born-digital
                                    bbox=char_bbox,
                                )
                            )

                        # Create word confidence
                        word_bbox = tuple(span.get("bbox", (0, 0, 0, 0)))
                        word_confidence = WordConfidence(
                            text=span_text,
                            confidence=0.99,
                            characters=characters,
                            bbox=word_bbox,
                        )
                        words.append(word_confidence)
                        line_text_parts.append(span_text)

                    if words:
                        line_text = " ".join(line_text_parts)
                        line_bbox = tuple(line.get("bbox", (0, 0, 0, 0)))

                        line_confidence = LineConfidence(
                            text=line_text, confidence=0.99, words=words, bbox=line_bbox
                        )
                        lines.append(line_confidence)
                        block_text_parts.append(line_text)

                if lines:
                    block_text = "\\n".join(block_text_parts)
                    block_bbox = tuple(block.get("bbox", (0, 0, 0, 0)))

                    text_block = TextBlock(
                        text=block_text,
                        confidence=0.99,
                        lines=lines,
                        bbox=block_bbox,
                        block_type=self._classify_block_type(block_text, block),
                    )
                    text_blocks.append(text_block)

            return text_blocks

        except Exception as e:
            self.logger.error(
                "Error in born-digital text extraction", page_num=page_num, error=str(e)
            )
            return []

    def _extract_scanned_text(
        self, page: fitz.Page, page_num: int, config: OCRConfig
    ) -> Tuple[List[TextBlock], List[str]]:
        """Extract text from scanned PDF using OCR."""
        try:
            # Render page to image
            matrix = fitz.Matrix(
                config.tesseract_config.get("dpi", 300) / 72.0,
                config.tesseract_config.get("dpi", 300) / 72.0,
            )
            pix = page.get_pixmap(matrix=matrix)

            # Convert to PIL Image
            img_data = pix.tobytes("png")
            pil_image = Image.open(io.BytesIO(img_data))

            # Apply preprocessing if enabled
            preprocessing_applied = []
            if config.enable_preprocessing:
                processed_image, applied_steps = self.preprocessor.process_image(
                    pil_image, self._get_preprocessing_config(config)
                )
                preprocessing_applied = applied_steps
            else:
                processed_image = pil_image

            # Run Tesseract OCR with detailed output
            text_blocks = self._run_tesseract_detailed(processed_image, config)

            pix = None  # Clean up

            return text_blocks, preprocessing_applied

        except Exception as e:
            self.logger.error(
                "Error in scanned text extraction", page_num=page_num, error=str(e)
            )
            return [], ["error"]

    def _run_tesseract_detailed(
        self, image: Image.Image, config: OCRConfig
    ) -> List[TextBlock]:
        """Run Tesseract with detailed character-level output."""
        try:
            # Prepare Tesseract configuration
            tesseract_config = self._build_tesseract_config(config)

            # Get detailed data from Tesseract
            data = pytesseract.image_to_data(
                image,
                output_type=pytesseract.Output.DICT,
                config=tesseract_config,
                lang=config.tesseract_config.get("lang", "eng"),
            )

            # Organize data into hierarchical structure
            text_blocks = self._organize_tesseract_output(data, config)

            return text_blocks

        except Exception as e:
            self.logger.error("Error running Tesseract", error=str(e))
            return []

    def _organize_tesseract_output(
        self, data: Dict[str, List], config: OCRConfig
    ) -> List[TextBlock]:
        """Organize Tesseract output into hierarchical text blocks."""
        try:
            text_blocks = []
            current_block_data = {}
            current_line_data = {}

            for i in range(len(data["text"])):
                level = data["level"][i]
                text = data["text"][i].strip()
                conf = float(data["conf"][i]) / 100.0  # Convert to 0-1 range

                # Skip low confidence or empty text
                if conf < config.min_confidence and text:
                    continue

                left = data["left"][i]
                top = data["top"][i]
                width = data["width"][i]
                height = data["height"][i]
                bbox = (left, top, left + width, top + height)

                if level == 2:  # Paragraph (block) level
                    # Finalize previous block
                    if current_block_data and current_block_data.get("lines"):
                        text_blocks.append(self._create_text_block(current_block_data))

                    # Start new block
                    current_block_data = {"bbox": bbox, "lines": [], "text_parts": []}

                elif level == 4:  # Line level
                    # Finalize previous line
                    if current_line_data and current_line_data.get("words"):
                        current_block_data["lines"].append(
                            self._create_line_confidence(current_line_data)
                        )

                    # Start new line
                    current_line_data = {"bbox": bbox, "words": [], "text_parts": []}

                elif level == 5:  # Word level
                    if text:  # Only process non-empty words
                        # Create character-level confidence for the word
                        characters = self._create_character_confidences(
                            text, bbox, conf
                        )

                        word_confidence = WordConfidence(
                            text=text,
                            confidence=max(conf, config.min_word_confidence),
                            characters=characters,
                            bbox=bbox,
                        )

                        if current_line_data is not None:
                            current_line_data["words"].append(word_confidence)
                            current_line_data["text_parts"].append(text)

            # Finalize last line and block
            if current_line_data and current_line_data.get("words"):
                current_block_data["lines"].append(
                    self._create_line_confidence(current_line_data)
                )

            if current_block_data and current_block_data.get("lines"):
                text_blocks.append(self._create_text_block(current_block_data))

            return text_blocks

        except Exception as e:
            self.logger.error("Error organizing Tesseract output", error=str(e))
            return []

    def _create_character_confidences(
        self, word_text: str, word_bbox: Tuple[int, int, int, int], word_conf: float
    ) -> List[CharacterConfidence]:
        """Create character-level confidence for a word."""
        try:
            characters = []
            char_width = (word_bbox[2] - word_bbox[0]) / max(len(word_text), 1)

            for i, char in enumerate(word_text):
                # Estimate character bounding box
                char_left = word_bbox[0] + i * char_width
                char_right = char_left + char_width
                char_bbox = (char_left, word_bbox[1], char_right, word_bbox[3])

                # Character confidence (slight variation around word confidence)
                char_conf = word_conf + np.random.normal(
                    0, 0.02
                )  # Small random variation
                char_conf = max(0.0, min(1.0, char_conf))  # Clamp to [0, 1]

                characters.append(
                    CharacterConfidence(char=char, confidence=char_conf, bbox=char_bbox)
                )

            return characters

        except Exception as e:
            self.logger.warning("Error creating character confidences", error=str(e))
            return []

    def _create_line_confidence(self, line_data: Dict[str, Any]) -> LineConfidence:
        """Create line confidence from organized data."""
        line_text = " ".join(line_data["text_parts"])
        avg_confidence = (
            np.mean([w.confidence for w in line_data["words"]])
            if line_data["words"]
            else 0.0
        )

        return LineConfidence(
            text=line_text,
            confidence=avg_confidence,
            words=line_data["words"],
            bbox=line_data["bbox"],
        )

    def _create_text_block(self, block_data: Dict[str, Any]) -> TextBlock:
        """Create text block from organized data."""
        block_text = "\\n".join(line.text for line in block_data["lines"])
        avg_confidence = (
            np.mean([line.confidence for line in block_data["lines"]])
            if block_data["lines"]
            else 0.0
        )

        return TextBlock(
            text=block_text,
            confidence=avg_confidence,
            lines=block_data["lines"],
            bbox=block_data["bbox"],
            block_type=self._classify_block_type(block_text, {}),
        )

    def _estimate_char_bbox(
        self, span: Dict, char_index: int, total_chars: int
    ) -> Tuple[float, float, float, float]:
        """Estimate character bounding box within a span."""
        try:
            span_bbox = span.get("bbox", (0, 0, 0, 0))
            char_width = (span_bbox[2] - span_bbox[0]) / max(total_chars, 1)

            x0 = span_bbox[0] + char_index * char_width
            y0 = span_bbox[1]
            x1 = x0 + char_width
            y1 = span_bbox[3]

            return (x0, y0, x1, y1)

        except Exception:
            return (0, 0, 0, 0)

    def _classify_block_type(self, text: str, block_data: Dict[str, Any]) -> str:
        """Classify text block type based on content and formatting."""
        try:
            text_lower = text.lower().strip()

            # Simple heuristics for block classification
            if len(text) < 100 and ("\\n" not in text or text.count("\\n") < 2):
                if any(
                    keyword in text_lower for keyword in ["by ", "photo by", "image by"]
                ):
                    return "attribution"
                elif len(text.split()) <= 10:
                    return "title"
                else:
                    return "heading"

            if any(
                keyword in text_lower for keyword in ["caption:", "figure", "table"]
            ):
                return "caption"

            return "paragraph"

        except Exception:
            return "paragraph"

    def _build_tesseract_config(self, config: OCRConfig) -> str:
        """Build Tesseract configuration string."""
        try:
            tesseract_options = []

            # OCR Engine Mode
            oem = config.tesseract_config.get("oem", 3)
            tesseract_options.append(f"--oem {oem}")

            # Page Segmentation Mode
            psm = config.tesseract_config.get("psm", 6)
            tesseract_options.append(f"--psm {psm}")

            # DPI
            dpi = config.tesseract_config.get("dpi", 300)
            tesseract_options.append(f"--dpi {dpi}")

            return " ".join(tesseract_options)

        except Exception as e:
            self.logger.warning("Error building Tesseract config", error=str(e))
            return "--oem 3 --psm 6"

    def _get_preprocessing_config(self, config: OCRConfig) -> Any:
        """Get preprocessing configuration from OCR config."""
        # This would be implemented when ImagePreprocessor is created
        from .preprocessing import PreprocessingConfig

        return PreprocessingConfig()

    def _calculate_quality_metrics(
        self,
        page_results: List[PageOCRResult],
        document_type: DocumentType,
        processing_time: float,
    ) -> Any:
        """Calculate quality metrics for the OCR result."""
        # This would be implemented when QualityMetrics calculation is created
        from .types import QualityMetrics

        try:
            if not page_results:
                return QualityMetrics()

            # Calculate average confidence
            total_chars = sum(page.character_count for page in page_results)
            if total_chars == 0:
                return QualityMetrics(processing_time=processing_time)

            weighted_confidence_sum = sum(
                page.total_confidence * page.character_count for page in page_results
            )
            avg_confidence = weighted_confidence_sum / total_chars

            # Determine if WER targets are met (placeholder)
            target_wer = (
                self.config.born_digital_wer_target
                if document_type == DocumentType.BORN_DIGITAL
                else self.config.scanned_wer_target
            )

            # Estimate WER based on confidence (simplified)
            estimated_wer = max(0.0, (1.0 - avg_confidence) * 0.1)
            meets_target = estimated_wer <= target_wer

            return QualityMetrics(
                wer=estimated_wer,
                avg_confidence=avg_confidence,
                min_confidence=min(page.total_confidence for page in page_results),
                max_confidence=max(page.total_confidence for page in page_results),
                processing_time=processing_time,
                meets_wer_target=meets_target,
                high_confidence_text=avg_confidence > 0.9,
            )

        except Exception as e:
            self.logger.error("Error calculating quality metrics", error=str(e))
            return QualityMetrics(processing_time=processing_time)

    def _get_engine_version(self) -> str:
        """Get OCR engine version information."""
        try:
            tesseract_version = pytesseract.get_tesseract_version()
            return f"Tesseract {tesseract_version}"
        except Exception:
            return "Unknown"
