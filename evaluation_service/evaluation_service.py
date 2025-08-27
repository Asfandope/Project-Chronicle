"""
Core evaluation service that processes XML comparisons and calculates accuracy.

This module provides the main evaluation logic that compares extracted XML
against ground truth and calculates field-level accuracy metrics.
"""

import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from synthetic_data.accuracy_calculator import AccuracyCalculator
from synthetic_data.ground_truth import GroundTruthGenerator
from synthetic_data.types import ArticleData, GroundTruthData, ImageElement, TextElement

from .models import ArticleEvaluation, DocumentEvaluation, EvaluationRun
from .schemas import (
    ArticleAccuracySchema,
    DocumentAccuracySchema,
    EvaluationType,
    ManualEvaluationRequest,
    TriggerSource,
)

logger = logging.getLogger(__name__)


class XMLParsingError(Exception):
    """Raised when XML parsing fails."""


class EvaluationService:
    """Core service for evaluating extraction accuracy against ground truth."""

    def __init__(self):
        self.accuracy_calculator = AccuracyCalculator()
        self.ground_truth_generator = GroundTruthGenerator()
        self.logger = logging.getLogger(__name__ + ".EvaluationService")

    def evaluate_single_document(
        self,
        session: Session,
        request: ManualEvaluationRequest,
        evaluation_run_id: Optional[str] = None,
    ) -> DocumentEvaluation:
        """Evaluate a single document against its ground truth."""

        start_time = time.time()

        try:
            # Parse XML content
            ground_truth_data = self._parse_ground_truth_xml(
                request.ground_truth_content
            )
            extracted_data = self._parse_extracted_xml(request.extracted_content)

            # Calculate accuracy
            document_accuracy = self.accuracy_calculator.calculate_document_accuracy(
                ground_truth_data, extracted_data
            )

            # Create database record
            doc_evaluation = self._create_document_evaluation(
                session=session,
                request=request,
                document_accuracy=document_accuracy,
                processing_time=time.time() - start_time,
                evaluation_run_id=evaluation_run_id,
            )

            self.logger.info(
                f"Evaluated document {request.document_id}: "
                f"accuracy={document_accuracy.document_weighted_accuracy:.3f}"
            )

            return doc_evaluation

        except Exception as e:
            self.logger.error(
                f"Error evaluating document {request.document_id}: {str(e)}"
            )

            # Create failed evaluation record
            doc_evaluation = DocumentEvaluation(
                evaluation_run_id=evaluation_run_id,
                document_id=request.document_id,
                brand_name=request.brand_name,
                complexity_level=request.complexity_level,
                edge_cases=request.edge_cases or [],
                extraction_successful=False,
                extraction_error=str(e),
                extraction_time_seconds=time.time() - start_time,
            )

            session.add(doc_evaluation)
            session.commit()

            return doc_evaluation

    def evaluate_batch(
        self,
        session: Session,
        documents: List[ManualEvaluationRequest],
        evaluation_type: EvaluationType = EvaluationType.BATCH,
        trigger_source: TriggerSource = TriggerSource.API_REQUEST,
    ) -> EvaluationRun:
        """Evaluate a batch of documents."""

        start_time = time.time()

        # Create evaluation run
        evaluation_run = EvaluationRun(
            evaluation_type=evaluation_type.value,
            trigger_source=trigger_source.value,
            document_count=len(documents),
            extractor_version="1.0.0",  # TODO: Get from request or config
            model_version="1.0.0",
        )

        session.add(evaluation_run)
        session.flush()  # Get the ID

        document_evaluations = []
        successful_extractions = 0
        failed_extractions = 0
        total_articles = 0

        # Aggregate accuracy metrics
        total_weighted_accuracy = 0.0
        total_title_accuracy = 0.0
        total_body_accuracy = 0.0
        total_contributors_accuracy = 0.0
        total_media_accuracy = 0.0

        for request in documents:
            try:
                doc_eval = self.evaluate_single_document(
                    session, request, evaluation_run.id
                )
                document_evaluations.append(doc_eval)

                if doc_eval.extraction_successful:
                    successful_extractions += 1
                    total_weighted_accuracy += doc_eval.weighted_overall_accuracy
                    total_title_accuracy += doc_eval.title_accuracy
                    total_body_accuracy += doc_eval.body_text_accuracy
                    total_contributors_accuracy += doc_eval.contributors_accuracy
                    total_media_accuracy += doc_eval.media_links_accuracy
                else:
                    failed_extractions += 1

                # Count articles
                total_articles += len(doc_eval.article_evaluations)

            except Exception as e:
                self.logger.error(
                    f"Failed to evaluate document {request.document_id}: {str(e)}"
                )
                failed_extractions += 1

        # Calculate averages
        successful_count = max(1, successful_extractions)
        evaluation_run.successful_extractions = successful_extractions
        evaluation_run.failed_extractions = failed_extractions
        evaluation_run.total_articles = total_articles
        evaluation_run.overall_weighted_accuracy = (
            total_weighted_accuracy / successful_count
        )
        evaluation_run.title_accuracy = total_title_accuracy / successful_count
        evaluation_run.body_text_accuracy = total_body_accuracy / successful_count
        evaluation_run.contributors_accuracy = (
            total_contributors_accuracy / successful_count
        )
        evaluation_run.media_links_accuracy = total_media_accuracy / successful_count
        evaluation_run.processing_time_seconds = time.time() - start_time

        session.commit()

        self.logger.info(
            f"Completed batch evaluation: {successful_extractions}/{len(documents)} successful, "
            f"avg_accuracy={evaluation_run.overall_weighted_accuracy:.3f}"
        )

        return evaluation_run

    def _parse_ground_truth_xml(self, xml_content: str) -> GroundTruthData:
        """Parse ground truth XML into structured data."""

        try:
            root = ET.fromstring(xml_content)

            # Extract document metadata
            doc_meta = root.find("document_metadata")
            if doc_meta is None:
                raise XMLParsingError("Missing document_metadata element")

            document_id = doc_meta.findtext("document_id", "")
            brand_name = doc_meta.findtext("brand_name", "")
            page_count = int(doc_meta.findtext("page_count", "1"))

            # Parse articles
            articles = []
            articles_elem = root.find("articles")
            if articles_elem is not None:
                for article_elem in articles_elem.findall("article"):
                    article = self._parse_article_element(article_elem)
                    articles.append(article)

            # Parse all elements
            all_text_elements = []
            all_image_elements = []

            elements_elem = root.find("all_elements")
            if elements_elem is not None:
                # Parse text elements
                text_elements_elem = elements_elem.find("text_elements")
                if text_elements_elem is not None:
                    for text_elem in text_elements_elem.findall("text_element"):
                        text_element = self._parse_text_element(text_elem)
                        all_text_elements.append(text_element)

                # Parse image elements
                image_elements_elem = elements_elem.find("image_elements")
                if image_elements_elem is not None:
                    for img_elem in image_elements_elem.findall("image_element"):
                        image_element = self._parse_image_element(img_elem)
                        all_image_elements.append(image_element)

            # Create GroundTruthData object
            ground_truth = GroundTruthData(
                document_id=document_id,
                brand_name=brand_name,
                generation_timestamp=datetime.now(timezone.utc),
                articles=articles,
                all_text_elements=all_text_elements,
                all_image_elements=all_image_elements,
                page_count=page_count,
            )

            return ground_truth

        except ET.ParseError as e:
            raise XMLParsingError(f"Invalid XML format: {str(e)}")
        except Exception as e:
            raise XMLParsingError(f"Error parsing ground truth XML: {str(e)}")

    def _parse_extracted_xml(self, xml_content: str) -> Dict[str, Any]:
        """Parse extracted XML into dictionary format."""

        try:
            root = ET.fromstring(xml_content)

            # Extract document info
            document_id = root.get("id", "")

            articles = []

            # Parse articles
            for article_elem in root.findall(".//article"):
                article = {
                    "article_id": article_elem.get("id", ""),
                    "title": article_elem.findtext("title", ""),
                    "text_content": self._extract_text_content(article_elem),
                    "contributors": self._extract_contributors(article_elem),
                    "media_elements": self._extract_media_elements(article_elem),
                }
                articles.append(article)

            return {"document_id": document_id, "articles": articles}

        except ET.ParseError as e:
            raise XMLParsingError(f"Invalid extracted XML format: {str(e)}")
        except Exception as e:
            raise XMLParsingError(f"Error parsing extracted XML: {str(e)}")

    def _parse_article_element(self, article_elem: ET.Element) -> ArticleData:
        """Parse article element from ground truth XML."""

        article_id = article_elem.get("id", "")
        title = article_elem.findtext("title", "")
        article_type = article_elem.findtext("article_type", "feature")

        # Parse page range
        page_range_elem = article_elem.find("page_range")
        page_range = (1, 1)
        if page_range_elem is not None:
            start = int(page_range_elem.get("start", "1"))
            end = int(page_range_elem.get("end", "1"))
            page_range = (start, end)

        # Parse contributors
        contributors = []
        contributors_elem = article_elem.find("contributors")
        if contributors_elem is not None:
            for contrib_elem in contributors_elem.findall("contributor"):
                contributor = {
                    "name": contrib_elem.get("name", ""),
                    "role": contrib_elem.get("role", ""),
                    "affiliation": contrib_elem.get("affiliation", ""),
                }
                contributors.append(contributor)

        # Parse text elements for this article
        text_elements = []
        text_elements_elem = article_elem.find("text_elements")
        if text_elements_elem is not None:
            for text_elem in text_elements_elem.findall("text_element"):
                text_element = self._parse_text_element(text_elem)
                text_elements.append(text_element)

        # Parse image elements for this article
        image_elements = []
        image_elements_elem = article_elem.find("image_elements")
        if image_elements_elem is not None:
            for img_elem in image_elements_elem.findall("image_element"):
                image_element = self._parse_image_element(img_elem)
                image_elements.append(image_element)

        return ArticleData(
            article_id=article_id,
            title=title,
            contributors=contributors,
            text_elements=text_elements,
            image_elements=image_elements,
            page_range=page_range,
            article_type=article_type,
        )

    def _parse_text_element(self, text_elem: ET.Element) -> TextElement:
        """Parse text element from XML."""

        element_id = text_elem.get("id", "")
        semantic_type = text_elem.get("type", "paragraph")
        page_number = int(text_elem.get("page", "1"))
        reading_order = int(text_elem.get("reading_order", "0"))

        # Parse bounding box
        bbox_elem = text_elem.find("bbox")
        bbox = (0, 0, 0, 0)
        if bbox_elem is not None:
            x0 = float(bbox_elem.get("x0", "0"))
            y0 = float(bbox_elem.get("y0", "0"))
            x1 = float(bbox_elem.get("x1", "0"))
            y1 = float(bbox_elem.get("y1", "0"))
            bbox = (x0, y0, x1, y1)

        # Parse content
        content_elem = text_elem.find("content")
        text_content = content_elem.text if content_elem is not None else ""

        # Parse font information
        font_elem = text_elem.find("font")
        font_family = "Arial"
        font_size = 12.0
        font_style = "normal"
        text_align = "left"

        if font_elem is not None:
            font_family = font_elem.get("family", "Arial")
            font_size = float(font_elem.get("size", "12"))
            font_style = font_elem.get("style", "normal")
            text_align = font_elem.get("align", "left")

        # Parse color
        color_elem = text_elem.find("color")
        text_color = (0.0, 0.0, 0.0)
        if color_elem is not None:
            r = float(color_elem.get("r", "0"))
            g = float(color_elem.get("g", "0"))
            b = float(color_elem.get("b", "0"))
            text_color = (r, g, b)

        # Parse extraction metadata
        extraction_elem = text_elem.find("extraction_metadata")
        confidence = 1.0
        difficulty = 0.0
        z_order = 0

        if extraction_elem is not None:
            confidence = float(extraction_elem.get("confidence", "1.0"))
            difficulty = float(extraction_elem.get("difficulty", "0.0"))
            z_order = int(extraction_elem.get("z_order", "0"))

        return TextElement(
            element_id=element_id,
            element_type="text",
            bbox=bbox,
            page_number=page_number,
            text_content=text_content,
            font_family=font_family,
            font_size=font_size,
            font_style=font_style,
            text_color=text_color,
            text_align=text_align,
            semantic_type=semantic_type,
            reading_order=reading_order,
            confidence=confidence,
            extraction_difficulty=difficulty,
            z_order=z_order,
        )

    def _parse_image_element(self, img_elem: ET.Element) -> ImageElement:
        """Parse image element from XML."""

        element_id = img_elem.get("id", "")
        page_number = int(img_elem.get("page", "1"))

        # Parse bounding box
        bbox_elem = img_elem.find("bbox")
        bbox = (0, 0, 0, 0)
        if bbox_elem is not None:
            x0 = float(bbox_elem.get("x0", "0"))
            y0 = float(bbox_elem.get("y0", "0"))
            x1 = float(bbox_elem.get("x1", "0"))
            y1 = float(bbox_elem.get("y1", "0"))
            bbox = (x0, y0, x1, y1)

        # Parse image properties
        props_elem = img_elem.find("image_properties")
        width = 0
        height = 0
        dpi = 300
        color_space = "RGB"

        if props_elem is not None:
            width = int(props_elem.get("width", "0"))
            height = int(props_elem.get("height", "0"))
            dpi = int(props_elem.get("dpi", "300"))
            color_space = props_elem.get("color_space", "RGB")

        # Parse alt text
        alt_text_elem = img_elem.find("alt_text")
        alt_text = alt_text_elem.text if alt_text_elem is not None else ""

        # Parse extraction metadata
        extraction_elem = img_elem.find("extraction_metadata")
        confidence = 1.0
        difficulty = 0.0
        z_order = 0

        if extraction_elem is not None:
            confidence = float(extraction_elem.get("confidence", "1.0"))
            difficulty = float(extraction_elem.get("difficulty", "0.0"))
            z_order = int(extraction_elem.get("z_order", "0"))

        return ImageElement(
            element_id=element_id,
            element_type="image",
            bbox=bbox,
            page_number=page_number,
            alt_text=alt_text,
            width=width,
            height=height,
            dpi=dpi,
            color_space=color_space,
            confidence=confidence,
            extraction_difficulty=difficulty,
            z_order=z_order,
        )

    def _extract_text_content(self, article_elem: ET.Element) -> str:
        """Extract all text content from article element."""

        text_parts = []

        # Get title
        title = article_elem.findtext("title", "")
        if title:
            text_parts.append(title)

        # Get all text elements
        for text_elem in article_elem.findall(".//text_element"):
            content = text_elem.findtext("content", "")
            if content:
                text_parts.append(content)

        # Get paragraph content
        for p_elem in article_elem.findall(".//paragraph"):
            content = p_elem.text or ""
            if content:
                text_parts.append(content)

        return " ".join(text_parts)

    def _extract_contributors(self, article_elem: ET.Element) -> List[Dict[str, str]]:
        """Extract contributor information from article element."""

        contributors = []

        # Check contributors section
        contributors_elem = article_elem.find("contributors")
        if contributors_elem is not None:
            for contrib_elem in contributors_elem.findall("contributor"):
                contributor = {
                    "name": contrib_elem.get("name", "") or contrib_elem.text or "",
                    "role": contrib_elem.get("role", "author"),
                }
                contributors.append(contributor)

        # Check byline elements
        for byline_elem in article_elem.findall(".//byline"):
            byline_text = byline_elem.text or ""
            if byline_text:
                # Simple parsing of byline (e.g., "By John Doe")
                if byline_text.lower().startswith("by "):
                    name = byline_text[3:].strip()
                    contributor = {"name": name, "role": "author"}
                    contributors.append(contributor)

        return contributors

    def _extract_media_elements(self, article_elem: ET.Element) -> List[Dict[str, Any]]:
        """Extract media element information from article element."""

        media_elements = []

        # Find image elements
        for img_elem in article_elem.findall(".//image"):
            # Parse bounding box if available
            bbox = None
            bbox_elem = img_elem.find("bbox")
            if bbox_elem is not None:
                bbox = (
                    float(bbox_elem.get("x0", "0")),
                    float(bbox_elem.get("y0", "0")),
                    float(bbox_elem.get("x1", "0")),
                    float(bbox_elem.get("y1", "0")),
                )

            # Find associated caption
            caption = ""
            caption_elem = img_elem.find("caption")
            if caption_elem is not None:
                caption = caption_elem.text or ""

            media_element = {
                "type": "image",
                "bbox": bbox,
                "caption": caption,
                "width": int(img_elem.get("width", "0")),
                "height": int(img_elem.get("height", "0")),
            }

            media_elements.append(media_element)

        return media_elements

    def _create_document_evaluation(
        self,
        session: Session,
        request: ManualEvaluationRequest,
        document_accuracy: DocumentAccuracySchema,
        processing_time: float,
        evaluation_run_id: Optional[str] = None,
    ) -> DocumentEvaluation:
        """Create DocumentEvaluation database record."""

        # Create document evaluation
        doc_evaluation = DocumentEvaluation(
            evaluation_run_id=evaluation_run_id,
            document_id=request.document_id,
            brand_name=request.brand_name,
            complexity_level=request.complexity_level,
            edge_cases=request.edge_cases or [],
            weighted_overall_accuracy=document_accuracy.document_weighted_accuracy,
            title_accuracy=document_accuracy.overall_title_accuracy.accuracy,
            body_text_accuracy=document_accuracy.overall_body_text_accuracy.accuracy,
            contributors_accuracy=document_accuracy.overall_contributors_accuracy.accuracy,
            media_links_accuracy=document_accuracy.overall_media_links_accuracy.accuracy,
            title_correct=document_accuracy.overall_title_accuracy.correct,
            title_total=document_accuracy.overall_title_accuracy.total,
            body_text_correct=document_accuracy.overall_body_text_accuracy.correct,
            body_text_total=document_accuracy.overall_body_text_accuracy.total,
            contributors_correct=document_accuracy.overall_contributors_accuracy.correct,
            contributors_total=document_accuracy.overall_contributors_accuracy.total,
            media_links_correct=document_accuracy.overall_media_links_accuracy.correct,
            media_links_total=document_accuracy.overall_media_links_accuracy.total,
            extraction_time_seconds=processing_time,
            extraction_successful=True,
            detailed_results=self._serialize_accuracy_results(document_accuracy),
        )

        session.add(doc_evaluation)
        session.flush()  # Get the ID

        # Create article evaluations
        for article_accuracy in document_accuracy.article_accuracies:
            article_eval = ArticleEvaluation(
                document_evaluation_id=doc_evaluation.id,
                article_id=article_accuracy.article_id,
                weighted_accuracy=article_accuracy.weighted_overall_accuracy,
                title_accuracy=article_accuracy.title_accuracy.accuracy,
                body_text_accuracy=article_accuracy.body_text_accuracy.accuracy,
                contributors_accuracy=article_accuracy.contributors_accuracy.accuracy,
                media_links_accuracy=article_accuracy.media_links_accuracy.accuracy,
                body_text_wer=article_accuracy.body_text_accuracy.details.get(
                    "word_error_rate"
                ),
                body_text_meets_threshold=article_accuracy.body_text_accuracy.details.get(
                    "meets_threshold"
                ),
                contributors_found=len(
                    article_accuracy.contributors_accuracy.details.get(
                        "match_details", []
                    )
                ),
                contributors_expected=article_accuracy.contributors_accuracy.total,
                contributors_matched=article_accuracy.contributors_accuracy.correct,
                field_details=self._serialize_article_details(article_accuracy),
            )

            session.add(article_eval)

        session.commit()
        return doc_evaluation

    def _serialize_accuracy_results(
        self, document_accuracy: DocumentAccuracySchema
    ) -> Dict[str, Any]:
        """Serialize accuracy results for database storage."""

        return {
            "document_weighted_accuracy": document_accuracy.document_weighted_accuracy,
            "field_accuracies": {
                "title": {
                    "accuracy": document_accuracy.overall_title_accuracy.accuracy,
                    "correct": document_accuracy.overall_title_accuracy.correct,
                    "total": document_accuracy.overall_title_accuracy.total,
                    "details": document_accuracy.overall_title_accuracy.details,
                },
                "body_text": {
                    "accuracy": document_accuracy.overall_body_text_accuracy.accuracy,
                    "correct": document_accuracy.overall_body_text_accuracy.correct,
                    "total": document_accuracy.overall_body_text_accuracy.total,
                    "details": document_accuracy.overall_body_text_accuracy.details,
                },
                "contributors": {
                    "accuracy": document_accuracy.overall_contributors_accuracy.accuracy,
                    "correct": document_accuracy.overall_contributors_accuracy.correct,
                    "total": document_accuracy.overall_contributors_accuracy.total,
                    "details": document_accuracy.overall_contributors_accuracy.details,
                },
                "media_links": {
                    "accuracy": document_accuracy.overall_media_links_accuracy.accuracy,
                    "correct": document_accuracy.overall_media_links_accuracy.correct,
                    "total": document_accuracy.overall_media_links_accuracy.total,
                    "details": document_accuracy.overall_media_links_accuracy.details,
                },
            },
            "article_count": len(document_accuracy.article_accuracies),
        }

    def _serialize_article_details(
        self, article_accuracy: ArticleAccuracySchema
    ) -> Dict[str, Any]:
        """Serialize article accuracy details for database storage."""

        return {
            "weighted_accuracy": article_accuracy.weighted_overall_accuracy,
            "field_details": {
                "title": {
                    "accuracy": article_accuracy.title_accuracy.accuracy,
                    "details": article_accuracy.title_accuracy.details,
                },
                "body_text": {
                    "accuracy": article_accuracy.body_text_accuracy.accuracy,
                    "details": article_accuracy.body_text_accuracy.details,
                },
                "contributors": {
                    "accuracy": article_accuracy.contributors_accuracy.accuracy,
                    "details": article_accuracy.contributors_accuracy.details,
                },
                "media_links": {
                    "accuracy": article_accuracy.media_links_accuracy.accuracy,
                    "details": article_accuracy.media_links_accuracy.details,
                },
            },
        }

    def validate_xml_format(
        self, xml_content: str, xml_type: str = "ground_truth"
    ) -> Tuple[bool, List[str]]:
        """Validate XML format and structure."""

        errors = []

        try:
            root = ET.fromstring(xml_content)

            if xml_type == "ground_truth":
                # Validate ground truth XML structure
                if root.tag != "magazine_ground_truth":
                    errors.append("Root element must be 'magazine_ground_truth'")

                # Check for required elements
                if root.find("document_metadata") is None:
                    errors.append("Missing 'document_metadata' element")

                if root.find("articles") is None:
                    errors.append("Missing 'articles' element")

            elif xml_type == "extracted":
                # Validate extracted XML structure
                if not root.findall(".//article"):
                    errors.append("No articles found in extracted XML")

            return len(errors) == 0, errors

        except ET.ParseError as e:
            errors.append(f"XML parsing error: {str(e)}")
            return False, errors
        except Exception as e:
            errors.append(f"XML validation error: {str(e)}")
            return False, errors
