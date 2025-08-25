"""
Ground truth generation for synthetic test data.

This module creates XML ground truth files that provide the known correct
extraction results for testing the magazine extraction pipeline.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import xml.etree.ElementTree as ET
from xml.dom import minidom
import json
import uuid

from .types import (
    ArticleData, GroundTruthData, TextElement, ImageElement,
    BrandConfiguration, GenerationConfig, EdgeCaseType,
    LayoutComplexity, SyntheticDataError
)


class GroundTruthGenerator:
    """Generates ground truth XML files for synthetic test data."""
    
    def __init__(self):
        self.confidence_calculator = ConfidenceCalculator()
    
    def generate_ground_truth(
        self,
        articles: List[ArticleData],
        brand_config: BrandConfiguration,
        generation_config: Optional[GenerationConfig] = None,
        document_id: Optional[str] = None
    ) -> GroundTruthData:
        """Generate complete ground truth data structure."""
        
        if not document_id:
            document_id = f"{brand_config.brand_name}_{uuid.uuid4().hex[:8]}"
        
        # Collect all elements
        all_text_elements = []
        all_image_elements = []
        edge_cases_present = set()
        
        for article in articles:
            all_text_elements.extend(article.text_elements)
            all_image_elements.extend(article.image_elements)
            edge_cases_present.update(article.edge_cases)
        
        # Calculate page count and dimensions
        max_page = max([elem.page_number for elem in all_text_elements + all_image_elements], default=1)
        page_dimensions = (612.0, 792.0)  # Default US Letter
        
        # Calculate difficulty metrics
        difficult_elements_count = sum(
            1 for elem in all_text_elements + all_image_elements
            if elem.extraction_difficulty > 0.5
        )
        
        # Calculate expected accuracy
        expected_accuracy = self.confidence_calculator.calculate_document_confidence(
            all_text_elements + all_image_elements,
            list(edge_cases_present)
        )
        
        return GroundTruthData(
            document_id=document_id,
            brand_name=brand_config.brand_name,
            generation_timestamp=datetime.now(),
            articles=articles,
            all_text_elements=all_text_elements,
            all_image_elements=all_image_elements,
            page_count=max_page,
            page_dimensions=page_dimensions,
            generation_config=generation_config,
            brand_config=brand_config,
            expected_extraction_accuracy=expected_accuracy,
            difficult_elements_count=difficult_elements_count,
            edge_cases_present=list(edge_cases_present)
        )
    
    def export_to_xml(
        self,
        ground_truth: GroundTruthData,
        output_path: Path,
        format_version: str = "1.0"
    ) -> bool:
        """Export ground truth to XML format."""
        try:
            root = ET.Element("magazine_ground_truth")
            root.set("version", format_version)
            root.set("generator", "synthetic_data_generator")
            
            # Document metadata
            doc_meta = ET.SubElement(root, "document_metadata")
            ET.SubElement(doc_meta, "document_id").text = ground_truth.document_id
            ET.SubElement(doc_meta, "brand_name").text = ground_truth.brand_name
            ET.SubElement(doc_meta, "generation_timestamp").text = ground_truth.generation_timestamp.isoformat()
            ET.SubElement(doc_meta, "page_count").text = str(ground_truth.page_count)
            
            # Page dimensions
            page_dims = ET.SubElement(doc_meta, "page_dimensions")
            page_dims.set("width", str(ground_truth.page_dimensions[0]))
            page_dims.set("height", str(ground_truth.page_dimensions[1]))
            page_dims.set("units", "points")
            
            # Quality metrics
            quality = ET.SubElement(doc_meta, "quality_metrics")
            ET.SubElement(quality, "expected_accuracy").text = str(ground_truth.expected_extraction_accuracy)
            ET.SubElement(quality, "difficult_elements").text = str(ground_truth.difficult_elements_count)
            
            # Edge cases
            if ground_truth.edge_cases_present:
                edge_cases = ET.SubElement(doc_meta, "edge_cases")
                for edge_case in ground_truth.edge_cases_present:
                    case_elem = ET.SubElement(edge_cases, "edge_case")
                    case_elem.set("type", edge_case.value if hasattr(edge_case, 'value') else str(edge_case))
            
            # Articles
            articles_elem = ET.SubElement(root, "articles")
            for article in ground_truth.articles:
                self._add_article_to_xml(articles_elem, article)
            
            # All elements (for easier processing)
            elements_elem = ET.SubElement(root, "all_elements")
            
            # Text elements
            text_elements = ET.SubElement(elements_elem, "text_elements")
            for text_elem in ground_truth.all_text_elements:
                self._add_text_element_to_xml(text_elements, text_elem)
            
            # Image elements
            image_elements = ET.SubElement(elements_elem, "image_elements")
            for img_elem in ground_truth.all_image_elements:
                self._add_image_element_to_xml(image_elements, img_elem)
            
            # Write to file with pretty formatting
            xml_string = ET.tostring(root, encoding='unicode')
            parsed = minidom.parseString(xml_string)
            pretty_xml = parsed.toprettyxml(indent="  ")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(pretty_xml)
            
            return True
            
        except Exception as e:
            raise SyntheticDataError(f"Failed to export ground truth XML: {str(e)}")
    
    def _add_article_to_xml(self, parent: ET.Element, article: ArticleData) -> None:
        """Add article data to XML."""
        article_elem = ET.SubElement(parent, "article")
        article_elem.set("id", article.article_id)
        
        # Basic info
        ET.SubElement(article_elem, "title").text = article.title
        ET.SubElement(article_elem, "article_type").text = article.article_type
        
        # Page range
        page_range = ET.SubElement(article_elem, "page_range")
        page_range.set("start", str(article.page_range[0]))
        page_range.set("end", str(article.page_range[1]))
        
        if article.is_split_article:
            page_range.set("split_article", "true")
            if article.continuation_pages:
                page_range.set("continuation_pages", ",".join(map(str, article.continuation_pages)))
        
        # Contributors
        if article.contributors:
            contributors = ET.SubElement(article_elem, "contributors")
            for contributor in article.contributors:
                contrib_elem = ET.SubElement(contributors, "contributor")
                contrib_elem.set("role", contributor.get("role", "author"))
                contrib_elem.set("name", contributor.get("name", ""))
                if "affiliation" in contributor:
                    contrib_elem.set("affiliation", contributor["affiliation"])
        
        # Complexity and edge cases
        complexity = ET.SubElement(article_elem, "complexity")
        complexity.set("level", article.complexity_level.value)
        
        if article.edge_cases:
            edge_cases = ET.SubElement(article_elem, "edge_cases")
            for edge_case in article.edge_cases:
                case_elem = ET.SubElement(edge_cases, "edge_case")
                case_elem.set("type", edge_case.value)
        
        # Text elements for this article
        if article.text_elements:
            text_elements = ET.SubElement(article_elem, "text_elements")
            for text_elem in article.text_elements:
                self._add_text_element_to_xml(text_elements, text_elem)
        
        # Image elements for this article
        if article.image_elements:
            image_elements = ET.SubElement(article_elem, "image_elements")
            for img_elem in article.image_elements:
                self._add_image_element_to_xml(image_elements, img_elem)
    
    def _add_text_element_to_xml(self, parent: ET.Element, element: TextElement) -> None:
        """Add text element to XML."""
        elem = ET.SubElement(parent, "text_element")
        elem.set("id", element.element_id)
        elem.set("type", element.semantic_type)
        elem.set("page", str(element.page_number))
        elem.set("reading_order", str(element.reading_order))
        
        # Bounding box
        bbox = ET.SubElement(elem, "bbox")
        x0, y0, x1, y1 = element.bbox
        bbox.set("x0", str(x0))
        bbox.set("y0", str(y0))
        bbox.set("x1", str(x1))
        bbox.set("y1", str(y1))
        
        # Text content
        content = ET.SubElement(elem, "content")
        content.text = element.text_content
        
        # Font information
        font = ET.SubElement(elem, "font")
        font.set("family", element.font_family)
        font.set("size", str(element.font_size))
        font.set("style", element.font_style)
        font.set("align", element.text_align)
        
        # Color
        color = ET.SubElement(elem, "color")
        r, g, b = element.text_color
        color.set("r", str(r))
        color.set("g", str(g))
        color.set("b", str(b))
        
        # Extraction metadata
        extraction = ET.SubElement(elem, "extraction_metadata")
        extraction.set("confidence", str(element.confidence))
        extraction.set("difficulty", str(element.extraction_difficulty))
        extraction.set("z_order", str(element.z_order))
    
    def _add_image_element_to_xml(self, parent: ET.Element, element: ImageElement) -> None:
        """Add image element to XML."""
        elem = ET.SubElement(parent, "image_element")
        elem.set("id", element.element_id)
        elem.set("page", str(element.page_number))
        
        # Bounding box
        bbox = ET.SubElement(elem, "bbox")
        x0, y0, x1, y1 = element.bbox
        bbox.set("x0", str(x0))
        bbox.set("y0", str(y0))
        bbox.set("x1", str(x1))
        bbox.set("y1", str(y1))
        
        # Image properties
        props = ET.SubElement(elem, "image_properties")
        props.set("width", str(element.width))
        props.set("height", str(element.height))
        props.set("dpi", str(element.dpi))
        props.set("color_space", element.color_space)
        
        # Alt text
        if element.alt_text:
            ET.SubElement(elem, "alt_text").text = element.alt_text
        
        # File reference
        if element.image_path:
            ET.SubElement(elem, "image_path").text = str(element.image_path)
        
        # Extraction metadata
        extraction = ET.SubElement(elem, "extraction_metadata")
        extraction.set("confidence", str(element.confidence))
        extraction.set("difficulty", str(element.extraction_difficulty))
        extraction.set("z_order", str(element.z_order))
    
    def export_to_json(self, ground_truth: GroundTruthData, output_path: Path) -> bool:
        """Export ground truth to JSON format."""
        try:
            data = {
                "document_id": ground_truth.document_id,
                "brand_name": ground_truth.brand_name,
                "generation_timestamp": ground_truth.generation_timestamp.isoformat(),
                "page_count": ground_truth.page_count,
                "page_dimensions": {
                    "width": ground_truth.page_dimensions[0],
                    "height": ground_truth.page_dimensions[1],
                    "units": "points"
                },
                "quality_metrics": {
                    "expected_accuracy": ground_truth.expected_extraction_accuracy,
                    "difficult_elements": ground_truth.difficult_elements_count
                },
                "edge_cases": [
                    edge_case.value if hasattr(edge_case, 'value') else str(edge_case)
                    for edge_case in ground_truth.edge_cases_present
                ],
                "articles": [self._article_to_dict(article) for article in ground_truth.articles],
                "all_text_elements": [self._text_element_to_dict(elem) for elem in ground_truth.all_text_elements],
                "all_image_elements": [self._image_element_to_dict(elem) for elem in ground_truth.all_image_elements]
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            raise SyntheticDataError(f"Failed to export ground truth JSON: {str(e)}")
    
    def _article_to_dict(self, article: ArticleData) -> Dict[str, Any]:
        """Convert article to dictionary."""
        return {
            "article_id": article.article_id,
            "title": article.title,
            "article_type": article.article_type,
            "page_range": {"start": article.page_range[0], "end": article.page_range[1]},
            "is_split_article": article.is_split_article,
            "continuation_pages": article.continuation_pages,
            "complexity_level": article.complexity_level.value,
            "edge_cases": [ec.value for ec in article.edge_cases],
            "contributors": article.contributors,
            "text_elements": [self._text_element_to_dict(elem) for elem in article.text_elements],
            "image_elements": [self._image_element_to_dict(elem) for elem in article.image_elements]
        }
    
    def _text_element_to_dict(self, element: TextElement) -> Dict[str, Any]:
        """Convert text element to dictionary."""
        return {
            "element_id": element.element_id,
            "element_type": element.element_type,
            "semantic_type": element.semantic_type,
            "page_number": element.page_number,
            "reading_order": element.reading_order,
            "bbox": {"x0": element.bbox[0], "y0": element.bbox[1], "x1": element.bbox[2], "y1": element.bbox[3]},
            "text_content": element.text_content,
            "font": {
                "family": element.font_family,
                "size": element.font_size,
                "style": element.font_style,
                "align": element.text_align
            },
            "color": {"r": element.text_color[0], "g": element.text_color[1], "b": element.text_color[2]},
            "extraction_metadata": {
                "confidence": element.confidence,
                "difficulty": element.extraction_difficulty,
                "z_order": element.z_order
            }
        }
    
    def _image_element_to_dict(self, element: ImageElement) -> Dict[str, Any]:
        """Convert image element to dictionary."""
        return {
            "element_id": element.element_id,
            "element_type": element.element_type,
            "page_number": element.page_number,
            "bbox": {"x0": element.bbox[0], "y0": element.bbox[1], "x1": element.bbox[2], "y1": element.bbox[3]},
            "image_properties": {
                "width": element.width,
                "height": element.height,
                "dpi": element.dpi,
                "color_space": element.color_space
            },
            "alt_text": element.alt_text,
            "image_path": str(element.image_path) if element.image_path else None,
            "extraction_metadata": {
                "confidence": element.confidence,
                "difficulty": element.extraction_difficulty,
                "z_order": element.z_order
            }
        }


class ConfidenceCalculator:
    """Calculates extraction confidence scores for elements and documents."""
    
    def __init__(self):
        # Base confidence penalties for different edge cases
        self.edge_case_penalties = {
            EdgeCaseType.SPLIT_ARTICLES: 0.15,
            EdgeCaseType.DECORATIVE_TITLES: 0.20,
            EdgeCaseType.MULTI_COLUMN_COMPLEX: 0.10,
            EdgeCaseType.OVERLAPPING_ELEMENTS: 0.25,
            EdgeCaseType.ROTATED_TEXT: 0.30,
            EdgeCaseType.WATERMARKS: 0.15,
            EdgeCaseType.ADVERTISEMENTS: 0.10,
            EdgeCaseType.CAPTION_AMBIGUITY: 0.20,
            EdgeCaseType.CONTRIBUTOR_COMPLEXITY: 0.15,
            EdgeCaseType.MIXED_LANGUAGES: 0.25
        }
    
    def calculate_element_confidence(
        self,
        element: Any,
        overlapping_elements: List[Any] = None,
        edge_cases: List[EdgeCaseType] = None
    ) -> float:
        """Calculate confidence score for individual element."""
        base_confidence = 1.0
        
        # Apply penalties based on element characteristics
        if hasattr(element, 'font_size'):
            # Very small text is harder to extract
            if element.font_size < 8:
                base_confidence -= 0.2
            elif element.font_size < 10:
                base_confidence -= 0.1
        
        # Check for overlaps
        if overlapping_elements:
            overlap_penalty = min(0.3, len(overlapping_elements) * 0.1)
            base_confidence -= overlap_penalty
        
        # Apply edge case penalties
        if edge_cases:
            for edge_case in edge_cases:
                penalty = self.edge_case_penalties.get(edge_case, 0.1)
                base_confidence -= penalty
        
        # Element-specific factors
        if hasattr(element, 'semantic_type'):
            if element.semantic_type in ['caption', 'byline']:
                # These are typically harder to extract correctly
                base_confidence -= 0.1
            elif element.semantic_type in ['title', 'heading']:
                # Titles are usually easier
                base_confidence += 0.05
        
        # Apply existing extraction difficulty
        if hasattr(element, 'extraction_difficulty'):
            base_confidence -= element.extraction_difficulty * 0.5
        
        return max(0.1, min(1.0, base_confidence))
    
    def calculate_document_confidence(
        self,
        elements: List[Any],
        edge_cases: List[EdgeCaseType] = None
    ) -> float:
        """Calculate overall document extraction confidence."""
        if not elements:
            return 1.0
        
        element_confidences = []
        
        for element in elements:
            # Find overlapping elements
            overlapping = [
                other for other in elements 
                if other != element and hasattr(element, 'overlaps') and element.overlaps(other)
            ]
            
            confidence = self.calculate_element_confidence(
                element, 
                overlapping, 
                edge_cases
            )
            element_confidences.append(confidence)
        
        # Weighted average based on element importance
        total_confidence = 0.0
        total_weight = 0.0
        
        for i, confidence in enumerate(element_confidences):
            element = elements[i]
            
            # Weight elements by importance
            weight = 1.0
            if hasattr(element, 'semantic_type'):
                if element.semantic_type == 'title':
                    weight = 2.0
                elif element.semantic_type in ['heading', 'byline']:
                    weight = 1.5
                elif element.semantic_type in ['caption', 'pullquote']:
                    weight = 0.8
            
            total_confidence += confidence * weight
            total_weight += weight
        
        document_confidence = total_confidence / max(1.0, total_weight)
        
        # Apply global edge case penalty
        if edge_cases:
            global_penalty = min(0.2, len(edge_cases) * 0.05)
            document_confidence -= global_penalty
        
        return max(0.1, min(1.0, document_confidence))
    
    def calculate_difficulty_distribution(
        self,
        elements: List[Any]
    ) -> Dict[str, int]:
        """Calculate distribution of extraction difficulties."""
        distribution = {"easy": 0, "medium": 0, "hard": 0, "very_hard": 0}
        
        for element in elements:
            difficulty = getattr(element, 'extraction_difficulty', 0.0)
            
            if difficulty < 0.2:
                distribution["easy"] += 1
            elif difficulty < 0.5:
                distribution["medium"] += 1
            elif difficulty < 0.8:
                distribution["hard"] += 1
            else:
                distribution["very_hard"] += 1
        
        return distribution


def create_test_ground_truth(output_dir: Path) -> Path:
    """Create a test ground truth file."""
    from .types import BrandConfiguration, TextElement, ArticleData
    
    # Create sample elements
    title_element = TextElement(
        element_id="test_title",
        element_type="text",
        bbox=(72, 650, 540, 700),
        page_number=1,
        text_content="Test Article Title",
        font_family="Helvetica",
        font_size=24,
        font_style="bold",
        semantic_type="title",
        confidence=0.95,
        extraction_difficulty=0.1
    )
    
    body_element = TextElement(
        element_id="test_body",
        element_type="text",
        bbox=(72, 400, 540, 600),
        page_number=1,
        text_content="Test body content for ground truth validation.",
        font_family="Times New Roman",
        font_size=12,
        semantic_type="paragraph",
        confidence=0.9,
        extraction_difficulty=0.2
    )
    
    # Create sample article
    test_article = ArticleData(
        article_id="test_001",
        title="Test Article Title",
        text_elements=[title_element, body_element]
    )
    
    # Generate ground truth
    generator = GroundTruthGenerator()
    brand_config = BrandConfiguration.create_tech_magazine()
    
    ground_truth = generator.generate_ground_truth(
        [test_article],
        brand_config,
        document_id="test_document"
    )
    
    # Export to XML
    xml_path = output_dir / "test_ground_truth.xml"
    generator.export_to_xml(ground_truth, xml_path)
    
    return xml_path