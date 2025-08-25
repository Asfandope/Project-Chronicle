"""
Main article to XML converter with constrained generation.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import structlog

try:
    from lxml import etree
    LXML_AVAILABLE = True
except ImportError:
    LXML_AVAILABLE = False
    etree = None

from .validator import SchemaValidator
from .formatter import XMLFormatter
from .types import (
    XMLConfig, FormattingOptions, ArticleData, ConversionResult,
    ValidationResult, XMLError
)


logger = structlog.get_logger(__name__)


class ArticleXMLConverter:
    """
    Main converter for article data to canonical XML with schema validation.
    
    Provides high-quality XML output with confidence scores, deterministic
    formatting, and comprehensive validation against article-v1.0.xsd schema.
    """
    
    def __init__(self, config: Optional[XMLConfig] = None):
        """
        Initialize article XML converter.
        
        Args:
            config: XML configuration
        """
        if not LXML_AVAILABLE:
            raise ImportError("lxml is required for XML conversion")
            
        self.config = config or XMLConfig()
        self.logger = logger.bind(component="ArticleXMLConverter")
        
        # Initialize components
        self.validator = SchemaValidator(self.config) if self.config.validate_output else None
        self.formatter = XMLFormatter(self.config, FormattingOptions())
        
        # Namespace setup
        self.namespace_map = {}
        if self.config.use_namespaces:
            self.namespace_map[self.config.namespace_prefix] = self.config.target_namespace
        
        # Conversion statistics
        self.stats = {
            "total_conversions": 0,
            "successful_conversions": 0,
            "failed_conversions": 0,
            "total_processing_time": 0.0,
            "elements_created": 0,
            "confidence_scores_added": 0
        }
        
        self.logger.info("Article XML converter initialized",
                        validate_output=self.config.validate_output,
                        schema_version=self.config.schema_version.value)
    
    def convert_article(self, article_data: Union[ArticleData, Dict[str, Any]]) -> ConversionResult:
        """
        Convert article data to XML format.
        
        Args:
            article_data: Article data to convert (ArticleData or dict)
            
        Returns:
            Conversion result with XML content and validation
        """
        try:
            start_time = time.time()
            
            self.logger.info("Starting article XML conversion",
                           article_id=self._get_article_id(article_data))
            
            # Validate input data
            if isinstance(article_data, dict):
                article_data = self._dict_to_article_data(article_data)
            
            data_errors = article_data.validate_data()
            if data_errors:
                raise XMLError(f"Invalid article data: {', '.join(data_errors)}")
            
            # Create XML document
            root = self._create_xml_document(article_data)
            
            # Convert to string with formatting
            xml_content = self._serialize_xml(root)
            
            # Validate against schema if enabled
            validation_result = ValidationResult(is_valid=True)
            if self.validator:
                validation_result = self.validator.validate_xml_string(xml_content)
            
            # Create conversion result
            conversion_time = time.time() - start_time
            result = ConversionResult(
                xml_content=xml_content,
                validation_result=validation_result,
                conversion_time=conversion_time,
                elements_created=self.stats["elements_created"],
                attributes_added=self._count_attributes_in_xml(xml_content),
                confidence_scores_included=self.stats["confidence_scores_added"]
            )
            
            # Update statistics
            self._update_conversion_stats(result, conversion_time)
            
            self.logger.info("Article XML conversion completed",
                           article_id=article_data.article_id,
                           is_successful=result.is_successful,
                           conversion_time=conversion_time,
                           elements_created=result.elements_created,
                           validation_passed=validation_result.is_valid)
            
            return result
            
        except Exception as e:
            self.logger.error("Error in article XML conversion", error=str(e), exc_info=True)
            self.stats["failed_conversions"] += 1
            
            # Return failed conversion result
            return ConversionResult(
                xml_content="",
                validation_result=ValidationResult(is_valid=False, errors=[str(e)]),
                conversion_time=time.time() - start_time
            )
    
    def _create_xml_document(self, article_data: ArticleData) -> etree.Element:
        """Create XML document structure from article data."""
        
        # Create root element with namespace
        if self.config.use_namespaces:
            root = etree.Element(
                f"{{{self.config.target_namespace}}}article",
                nsmap={self.config.namespace_prefix: self.config.target_namespace}
            )
        else:
            root = etree.Element("article")
        
        # Add root attributes
        root.set("id", article_data.article_id)
        root.set("brand", article_data.brand)
        root.set("issue", article_data.issue_date.strftime("%Y-%m-%d"))
        root.set("page_start", str(article_data.page_start))
        root.set("page_end", str(article_data.page_end))
        
        self.stats["elements_created"] += 1
        
        # Add title
        title_elem = self._create_title_element(article_data)
        root.append(title_elem)
        
        # Add contributors if present
        if article_data.contributors:
            contributors_elem = self._create_contributors_element(article_data.contributors)
            root.append(contributors_elem)
        
        # Add body content
        body_elem = self._create_body_element(article_data.text_blocks)
        root.append(body_elem)
        
        # Add media if present
        if article_data.images:
            media_elem = self._create_media_element(article_data.images)
            root.append(media_elem)
        
        # Add provenance information
        provenance_elem = self._create_provenance_element(article_data)
        root.append(provenance_elem)
        
        return root
    
    def _create_title_element(self, article_data: ArticleData) -> etree.Element:
        """Create title element with confidence."""
        title_elem = etree.Element("title")
        title_elem.text = article_data.title
        
        confidence_str = self._format_confidence(article_data.title_confidence)
        title_elem.set("confidence", confidence_str)
        
        self.stats["elements_created"] += 1
        self.stats["confidence_scores_added"] += 1
        
        return title_elem
    
    def _create_contributors_element(self, contributors_data: List[Dict[str, Any]]) -> etree.Element:
        """Create contributors element with individual contributor entries."""
        contributors_elem = etree.Element("contributors")
        
        for contributor_data in contributors_data:
            contributor_elem = etree.Element("contributor")
            
            # Add contributor attributes
            if "role" in contributor_data:
                contributor_elem.set("role", str(contributor_data["role"]))
            
            if "confidence" in contributor_data:
                confidence_str = self._format_confidence(contributor_data["confidence"])
                contributor_elem.set("confidence", confidence_str)
                self.stats["confidence_scores_added"] += 1
            
            # Add name elements
            if "name" in contributor_data:
                name_elem = etree.Element("name")
                name_elem.text = str(contributor_data["name"])
                contributor_elem.append(name_elem)
                self.stats["elements_created"] += 1
            
            if "normalized_name" in contributor_data:
                normalized_elem = etree.Element("normalized_name")
                normalized_elem.text = str(contributor_data["normalized_name"])
                contributor_elem.append(normalized_elem)
                self.stats["elements_created"] += 1
            
            contributors_elem.append(contributor_elem)
            self.stats["elements_created"] += 1
        
        self.stats["elements_created"] += 1
        return contributors_elem
    
    def _create_body_element(self, text_blocks: List[Dict[str, Any]]) -> etree.Element:
        """Create body element with text blocks."""
        body_elem = etree.Element("body")
        
        for block_data in text_blocks:
            block_type = block_data.get("type", "paragraph")
            
            if block_type == "paragraph":
                block_elem = etree.Element("paragraph")
            elif block_type == "pullquote":
                block_elem = etree.Element("pullquote")
            else:
                block_elem = etree.Element("paragraph")  # Default fallback
            
            # Add text content
            if "text" in block_data:
                block_elem.text = str(block_data["text"])
            
            # Add confidence if available
            if "confidence" in block_data:
                confidence_str = self._format_confidence(block_data["confidence"])
                block_elem.set("confidence", confidence_str)
                self.stats["confidence_scores_added"] += 1
            
            # Add other attributes
            for attr in ["id", "page", "position"]:
                if attr in block_data:
                    block_elem.set(attr, str(block_data[attr]))
            
            body_elem.append(block_elem)
            self.stats["elements_created"] += 1
        
        self.stats["elements_created"] += 1
        return body_elem
    
    def _create_media_element(self, images_data: List[Dict[str, Any]]) -> etree.Element:
        """Create media element with image entries."""
        media_elem = etree.Element("media")
        
        for image_data in images_data:
            image_elem = etree.Element("image")
            
            # Required src attribute
            if "filename" in image_data:
                image_elem.set("src", str(image_data["filename"]))
            elif "src" in image_data:
                image_elem.set("src", str(image_data["src"]))
            
            # Add confidence
            if "confidence" in image_data:
                confidence_str = self._format_confidence(image_data["confidence"])
                image_elem.set("confidence", confidence_str)
                self.stats["confidence_scores_added"] += 1
            elif "pairing_confidence" in image_data:
                confidence_str = self._format_confidence(image_data["pairing_confidence"])
                image_elem.set("confidence", confidence_str)
                self.stats["confidence_scores_added"] += 1
            
            # Add caption if present
            if "caption" in image_data and image_data["caption"]:
                caption_elem = etree.Element("caption")
                caption_elem.text = str(image_data["caption"])
                image_elem.append(caption_elem)
                self.stats["elements_created"] += 1
            
            # Add credit if present
            if "credit" in image_data and image_data["credit"]:
                credit_elem = etree.Element("credit")
                credit_elem.text = str(image_data["credit"])
                image_elem.append(credit_elem)
                self.stats["elements_created"] += 1
            
            media_elem.append(image_elem)
            self.stats["elements_created"] += 1
        
        self.stats["elements_created"] += 1
        return media_elem
    
    def _create_provenance_element(self, article_data: ArticleData) -> etree.Element:
        """Create provenance element with processing metadata."""
        provenance_elem = etree.Element("provenance")
        
        # Extraction timestamp
        extracted_at_elem = etree.Element("extracted_at")
        extracted_at_elem.text = article_data.extraction_timestamp.isoformat()
        provenance_elem.append(extracted_at_elem)
        
        # Model version
        model_version_elem = etree.Element("model_version")
        model_version_elem.text = article_data.processing_pipeline_version
        provenance_elem.append(model_version_elem)
        
        # Overall confidence
        confidence_elem = etree.Element("confidence_overall")
        confidence_str = self._format_confidence(article_data.extraction_confidence)
        confidence_elem.text = confidence_str
        provenance_elem.append(confidence_elem)
        
        self.stats["elements_created"] += 3  # provenance + 3 child elements
        self.stats["confidence_scores_added"] += 1
        
        return provenance_elem
    
    def _format_confidence(self, confidence: float) -> str:
        """Format confidence score to consistent precision."""
        if not isinstance(confidence, (int, float)):
            return "0.000"
        
        # Clamp to valid range
        confidence = max(0.0, min(1.0, float(confidence)))
        
        # Format with configured precision
        return f"{confidence:.{self.config.confidence_precision}f}"
    
    def _serialize_xml(self, root: etree.Element) -> str:
        """Serialize XML element to string with formatting."""
        
        # Convert to string first
        xml_bytes = etree.tostring(
            root,
            method='xml',
            xml_declaration=self.config.xml_declaration,
            encoding=self.config.encoding,
            pretty_print=False  # We'll handle formatting separately
        )
        
        xml_content = xml_bytes.decode(self.config.encoding)
        
        # Apply formatting
        formatted_xml = self.formatter.format_xml(xml_content)
        
        return formatted_xml
    
    def _dict_to_article_data(self, data_dict: Dict[str, Any]) -> ArticleData:
        """Convert dictionary to ArticleData object."""
        
        # Extract required fields
        article_id = data_dict.get("id") or data_dict.get("article_id", "")
        title = data_dict.get("title", "")
        title_confidence = float(data_dict.get("title_confidence", 1.0))
        brand = data_dict.get("brand", "")
        
        # Parse issue date
        issue_date = data_dict.get("issue_date")
        if isinstance(issue_date, str):
            try:
                issue_date = datetime.fromisoformat(issue_date.replace('Z', '+00:00'))
            except:
                issue_date = datetime.now()
        elif not isinstance(issue_date, datetime):
            issue_date = datetime.now()
        
        # Extract page info
        page_start = int(data_dict.get("page_start", 1))
        page_end = int(data_dict.get("page_end", page_start))
        
        # Create ArticleData object
        article_data = ArticleData(
            article_id=article_id,
            title=title,
            title_confidence=title_confidence,
            brand=brand,
            issue_date=issue_date,
            page_start=page_start,
            page_end=page_end,
            contributors=data_dict.get("contributors", []),
            text_blocks=data_dict.get("text_blocks", []),
            images=data_dict.get("images", []),
            extraction_confidence=float(data_dict.get("extraction_confidence", 1.0)),
            processing_pipeline_version=data_dict.get("processing_pipeline_version", "1.0")
        )
        
        return article_data
    
    def _get_article_id(self, article_data: Union[ArticleData, Dict[str, Any]]) -> str:
        """Get article ID from data."""
        if isinstance(article_data, ArticleData):
            return article_data.article_id
        else:
            return article_data.get("id") or article_data.get("article_id", "unknown")
    
    def _count_attributes_in_xml(self, xml_content: str) -> int:
        """Count total attributes in XML content."""
        # Simple regex-based counting (could be more sophisticated)
        import re
        attr_pattern = r'\s+\w+\s*=\s*["\'][^"\']*["\']'
        return len(re.findall(attr_pattern, xml_content))
    
    def _update_conversion_stats(self, result: ConversionResult, processing_time: float) -> None:
        """Update conversion statistics."""
        self.stats["total_conversions"] += 1
        self.stats["total_processing_time"] += processing_time
        
        if result.is_successful:
            self.stats["successful_conversions"] += 1
        else:
            self.stats["failed_conversions"] += 1
    
    def get_conversion_statistics(self) -> Dict[str, Any]:
        """Get conversion performance statistics."""
        total_conversions = self.stats["total_conversions"]
        
        return {
            "total_conversions": total_conversions,
            "successful_conversions": self.stats["successful_conversions"],
            "failed_conversions": self.stats["failed_conversions"],
            "success_rate": (
                self.stats["successful_conversions"] / max(1, total_conversions)
            ),
            "average_processing_time": (
                self.stats["total_processing_time"] / max(1, total_conversions)
            ),
            "total_elements_created": self.stats["elements_created"],
            "total_confidence_scores": self.stats["confidence_scores_added"],
            "schema_validation_enabled": self.validator is not None,
            "output_format": self.config.output_format.value
        }
    
    def validate_xml_string(self, xml_content: str) -> ValidationResult:
        """Validate XML string against schema."""
        if not self.validator:
            return ValidationResult(is_valid=True, warnings=["Validation disabled"])
        
        return self.validator.validate_xml_string(xml_content)
    
    def reload_schema(self) -> None:
        """Reload XML schema (useful for development)."""
        if self.validator:
            self.validator.reload_schema()
            self.logger.info("Schema reloaded successfully")