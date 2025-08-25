"""
XML schema validation using lxml.
"""

import time
from pathlib import Path
from typing import Optional
import structlog

try:
    from lxml import etree
    LXML_AVAILABLE = True
except ImportError:
    LXML_AVAILABLE = False
    etree = None

from .types import XMLConfig, ValidationResult, ValidationError


logger = structlog.get_logger(__name__)


class SchemaValidator:
    """
    XML schema validator using lxml for constrained generation.
    
    Provides high-performance schema validation with detailed error reporting.
    """
    
    def __init__(self, config: Optional[XMLConfig] = None):
        """
        Initialize schema validator.
        
        Args:
            config: XML configuration
        """
        if not LXML_AVAILABLE:
            raise ImportError("lxml is required for XML validation")
            
        self.config = config or XMLConfig()
        self.logger = logger.bind(component="SchemaValidator")
        
        # Validation state
        self._schema = None
        self._schema_path = None
        self._validation_stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "average_validation_time": 0.0
        }
        
        # Load schema
        self._load_schema()
        
        self.logger.info("Schema validator initialized")
    
    def _load_schema(self) -> None:
        """Load XSD schema for validation."""
        try:
            # Determine schema path
            if self.config.schema_location:
                schema_path = self.config.schema_location
            else:
                # Use default schema location
                schema_path = Path(__file__).parent.parent.parent / "schemas" / "article-v1.0.xsd"
            
            if not schema_path.exists():
                raise FileNotFoundError(f"Schema file not found: {schema_path}")
            
            self.logger.info("Loading XML schema", schema_path=str(schema_path))
            
            # Parse and compile schema
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_doc = etree.parse(f)
            
            self._schema = etree.XMLSchema(schema_doc)
            self._schema_path = schema_path
            
            self.logger.info("Schema loaded successfully", 
                           schema_version=self.config.schema_version.value)
            
        except Exception as e:
            self.logger.error("Failed to load schema", error=str(e))
            raise ValidationError(f"Failed to load schema: {e}")
    
    def validate_xml_string(self, xml_content: str) -> ValidationResult:
        """
        Validate XML content against schema.
        
        Args:
            xml_content: XML content as string
            
        Returns:
            Validation result with detailed error information
        """
        if not self._schema:
            raise ValidationError("Schema not loaded")
        
        start_time = time.time()
        
        try:
            self.logger.debug("Starting XML validation", 
                            content_length=len(xml_content))
            
            # Parse XML document
            try:
                xml_doc = etree.fromstring(xml_content.encode('utf-8'))
            except etree.XMLSyntaxError as e:
                return self._create_parse_error_result(e, time.time() - start_time)
            
            # Validate against schema
            result = ValidationResult(is_valid=True)
            
            if not self._schema.validate(xml_doc):
                result.is_valid = False
                
                # Collect detailed error information
                for error in self._schema.error_log:
                    result.add_error(
                        message=error.message,
                        line=error.line,
                        column=error.column
                    )
                    
                    self.logger.debug("Validation error",
                                    message=error.message,
                                    line=error.line,
                                    column=error.column)
            
            # Additional custom validations
            if self.config.strict_validation:
                self._perform_strict_validation(xml_doc, result)
            
            result.validation_time = time.time() - start_time
            
            # Update statistics
            self._update_validation_stats(result)
            
            self.logger.info("XML validation completed",
                           is_valid=result.is_valid,
                           errors=len(result.errors),
                           warnings=len(result.warnings),
                           validation_time=result.validation_time)
            
            return result
            
        except Exception as e:
            self.logger.error("Error during validation", error=str(e))
            result = ValidationResult(is_valid=False)
            result.add_error(f"Validation error: {e}")
            result.validation_time = time.time() - start_time
            return result
    
    def validate_xml_file(self, xml_path: Path) -> ValidationResult:
        """
        Validate XML file against schema.
        
        Args:
            xml_path: Path to XML file
            
        Returns:
            Validation result
        """
        try:
            with open(xml_path, 'r', encoding='utf-8') as f:
                xml_content = f.read()
            
            return self.validate_xml_string(xml_content)
            
        except Exception as e:
            self.logger.error("Error reading XML file", path=str(xml_path), error=str(e))
            result = ValidationResult(is_valid=False)
            result.add_error(f"Failed to read XML file: {e}")
            return result
    
    def _create_parse_error_result(self, parse_error: etree.XMLSyntaxError, validation_time: float) -> ValidationResult:
        """Create validation result for XML parse errors."""
        result = ValidationResult(is_valid=False)
        result.add_error(
            message=f"XML parsing error: {parse_error.msg}",
            line=parse_error.lineno,
            column=parse_error.offset
        )
        result.validation_time = validation_time
        
        self.logger.warning("XML parse error",
                          message=parse_error.msg,
                          line=parse_error.lineno,
                          column=parse_error.offset)
        
        return result
    
    def _perform_strict_validation(self, xml_doc: etree.Element, result: ValidationResult) -> None:
        """Perform additional strict validation checks."""
        
        # Check confidence score ranges
        confidence_attrs = [
            'confidence', 'extraction_confidence', 'pairing_confidence',
            'normalization_confidence', 'overall_confidence'
        ]
        
        for elem in xml_doc.iter():
            for attr_name in confidence_attrs:
                if attr_name in elem.attrib:
                    try:
                        confidence_value = float(elem.attrib[attr_name])
                        if not (0.0 <= confidence_value <= 1.0):
                            result.add_warning(
                                f"Confidence score {confidence_value} out of range [0.0, 1.0] "
                                f"in element {elem.tag}"
                            )
                    except ValueError:
                        result.add_error(
                            f"Invalid confidence score format in {elem.tag}.{attr_name}: "
                            f"{elem.attrib[attr_name]}"
                        )
        
        # Check required IDs are present and unique
        ids_found = set()
        for elem in xml_doc.iter():
            if 'id' in elem.attrib:
                elem_id = elem.attrib['id']
                if elem_id in ids_found:
                    result.add_error(f"Duplicate ID found: {elem_id}")
                else:
                    ids_found.add(elem_id)
        
        # Check bounding box format
        for elem in xml_doc.iter():
            if 'bbox' in elem.attrib:
                bbox_str = elem.attrib['bbox']
                try:
                    # Should be "x0,y0,x1,y1" format
                    coords = [float(x.strip()) for x in bbox_str.split(',')]
                    if len(coords) != 4:
                        result.add_error(f"Invalid bounding box format in {elem.tag}: {bbox_str}")
                    elif coords[2] <= coords[0] or coords[3] <= coords[1]:
                        result.add_warning(f"Invalid bounding box coordinates in {elem.tag}: {bbox_str}")
                except (ValueError, AttributeError):
                    result.add_error(f"Invalid bounding box format in {elem.tag}: {bbox_str}")
    
    def _update_validation_stats(self, result: ValidationResult) -> None:
        """Update validation statistics."""
        self._validation_stats["total_validations"] += 1
        
        if result.is_valid:
            self._validation_stats["successful_validations"] += 1
        else:
            self._validation_stats["failed_validations"] += 1
        
        # Update average validation time
        total = self._validation_stats["total_validations"]
        current_avg = self._validation_stats["average_validation_time"]
        new_avg = ((current_avg * (total - 1)) + result.validation_time) / total
        self._validation_stats["average_validation_time"] = new_avg
    
    def get_schema_info(self) -> dict:
        """Get information about the loaded schema."""
        return {
            "schema_path": str(self._schema_path) if self._schema_path else None,
            "schema_version": self.config.schema_version.value,
            "target_namespace": self.config.target_namespace,
            "validation_stats": self._validation_stats.copy(),
            "strict_validation_enabled": self.config.strict_validation
        }
    
    def reload_schema(self) -> None:
        """Reload the schema (useful for development)."""
        self.logger.info("Reloading XML schema")
        self._schema = None
        self._load_schema()
    
    def is_schema_loaded(self) -> bool:
        """Check if schema is loaded and ready for validation."""
        return self._schema is not None