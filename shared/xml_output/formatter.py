"""
XML formatting and canonicalization for deterministic output.
"""

import re
import time
from typing import Dict, List, Optional
import structlog

try:
    from lxml import etree
    LXML_AVAILABLE = True
except ImportError:
    LXML_AVAILABLE = False
    etree = None

from .types import XMLConfig, FormattingOptions, OutputFormat, FormattingError


logger = structlog.get_logger(__name__)


class XMLFormatter:
    """
    XML formatter that ensures deterministic, canonical output.
    
    Provides consistent formatting with sorted attributes and elements
    for reliable, reproducible XML generation.
    """
    
    def __init__(self, config: Optional[XMLConfig] = None, options: Optional[FormattingOptions] = None):
        """
        Initialize XML formatter.
        
        Args:
            config: XML configuration
            options: Formatting options
        """
        if not LXML_AVAILABLE:
            raise ImportError("lxml is required for XML formatting")
            
        self.config = config or XMLConfig()
        self.options = options or FormattingOptions()
        self.logger = logger.bind(component="XMLFormatter")
        
        # Formatting statistics
        self.stats = {
            "documents_formatted": 0,
            "elements_sorted": 0,
            "attributes_sorted": 0,
            "whitespace_normalized": 0
        }
        
        self.logger.info("XML formatter initialized", 
                        output_format=self.config.output_format.value)
    
    def format_xml(self, xml_content: str) -> str:
        """
        Format XML content according to configuration.
        
        Args:
            xml_content: Raw XML content
            
        Returns:
            Formatted XML content
        """
        try:
            self.logger.debug("Starting XML formatting", 
                            content_length=len(xml_content),
                            format_type=self.config.output_format.value)
            
            # Parse XML
            try:
                root = etree.fromstring(xml_content.encode('utf-8'))
            except etree.XMLSyntaxError as e:
                raise FormattingError(f"Invalid XML content: {e}")
            
            # Apply formatting based on output format
            if self.config.output_format == OutputFormat.CANONICAL:
                formatted_xml = self._format_canonical(root)
            elif self.config.output_format == OutputFormat.PRETTY:
                formatted_xml = self._format_pretty(root)
            elif self.config.output_format == OutputFormat.DEBUG:
                formatted_xml = self._format_debug(root)
            else:  # COMPACT
                formatted_xml = self._format_compact(root)
            
            # Post-processing
            formatted_xml = self._post_process(formatted_xml)
            
            # Update statistics
            self.stats["documents_formatted"] += 1
            
            self.logger.debug("XML formatting completed",
                            original_length=len(xml_content),
                            formatted_length=len(formatted_xml))
            
            return formatted_xml
            
        except Exception as e:
            self.logger.error("Error formatting XML", error=str(e))
            raise FormattingError(f"XML formatting failed: {e}")
    
    def _format_canonical(self, root: etree.Element) -> str:
        """Format XML in canonical form (C14N)."""
        # Apply deterministic ordering
        self._sort_elements_recursively(root)
        self._sort_attributes_recursively(root)
        self._normalize_whitespace_recursively(root)
        
        # Use C14N canonicalization
        try:
            xml_bytes = etree.tostring(
                root,
                method='c14n',
                xml_declaration=self.config.xml_declaration,
                encoding=self.config.encoding
            )
            return xml_bytes.decode(self.config.encoding)
        except Exception:
            # Fallback to manual canonicalization
            return self._manual_canonicalization(root)
    
    def _format_pretty(self, root: etree.Element) -> str:
        """Format XML with pretty-printing."""
        # Apply sorting and normalization
        if self.config.sort_elements:
            self._sort_elements_recursively(root)
        
        if self.config.sort_attributes:
            self._sort_attributes_recursively(root)
        
        if self.config.normalize_whitespace:
            self._normalize_whitespace_recursively(root)
        
        # Pretty print with indentation
        etree.indent(root, space=self.options.indent_char * self.options.indent_size)
        
        xml_bytes = etree.tostring(
            root,
            pretty_print=True,
            xml_declaration=self.config.xml_declaration,
            encoding=self.config.encoding
        )
        
        return xml_bytes.decode(self.config.encoding)
    
    def _format_debug(self, root: etree.Element) -> str:
        """Format XML with debug information."""
        # Add debug attributes if configured
        if self.config.include_debug_info:
            self._add_debug_attributes(root)
        
        # Apply pretty formatting
        return self._format_pretty(root)
    
    def _format_compact(self, root: etree.Element) -> str:
        """Format XML in compact form."""
        # Apply sorting for deterministic output
        if self.config.sort_elements:
            self._sort_elements_recursively(root)
        
        if self.config.sort_attributes:
            self._sort_attributes_recursively(root)
        
        # Remove unnecessary whitespace
        self._remove_whitespace_recursively(root)
        
        xml_bytes = etree.tostring(
            root,
            method='xml',
            xml_declaration=self.config.xml_declaration,
            encoding=self.config.encoding
        )
        
        return xml_bytes.decode(self.config.encoding)
    
    def _sort_elements_recursively(self, element: etree.Element) -> None:
        """Sort child elements deterministically."""
        if not self.config.sort_elements:
            return
        
        # Get all child elements
        children = list(element)
        if len(children) <= 1:
            return
        
        # Sort by tag name, then by id attribute if present
        def sort_key(elem):
            tag_priority = self._get_tag_priority(elem.tag)
            elem_id = elem.get('id', '')
            return (tag_priority, elem.tag, elem_id)
        
        sorted_children = sorted(children, key=sort_key)
        
        # Clear and re-add in sorted order
        element[:] = sorted_children
        
        # Recursively sort child elements
        for child in element:
            self._sort_elements_recursively(child)
        
        self.stats["elements_sorted"] += len(children)
    
    def _sort_attributes_recursively(self, element: etree.Element) -> None:
        """Sort attributes deterministically."""
        if not self.config.sort_attributes:
            return
        
        # Sort attributes according to priority order
        if element.attrib:
            sorted_attrs = self._sort_attributes_dict(element.attrib)
            element.attrib.clear()
            element.attrib.update(sorted_attrs)
            
            self.stats["attributes_sorted"] += len(sorted_attrs)
        
        # Recursively sort attributes in child elements
        for child in element:
            self._sort_attributes_recursively(child)
    
    def _sort_attributes_dict(self, attrs: Dict[str, str]) -> Dict[str, str]:
        """Sort attributes dictionary according to priority order."""
        # Create priority mapping
        priority_map = {attr: i for i, attr in enumerate(self.options.attribute_sort_order)}
        
        def attr_sort_key(item):
            attr_name, attr_value = item
            priority = priority_map.get(attr_name, 1000)  # Unknown attributes go last
            return (priority, attr_name)
        
        return dict(sorted(attrs.items(), key=attr_sort_key))
    
    def _normalize_whitespace_recursively(self, element: etree.Element) -> None:
        """Normalize whitespace in text content."""
        if not self.config.normalize_whitespace:
            return
        
        # Normalize text content
        if element.text:
            if self.options.trim_text_content:
                element.text = element.text.strip()
            if self.options.normalize_line_endings:
                element.text = re.sub(r'\r\n|\r', '\n', element.text)
        
        # Normalize tail content
        if element.tail:
            if self.options.trim_text_content:
                element.tail = element.tail.strip()
            if self.options.normalize_line_endings:
                element.tail = re.sub(r'\r\n|\r', '\n', element.tail)
        
        # Recursively normalize child elements
        for child in element:
            self._normalize_whitespace_recursively(child)
        
        self.stats["whitespace_normalized"] += 1
    
    def _remove_whitespace_recursively(self, element: etree.Element) -> None:
        """Remove unnecessary whitespace for compact output."""
        # Remove whitespace-only text content
        if element.text and element.text.isspace():
            element.text = None
        
        if element.tail and element.tail.isspace():
            element.tail = None
        
        # Recursively process child elements
        for child in element:
            self._remove_whitespace_recursively(child)
    
    def _add_debug_attributes(self, element: etree.Element) -> None:
        """Add debug attributes to elements."""
        # Add debug info to root element
        if element.getparent() is None:  # Root element
            element.set('debug_formatted_at', str(int(time.time())))
            element.set('debug_formatter_version', '1.0')
        
        # Add element count to containers
        child_count = len(list(element))
        if child_count > 0:
            element.set('debug_child_count', str(child_count))
        
        # Recursively add debug info
        for child in element:
            self._add_debug_attributes(child)
    
    def _get_tag_priority(self, tag: str) -> int:
        """Get sorting priority for element tags."""
        # Remove namespace prefix if present
        local_tag = tag.split('}')[-1] if '}' in tag else tag
        
        # Define tag priority order
        tag_priorities = {
            'metadata': 1,
            'title': 2,
            'contributors': 3,
            'contributor': 4,
            'content': 5,
            'text_blocks': 6,
            'block': 7,
            'media': 8,
            'images': 9,
            'image': 10,
            'layout': 11,
            'processing': 12
        }
        
        return tag_priorities.get(local_tag, 100)  # Default priority for unknown tags
    
    def _manual_canonicalization(self, root: etree.Element) -> str:
        """Manual canonicalization when C14N is not available."""
        # This is a simplified canonicalization
        # In production, should use proper C14N implementation
        
        xml_bytes = etree.tostring(
            root,
            method='xml',
            xml_declaration=self.config.xml_declaration,
            encoding=self.config.encoding
        )
        
        return xml_bytes.decode(self.config.encoding)
    
    def _post_process(self, xml_content: str) -> str:
        """Apply post-processing to formatted XML."""
        
        # Ensure consistent line endings
        if self.options.normalize_line_endings:
            xml_content = re.sub(r'\r\n|\r', '\n', xml_content)
        
        # Ensure file ends with newline for POSIX compliance
        if not xml_content.endswith('\n'):
            xml_content += '\n'
        
        # Apply quote character consistency
        if self.options.quote_char == "'":
            # Convert double quotes to single quotes in attributes
            xml_content = re.sub(r'="([^"]*)"', r"='\1'", xml_content)
        
        return xml_content
    
    def get_formatting_stats(self) -> Dict[str, int]:
        """Get formatting statistics."""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """Reset formatting statistics."""
        for key in self.stats:
            self.stats[key] = 0