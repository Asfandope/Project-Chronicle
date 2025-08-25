"""
Variation engine for adding layout and content variations.
"""

import random
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import structlog

from .types import (
    LayoutElement, TextElement, ImageElement, BrandConfiguration,
    LayoutComplexity, EdgeCaseType, SyntheticDataError
)


logger = structlog.get_logger(__name__)


@dataclass
class FontVariation:
    """Font variation parameters."""
    
    font_family: str
    size_multiplier: float = 1.0
    weight_variation: str = "normal"  # normal, bold, light
    style_variation: str = "normal"   # normal, italic
    letter_spacing: float = 0.0


@dataclass
class ColorVariation:
    """Color variation parameters."""
    
    text_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    background_color: Optional[Tuple[float, float, float]] = None
    accent_color: Tuple[float, float, float] = (0.2, 0.4, 0.8)


@dataclass
class LayoutVariation:
    """Layout variation parameters."""
    
    column_count_variation: int = 0     # +/- columns from default
    margin_adjustment: float = 0.0      # +/- points from default margins
    spacing_multiplier: float = 1.0     # Multiply all spacing by this factor
    element_rotation: float = 0.0       # Degrees of rotation for elements
    overlap_probability: float = 0.0    # Probability of element overlaps


class VariationEngine:
    """
    Applies systematic variations to magazine layouts.
    
    Creates diverse layouts by varying fonts, colors, spacing, positioning,
    and other design elements while maintaining readability.
    """
    
    def __init__(self):
        self.logger = logger.bind(component="VariationEngine")
        
        # Font families for variation
        self.font_families = {
            "serif": ["Times New Roman", "Georgia", "Garamond", "Baskerville", "Minion Pro"],
            "sans_serif": ["Arial", "Helvetica", "Futura", "Avenir", "Source Sans Pro"],
            "display": ["Impact", "Franklin Gothic", "Bebas Neue", "Oswald", "Montserrat"],
            "script": ["Brush Script", "Pacifico", "Dancing Script", "Amatic SC"]
        }
        
        # Color palettes for different brand styles
        self.color_palettes = {
            "tech": [
                ((0.0, 0.2, 0.4), (0.8, 0.9, 1.0), (0.0, 0.6, 1.0)),  # Blue theme
                ((0.1, 0.1, 0.1), (0.95, 0.95, 0.95), (0.2, 0.8, 0.2)),  # Green tech
                ((0.3, 0.0, 0.5), (0.95, 0.95, 1.0), (0.6, 0.2, 0.8))   # Purple tech
            ],
            "fashion": [
                ((0.0, 0.0, 0.0), (1.0, 0.98, 0.95), (0.8, 0.2, 0.4)),  # Elegant
                ((0.3, 0.2, 0.1), (0.98, 0.95, 0.9), (0.9, 0.6, 0.3)),  # Warm
                ((0.1, 0.1, 0.1), (0.98, 0.98, 0.98), (0.7, 0.7, 0.7))  # Minimal
            ],
            "news": [
                ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.6, 0.0, 0.0)),    # Classic
                ((0.1, 0.1, 0.1), (0.95, 0.95, 0.95), (0.0, 0.4, 0.8)), # Modern
                ((0.2, 0.2, 0.2), (0.98, 0.98, 0.95), (0.8, 0.6, 0.0))  # Traditional
            ]
        }
        
        # Variation statistics
        self.variation_stats = {
            "font_variations_applied": 0,
            "color_variations_applied": 0,
            "layout_variations_applied": 0,
            "edge_case_variations_applied": 0
        }
        
        self.logger.info("Variation engine initialized")
    
    def apply_font_variations(
        self,
        elements: List[LayoutElement],
        brand_config: BrandConfiguration,
        variation_strength: float = 0.3
    ) -> List[LayoutElement]:
        """
        Apply font variations to text elements.
        
        Args:
            elements: Layout elements to modify
            brand_config: Brand configuration
            variation_strength: How much to vary (0.0 to 1.0)
            
        Returns:
            Modified layout elements
        """
        try:
            self.logger.debug("Applying font variations", 
                            elements=len(elements),
                            strength=variation_strength)
            
            varied_elements = []
            
            for element in elements:
                if isinstance(element, TextElement):
                    # Create font variation
                    font_variation = self._create_font_variation(
                        element, brand_config, variation_strength
                    )
                    
                    # Apply font variation
                    varied_element = self._apply_font_variation(element, font_variation)
                    varied_elements.append(varied_element)
                    
                    self.variation_stats["font_variations_applied"] += 1
                else:
                    varied_elements.append(element)
            
            self.logger.debug("Font variations applied",
                            text_elements_modified=sum(1 for e in varied_elements 
                                                     if isinstance(e, TextElement)))
            
            return varied_elements
            
        except Exception as e:
            self.logger.error("Error applying font variations", error=str(e))
            return elements
    
    def apply_color_variations(
        self,
        elements: List[LayoutElement],
        brand_config: BrandConfiguration,
        variation_strength: float = 0.3
    ) -> List[LayoutElement]:
        """Apply color variations to elements."""
        
        try:
            self.logger.debug("Applying color variations",
                            elements=len(elements),
                            strength=variation_strength)
            
            # Select color palette
            palette_key = brand_config.brand_style.value
            if palette_key in self.color_palettes:
                color_palette = random.choice(self.color_palettes[palette_key])
            else:
                color_palette = random.choice(self.color_palettes["news"])
            
            text_color, background_color, accent_color = color_palette
            
            varied_elements = []
            
            for element in elements:
                if isinstance(element, TextElement):
                    # Apply color variation based on semantic type
                    varied_element = self._apply_text_color_variation(
                        element, text_color, accent_color, variation_strength
                    )
                    varied_elements.append(varied_element)
                    
                    self.variation_stats["color_variations_applied"] += 1
                else:
                    varied_elements.append(element)
            
            return varied_elements
            
        except Exception as e:
            self.logger.error("Error applying color variations", error=str(e))
            return elements
    
    def apply_layout_variations(
        self,
        elements: List[LayoutElement],
        brand_config: BrandConfiguration,
        page_size: Tuple[float, float],
        variation_strength: float = 0.3
    ) -> List[LayoutElement]:
        """Apply layout variations to element positioning."""
        
        try:
            self.logger.debug("Applying layout variations",
                            elements=len(elements),
                            strength=variation_strength)
            
            # Create layout variation parameters
            layout_variation = LayoutVariation(
                margin_adjustment=random.uniform(-20, 20) * variation_strength,
                spacing_multiplier=1.0 + random.uniform(-0.3, 0.3) * variation_strength,
                element_rotation=random.uniform(-5, 5) * variation_strength,
                overlap_probability=0.05 * variation_strength
            )
            
            varied_elements = []
            
            for i, element in enumerate(elements):
                # Apply positioning variations
                varied_element = self._apply_position_variation(
                    element, layout_variation, page_size, i
                )
                varied_elements.append(varied_element)
                
                self.variation_stats["layout_variations_applied"] += 1
            
            # Resolve overlaps if they occur
            varied_elements = self._resolve_overlaps_if_needed(
                varied_elements, layout_variation.overlap_probability
            )
            
            return varied_elements
            
        except Exception as e:
            self.logger.error("Error applying layout variations", error=str(e))
            return elements
    
    def apply_edge_case_variations(
        self,
        elements: List[LayoutElement],
        edge_cases: List[EdgeCaseType],
        brand_config: BrandConfiguration,
        page_size: Tuple[float, float]
    ) -> List[LayoutElement]:
        """Apply variations specific to edge cases."""
        
        try:
            self.logger.debug("Applying edge case variations",
                            edge_cases=[ec.value for ec in edge_cases])
            
            varied_elements = list(elements)
            
            for edge_case in edge_cases:
                if edge_case == EdgeCaseType.DECORATIVE_TITLES:
                    varied_elements = self._apply_decorative_title_variation(
                        varied_elements, brand_config
                    )
                
                elif edge_case == EdgeCaseType.OVERLAPPING_ELEMENTS:
                    varied_elements = self._apply_overlapping_elements_variation(
                        varied_elements, page_size
                    )
                
                elif edge_case == EdgeCaseType.ROTATED_TEXT:
                    varied_elements = self._apply_rotated_text_variation(
                        varied_elements
                    )
                
                elif edge_case == EdgeCaseType.MULTI_COLUMN_COMPLEX:
                    varied_elements = self._apply_complex_column_variation(
                        varied_elements, page_size
                    )
                
                self.variation_stats["edge_case_variations_applied"] += 1
            
            return varied_elements
            
        except Exception as e:
            self.logger.error("Error applying edge case variations", error=str(e))
            return elements
    
    def _create_font_variation(
        self,
        element: TextElement,
        brand_config: BrandConfiguration,
        variation_strength: float
    ) -> FontVariation:
        """Create font variation for a text element."""
        
        # Select font family variation
        if element.semantic_type == "title":
            font_category = "display" if random.random() < 0.3 else "sans_serif"
        elif element.semantic_type in ["heading", "byline"]:
            font_category = "sans_serif"
        else:
            font_category = random.choice(["serif", "sans_serif"])
        
        font_family = random.choice(self.font_families[font_category])
        
        # Size variation
        if element.semantic_type == "title":
            size_multiplier = 1.0 + random.uniform(-0.2, 0.4) * variation_strength
        else:
            size_multiplier = 1.0 + random.uniform(-0.15, 0.15) * variation_strength
        
        # Weight variation
        if element.semantic_type in ["title", "heading"]:
            weight_options = ["bold", "normal"]
        else:
            weight_options = ["normal", "light"]
        
        weight_variation = random.choice(weight_options)
        
        # Style variation
        style_variation = "italic" if random.random() < 0.2 * variation_strength else "normal"
        
        return FontVariation(
            font_family=font_family,
            size_multiplier=size_multiplier,
            weight_variation=weight_variation,
            style_variation=style_variation
        )
    
    def _apply_font_variation(self, element: TextElement, variation: FontVariation) -> TextElement:
        """Apply font variation to text element."""
        
        # Create new element with variations
        varied_element = TextElement(
            element_id=element.element_id,
            element_type=element.element_type,
            bbox=element.bbox,
            page_number=element.page_number,
            text_content=element.text_content,
            font_family=variation.font_family,
            font_size=element.font_size * variation.size_multiplier,
            font_style=variation.weight_variation,
            text_color=element.text_color,
            text_align=element.text_align,
            semantic_type=element.semantic_type,
            reading_order=element.reading_order,
            extraction_difficulty=element.extraction_difficulty
        )
        
        # Adjust extraction difficulty based on variations
        if variation.font_family != element.font_family:
            varied_element.extraction_difficulty += 0.05
        
        if abs(variation.size_multiplier - 1.0) > 0.2:
            varied_element.extraction_difficulty += 0.03
        
        return varied_element
    
    def _apply_text_color_variation(
        self,
        element: TextElement,
        base_color: Tuple[float, float, float],
        accent_color: Tuple[float, float, float],
        variation_strength: float
    ) -> TextElement:
        """Apply color variation to text element."""
        
        # Select color based on semantic type
        if element.semantic_type == "title":
            text_color = accent_color if random.random() < 0.4 else base_color
        elif element.semantic_type == "heading":
            text_color = accent_color if random.random() < 0.3 else base_color
        else:
            text_color = base_color
        
        # Apply slight color variation
        varied_color = tuple(
            max(0.0, min(1.0, c + random.uniform(-0.1, 0.1) * variation_strength))
            for c in text_color
        )
        
        # Create varied element
        varied_element = TextElement(
            element_id=element.element_id,
            element_type=element.element_type,
            bbox=element.bbox,
            page_number=element.page_number,
            text_content=element.text_content,
            font_family=element.font_family,
            font_size=element.font_size,
            font_style=element.font_style,
            text_color=varied_color,
            text_align=element.text_align,
            semantic_type=element.semantic_type,
            reading_order=element.reading_order,
            extraction_difficulty=element.extraction_difficulty
        )
        
        return varied_element
    
    def _apply_position_variation(
        self,
        element: LayoutElement,
        variation: LayoutVariation,
        page_size: Tuple[float, float],
        element_index: int
    ) -> LayoutElement:
        """Apply position variation to element."""
        
        x0, y0, x1, y1 = element.bbox
        
        # Apply margin adjustments
        if x0 < 100:  # Left margin
            x0 += variation.margin_adjustment
        if x1 > page_size[0] - 100:  # Right margin
            x1 += variation.margin_adjustment
        
        # Apply spacing adjustments
        if element_index > 0:
            spacing_adjustment = (variation.spacing_multiplier - 1.0) * 10
            y0 += spacing_adjustment
            y1 += spacing_adjustment
        
        # Small random position adjustments
        position_jitter = 3.0
        x_offset = random.uniform(-position_jitter, position_jitter)
        y_offset = random.uniform(-position_jitter, position_jitter)
        
        x0 += x_offset
        x1 += x_offset
        y0 += y_offset
        y1 += y_offset
        
        # Keep within page bounds
        x0 = max(0, min(x0, page_size[0] - (x1 - x0)))
        x1 = max(x0, min(x1, page_size[0]))
        y0 = max(0, min(y0, page_size[1] - (y1 - y0)))
        y1 = max(y0, min(y1, page_size[1]))
        
        # Create varied element
        if isinstance(element, TextElement):
            varied_element = TextElement(
                element_id=element.element_id,
                element_type=element.element_type,
                bbox=(x0, y0, x1, y1),
                page_number=element.page_number,
                text_content=element.text_content,
                font_family=element.font_family,
                font_size=element.font_size,
                font_style=element.font_style,
                text_color=element.text_color,
                text_align=element.text_align,
                semantic_type=element.semantic_type,
                reading_order=element.reading_order,
                extraction_difficulty=element.extraction_difficulty
            )
        elif isinstance(element, ImageElement):
            varied_element = ImageElement(
                element_id=element.element_id,
                element_type=element.element_type,
                bbox=(x0, y0, x1, y1),
                page_number=element.page_number,
                alt_text=element.alt_text,
                width=element.width,
                height=element.height,
                extraction_difficulty=element.extraction_difficulty
            )
        else:
            # Generic layout element
            varied_element = LayoutElement(
                element_id=element.element_id,
                element_type=element.element_type,
                bbox=(x0, y0, x1, y1),
                page_number=element.page_number,
                extraction_difficulty=element.extraction_difficulty
            )
        
        return varied_element
    
    def _apply_decorative_title_variation(
        self,
        elements: List[LayoutElement],
        brand_config: BrandConfiguration
    ) -> List[LayoutElement]:
        """Apply decorative variations to title elements."""
        
        varied_elements = []
        
        for element in elements:
            if isinstance(element, TextElement) and element.semantic_type == "title":
                # Increase font size for decorative effect
                varied_element = TextElement(
                    element_id=element.element_id,
                    element_type=element.element_type,
                    bbox=element.bbox,
                    page_number=element.page_number,
                    text_content=element.text_content,
                    font_family="Impact",  # More decorative font
                    font_size=element.font_size * 1.3,
                    font_style="bold",
                    text_color=brand_config.accent_color,
                    text_align=element.text_align,
                    semantic_type=element.semantic_type,
                    reading_order=element.reading_order,
                    extraction_difficulty=element.extraction_difficulty * 1.4  # Harder to extract
                )
                varied_elements.append(varied_element)
            else:
                varied_elements.append(element)
        
        return varied_elements
    
    def _apply_overlapping_elements_variation(
        self,
        elements: List[LayoutElement],
        page_size: Tuple[float, float]
    ) -> List[LayoutElement]:
        """Create intentional element overlaps."""
        
        if len(elements) < 2:
            return elements
        
        # Select 2-3 elements to overlap
        overlap_candidates = random.sample(elements, min(3, len(elements)))
        
        # Move elements closer to create overlaps
        for i, element in enumerate(overlap_candidates[1:], 1):
            reference_element = overlap_candidates[i-1]
            
            # Move element to partially overlap with reference
            ref_x0, ref_y0, ref_x1, ref_y1 = reference_element.bbox
            elem_x0, elem_y0, elem_x1, elem_y1 = element.bbox
            
            # Calculate overlap position
            overlap_x = ref_x1 - (ref_x1 - ref_x0) * 0.3
            overlap_y = ref_y1 - 10
            
            new_bbox = (
                overlap_x,
                overlap_y,
                overlap_x + (elem_x1 - elem_x0),
                overlap_y + (elem_y1 - elem_y0)
            )
            
            # Update element bbox
            element.bbox = new_bbox
            element.extraction_difficulty *= 1.5  # Much harder to extract
        
        return elements
    
    def _apply_rotated_text_variation(self, elements: List[LayoutElement]) -> List[LayoutElement]:
        """Apply rotation effect to some text elements."""
        
        varied_elements = []
        
        for element in elements:
            if (isinstance(element, TextElement) and 
                element.semantic_type in ["sidebar", "caption", "byline"] and
                random.random() < 0.3):
                
                # Simulate rotation by adjusting bbox and difficulty
                x0, y0, x1, y1 = element.bbox
                
                # For rotated text, swap width/height proportions
                width = x1 - x0
                height = y1 - y0
                
                # Adjust to simulate 90-degree rotation
                new_width = height * 0.8
                new_height = width * 1.2
                
                rotated_bbox = (x0, y0, x0 + new_width, y0 + new_height)
                
                varied_element = TextElement(
                    element_id=element.element_id,
                    element_type=element.element_type,
                    bbox=rotated_bbox,
                    page_number=element.page_number,
                    text_content=element.text_content,
                    font_family=element.font_family,
                    font_size=element.font_size,
                    font_style=element.font_style,
                    text_color=element.text_color,
                    text_align=element.text_align,
                    semantic_type=element.semantic_type,
                    reading_order=element.reading_order,
                    extraction_difficulty=element.extraction_difficulty * 1.6  # Much harder
                )
                
                varied_elements.append(varied_element)
            else:
                varied_elements.append(element)
        
        return varied_elements
    
    def _apply_complex_column_variation(
        self,
        elements: List[LayoutElement],
        page_size: Tuple[float, float]
    ) -> List[LayoutElement]:
        """Create complex multi-column layouts."""
        
        # This is a simplified version - in practice would reorganize entire layout
        varied_elements = []
        
        # Split elements into multiple column groups
        text_elements = [e for e in elements if isinstance(e, TextElement)]
        other_elements = [e for e in elements if not isinstance(e, TextElement)]
        
        if len(text_elements) >= 4:
            # Rearrange text elements into more complex column structure
            columns = 3
            column_width = (page_size[0] - 120) / columns  # Account for margins
            gutter = 20
            
            for i, element in enumerate(text_elements):
                column_idx = i % columns
                column_x = 60 + column_idx * (column_width + gutter)
                
                # Adjust element position to fit column
                x0, y0, x1, y1 = element.bbox
                height = y1 - y0
                
                new_bbox = (column_x, y0, column_x + column_width, y0 + height)
                element.bbox = new_bbox
                element.extraction_difficulty *= 1.2  # Slightly harder
                
                varied_elements.append(element)
        else:
            varied_elements.extend(text_elements)
        
        varied_elements.extend(other_elements)
        return varied_elements
    
    def _resolve_overlaps_if_needed(
        self,
        elements: List[LayoutElement],
        overlap_probability: float
    ) -> List[LayoutElement]:
        """Resolve overlaps unless they're intentional."""
        
        if overlap_probability > 0.1:
            # Overlaps are intentional, keep them
            return elements
        
        # Simple overlap resolution
        for i, element in enumerate(elements):
            for j, other_element in enumerate(elements[:i]):
                if element.overlaps(other_element):
                    # Move element down to avoid overlap
                    x0, y0, x1, y1 = element.bbox
                    offset = other_element.bbox[3] - y0 + 5
                    element.bbox = (x0, y0 + offset, x1, y1 + offset)
        
        return elements
    
    def get_variation_statistics(self) -> Dict[str, Any]:
        """Get variation statistics."""
        
        return {
            "font_variations_applied": self.variation_stats["font_variations_applied"],
            "color_variations_applied": self.variation_stats["color_variations_applied"],
            "layout_variations_applied": self.variation_stats["layout_variations_applied"],
            "edge_case_variations_applied": self.variation_stats["edge_case_variations_applied"],
            "total_variations": sum(self.variation_stats.values()),
            "font_families_available": sum(len(fonts) for fonts in self.font_families.values()),
            "color_palettes_available": sum(len(palettes) for palettes in self.color_palettes.values())
        }