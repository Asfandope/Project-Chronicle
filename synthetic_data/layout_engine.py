"""
Layout engine for generating realistic magazine layouts.
"""

import random
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import structlog

from .types import (
    LayoutElement, TextElement, ImageElement, BrandConfiguration,
    LayoutComplexity, EdgeCaseType, SyntheticDataError
)


logger = structlog.get_logger(__name__)


@dataclass
class ColumnLayout:
    """Column layout configuration."""
    
    column_count: int
    column_width: float
    gutter_width: float
    columns: List[Tuple[float, float]]  # (left_x, right_x) for each column
    
    @classmethod
    def create_layout(cls, page_width: float, margins: Tuple[float, float], 
                      column_count: int, gutter_width: float) -> "ColumnLayout":
        """Create column layout from page parameters."""
        
        left_margin, right_margin = margins
        available_width = page_width - left_margin - right_margin
        
        if column_count == 1:
            gutter_width = 0
        
        total_gutter = gutter_width * (column_count - 1)
        column_width = (available_width - total_gutter) / column_count
        
        columns = []
        current_x = left_margin
        
        for i in range(column_count):
            columns.append((current_x, current_x + column_width))
            current_x += column_width + gutter_width
        
        return cls(
            column_count=column_count,
            column_width=column_width,
            gutter_width=gutter_width,
            columns=columns
        )


@dataclass
class MagazineStyle:
    """Magazine visual style configuration."""
    
    # Typography hierarchy
    title_font_size: float = 24.0
    subtitle_font_size: float = 18.0
    heading_font_size: float = 16.0
    body_font_size: float = 11.0
    caption_font_size: float = 9.0
    byline_font_size: float = 10.0
    
    # Spacing
    title_spacing: float = 20.0
    paragraph_spacing: float = 12.0
    line_height_multiplier: float = 1.2
    
    # Visual elements
    use_drop_caps: bool = False
    use_pull_quotes: bool = True
    use_decorative_elements: bool = False
    
    # Layout preferences
    preferred_image_sizes: List[Tuple[float, float]] = field(default_factory=lambda: [
        (200, 150), (300, 200), (400, 300), (150, 200)  # width, height
    ])
    
    @classmethod
    def create_tech_style(cls) -> "MagazineStyle":
        """Create tech magazine style."""
        return cls(
            title_font_size=28.0,
            body_font_size=10.5,
            use_drop_caps=False,
            use_pull_quotes=True,
            preferred_image_sizes=[
                (250, 180), (320, 240), (400, 250), (180, 120)
            ]
        )
    
    @classmethod
    def create_fashion_style(cls) -> "MagazineStyle":
        """Create fashion magazine style."""
        return cls(
            title_font_size=32.0,
            body_font_size=11.5,
            use_drop_caps=True,
            use_pull_quotes=True,
            use_decorative_elements=True,
            preferred_image_sizes=[
                (300, 400), (250, 350), (400, 500), (200, 300)
            ]
        )


@dataclass
class LayoutTemplate:
    """Template for magazine page layout."""
    
    template_name: str
    complexity: LayoutComplexity
    column_configurations: List[int]  # Possible column counts
    
    # Element placement rules
    title_placement: str = "top"  # top, center, custom
    image_placement_options: List[str] = field(default_factory=lambda: ["top", "middle", "bottom", "side"])
    
    # Layout constraints
    min_text_height: float = 100.0
    min_image_size: Tuple[float, float] = (100.0, 100.0)
    max_elements_per_page: int = 15
    
    # Edge case support
    supports_split_articles: bool = True
    supports_decorative_titles: bool = False
    supports_overlapping_elements: bool = False


class LayoutEngine:
    """
    Generates realistic magazine layouts with configurable complexity.
    
    Creates authentic-looking magazine pages with proper typography,
    column layouts, and element positioning.
    """
    
    def __init__(self):
        self.logger = logger.bind(component="LayoutEngine")
        
        # Standard page sizes (width, height in points)
        self.page_sizes = {
            "US_LETTER": (612.0, 792.0),
            "A4": (595.276, 841.89),
            "TABLOID": (792.0, 1224.0),
            "MAGAZINE": (540.0, 720.0)  # Typical magazine size
        }
        
        # Initialize layout templates
        self.templates = self._create_layout_templates()
        
        # Layout state
        self._current_y_position = {}  # Track y position per column
        self._element_counter = 0
        
        self.logger.info("Layout engine initialized", templates=len(self.templates))
    
    def _create_layout_templates(self) -> Dict[str, LayoutTemplate]:
        """Create predefined layout templates."""
        
        templates = {}
        
        # Simple layouts
        templates["single_column"] = LayoutTemplate(
            template_name="single_column",
            complexity=LayoutComplexity.SIMPLE,
            column_configurations=[1],
            supports_decorative_titles=False,
            supports_overlapping_elements=False
        )
        
        templates["two_column"] = LayoutTemplate(
            template_name="two_column", 
            complexity=LayoutComplexity.SIMPLE,
            column_configurations=[2],
            image_placement_options=["top", "middle", "side"]
        )
        
        # Moderate complexity
        templates["magazine_standard"] = LayoutTemplate(
            template_name="magazine_standard",
            complexity=LayoutComplexity.MODERATE,
            column_configurations=[2, 3],
            supports_decorative_titles=True,
            image_placement_options=["top", "middle", "bottom", "side", "wrapped"]
        )
        
        templates["news_layout"] = LayoutTemplate(
            template_name="news_layout",
            complexity=LayoutComplexity.MODERATE,
            column_configurations=[3, 4],
            supports_split_articles=True
        )
        
        # Complex layouts
        templates["fashion_spread"] = LayoutTemplate(
            template_name="fashion_spread",
            complexity=LayoutComplexity.COMPLEX,
            column_configurations=[1, 2, 3],
            supports_decorative_titles=True,
            supports_overlapping_elements=True,
            image_placement_options=["full_page", "wrapped", "side", "overlay"]
        )
        
        templates["tech_feature"] = LayoutTemplate(
            template_name="tech_feature",
            complexity=LayoutComplexity.COMPLEX,
            column_configurations=[2, 3, 4],
            max_elements_per_page=20
        )
        
        # Chaotic layouts for stress testing
        templates["experimental"] = LayoutTemplate(
            template_name="experimental",
            complexity=LayoutComplexity.CHAOTIC,
            column_configurations=[1, 2, 3, 4, 5],
            supports_decorative_titles=True,
            supports_overlapping_elements=True,
            image_placement_options=["random", "overlay", "rotated", "scattered"]
        )
        
        return templates
    
    def generate_page_layout(
        self, 
        brand_config: BrandConfiguration,
        magazine_style: MagazineStyle,
        page_number: int,
        content_requirements: Dict[str, Any],
        complexity: Optional[LayoutComplexity] = None,
        edge_cases: Optional[List[EdgeCaseType]] = None
    ) -> Tuple[ColumnLayout, List[LayoutElement]]:
        """
        Generate a complete page layout.
        
        Args:
            brand_config: Brand configuration
            magazine_style: Visual style
            page_number: Page number in document
            content_requirements: Required content (text blocks, images, etc.)
            complexity: Desired complexity level
            edge_cases: Edge cases to include
            
        Returns:
            Tuple of (column_layout, layout_elements)
        """
        try:
            self.logger.debug("Generating page layout", 
                            page=page_number, 
                            complexity=complexity.value if complexity else "auto")
            
            # Select appropriate template
            template = self._select_template(complexity, edge_cases or [])
            
            # Determine page size
            page_size = self.page_sizes.get(brand_config.brand_style.value.upper() + "_SIZE", 
                                          self.page_sizes["MAGAZINE"])
            
            # Create column layout
            column_count = random.choice(template.column_configurations)
            if hasattr(brand_config, 'default_columns'):
                column_count = brand_config.default_columns
            
            column_layout = ColumnLayout.create_layout(
                page_width=page_size[0],
                margins=(brand_config.margin_left, brand_config.margin_right),
                column_count=column_count,
                gutter_width=brand_config.column_gap
            )
            
            # Generate layout elements
            elements = self._generate_layout_elements(
                template, column_layout, page_size, brand_config, 
                magazine_style, page_number, content_requirements, edge_cases or []
            )
            
            self.logger.info("Page layout generated",
                           page=page_number,
                           template=template.template_name,
                           columns=column_count,
                           elements=len(elements))
            
            return column_layout, elements
            
        except Exception as e:
            self.logger.error("Error generating page layout", error=str(e))
            raise SyntheticDataError(f"Failed to generate page layout: {e}")
    
    def _select_template(self, complexity: Optional[LayoutComplexity], 
                        edge_cases: List[EdgeCaseType]) -> LayoutTemplate:
        """Select appropriate layout template."""
        
        if complexity is None:
            complexity = random.choice(list(LayoutComplexity))
        
        # Filter templates by complexity
        suitable_templates = [
            template for template in self.templates.values()
            if template.complexity == complexity
        ]
        
        # Filter by edge case support if needed
        if edge_cases:
            filtered_templates = []
            for template in suitable_templates:
                if EdgeCaseType.DECORATIVE_TITLES in edge_cases and not template.supports_decorative_titles:
                    continue
                if EdgeCaseType.OVERLAPPING_ELEMENTS in edge_cases and not template.supports_overlapping_elements:
                    continue
                if EdgeCaseType.SPLIT_ARTICLES in edge_cases and not template.supports_split_articles:
                    continue
                filtered_templates.append(template)
            
            if filtered_templates:
                suitable_templates = filtered_templates
        
        if not suitable_templates:
            # Fallback to any template
            suitable_templates = list(self.templates.values())
        
        return random.choice(suitable_templates)
    
    def _generate_layout_elements(
        self,
        template: LayoutTemplate,
        column_layout: ColumnLayout, 
        page_size: Tuple[float, float],
        brand_config: BrandConfiguration,
        magazine_style: MagazineStyle,
        page_number: int,
        content_requirements: Dict[str, Any],
        edge_cases: List[EdgeCaseType]
    ) -> List[LayoutElement]:
        """Generate layout elements for the page."""
        
        elements = []
        page_width, page_height = page_size
        
        # Initialize column tracking
        column_y_positions = [brand_config.margin_top] * column_layout.column_count
        
        # Generate title element
        if content_requirements.get("title"):
            title_element = self._create_title_element(
                content_requirements["title"],
                column_layout, page_size, brand_config, magazine_style,
                page_number, EdgeCaseType.DECORATIVE_TITLES in edge_cases
            )
            elements.append(title_element)
            
            # Update column positions after title
            title_bottom = title_element.bbox[3] + magazine_style.title_spacing
            column_y_positions = [max(pos, title_bottom) for pos in column_y_positions]
        
        # Generate contributor/byline elements
        if content_requirements.get("contributors"):
            byline_elements = self._create_byline_elements(
                content_requirements["contributors"],
                column_layout, column_y_positions, brand_config, magazine_style,
                page_number, EdgeCaseType.CONTRIBUTOR_COMPLEXITY in edge_cases
            )
            elements.extend(byline_elements)
            
            # Update column positions
            if byline_elements:
                byline_bottom = max(elem.bbox[3] for elem in byline_elements) + 10
                column_y_positions = [max(pos, byline_bottom) for pos in column_y_positions]
        
        # Generate text content elements
        text_blocks = content_requirements.get("text_blocks", [])
        for i, text_block in enumerate(text_blocks):
            # Select column for this text block
            column_idx = self._select_text_column(i, column_layout, column_y_positions)
            
            text_element = self._create_text_element(
                text_block, column_idx, column_layout, column_y_positions,
                brand_config, magazine_style, page_number
            )
            elements.append(text_element)
            
            # Update column position
            column_y_positions[column_idx] = text_element.bbox[3] + magazine_style.paragraph_spacing
        
        # Generate image elements
        images = content_requirements.get("images", [])
        for i, image_info in enumerate(images):
            # Determine image placement
            placement_style = self._select_image_placement(template, i, len(images))
            
            image_element = self._create_image_element(
                image_info, placement_style, column_layout, column_y_positions,
                page_size, brand_config, magazine_style, page_number,
                EdgeCaseType.CAPTION_AMBIGUITY in edge_cases
            )
            elements.append(image_element)
            
            # Create caption if present
            if image_info.get("caption"):
                caption_element = self._create_caption_element(
                    image_info["caption"], image_element, column_layout,
                    brand_config, magazine_style, page_number
                )
                elements.append(caption_element)
        
        # Add edge case elements
        edge_case_elements = self._generate_edge_case_elements(
            edge_cases, column_layout, page_size, brand_config, 
            magazine_style, page_number
        )
        elements.extend(edge_case_elements)
        
        # Apply final positioning and overlap resolution
        elements = self._finalize_element_positions(elements, template, page_size)
        
        return elements
    
    def _create_title_element(
        self, title_text: str, column_layout: ColumnLayout,
        page_size: Tuple[float, float], brand_config: BrandConfiguration,
        magazine_style: MagazineStyle, page_number: int, 
        decorative: bool = False
    ) -> TextElement:
        """Create title element with optional decorative styling."""
        
        font_size = magazine_style.title_font_size
        if decorative:
            font_size *= random.uniform(1.2, 1.8)
        
        # Position title across columns or centered
        if decorative or column_layout.column_count == 1:
            # Full width title
            left_x = brand_config.margin_left
            right_x = page_size[0] - brand_config.margin_right
        else:
            # Span 2-3 columns
            span_columns = min(3, column_layout.column_count)
            left_x = column_layout.columns[0][0]
            right_x = column_layout.columns[span_columns - 1][1]
        
        # Calculate title height
        title_height = font_size * 1.5
        if decorative:
            title_height *= random.uniform(1.3, 2.0)
        
        top_y = brand_config.margin_top
        bottom_y = top_y + title_height
        
        title_element = TextElement(
            element_id=f"title_{page_number}_{self._next_element_id()}",
            element_type="text",
            bbox=(left_x, top_y, right_x, bottom_y),
            page_number=page_number,
            text_content=title_text,
            font_family=brand_config.title_font,
            font_size=font_size,
            font_style="bold" if not decorative else random.choice(["bold", "bold-italic"]),
            semantic_type="title",
            reading_order=1,
            extraction_difficulty=0.3 if decorative else 0.1
        )
        
        return title_element
    
    def _create_byline_elements(
        self, contributors: List[Dict[str, Any]], column_layout: ColumnLayout,
        column_y_positions: List[float], brand_config: BrandConfiguration,
        magazine_style: MagazineStyle, page_number: int, 
        complex_bylines: bool = False
    ) -> List[TextElement]:
        """Create byline elements for contributors."""
        
        byline_elements = []
        
        if complex_bylines:
            # Create complex bylines with multiple formats
            byline_formats = [
                "By {name}",
                "Written by {name}",
                "{name}, Staff Writer", 
                "{role}: {name}",
                "{name} | {role}"
            ]
        else:
            byline_formats = ["By {name}", "Written by {name}"]
        
        for i, contributor in enumerate(contributors):
            byline_format = random.choice(byline_formats)
            
            # Format byline text
            byline_text = byline_format.format(
                name=contributor.get("name", "Unknown"),
                role=contributor.get("role", "Writer").title()
            )
            
            # Position byline
            column_idx = 0  # Usually in first column
            left_x, right_x = column_layout.columns[column_idx]
            top_y = column_y_positions[column_idx]
            
            byline_height = magazine_style.byline_font_size * 1.3
            bottom_y = top_y + byline_height
            
            byline_element = TextElement(
                element_id=f"byline_{page_number}_{i}_{self._next_element_id()}",
                element_type="text",
                bbox=(left_x, top_y, right_x, bottom_y),
                page_number=page_number,
                text_content=byline_text,
                font_family=brand_config.secondary_font,
                font_size=magazine_style.byline_font_size,
                font_style="italic",
                semantic_type="byline",
                reading_order=2 + i,
                extraction_difficulty=0.2 if complex_bylines else 0.1
            )
            
            byline_elements.append(byline_element)
            
            # Update position for next byline
            column_y_positions[column_idx] = bottom_y + 5
        
        return byline_elements
    
    def _create_text_element(
        self, text_block: Dict[str, Any], column_idx: int,
        column_layout: ColumnLayout, column_y_positions: List[float],
        brand_config: BrandConfiguration, magazine_style: MagazineStyle,
        page_number: int
    ) -> TextElement:
        """Create text element for paragraph content."""
        
        text_content = text_block.get("text", "")
        text_type = text_block.get("type", "paragraph")
        
        # Select font size based on text type
        if text_type == "pullquote":
            font_size = magazine_style.heading_font_size
            font_style = "italic"
        elif text_type == "heading":
            font_size = magazine_style.heading_font_size
            font_style = "bold"
        else:
            font_size = magazine_style.body_font_size
            font_style = "normal"
        
        # Calculate text height (rough estimation)
        line_height = font_size * magazine_style.line_height_multiplier
        char_per_line = int((column_layout.column_width - 20) / (font_size * 0.6))
        lines_needed = max(1, len(text_content) // char_per_line)
        text_height = lines_needed * line_height
        
        # Position in selected column
        left_x, right_x = column_layout.columns[column_idx]
        top_y = column_y_positions[column_idx]
        bottom_y = top_y + text_height
        
        text_element = TextElement(
            element_id=f"text_{page_number}_{column_idx}_{self._next_element_id()}",
            element_type="text",
            bbox=(left_x, top_y, right_x, bottom_y),
            page_number=page_number,
            text_content=text_content,
            font_family=brand_config.primary_font,
            font_size=font_size,
            font_style=font_style,
            semantic_type=text_type,
            reading_order=10 + column_idx * 100 + len(column_y_positions),
            extraction_difficulty=0.05 if text_type == "paragraph" else 0.15
        )
        
        return text_element
    
    def _create_image_element(
        self, image_info: Dict[str, Any], placement_style: str,
        column_layout: ColumnLayout, column_y_positions: List[float],
        page_size: Tuple[float, float], brand_config: BrandConfiguration,
        magazine_style: MagazineStyle, page_number: int,
        ambiguous_captions: bool = False
    ) -> ImageElement:
        """Create image element with specified placement."""
        
        # Select image size
        if placement_style == "full_page":
            width = page_size[0] - brand_config.margin_left - brand_config.margin_right
            height = width * 0.75  # 4:3 aspect ratio
        elif placement_style == "wrapped":
            width, height = random.choice(magazine_style.preferred_image_sizes)
            width = min(width, column_layout.column_width)
        else:
            width, height = random.choice(magazine_style.preferred_image_sizes)
            width = min(width, column_layout.column_width * 2)
        
        # Determine position based on placement style
        if placement_style == "full_page":
            left_x = brand_config.margin_left
            top_y = min(column_y_positions) + 20
        elif placement_style == "side":
            # Place in rightmost column
            column_idx = column_layout.column_count - 1
            left_x, right_x = column_layout.columns[column_idx]
            width = min(width, right_x - left_x)
            top_y = column_y_positions[column_idx]
        else:
            # Default placement in first available column
            column_idx = column_y_positions.index(min(column_y_positions))
            left_x, right_x = column_layout.columns[column_idx]
            width = min(width, right_x - left_x)
            top_y = column_y_positions[column_idx]
        
        right_x = left_x + width
        bottom_y = top_y + height
        
        image_element = ImageElement(
            element_id=f"image_{page_number}_{self._next_element_id()}",
            element_type="image",
            bbox=(left_x, top_y, right_x, bottom_y),
            page_number=page_number,
            alt_text=image_info.get("alt_text", ""),
            width=int(width),
            height=int(height),
            extraction_difficulty=0.1 if not ambiguous_captions else 0.4
        )
        
        return image_element
    
    def _create_caption_element(
        self, caption_text: str, image_element: ImageElement,
        column_layout: ColumnLayout, brand_config: BrandConfiguration,
        magazine_style: MagazineStyle, page_number: int
    ) -> TextElement:
        """Create caption element positioned near image."""
        
        # Position caption below image
        left_x = image_element.bbox[0]
        right_x = image_element.bbox[2]
        top_y = image_element.bbox[3] + 5
        
        caption_height = magazine_style.caption_font_size * 2  # Assume 2 lines
        bottom_y = top_y + caption_height
        
        caption_element = TextElement(
            element_id=f"caption_{page_number}_{self._next_element_id()}",
            element_type="text",
            bbox=(left_x, top_y, right_x, bottom_y),
            page_number=page_number,
            text_content=caption_text,
            font_family=brand_config.primary_font,
            font_size=magazine_style.caption_font_size,
            font_style="italic",
            semantic_type="caption",
            reading_order=image_element.reading_order + 1,
            extraction_difficulty=0.15
        )
        
        return caption_element
    
    def _select_text_column(self, text_index: int, column_layout: ColumnLayout,
                           column_y_positions: List[float]) -> int:
        """Select best column for text placement."""
        
        # Simple strategy: use column with lowest y position
        return column_y_positions.index(min(column_y_positions))
    
    def _select_image_placement(self, template: LayoutTemplate, 
                              image_index: int, total_images: int) -> str:
        """Select image placement style."""
        
        placement_options = template.image_placement_options
        
        # First image gets preferred placement
        if image_index == 0 and "top" in placement_options:
            return "top"
        
        return random.choice(placement_options)
    
    def _generate_edge_case_elements(
        self, edge_cases: List[EdgeCaseType], column_layout: ColumnLayout,
        page_size: Tuple[float, float], brand_config: BrandConfiguration,
        magazine_style: MagazineStyle, page_number: int
    ) -> List[LayoutElement]:
        """Generate elements for specific edge cases."""
        
        edge_elements = []
        
        if EdgeCaseType.WATERMARKS in edge_cases:
            # Add watermark element
            watermark = TextElement(
                element_id=f"watermark_{page_number}_{self._next_element_id()}",
                element_type="text",
                bbox=(100, 400, 500, 430),
                page_number=page_number,
                text_content="CONFIDENTIAL",
                font_family="Arial",
                font_size=48,
                font_style="normal",
                text_color=(0.9, 0.9, 0.9),  # Light gray
                semantic_type="watermark",
                z_order=-1,  # Behind other elements
                extraction_difficulty=0.8
            )
            edge_elements.append(watermark)
        
        if EdgeCaseType.ROTATED_TEXT in edge_cases:
            # Add rotated text element (simulated by changing difficulty)
            rotated_text = TextElement(
                element_id=f"rotated_{page_number}_{self._next_element_id()}",
                element_type="text",
                bbox=(50, 200, 100, 600),  # Tall, narrow bbox to simulate rotation
                page_number=page_number,
                text_content="SIDEBAR TEXT",
                font_family=brand_config.primary_font,
                font_size=10,
                semantic_type="sidebar",
                extraction_difficulty=0.6  # Higher difficulty for rotated text
            )
            edge_elements.append(rotated_text)
        
        return edge_elements
    
    def _finalize_element_positions(
        self, elements: List[LayoutElement], template: LayoutTemplate,
        page_size: Tuple[float, float]
    ) -> List[LayoutElement]:
        """Apply final positioning and resolve overlaps."""
        
        # Sort elements by z-order
        elements.sort(key=lambda e: e.z_order)
        
        # Resolve overlaps if template doesn't support them
        if not template.supports_overlapping_elements:
            elements = self._resolve_overlaps(elements)
        
        # Ensure elements stay within page bounds
        for element in elements:
            x0, y0, x1, y1 = element.bbox
            x0 = max(0, min(x0, page_size[0]))
            y0 = max(0, min(y0, page_size[1]))
            x1 = max(x0, min(x1, page_size[0]))
            y1 = max(y0, min(y1, page_size[1]))
            element.bbox = (x0, y0, x1, y1)
        
        return elements
    
    def _resolve_overlaps(self, elements: List[LayoutElement]) -> List[LayoutElement]:
        """Resolve overlapping elements by adjusting positions."""
        
        # Simple overlap resolution: move overlapping elements down
        for i, element in enumerate(elements):
            for j, other_element in enumerate(elements[:i]):
                if element.overlaps(other_element):
                    # Move current element below the other
                    overlap_offset = other_element.bbox[3] - element.bbox[1] + 10
                    element.bbox = (
                        element.bbox[0],
                        element.bbox[1] + overlap_offset,
                        element.bbox[2],
                        element.bbox[3] + overlap_offset
                    )
        
        return elements
    
    def _next_element_id(self) -> int:
        """Get next element ID."""
        self._element_counter += 1
        return self._element_counter