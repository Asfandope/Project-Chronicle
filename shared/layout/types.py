"""
Type definitions for layout analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union
import json


class LayoutError(Exception):
    """Base exception for layout analysis errors."""
    
    def __init__(self, message: str, page_num: Optional[int] = None, pdf_path: Optional[Path] = None):
        self.page_num = page_num
        self.pdf_path = pdf_path
        super().__init__(message)


class BlockType(Enum):
    """Text block classification types."""
    TITLE = "title"
    SUBTITLE = "subtitle"
    HEADING = "heading"
    BODY = "body"
    CAPTION = "caption"
    HEADER = "header"
    FOOTER = "footer"
    BYLINE = "byline"
    QUOTE = "quote"
    SIDEBAR = "sidebar"
    ADVERTISEMENT = "advertisement"
    PAGE_NUMBER = "page_number"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """Bounding box with coordinates and utility methods."""
    x0: float
    y0: float
    x1: float
    y1: float
    
    @property
    def width(self) -> float:
        return self.x1 - self.x0
    
    @property
    def height(self) -> float:
        return self.y1 - self.y0
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)
    
    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2
    
    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2
    
    def overlaps(self, other: "BoundingBox") -> bool:
        """Check if this bounding box overlaps with another."""
        return not (self.x1 < other.x0 or other.x1 < self.x0 or 
                   self.y1 < other.y0 or other.y1 < self.y0)
    
    def intersection_area(self, other: "BoundingBox") -> float:
        """Calculate intersection area with another bounding box."""
        if not self.overlaps(other):
            return 0.0
        
        x_overlap = min(self.x1, other.x1) - max(self.x0, other.x0)
        y_overlap = min(self.y1, other.y1) - max(self.y0, other.y0)
        return x_overlap * y_overlap
    
    def distance_to(self, other: "BoundingBox") -> float:
        """Calculate minimum distance between two bounding boxes."""
        # If they overlap, distance is 0
        if self.overlaps(other):
            return 0.0
        
        # Calculate horizontal and vertical distances
        h_distance = max(0, max(self.x0 - other.x1, other.x0 - self.x1))
        v_distance = max(0, max(self.y0 - other.y1, other.y0 - self.y1))
        
        return (h_distance ** 2 + v_distance ** 2) ** 0.5
    
    def expand(self, margin: float) -> "BoundingBox":
        """Expand bounding box by margin in all directions."""
        return BoundingBox(
            self.x0 - margin,
            self.y0 - margin,
            self.x1 + margin,
            self.y1 + margin
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary representation."""
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "width": self.width,
            "height": self.height,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "area": self.area
        }


@dataclass
class TextBlock:
    """Text block with classification and metadata."""
    
    # Core properties
    text: str
    bbox: BoundingBox
    block_type: BlockType = BlockType.UNKNOWN
    confidence: float = 0.0
    
    # Font properties
    font_size: Optional[float] = None
    font_family: Optional[str] = None
    is_bold: bool = False
    is_italic: bool = False
    
    # Layout properties
    page_num: int = 0
    reading_order: int = 0
    column: Optional[int] = None
    
    # Classification metadata
    classification_features: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def word_count(self) -> int:
        return len(self.text.split())
    
    @property
    def char_count(self) -> int:
        return len(self.text.strip())
    
    @property
    def line_count(self) -> int:
        return len(self.text.split('\n'))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "text": self.text,
            "bbox": self.bbox.to_dict(),
            "block_type": self.block_type.value,
            "confidence": self.confidence,
            "font_size": self.font_size,
            "font_family": self.font_family,
            "is_bold": self.is_bold,
            "is_italic": self.is_italic,
            "page_num": self.page_num,
            "reading_order": self.reading_order,
            "column": self.column,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "line_count": self.line_count,
            "classification_features": self.classification_features
        }


@dataclass
class PageLayout:
    """Layout analysis result for a single page."""
    
    page_num: int
    page_width: float
    page_height: float
    text_blocks: List[TextBlock] = field(default_factory=list)
    processing_time: float = 0.0
    
    @property
    def block_count(self) -> int:
        return len(self.text_blocks)
    
    @property
    def blocks_by_type(self) -> Dict[BlockType, List[TextBlock]]:
        """Group blocks by type."""
        grouped = {}
        for block in self.text_blocks:
            if block.block_type not in grouped:
                grouped[block.block_type] = []
            grouped[block.block_type].append(block)
        return grouped
    
    @property
    def reading_order_blocks(self) -> List[TextBlock]:
        """Get blocks sorted by reading order."""
        return sorted(self.text_blocks, key=lambda b: b.reading_order)
    
    def get_blocks_by_type(self, block_type: BlockType) -> List[TextBlock]:
        """Get all blocks of a specific type."""
        return [block for block in self.text_blocks if block.block_type == block_type]
    
    def get_main_content_blocks(self) -> List[TextBlock]:
        """Get main content blocks (titles, headings, body)."""
        content_types = {BlockType.TITLE, BlockType.SUBTITLE, BlockType.HEADING, BlockType.BODY}
        return [block for block in self.text_blocks if block.block_type in content_types]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "page_num": self.page_num,
            "page_width": self.page_width,
            "page_height": self.page_height,
            "block_count": self.block_count,
            "processing_time": self.processing_time,
            "text_blocks": [block.to_dict() for block in self.text_blocks],
            "blocks_by_type": {
                block_type.value: len(blocks) 
                for block_type, blocks in self.blocks_by_type.items()
            }
        }


@dataclass
class LayoutResult:
    """Complete layout analysis result for a document."""
    
    pdf_path: Path
    pages: List[PageLayout] = field(default_factory=list)
    total_processing_time: float = 0.0
    analysis_config: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def page_count(self) -> int:
        return len(self.pages)
    
    @property
    def total_blocks(self) -> int:
        return sum(page.block_count for page in self.pages)
    
    @property
    def blocks_by_type_total(self) -> Dict[BlockType, int]:
        """Count of all blocks by type across all pages."""
        totals = {}
        for page in self.pages:
            for block_type, blocks in page.blocks_by_type.items():
                totals[block_type] = totals.get(block_type, 0) + len(blocks)
        return totals
    
    def get_page(self, page_num: int) -> Optional[PageLayout]:
        """Get layout for specific page."""
        for page in self.pages:
            if page.page_num == page_num:
                return page
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "pdf_path": str(self.pdf_path),
            "page_count": self.page_count,
            "total_blocks": self.total_blocks,
            "total_processing_time": self.total_processing_time,
            "timestamp": self.timestamp.isoformat(),
            "blocks_by_type_total": {
                block_type.value: count 
                for block_type, count in self.blocks_by_type_total.items()
            },
            "pages": [page.to_dict() for page in self.pages],
            "analysis_config": self.analysis_config
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def save_json(self, output_path: Path):
        """Save layout result to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())


@dataclass
class ClassificationRule:
    """Rule for classifying text blocks."""
    
    name: str
    block_type: BlockType
    conditions: Dict[str, Any]
    priority: int = 0
    confidence: float = 1.0
    
    def matches(self, block: TextBlock, page_layout: PageLayout) -> Tuple[bool, float]:
        """
        Check if this rule matches a text block.
        
        Returns:
            Tuple of (matches, confidence_score)
        """
        try:
            # Font size conditions
            if "font_size_min" in self.conditions:
                if not block.font_size or block.font_size < self.conditions["font_size_min"]:
                    return False, 0.0
            
            if "font_size_max" in self.conditions:
                if not block.font_size or block.font_size > self.conditions["font_size_max"]:
                    return False, 0.0
            
            # Position conditions
            if "position_top_threshold" in self.conditions:
                if block.bbox.y0 > self.conditions["position_top_threshold"]:
                    return False, 0.0
            
            if "position_bottom_threshold" in self.conditions:
                if block.bbox.y1 < (page_layout.page_height - self.conditions["position_bottom_threshold"]):
                    return False, 0.0
            
            # Text length conditions
            if "max_words" in self.conditions:
                if block.word_count > self.conditions["max_words"]:
                    return False, 0.0
            
            if "min_words" in self.conditions:
                if block.word_count < self.conditions["min_words"]:
                    return False, 0.0
            
            # Text pattern conditions
            if "text_patterns" in self.conditions:
                import re
                patterns = self.conditions["text_patterns"]
                if not any(re.search(pattern, block.text, re.IGNORECASE) for pattern in patterns):
                    return False, 0.0
            
            # Formatting conditions
            if "requires_bold" in self.conditions:
                if self.conditions["requires_bold"] and not block.is_bold:
                    return False, 0.0
            
            if "requires_italic" in self.conditions:
                if self.conditions["requires_italic"] and not block.is_italic:
                    return False, 0.0
            
            # All conditions passed
            return True, self.confidence
            
        except Exception:
            return False, 0.0


@dataclass
class LayoutConfig:
    """Configuration for layout analysis."""
    
    # Text extraction settings
    min_text_length: int = 3
    merge_nearby_blocks: bool = True
    merge_distance_threshold: float = 10.0
    
    # Classification settings
    classification_rules: List[ClassificationRule] = field(default_factory=list)
    enable_reading_order: bool = True
    enable_column_detection: bool = True
    
    # Font analysis settings
    analyze_font_properties: bool = True
    normalize_font_sizes: bool = True
    
    # Page layout settings
    header_footer_margin: float = 50.0  # pixels from edge
    title_position_threshold: float = 0.3  # fraction of page height from top
    
    @classmethod
    def get_default_rules(cls) -> List[ClassificationRule]:
        """Get default classification rules."""
        return [
            # Title rules (highest priority)
            ClassificationRule(
                name="large_bold_title",
                block_type=BlockType.TITLE,
                conditions={
                    "font_size_min": 18,
                    "position_top_threshold": 200,
                    "max_words": 20,
                    "requires_bold": True
                },
                priority=10,
                confidence=0.9
            ),
            
            # Header rules
            ClassificationRule(
                name="top_header",
                block_type=BlockType.HEADER,
                conditions={
                    "position_top_threshold": 50,
                    "max_words": 15
                },
                priority=9,
                confidence=0.8
            ),
            
            # Footer rules
            ClassificationRule(
                name="bottom_footer",
                block_type=BlockType.FOOTER,
                conditions={
                    "position_bottom_threshold": 50,
                    "max_words": 15
                },
                priority=9,
                confidence=0.8
            ),
            
            # Page number rules
            ClassificationRule(
                name="page_number",
                block_type=BlockType.PAGE_NUMBER,
                conditions={
                    "text_patterns": [r'^\d+$', r'^Page\s+\d+', r'^\d+\s*/\s*\d+$'],
                    "max_words": 3
                },
                priority=8,
                confidence=0.9
            ),
            
            # Byline rules
            ClassificationRule(
                name="byline",
                block_type=BlockType.BYLINE,
                conditions={
                    "text_patterns": [r'^By\s+', r'^\w+\s+\w+\s+reports?', r'^Author:'],
                    "max_words": 10
                },
                priority=7,
                confidence=0.8
            ),
            
            # Caption rules
            ClassificationRule(
                name="caption",
                block_type=BlockType.CAPTION,
                conditions={
                    "font_size_max": 10,
                    "max_words": 50
                },
                priority=6,
                confidence=0.7
            ),
            
            # Heading rules
            ClassificationRule(
                name="medium_heading",
                block_type=BlockType.HEADING,
                conditions={
                    "font_size_min": 14,
                    "max_words": 15
                },
                priority=5,
                confidence=0.7
            ),
            
            # Body text (lowest priority - catch-all)
            ClassificationRule(
                name="body_text",
                block_type=BlockType.BODY,
                conditions={
                    "min_words": 5
                },
                priority=1,
                confidence=0.6
            )
        ]
    
    @classmethod
    def create_default(cls) -> "LayoutConfig":
        """Create default configuration."""
        return cls(classification_rules=cls.get_default_rules())


@dataclass
class VisualizationConfig:
    """Configuration for layout visualization."""
    
    # Color scheme for different block types
    colors: Dict[BlockType, str] = field(default_factory=lambda: {
        BlockType.TITLE: "#FF6B6B",        # Red
        BlockType.SUBTITLE: "#4ECDC4",     # Teal
        BlockType.HEADING: "#45B7D1",      # Blue
        BlockType.BODY: "#96CEB4",         # Green
        BlockType.CAPTION: "#FFEAA7",      # Yellow
        BlockType.HEADER: "#DDA0DD",       # Plum
        BlockType.FOOTER: "#DDA0DD",       # Plum
        BlockType.BYLINE: "#FFB347",       # Orange
        BlockType.QUOTE: "#F8C471",        # Light Orange
        BlockType.SIDEBAR: "#AED6F1",      # Light Blue
        BlockType.ADVERTISEMENT: "#F1948A", # Light Red
        BlockType.PAGE_NUMBER: "#D5DBDB",   # Light Gray
        BlockType.UNKNOWN: "#BDC3C7"       # Gray
    })
    
    # Visualization settings
    show_text: bool = True
    show_bounding_boxes: bool = True
    show_reading_order: bool = True
    show_confidence: bool = True
    
    # Style settings
    box_opacity: float = 0.3
    text_size: int = 8
    line_width: float = 2.0
    
    # Output settings
    output_format: str = "PDF"  # PDF, PNG, etc.
    dpi: int = 150
    
    def get_color(self, block_type: BlockType) -> str:
        """Get color for a block type."""
        return self.colors.get(block_type, self.colors[BlockType.UNKNOWN])