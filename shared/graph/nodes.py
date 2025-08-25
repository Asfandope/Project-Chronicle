"""
Node implementations for semantic graph.
"""

from typing import Any, Dict, Optional, Tuple
import uuid

from ..layout.types import BoundingBox, BlockType, TextBlock
from .types import GraphNodeData, NodeType


class BaseNode:
    """Base class for graph nodes."""
    
    def __init__(self, node_id: Optional[str] = None):
        self.node_id = node_id or str(uuid.uuid4())
    
    def to_graph_data(self) -> GraphNodeData:
        """Convert to GraphNodeData representation."""
        raise NotImplementedError
    
    @classmethod
    def from_graph_data(cls, data: GraphNodeData) -> "BaseNode":
        """Create node from GraphNodeData."""
        raise NotImplementedError


class TextBlockNode(BaseNode):
    """Node representing a text block in the document."""
    
    def __init__(
        self,
        text: str,
        bbox: BoundingBox,
        page_num: int,
        confidence: float = 1.0,
        classification: Optional[BlockType] = None,
        font_size: Optional[float] = None,
        font_family: Optional[str] = None,
        is_bold: bool = False,
        is_italic: bool = False,
        reading_order: int = 0,
        column: Optional[int] = None,
        node_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__(node_id)
        self.text = text
        self.bbox = bbox
        self.page_num = page_num
        self.confidence = confidence
        self.classification = classification
        self.font_size = font_size
        self.font_family = font_family
        self.is_bold = is_bold
        self.is_italic = is_italic
        self.reading_order = reading_order
        self.column = column
        self.metadata = metadata or {}
    
    @classmethod
    def from_text_block(cls, text_block: TextBlock, node_id: Optional[str] = None) -> "TextBlockNode":
        """Create TextBlockNode from layout TextBlock."""
        return cls(
            text=text_block.text,
            bbox=text_block.bbox,
            page_num=text_block.page_num,
            confidence=text_block.confidence,
            classification=text_block.block_type,
            font_size=text_block.font_size,
            font_family=text_block.font_family,
            is_bold=text_block.is_bold,
            is_italic=text_block.is_italic,
            reading_order=text_block.reading_order,
            column=text_block.column,
            node_id=node_id,
            metadata=text_block.classification_features.copy()
        )
    
    def to_graph_data(self) -> GraphNodeData:
        """Convert to GraphNodeData representation."""
        # Combine text properties in metadata
        text_metadata = self.metadata.copy()
        text_metadata.update({
            "font_size": self.font_size,
            "font_family": self.font_family,
            "is_bold": self.is_bold,
            "is_italic": self.is_italic,
            "reading_order": self.reading_order,
            "column": self.column,
            "word_count": len(self.text.split()),
            "char_count": len(self.text.strip())
        })
        
        return GraphNodeData(
            node_id=self.node_id,
            node_type=NodeType.TEXT_BLOCK,
            bbox=self.bbox,
            confidence=self.confidence,
            page_num=self.page_num,
            text=self.text,
            classification=self.classification,
            metadata=text_metadata
        )
    
    @classmethod
    def from_graph_data(cls, data: GraphNodeData) -> "TextBlockNode":
        """Create TextBlockNode from GraphNodeData."""
        if data.node_type != NodeType.TEXT_BLOCK:
            raise ValueError(f"Invalid node type: {data.node_type}")
        
        metadata = data.metadata.copy()
        
        return cls(
            text=data.text or "",
            bbox=data.bbox,
            page_num=data.page_num,
            confidence=data.confidence,
            classification=data.classification,
            font_size=metadata.pop("font_size", None),
            font_family=metadata.pop("font_family", None),
            is_bold=metadata.pop("is_bold", False),
            is_italic=metadata.pop("is_italic", False),
            reading_order=metadata.pop("reading_order", 0),
            column=metadata.pop("column", None),
            node_id=data.node_id,
            metadata=metadata
        )
    
    @property
    def word_count(self) -> int:
        """Get word count."""
        return len(self.text.split())
    
    @property
    def char_count(self) -> int:
        """Get character count."""
        return len(self.text.strip())
    
    def __repr__(self) -> str:
        return f"TextBlockNode(id={self.node_id[:8]}, type={self.classification}, text='{self.text[:50]}...')"


class ImageNode(BaseNode):
    """Node representing an image in the document."""
    
    def __init__(
        self,
        bbox: BoundingBox,
        page_num: int,
        image_path: Optional[str] = None,
        image_format: Optional[str] = None,
        image_size: Optional[Tuple[int, int]] = None,
        confidence: float = 1.0,
        description: Optional[str] = None,
        node_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__(node_id)
        self.bbox = bbox
        self.page_num = page_num
        self.image_path = image_path
        self.image_format = image_format
        self.image_size = image_size
        self.confidence = confidence
        self.description = description
        self.metadata = metadata or {}
    
    def to_graph_data(self) -> GraphNodeData:
        """Convert to GraphNodeData representation."""
        image_metadata = self.metadata.copy()
        if self.description:
            image_metadata["description"] = self.description
        
        return GraphNodeData(
            node_id=self.node_id,
            node_type=NodeType.IMAGE,
            bbox=self.bbox,
            confidence=self.confidence,
            page_num=self.page_num,
            image_path=self.image_path,
            image_format=self.image_format,
            image_size=self.image_size,
            metadata=image_metadata
        )
    
    @classmethod
    def from_graph_data(cls, data: GraphNodeData) -> "ImageNode":
        """Create ImageNode from GraphNodeData."""
        if data.node_type != NodeType.IMAGE:
            raise ValueError(f"Invalid node type: {data.node_type}")
        
        metadata = data.metadata.copy()
        description = metadata.pop("description", None)
        
        return cls(
            bbox=data.bbox,
            page_num=data.page_num,
            image_path=data.image_path,
            image_format=data.image_format,
            image_size=data.image_size,
            confidence=data.confidence,
            description=description,
            node_id=data.node_id,
            metadata=metadata
        )
    
    @property
    def width(self) -> Optional[int]:
        """Get image width."""
        return self.image_size[0] if self.image_size else None
    
    @property
    def height(self) -> Optional[int]:
        """Get image height."""
        return self.image_size[1] if self.image_size else None
    
    @property
    def aspect_ratio(self) -> Optional[float]:
        """Get image aspect ratio."""
        if self.image_size and self.image_size[1] > 0:
            return self.image_size[0] / self.image_size[1]
        return None
    
    def __repr__(self) -> str:
        size_str = f"{self.width}x{self.height}" if self.image_size else "unknown"
        return f"ImageNode(id={self.node_id[:8]}, size={size_str}, page={self.page_num})"


class PageBreakNode(BaseNode):
    """Node representing a page break in the document."""
    
    def __init__(
        self,
        page_num: int,
        confidence: float = 1.0,
        break_type: str = "standard",
        node_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__(node_id)
        self.page_num = page_num
        self.confidence = confidence
        self.break_type = break_type  # standard, column, section
        self.metadata = metadata or {}
        
        # Create a minimal bounding box at page boundary
        # This is a virtual boundary, actual coordinates depend on page size
        self.bbox = BoundingBox(x0=0, y0=0, x1=0, y1=0)
    
    def set_page_dimensions(self, page_width: float, page_height: float):
        """Set page dimensions for the break."""
        # Position at bottom of previous page or top of current page
        self.bbox = BoundingBox(
            x0=0,
            y0=page_height,
            x1=page_width,
            y1=page_height
        )
        self.metadata["page_width"] = page_width
        self.metadata["page_height"] = page_height
    
    def to_graph_data(self) -> GraphNodeData:
        """Convert to GraphNodeData representation."""
        break_metadata = self.metadata.copy()
        break_metadata["break_type"] = self.break_type
        
        return GraphNodeData(
            node_id=self.node_id,
            node_type=NodeType.PAGE_BREAK,
            bbox=self.bbox,
            confidence=self.confidence,
            page_num=self.page_num,
            metadata=break_metadata
        )
    
    @classmethod
    def from_graph_data(cls, data: GraphNodeData) -> "PageBreakNode":
        """Create PageBreakNode from GraphNodeData."""
        if data.node_type != NodeType.PAGE_BREAK:
            raise ValueError(f"Invalid node type: {data.node_type}")
        
        metadata = data.metadata.copy()
        break_type = metadata.pop("break_type", "standard")
        
        node = cls(
            page_num=data.page_num,
            confidence=data.confidence,
            break_type=break_type,
            node_id=data.node_id,
            metadata=metadata
        )
        node.bbox = data.bbox
        return node
    
    def __repr__(self) -> str:
        return f"PageBreakNode(id={self.node_id[:8]}, page={self.page_num}, type={self.break_type})"