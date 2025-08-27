"""
Type definitions for semantic graph module.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..layout.types import BlockType
from ..types import BoundingBox


class GraphError(Exception):
    """Base exception for graph-related errors."""

    def __init__(
        self,
        message: str,
        graph_id: Optional[str] = None,
        node_id: Optional[str] = None,
    ):
        self.graph_id = graph_id
        self.node_id = node_id
        super().__init__(message)


class NodeType(Enum):
    """Types of nodes in the semantic graph."""

    TEXT_BLOCK = "text_block"
    IMAGE = "image"
    PAGE_BREAK = "page_break"


class EdgeType(Enum):
    """Types of edges in the semantic graph."""

    FOLLOWS = "follows"  # Sequential relationship
    BELONGS_TO = "belongs_to"  # Hierarchical relationship
    CONTINUES_ON = "continues_on"  # Content continuation across pages
    CAPTION_OF = "caption_of"  # Caption describes image/table

    # Spatial relationships
    ABOVE = "above"  # Spatial: node is above another
    BELOW = "below"  # Spatial: node is below another
    LEFT_OF = "left_of"  # Spatial: node is left of another
    RIGHT_OF = "right_of"  # Spatial: node is right of another


@dataclass
class GraphNodeData:
    """Base data structure for graph nodes."""

    # Core attributes
    node_id: str
    node_type: NodeType
    bbox: BoundingBox
    confidence: float
    page_num: int

    # Content attributes
    text: Optional[str] = None
    classification: Optional[BlockType] = None

    # Image-specific attributes
    image_path: Optional[str] = None
    image_format: Optional[str] = None
    image_size: Optional[Tuple[int, int]] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "bbox": self.bbox.to_dict(),
            "confidence": self.confidence,
            "page_num": self.page_num,
            "metadata": self.metadata,
        }

        # Add optional fields if present
        if self.text is not None:
            data["text"] = self.text
        if self.classification is not None:
            data["classification"] = self.classification.value
        if self.image_path is not None:
            data["image_path"] = self.image_path
        if self.image_format is not None:
            data["image_format"] = self.image_format
        if self.image_size is not None:
            data["image_size"] = self.image_size

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphNodeData":
        """Create from dictionary representation."""
        # Parse bbox
        bbox_data = data["bbox"]
        bbox = BoundingBox(
            x0=bbox_data["x0"],
            y0=bbox_data["y0"],
            x1=bbox_data["x1"],
            y1=bbox_data["y1"],
        )

        # Parse optional classification
        classification = None
        if "classification" in data:
            classification = BlockType(data["classification"])

        return cls(
            node_id=data["node_id"],
            node_type=NodeType(data["node_type"]),
            bbox=bbox,
            confidence=data["confidence"],
            page_num=data["page_num"],
            text=data.get("text"),
            classification=classification,
            image_path=data.get("image_path"),
            image_format=data.get("image_format"),
            image_size=data.get("image_size"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class GraphEdgeData:
    """Data structure for graph edges."""

    # Core attributes
    source_id: str
    target_id: str
    edge_type: EdgeType
    confidence: float

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphEdgeData":
        """Create from dictionary representation."""
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            edge_type=EdgeType(data["edge_type"]),
            confidence=data["confidence"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class SerializedGraph:
    """Serialized representation of a semantic graph."""

    # Graph metadata
    graph_id: str
    document_path: str
    creation_time: str

    # Graph data
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]

    # Statistics
    node_count: int
    edge_count: int
    page_count: int

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.__dict__, indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "SerializedGraph":
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls(**data)

    def save_json(self, output_path: str):
        """Save to JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def load_json(cls, file_path: str) -> "SerializedGraph":
        """Load from JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return cls.from_json(f.read())


@dataclass
class GraphStats:
    """Statistics about a semantic graph."""

    # Node counts by type
    text_block_count: int
    image_count: int
    page_break_count: int

    # Edge counts by type
    follows_count: int
    belongs_to_count: int
    continues_on_count: int
    caption_of_count: int

    # Layout statistics
    page_count: int
    avg_blocks_per_page: float

    # Quality metrics
    avg_node_confidence: float
    avg_edge_confidence: float
    unconnected_nodes: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "node_counts": {
                "text_blocks": self.text_block_count,
                "images": self.image_count,
                "page_breaks": self.page_break_count,
                "total": self.text_block_count
                + self.image_count
                + self.page_break_count,
            },
            "edge_counts": {
                "follows": self.follows_count,
                "belongs_to": self.belongs_to_count,
                "continues_on": self.continues_on_count,
                "caption_of": self.caption_of_count,
                "total": self.follows_count
                + self.belongs_to_count
                + self.continues_on_count
                + self.caption_of_count,
            },
            "layout_stats": {
                "page_count": self.page_count,
                "avg_blocks_per_page": self.avg_blocks_per_page,
            },
            "quality_metrics": {
                "avg_node_confidence": self.avg_node_confidence,
                "avg_edge_confidence": self.avg_edge_confidence,
                "unconnected_nodes": self.unconnected_nodes,
            },
        }
