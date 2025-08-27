"""
Semantic Graph Module for Document Analysis.

This module implements a semantic graph representation of document structure
using NetworkX, providing nodes for text blocks, images, and page breaks,
with edges representing semantic relationships.
"""

from .factory import GraphFactory
from .graph import SemanticGraph
from .nodes import ImageNode, PageBreakNode, TextBlockNode
from .types import (
    EdgeType,
    GraphEdgeData,
    GraphError,
    GraphNodeData,
    NodeType,
    SerializedGraph,
)
from .visualizer import GraphVisualizer

__all__ = [
    # Core classes
    "SemanticGraph",
    "GraphFactory",
    "GraphVisualizer",
    # Node types
    "TextBlockNode",
    "ImageNode",
    "PageBreakNode",
    # Enums
    "NodeType",
    "EdgeType",
    # Data types
    "GraphNodeData",
    "GraphEdgeData",
    "SerializedGraph",
    # Exceptions
    "GraphError",
]
