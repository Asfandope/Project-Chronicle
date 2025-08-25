"""
Semantic Graph Module for Document Analysis.

This module implements a semantic graph representation of document structure
using NetworkX, providing nodes for text blocks, images, and page breaks,
with edges representing semantic relationships.
"""

from .graph import SemanticGraph
from .types import NodeType, EdgeType
from .nodes import TextBlockNode, ImageNode, PageBreakNode
from .factory import GraphFactory
from .visualizer import GraphVisualizer
from .types import (
    GraphNodeData,
    GraphEdgeData,
    SerializedGraph,
    GraphError
)

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