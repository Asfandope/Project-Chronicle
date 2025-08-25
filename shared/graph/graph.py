"""
Main semantic graph implementation using NetworkX.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union
import json
import uuid

import networkx as nx
import structlog

from .types import (
    GraphNodeData, GraphEdgeData, SerializedGraph, GraphStats,
    NodeType, EdgeType, GraphError
)
from .nodes import BaseNode, TextBlockNode, ImageNode, PageBreakNode


logger = structlog.get_logger(__name__)


class SemanticGraph:
    """
    Semantic graph representation of document structure using NetworkX.
    
    Provides nodes for text blocks, images, and page breaks with edges
    representing semantic relationships like reading order, hierarchies,
    and content associations.
    """
    
    def __init__(self, graph_id: Optional[str] = None, document_path: Optional[str] = None):
        """
        Initialize semantic graph.
        
        Args:
            graph_id: Unique identifier for the graph
            document_path: Path to source document
        """
        self.graph_id = graph_id or str(uuid.uuid4())
        self.document_path = document_path
        self.creation_time = datetime.now()
        
        # Initialize NetworkX directed graph
        self.graph = nx.DiGraph()
        
        # Metadata
        self.metadata: Dict[str, Any] = {}
        
        self.logger = logger.bind(
            component="SemanticGraph",
            graph_id=self.graph_id[:8]
        )
        
        self.logger.debug("Initialized semantic graph")
    
    # Node management
    def add_node(self, node: BaseNode) -> str:
        """
        Add a node to the graph.
        
        Args:
            node: Node to add
            
        Returns:
            Node ID
        """
        try:
            node_data = node.to_graph_data()
            
            # Add to NetworkX graph with all attributes
            self.graph.add_node(
                node.node_id,
                **node_data.to_dict()
            )
            
            self.logger.debug(
                "Added node to graph",
                node_id=node.node_id[:8],
                node_type=node_data.node_type.value,
                page=node_data.page_num
            )
            
            return node.node_id
            
        except Exception as e:
            self.logger.error("Error adding node", error=str(e), exc_info=True)
            raise GraphError(f"Failed to add node: {e}", self.graph_id, node.node_id)
    
    def get_node(self, node_id: str) -> Optional[BaseNode]:
        """
        Get a node by ID.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Node instance or None if not found
        """
        try:
            if node_id not in self.graph:
                return None
            
            node_attrs = self.graph.nodes[node_id]
            node_data = GraphNodeData.from_dict(node_attrs)
            
            # Create appropriate node type
            if node_data.node_type == NodeType.TEXT_BLOCK:
                return TextBlockNode.from_graph_data(node_data)
            elif node_data.node_type == NodeType.IMAGE:
                return ImageNode.from_graph_data(node_data)
            elif node_data.node_type == NodeType.PAGE_BREAK:
                return PageBreakNode.from_graph_data(node_data)
            else:
                raise GraphError(f"Unknown node type: {node_data.node_type}")
                
        except Exception as e:
            self.logger.error("Error getting node", node_id=node_id[:8], error=str(e))
            raise GraphError(f"Failed to get node {node_id}: {e}", self.graph_id, node_id)
    
    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node from the graph.
        
        Args:
            node_id: Node identifier
            
        Returns:
            True if node was removed
        """
        try:
            if node_id in self.graph:
                self.graph.remove_node(node_id)
                self.logger.debug("Removed node from graph", node_id=node_id[:8])
                return True
            return False
            
        except Exception as e:
            self.logger.error("Error removing node", node_id=node_id[:8], error=str(e))
            raise GraphError(f"Failed to remove node {node_id}: {e}", self.graph_id, node_id)
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[BaseNode]:
        """Get all nodes of a specific type."""
        try:
            nodes = []
            for node_id in self.graph.nodes():
                node_attrs = self.graph.nodes[node_id]
                if NodeType(node_attrs["node_type"]) == node_type:
                    node = self.get_node(node_id)
                    if node:
                        nodes.append(node)
            
            return nodes
            
        except Exception as e:
            self.logger.error("Error getting nodes by type", node_type=node_type.value, error=str(e))
            raise GraphError(f"Failed to get nodes by type {node_type}: {e}", self.graph_id)
    
    def get_nodes_by_page(self, page_num: int) -> List[BaseNode]:
        """Get all nodes on a specific page."""
        try:
            nodes = []
            for node_id in self.graph.nodes():
                node_attrs = self.graph.nodes[node_id]
                if node_attrs["page_num"] == page_num:
                    node = self.get_node(node_id)
                    if node:
                        nodes.append(node)
            
            return nodes
            
        except Exception as e:
            self.logger.error("Error getting nodes by page", page_num=page_num, error=str(e))
            raise GraphError(f"Failed to get nodes for page {page_num}: {e}", self.graph_id)
    
    # Edge management
    def add_edge(self, source_id: str, target_id: str, edge_type: EdgeType, 
                 confidence: float = 1.0, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add an edge between two nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Type of relationship
            confidence: Confidence score
            metadata: Additional edge metadata
            
        Returns:
            True if edge was added
        """
        try:
            # Validate nodes exist
            if source_id not in self.graph or target_id not in self.graph:
                raise GraphError(f"Source or target node not found")
            
            edge_data = GraphEdgeData(
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                confidence=confidence,
                metadata=metadata or {}
            )
            
            # Add to NetworkX graph
            self.graph.add_edge(
                source_id,
                target_id,
                **edge_data.to_dict()
            )
            
            self.logger.debug(
                "Added edge to graph",
                source=source_id[:8],
                target=target_id[:8],
                edge_type=edge_type.value,
                confidence=confidence
            )
            
            return True
            
        except Exception as e:
            self.logger.error("Error adding edge", error=str(e), exc_info=True)
            raise GraphError(f"Failed to add edge: {e}", self.graph_id)
    
    def get_edges_by_type(self, edge_type: EdgeType) -> List[GraphEdgeData]:
        """Get all edges of a specific type."""
        try:
            edges = []
            for source, target, attrs in self.graph.edges(data=True):
                if EdgeType(attrs["edge_type"]) == edge_type:
                    edges.append(GraphEdgeData.from_dict(attrs))
            
            return edges
            
        except Exception as e:
            self.logger.error("Error getting edges by type", edge_type=edge_type.value, error=str(e))
            raise GraphError(f"Failed to get edges by type {edge_type}: {e}", self.graph_id)
    
    def get_successors(self, node_id: str, edge_type: Optional[EdgeType] = None) -> List[str]:
        """Get successor nodes, optionally filtered by edge type."""
        try:
            successors = []
            for target in self.graph.successors(node_id):
                if edge_type is None:
                    successors.append(target)
                else:
                    edge_attrs = self.graph.edges[node_id, target]
                    if EdgeType(edge_attrs["edge_type"]) == edge_type:
                        successors.append(target)
            
            return successors
            
        except Exception as e:
            self.logger.error("Error getting successors", node_id=node_id[:8], error=str(e))
            raise GraphError(f"Failed to get successors for {node_id}: {e}", self.graph_id)
    
    def get_predecessors(self, node_id: str, edge_type: Optional[EdgeType] = None) -> List[str]:
        """Get predecessor nodes, optionally filtered by edge type."""
        try:
            predecessors = []
            for source in self.graph.predecessors(node_id):
                if edge_type is None:
                    predecessors.append(source)
                else:
                    edge_attrs = self.graph.edges[source, node_id]
                    if EdgeType(edge_attrs["edge_type"]) == edge_type:
                        predecessors.append(source)
            
            return predecessors
            
        except Exception as e:
            self.logger.error("Error getting predecessors", node_id=node_id[:8], error=str(e))
            raise GraphError(f"Failed to get predecessors for {node_id}: {e}", self.graph_id)
    
    # Graph analysis
    def get_reading_order(self, page_num: Optional[int] = None) -> List[str]:
        """
        Get nodes in reading order.
        
        Args:
            page_num: Specific page number, or None for all pages
            
        Returns:
            List of node IDs in reading order
        """
        try:
            # Get all follows edges
            follows_edges = self.get_edges_by_type(EdgeType.FOLLOWS)
            
            # Filter by page if specified
            if page_num is not None:
                page_nodes = {node.node_id for node in self.get_nodes_by_page(page_num)}
                follows_edges = [e for e in follows_edges 
                               if e.source_id in page_nodes and e.target_id in page_nodes]
            
            # Build subgraph of follows relationships
            follows_graph = nx.DiGraph()
            for edge in follows_edges:
                follows_graph.add_edge(edge.source_id, edge.target_id)
            
            # Topological sort for reading order
            if follows_graph.nodes():
                return list(nx.topological_sort(follows_graph))
            else:
                # Fall back to page order if no follows edges
                nodes = self.get_nodes_by_page(page_num) if page_num else list(self.graph.nodes())
                return [n.node_id if hasattr(n, 'node_id') else n for n in nodes]
            
        except Exception as e:
            self.logger.error("Error getting reading order", page_num=page_num, error=str(e))
            raise GraphError(f"Failed to get reading order: {e}", self.graph_id)
    
    def find_connected_components(self) -> List[Set[str]]:
        """Find weakly connected components in the graph."""
        try:
            components = list(nx.weakly_connected_components(self.graph))
            self.logger.debug("Found connected components", count=len(components))
            return components
            
        except Exception as e:
            self.logger.error("Error finding connected components", error=str(e))
            raise GraphError(f"Failed to find connected components: {e}", self.graph_id)
    
    def get_statistics(self) -> GraphStats:
        """Get comprehensive graph statistics."""
        try:
            # Count nodes by type
            text_blocks = len(self.get_nodes_by_type(NodeType.TEXT_BLOCK))
            images = len(self.get_nodes_by_type(NodeType.IMAGE))
            page_breaks = len(self.get_nodes_by_type(NodeType.PAGE_BREAK))
            
            # Count edges by type
            follows_edges = len(self.get_edges_by_type(EdgeType.FOLLOWS))
            belongs_to_edges = len(self.get_edges_by_type(EdgeType.BELONGS_TO))
            continues_on_edges = len(self.get_edges_by_type(EdgeType.CONTINUES_ON))
            caption_of_edges = len(self.get_edges_by_type(EdgeType.CAPTION_OF))
            
            # Calculate page statistics
            page_nums = set()
            node_confidences = []
            for node_id in self.graph.nodes():
                node_attrs = self.graph.nodes[node_id]
                page_nums.add(node_attrs["page_num"])
                node_confidences.append(node_attrs["confidence"])
            
            page_count = len(page_nums)
            avg_blocks_per_page = (text_blocks + images) / max(page_count, 1)
            
            # Calculate confidence statistics
            edge_confidences = []
            for _, _, attrs in self.graph.edges(data=True):
                edge_confidences.append(attrs["confidence"])
            
            avg_node_confidence = sum(node_confidences) / max(len(node_confidences), 1)
            avg_edge_confidence = sum(edge_confidences) / max(len(edge_confidences), 1)
            
            # Find unconnected nodes
            components = self.find_connected_components()
            unconnected_nodes = sum(1 for comp in components if len(comp) == 1)
            
            return GraphStats(
                text_block_count=text_blocks,
                image_count=images,
                page_break_count=page_breaks,
                follows_count=follows_edges,
                belongs_to_count=belongs_to_edges,
                continues_on_count=continues_on_edges,
                caption_of_count=caption_of_edges,
                page_count=page_count,
                avg_blocks_per_page=avg_blocks_per_page,
                avg_node_confidence=avg_node_confidence,
                avg_edge_confidence=avg_edge_confidence,
                unconnected_nodes=unconnected_nodes
            )
            
        except Exception as e:
            self.logger.error("Error calculating statistics", error=str(e))
            raise GraphError(f"Failed to calculate statistics: {e}", self.graph_id)
    
    # Serialization
    def to_serialized(self) -> SerializedGraph:
        """Convert graph to serialized representation."""
        try:
            # Extract nodes and edges
            nodes = []
            for node_id in self.graph.nodes():
                node_attrs = self.graph.nodes[node_id]
                nodes.append(node_attrs)
            
            edges = []
            for source, target, attrs in self.graph.edges(data=True):
                edges.append(attrs)
            
            # Get statistics
            stats = self.get_statistics()
            
            return SerializedGraph(
                graph_id=self.graph_id,
                document_path=self.document_path or "",
                creation_time=self.creation_time.isoformat(),
                nodes=nodes,
                edges=edges,
                node_count=len(nodes),
                edge_count=len(edges),
                page_count=stats.page_count,
                metadata=self.metadata
            )
            
        except Exception as e:
            self.logger.error("Error serializing graph", error=str(e))
            raise GraphError(f"Failed to serialize graph: {e}", self.graph_id)
    
    @classmethod
    def from_serialized(cls, serialized: SerializedGraph) -> "SemanticGraph":
        """Create graph from serialized representation."""
        try:
            graph = cls(
                graph_id=serialized.graph_id,
                document_path=serialized.document_path
            )
            graph.creation_time = datetime.fromisoformat(serialized.creation_time)
            graph.metadata = serialized.metadata
            
            # Add nodes
            for node_data in serialized.nodes:
                graph.graph.add_node(node_data["node_id"], **node_data)
            
            # Add edges
            for edge_data in serialized.edges:
                graph.graph.add_edge(
                    edge_data["source_id"],
                    edge_data["target_id"],
                    **edge_data
                )
            
            logger.debug(
                "Loaded serialized graph",
                graph_id=graph.graph_id[:8],
                nodes=len(serialized.nodes),
                edges=len(serialized.edges)
            )
            
            return graph
            
        except Exception as e:
            logger.error("Error loading serialized graph", error=str(e))
            raise GraphError(f"Failed to load serialized graph: {e}")
    
    def save_json(self, output_path: Union[str, Path]):
        """Save graph to JSON file."""
        try:
            serialized = self.to_serialized()
            serialized.save_json(str(output_path))
            
            self.logger.info(
                "Saved graph to JSON",
                output_path=str(output_path),
                nodes=serialized.node_count,
                edges=serialized.edge_count
            )
            
        except Exception as e:
            self.logger.error("Error saving graph JSON", error=str(e))
            raise GraphError(f"Failed to save graph: {e}", self.graph_id)
    
    @classmethod
    def load_json(cls, file_path: Union[str, Path]) -> "SemanticGraph":
        """Load graph from JSON file."""
        try:
            serialized = SerializedGraph.load_json(str(file_path))
            return cls.from_serialized(serialized)
            
        except Exception as e:
            logger.error("Error loading graph JSON", file_path=str(file_path), error=str(e))
            raise GraphError(f"Failed to load graph from {file_path}: {e}")
    
    # Properties
    @property
    def node_count(self) -> int:
        """Get total number of nodes."""
        return self.graph.number_of_nodes()
    
    @property
    def edge_count(self) -> int:
        """Get total number of edges."""
        return self.graph.number_of_edges()
    
    @property
    def is_empty(self) -> bool:
        """Check if graph is empty."""
        return self.node_count == 0
    
    def __len__(self) -> int:
        """Get node count."""
        return self.node_count
    
    def __contains__(self, node_id: str) -> bool:
        """Check if node exists in graph."""
        return node_id in self.graph
    
    def __repr__(self) -> str:
        return f"SemanticGraph(id={self.graph_id[:8]}, nodes={self.node_count}, edges={self.edge_count})"