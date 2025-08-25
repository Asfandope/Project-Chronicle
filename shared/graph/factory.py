"""
Factory methods for creating common semantic graph patterns.
"""

from typing import List, Optional, Dict, Any, Tuple
import structlog

from ..layout.types import LayoutResult, PageLayout, TextBlock, BlockType
from .graph import SemanticGraph
from .nodes import TextBlockNode, ImageNode, PageBreakNode
from .types import EdgeType, NodeType


logger = structlog.get_logger(__name__)


class GraphFactory:
    """
    Factory for creating semantic graphs with common patterns.
    
    Provides convenience methods for building graphs from layout analysis
    results and creating typical document structure patterns.
    """
    
    @staticmethod
    def from_layout_result(layout_result: LayoutResult, 
                          include_page_breaks: bool = True) -> SemanticGraph:
        """
        Create semantic graph from layout analysis result.
        
        Args:
            layout_result: Layout analysis result
            include_page_breaks: Whether to add page break nodes
            
        Returns:
            Semantic graph with nodes and basic relationships
        """
        logger.debug(
            "Creating graph from layout result",
            pages=len(layout_result.pages),
            total_blocks=layout_result.total_blocks
        )
        
        graph = SemanticGraph(
            document_path=str(layout_result.pdf_path)
        )
        
        # Add metadata
        graph.metadata = {
            "source": "layout_analyzer",
            "total_processing_time": layout_result.total_processing_time,
            "analysis_config": layout_result.analysis_config
        }
        
        prev_page_last_node = None
        
        for page in layout_result.pages:
            # Add page break node if not first page
            if include_page_breaks and page.page_num > 1:
                page_break = PageBreakNode(
                    page_num=page.page_num,
                    break_type="standard"
                )
                page_break.set_page_dimensions(page.page_width, page.page_height)
                graph.add_node(page_break)
                
                # Connect to last node of previous page
                if prev_page_last_node:
                    graph.add_edge(
                        prev_page_last_node,
                        page_break.node_id,
                        EdgeType.CONTINUES_ON,
                        confidence=0.9
                    )
            
            # Add text blocks and create reading order
            page_nodes = []
            for block in page.reading_order_blocks:
                text_node = TextBlockNode.from_text_block(block)
                graph.add_node(text_node)
                page_nodes.append(text_node.node_id)
            
            # Create follows edges for reading order
            for i in range(len(page_nodes) - 1):
                graph.add_edge(
                    page_nodes[i],
                    page_nodes[i + 1],
                    EdgeType.FOLLOWS,
                    confidence=0.8
                )
            
            # Remember last node for page continuation
            if page_nodes:
                prev_page_last_node = page_nodes[-1]
        
        logger.info(
            "Created graph from layout result",
            graph_id=graph.graph_id[:8],
            nodes=graph.node_count,
            edges=graph.edge_count
        )
        
        return graph
    
    @staticmethod
    def create_article_structure(title_text: str, 
                               body_blocks: List[TextBlock],
                               byline: Optional[str] = None,
                               page_num: int = 1) -> SemanticGraph:
        """
        Create a semantic graph representing a typical article structure.
        
        Args:
            title_text: Article title
            body_blocks: List of body text blocks
            byline: Optional byline text
            page_num: Page number
            
        Returns:
            Semantic graph with article hierarchy
        """
        logger.debug("Creating article structure", title=title_text[:50])
        
        graph = SemanticGraph()
        graph.metadata = {"pattern": "article_structure"}
        
        # Create title node
        from ..layout.types import BoundingBox
        title_bbox = BoundingBox(0, 0, 400, 50)  # Placeholder coordinates
        
        title_node = TextBlockNode(
            text=title_text,
            bbox=title_bbox,
            page_num=page_num,
            confidence=1.0,
            classification=BlockType.TITLE,
            font_size=24,
            is_bold=True
        )
        graph.add_node(title_node)
        
        # Create byline node if provided
        byline_node_id = None
        if byline:
            byline_bbox = BoundingBox(0, 60, 300, 80)
            byline_node = TextBlockNode(
                text=byline,
                bbox=byline_bbox,
                page_num=page_num,
                confidence=1.0,
                classification=BlockType.BYLINE,
                font_size=12
            )
            graph.add_node(byline_node)
            byline_node_id = byline_node.node_id
            
            # Connect title to byline
            graph.add_edge(
                title_node.node_id,
                byline_node.node_id,
                EdgeType.FOLLOWS,
                confidence=0.9
            )
        
        # Add body blocks
        prev_node_id = byline_node_id if byline_node_id else title_node.node_id
        
        for i, block in enumerate(body_blocks):
            body_node = TextBlockNode.from_text_block(block)
            graph.add_node(body_node)
            
            # Connect in reading order
            if prev_node_id:
                graph.add_edge(
                    prev_node_id,
                    body_node.node_id,
                    EdgeType.FOLLOWS,
                    confidence=0.8
                )
            
            # Add hierarchical relationship to title
            graph.add_edge(
                body_node.node_id,
                title_node.node_id,
                EdgeType.BELONGS_TO,
                confidence=0.7
            )
            
            prev_node_id = body_node.node_id
        
        logger.debug(
            "Created article structure",
            graph_id=graph.graph_id[:8],
            blocks=len(body_blocks) + 1 + (1 if byline else 0)
        )
        
        return graph
    
    @staticmethod
    def add_image_with_caption(graph: SemanticGraph,
                             image_bbox: "BoundingBox",
                             caption_text: str,
                             page_num: int,
                             image_path: Optional[str] = None) -> Tuple[str, str]:
        """
        Add an image node with its caption to the graph.
        
        Args:
            graph: Target semantic graph
            image_bbox: Image bounding box
            caption_text: Caption text
            page_num: Page number
            image_path: Optional path to image file
            
        Returns:
            Tuple of (image_node_id, caption_node_id)
        """
        logger.debug("Adding image with caption", caption=caption_text[:50])
        
        # Create image node
        image_node = ImageNode(
            bbox=image_bbox,
            page_num=page_num,
            image_path=image_path,
            confidence=1.0
        )
        image_id = graph.add_node(image_node)
        
        # Create caption node
        from ..layout.types import BoundingBox
        caption_bbox = BoundingBox(
            image_bbox.x0,
            image_bbox.y1 + 5,  # Below image
            image_bbox.x1,
            image_bbox.y1 + 25
        )
        
        caption_node = TextBlockNode(
            text=caption_text,
            bbox=caption_bbox,
            page_num=page_num,
            confidence=1.0,
            classification=BlockType.CAPTION,
            font_size=9
        )
        caption_id = graph.add_node(caption_node)
        
        # Connect caption to image
        graph.add_edge(
            caption_id,
            image_id,
            EdgeType.CAPTION_OF,
            confidence=0.9
        )
        
        logger.debug(
            "Added image with caption",
            image_id=image_id[:8],
            caption_id=caption_id[:8]
        )
        
        return image_id, caption_id
    
    @staticmethod
    def create_multi_column_layout(columns: List[List[TextBlock]],
                                 page_num: int = 1) -> SemanticGraph:
        """
        Create a semantic graph for multi-column layout.
        
        Args:
            columns: List of columns, each containing text blocks
            page_num: Page number
            
        Returns:
            Semantic graph with column structure
        """
        logger.debug("Creating multi-column layout", columns=len(columns))
        
        graph = SemanticGraph()
        graph.metadata = {
            "pattern": "multi_column_layout",
            "column_count": len(columns)
        }
        
        column_last_nodes = []
        
        for col_idx, column_blocks in enumerate(columns):
            prev_node_id = None
            
            for block in column_blocks:
                # Update block column information
                block.column = col_idx
                
                text_node = TextBlockNode.from_text_block(block)
                graph.add_node(text_node)
                
                # Connect within column
                if prev_node_id:
                    graph.add_edge(
                        prev_node_id,
                        text_node.node_id,
                        EdgeType.FOLLOWS,
                        confidence=0.8
                    )
                
                prev_node_id = text_node.node_id
            
            # Remember last node of column
            if prev_node_id:
                column_last_nodes.append(prev_node_id)
        
        # Connect columns (left to right reading order)
        for i in range(len(column_last_nodes) - 1):
            # Get first node of next column
            next_column_first = GraphFactory._get_first_node_in_column(
                graph, columns[i + 1], i + 1
            )
            
            if next_column_first:
                graph.add_edge(
                    column_last_nodes[i],
                    next_column_first,
                    EdgeType.CONTINUES_ON,
                    confidence=0.6
                )
        
        logger.debug(
            "Created multi-column layout",
            graph_id=graph.graph_id[:8],
            columns=len(columns)
        )
        
        return graph
    
    @staticmethod
    def _get_first_node_in_column(graph: SemanticGraph, 
                                column_blocks: List[TextBlock],
                                column_idx: int) -> Optional[str]:
        """Helper to find first node in a column."""
        if not column_blocks:
            return None
        
        # Find node matching first block
        first_block = column_blocks[0]
        for node_id in graph.graph.nodes():
            node_attrs = graph.graph.nodes[node_id]
            if (node_attrs.get("column") == column_idx and 
                node_attrs.get("text") == first_block.text):
                return node_id
        
        return None
    
    @staticmethod
    def add_hierarchical_headings(graph: SemanticGraph,
                                heading_hierarchy: List[Tuple[str, int, List[str]]]) -> List[str]:
        """
        Add hierarchical heading structure to graph.
        
        Args:
            graph: Target semantic graph
            heading_hierarchy: List of (heading_text, level, child_node_ids)
            
        Returns:
            List of heading node IDs
        """
        logger.debug("Adding hierarchical headings", count=len(heading_hierarchy))
        
        heading_nodes = []
        
        for heading_text, level, child_ids in heading_hierarchy:
            # Create heading node
            from ..layout.types import BoundingBox
            heading_bbox = BoundingBox(0, 0, 400, 30)  # Placeholder
            
            heading_node = TextBlockNode(
                text=heading_text,
                bbox=heading_bbox,
                page_num=1,  # Will be updated based on children
                confidence=1.0,
                classification=BlockType.HEADING,
                font_size=16 - level * 2,  # Smaller font for deeper levels
                is_bold=True
            )
            
            heading_id = graph.add_node(heading_node)
            heading_nodes.append(heading_id)
            
            # Connect children to heading
            for child_id in child_ids:
                if child_id in graph:
                    graph.add_edge(
                        child_id,
                        heading_id,
                        EdgeType.BELONGS_TO,
                        confidence=0.8
                    )
        
        logger.debug(
            "Added hierarchical headings",
            heading_count=len(heading_nodes)
        )
        
        return heading_nodes
    
    @staticmethod
    def merge_graphs(graphs: List[SemanticGraph]) -> SemanticGraph:
        """
        Merge multiple semantic graphs into one.
        
        Args:
            graphs: List of graphs to merge
            
        Returns:
            Merged semantic graph
        """
        if not graphs:
            return SemanticGraph()
        
        if len(graphs) == 1:
            return graphs[0]
        
        logger.debug("Merging graphs", count=len(graphs))
        
        # Create new graph
        merged = SemanticGraph()
        merged.metadata = {
            "pattern": "merged_graphs",
            "source_graphs": [g.graph_id for g in graphs]
        }
        
        # Copy all nodes and edges
        for graph in graphs:
            # Copy nodes
            for node_id in graph.graph.nodes():
                node_attrs = graph.graph.nodes[node_id]
                merged.graph.add_node(node_id, **node_attrs)
            
            # Copy edges
            for source, target, attrs in graph.graph.edges(data=True):
                merged.graph.add_edge(source, target, **attrs)
        
        logger.info(
            "Merged graphs",
            merged_id=merged.graph_id[:8],
            total_nodes=merged.node_count,
            total_edges=merged.edge_count
        )
        
        return merged