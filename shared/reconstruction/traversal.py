"""
Graph traversal algorithms for article reconstruction.
"""

import re
import uuid
from typing import Any, List, Optional, Set

import structlog

from ..graph import EdgeType, NodeType, SemanticGraph
from ..layout.types import BlockType
from .types import (
    ContinuationMarker,
    ContinuationType,
    ReconstructionConfig,
    ReconstructionError,
    TraversalPath,
)

logger = structlog.get_logger(__name__)


class GraphTraversal:
    """
    Graph traversal algorithms for article reconstruction.

    Provides methods for identifying article start nodes and traversing
    the semantic graph to collect article components.
    """

    def __init__(self, config: Optional[ReconstructionConfig] = None):
        """
        Initialize graph traversal.

        Args:
            config: Reconstruction configuration
        """
        self.config = config or ReconstructionConfig()
        self.logger = logger.bind(component="GraphTraversal")

        # Traversal state
        self._visited_nodes: Set[str] = set()
        self._current_path: Optional[TraversalPath] = None

        self.logger.debug("Initialized graph traversal")

    def identify_article_starts(self, graph: SemanticGraph) -> List[str]:
        """
        Identify potential article start nodes (titles).

        Args:
            graph: Semantic graph to analyze

        Returns:
            List of node IDs that could be article starts
        """
        try:
            self.logger.debug("Identifying article start nodes")

            article_starts = []

            # Get all nodes in the graph
            for node_id in graph.graph.nodes():
                node = graph.get_node(node_id)
                if not node:
                    continue

                node_data = node.to_graph_data()

                # Check if this could be an article start
                if self._is_potential_article_start(node_data, graph):
                    article_starts.append(node_id)

            # Sort by confidence and position
            article_starts = self._rank_article_starts(article_starts, graph)

            self.logger.info(
                "Identified article start candidates", count=len(article_starts)
            )

            return article_starts

        except Exception as e:
            self.logger.error("Error identifying article starts", error=str(e))
            raise ReconstructionError(f"Failed to identify article starts: {e}")

    def _is_potential_article_start(self, node_data: Any, graph: SemanticGraph) -> bool:
        """Check if a node could be an article start."""
        try:
            # Must be a text block
            if node_data.node_type != NodeType.TEXT_BLOCK:
                return False

            # Must have title classification or high confidence
            if node_data.classification == BlockType.TITLE:
                return node_data.confidence >= self.config.min_title_confidence

            # Could be a subtitle or heading with very high confidence
            if node_data.classification in [BlockType.SUBTITLE, BlockType.HEADING]:
                return node_data.confidence >= 0.9

            # Check for title-like characteristics even without classification
            if not node_data.text:
                return False

            text = node_data.text.strip()

            # Title-like heuristics
            word_count = len(text.split())

            # Reasonable title length
            if not (3 <= word_count <= 20):
                return False

            # Check position (should be near top of page)
            if node_data.bbox and node_data.bbox.y0 > 300:  # Not in top portion
                return False

            # Check font size if available
            font_size = node_data.metadata.get("font_size")
            if font_size and font_size < 14:  # Too small for title
                return False

            # Check for title capitalization patterns
            if self._has_title_capitalization(text):
                return True

            return False

        except Exception as e:
            self.logger.warning("Error checking article start", error=str(e))
            return False

    def _has_title_capitalization(self, text: str) -> bool:
        """Check if text has title-like capitalization."""
        words = text.split()
        if len(words) < 2:
            return False

        # Count capitalized words (excluding common articles/prepositions)
        minor_words = {
            "a",
            "an",
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
        }
        capitalized_count = 0

        for i, word in enumerate(words):
            # First word should always be capitalized
            if i == 0:
                if word[0].isupper():
                    capitalized_count += 1
            else:
                # Other words: capitalize unless minor word
                clean_word = re.sub(r"[^a-zA-Z]", "", word.lower())
                if clean_word not in minor_words and word[0].isupper():
                    capitalized_count += 1
                elif clean_word in minor_words and not word[0].isupper():
                    # Minor words should be lowercase (except first word)
                    capitalized_count += 0.5

        # At least 60% proper capitalization
        return (capitalized_count / len(words)) >= 0.6

    def _rank_article_starts(
        self, start_nodes: List[str], graph: SemanticGraph
    ) -> List[str]:
        """Rank article start nodes by likelihood."""
        try:
            node_scores = []

            for node_id in start_nodes:
                node = graph.get_node(node_id)
                if not node:
                    continue

                node_data = node.to_graph_data()
                score = self._calculate_start_score(node_data, graph)
                node_scores.append((node_id, score))

            # Sort by score descending
            node_scores.sort(key=lambda x: x[1], reverse=True)

            return [node_id for node_id, _ in node_scores]

        except Exception as e:
            self.logger.warning("Error ranking article starts", error=str(e))
            return start_nodes

    def _calculate_start_score(self, node_data: Any, graph: SemanticGraph) -> float:
        """Calculate likelihood score for article start node."""
        score = 0.0

        # Base confidence
        score += node_data.confidence * 0.4

        # Classification bonus
        if node_data.classification == BlockType.TITLE:
            score += 0.3
        elif node_data.classification == BlockType.SUBTITLE:
            score += 0.2
        elif node_data.classification == BlockType.HEADING:
            score += 0.1

        # Position bonus (higher on page is better)
        if node_data.bbox:
            # Normalize y position (0 = top of page)
            position_score = max(0, 1 - node_data.bbox.y0 / 500)
            score += position_score * 0.2

        # Font size bonus
        font_size = node_data.metadata.get("font_size", 12)
        if font_size > 16:
            score += 0.1

        # Text characteristics
        if node_data.text:
            text = node_data.text.strip()
            word_count = len(text.split())

            # Optimal title length
            if 5 <= word_count <= 12:
                score += 0.1
            elif 3 <= word_count <= 20:
                score += 0.05

            # Title capitalization
            if self._has_title_capitalization(text):
                score += 0.1

        return min(1.0, score)

    def traverse_article(
        self,
        start_node_id: str,
        graph: SemanticGraph,
        visited_nodes: Optional[Set[str]] = None,
    ) -> TraversalPath:
        """
        Traverse graph to collect article components starting from a title.

        Args:
            start_node_id: Starting node ID (typically a title)
            graph: Semantic graph to traverse
            visited_nodes: Previously visited nodes to avoid

        Returns:
            TraversalPath containing the article components
        """
        try:
            self.logger.debug(
                "Starting article traversal", start_node=start_node_id[:8]
            )

            # Initialize traversal state
            if visited_nodes is None:
                visited_nodes = set()

            path = TraversalPath(
                path_id=str(uuid.uuid4()),
                traversal_method="depth_first_with_continuation",
            )

            # Start traversal from the title node
            start_node = graph.get_node(start_node_id)
            if not start_node:
                raise ReconstructionError(f"Start node not found: {start_node_id}")

            start_data = start_node.to_graph_data()
            path.start_page = start_data.page_num
            path.end_page = start_data.page_num

            # Add start node to path
            path.add_node(start_node_id, confidence=start_data.confidence)
            if start_data.classification:
                path.component_types.append(start_data.classification)

            visited_nodes.add(start_node_id)

            # Traverse using depth-first search with continuation awareness
            self._traverse_depth_first(start_node_id, graph, path, visited_nodes)

            # Handle continuation markers
            self._process_continuations(path, graph, visited_nodes)

            self.logger.debug(
                "Article traversal completed",
                path_length=path.path_length,
                pages=f"{path.start_page}-{path.end_page}",
                avg_confidence=path.average_confidence,
            )

            return path

        except Exception as e:
            self.logger.error("Error in article traversal", error=str(e), exc_info=True)
            raise ReconstructionError(f"Failed to traverse article: {e}")

    def _traverse_depth_first(
        self,
        current_node_id: str,
        graph: SemanticGraph,
        path: TraversalPath,
        visited_nodes: Set[str],
    ):
        """Perform depth-first traversal to collect article components."""
        try:
            if path.path_length >= self.config.max_article_components:
                return

            # Get successors with relevant edge types
            relevant_edges = [
                EdgeType.FOLLOWS,
                EdgeType.BELONGS_TO,
                EdgeType.CONTINUES_ON,
            ]

            successors = []
            for edge_type in relevant_edges:
                edge_successors = graph.get_successors(current_node_id, edge_type)
                for successor_id in edge_successors:
                    if successor_id not in visited_nodes:
                        # Get edge confidence
                        edge_attrs = graph.graph.edges.get(
                            (current_node_id, successor_id), {}
                        )
                        confidence = edge_attrs.get("confidence", 0.0)

                        if confidence >= self.config.min_connection_confidence:
                            successors.append((successor_id, edge_type, confidence))

            # Sort successors by confidence
            successors.sort(key=lambda x: x[2], reverse=True)

            # Process each successor
            for successor_id, edge_type, confidence in successors:
                if successor_id in visited_nodes:
                    continue

                successor_node = graph.get_node(successor_id)
                if not successor_node:
                    continue

                successor_data = successor_node.to_graph_data()

                # Check if this is a valid article component
                if not self._is_valid_article_component(successor_data):
                    continue

                # Add to path
                path.add_node(successor_id, edge_type, confidence)
                visited_nodes.add(successor_id)

                # Update path page range
                if successor_data.page_num < path.start_page:
                    path.start_page = successor_data.page_num
                if successor_data.page_num > path.end_page:
                    path.end_page = successor_data.page_num

                # Update component types
                if successor_data.classification:
                    path.component_types.append(successor_data.classification)

                # Continue traversal from this node
                self._traverse_depth_first(successor_id, graph, path, visited_nodes)

        except Exception as e:
            self.logger.warning("Error in depth-first traversal", error=str(e))

    def _is_valid_article_component(self, node_data: Any) -> bool:
        """Check if a node is a valid article component."""
        try:
            # Must be text block
            if node_data.node_type != NodeType.TEXT_BLOCK:
                return False

            # Must have reasonable confidence
            if node_data.confidence < self.config.min_connection_confidence:
                return False

            # Must have meaningful text
            if not node_data.text or len(node_data.text.strip()) < 10:
                return False

            # Exclude certain block types
            excluded_types = {BlockType.HEADER, BlockType.FOOTER, BlockType.PAGE_NUMBER}
            if node_data.classification in excluded_types:
                return False

            # Filter out advertisements if configured
            if (
                self.config.filter_advertisements
                and node_data.classification == BlockType.ADVERTISEMENT
            ):
                return False

            return True

        except Exception:
            return False

    def _process_continuations(
        self, path: TraversalPath, graph: SemanticGraph, visited_nodes: Set[str]
    ):
        """Process continuation markers to extend article across pages."""
        try:
            continuation_markers = self._find_continuation_markers(path, graph)

            for marker in continuation_markers:
                if marker.marker_type == ContinuationType.FORWARD:
                    # Look for continuation on target page
                    self._follow_continuation(marker, path, graph, visited_nodes)
                elif marker.marker_type == ContinuationType.JUMP:
                    # Handle page jumps
                    self._follow_page_jump(marker, path, graph, visited_nodes)

        except Exception as e:
            self.logger.warning("Error processing continuations", error=str(e))

    def _find_continuation_markers(
        self, path: TraversalPath, graph: SemanticGraph
    ) -> List[ContinuationMarker]:
        """Find continuation markers in the current path."""
        markers = []

        try:
            for node_id in path.node_ids:
                node = graph.get_node(node_id)
                if not node:
                    continue

                node_data = node.to_graph_data()
                if not node_data.text:
                    continue

                # Check text for continuation patterns
                for pattern in self.config.continuation_patterns:
                    matches = re.finditer(pattern, node_data.text, re.IGNORECASE)

                    for match in matches:
                        marker = self._create_continuation_marker(
                            match, pattern, node_data
                        )
                        if (
                            marker
                            and marker.confidence
                            >= self.config.min_continuation_confidence
                        ):
                            markers.append(marker)

            return markers

        except Exception as e:
            self.logger.warning("Error finding continuation markers", error=str(e))
            return []

    def _create_continuation_marker(
        self, match: re.Match, pattern: str, node_data: Any
    ) -> Optional[ContinuationMarker]:
        """Create continuation marker from regex match."""
        try:
            marker_text = match.group(0)

            # Determine continuation type and target page
            if "continued" in marker_text.lower() or "cont" in marker_text.lower():
                if "from" in marker_text.lower():
                    marker_type = ContinuationType.BACKWARD
                else:
                    marker_type = ContinuationType.FORWARD
            elif "see" in marker_text.lower() or "turn" in marker_text.lower():
                marker_type = ContinuationType.JUMP
            else:
                marker_type = ContinuationType.FORWARD

            # Extract target page number
            target_page = None
            if match.groups():
                try:
                    target_page = int(match.group(1))
                except (ValueError, IndexError):
                    pass

            # Calculate confidence based on pattern specificity
            confidence = self._calculate_continuation_confidence(marker_text, pattern)

            return ContinuationMarker(
                marker_type=marker_type,
                source_page=node_data.page_num,
                target_page=target_page,
                marker_text=marker_text,
                confidence=confidence,
                pattern_used=pattern,
                extraction_method="regex",
            )

        except Exception as e:
            self.logger.warning("Error creating continuation marker", error=str(e))
            return None

    def _calculate_continuation_confidence(
        self, marker_text: str, pattern: str
    ) -> float:
        """Calculate confidence for continuation marker."""
        confidence = 0.5  # Base confidence

        # Specific keywords boost confidence
        if "continued" in marker_text.lower():
            confidence += 0.3
        if "page" in marker_text.lower():
            confidence += 0.2
        if re.search(r"\d+", marker_text):  # Has page number
            confidence += 0.2

        # Pattern specificity
        if "\\d+" in pattern:  # Pattern includes page number
            confidence += 0.1

        return min(1.0, confidence)

    def _follow_continuation(
        self,
        marker: ContinuationMarker,
        path: TraversalPath,
        graph: SemanticGraph,
        visited_nodes: Set[str],
    ):
        """Follow a continuation marker to extend the article."""
        try:
            target_page = marker.target_page or (marker.source_page + 1)

            # Find nodes on target page that could be continuations
            page_nodes = graph.get_nodes_by_page(target_page)

            for node in page_nodes:
                if node.node_id in visited_nodes:
                    continue

                node_data = node.to_graph_data()

                # Check if this could be the continuation
                if self._is_likely_continuation(node_data, marker):
                    # Add to path and continue traversal
                    path.add_node(
                        node.node_id, EdgeType.CONTINUES_ON, marker.confidence
                    )
                    visited_nodes.add(node.node_id)

                    if node_data.classification:
                        path.component_types.append(node_data.classification)

                    # Update page range
                    if target_page > path.end_page:
                        path.end_page = target_page

                    # Continue traversal from continuation
                    self._traverse_depth_first(node.node_id, graph, path, visited_nodes)
                    break

        except Exception as e:
            self.logger.warning("Error following continuation", error=str(e))

    def _follow_page_jump(
        self,
        marker: ContinuationMarker,
        path: TraversalPath,
        graph: SemanticGraph,
        visited_nodes: Set[str],
    ):
        """Follow a page jump continuation."""
        # Similar to _follow_continuation but specifically for jumps
        self._follow_continuation(marker, path, graph, visited_nodes)

    def _is_likely_continuation(
        self, node_data: Any, marker: ContinuationMarker
    ) -> bool:
        """Check if a node is likely a continuation of an article."""
        try:
            # Must be valid article component
            if not self._is_valid_article_component(node_data):
                return False

            # Should be body text or heading
            if node_data.classification not in [
                BlockType.BODY,
                BlockType.HEADING,
                BlockType.SUBTITLE,
            ]:
                return False

            # Check for continuation indicators in text
            if node_data.text:
                text = node_data.text.lower()

                # Positive indicators
                if any(
                    phrase in text
                    for phrase in ["continued from", "cont. from", "from page"]
                ):
                    return True

                # Should not start with title-like text
                if self._has_title_capitalization(node_data.text):
                    return False

            # Position should be reasonable (not at very top unless it's a clear continuation)
            if node_data.bbox and node_data.bbox.y0 < 100:
                # Very top of page - check for continuation markers
                return (
                    "continued" in node_data.text.lower() if node_data.text else False
                )

            return True

        except Exception:
            return False
