"""
Ambiguity resolution for article reconstruction.

This module provides algorithms for resolving ambiguous connections
during graph traversal using confidence scores and spatial relationships.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog

from ..graph import EdgeType, SemanticGraph
from ..layout.types import BlockType
from .types import (
    ConnectionScore,
    ReconstructionConfig,
    ReconstructionError,
    TraversalPath,
)

logger = structlog.get_logger(__name__)


class AmbiguityResolver:
    """
    Resolver for ambiguous connections during article reconstruction.

    Uses confidence scores, spatial relationships, and content analysis
    to resolve cases where multiple traversal paths are possible.
    """

    def __init__(self, config: Optional[ReconstructionConfig] = None):
        """
        Initialize ambiguity resolver.

        Args:
            config: Reconstruction configuration
        """
        self.config = config or ReconstructionConfig()
        self.logger = logger.bind(component="AmbiguityResolver")

        # Resolution statistics
        self.resolution_stats = {
            "total_ambiguities": 0,
            "resolved_by_confidence": 0,
            "resolved_by_spatial": 0,
            "resolved_by_content": 0,
            "unresolved": 0,
        }

        self.logger.debug("Initialized ambiguity resolver")

    def resolve_connection_ambiguity(
        self, source_node_id: str, candidate_targets: List[str], graph: SemanticGraph
    ) -> Tuple[Optional[str], ConnectionScore]:
        """
        Resolve ambiguity when multiple target nodes are possible.

        Args:
            source_node_id: Source node ID
            candidate_targets: List of candidate target node IDs
            graph: Semantic graph

        Returns:
            Tuple of (best_target_id, connection_score)
        """
        try:
            self.logger.debug(
                "Resolving connection ambiguity",
                source=source_node_id[:8],
                candidates=len(candidate_targets),
            )

            self.resolution_stats["total_ambiguities"] += 1

            if not candidate_targets:
                return None, ConnectionScore(0, 0, 0, 0, 0, EdgeType.FOLLOWS)

            if len(candidate_targets) == 1:
                # No ambiguity - return the single candidate
                target_id = candidate_targets[0]
                score = self._calculate_connection_score(
                    source_node_id, target_id, graph
                )
                return target_id, score

            # Calculate scores for all candidates
            candidate_scores = []
            for target_id in candidate_targets:
                score = self._calculate_connection_score(
                    source_node_id, target_id, graph
                )
                candidate_scores.append((target_id, score))

            # Sort by total score descending
            candidate_scores.sort(key=lambda x: x[1].total_score, reverse=True)

            # Check if there's a clear winner
            best_target, best_score = candidate_scores[0]

            if len(candidate_scores) > 1:
                second_score = candidate_scores[1][1].total_score
                score_difference = best_score.total_score - second_score

                # If scores are very close, apply additional disambiguation
                if score_difference < 0.1:
                    best_target, best_score = self._disambiguate_close_scores(
                        candidate_scores[:3], source_node_id, graph
                    )

            # Update resolution statistics
            self._update_resolution_stats(best_score)

            self.logger.debug(
                "Resolved connection ambiguity",
                chosen_target=best_target[:8],
                score=best_score.total_score,
                reasoning=best_score.reasoning,
            )

            return best_target, best_score

        except Exception as e:
            self.logger.error("Error resolving connection ambiguity", error=str(e))
            self.resolution_stats["unresolved"] += 1

            # Return first candidate as fallback
            if candidate_targets:
                fallback_score = ConnectionScore(0.3, 0, 0, 0, 0.3, EdgeType.FOLLOWS)
                return candidate_targets[0], fallback_score

            return None, ConnectionScore(0, 0, 0, 0, 0, EdgeType.FOLLOWS)

    def _calculate_connection_score(
        self, source_id: str, target_id: str, graph: SemanticGraph
    ) -> ConnectionScore:
        """Calculate connection score between two nodes."""
        try:
            source_node = graph.get_node(source_id)
            target_node = graph.get_node(target_id)

            if not source_node or not target_node:
                return ConnectionScore(0, 0, 0, 0, 0, EdgeType.FOLLOWS)

            source_data = source_node.to_graph_data()
            target_data = target_node.to_graph_data()

            # Get edge information if edge exists
            edge_confidence = 0.0
            connection_type = EdgeType.FOLLOWS

            if graph.graph.has_edge(source_id, target_id):
                edge_attrs = graph.graph.edges[source_id, target_id]
                edge_confidence = edge_attrs.get("confidence", 0.0)
                connection_type = EdgeType(edge_attrs.get("edge_type", "follows"))

            # Calculate spatial proximity
            spatial_proximity = self._calculate_spatial_proximity(
                source_data.bbox, target_data.bbox
            )

            # Calculate semantic similarity
            semantic_similarity = self._calculate_semantic_similarity(
                source_data, target_data
            )

            # Check for continuation markers
            has_continuation = self._has_continuation_marker(source_data, target_data)

            # Calculate combined score
            score = ConnectionScore.calculate(
                confidence=edge_confidence,
                spatial_proximity=spatial_proximity,
                semantic_similarity=semantic_similarity,
                has_continuation=has_continuation,
                connection_type=connection_type,
                config=self.config,
            )

            return score

        except Exception as e:
            self.logger.warning("Error calculating connection score", error=str(e))
            return ConnectionScore(0, 0, 0, 0, 0, EdgeType.FOLLOWS)

    def _calculate_spatial_proximity(self, bbox1: Any, bbox2: Any) -> float:
        """Calculate spatial proximity between two bounding boxes."""
        try:
            if not bbox1 or not bbox2:
                return 1000.0  # Large distance for missing bounding boxes

            # Calculate center points
            center1_x = (bbox1.x0 + bbox1.x1) / 2
            center1_y = (bbox1.y0 + bbox1.y1) / 2
            center2_x = (bbox2.x0 + bbox2.x1) / 2
            center2_y = (bbox2.y0 + bbox2.y1) / 2

            # Euclidean distance
            distance = np.sqrt(
                (center2_x - center1_x) ** 2 + (center2_y - center1_y) ** 2
            )

            return float(distance)

        except Exception:
            return 1000.0

    def _calculate_semantic_similarity(self, data1: Any, data2: Any) -> float:
        """Calculate semantic similarity between two text blocks."""
        try:
            if not data1.text or not data2.text:
                return 0.0

            text1 = data1.text.lower().strip()
            text2 = data2.text.lower().strip()

            # Simple similarity metrics
            similarity = 0.0

            # Classification compatibility
            if data1.classification and data2.classification:
                compatible_pairs = [
                    (BlockType.TITLE, BlockType.SUBTITLE),
                    (BlockType.TITLE, BlockType.BYLINE),
                    (BlockType.SUBTITLE, BlockType.BODY),
                    (BlockType.BYLINE, BlockType.BODY),
                    (BlockType.BODY, BlockType.BODY),
                    (BlockType.HEADING, BlockType.BODY),
                    (BlockType.QUOTE, BlockType.BODY),
                ]

                pair = (data1.classification, data2.classification)
                reverse_pair = (data2.classification, data1.classification)

                if pair in compatible_pairs or reverse_pair in compatible_pairs:
                    similarity += 0.4

            # Content continuity indicators
            if self._suggests_continuation(text1, text2):
                similarity += 0.3

            # Length compatibility (similar lengths suggest related content)
            len1, len2 = len(text1.split()), len(text2.split())
            if len1 > 0 and len2 > 0:
                length_ratio = min(len1, len2) / max(len1, len2)
                similarity += length_ratio * 0.2

            # Common word overlap (simple)
            words1 = set(text1.split())
            words2 = set(text2.split())
            if words1 and words2:
                overlap = len(words1.intersection(words2))
                union = len(words1.union(words2))
                jaccard = overlap / union if union > 0 else 0
                similarity += jaccard * 0.1

            return min(1.0, similarity)

        except Exception:
            return 0.0

    def _suggests_continuation(self, text1: str, text2: str) -> bool:
        """Check if text2 suggests continuation of text1."""
        # Check if text1 ends mid-sentence
        if text1.endswith((",", ";", "-", "and", "or", "but")):
            return True

        # Check if text2 starts with continuation words
        continuation_starters = [
            "however",
            "moreover",
            "furthermore",
            "additionally",
            "meanwhile",
            "consequently",
            "therefore",
            "thus",
        ]

        text2_start = text2.split()[:2]
        for starter in continuation_starters:
            if starter in [word.lower().strip(".,;:") for word in text2_start]:
                return True

        return False

    def _has_continuation_marker(self, data1: Any, data2: Any) -> bool:
        """Check for explicit continuation markers."""
        if not data1.text or not data2.text:
            return False

        text1 = data1.text.lower()
        text2 = data2.text.lower()

        # Check for continuation patterns
        continuation_patterns = [
            "continued on page",
            "see page",
            "turn to page",
            "continued from",
            "from page",
        ]

        for pattern in continuation_patterns:
            if pattern in text1 or pattern in text2:
                return True

        return False

    def _disambiguate_close_scores(
        self,
        candidates: List[Tuple[str, ConnectionScore]],
        source_id: str,
        graph: SemanticGraph,
    ) -> Tuple[str, ConnectionScore]:
        """Disambiguate when connection scores are very close."""
        try:
            # Apply additional criteria for tie-breaking

            enhanced_scores = []

            for target_id, score in candidates:
                enhanced_score = score.total_score

                # Prefer sequential page order
                if self.config.prefer_sequential_pages:
                    source_node = graph.get_node(source_id)
                    target_node = graph.get_node(target_id)

                    if source_node and target_node:
                        source_data = source_node.to_graph_data()
                        target_data = target_node.to_graph_data()

                        # Bonus for next page
                        if target_data.page_num == source_data.page_num + 1:
                            enhanced_score += 0.15
                        # Bonus for same page
                        elif target_data.page_num == source_data.page_num:
                            enhanced_score += 0.1
                        # Penalty for page gaps
                        elif abs(target_data.page_num - source_data.page_num) > 2:
                            enhanced_score -= 0.1

                # Prefer stronger edge types
                if score.connection_type == EdgeType.FOLLOWS:
                    enhanced_score += 0.05
                elif score.connection_type == EdgeType.CONTINUES_ON:
                    enhanced_score += 0.1

                enhanced_scores.append((target_id, enhanced_score, score))

            # Return best enhanced score
            best = max(enhanced_scores, key=lambda x: x[1])

            # Update reasoning
            best_score = best[2]
            best_score.reasoning.append("tie_breaking_applied")

            return best[0], best_score

        except Exception as e:
            self.logger.warning("Error in disambiguation", error=str(e))
            return candidates[0]

    def _update_resolution_stats(self, score: ConnectionScore):
        """Update resolution statistics based on scoring factors."""
        if score.confidence_score > 0.7:
            self.resolution_stats["resolved_by_confidence"] += 1
        elif score.spatial_score > 0.7:
            self.resolution_stats["resolved_by_spatial"] += 1
        elif score.semantic_score > 0.6:
            self.resolution_stats["resolved_by_content"] += 1

    def resolve_path_conflicts(
        self, competing_paths: List[TraversalPath], graph: SemanticGraph
    ) -> TraversalPath:
        """
        Resolve conflicts between competing traversal paths.

        Args:
            competing_paths: List of competing traversal paths
            graph: Semantic graph

        Returns:
            Best path among the competitors
        """
        try:
            self.logger.debug("Resolving path conflicts", paths=len(competing_paths))

            if not competing_paths:
                raise ReconstructionError("No paths to resolve")

            if len(competing_paths) == 1:
                return competing_paths[0]

            # Score each path
            path_scores = []

            for path in competing_paths:
                score = self._score_traversal_path(path, graph)
                path_scores.append((path, score))

            # Sort by score descending
            path_scores.sort(key=lambda x: x[1], reverse=True)

            best_path = path_scores[0][0]

            self.logger.debug(
                "Resolved path conflict",
                chosen_path=best_path.path_id[:8],
                score=path_scores[0][1],
                length=best_path.path_length,
            )

            return best_path

        except Exception as e:
            self.logger.error("Error resolving path conflicts", error=str(e))
            return competing_paths[0]  # Return first path as fallback

    def _score_traversal_path(self, path: TraversalPath, graph: SemanticGraph) -> float:
        """Score a traversal path for quality assessment."""
        try:
            score = 0.0

            # Base confidence score
            score += path.average_confidence * 0.4

            # Path length bonus (prefer longer, more complete articles)
            length_score = min(1.0, path.path_length / 20)  # Normalize to 20 components
            score += length_score * 0.2

            # Component diversity bonus
            unique_types = len(set(path.component_types))
            diversity_score = min(1.0, unique_types / 5)  # Up to 5 different types
            score += diversity_score * 0.15

            # Sequential page bonus
            if path.spans_multiple_pages:
                page_range = path.end_page - path.start_page + 1
                expected_pages = path.path_length / 10  # Rough estimate
                page_score = min(1.0, page_range / max(1, expected_pages))
                score += page_score * 0.1

            # Structural completeness
            has_title = BlockType.TITLE in path.component_types
            has_body = BlockType.BODY in path.component_types

            if has_title:
                score += 0.1
            if has_body:
                score += 0.05
            if has_title and has_body:
                score += 0.1  # Bonus for having both

            return min(1.0, score)

        except Exception as e:
            self.logger.warning("Error scoring traversal path", error=str(e))
            return 0.0

    def get_resolution_statistics(self) -> Dict[str, Any]:
        """Get ambiguity resolution statistics."""
        total = max(1, self.resolution_stats["total_ambiguities"])

        return {
            "total_ambiguities": self.resolution_stats["total_ambiguities"],
            "resolution_breakdown": {
                "by_confidence": self.resolution_stats["resolved_by_confidence"]
                / total,
                "by_spatial": self.resolution_stats["resolved_by_spatial"] / total,
                "by_content": self.resolution_stats["resolved_by_content"] / total,
                "unresolved": self.resolution_stats["unresolved"] / total,
            },
            "success_rate": 1 - (self.resolution_stats["unresolved"] / total),
        }
