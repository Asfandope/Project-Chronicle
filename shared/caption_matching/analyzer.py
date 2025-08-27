"""
Spatial analysis for image-caption matching.
"""

from typing import Dict, List, Optional

import structlog

from ..graph.types import GraphNode, NodeType, SemanticGraph
from .types import (
    BoundingBox,
    MatchingError,
    MatchType,
    ProximityScore,
    SpatialConfig,
)

logger = structlog.get_logger(__name__)


class SpatialAnalyzer:
    """Analyzes spatial relationships between images and text for caption matching."""

    def __init__(self, config: Optional[SpatialConfig] = None):
        """
        Initialize spatial analyzer.

        Args:
            config: Spatial matching configuration
        """
        self.config = config or SpatialConfig()
        self.logger = logger.bind(component="SpatialAnalyzer")

        # Cache for performance
        self._image_nodes_cache = {}
        self._caption_nodes_cache = {}
        self._bbox_cache = {}

    def find_image_nodes(self, graph: SemanticGraph) -> List[GraphNode]:
        """
        Find all image nodes in the semantic graph.

        Args:
            graph: Semantic graph to analyze

        Returns:
            List of image nodes found
        """
        try:
            self.logger.debug("Finding image nodes in graph")

            # Check cache first
            cache_key = id(graph)
            if cache_key in self._image_nodes_cache:
                return self._image_nodes_cache[cache_key]

            image_nodes = []

            for node_id, node in graph.nodes.items():
                if self._is_image_node(node):
                    image_nodes.append(node)

                    self.logger.debug(
                        "Found image node",
                        node_id=node_id,
                        node_type=node.node_type.value,
                        bbox=self._extract_bbox_from_metadata(node.metadata),
                    )

            # Cache result
            self._image_nodes_cache[cache_key] = image_nodes

            self.logger.info(
                "Image node discovery completed",
                total_images=len(image_nodes),
                node_types=[n.node_type.value for n in image_nodes],
            )

            return image_nodes

        except Exception as e:
            self.logger.error("Error finding image nodes", error=str(e))
            raise MatchingError(f"Failed to find image nodes: {e}")

    def find_potential_caption_nodes(
        self, graph: SemanticGraph, image_node: GraphNode
    ) -> List[GraphNode]:
        """
        Find potential caption nodes near an image.

        Args:
            graph: Semantic graph to analyze
            image_node: Image node to find captions for

        Returns:
            List of potential caption nodes
        """
        try:
            self.logger.debug(
                "Finding caption candidates for image", image_node_id=image_node.node_id
            )

            image_bbox = self._get_node_bbox(image_node)
            if not image_bbox:
                self.logger.warning(
                    "No bounding box for image node", node_id=image_node.node_id
                )
                return []

            caption_candidates = []

            # Search all text nodes for potential captions
            for node_id, node in graph.nodes.items():
                if self._is_potential_caption_node(node, image_node):
                    caption_bbox = self._get_node_bbox(node)
                    if not caption_bbox:
                        continue

                    # Check if within search distance
                    distance = image_bbox.distance_to(caption_bbox)
                    if distance <= self.config.max_search_distance:
                        caption_candidates.append(node)

                        self.logger.debug(
                            "Found caption candidate",
                            caption_node_id=node_id,
                            distance=distance,
                            text_preview=node.content[:50] if node.content else "",
                        )

            # Sort by distance (closest first)
            caption_candidates.sort(
                key=lambda n: image_bbox.distance_to(self._get_node_bbox(n))
            )

            self.logger.info(
                "Caption candidate discovery completed",
                image_node_id=image_node.node_id,
                candidates_found=len(caption_candidates),
            )

            return caption_candidates

        except Exception as e:
            self.logger.error(
                "Error finding caption candidates",
                image_node_id=image_node.node_id,
                error=str(e),
            )
            raise MatchingError(
                f"Failed to find caption candidates: {e}",
                image_node_id=image_node.node_id,
            )

    def calculate_proximity_score(
        self,
        image_bbox: BoundingBox,
        caption_bbox: BoundingBox,
        graph: Optional[SemanticGraph] = None,
    ) -> ProximityScore:
        """
        Calculate spatial proximity score between image and caption.

        Args:
            image_bbox: Bounding box of image
            caption_bbox: Bounding box of caption
            graph: Optional graph for layout analysis

        Returns:
            Detailed proximity score
        """
        try:
            score = ProximityScore()

            # Basic distance measurements
            score.euclidean_distance = image_bbox.distance_to(caption_bbox)
            score.vertical_distance = abs(caption_bbox.center_y - image_bbox.center_y)
            score.horizontal_distance = abs(caption_bbox.center_x - image_bbox.center_x)

            # Calculate alignment score
            score.alignment_score = self._calculate_alignment_score(
                image_bbox, caption_bbox
            )

            # Calculate relative position score
            score.relative_position_score = self._calculate_position_score(
                image_bbox, caption_bbox
            )

            # Calculate containment score
            score.containment_score = self._calculate_containment_score(
                image_bbox, caption_bbox
            )

            # Calculate reading order score
            score.reading_order_score = self._calculate_reading_order_score(
                image_bbox, caption_bbox
            )

            # Calculate column awareness score (if graph available)
            if graph:
                score.column_awareness_score = self._calculate_column_awareness_score(
                    image_bbox, caption_bbox, graph
                )
            else:
                score.column_awareness_score = 0.5  # Neutral score

            # Calculate overall score
            score.calculate_overall_score()

            self.logger.debug(
                "Proximity score calculated",
                euclidean_distance=score.euclidean_distance,
                alignment_score=score.alignment_score,
                position_score=score.relative_position_score,
                overall_score=score.overall_score,
            )

            return score

        except Exception as e:
            self.logger.error("Error calculating proximity score", error=str(e))
            # Return default score on error
            return ProximityScore()

    def determine_match_type(
        self, image_bbox: BoundingBox, caption_bbox: BoundingBox
    ) -> MatchType:
        """
        Determine the type of spatial match between image and caption.

        Args:
            image_bbox: Bounding box of image
            caption_bbox: Bounding box of caption

        Returns:
            Type of spatial match
        """
        try:
            # Check if caption is contained within or overlapping image
            if caption_bbox.overlaps(image_bbox):
                return MatchType.EMBEDDED

            # Check vertical relationships
            vertical_threshold = min(image_bbox.height, caption_bbox.height) * 0.5
            horizontal_alignment_threshold = (
                min(image_bbox.width, caption_bbox.width) * 0.3
            )

            # Check if relatively aligned horizontally
            horizontal_overlap = min(image_bbox.x1, caption_bbox.x1) - max(
                image_bbox.x0, caption_bbox.x0
            )
            is_horizontally_aligned = (
                horizontal_overlap >= horizontal_alignment_threshold
            )

            # Caption below image
            if (
                caption_bbox.y0 >= image_bbox.y1 - vertical_threshold
                and caption_bbox.y0 <= image_bbox.y1 + vertical_threshold * 2
                and is_horizontally_aligned
            ):
                return MatchType.DIRECT_BELOW

            # Caption above image
            if (
                caption_bbox.y1 <= image_bbox.y0 + vertical_threshold
                and caption_bbox.y1 >= image_bbox.y0 - vertical_threshold * 2
                and is_horizontally_aligned
            ):
                return MatchType.DIRECT_ABOVE

            # Side by side
            side_threshold = min(image_bbox.width, caption_bbox.width) * 0.5
            vertical_alignment_threshold = (
                min(image_bbox.height, caption_bbox.height) * 0.3
            )

            vertical_overlap = min(image_bbox.y1, caption_bbox.y1) - max(
                image_bbox.y0, caption_bbox.y0
            )
            is_vertically_aligned = vertical_overlap >= vertical_alignment_threshold

            if (
                (
                    caption_bbox.x0 >= image_bbox.x1 - side_threshold
                    and caption_bbox.x0 <= image_bbox.x1 + side_threshold * 2
                )
                or (
                    caption_bbox.x1 <= image_bbox.x0 + side_threshold
                    and caption_bbox.x1 >= image_bbox.x0 - side_threshold * 2
                )
            ) and is_vertically_aligned:
                return MatchType.SIDE_BY_SIDE

            # If close enough, consider it grouped
            distance = image_bbox.distance_to(caption_bbox)
            if distance <= self.config.preferred_caption_distance * 2:
                return MatchType.GROUPED

            # Otherwise it's distant
            return MatchType.DISTANT

        except Exception as e:
            self.logger.warning("Error determining match type", error=str(e))
            return MatchType.DISTANT

    def _is_image_node(self, node: GraphNode) -> bool:
        """Check if node represents an image."""
        # Check node type
        if node.node_type in [NodeType.IMAGE, NodeType.FIGURE]:
            return True

        # Check metadata for image indicators
        metadata = node.metadata or {}

        # Look for image file extensions
        if "filename" in metadata:
            filename = metadata["filename"].lower()
            image_extensions = {
                ".jpg",
                ".jpeg",
                ".png",
                ".gif",
                ".bmp",
                ".webp",
                ".svg",
            }
            if any(filename.endswith(ext) for ext in image_extensions):
                return True

        # Look for image-related tags or attributes
        image_indicators = {"img", "image", "figure", "photo", "picture"}
        if any(indicator in str(metadata).lower() for indicator in image_indicators):
            return True

        return False

    def _is_potential_caption_node(
        self, node: GraphNode, image_node: GraphNode
    ) -> bool:
        """Check if node could be a caption for the given image."""
        # Must be text content
        if not node.content or not node.content.strip():
            return False

        # Exclude certain node types that are unlikely to be captions
        excluded_types = {NodeType.TITLE, NodeType.HEADING, NodeType.NAVIGATION}
        if node.node_type in excluded_types:
            return False

        # Check for caption indicators in content or metadata
        content_lower = node.content.lower()
        caption_indicators = {
            "photo",
            "image",
            "picture",
            "figure",
            "caption",
            "credit",
            "courtesy",
            "source",
            "by ",
            "photographer",
            "illustration",
        }

        # Strong indicators suggest it's likely a caption
        has_caption_indicators = any(
            indicator in content_lower for indicator in caption_indicators
        )

        # Check length - captions are usually short to medium length
        word_count = len(node.content.split())
        is_reasonable_length = 2 <= word_count <= 100  # Reasonable caption length

        # Check metadata for caption indicators
        metadata = node.metadata or {}
        metadata_str = str(metadata).lower()
        has_metadata_indicators = any(
            indicator in metadata_str for indicator in caption_indicators
        )

        return (
            has_caption_indicators or has_metadata_indicators
        ) and is_reasonable_length

    def _get_node_bbox(self, node: GraphNode) -> Optional[BoundingBox]:
        """Extract bounding box from node metadata."""
        cache_key = node.node_id
        if cache_key in self._bbox_cache:
            return self._bbox_cache[cache_key]

        bbox = self._extract_bbox_from_metadata(node.metadata)
        self._bbox_cache[cache_key] = bbox
        return bbox

    def _extract_bbox_from_metadata(
        self, metadata: Optional[Dict]
    ) -> Optional[BoundingBox]:
        """Extract bounding box coordinates from metadata."""
        if not metadata:
            return None

        # Try different possible bbox formats in metadata
        bbox_keys = ["bbox", "bounding_box", "bounds", "rect", "coordinates"]

        for key in bbox_keys:
            if key in metadata:
                bbox_data = metadata[key]

                # Handle different bbox formats
                try:
                    if isinstance(bbox_data, (list, tuple)) and len(bbox_data) == 4:
                        return BoundingBox(
                            x0=float(bbox_data[0]),
                            y0=float(bbox_data[1]),
                            x1=float(bbox_data[2]),
                            y1=float(bbox_data[3]),
                        )
                    elif isinstance(bbox_data, dict):
                        # Handle {x0, y0, x1, y1} or {left, top, right, bottom} format
                        if all(k in bbox_data for k in ["x0", "y0", "x1", "y1"]):
                            return BoundingBox(
                                x0=float(bbox_data["x0"]),
                                y0=float(bbox_data["y0"]),
                                x1=float(bbox_data["x1"]),
                                y1=float(bbox_data["y1"]),
                            )
                        elif all(
                            k in bbox_data for k in ["left", "top", "right", "bottom"]
                        ):
                            return BoundingBox(
                                x0=float(bbox_data["left"]),
                                y0=float(bbox_data["top"]),
                                x1=float(bbox_data["right"]),
                                y1=float(bbox_data["bottom"]),
                            )
                        elif all(k in bbox_data for k in ["x", "y", "width", "height"]):
                            return BoundingBox(
                                x0=float(bbox_data["x"]),
                                y0=float(bbox_data["y"]),
                                x1=float(bbox_data["x"]) + float(bbox_data["width"]),
                                y1=float(bbox_data["y"]) + float(bbox_data["height"]),
                            )
                except (ValueError, KeyError, TypeError):
                    continue

        return None

    def _calculate_alignment_score(
        self, image_bbox: BoundingBox, caption_bbox: BoundingBox
    ) -> float:
        """Calculate how well image and caption are aligned."""
        # Check horizontal alignment
        horizontal_overlap = min(image_bbox.x1, caption_bbox.x1) - max(
            image_bbox.x0, caption_bbox.x0
        )
        max_horizontal = max(image_bbox.width, caption_bbox.width)
        horizontal_alignment = (
            max(0.0, horizontal_overlap / max_horizontal) if max_horizontal > 0 else 0.0
        )

        # Check vertical alignment
        vertical_overlap = min(image_bbox.y1, caption_bbox.y1) - max(
            image_bbox.y0, caption_bbox.y0
        )
        max_vertical = max(image_bbox.height, caption_bbox.height)
        vertical_alignment = (
            max(0.0, vertical_overlap / max_vertical) if max_vertical > 0 else 0.0
        )

        # Prefer horizontal alignment for typical caption layouts
        return horizontal_alignment * 0.7 + vertical_alignment * 0.3

    def _calculate_position_score(
        self, image_bbox: BoundingBox, caption_bbox: BoundingBox
    ) -> float:
        """Calculate score based on expected caption position."""
        score = 0.5  # Neutral base score

        # Prefer captions below images (common convention)
        if self.config.prefer_below_captions:
            if caption_bbox.center_y > image_bbox.center_y:
                score += 0.3
            else:
                score -= 0.1

        # Bonus for close vertical proximity
        vertical_gap = abs(caption_bbox.center_y - image_bbox.center_y)
        if vertical_gap <= self.config.preferred_caption_distance:
            score += 0.2 * (1.0 - vertical_gap / self.config.preferred_caption_distance)

        return max(0.0, min(1.0, score))

    def _calculate_containment_score(
        self, image_bbox: BoundingBox, caption_bbox: BoundingBox
    ) -> float:
        """Calculate score for caption containment within image bounds."""
        if caption_bbox.overlaps(image_bbox):
            # Calculate overlap percentage
            overlap_x = min(image_bbox.x1, caption_bbox.x1) - max(
                image_bbox.x0, caption_bbox.x0
            )
            overlap_y = min(image_bbox.y1, caption_bbox.y1) - max(
                image_bbox.y0, caption_bbox.y0
            )
            overlap_area = overlap_x * overlap_y

            caption_area = caption_bbox.area
            if caption_area > 0:
                overlap_ratio = overlap_area / caption_area
                return (
                    overlap_ratio * 0.8
                )  # Contained captions get some bonus but not full

        return 0.0

    def _calculate_reading_order_score(
        self, image_bbox: BoundingBox, caption_bbox: BoundingBox
    ) -> float:
        """Calculate score based on natural reading order."""
        score = 0.5  # Neutral base

        # Left-to-right, top-to-bottom reading order
        # Captions should generally be below or to the right of images

        if (
            caption_bbox.center_y >= image_bbox.center_y
        ):  # Caption below or at same level
            score += 0.2

        if (
            caption_bbox.center_x >= image_bbox.center_x
        ):  # Caption to right or at same position
            score += 0.1

        # Penalize captions that are above and to the left (against reading order)
        if (
            caption_bbox.center_y < image_bbox.center_y
            and caption_bbox.center_x < image_bbox.center_x
        ):
            score -= 0.2

        return max(0.0, min(1.0, score))

    def _calculate_column_awareness_score(
        self, image_bbox: BoundingBox, caption_bbox: BoundingBox, graph: SemanticGraph
    ) -> float:
        """Calculate score considering column layout."""
        # This is a simplified implementation
        # In practice, would analyze the graph for column structures

        # For now, assume single column and give neutral score
        # TODO: Implement proper column detection from graph structure
        return 0.5

    def clear_cache(self) -> None:
        """Clear internal caches."""
        self._image_nodes_cache.clear()
        self._caption_nodes_cache.clear()
        self._bbox_cache.clear()
