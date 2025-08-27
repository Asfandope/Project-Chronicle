"""
Type definitions for article reconstruction.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from ..graph.types import EdgeType
from ..layout.types import BlockType


class ReconstructionError(Exception):
    """Base exception for article reconstruction errors."""

    def __init__(
        self,
        message: str,
        article_id: Optional[str] = None,
        node_id: Optional[str] = None,
    ):
        self.article_id = article_id
        self.node_id = node_id
        super().__init__(message)


class ContinuationType(Enum):
    """Types of article continuation."""

    NONE = "none"  # Complete article on single page
    FORWARD = "forward"  # Continues to next page
    BACKWARD = "backward"  # Continued from previous page
    JUMP = "jump"  # Jumps to specific page
    RETURN = "return"  # Returns from jump


@dataclass
class ContinuationMarker:
    """Marker indicating article continuation."""

    # Core properties
    marker_type: ContinuationType
    source_page: int
    target_page: Optional[int] = None

    # Extracted text
    marker_text: str = ""
    confidence: float = 0.0

    # Position information
    position: str = "unknown"  # top, bottom, inline

    # Pattern information
    pattern_used: str = ""
    extraction_method: str = ""

    def __post_init__(self):
        """Validate continuation marker."""
        if self.marker_type == ContinuationType.JUMP and self.target_page is None:
            raise ValueError("Jump continuation must specify target page")


@dataclass
class ArticleBoundary:
    """Boundary definition for a reconstructed article."""

    # Article identification
    article_id: str
    title: str

    # Page range information
    start_page: int
    end_page: int
    total_pages: int

    # Content boundaries
    start_node_id: str
    end_node_id: str
    component_count: int

    # Continuation information
    continuation_markers: List[ContinuationMarker] = field(default_factory=list)
    is_split_article: bool = False
    split_pages: List[int] = field(default_factory=list)

    # Quality metrics
    completeness_score: float = 1.0
    confidence_score: float = 1.0
    reconstruction_quality: str = "high"

    # Metadata
    byline: Optional[str] = None
    word_count: int = 0
    estimated_reading_time: float = 0.0
    extraction_timestamp: datetime = field(default_factory=datetime.now)

    @property
    def page_range(self) -> str:
        """Get formatted page range string."""
        if self.start_page == self.end_page:
            return str(self.start_page)
        return f"{self.start_page}-{self.end_page}"

    @property
    def is_complete(self) -> bool:
        """Check if article appears complete."""
        return self.completeness_score >= 0.8 and not self.has_dangling_continuations

    @property
    def has_dangling_continuations(self) -> bool:
        """Check for unresolved continuation markers."""
        for marker in self.continuation_markers:
            if marker.marker_type in [ContinuationType.FORWARD, ContinuationType.JUMP]:
                # Check if continuation was resolved
                if marker.target_page and marker.target_page not in self.split_pages:
                    return True
        return False


@dataclass
class ReconstructionConfig:
    """Configuration for article reconstruction."""

    # Traversal parameters
    max_traversal_depth: int = 50
    min_article_components: int = 2
    max_article_components: int = 200

    # Confidence thresholds
    min_connection_confidence: float = 0.3
    min_title_confidence: float = 0.7
    min_continuation_confidence: float = 0.6

    # Continuation detection
    continuation_patterns: List[str] = field(
        default_factory=lambda: [
            r"continued?\s+on\s+page\s+(\d+)",
            r"see\s+page\s+(\d+)",
            r"turn\s+to\s+page\s+(\d+)",
            r"continued?\s+(?:on\s+)?p\.?\s*(\d+)",
            r"from\s+page\s+(\d+)",
            r"continued?\s+from\s+p\.?\s*(\d+)",
            r"\(continued?\)",
            r"\(cont\.?\)",
            r"more\s+on\s+page\s+(\d+)",
        ]
    )

    # Ambiguity resolution
    use_confidence_scoring: bool = True
    prefer_sequential_pages: bool = True
    spatial_proximity_weight: float = 0.3
    content_similarity_weight: float = 0.4
    continuation_marker_weight: float = 0.3

    # Quality filters
    filter_short_articles: bool = True
    min_article_words: int = 50
    filter_advertisements: bool = True
    require_title: bool = True

    # Output options
    include_metadata: bool = True
    calculate_reading_time: bool = True
    words_per_minute: int = 250  # For reading time estimation

    @classmethod
    def create_conservative(cls) -> "ReconstructionConfig":
        """Create conservative configuration for high precision."""
        return cls(
            min_connection_confidence=0.6,
            min_title_confidence=0.8,
            min_continuation_confidence=0.7,
            min_article_words=100,
            require_title=True,
        )

    @classmethod
    def create_aggressive(cls) -> "ReconstructionConfig":
        """Create aggressive configuration for high recall."""
        return cls(
            min_connection_confidence=0.2,
            min_title_confidence=0.5,
            min_continuation_confidence=0.4,
            min_article_words=25,
            require_title=False,
        )


@dataclass
class ConnectionScore:
    """Score for connection between graph nodes."""

    # Basic scores
    confidence_score: float
    spatial_score: float
    semantic_score: float
    continuation_score: float

    # Combined score
    total_score: float

    # Metadata
    connection_type: EdgeType
    reasoning: List[str] = field(default_factory=list)

    @classmethod
    def calculate(
        cls,
        confidence: float,
        spatial_proximity: float,
        semantic_similarity: float,
        has_continuation: bool,
        connection_type: EdgeType,
        config: ReconstructionConfig,
    ) -> "ConnectionScore":
        """Calculate connection score using weighted components."""

        # Normalize spatial proximity (0-1 range)
        spatial_score = max(0, min(1, 1 - spatial_proximity / 1000))

        # Continuation bonus
        continuation_score = 1.0 if has_continuation else 0.0

        # Weight the components
        total_score = (
            confidence
            * (
                1
                - config.spatial_proximity_weight
                - config.content_similarity_weight
                - config.continuation_marker_weight
            )
            + spatial_score * config.spatial_proximity_weight
            + semantic_similarity * config.content_similarity_weight
            + continuation_score * config.continuation_marker_weight
        )

        reasoning = []
        if confidence > 0.8:
            reasoning.append("high_confidence_edge")
        if spatial_score > 0.7:
            reasoning.append("spatial_proximity")
        if semantic_similarity > 0.6:
            reasoning.append("semantic_similarity")
        if has_continuation:
            reasoning.append("continuation_marker")

        return cls(
            confidence_score=confidence,
            spatial_score=spatial_score,
            semantic_score=semantic_similarity,
            continuation_score=continuation_score,
            total_score=total_score,
            connection_type=connection_type,
            reasoning=reasoning,
        )


@dataclass
class TraversalPath:
    """Path through the semantic graph during article reconstruction."""

    # Path information
    path_id: str
    node_ids: List[str] = field(default_factory=list)
    edge_types: List[EdgeType] = field(default_factory=list)

    # Quality metrics
    total_confidence: float = 0.0
    path_length: int = 0

    # Content information
    start_page: int = 0
    end_page: int = 0
    component_types: List[BlockType] = field(default_factory=list)

    # Traversal metadata
    traversal_method: str = ""
    branch_points: List[int] = field(default_factory=list)
    ambiguous_connections: List[int] = field(default_factory=list)

    def add_node(
        self,
        node_id: str,
        edge_type: Optional[EdgeType] = None,
        confidence: float = 1.0,
    ):
        """Add a node to the traversal path."""
        self.node_ids.append(node_id)
        if edge_type:
            self.edge_types.append(edge_type)

        self.total_confidence += confidence
        self.path_length = len(self.node_ids)

    @property
    def average_confidence(self) -> float:
        """Get average confidence along the path."""
        return self.total_confidence / max(1, self.path_length)

    @property
    def spans_multiple_pages(self) -> bool:
        """Check if path spans multiple pages."""
        return self.end_page > self.start_page

    def get_path_summary(self) -> Dict[str, Any]:
        """Get summary of the traversal path."""
        return {
            "path_id": self.path_id,
            "length": self.path_length,
            "pages": f"{self.start_page}-{self.end_page}"
            if self.spans_multiple_pages
            else str(self.start_page),
            "avg_confidence": self.average_confidence,
            "components": len(set(self.component_types)),
            "method": self.traversal_method,
            "has_ambiguity": len(self.ambiguous_connections) > 0,
        }
