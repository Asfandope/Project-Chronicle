"""
Type definitions for caption matching module.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import re


class MatchingError(Exception):
    """Base exception for caption matching errors."""
    
    def __init__(self, message: str, image_node_id: Optional[str] = None, caption_node_id: Optional[str] = None):
        self.image_node_id = image_node_id
        self.caption_node_id = caption_node_id
        super().__init__(message)


class MatchType(Enum):
    """Types of image-caption matches."""
    DIRECT_BELOW = "direct_below"        # Caption directly below image
    DIRECT_ABOVE = "direct_above"        # Caption directly above image
    SIDE_BY_SIDE = "side_by_side"       # Caption beside image
    GROUPED = "grouped"                  # Multiple images with shared caption
    EMBEDDED = "embedded"                # Caption embedded within image area
    DISTANT = "distant"                  # Caption far from image but matched by keywords


class FilenameFormat(Enum):
    """Filename generation formats."""
    SEQUENTIAL = "sequential"            # img_001.jpg, img_002.jpg
    DESCRIPTIVE = "descriptive"         # photo_by_john_smith.jpg
    HYBRID = "hybrid"                   # img_001_photo_by_john_smith.jpg
    ARTICLE_BASED = "article_based"     # article_title_img_001.jpg


@dataclass
class BoundingBox:
    """Bounding box coordinates."""
    x0: float
    y0: float
    x1: float
    y1: float
    
    @property
    def width(self) -> float:
        return self.x1 - self.x0
    
    @property  
    def height(self) -> float:
        return self.y1 - self.y0
    
    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2
    
    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def overlaps(self, other: "BoundingBox") -> bool:
        """Check if this box overlaps with another."""
        return not (self.x1 <= other.x0 or other.x1 <= self.x0 or
                   self.y1 <= other.y0 or other.y1 <= self.y0)
    
    def distance_to(self, other: "BoundingBox") -> float:
        """Calculate distance between centers of two boxes."""
        dx = self.center_x - other.center_x
        dy = self.center_y - other.center_y
        return (dx * dx + dy * dy) ** 0.5


@dataclass
class ProximityScore:
    """Score for spatial proximity between image and caption."""
    
    # Distance metrics
    euclidean_distance: float = 0.0
    vertical_distance: float = 0.0
    horizontal_distance: float = 0.0
    
    # Spatial relationship scores
    alignment_score: float = 0.0      # How well aligned vertically/horizontally
    relative_position_score: float = 0.0  # Score for expected position (below/above)
    containment_score: float = 0.0     # If caption is within image bounds
    
    # Layout scores
    reading_order_score: float = 0.0   # Follows natural reading order
    column_awareness_score: float = 0.0  # Respects column layout
    
    # Combined score
    overall_score: float = 0.0
    
    def calculate_overall_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate weighted overall proximity score."""
        default_weights = {
            'distance': 0.3,
            'alignment': 0.2,
            'position': 0.2,
            'reading_order': 0.15,
            'column_awareness': 0.1,
            'containment': 0.05
        }
        
        weights = weights or default_weights
        
        # Normalize distance (closer = higher score)
        distance_score = max(0.0, 1.0 - (self.euclidean_distance / 1000.0))
        
        self.overall_score = (
            distance_score * weights['distance'] +
            self.alignment_score * weights['alignment'] +
            self.relative_position_score * weights['position'] +
            self.reading_order_score * weights['reading_order'] +
            self.column_awareness_score * weights['column_awareness'] +
            self.containment_score * weights['containment']
        )
        
        return self.overall_score


@dataclass
class MatchConfidence:
    """Confidence assessment for an image-caption match."""
    
    # Component confidences
    spatial_confidence: float = 0.0     # How good is spatial match
    semantic_confidence: float = 0.0    # How well keywords match
    layout_confidence: float = 0.0      # How well it fits layout patterns
    uniqueness_confidence: float = 0.0  # How unique/unambiguous the match is
    
    # Meta information
    match_type: MatchType = MatchType.DISTANT
    ambiguity_level: float = 0.0        # 0 = unambiguous, 1 = highly ambiguous
    competing_matches: int = 0          # Number of other possible matches
    
    # Overall confidence
    overall_confidence: float = 0.0
    
    def calculate_overall_confidence(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate weighted overall confidence."""
        default_weights = {
            'spatial': 0.4,
            'semantic': 0.25,
            'layout': 0.2,
            'uniqueness': 0.15
        }
        
        weights = weights or default_weights
        
        self.overall_confidence = (
            self.spatial_confidence * weights['spatial'] +
            self.semantic_confidence * weights['semantic'] +
            self.layout_confidence * weights['layout'] +
            self.uniqueness_confidence * weights['uniqueness']
        )
        
        # Apply ambiguity penalty
        ambiguity_penalty = 1.0 - (self.ambiguity_level * 0.3)
        self.overall_confidence *= ambiguity_penalty
        
        return self.overall_confidence
    
    @property
    def is_high_confidence(self) -> bool:
        """Check if this is a high-confidence match."""
        return (self.overall_confidence >= 0.9 and 
                self.ambiguity_level <= 0.3 and
                self.competing_matches <= 1)


@dataclass
class SpatialMatch:
    """A potential spatial match between image and caption."""
    
    # Node identifiers
    image_node_id: str
    caption_node_id: str
    
    # Spatial information
    image_bbox: BoundingBox
    caption_bbox: BoundingBox
    proximity_score: ProximityScore
    
    # Match assessment
    match_confidence: MatchConfidence
    keywords_found: List[str] = field(default_factory=list)
    
    # Processing metadata
    detection_method: str = ""
    processing_timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def is_valid_match(self) -> bool:
        """Check if this is a valid match based on thresholds."""
        return (self.match_confidence.overall_confidence >= 0.8 and
                self.proximity_score.overall_score >= 0.7)


@dataclass
class ImageCaptionPair:
    """Final paired image-caption result."""
    
    # Core pairing information
    image_node_id: str
    caption_node_id: str
    caption_text: str
    
    # Spatial match details
    spatial_match: SpatialMatch
    
    # Generated metadata
    filename: str = ""
    alt_text: str = ""
    
    # Quality metrics
    pairing_confidence: float = 0.0
    quality_score: float = 0.0
    
    # Processing metadata
    pairing_method: str = ""
    pairing_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "image_node_id": self.image_node_id,
            "caption_node_id": self.caption_node_id,
            "caption_text": self.caption_text,
            "filename": self.filename,
            "alt_text": self.alt_text,
            "pairing_confidence": self.pairing_confidence,
            "quality_score": self.quality_score,
            "pairing_method": self.pairing_method,
            "match_type": self.spatial_match.match_confidence.match_type.value,
            "spatial_score": self.spatial_match.proximity_score.overall_score,
            "pairing_timestamp": self.pairing_timestamp.isoformat()
        }


@dataclass
class SpatialConfig:
    """Configuration for spatial caption matching."""
    
    # Distance thresholds
    max_search_distance: float = 500.0    # Max pixels to search for captions
    preferred_caption_distance: float = 100.0  # Preferred distance for captions
    
    # Confidence thresholds  
    min_spatial_confidence: float = 0.8
    min_semantic_confidence: float = 0.7
    min_overall_confidence: float = 0.85
    
    # Matching preferences
    prefer_below_captions: bool = True     # Prefer captions below images
    prefer_aligned_captions: bool = True   # Prefer aligned captions
    enable_keyword_matching: bool = True   # Use semantic keyword matching
    
    # Layout analysis
    respect_column_layout: bool = True     # Consider column boundaries
    reading_order_weight: float = 0.15     # Weight for reading order
    
    # Ambiguity resolution
    max_ambiguous_matches: int = 3         # Max matches to consider before disambiguation
    ambiguity_threshold: float = 0.1       # Threshold for considering matches ambiguous
    
    # Filename generation
    filename_format: FilenameFormat = FilenameFormat.HYBRID
    filename_max_length: int = 100
    filename_sanitize: bool = True
    
    @classmethod
    def create_high_precision(cls) -> "SpatialConfig":
        """Create configuration optimized for high precision."""
        return cls(
            min_spatial_confidence=0.9,
            min_semantic_confidence=0.85,
            min_overall_confidence=0.95,
            max_ambiguous_matches=2,
            ambiguity_threshold=0.05
        )
    
    @classmethod
    def create_high_recall(cls) -> "SpatialConfig":
        """Create configuration optimized for high recall."""
        return cls(
            min_spatial_confidence=0.7,
            min_semantic_confidence=0.6,
            min_overall_confidence=0.75,
            max_search_distance=800.0,
            max_ambiguous_matches=5,
            ambiguity_threshold=0.2
        )


@dataclass
class MatchingResult:
    """Complete result of caption matching process."""
    
    # Successful pairings
    successful_pairs: List[ImageCaptionPair] = field(default_factory=list)
    
    # Unmatched items
    unmatched_images: List[str] = field(default_factory=list)
    unmatched_captions: List[str] = field(default_factory=list)
    
    # Ambiguous cases
    ambiguous_matches: List[List[SpatialMatch]] = field(default_factory=list)
    
    # Processing metrics
    processing_time: float = 0.0
    total_images: int = 0
    total_captions: int = 0
    
    # Quality assessment
    matching_quality: str = "unknown"  # high, medium, low
    
    @property
    def matching_rate(self) -> float:
        """Calculate the matching rate."""
        if self.total_images == 0:
            return 0.0
        return len(self.successful_pairs) / self.total_images
    
    @property
    def average_confidence(self) -> float:
        """Calculate average pairing confidence."""
        if not self.successful_pairs:
            return 0.0
        return sum(pair.pairing_confidence for pair in self.successful_pairs) / len(self.successful_pairs)
    
    @property
    def meets_target(self) -> bool:
        """Check if result meets 99% target."""
        return self.matching_rate >= 0.99 and self.average_confidence >= 0.9


@dataclass
class MatchingMetrics:
    """Metrics for evaluating matching performance."""
    
    # Performance metrics
    total_matches_attempted: int = 0
    successful_matches: int = 0
    failed_matches: int = 0
    ambiguous_cases: int = 0
    
    # Accuracy metrics
    correct_matches: int = 0       # When ground truth is available
    incorrect_matches: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Confidence distribution
    high_confidence_matches: int = 0    # >= 0.9
    medium_confidence_matches: int = 0  # 0.8-0.9
    low_confidence_matches: int = 0     # < 0.8
    
    # Spatial analysis
    average_match_distance: float = 0.0
    match_type_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Target achievement
    target_achievement_rate: float = 0.0  # How close to 99% target
    
    def calculate_performance_scores(self) -> None:
        """Calculate derived performance metrics."""
        if self.correct_matches + self.incorrect_matches > 0:
            self.precision = self.correct_matches / (self.correct_matches + self.incorrect_matches)
        
        if self.total_matches_attempted > 0:
            self.recall = self.correct_matches / self.total_matches_attempted
        
        if self.precision + self.recall > 0:
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        
        # Calculate target achievement
        if self.total_matches_attempted > 0:
            accuracy = self.successful_matches / self.total_matches_attempted
            self.target_achievement_rate = accuracy / 0.99  # Relative to 99% target
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "matching_performance": {
                "total_attempted": self.total_matches_attempted,
                "successful": self.successful_matches,
                "failed": self.failed_matches,
                "success_rate": self.successful_matches / max(1, self.total_matches_attempted)
            },
            "accuracy_metrics": {
                "precision": self.precision,
                "recall": self.recall,
                "f1_score": self.f1_score,
                "correct_matches": self.correct_matches,
                "incorrect_matches": self.incorrect_matches
            },
            "confidence_distribution": {
                "high_confidence": self.high_confidence_matches,
                "medium_confidence": self.medium_confidence_matches,
                "low_confidence": self.low_confidence_matches
            },
            "spatial_analysis": {
                "average_distance": self.average_match_distance,
                "match_types": self.match_type_distribution
            },
            "target_metrics": {
                "target_achievement_rate": self.target_achievement_rate,
                "meets_target": self.target_achievement_rate >= 1.0
            }
        }