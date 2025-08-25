"""
Ambiguity resolution for image-caption matching.
"""

import itertools
from typing import Dict, List, Optional, Set, Tuple
import structlog

from ..graph.types import SemanticGraph
from .types import (
    SpatialMatch, MatchConfidence, ImageCaptionPair, SpatialConfig,
    MatchingResult, MatchingError
)


logger = structlog.get_logger(__name__)


class AmbiguityResolver:
    """
    Resolves ambiguous image-caption matches using advanced algorithms.
    
    Handles cases where multiple images could match the same caption,
    or multiple captions could match the same image.
    """
    
    def __init__(self, config: Optional[SpatialConfig] = None):
        """
        Initialize ambiguity resolver.
        
        Args:
            config: Spatial matching configuration
        """
        self.config = config or SpatialConfig()
        self.logger = logger.bind(component="AmbiguityResolver")
        
        # Resolution statistics
        self.stats = {
            "ambiguous_cases_resolved": 0,
            "perfect_assignments": 0,
            "compromise_assignments": 0,
            "unresolvable_cases": 0
        }
    
    def resolve_ambiguities(
        self, 
        matches: List[SpatialMatch],
        graph: SemanticGraph
    ) -> MatchingResult:
        """
        Resolve ambiguities and create final image-caption pairs.
        
        Args:
            matches: List of all potential matches
            graph: Semantic graph for additional context
            
        Returns:
            Final matching result with resolved ambiguities
        """
        try:
            self.logger.info("Starting ambiguity resolution", total_matches=len(matches))
            
            # Group matches by potential conflicts
            conflict_groups = self._identify_conflicts(matches)
            
            # Resolve each conflict group
            resolved_pairs = []
            ambiguous_cases = []
            
            for group in conflict_groups:
                if self._is_ambiguous_group(group):
                    self.logger.debug(
                        "Resolving ambiguous group",
                        group_size=len(group),
                        images=list(set(m.image_node_id for m in group)),
                        captions=list(set(m.caption_node_id for m in group))
                    )
                    
                    resolution_result = self._resolve_conflict_group(group, graph)
                    resolved_pairs.extend(resolution_result['resolved'])
                    ambiguous_cases.extend(resolution_result['ambiguous'])
                    
                    self.stats["ambiguous_cases_resolved"] += 1
                else:
                    # No conflicts, use all matches
                    pairs = [self._create_image_caption_pair(match, graph) for match in group]
                    resolved_pairs.extend(pairs)
            
            # Update uniqueness confidence scores
            self._update_uniqueness_confidence(resolved_pairs, ambiguous_cases)
            
            # Create final result
            result = self._create_final_result(resolved_pairs, ambiguous_cases, matches, graph)
            
            self.logger.info(
                "Ambiguity resolution completed",
                resolved_pairs=len(resolved_pairs),
                ambiguous_cases=len(ambiguous_cases),
                resolution_quality=result.matching_quality
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Error in ambiguity resolution", error=str(e), exc_info=True)
            raise MatchingError(f"Ambiguity resolution failed: {e}")
    
    def _identify_conflicts(self, matches: List[SpatialMatch]) -> List[List[SpatialMatch]]:
        """Identify groups of matches that conflict with each other."""
        
        # Create graph of conflicts
        image_to_matches = {}
        caption_to_matches = {}
        
        for match in matches:
            # Group by image
            if match.image_node_id not in image_to_matches:
                image_to_matches[match.image_node_id] = []
            image_to_matches[match.image_node_id].append(match)
            
            # Group by caption
            if match.caption_node_id not in caption_to_matches:
                caption_to_matches[match.caption_node_id] = []
            caption_to_matches[match.caption_node_id].append(match)
        
        # Find connected components of conflicts
        conflict_groups = []
        processed_matches = set()
        
        for match in matches:
            if id(match) in processed_matches:
                continue
            
            # Find all matches that conflict with this one
            conflict_group = set()
            to_process = [match]
            
            while to_process:
                current_match = to_process.pop(0)
                if id(current_match) in processed_matches:
                    continue
                
                conflict_group.add(current_match)
                processed_matches.add(id(current_match))
                
                # Add all matches for the same image
                for related_match in image_to_matches.get(current_match.image_node_id, []):
                    if id(related_match) not in processed_matches:
                        to_process.append(related_match)
                
                # Add all matches for the same caption
                for related_match in caption_to_matches.get(current_match.caption_node_id, []):
                    if id(related_match) not in processed_matches:
                        to_process.append(related_match)
            
            if conflict_group:
                conflict_groups.append(list(conflict_group))
        
        return conflict_groups
    
    def _is_ambiguous_group(self, group: List[SpatialMatch]) -> bool:
        """Check if a group of matches represents an ambiguous case."""
        if len(group) <= 1:
            return False
        
        # Check if there are multiple images or multiple captions
        unique_images = set(m.image_node_id for m in group)
        unique_captions = set(m.caption_node_id for m in group)
        
        # Ambiguous if multiple images compete for same caption 
        # OR multiple captions compete for same image
        return len(unique_images) > 1 or len(unique_captions) > 1
    
    def _resolve_conflict_group(
        self, 
        group: List[SpatialMatch], 
        graph: SemanticGraph
    ) -> Dict[str, List]:
        """Resolve conflicts within a group of matches."""
        
        # Try different resolution strategies
        strategies = [
            self._resolve_by_confidence,
            self._resolve_by_spatial_preference,
            self._resolve_by_semantic_strength,
            self._resolve_by_layout_patterns
        ]
        
        for strategy in strategies:
            try:
                result = strategy(group, graph)
                if result['resolved']:  # Strategy found a solution
                    self.logger.debug(
                        "Conflict resolved by strategy",
                        strategy=strategy.__name__,
                        resolved_count=len(result['resolved'])
                    )
                    return result
            except Exception as e:
                self.logger.warning(
                    "Strategy failed",
                    strategy=strategy.__name__,
                    error=str(e)
                )
                continue
        
        # If no strategy worked, mark as unresolvable
        self.stats["unresolvable_cases"] += 1
        return {
            'resolved': [],
            'ambiguous': [group]
        }
    
    def _resolve_by_confidence(
        self, 
        group: List[SpatialMatch], 
        graph: SemanticGraph
    ) -> Dict[str, List]:
        """Resolve conflicts by selecting highest confidence matches."""
        
        # Sort matches by overall confidence
        sorted_matches = sorted(
            group, 
            key=lambda m: m.match_confidence.overall_confidence,
            reverse=True
        )
        
        resolved = []
        used_images = set()
        used_captions = set()
        ambiguous = []
        
        for match in sorted_matches:
            # Check if this match conflicts with already resolved ones
            if (match.image_node_id in used_images or 
                match.caption_node_id in used_captions):
                continue
            
            # Check if confidence is above threshold
            if match.match_confidence.overall_confidence >= self.config.min_overall_confidence:
                pair = self._create_image_caption_pair(match, graph)
                resolved.append(pair)
                used_images.add(match.image_node_id)
                used_captions.add(match.caption_node_id)
                
                self.stats["perfect_assignments"] += 1
            else:
                # Low confidence, mark as ambiguous
                ambiguous.append([match])
        
        return {
            'resolved': resolved,
            'ambiguous': ambiguous
        }
    
    def _resolve_by_spatial_preference(
        self, 
        group: List[SpatialMatch], 
        graph: SemanticGraph
    ) -> Dict[str, List]:
        """Resolve conflicts by preferring better spatial arrangements."""
        
        # Score matches by spatial quality
        def spatial_quality_score(match: SpatialMatch) -> float:
            proximity_score = match.proximity_score.overall_score
            layout_bonus = 0.0
            
            # Bonus for preferred caption positions
            if self.config.prefer_below_captions:
                if match.caption_bbox.center_y > match.image_bbox.center_y:
                    layout_bonus += 0.1
            
            if self.config.prefer_aligned_captions:
                # Simple horizontal alignment check
                horizontal_overlap = (
                    min(match.image_bbox.x1, match.caption_bbox.x1) - 
                    max(match.image_bbox.x0, match.caption_bbox.x0)
                )
                if horizontal_overlap > 0:
                    layout_bonus += 0.1
            
            return proximity_score + layout_bonus
        
        # Sort by spatial quality
        sorted_matches = sorted(
            group,
            key=spatial_quality_score,
            reverse=True
        )
        
        # Use greedy assignment
        resolved = []
        used_images = set()
        used_captions = set()
        
        for match in sorted_matches:
            if (match.image_node_id not in used_images and 
                match.caption_node_id not in used_captions):
                
                pair = self._create_image_caption_pair(match, graph)
                resolved.append(pair)
                used_images.add(match.image_node_id)
                used_captions.add(match.caption_node_id)
        
        # Remaining matches are ambiguous
        remaining_matches = [
            m for m in group 
            if (m.image_node_id not in used_images or 
                m.caption_node_id not in used_captions)
        ]
        
        return {
            'resolved': resolved,
            'ambiguous': [remaining_matches] if remaining_matches else []
        }
    
    def _resolve_by_semantic_strength(
        self, 
        group: List[SpatialMatch], 
        graph: SemanticGraph
    ) -> Dict[str, List]:
        """Resolve conflicts by semantic matching strength."""
        
        def semantic_strength(match: SpatialMatch) -> float:
            keyword_score = len(match.keywords_found) * 0.1
            semantic_confidence = match.match_confidence.semantic_confidence
            
            # Bonus for strong caption indicators
            strong_indicators = [
                'photo_credit', 'illustration_credit', 'caption_indicator'
            ]
            strong_keyword_bonus = sum(
                0.2 for keyword in match.keywords_found 
                if keyword in strong_indicators
            )
            
            return semantic_confidence + keyword_score + strong_keyword_bonus
        
        # Sort by semantic strength
        sorted_matches = sorted(
            group,
            key=semantic_strength,
            reverse=True
        )
        
        # Use greedy assignment based on semantic strength
        resolved = []
        used_images = set()
        used_captions = set()
        
        for match in sorted_matches:
            if (match.image_node_id not in used_images and 
                match.caption_node_id not in used_captions and
                semantic_strength(match) >= 0.5):  # Minimum semantic threshold
                
                pair = self._create_image_caption_pair(match, graph)
                resolved.append(pair)
                used_images.add(match.image_node_id)
                used_captions.add(match.caption_node_id)
        
        return {
            'resolved': resolved,
            'ambiguous': []
        }
    
    def _resolve_by_layout_patterns(
        self, 
        group: List[SpatialMatch], 
        graph: SemanticGraph
    ) -> Dict[str, List]:
        """Resolve conflicts by analyzing common layout patterns."""
        
        # Analyze group for common layout patterns
        images = list(set(m.image_node_id for m in group))
        captions = list(set(m.caption_node_id for m in group))
        
        # Pattern: Multiple images sharing one caption
        if len(images) > 1 and len(captions) == 1:
            return self._handle_shared_caption_pattern(group, graph)
        
        # Pattern: One image with multiple potential captions
        if len(images) == 1 and len(captions) > 1:
            return self._handle_multiple_captions_pattern(group, graph)
        
        # Pattern: Grid or gallery layout
        if len(images) > 2 and len(captions) > 1:
            return self._handle_gallery_pattern(group, graph)
        
        # Default: fall back to confidence-based resolution
        return self._resolve_by_confidence(group, graph)
    
    def _handle_shared_caption_pattern(
        self, 
        group: List[SpatialMatch], 
        graph: SemanticGraph
    ) -> Dict[str, List]:
        """Handle case where multiple images share one caption."""
        
        # Find the best image-caption match
        best_match = max(
            group, 
            key=lambda m: m.match_confidence.overall_confidence
        )
        
        # Create one pair for the shared caption
        pair = self._create_image_caption_pair(best_match, graph)
        
        # Add references to other images in the metadata
        other_images = [
            m.image_node_id for m in group 
            if m.image_node_id != best_match.image_node_id
        ]
        
        # Store additional context in the pair
        if hasattr(pair.spatial_match, 'additional_images'):
            pair.spatial_match.additional_images = other_images
        
        return {
            'resolved': [pair],
            'ambiguous': []
        }
    
    def _handle_multiple_captions_pattern(
        self, 
        group: List[SpatialMatch], 
        graph: SemanticGraph
    ) -> Dict[str, List]:
        """Handle case where one image has multiple potential captions."""
        
        # Select the best caption based on multiple factors
        def caption_score(match: SpatialMatch) -> float:
            base_score = match.match_confidence.overall_confidence
            
            # Prefer shorter captions (more likely to be actual captions)
            caption_node = graph.get_node(match.caption_node_id)
            if caption_node and caption_node.content:
                word_count = len(caption_node.content.split())
                length_factor = max(0.5, 1.0 - (word_count - 10) * 0.02)  # Prefer 10 words or fewer
                base_score *= length_factor
            
            # Prefer captions with strong indicators
            if match.keywords_found:
                keyword_bonus = min(0.2, len(match.keywords_found) * 0.05)
                base_score += keyword_bonus
            
            return base_score
        
        best_match = max(group, key=caption_score)
        pair = self._create_image_caption_pair(best_match, graph)
        
        return {
            'resolved': [pair],
            'ambiguous': []
        }
    
    def _handle_gallery_pattern(
        self, 
        group: List[SpatialMatch], 
        graph: SemanticGraph
    ) -> Dict[str, List]:
        """Handle gallery or grid layout with multiple images and captions."""
        
        # Use Hungarian algorithm approach for optimal assignment
        # For simplicity, using a greedy approach here
        
        # Sort all matches by confidence
        sorted_matches = sorted(
            group,
            key=lambda m: m.match_confidence.overall_confidence,
            reverse=True
        )
        
        resolved = []
        used_images = set()
        used_captions = set()
        
        for match in sorted_matches:
            if (match.image_node_id not in used_images and 
                match.caption_node_id not in used_captions):
                
                pair = self._create_image_caption_pair(match, graph)
                resolved.append(pair)
                used_images.add(match.image_node_id)
                used_captions.add(match.caption_node_id)
        
        return {
            'resolved': resolved,
            'ambiguous': []
        }
    
    def _create_image_caption_pair(
        self, 
        match: SpatialMatch, 
        graph: SemanticGraph
    ) -> ImageCaptionPair:
        """Create final image-caption pair from a spatial match."""
        
        # Get caption text
        caption_node = graph.get_node(match.caption_node_id)
        caption_text = caption_node.content if caption_node else ""
        
        # Create pair
        pair = ImageCaptionPair(
            image_node_id=match.image_node_id,
            caption_node_id=match.caption_node_id,
            caption_text=caption_text,
            spatial_match=match,
            pairing_confidence=match.match_confidence.overall_confidence,
            quality_score=self._calculate_pair_quality_score(match),
            pairing_method="ambiguity_resolved"
        )
        
        return pair
    
    def _calculate_pair_quality_score(self, match: SpatialMatch) -> float:
        """Calculate overall quality score for an image-caption pair."""
        
        # Combine multiple quality factors
        spatial_quality = match.proximity_score.overall_score
        confidence_quality = match.match_confidence.overall_confidence
        keyword_quality = min(1.0, len(match.keywords_found) * 0.2)
        
        # Weighted combination
        quality_score = (
            spatial_quality * 0.4 +
            confidence_quality * 0.4 +
            keyword_quality * 0.2
        )
        
        return quality_score
    
    def _update_uniqueness_confidence(
        self, 
        resolved_pairs: List[ImageCaptionPair],
        ambiguous_cases: List[List[SpatialMatch]]
    ) -> None:
        """Update uniqueness confidence scores based on resolution results."""
        
        # Count competing matches for each image and caption
        image_competition = {}
        caption_competition = {}
        
        for case in ambiguous_cases:
            for match in case:
                image_id = match.image_node_id
                caption_id = match.caption_node_id
                
                image_competition[image_id] = image_competition.get(image_id, 0) + 1
                caption_competition[caption_id] = caption_competition.get(caption_id, 0) + 1
        
        # Update uniqueness confidence for resolved pairs
        for pair in resolved_pairs:
            match = pair.spatial_match
            
            image_competitors = image_competition.get(match.image_node_id, 0)
            caption_competitors = caption_competition.get(match.caption_node_id, 0)
            
            total_competitors = image_competitors + caption_competitors
            
            # Higher uniqueness if fewer competitors
            if total_competitors == 0:
                uniqueness = 1.0
            elif total_competitors <= 2:
                uniqueness = 0.8
            elif total_competitors <= 5:
                uniqueness = 0.6
            else:
                uniqueness = 0.4
            
            match.match_confidence.uniqueness_confidence = uniqueness
            match.match_confidence.competing_matches = total_competitors
            
            # Recalculate overall confidence
            match.match_confidence.calculate_overall_confidence()
            pair.pairing_confidence = match.match_confidence.overall_confidence
    
    def _create_final_result(
        self, 
        resolved_pairs: List[ImageCaptionPair],
        ambiguous_cases: List[List[SpatialMatch]],
        original_matches: List[SpatialMatch],
        graph: SemanticGraph
    ) -> MatchingResult:
        """Create final matching result."""
        
        result = MatchingResult(
            successful_pairs=resolved_pairs,
            ambiguous_matches=ambiguous_cases
        )
        
        # Calculate unmatched items
        matched_images = {pair.image_node_id for pair in resolved_pairs}
        matched_captions = {pair.caption_node_id for pair in resolved_pairs}
        
        all_images = set(m.image_node_id for m in original_matches)
        all_captions = set(m.caption_node_id for m in original_matches)
        
        result.unmatched_images = list(all_images - matched_images)
        result.unmatched_captions = list(all_captions - matched_captions)
        
        result.total_images = len(all_images)
        result.total_captions = len(all_captions)
        
        # Assess matching quality
        if result.total_images > 0:
            match_rate = len(resolved_pairs) / result.total_images
            avg_confidence = (
                sum(pair.pairing_confidence for pair in resolved_pairs) / 
                len(resolved_pairs) if resolved_pairs else 0.0
            )
            
            if match_rate >= 0.99 and avg_confidence >= 0.95:
                result.matching_quality = "high"
            elif match_rate >= 0.9 and avg_confidence >= 0.8:
                result.matching_quality = "medium"
            else:
                result.matching_quality = "low"
        else:
            result.matching_quality = "low"
        
        return result
    
    def get_resolution_statistics(self) -> Dict[str, any]:
        """Get ambiguity resolution statistics."""
        return {
            "ambiguous_cases_resolved": self.stats["ambiguous_cases_resolved"],
            "perfect_assignments": self.stats["perfect_assignments"],
            "compromise_assignments": self.stats["compromise_assignments"],
            "unresolvable_cases": self.stats["unresolvable_cases"],
            "resolution_success_rate": (
                self.stats["perfect_assignments"] + self.stats["compromise_assignments"]
            ) / max(1, self.stats["ambiguous_cases_resolved"])
        }