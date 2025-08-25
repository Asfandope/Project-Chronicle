"""
Deterministic filename generation for image-caption pairs.
"""

import re
import hashlib
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import structlog

from ..graph.types import SemanticGraph
from .types import (
    ImageCaptionPair, FilenameFormat, SpatialConfig, MatchingError
)


logger = structlog.get_logger(__name__)


@dataclass
class FilenameStrategy:
    """Strategy configuration for filename generation."""
    
    format_type: FilenameFormat = FilenameFormat.HYBRID
    max_length: int = 100
    include_page_number: bool = True
    include_sequence: bool = True
    include_descriptor: bool = True
    sanitize_names: bool = True
    preserve_case: bool = False
    
    # Descriptor options
    use_caption_keywords: bool = True
    use_contributor_names: bool = True
    use_image_type: bool = True
    
    # Sequence options
    sequence_padding: int = 3  # 001, 002, etc.
    sequence_start: int = 1
    
    # Article context
    include_article_context: bool = True
    article_name_max_words: int = 3


class FilenameGenerator:
    """
    Generates deterministic, descriptive filenames for image-caption pairs.
    
    Creates consistent filenames that are both human-readable and machine-processable.
    """
    
    def __init__(self, config: Optional[SpatialConfig] = None, strategy: Optional[FilenameStrategy] = None):
        """
        Initialize filename generator.
        
        Args:
            config: Spatial matching configuration
            strategy: Filename generation strategy
        """
        self.config = config or SpatialConfig()
        self.strategy = strategy or FilenameStrategy()
        self.logger = logger.bind(component="FilenameGenerator")
        
        # Track generated filenames to ensure uniqueness
        self._generated_filenames: Set[str] = set()
        self._filename_counters: Dict[str, int] = {}
        
        # Initialize filename patterns and sanitization rules
        self._initialize_patterns()
        
        # Statistics
        self.stats = {
            "filenames_generated": 0,
            "unique_filenames": 0,
            "collisions_resolved": 0,
            "format_distribution": {}
        }
    
    def _initialize_patterns(self) -> None:
        """Initialize patterns for filename generation."""
        
        # Words to remove from descriptive parts
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'these', 'those'
        }
        
        # Common image type indicators
        self.image_type_patterns = {
            r'\bphoto(?:graph)?\b': 'photo',
            r'\bpicture\b': 'photo', 
            r'\bimage\b': 'image',
            r'\billustration\b': 'illustration',
            r'\bdrawing\b': 'drawing',
            r'\bgraphic\b': 'graphic',
            r'\bchart\b': 'chart',
            r'\bgraph\b': 'graph',
            r'\bdiagram\b': 'diagram',
            r'\bmap\b': 'map'
        }
        
        # Compile patterns
        self.image_type_regex = {
            re.compile(pattern, re.IGNORECASE): image_type 
            for pattern, image_type in self.image_type_patterns.items()
        }
        
        # File extension patterns
        self.image_extensions = {
            'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'svg', 'tiff'
        }
    
    def generate_filenames(
        self, 
        pairs: List[ImageCaptionPair],
        graph: SemanticGraph,
        article_context: Optional[Dict[str, any]] = None
    ) -> List[ImageCaptionPair]:
        """
        Generate filenames for a list of image-caption pairs.
        
        Args:
            pairs: List of image-caption pairs
            graph: Semantic graph for context
            article_context: Optional article context information
            
        Returns:
            Updated pairs with generated filenames
        """
        try:
            self.logger.info(
                "Generating filenames",
                pair_count=len(pairs),
                format_type=self.strategy.format_type.value
            )
            
            # Clear previous generation state
            self._generated_filenames.clear()
            self._filename_counters.clear()
            
            # Sort pairs for consistent ordering
            sorted_pairs = self._sort_pairs_for_naming(pairs, graph)
            
            # Generate filename for each pair
            updated_pairs = []
            for i, pair in enumerate(sorted_pairs):
                try:
                    filename = self._generate_single_filename(
                        pair, graph, i + self.strategy.sequence_start, article_context
                    )
                    
                    # Ensure uniqueness
                    filename = self._ensure_unique_filename(filename)
                    
                    # Update pair
                    pair.filename = filename
                    updated_pairs.append(pair)
                    
                    self.stats["filenames_generated"] += 1
                    
                    self.logger.debug(
                        "Generated filename",
                        image_id=pair.image_node_id,
                        filename=filename,
                        method=self._get_generation_method(pair)
                    )
                    
                except Exception as e:
                    self.logger.warning(
                        "Failed to generate filename for pair",
                        image_id=pair.image_node_id,
                        error=str(e)
                    )
                    # Generate fallback filename
                    fallback_filename = self._generate_fallback_filename(i + self.strategy.sequence_start)
                    pair.filename = self._ensure_unique_filename(fallback_filename)
                    updated_pairs.append(pair)
            
            # Update statistics
            self.stats["unique_filenames"] = len(self._generated_filenames)
            format_type = self.strategy.format_type.value
            self.stats["format_distribution"][format_type] = len(updated_pairs)
            
            self.logger.info(
                "Filename generation completed",
                generated_count=len(updated_pairs),
                unique_count=self.stats["unique_filenames"],
                collisions=self.stats["collisions_resolved"]
            )
            
            return updated_pairs
            
        except Exception as e:
            self.logger.error("Error in filename generation", error=str(e), exc_info=True)
            raise MatchingError(f"Filename generation failed: {e}")
    
    def _sort_pairs_for_naming(
        self, 
        pairs: List[ImageCaptionPair], 
        graph: SemanticGraph
    ) -> List[ImageCaptionPair]:
        """Sort pairs for consistent filename generation order."""
        
        def sort_key(pair: ImageCaptionPair) -> Tuple:
            # Sort by spatial position (top to bottom, left to right)
            image_bbox = pair.spatial_match.image_bbox
            
            # Primary: page number (if available)
            image_node = graph.get_node(pair.image_node_id)
            page_num = 0
            if image_node and image_node.metadata:
                page_num = image_node.metadata.get('page', 0)
            
            # Secondary: vertical position
            vertical_pos = image_bbox.center_y
            
            # Tertiary: horizontal position  
            horizontal_pos = image_bbox.center_x
            
            return (page_num, vertical_pos, horizontal_pos)
        
        return sorted(pairs, key=sort_key)
    
    def _generate_single_filename(
        self, 
        pair: ImageCaptionPair,
        graph: SemanticGraph,
        sequence_number: int,
        article_context: Optional[Dict[str, any]] = None
    ) -> str:
        """Generate filename for a single image-caption pair."""
        
        format_type = self.strategy.format_type
        
        if format_type == FilenameFormat.SEQUENTIAL:
            return self._generate_sequential_filename(sequence_number)
        elif format_type == FilenameFormat.DESCRIPTIVE:
            return self._generate_descriptive_filename(pair, graph, article_context)
        elif format_type == FilenameFormat.HYBRID:
            return self._generate_hybrid_filename(pair, graph, sequence_number, article_context)
        elif format_type == FilenameFormat.ARTICLE_BASED:
            return self._generate_article_based_filename(pair, graph, sequence_number, article_context)
        else:
            return self._generate_sequential_filename(sequence_number)
    
    def _generate_sequential_filename(self, sequence_number: int) -> str:
        """Generate simple sequential filename."""
        
        sequence_str = str(sequence_number).zfill(self.strategy.sequence_padding)
        base_filename = f"img_{sequence_str}"
        
        # Add page number if available and requested
        if self.strategy.include_page_number:
            # Page number would be extracted from metadata
            pass
        
        return self._add_extension(base_filename)
    
    def _generate_descriptive_filename(
        self, 
        pair: ImageCaptionPair,
        graph: SemanticGraph,
        article_context: Optional[Dict[str, any]] = None
    ) -> str:
        """Generate descriptive filename based on caption content."""
        
        components = []
        
        # Extract image type
        if self.strategy.use_image_type:
            image_type = self._extract_image_type(pair.caption_text)
            if image_type:
                components.append(image_type)
        
        # Extract contributor names
        if self.strategy.use_contributor_names:
            contributor_names = self._extract_contributor_names(pair.caption_text)
            if contributor_names:
                components.extend(contributor_names[:2])  # Max 2 names
        
        # Extract descriptive keywords
        if self.strategy.use_caption_keywords:
            keywords = self._extract_descriptive_keywords(pair.caption_text)
            if keywords:
                components.extend(keywords[:3])  # Max 3 keywords
        
        # Join components
        if components:
            base_filename = "_".join(components)
        else:
            base_filename = "image"
        
        # Sanitize
        base_filename = self._sanitize_filename(base_filename)
        
        return self._add_extension(base_filename)
    
    def _generate_hybrid_filename(
        self, 
        pair: ImageCaptionPair,
        graph: SemanticGraph,
        sequence_number: int,
        article_context: Optional[Dict[str, any]] = None
    ) -> str:
        """Generate hybrid filename with sequence and description."""
        
        # Start with sequence
        sequence_str = str(sequence_number).zfill(self.strategy.sequence_padding)
        components = [f"img_{sequence_str}"]
        
        # Add descriptive elements
        if self.strategy.use_image_type:
            image_type = self._extract_image_type(pair.caption_text)
            if image_type and image_type != "image":  # Don't repeat "image"
                components.append(image_type)
        
        if self.strategy.use_contributor_names:
            contributor_names = self._extract_contributor_names(pair.caption_text)
            if contributor_names:
                # Use just last names for brevity
                last_names = [name.split()[-1] for name in contributor_names[:2]]
                components.extend(last_names)
        
        # Join and sanitize
        base_filename = "_".join(components)
        base_filename = self._sanitize_filename(base_filename)
        
        return self._add_extension(base_filename)
    
    def _generate_article_based_filename(
        self, 
        pair: ImageCaptionPair,
        graph: SemanticGraph,
        sequence_number: int,
        article_context: Optional[Dict[str, any]] = None
    ) -> str:
        """Generate filename based on article context."""
        
        components = []
        
        # Add article context if available
        if article_context and self.strategy.include_article_context:
            article_title = article_context.get('title', '')
            if article_title:
                # Extract key words from article title
                title_words = self._extract_title_keywords(article_title)
                if title_words:
                    components.extend(title_words[:self.strategy.article_name_max_words])
        
        # Add sequence number
        if self.strategy.include_sequence:
            sequence_str = str(sequence_number).zfill(self.strategy.sequence_padding)
            components.append(f"img_{sequence_str}")
        
        # Add page number if available
        if self.strategy.include_page_number:
            image_node = graph.get_node(pair.image_node_id)
            if image_node and image_node.metadata:
                page_num = image_node.metadata.get('page')
                if page_num:
                    components.append(f"p{page_num}")
        
        # Join components
        base_filename = "_".join(components) if components else f"img_{sequence_number:03d}"
        base_filename = self._sanitize_filename(base_filename)
        
        return self._add_extension(base_filename)
    
    def _extract_image_type(self, caption_text: str) -> Optional[str]:
        """Extract image type from caption text."""
        if not caption_text:
            return None
        
        caption_lower = caption_text.lower()
        
        # Check each pattern
        for pattern, image_type in self.image_type_regex.items():
            if pattern.search(caption_text):
                return image_type
        
        return None
    
    def _extract_contributor_names(self, caption_text: str) -> List[str]:
        """Extract contributor names from caption text."""
        if not caption_text:
            return []
        
        names = []
        
        # Pattern for "by [Name]" or "photo by [Name]"
        by_pattern = re.compile(r'\b(?:photo\s+)?by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', re.IGNORECASE)
        by_matches = by_pattern.findall(caption_text)
        names.extend(by_matches)
        
        # Pattern for "photographer: [Name]" or similar
        credit_pattern = re.compile(r'\b(?:photographer|illustrator|artist)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', re.IGNORECASE)
        credit_matches = credit_pattern.findall(caption_text)
        names.extend(credit_matches)
        
        # Clean and deduplicate names
        cleaned_names = []
        seen_names = set()
        
        for name in names:
            cleaned_name = re.sub(r'[^\w\s]', '', name).strip()
            if (cleaned_name and 
                len(cleaned_name.split()) <= 3 and  # Max 3 words for a name
                cleaned_name.lower() not in seen_names):
                cleaned_names.append(cleaned_name)
                seen_names.add(cleaned_name.lower())
        
        return cleaned_names[:3]  # Max 3 names
    
    def _extract_descriptive_keywords(self, caption_text: str) -> List[str]:
        """Extract descriptive keywords from caption text."""
        if not caption_text:
            return []
        
        # Remove common caption prefixes
        text = re.sub(r'^(?:photo|image|picture|illustration)\s*(?:by|of|shows?)?\s*:?\s*', '', caption_text, flags=re.IGNORECASE)
        
        # Remove credit information
        text = re.sub(r'\b(?:photo|image)\s+(?:by|courtesy|credit)\s+[^.]*', '', text, flags=re.IGNORECASE)
        
        # Extract meaningful words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)  # Words with 3+ letters
        
        # Filter out stop words and common photo terms
        filtered_words = []
        photo_terms = {'photo', 'image', 'picture', 'shows', 'depicts', 'taken', 'captured'}
        
        for word in words:
            word_lower = word.lower()
            if (word_lower not in self.stop_words and 
                word_lower not in photo_terms and
                len(word_lower) >= 3):
                filtered_words.append(word_lower)
        
        # Return first few meaningful words
        return filtered_words[:4]
    
    def _extract_title_keywords(self, title: str) -> List[str]:
        """Extract key words from article title."""
        if not title:
            return []
        
        # Remove common article words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', title)
        
        filtered_words = []
        for word in words:
            word_lower = word.lower()
            if word_lower not in self.stop_words and len(word_lower) >= 3:
                filtered_words.append(word_lower)
        
        return filtered_words
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to be filesystem-safe."""
        if not self.strategy.sanitize_names:
            return filename
        
        # Convert to lowercase unless preserving case
        if not self.strategy.preserve_case:
            filename = filename.lower()
        
        # Replace spaces and special characters with underscores
        filename = re.sub(r'[^\w\-_.]', '_', filename)
        
        # Remove multiple consecutive underscores
        filename = re.sub(r'_+', '_', filename)
        
        # Remove leading/trailing underscores
        filename = filename.strip('_')
        
        # Ensure reasonable length
        if len(filename) > self.strategy.max_length - 10:  # Leave room for extension and uniqueness suffix
            filename = filename[:self.strategy.max_length - 10]
        
        return filename
    
    def _add_extension(self, base_filename: str) -> str:
        """Add appropriate file extension."""
        # For now, default to .jpg
        # In practice, this would be determined from the actual image file
        return f"{base_filename}.jpg"
    
    def _ensure_unique_filename(self, filename: str) -> str:
        """Ensure filename is unique by adding suffix if needed."""
        if filename not in self._generated_filenames:
            self._generated_filenames.add(filename)
            return filename
        
        # Generate unique variant
        base_name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        counter = self._filename_counters.get(base_name, 1)
        
        while True:
            if ext:
                candidate = f"{base_name}_{counter:02d}.{ext}"
            else:
                candidate = f"{base_name}_{counter:02d}"
            
            if candidate not in self._generated_filenames:
                self._generated_filenames.add(candidate)
                self._filename_counters[base_name] = counter + 1
                self.stats["collisions_resolved"] += 1
                return candidate
            
            counter += 1
    
    def _generate_fallback_filename(self, sequence_number: int) -> str:
        """Generate fallback filename when other methods fail."""
        sequence_str = str(sequence_number).zfill(self.strategy.sequence_padding)
        return f"image_{sequence_str}.jpg"
    
    def _get_generation_method(self, pair: ImageCaptionPair) -> str:
        """Get the method used to generate filename for logging."""
        return f"{self.strategy.format_type.value}_generation"
    
    def get_generation_statistics(self) -> Dict[str, any]:
        """Get filename generation statistics."""
        return {
            "filenames_generated": self.stats["filenames_generated"],
            "unique_filenames": self.stats["unique_filenames"],
            "collisions_resolved": self.stats["collisions_resolved"],
            "uniqueness_rate": (
                self.stats["unique_filenames"] / 
                max(1, self.stats["filenames_generated"])
            ),
            "format_distribution": self.stats["format_distribution"],
            "strategy_config": {
                "format_type": self.strategy.format_type.value,
                "max_length": self.strategy.max_length,
                "include_sequence": self.strategy.include_sequence,
                "include_descriptor": self.strategy.include_descriptor
            }
        }
    
    def clear_generation_state(self) -> None:
        """Clear generation state for new batch."""
        self._generated_filenames.clear()
        self._filename_counters.clear()