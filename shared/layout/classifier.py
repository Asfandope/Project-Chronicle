"""
Rule-based block classifier for layout analysis.
"""

import re
from typing import Any, Dict, List, Tuple

import structlog

from .types import BlockType, BoundingBox, LayoutConfig, TextBlock

logger = structlog.get_logger(__name__)


class BlockClassifier:
    """
    Rule-based text block classifier.

    Uses configurable rules to classify text blocks into different types
    based on font properties, position, content, and other features.
    """

    def __init__(self, config: LayoutConfig):
        """
        Initialize block classifier.

        Args:
            config: Layout configuration with classification rules
        """
        self.config = config
        self.rules = sorted(
            config.classification_rules, key=lambda r: r.priority, reverse=True
        )
        self.logger = logger.bind(component="BlockClassifier")

    def classify_blocks(
        self, text_blocks: List[TextBlock], page_rect: Any
    ) -> List[TextBlock]:
        """
        Classify a list of text blocks.

        Args:
            text_blocks: List of text blocks to classify
            page_rect: Page rectangle for position-based rules

        Returns:
            List of classified text blocks
        """
        try:
            self.logger.debug("Classifying text blocks", block_count=len(text_blocks))

            # Create a mock page layout for rule evaluation
            from .types import PageLayout

            mock_page = PageLayout(
                page_num=text_blocks[0].page_num if text_blocks else 1,
                page_width=page_rect.width,
                page_height=page_rect.height,
                text_blocks=text_blocks,
            )

            classified_blocks = []

            for block in text_blocks:
                # Classify individual block
                block_type, confidence = self._classify_single_block(block, mock_page)

                # Update block with classification
                block.block_type = block_type
                block.confidence = confidence

                classified_blocks.append(block)

            # Post-process classifications
            classified_blocks = self._post_process_classifications(
                classified_blocks, mock_page
            )

            self.logger.debug(
                "Block classification completed",
                classified_count=len(classified_blocks),
            )

            return classified_blocks

        except Exception as e:
            self.logger.error("Error classifying blocks", error=str(e), exc_info=True)
            # Return blocks with unknown classification
            for block in text_blocks:
                if block.block_type == BlockType.UNKNOWN:
                    block.block_type = BlockType.UNKNOWN
                    block.confidence = 0.0
            return text_blocks

    def _classify_single_block(
        self, block: TextBlock, page_layout: Any
    ) -> Tuple[BlockType, float]:
        """
        Classify a single text block using rules.

        Args:
            block: Text block to classify
            page_layout: Page layout context

        Returns:
            Tuple of (block_type, confidence)
        """
        try:
            best_match = None
            best_confidence = 0.0

            # Try each rule in priority order
            for rule in self.rules:
                matches, confidence = rule.matches(block, page_layout)

                if matches and confidence > best_confidence:
                    best_match = rule.block_type
                    best_confidence = confidence

                    # If we found a high-confidence match with high priority, use it
                    if confidence > 0.8 and rule.priority > 7:
                        break

            # Apply additional heuristics if no rule matched well
            if best_confidence < 0.5:
                heuristic_type, heuristic_confidence = self._apply_heuristics(
                    block, page_layout
                )
                if heuristic_confidence > best_confidence:
                    best_match = heuristic_type
                    best_confidence = heuristic_confidence

            return best_match or BlockType.UNKNOWN, best_confidence

        except Exception as e:
            self.logger.warning("Error classifying single block", error=str(e))
            return BlockType.UNKNOWN, 0.0

    def _apply_heuristics(
        self, block: TextBlock, page_layout: Any
    ) -> Tuple[BlockType, float]:
        """
        Apply additional heuristics for classification.

        Args:
            block: Text block to classify
            page_layout: Page layout context

        Returns:
            Tuple of (block_type, confidence)
        """
        try:
            text = block.text.strip()
            text.lower()

            # Page number heuristics
            if self._is_likely_page_number(text):
                return BlockType.PAGE_NUMBER, 0.8

            # Header/footer position heuristics
            if block.bbox.y0 < self.config.header_footer_margin:
                if not self._contains_main_content_indicators(text):
                    return BlockType.HEADER, 0.7

            if block.bbox.y1 > (
                page_layout.page_height - self.config.header_footer_margin
            ):
                if not self._contains_main_content_indicators(text):
                    return BlockType.FOOTER, 0.7

            # Title heuristics (position + properties)
            if (
                block.bbox.y0
                < page_layout.page_height * self.config.title_position_threshold
                and block.font_size
                and block.font_size > 16
                and block.word_count <= 15
            ):
                return BlockType.TITLE, 0.6

            # Byline heuristics
            if self._is_likely_byline(text):
                return BlockType.BYLINE, 0.7

            # Caption heuristics
            if self._is_likely_caption(text, block):
                return BlockType.CAPTION, 0.6

            # Quote heuristics
            if self._is_likely_quote(text):
                return BlockType.QUOTE, 0.6

            # Advertisement heuristics
            if self._is_likely_advertisement(text):
                return BlockType.ADVERTISEMENT, 0.7

            # Heading vs body heuristics
            if block.font_size and block.font_size > 12 and block.word_count <= 20:
                return BlockType.HEADING, 0.5

            # Default to body for substantial text
            if block.word_count >= 10:
                return BlockType.BODY, 0.4

            return BlockType.UNKNOWN, 0.0

        except Exception as e:
            self.logger.warning("Error applying heuristics", error=str(e))
            return BlockType.UNKNOWN, 0.0

    def _is_likely_page_number(self, text: str) -> bool:
        """Check if text is likely a page number."""
        text = text.strip()

        # Simple numeric page numbers
        if re.match(r"^\d+$", text):
            return True

        # Page X format
        if re.match(r"^Page\s+\d+$", text, re.IGNORECASE):
            return True

        # X of Y format
        if re.match(r"^\d+\s*/\s*\d+$", text):
            return True

        # Roman numerals
        if re.match(r"^[ivxlcdm]+$", text, re.IGNORECASE) and len(text) <= 6:
            return True

        return False

    def _contains_main_content_indicators(self, text: str) -> bool:
        """Check if text contains indicators of main content."""
        text_lower = text.lower()

        # Long text is likely main content
        if len(text.split()) > 20:
            return True

        # Common content patterns
        content_patterns = [
            r"\b(article|story|report|analysis|interview)\b",
            r"\b(said|according to|reported|stated)\b",
            r"\b(however|therefore|furthermore|moreover)\b",
        ]

        return any(re.search(pattern, text_lower) for pattern in content_patterns)

    def _is_likely_byline(self, text: str) -> bool:
        """Check if text is likely a byline."""
        text_lower = text.lower()

        # By + name patterns
        if re.search(r"^by\s+[a-z]+(?:\s+[a-z]+)*$", text_lower):
            return True

        # Reporter patterns
        if re.search(r"\b(?:reports?|correspondent|editor)\b", text_lower):
            return True

        # Author patterns
        if re.search(r"^author:\s*[a-z]+", text_lower):
            return True

        # Name + title patterns
        if re.search(
            r"^[A-Z][a-z]+\s+[A-Z][a-z]+,?\s+(?:reports?|correspondent)", text
        ):
            return True

        return False

    def _is_likely_caption(self, text: str, block: TextBlock) -> bool:
        """Check if text is likely a caption."""
        # Small font size
        if block.font_size and block.font_size < 10:
            return True

        # Short text near potential images
        if len(text.split()) <= 30:
            # Caption patterns
            if re.search(r"^(?:Photo|Image|Figure|Chart):", text, re.IGNORECASE):
                return True

            # Photo credit patterns
            if re.search(r"(?:photo|courtesy|credit):\s*", text, re.IGNORECASE):
                return True

        return False

    def _is_likely_quote(self, text: str) -> bool:
        """Check if text is likely a quote."""
        # Quoted text
        if (text.startswith('"') and text.endswith('"')) or (
            text.startswith("'") and text.endswith("'")
        ):
            return True

        # Quotation patterns
        if re.search(r'^"[^"]*"$', text):
            return True

        # Pull quote patterns
        if len(text.split()) <= 50 and '"' in text:
            return True

        return False

    def _is_likely_advertisement(self, text: str) -> bool:
        """Check if text is likely an advertisement."""
        text_lower = text.lower()

        # Advertisement keywords
        ad_keywords = [
            "advertisement",
            "sponsored",
            "subscribe",
            "buy now",
            "call now",
            "visit our website",
            "www.",
            "offer expires",
            "limited time",
        ]

        return any(keyword in text_lower for keyword in ad_keywords)

    def _post_process_classifications(
        self, blocks: List[TextBlock], page_layout: Any
    ) -> List[TextBlock]:
        """
        Post-process classifications to fix common errors.

        Args:
            blocks: Classified text blocks
            page_layout: Page layout context

        Returns:
            Post-processed text blocks
        """
        try:
            # Ensure only one main title per page
            titles = [b for b in blocks if b.block_type == BlockType.TITLE]
            if len(titles) > 1:
                # Keep the largest/highest confidence title
                best_title = max(titles, key=lambda t: (t.confidence, t.font_size or 0))
                for title in titles:
                    if title != best_title:
                        title.block_type = BlockType.HEADING
                        title.confidence *= 0.8

            # Convert very short "body" text to captions or headings
            for block in blocks:
                if block.block_type == BlockType.BODY and block.word_count < 5:
                    if block.font_size and block.font_size < 10:
                        block.block_type = BlockType.CAPTION
                    else:
                        block.block_type = BlockType.HEADING
                    block.confidence *= 0.9

            # Group adjacent similar blocks
            blocks = self._group_similar_adjacent_blocks(blocks)

            return blocks

        except Exception as e:
            self.logger.warning("Error in post-processing", error=str(e))
            return blocks

    def _group_similar_adjacent_blocks(
        self, blocks: List[TextBlock]
    ) -> List[TextBlock]:
        """Group adjacent blocks of the same type if appropriate."""
        try:
            if len(blocks) <= 1:
                return blocks

            # Sort by reading order
            sorted_blocks = sorted(blocks, key=lambda b: (b.bbox.y0, b.bbox.x0))

            grouped_blocks = []
            i = 0

            while i < len(sorted_blocks):
                current_block = sorted_blocks[i]

                # Look for similar adjacent blocks to group
                if current_block.block_type == BlockType.BODY and i + 1 < len(
                    sorted_blocks
                ):
                    next_block = sorted_blocks[i + 1]

                    # Check if blocks should be grouped
                    if (
                        next_block.block_type == BlockType.BODY
                        and self._should_group_blocks(current_block, next_block)
                    ):
                        # Merge blocks
                        merged_block = self._merge_adjacent_blocks(
                            current_block, next_block
                        )
                        grouped_blocks.append(merged_block)
                        i += 2  # Skip both blocks
                        continue

                grouped_blocks.append(current_block)
                i += 1

            return grouped_blocks

        except Exception as e:
            self.logger.warning("Error grouping similar blocks", error=str(e))
            return blocks

    def _should_group_blocks(self, block1: TextBlock, block2: TextBlock) -> bool:
        """Check if two blocks should be grouped together."""
        try:
            # Must be same type
            if block1.block_type != block2.block_type:
                return False

            # Must be reasonably close
            distance = block1.bbox.distance_to(block2.bbox)
            if distance > 50:  # 50 pixels
                return False

            # Similar font properties
            if (
                block1.font_size
                and block2.font_size
                and abs(block1.font_size - block2.font_size) > 2
            ):
                return False

            # Check vertical alignment for body text
            if block1.block_type == BlockType.BODY:
                # Should be in same column or similar x-position
                x_overlap = min(block1.bbox.x1, block2.bbox.x1) - max(
                    block1.bbox.x0, block2.bbox.x0
                )
                min_width = min(block1.bbox.width, block2.bbox.width)
                if x_overlap < min_width * 0.7:  # 70% overlap
                    return False

            return True

        except Exception:
            return False

    def _merge_adjacent_blocks(self, block1: TextBlock, block2: TextBlock) -> TextBlock:
        """Merge two adjacent blocks."""
        try:
            # Combine text
            combined_text = block1.text + "\n" + block2.text

            # Combine bounding boxes
            combined_bbox = BoundingBox(
                x0=min(block1.bbox.x0, block2.bbox.x0),
                y0=min(block1.bbox.y0, block2.bbox.y0),
                x1=max(block1.bbox.x1, block2.bbox.x1),
                y1=max(block1.bbox.y1, block2.bbox.y1),
            )

            # Use average confidence
            avg_confidence = (block1.confidence + block2.confidence) / 2

            # Merge classification features
            merged_features = block1.classification_features.copy()
            merged_features["merged_adjacent"] = True

            return TextBlock(
                text=combined_text,
                bbox=combined_bbox,
                block_type=block1.block_type,
                confidence=avg_confidence,
                font_size=block1.font_size or block2.font_size,
                font_family=block1.font_family or block2.font_family,
                is_bold=block1.is_bold or block2.is_bold,
                is_italic=block1.is_italic or block2.is_italic,
                page_num=block1.page_num,
                reading_order=block1.reading_order,
                column=block1.column,
                classification_features=merged_features,
            )

        except Exception as e:
            self.logger.warning("Error merging adjacent blocks", error=str(e))
            return block1

    def get_classification_stats(self, blocks: List[TextBlock]) -> Dict[str, Any]:
        """Get classification statistics."""
        try:
            stats = {
                "total_blocks": len(blocks),
                "blocks_by_type": {},
                "avg_confidence": 0.0,
                "min_confidence": 1.0,
                "max_confidence": 0.0,
                "unclassified_count": 0,
            }

            confidences = []

            for block in blocks:
                # Count by type
                block_type = block.block_type.value
                stats["blocks_by_type"][block_type] = (
                    stats["blocks_by_type"].get(block_type, 0) + 1
                )

                # Confidence stats
                confidences.append(block.confidence)

                # Count unclassified
                if block.block_type == BlockType.UNKNOWN:
                    stats["unclassified_count"] += 1

            if confidences:
                stats["avg_confidence"] = sum(confidences) / len(confidences)
                stats["min_confidence"] = min(confidences)
                stats["max_confidence"] = max(confidences)

            stats["classification_rate"] = (
                len(blocks) - stats["unclassified_count"]
            ) / max(len(blocks), 1)

            return stats

        except Exception as e:
            self.logger.error("Error getting classification stats", error=str(e))
            return {"error": str(e)}
