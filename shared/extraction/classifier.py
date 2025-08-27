"""
Role classification for extracted contributors.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .types import ContributorMatch, ContributorRole, ExtractionConfig


@dataclass
class RoleContext:
    """Context information for role classification."""

    preceding_text: str = ""
    following_text: str = ""
    full_context: str = ""
    position_in_text: int = 0
    surrounding_keywords: List[str] = field(default_factory=list)


class RoleClassifier:
    """Classifies contributor roles using pattern matching and context analysis."""

    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.config = config or ExtractionConfig()
        self._initialize_patterns()

    def _initialize_patterns(self) -> None:
        """Initialize role classification patterns."""

        # Author patterns - indicators that suggest author role
        self.author_patterns = {
            "byline_indicators": [
                r"\bby\b\s*",
                r"\bwritten\s+by\b",
                r"\bauthor(?:ed)?\s*(?:by)?\b",
                r"\breport(?:ed|er)\s*(?:by)?\b",
                r"\bcorrespondent\b",
                r"\bcolumnist\b",
                r"\bstaff\s+writer\b",
                r"\bcontributing\s+(?:writer|author|editor)\b",
            ],
            "position_indicators": [
                r"^\s*by\s+",  # Line starting with "by"
                r"--\s*",  # Em dash before name
                r":\s*",  # Colon before name in headlines
            ],
            "role_titles": [
                r"\bsenior\s+(?:writer|correspondent|editor)\b",
                r"\bstaff\s+(?:writer|reporter)\b",
                r"\bspecial\s+correspondent\b",
                r"\bassociate\s+editor\b",
                r"\bcontributing\s+editor\b",
                r"\bguest\s+columnist\b",
            ],
        }

        # Photographer patterns
        self.photographer_patterns = {
            "credit_indicators": [
                r"\bphoto(?:graph)?\s*(?:by|credit|courtesy)\b",
                r"\bpicture\s*(?:by|credit)\b",
                r"\bimage\s*(?:by|credit|courtesy)\b",
                r"\bshot\s*by\b",
                r"\bcaptured\s*by\b",
                r"\bphotographer\b",
            ],
            "position_indicators": [
                r"\(photo\s*(?:by|credit)\s*[:\)]",
                r"\[photo\s*(?:by|credit)\s*[:]\]",
                r"photo\s*credit\s*:",
                r"image\s*(?:courtesy|credit)\s*(?:of|:)",
            ],
            "role_titles": [
                r"\bstaff\s+photographer\b",
                r"\bsenior\s+photographer\b",
                r"\bfreelance\s+photographer\b",
                r"\bphoto\s+editor\b",
                r"\bcontributing\s+photographer\b",
            ],
        }

        # Illustrator patterns
        self.illustrator_patterns = {
            "credit_indicators": [
                r"\billustration\s*(?:by|credit)\b",
                r"\bdrawing\s*(?:by|credit)\b",
                r"\bartwork\s*(?:by|credit)\b",
                r"\bgraphic\s*(?:by|credit)\b",
                r"\bdesign\s*(?:by|credit)\b",
                r"\billustrator\b",
            ],
            "position_indicators": [
                r"\(illustration\s*(?:by|credit)\s*[:\)]",
                r"\[graphic\s*(?:by|credit)\s*[:]\]",
                r"illustration\s*credit\s*:",
                r"artwork\s*(?:courtesy|credit)\s*(?:of|:)",
            ],
            "role_titles": [
                r"\bstaff\s+illustrator\b",
                r"\bsenior\s+illustrator\b",
                r"\bfreelance\s+illustrator\b",
                r"\bgraphic\s+(?:designer|artist)\b",
                r"\bart\s+director\b",
            ],
        }

        # Editor patterns
        self.editor_patterns = {
            "role_indicators": [
                r"\bedited\s+by\b",
                r"\beditor\b",
                r"\beditorial\s+(?:by|credit)\b",
                r"\bmanaging\s+editor\b",
                r"\bexecutive\s+editor\b",
                r"\bassistant\s+editor\b",
            ]
        }

        # Compiled patterns for efficiency
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for better performance."""
        self.compiled_patterns = {}

        for role in ["author", "photographer", "illustrator", "editor"]:
            role_patterns = getattr(self, f"{role}_patterns", {})
            self.compiled_patterns[role] = {}

            for category, patterns in role_patterns.items():
                self.compiled_patterns[role][category] = [
                    re.compile(pattern, re.IGNORECASE) for pattern in patterns
                ]

    def classify_role(
        self, match: ContributorMatch, full_text: str
    ) -> Tuple[ContributorRole, float]:
        """
        Classify the role of a contributor match.

        Args:
            match: ContributorMatch to classify
            full_text: Full document text for context analysis

        Returns:
            Tuple of (role, confidence_score)
        """
        # Extract context around the match
        context = self._extract_context(match, full_text)

        # Calculate role scores
        role_scores = self._calculate_role_scores(match, context)

        # Determine best role and confidence
        best_role, confidence = self._select_best_role(role_scores)

        return best_role, confidence

    def _extract_context(self, match: ContributorMatch, full_text: str) -> RoleContext:
        """Extract context information around a match."""
        window_size = self.config.context_window_size

        # Get surrounding text
        start_pos = max(0, match.start_pos - window_size)
        end_pos = min(len(full_text), match.end_pos + window_size)

        preceding = full_text[start_pos : match.start_pos]
        following = full_text[match.end_pos : end_pos]
        full_context = full_text[start_pos:end_pos]

        # Extract keywords from context
        keywords = self._extract_keywords(full_context)

        return RoleContext(
            preceding_text=preceding,
            following_text=following,
            full_context=full_context,
            position_in_text=match.start_pos,
            surrounding_keywords=keywords,
        )

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from context text."""
        # Keywords that might indicate role
        role_keywords = [
            "by",
            "photo",
            "photograph",
            "image",
            "picture",
            "shot",
            "credit",
            "illustration",
            "drawing",
            "artwork",
            "graphic",
            "design",
            "writer",
            "author",
            "reporter",
            "correspondent",
            "columnist",
            "photographer",
            "illustrator",
            "designer",
            "editor",
            "staff",
            "freelance",
            "senior",
            "contributing",
            "special",
            "courtesy",
        ]

        found_keywords = []
        text_lower = text.lower()

        for keyword in role_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)

        return found_keywords

    def _calculate_role_scores(
        self, match: ContributorMatch, context: RoleContext
    ) -> Dict[ContributorRole, float]:
        """Calculate confidence scores for each possible role."""
        scores = {
            ContributorRole.AUTHOR: 0.0,
            ContributorRole.PHOTOGRAPHER: 0.0,
            ContributorRole.ILLUSTRATOR: 0.0,
            ContributorRole.EDITOR: 0.0,
        }

        # Score each role based on patterns and context
        for role_name in ["author", "photographer", "illustrator", "editor"]:
            role_enum = getattr(ContributorRole, role_name.upper())
            score = self._score_role_patterns(role_name, match, context)
            scores[role_enum] = score

        return scores

    def _score_role_patterns(
        self, role_name: str, match: ContributorMatch, context: RoleContext
    ) -> float:
        """Score a specific role based on pattern matching."""
        if role_name not in self.compiled_patterns:
            return 0.0

        total_score = 0.0
        max_possible_score = 0.0

        role_patterns = self.compiled_patterns[role_name]

        # Weight different pattern categories
        category_weights = {
            "byline_indicators": 1.0,
            "credit_indicators": 1.0,
            "role_indicators": 1.0,
            "position_indicators": 0.8,
            "role_titles": 0.9,
        }

        for category, patterns in role_patterns.items():
            weight = category_weights.get(category, 0.5)
            max_possible_score += weight

            # Check if any pattern in this category matches
            category_matched = False
            for pattern in patterns:
                if pattern.search(context.full_context):
                    category_matched = True
                    break

            if category_matched:
                total_score += weight

        # Normalize score
        if max_possible_score > 0:
            normalized_score = total_score / max_possible_score
        else:
            normalized_score = 0.0

        # Boost score based on context keywords
        keyword_boost = self._calculate_keyword_boost(
            role_name, context.surrounding_keywords
        )
        final_score = min(1.0, normalized_score + keyword_boost)

        return final_score

    def _calculate_keyword_boost(self, role_name: str, keywords: List[str]) -> float:
        """Calculate boost score based on relevant keywords in context."""
        role_keyword_maps = {
            "author": [
                "by",
                "writer",
                "author",
                "reporter",
                "correspondent",
                "columnist",
            ],
            "photographer": [
                "photo",
                "photograph",
                "image",
                "picture",
                "shot",
                "photographer",
            ],
            "illustrator": [
                "illustration",
                "drawing",
                "artwork",
                "graphic",
                "design",
                "illustrator",
            ],
            "editor": ["editor", "edited", "editorial"],
        }

        relevant_keywords = role_keyword_maps.get(role_name, [])
        keyword_matches = sum(1 for kw in keywords if kw in relevant_keywords)

        # Convert to boost score (max 0.2 boost)
        if relevant_keywords:
            boost = min(0.2, (keyword_matches / len(relevant_keywords)) * 0.2)
        else:
            boost = 0.0

        return boost

    def _select_best_role(
        self, role_scores: Dict[ContributorRole, float]
    ) -> Tuple[ContributorRole, float]:
        """Select the best role based on scores."""
        # Find the role with highest score
        best_role = max(role_scores.keys(), key=lambda r: role_scores[r])
        best_score = role_scores[best_role]

        # If no role has sufficient confidence, return UNKNOWN
        if best_score < self.config.min_role_confidence:
            return ContributorRole.UNKNOWN, best_score

        # Check if there's ambiguity (multiple high scores)
        sorted_scores = sorted(role_scores.values(), reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[0] - sorted_scores[1] < 0.1:
            # Ambiguous case - reduce confidence
            confidence = best_score * 0.8
        else:
            confidence = best_score

        return best_role, confidence

    def classify_multiple(
        self, matches: List[ContributorMatch], full_text: str
    ) -> List[Tuple[ContributorRole, float]]:
        """Classify multiple matches efficiently."""
        results = []

        for match in matches:
            role, confidence = self.classify_role(match, full_text)
            results.append((role, confidence))

        return results

    def get_role_statistics(
        self, matches: List[ContributorMatch], full_text: str
    ) -> Dict[str, any]:
        """Get statistics about role classification for a set of matches."""
        classifications = self.classify_multiple(matches, full_text)

        role_counts = {}
        total_confidence = 0.0
        high_confidence_count = 0

        for role, confidence in classifications:
            role_name = role.value
            role_counts[role_name] = role_counts.get(role_name, 0) + 1
            total_confidence += confidence

            if confidence >= self.config.min_role_confidence:
                high_confidence_count += 1

        return {
            "total_matches": len(matches),
            "role_distribution": role_counts,
            "average_confidence": total_confidence / len(matches) if matches else 0.0,
            "high_confidence_rate": high_confidence_count / len(matches)
            if matches
            else 0.0,
            "classification_accuracy": high_confidence_count / len(matches)
            if matches
            else 0.0,
        }
