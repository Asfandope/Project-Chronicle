"""
Pattern matching for bylines and photo credits.
"""

import re
from typing import Dict, List, Optional, Pattern

import structlog

from .types import ContributorMatch, ContributorRole

logger = structlog.get_logger(__name__)


class BylinePatterns:
    """
    Pattern matching for bylines in article text.

    Handles various byline formats and structures commonly found
    in newspapers, magazines, and online publications.
    """

    def __init__(self):
        """Initialize byline patterns."""
        self.logger = logger.bind(component="BylinePatterns")

        # Compile patterns for efficiency
        self.patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[str, Pattern]:
        """Compile regex patterns for byline detection."""
        patterns = {}

        # Standard "By [Name]" patterns
        patterns["by_simple"] = re.compile(
            r"(?:^|\\n|\\s)By\\s+([A-Z][a-z]+(?:\\s+[A-Z][a-z]*)*(?:\\s+[A-Z][a-z]+))",
            re.IGNORECASE | re.MULTILINE,
        )

        patterns["by_with_title"] = re.compile(
            r"(?:^|\\n|\\s)By\\s+([A-Z][a-z]+(?:\\s+[A-Z][a-z]*)*(?:\\s+[A-Z][a-z]+))(?:,\\s*([^\\n]+?))?$",
            re.IGNORECASE | re.MULTILINE,
        )

        # Multiple authors
        patterns["by_multiple"] = re.compile(
            r"By\\s+([A-Z][a-z]+(?:\\s+[A-Z][a-z]*)*(?:\\s+[A-Z][a-z]+))(?:\\s+and\\s+([A-Z][a-z]+(?:\\s+[A-Z][a-z]*)*(?:\\s+[A-Z][a-z]+)))+",
            re.IGNORECASE,
        )

        patterns["by_comma_separated"] = re.compile(
            r"By\\s+([A-Z][a-z]+(?:\\s+[A-Z][a-z]*)*(?:\\s+[A-Z][a-z]+))(?:,\\s*([A-Z][a-z]+(?:\\s+[A-Z][a-z]*)*(?:\\s+[A-Z][a-z]+)))*",
            re.IGNORECASE,
        )

        # Reporter/correspondent patterns
        patterns["reporter"] = re.compile(
            r"([A-Z][a-z]+(?:\\s+[A-Z][a-z]*)*(?:\\s+[A-Z][a-z]+))\\s*,?\\s*(?:reports?|correspondent|staff\\s+writer)",
            re.IGNORECASE,
        )

        patterns["correspondent"] = re.compile(
            r"([A-Z][a-z]+(?:\\s+[A-Z][a-z]*)*(?:\\s+[A-Z][a-z]+))\\s*,?\\s*(?:correspondent|reporter)(?:\\s+in\\s+([A-Z][a-z]+))?",
            re.IGNORECASE,
        )

        # Publication-specific patterns
        patterns["from_location"] = re.compile(
            r"From\\s+([A-Z][a-z]+(?:\\s+[A-Z][a-z]*)*(?:\\s+[A-Z][a-z]+))\\s+in\\s+([A-Z][a-z]+)",
            re.IGNORECASE,
        )

        patterns["dateline"] = re.compile(
            r"([A-Z][A-Z\\s]+)\\s*[-–—]\\s*([A-Z][a-z]+(?:\\s+[A-Z][a-z]*)*(?:\\s+[A-Z][a-z]+))",
            re.IGNORECASE,
        )

        # Author attribution at end
        patterns["author_suffix"] = re.compile(
            r"Author:\\s*([A-Z][a-z]+(?:\\s+[A-Z][a-z]*)*(?:\\s+[A-Z][a-z]+))",
            re.IGNORECASE,
        )

        patterns["written_by"] = re.compile(
            r"Written\\s+by\\s+([A-Z][a-z]+(?:\\s+[A-Z][a-z]*)*(?:\\s+[A-Z][a-z]+))",
            re.IGNORECASE,
        )

        # Special formats
        patterns["name_title"] = re.compile(
            r"([A-Z][a-z]+(?:\\s+[A-Z][a-z]*)*(?:\\s+[A-Z][a-z]+))\\s*,\\s*([^\\n]+?)(?:for|at|of)\\s+([A-Z][a-z][^\\n]*)",
            re.IGNORECASE,
        )

        patterns["contributing"] = re.compile(
            r"([A-Z][a-z]+(?:\\s+[A-Z][a-z]*)*(?:\\s+[A-Z][a-z]+))\\s*(?:contributed|contributing)\\s+(?:to\\s+this\\s+report|reporting)",
            re.IGNORECASE,
        )

        return patterns

    def find_bylines(self, text: str) -> List[ContributorMatch]:
        """
        Find all byline matches in text.

        Args:
            text: Input text to search

        Returns:
            List of contributor matches
        """
        matches = []

        try:
            for pattern_name, pattern in self.patterns.items():
                for match in pattern.finditer(text):
                    contributor_match = self._create_match_from_regex(
                        match, text, pattern_name, ContributorRole.AUTHOR
                    )
                    if contributor_match:
                        matches.append(contributor_match)

            # Deduplicate overlapping matches
            matches = self._deduplicate_matches(matches)

            self.logger.debug("Found byline matches", count=len(matches))
            return matches

        except Exception as e:
            self.logger.error("Error finding bylines", error=str(e))
            return []

    def _create_match_from_regex(
        self, match: re.Match, text: str, pattern_name: str, role: ContributorRole
    ) -> Optional[ContributorMatch]:
        """Create ContributorMatch from regex match."""
        try:
            match_text = match.group(0)
            start_pos = match.start()
            end_pos = match.end()

            # Extract names from groups
            extracted_names = []
            for i in range(1, match.lastindex + 1 if match.lastindex else 1):
                group = match.group(i)
                if group and self._is_likely_name(group):
                    extracted_names.append(group.strip())

            if not extracted_names:
                return None

            # Get context
            context_start = max(0, start_pos - 50)
            context_end = min(len(text), end_pos + 50)
            context_before = text[context_start:start_pos]
            context_after = text[end_pos:context_end]

            # Calculate confidence based on pattern strength
            confidence = self._calculate_pattern_confidence(pattern_name, match_text)

            return ContributorMatch(
                text=match_text,
                start_pos=start_pos,
                end_pos=end_pos,
                role=role,
                role_confidence=confidence,
                extracted_names=extracted_names,
                context_before=context_before,
                context_after=context_after,
                pattern_used=pattern_name,
                extraction_method="regex_pattern",
                extraction_confidence=confidence,
            )

        except Exception as e:
            self.logger.warning("Error creating match from regex", error=str(e))
            return None

    def _is_likely_name(self, text: str) -> bool:
        """Check if text is likely a person's name."""
        if not text or len(text.strip()) < 2:
            return False

        text = text.strip()

        # Basic name validation
        if not re.match(r"^[A-Za-z][A-Za-z\\s\\.'\\-]+$", text):
            return False

        # Check for reasonable name patterns
        words = text.split()
        if len(words) > 5:  # Too many words
            return False

        # Check for title words that suggest not a name
        title_words = {
            "editor",
            "reporter",
            "correspondent",
            "writer",
            "journalist",
            "photographer",
            "staff",
            "bureau",
            "chief",
            "senior",
            "associate",
        }

        if any(word.lower() in title_words for word in words):
            return False

        return True

    def _calculate_pattern_confidence(
        self, pattern_name: str, match_text: str
    ) -> float:
        """Calculate confidence score for pattern match."""
        base_confidence = {
            "by_simple": 0.95,
            "by_with_title": 0.93,
            "by_multiple": 0.90,
            "by_comma_separated": 0.88,
            "reporter": 0.85,
            "correspondent": 0.87,
            "from_location": 0.82,
            "dateline": 0.75,
            "author_suffix": 0.90,
            "written_by": 0.92,
            "name_title": 0.80,
            "contributing": 0.78,
        }.get(pattern_name, 0.70)

        # Adjust based on match characteristics
        if "By " in match_text and match_text.startswith("By "):
            base_confidence += 0.05

        if len(match_text.split()) <= 3:  # Short, clean match
            base_confidence += 0.03

        return min(1.0, base_confidence)

    def _deduplicate_matches(
        self, matches: List[ContributorMatch]
    ) -> List[ContributorMatch]:
        """Remove overlapping or duplicate matches."""
        if not matches:
            return matches

        # Sort by start position
        matches.sort(key=lambda m: m.start_pos)

        deduplicated = []
        for match in matches:
            # Check for overlap with existing matches
            overlaps = False
            for existing in deduplicated:
                if (
                    match.start_pos < existing.end_pos
                    and match.end_pos > existing.start_pos
                ):
                    # Overlapping - keep the higher confidence one
                    if match.extraction_confidence > existing.extraction_confidence:
                        deduplicated.remove(existing)
                        deduplicated.append(match)
                    overlaps = True
                    break

            if not overlaps:
                deduplicated.append(match)

        return deduplicated


class CreditPatterns:
    """
    Pattern matching for photo credits and other media credits.

    Handles various credit formats for photographers, illustrators,
    and graphic designers.
    """

    def __init__(self):
        """Initialize credit patterns."""
        self.logger = logger.bind(component="CreditPatterns")
        self.patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[str, Pattern]:
        """Compile regex patterns for credit detection."""
        patterns = {}

        # Photo credit patterns
        patterns["photo_credit"] = re.compile(
            r"(?:Photo|Image|Picture)\\s*(?:by|credit|courtesy)?:?\\s*([A-Z][a-z]+(?:\\s+[A-Z][a-z]*)*(?:\\s+[A-Z][a-z]+))",
            re.IGNORECASE,
        )

        patterns["photo_by"] = re.compile(
            r"Photo\\s+by\\s+([A-Z][a-z]+(?:\\s+[A-Z][a-z]*)*(?:\\s+[A-Z][a-z]+))",
            re.IGNORECASE,
        )

        patterns["photography"] = re.compile(
            r"Photography\\s*:?\\s*([A-Z][a-z]+(?:\\s+[A-Z][a-z]*)*(?:\\s+[A-Z][a-z]+))",
            re.IGNORECASE,
        )

        patterns["photographer"] = re.compile(
            r"Photographer\\s*:?\\s*([A-Z][a-z]+(?:\\s+[A-Z][a-z]*)*(?:\\s+[A-Z][a-z]+))",
            re.IGNORECASE,
        )

        # Illustration patterns
        patterns["illustration"] = re.compile(
            r"Illustration\\s*(?:by)?\\s*:?\\s*([A-Z][a-z]+(?:\\s+[A-Z][a-z]*)*(?:\\s+[A-Z][a-z]+))",
            re.IGNORECASE,
        )

        patterns["illustrator"] = re.compile(
            r"Illustrator\\s*:?\\s*([A-Z][a-z]+(?:\\s+[A-Z][a-z]*)*(?:\\s+[A-Z][a-z]+))",
            re.IGNORECASE,
        )

        patterns["graphic_by"] = re.compile(
            r"Graphic\\s+by\\s+([A-Z][a-z]+(?:\\s+[A-Z][a-z]*)*(?:\\s+[A-Z][a-z]+))",
            re.IGNORECASE,
        )

        # General credit patterns
        patterns["credit_colon"] = re.compile(
            r"Credit\\s*:?\\s*([A-Z][a-z]+(?:\\s+[A-Z][a-z]*)*(?:\\s+[A-Z][a-z]+))",
            re.IGNORECASE,
        )

        patterns["courtesy"] = re.compile(
            r"Courtesy\\s+(?:of\\s+)?([A-Z][a-z]+(?:\\s+[A-Z][a-z]*)*(?:\\s+[A-Z][a-z]+))",
            re.IGNORECASE,
        )

        # Source attribution
        patterns["source"] = re.compile(
            r"Source\\s*:?\\s*([A-Z][a-z]+(?:\\s+[A-Z][a-z]*)*(?:\\s+[A-Z][a-z]+))",
            re.IGNORECASE,
        )

        return patterns

    def find_credits(self, text: str) -> List[ContributorMatch]:
        """
        Find all credit matches in text.

        Args:
            text: Input text to search

        Returns:
            List of contributor matches
        """
        matches = []

        try:
            for pattern_name, pattern in self.patterns.items():
                role = self._determine_role_from_pattern(pattern_name)

                for match in pattern.finditer(text):
                    contributor_match = self._create_credit_match(
                        match, text, pattern_name, role
                    )
                    if contributor_match:
                        matches.append(contributor_match)

            # Deduplicate overlapping matches
            matches = self._deduplicate_credit_matches(matches)

            self.logger.debug("Found credit matches", count=len(matches))
            return matches

        except Exception as e:
            self.logger.error("Error finding credits", error=str(e))
            return []

    def _determine_role_from_pattern(self, pattern_name: str) -> ContributorRole:
        """Determine contributor role from pattern name."""
        role_mapping = {
            "photo_credit": ContributorRole.PHOTOGRAPHER,
            "photo_by": ContributorRole.PHOTOGRAPHER,
            "photography": ContributorRole.PHOTOGRAPHER,
            "photographer": ContributorRole.PHOTOGRAPHER,
            "illustration": ContributorRole.ILLUSTRATOR,
            "illustrator": ContributorRole.ILLUSTRATOR,
            "graphic_by": ContributorRole.GRAPHIC_DESIGNER,
            "credit_colon": ContributorRole.UNKNOWN,
            "courtesy": ContributorRole.UNKNOWN,
            "source": ContributorRole.UNKNOWN,
        }

        return role_mapping.get(pattern_name, ContributorRole.UNKNOWN)

    def _create_credit_match(
        self, match: re.Match, text: str, pattern_name: str, role: ContributorRole
    ) -> Optional[ContributorMatch]:
        """Create ContributorMatch from credit regex match."""
        try:
            match_text = match.group(0)
            start_pos = match.start()
            end_pos = match.end()

            # Extract name from first group
            name = match.group(1).strip()

            if not self._is_valid_credit_name(name):
                return None

            # Get context
            context_start = max(0, start_pos - 30)
            context_end = min(len(text), end_pos + 30)
            context_before = text[context_start:start_pos]
            context_after = text[end_pos:context_end]

            # Calculate confidence
            confidence = self._calculate_credit_confidence(pattern_name, match_text)

            return ContributorMatch(
                text=match_text,
                start_pos=start_pos,
                end_pos=end_pos,
                role=role,
                role_confidence=confidence,
                extracted_names=[name],
                context_before=context_before,
                context_after=context_after,
                pattern_used=pattern_name,
                extraction_method="credit_pattern",
                extraction_confidence=confidence,
            )

        except Exception as e:
            self.logger.warning("Error creating credit match", error=str(e))
            return None

    def _is_valid_credit_name(self, name: str) -> bool:
        """Check if extracted name is valid for a credit."""
        if not name or len(name.strip()) < 2:
            return False

        name = name.strip()

        # Basic validation
        if not re.match(r"^[A-Za-z][A-Za-z\\s\\.'\\-]+$", name):
            return False

        # Check for organization/agency names that are not person names
        org_indicators = {
            "ap",
            "reuters",
            "getty",
            "afp",
            "shutterstock",
            "stock",
            "news",
            "press",
            "agency",
            "service",
            "media",
            "photo",
            "images",
            "pictures",
            "wire",
            "bureau",
        }

        name_lower = name.lower()
        if any(indicator in name_lower for indicator in org_indicators):
            return False

        # Check for reasonable length
        words = name.split()
        return 1 <= len(words) <= 4

    def _calculate_credit_confidence(self, pattern_name: str, match_text: str) -> float:
        """Calculate confidence score for credit match."""
        base_confidence = {
            "photo_credit": 0.92,
            "photo_by": 0.95,
            "photography": 0.90,
            "photographer": 0.93,
            "illustration": 0.90,
            "illustrator": 0.93,
            "graphic_by": 0.88,
            "credit_colon": 0.75,
            "courtesy": 0.70,
            "source": 0.65,
        }.get(pattern_name, 0.60)

        # Adjust based on match characteristics
        if "by" in match_text.lower():
            base_confidence += 0.05

        if len(match_text.split()) <= 4:  # Concise credit
            base_confidence += 0.03

        return min(1.0, base_confidence)

    def _deduplicate_credit_matches(
        self, matches: List[ContributorMatch]
    ) -> List[ContributorMatch]:
        """Remove overlapping or duplicate credit matches."""
        # Similar to byline deduplication but optimized for credits
        if not matches:
            return matches

        matches.sort(key=lambda m: m.start_pos)

        deduplicated = []
        for match in matches:
            overlaps = False
            for existing in deduplicated:
                if (
                    match.start_pos < existing.end_pos
                    and match.end_pos > existing.start_pos
                ):
                    # For credits, prefer more specific roles
                    role_priority = {
                        ContributorRole.PHOTOGRAPHER: 3,
                        ContributorRole.ILLUSTRATOR: 3,
                        ContributorRole.GRAPHIC_DESIGNER: 2,
                        ContributorRole.UNKNOWN: 1,
                    }

                    match_priority = role_priority.get(match.role, 0)
                    existing_priority = role_priority.get(existing.role, 0)

                    if match_priority > existing_priority or (
                        match_priority == existing_priority
                        and match.extraction_confidence > existing.extraction_confidence
                    ):
                        deduplicated.remove(existing)
                        deduplicated.append(match)
                    overlaps = True
                    break

            if not overlaps:
                deduplicated.append(match)

        return deduplicated
