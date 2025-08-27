"""
Edge case handling for contributor extraction.
"""

import re
from dataclasses import dataclass
from typing import List, Set, Tuple

from .normalizer import NameNormalizer
from .types import (
    ContributorMatch,
    ContributorRole,
    ExtractedContributor,
    NormalizedName,
)


@dataclass
class EdgeCasePattern:
    """Definition of an edge case pattern."""

    name: str
    pattern: re.Pattern
    handler: str
    priority: int = 1


class EdgeCaseHandler:
    """Handles edge cases in contributor extraction."""

    def __init__(self):
        self.name_normalizer = NameNormalizer()
        self._initialize_edge_case_patterns()

    def _initialize_edge_case_patterns(self) -> None:
        """Initialize edge case detection patterns."""

        self.edge_case_patterns = [
            # Multiple authors with "and"
            EdgeCasePattern(
                "multiple_authors_and",
                re.compile(
                    r"\b([A-Z][a-z]+(?: [A-Z][a-z]*)*)\s+and\s+([A-Z][a-z]+(?: [A-Z][a-z]*)*)\b"
                ),
                "handle_multiple_authors_and",
                priority=3,
            ),
            # Multiple authors with comma separation
            EdgeCasePattern(
                "multiple_authors_comma",
                re.compile(
                    r"\b([A-Z][a-z]+(?: [A-Z][a-z]*)*)\s*,\s+([A-Z][a-z]+(?: [A-Z][a-z]*)*)\s*(?:,\s+and\s+([A-Z][a-z]+(?: [A-Z][a-z]*)*))?\b"
                ),
                "handle_multiple_authors_comma",
                priority=3,
            ),
            # Names with multiple titles/prefixes
            EdgeCasePattern(
                "multiple_titles",
                re.compile(
                    r"\b(?:Dr\.?\s+Prof\.?\s+|Prof\.?\s+Dr\.?\s+|Rev\.?\s+Dr\.?\s+)([A-Z][a-z]+(?: [A-Z][a-z]*)*)\b"
                ),
                "handle_multiple_titles",
                priority=2,
            ),
            # Names with complex suffixes
            EdgeCasePattern(
                "complex_suffixes",
                re.compile(
                    r"\b([A-Z][a-z]+(?: [A-Z][a-z]*)*)\s*,?\s*(Jr\.?|Sr\.?|III?|IV|V|VI|VII|VIII|IX|X)(?:\s*,?\s*(Ph\.?D\.?|M\.?D\.?|J\.?D\.?|Esq\.?))?\b"
                ),
                "handle_complex_suffixes",
                priority=2,
            ),
            # Hyphenated surnames
            EdgeCasePattern(
                "hyphenated_surnames",
                re.compile(
                    r"\b([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)*)\s+([A-Z][a-z]+-[A-Z][a-z]+(?:-[A-Z][a-z]+)*)\b"
                ),
                "handle_hyphenated_surnames",
                priority=2,
            ),
            # Names with apostrophes (O'Connor, D'Angelo)
            EdgeCasePattern(
                "apostrophe_names",
                re.compile(r"\b([A-Z][a-z]*'[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*)\b"),
                "handle_apostrophe_names",
                priority=2,
            ),
            # Professional titles after names
            EdgeCasePattern(
                "titles_after_names",
                re.compile(
                    r"\b([A-Z][a-z]+(?: [A-Z][a-z]*)*)\s*,\s*((?:Staff\s+)?(?:Writer|Reporter|Photographer|Correspondent|Editor|Columnist))\b"
                ),
                "handle_titles_after_names",
                priority=2,
            ),
            # Photo credits with "courtesy of"
            EdgeCasePattern(
                "courtesy_credits",
                re.compile(
                    r"\b(?:Photo|Image|Picture)\s+(?:courtesy\s+of\s+|credit:\s*)([A-Z][a-z]+(?: [A-Z][a-z]*)*)\b"
                ),
                "handle_courtesy_credits",
                priority=2,
            ),
            # Names with middle initials
            EdgeCasePattern(
                "middle_initials",
                re.compile(
                    r"\b([A-Z][a-z]+)\s+([A-Z]\.(?:\s*[A-Z]\.)*)\s+([A-Z][a-z]+)\b"
                ),
                "handle_middle_initials",
                priority=1,
            ),
            # Initials only
            EdgeCasePattern(
                "initials_only",
                re.compile(r"\b([A-Z]\.(?:\s*[A-Z]\.)*)\s+([A-Z][a-z]+)\b"),
                "handle_initials_only",
                priority=1,
            ),
        ]

        # Sort by priority (higher priority first)
        self.edge_case_patterns.sort(key=lambda x: x.priority, reverse=True)

    def handle_edge_cases(
        self, text: str, existing_matches: List[ContributorMatch]
    ) -> List[ContributorMatch]:
        """
        Detect and handle edge cases in text.

        Args:
            text: Text to process
            existing_matches: Already found matches

        Returns:
            List of additional matches from edge case handling
        """
        additional_matches = []
        processed_spans = set()

        # Get spans of existing matches to avoid overlap
        existing_spans = {(m.start_pos, m.end_pos) for m in existing_matches}

        for pattern_def in self.edge_case_patterns:
            matches = self._find_pattern_matches(
                text, pattern_def, existing_spans, processed_spans
            )
            additional_matches.extend(matches)

            # Track processed spans
            for match in matches:
                processed_spans.add((match.start_pos, match.end_pos))

        return additional_matches

    def _find_pattern_matches(
        self,
        text: str,
        pattern_def: EdgeCasePattern,
        existing_spans: Set[Tuple[int, int]],
        processed_spans: Set[Tuple[int, int]],
    ) -> List[ContributorMatch]:
        """Find matches for a specific edge case pattern."""
        matches = []

        for match in pattern_def.pattern.finditer(text):
            start_pos = match.start()
            end_pos = match.end()

            # Skip if overlaps with existing matches or already processed
            if self._overlaps_with_spans(
                start_pos, end_pos, existing_spans | processed_spans
            ):
                continue

            # Handle the specific edge case
            handler_method = getattr(self, pattern_def.handler, None)
            if handler_method:
                edge_case_matches = handler_method(match, text)
                matches.extend(edge_case_matches)

        return matches

    def _overlaps_with_spans(
        self, start: int, end: int, spans: Set[Tuple[int, int]]
    ) -> bool:
        """Check if a span overlaps with existing spans."""
        for span_start, span_end in spans:
            # Check for any overlap
            if not (end <= span_start or start >= span_end):
                return True
        return False

    def handle_multiple_authors_and(
        self, match: re.Match, text: str
    ) -> List[ContributorMatch]:
        """Handle multiple authors connected with 'and'."""
        matches = []

        # Extract author names
        author1 = match.group(1).strip()
        author2 = match.group(2).strip()

        # Create matches for each author
        for i, author_name in enumerate([author1, author2], 1):
            if self._is_valid_name(author_name):
                # Find position of this specific author
                author_start = text.find(author_name, match.start())
                author_end = author_start + len(author_name)

                contributor_match = ContributorMatch(
                    text=author_name,
                    start_pos=author_start,
                    end_pos=author_end,
                    role=ContributorRole.AUTHOR,
                    role_confidence=0.9,  # High confidence for multiple author pattern
                    extracted_names=[author_name],
                    context_before=text[max(0, match.start() - 50) : match.start()],
                    context_after=text[match.end() : match.end() + 50],
                    pattern_used="multiple_authors_and",
                    extraction_method="edge_case_handler",
                    extraction_confidence=0.85,
                )

                matches.append(contributor_match)

        return matches

    def handle_multiple_authors_comma(
        self, match: re.Match, text: str
    ) -> List[ContributorMatch]:
        """Handle multiple authors with comma separation."""
        matches = []

        # Extract all author names from groups
        authors = []
        for i in range(1, match.lastindex + 1 if match.lastindex else 1):
            author = match.group(i)
            if author and author.strip():
                authors.append(author.strip())

        # Create matches for each author
        for author_name in authors:
            if self._is_valid_name(author_name):
                # Find position of this specific author
                author_start = text.find(author_name, match.start())
                author_end = author_start + len(author_name)

                contributor_match = ContributorMatch(
                    text=author_name,
                    start_pos=author_start,
                    end_pos=author_end,
                    role=ContributorRole.AUTHOR,
                    role_confidence=0.9,
                    extracted_names=[author_name],
                    context_before=text[max(0, match.start() - 50) : match.start()],
                    context_after=text[match.end() : match.end() + 50],
                    pattern_used="multiple_authors_comma",
                    extraction_method="edge_case_handler",
                    extraction_confidence=0.85,
                )

                matches.append(contributor_match)

        return matches

    def handle_multiple_titles(
        self, match: re.Match, text: str
    ) -> List[ContributorMatch]:
        """Handle names with multiple titles/prefixes."""
        full_text = match.group(0)
        name_part = match.group(1)

        contributor_match = ContributorMatch(
            text=full_text,
            start_pos=match.start(),
            end_pos=match.end(),
            role=ContributorRole.AUTHOR,  # Multiple titles often indicate authors
            role_confidence=0.8,
            extracted_names=[name_part],
            context_before=text[max(0, match.start() - 50) : match.start()],
            context_after=text[match.end() : match.end() + 50],
            pattern_used="multiple_titles",
            extraction_method="edge_case_handler",
            extraction_confidence=0.8,
        )

        return [contributor_match]

    def handle_complex_suffixes(
        self, match: re.Match, text: str
    ) -> List[ContributorMatch]:
        """Handle names with complex suffixes."""
        full_text = match.group(0)
        base_name = match.group(1)

        contributor_match = ContributorMatch(
            text=full_text,
            start_pos=match.start(),
            end_pos=match.end(),
            role=ContributorRole.UNKNOWN,  # Let role classifier determine
            role_confidence=0.5,
            extracted_names=[base_name],
            context_before=text[max(0, match.start() - 50) : match.start()],
            context_after=text[match.end() : match.end() + 50],
            pattern_used="complex_suffixes",
            extraction_method="edge_case_handler",
            extraction_confidence=0.85,
        )

        return [contributor_match]

    def handle_hyphenated_surnames(
        self, match: re.Match, text: str
    ) -> List[ContributorMatch]:
        """Handle hyphenated surnames."""
        full_text = match.group(0)
        first_name = match.group(1)
        last_name = match.group(2)
        full_name = f"{first_name} {last_name}"

        contributor_match = ContributorMatch(
            text=full_text,
            start_pos=match.start(),
            end_pos=match.end(),
            role=ContributorRole.UNKNOWN,
            role_confidence=0.5,
            extracted_names=[full_name],
            context_before=text[max(0, match.start() - 50) : match.start()],
            context_after=text[match.end() : match.end() + 50],
            pattern_used="hyphenated_surnames",
            extraction_method="edge_case_handler",
            extraction_confidence=0.8,
        )

        return [contributor_match]

    def handle_apostrophe_names(
        self, match: re.Match, text: str
    ) -> List[ContributorMatch]:
        """Handle names with apostrophes."""
        full_text = match.group(0)
        name = match.group(1)

        contributor_match = ContributorMatch(
            text=full_text,
            start_pos=match.start(),
            end_pos=match.end(),
            role=ContributorRole.UNKNOWN,
            role_confidence=0.5,
            extracted_names=[name],
            context_before=text[max(0, match.start() - 50) : match.start()],
            context_after=text[match.end() : match.end() + 50],
            pattern_used="apostrophe_names",
            extraction_method="edge_case_handler",
            extraction_confidence=0.8,
        )

        return [contributor_match]

    def handle_titles_after_names(
        self, match: re.Match, text: str
    ) -> List[ContributorMatch]:
        """Handle professional titles after names."""
        full_text = match.group(0)
        name = match.group(1)
        title = match.group(2).lower()

        # Determine role from title
        role = ContributorRole.UNKNOWN
        role_confidence = 0.9

        if any(keyword in title for keyword in ["writer", "author", "columnist"]):
            role = ContributorRole.AUTHOR
        elif "photographer" in title:
            role = ContributorRole.PHOTOGRAPHER
        elif any(keyword in title for keyword in ["correspondent", "reporter"]):
            role = ContributorRole.CORRESPONDENT
        elif "editor" in title:
            role = ContributorRole.EDITOR

        contributor_match = ContributorMatch(
            text=full_text,
            start_pos=match.start(),
            end_pos=match.end(),
            role=role,
            role_confidence=role_confidence,
            extracted_names=[name],
            context_before=text[max(0, match.start() - 50) : match.start()],
            context_after=text[match.end() : match.end() + 50],
            pattern_used="titles_after_names",
            extraction_method="edge_case_handler",
            extraction_confidence=0.9,
        )

        return [contributor_match]

    def handle_courtesy_credits(
        self, match: re.Match, text: str
    ) -> List[ContributorMatch]:
        """Handle photo credits with 'courtesy of'."""
        full_text = match.group(0)
        name = match.group(1)

        contributor_match = ContributorMatch(
            text=full_text,
            start_pos=match.start(),
            end_pos=match.end(),
            role=ContributorRole.PHOTOGRAPHER,  # Usually photographers
            role_confidence=0.8,
            extracted_names=[name],
            context_before=text[max(0, match.start() - 50) : match.start()],
            context_after=text[match.end() : match.end() + 50],
            pattern_used="courtesy_credits",
            extraction_method="edge_case_handler",
            extraction_confidence=0.8,
        )

        return [contributor_match]

    def handle_middle_initials(
        self, match: re.Match, text: str
    ) -> List[ContributorMatch]:
        """Handle names with middle initials."""
        full_text = match.group(0)
        first_name = match.group(1)
        middle_initials = match.group(2)
        last_name = match.group(3)
        full_name = f"{first_name} {middle_initials} {last_name}"

        contributor_match = ContributorMatch(
            text=full_text,
            start_pos=match.start(),
            end_pos=match.end(),
            role=ContributorRole.UNKNOWN,
            role_confidence=0.5,
            extracted_names=[full_name],
            context_before=text[max(0, match.start() - 50) : match.start()],
            context_after=text[match.end() : match.end() + 50],
            pattern_used="middle_initials",
            extraction_method="edge_case_handler",
            extraction_confidence=0.8,
        )

        return [contributor_match]

    def handle_initials_only(
        self, match: re.Match, text: str
    ) -> List[ContributorMatch]:
        """Handle names that are initials only."""
        full_text = match.group(0)
        initials = match.group(1)
        last_name = match.group(2)
        full_name = f"{initials} {last_name}"

        contributor_match = ContributorMatch(
            text=full_text,
            start_pos=match.start(),
            end_pos=match.end(),
            role=ContributorRole.UNKNOWN,
            role_confidence=0.5,
            extracted_names=[full_name],
            context_before=text[max(0, match.start() - 50) : match.start()],
            context_after=text[match.end() : match.end() + 50],
            pattern_used="initials_only",
            extraction_method="edge_case_handler",
            extraction_confidence=0.7,
        )

        return [contributor_match]

    def _is_valid_name(self, name: str) -> bool:
        """Check if extracted text looks like a valid name."""
        if not name or len(name.strip()) < 2:
            return False

        name = name.strip()

        # Must contain at least one letter
        if not any(c.isalpha() for c in name):
            return False

        # Length checks
        if len(name) > 50:
            return False

        # Word count check
        words = name.split()
        if len(words) > 5:
            return False

        # Check for obvious non-names
        non_names = {
            "and",
            "or",
            "the",
            "by",
            "from",
            "with",
            "photo",
            "image",
            "article",
            "story",
            "news",
            "report",
            "staff",
        }

        if name.lower() in non_names:
            return False

        return True

    def optimize_contributor_list(
        self, contributors: List[ExtractedContributor]
    ) -> List[ExtractedContributor]:
        """
        Optimize contributor list by handling edge cases in the final results.
        """
        if not contributors:
            return contributors

        optimized = []
        processed_names = set()

        # Group contributors by similar names
        name_groups = self._group_similar_contributors(contributors)

        for group in name_groups:
            # Select best contributor from each group
            best = max(group, key=lambda c: c.overall_confidence)

            # Apply edge case optimizations
            best = self._optimize_single_contributor(best)

            if best.name.full_name.lower() not in processed_names:
                optimized.append(best)
                processed_names.add(best.name.full_name.lower())

        return optimized

    def _group_similar_contributors(
        self, contributors: List[ExtractedContributor]
    ) -> List[List[ExtractedContributor]]:
        """Group contributors with similar names."""
        groups = []
        processed = set()

        for i, contrib in enumerate(contributors):
            if i in processed:
                continue

            group = [contrib]
            processed.add(i)

            # Find similar contributors
            for j, other_contrib in enumerate(contributors[i + 1 :], i + 1):
                if j in processed:
                    continue

                if self._are_similar_names(contrib.name, other_contrib.name):
                    group.append(other_contrib)
                    processed.add(j)

            groups.append(group)

        return groups

    def _are_similar_names(self, name1: NormalizedName, name2: NormalizedName) -> bool:
        """Check if two normalized names are similar."""
        # Compare full names
        full1 = name1.full_name.lower()
        full2 = name2.full_name.lower()

        if full1 == full2:
            return True

        # Compare last names and first initials
        if (
            name1.last_name.lower() == name2.last_name.lower()
            and name1.first_name
            and name2.first_name
            and name1.first_name[0].lower() == name2.first_name[0].lower()
        ):
            return True

        return False

    def _optimize_single_contributor(
        self, contributor: ExtractedContributor
    ) -> ExtractedContributor:
        """Apply optimizations to a single contributor."""
        # Re-normalize name with edge case handling
        optimized_name = self.name_normalizer.normalize_name(contributor.source_text)

        # Update if optimization improved confidence
        if optimized_name.confidence > contributor.name.confidence:
            contributor.name = optimized_name
            contributor.name_confidence = optimized_name.confidence

        return contributor
