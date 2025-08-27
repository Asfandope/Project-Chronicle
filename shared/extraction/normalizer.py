"""
Name normalization for extracted contributors.
"""

import re
from typing import Dict, List, Optional

from .types import ExtractionConfig, NormalizedName


class NameNormalizer:
    """Normalizes contributor names to standardized format."""

    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.config = config or ExtractionConfig()
        self._initialize_patterns()
        self._initialize_name_data()

    def _initialize_patterns(self) -> None:
        """Initialize name parsing patterns."""

        # Title/prefix patterns
        self.prefix_patterns = [
            r"\b(?:Dr|Doctor)\.?\b",
            r"\b(?:Prof|Professor)\.?\b",
            r"\b(?:Mr|Mister)\.?\b",
            r"\b(?:Mrs|Misses)\.?\b",
            r"\b(?:Ms|Miss)\.?\b",
            r"\b(?:Rev|Reverend)\.?\b",
            r"\b(?:Hon|Honorable)\.?\b",
            r"\b(?:Sen|Senator)\.?\b",
            r"\b(?:Rep|Representative)\.?\b",
            r"\b(?:Judge|Justice)\.?\b",
            r"\b(?:Chief|Capt|Captain|Col|Colonel|Major|Lt|Lieutenant)\.?\b",
        ]

        # Suffix patterns
        self.suffix_patterns = [
            r"\b(?:Jr|Junior)\.?\b",
            r"\b(?:Sr|Senior)\.?\b",
            r"\b(?:II|III|IV|V|VI|VII|VIII|IX|X)\b",
            r"\b(?:2nd|3rd|4th|5th|6th|7th|8th|9th)\b",
            r"\b(?:Ph\.?D|PhD)\.?\b",
            r"\b(?:M\.?D|MD)\.?\b",
            r"\b(?:J\.?D|JD)\.?\b",
            r"\b(?:M\.?A|MA)\.?\b",
            r"\b(?:B\.?A|BA)\.?\b",
            r"\b(?:M\.?S|MS)\.?\b",
            r"\b(?:B\.?S|BS)\.?\b",
            r"\b(?:M\.?B\.?A|MBA)\.?\b",
            r"\b(?:Esq|Esquire)\.?\b",
            r"\b(?:CPA|RN|PE)\.?\b",
        ]

        # Compile patterns for efficiency
        self.prefix_regex = re.compile("|".join(self.prefix_patterns), re.IGNORECASE)
        self.suffix_regex = re.compile("|".join(self.suffix_patterns), re.IGNORECASE)

        # Name component patterns
        self.initial_pattern = re.compile(r"\b[A-Z]\.?\b")
        self.hyphenated_name_pattern = re.compile(r"\b\w+(?:-\w+)+\b")
        self.apostrophe_name_pattern = re.compile(r"\b\w+'\w+\b")

        # Special parsing patterns
        self.last_first_pattern = re.compile(r"^([^,]+),\s*([^,]+)(?:,\s*(.+))?$")
        self.parenthetical_pattern = re.compile(r"\([^)]*\)")

    def _initialize_name_data(self) -> None:
        """Initialize name parsing data and lookup tables."""

        # Common prefixes that should be preserved
        self.known_prefixes = {
            "dr",
            "doctor",
            "prof",
            "professor",
            "mr",
            "mister",
            "mrs",
            "misses",
            "ms",
            "miss",
            "rev",
            "reverend",
            "hon",
            "honorable",
            "sen",
            "senator",
            "rep",
            "representative",
            "judge",
            "justice",
            "chief",
            "capt",
            "captain",
            "col",
            "colonel",
            "major",
            "lt",
            "lieutenant",
        }

        # Common suffixes that should be preserved
        self.known_suffixes = {
            "jr",
            "junior",
            "sr",
            "senior",
            "ii",
            "iii",
            "iv",
            "v",
            "vi",
            "vii",
            "viii",
            "ix",
            "x",
            "2nd",
            "3rd",
            "4th",
            "5th",
            "6th",
            "7th",
            "8th",
            "9th",
            "phd",
            "ph.d",
            "md",
            "m.d",
            "jd",
            "j.d",
            "ma",
            "m.a",
            "ba",
            "b.a",
            "ms",
            "m.s",
            "bs",
            "b.s",
            "mba",
            "m.b.a",
            "esq",
            "esquire",
            "cpa",
            "rn",
            "pe",
        }

        # Nickname mappings for expansion
        if self.config.expand_nicknames:
            self.nickname_mappings = {
                "bob": "robert",
                "rob": "robert",
                "bobby": "robert",
                "bill": "william",
                "will": "william",
                "billy": "william",
                "dick": "richard",
                "rick": "richard",
                "rich": "richard",
                "jim": "james",
                "jimmy": "james",
                "mike": "michael",
                "mick": "michael",
                "mickey": "michael",
                "tom": "thomas",
                "tommy": "thomas",
                "dave": "david",
                "davy": "david",
                "joe": "joseph",
                "joey": "joseph",
                "dan": "daniel",
                "danny": "daniel",
                "sam": "samuel",
                "sammy": "samuel",
                "ben": "benjamin",
                "benny": "benjamin",
                "matt": "matthew",
                "matty": "matthew",
                "chris": "christopher",
                "christy": "christopher",
                "steve": "stephen",
                "stevie": "stephen",
                "tony": "anthony",
                "beth": "elizabeth",
                "liz": "elizabeth",
                "betty": "elizabeth",
                "sue": "susan",
                "susie": "susan",
                "suzy": "susan",
                "kate": "katherine",
                "kathy": "katherine",
                "katie": "katherine",
                "jen": "jennifer",
                "jenny": "jennifer",
                "meg": "margaret",
                "maggie": "margaret",
                "peggy": "margaret",
            }
        else:
            self.nickname_mappings = {}

    def normalize_name(self, name_text: str) -> NormalizedName:
        """
        Normalize a name string to standardized format.

        Args:
            name_text: Raw name string to normalize

        Returns:
            NormalizedName object with parsed components
        """
        if not name_text or not name_text.strip():
            return NormalizedName(original_text=name_text)

        # Clean and prepare the name
        cleaned_name = self._clean_name_text(name_text)

        # Try different parsing strategies
        normalized = self._parse_name(cleaned_name)
        normalized.original_text = name_text

        # Apply post-processing
        normalized = self._post_process_name(normalized)

        # Calculate confidence
        normalized.confidence = self._calculate_confidence(normalized, name_text)

        return normalized

    def _clean_name_text(self, name_text: str) -> str:
        """Clean and prepare name text for parsing."""
        # Remove extra whitespace
        cleaned = re.sub(r"\s+", " ", name_text.strip())

        # Remove parenthetical content (often job titles)
        cleaned = self.parenthetical_pattern.sub("", cleaned)

        # Remove common prefixes like "By " at the start
        cleaned = re.sub(
            r"^(?:by|from|written by|photo by)\s+", "", cleaned, flags=re.IGNORECASE
        )

        # Remove quotes around names
        cleaned = re.sub(r'^["\'](.+)["\']$', r"\1", cleaned)

        return cleaned.strip()

    def _parse_name(self, name_text: str) -> NormalizedName:
        """Parse a cleaned name text into components."""

        # First, try to parse if it's already in "Last, First" format
        last_first_match = self.last_first_pattern.match(name_text)
        if last_first_match:
            return self._parse_last_first_format(last_first_match)

        # Otherwise, parse as natural order name
        return self._parse_natural_order(name_text)

    def _parse_last_first_format(self, match: re.Match) -> NormalizedName:
        """Parse name already in 'Last, First [Middle/Suffix]' format."""
        last_part = match.group(1).strip()
        first_part = match.group(2).strip()
        additional_part = match.group(3).strip() if match.group(3) else ""

        normalized = NormalizedName(normalization_method="last_first_parsing")

        # Parse last name
        normalized.last_name = last_part

        # Parse first name and additional components
        first_tokens = first_part.split()
        if first_tokens:
            # Check if first token is a prefix
            if self._is_prefix(first_tokens[0]):
                normalized.prefixes.append(first_tokens[0])
                first_tokens = first_tokens[1:]

            # First remaining token is the first name
            if first_tokens:
                normalized.first_name = first_tokens[0]

                # Remaining tokens could be middle names or suffixes
                for token in first_tokens[1:]:
                    if self._is_suffix(token):
                        normalized.suffixes.append(token)
                    else:
                        normalized.middle_names.append(token)

        # Process additional part (usually suffixes)
        if additional_part:
            additional_tokens = additional_part.split()
            for token in additional_tokens:
                if self._is_suffix(token):
                    normalized.suffixes.append(token)
                elif self._is_prefix(token):
                    normalized.prefixes.append(token)
                else:
                    normalized.middle_names.append(token)

        return normalized

    def _parse_natural_order(self, name_text: str) -> NormalizedName:
        """Parse name in natural order (First Middle Last)."""
        tokens = name_text.split()
        if not tokens:
            return NormalizedName(normalization_method="natural_order_parsing")

        normalized = NormalizedName(normalization_method="natural_order_parsing")

        # Process tokens from left to right
        i = 0

        # Extract prefixes at the beginning
        while i < len(tokens) and self._is_prefix(tokens[i]):
            normalized.prefixes.append(tokens[i])
            i += 1

        # Must have at least one remaining token for a name
        if i >= len(tokens):
            return normalized

        # Extract suffixes at the end
        j = len(tokens) - 1
        while j > i and self._is_suffix(tokens[j]):
            normalized.suffixes.insert(0, tokens[j])
            j -= 1

        # Process remaining tokens as names
        name_tokens = tokens[i : j + 1]

        if len(name_tokens) == 1:
            # Single name - treat as first name
            normalized.first_name = name_tokens[0]
        elif len(name_tokens) == 2:
            # Two names - first and last
            normalized.first_name = name_tokens[0]
            normalized.last_name = name_tokens[1]
        else:
            # Multiple names - first, middle(s), last
            normalized.first_name = name_tokens[0]
            normalized.last_name = name_tokens[-1]
            normalized.middle_names = name_tokens[1:-1]

        return normalized

    def _is_prefix(self, token: str) -> bool:
        """Check if a token is a name prefix."""
        clean_token = re.sub(r"[.,]", "", token.lower())
        return clean_token in self.known_prefixes

    def _is_suffix(self, token: str) -> bool:
        """Check if a token is a name suffix."""
        clean_token = re.sub(r"[.,]", "", token.lower())
        return clean_token in self.known_suffixes

    def _post_process_name(self, normalized: NormalizedName) -> NormalizedName:
        """Apply post-processing to normalized name."""

        # Expand nicknames if configured
        if self.config.expand_nicknames and normalized.first_name:
            first_lower = normalized.first_name.lower()
            if first_lower in self.nickname_mappings:
                normalized.first_name = self.nickname_mappings[first_lower].title()

        # Handle initials if configured
        if self.config.handle_initials:
            normalized = self._process_initials(normalized)

        # Capitalize names properly
        normalized = self._capitalize_names(normalized)

        return normalized

    def _process_initials(self, normalized: NormalizedName) -> NormalizedName:
        """Process initials in names."""

        # Convert single letters to proper initials
        if normalized.first_name and len(normalized.first_name) == 1:
            normalized.first_name = f"{normalized.first_name.upper()}."

        # Process middle names/initials
        processed_middle = []
        for middle in normalized.middle_names:
            if len(middle) == 1 or (len(middle) == 2 and middle.endswith(".")):
                # Single letter or already formatted initial
                initial = middle[0].upper() + "."
                processed_middle.append(initial)
            else:
                processed_middle.append(middle)

        normalized.middle_names = processed_middle
        return normalized

    def _capitalize_names(self, normalized: NormalizedName) -> NormalizedName:
        """Apply proper capitalization to name components."""

        def capitalize_name_part(name: str) -> str:
            """Capitalize a name part handling special cases."""
            if not name:
                return name

            # Handle hyphenated names
            if "-" in name:
                parts = name.split("-")
                return "-".join(part.capitalize() for part in parts)

            # Handle apostrophe names (O'Connor, D'Angelo)
            if "'" in name:
                parts = name.split("'")
                return "'".join(part.capitalize() for part in parts)

            # Handle Scottish/Irish prefixes
            if name.lower().startswith(("mc", "mac")):
                if len(name) > 2:
                    return name[:2].capitalize() + name[2:].capitalize()

            return name.capitalize()

        # Apply capitalization
        if normalized.first_name:
            normalized.first_name = capitalize_name_part(normalized.first_name)

        if normalized.last_name:
            normalized.last_name = capitalize_name_part(normalized.last_name)

        normalized.middle_names = [
            capitalize_name_part(name) for name in normalized.middle_names
        ]

        # Don't change capitalization of prefixes/suffixes as they have standard forms

        return normalized

    def _calculate_confidence(self, normalized: NormalizedName, original: str) -> float:
        """Calculate confidence score for normalization."""
        confidence = 1.0

        # Reduce confidence for incomplete names if required
        if self.config.require_complete_names and not normalized.is_complete:
            confidence *= 0.7

        # Reduce confidence for very short names
        if len(original.strip()) < 3:
            confidence *= 0.6

        # Reduce confidence if no clear name components were extracted
        if not normalized.first_name and not normalized.last_name:
            confidence *= 0.3

        # Boost confidence for well-structured names
        if normalized.is_complete and (normalized.prefixes or normalized.suffixes):
            confidence = min(1.0, confidence * 1.1)

        return confidence

    def normalize_multiple(self, name_texts: List[str]) -> List[NormalizedName]:
        """Normalize multiple names efficiently."""
        return [self.normalize_name(name) for name in name_texts]

    def get_normalization_statistics(self, names: List[str]) -> Dict[str, any]:
        """Get statistics about name normalization."""
        if not names:
            return {"total_names": 0}

        normalized_names = self.normalize_multiple(names)

        complete_names = sum(1 for name in normalized_names if name.is_complete)
        high_confidence = sum(1 for name in normalized_names if name.confidence >= 0.8)
        with_prefixes = sum(1 for name in normalized_names if name.prefixes)
        with_suffixes = sum(1 for name in normalized_names if name.suffixes)
        with_middle = sum(1 for name in normalized_names if name.middle_names)

        avg_confidence = sum(name.confidence for name in normalized_names) / len(
            normalized_names
        )

        return {
            "total_names": len(names),
            "complete_names": complete_names,
            "completion_rate": complete_names / len(names),
            "high_confidence_count": high_confidence,
            "high_confidence_rate": high_confidence / len(names),
            "average_confidence": avg_confidence,
            "names_with_prefixes": with_prefixes,
            "names_with_suffixes": with_suffixes,
            "names_with_middle_names": with_middle,
            "prefix_rate": with_prefixes / len(names),
            "suffix_rate": with_suffixes / len(names),
        }
