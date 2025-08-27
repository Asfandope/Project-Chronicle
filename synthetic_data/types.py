"""
Type definitions for synthetic test data generation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class SyntheticDataError(Exception):
    """Base exception for synthetic data generation errors."""


class EdgeCaseType(Enum):
    """Types of edge cases to generate."""

    SPLIT_ARTICLES = "split_articles"  # Articles spanning multiple pages
    DECORATIVE_TITLES = "decorative_titles"  # Stylized titles with graphics
    MULTI_COLUMN_COMPLEX = "multi_column_complex"  # Complex column layouts
    OVERLAPPING_ELEMENTS = "overlapping_elements"  # Elements that overlap
    ROTATED_TEXT = "rotated_text"  # Text at angles
    WATERMARKS = "watermarks"  # Background watermarks
    ADVERTISEMENTS = "advertisements"  # Ad placement variations
    CAPTION_AMBIGUITY = "caption_ambiguity"  # Multiple possible caption matches
    CONTRIBUTOR_COMPLEXITY = "contributor_complexity"  # Complex bylines
    MIXED_LANGUAGES = "mixed_languages"  # Multiple languages


class LayoutComplexity(Enum):
    """Layout complexity levels."""

    SIMPLE = "simple"  # Single column, basic layout
    MODERATE = "moderate"  # 2-3 columns, some complexity
    COMPLEX = "complex"  # Multi-column, advanced layouts
    CHAOTIC = "chaotic"  # Intentionally complex for stress testing


class BrandStyle(Enum):
    """Magazine brand styles."""

    TECH = "tech"  # Technology magazine style
    FASHION = "fashion"  # Fashion magazine style
    NEWS = "news"  # News magazine style
    LIFESTYLE = "lifestyle"  # Lifestyle magazine style
    ACADEMIC = "academic"  # Academic journal style
    TABLOID = "tabloid"  # Tabloid newspaper style


@dataclass
class BrandConfiguration:
    """Configuration for a specific magazine brand."""

    # Brand identity
    brand_name: str
    brand_style: BrandStyle

    # Typography
    primary_font: str = "Arial"
    secondary_font: str = "Times New Roman"
    title_font: str = "Helvetica"

    # Layout preferences
    default_columns: int = 2
    column_gap: float = 20.0  # points
    margin_top: float = 72.0  # points (1 inch)
    margin_bottom: float = 72.0
    margin_left: float = 54.0  # points (0.75 inch)
    margin_right: float = 54.0

    # Color scheme
    primary_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # RGB 0-1
    accent_color: Tuple[float, float, float] = (0.2, 0.4, 0.8)
    background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Content preferences
    typical_article_length: Tuple[int, int] = (500, 2000)  # word range
    image_frequency: float = 0.3  # images per 100 words
    pullquote_frequency: float = 0.1  # pullquotes per 100 words

    # Edge case probabilities
    split_article_probability: float = 0.15
    decorative_title_probability: float = 0.25
    complex_layout_probability: float = 0.3

    @classmethod
    def create_tech_magazine(cls) -> "BrandConfiguration":
        """Create configuration for tech magazine."""
        return cls(
            brand_name="TechWeekly",
            brand_style=BrandStyle.TECH,
            primary_font="Arial",
            secondary_font="Helvetica",
            title_font="Impact",
            default_columns=3,
            accent_color=(0.0, 0.4, 0.8),
            typical_article_length=(800, 1500),
            image_frequency=0.4,
            complex_layout_probability=0.4,
        )

    @classmethod
    def create_fashion_magazine(cls) -> "BrandConfiguration":
        """Create configuration for fashion magazine."""
        return cls(
            brand_name="StyleMag",
            brand_style=BrandStyle.FASHION,
            primary_font="Garamond",
            secondary_font="Didot",
            title_font="Futura",
            default_columns=2,
            accent_color=(0.8, 0.2, 0.4),
            typical_article_length=(300, 1000),
            image_frequency=0.6,
            decorative_title_probability=0.4,
        )

    @classmethod
    def create_news_magazine(cls) -> "BrandConfiguration":
        """Create configuration for news magazine."""
        return cls(
            brand_name="NewsToday",
            brand_style=BrandStyle.NEWS,
            primary_font="Times New Roman",
            secondary_font="Arial",
            title_font="Franklin Gothic",
            default_columns=4,
            accent_color=(0.6, 0.0, 0.0),
            typical_article_length=(1000, 3000),
            image_frequency=0.2,
            split_article_probability=0.25,
        )


@dataclass
class GenerationConfig:
    """Configuration for synthetic data generation."""

    # Output settings
    output_directory: Path
    generate_pdfs: bool = True
    generate_ground_truth: bool = True

    # Generation parameters
    documents_per_brand: int = 100
    pages_per_document: Tuple[int, int] = (2, 8)  # min, max pages

    # Content variation
    layout_complexity_distribution: Dict[LayoutComplexity, float] = field(
        default_factory=lambda: {
            LayoutComplexity.SIMPLE: 0.3,
            LayoutComplexity.MODERATE: 0.4,
            LayoutComplexity.COMPLEX: 0.25,
            LayoutComplexity.CHAOTIC: 0.05,
        }
    )

    # Edge case generation
    edge_case_probability: float = 0.3  # Overall edge case probability
    edge_case_distribution: Dict[EdgeCaseType, float] = field(
        default_factory=lambda: {
            EdgeCaseType.SPLIT_ARTICLES: 0.2,
            EdgeCaseType.DECORATIVE_TITLES: 0.15,
            EdgeCaseType.MULTI_COLUMN_COMPLEX: 0.15,
            EdgeCaseType.OVERLAPPING_ELEMENTS: 0.1,
            EdgeCaseType.ROTATED_TEXT: 0.05,
            EdgeCaseType.WATERMARKS: 0.05,
            EdgeCaseType.ADVERTISEMENTS: 0.1,
            EdgeCaseType.CAPTION_AMBIGUITY: 0.1,
            EdgeCaseType.CONTRIBUTOR_COMPLEXITY: 0.08,
            EdgeCaseType.MIXED_LANGUAGES: 0.02,
        }
    )

    # Quality settings
    pdf_dpi: int = 300
    image_quality: float = 0.85
    text_rendering_quality: str = "high"

    # Validation settings
    validate_ground_truth: bool = True
    validate_pdfs: bool = True

    @classmethod
    def create_comprehensive_test(cls, output_dir: Path) -> "GenerationConfig":
        """Create configuration for comprehensive testing."""
        return cls(
            output_directory=output_dir,
            documents_per_brand=150,
            pages_per_document=(1, 12),
            edge_case_probability=0.4,
            layout_complexity_distribution={
                LayoutComplexity.SIMPLE: 0.2,
                LayoutComplexity.MODERATE: 0.3,
                LayoutComplexity.COMPLEX: 0.35,
                LayoutComplexity.CHAOTIC: 0.15,
            },
        )

    @classmethod
    def create_edge_case_focused(cls, output_dir: Path) -> "GenerationConfig":
        """Create configuration focused on edge cases."""
        return cls(
            output_directory=output_dir,
            documents_per_brand=200,
            edge_case_probability=0.8,
            layout_complexity_distribution={
                LayoutComplexity.SIMPLE: 0.1,
                LayoutComplexity.MODERATE: 0.2,
                LayoutComplexity.COMPLEX: 0.4,
                LayoutComplexity.CHAOTIC: 0.3,
            },
        )


@dataclass
class LayoutElement:
    """Base class for layout elements."""

    element_id: str = ""
    element_type: str = ""
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # x0, y0, x1, y1
    page_number: int = 0
    z_order: int = 0  # Stacking order

    # Metadata
    confidence: float = 1.0  # Ground truth confidence
    extraction_difficulty: float = 0.0  # 0=easy, 1=very difficult

    def overlaps(self, other: "LayoutElement") -> bool:
        """Check if this element overlaps with another."""
        x0, y0, x1, y1 = self.bbox
        ox0, oy0, ox1, oy1 = other.bbox

        return not (x1 <= ox0 or ox1 <= x0 or y1 <= oy0 or oy1 <= y0)

    def area(self) -> float:
        """Calculate element area."""
        x0, y0, x1, y1 = self.bbox
        return (x1 - x0) * (y1 - y0)


@dataclass
class TextElement(LayoutElement):
    """Text layout element."""

    text_content: str = ""
    font_family: str = "Arial"
    font_size: float = 12.0
    font_style: str = "normal"  # normal, bold, italic, bold-italic
    text_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    text_align: str = "left"  # left, center, right, justify

    # Semantic information
    semantic_type: str = "paragraph"  # title, heading, paragraph, caption, byline, etc.
    reading_order: int = 0

    def __post_init__(self):
        if not self.element_type:
            self.element_type = "text"


@dataclass
class ImageElement(LayoutElement):
    """Image layout element."""

    image_path: Optional[Path] = None
    image_data: Optional[bytes] = None
    alt_text: str = ""

    # Image properties
    width: int = 0
    height: int = 0
    dpi: int = 300
    color_space: str = "RGB"

    def __post_init__(self):
        if not self.element_type:
            self.element_type = "image"


@dataclass
class ArticleData:
    """Complete article data with ground truth."""

    article_id: str
    title: str
    contributors: List[Dict[str, Any]] = field(default_factory=list)

    # Content elements
    text_elements: List[TextElement] = field(default_factory=list)
    image_elements: List[ImageElement] = field(default_factory=list)

    # Article metadata
    page_range: Tuple[int, int] = (1, 1)
    is_split_article: bool = False
    continuation_pages: List[int] = field(default_factory=list)

    # Classification
    article_type: str = "feature"  # feature, news, editorial, review, etc.
    complexity_level: LayoutComplexity = LayoutComplexity.SIMPLE
    edge_cases: List[EdgeCaseType] = field(default_factory=list)


@dataclass
class GroundTruthData:
    """Ground truth data for generated document."""

    document_id: str
    brand_name: str
    generation_timestamp: datetime

    # Document structure
    articles: List[ArticleData] = field(default_factory=list)
    all_text_elements: List[TextElement] = field(default_factory=list)
    all_image_elements: List[ImageElement] = field(default_factory=list)

    # Page information
    page_count: int = 1
    page_dimensions: Tuple[float, float] = (612.0, 792.0)  # US Letter

    # Generation metadata
    generation_config: Optional[GenerationConfig] = None
    brand_config: Optional[BrandConfiguration] = None

    # Quality metrics for testing
    expected_extraction_accuracy: float = 1.0
    difficult_elements_count: int = 0
    edge_cases_present: List[EdgeCaseType] = field(default_factory=list)

    def to_xml(self) -> str:
        """Convert ground truth to XML format."""
        # This would use the XML output system we built earlier

    def validate(self) -> List[str]:
        """Validate ground truth data integrity."""
        errors = []

        if not self.document_id:
            errors.append("Missing document ID")

        if not self.articles:
            errors.append("No articles in document")

        # Check for overlapping article page ranges
        page_ranges = [article.page_range for article in self.articles]
        for i, range1 in enumerate(page_ranges):
            for j, range2 in enumerate(page_ranges[i + 1 :], i + 1):
                if self._ranges_overlap(range1, range2):
                    errors.append(f"Articles {i} and {j} have overlapping page ranges")

        # Validate element consistency
        total_text_elements = sum(
            len(article.text_elements) for article in self.articles
        )
        if total_text_elements != len(self.all_text_elements):
            errors.append(
                "Text element count mismatch between articles and global list"
            )

        return errors

    def _ranges_overlap(self, range1: Tuple[int, int], range2: Tuple[int, int]) -> bool:
        """Check if two page ranges overlap."""
        return not (range1[1] < range2[0] or range2[1] < range1[0])


@dataclass
class GeneratedDocument:
    """Complete generated document with PDF and ground truth."""

    document_id: str
    brand_name: str

    # Generated files
    pdf_path: Optional[Path] = None
    ground_truth_path: Optional[Path] = None

    # Document data
    ground_truth: Optional[GroundTruthData] = None

    # Generation metadata
    generation_time: float = 0.0
    generation_timestamp: datetime = field(default_factory=datetime.now)

    # Quality indicators
    generation_successful: bool = False
    validation_passed: bool = False
    validation_errors: List[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """Check if document generation was complete."""
        return (
            self.generation_successful
            and self.pdf_path
            and self.pdf_path.exists()
            and self.ground_truth_path
            and self.ground_truth_path.exists()
        )


@dataclass
class TestSuite:
    """Complete test suite with multiple documents."""

    suite_name: str
    generation_config: GenerationConfig

    # Generated documents by brand
    documents_by_brand: Dict[str, List[GeneratedDocument]] = field(default_factory=dict)

    # Suite metadata
    total_documents: int = 0
    successful_generations: int = 0
    generation_start_time: datetime = field(default_factory=datetime.now)
    generation_end_time: Optional[datetime] = None

    # Quality metrics
    complexity_distribution: Dict[LayoutComplexity, int] = field(default_factory=dict)
    edge_case_distribution: Dict[EdgeCaseType, int] = field(default_factory=dict)

    def add_document(self, document: GeneratedDocument) -> None:
        """Add a document to the test suite."""
        if document.brand_name not in self.documents_by_brand:
            self.documents_by_brand[document.brand_name] = []

        self.documents_by_brand[document.brand_name].append(document)
        self.total_documents += 1

        if document.generation_successful:
            self.successful_generations += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get test suite summary statistics."""
        return {
            "suite_name": self.suite_name,
            "total_documents": self.total_documents,
            "successful_generations": self.successful_generations,
            "success_rate": self.successful_generations / max(1, self.total_documents),
            "brands": list(self.documents_by_brand.keys()),
            "documents_per_brand": {
                brand: len(docs) for brand, docs in self.documents_by_brand.items()
            },
            "complexity_distribution": dict(self.complexity_distribution),
            "edge_case_distribution": {
                ec.value: count for ec, count in self.edge_case_distribution.items()
            },
            "generation_duration": (
                (self.generation_end_time - self.generation_start_time).total_seconds()
                if self.generation_end_time
                else None
            ),
        }
