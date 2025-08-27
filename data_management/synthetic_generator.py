"""
Enhanced synthetic data generator for gold standard dataset creation.

Creates realistic magazine test data with XML ground truth that integrates
with the gold standard dataset infrastructure and validation system.
"""

import json
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple
from xml.dom import minidom

import structlog

# Try to import synthetic data modules
try:
    pass

    synthetic_data_available = True
except ImportError:
    synthetic_data_available = False

from .ingestion import DataIngestionManager
from .schema_validator import DatasetValidator

logger = structlog.get_logger(__name__)


@dataclass
class SyntheticArticle:
    """Represents a synthetic article for testing."""

    article_id: str
    title: str
    body_paragraphs: List[str]
    contributors: List[Dict[str, str]]
    start_page: int
    end_page: int
    layout_complexity: str
    confidence_scores: Dict[str, float]

    def to_xml_element(self) -> ET.Element:
        """Convert to XML element for ground truth."""
        article_elem = ET.Element("article")
        article_elem.set("id", self.article_id)
        article_elem.set("start_page", str(self.start_page))
        article_elem.set("end_page", str(self.end_page))

        # Add title
        title_elem = ET.SubElement(article_elem, "title")
        title_elem.text = self.title
        title_elem.set("confidence", str(self.confidence_scores.get("title", 0.95)))

        # Add body paragraphs
        for i, paragraph in enumerate(self.body_paragraphs):
            body_elem = ET.SubElement(article_elem, "body")
            body_elem.text = paragraph
            body_elem.set("confidence", str(self.confidence_scores.get("body", 0.92)))
            body_elem.set("paragraph_index", str(i))

        # Add contributors
        if self.contributors:
            contributors_elem = ET.SubElement(article_elem, "contributors")
            for contrib in self.contributors:
                contrib_elem = ET.SubElement(contributors_elem, "contributor")
                contrib_elem.set("name", contrib.get("name", "Unknown"))
                contrib_elem.set("role", contrib.get("role", "author"))
                contrib_elem.set("confidence", str(contrib.get("confidence", 0.88)))

        return article_elem


@dataclass
class SyntheticDocument:
    """Represents a complete synthetic magazine document."""

    document_id: str
    brand: str
    issue_date: str
    page_count: int
    articles: List[SyntheticArticle]
    metadata: Dict[str, Any]

    def to_xml_ground_truth(self) -> str:
        """Convert to XML ground truth format."""
        # Create root element
        root = ET.Element("magazine")
        root.set("brand", self.brand)
        root.set("issue_date", self.issue_date)
        root.set("total_pages", str(self.page_count))
        root.set("document_id", self.document_id)

        # Add articles
        for article in self.articles:
            root.append(article.to_xml_element())

        # Format XML with pretty printing
        rough_string = ET.tostring(root, "utf-8")
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")


class GoldStandardSyntheticGenerator:
    """Generates synthetic test data for gold standard datasets."""

    def __init__(self, data_root: Path = None):
        """
        Initialize synthetic generator.

        Args:
            data_root: Root directory for gold standard datasets
        """
        self.data_root = data_root or Path("data/gold_sets")
        self.ingestion_manager = DataIngestionManager(self.data_root)
        self.validator = DatasetValidator(self.data_root)
        self.logger = logger.bind(component="SyntheticGenerator")

        # Load brand configurations
        self.brand_configs = self._load_brand_configs()

        # Content templates for realistic generation
        self.content_templates = self._initialize_content_templates()

        self.logger.info("Initialized synthetic data generator")

    def _load_brand_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load brand configurations from config files."""
        configs = {}
        config_path = Path("configs/brands")

        if config_path.exists():
            for brand_file in config_path.glob("*.yaml"):
                try:
                    import yaml

                    with open(brand_file, "r") as f:
                        config = yaml.safe_load(f)
                        brand_name = brand_file.stem
                        configs[brand_name] = config
                        self.logger.debug("Loaded brand config", brand=brand_name)
                except Exception as e:
                    self.logger.warning(
                        "Failed to load brand config",
                        file=str(brand_file),
                        error=str(e),
                    )

        # Add default configs if none found
        if not configs:
            configs = self._create_default_brand_configs()

        return configs

    def _create_default_brand_configs(self) -> Dict[str, Dict[str, Any]]:
        """Create default brand configurations."""
        return {
            "economist": {
                "brand": "economist",
                "title_patterns": [
                    "Economic Analysis",
                    "Market Report",
                    "Global Update",
                ],
                "content_themes": [
                    "economics",
                    "politics",
                    "business",
                    "international",
                ],
                "layout_hints": {"column_count": [2, 3]},
                "confidence_overrides": {"title": 0.95, "body": 0.92},
            },
            "time": {
                "brand": "time",
                "title_patterns": ["Breaking News", "Feature Story", "Cover Story"],
                "content_themes": ["news", "politics", "culture", "technology"],
                "layout_hints": {"column_count": [2, 3]},
                "confidence_overrides": {"title": 0.93, "body": 0.90},
            },
            "newsweek": {
                "brand": "newsweek",
                "title_patterns": ["Weekly Report", "News Analysis", "Special Report"],
                "content_themes": ["current_events", "politics", "society", "health"],
                "layout_hints": {"column_count": [2, 3]},
                "confidence_overrides": {"title": 0.92, "body": 0.89},
            },
            "vogue": {
                "brand": "vogue",
                "title_patterns": [
                    "Fashion Forward",
                    "Style Guide",
                    "Designer Profile",
                ],
                "content_themes": ["fashion", "beauty", "lifestyle", "culture"],
                "layout_hints": {"column_count": [1, 2, 3]},
                "confidence_overrides": {"title": 0.94, "body": 0.91},
            },
        }

    def _initialize_content_templates(self) -> Dict[str, List[str]]:
        """Initialize content templates for realistic text generation."""
        return {
            "economics": [
                "The global economy continues to show signs of recovery following recent market volatility.",
                "Central banks worldwide are reassessing their monetary policy strategies.",
                "Inflation rates have stabilized in most developed economies, though concerns remain.",
                "Supply chain disruptions have created new challenges for international trade.",
                "Emerging markets are experiencing unprecedented growth opportunities.",
            ],
            "politics": [
                "Political leaders gathered today to discuss the implications of recent policy changes.",
                "Voter sentiment has shifted significantly in key demographic groups.",
                "Congressional debates continue over proposed legislative reforms.",
                "International diplomatic relations face new challenges and opportunities.",
                "Campaign strategies are evolving in response to changing public opinion.",
            ],
            "technology": [
                "Artificial intelligence applications are transforming industry practices.",
                "Cybersecurity concerns have prompted new regulatory frameworks.",
                "Digital transformation initiatives are accelerating across sectors.",
                "Cloud computing adoption continues to reshape enterprise operations.",
                "Innovation in mobile technologies opens new market possibilities.",
            ],
            "fashion": [
                "This season's trends reflect a return to classic silhouettes with modern twists.",
                "Sustainable fashion continues to gain momentum among conscious consumers.",
                "Designer collaborations are creating exciting new aesthetic directions.",
                "Street style influences are increasingly visible on runway collections.",
                "Luxury brands are reimagining their approach to digital engagement.",
            ],
        }

    def generate_brand_dataset(
        self,
        brand: str,
        num_documents: int = 10,
        articles_per_document: Tuple[int, int] = (3, 8),
        pages_per_document: Tuple[int, int] = (10, 25),
    ) -> Dict[str, Any]:
        """
        Generate a complete synthetic dataset for a brand.

        Args:
            brand: Brand name
            num_documents: Number of documents to generate
            articles_per_document: Range of articles per document (min, max)
            pages_per_document: Range of pages per document (min, max)

        Returns:
            Generation report with statistics and file paths
        """
        start_time = datetime.now()

        self.logger.info(
            "Starting synthetic dataset generation",
            brand=brand,
            num_documents=num_documents,
        )

        if brand not in self.brand_configs:
            raise ValueError(f"Unknown brand: {brand}")

        brand_config = self.brand_configs[brand]
        generated_files = []
        errors = []

        # Generate documents
        for i in range(num_documents):
            try:
                document = self._generate_single_document(
                    brand,
                    brand_config,
                    random.randint(*articles_per_document),
                    random.randint(*pages_per_document),
                    document_index=i,
                )

                # Save document files
                files = self._save_document_files(document)
                generated_files.extend(files)

                self.logger.debug(
                    "Generated document",
                    brand=brand,
                    document_id=document.document_id,
                    articles=len(document.articles),
                )

            except Exception as e:
                error_msg = f"Failed to generate document {i}: {str(e)}"
                errors.append(error_msg)
                self.logger.error(
                    "Document generation failed", document_index=i, error=str(e)
                )

        generation_time = datetime.now() - start_time

        # Validate generated dataset
        validation_report = self.validator.validate_brand_dataset(brand)

        report = {
            "brand": brand,
            "generation_timestamp": start_time.isoformat(),
            "generation_time_seconds": generation_time.total_seconds(),
            "requested_documents": num_documents,
            "generated_documents": len(generated_files)
            // 2,  # Each doc creates XML + metadata
            "generated_files": len(generated_files),
            "errors": errors,
            "validation_report": {
                "total_files": validation_report.total_files,
                "valid_files": validation_report.valid_files,
                "validation_rate": validation_report.validation_rate,
                "average_quality_score": validation_report.average_quality_score,
            },
            "files_generated": [str(f) for f in generated_files],
        }

        self.logger.info(
            "Synthetic dataset generation completed",
            brand=brand,
            generated_documents=report["generated_documents"],
            generation_time=generation_time.total_seconds(),
            validation_rate=validation_report.validation_rate,
        )

        return report

    def _generate_single_document(
        self,
        brand: str,
        brand_config: Dict[str, Any],
        num_articles: int,
        num_pages: int,
        document_index: int = 0,
    ) -> SyntheticDocument:
        """Generate a single synthetic document."""

        # Create document metadata
        document_id = (
            f"{brand}_{datetime.now().strftime('%Y%m%d')}_{document_index:03d}"
        )
        issue_date = (datetime.now() - timedelta(days=random.randint(0, 365))).strftime(
            "%Y-%m-%d"
        )

        # Generate articles
        articles = []
        current_page = 1

        for article_idx in range(num_articles):
            article = self._generate_article(
                brand, brand_config, article_idx, current_page, max_pages=num_pages
            )
            articles.append(article)
            current_page = article.end_page + 1

            # Don't exceed total pages
            if current_page > num_pages:
                break

        return SyntheticDocument(
            document_id=document_id,
            brand=brand,
            issue_date=issue_date,
            page_count=num_pages,
            articles=articles,
            metadata={
                "generated_timestamp": datetime.now().isoformat(),
                "generation_method": "synthetic",
                "layout_complexity": "standard",
                "content_themes": brand_config.get("content_themes", []),
            },
        )

    def _generate_article(
        self,
        brand: str,
        brand_config: Dict[str, Any],
        article_index: int,
        start_page: int,
        max_pages: int,
    ) -> SyntheticArticle:
        """Generate a single synthetic article."""

        # Generate article length (1-3 pages typically)
        article_pages = min(random.randint(1, 3), max_pages - start_page + 1)
        end_page = start_page + article_pages - 1

        # Generate title
        title_patterns = brand_config.get("title_patterns", ["Article"])
        base_title = random.choice(title_patterns)
        title = f"{base_title}: {self._generate_title_variation()}"

        # Generate body paragraphs (2-6 paragraphs per page)
        paragraphs_per_page = random.randint(2, 6)
        total_paragraphs = paragraphs_per_page * article_pages

        # Select content theme
        themes = brand_config.get("content_themes", ["general"])
        theme = random.choice(themes)

        body_paragraphs = []
        for i in range(total_paragraphs):
            paragraph = self._generate_paragraph(theme, length=random.randint(3, 8))
            body_paragraphs.append(paragraph)

        # Generate contributors (0-2 contributors typically)
        contributors = []
        if random.random() < 0.8:  # 80% chance of having contributors
            num_contributors = random.randint(1, 2)
            for i in range(num_contributors):
                contributor = {
                    "name": self._generate_contributor_name(),
                    "role": random.choice(["author", "correspondent", "editor"]),
                    "confidence": random.uniform(0.85, 0.95),
                }
                contributors.append(contributor)

        # Get confidence overrides from brand config
        confidence_overrides = brand_config.get("confidence_overrides", {})
        confidence_scores = {
            "title": confidence_overrides.get("title", 0.93),
            "body": confidence_overrides.get("body", 0.90),
            "contributors": confidence_overrides.get("contributors", 0.87),
        }

        return SyntheticArticle(
            article_id=f"article_{article_index:03d}",
            title=title,
            body_paragraphs=body_paragraphs,
            contributors=contributors,
            start_page=start_page,
            end_page=end_page,
            layout_complexity="standard",
            confidence_scores=confidence_scores,
        )

    def _generate_title_variation(self) -> str:
        """Generate varied title endings."""
        variations = [
            "Market Trends and Analysis",
            "Latest Developments",
            "Key Insights and Perspectives",
            "Strategic Overview",
            "Current Situation Report",
            "Expert Analysis",
            "Comprehensive Review",
            "Future Outlook",
            "Impact Assessment",
            "Industry Update",
        ]
        return random.choice(variations)

    def _generate_paragraph(self, theme: str, length: int = 5) -> str:
        """Generate a paragraph with realistic content."""

        # Get template sentences for the theme
        templates = self.content_templates.get(
            theme, self.content_templates["economics"]
        )

        # Generate sentences
        sentences = []
        for _ in range(length):
            base_sentence = random.choice(templates)

            # Add some variation
            variations = [
                base_sentence,
                f"Furthermore, {base_sentence.lower()}",
                f"However, {base_sentence.lower()}",
                f"According to experts, {base_sentence.lower()}",
                f"Recent studies suggest that {base_sentence.lower()}",
            ]

            sentence = random.choice(variations)
            sentences.append(sentence)

        return " ".join(sentences)

    def _generate_contributor_name(self) -> str:
        """Generate realistic contributor names."""
        first_names = [
            "Sarah",
            "Michael",
            "Jennifer",
            "David",
            "Emily",
            "Robert",
            "Lisa",
            "James",
            "Maria",
            "John",
            "Anna",
            "William",
            "Jessica",
            "Thomas",
            "Michelle",
            "Christopher",
            "Amanda",
            "Daniel",
        ]

        last_names = [
            "Johnson",
            "Smith",
            "Williams",
            "Brown",
            "Jones",
            "Garcia",
            "Miller",
            "Davis",
            "Rodriguez",
            "Martinez",
            "Hernandez",
            "Lopez",
            "Gonzalez",
            "Wilson",
            "Anderson",
            "Thomas",
            "Taylor",
            "Moore",
        ]

        return f"{random.choice(first_names)} {random.choice(last_names)}"

    def _save_document_files(self, document: SyntheticDocument) -> List[Path]:
        """
        Save document as XML ground truth and metadata files.

        Args:
            document: SyntheticDocument to save

        Returns:
            List of created file paths
        """
        brand_path = self.data_root / document.brand

        # Ensure directories exist
        (brand_path / "ground_truth").mkdir(parents=True, exist_ok=True)
        (brand_path / "metadata").mkdir(parents=True, exist_ok=True)

        created_files = []

        # Save XML ground truth
        xml_content = document.to_xml_ground_truth()
        xml_path = brand_path / "ground_truth" / f"{document.document_id}.xml"

        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(xml_content)
        created_files.append(xml_path)

        # Save metadata
        metadata = {
            "dataset_info": {
                "brand": document.brand,
                "filename": f"{document.document_id}.xml",
                "creation_date": datetime.now().isoformat(),
                "file_type": "synthetic_ground_truth",
            },
            "quality_metrics": {
                "manual_validation": False,
                "annotation_quality": 0.95,  # High for synthetic data
                "completeness_score": 0.98,
            },
            "content_info": {
                "page_count": document.page_count,
                "article_count": len(document.articles),
                "layout_complexity": "standard",
            },
            "synthetic_metadata": {
                "generation_method": "gold_standard_synthetic",
                "content_themes": document.metadata.get("content_themes", []),
                "document_id": document.document_id,
                "article_details": [
                    {
                        "article_id": article.article_id,
                        "title": article.title,
                        "page_range": [article.start_page, article.end_page],
                        "paragraph_count": len(article.body_paragraphs),
                        "contributor_count": len(article.contributors),
                    }
                    for article in document.articles
                ],
            },
        }

        metadata_path = (
            brand_path / "metadata" / f"{document.document_id}_metadata.json"
        )
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        created_files.append(metadata_path)

        return created_files

    def generate_all_brands(
        self, documents_per_brand: int = 5, **kwargs
    ) -> Dict[str, Any]:
        """
        Generate synthetic datasets for all configured brands.

        Args:
            documents_per_brand: Number of documents to generate per brand
            **kwargs: Additional arguments for generation

        Returns:
            Comprehensive generation report
        """
        start_time = datetime.now()

        reports = {}
        total_documents = 0
        total_errors = 0

        for brand in self.brand_configs.keys():
            try:
                self.logger.info("Generating synthetic data for brand", brand=brand)

                brand_report = self.generate_brand_dataset(
                    brand, documents_per_brand, **kwargs
                )

                reports[brand] = brand_report
                total_documents += brand_report["generated_documents"]
                total_errors += len(brand_report["errors"])

            except Exception as e:
                self.logger.error("Brand generation failed", brand=brand, error=str(e))
                reports[brand] = {"error": str(e), "generated_documents": 0}
                total_errors += 1

        total_time = datetime.now() - start_time

        summary_report = {
            "generation_timestamp": start_time.isoformat(),
            "total_generation_time": total_time.total_seconds(),
            "brands_processed": len(self.brand_configs),
            "total_documents_generated": total_documents,
            "total_errors": total_errors,
            "brand_reports": reports,
            "success_rate": (
                total_documents / (len(self.brand_configs) * documents_per_brand)
            )
            * 100,
        }

        self.logger.info(
            "All-brand synthetic generation completed",
            brands=len(self.brand_configs),
            total_documents=total_documents,
            generation_time=total_time.total_seconds(),
            success_rate=summary_report["success_rate"],
        )

        return summary_report


# CLI utilities
def generate_test_dataset(brand: str = None, num_docs: int = 5) -> None:
    """CLI utility to generate test datasets."""
    generator = GoldStandardSyntheticGenerator()

    if brand:
        print(f"Generating synthetic dataset for {brand}...")
        report = generator.generate_brand_dataset(brand, num_docs)

        print(f"\n=== Generation Report ===")
        print(f"Brand: {report['brand']}")
        print(f"Documents generated: {report['generated_documents']}")
        print(f"Files created: {report['generated_files']}")
        print(f"Generation time: {report['generation_time_seconds']:.2f}s")
        print(f"Validation rate: {report['validation_report']['validation_rate']:.1f}%")

        if report["errors"]:
            print(f"Errors: {len(report['errors'])}")
    else:
        print("Generating synthetic datasets for all brands...")
        report = generator.generate_all_brands(num_docs)

        print(f"\n=== All-Brands Generation Report ===")
        print(f"Total documents: {report['total_documents_generated']}")
        print(f"Success rate: {report['success_rate']:.1f}%")
        print(f"Generation time: {report['total_generation_time']:.2f}s")


if __name__ == "__main__":
    import sys

    brand = sys.argv[1] if len(sys.argv) > 1 else None
    num_docs = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    generate_test_dataset(brand, num_docs)
