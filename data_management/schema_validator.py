"""
Schema validation for gold standard dataset files.

Validates XML ground truth files, JSON annotations, and metadata against
defined schemas to ensure data quality and consistency.
"""

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of validating a gold standard file."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    quality_score: float
    metadata: Dict[str, Any]

    def __post_init__(self):
        """Calculate overall validation status."""
        self.has_errors = len(self.errors) > 0
        self.has_warnings = len(self.warnings) > 0
        self.validation_passed = self.is_valid and not self.has_errors


@dataclass
class DatasetValidationReport:
    """Comprehensive validation report for a dataset."""

    brand: str
    total_files: int
    valid_files: int
    invalid_files: int
    file_results: List[ValidationResult]
    coverage_metrics: Dict[str, float]
    quality_metrics: Dict[str, float]
    recommendations: List[str]
    validation_timestamp: datetime

    @property
    def validation_rate(self) -> float:
        """Percentage of files that passed validation."""
        return (
            (self.valid_files / self.total_files) * 100 if self.total_files > 0 else 0.0
        )

    @property
    def average_quality_score(self) -> float:
        """Average quality score across all files."""
        scores = [r.quality_score for r in self.file_results if r.quality_score > 0]
        return sum(scores) / len(scores) if scores else 0.0


class GroundTruthSchemaValidator:
    """Validates XML ground truth files against magazine extraction schema."""

    def __init__(self, schema_path: Optional[Path] = None):
        """
        Initialize schema validator.

        Args:
            schema_path: Path to XSD schema file (defaults to project schema)
        """
        self.schema_path = schema_path or Path("schemas/article-v1.0.xsd")
        self.logger = logger.bind(component="GroundTruthValidator")

        # Define required elements and their constraints
        self.required_elements = {
            "magazine": {"required": True, "max_count": 1},
            "article": {"required": True, "min_count": 1},
            "title": {"required": True, "max_count": 1, "parent": "article"},
            "body": {"required": True, "min_count": 1, "parent": "article"},
            "contributors": {"required": False, "parent": "article"},
            "images": {"required": False, "parent": "article"},
        }

        # Define attribute requirements
        self.required_attributes = {
            "magazine": ["brand", "issue_date", "total_pages"],
            "article": ["id", "start_page", "end_page"],
            "title": ["confidence"],
            "body": ["confidence"],
            "contributor": ["name", "role", "confidence"],
            "image": ["id", "page", "bbox"],
        }

    def validate_xml_structure(self, xml_path: Path) -> ValidationResult:
        """
        Validate XML ground truth file structure and content.

        Args:
            xml_path: Path to XML file to validate

        Returns:
            ValidationResult with validation status and details
        """
        errors = []
        warnings = []
        metadata = {}

        try:
            self.logger.info("Validating XML structure", file=str(xml_path))

            # Check file exists and is readable
            if not xml_path.exists():
                return ValidationResult(
                    is_valid=False,
                    errors=[f"File does not exist: {xml_path}"],
                    warnings=[],
                    quality_score=0.0,
                    metadata={},
                )

            # Parse XML
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                metadata["root_element"] = root.tag
            except ET.ParseError as e:
                return ValidationResult(
                    is_valid=False,
                    errors=[f"XML parse error: {str(e)}"],
                    warnings=[],
                    quality_score=0.0,
                    metadata={},
                )

            # Validate root element
            if root.tag != "magazine":
                errors.append(f"Root element must be 'magazine', found '{root.tag}'")

            # Validate required attributes
            self._validate_attributes(root, "magazine", errors, warnings)

            # Validate magazine-level elements
            articles = root.findall("article")
            if not articles:
                errors.append("No articles found - at least one article is required")
            else:
                metadata["article_count"] = len(articles)

                # Validate each article
                for i, article in enumerate(articles):
                    self._validate_article(article, i, errors, warnings)

            # Validate page numbering consistency
            self._validate_page_consistency(articles, errors, warnings)

            # Calculate quality score
            quality_score = self._calculate_quality_score(root, errors, warnings)
            metadata["quality_score"] = quality_score

            # Check for additional quality indicators
            self._check_quality_indicators(root, warnings, metadata)

            is_valid = len(errors) == 0

            self.logger.info(
                "XML validation completed",
                file=str(xml_path),
                is_valid=is_valid,
                errors=len(errors),
                warnings=len(warnings),
                quality_score=quality_score,
            )

            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                quality_score=quality_score,
                metadata=metadata,
            )

        except Exception as e:
            self.logger.error(
                "Unexpected error in XML validation",
                file=str(xml_path),
                error=str(e),
                exc_info=True,
            )
            return ValidationResult(
                is_valid=False,
                errors=[f"Unexpected validation error: {str(e)}"],
                warnings=[],
                quality_score=0.0,
                metadata={},
            )

    def _validate_attributes(
        self,
        element: ET.Element,
        element_type: str,
        errors: List[str],
        warnings: List[str],
    ) -> None:
        """Validate required attributes for an element."""
        required_attrs = self.required_attributes.get(element_type, [])

        for attr in required_attrs:
            if attr not in element.attrib:
                errors.append(
                    f"{element_type} element missing required attribute: {attr}"
                )
            else:
                # Validate attribute values
                attr_value = element.attrib[attr]

                if attr == "confidence":
                    try:
                        conf = float(attr_value)
                        if not (0.0 <= conf <= 1.0):
                            errors.append(
                                f"Confidence value must be between 0.0 and 1.0, got {conf}"
                            )
                        elif conf < 0.5:
                            warnings.append(
                                f"Low confidence value ({conf}) in {element_type}"
                            )
                    except ValueError:
                        errors.append(
                            f"Confidence must be a number, got '{attr_value}'"
                        )

                elif attr in ["start_page", "end_page", "page"]:
                    try:
                        page = int(attr_value)
                        if page <= 0:
                            errors.append(f"Page numbers must be positive, got {page}")
                    except ValueError:
                        errors.append(
                            f"Page number must be an integer, got '{attr_value}'"
                        )

                elif attr == "total_pages":
                    try:
                        total = int(attr_value)
                        if total <= 0:
                            errors.append(f"Total pages must be positive, got {total}")
                    except ValueError:
                        errors.append(
                            f"Total pages must be an integer, got '{attr_value}'"
                        )

    def _validate_article(
        self,
        article: ET.Element,
        article_index: int,
        errors: List[str],
        warnings: List[str],
    ) -> None:
        """Validate individual article structure and content."""
        article_id = article.get("id", f"article_{article_index}")

        # Validate required attributes
        self._validate_attributes(article, "article", errors, warnings)

        # Validate page range
        try:
            start_page = int(article.get("start_page", "0"))
            end_page = int(article.get("end_page", "0"))

            if start_page > end_page:
                errors.append(
                    f"Article {article_id}: start_page ({start_page}) > end_page ({end_page})"
                )
            elif end_page - start_page > 10:
                warnings.append(
                    f"Article {article_id}: spans many pages ({start_page}-{end_page})"
                )
        except (ValueError, TypeError):
            pass  # Error already caught in attribute validation

        # Validate title
        titles = article.findall("title")
        if not titles:
            errors.append(f"Article {article_id}: missing required title element")
        elif len(titles) > 1:
            errors.append(f"Article {article_id}: multiple title elements found")
        else:
            title = titles[0]
            self._validate_attributes(title, "title", errors, warnings)

            title_text = title.text or ""
            if len(title_text.strip()) == 0:
                errors.append(f"Article {article_id}: title is empty")
            elif len(title_text) > 300:
                warnings.append(
                    f"Article {article_id}: very long title ({len(title_text)} chars)"
                )

        # Validate body paragraphs
        body_elements = article.findall("body")
        if not body_elements:
            errors.append(f"Article {article_id}: missing required body element(s)")
        else:
            for i, body in enumerate(body_elements):
                self._validate_attributes(body, "body", errors, warnings)

                body_text = body.text or ""
                if len(body_text.strip()) == 0:
                    warnings.append(
                        f"Article {article_id}: body paragraph {i+1} is empty"
                    )
                elif len(body_text) < 50:
                    warnings.append(
                        f"Article {article_id}: body paragraph {i+1} is very short"
                    )

        # Validate contributors
        contributors = article.findall("contributors/contributor")
        for i, contrib in enumerate(contributors):
            self._validate_attributes(contrib, "contributor", errors, warnings)

            name = contrib.get("name", "")
            if not name.strip():
                errors.append(f"Article {article_id}: contributor {i+1} has empty name")

            role = contrib.get("role", "")
            valid_roles = [
                "author",
                "photographer",
                "illustrator",
                "editor",
                "correspondent",
            ]
            if role not in valid_roles:
                warnings.append(
                    f"Article {article_id}: unusual contributor role '{role}'"
                )

        # Validate images
        images = article.findall("images/image")
        for i, image in enumerate(images):
            self._validate_attributes(image, "image", errors, warnings)

            # Validate bounding box format
            bbox = image.get("bbox", "")
            if bbox:
                try:
                    coords = [float(x.strip()) for x in bbox.split(",")]
                    if len(coords) != 4:
                        errors.append(
                            f"Article {article_id}: image {i+1} bbox must have 4 coordinates"
                        )
                    elif coords[0] >= coords[2] or coords[1] >= coords[3]:
                        errors.append(
                            f"Article {article_id}: image {i+1} invalid bbox coordinates"
                        )
                except ValueError:
                    errors.append(
                        f"Article {article_id}: image {i+1} invalid bbox format"
                    )

    def _validate_page_consistency(
        self, articles: List[ET.Element], errors: List[str], warnings: List[str]
    ) -> None:
        """Validate page numbering consistency across articles."""
        page_ranges = []

        for article in articles:
            try:
                start_page = int(article.get("start_page", "0"))
                end_page = int(article.get("end_page", "0"))
                article_id = article.get("id", "unknown")

                page_ranges.append((start_page, end_page, article_id))
            except ValueError:
                continue  # Skip invalid page numbers

        # Sort by start page
        page_ranges.sort(key=lambda x: x[0])

        # Check for overlaps
        for i in range(len(page_ranges) - 1):
            current = page_ranges[i]
            next_range = page_ranges[i + 1]

            if current[1] >= next_range[0]:
                warnings.append(
                    f"Page overlap between articles {current[2]} and {next_range[2]}: "
                    f"pages {current[0]}-{current[1]} and {next_range[0]}-{next_range[1]}"
                )

    def _calculate_quality_score(
        self, root: ET.Element, errors: List[str], warnings: List[str]
    ) -> float:
        """Calculate overall quality score for the ground truth file."""
        if errors:
            return 0.0  # Any errors result in 0 quality score

        score = 1.0

        # Deduct for warnings
        score -= len(warnings) * 0.05

        # Check for completeness indicators
        articles = root.findall("article")
        if articles:
            # Bonus for having contributors
            contributors_count = sum(
                len(article.findall("contributors/contributor")) for article in articles
            )
            if contributors_count > 0:
                score += 0.1

            # Bonus for having images with captions
            images_with_captions = sum(
                len(
                    [
                        img
                        for img in article.findall("images/image")
                        if img.find("caption") is not None
                    ]
                )
                for article in articles
            )
            if images_with_captions > 0:
                score += 0.1

            # Check confidence values
            confidence_elements = root.findall(".//*[@confidence]")
            if confidence_elements:
                confidences = []
                for elem in confidence_elements:
                    try:
                        conf = float(elem.get("confidence", "0"))
                        confidences.append(conf)
                    except ValueError:
                        continue

                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
                    score = score * avg_confidence  # Scale by average confidence

        return min(1.0, max(0.0, score))

    def _check_quality_indicators(
        self, root: ET.Element, warnings: List[str], metadata: Dict[str, Any]
    ) -> None:
        """Check for additional quality indicators and add to metadata."""

        # Count elements by type
        articles = root.findall("article")
        total_contributors = sum(
            len(article.findall("contributors/contributor")) for article in articles
        )
        total_images = sum(len(article.findall("images/image")) for article in articles)
        total_body_paragraphs = sum(
            len(article.findall("body")) for article in articles
        )

        metadata.update(
            {
                "total_contributors": total_contributors,
                "total_images": total_images,
                "total_body_paragraphs": total_body_paragraphs,
                "avg_contributors_per_article": total_contributors / len(articles)
                if articles
                else 0,
                "avg_images_per_article": total_images / len(articles)
                if articles
                else 0,
                "avg_paragraphs_per_article": total_body_paragraphs / len(articles)
                if articles
                else 0,
            }
        )

        # Check for potential quality issues
        if total_contributors == 0:
            warnings.append(
                "No contributors found - may indicate incomplete annotation"
            )

        if total_images == 0:
            warnings.append(
                "No images found - may indicate text-only content or incomplete annotation"
            )

        if metadata["avg_paragraphs_per_article"] < 2:
            warnings.append(
                "Very few body paragraphs per article - may indicate incomplete content"
            )


class MetadataValidator:
    """Validates JSON metadata files for gold standard datasets."""

    def __init__(self):
        self.logger = logger.bind(component="MetadataValidator")

        self.required_fields = {
            "dataset_info": {
                "brand": str,
                "filename": str,
                "creation_date": str,
                "file_type": str,
            },
            "quality_metrics": {
                "manual_validation": bool,
                "annotation_quality": float,
                "completeness_score": float,
            },
            "content_info": {
                "page_count": int,
                "article_count": int,
                "layout_complexity": str,
            },
        }

    def validate_metadata(self, metadata_path: Path) -> ValidationResult:
        """
        Validate JSON metadata file.

        Args:
            metadata_path: Path to metadata JSON file

        Returns:
            ValidationResult with validation status
        """
        errors = []
        warnings = []
        metadata = {}

        try:
            if not metadata_path.exists():
                return ValidationResult(
                    is_valid=False,
                    errors=[f"Metadata file does not exist: {metadata_path}"],
                    warnings=[],
                    quality_score=0.0,
                    metadata={},
                )

            # Load JSON
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                return ValidationResult(
                    is_valid=False,
                    errors=[f"Invalid JSON format: {str(e)}"],
                    warnings=[],
                    quality_score=0.0,
                    metadata={},
                )

            # Validate required sections
            for section, fields in self.required_fields.items():
                if section not in data:
                    errors.append(f"Missing required section: {section}")
                    continue

                section_data = data[section]
                for field, expected_type in fields.items():
                    if field not in section_data:
                        errors.append(f"Missing field {section}.{field}")
                    else:
                        value = section_data[field]
                        if not isinstance(value, expected_type):
                            errors.append(
                                f"Field {section}.{field} should be {expected_type.__name__}, "
                                f"got {type(value).__name__}"
                            )

            # Validate specific values
            if "quality_metrics" in data:
                metrics = data["quality_metrics"]

                # Check confidence scores
                for score_field in ["annotation_quality", "completeness_score"]:
                    if score_field in metrics:
                        score = metrics[score_field]
                        if isinstance(score, (int, float)):
                            if not (0.0 <= score <= 1.0):
                                errors.append(
                                    f"{score_field} must be between 0.0 and 1.0"
                                )
                            elif score < 0.8:
                                warnings.append(f"Low {score_field}: {score}")

            # Check creation date format
            if "dataset_info" in data and "creation_date" in data["dataset_info"]:
                date_str = data["dataset_info"]["creation_date"]
                try:
                    datetime.fromisoformat(date_str)
                except ValueError:
                    errors.append(
                        "creation_date must be in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)"
                    )

            # Calculate quality score
            quality_score = 1.0 - (len(errors) * 0.2) - (len(warnings) * 0.1)
            quality_score = max(0.0, min(1.0, quality_score))

            metadata.update(
                {
                    "validation_timestamp": datetime.now().isoformat(),
                    "field_completeness": self._calculate_field_completeness(data),
                    "data_size": len(json.dumps(data)),
                }
            )

            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                quality_score=quality_score,
                metadata=metadata,
            )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Unexpected error: {str(e)}"],
                warnings=[],
                quality_score=0.0,
                metadata={},
            )

    def _calculate_field_completeness(self, data: Dict[str, Any]) -> float:
        """Calculate how complete the metadata is based on required fields."""
        total_fields = sum(len(fields) for fields in self.required_fields.values())
        present_fields = 0

        for section, fields in self.required_fields.items():
            if section in data:
                section_data = data[section]
                for field in fields.keys():
                    if field in section_data:
                        present_fields += 1

        return present_fields / total_fields if total_fields > 0 else 0.0


class DatasetValidator:
    """Main validator for complete gold standard datasets."""

    def __init__(self, data_root: Path = None):
        """
        Initialize dataset validator.

        Args:
            data_root: Root path to gold standard datasets
        """
        self.data_root = data_root or Path("data/gold_sets")
        self.xml_validator = GroundTruthSchemaValidator()
        self.metadata_validator = MetadataValidator()
        self.logger = logger.bind(component="DatasetValidator")

    def validate_brand_dataset(self, brand: str) -> DatasetValidationReport:
        """
        Validate all files for a specific brand dataset.

        Args:
            brand: Brand name (e.g., 'economist', 'time')

        Returns:
            Comprehensive validation report
        """
        brand_path = self.data_root / brand

        if not brand_path.exists():
            return DatasetValidationReport(
                brand=brand,
                total_files=0,
                valid_files=0,
                invalid_files=0,
                file_results=[],
                coverage_metrics={},
                quality_metrics={},
                recommendations=[f"Brand directory does not exist: {brand_path}"],
                validation_timestamp=datetime.now(),
            )

        self.logger.info("Validating brand dataset", brand=brand, path=str(brand_path))

        # Find all files to validate
        xml_files = list((brand_path / "ground_truth").glob("*.xml"))
        metadata_files = list((brand_path / "metadata").glob("*.json"))

        file_results = []
        valid_count = 0

        # Validate XML ground truth files
        for xml_file in xml_files:
            result = self.xml_validator.validate_xml_structure(xml_file)
            result.metadata["file_type"] = "ground_truth"
            result.metadata["file_path"] = str(xml_file)
            file_results.append(result)

            if result.validation_passed:
                valid_count += 1

        # Validate metadata files
        for meta_file in metadata_files:
            result = self.metadata_validator.validate_metadata(meta_file)
            result.metadata["file_type"] = "metadata"
            result.metadata["file_path"] = str(meta_file)
            file_results.append(result)

            if result.validation_passed:
                valid_count += 1

        total_files = len(file_results)
        invalid_count = total_files - valid_count

        # Calculate coverage and quality metrics
        coverage_metrics = self._calculate_coverage_metrics(brand_path, file_results)
        quality_metrics = self._calculate_quality_metrics(file_results)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            brand, file_results, coverage_metrics
        )

        report = DatasetValidationReport(
            brand=brand,
            total_files=total_files,
            valid_files=valid_count,
            invalid_files=invalid_count,
            file_results=file_results,
            coverage_metrics=coverage_metrics,
            quality_metrics=quality_metrics,
            recommendations=recommendations,
            validation_timestamp=datetime.now(),
        )

        self.logger.info(
            "Brand dataset validation completed",
            brand=brand,
            total_files=total_files,
            valid_files=valid_count,
            validation_rate=report.validation_rate,
            avg_quality=report.average_quality_score,
        )

        return report

    def _calculate_coverage_metrics(
        self, brand_path: Path, results: List[ValidationResult]
    ) -> Dict[str, float]:
        """Calculate dataset coverage metrics."""
        pdf_count = len(list((brand_path / "pdfs").glob("*.pdf")))
        xml_count = len(
            [r for r in results if r.metadata.get("file_type") == "ground_truth"]
        )
        metadata_count = len(
            [r for r in results if r.metadata.get("file_type") == "metadata"]
        )

        # Calculate paired file ratios
        pdf_xml_ratio = xml_count / pdf_count if pdf_count > 0 else 0.0
        xml_metadata_ratio = metadata_count / xml_count if xml_count > 0 else 0.0

        return {
            "pdf_count": pdf_count,
            "xml_count": xml_count,
            "metadata_count": metadata_count,
            "pdf_xml_coverage": pdf_xml_ratio,
            "xml_metadata_coverage": xml_metadata_ratio,
            "complete_triplets": min(pdf_count, xml_count, metadata_count),
        }

    def _calculate_quality_metrics(
        self, results: List[ValidationResult]
    ) -> Dict[str, float]:
        """Calculate overall quality metrics."""
        if not results:
            return {}

        error_counts = [len(r.errors) for r in results]
        warning_counts = [len(r.warnings) for r in results]
        quality_scores = [r.quality_score for r in results if r.quality_score > 0]

        return {
            "avg_quality_score": sum(quality_scores) / len(quality_scores)
            if quality_scores
            else 0.0,
            "avg_errors_per_file": sum(error_counts) / len(error_counts),
            "avg_warnings_per_file": sum(warning_counts) / len(warning_counts),
            "error_free_files": sum(1 for c in error_counts if c == 0),
            "warning_free_files": sum(1 for c in warning_counts if c == 0),
        }

    def _generate_recommendations(
        self, brand: str, results: List[ValidationResult], coverage: Dict[str, float]
    ) -> List[str]:
        """Generate actionable recommendations for improving dataset quality."""
        recommendations = []

        # Coverage recommendations
        if coverage.get("pdf_count", 0) == 0:
            recommendations.append("Add PDF files to start dataset creation")

        if coverage.get("pdf_xml_coverage", 0) < 1.0:
            missing = coverage.get("pdf_count", 0) - coverage.get("xml_count", 0)
            recommendations.append(f"Create {missing} missing XML ground truth files")

        if coverage.get("xml_metadata_coverage", 0) < 1.0:
            missing = coverage.get("xml_count", 0) - coverage.get("metadata_count", 0)
            recommendations.append(f"Create {missing} missing metadata files")

        # Quality recommendations
        error_files = [r for r in results if r.errors]
        if error_files:
            recommendations.append(f"Fix validation errors in {len(error_files)} files")

        low_quality_files = [r for r in results if r.quality_score < 0.8]
        if low_quality_files:
            recommendations.append(
                f"Improve quality of {len(low_quality_files)} files with low scores"
            )

        # Dataset size recommendations
        total_files = coverage.get("complete_triplets", 0)
        if total_files < 25:
            recommendations.append(
                f"Expand dataset to at least 25 complete files (currently {total_files})"
            )
        elif total_files < 100:
            recommendations.append(
                f"Consider expanding to 100+ files for training (currently {total_files})"
            )

        return recommendations
