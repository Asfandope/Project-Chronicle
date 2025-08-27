"""
Configuration validator that ensures all YAMLs conform to schema.
This is the single source of truth for configuration validation.
"""

from pathlib import Path
from typing import Any, Dict, List

import structlog
import yaml
from pydantic import ValidationError

from shared.models.brand_config import BrandConfig, validate_brand_config_yaml

logger = structlog.get_logger()


class ConfigValidator:
    """
    Validates all configuration files against their schemas.
    Ensures DRY principles by being the single source of validation logic.
    """

    def __init__(self, config_base_path: str = "configs"):
        self.config_base_path = Path(config_base_path)
        self.brands_path = self.config_base_path / "brands"
        self.schemas_path = Path("schemas")
        self.logger = logger.bind(component="config_validator")

    def validate_all_configs(self) -> Dict[str, Any]:
        """
        Validate all configuration files in the system.
        Returns validation results with pass/fail status and details.
        """
        results = {
            "overall_status": "pass",
            "validation_timestamp": "placeholder",  # Would use actual timestamp
            "brand_configs": {},
            "xml_schema": {},
            "errors": [],
            "warnings": [],
        }

        # Validate XML schema
        xml_schema_result = self._validate_xml_schema()
        results["xml_schema"] = xml_schema_result

        if not xml_schema_result["valid"]:
            results["overall_status"] = "fail"
            results["errors"].extend(xml_schema_result["errors"])

        # Validate all brand configurations
        brand_results = self._validate_all_brand_configs()
        results["brand_configs"] = brand_results

        # Check if any brands failed
        failed_brands = [
            brand for brand, result in brand_results.items() if not result["valid"]
        ]
        if failed_brands:
            results["overall_status"] = "fail"
            results["errors"].append(f"Failed brand validations: {failed_brands}")

        # Cross-validation checks
        cross_validation_result = self._cross_validate_configs(brand_results)
        if cross_validation_result["warnings"]:
            results["warnings"].extend(cross_validation_result["warnings"])

        self.logger.info(
            "Configuration validation completed",
            overall_status=results["overall_status"],
            total_brands=len(brand_results),
            failed_brands=len(failed_brands),
        )

        return results

    def validate_brand_config(self, brand_name: str) -> Dict[str, Any]:
        """
        Validate a specific brand configuration.
        Returns detailed validation results.
        """
        config_file = self.brands_path / f"{brand_name}.yaml"

        if not config_file.exists():
            return {
                "valid": False,
                "brand": brand_name,
                "file_path": str(config_file),
                "errors": [f"Configuration file not found: {config_file}"],
                "warnings": [],
            }

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Validate YAML syntax
            try:
                yaml.safe_load(content)
            except yaml.YAMLError as e:
                return {
                    "valid": False,
                    "brand": brand_name,
                    "file_path": str(config_file),
                    "errors": [f"Invalid YAML syntax: {e}"],
                    "warnings": [],
                }

            # Validate against Pydantic schema
            try:
                config = validate_brand_config_yaml(content)

                # Additional validations
                warnings = self._validate_brand_config_contents(config, brand_name)

                return {
                    "valid": True,
                    "brand": brand_name,
                    "file_path": str(config_file),
                    "config": config.dict(),
                    "errors": [],
                    "warnings": warnings,
                }

            except ValidationError as e:
                return {
                    "valid": False,
                    "brand": brand_name,
                    "file_path": str(config_file),
                    "errors": [f"Schema validation failed: {e}"],
                    "warnings": [],
                }

        except Exception as e:
            return {
                "valid": False,
                "brand": brand_name,
                "file_path": str(config_file),
                "errors": [f"Unexpected error: {e}"],
                "warnings": [],
            }

    def _validate_xml_schema(self) -> Dict[str, Any]:
        """Validate the canonical XML schema file."""
        schema_file = self.schemas_path / "article-v1.0.xsd"

        if not schema_file.exists():
            return {
                "valid": False,
                "file_path": str(schema_file),
                "errors": [f"XML schema file not found: {schema_file}"],
            }

        try:
            # Basic XML syntax validation
            import xml.etree.ElementTree as ET

            with open(schema_file, "r", encoding="utf-8") as f:
                ET.parse(f)

            # Additional schema-specific validations could go here
            # For now, basic XML parsing is sufficient

            return {"valid": True, "file_path": str(schema_file), "errors": []}

        except ET.ParseError as e:
            return {
                "valid": False,
                "file_path": str(schema_file),
                "errors": [f"XML schema parse error: {e}"],
            }
        except Exception as e:
            return {
                "valid": False,
                "file_path": str(schema_file),
                "errors": [f"XML schema validation error: {e}"],
            }

    def _validate_all_brand_configs(self) -> Dict[str, Dict[str, Any]]:
        """Validate all brand configuration files."""
        results = {}

        if not self.brands_path.exists():
            self.logger.warning(
                "Brands configuration directory not found", path=str(self.brands_path)
            )
            return results

        for config_file in self.brands_path.glob("*.yaml"):
            brand_name = config_file.stem
            results[brand_name] = self.validate_brand_config(brand_name)

        return results

    def _validate_brand_config_contents(
        self, config: BrandConfig, expected_brand_name: str
    ) -> List[str]:
        """
        Additional content validation for brand configurations.
        Returns list of warnings.
        """
        warnings = []

        # Check brand name consistency
        if config.brand != expected_brand_name:
            warnings.append(
                f"Brand name mismatch: config says '{config.brand}', filename says '{expected_brand_name}'"
            )

        # Check confidence threshold consistency
        overrides = config.confidence_overrides
        if overrides.title and overrides.body and overrides.title <= overrides.body:
            warnings.append(
                "Title confidence threshold should typically be higher than body threshold"
            )

        # Check reconstruction rules consistency
        rules = config.reconstruction_rules
        if rules.min_title_length >= rules.max_title_length:
            warnings.append("min_title_length should be less than max_title_length")

        # Check OCR settings
        ocr = config.ocr_preprocessing
        if ocr.denoise_level > 3 and ocr.enhance_contrast:
            warnings.append(
                "High denoise level with contrast enhancement may over-process images"
            )

        # Check spatial thresholds
        if rules.spatial_threshold_pixels > 200:
            warnings.append(
                "Large spatial threshold may incorrectly link distant blocks"
            )

        # Check pattern validity (basic regex syntax check)
        all_patterns = []
        if config.contributor_patterns:
            all_patterns.extend(config.contributor_patterns.author or [])
            all_patterns.extend(config.contributor_patterns.photographer or [])
            all_patterns.extend(config.contributor_patterns.illustrator or [])

        for i, pattern in enumerate(all_patterns):
            try:
                import re

                re.compile(pattern)
            except re.error as e:
                warnings.append(f"Invalid regex pattern #{i+1}: {pattern} - {e}")

        return warnings

    def _cross_validate_configs(
        self, brand_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Cross-validate configurations to ensure consistency across brands.
        Returns warnings about potential inconsistencies.
        """
        warnings = []
        valid_configs = {}

        # Collect valid configurations
        for brand, result in brand_results.items():
            if result["valid"] and "config" in result:
                valid_configs[brand] = result["config"]

        if len(valid_configs) < 2:
            return {"warnings": warnings}

        # Check for extreme outliers in confidence thresholds
        all_title_thresholds = []
        all_body_thresholds = []

        for brand, config in valid_configs.items():
            overrides = config.get("confidence_overrides", {})
            if overrides.get("title"):
                all_title_thresholds.append((brand, overrides["title"]))
            if overrides.get("body"):
                all_body_thresholds.append((brand, overrides["body"]))

        # Check for outliers (simple statistical check)
        if all_title_thresholds:
            values = [t[1] for t in all_title_thresholds]
            mean_val = sum(values) / len(values)
            for brand, val in all_title_thresholds:
                if abs(val - mean_val) > 0.15:  # More than 15% deviation
                    warnings.append(
                        f"Brand '{brand}' has unusual title confidence threshold: {val} (mean: {mean_val:.3f})"
                    )

        # Check for consistent naming patterns
        brand_naming_styles = {}
        for brand in valid_configs.keys():
            if "_" in brand:
                brand_naming_styles[brand] = "underscore"
            elif "-" in brand:
                brand_naming_styles[brand] = "hyphen"
            else:
                brand_naming_styles[brand] = "simple"

        style_counts = {}
        for style in brand_naming_styles.values():
            style_counts[style] = style_counts.get(style, 0) + 1

        if len(style_counts) > 1:
            warnings.append(f"Inconsistent brand naming styles found: {style_counts}")

        return {"warnings": warnings}

    def get_brand_config(self, brand_name: str) -> BrandConfig:
        """
        Load and return a validated brand configuration.
        Raises ValueError if configuration is invalid.
        """
        validation_result = self.validate_brand_config(brand_name)

        if not validation_result["valid"]:
            errors = "; ".join(validation_result["errors"])
            raise ValueError(
                f"Invalid configuration for brand '{brand_name}': {errors}"
            )

        config_file = self.brands_path / f"{brand_name}.yaml"
        with open(config_file, "r", encoding="utf-8") as f:
            content = f.read()

        return validate_brand_config_yaml(content)

    def list_available_brands(self) -> List[str]:
        """Return list of available brand configurations."""
        if not self.brands_path.exists():
            return []

        brands = []
        for config_file in self.brands_path.glob("*.yaml"):
            validation_result = self.validate_brand_config(config_file.stem)
            if validation_result["valid"]:
                brands.append(config_file.stem)

        return sorted(brands)

    def generate_config_template(self, brand_name: str) -> str:
        """Generate a YAML template for a new brand configuration."""
        template_config = BrandConfig(
            brand=brand_name,
            version="1.0",
            description=f"Configuration for {brand_name.title()} magazine",
            layout_hints={
                "column_count": [2, 3],
                "title_patterns": ["^[A-Z][a-z]+.*"],
                "jump_indicators": ["continued on page", "see page"],
            },
            ocr_preprocessing={
                "deskew": True,
                "denoise_level": 2,
                "enhance_contrast": True,
                "tesseract_config": "--oem 3 --psm 6",
                "confidence_threshold": 0.7,
                "languages": ["eng"],
            },
            confidence_overrides={
                "title": 0.90,
                "body": 0.88,
                "contributors": 0.85,
                "images": 0.85,
            },
            reconstruction_rules={
                "min_title_length": 5,
                "max_title_length": 200,
                "min_body_paragraphs": 1,
                "spatial_threshold_pixels": 50,
                "allow_cross_page_articles": True,
                "max_jump_distance_pages": 5,
            },
        )

        return yaml.dump(
            template_config.dict(), default_flow_style=False, sort_keys=False
        )


# Global validator instance
config_validator = ConfigValidator()


# Convenience functions for external use
def validate_all_configs() -> Dict[str, Any]:
    """Global function to validate all configurations."""
    return config_validator.validate_all_configs()


def get_brand_config(brand_name: str) -> BrandConfig:
    """Global function to get a validated brand configuration."""
    return config_validator.get_brand_config(brand_name)


def list_available_brands() -> List[str]:
    """Global function to list available brands."""
    return config_validator.list_available_brands()


def validate_brand_config_file(brand_name: str) -> Dict[str, Any]:
    """Global function to validate a specific brand configuration."""
    return config_validator.validate_brand_config(brand_name)
