"""
Brand-specific configuration loader for layout understanding.

This module provides utilities for loading and managing brand-specific
configurations for optimal layout understanding performance.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
import yaml

logger = structlog.get_logger(__name__)


class BrandConfigLoader:
    """
    Loader for brand-specific configuration files.

    Manages loading, validation, and merging of brand configurations
    for layout understanding optimization.
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize brand config loader.

        Args:
            config_dir: Directory containing brand configuration files
        """
        if config_dir is None:
            # Default to configs/brands directory
            config_dir = Path(__file__).parent.parent.parent / "configs" / "brands"

        self.config_dir = Path(config_dir)
        self.logger = logger.bind(component="BrandConfigLoader")

        # Cache for loaded configurations
        self._config_cache: Dict[str, Dict[str, Any]] = {}

        self.logger.debug(
            "Initialized brand config loader", config_dir=str(self.config_dir)
        )

    def load_brand_config(self, brand_name: str) -> Dict[str, Any]:
        """
        Load brand-specific configuration.

        Args:
            brand_name: Name of the brand

        Returns:
            Brand configuration dictionary
        """
        try:
            # Check cache first
            if brand_name in self._config_cache:
                return self._config_cache[brand_name]

            # Try to load from file
            config_path = self.config_dir / f"{brand_name.lower()}.yaml"

            if config_path.exists():
                config = self._load_yaml_config(config_path)
                self.logger.info(
                    "Loaded brand config from file",
                    brand=brand_name,
                    path=str(config_path),
                )
            else:
                # Fall back to default configuration
                config = self._get_default_config(brand_name)
                self.logger.info("Using default brand config", brand=brand_name)

            # Validate and process configuration
            config = self._validate_and_process_config(config, brand_name)

            # Cache the configuration
            self._config_cache[brand_name] = config

            return config

        except Exception as e:
            self.logger.error(
                "Error loading brand config", brand=brand_name, error=str(e)
            )
            return self._get_fallback_config(brand_name)

    def _load_yaml_config(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            if not isinstance(config, dict):
                raise ValueError("Configuration must be a dictionary")

            return config

        except Exception as e:
            self.logger.error(
                "Error loading YAML config", path=str(config_path), error=str(e)
            )
            raise

    def _validate_and_process_config(
        self, config: Dict[str, Any], brand_name: str
    ) -> Dict[str, Any]:
        """Validate and process brand configuration."""
        try:
            # Ensure required sections exist
            required_sections = ["layout_understanding"]
            for section in required_sections:
                if section not in config:
                    config[section] = {}

            # Process layout understanding configuration
            layout_config = config["layout_understanding"]

            # Set defaults for layout understanding
            if "model" not in layout_config:
                layout_config["model"] = {
                    "name": "microsoft/layoutlmv3-base",
                    "confidence_threshold": 0.95,
                    "device": "auto",
                }

            if "spatial_relationships" not in layout_config:
                layout_config[
                    "spatial_relationships"
                ] = self._get_default_spatial_config()

            # Normalize confidence adjustments
            if "confidence_adjustments" in layout_config:
                layout_config[
                    "confidence_adjustments"
                ] = self._normalize_confidence_adjustments(
                    layout_config["confidence_adjustments"]
                )

            # Add brand metadata
            config["_metadata"] = {
                "brand_name": brand_name,
                "loaded_at": "runtime",
                "version": config.get("version", "1.0"),
            }

            return config

        except Exception as e:
            self.logger.error("Error validating config", brand=brand_name, error=str(e))
            raise

    def _normalize_confidence_adjustments(
        self, adjustments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Normalize confidence adjustment configuration."""
        normalized = {}

        for block_type, adjustment in adjustments.items():
            if isinstance(adjustment, dict):
                normalized_adjustment = {
                    "confidence_multiplier": adjustment.get(
                        "confidence_multiplier", 1.0
                    ),
                    "confidence_bias": adjustment.get("confidence_bias", 0.0),
                }

                # Handle pattern overrides
                if "pattern_overrides" in adjustment:
                    normalized_adjustment["pattern_overrides"] = adjustment[
                        "pattern_overrides"
                    ]

                normalized[block_type] = normalized_adjustment
            else:
                # Simple numeric multiplier
                normalized[block_type] = {
                    "confidence_multiplier": float(adjustment),
                    "confidence_bias": 0.0,
                }

        return normalized

    def _get_default_spatial_config(self) -> Dict[str, Any]:
        """Get default spatial relationship configuration."""
        return {
            "proximity_threshold": 50,
            "alignment_threshold": 10,
            "column_gap_threshold": 30,
            "above": {"multiplier": 1.0},
            "below": {"multiplier": 1.0},
            "left_of": {"multiplier": 1.0},
            "right_of": {"multiplier": 1.0},
        }

    def _get_default_config(self, brand_name: str) -> Dict[str, Any]:
        """Get default configuration for unknown brands."""
        return {
            "brand": brand_name,
            "version": "1.0",
            "description": f"Default configuration for {brand_name}",
            "layout_understanding": {
                "model": {
                    "name": "microsoft/layoutlmv3-base",
                    "confidence_threshold": 0.95,
                    "device": "auto",
                },
                "confidence_adjustments": {
                    "title": {"confidence_multiplier": 1.0},
                    "body": {"confidence_multiplier": 1.0},
                    "byline": {"confidence_multiplier": 1.0},
                },
                "spatial_relationships": self._get_default_spatial_config(),
                "post_processing": {"min_edge_confidence": 0.3},
            },
            "accuracy_optimization": {"target_accuracy": 0.995},
        }

    def _get_fallback_config(self, brand_name: str) -> Dict[str, Any]:
        """Get minimal fallback configuration."""
        return {
            "brand": brand_name,
            "layout_understanding": {
                "model": {
                    "name": "microsoft/layoutlmv3-base",
                    "confidence_threshold": 0.90,  # Lower threshold for fallback
                    "device": "auto",
                }
            },
        }

    def get_available_brands(self) -> List[str]:
        """Get list of available brand configurations."""
        try:
            brands = []
            if self.config_dir.exists():
                for config_file in self.config_dir.glob("*.yaml"):
                    brand_name = config_file.stem
                    brands.append(brand_name)

            return sorted(brands)

        except Exception as e:
            self.logger.error("Error getting available brands", error=str(e))
            return []

    def extract_layoutlm_config(self, brand_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract LayoutLM-specific configuration from brand config.

        Args:
            brand_config: Full brand configuration

        Returns:
            LayoutLM configuration dictionary
        """
        layout_config = brand_config.get("layout_understanding", {})

        return {
            "name": brand_config.get("brand", "unknown"),
            "model_config": layout_config.get("model", {}),
            "confidence_adjustments": layout_config.get("confidence_adjustments", {}),
            "classification_adjustments": layout_config.get(
                "classification_adjustments", {}
            ),
            "spatial_relationships": layout_config.get("spatial_relationships", {}),
            "post_processing": layout_config.get("post_processing", {}),
        }

    def extract_spatial_config(self, brand_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract spatial relationship configuration from brand config.

        Args:
            brand_config: Full brand configuration

        Returns:
            Spatial configuration dictionary
        """
        spatial_config = brand_config.get("layout_understanding", {}).get(
            "spatial_relationships", {}
        )

        # Merge with defaults
        default_config = self._get_default_spatial_config()
        default_config.update(spatial_config)

        return default_config

    def get_accuracy_target(self, brand_config: Dict[str, Any]) -> float:
        """Get accuracy target from brand configuration."""
        return brand_config.get("accuracy_optimization", {}).get(
            "target_accuracy", 0.995
        )

    def clear_cache(self):
        """Clear the configuration cache."""
        self._config_cache.clear()
        self.logger.debug("Cleared brand config cache")

    def reload_brand_config(self, brand_name: str) -> Dict[str, Any]:
        """
        Reload brand configuration, bypassing cache.

        Args:
            brand_name: Name of the brand to reload

        Returns:
            Reloaded brand configuration
        """
        # Remove from cache if present
        if brand_name in self._config_cache:
            del self._config_cache[brand_name]

        # Load fresh configuration
        return self.load_brand_config(brand_name)


# Global instance for convenient access
_global_loader: Optional[BrandConfigLoader] = None


def get_brand_loader(config_dir: Optional[Path] = None) -> BrandConfigLoader:
    """Get global brand configuration loader instance."""
    global _global_loader

    if _global_loader is None or config_dir is not None:
        _global_loader = BrandConfigLoader(config_dir)

    return _global_loader


def load_brand_config(
    brand_name: str, config_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Convenience function to load brand configuration.

    Args:
        brand_name: Name of the brand
        config_dir: Optional config directory override

    Returns:
        Brand configuration dictionary
    """
    loader = get_brand_loader(config_dir)
    return loader.load_brand_config(brand_name)
