"""
Configuration Manager - Single source of truth for all configuration access.
Enforces DRY principles by centralizing configuration logic.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from functools import lru_cache
import structlog

from shared.models.brand_config import BrandConfig
from shared.config.validator import config_validator

logger = structlog.get_logger()

class ConfigurationManager:
    """
    Central configuration manager that serves as the single source of truth.
    No hardcoded brand logic allowed - everything must come from configurations.
    """
    
    def __init__(self, config_base_path: str = "configs"):
        self.config_base_path = Path(config_base_path)
        self.logger = logger.bind(component="config_manager")
        self._cache = {}
        
        # Validate all configurations on startup
        validation_results = config_validator.validate_all_configs()
        if validation_results["overall_status"] != "pass":
            self.logger.warning("Configuration validation issues found", 
                              errors=validation_results["errors"],
                              warnings=validation_results["warnings"])
    
    @lru_cache(maxsize=32)
    def get_brand_config(self, brand_name: str) -> BrandConfig:
        """
        Get brand configuration with caching.
        This is the ONLY way to access brand configurations.
        """
        return config_validator.get_brand_config(brand_name)
    
    def get_layout_hints(self, brand_name: str) -> Dict[str, Any]:
        """Get layout hints for a brand."""
        config = self.get_brand_config(brand_name)
        return config.layout_hints.dict()
    
    def get_ocr_settings(self, brand_name: str) -> Dict[str, Any]:
        """Get OCR settings for a brand."""
        config = self.get_brand_config(brand_name)
        return config.ocr_preprocessing.dict()
    
    def get_confidence_thresholds(self, brand_name: str) -> Dict[str, float]:
        """Get confidence thresholds for a brand."""
        config = self.get_brand_config(brand_name)
        overrides = config.confidence_overrides.dict()
        
        # Remove None values and provide defaults
        defaults = {
            "title": 0.90,
            "body": 0.88,
            "contributors": 0.85,
            "images": 0.85,
            "captions": 0.83
        }
        
        result = {}
        for field, default_value in defaults.items():
            result[field] = overrides.get(field) or default_value
        
        return result
    
    def get_reconstruction_rules(self, brand_name: str) -> Dict[str, Any]:
        """Get article reconstruction rules for a brand."""
        config = self.get_brand_config(brand_name)
        return config.reconstruction_rules.dict()
    
    def get_contributor_patterns(self, brand_name: str) -> Dict[str, List[str]]:
        """Get contributor extraction patterns for a brand."""
        config = self.get_brand_config(brand_name)
        if config.contributor_patterns:
            return config.contributor_patterns.dict()
        
        # Return empty patterns if not configured
        return {"author": [], "photographer": [], "illustrator": []}
    
    def get_ad_filtering_config(self, brand_name: str) -> Dict[str, Any]:
        """Get ad filtering configuration for a brand."""
        config = self.get_brand_config(brand_name)
        if config.ad_filtering:
            return config.ad_filtering.dict()
        
        # Return default ad filtering if not configured
        return {
            "visual_indicators": ["high_image_ratio", "border_box"],
            "text_patterns": ["Advertisement", "Sponsored", "Ad"],
            "confidence_threshold": 0.8
        }
    
    def get_image_processing_config(self, brand_name: str) -> Dict[str, Any]:
        """Get image processing configuration for a brand."""
        config = self.get_brand_config(brand_name)
        if config.image_processing:
            return config.image_processing.dict()
        
        # Return defaults if not configured
        return {
            "min_dimensions": [100, 100],
            "max_dimensions": [2048, 2048],
            "supported_formats": ["JPEG", "PNG", "TIFF"],
            "caption_linking_distance": 100
        }
    
    def get_custom_settings(self, brand_name: str) -> Dict[str, Any]:
        """Get custom brand-specific settings."""
        config = self.get_brand_config(brand_name)
        return config.custom_settings
    
    def is_feature_enabled(self, brand_name: str, feature_name: str) -> bool:
        """Check if a custom feature is enabled for a brand."""
        custom_settings = self.get_custom_settings(brand_name)
        return custom_settings.get(feature_name, False)
    
    def get_tesseract_config(self, brand_name: str) -> str:
        """Get Tesseract configuration string for a brand."""
        ocr_settings = self.get_ocr_settings(brand_name)
        return ocr_settings["tesseract_config"]
    
    def should_deskew_images(self, brand_name: str) -> bool:
        """Check if images should be deskewed for a brand."""
        ocr_settings = self.get_ocr_settings(brand_name)
        return ocr_settings["deskew"]
    
    def get_column_count_hints(self, brand_name: str) -> List[int]:
        """Get typical column counts for a brand."""
        layout_hints = self.get_layout_hints(brand_name)
        return layout_hints["column_count"]
    
    def get_title_patterns(self, brand_name: str) -> List[str]:
        """Get title detection patterns for a brand."""
        layout_hints = self.get_layout_hints(brand_name)
        return layout_hints["title_patterns"]
    
    def get_jump_indicators(self, brand_name: str) -> List[str]:
        """Get jump reference indicators for a brand."""
        layout_hints = self.get_layout_hints(brand_name)
        return layout_hints["jump_indicators"]
    
    def get_spatial_threshold(self, brand_name: str) -> int:
        """Get spatial threshold for block relationships."""
        rules = self.get_reconstruction_rules(brand_name)
        return rules["spatial_threshold_pixels"]
    
    def allows_cross_page_articles(self, brand_name: str) -> bool:
        """Check if brand allows articles spanning multiple pages."""
        rules = self.get_reconstruction_rules(brand_name)
        return rules["allow_cross_page_articles"]
    
    def get_max_jump_distance(self, brand_name: str) -> int:
        """Get maximum page distance for jump references."""
        rules = self.get_reconstruction_rules(brand_name)
        return rules["max_jump_distance_pages"]
    
    def get_confidence_threshold(self, brand_name: str, field: str) -> float:
        """Get confidence threshold for a specific field."""
        thresholds = self.get_confidence_thresholds(brand_name)
        return thresholds.get(field, 0.85)  # Default fallback
    
    def list_configured_brands(self) -> List[str]:
        """List all brands with valid configurations."""
        return config_validator.list_available_brands()
    
    def validate_brand_exists(self, brand_name: str) -> bool:
        """Check if a brand configuration exists and is valid."""
        try:
            self.get_brand_config(brand_name)
            return True
        except (ValueError, FileNotFoundError):
            return False
    
    def get_brand_description(self, brand_name: str) -> Optional[str]:
        """Get human-readable description for a brand."""
        config = self.get_brand_config(brand_name)
        return config.description
    
    def get_config_version(self, brand_name: str) -> str:
        """Get configuration version for a brand."""
        config = self.get_brand_config(brand_name)
        return config.version
    
    def clear_cache(self):
        """Clear configuration cache - useful for testing or config updates."""
        self.get_brand_config.cache_clear()
        self._cache.clear()
        self.logger.info("Configuration cache cleared")
    
    def get_all_field_weights(self) -> Dict[str, float]:
        """
        Get standard field weights for accuracy calculation.
        These are defined in the PRD and should be consistent across brands.
        """
        return {
            "title": 0.30,     # 30% weight - Title text accuracy
            "body": 0.40,      # 40% weight - Body text WER < 0.1%
            "contributors": 0.20,  # 20% weight - Name + role accuracy
            "media": 0.10      # 10% weight - Image-caption linking
        }
    
    def get_global_accuracy_threshold(self) -> float:
        """Get the global accuracy threshold from PRD (99.9%)."""
        return 0.999
    
    def get_brand_pass_rate_threshold(self) -> float:
        """Get the brand pass rate threshold from PRD (95%)."""
        return 0.95
    
    def get_quarantine_threshold(self) -> float:
        """Get the quarantine threshold from PRD."""
        return 0.95

# Global configuration manager instance
config_manager = ConfigurationManager()

# Convenience functions for external use - enforce DRY by using single manager
def get_brand_config(brand_name: str) -> BrandConfig:
    """Get brand configuration - SINGLE SOURCE OF TRUTH."""
    return config_manager.get_brand_config(brand_name)

def get_confidence_threshold(brand_name: str, field: str) -> float:
    """Get confidence threshold for field - NO HARDCODED VALUES."""
    return config_manager.get_confidence_threshold(brand_name, field)

def get_layout_hints(brand_name: str) -> Dict[str, Any]:
    """Get layout hints - NO HARDCODED PATTERNS."""
    return config_manager.get_layout_hints(brand_name)

def get_ocr_settings(brand_name: str) -> Dict[str, Any]:
    """Get OCR settings - NO HARDCODED PARAMETERS."""
    return config_manager.get_ocr_settings(brand_name)

def is_feature_enabled(brand_name: str, feature: str) -> bool:
    """Check custom feature - NO HARDCODED FEATURE FLAGS."""
    return config_manager.is_feature_enabled(brand_name, feature)

def list_available_brands() -> List[str]:
    """List available brands - DISCOVER FROM CONFIGS."""
    return config_manager.list_configured_brands()

# Validation functions
def validate_brand_name(brand_name: str) -> bool:
    """Validate that brand name has valid configuration."""
    return config_manager.validate_brand_exists(brand_name)

def validate_all_configurations() -> Dict[str, Any]:
    """Validate all configuration files."""
    return config_validator.validate_all_configs()