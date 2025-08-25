"""
Test the configuration system to ensure DRY principles and schema compliance.
"""

import pytest
import yaml
from pathlib import Path
from pydantic import ValidationError

from shared.models.brand_config import BrandConfig, validate_brand_config_yaml
from shared.models.article import Article, ConfidenceText
from shared.config.validator import config_validator
from shared.config.manager import config_manager

class TestXMLSchemaCompliance:
    """Test that Pydantic models match XML schema exactly."""
    
    def test_confidence_type_validation(self):
        """Test confidence type validates range and precision."""
        # Valid confidence values
        valid_values = [0.0, 0.5, 1.0, 0.123, 0.999]
        for value in valid_values:
            text = ConfidenceText(content="test", confidence=value)
            assert text.confidence == round(value, 3)
        
        # Invalid confidence values
        invalid_values = [-0.1, 1.1, 2.0]
        for value in invalid_values:
            with pytest.raises(ValueError):
                ConfidenceText(content="test", confidence=value)
    
    def test_contributor_role_enum(self):
        """Test contributor roles match XSD enumeration exactly."""
        from shared.models.article import ContributorRole
        
        # These must match the XSD enumeration exactly
        expected_roles = ["author", "photographer", "illustrator"]
        actual_roles = [role.value for role in ContributorRole]
        
        assert set(actual_roles) == set(expected_roles)
    
    def test_article_structure_matches_xsd(self):
        """Test Article model structure matches XSD exactly."""
        from datetime import datetime, date
        
        # Create a complete article that should match XSD
        article_data = {
            "id": "test_article_001",
            "brand": "economist",
            "issue": date(2023, 10, 15),
            "page_start": 1,
            "page_end": 2,
            "title": {
                "content": "Test Article Title",
                "confidence": 0.95
            },
            "body": {
                "content": [
                    {
                        "content": "First paragraph of the article.",
                        "confidence": 0.92
                    }
                ]
            },
            "provenance": {
                "extracted_at": datetime(2023, 10, 15, 14, 30, 0),
                "model_version": "v1.2.3",
                "confidence_overall": 0.94
            }
        }
        
        # This should not raise any validation errors
        article = Article(**article_data)
        
        # Verify required fields
        assert article.id == "test_article_001"
        assert article.brand == "economist"
        assert article.page_start == 1
        assert article.page_end == 2
        assert article.title.content == "Test Article Title"
        assert article.title.confidence == 0.95

class TestBrandConfigurationSchema:
    """Test brand configuration schema validation."""
    
    def test_valid_brand_config_creation(self):
        """Test creating valid brand configuration."""
        config_data = {
            "brand": "test_brand",
            "version": "1.0",
            "layout_hints": {
                "column_count": [2, 3],
                "title_patterns": ["^[A-Z][a-z]+.*"],
                "jump_indicators": ["continued on page"]
            },
            "ocr_preprocessing": {
                "deskew": True,
                "denoise_level": 2,
                "tesseract_config": "--oem 3 --psm 6",
                "confidence_threshold": 0.7
            },
            "confidence_overrides": {
                "title": 0.95,
                "body": 0.92
            },
            "reconstruction_rules": {
                "min_title_length": 5,
                "max_title_length": 200,
                "spatial_threshold_pixels": 50
            }
        }
        
        config = BrandConfig(**config_data)
        assert config.brand == "test_brand"
        assert config.layout_hints.column_count == [2, 3]
    
    def test_invalid_confidence_thresholds(self):
        """Test validation of confidence thresholds."""
        invalid_configs = [
            {"title": 1.5},  # > 1.0
            {"body": -0.1},  # < 0.0
            {"contributors": "invalid"},  # not a number
        ]
        
        for invalid_override in invalid_configs:
            with pytest.raises(ValidationError):
                BrandConfig(
                    brand="test",
                    layout_hints={"column_count": [2], "title_patterns": [], "jump_indicators": []},
                    ocr_preprocessing={"tesseract_config": "--oem 3", "confidence_threshold": 0.7},
                    confidence_overrides=invalid_override,
                    reconstruction_rules={"spatial_threshold_pixels": 50}
                )
    
    def test_brand_name_validation(self):
        """Test brand name validation rules."""
        valid_names = ["economist", "time", "vogue", "test_brand", "brand-name"]
        invalid_names = ["", "Brand With Spaces", "brand@special", "123"]
        
        base_config = {
            "layout_hints": {"column_count": [2], "title_patterns": [], "jump_indicators": []},
            "ocr_preprocessing": {"tesseract_config": "--oem 3", "confidence_threshold": 0.7},
            "confidence_overrides": {},
            "reconstruction_rules": {"spatial_threshold_pixels": 50}
        }
        
        for valid_name in valid_names:
            config = BrandConfig(brand=valid_name, **base_config)
            assert config.brand == valid_name.lower()
        
        for invalid_name in invalid_names:
            with pytest.raises(ValidationError):
                BrandConfig(brand=invalid_name, **base_config)

class TestConfigurationValidator:
    """Test the configuration validator."""
    
    def test_validate_existing_brand_configs(self):
        """Test validation of existing brand configuration files."""
        # Test validation of actual config files
        brands_to_test = ["economist", "time", "vogue"]
        
        for brand in brands_to_test:
            result = config_validator.validate_brand_config(brand)
            
            # Print details if validation fails for debugging
            if not result["valid"]:
                print(f"Validation failed for {brand}:")
                print(f"Errors: {result['errors']}")
                print(f"Warnings: {result['warnings']}")
            
            assert result["valid"], f"Brand config '{brand}' should be valid"
            assert result["brand"] == brand
            assert "config" in result
    
    def test_validate_all_configs(self):
        """Test validation of all configurations."""
        results = config_validator.validate_all_configs()
        
        # Print details if validation fails for debugging
        if results["overall_status"] != "pass":
            print("Overall validation failed:")
            print(f"Errors: {results['errors']}")
            print(f"Warnings: {results['warnings']}")
            for brand, result in results["brand_configs"].items():
                if not result["valid"]:
                    print(f"  {brand}: {result['errors']}")
        
        assert results["overall_status"] == "pass"
        assert results["xml_schema"]["valid"]
    
    def test_yaml_syntax_validation(self):
        """Test YAML syntax validation."""
        valid_yaml = """
brand: test
version: "1.0"
layout_hints:
  column_count: [2, 3]
  title_patterns: []
  jump_indicators: []
ocr_preprocessing:
  tesseract_config: "--oem 3"
  confidence_threshold: 0.7
confidence_overrides: {}
reconstruction_rules:
  spatial_threshold_pixels: 50
"""
        
        invalid_yaml = """
brand: test
version: "1.0"
layout_hints:
  column_count: [2, 3
  # Missing closing bracket - invalid YAML
"""
        
        # Valid YAML should parse successfully
        config = validate_brand_config_yaml(valid_yaml)
        assert config.brand == "test"
        
        # Invalid YAML should raise error
        with pytest.raises(ValueError, match="Invalid YAML format"):
            validate_brand_config_yaml(invalid_yaml)

class TestConfigurationManager:
    """Test the configuration manager DRY principles."""
    
    def test_no_hardcoded_brand_logic(self):
        """Test that no hardcoded brand logic exists."""
        # All brand-specific logic should come from configs
        brands = config_manager.list_configured_brands()
        
        for brand in brands:
            # These should all come from config, not hardcoded
            confidence_thresholds = config_manager.get_confidence_thresholds(brand)
            layout_hints = config_manager.get_layout_hints(brand)
            ocr_settings = config_manager.get_ocr_settings(brand)
            
            # Verify we get actual config data, not defaults
            assert isinstance(confidence_thresholds, dict)
            assert isinstance(layout_hints, dict)
            assert isinstance(ocr_settings, dict)
            
            # These should be from config files
            assert "title_patterns" in layout_hints
            assert "tesseract_config" in ocr_settings
    
    def test_configuration_caching(self):
        """Test that configuration caching works correctly."""
        brand = "economist"
        
        # First call should load from file
        config1 = config_manager.get_brand_config(brand)
        
        # Second call should use cache
        config2 = config_manager.get_brand_config(brand)
        
        # Should be the same object due to caching
        assert config1 is config2
        
        # Clear cache and reload
        config_manager.clear_cache()
        config3 = config_manager.get_brand_config(brand)
        
        # Should be different object but same content
        assert config1 is not config3
        assert config1.dict() == config3.dict()
    
    def test_single_source_of_truth(self):
        """Test that configuration manager is single source of truth."""
        brand = "economist"
        
        # All these should return consistent data from same source
        title_threshold1 = config_manager.get_confidence_threshold(brand, "title")
        title_threshold2 = config_manager.get_confidence_thresholds(brand)["title"]
        
        config = config_manager.get_brand_config(brand)
        title_threshold3 = config.confidence_overrides.title
        
        assert title_threshold1 == title_threshold2 == title_threshold3
    
    def test_feature_flag_system(self):
        """Test custom feature flag system."""
        brand = "economist"
        
        # Should be able to check custom features
        custom_settings = config_manager.get_custom_settings(brand)
        
        # Test specific feature check
        if "uk_spelling_preference" in custom_settings:
            uk_spelling = config_manager.is_feature_enabled(brand, "uk_spelling_preference")
            assert isinstance(uk_spelling, bool)
            assert uk_spelling == custom_settings["uk_spelling_preference"]

class TestDRYPrinciples:
    """Test that DRY principles are enforced."""
    
    def test_no_duplicate_configuration_logic(self):
        """Test that configuration logic is not duplicated."""
        # All services should use the same configuration manager
        from shared.config.manager import config_manager as manager1
        from shared.config.manager import get_brand_config as func1
        from shared.config.validator import config_validator as validator1
        
        # These should be the same instances
        assert config_manager is manager1
        
        # Functions should use the same underlying manager
        brand = "economist"
        config1 = func1(brand)
        config2 = config_manager.get_brand_config(brand)
        
        assert config1 is config2  # Should be same cached object
    
    def test_configuration_consistency_across_brands(self):
        """Test configuration consistency across brands."""
        brands = config_manager.list_configured_brands()
        
        # All brands should follow same schema
        for brand in brands:
            config = config_manager.get_brand_config(brand)
            
            # All should have required sections
            assert hasattr(config, 'layout_hints')
            assert hasattr(config, 'ocr_preprocessing')
            assert hasattr(config, 'confidence_overrides')
            assert hasattr(config, 'reconstruction_rules')
            
            # All should have valid version format
            assert config.version.count('.') >= 1  # At least "1.0"
    
    def test_global_constants_consistency(self):
        """Test that global constants are consistent."""
        # These should be defined in one place and consistent
        field_weights = config_manager.get_all_field_weights()
        
        # PRD specifies exact weights
        assert field_weights["title"] == 0.30
        assert field_weights["body"] == 0.40
        assert field_weights["contributors"] == 0.20
        assert field_weights["media"] == 0.10
        
        # Should sum to 1.0
        assert abs(sum(field_weights.values()) - 1.0) < 0.001
        
        # Global thresholds from PRD
        assert config_manager.get_global_accuracy_threshold() == 0.999
        assert config_manager.get_brand_pass_rate_threshold() == 0.95