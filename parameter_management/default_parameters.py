"""
Default parameter definitions for initial system setup.

This module defines all the default parameters that replace hardcoded values
throughout the system. These parameters are loaded during initial setup.
"""

from typing import Any, Dict, List

from .models import ParameterType

# Default parameters organized by category
DEFAULT_PARAMETERS = {
    # Accuracy calculation parameters (PRD Section 6)
    "accuracy": {
        "accuracy.title_weight": {
            "name": "Title Accuracy Weight",
            "description": "Weight for title matching in overall accuracy calculation (PRD Section 6: 30%)",
            "parameter_type": ParameterType.THRESHOLD,
            "data_type": "float",
            "default_value": 0.30,
            "validation_rules": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "impact_level": "high",
        },
        "accuracy.body_text_weight": {
            "name": "Body Text Accuracy Weight",
            "description": "Weight for body text WER in overall accuracy calculation (PRD Section 6: 40%)",
            "parameter_type": ParameterType.THRESHOLD,
            "data_type": "float",
            "default_value": 0.40,
            "validation_rules": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "impact_level": "high",
        },
        "accuracy.contributors_weight": {
            "name": "Contributors Accuracy Weight",
            "description": "Weight for contributors matching in overall accuracy calculation (PRD Section 6: 20%)",
            "parameter_type": ParameterType.THRESHOLD,
            "data_type": "float",
            "default_value": 0.20,
            "validation_rules": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "impact_level": "high",
        },
        "accuracy.media_links_weight": {
            "name": "Media Links Accuracy Weight",
            "description": "Weight for media links matching in overall accuracy calculation (PRD Section 6: 10%)",
            "parameter_type": ParameterType.THRESHOLD,
            "data_type": "float",
            "default_value": 0.10,
            "validation_rules": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "impact_level": "high",
        },
        "accuracy.wer_threshold": {
            "name": "Word Error Rate Threshold",
            "description": "Maximum WER for body text to be considered accurate (PRD Section 6: < 0.1%)",
            "parameter_type": ParameterType.THRESHOLD,
            "data_type": "float",
            "default_value": 0.001,
            "validation_rules": {"type": "number", "minimum": 0.0, "maximum": 0.1},
            "impact_level": "critical",
        },
    },
    # Drift detection parameters
    "drift": {
        "drift.window_size": {
            "name": "Drift Detection Window Size",
            "description": "Number of recent evaluations to consider for drift detection (rolling window)",
            "parameter_type": ParameterType.PROCESSING_CONFIG,
            "data_type": "integer",
            "default_value": 10,
            "validation_rules": {"type": "integer", "minimum": 5, "maximum": 50},
            "impact_level": "medium",
        },
        "drift.threshold": {
            "name": "Drift Detection Threshold",
            "description": "Accuracy drop threshold to trigger drift detection (5% default)",
            "parameter_type": ParameterType.THRESHOLD,
            "data_type": "float",
            "default_value": 0.05,
            "validation_rules": {"type": "number", "minimum": 0.01, "maximum": 0.5},
            "impact_level": "high",
        },
        "drift.alert_threshold": {
            "name": "Drift Alert Threshold",
            "description": "Accuracy drop threshold to trigger alerts (10% default)",
            "parameter_type": ParameterType.THRESHOLD,
            "data_type": "float",
            "default_value": 0.10,
            "validation_rules": {"type": "number", "minimum": 0.01, "maximum": 0.5},
            "impact_level": "high",
        },
        "drift.auto_tuning_threshold": {
            "name": "Auto-Tuning Threshold",
            "description": "Accuracy drop threshold to trigger automatic tuning (15% default)",
            "parameter_type": ParameterType.THRESHOLD,
            "data_type": "float",
            "default_value": 0.15,
            "validation_rules": {"type": "number", "minimum": 0.05, "maximum": 0.5},
            "impact_level": "critical",
        },
        "drift.confidence_level": {
            "name": "Statistical Confidence Level",
            "description": "Confidence level for statistical significance testing",
            "parameter_type": ParameterType.THRESHOLD,
            "data_type": "float",
            "default_value": 0.95,
            "validation_rules": {"type": "number", "minimum": 0.8, "maximum": 0.99},
            "impact_level": "medium",
        },
        "drift.baseline_lookback_days": {
            "name": "Baseline Lookback Days",
            "description": "Number of days to look back for baseline accuracy calculation",
            "parameter_type": ParameterType.PROCESSING_CONFIG,
            "data_type": "integer",
            "default_value": 30,
            "validation_rules": {"type": "integer", "minimum": 7, "maximum": 90},
            "impact_level": "medium",
        },
        "drift.enable_statistical_tests": {
            "name": "Enable Statistical Significance Testing",
            "description": "Whether to perform statistical significance tests for drift detection",
            "parameter_type": ParameterType.FEATURE_FLAG,
            "data_type": "boolean",
            "default_value": True,
            "impact_level": "medium",
        },
    },
    # Processing and performance parameters
    "processing": {
        "processing.batch_size": {
            "name": "Batch Processing Size",
            "description": "Number of documents to process in a single batch",
            "parameter_type": ParameterType.PROCESSING_CONFIG,
            "data_type": "integer",
            "default_value": 32,
            "validation_rules": {"type": "integer", "minimum": 1, "maximum": 100},
            "impact_level": "medium",
        },
        "processing.timeout_seconds": {
            "name": "Processing Timeout",
            "description": "Maximum time in seconds to wait for document processing",
            "parameter_type": ParameterType.PROCESSING_CONFIG,
            "data_type": "integer",
            "default_value": 300,
            "validation_rules": {"type": "integer", "minimum": 30, "maximum": 3600},
            "impact_level": "medium",
        },
        "processing.max_retries": {
            "name": "Maximum Retries",
            "description": "Maximum number of retry attempts for failed processing",
            "parameter_type": ParameterType.PROCESSING_CONFIG,
            "data_type": "integer",
            "default_value": 3,
            "validation_rules": {"type": "integer", "minimum": 0, "maximum": 10},
            "impact_level": "low",
        },
        "processing.retry_delay": {
            "name": "Retry Delay Seconds",
            "description": "Delay in seconds between retry attempts",
            "parameter_type": ParameterType.PROCESSING_CONFIG,
            "data_type": "float",
            "default_value": 2.0,
            "validation_rules": {"type": "number", "minimum": 0.1, "maximum": 60.0},
            "impact_level": "low",
        },
        "processing.max_workers": {
            "name": "Maximum Worker Threads",
            "description": "Maximum number of worker threads for parallel processing",
            "parameter_type": ParameterType.PROCESSING_CONFIG,
            "data_type": "integer",
            "default_value": 4,
            "validation_rules": {"type": "integer", "minimum": 1, "maximum": 16},
            "impact_level": "medium",
        },
    },
    # Model configuration parameters
    "model": {
        "model.extraction_confidence_threshold": {
            "name": "Extraction Confidence Threshold",
            "description": "Minimum confidence score for accepting extraction results",
            "parameter_type": ParameterType.THRESHOLD,
            "data_type": "float",
            "default_value": 0.75,
            "validation_rules": {"type": "number", "minimum": 0.1, "maximum": 0.99},
            "impact_level": "high",
        },
        "model.ocr_dpi": {
            "name": "OCR Resolution DPI",
            "description": "DPI resolution for OCR processing",
            "parameter_type": ParameterType.MODEL_CONFIG,
            "data_type": "integer",
            "default_value": 300,
            "validation_rules": {"type": "integer", "minimum": 150, "maximum": 600},
            "impact_level": "medium",
        },
        "model.max_image_size": {
            "name": "Maximum Image Size",
            "description": "Maximum image dimensions for processing (pixels)",
            "parameter_type": ParameterType.MODEL_CONFIG,
            "data_type": "integer",
            "default_value": 4096,
            "validation_rules": {"type": "integer", "minimum": 1024, "maximum": 8192},
            "impact_level": "medium",
        },
        "model.language_detection_threshold": {
            "name": "Language Detection Confidence",
            "description": "Minimum confidence for language detection",
            "parameter_type": ParameterType.THRESHOLD,
            "data_type": "float",
            "default_value": 0.8,
            "validation_rules": {"type": "number", "minimum": 0.5, "maximum": 0.95},
            "impact_level": "low",
        },
    },
    # UI and messaging parameters
    "ui": {
        "ui.error_message_template": {
            "name": "Error Message Template",
            "description": "Template for error messages shown to users",
            "parameter_type": ParameterType.PROMPT,
            "data_type": "string",
            "default_value": "Error: {error_type} - {error_message}. Please try again or contact support.",
            "validation_rules": {"type": "string", "minLength": 10, "maxLength": 500},
            "impact_level": "low",
        },
        "ui.success_message_template": {
            "name": "Success Message Template",
            "description": "Template for success messages shown to users",
            "parameter_type": ParameterType.PROMPT,
            "data_type": "string",
            "default_value": "Success: {operation} completed successfully. {details}",
            "validation_rules": {"type": "string", "minLength": 10, "maxLength": 500},
            "impact_level": "low",
        },
        "ui.pagination_default_size": {
            "name": "Default Pagination Size",
            "description": "Default number of items per page in paginated lists",
            "parameter_type": ParameterType.UI_CONFIG,
            "data_type": "integer",
            "default_value": 20,
            "validation_rules": {"type": "integer", "minimum": 5, "maximum": 100},
            "impact_level": "low",
        },
        "ui.pagination_max_size": {
            "name": "Maximum Pagination Size",
            "description": "Maximum allowed number of items per page",
            "parameter_type": ParameterType.UI_CONFIG,
            "data_type": "integer",
            "default_value": 100,
            "validation_rules": {"type": "integer", "minimum": 10, "maximum": 1000},
            "impact_level": "low",
        },
    },
    # Feature flags
    "feature": {
        "feature.drift_detection_enabled": {
            "name": "Drift Detection Enabled",
            "description": "Enable/disable drift detection functionality",
            "parameter_type": ParameterType.FEATURE_FLAG,
            "data_type": "boolean",
            "default_value": True,
            "impact_level": "high",
        },
        "feature.auto_tuning_enabled": {
            "name": "Auto-Tuning Enabled",
            "description": "Enable/disable automatic tuning when thresholds are breached",
            "parameter_type": ParameterType.FEATURE_FLAG,
            "data_type": "boolean",
            "default_value": True,
            "impact_level": "critical",
        },
        "feature.brand_overrides_enabled": {
            "name": "Brand Overrides Enabled",
            "description": "Enable/disable brand-specific parameter overrides",
            "parameter_type": ParameterType.FEATURE_FLAG,
            "data_type": "boolean",
            "default_value": True,
            "impact_level": "medium",
        },
        "feature.statistical_significance_enabled": {
            "name": "Statistical Significance Testing",
            "description": "Enable/disable statistical significance testing in drift detection",
            "parameter_type": ParameterType.FEATURE_FLAG,
            "data_type": "boolean",
            "default_value": True,
            "impact_level": "medium",
        },
        "feature.parameter_audit_logging_enabled": {
            "name": "Parameter Audit Logging",
            "description": "Enable/disable detailed audit logging of parameter changes",
            "parameter_type": ParameterType.FEATURE_FLAG,
            "data_type": "boolean",
            "default_value": True,
            "impact_level": "low",
        },
    },
    # System-level parameters
    "system": {
        "system.database_pool_size": {
            "name": "Database Connection Pool Size",
            "description": "Number of database connections to maintain in the pool",
            "parameter_type": ParameterType.PROCESSING_CONFIG,
            "data_type": "integer",
            "default_value": 10,
            "validation_rules": {"type": "integer", "minimum": 1, "maximum": 50},
            "impact_level": "medium",
            "requires_restart": True,
        },
        "system.cache_ttl_seconds": {
            "name": "Cache TTL Seconds",
            "description": "Time-to-live for cached parameter values",
            "parameter_type": ParameterType.PROCESSING_CONFIG,
            "data_type": "integer",
            "default_value": 300,
            "validation_rules": {"type": "integer", "minimum": 10, "maximum": 3600},
            "impact_level": "low",
        },
        "system.health_check_interval": {
            "name": "Health Check Interval",
            "description": "Interval in seconds between system health checks",
            "parameter_type": ParameterType.PROCESSING_CONFIG,
            "data_type": "integer",
            "default_value": 60,
            "validation_rules": {"type": "integer", "minimum": 10, "maximum": 300},
            "impact_level": "low",
        },
    },
}


# Brand-specific default overrides
BRAND_SPECIFIC_OVERRIDES = {
    "TechWeekly": {
        "brand.default_columns": {
            "name": "Default Column Count",
            "description": "Default number of columns for TechWeekly layout",
            "parameter_type": ParameterType.UI_CONFIG,
            "data_type": "integer",
            "default_value": 3,
            "scope": "brand_specific",
            "scope_identifier": "TechWeekly",
        },
        "brand.primary_color": {
            "name": "Primary Brand Color",
            "description": "Primary color for TechWeekly branding (RGB)",
            "parameter_type": ParameterType.UI_CONFIG,
            "data_type": "array",
            "default_value": [0.0, 0.4, 0.8],
            "scope": "brand_specific",
            "scope_identifier": "TechWeekly",
        },
        "accuracy.wer_threshold": {
            "name": "TechWeekly WER Threshold",
            "description": "Stricter WER threshold for technical content",
            "parameter_type": ParameterType.THRESHOLD,
            "data_type": "float",
            "default_value": 0.0005,  # Even stricter for tech content
            "scope": "brand_specific",
            "scope_identifier": "TechWeekly",
        },
    },
    "StyleMag": {
        "brand.default_columns": {
            "name": "Default Column Count",
            "description": "Default number of columns for StyleMag layout",
            "parameter_type": ParameterType.UI_CONFIG,
            "data_type": "integer",
            "default_value": 2,
            "scope": "brand_specific",
            "scope_identifier": "StyleMag",
        },
        "brand.primary_color": {
            "name": "Primary Brand Color",
            "description": "Primary color for StyleMag branding (RGB)",
            "parameter_type": ParameterType.UI_CONFIG,
            "data_type": "array",
            "default_value": [0.8, 0.2, 0.4],
            "scope": "brand_specific",
            "scope_identifier": "StyleMag",
        },
        "brand.image_frequency": {
            "name": "Image Frequency",
            "description": "Expected frequency of images per 100 words for fashion content",
            "parameter_type": ParameterType.THRESHOLD,
            "data_type": "float",
            "default_value": 0.6,
            "scope": "brand_specific",
            "scope_identifier": "StyleMag",
        },
    },
    "NewsToday": {
        "brand.default_columns": {
            "name": "Default Column Count",
            "description": "Default number of columns for NewsToday layout",
            "parameter_type": ParameterType.UI_CONFIG,
            "data_type": "integer",
            "default_value": 4,
            "scope": "brand_specific",
            "scope_identifier": "NewsToday",
        },
        "brand.primary_color": {
            "name": "Primary Brand Color",
            "description": "Primary color for NewsToday branding (RGB)",
            "parameter_type": ParameterType.UI_CONFIG,
            "data_type": "array",
            "default_value": [0.6, 0.0, 0.0],
            "scope": "brand_specific",
            "scope_identifier": "NewsToday",
        },
        "processing.batch_size": {
            "name": "News Processing Batch Size",
            "description": "Larger batch size for high-volume news processing",
            "parameter_type": ParameterType.PROCESSING_CONFIG,
            "data_type": "integer",
            "default_value": 50,
            "scope": "brand_specific",
            "scope_identifier": "NewsToday",
        },
    },
}


def get_all_default_parameters() -> Dict[str, Any]:
    """Get all default parameters flattened into a single dictionary."""
    all_params = {}

    for category, params in DEFAULT_PARAMETERS.items():
        all_params.update(params)

    return all_params


def get_brand_overrides_for_brand(brand_name: str) -> Dict[str, Any]:
    """Get brand-specific parameter overrides for a specific brand."""
    return BRAND_SPECIFIC_OVERRIDES.get(brand_name, {})


def get_parameters_by_category(category: str) -> Dict[str, Any]:
    """Get all default parameters for a specific category."""
    return DEFAULT_PARAMETERS.get(category, {})


def get_critical_parameters() -> List[str]:
    """Get list of parameter keys that have critical impact level."""
    critical_params = []

    for category, params in DEFAULT_PARAMETERS.items():
        for key, config in params.items():
            if config.get("impact_level") == "critical":
                critical_params.append(key)

    return critical_params


def get_parameters_requiring_restart() -> List[str]:
    """Get list of parameter keys that require system restart when changed."""
    restart_params = []

    for category, params in DEFAULT_PARAMETERS.items():
        for key, config in params.items():
            if config.get("requires_restart", False):
                restart_params.append(key)

    return restart_params
