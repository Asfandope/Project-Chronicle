"""
Centralized Parameter Management System.

This package provides comprehensive parameter management with:
- Database-backed parameter storage with versioning
- Brand-specific overrides and scoping
- Rollback capabilities to previous versions or snapshots
- Migration tools for converting hardcoded values
- REST API for parameter management
"""

from .api import create_parameter_management_api, mount_parameter_management
from .migrator import (
    CodeAnalyzer,
    HardcodedValue,
    MigrationPlan,
    ParameterMigrator,
    analyze_and_migrate_codebase,
)
from .models import (
    Parameter,
    ParameterAuditLog,
    ParameterChangeRequest,
    ParameterOverride,
    ParameterScope,
    ParameterSnapshot,
    ParameterStatus,
    ParameterTemplate,
    ParameterType,
    ParameterVersion,
    create_parameter_snapshot,
    get_active_parameter_value,
    get_parameter_history,
)
from .service import (
    ParameterOverrideRequest,
    ParameterService,
    ParameterUpdateRequest,
    ParameterValidator,
    RollbackRequest,
    parameter_service,
)

# Global service instance
_parameter_service = None
_session_factory = None


def initialize_parameter_management(database_session_factory):
    """Initialize the parameter management system with database session factory."""
    global _parameter_service, _session_factory
    _session_factory = database_session_factory
    _parameter_service = ParameterService()


def get_parameter(
    key: str, brand: str = None, context: dict = None, default: any = None
):
    """
    Get a parameter value with brand and context support.

    This is the main function that should be used throughout the application
    to retrieve parameter values instead of hardcoded constants.

    Args:
        key: Parameter key (e.g., 'accuracy.drift_threshold')
        brand: Brand name for brand-specific overrides
        context: Additional context for conditional parameters
        default: Default value if parameter not found

    Returns:
        The resolved parameter value

    Examples:
        # Basic usage
        threshold = get_parameter('accuracy.drift_threshold')

        # Brand-specific
        threshold = get_parameter('accuracy.drift_threshold', brand='TechWeekly')

        # With default
        batch_size = get_parameter('processing.batch_size', default=32)
    """
    if not _parameter_service or not _session_factory:
        raise RuntimeError(
            "Parameter management not initialized. Call initialize_parameter_management() first."
        )

    try:
        session = _session_factory()
        try:
            return _parameter_service.get_parameter_value(session, key, brand, context)
        finally:
            session.close()
    except Exception:
        if default is not None:
            return default
        raise


def get_category_parameters(category: str, brand: str = None):
    """
    Get all parameters in a category as a dictionary.

    Args:
        category: Parameter category (e.g., 'accuracy', 'processing')
        brand: Brand name for brand-specific overrides

    Returns:
        Dictionary of parameter_key: value pairs
    """
    if not _parameter_service or not _session_factory:
        raise RuntimeError("Parameter management not initialized.")

    session = _session_factory()
    try:
        return _parameter_service.get_parameter_configuration(
            session=session, category=category, brand=brand, include_overrides=False
        )
    finally:
        session.close()


def create_parameter_snapshot(name: str, description: str = ""):
    """
    Create a snapshot of current parameter state.

    Args:
        name: Snapshot name
        description: Optional description

    Returns:
        Snapshot ID
    """
    if not _parameter_service or not _session_factory:
        raise RuntimeError("Parameter management not initialized.")

    session = _session_factory()
    try:
        snapshot = _parameter_service.create_snapshot(
            session=session, name=name, description=description, created_by="system"
        )
        return str(snapshot.id)
    finally:
        session.close()


# Pre-defined parameter keys to prevent typos and enable IDE auto-completion
class ParameterKeys:
    """Centralized parameter key definitions."""

    # Accuracy and scoring parameters
    ACCURACY_TITLE_WEIGHT = "accuracy.title_weight"
    ACCURACY_BODY_TEXT_WEIGHT = "accuracy.body_text_weight"
    ACCURACY_CONTRIBUTORS_WEIGHT = "accuracy.contributors_weight"
    ACCURACY_MEDIA_LINKS_WEIGHT = "accuracy.media_links_weight"
    ACCURACY_WER_THRESHOLD = "accuracy.wer_threshold"

    # Drift detection parameters
    DRIFT_WINDOW_SIZE = "drift.window_size"
    DRIFT_THRESHOLD = "drift.threshold"
    DRIFT_ALERT_THRESHOLD = "drift.alert_threshold"
    DRIFT_AUTO_TUNING_THRESHOLD = "drift.auto_tuning_threshold"
    DRIFT_CONFIDENCE_LEVEL = "drift.confidence_level"
    DRIFT_BASELINE_LOOKBACK_DAYS = "drift.baseline_lookback_days"

    # Processing parameters
    PROCESSING_BATCH_SIZE = "processing.batch_size"
    PROCESSING_TIMEOUT_SECONDS = "processing.timeout_seconds"
    PROCESSING_MAX_RETRIES = "processing.max_retries"
    PROCESSING_RETRY_DELAY = "processing.retry_delay"
    PROCESSING_MAX_WORKERS = "processing.max_workers"

    # Model configuration
    MODEL_EXTRACTION_CONFIDENCE_THRESHOLD = "model.extraction_confidence_threshold"
    MODEL_OCR_DPI = "model.ocr_dpi"
    MODEL_MAX_IMAGE_SIZE = "model.max_image_size"
    MODEL_LANGUAGE_DETECTION_THRESHOLD = "model.language_detection_threshold"

    # UI and messaging
    UI_ERROR_MESSAGE_TEMPLATE = "ui.error_message_template"
    UI_SUCCESS_MESSAGE_TEMPLATE = "ui.success_message_template"
    UI_PAGINATION_DEFAULT_SIZE = "ui.pagination_default_size"
    UI_PAGINATION_MAX_SIZE = "ui.pagination_max_size"

    # Feature flags
    FEATURE_DRIFT_DETECTION_ENABLED = "feature.drift_detection_enabled"
    FEATURE_AUTO_TUNING_ENABLED = "feature.auto_tuning_enabled"
    FEATURE_BRAND_OVERRIDES_ENABLED = "feature.brand_overrides_enabled"
    FEATURE_STATISTICAL_SIGNIFICANCE_ENABLED = (
        "feature.statistical_significance_enabled"
    )

    # Brand-specific configurations
    BRAND_DEFAULT_COLUMNS = "brand.default_columns"
    BRAND_PRIMARY_COLOR = "brand.primary_color"
    BRAND_ACCENT_COLOR = "brand.accent_color"
    BRAND_TYPICAL_ARTICLE_LENGTH = "brand.typical_article_length"
    BRAND_IMAGE_FREQUENCY = "brand.image_frequency"


# Convenience functions for commonly used parameter categories
def get_accuracy_parameters(brand: str = None) -> dict:
    """Get all accuracy-related parameters."""
    return get_category_parameters("accuracy", brand)


def get_drift_parameters(brand: str = None) -> dict:
    """Get all drift detection parameters."""
    return get_category_parameters("drift", brand)


def get_processing_parameters(brand: str = None) -> dict:
    """Get all processing parameters."""
    return get_category_parameters("processing", brand)


def get_model_parameters(brand: str = None) -> dict:
    """Get all model configuration parameters."""
    return get_category_parameters("model", brand)


def get_feature_flags(brand: str = None) -> dict:
    """Get all feature flag parameters."""
    return get_category_parameters("feature", brand)


__version__ = "1.0.0"

__all__ = [
    # Core models
    "Parameter",
    "ParameterVersion",
    "ParameterOverride",
    "ParameterChangeRequest",
    "ParameterSnapshot",
    "ParameterAuditLog",
    "ParameterTemplate",
    # Enums
    "ParameterType",
    "ParameterScope",
    "ParameterStatus",
    # Service layer
    "ParameterService",
    "ParameterValidator",
    "ParameterUpdateRequest",
    "ParameterOverrideRequest",
    "RollbackRequest",
    "parameter_service",
    # API
    "create_parameter_management_api",
    "mount_parameter_management",
    # Migration tools
    "CodeAnalyzer",
    "ParameterMigrator",
    "HardcodedValue",
    "MigrationPlan",
    "analyze_and_migrate_codebase",
    # Main interface functions
    "initialize_parameter_management",
    "get_parameter",
    "get_category_parameters",
    "create_parameter_snapshot",
    # Parameter keys
    "ParameterKeys",
    # Convenience functions
    "get_accuracy_parameters",
    "get_drift_parameters",
    "get_processing_parameters",
    "get_model_parameters",
    "get_feature_flags",
    # Database utilities
    "get_active_parameter_value",
    "get_parameter_history",
    "create_parameter_snapshot",
]
