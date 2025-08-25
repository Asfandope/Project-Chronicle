"""
Pydantic schemas for the evaluation service API.

This module defines request/response models for all FastAPI endpoints,
including evaluation requests, accuracy results, and drift detection.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator, computed_field
from enum import Enum
import uuid


class EvaluationType(str, Enum):
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    BATCH = "batch"


class TriggerSource(str, Enum):
    DRIFT_DETECTION = "drift_detection"
    MANUAL_UPLOAD = "manual_upload"
    SCHEDULED = "scheduled"
    API_REQUEST = "api_request"


class MetricType(str, Enum):
    OVERALL = "overall"
    TITLE = "title"
    BODY_TEXT = "body_text"
    CONTRIBUTORS = "contributors"
    MEDIA_LINKS = "media_links"


class DriftStatus(str, Enum):
    NO_DRIFT = "no_drift"
    DRIFT_DETECTED = "drift_detected"
    ALERT_TRIGGERED = "alert_triggered"
    AUTO_TUNING_TRIGGERED = "auto_tuning_triggered"


class TuningStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# Request/Response schemas
class FieldAccuracySchema(BaseModel):
    """Schema for individual field accuracy metrics."""
    field_name: str
    correct: int = 0
    total: int = 0
    accuracy: float = 0.0
    accuracy_percentage: float = 0.0
    details: Dict[str, Any] = {}

    class Config:
        from_attributes = True


class ArticleAccuracySchema(BaseModel):
    """Schema for article-level accuracy results."""
    article_id: str
    title_accuracy: FieldAccuracySchema
    body_text_accuracy: FieldAccuracySchema
    contributors_accuracy: FieldAccuracySchema
    media_links_accuracy: FieldAccuracySchema
    weighted_overall_accuracy: float = 0.0
    
    class Config:
        from_attributes = True


class DocumentAccuracySchema(BaseModel):
    """Schema for document-level accuracy results."""
    document_id: str
    article_accuracies: List[ArticleAccuracySchema]
    overall_title_accuracy: FieldAccuracySchema
    overall_body_text_accuracy: FieldAccuracySchema
    overall_contributors_accuracy: FieldAccuracySchema
    overall_media_links_accuracy: FieldAccuracySchema
    document_weighted_accuracy: float = 0.0

    class Config:
        from_attributes = True


class ManualEvaluationRequest(BaseModel):
    """Request schema for manual evaluation uploads."""
    document_id: str = Field(..., description="Unique identifier for the document")
    ground_truth_content: str = Field(..., description="Ground truth XML content")
    extracted_content: str = Field(..., description="Extracted output XML content")
    
    # Optional metadata
    brand_name: Optional[str] = None
    complexity_level: Optional[str] = None
    edge_cases: Optional[List[str]] = []
    document_path: Optional[str] = None
    notes: Optional[str] = None
    
    # System version info
    extractor_version: Optional[str] = None
    model_version: Optional[str] = None

    @validator('ground_truth_content', 'extracted_content')
    def validate_xml_content(cls, v):
        if not v or not v.strip():
            raise ValueError("XML content cannot be empty")
        return v.strip()


class BatchEvaluationRequest(BaseModel):
    """Request schema for batch evaluation."""
    evaluation_name: str = Field(..., description="Name for this batch evaluation")
    documents: List[ManualEvaluationRequest] = Field(..., min_items=1, max_items=100)
    
    # Batch settings
    parallel_processing: bool = True
    fail_on_error: bool = False
    
    # Drift detection settings
    enable_drift_detection: bool = True
    drift_threshold: float = Field(0.05, ge=0.0, le=1.0, description="Drift detection threshold")


class EvaluationRunResponse(BaseModel):
    """Response schema for evaluation run results."""
    id: uuid.UUID
    created_at: datetime
    evaluation_type: EvaluationType
    trigger_source: Optional[TriggerSource] = None
    
    # Document counts
    document_count: int
    total_articles: int
    successful_extractions: int
    failed_extractions: int
    
    # Overall metrics
    overall_weighted_accuracy: float
    title_accuracy: float
    body_text_accuracy: float
    contributors_accuracy: float
    media_links_accuracy: float
    
    # Processing info
    processing_time_seconds: Optional[float] = None
    extractor_version: Optional[str] = None
    model_version: Optional[str] = None
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = Field(default={}, alias="evaluation_metadata")
    notes: Optional[str] = None

    class Config:
        from_attributes = True
        populate_by_name = True


class DocumentEvaluationResponse(BaseModel):
    """Response schema for individual document evaluation."""
    id: uuid.UUID
    evaluation_run_id: uuid.UUID
    created_at: datetime
    
    # Document info
    document_id: str
    brand_name: Optional[str] = None
    page_count: Optional[int] = None
    complexity_level: Optional[str] = None
    edge_cases: Optional[List[str]] = []
    
    # Accuracy metrics
    weighted_overall_accuracy: float
    title_accuracy: float
    body_text_accuracy: float
    contributors_accuracy: float
    media_links_accuracy: float
    
    # Field statistics
    title_correct: int
    title_total: int
    body_text_correct: int
    body_text_total: int
    contributors_correct: int
    contributors_total: int
    media_links_correct: int
    media_links_total: int
    
    # Processing info
    extraction_time_seconds: Optional[float] = None
    extraction_successful: bool = True
    extraction_error: Optional[str] = None

    class Config:
        from_attributes = True


class DriftDetectionResponse(BaseModel):
    """Response schema for drift detection results."""
    id: uuid.UUID
    evaluation_run_id: uuid.UUID
    created_at: datetime
    
    # Detection parameters
    window_size: int
    metric_type: MetricType
    
    # Metrics
    current_accuracy: float
    baseline_accuracy: float
    accuracy_drop: float
    
    # Thresholds
    drift_threshold: float
    alert_threshold: float
    
    # Status
    drift_detected: bool
    alert_triggered: bool
    auto_tuning_triggered: bool
    
    # Statistical data
    p_value: Optional[float] = None
    confidence_interval: Optional[List[float]] = None
    trend_direction: Optional[str] = None
    
    # Actions
    actions_triggered: Optional[List[str]] = []
    notification_sent: bool = False

    class Config:
        from_attributes = True


class AutoTuningEventResponse(BaseModel):
    """Response schema for auto-tuning events."""
    id: uuid.UUID
    created_at: datetime
    
    # Trigger info
    trigger_accuracy_drop: float
    trigger_metric_type: MetricType
    
    # Tuning details
    tuning_type: str
    tuning_parameters: Optional[Dict[str, Any]] = {}
    
    # Status
    status: TuningStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    pre_tuning_accuracy: Optional[float] = None
    post_tuning_accuracy: Optional[float] = None
    improvement: Optional[float] = None
    
    # Error info
    error_message: Optional[str] = None

    class Config:
        from_attributes = True


class DriftDetectionSettings(BaseModel):
    """Settings for drift detection configuration."""
    enabled: bool = True
    window_size: int = Field(10, ge=5, le=50, description="Rolling window size")
    
    # Thresholds
    drift_threshold: float = Field(0.05, ge=0.01, le=0.5, description="Drift detection threshold")
    alert_threshold: float = Field(0.10, ge=0.01, le=0.5, description="Alert threshold")
    
    # Metrics to monitor
    monitor_overall_accuracy: bool = True
    monitor_title_accuracy: bool = True
    monitor_body_text_accuracy: bool = True
    monitor_contributors_accuracy: bool = True
    monitor_media_links_accuracy: bool = False
    
    # Auto-tuning settings
    enable_auto_tuning: bool = True
    auto_tuning_threshold: float = Field(0.15, ge=0.05, le=0.5, description="Auto-tuning trigger threshold")
    
    @validator('alert_threshold')
    def alert_threshold_greater_than_drift(cls, v, values):
        drift_threshold = values.get('drift_threshold', 0.05)
        if v <= drift_threshold:
            raise ValueError('Alert threshold must be greater than drift threshold')
        return v
    
    @validator('auto_tuning_threshold')
    def auto_tuning_threshold_greater_than_alert(cls, v, values):
        alert_threshold = values.get('alert_threshold', 0.10)
        if v <= alert_threshold:
            raise ValueError('Auto-tuning threshold must be greater than alert threshold')
        return v


class SystemHealthResponse(BaseModel):
    """Response schema for system health metrics."""
    recorded_at: datetime
    period_start: datetime
    period_end: datetime
    
    # Volume metrics
    documents_processed: int
    articles_processed: int
    average_processing_time: Optional[float] = None
    
    # Accuracy metrics
    average_overall_accuracy: Optional[float] = None
    average_title_accuracy: Optional[float] = None
    average_body_text_accuracy: Optional[float] = None
    average_contributors_accuracy: Optional[float] = None
    average_media_links_accuracy: Optional[float] = None
    
    # Quality metrics
    extraction_success_rate: Optional[float] = None
    low_confidence_extractions: int = 0
    
    # Drift indicators
    drift_alerts_count: int = 0
    auto_tuning_events_count: int = 0
    
    # System performance
    average_response_time: Optional[float] = None
    error_rate: Optional[float] = None
    uptime_percentage: Optional[float] = None
    
    # Alerts
    critical_alerts_count: int = 0
    warning_alerts_count: int = 0

    class Config:
        from_attributes = True


class AccuracyTrendResponse(BaseModel):
    """Response schema for accuracy trend analysis."""
    metric_type: MetricType
    time_period_days: int
    
    # Data points
    data_points: List[Dict[str, Any]] = Field(..., description="Time series data points")
    
    # Statistical summary
    current_accuracy: float
    average_accuracy: float
    trend_slope: Optional[float] = None
    trend_direction: Optional[str] = None  # 'improving', 'declining', 'stable'
    
    # Variability
    std_deviation: Optional[float] = None
    min_accuracy: Optional[float] = None
    max_accuracy: Optional[float] = None
    
    # Recent performance
    last_7_days_average: Optional[float] = None
    last_30_days_average: Optional[float] = None


class ComparisonAnalysisRequest(BaseModel):
    """Request schema for comparative analysis."""
    baseline_period_start: datetime
    baseline_period_end: datetime
    comparison_period_start: datetime
    comparison_period_end: datetime
    
    # Filters
    brand_filter: Optional[str] = None
    complexity_filter: Optional[str] = None
    metric_types: List[MetricType] = [MetricType.OVERALL]


class ComparisonAnalysisResponse(BaseModel):
    """Response schema for comparative analysis results."""
    baseline_period: Dict[str, datetime]
    comparison_period: Dict[str, datetime]
    
    # Comparison results per metric
    metric_comparisons: Dict[str, Dict[str, Any]]
    
    # Overall summary
    overall_improvement: bool
    significant_changes: List[Dict[str, Any]]
    recommendations: List[str]


class ErrorResponse(BaseModel):
    """Standard error response schema."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None


class SuccessResponse(BaseModel):
    """Standard success response schema."""
    success: bool = True
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Validation schemas
class XMLValidationRequest(BaseModel):
    """Request schema for XML validation."""
    xml_content: str = Field(..., description="XML content to validate")
    xml_type: str = Field(..., pattern="^(ground_truth|extracted)$", description="Type of XML content")


class XMLValidationResponse(BaseModel):
    """Response schema for XML validation."""
    is_valid: bool
    validation_errors: List[str] = []
    element_count: Optional[int] = None
    article_count: Optional[int] = None


class HealthCheckResponse(BaseModel):
    """Health check response schema."""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str
    database_connected: bool
    redis_connected: bool = False
    external_services: Dict[str, bool] = {}


# Pagination schemas
class PaginationParams(BaseModel):
    """Parameters for paginated requests."""
    page: int = Field(1, ge=1, description="Page number (1-based)")
    page_size: int = Field(20, ge=1, le=100, description="Items per page")
    sort_by: Optional[str] = "created_at"
    sort_order: Optional[str] = Field("desc", pattern="^(asc|desc)$")


class PaginatedResponse(BaseModel):
    """Generic paginated response wrapper."""
    items: List[Any]
    total_count: int
    page: int
    page_size: int
    
    @computed_field
    @property
    def total_pages(self) -> int:
        return (self.total_count + self.page_size - 1) // self.page_size if self.page_size > 0 else 0
    
    @computed_field
    @property
    def has_next(self) -> bool:
        return self.page < self.total_pages
    
    @computed_field  
    @property
    def has_previous(self) -> bool:
        return self.page > 1