"""
Magazine Extraction Evaluation Service.

This package provides a comprehensive FastAPI service for evaluating
magazine extraction accuracy, detecting drift, and triggering auto-tuning.
"""

from .main import app
from .evaluation_service import EvaluationService
from .drift_detector import DriftDetector, DriftDetectionConfig
from .models import (
    EvaluationRun, DocumentEvaluation, ArticleEvaluation,
    DriftDetection, AutoTuningEvent, SystemHealth
)
from .schemas import (
    ManualEvaluationRequest, BatchEvaluationRequest,
    EvaluationRunResponse, DocumentEvaluationResponse,
    DriftDetectionResponse, AutoTuningEventResponse
)

__version__ = "1.0.0"

__all__ = [
    "app",
    "EvaluationService", 
    "DriftDetector",
    "DriftDetectionConfig",
    "EvaluationRun",
    "DocumentEvaluation", 
    "ArticleEvaluation",
    "DriftDetection",
    "AutoTuningEvent",
    "SystemHealth",
    "ManualEvaluationRequest",
    "BatchEvaluationRequest",
    "EvaluationRunResponse",
    "DocumentEvaluationResponse",
    "DriftDetectionResponse",
    "AutoTuningEventResponse"
]