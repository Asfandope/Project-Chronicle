"""
Magazine Extraction Evaluation Service.

This package provides a comprehensive FastAPI service for evaluating
magazine extraction accuracy, detecting drift, and triggering auto-tuning.
"""

from .drift_detector import DriftDetectionConfig, DriftDetector
from .evaluation_service import EvaluationService
from .main import app
from .models import (
    ArticleEvaluation,
    AutoTuningEvent,
    DocumentEvaluation,
    DriftDetection,
    EvaluationRun,
    SystemHealth,
)
from .schemas import (
    AutoTuningEventResponse,
    BatchEvaluationRequest,
    DocumentEvaluationResponse,
    DriftDetectionResponse,
    EvaluationRunResponse,
    ManualEvaluationRequest,
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
    "AutoTuningEventResponse",
]
