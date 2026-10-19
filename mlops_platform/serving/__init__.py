"""
Serving Package
===============
FastAPI model inference microservice, Pydantic schemas, and SHAP explainability.
"""

from mlops_platform.serving.app import app
from mlops_platform.serving.explainer import SHAPExplainer
from mlops_platform.serving.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    FeatureExplanation,
    LivenessResponse,
    PredictionResponse,
    ReadinessResponse,
)

__all__ = [
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    "FeatureExplanation",
    "LivenessResponse",
    "PredictionResponse",
    "ReadinessResponse",
    "SHAPExplainer",
    "app",
]
