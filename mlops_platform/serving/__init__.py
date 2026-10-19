"""
Serving Package
===============
FastAPI model inference microservice, Pydantic schemas, and SHAP explainability.
"""

from mlops_platform.serving.app import app
from mlops_platform.serving.schemas import (
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    FeatureExplanation,
    LivenessResponse,
    ReadinessResponse,
)
from mlops_platform.serving.explainer import SHAPExplainer

__all__ = [
    "app",
    "PredictionResponse",
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    "FeatureExplanation",
    "LivenessResponse",
    "ReadinessResponse",
    "SHAPExplainer",
]
