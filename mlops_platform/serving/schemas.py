"""
Serving Request & Response Schemas
==================================
Pydantic v2 schemas for single and batch predictions, SHAP explanations,
and Kubernetes liveness/readiness health probes.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from mlops_platform.data.contracts import InferencePayload


class FeatureExplanation(BaseModel):
    """Represents a single feature's SHAP attribution impact."""

    feature: str = Field(..., description="Feature name")
    shap_value: float = Field(..., description="Raw SHAP value contribution")
    impact_percentage: float = Field(..., description="Relative percentage contribution to prediction")
    direction: Literal["increases_risk", "decreases_risk"] = Field(
        ..., description="Whether this feature increased or decreased churn risk"
    )


class PredictionResponse(BaseModel):
    """Response payload for a single customer churn prediction."""

    customer_id: str
    churn_prediction: int = Field(..., description="Binary classification (1 = Churn, 0 = Retain)")
    churn_probability: float = Field(..., description="Predicted probability of churning [0.0, 1.0]")
    risk_tier: Literal["LOW", "MEDIUM", "HIGH"] = Field(..., description="Customer churn risk tier")
    engine_used: Literal["onnx", "sklearn"] = Field(..., description="Inference engine that computed prediction")
    model_version: str = Field(..., description="Active production model version")
    latency_ms: float = Field(..., description="Total inference compute time in milliseconds")
    explanations: Optional[List[FeatureExplanation]] = Field(
        None, description="Top driving SHAP feature attributions explaining the prediction"
    )


class BatchPredictionRequest(BaseModel):
    """Request payload for bulk inference."""

    customers: List[InferencePayload] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    """Response payload for bulk inference."""

    predictions: List[PredictionResponse]
    total_processed: int
    batch_latency_ms: float
    engine_used: str


class LivenessResponse(BaseModel):
    """Kubernetes liveness probe response."""

    status: str = "alive"
    timestamp: str


class ReadinessResponse(BaseModel):
    """Kubernetes readiness probe response."""

    status: str = "ready"
    model_name: str
    model_version: str
    sklearn_loaded: bool
    onnx_loaded: bool
    preprocessor_loaded: bool
    explainer_loaded: bool
