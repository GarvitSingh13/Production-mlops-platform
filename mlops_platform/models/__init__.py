"""
Models Package
==============
Model training, evaluation, experiment tracking, artifact generation,
Model Registry governance, and ONNX runtime acceleration.
"""

from mlops_platform.models.evaluator import EvaluationResult, ModelEvaluator
from mlops_platform.models.onnx_converter import ONNXConverter
from mlops_platform.models.registry import ModelGovernanceError, ModelRegistryManager
from mlops_platform.models.train import ModelTrainer, run_training_pipeline

__all__ = [
    "EvaluationResult",
    "ModelEvaluator",
    "ModelGovernanceError",
    "ModelRegistryManager",
    "ModelTrainer",
    "ONNXConverter",
    "run_training_pipeline",
]
