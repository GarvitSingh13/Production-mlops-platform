"""
Unit & Integration Tests for Stage 3: Model Registry & ONNX Acceleration
========================================================================
Verifies MLflow Model Registry operations, quality gates, stage transitions,
ONNX graph compilation, and numerical parity checks.
"""

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from mlops_platform.config.settings import settings
from mlops_platform.models.onnx_converter import ONNXConverter
from mlops_platform.models.registry import ModelGovernanceError, ModelRegistryManager


class TestModelRegistry:
    """Tests for MLflow Model Registry governance and promotion gates."""

    @pytest.fixture
    def setup_mock_run(self):
        """Create a mock MLflow run with known metrics for testing."""
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment("test_registry_experiment")

        model = LogisticRegression().fit([[1, 2], [3, 4], [5, 6], [7, 8]], [0, 0, 1, 1])

        with mlflow.start_run(run_name="mock_candidate") as run:
            run_id = run.info.run_id
            mlflow.log_metrics({
                "val_roc_auc": 0.88,
                "test_roc_auc": 0.84,
                "p95_latency_ms": 1.2,
            })
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                serialization_format="cloudpickle",
            )
        return run_id, model

    def test_quality_gate_pass(self, setup_mock_run):
        run_id, _ = setup_mock_run
        manager = ModelRegistryManager(model_name="TestRegistryModel")
        gate_res = manager.evaluate_quality_gate(run_id)

        assert gate_res["all_passed"] is True
        assert gate_res["checks"]["val_roc_auc"]["passed"] is True
        assert gate_res["checks"]["test_roc_auc"]["passed"] is True
        assert gate_res["checks"]["p95_latency_ms"]["passed"] is True

    def test_quality_gate_rejection_on_low_metric(self):
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment("test_registry_experiment")

        with mlflow.start_run(run_name="poor_candidate") as run:
            run_id = run.info.run_id
            mlflow.log_metrics({
                "val_roc_auc": 0.62,  # Below 0.80 threshold
                "test_roc_auc": 0.60,
                "p95_latency_ms": 50.0,  # Above 25ms threshold
            })

        manager = ModelRegistryManager(model_name="TestRegistryModel")
        gate_res = manager.evaluate_quality_gate(run_id)

        assert gate_res["all_passed"] is False
        assert gate_res["checks"]["val_roc_auc"]["passed"] is False

        with pytest.raises(ModelGovernanceError):
            manager.promote_to_production(version="1", run_id=run_id)

    def test_registration_and_production_promotion(self, setup_mock_run):
        run_id, _ = setup_mock_run
        model_name = "TestPromotionModel"
        manager = ModelRegistryManager(model_name=model_name)

        # 1. Register Model
        v1 = manager.register_model_from_run(run_id, description="Initial candidate test model")
        assert v1.name == model_name

        # 2. Promote to Production
        promoted = manager.promote_to_production(version=v1.version, run_id=run_id)
        assert promoted["stage"] == "Production"

        # 3. Verify Latest Production
        latest = manager.get_latest_production_model()
        assert latest is not None
        assert str(latest["version"]) == str(v1.version)
        assert latest["stage"] == "Production"


class TestONNXAcceleration:
    """Tests for ONNX compilation, loading, and parity verification."""

    def test_onnx_conversion_and_numerical_parity(self, tmp_path):
        # 1. Fit a test model
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.normal(size=(100, 8)), columns=[f"feat_{i}" for i in range(8)])
        y = rng.integers(0, 2, size=100)

        model = LogisticRegression().fit(X, y)

        # 2. Convert to ONNX
        converter = ONNXConverter(output_dir=tmp_path)
        onnx_file = converter.convert(model, n_features=8)
        assert onnx_file.exists()
        assert onnx_file.stat().st_size > 200

        # 3. Benchmark and verify numerical parity
        res = converter.benchmark_and_verify(model, X, n_iterations=50)

        assert res["numerical_parity_verified"] is True
        assert res["max_absolute_difference"] < 1e-4
        assert res["onnx_latency_p95_ms"] >= 0.0
        assert res["p95_speedup_factor"] > 0.0
