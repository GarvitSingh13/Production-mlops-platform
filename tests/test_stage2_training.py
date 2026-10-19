"""
Unit & Integration Tests for Stage 2: Model Training & MLflow Experiment Tracking
================================================================================
Verifies metrics computation, artifact plotting, MLflow experiment logging,
and champion model selection logic.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import mlflow

from mlops_platform.config.settings import settings
from mlops_platform.models.evaluator import ModelEvaluator, EvaluationResult
from mlops_platform.models.train import ModelTrainer


class TestModelEvaluator:
    """Tests for classification evaluation and visual artifact generation."""

    def test_metrics_calculation_sanity(self):
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 0, 0, 1])  # 7 correct, 1 FN
        y_prob = np.array([0.1, 0.2, 0.85, 0.9, 0.15, 0.45, 0.25, 0.95])

        result = ModelEvaluator.evaluate(y_true, y_pred, y_prob)

        assert 0.0 <= result.accuracy <= 1.0
        assert 0.0 <= result.precision <= 1.0
        assert 0.0 <= result.recall <= 1.0
        assert 0.0 <= result.f1 <= 1.0
        assert 0.5 <= result.roc_auc <= 1.0
        assert result.true_negatives == 4
        assert result.false_positives == 0
        assert result.false_negatives == 1
        assert result.true_positives == 3

    def test_artifact_plots_generation(self, tmp_path):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
        features = ["feature_a", "feature_b", "feature_c"]
        importances = np.array([0.1, 0.6, 0.3])
        cm_dict = {"true_negatives": 3, "false_positives": 0, "false_negatives": 0, "true_positives": 3}

        roc_file = tmp_path / "roc.png"
        pr_file = tmp_path / "pr.png"
        cm_file = tmp_path / "cm.png"
        fi_file = tmp_path / "fi.png"

        ModelEvaluator.plot_roc_curve(y_true, y_prob, roc_file)
        ModelEvaluator.plot_precision_recall_curve(y_true, y_prob, pr_file)
        ModelEvaluator.plot_confusion_matrix(cm_dict, cm_file)
        ModelEvaluator.plot_feature_importance(features, importances, fi_file)

        for plot_file in [roc_file, pr_file, cm_file, fi_file]:
            assert plot_file.exists()
            assert plot_file.stat().st_size > 1000  # Non-trivial PNG file


class TestModelTrainingAndMLflow:
    """Tests for model training execution and MLflow experiment logging."""

    def test_candidate_models_definition(self):
        trainer = ModelTrainer()
        candidates = trainer.get_candidate_models()

        assert "LogisticRegression" in candidates
        assert "RandomForest" in candidates
        assert "GradientBoosting" in candidates

        for name, cfg in candidates.items():
            assert "estimator" in cfg
            assert "params" in cfg
            assert "type" in cfg

    def test_train_and_log_single_model(self, tmp_path):
        trainer = ModelTrainer(experiment_name="test_mlops_experiment")
        candidates = trainer.get_candidate_models()

        # Create minimal synthetic data
        rng = np.random.default_rng(42)
        X = pd.DataFrame({f"f_{i}": rng.normal(size=120) for i in range(5)})
        y = rng.integers(0, 2, size=120)

        X_train, y_train = X.iloc[:80], y[:80]
        X_val, y_val = X.iloc[80:100], y[80:100]
        X_test, y_test = X.iloc[100:], y[100:]

        log_res = trainer.train_and_log_model(
            "LogisticRegression",
            candidates["LogisticRegression"],
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            tmp_path,
        )

        assert "run_id" in log_res
        assert log_res["val_roc_auc"] >= 0.0
        assert log_res["p95_latency_ms"] >= 0.0

        # Query MLflow to verify run actually exists
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(log_res["run_id"])
        assert run.info.run_id == log_res["run_id"]
        assert "val_roc_auc" in run.data.metrics
        assert "p95_latency_ms" in run.data.metrics
        assert run.data.tags.get("algorithm") == "LogisticRegression"
