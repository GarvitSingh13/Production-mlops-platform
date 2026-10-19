"""
Model Training & MLflow Experiment Tracking Pipeline
====================================================
Trains baseline and advanced ensemble models, evaluates metrics, logs
hyperparameters, metrics, and visual artifacts to MLflow, and identifies
the candidate champion model.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import time
import json
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from mlops_platform.config.settings import settings
from mlops_platform.models.evaluator import ModelEvaluator, EvaluationResult


class ModelTrainer:
    """Orchestrates multi-algorithm model training, MLflow logging, and champion selection."""

    def __init__(
        self,
        tracking_uri: str = settings.MLFLOW_TRACKING_URI,
        experiment_name: str = settings.MLFLOW_EXPERIMENT_NAME,
        random_seed: int = settings.RANDOM_SEED,
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.random_seed = random_seed
        self._setup_mlflow()

    def _setup_mlflow(self):
        """Initialize MLflow tracking server and experiment."""
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def load_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load preprocessed Parquet splits."""
        if not settings.TRAIN_DATA_PATH.exists():
            raise FileNotFoundError(
                f"Training data not found at {settings.TRAIN_DATA_PATH}. Please run Stage 1 first."
            )
        train_df = pd.read_parquet(settings.TRAIN_DATA_PATH)
        val_df = pd.read_parquet(settings.VAL_DATA_PATH)
        test_df = pd.read_parquet(settings.TEST_DATA_PATH)
        return train_df, val_df, test_df

    def get_candidate_models(self) -> Dict[str, Dict[str, Any]]:
        """Define candidate models with their respective hyperparameters."""
        return {
            "LogisticRegression": {
                "estimator": LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    random_state=self.random_seed,
                    solver="lbfgs",
                ),
                "params": {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
                "type": "linear",
            },
            "RandomForest": {
                "estimator": RandomForestClassifier(
                    n_estimators=100,
                    max_depth=8,
                    min_samples_split=4,
                    random_state=self.random_seed,
                    n_jobs=-1,
                ),
                "params": {"n_estimators": 100, "max_depth": 8, "min_samples_split": 4},
                "type": "ensemble_bagging",
            },
            "GradientBoosting": {
                "estimator": GradientBoostingClassifier(
                    n_estimators=120,
                    learning_rate=0.08,
                    max_depth=4,
                    min_samples_split=4,
                    random_state=self.random_seed,
                ),
                "params": {"n_estimators": 120, "learning_rate": 0.08, "max_depth": 4},
                "type": "ensemble_boosting",
            },
        }

    def train_and_log_model(
        self,
        model_name: str,
        config: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        artifact_temp_dir: Path,
    ) -> Dict[str, Any]:
        """Train a single model, compute metrics, and log everything to MLflow."""
        estimator = config["estimator"]
        params = config["params"]

        # 1. Fit Model and Measure Latency
        start_train = time.perf_counter()
        estimator.fit(X_train, y_train)
        train_time_sec = time.perf_counter() - start_train

        # Measure Single-Sample Inference Latency (p95 over 100 samples)
        latencies_ms = []
        sample_rows = X_val.iloc[:100]
        for i in range(len(sample_rows)):
            row_df = sample_rows.iloc[[i]]  # Retain DataFrame with feature names
            t0 = time.perf_counter()
            _ = estimator.predict_proba(row_df)
            latencies_ms.append((time.perf_counter() - t0) * 1000)
        p95_latency_ms = float(np.percentile(latencies_ms, 95))

        # 2. Evaluate Performance on All Splits
        val_preds = estimator.predict(X_val)
        val_probs = estimator.predict_proba(X_val)[:, 1]
        val_res = ModelEvaluator.evaluate(y_val, val_preds, val_probs)

        test_preds = estimator.predict(X_test)
        test_probs = estimator.predict_proba(X_test)[:, 1]
        test_res = ModelEvaluator.evaluate(y_test, test_preds, test_probs)

        # 3. Generate Artifact Plots
        model_plot_dir = artifact_temp_dir / model_name
        model_plot_dir.mkdir(parents=True, exist_ok=True)

        roc_path = model_plot_dir / "roc_curve.png"
        ModelEvaluator.plot_roc_curve(y_val, val_probs, roc_path, title=f"ROC Curve - {model_name}")

        pr_path = model_plot_dir / "pr_curve.png"
        ModelEvaluator.plot_precision_recall_curve(y_val, val_probs, pr_path)

        cm_path = model_plot_dir / "confusion_matrix.png"
        ModelEvaluator.plot_confusion_matrix(val_res.to_dict(), cm_path)

        # Feature Importance (if supported by model)
        feature_importance_path = None
        if hasattr(estimator, "feature_importances_"):
            feature_importance_path = model_plot_dir / "feature_importance.png"
            ModelEvaluator.plot_feature_importance(
                X_train.columns.tolist(), estimator.feature_importances_, feature_importance_path
            )
        elif hasattr(estimator, "coef_"):
            feature_importance_path = model_plot_dir / "feature_importance.png"
            ModelEvaluator.plot_feature_importance(
                X_train.columns.tolist(), np.abs(estimator.coef_[0]), feature_importance_path
            )

        # 4. Log to MLflow
        with mlflow.start_run(run_name=f"{model_name}") as run:
            run_id = run.info.run_id

            # Log Tags & Params
            mlflow.set_tag("model_family", config["type"])
            mlflow.set_tag("algorithm", model_name)
            mlflow.log_params(params)
            mlflow.log_param("train_records", len(X_train))
            mlflow.log_param("feature_count", X_train.shape[1])

            # Log Metrics
            mlflow.log_metrics({
                "val_roc_auc": val_res.roc_auc,
                "val_f1": val_res.f1,
                "val_precision": val_res.precision,
                "val_recall": val_res.recall,
                "val_accuracy": val_res.accuracy,
                "val_log_loss": val_res.log_loss,
                "test_roc_auc": test_res.roc_auc,
                "test_f1": test_res.f1,
                "test_precision": test_res.precision,
                "test_recall": test_res.recall,
                "test_accuracy": test_res.accuracy,
                "train_time_sec": train_time_sec,
                "p95_latency_ms": p95_latency_ms,
            })

            # Log Plot Artifacts
            mlflow.log_artifact(str(roc_path), artifact_path="plots")
            mlflow.log_artifact(str(pr_path), artifact_path="plots")
            mlflow.log_artifact(str(cm_path), artifact_path="plots")
            if feature_importance_path and feature_importance_path.exists():
                mlflow.log_artifact(str(feature_importance_path), artifact_path="plots")

            # Model Signature & Model Artifact
            signature = infer_signature(X_val.head(5), estimator.predict(X_val.head(5)))
            input_example = X_val.head(1)

            # Use cloudpickle serialization format to avoid skops untrusted type security exceptions
            mlflow.sklearn.log_model(
                sk_model=estimator,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                serialization_format="cloudpickle",
            )

        return {
            "model_name": model_name,
            "run_id": run_id,
            "estimator": estimator,
            "val_roc_auc": val_res.roc_auc,
            "val_f1": val_res.f1,
            "val_precision": val_res.precision,
            "val_recall": val_res.recall,
            "test_roc_auc": test_res.roc_auc,
            "test_f1": test_res.f1,
            "p95_latency_ms": p95_latency_ms,
            "train_time_sec": train_time_sec,
        }

    def run(self) -> Dict[str, Any]:
        """Execute full training suite across all candidate models and pick champion."""
        train_df, val_df, test_df = self.load_splits()

        feature_cols = [c for c in train_df.columns if c != settings.TARGET_COLUMN]
        X_train, y_train = train_df[feature_cols], train_df[settings.TARGET_COLUMN].values
        X_val, y_val = val_df[feature_cols], val_df[settings.TARGET_COLUMN].values
        X_test, y_test = test_df[feature_cols], test_df[settings.TARGET_COLUMN].values

        models = self.get_candidate_models()
        temp_artifact_dir = settings.PROJECT_ROOT / "metrics" / "temp_artifacts"
        temp_artifact_dir.mkdir(parents=True, exist_ok=True)

        results: List[Dict[str, Any]] = []
        for name, config in models.items():
            print(f"  --> Training and evaluating candidate: {name}...")
            res = self.train_and_log_model(
                name, config, X_train, y_train, X_val, y_val, X_test, y_test, temp_artifact_dir
            )
            results.append(res)

        # Select Champion based on Validation ROC-AUC
        champion = max(results, key=lambda x: x["val_roc_auc"])

        # Tag Champion in MLflow
        with mlflow.start_run(run_id=champion["run_id"]):
            mlflow.set_tag("candidate_champion", "true")

        # Persist Champion locally to models/champion/ for quick local serving/reference
        champion_dir = settings.MODELS_DIR / "champion"
        champion_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(champion["estimator"], champion_dir / "model.joblib")
        champion_metadata = {
            "model_name": champion["model_name"],
            "run_id": champion["run_id"],
            "val_roc_auc": champion["val_roc_auc"],
            "val_f1": champion["val_f1"],
            "test_roc_auc": champion["test_roc_auc"],
            "p95_latency_ms": champion["p95_latency_ms"],
            "feature_names": feature_cols,
        }
        with open(champion_dir / "metadata.json", "w") as f:
            json.dump(champion_metadata, f, indent=2)

        return {
            "results": results,
            "champion": champion,
            "champion_metadata_path": champion_dir / "metadata.json",
        }


def run_training_pipeline() -> Dict[str, Any]:
    """Helper entrypoint to execute training pipeline."""
    trainer = ModelTrainer()
    return trainer.run()


if __name__ == "__main__":
    out = run_training_pipeline()
    champ = out["champion"]
    print(f"\n[OK] Training completed. Champion: {champ['model_name']} (Val ROC-AUC: {champ['val_roc_auc']:.4f})")
