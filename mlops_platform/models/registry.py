"""
Model Governance & MLflow Model Registry Management
===================================================
Manages model registration, semantic versioning, automated quality gates,
and lifecycle stage transitions (None -> Staging -> Production -> Archived).
"""

from datetime import UTC, datetime
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from mlops_platform.config.settings import settings


class ModelGovernanceError(Exception):
    """Raised when a candidate model fails production quality gate thresholds."""


class ModelRegistryManager:
    """Manages model registry operations and promotion governance in MLflow."""

    def __init__(
        self,
        tracking_uri: str = settings.MLFLOW_TRACKING_URI,
        model_name: str = settings.MLFLOW_REGISTERED_MODEL_NAME,
        min_val_roc_auc: float = 0.80,
        min_test_roc_auc: float = 0.75,
        max_latency_p95_ms: float = 25.0,
    ):
        self.tracking_uri = tracking_uri
        self.model_name = model_name
        self.min_val_roc_auc = min_val_roc_auc
        self.min_test_roc_auc = min_test_roc_auc
        self.max_latency_p95_ms = max_latency_p95_ms

        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient()

    def register_model_from_run(self, run_id: str, description: str | None = None) -> Any:
        """
        Register a trained model from an MLflow run into the central Model Registry.

        Args:
            run_id: MLflow run identifier containing the logged model artifact.
            description: Optional human-readable description for this version.

        Returns:
            ModelVersion object representing the newly registered model version.
        """
        model_uri = f"runs:/{run_id}/model"
        print(f"  [INFO] Registering model from URI: {model_uri} as '{self.model_name}'...")

        model_version = mlflow.register_model(model_uri=model_uri, name=self.model_name)

        if description:
            self.client.update_model_version(
                name=self.model_name,
                version=model_version.version,
                description=description,
            )

        return model_version

    def evaluate_quality_gate(self, run_id: str) -> dict[str, Any]:
        """
        Verify that a candidate run passes all strict production deployment thresholds.

        Checks:
          1. Validation ROC-AUC >= min_val_roc_auc (0.80)
          2. Test ROC-AUC >= min_test_roc_auc (0.75)
          3. p95 Inference Latency <= max_latency_p95_ms (25ms)
        """
        run = self.client.get_run(run_id)
        metrics = run.data.metrics

        val_auc = metrics.get("val_roc_auc", 0.0)
        test_auc = metrics.get("test_roc_auc", 0.0)
        p95_latency = metrics.get("p95_latency_ms", 999.0)

        checks = {
            "val_roc_auc": {
                "value": val_auc,
                "threshold": self.min_val_roc_auc,
                "passed": val_auc >= self.min_val_roc_auc,
            },
            "test_roc_auc": {
                "value": test_auc,
                "threshold": self.min_test_roc_auc,
                "passed": test_auc >= self.min_test_roc_auc,
            },
            "p95_latency_ms": {
                "value": p95_latency,
                "threshold": self.max_latency_p95_ms,
                "passed": p95_latency <= self.max_latency_p95_ms,
            },
        }

        all_passed = all(check["passed"] for check in checks.values())
        return {
            "all_passed": all_passed,
            "checks": checks,
            "run_id": run_id,
        }

    def promote_to_production(self, version: str, run_id: str) -> dict[str, Any]:
        """
        Promotes a model version to 'Production' stage after verifying quality gates.
        Archives any previously active production version.
        """
        # 1. Enforce Quality Gate
        gate_result = self.evaluate_quality_gate(run_id)
        if not gate_result["all_passed"]:
            failed_checks = [k for k, v in gate_result["checks"].items() if not v["passed"]]
            raise ModelGovernanceError(
                f"Model version {version} failed deployment quality gates on: {failed_checks}. "
                f"Gate details: {gate_result['checks']}"
            )

        # 2. Archive Existing Production Versions
        existing_versions = self.client.search_model_versions(f"name='{self.model_name}'")
        for v in existing_versions:
            if v.current_stage == "Production" and v.version != str(version):
                print(f"  [INFO] Archiving previous production model version: {v.version}")
                self.client.transition_model_version_stage(
                    name=self.model_name,
                    version=v.version,
                    stage="Archived",
                    archive_existing_versions=False,
                )

        # 3. Transition New Model Version to Production
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=str(version),
            stage="Production",
            archive_existing_versions=False,
        )

        # 4. Set Metadata Tags
        self.client.set_model_version_tag(
            name=self.model_name,
            version=str(version),
            key="promoted_by",
            value="automated_ci_quality_gate",
        )
        self.client.set_model_version_tag(
            name=self.model_name,
            version=str(version),
            key="promoted_at",
            value=datetime.now(UTC).isoformat(),
        )

        return {
            "name": self.model_name,
            "version": str(version),
            "stage": "Production",
            "run_id": run_id,
            "gate_checks": gate_result["checks"],
        }

    def get_latest_production_model(self) -> dict[str, Any] | None:
        """Fetch current active Production model details."""
        versions = self.client.get_latest_versions(self.model_name, stages=["Production"])
        if not versions:
            return None
        latest = versions[0]
        return {
            "name": latest.name,
            "version": str(latest.version),
            "stage": latest.current_stage,
            "run_id": latest.run_id,
            "source": latest.source,
        }
