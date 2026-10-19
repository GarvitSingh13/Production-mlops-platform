"""
Stage 2 Verification Runner: MLflow Experiment Tracking & Training Pipeline
===========================================================================
Executes and verifies:
  1. Ingestion of Stage 1 preprocessed Parquet splits.
  2. Multi-model candidate training (Logistic Regression, Random Forest, Gradient Boosting).
  3. Metric evaluation & visual plot artifact generation (ROC, PR, Confusion Matrix, Feature Importance).
  4. Full MLflow experiment tracking (parameters, metrics, plots, model signatures).
  5. Automated Champion model selection and local serialization to models/champion/.
  6. Programmatic execution of pytest test suite for Stage 2.
"""

import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles to prevent cp1252 encoding crashes
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import mlflow
from mlops_platform.config.settings import settings
from mlops_platform.models.train import ModelTrainer


def print_banner(text: str, char: str = "="):
    line = char * 78
    print(f"\n{line}")
    print(f" {text}")
    print(f"{line}")


def main():
    print_banner("STAGE 2: MLFLOW EXPERIMENT TRACKING & TRAINING PIPELINE")

    # ---------------------------------------------------------
    # Step 1: Verify Stage 1 Data Splits Exist
    # ---------------------------------------------------------
    print("\n[Step 1/5] Checking Preprocessed Data Splits...")
    for path in [settings.TRAIN_DATA_PATH, settings.VAL_DATA_PATH, settings.TEST_DATA_PATH]:
        assert path.exists(), f"Missing split: {path}. Please run Stage 1 first."
        print(f"  [OK] Found {path.name:<20} ({path.stat().st_size / 1024:6.1f} KB)")

    # ---------------------------------------------------------
    # Step 2: Initialize MLflow Trainer
    # ---------------------------------------------------------
    print("\n[Step 2/5] Initializing MLflow Tracking Server...")
    trainer = ModelTrainer()
    print(f"  [OK] MLflow Tracking URI: {settings.MLFLOW_TRACKING_URI}")
    print(f"  [OK] MLflow Experiment:   {settings.MLFLOW_EXPERIMENT_NAME}")

    # ---------------------------------------------------------
    # Step 3: Train Candidates & Log to MLflow
    # ---------------------------------------------------------
    print("\n[Step 3/5] Training Candidate Models & Logging Artifacts to MLflow...")
    summary = trainer.run()
    results = summary["results"]
    champion = summary["champion"]

    print("\n  [Results Summary Table]")
    print("  " + "-" * 74)
    print(f"  {'Algorithm':<20} | {'Val ROC-AUC':<11} | {'Val F1':<8} | {'Test ROC-AUC':<12} | {'p95 Latency':<11}")
    print("  " + "-" * 74)
    for res in results:
        is_champ = "🏆 CHAMPION" if res["model_name"] == champion["model_name"] else ""
        print(
            f"  {res['model_name']:<20} | {res['val_roc_auc']:11.4f} | {res['val_f1']:8.4f} | "
            f"{res['test_roc_auc']:12.4f} | {res['p95_latency_ms']:8.2f} ms {is_champ}"
        )
    print("  " + "-" * 74)

    # ---------------------------------------------------------
    # Step 4: Verify Champion Artifacts & MLflow Records
    # ---------------------------------------------------------
    print("\n[Step 4/5] Verifying Champion Model Artifacts...")
    champion_model_file = settings.MODELS_DIR / "champion" / "model.joblib"
    champion_meta_file = settings.MODELS_DIR / "champion" / "metadata.json"

    assert champion_model_file.exists(), "Missing champion model binary!"
    assert champion_meta_file.exists(), "Missing champion metadata JSON!"
    print(f"  [OK] Champion model serialized to: {champion_model_file}")
    print(f"  [OK] Champion metadata saved to:   {champion_meta_file}")

    # Verify MLflow recorded the run
    client = mlflow.tracking.MlflowClient()
    champ_run = client.get_run(champion["run_id"])
    assert champ_run.info.run_id == champion["run_id"]
    print(f"  [OK] MLflow Run ID verified:       {champion['run_id']}")
    print(f"  [OK] MLflow candidate_champion:    {champ_run.data.tags.get('candidate_champion')}")

    # ---------------------------------------------------------
    # Step 5: Run Automated Pytest Test Suite
    # ---------------------------------------------------------
    print("\n[Step 5/5] Executing Pytest Test Suite for Stage 2...")
    test_file = str(PROJECT_ROOT / "tests" / "test_stage2_training.py")
    exit_code = pytest.main(["-v", test_file])
    assert exit_code == 0, f"Pytest suite failed with exit code: {exit_code}"

    # ---------------------------------------------------------
    # Stage 2 Summary Report
    # ---------------------------------------------------------
    print_banner("STAGE 2 COMPLETE: ALL VERIFICATION CHECKS PASSED [100%]")
    print(f"Champion Algorithm:       {champion['model_name']}")
    print(f"Champion Run ID:          {champion['run_id']}")
    print(f"Validation ROC-AUC:       {champion['val_roc_auc']:.4f}")
    print(f"Validation F1-Score:      {champion['val_f1']:.4f}")
    print(f"Test ROC-AUC:             {champion['test_roc_auc']:.4f}")
    print(f"Inference Latency (p95):  {champion['p95_latency_ms']:.2f} ms")
    print(f"MLflow Tracking DB:       {settings.MLFLOW_TRACKING_URI}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
