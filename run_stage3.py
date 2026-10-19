"""
Stage 3 Verification Runner: MLflow Model Registry & ONNX Acceleration
======================================================================
Executes and verifies:
  1. Ingestion of Stage 2 Champion model artifacts and MLflow run.
  2. Registration in central MLflow Model Registry.
  3. Automated Quality Gate enforcement (Validation ROC-AUC, Latency thresholds).
  4. Promotion to 'Production' stage and archival of previous versions.
  5. Model graph conversion to ONNX format and runtime latency benchmarking.
  6. Numerical probability parity verification (Scikit-learn vs. ONNX Runtime).
  7. Programmatic execution of pytest test suite for Stage 3.
"""

import sys
import json
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
import joblib
import pandas as pd
import mlflow
from mlops_platform.config.settings import settings
from mlops_platform.models.registry import ModelRegistryManager
from mlops_platform.models.onnx_converter import ONNXConverter


def print_banner(text: str, char: str = "="):
    line = char * 78
    print(f"\n{line}")
    print(f" {text}")
    print(f"{line}")


def main():
    print_banner("STAGE 3: MLFLOW MODEL REGISTRY, VERSIONING & ONNX ACCELERATION")

    # ---------------------------------------------------------
    # Step 1: Ingest Champion Artifacts from Stage 2
    # ---------------------------------------------------------
    print("\n[Step 1/6] Ingesting Stage 2 Champion Model Artifacts...")
    meta_path = settings.MODELS_DIR / "champion" / "metadata.json"
    model_path = settings.MODELS_DIR / "champion" / "model.joblib"

    assert meta_path.exists(), f"Missing champion metadata at {meta_path}. Please run Stage 2 first."
    assert model_path.exists(), f"Missing champion model at {model_path}. Please run Stage 2 first."

    with open(meta_path, "r") as f:
        champion_meta = json.load(f)

    champion_model = joblib.load(model_path)
    champion_run_id = champion_meta["run_id"]
    print(f"  [OK] Champion Algorithm: {champion_meta['model_name']}")
    print(f"  [OK] Champion Run ID:    {champion_run_id}")
    print(f"  [OK] Val ROC-AUC:        {champion_meta['val_roc_auc']:.4f}")

    # ---------------------------------------------------------
    # Step 2: Register Champion in MLflow Model Registry
    # ---------------------------------------------------------
    print("\n[Step 2/6] Registering Model in MLflow Model Registry...")
    registry_mgr = ModelRegistryManager(
        tracking_uri=settings.MLFLOW_TRACKING_URI,
        model_name=settings.MLFLOW_REGISTERED_MODEL_NAME,
        min_val_roc_auc=0.80,
        min_test_roc_auc=0.75,
        max_latency_p95_ms=25.0,
    )

    registered_version = registry_mgr.register_model_from_run(
        run_id=champion_run_id,
        description=f"Automated champion model ({champion_meta['model_name']}) with ROC-AUC {champion_meta['val_roc_auc']:.4f}",
    )
    print(f"  [OK] Registered Model Name:    {registered_version.name}")
    print(f"  [OK] Model Version Generated:  v{registered_version.version}")

    # ---------------------------------------------------------
    # Step 3: Run Quality Gate Checks
    # ---------------------------------------------------------
    print("\n[Step 3/6] Evaluating Production Quality Deployment Gates...")
    gate_res = registry_mgr.evaluate_quality_gate(champion_run_id)

    print("  [Gate Criteria Check Table]")
    print("  " + "-" * 70)
    print(f"  {'Metric':<20} | {'Observed':<12} | {'Threshold':<12} | {'Gate Status'}")
    print("  " + "-" * 70)
    for metric_name, info in gate_res["checks"].items():
        status = "PASSED [OK]" if info["passed"] else "FAILED [X]"
        print(f"  {metric_name:<20} | {info['value']:12.4f} | {info['threshold']:12.4f} | {status}")
    print("  " + "-" * 70)
    assert gate_res["all_passed"] is True, "Model failed quality gate criteria!"

    # ---------------------------------------------------------
    # Step 4: Promote Model to Production Stage
    # ---------------------------------------------------------
    print("\n[Step 4/6] Promoting Model Version to Production Stage...")
    promoted_info = registry_mgr.promote_to_production(
        version=registered_version.version,
        run_id=champion_run_id,
    )
    print(f"  [OK] Model Version {promoted_info['version']} promoted to: {promoted_info['stage']}")
    print(f"  [OK] Tagged with: promoted_by=automated_ci_quality_gate")

    # ---------------------------------------------------------
    # Step 5: Convert to ONNX & Benchmark Latency Acceleration
    # ---------------------------------------------------------
    print("\n[Step 5/6] Converting to ONNX & Benchmarking Inference Speedup...")
    test_df = pd.read_parquet(settings.TEST_DATA_PATH)
    feature_cols = [c for c in test_df.columns if c != settings.TARGET_COLUMN]
    X_test_sample = test_df[feature_cols].head(300)

    onnx_converter = ONNXConverter(output_dir=settings.MODELS_DIR / "champion")
    onnx_path = onnx_converter.convert(champion_model, n_features=len(feature_cols))

    benchmark_res = onnx_converter.benchmark_and_verify(
        sklearn_model=champion_model,
        sample_df=X_test_sample,
        n_iterations=250,
    )

    print(f"  [OK] Numerical Parity Verified: {benchmark_res['numerical_parity_verified']} (Max Diff: {benchmark_res['max_absolute_difference']:.2e})")
    print(f"  [OK] Scikit-Learn p95 Latency:  {benchmark_res['sklearn_latency_p95_ms']:.4f} ms")
    print(f"  [OK] ONNX Runtime p95 Latency:   {benchmark_res['onnx_latency_p95_ms']:.4f} ms")
    print(f"  [OK] Latency Speedup Factor:     {benchmark_res['p95_speedup_factor']:.2f}x faster with ONNX!")

    # ---------------------------------------------------------
    # Step 6: Run Automated Pytest Test Suite
    # ---------------------------------------------------------
    print("\n[Step 6/6] Executing Pytest Test Suite for Stage 3...")
    test_file = str(PROJECT_ROOT / "tests" / "test_stage3_registry.py")
    exit_code = pytest.main(["-v", test_file])
    assert exit_code == 0, f"Pytest suite failed with exit code: {exit_code}"

    # ---------------------------------------------------------
    # Stage 3 Summary Report
    # ---------------------------------------------------------
    print_banner("STAGE 3 COMPLETE: ALL VERIFICATION CHECKS PASSED [100%]")
    print(f"Registered Model:         {registered_version.name}")
    print(f"Production Version:       v{promoted_info['version']}")
    print(f"Quality Gate Status:      PASSED (All 3 thresholds satisfied)")
    print(f"ONNX Model Serialized:    {onnx_path}")
    print(f"ONNX Speedup Ratio:       {benchmark_res['p95_speedup_factor']:.2f}x faster inference")
    print(f"MLflow Active Stage:      Production")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
