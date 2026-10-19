"""
Stage 1 Verification Runner: Data Pipeline, Preprocessing & Feature Contracts
=============================================================================
Executes and verifies:
  1. Synthetic enterprise dataset generation (baseline + drifted).
  2. Pydantic v2 data contract boundary validation and error rejection.
  3. Stratified train/val/test splitting and leak-free feature transformations.
  4. Disk serialization of Parquet splits and preprocessor ColumnTransformer.
  5. Programmatic execution of pytest unit test suite.
"""

import sys
import os
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

import pandas as pd
import pytest
from mlops_platform.config.settings import settings
from mlops_platform.data.contracts import (
    validate_dataframe_contract,
    ContractValidationError,
    InferencePayload,
)
from mlops_platform.data.generator import generate_synthetic_churn_data
from mlops_platform.data.pipeline import DataPipeline


def print_banner(text: str, char: str = "="):
    line = char * 78
    print(f"\n{line}")
    print(f" {text}")
    print(f"{line}")


def main():
    print_banner("STAGE 1: DATA PIPELINE, PREPROCESSING & FEATURE CONTRACTS")

    # ---------------------------------------------------------
    # Step 1: Generate Datasets
    # ---------------------------------------------------------
    print("\n[Step 1/5] Generating Enterprise Datasets...")
    settings.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df_baseline = generate_synthetic_churn_data(
        n_samples=6000,
        random_seed=settings.RANDOM_SEED,
        drift_mode=False,
        output_path=settings.BASELINE_DATA_PATH,
    )
    print(f"  [OK] Baseline dataset generated: {df_baseline.shape} -> {settings.BASELINE_DATA_PATH.name}")

    df_drifted = generate_synthetic_churn_data(
        n_samples=1500,
        random_seed=settings.RANDOM_SEED + 99,
        drift_mode=True,
        output_path=settings.DRIFTED_DATA_PATH,
    )
    print(f"  [OK] Drifted dataset generated:  {df_drifted.shape} -> {settings.DRIFTED_DATA_PATH.name}")

    # ---------------------------------------------------------
    # Step 2: Validate Data Contract & Security Bounds
    # ---------------------------------------------------------
    print("\n[Step 2/5] Enforcing Data Contracts & Bounds Validation...")
    errors = validate_dataframe_contract(df_baseline, is_training=True, raise_on_error=False)
    assert len(errors) == 0, f"Baseline dataset failed contract validation: {errors}"
    print("  [OK] Baseline dataset passed all schema, null, and numerical range checks.")

    # Test intentional contract violation rejection
    bad_df = df_baseline.head(10).copy()
    bad_df.loc[0, "age"] = -5  # Invalid age
    bad_df.loc[1, "contract_type"] = "decade-plan"  # Invalid category
    rejection_errors = validate_dataframe_contract(bad_df, is_training=True, raise_on_error=False)
    assert len(rejection_errors) == 2, "Expected 2 contract rejections for corrupted sample."
    print("  [OK] Data Contract successfully caught and rejected invalid values (negative age & bad category).")

    # ---------------------------------------------------------
    # Step 3: Run Stratified Splitting & Feature Transformation
    # ---------------------------------------------------------
    print("\n[Step 3/5] Running Data Pipeline (Stratified Split & Preprocessing)...")
    pipeline = DataPipeline(random_seed=settings.RANDOM_SEED)
    summary = pipeline.run(raw_data_path=settings.BASELINE_DATA_PATH, save_artifacts=True)

    print(f"  [OK] Stratified Train split: {summary['train_shape']} (Churn: {summary['train_churn_rate']:.1%})")
    print(f"  [OK] Stratified Val split:   {summary['val_shape']} (Churn: {summary['val_churn_rate']:.1%})")
    print(f"  [OK] Stratified Test split:  {summary['test_shape']} (Churn: {summary['test_churn_rate']:.1%})")
    print(f"  [OK] Transformed Features ({summary['num_features']}): {summary['feature_names'][:6]}...")

    # ---------------------------------------------------------
    # Step 4: Verify Saved Artifacts on Disk
    # ---------------------------------------------------------
    print("\n[Step 4/5] Verifying Saved Artifacts on Disk...")
    for artifact_path in [
        settings.TRAIN_DATA_PATH,
        settings.VAL_DATA_PATH,
        settings.TEST_DATA_PATH,
        settings.PREPROCESSOR_PATH,
    ]:
        assert artifact_path.exists(), f"Missing artifact: {artifact_path}"
        size_kb = artifact_path.stat().st_size / 1024
        print(f"  [OK] Found {artifact_path.name:<25} ({size_kb:6.1f} KB)")

    # ---------------------------------------------------------
    # Step 5: Run Automated Pytest Test Suite
    # ---------------------------------------------------------
    print("\n[Step 5/5] Executing Pytest Test Suite for Stage 1...")
    test_file = str(PROJECT_ROOT / "tests" / "test_stage1_data.py")
    exit_code = pytest.main(["-v", test_file])
    assert exit_code == 0, f"Pytest suite failed with exit code: {exit_code}"

    # ---------------------------------------------------------
    # Stage 1 Summary Report
    # ---------------------------------------------------------
    print_banner("STAGE 1 COMPLETE: ALL VERIFICATION CHECKS PASSED [100%]")
    print(f"Total Raw Records:        {summary['total_records']:,}")
    print(f"Train Set Shape:          {summary['train_shape']} (70%)")
    print(f"Validation Set Shape:     {summary['val_shape']} (15%)")
    print(f"Test Set Shape:           {summary['test_shape']} (15%)")
    print(f"Transformed Feature Dim:  {summary['num_features']} columns")
    print(f"Data Contract Gate:       ENFORCED & ACTIVE")
    print(f"Preprocessors Saved:      {settings.PREPROCESSOR_PATH}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
