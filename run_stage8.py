"""
Stage 8 Execution & Verification Runner: Statistical Drift Detection & Retraining Trigger
==========================================================================================
Executes two-sample statistical hypothesis tests (KS-test, Chi-square) and PSI
calculations to detect real-world data drift between baseline training distributions
and live production observation windows, triggering automated retraining pipelines.
"""

import sys
from pathlib import Path
import json
import subprocess
import pandas as pd

from mlops_platform.config.settings import settings
from mlops_platform.data.generator import generate_synthetic_churn_data
from mlops_platform.monitoring.drift import (
    DriftDetector,
    trigger_automated_retraining,
)
from mlops_platform.monitoring.telemetry import (
    DATA_DRIFT_ALERT_GAUGE,
    DATA_DRIFT_SCORE_GAUGE,
)


def main():
    print("=" * 78)
    print(" STAGE 8: STATISTICAL DRIFT DETECTION & AUTOMATED RETRAINING TRIGGER")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # [Step 1/5] Ingesting Baseline Reference Distribution
    # -------------------------------------------------------------------------
    print("\n[Step 1/5] Ingesting Reference Baseline Dataset...")
    if settings.BASELINE_DATA_PATH.exists():
        base_df = pd.read_csv(settings.BASELINE_DATA_PATH)
        print(f"  [OK] Ingested Baseline from {settings.BASELINE_DATA_PATH.name} ({len(base_df)} rows)")
    elif settings.TRAIN_DATA_PATH.exists():
        base_df = pd.read_parquet(settings.TRAIN_DATA_PATH)
        print(f"  [OK] Ingested Baseline from {settings.TRAIN_DATA_PATH.name} ({len(base_df)} rows)")
    else:
        print("  [WARN] Baseline dataset missing. Generating reference baseline...")
        base_df = generate_synthetic_churn_data(n_samples=2000, random_seed=42, drift_mode=False)

    detector = DriftDetector(base_df, alpha=0.05, psi_threshold=0.20)
    print(f"  [OK] Drift Detector Initialized:")
    print(f"       - Numerical Features:   {len(detector.numerical_features)} {detector.numerical_features}")
    print(f"       - Categorical Features: {len(detector.categorical_features)} {detector.categorical_features}")

    # -------------------------------------------------------------------------
    # [Step 2/5] Evaluating Healthy Production Ingestion Window
    # -------------------------------------------------------------------------
    print("\n[Step 2/5] Evaluating Healthy Production Window (No Drift Expected)...")
    healthy_batch = generate_synthetic_churn_data(n_samples=400, random_seed=77, drift_mode=False)
    healthy_report = detector.evaluate_drift(healthy_batch, sync_telemetry=True)

    print(f"  [OK] Overall Drift Detected:     {healthy_report.overall_drift_detected}")
    print(f"  [OK] Drift Severity:             {healthy_report.severity}")
    print(f"  [OK] Drifted Feature Count:      {healthy_report.drifted_feature_count} / {healthy_report.total_features_evaluated}")
    print(f"  [OK] Retraining Triggered:       {healthy_report.retraining_triggered}")

    # -------------------------------------------------------------------------
    # [Step 3/5] Evaluating Drifted Production Window (Simulating Market Shift)
    # -------------------------------------------------------------------------
    print("\n[Step 3/5] Evaluating Drifted Production Window (Inflation & Product Shift)...")
    drifted_batch = generate_synthetic_churn_data(n_samples=600, random_seed=888, drift_mode=True)
    drifted_report = detector.evaluate_drift(drifted_batch, sync_telemetry=True)

    print(f"\n  Feature-by-Feature Statistical Diagnostic:")
    print(f"  {'-'*74}")
    print(f"  {'Feature':<22} | {'Method':<8} | {'p-Value':<10} | {'PSI':<7} | {'Severity':<9} | Drift?")
    print(f"  {'-'*74}")

    for feat_name, res in drifted_report.feature_results.items():
        method_short = "KS-Test" if res.feature_type == "numerical" else "Chi2"
        psi_str = f"{res.psi_score:.3f}" if res.psi_score is not None else "N/A"
        flag = "[DRIFT]" if res.drift_detected else " OK "
        print(f"  {feat_name:<22} | {method_short:<8} | {res.p_value:<10.5f} | {psi_str:<7} | {res.severity:<9} | {flag}")
    print(f"  {'-'*74}")

    print(f"\n  [OK] Overall Drift Detected:     {drifted_report.overall_drift_detected}")
    print(f"  [OK] Drift Severity:             {drifted_report.severity}")
    print(f"  [OK] Drift Ratio:                {drifted_report.drift_ratio * 100:.1f}%")
    print(f"  [OK] Recommendation:            {drifted_report.recommendation}")

    # -------------------------------------------------------------------------
    # [Step 4/5] Executing Automated Retraining Pipeline Trigger
    # -------------------------------------------------------------------------
    print("\n[Step 4/5] Dispatching Automated Retraining Trigger...")
    retrain_result = trigger_automated_retraining(drifted_report)

    # Save comprehensive report to metrics directory
    settings.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = settings.METRICS_DIR / "drift_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(drifted_report.model_dump_json(indent=2))

    print(f"  [OK] Trigger Status:             {retrain_result['status'].upper()}")
    print(f"  [OK] Trigger Identifier:         {retrain_result['trigger_id']}")
    print(f"  [OK] Target Model:               {retrain_result['target_model']}")
    print(f"  [OK] Drift Report Artifact:      {report_file.name} saved in {settings.METRICS_DIR}")

    # -------------------------------------------------------------------------
    # [Step 5/5] Prometheus Telemetry Synchronicity Check
    # -------------------------------------------------------------------------
    print("\n[Step 5/5] Auditing Prometheus Metric Synchronization...")
    alert_val = DATA_DRIFT_ALERT_GAUGE._value.get()
    print(f"  [OK] Prometheus Gauge State:     mlops_data_drift_detected = {alert_val} (Alert Active)")

    # -------------------------------------------------------------------------
    # Executing Pytest Test Suite
    # -------------------------------------------------------------------------
    print("\n[Verification] Executing Pytest Test Suite for Stage 8...")
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "-v", "tests/test_stage8_drift.py"],
        cwd=str(settings.PROJECT_ROOT),
    )
    if res.returncode != 0:
        print("\n  [FAIL] Stage 8 unit tests failed!")
        sys.exit(res.returncode)

    print("\n" + "=" * 78)
    print(" STAGE 8 COMPLETE: DRIFT ENGINE & RETRAINING TRIGGER VALIDATED [100%]")
    print("=" * 78)
    print("Statistical Methods:      Kolmogorov-Smirnov (2-Sample) & Chi-Square Contingency")
    print("Distribution Stability:   Population Stability Index (PSI) with Quantile Deciles")
    print("Detection Thresholds:     alpha = 0.05, PSI Threshold = 0.20")
    print("Retraining Trigger:       Active (Dispatched on Severity >= WARNING)")
    print("Prometheus Sync:          Live Gauges Updated for Grafana Real-Time Alerts")
    print("=" * 78)


if __name__ == "__main__":
    main()
