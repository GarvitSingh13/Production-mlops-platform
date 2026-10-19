"""
Unit & Integration Tests for Stage 8: Statistical Drift Detection & Retraining Trigger
======================================================================================
Verifies Kolmogorov-Smirnov (KS-test), Population Stability Index (PSI), Chi-Square
categorical tests, drift severity grading, and automated retraining dispatchers.
"""

import numpy as np
import pandas as pd
import pytest

from mlops_platform.config.settings import settings
from mlops_platform.data.generator import generate_synthetic_churn_data
from mlops_platform.monitoring.drift import (
    DriftDetector,
    calculate_psi,
    trigger_automated_retraining,
)


@pytest.fixture(scope="module")
def baseline_dataset():
    """Deterministic clean baseline dataset."""
    return generate_synthetic_churn_data(n_samples=500, random_seed=42, drift_mode=False)


@pytest.fixture(scope="module")
def drifted_dataset():
    """Drifted dataset simulating inflation and customer dissatisfaction."""
    return generate_synthetic_churn_data(n_samples=500, random_seed=999, drift_mode=True)


class TestPopulationStabilityIndex:
    """Verifies PSI algorithm accuracy against standard statistical benchmarks."""

    def test_psi_identical_distributions(self):
        rng = np.random.default_rng(42)
        dist = rng.normal(100, 15, size=1000)
        psi = calculate_psi(dist, dist)
        assert psi < 0.01, f"PSI for identical data should be ~0.0, got: {psi}"

    def test_psi_stable_variation(self):
        rng = np.random.default_rng(42)
        base = rng.normal(100, 15, size=1000)
        target = rng.normal(101, 15.5, size=1000)
        psi = calculate_psi(base, target)
        assert psi < 0.10, f"Expected stable PSI < 0.10, got: {psi}"

    def test_psi_significant_drift(self):
        rng = np.random.default_rng(42)
        base = rng.normal(100, 15, size=1000)
        drifted = rng.normal(130, 25, size=1000)
        psi = calculate_psi(base, drifted)
        assert psi >= 0.25, f"Expected critical PSI >= 0.25, got: {psi}"


class TestStatisticalHypothesisTests:
    """Verifies KS-test and Chi-square feature-level evaluations."""

    def test_numerical_ks_test_no_drift(self, baseline_dataset):
        detector = DriftDetector(baseline_dataset, alpha=0.05)
        # Sliced split from same distribution
        clean_target = baseline_dataset["monthly_charges"].sample(200, random_state=7)
        res = detector.test_numerical_feature("monthly_charges", clean_target)

        assert res.drift_detected is False
        assert res.p_value >= 0.05
        assert res.severity == "NONE"

    def test_numerical_ks_test_with_drift(self, baseline_dataset):
        detector = DriftDetector(baseline_dataset, alpha=0.05)
        # Shifted support tickets (heavy spike)
        drifted_tickets = baseline_dataset["num_support_tickets"] * 3 + 4
        res = detector.test_numerical_feature("num_support_tickets", drifted_tickets)

        assert res.drift_detected is True
        assert res.p_value < 0.01
        assert res.severity in ["MODERATE", "CRITICAL"]

    def test_categorical_chi_square_test(self, baseline_dataset):
        detector = DriftDetector(baseline_dataset, alpha=0.05)

        # 1. Healthy categorical sample
        clean_contract = baseline_dataset["contract_type"].sample(200, random_state=42)
        clean_res = detector.test_categorical_feature("contract_type", clean_contract)
        assert clean_res.drift_detected is False

        # 2. Skewed categorical distribution (100% month-to-month)
        drifted_contract = pd.Series(["month-to-month"] * 200)
        drifted_res = detector.test_categorical_feature("contract_type", drifted_contract)
        assert drifted_res.drift_detected is True
        assert drifted_res.p_value < 0.001


class TestEndToEndDatasetDriftEvaluation:
    """Verifies full dataset drift report and decision matrix."""

    def test_healthy_dataset_evaluation(self, baseline_dataset):
        detector = DriftDetector(baseline_dataset)
        # Fresh draw from baseline generator
        healthy_target = generate_synthetic_churn_data(n_samples=400, random_seed=101, drift_mode=False)

        report = detector.evaluate_drift(healthy_target, sync_telemetry=False)

        assert report.overall_drift_detected is False
        assert report.severity == "HEALTHY"
        assert report.retraining_triggered is False
        assert report.drift_ratio < 0.33

    def test_drifted_dataset_evaluation(self, baseline_dataset, drifted_dataset):
        detector = DriftDetector(baseline_dataset)
        report = detector.evaluate_drift(drifted_dataset, sync_telemetry=False)

        assert report.overall_drift_detected is True
        assert report.severity in ["WARNING", "CRITICAL"]
        assert report.retraining_triggered is True
        assert report.drifted_feature_count >= 2

        # Ensure high priority features are flagged
        assert "monthly_charges" in report.feature_results
        assert report.feature_results["monthly_charges"].drift_detected is True


class TestAutomatedRetrainingTrigger:
    """Verifies retraining event emission and artifact recording."""

    def test_retraining_skipped_when_healthy(self, baseline_dataset):
        detector = DriftDetector(baseline_dataset)
        healthy_target = generate_synthetic_churn_data(n_samples=200, random_seed=55, drift_mode=False)
        report = detector.evaluate_drift(healthy_target, sync_telemetry=False)

        result = trigger_automated_retraining(report)
        assert result["status"] == "skipped"

    def test_retraining_triggered_on_drift(self, baseline_dataset, drifted_dataset):
        detector = DriftDetector(baseline_dataset)
        report = detector.evaluate_drift(drifted_dataset, sync_telemetry=False)

        result = trigger_automated_retraining(report)
        assert result["status"] == "triggered"
        assert "RETRAIN-" in result["trigger_id"]
        assert len(result["drifted_features"]) > 0

        # Verify trigger artifact was written
        trigger_file = settings.METRICS_DIR / "automated_retrain_trigger.json"
        assert trigger_file.exists()
