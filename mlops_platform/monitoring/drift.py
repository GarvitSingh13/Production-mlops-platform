"""
Enterprise Statistical Drift Detection Engine & Automated Retraining Trigger
=============================================================================
Performs statistical hypothesis testing (Kolmogorov-Smirnov, Chi-Square) and
Population Stability Index (PSI) calculations to detect covariate shift, concept drift,
and automatically trigger MLOps model retraining pipelines.
"""

from typing import Dict, List, Literal, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

from mlops_platform.config.settings import settings
from mlops_platform.monitoring.telemetry import update_drift_telemetry


class FeatureDriftResult(BaseModel):
    """Statistical drift evaluation for an individual feature."""

    feature_name: str
    feature_type: Literal["numerical", "categorical"]
    statistical_test: str
    test_statistic: float
    p_value: float
    psi_score: Optional[float] = None
    drift_detected: bool
    severity: Literal["NONE", "LOW", "MODERATE", "CRITICAL"]


class DriftReport(BaseModel):
    """Comprehensive dataset drift report and retraining decision."""

    timestamp: str
    baseline_sample_size: int
    target_sample_size: int
    total_features_evaluated: int
    drifted_feature_count: int
    drift_ratio: float
    overall_drift_detected: bool
    severity: Literal["HEALTHY", "WARNING", "CRITICAL"]
    retraining_triggered: bool
    feature_results: Dict[str, FeatureDriftResult]
    recommendation: str


def calculate_psi(
    baseline: np.ndarray,
    target: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """
    Calculate Population Stability Index (PSI) between baseline and target distributions.

    PSI Formula:
        PSI = sum((Actual% - Expected%) * ln(Actual% / Expected%))

    Rule of Thumb:
        PSI < 0.10: Insignificant change (Stable)
        0.10 <= PSI < 0.25: Moderate shift (Warning)
        PSI >= 0.25: Significant shift (Critical Action Required)
    """
    baseline = np.asarray(baseline, dtype=float)
    target = np.asarray(target, dtype=float)

    # Filter out NaNs
    baseline = baseline[~np.isnan(baseline)]
    target = target[~np.isnan(target)]

    if len(baseline) == 0 or len(target) == 0:
        return 0.0

    # Determine quantile bin edges from baseline
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(baseline, quantiles)
    bin_edges = np.unique(bin_edges)

    # If distribution is discrete or low-variance, fallback to linear spacing
    if len(bin_edges) < 3:
        bin_edges = np.linspace(
            min(baseline.min(), target.min()),
            max(baseline.max(), target.max()),
            num_bins + 1,
        )

    # Adjust outermost edges to capture boundary samples
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5

    # Compute frequency counts in each bin
    base_counts, _ = np.histogram(baseline, bins=bin_edges)
    target_counts, _ = np.histogram(target, bins=bin_edges)

    # Convert to proportions and apply epsilon smoothing
    base_props = (base_counts / len(baseline)) + epsilon
    target_props = (target_counts / len(target)) + epsilon

    # Normalize proportions so they sum to 1.0
    base_props /= np.sum(base_props)
    target_props /= np.sum(target_props)

    # Calculate PSI summation
    psi_value = np.sum((target_props - base_props) * np.log(target_props / base_props))
    return float(max(0.0, psi_value))


class DriftDetector:
    """
    Production-grade Data Drift Detector combining Kolmogorov-Smirnov (KS-test),
    Population Stability Index (PSI), and Chi-Square categorical tests.
    """

    def __init__(
        self,
        baseline_df: pd.DataFrame,
        alpha: float = 0.05,
        psi_threshold: float = 0.20,
    ):
        """
        Initialize detector with reference baseline dataset.

        Args:
            baseline_df: Clean historical baseline (typically the training dataset).
            alpha: Significance level for p-value hypothesis testing (default 0.05).
            psi_threshold: PSI threshold for determining distribution divergence.
        """
        self.baseline_df = baseline_df.copy()
        self.alpha = alpha
        self.psi_threshold = psi_threshold

        self.numerical_features = [
            f for f in settings.NUMERICAL_FEATURES if f in self.baseline_df.columns
        ]
        self.categorical_features = [
            f for f in settings.CATEGORICAL_FEATURES if f in self.baseline_df.columns
        ]

    def test_numerical_feature(
        self, feature_name: str, target_series: pd.Series
    ) -> FeatureDriftResult:
        """Evaluate continuous distribution shift using two-sample KS-Test and PSI."""
        base_series = self.baseline_df[feature_name].dropna()
        target_clean = target_series.dropna()

        # Two-Sample Kolmogorov-Smirnov Test
        ks_res = stats.ks_2samp(base_series, target_clean)
        ks_stat = float(ks_res.statistic)
        p_val = float(ks_res.pvalue)

        # Population Stability Index
        psi_score = calculate_psi(base_series.values, target_clean.values)

        # Determine drift flag and severity
        is_drifted = (p_val < self.alpha) or (psi_score >= self.psi_threshold)

        if psi_score >= 0.25 or p_val < 0.001:
            severity = "CRITICAL"
        elif psi_score >= 0.10 or p_val < self.alpha:
            severity = "MODERATE"
        elif psi_score >= 0.05:
            severity = "LOW"
        else:
            severity = "NONE"

        return FeatureDriftResult(
            feature_name=feature_name,
            feature_type="numerical",
            statistical_test="Kolmogorov-Smirnov (2-Sample) + PSI",
            test_statistic=round(ks_stat, 4),
            p_value=round(p_val, 6),
            psi_score=round(psi_score, 4),
            drift_detected=is_drifted,
            severity=severity,
        )

    def test_categorical_feature(
        self, feature_name: str, target_series: pd.Series
    ) -> FeatureDriftResult:
        """Evaluate categorical frequency shifts using Chi-Square Contingency Test."""
        base_series = self.baseline_df[feature_name].dropna().astype(str)
        target_clean = target_series.dropna().astype(str)

        all_categories = sorted(list(set(base_series.unique()).union(set(target_clean.unique()))))

        base_freq = base_series.value_counts().reindex(all_categories, fill_value=0)
        target_freq = target_clean.value_counts().reindex(all_categories, fill_value=0)

        contingency_table = np.array([base_freq.values, target_freq.values])

        chi2_res = stats.chi2_contingency(contingency_table)
        chi2_stat = float(chi2_res.statistic)
        p_val = float(chi2_res.pvalue)

        is_drifted = p_val < self.alpha

        if p_val < 0.001:
            severity = "CRITICAL"
        elif p_val < self.alpha:
            severity = "MODERATE"
        else:
            severity = "NONE"

        return FeatureDriftResult(
            feature_name=feature_name,
            feature_type="categorical",
            statistical_test="Chi-Square Contingency Test",
            test_statistic=round(chi2_stat, 4),
            p_value=round(p_val, 6),
            psi_score=None,
            drift_detected=is_drifted,
            severity=severity,
        )

    def evaluate_drift(
        self,
        target_df: pd.DataFrame,
        sync_telemetry: bool = True,
    ) -> DriftReport:
        """
        Evaluate full dataset drift across all numerical and categorical features.

        Args:
            target_df: Batch of production inference observations.
            sync_telemetry: Whether to publish p-values to Prometheus gauges.

        Returns:
            Structured DriftReport with retraining decision.
        """
        feature_results: Dict[str, FeatureDriftResult] = {}
        p_values_dict: Dict[str, float] = {}

        # 1. Evaluate Numerical Features
        for feat in self.numerical_features:
            if feat in target_df.columns:
                res = self.test_numerical_feature(feat, target_df[feat])
                feature_results[feat] = res
                p_values_dict[feat] = res.p_value

        # 2. Evaluate Categorical Features
        for feat in self.categorical_features:
            if feat in target_df.columns:
                res = self.test_categorical_feature(feat, target_df[feat])
                feature_results[feat] = res
                p_values_dict[feat] = res.p_value

        # 3. Aggregate Dataset-Level Drift Metrics
        total_feats = len(feature_results)
        drifted_count = sum(1 for r in feature_results.values() if r.drift_detected)
        critical_count = sum(1 for r in feature_results.values() if r.severity == "CRITICAL")
        drift_ratio = round(drifted_count / total_feats, 3) if total_feats > 0 else 0.0

        # High priority features whose drift immediately triggers action
        critical_features = {"monthly_charges", "num_support_tickets", "credit_score"}
        critical_feat_drifted = any(
            feature_results[f].severity == "CRITICAL"
            for f in critical_features
            if f in feature_results
        )

        overall_drift = (drift_ratio >= 0.33) or critical_feat_drifted or (critical_count >= 2)

        if critical_count >= 2 or critical_feat_drifted:
            severity = "CRITICAL"
            recommendation = (
                "CRITICAL DATA DRIFT DETECTED: Significant statistical shift in core business features. "
                "Immediate automated retraining pipeline triggered."
            )
            trigger_retrain = True
        elif overall_drift:
            severity = "WARNING"
            recommendation = (
                "MODERATE DRIFT DETECTED: Elevated divergence from baseline. "
                "Trigger retraining workflow and monitor validation gate."
            )
            trigger_retrain = True
        else:
            severity = "HEALTHY"
            recommendation = (
                "NO SIGNIFICANT DRIFT: Feature distributions remain stable within baseline parameters."
            )
            trigger_retrain = False

        # 4. Sync to Prometheus Telemetry
        if sync_telemetry:
            update_drift_telemetry(p_values_dict, overall_drift)

        return DriftReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            baseline_sample_size=len(self.baseline_df),
            target_sample_size=len(target_df),
            total_features_evaluated=total_feats,
            drifted_feature_count=drifted_count,
            drift_ratio=drift_ratio,
            overall_drift_detected=overall_drift,
            severity=severity,
            retraining_triggered=trigger_retrain,
            feature_results=feature_results,
            recommendation=recommendation,
        )


def trigger_automated_retraining(report: DriftReport) -> Dict[str, Any]:
    """
    Automated Retraining Trigger Dispatcher.

    Generates audit records and triggers downstream CI/CD training jobs
    when data drift exceeds enterprise tolerance.
    """
    if not report.retraining_triggered:
        return {
            "status": "skipped",
            "message": "Model retraining skipped: Feature distributions are healthy.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    trigger_payload = {
        "status": "triggered",
        "trigger_id": f"RETRAIN-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "drift_severity": report.severity,
        "drifted_features": [
            k for k, v in report.feature_results.items() if v.drift_detected
        ],
        "drift_ratio": report.drift_ratio,
        "action": "trigger_model_training_pipeline",
        "target_model": settings.MLFLOW_REGISTERED_MODEL_NAME,
    }

    # Save trigger record in metrics directory
    settings.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    trigger_file = settings.METRICS_DIR / "automated_retrain_trigger.json"
    with open(trigger_file, "w", encoding="utf-8") as f:
        json.dump(trigger_payload, f, indent=2)

    return trigger_payload
