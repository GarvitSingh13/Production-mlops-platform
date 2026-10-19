"""
Real-Time Telemetry, Prometheus Metrics & Drift Monitoring
===========================================================
"""

from mlops_platform.monitoring.drift import (
    DriftDetector,
    DriftReport,
    FeatureDriftResult,
    calculate_psi,
    trigger_automated_retraining,
)
from mlops_platform.monitoring.telemetry import (
    ACTIVE_REQUESTS_GAUGE,
    DATA_DRIFT_ALERT_GAUGE,
    DATA_DRIFT_SCORE_GAUGE,
    INFERENCE_LATENCY_SECONDS,
    INFERENCE_REQUESTS_TOTAL,
    MODEL_INFO_GAUGE,
    PREDICTION_PROBABILITY_HISTOGRAM,
    get_prometheus_metrics,
    record_batch_metrics,
    record_inference_metric,
)

__all__ = [
    "ACTIVE_REQUESTS_GAUGE",
    "DATA_DRIFT_ALERT_GAUGE",
    "DATA_DRIFT_SCORE_GAUGE",
    "INFERENCE_LATENCY_SECONDS",
    "INFERENCE_REQUESTS_TOTAL",
    "MODEL_INFO_GAUGE",
    "PREDICTION_PROBABILITY_HISTOGRAM",
    "DriftDetector",
    "DriftReport",
    "FeatureDriftResult",
    "calculate_psi",
    "get_prometheus_metrics",
    "record_batch_metrics",
    "record_inference_metric",
    "trigger_automated_retraining",
]
