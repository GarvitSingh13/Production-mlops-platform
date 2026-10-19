"""
Real-Time Telemetry, Prometheus Metrics & Drift Monitoring
===========================================================
"""

from mlops_platform.monitoring.telemetry import (
    INFERENCE_REQUESTS_TOTAL,
    INFERENCE_LATENCY_SECONDS,
    PREDICTION_PROBABILITY_HISTOGRAM,
    ACTIVE_REQUESTS_GAUGE,
    MODEL_INFO_GAUGE,
    DATA_DRIFT_SCORE_GAUGE,
    DATA_DRIFT_ALERT_GAUGE,
    record_inference_metric,
    record_batch_metrics,
    get_prometheus_metrics,
)

__all__ = [
    "INFERENCE_REQUESTS_TOTAL",
    "INFERENCE_LATENCY_SECONDS",
    "PREDICTION_PROBABILITY_HISTOGRAM",
    "ACTIVE_REQUESTS_GAUGE",
    "MODEL_INFO_GAUGE",
    "DATA_DRIFT_SCORE_GAUGE",
    "DATA_DRIFT_ALERT_GAUGE",
    "record_inference_metric",
    "record_batch_metrics",
    "get_prometheus_metrics",
]
