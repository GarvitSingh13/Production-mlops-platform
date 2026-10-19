"""
Prometheus Telemetry Instrumentation & Metrics Collector
=========================================================
Exports real-time operational SLO telemetry, request latency histograms,
engine-specific performance metrics, prediction distributions, and drift alerts.
"""

from prometheus_client import (
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# -----------------------------------------------------------------------------
# 1. Traffic & Request Counter Metrics
# -----------------------------------------------------------------------------
INFERENCE_REQUESTS_TOTAL = Counter(
    "mlops_inference_requests_total",
    "Total count of inference requests processed by the microservice.",
    labelnames=["endpoint", "status", "engine", "prediction_class"],
)

# -----------------------------------------------------------------------------
# 2. SLO Latency Histograms (Sub-millisecond to 2.5s)
# -----------------------------------------------------------------------------
SLO_LATENCY_BUCKETS = (
    0.0005,  # 0.5 ms
    0.0010,  # 1.0 ms (Target ONNX P50)
    0.0025,  # 2.5 ms
    0.0050,  # 5.0 ms (Target ONNX P95)
    0.0100,  # 10.0 ms (Enterprise SLO Ceiling)
    0.0250,  # 25.0 ms
    0.0500,  # 50.0 ms
    0.1000,  # 100.0 ms
    0.2500,  # 250.0 ms
    0.5000,  # 500.0 ms
    1.0000,  # 1.0 s
    2.5000,  # 2.5 s
)

INFERENCE_LATENCY_SECONDS = Histogram(
    "mlops_inference_latency_seconds",
    "End-to-end inference execution duration in seconds.",
    labelnames=["endpoint", "engine"],
    buckets=SLO_LATENCY_BUCKETS,
)

# -----------------------------------------------------------------------------
# 3. Model Prediction Distribution Histogram
# -----------------------------------------------------------------------------
PREDICTION_PROBABILITY_HISTOGRAM = Histogram(
    "mlops_prediction_probability",
    "Distribution of churn prediction probabilities output by the model.",
    labelnames=["engine", "risk_tier"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# -----------------------------------------------------------------------------
# 4. In-Flight Concurrency & System State Gauges
# -----------------------------------------------------------------------------
ACTIVE_REQUESTS_GAUGE = Gauge(
    "mlops_active_requests",
    "Number of active in-flight requests currently being executed.",
    labelnames=["endpoint"],
)

MODEL_INFO_GAUGE = Gauge(
    "mlops_model_info",
    "Active production champion model deployment metadata.",
    labelnames=["model_name", "model_version", "onnx_accelerated"],
)

# -----------------------------------------------------------------------------
# 5. Data Drift & Retraining Alert Gauges
# -----------------------------------------------------------------------------
DATA_DRIFT_SCORE_GAUGE = Gauge(
    "mlops_data_drift_pvalue",
    "Kolmogorov-Smirnov statistical test p-value per feature (p < 0.05 indicates drift).",
    labelnames=["feature_name"],
)

DATA_DRIFT_ALERT_GAUGE = Gauge(
    "mlops_data_drift_detected",
    "Binary indicator (1 = Significant Drift Detected, 0 = Healthy Baseline).",
)


# -----------------------------------------------------------------------------
# Telemetry Recording Helpers
# -----------------------------------------------------------------------------
def record_inference_metric(
    endpoint: str,
    engine: str,
    status_code: int,
    prediction_class: int,
    probability: float,
    risk_tier: str,
    latency_seconds: float,
) -> None:
    """Record telemetry observations for a single inference request."""
    status_str = str(status_code)
    pred_str = str(prediction_class)

    # Increment request counter
    INFERENCE_REQUESTS_TOTAL.labels(
        endpoint=endpoint,
        status=status_str,
        engine=engine,
        prediction_class=pred_str,
    ).inc()

    # Record latency in histogram
    INFERENCE_LATENCY_SECONDS.labels(
        endpoint=endpoint,
        engine=engine,
    ).observe(latency_seconds)

    # Record probability distribution
    PREDICTION_PROBABILITY_HISTOGRAM.labels(
        engine=engine,
        risk_tier=risk_tier,
    ).observe(probability)


def record_batch_metrics(
    endpoint: str,
    engine: str,
    status_code: int,
    total_processed: int,
    churn_count: int,
    probabilities: list[float],
    latency_seconds: float,
) -> None:
    """Record telemetry observations for a vectorized batch prediction request."""
    status_str = str(status_code)
    non_churn_count = total_processed - churn_count

    # Increment counters for churn and non-churn items
    if churn_count > 0:
        INFERENCE_REQUESTS_TOTAL.labels(
            endpoint=endpoint,
            status=status_str,
            engine=engine,
            prediction_class="1",
        ).inc(churn_count)

    if non_churn_count > 0:
        INFERENCE_REQUESTS_TOTAL.labels(
            endpoint=endpoint,
            status=status_str,
            engine=engine,
            prediction_class="0",
        ).inc(non_churn_count)

    # Record batch latency
    INFERENCE_LATENCY_SECONDS.labels(
        endpoint=endpoint,
        engine=engine,
    ).observe(latency_seconds)

    # Record probabilities
    for p in probabilities:
        tier = "HIGH" if p >= 0.70 else ("MEDIUM" if p >= 0.40 else "LOW")
        PREDICTION_PROBABILITY_HISTOGRAM.labels(
            engine=engine,
            risk_tier=tier,
        ).observe(p)


def set_model_info(model_name: str, model_version: str, onnx_accelerated: bool) -> None:
    """Publish active model metadata gauge to Prometheus."""
    MODEL_INFO_GAUGE.labels(
        model_name=model_name,
        model_version=model_version,
        onnx_accelerated=str(onnx_accelerated).lower(),
    ).set(1)


def update_drift_telemetry(p_values: dict[str, float], drift_detected: bool) -> None:
    """Update Prometheus drift gauges with fresh statistical tests."""
    for feature_name, p_val in p_values.items():
        DATA_DRIFT_SCORE_GAUGE.labels(feature_name=feature_name).set(p_val)
    DATA_DRIFT_ALERT_GAUGE.set(1 if drift_detected else 0)


def get_prometheus_metrics() -> bytes:
    """Expose Prometheus formatted metric exposition string."""
    return generate_latest(REGISTRY)
