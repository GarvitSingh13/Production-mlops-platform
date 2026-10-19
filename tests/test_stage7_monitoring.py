"""
Unit & Integration Tests for Stage 7: Real-Time Observability (Prometheus & Grafana)
===================================================================================
Verifies Prometheus telemetry instrumentation, /metrics scraper endpoint,
latency histogram buckets, model info gauges, alerting rules, and Grafana dashboard.
"""

import json

import pytest
import yaml
from fastapi.testclient import TestClient

from mlops_platform.config.settings import settings
from mlops_platform.monitoring.telemetry import update_drift_telemetry
from mlops_platform.serving.app import app


@pytest.fixture
def test_client():
    """Context-managed FastAPI test client ensuring lifespan handlers execute."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_payload():
    return {
        "customer_id": "CUST-PROM-001",
        "credit_score": 680,
        "age": 42,
        "tenure_months": 24,
        "monthly_charges": 75.50,
        "total_charges": 1812.00,
        "num_support_tickets": 2,
        "contract_type": "one-year",
        "payment_method": "credit_card",
        "has_paperless_billing": 1,
    }


class TestPrometheusConfiguration:
    """Verifies Prometheus server configuration, alerting rules, and Grafana dashboard."""

    def test_prometheus_server_config(self):
        prom_file = settings.PROJECT_ROOT / "monitoring" / "prometheus.yml"
        assert prom_file.exists(), "Missing monitoring/prometheus.yml!"

        with open(prom_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        assert "scrape_configs" in cfg
        job_names = [j["job_name"] for j in cfg["scrape_configs"]]
        assert "mlops-inference-service" in job_names

    def test_alerting_rules_syntax(self):
        alert_file = settings.PROJECT_ROOT / "monitoring" / "alert_rules.yml"
        assert alert_file.exists(), "Missing monitoring/alert_rules.yml!"

        with open(alert_file, "r", encoding="utf-8") as f:
            rules_cfg = yaml.safe_load(f)

        assert "groups" in rules_cfg
        alerts = [r["alert"] for r in rules_cfg["groups"][0]["rules"]]
        assert "P99LatencySLOBreached" in alerts
        assert "HighInferenceErrorRate" in alerts
        assert "StatisticalDataDriftDetected" in alerts

    def test_grafana_dashboard_schema(self):
        dash_file = settings.PROJECT_ROOT / "monitoring" / "grafana_dashboard.json"
        assert dash_file.exists(), "Missing monitoring/grafana_dashboard.json!"

        with open(dash_file, "r", encoding="utf-8") as f:
            dashboard = json.load(f)

        assert dashboard["title"].startswith("Enterprise MLOps Command Center")
        assert len(dashboard["panels"]) >= 7, "Dashboard must have at least 7 observability panels!"

        # Check PromQL expressions exist in panels
        has_promql = False
        for panel in dashboard["panels"]:
            targets = panel.get("targets", [])
            for target in targets:
                if "expr" in target:
                    has_promql = True
                    break
        assert has_promql, "Grafana dashboard must contain valid PromQL metric queries!"


class TestFastAPIMetricsEndpoint:
    """Verifies the /metrics scraper endpoint adheres to Prometheus exposition standards."""

    def test_metrics_endpoint_status_and_format(self, test_client):
        response = test_client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]

        body = response.text
        assert "mlops_inference_requests_total" in body
        assert "mlops_inference_latency_seconds" in body
        assert "mlops_model_info" in body

    def test_model_info_gauge_registered(self, test_client):
        response = test_client.get("/metrics")
        body = response.text
        assert 'model_name="LogisticRegression"' in body
        assert "mlops_model_info" in body


class TestInferenceTelemetryCollection:
    """Verifies live requests correctly update counters, histograms, and engine labels."""

    def test_single_prediction_telemetry_onnx(self, test_client, sample_payload):
        # 1. Trigger ONNX inference
        res = test_client.post("/predict?engine=onnx&explain=false", json=sample_payload)
        assert res.status_code == 200

        # 2. Scrape Prometheus metrics
        metrics_res = test_client.get("/metrics")
        body = metrics_res.text

        # Ensure ONNX engine, /predict endpoint, and latency buckets were captured
        assert 'engine="onnx"' in body
        assert 'endpoint="/predict"' in body
        assert "mlops_inference_latency_seconds_bucket" in body

    def test_single_prediction_telemetry_sklearn(self, test_client, sample_payload):
        # 1. Trigger Scikit-Learn inference
        res = test_client.post("/predict?engine=sklearn&explain=false", json=sample_payload)
        assert res.status_code == 200

        # 2. Scrape Prometheus metrics
        metrics_res = test_client.get("/metrics")
        body = metrics_res.text

        assert 'engine="sklearn"' in body
        assert 'endpoint="/predict"' in body

    def test_batch_prediction_telemetry(self, test_client, sample_payload):
        batch_payload = {"customers": [sample_payload, sample_payload]}
        res = test_client.post("/predict_batch?engine=onnx", json=batch_payload)
        assert res.status_code == 200

        metrics_res = test_client.get("/metrics")
        body = metrics_res.text
        assert 'endpoint="/predict_batch"' in body

    def test_drift_telemetry_update(self, test_client):
        # Simulate drift detection event
        update_drift_telemetry(p_values={"monthly_charges": 0.0042, "credit_score": 0.0089}, drift_detected=True)

        metrics_res = test_client.get("/metrics")
        body = metrics_res.text

        assert "mlops_data_drift_detected 1.0" in body
        assert 'feature_name="monthly_charges"' in body
