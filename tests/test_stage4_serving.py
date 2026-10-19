"""
Unit & Integration Tests for Stage 4: FastAPI Model Serving Microservice
========================================================================
Verifies health probes, dual-engine inference, numerical parity, SHAP
explainability attributions, batch processing, and schema error rejection.
"""

import pytest
from fastapi.testclient import TestClient
from mlops_platform.serving.app import app


@pytest.fixture(scope="module")
def client():
    """Create a test client that triggers lifespan startup and shutdown."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_payload():
    """Valid sample customer payload for testing."""
    return {
        "customer_id": "CUST-TEST-001",
        "credit_score": 710,
        "age": 36,
        "tenure_months": 24,
        "monthly_charges": 68.50,
        "total_charges": 1644.00,
        "num_support_tickets": 1,
        "contract_type": "one-year",
        "payment_method": "credit_card",
        "has_paperless_billing": 1,
    }


class TestServingHealthAndMetadata:
    """Tests for system health probes and metadata endpoints."""

    def test_root_endpoint(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "online"
        assert "predict_single" in data["endpoints"]

    def test_liveness_probe(self, client):
        resp = client.get("/health/live")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "alive"
        assert "timestamp" in data

    def test_readiness_probe(self, client):
        resp = client.get("/health/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["sklearn_loaded"] is True
        assert data["onnx_loaded"] is True
        assert data["preprocessor_loaded"] is True

    def test_model_metadata(self, client):
        resp = client.get("/model/metadata")
        assert resp.status_code == 200
        data = resp.json()
        assert data["feature_count"] > 5
        assert "numerical_inputs" in data


class TestServingInference:
    """Tests for prediction endpoints, dual engines, and SHAP explainability."""

    def test_predict_single_onnx_engine(self, client, sample_payload):
        resp = client.post("/predict?engine=onnx&explain=false", json=sample_payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["customer_id"] == "CUST-TEST-001"
        assert data["churn_prediction"] in [0, 1]
        assert 0.0 <= data["churn_probability"] <= 1.0
        assert data["risk_tier"] in ["LOW", "MEDIUM", "HIGH"]
        assert data["engine_used"] == "onnx"
        assert data["latency_ms"] >= 0.0

    def test_predict_single_sklearn_engine_and_parity(self, client, sample_payload):
        resp_onnx = client.post("/predict?engine=onnx&explain=false", json=sample_payload)
        resp_sk = client.post("/predict?engine=sklearn&explain=false", json=sample_payload)

        assert resp_onnx.status_code == 200
        assert resp_sk.status_code == 200

        data_onnx = resp_onnx.json()
        data_sk = resp_sk.json()

        assert data_sk["engine_used"] == "sklearn"
        # Mathematical parity check: probabilities should match within 1e-4
        assert abs(data_onnx["churn_probability"] - data_sk["churn_probability"]) < 1e-4
        assert data_onnx["churn_prediction"] == data_sk["churn_prediction"]

    def test_predict_with_shap_explanations(self, client, sample_payload):
        resp = client.post("/predict?engine=onnx&explain=true", json=sample_payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["explanations"] is not None
        assert len(data["explanations"]) == 3  # Top 3 driving features

        top_feat = data["explanations"][0]
        assert "feature" in top_feat
        assert "shap_value" in top_feat
        assert "impact_percentage" in top_feat
        assert top_feat["direction"] in ["increases_risk", "decreases_risk"]

    def test_predict_batch(self, client, sample_payload):
        # Create batch of 3 customers
        batch_payload = {
            "customers": [
                {**sample_payload, "customer_id": f"CUST-BATCH-{i}"}
                for i in range(3)
            ]
        }
        resp = client.post("/predict_batch?engine=onnx", json=batch_payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["total_processed"] == 3
        assert len(data["predictions"]) == 3
        assert data["batch_latency_ms"] >= 0.0

    def test_invalid_payload_rejected_with_422(self, client, sample_payload):
        # Corrupt the payload with negative age and invalid credit score
        bad_payload = {**sample_payload, "age": -10, "credit_score": 999}
        resp = client.post("/predict", json=bad_payload)

        # FastAPI Pydantic validator must reject before execution with HTTP 422
        assert resp.status_code == 422
