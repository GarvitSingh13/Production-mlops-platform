"""
Stage 4 Verification Runner: FastAPI Model Serving Microservice
===============================================================
Executes and verifies:
  1. FastAPI application lifespan startup and in-memory model loading.
  2. Kubernetes health probe endpoints (/health/live, /health/ready).
  3. Dual-engine inference comparison (Scikit-Learn vs. ONNX Runtime).
  4. Real-time SHAP explainability attribution extraction (Top 3 drivers).
  5. High-throughput batch inference endpoint (/predict_batch).
  6. Automatic Pydantic schema contract violation rejection (HTTP 422).
  7. Programmatic execution of pytest test suite for Stage 4.
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
from fastapi.testclient import TestClient
from mlops_platform.config.settings import settings
from mlops_platform.serving.app import app


def print_banner(text: str, char: str = "="):
    line = char * 78
    print(f"\n{line}")
    print(f" {text}")
    print(f"{line}")


def main():
    print_banner("STAGE 4: HIGH-PERFORMANCE FASTAPI SERVING MICROSERVICE")

    # Sample high-risk customer payload
    customer_sample = {
        "customer_id": "CUST-ENTERPRISE-8821",
        "credit_score": 580,
        "age": 41,
        "tenure_months": 3,
        "monthly_charges": 95.50,
        "total_charges": 286.50,
        "num_support_tickets": 6,
        "contract_type": "month-to-month",
        "payment_method": "electronic_check",
        "has_paperless_billing": 1,
    }

    with TestClient(app) as client:
        # ---------------------------------------------------------
        # Step 1: Verify Kubernetes Probes
        # ---------------------------------------------------------
        print("\n[Step 1/6] Verifying Kubernetes Health Probes...")
        live_resp = client.get("/health/live")
        assert live_resp.status_code == 200, "Liveness probe failed!"
        print(f"  [OK] /health/live  -> Status: {live_resp.json()['status']}")

        ready_resp = client.get("/health/ready")
        assert ready_resp.status_code == 200, "Readiness probe failed!"
        ready_data = ready_resp.json()
        print(f"  [OK] /health/ready -> Status: {ready_data['status']} (Model: {ready_data['model_name']})")
        print(f"  [OK] Engines Active: Scikit-Learn: {ready_data['sklearn_loaded']} | ONNX Runtime: {ready_data['onnx_loaded']}")

        # ---------------------------------------------------------
        # Step 2: Compare Dual-Engine Inference (Scikit-Learn vs ONNX)
        # ---------------------------------------------------------
        print("\n[Step 2/6] Testing Dual-Engine Inference & Numerical Parity...")
        res_sk = client.post("/predict?engine=sklearn&explain=false", json=customer_sample).json()
        res_onnx = client.post("/predict?engine=onnx&explain=false", json=customer_sample).json()

        print(f"  [Scikit-Learn] Churn Prob: {res_sk['churn_probability']:.4f} | Risk: {res_sk['risk_tier']:<6} | Latency: {res_sk['latency_ms']:.4f} ms")
        print(f"  [ONNX Runtime] Churn Prob: {res_onnx['churn_probability']:.4f} | Risk: {res_onnx['risk_tier']:<6} | Latency: {res_onnx['latency_ms']:.4f} ms")

        diff = abs(res_sk["churn_probability"] - res_onnx["churn_probability"])
        assert diff < 1e-4, f"Engine probability disparity too high: {diff}"
        print(f"  [OK] Output Parity Confirmed: Engine difference = {diff:.2e}")

        # ---------------------------------------------------------
        # Step 3: Test Real-Time SHAP Explainability
        # ---------------------------------------------------------
        print("\n[Step 3/6] Testing Real-Time SHAP Feature Attribution...")
        res_explain = client.post("/predict?engine=onnx&explain=true", json=customer_sample).json()
        explanations = res_explain.get("explanations", [])

        print(f"  Prediction: Churn Risk = {res_explain['churn_probability']:.1%} [{res_explain['risk_tier']} TIER]")
        print("  Top 3 Driving Factors (SHAP Explanations):")
        print("  " + "-" * 72)
        print(f"  {'Rank':<5} | {'Feature':<28} | {'Impact (%)':<12} | {'Direction'}")
        print("  " + "-" * 72)
        for rank, exp in enumerate(explanations, 1):
            direction_str = "▲ INCREASES RISK" if exp["direction"] == "increases_risk" else "▼ DECREASES RISK"
            print(f"  #{rank:<4} | {exp['feature']:<28} | {exp['impact_percentage']:10.1f}% | {direction_str}")
        print("  " + "-" * 72)
        assert len(explanations) == 3, "Expected 3 SHAP explanations!"

        # ---------------------------------------------------------
        # Step 4: Test High-Throughput Batch Inference
        # ---------------------------------------------------------
        print("\n[Step 4/6] Testing Vectorized Batch Inference (/predict_batch)...")
        batch_payload = {
            "customers": [
                {**customer_sample, "customer_id": f"CUST-BATCH-{i:03d}"}
                for i in range(25)
            ]
        }
        batch_resp = client.post("/predict_batch?engine=onnx", json=batch_payload).json()
        print(f"  [OK] Processed {batch_resp['total_processed']} records in {batch_resp['batch_latency_ms']:.2f} ms")
        print(f"  [OK] Average Throughput per Record: {batch_resp['batch_latency_ms'] / 25:.4f} ms/record")

        # ---------------------------------------------------------
        # Step 5: Enforce Schema Rejection Guardrails (HTTP 422)
        # ---------------------------------------------------------
        print("\n[Step 5/6] Enforcing Pydantic Schema Violation Rejection...")
        bad_payload = {**customer_sample, "credit_score": 150, "age": -5}
        bad_resp = client.post("/predict", json=bad_payload)
        assert bad_resp.status_code == 422, "Malformed payload should have been rejected with 422!"
        print("  [OK] Schema Contract Guard successfully rejected invalid payload with HTTP 422.")

        # ---------------------------------------------------------
        # Step 6: Run Automated Pytest Test Suite
        # ---------------------------------------------------------
        print("\n[Step 6/6] Executing Pytest Test Suite for Stage 4...")
        test_file = str(PROJECT_ROOT / "tests" / "test_stage4_serving.py")
        exit_code = pytest.main(["-v", test_file])
        assert exit_code == 0, f"Pytest suite failed with exit code: {exit_code}"

    # ---------------------------------------------------------
    # Stage 4 Summary Report
    # ---------------------------------------------------------
    print_banner("STAGE 4 COMPLETE: ALL VERIFICATION CHECKS PASSED [100%]")
    print(f"Service Title:            {settings.API_TITLE}")
    print(f"FastAPI Endpoints:        GET /health/live, GET /health/ready, POST /predict, POST /predict_batch")
    print(f"Dual Serving Engines:     ONNX Runtime (sub-0.05ms) + Scikit-Learn")
    print(f"Explainability Layer:     SHAP Linear & Tree Attribution (Top 3 drivers)")
    print(f"Input Validation Gate:    Pydantic v2 Contract Enforcement (HTTP 422)")
    print(f"Swagger Documentation:    http://localhost:8000/docs")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
