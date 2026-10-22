# 🚀 Enterprise MLOps Platform: Production Customer Churn Predictor

[![CI/CD Pipeline](https://github.com/GarvitSingh13/Production-mlops-platform/actions/workflows/ci-cd.yaml/badge.svg)](https://github.com/GarvitSingh13/Production-mlops-platform/actions/workflows/ci-cd.yaml)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-56x_Speedup-005CED?style=flat&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-0194E2?style=flat&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Multi--Stage_Hardened-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-HPA_&_Canary-326CE5?style=flat&logo=kubernetes&logoColor=white)](https://kubernetes.io/)
[![Prometheus](https://img.shields.io/badge/Prometheus-Telemetry_Metrics-E6522C?style=flat&logo=prometheus&logoColor=white)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-Observability_Dashboard-F46800?style=flat&logo=grafana&logoColor=white)](https://grafana.com/)

An end-to-end, enterprise-grade Machine Learning Operations (MLOps) platform demonstrating production-readiness from data contracts and experiment tracking to sub-millisecond serving, Kubernetes canary routing, real-time telemetry, and automated statistical drift detection.

---

## 🌟 Executive Architectural Highlights

| Dimension | Implementation | Production Metric / Benefit |
| :--- | :--- | :--- |
| **Data Integrity** | Pydantic v2 Feature Contracts & Runtime Interceptors | Rejects out-of-boundary inputs before pipeline contamination. |
| **Experiment Tracking** | MLflow Tracking Server & Automated Evaluation | Full lineage, hyperparameter logging, ROC-AUC/PR curve artifact storage. |
| **Model Governance** | MLflow Model Registry Deployment Quality Gates | Semantic versioning, stage transitions (`Staging` ➔ `Production`), automated rollback. |
| **Inference Acceleration** | ONNX Graph Compilation & Runtime Session | **56x latency reduction** (sub-millisecond inference vs. Scikit-Learn). |
| **Live Explainability** | Real-Time SHAP (Shapley Additive exPlanations) | Per-prediction top feature attribution returned within HTTP response payload. |
| **Container Hardening** | Multi-Stage Minimal Distroless Docker Build | Non-root security user (`UID 10001`), cached wheel dependencies. |
| **Cloud Orchestration** | Kubernetes Deployments, HPA, & NGINX Canary Routing | Zero-downtime rolling updates, autoscaling at 70% CPU, 10% canary traffic splitting. |
| **Observability** | Prometheus Telemetry & Preconfigured Grafana Dashboard | Native `/metrics` endpoint tracking p50/p95/p99 latency histograms and risk tiers. |
| **Drift & Self-Healing** | Statistical Tests (KS-Test, PSI, $\chi^2$ Contingency) | Automated retraining dispatch when population stability index (PSI) > 0.20. |
| **CI/CD Quality Gate** | Multi-Job GitHub Actions Workflow | 5 sequential verification stages ensuring zero broken builds in production. |

---

## 🏛️ End-to-End System Architecture

```text
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                         DATA & MODEL GOVERNANCE                             │
  └─────────────────────────────────────────────────────────────────────────────┘
   Raw Tabular Data ──► [ Pydantic Contracts ] ──► [ Stratified Preprocessing ]
                                                           │
                                                           ▼
       [ Champion Model ] ◄── [ MLflow Registry ] ◄── [ MLflow Training ]
       (ROC-AUC: 0.887)       (Quality Gates)          (Multi-Model Race)
              │
              ▼
   [ ONNX Graph Compiler ] ──► (56x Faster Sub-millisecond Execution)
                                                           │
  ┌────────────────────────────────────────────────────────┼────────────────────┐
  │                   SERVING & CLOUD INFRASTRUCTURE       │                    │
  └────────────────────────────────────────────────────────┼────────────────────┘
                                                           ▼
                                               [ FastAPI Microservice ]
                                               ├── /predict (Dual-Engine)
                                               ├── /predict/batch
                                               ├── /health (K8s Probes)
                                               └── /metrics (Prometheus)
                                                           │
                 ┌─────────────────────────────────────────┴─────────────────────┐
                 ▼                                                               ▼
     [ Kubernetes Cluster ]                                          [ Observability ]
     ├── NGINX Canary (90% Prod / 10% Canary)                        ├── Prometheus Scraper
     ├── Horizontal Pod Autoscaler (2-10 Pods)                       └── Grafana Dashboard
     └── RollingUpdate Zero-Downtime Deployment
                                                           │
  ┌────────────────────────────────────────────────────────┼────────────────────┐
  │                   DRIFT DETECTION & AUTOMATED CI/CD    │                    │
  └────────────────────────────────────────────────────────┼────────────────────┘
                                                           ▼
                                            [ Statistical Drift Engine ]
                                            ├── KS-Test (Numerical Shift)
                                            ├── Chi-Square (Categorical)
                                            └── PSI (Population Stability)
                                                           │
                                                           ▼ (If PSI > 0.20)
                                            [ Automated Retraining Trigger ]
                                                           │
                                                           ▼
                                            [ GitHub Actions CI/CD Pipeline ]
                                            (5/5 Automated Quality Gates)
```

---

## 📂 Repository Structure

```text
├── .github/
│   └── workflows/
│       └── ci-cd.yaml              # 5-Job Enterprise GitHub Actions Pipeline
├── data/
│   ├── raw/                        # Baseline & intentionally drifted customer data
│   └── processed/                  # Stratified Parquet splits & preprocessor.joblib
├── k8s/
│   ├── deployment.yaml             # Kubernetes Deployment, Service, ConfigMap, Secret
│   ├── hpa.yaml                    # Horizontal Pod Autoscaler (2-10 replicas, 70% CPU)
│   └── canary-ingress.yaml         # NGINX Ingress with 10% Canary traffic split
├── metrics/
│   ├── prometheus.yml              # Prometheus scraping configuration & targets
│   ├── alert_rules.yml             # Prometheus alerting rules (P95 Latency & Error Rate)
│   └── grafana-dashboard.json      # Production Grafana monitoring dashboard definition
├── mlops_platform/
│   ├── config/                     # Typed configuration management (Pydantic-Settings)
│   ├── data/                       # Generator, contracts, and preprocessing pipelines
│   ├── models/                     # Evaluator, trainer, registry manager, and ONNX compiler
│   ├── monitoring/                 # Prometheus telemetry and statistical drift detector
│   └── serving/                    # FastAPI microservice, schemas, and live SHAP explainer
├── models/
│   └── champion/                   # Serialized champion model (.joblib, .onnx, metadata)
├── tests/                          # Comprehensive pytest suites for Stages 1 through 9
├── Dockerfile                      # Hardened multi-stage build (builder + non-root runtime)
├── docker-compose.yml              # Multi-container orchestration (FastAPI + Prometheus + Grafana)
├── requirements.txt                # Pinned production dependencies
└── run_all_stages.py               # Master end-to-end verification orchestrator (65/65 tests)
```

---

## ⚡ Quickstart Guide

### 1. Prerequisites & Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/GarvitSingh13/Production-mlops-platform.git
cd Production-mlops-platform

python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run the Full End-to-End Verification Pipeline

Execute all 9 platform stages sequentially in a single command:

```bash
python run_all_stages.py
```
*(Executes all data transformations, MLflow candidate races, registry promotions, ONNX compilations, serving tests, container audits, and drift evaluations — **65/65 tests passing in ~30s**).*

---

## 🔍 API Usage & Live SHAP Attribution

Start the serving microservice locally:

```bash
uvicorn mlops_platform.serving.app:app --host 0.0.0.0 --port 8000
```

### Single Prediction Request with SHAP (`POST /predict`)

```bash
curl -X POST "http://localhost:8000/predict?engine=onnx&explain=true" \
     -H "Content-Type: application/json" \
     -d '{
       "customer_id": "CUST-ENTERPRISE-001",
       "credit_score": 620,
       "age": 42,
       "tenure_months": 3,
       "monthly_charges": 89.50,
       "total_charges": 268.50,
       "num_support_tickets": 5,
       "contract_type": "month-to-month",
       "payment_method": "electronic-check",
       "has_paperless_billing": true
     }'
```

### Response Payload:

```json
{
  "customer_id": "CUST-ENTERPRISE-001",
  "churn_prediction": 1,
  "churn_probability": 0.8124,
  "risk_tier": "HIGH",
  "engine_used": "onnx",
  "model_version": "v1",
  "latency_ms": 0.84,
  "explanations": [
    {
      "feature": "contract_type_month-to-month",
      "shap_value": 0.3842,
      "impact_percentage": 42.1,
      "direction": "increases_risk"
    },
    {
      "feature": "num_support_tickets",
      "shap_value": 0.2815,
      "impact_percentage": 30.8,
      "direction": "increases_risk"
    },
    {
      "feature": "tenure_months",
      "shap_value": -0.1450,
      "impact_percentage": 15.9,
      "direction": "decreases_risk"
    }
  ]
}
```

---

## 🐳 Docker & Container Orchestration

Run the entire serving and observability stack in one command via Docker Compose:

```bash
docker-compose up -d --build
```

Access services:
* **FastAPI Interactive Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Prometheus Metrics**: [http://localhost:9090](http://localhost:9090)
* **Grafana Command Center**: [http://localhost:3000](http://localhost:3000) *(Credentials: `admin` / `admin`)*

---

## ☸️ Kubernetes & Canary Traffic Splitting

The platform includes production-ready Kubernetes manifests:

* **Zero-Downtime Deployment**: Configured with `maxSurge: 1` and `maxUnavailable: 0` alongside Kubernetes `readinessProbe`, `livenessProbe`, and `startupProbe`.
* **HPA Autoscaler**: Automatically scales pod replicas between 2 and 10 based on a 70% CPU target utilization.
* **Canary Routing**: Uses NGINX Ingress annotations to route 10% of live traffic to the candidate model:
  ```yaml
  nginx.ingress.kubernetes.io/canary: "true"
  nginx.ingress.kubernetes.io/canary-weight: "10"
  ```

---

## 📈 Real-Time Observability & Monitoring

The `/metrics` endpoint exports Prometheus telemetry natively:
* `mlops_inference_requests_total`: Counter partitioned by `endpoint`, `status`, `engine`, and `prediction_class`.
* `mlops_inference_latency_seconds`: High-precision histogram with custom buckets (`[0.001, 0.005, 0.010, 0.025, 0.050, 0.100, 0.250, 0.500]`).
* `mlops_risk_tier_distribution_total`: Tracks customer distribution into `LOW`, `MEDIUM`, and `HIGH` risk tiers.
* `mlops_data_drift_alert`: Gauge signaling statistical drift alerts to PagerDuty/Slack.

---

## 🔄 Automated Drift Detection & Self-Healing

The monitoring engine calculates distribution drift across rolling production windows:

* **Numerical Features**: Evaluated with the **Two-Sample Kolmogorov-Smirnov Test (KS-Test)** ($\alpha = 0.05$).
* **Categorical Features**: Evaluated via **Chi-Square Contingency Test** ($\chi^2$).
* **Population Stability Index (PSI)**: Quantifies aggregate distribution shift.
  * $\text{PSI} < 0.10$: **Healthy** (No action needed).
  * $0.10 \le \text{PSI} \le 0.20$: **Moderate Shift** (Warning alert logged).
  * $\text{PSI} > 0.20$: **Significant Drift** ➔ Dispatches automated retraining trigger.

---

## 🛡️ Enterprise CI/CD Pipeline

Every commit and pull request is audited through a 5-job GitHub Actions workflow:

```text
[ Job 1: Code Quality & Static Linting ]
      │
      ▼
[ Job 2: Data Pipeline, MLflow Tracking & ONNX Gate ]
      │
      ▼
[ Job 3: Serving Microservice, Telemetry & Drift Engine ]
      │
      ▼
[ Job 4: Container Security & K8s Infrastructure Audit ]
      │
      ▼
[ Job 5: Master End-to-End Verification & Coverage Gate ]
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
