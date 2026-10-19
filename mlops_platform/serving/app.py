"""
FastAPI High-Performance Inference Microservice
===============================================
Asynchronous REST API providing single and batch predictions, dual-engine
execution (Scikit-Learn vs. ONNX Runtime), live SHAP explainability, and
Kubernetes health probes.
"""

import json
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, Literal

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Response, status

from mlops_platform.config.settings import settings
from mlops_platform.data.contracts import InferencePayload
from mlops_platform.monitoring.telemetry import (
    ACTIVE_REQUESTS_GAUGE,
    CONTENT_TYPE_LATEST,
    get_prometheus_metrics,
    record_batch_metrics,
    record_inference_metric,
    set_model_info,
)
from mlops_platform.serving.explainer import SHAPExplainer
from mlops_platform.serving.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    LivenessResponse,
    PredictionResponse,
    ReadinessResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle event handler to load models, preprocessors, ONNX sessions,
    and warm up SHAP explainers into memory on application startup.
    """
    print("[STARTUP] Initializing Model Serving Microservice...")

    # 1. Load Scikit-Learn Preprocessor
    if not settings.PREPROCESSOR_PATH.exists():
        raise RuntimeError(f"Missing preprocessor at {settings.PREPROCESSOR_PATH}. Run Stage 1 first.")
    app.state.preprocessor = joblib.load(settings.PREPROCESSOR_PATH)

    # 2. Load Champion Scikit-Learn Model & Metadata
    champion_dir = settings.MODELS_DIR / "champion"
    model_path = champion_dir / "model.joblib"
    meta_path = champion_dir / "metadata.json"

    if not model_path.exists():
        raise RuntimeError(f"Missing champion model at {model_path}. Run Stage 2 first.")
    app.state.sklearn_model = joblib.load(model_path)

    if meta_path.exists():
        with open(meta_path, "r") as f:  # noqa: ASYNC230
            app.state.model_meta = json.load(f)
    else:
        app.state.model_meta = {"model_name": "LogisticRegression", "run_id": "unknown"}

    # 3. Load ONNX Runtime Session
    onnx_path = champion_dir / "model.onnx"
    if onnx_path.exists():
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        app.state.onnx_session = session
        app.state.onnx_input_name = session.get_inputs()[0].name
        app.state.onnx_output_name = session.get_outputs()[1].name  # Probabilities tensor
        app.state.onnx_loaded = True
    else:
        app.state.onnx_session = None
        app.state.onnx_loaded = False

    # 4. Extract Feature Names
    cat_step = app.state.preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_features = cat_step.get_feature_names_out(settings.CATEGORICAL_FEATURES).tolist()
    app.state.feature_names = settings.NUMERICAL_FEATURES + cat_features

    # 5. Initialize SHAP Explainer with background sample
    bg_df = None
    if settings.TRAIN_DATA_PATH.exists():
        train_df = pd.read_parquet(settings.TRAIN_DATA_PATH)
        bg_features = [c for c in train_df.columns if c != settings.TARGET_COLUMN]
        bg_df = train_df[bg_features].head(50)

    app.state.explainer = SHAPExplainer(
        model=app.state.sklearn_model,
        feature_names=app.state.feature_names,
        background_data=bg_df,
    )

    # 6. Set Model Info Gauge in Prometheus
    set_model_info(
        model_name=app.state.model_meta.get("model_name", "LogisticRegression"),
        model_version=str(app.state.model_meta.get("run_id", "v1"))[:8],
        onnx_accelerated=app.state.onnx_loaded,
    )

    print(f"[STARTUP] Serving service ready. Active Model: {app.state.model_meta.get('model_name')}")
    yield
    print("[SHUTDOWN] Cleaning up model serving resources.")


# Initialize FastAPI App
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=(
        "Enterprise-grade MLOps model serving API with dual Scikit-Learn/ONNX execution, "
        "real-time SHAP explainability, and Kubernetes observability."
    ),
    lifespan=lifespan,
)


@app.get("/", tags=["System"])
async def root() -> dict[str, Any]:
    """Root metadata endpoint."""
    return {
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "online",
        "documentation": "/docs",
        "endpoints": {
            "predict_single": "POST /predict",
            "predict_batch": "POST /predict_batch",
            "liveness_probe": "GET /health/live",
            "readiness_probe": "GET /health/ready",
            "model_metadata": "GET /model/metadata",
            "prometheus_metrics": "GET /metrics",
        },
    }


@app.get("/metrics", tags=["Observability"])
async def metrics():
    """Prometheus exposition format metrics scraper endpoint."""
    return Response(content=get_prometheus_metrics(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health/live", response_model=LivenessResponse, tags=["Health"])
async def liveness_probe():
    """Kubernetes liveness probe ensuring the server process is alive."""
    return LivenessResponse(
        status="alive",
        timestamp=datetime.now(UTC).isoformat(),
    )


@app.get("/health/ready", response_model=ReadinessResponse, tags=["Health"])
async def readiness_probe():
    """Kubernetes readiness probe verifying models and preprocessors are loaded in memory."""
    is_ready = (
        hasattr(app.state, "sklearn_model")
        and app.state.sklearn_model is not None
        and hasattr(app.state, "preprocessor")
        and app.state.preprocessor is not None
    )

    if not is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Serving dependencies are not yet loaded in memory.",
        )

    return ReadinessResponse(
        status="ready",
        model_name=app.state.model_meta.get("model_name", "Unknown"),
        model_version=str(app.state.model_meta.get("run_id", "v1")),
        sklearn_loaded=app.state.sklearn_model is not None,
        onnx_loaded=app.state.onnx_loaded,
        preprocessor_loaded=app.state.preprocessor is not None,
        explainer_loaded=app.state.explainer is not None,
    )


@app.get("/model/metadata", tags=["Model"])
async def get_model_metadata():
    """Retrieve active production model details, feature schemas, and benchmarks."""
    benchmark_data = None
    bench_file = settings.MODELS_DIR / "champion" / "onnx_benchmark.json"
    if bench_file.exists():
        with open(bench_file, "r") as f:  # noqa: ASYNC230
            benchmark_data = json.load(f)

    return {
        "model_metadata": app.state.model_meta,
        "feature_count": len(app.state.feature_names),
        "feature_names": app.state.feature_names,
        "numerical_inputs": settings.NUMERICAL_FEATURES,
        "categorical_inputs": settings.CATEGORICAL_FEATURES,
        "onnx_acceleration_benchmark": benchmark_data,
    }


def _payload_to_dataframe(payload: InferencePayload) -> pd.DataFrame:
    """Helper to convert Pydantic InferencePayload into a pandas DataFrame."""
    return pd.DataFrame([payload.model_dump()])


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_single(
    payload: InferencePayload,
    engine: Literal["onnx", "sklearn"] = Query("onnx", description="Inference engine to use"),
    explain: bool = Query(True, description="Compute top 3 SHAP feature attributions"),
):
    """
    Generate churn prediction for a single customer record with optional SHAP explainability.
    """
    ACTIVE_REQUESTS_GAUGE.labels(endpoint="/predict").inc()
    t0 = time.perf_counter()
    try:
        # 1. Transform raw payload using ColumnTransformer
        df_raw = _payload_to_dataframe(payload)
        X_raw = df_raw[settings.NUMERICAL_FEATURES + settings.CATEGORICAL_FEATURES]
        X_transformed = app.state.preprocessor.transform(X_raw)

        # 2. Compute Probability based on selected engine
        if engine == "onnx" and app.state.onnx_loaded:
            onnx_input = {app.state.onnx_input_name: X_transformed.astype(np.float32)}
            raw_output = app.state.onnx_session.run([app.state.onnx_output_name], onnx_input)[0]
            churn_prob = float(raw_output[0, 1])
            engine_used = "onnx"
        else:
            # Fallback or explicit Scikit-Learn
            df_transformed = pd.DataFrame(X_transformed, columns=app.state.feature_names)
            churn_prob = float(app.state.sklearn_model.predict_proba(df_transformed)[0, 1])
            engine_used = "sklearn"

        # 3. Classify and Tier Risk
        prediction = 1 if churn_prob >= 0.50 else 0
        if churn_prob >= 0.70:
            risk_tier = "HIGH"
        elif churn_prob >= 0.40:
            risk_tier = "MEDIUM"
        else:
            risk_tier = "LOW"

        # 4. Compute SHAP Explanations (if requested)
        explanations = None
        if explain and app.state.explainer is not None:
            try:
                explanations = app.state.explainer.explain_instance(X_transformed[0], top_n=3)
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] Failed to compute SHAP attribution: {e}")

        compute_latency_s = time.perf_counter() - t0
        compute_latency_ms = round(compute_latency_s * 1000, 4)

        # 5. Record Prometheus Metrics
        record_inference_metric(
            endpoint="/predict",
            engine=engine_used,
            status_code=200,
            prediction_class=prediction,
            probability=churn_prob,
            risk_tier=risk_tier,
            latency_seconds=compute_latency_s,
        )

        return PredictionResponse(
            customer_id=payload.customer_id,
            churn_prediction=prediction,
            churn_probability=round(churn_prob, 4),
            risk_tier=risk_tier,
            engine_used=engine_used,
            model_version=str(app.state.model_meta.get("run_id", "v1"))[:8],
            latency_ms=compute_latency_ms,
            explanations=explanations,
        )
    finally:
        ACTIVE_REQUESTS_GAUGE.labels(endpoint="/predict").dec()


@app.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch(
    request: BatchPredictionRequest,
    engine: Literal["onnx", "sklearn"] = Query("onnx", description="Inference engine to use"),
):
    """
    High-throughput vectorized bulk prediction for up to 1,000 customers in a single request.
    """
    ACTIVE_REQUESTS_GAUGE.labels(endpoint="/predict_batch").inc()
    t0 = time.perf_counter()
    try:
        # 1. Vectorized DataFrame Assembly
        records = [c.model_dump() for c in request.customers]
        df_raw = pd.DataFrame(records)
        X_raw = df_raw[settings.NUMERICAL_FEATURES + settings.CATEGORICAL_FEATURES]
        X_transformed = app.state.preprocessor.transform(X_raw)

        # 2. Vectorized Probability Calculation
        if engine == "onnx" and app.state.onnx_loaded:
            onnx_input = {app.state.onnx_input_name: X_transformed.astype(np.float32)}
            raw_output = app.state.onnx_session.run([app.state.onnx_output_name], onnx_input)[0]
            churn_probs = raw_output[:, 1].tolist()
            engine_used = "onnx"
        else:
            df_transformed = pd.DataFrame(X_transformed, columns=app.state.feature_names)
            churn_probs = app.state.sklearn_model.predict_proba(df_transformed)[:, 1].tolist()
            engine_used = "sklearn"

        # 3. Assemble Responses
        predictions: list[PredictionResponse] = []
        version_str = str(app.state.model_meta.get("run_id", "v1"))[:8]

        for idx, prob in enumerate(churn_probs):
            pred = 1 if prob >= 0.50 else 0
            if prob >= 0.70:
                tier = "HIGH"
            elif prob >= 0.40:
                tier = "MEDIUM"
            else:
                tier = "LOW"

            predictions.append(
                PredictionResponse(
                    customer_id=request.customers[idx].customer_id,
                    churn_prediction=pred,
                    churn_probability=round(prob, 4),
                    risk_tier=tier,
                    engine_used=engine_used,
                    model_version=version_str,
                    latency_ms=0.0,
                    explanations=None,
                )
            )

        batch_latency_s = time.perf_counter() - t0
        batch_latency_ms = round(batch_latency_s * 1000, 3)

        # 4. Record Prometheus Telemetry
        churn_count = sum(1 for p in churn_probs if p >= 0.50)
        record_batch_metrics(
            endpoint="/predict_batch",
            engine=engine_used,
            status_code=200,
            total_processed=len(predictions),
            churn_count=churn_count,
            probabilities=churn_probs,
            latency_seconds=batch_latency_s,
        )

        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions),
            batch_latency_ms=batch_latency_ms,
            engine_used=engine_used,
        )
    finally:
        ACTIVE_REQUESTS_GAUGE.labels(endpoint="/predict_batch").dec()
