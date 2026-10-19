"""
ONNX Model Acceleration & Benchmark Engine
==========================================
Converts Scikit-Learn models to ONNX format (Open Neural Network Exchange),
validates numerical output parity, and benchmarks CPU inference acceleration.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import time
import json
import numpy as np
import pandas as pd
import joblib

import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from mlops_platform.config.settings import settings


class ONNXConverter:
    """Converts, validates, and benchmarks ONNX runtime performance for tabular models."""

    def __init__(self, output_dir: Path = settings.MODELS_DIR / "champion"):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.onnx_path = self.output_dir / "model.onnx"
        self.benchmark_path = self.output_dir / "onnx_benchmark.json"

    def convert(self, sklearn_model: Any, n_features: int) -> Path:
        """
        Convert scikit-learn model to ONNX graph with standard 2D array output.

        Args:
            sklearn_model: Fitted Scikit-learn estimator.
            n_features: Number of input features in transformed space.

        Returns:
            Path to serialized model.onnx.
        """
        initial_type = [("float_input", FloatTensorType([None, n_features]))]

        # zipmap=False ensures probabilities are output as a clean float32 [N, 2] array
        options = {type(sklearn_model): {"zipmap": False}}

        onnx_model = convert_sklearn(
            sklearn_model,
            name="customer_churn_onnx",
            initial_types=initial_type,
            options=options,
            target_opset=12,
        )

        with open(self.onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        print(f"  [OK] Successfully serialized ONNX model: {self.onnx_path} ({self.onnx_path.stat().st_size / 1024:.1f} KB)")
        return self.onnx_path

    def benchmark_and_verify(
        self,
        sklearn_model: Any,
        sample_df: pd.DataFrame,
        n_iterations: int = 300,
    ) -> Dict[str, Any]:
        """
        Verify numerical parity between Scikit-learn and ONNX Runtime and benchmark latency.

        Args:
            sklearn_model: Baseline Scikit-learn estimator.
            sample_df: DataFrame with test feature rows.
            n_iterations: Number of single-row inference passes to benchmark.

        Returns:
            Dictionary with parity status and latency comparison statistics.
        """
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {self.onnx_path}. Run convert() first.")

        # 1. Initialize ONNX Runtime Session
        session = ort.InferenceSession(str(self.onnx_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[1].name  # Output 1 is probabilities array

        # 2. Verify Numerical Parity across batch
        X_batch_float32 = sample_df.values.astype(np.float32)
        sklearn_probs = sklearn_model.predict_proba(sample_df)[:, 1]
        onnx_raw = session.run([output_name], {input_name: X_batch_float32})[0]
        onnx_probs = onnx_raw[:, 1]

        max_abs_diff = float(np.max(np.abs(sklearn_probs - onnx_probs)))
        parity_verified = max_abs_diff < 1e-4

        if not parity_verified:
            raise ValueError(f"Numerical disparity between Scikit-learn and ONNX: max diff = {max_abs_diff}")

        # 3. Single-Sample Latency Benchmark (p50, p95)
        benchmark_samples = sample_df.iloc[:min(n_iterations, len(sample_df))]

        # Scikit-learn latency
        sk_latencies = []
        for i in range(len(benchmark_samples)):
            row = benchmark_samples.iloc[[i]]
            t0 = time.perf_counter()
            _ = sklearn_model.predict_proba(row)
            sk_latencies.append((time.perf_counter() - t0) * 1000)

        # ONNX Runtime latency
        onnx_latencies = []
        for i in range(len(benchmark_samples)):
            row_arr = benchmark_samples.iloc[i].values.reshape(1, -1).astype(np.float32)
            t0 = time.perf_counter()
            _ = session.run([output_name], {input_name: row_arr})
            onnx_latencies.append((time.perf_counter() - t0) * 1000)

        sk_p50 = float(np.percentile(sk_latencies, 50))
        sk_p95 = float(np.percentile(sk_latencies, 95))
        onnx_p50 = float(np.percentile(onnx_latencies, 50))
        onnx_p95 = float(np.percentile(onnx_latencies, 95))

        speedup = sk_p95 / max(onnx_p95, 0.001)

        summary = {
            "numerical_parity_verified": parity_verified,
            "max_absolute_difference": max_abs_diff,
            "benchmark_sample_count": len(benchmark_samples),
            "sklearn_latency_p50_ms": round(sk_p50, 4),
            "sklearn_latency_p95_ms": round(sk_p95, 4),
            "onnx_latency_p50_ms": round(onnx_p50, 4),
            "onnx_latency_p95_ms": round(onnx_p95, 4),
            "p95_speedup_factor": round(speedup, 2),
            "onnx_model_path": str(self.onnx_path),
        }

        with open(self.benchmark_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary
