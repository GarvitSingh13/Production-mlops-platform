# -----------------------------------------------------------------------------
# Stage 1: Build & Dependency Wheel Cache
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install minimal build tools for C extensions (e.g. OpenMP for Scikit-Learn/ONNX)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pre-compile Python wheels to avoid shipping build compilers in final image
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Minimal Production Runtime
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8000

# Install runtime-only dependencies (curl for container healthchecks, OpenMP runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Security Hardening: Create non-root user (UID 10001) to prevent container escape
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/sh -m appuser

# Install pre-built wheels from builder stage
COPY --from=builder /app/wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir /wheels/* && \
    rm -rf /wheels

# Copy application code, preprocessors, and champion models
COPY mlops_platform/ mlops_platform/
COPY data/processed/preprocessor.joblib data/processed/preprocessor.joblib
COPY data/processed/train.parquet data/processed/train.parquet
COPY models/champion/ models/champion/

# Set strict permissions and switch to non-root user
RUN chown -R appuser:appgroup /app
USER appuser

# Container Healthcheck polling the FastAPI liveness probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

EXPOSE 8000

# Start production ASGI server with multiple workers
CMD ["uvicorn", "mlops_platform.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
