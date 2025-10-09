# Simple Dockerfile for Hugging Face Docker Space (FastAPI backend)

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    CUDA_VISIBLE_DEVICES=-1 \
    PORT=7860

# System deps (slim, no dev tools; add build-essential only if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (leveraging layer cache)
COPY requirements.txt /app/requirements.txt
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy backend and entrypoint
COPY backend /app/backend
COPY app.py /app/app.py

# Copy artifacts (required for inference)
COPY notebooks/artifacts_wp /app/notebooks/artifacts_wp
COPY notebooks/artifacts_comp /app/notebooks/artifacts_comp
COPY notebooks/artifacts_coach /app/notebooks/artifacts_coach

EXPOSE 7860

# Healthcheck against the FastAPI health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=5 \
  CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

