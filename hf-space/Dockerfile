FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_VISIBLE_DEVICES=-1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    TF_NUM_INTRAOP_THREADS=1 \
    TF_NUM_INTEROP_THREADS=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    CROPGUARD_API_URL=http://127.0.0.1:8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/backend /app/frontend /app/models

ARG GITHUB_RAW=https://raw.githubusercontent.com/kingsohad37-stack/Cropguard.ai/main

RUN curl -fL "$GITHUB_RAW/requirements-api.txt" -o /app/requirements-api.txt \
    && curl -fL "$GITHUB_RAW/requirements-ui.txt" -o /app/requirements-ui.txt \
    && pip install --no-cache-dir -r /app/requirements-api.txt -r /app/requirements-ui.txt

RUN curl -fL "$GITHUB_RAW/backend/__init__.py" -o /app/backend/__init__.py \
    && curl -fL "$GITHUB_RAW/backend/main.py" -o /app/backend/main.py \
    && curl -fL "$GITHUB_RAW/backend/inference.py" -o /app/backend/inference.py \
    && curl -fL "$GITHUB_RAW/backend/treatments.json" -o /app/backend/treatments.json \
    && curl -fL "$GITHUB_RAW/frontend/app.py" -o /app/frontend/app.py \
    && curl -fL "$GITHUB_RAW/models/labels.json" -o /app/models/labels.json \
    && curl -fL "$GITHUB_RAW/models/metrics.json" -o /app/models/metrics.json || true

RUN curl -fL "$GITHUB_RAW/models/plantvillage_best.keras" -o /app/models/plantvillage_best.keras

EXPOSE 7860

COPY start_hf.sh /app/start_hf.sh

CMD ["bash", "/app/start_hf.sh"]
