# Build stage: TensorFlow is used only to convert the committed genuine Keras
# checkpoint into a deployment-friendly TFLite representation.
FROM python:3.11-slim AS model-builder
WORKDIR /work
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=-1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    TF_NUM_INTRAOP_THREADS=1 \
    TF_NUM_INTEROP_THREADS=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

RUN pip install --no-cache-dir tensorflow==2.18.0
COPY models/plantvillage_best.keras models/plantvillage_best.keras
COPY models/labels.json models/labels.json
COPY scripts/convert_model_to_tflite.py scripts/convert_model_to_tflite.py
RUN python scripts/convert_model_to_tflite.py

# Runtime stage: only the lightweight TFLite interpreter is installed.
FROM python:3.11-slim
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=-1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    CROPGUARD_API_URL=http://127.0.0.1:8000

COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

COPY requirements-ui.txt ./
RUN pip install --no-cache-dir -r requirements-ui.txt

COPY . /app
COPY --from=model-builder /work/models/plantvillage_best.tflite /app/models/plantvillage_best.tflite
COPY --from=model-builder /work/models/tflite_metadata.json /app/models/tflite_metadata.json

EXPOSE 7860
CMD ["bash", "/app/start_hf.sh"]
