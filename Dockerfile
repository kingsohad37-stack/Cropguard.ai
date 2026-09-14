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

COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

COPY requirements-ui.txt ./
RUN pip install --no-cache-dir -r requirements-ui.txt

COPY . /app

EXPOSE 7860

CMD ["bash", "/app/start_hf.sh"]
