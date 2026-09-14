#!/usr/bin/env bash
set -e

# Render exposes one public HTTP port via $PORT. Keep FastAPI internal on 8000
# and expose Streamlit on Render's assigned public port.
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --workers 1 &
API_PID=$!

streamlit run frontend/app.py \
  --server.address=0.0.0.0 \
  --server.port="${PORT:-7860}" \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false &
UI_PID=$!

cleanup() {
  kill "$API_PID" "$UI_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait -n "$API_PID" "$UI_PID"
