# CropGuard AI

CropGuard AI is a real PlantVillage leaf-classification demo: TensorFlow MobileNetV2 inference, request-specific GradientTape Grad-CAM, an image-derived severity estimate, and SQLite scan history. It deliberately ships without a model, dataset, history, or measured metrics. It will return HTTP 503 for prediction until a genuine model has been trained.

## Install

Windows:
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Kaggle and data preparation

Create a Kaggle API token in Kaggle account settings. Put `kaggle.json` in `%USERPROFILE%\.kaggle\` (or configure the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables), then run:
```powershell
python scripts/data_pipeline.py --dataset mohitsingh1804/plantvillage
```

The script downloads the actual [PlantVillage Kaggle dataset](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage), discovers images and labels from folders, removes exact duplicate content before splitting, creates reproducible stratified train/validation/test CSVs, and writes the discovered labels plus `data/splits/summary.json`. The dataset contents—not an assumed class count—are authoritative.

## Train and evaluate

```powershell
python scripts/train.py --epochs 8 --fine-tune-epochs 4 --batch-size 32 --seed 42
python scripts/evaluate.py
```

Training uses ImageNet-initialized MobileNetV2, augmentation, a checkpoint, early stopping, a frozen-backbone stage, and upper-backbone fine tuning. It creates `models/plantvillage_best.keras`, optional H5 export, training log, and held-out metrics. Evaluation writes actual test loss/accuracy, precision, recall, F1, per-class metrics, and a confusion matrix. Metrics are valid only after real training/evaluation—this repository contains no claimed accuracy.

## Run

```powershell
uvicorn backend.main:app --reload --port 8000
streamlit run frontend/app.py
```

### One-click Windows startup

After installation, double-click `Start_CropGuard.cmd` in the project folder.
It opens the API and dashboard in separate windows and launches the dashboard in
your browser. Keep those two windows open (or minimized) while using CropGuard;
a local web app needs its server processes running to respond to the browser.

The API exposes `/health`, `/metrics`, `/history`, and `/predict`. It reads uploaded image bytes directly, validates them with Pillow, returns their SHA-256 digest, and never writes uploads to disk. `/predict` emits the real prediction vector’s top three labels and probabilities; Grad-CAM is calculated using `tf.GradientTape` on that request’s model tensor. Treatment guidance is label-normalized and shows an explicit unavailable message if no matching record exists.

## Docker

```powershell
docker compose up --build
```

The mounted project must already contain a trained model and labels before the API can analyze images.

## Render + Supabase deployment

`render.yaml` defines two Render services: a FastAPI inference API and a
Streamlit dashboard. Both services require the trained `models/` artifacts to
be committed to Git. The dashboard needs `CROPGUARD_API_URL` set to the public
URL of the API after the API is deployed.

To use Supabase scan history, run `supabase/migrations/001_create_scans.sql`
in the Supabase SQL Editor, then set `SUPABASE_URL` and
`SUPABASE_SERVICE_ROLE_KEY` **only on the API service**. Never put the
service-role key in the Streamlit service, browser code, Git, or `.env` files.
Without those variables the app uses local SQLite for development.

The Blueprint uses Render's free plan. TensorFlow can require more memory than
a free instance provides; if Render reports an out-of-memory startup failure,
choose a paid instance only after reviewing Render's current price.

## Verification

```powershell
python -m compileall -q backend frontend scripts tests
pytest -q
```

To verify this is not a mock — upload a clearly healthy leaf and a clearly diseased leaf; the app must produce two DIFFERENT diagnoses and two DIFFERENT heatmaps, proving the model is actually running inference, not returning canned output.

PlantVillage is mainly controlled-condition imagery; field performance may differ. This is not a substitute for professional agricultural diagnosis. Grad-CAM indicates model attention, not definitive biological causation. Severity is an AI-derived estimate based on activation over an image-derived leaf/foreground area, not laboratory-grade measurement.
