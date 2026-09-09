"""Standalone real inference + Grad-CAM command."""
import argparse
from pathlib import Path

from backend.inference import InferenceEngine


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image", help="Path to any real JPG/PNG leaf image")
    p.add_argument("--model", default="models/plantvillage_best.h5")
    p.add_argument("--labels", default="models/labels.json")
    p.add_argument("--output", default="reports/gradcam_overlay.png")
    args = p.parse_args()

    raw = Path(args.image).read_bytes()
    result = InferenceEngine(args.model, args.labels).predict(raw)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(result.pop("heatmap_png"))
    print("REAL INFERENCE RESULT")
    for k, v in result.items():
        print(f"{k}: {v}")
    print(f"heatmap: {out}")


if __name__ == "__main__":
    main()
