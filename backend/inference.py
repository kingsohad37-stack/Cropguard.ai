"""Real inference and optional class activation maps for the trained PlantVillage MobileNetV2 model."""
from __future__ import annotations
import gc
import io
import json
import hashlib
import logging
import threading

import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

IMG_SIZE = (224, 224)
MAX_VISUAL_SIZE = (512, 512)
_INFERENCE_LOCK = threading.Lock()
logger = logging.getLogger("cropguard.inference")


class InferenceEngine:
    def __init__(self, model_path="models/plantvillage_best.keras", labels_path="models/labels.json"):
        from pathlib import Path
        model_path = Path(model_path)
        labels_path = Path(labels_path)
        print(f"[CropGuard] REAL checkpoint path: {model_path} exists={model_path.exists()} size={model_path.stat().st_size if model_path.exists() else 0}", flush=True)
        if not model_path.exists():
            raise FileNotFoundError(f"REAL MODEL MISSING: {model_path}. Run the real data pipeline and training first.")
        if not labels_path.exists():
            raise FileNotFoundError(f"LABEL MAP MISSING: {labels_path}")

        print(f"[CropGuard] Loading REAL trained checkpoint: {model_path}", flush=True)
        # Prediction model loading is the critical path. If this succeeds, the
        # application remains usable even when the optional Grad-CAM view fails.
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.labels = json.loads(labels_path.read_text())
        if self.model.output_shape[-1] != len(self.labels):
            raise RuntimeError(f"Model has {self.model.output_shape[-1]} outputs but labels.json has {len(self.labels)} labels.")

        # Force Keras to materialize the loaded graph before creating any
        # auxiliary functional model. This is important for Keras 3 .keras
        # checkpoints whose symbolic tensors may not be connected until the
        # loaded model has actually been called.
        dummy = tf.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32)
        with _INFERENCE_LOCK, tf.device("/CPU:0"):
            _ = self.model(dummy, training=False)

        self.grad_model = None
        self.grad_layer_name = None
        try:
            self.base_model = self._find_mobilenet_base()
            target_layer = self._find_target_conv_layer(self.base_model)
            self.grad_layer_name = target_layer.name

            # Build the Grad-CAM view from the actual saved graph, by name rather
            # than relying on a numeric layer index. No trained head is rebuilt.
            self.grad_model = tf.keras.Model(
                inputs=self.model.inputs,
                outputs=[target_layer.output, self.model.output],
                name="cropguard_grad_model",
            )
            # Validate connectivity immediately. If Keras rejects the symbolic
            # connection, Grad-CAM is disabled but the prediction model remains live.
            _ = self.grad_model(dummy, training=False)
            print(f"[CropGuard] Grad-CAM enabled using saved layer: {self.grad_layer_name}", flush=True)
        except Exception as exc:
            self.grad_model = None
            self.grad_layer_name = None
            logger.warning("Grad-CAM disabled; real prediction model remains available: %s", exc, exc_info=True)
            print(f"[CropGuard] WARNING: Grad-CAM unavailable; prediction model remains usable: {exc}", flush=True)

        del dummy
        gc.collect()
        print(f"[CropGuard] REAL TRAINED MODEL LOADED successfully: {model_path}", flush=True)

    def _find_mobilenet_base(self):
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.Model) and "mobilenet" in layer.name.lower():
                return layer
        for layer in self.model.layers:
            if "mobilenet" in layer.name.lower() and hasattr(layer, "layers"):
                return layer
        raise RuntimeError("Trained MobileNetV2 backbone was not found in the checkpoint.")

    @staticmethod
    def _find_target_conv_layer(base_model):
        # Pick the final convolutional feature layer by its actual Keras name.
        # MobileNetV2 has BatchNorm/Activation layers after some convolutions, so
        # the last Conv2D is a stable feature-map target without relying on index.
        conv_layers = [layer for layer in base_model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        if not conv_layers:
            raise RuntimeError("No Conv2D target layer was found in the MobileNetV2 backbone.")
        return conv_layers[-1]

    @staticmethod
    def _preprocess(raw: bytes):
        try:
            with Image.open(io.BytesIO(raw)) as decoded:
                decoded.verify()
            with Image.open(io.BytesIO(raw)) as decoded:
                original = ImageOps.exif_transpose(decoded).convert("RGB")
        except Exception as exc:
            raise ValueError("Upload is not a valid decodable image.") from exc
        original.thumbnail(MAX_VISUAL_SIZE, Image.Resampling.LANCZOS)
        resized = original.resize(IMG_SIZE, Image.Resampling.BILINEAR)
        x = np.asarray(resized, dtype=np.float32)
        return original, tf.convert_to_tensor(x[None, ...], dtype=tf.float32)

    def predict(self, raw: bytes) -> dict:
        with _INFERENCE_LOCK:
            try:
                original, x = self._preprocess(raw)
                with tf.device("/CPU:0"):
                    if self.grad_model is not None:
                        # The saved model already contains MobileNetV2
                        # preprocess_input, so feed the original 0..255 tensor.
                        with tf.GradientTape() as tape:
                            feature_maps, predictions = self.grad_model(x, training=False)
                            idx_tensor = tf.argmax(predictions[0], axis=-1)
                            target = predictions[:, idx_tensor]
                        gradients = tape.gradient(target, feature_maps)
                    else:
                        predictions = self.model(x, training=False)
                        idx_tensor = tf.argmax(predictions[0], axis=-1)
                        feature_maps = None
                        gradients = None

                idx = int(idx_tensor.numpy())
                vector = predictions[0].numpy()
                top_indices = np.argsort(vector)[::-1][: min(3, len(self.labels))]

                result = {
                    "label": self.labels[idx],
                    "crop": self._split_label(self.labels[idx])[0],
                    "disease": self._split_label(self.labels[idx])[1],
                    "class_index": idx,
                    "confidence": float(predictions[0, idx].numpy()),
                    "top_predictions": [{"label": self.labels[int(i)], "probability": float(vector[int(i)])} for i in top_indices],
                    "image_sha256": hashlib.sha256(raw).hexdigest(),
                }

                # Grad-CAM is an enhancement, not a prerequisite for genuine
                # classification. Return a valid prediction if the optional view
                # or gradients are unavailable.
                if self.grad_model is not None and gradients is not None:
                    pooled_gradients = tf.reduce_mean(gradients, axis=(1, 2))
                    cam = tf.reduce_sum(feature_maps * pooled_gradients[:, None, None, :], axis=-1)[0]
                    cam = tf.maximum(cam, 0.0)
                    max_value = tf.reduce_max(cam)
                    cam = cam / (max_value + tf.keras.backend.epsilon())
                    cam_np = cam.numpy().astype(np.float32)

                    heat = Image.fromarray(np.uint8(cam_np * 255), mode="L").resize(original.size, Image.Resampling.BILINEAR)
                    heat_np = np.asarray(heat, dtype=np.float32) / 255.0
                    rgb = np.asarray(original).astype(np.float32) / 255.0
                    mx = rgb.max(axis=2)
                    mn = rgb.min(axis=2)
                    sat = (mx - mn) / (mx + 1e-6)
                    green = (rgb[..., 1] > rgb[..., 0] * 0.72) & (rgb[..., 1] > rgb[..., 2] * 0.72)
                    leaf_mask = (green | (sat > 0.18)) & (mx < 0.97)
                    if leaf_mask.mean() < 0.01:
                        leaf_mask = mx < 0.97
                    if leaf_mask.mean() < 0.01:
                        leaf_mask = np.ones(leaf_mask.shape, dtype=bool)
                    leaf_values = heat_np[leaf_mask]
                    result["severity_score"] = float(np.mean(leaf_values) * 100.0)
                    result["heatmap_coverage_percent"] = float(np.mean(leaf_values >= 0.50) * 100.0)
                    result["heatmap_png"] = self._overlay(original, cam_np)
                    result["gradcam_available"] = True
                else:
                    result["gradcam_available"] = False
                    result["heatmap_png"] = None

                return result
            finally:
                gc.collect()

    @staticmethod
    def _split_label(label: str):
        parts = label.split("___", 1)
        if len(parts) == 2:
            return parts[0], parts[1].replace("_", " ")
        return label.split("_", 1)[0], label

    @staticmethod
    def _overlay(image: Image.Image, cam: np.ndarray) -> bytes:
        heat = Image.fromarray(np.uint8(cam * 255), mode="L").resize(image.size, Image.Resampling.BILINEAR)
        arr = np.asarray(heat, dtype=np.float32) / 255.0
        r = np.clip(1.5 - np.abs(4.0 * arr - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * arr - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * arr - 1.0), 0.0, 1.0)
        heat_rgb = np.stack([r, g, b], axis=-1)
        base = np.asarray(image, dtype=np.float32) / 255.0
        overlay = np.clip(base * 0.58 + heat_rgb * 0.42, 0.0, 1.0)
        out = Image.fromarray(np.uint8(overlay * 255.0), mode="RGB")
        buf = io.BytesIO()
        out.save(buf, format="PNG", optimize=True)
        return buf.getvalue()
