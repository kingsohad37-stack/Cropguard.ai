"""Real inference and Grad-CAM for the trained PlantVillage MobileNetV2 model."""
from __future__ import annotations
import io
import json
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf

IMG_SIZE = (224, 224)


class InferenceEngine:
    def __init__(self, model_path="models/plantvillage_best.h5", labels_path="models/labels.json"):
        model_path = Path(model_path)
        labels_path = Path(labels_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"REAL MODEL MISSING: {model_path}. Run the real data pipeline and training first."
            )
        if not labels_path.exists():
            raise FileNotFoundError(f"LABEL MAP MISSING: {labels_path}")

        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.labels = json.loads(labels_path.read_text())
        if self.model.output_shape[-1] != len(self.labels):
            raise RuntimeError(
                f"Model has {self.model.output_shape[-1]} outputs but labels.json has {len(self.labels)} labels."
            )
        self.grad_layer = self._find_last_conv_like_layer()
        # Keras 3 can deserialize a nested MobileNet backbone whose stored
        # ``layer.output`` belongs to an old internal graph. Rebuild the
        # inference-only tail around the same loaded layers instead of using
        # that stale symbolic output. Inputs are already MobileNet-preprocessed
        # by ``_preprocess``; augmentation is an identity during inference.
        input_tensor = tf.keras.Input(shape=(*IMG_SIZE, 3), name="gradcam_input")
        features = self.grad_layer(input_tensor, training=False)
        output = features
        passed_backbone = False
        for layer in self.model.layers:
            if layer is self.grad_layer:
                passed_backbone = True
                continue
            if passed_backbone:
                output = layer(output, training=False)
        self.grad_model = tf.keras.Model(input_tensor, [features, output], name="cropguard_gradcam")

    def _find_last_conv_like_layer(self):
        # Search nested model layers from the top level. The trained MobileNetV2
        # backbone is the last 4-D feature-producing layer in this architecture.
        for layer in reversed(self.model.layers):
            try:
                shape = layer.output.shape
                if len(shape) == 4:
                    return layer
            except (AttributeError, TypeError):
                continue
        raise RuntimeError("No 4-D convolutional feature map exists for Grad-CAM.")

    @staticmethod
    def _preprocess(raw: bytes):
        try:
            with Image.open(io.BytesIO(raw)) as decoded:
                decoded.verify()
            original = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:
            raise ValueError("Upload is not a valid decodable image.") from exc
        resized = original.resize(IMG_SIZE, Image.Resampling.BILINEAR)
        x = np.asarray(resized, dtype=np.float32)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        return original, tf.convert_to_tensor(x[None, ...], dtype=tf.float32)

    @staticmethod
    def _leaf_mask(image: Image.Image) -> np.ndarray:
        """Estimate leaf pixels from the uploaded image itself.

        This is intentionally an image-derived mask, not a stored mask or score.
        It is conservative: green vegetation is preferred, while sufficiently
        saturated non-background pixels are retained for yellow/brown lesions.
        """
        rgb = np.asarray(image).astype(np.float32) / 255.0
        mx = rgb.max(axis=2)
        mn = rgb.min(axis=2)
        sat = (mx - mn) / (mx + 1e-6)
        green = (rgb[..., 1] > rgb[..., 0] * 0.72) & (rgb[..., 1] > rgb[..., 2] * 0.72)
        nonwhite = mx < 0.97
        mask = (green | (sat > 0.18)) & nonwhite
        # Reject tiny speckles while keeping the computation deterministic.
        if mask.mean() < 0.01:
            # A neutral fallback means "foreground unavailable" rather than
            # fabricating a disease-specific area. The severity remains derived
            # entirely from the actual Grad-CAM values.
            mask = nonwhite
        if mask.mean() < 0.01:
            mask = np.ones(mask.shape, dtype=bool)
        return mask

    def predict(self, raw: bytes) -> dict:
        original, x = self._preprocess(raw)

        # The class and heatmap below are produced by this exact uploaded tensor.
        with tf.GradientTape() as tape:
            conv_features, predictions = self.grad_model(x, training=False)
            class_index = tf.argmax(predictions[0], axis=-1)
            class_score = predictions[:, class_index]

        gradients = tape.gradient(class_score, conv_features)
        if gradients is None:
            raise RuntimeError("Gradient computation returned None; cannot create Grad-CAM.")

        weights = tf.reduce_mean(gradients, axis=(1, 2))
        cam = tf.reduce_sum(conv_features * weights[:, None, None, :], axis=-1)[0]
        cam = tf.maximum(cam, 0.0)
        cam = cam / (tf.reduce_max(cam) + tf.keras.backend.epsilon())
        cam_np = cam.numpy().astype(np.float32)

        heat = Image.fromarray(np.uint8(cam_np * 255)).resize(
            original.size, Image.Resampling.BILINEAR
        )
        heat_np = np.asarray(heat, dtype=np.float32) / 255.0
        leaf_mask = self._leaf_mask(original)
        leaf_values = heat_np[leaf_mask]
        severity = float(np.mean(leaf_values) * 100.0)
        coverage = float(np.mean(leaf_values >= 0.50) * 100.0)

        idx = int(class_index.numpy())
        label = self.labels[idx]
        crop, disease = self._split_label(label)
        vector = predictions[0].numpy()
        top_indices = np.argsort(vector)[::-1][: min(3, len(self.labels))]

        return {
            "label": label,
            "crop": crop,
            "disease": disease,
            "class_index": idx,
            "confidence": float(predictions[0, idx].numpy()),
            "top_predictions": [
                {"label": self.labels[int(i)], "probability": float(vector[int(i)])}
                for i in top_indices
            ],
            "image_sha256": hashlib.sha256(raw).hexdigest(),
            "severity_score": severity,
            "heatmap_coverage_percent": coverage,
            "heatmap_png": self._overlay(original, cam_np),
        }

    @staticmethod
    def _split_label(label: str):
        parts = label.split("___", 1)
        if len(parts) == 2:
            return parts[0], parts[1].replace("_", " ")
        return label.split("_", 1)[0], label

    @staticmethod
    def _overlay(image: Image.Image, cam: np.ndarray) -> bytes:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import io

        fig = plt.figure(figsize=(7, 7), frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(image)
        ax.imshow(cam, cmap="jet", alpha=0.42, vmin=0, vmax=1)
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return buf.getvalue()
