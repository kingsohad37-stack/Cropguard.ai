"""Evaluate a real trained CropGuard model against the held-out test split."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
IMG_SIZE = (224, 224)


def make_dataset(frame: pd.DataFrame, labels: list[str], batch_size: int = 32):
    index = {label: i for i, label in enumerate(labels)}
    ds = tf.data.Dataset.from_tensor_slices(
        (frame.path.to_numpy(), frame.label.map(index).to_numpy(dtype=np.int32))
    )

    def load(path, target):
        image = tf.io.decode_image(
            tf.io.read_file(path), channels=3, expand_animations=False
        )
        image.set_shape([None, None, 3])
        return tf.image.resize(tf.cast(image, tf.float32), IMG_SIZE), target

    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main():
    model_path = ROOT / "models/plantvillage_best.keras"
    labels_path = ROOT / "models/labels.json"
    split_path = ROOT / "data/splits/test.csv"
    if not all(path.exists() for path in (model_path, labels_path, split_path)):
        raise FileNotFoundError("Real model, labels, and held-out test split are required before evaluation.")

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(split_path)
    ds = make_dataset(frame, labels)

    # The training checkpoint contains a custom Keras-3 loss. Loading it with
    # compile=True requires deserializing that training-only loss and was the
    # reason the previous evaluation stage failed. Evaluation only needs the
    # saved inference graph, so load without the training configuration.
    model = tf.keras.models.load_model(model_path, compile=False)
    probabilities = model.predict(ds, verbose=1)

    actual = frame.label.map({name: i for i, name in enumerate(labels)}).to_numpy(dtype=np.int32)
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)

    accuracy = float(np.mean(predicted == actual))
    eps = np.finfo(np.float32).eps
    test_loss = float(
        tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(actual, np.clip(probabilities, eps, 1.0))
        ).numpy()
    )

    report = classification_report(
        actual,
        predicted,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(
        [{"label": name, **report[name]} for name in labels]
    ).to_csv(ROOT / "models/per_class_metrics.csv", index=False)

    matrix = confusion_matrix(actual, predicted, labels=list(range(len(labels))))
    np.save(ROOT / "models/confusion_matrix.npy", matrix)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(matrix, cmap="Blues")
    ax.set(xlabel="Predicted", ylabel="Actual", title="PlantVillage confusion matrix")
    fig.tight_layout()
    fig.savefig(ROOT / "models/confusion_matrix.png", dpi=160)
    plt.close(fig)

    result = {
        "test_loss": test_loss,
        "test_accuracy": accuracy,
        "precision_weighted": float(report["weighted avg"]["precision"]),
        "recall_weighted": float(report["weighted avg"]["recall"]),
        "f1_weighted": float(report["weighted avg"]["f1-score"]),
        "mean_confidence": float(np.mean(confidence)),
        "test_examples": len(frame),
        "labels": labels,
        "evaluation_method": "compile-free inference against held-out PlantVillage test split",
    }
    (ROOT / "models/evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
