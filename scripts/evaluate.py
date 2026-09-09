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
    ds = tf.data.Dataset.from_tensor_slices((frame.path.to_numpy(), frame.label.map(index).to_numpy()))
    def load(path, target):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        return tf.image.resize(tf.cast(image, tf.float32), IMG_SIZE), target
    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def main():
    model_path, labels_path, split_path = ROOT / 'models/plantvillage_best.keras', ROOT / 'models/labels.json', ROOT / 'data/splits/test.csv'
    if not all(path.exists() for path in (model_path, labels_path, split_path)):
        raise FileNotFoundError('Real model, labels, and held-out test split are required before evaluation.')
    labels = json.loads(labels_path.read_text(encoding='utf-8'))
    frame = pd.read_csv(split_path)
    model = tf.keras.models.load_model(model_path, compile=True)
    loss, accuracy = model.evaluate(make_dataset(frame, labels), verbose=1)
    probabilities = model.predict(make_dataset(frame, labels), verbose=1)
    actual = frame.label.map({name: i for i, name in enumerate(labels)}).to_numpy()
    predicted = probabilities.argmax(axis=1)
    report = classification_report(actual, predicted, labels=list(range(len(labels))), target_names=labels, output_dict=True, zero_division=0)
    pd.DataFrame([{'label': name, **report[name]} for name in labels]).to_csv(ROOT / 'models/per_class_metrics.csv', index=False)
    matrix = confusion_matrix(actual, predicted, labels=list(range(len(labels))))
    np.save(ROOT / 'models/confusion_matrix.npy', matrix)
    fig, ax = plt.subplots(figsize=(12, 10)); ax.imshow(matrix, cmap='Blues'); ax.set(xlabel='Predicted', ylabel='Actual', title='PlantVillage confusion matrix'); fig.tight_layout(); fig.savefig(ROOT / 'models/confusion_matrix.png', dpi=160); plt.close(fig)
    result = {'test_loss': float(loss), 'test_accuracy': float(accuracy), 'precision_weighted': report['weighted avg']['precision'], 'recall_weighted': report['weighted avg']['recall'], 'f1_weighted': report['weighted avg']['f1-score'], 'test_examples': len(frame), 'labels': labels}
    (ROOT / 'models/evaluation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
