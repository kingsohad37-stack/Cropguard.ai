"""Advanced, reproducible MobileNetV2 transfer-learning training for CropGuard."""
from __future__ import annotations
import argparse, json, os, random
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMG = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE


def ds_from_csv(csv_path, labels, batch, shuffle, seed, max_per_class=0):
    df = pd.read_csv(csv_path)
    required = {"path", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"{csv_path} must contain columns: {sorted(required)}")
    idx = {x: i for i, x in enumerate(labels)}
    unknown = sorted(set(df.label) - set(labels))
    if unknown:
        raise ValueError(f"Unknown labels in {csv_path}: {unknown[:5]}")
    if max_per_class:
        parts = []
        for label, group in df.groupby("label"):
            parts.append(group.sample(n=min(len(group), max_per_class), random_state=seed))
        df = pd.concat(parts, ignore_index=True)
    if df.empty:
        raise ValueError(f"No samples found in {csv_path}")
    missing = [p for p in df.path if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing image referenced by {csv_path}: {missing[0]}")

    paths = df.path.to_numpy()
    ys = df.label.map(idx).to_numpy(dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, ys))
    if shuffle:
        ds = ds.shuffle(min(len(df), 10000), seed=seed, reshuffle_each_iteration=True)

    def load(path, y):
        b = tf.io.read_file(path)
        x = tf.io.decode_image(b, channels=3, expand_animations=False)
        x.set_shape([None, None, 3])
        x = tf.image.resize(x, IMG, antialias=True)
        return tf.cast(x, tf.float32), y

    return ds.map(load, num_parallel_calls=AUTOTUNE).batch(batch).prefetch(AUTOTUNE), df


def class_weights(frame, labels):
    counts = frame.label.value_counts()
    total = len(frame)
    n = len(labels)
    return {i: float(total / (n * max(1, counts.get(label, 0)))) for i, label in enumerate(labels)}


def check_split_integrity(train_frame, val_frame, test_frame):
    def hashes(frame):
        return set(str(Path(p).resolve()) for p in frame.path)
    tr, va, te = hashes(train_frame), hashes(val_frame), hashes(test_frame)
    overlaps = {"train_val": tr & va, "train_test": tr & te, "val_test": va & te}
    bad = {k: len(v) for k, v in overlaps.items() if v}
    if bad:
        raise RuntimeError(f"DATA LEAKAGE detected across splits: {bad}")


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--fine-tune-epochs', type=int, default=8)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--fine-tune-lr', type=float, default=1e-5)
    ap.add_argument('--label-smoothing', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-per-class', type=int, default=0)
    args = ap.parse_args()

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    labels = json.loads(Path('models/labels.json').read_text(encoding='utf-8'))
    n = len(labels)
    train, train_frame = ds_from_csv('data/splits/train.csv', labels, args.batch_size, True, args.seed, args.max_per_class)
    val, val_frame = ds_from_csv('data/splits/val.csv', labels, args.batch_size, False, args.seed, args.max_per_class)
    test, test_frame = ds_from_csv('data/splits/test.csv', labels, args.batch_size, False, args.seed, args.max_per_class)
    check_split_integrity(train_frame, val_frame, test_frame)
    weights = class_weights(train_frame, labels)

    # Field-oriented augmentation: geometry + illumination/color variation.
    # These layers run only during training; validation/test remain untouched.
    aug = keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.10),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.08, 0.08),
        layers.RandomContrast(0.15),
        layers.RandomBrightness(0.12),
    ], name='augmentation')

    base = keras.applications.MobileNetV2(input_shape=(*IMG, 3), include_top=False, weights='imagenet')
    base.trainable = False
    inp = keras.Input((*IMG, 3))
    x = aug(inp)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    # Keep a direct classifier after GAP so the production CAM implementation
    # can use the learned classifier weights without constructing a new graph.
    out = layers.Dense(n, activation='softmax', name='predictions')(x)
    model = keras.Model(inp, out)

    loss = keras.losses.SparseCategoricalCrossentropy(label_smoothing=args.label_smoothing)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=1e-4, clipnorm=1.0),
        loss=loss,
        metrics=[keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
    )

    Path('models').mkdir(exist_ok=True)
    ckpt = 'models/plantvillage_best.keras'
    cb = [
        keras.callbacks.ModelCheckpoint(ckpt, monitor='val_accuracy', save_best_only=True, mode='max'),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.35, patience=2, min_lr=1e-7, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True),
        keras.callbacks.CSVLogger('models/training_log.csv', append=False),
    ]

    print(f'Train/val/test samples: {len(train_frame)}/{len(val_frame)}/{len(test_frame)}')
    print(f'Classes: {n}; class-balanced weighting: enabled; split leakage check: passed')
    print('Stage 1: advanced head training with field-oriented augmentation')
    model.fit(train, validation_data=val, epochs=args.epochs, callbacks=cb, class_weight=weights)

    print('Stage 2: fine-tuning upper MobileNetV2 layers')
    base.trainable = True
    for layer in base.layers[:-50]:
        layer.trainable = False
    # Keep BatchNorm statistics stable on the deployment-oriented dataset.
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=args.fine_tune_lr, weight_decay=5e-5, clipnorm=1.0),
        loss=loss,
        metrics=[keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
    )
    model.fit(
        train,
        validation_data=val,
        initial_epoch=args.epochs,
        epochs=args.epochs + args.fine_tune_epochs,
        callbacks=cb,
        class_weight=weights,
    )

    best = keras.models.load_model(ckpt)
    test_loss, test_acc = best.evaluate(test, verbose=1)
    metrics = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'best_checkpoint': ckpt,
        'classes': n,
        'train_samples': len(train_frame),
        'val_samples': len(val_frame),
        'test_samples': len(test_frame),
        'epochs_stage1': args.epochs,
        'epochs_finetune': args.fine_tune_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'fine_tune_learning_rate': args.fine_tune_lr,
        'label_smoothing': args.label_smoothing,
        'seed': args.seed,
        'max_per_class': args.max_per_class,
        'class_weighting': True,
        'field_augmentation': ['flip', 'rotation', 'zoom', 'translation', 'contrast', 'brightness'],
        'fine_tuned_layers': 50,
        'evaluated_on': 'data/splits/test.csv',
    }
    Path('models/metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print('REAL TEST METRICS:', json.dumps(metrics, indent=2))

    try:
        best.save('models/plantvillage_best.h5', include_optimizer=False)
        print('Saved compatibility H5 model to models/plantvillage_best.h5')
    except Exception as exc:
        h5_path = Path('models/plantvillage_best.h5')
        if h5_path.exists():
            h5_path.unlink()
        print(f'H5 compatibility export skipped: {exc}')


if __name__ == '__main__':
    main()
