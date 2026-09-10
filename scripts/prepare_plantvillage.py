"""Prepare real PlantVillage images into CropGuard CSV train/val/test splits."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def norm(s: str) -> str:
    return s.replace('\\', '/').lstrip('./')


def resolve_entries(root: Path, entries: list[str]) -> list[tuple[str, str]]:
    files = [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    by_rel = {norm(str(p.relative_to(root))): p for p in files}
    by_name = {}
    for p in files:
        by_name.setdefault(p.name.lower(), []).append(p)

    resolved = []
    missing = []
    for raw in entries:
        key = norm(raw)
        candidates = [key]
        if key.startswith('color/'):
            candidates.append(key[6:])
        found = None
        for candidate in candidates:
            if candidate in by_rel:
                found = by_rel[candidate]
                break
        if found is None:
            base = Path(key).name.lower()
            hits = by_name.get(base, [])
            if len(hits) == 1:
                found = hits[0]
        if found is None:
            missing.append(raw)
            continue
        parts = found.relative_to(root).parts
        label = next((p for p in parts if '___' in p), None)
        if not label:
            missing.append(raw)
            continue
        resolved.append((str(found), label))
    if missing:
        raise FileNotFoundError(f'Could not resolve {len(missing)} dataset entries; first: {missing[0]}')
    return resolved


def read_list(path: Path) -> list[str]:
    return [x.strip() for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--train-list', required=True)
    ap.add_argument('--test-list', required=True)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    train_entries = resolve_entries(root, read_list(Path(args.train_list)))
    test_entries = resolve_entries(root, read_list(Path(args.test_list)))
    train_df = pd.DataFrame(train_entries, columns=['path', 'label'])
    test_df = pd.DataFrame(test_entries, columns=['path', 'label'])

    # Hold out 10% of the official training partition for validation.
    train_parts, val_parts = [], []
    for label, group in train_df.groupby('label'):
        if len(group) < 2:
            train_parts.append(group)
            continue
        tr, va = train_test_split(group, test_size=0.10, random_state=42, shuffle=True)
        train_parts.append(tr)
        val_parts.append(va)
    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = pd.concat(val_parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

    labels = json.loads(Path('models/labels.json').read_text(encoding='utf-8'))
    found_labels = set(train_df.label) | set(val_df.label) | set(test_df.label)
    if found_labels != set(labels):
        raise RuntimeError(f'Label mismatch. Dataset has {len(found_labels)} classes, labels.json has {len(labels)}.')

    out = Path('data/splits')
    out.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out / 'train.csv', index=False)
    val_df.to_csv(out / 'val.csv', index=False)
    test_df.to_csv(out / 'test.csv', index=False)
    print(f'Prepared real PlantVillage splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}, classes={len(labels)}')


if __name__ == '__main__':
    main()
