"""Download/prepare PlantVillage and create a leakage-resistant stratified split."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

EXTS={".jpg", ".jpeg", ".png", ".webp"}

def maybe_download(out: Path, dataset: str, kaggle_config_dir: str | None):
    out.mkdir(parents=True, exist_ok=True)
    if any(p.suffix.lower() in EXTS for p in out.rglob('*') if p.is_file()):
        return
    # Use the virtual environment's CLI executable explicitly. The Kaggle
    # package version used here has no ``python -m kaggle`` entry point.
    kaggle_executable = Path(sys.executable).with_name("kaggle.exe") if os.name == "nt" else "kaggle"
    cmd=[str(kaggle_executable), "datasets", "download", "-d",dataset,"-p",str(out),"--unzip"]
    print("Downloading real PlantVillage dataset:", " ".join(cmd))
    environment = os.environ.copy()
    if kaggle_config_dir:
        environment["KAGGLE_CONFIG_DIR"] = kaggle_config_dir
    subprocess.run(cmd, check=True, env=environment)

def discover(root: Path):
    rows=[]
    candidates = 0
    # A content digest stops exact duplicate files entering separate splits.
    # The directory immediately containing each image is the discovered label.
    seen = set()
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS and p.parent.name:
            candidates += 1
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            if digest not in seen:
                seen.add(digest)
                rows.append((str(p.resolve()), p.parent.name, digest))
    if not rows:
        raise RuntimeError(f"No image files found under {root}")
    return pd.DataFrame(rows, columns=["path","label","sha256"]), candidates - len(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--dataset", default="mohitsingh1804/plantvillage")
    ap.add_argument("--kaggle-config-dir", default=os.environ.get("KAGGLE_CONFIG_DIR"), help="Directory containing kaggle.json; never commit it.")
    ap.add_argument("--seed", type=int, default=42)
    args=ap.parse_args()
    root=Path(args.data_dir)
    maybe_download(root,args.dataset,args.kaggle_config_dir)
    df, duplicates_removed=discover(root)
    # Remove the optional background class if present: it is not a crop/disease class.
    df=df[df.label!="Background_without_leaves"].reset_index(drop=True)
    counts=df.label.value_counts()
    if len(counts)!=38:
        print(f"WARNING: discovered {len(counts)} classes, not 38. Proceeding with the actual dataset found.")
    if (counts < 3).any():
        raise RuntimeError("Every discovered class needs at least 3 unique images for a stratified split.")
    train_val,test=train_test_split(df,test_size=0.10,stratify=df.label,random_state=args.seed)
    train,val=train_test_split(train_val,test_size=0.1111111111,stratify=train_val.label,random_state=args.seed)
    out=Path("data/splits"); out.mkdir(parents=True,exist_ok=True)
    assert not (set(train.sha256) & set(val.sha256) or set(train.sha256) & set(test.sha256) or set(val.sha256) & set(test.sha256))
    train.to_csv(out/"train.csv",index=False); val.to_csv(out/"val.csv",index=False); test.to_csv(out/"test.csv",index=False)
    labels=sorted(df.label.unique())
    (Path("models")).mkdir(exist_ok=True)
    (Path("models") / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    summary={"images":len(df),"classes":len(labels),"images_per_class":counts.to_dict(),"train_count":len(train),"validation_count":len(val),"test_count":len(test),"seed":args.seed,"duplicates_removed":duplicates_removed}
    (out / "summary.json").write_text(json.dumps(summary,indent=2), encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
