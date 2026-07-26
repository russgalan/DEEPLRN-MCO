"""
Build stratified 70/15/15 train/val/test splits from a TrashNet folder tree.

Expected input layout:
    raw_dir/
        cardboard/*.jpg
        glass/*.jpg
        metal/*.jpg
        paper/*.jpg
        plastic/*.jpg
        trash/*.jpg

Writes train.csv / val.csv / test.csv (columns: filepath,label) into out_dir.
This is run ONCE. The test split is not touched again until src/evaluate.py,
so validation-set decisions (hyperparameter tuning, ablation) never leak
into the reported test performance -- per Section 4.1 of the paper.
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import CLASS_NAMES, set_seed

VALID_EXT = {".jpg", ".jpeg", ".png"}


def collect_files(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for cls in CLASS_NAMES:
        cls_dir = raw_dir / cls
        if not cls_dir.exists():
            raise FileNotFoundError(
                f"Expected class folder '{cls_dir}' not found. "
                f"Check that --raw_dir points at the unzipped TrashNet root."
            )
        for p in cls_dir.iterdir():
            if p.suffix.lower() in VALID_EXT:
                rows.append({"filepath": str(p.resolve()), "label": cls})
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No images found -- check --raw_dir path and folder names.")
    return df


def make_splits(df: pd.DataFrame, seed: int):
    # First peel off 70% train, 30% temp; then split temp 50/50 -> 15%/15%.
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=seed
    )
    return train_df, val_df, test_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="Path to unzipped TrashNet root")
    ap.add_argument("--out_dir", default="data/splits")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    df = collect_files(Path(args.raw_dir))

    print("Class distribution (full dataset):")
    print(df["label"].value_counts(), "\n")

    train_df, val_df, test_df = make_splits(df, args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"{name}: {len(split)} images")
        print(split["label"].value_counts(normalize=True).round(3), "\n")

    print(f"Splits written to {out_dir}/. The test.csv split should not be "
          f"opened again until final evaluation.")


if __name__ == "__main__":
    main()
