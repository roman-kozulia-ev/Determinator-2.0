"""
train_model.py — train / evaluate the shipped 9-class XGBoost model

Writes artifacts/models/xgboost_9_class_13_feat.joblib
and C++ export JSON under artifacts/models/export/.

Labels are the nine raw Haddock scores (1.0 … 5.0 step 0.5).
Features come from a local NPZ (not shipped): artifacts/features/features.npz

Examples
  python train_model.py --include-test
  python train_model.py --make-splits --csv artifacts/train_sample.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dataset import (
    DEFAULT_CSV,
    DEFAULT_SPLIT_DIR,
    load_dataset,
    save_splits,
    split_dataset,
)
from modeling import (
    DEFAULT_FEATURES,
    DEFAULT_MODEL_DIR,
    SELECTED_FEATURES,
    SHIPPED_MODEL_NAME,
    train_and_evaluate,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Determinator 2.0 XGBoost (9 Haddock classes, 13 features)",
    )
    p.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES,
        help="Feature NPZ with train/val/test splits",
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Where to save the joblib and C++ export JSON",
    )
    p.add_argument(
        "--model-name",
        default=SHIPPED_MODEL_NAME,
        help=f"Output stem (default: {SHIPPED_MODEL_NAME})",
    )
    p.add_argument(
        "--include-test",
        action="store_true",
        help="Also score the held-out TEST split",
    )
    p.add_argument(
        "--balanced-class-weight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Inverse-frequency sample weights (default: on)",
    )
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--make-splits",
        action="store_true",
        help="Build train/val/test CSVs from a sample CSV (no training)",
    )
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--train-frac", type=float, default=0.80)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    return p.parse_args()


def run_make_splits(args: argparse.Namespace) -> None:
    csv_path = args.csv.resolve()
    split_dir = args.split_dir.resolve()
    print("train_model — split dataset")
    print(f"  csv: {csv_path}")
    train_df, val_df, test_df = split_dataset(
        load_dataset(csv_path),
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    paths = save_splits(train_df, val_df, test_df, out_dir=split_dir, stem=csv_path.stem)
    for role, path in paths.items():
        print(f"  {role:5} -> {path.resolve()}")


def main() -> None:
    args = parse_args()

    if args.make_splits:
        run_make_splits(args)
        return

    features = args.features.resolve()
    model_name = args.model_name

    print("train_model — fit tuned XGBoost (SELECTED_FEATURES)")
    print(f"  model-name: {model_name}")
    print(f"  features:   {features}")
    print(f"  model-dir:  {args.model_dir.resolve()}")
    print(f"  selected features ({len(SELECTED_FEATURES)}):")
    for i, name in enumerate(SELECTED_FEATURES, 1):
        print(f"    {i:2d}. {name}")

    train_and_evaluate(
        features_npz=features,
        out_dir=args.model_dir,
        seed=args.seed,
        include_test=args.include_test,
        use_balanced_class_weight=args.balanced_class_weight,
        model_name=model_name,
    )


if __name__ == "__main__":
    main()
