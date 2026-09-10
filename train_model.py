"""
train_model.py — Determinator 2.0 training entry point

Default: train tuned XGBoost on the locked top-15 feature subset
(see modeling.SELECTED_FEATURES / build_models), evaluate on VAL,
save model + comparison JSON.

Also supports dataset split utilities:
  --make-splits / --validate-splits
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
    spot_check_paths,
    summarize,
    validate_splits,
)
from modeling import (
    DEFAULT_FEATURES,
    DEFAULT_MODEL_DIR,
    SELECTED_FEATURES,
    train_and_evaluate,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Determinator focus model")
    p.add_argument(
        "--make-splits",
        action="store_true",
        help="Build train/val/test CSVs from sample CSV (no model training)",
    )
    p.add_argument(
        "--validate-splits",
        action="store_true",
        help="Validate existing train/val/test CSVs (no model training)",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Training sample CSV (path, group, rating_raw)",
    )
    p.add_argument("--train-frac", type=float, default=0.80)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    p.add_argument("--check-paths", action="store_true")
    p.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES,
        help="Wide features NPZ",
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Where to save trained models",
    )
    p.add_argument(
        "--include-test",
        action="store_true",
        help="Also score the held-out TEST split (use sparingly)",
    )
    p.add_argument(
        "--balanced-class-weight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use inverse-frequency balanced weights (default: on for imbalanced five-bin data)",
    )
    p.add_argument(
        "--model-name",
        default=None,
        help="Output stem (default: xgboost_5bin or xgboost_identity9 from class count)",
    )
    return p.parse_args()


def run_make_splits(args: argparse.Namespace) -> None:
    csv_path = args.csv.resolve()
    stem = csv_path.stem
    split_dir = args.split_dir.resolve()

    print("train_model — split dataset")
    print(f"  csv: {csv_path}")
    print(
        f"  fractions: train={args.train_frac}  "
        f"val={args.val_frac}  test={args.test_frac}"
    )

    df = load_dataset(csv_path)
    summarize(df, title="FULL SAMPLE")
    if args.check_paths:
        spot_check_paths(df)

    train_df, val_df, test_df = split_dataset(
        df,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    summarize(train_df, title="TRAIN")
    summarize(val_df, title="VAL")
    summarize(test_df, title="TEST")

    paths = save_splits(train_df, val_df, test_df, out_dir=split_dir, stem=stem)
    for role, path in paths.items():
        print(f"  {role:5} -> {path.resolve()}")
    validate_splits(split_dir, stem, check_files_exist=False)


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()
    stem = csv_path.stem
    split_dir = args.split_dir.resolve()

    if args.validate_splits:
        print("train_model — validate splits")
        validate_splits(split_dir, stem, check_files_exist=args.check_paths)
        return

    if args.make_splits:
        run_make_splits(args)
        return

    print("train_model — fit tuned XGBoost (top-15 features)")
    print(f"  features: {args.features.resolve()}")
    print(f"  model-dir: {args.model_dir.resolve()}")
    print(f"  selected features ({len(SELECTED_FEATURES)}):")
    for i, name in enumerate(SELECTED_FEATURES, 1):
        print(f"    {i:2d}. {name}")
    train_and_evaluate(
        features_npz=args.features,
        out_dir=args.model_dir,
        seed=args.seed,
        include_test=args.include_test,
        use_balanced_class_weight=args.balanced_class_weight,
        selected_features=SELECTED_FEATURES,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
