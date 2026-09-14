"""
dataset.py — load a sample CSV and write stratified train/val/test splits.

Expected columns: path, group, rating_raw
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = REPO_ROOT / "artifacts" / "train_sample.csv"
DEFAULT_SPLIT_DIR = REPO_ROOT / "artifacts" / "splits"
REQUIRED_COLUMNS = {"path", "group", "rating_raw"}


def load_dataset(csv_path: Path | str = DEFAULT_CSV) -> pd.DataFrame:
    """Read a sample CSV and check required columns plus unique paths."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    df = df.copy()
    df["group"] = df["group"].astype(str)
    df["rating_raw"] = df["rating_raw"].astype(float)
    if df["path"].duplicated().any():
        n = int(df["path"].duplicated().sum())
        raise ValueError(f"dataset has {n} duplicate paths")
    return df


def assert_disjoint_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Raise if any image path appears in more than one split."""
    train_p, val_p, test_p = (
        set(train_df["path"]),
        set(val_df["path"]),
        set(test_df["path"]),
    )
    if train_p & val_p or train_p & test_p or val_p & test_p:
        raise RuntimeError("overlapping paths between train/val/test")


def split_dataset(
    df: pd.DataFrame,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test split by Haddock `group`. Fractions must sum to 1."""
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1.0")
    if min(train_frac, val_frac, test_frac) <= 0:
        raise ValueError("train/val/test fractions must all be > 0")

    holdout_frac = val_frac + test_frac
    train_df, holdout_df = train_test_split(
        df, test_size=holdout_frac, random_state=seed, stratify=df["group"]
    )
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=test_frac / holdout_frac,
        random_state=seed,
        stratify=holdout_df["group"],
    )
    train_df = train_df.reset_index(drop=True).assign(split="train")
    val_df = val_df.reset_index(drop=True).assign(split="val")
    test_df = test_df.reset_index(drop=True).assign(split="test")
    assert_disjoint_splits(train_df, val_df, test_df)
    return train_df, val_df, test_df


def split_paths_for_stem(out_dir: Path | str, stem: str) -> dict[str, Path]:
    """Return `{train,val,test}` CSV paths for a stem under `out_dir`."""
    out_dir = Path(out_dir)
    return {
        "train": out_dir / f"{stem}_train.csv",
        "val": out_dir / f"{stem}_val.csv",
        "test": out_dir / f"{stem}_test.csv",
    }


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path | str,
    stem: str,
) -> dict[str, Path]:
    """Write `{stem}_train.csv`, `{stem}_val.csv`, `{stem}_test.csv`."""
    assert_disjoint_splits(train_df, val_df, test_df)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = split_paths_for_stem(out_dir, stem)
    train_df.to_csv(paths["train"], index=False)
    val_df.to_csv(paths["val"], index=False)
    test_df.to_csv(paths["test"], index=False)
    return paths
