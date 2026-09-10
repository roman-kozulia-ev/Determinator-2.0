"""
dataset.py — load, validate, and split training sample CSVs.

Expected columns: path, group, rating_raw

Splits are written to separate files (train / val / test) and checked for
zero path overlap so sets cannot get mixed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = REPO_ROOT / "artifacts" / "train_sample_five.csv"
DEFAULT_SPLIT_DIR = REPO_ROOT / "artifacts" / "splits"

REQUIRED_COLUMNS = {"path", "group", "rating_raw"}


def load_dataset(csv_path: Path | str = DEFAULT_CSV) -> pd.DataFrame:
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
        raise ValueError(f"dataset has {n} duplicate paths; refuse to split")

    return df


def summarize(df: pd.DataFrame, title: str | None = None) -> None:
    if title:
        print(f"\n=== {title} ===")
    print(f"rows: {len(df)}")
    print(f"unique paths: {df['path'].nunique()}")
    print("counts by group:")
    counts = df["group"].value_counts()

    def sort_key(g: str):
        try:
            return (0, float(g))
        except ValueError:
            return (1, g)

    for group in sorted(counts.index, key=sort_key):
        print(f"  {group:>8}: {counts[group]}")


def spot_check_paths(df: pd.DataFrame, n: int = 3, seed: int = 42) -> None:
    """Confirm a few crop files exist on disk."""
    sample = df.sample(n=min(n, len(df)), random_state=seed)
    print(f"\nspot-check {len(sample)} paths:")
    for path in sample["path"]:
        ok = Path(path).is_file()
        print(f"  [{'OK' if ok else 'MISSING'}] {path}")
        if not ok:
            raise FileNotFoundError(path)


def assert_disjoint_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Hard guarantee: no shared image paths across splits."""
    train_p = set(train_df["path"])
    val_p = set(val_df["path"])
    test_p = set(test_df["path"])

    overlap_tv = train_p & val_p
    overlap_tt = train_p & test_p
    overlap_vt = val_p & test_p

    if overlap_tv or overlap_tt or overlap_vt:
        raise RuntimeError(
            "SPLIT LEAKAGE: overlapping paths between splits "
            f"(train∩val={len(overlap_tv)}, train∩test={len(overlap_tt)}, "
            f"val∩test={len(overlap_vt)})"
        )

    total = len(train_df) + len(val_df) + len(test_df)
    unique = len(train_p | val_p | test_p)
    if total != unique:
        raise RuntimeError(
            f"SPLIT ERROR: row count {total} != unique paths {unique}"
        )


def split_dataset(
    df: pd.DataFrame,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split by `group` into train / val / test.

    Fractions must sum to 1.0.
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-9:
        raise ValueError(
            f"fractions must sum to 1.0, got "
            f"{train_frac + val_frac + test_frac}"
        )
    if min(train_frac, val_frac, test_frac) <= 0:
        raise ValueError("train/val/test fractions must all be > 0")

    # First cut: train vs (val+test)
    holdout_frac = val_frac + test_frac
    train_df, holdout_df = train_test_split(
        df,
        test_size=holdout_frac,
        random_state=seed,
        stratify=df["group"],
    )

    # Second cut: val vs test from holdout
    test_in_holdout = test_frac / holdout_frac
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=test_in_holdout,
        random_state=seed,
        stratify=holdout_df["group"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    assert_disjoint_splits(train_df, val_df, test_df)

    # Tag rows so a mistaken concat is still identifiable
    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")

    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path | str,
    stem: str,
) -> dict[str, Path]:
    """
    Write three separate CSVs. Filenames include the split role so they
    cannot be confused (e.g. ..._train.csv vs ..._test.csv).
    """
    assert_disjoint_splits(train_df, val_df, test_df)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": out_dir / f"{stem}_train.csv",
        "val": out_dir / f"{stem}_val.csv",
        "test": out_dir / f"{stem}_test.csv",
    }
    train_df.to_csv(paths["train"], index=False)
    val_df.to_csv(paths["val"], index=False)
    test_df.to_csv(paths["test"], index=False)
    return paths


def split_paths_for_stem(out_dir: Path | str, stem: str) -> dict[str, Path]:
    out_dir = Path(out_dir)
    return {
        "train": out_dir / f"{stem}_train.csv",
        "val": out_dir / f"{stem}_val.csv",
        "test": out_dir / f"{stem}_test.csv",
    }


def validate_splits(
    out_dir: Path | str,
    stem: str,
    check_files_exist: bool = False,
    spot_n: int = 2,
) -> dict[str, int]:
    """
    Reload saved train/val/test CSVs and verify they are safe to use.

    Checks:
      - all three files exist
      - required columns present
      - each file's `split` column matches its role
      - zero overlapping paths across splits
      - optional: spot-check image files on disk
    """
    paths = split_paths_for_stem(out_dir, stem)
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "split files missing:\n  " + "\n  ".join(missing)
        )

    frames = {role: load_dataset(path) for role, path in paths.items()}

    for role, df in frames.items():
        if "split" not in df.columns:
            raise ValueError(f"{paths[role]} missing 'split' column")
        bad = df.loc[df["split"] != role]
        if len(bad):
            raise ValueError(
                f"{paths[role]} has {len(bad)} rows with split != {role!r}"
            )

    assert_disjoint_splits(frames["train"], frames["val"], frames["test"])

    if check_files_exist:
        for role, df in frames.items():
            print(f"\npath spot-check [{role}]")
            spot_check_paths(df, n=spot_n)

    counts = {role: len(df) for role, df in frames.items()}
    print("\nVALIDATE SPLITS — PASSED")
    print(f"  stem: {stem}")
    for role, n in counts.items():
        print(f"  {role:5}: {n:5d}  -> {paths[role]}")
    print("  overlaps: train&val=0  train&test=0  val&test=0")
    return counts
