"""
features.py — Python training-time extract of the locked 13-feature vector.

Production (on-aircraft) feature extraction is C++ in libDeterminator.
This module is only a port so we can train from labeled focusM ROIs.

Order is the model / C++ input layout. Do not reorder without re-exporting trees.
CompLIB-style values use Chris's post-norm: v /= (w*h); if v > 0: v = sqrt(v).
Not bit-identical to C++; same operators, names, and order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset import DEFAULT_SPLIT_DIR, load_dataset, split_paths_for_stem

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_FEATURE_DIR = REPO_ROOT / "artifacts" / "features"

# Locked inference / training feature set. Same order as the shipped model.
FEATURE_NAMES: list[str] = [
    "fft_high_freq_ratio",
    "Sobel2ndOrder5x5",
    "grad_mean",
    "Laplacian3x3",
    "ThresholdGradient",
    "Sobel2ndOrder3x3",
    "roi_std",
    "Vollath5",
    "LaplacianOfGaussian",
    "Sobel2ndOrder3x3Cross",
    "entropy",
    "modified_laplacian",
    "Sobel2ndOrder5x5Cross",
]


def _as_gray_u8(image: np.ndarray) -> np.ndarray:
    """Convert an ROI to 8-bit grayscale."""
    if image is None:
        raise ValueError("image is None")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image


def _chris_norm(v: float, n_pixels: int) -> float:
    """CompLIB per-pixel norm, then sqrt if positive."""
    v = float(v) / float(n_pixels)
    if v > 0.0:
        v = float(np.sqrt(v))
    return v


def _grad_energy(gray: np.ndarray, kx: np.ndarray, ky: np.ndarray) -> float:
    """Sum of squared responses of a pair of derivative kernels."""
    gx = cv2.filter2D(gray, cv2.CV_64F, kx)
    gy = cv2.filter2D(gray, cv2.CV_64F, ky)
    return float(np.sum(gx * gx + gy * gy))


def _kernel_energy(gray: np.ndarray, k: np.ndarray) -> float:
    """Sum of squared responses of one kernel."""
    r = cv2.filter2D(gray, cv2.CV_64F, k)
    return float(np.sum(r * r))


def _feature_values(gray_u8: np.ndarray) -> dict[str, float]:
    """Compute the 13 locked measures on a grayscale ROI."""
    g = gray_u8.astype(np.float64)
    h, w = g.shape
    n = h * w

    so3x = np.array([[1, 2, 1], [-2, -4, -2], [1, 2, 1]], dtype=np.float64)
    so3y = np.array([[1, -2, 1], [2, -4, 2], [1, -2, 1]], dtype=np.float64)
    so5x = np.array(
        [
            [1, 4, 6, 4, 1],
            [0, 0, 0, 0, 0],
            [-2, -8, -12, -8, -2],
            [0, 0, 0, 0, 0],
            [1, 4, 6, 4, 1],
        ],
        dtype=np.float64,
    )
    lap3 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)
    cross3 = np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]], dtype=np.float64)
    cross5 = np.array(
        [
            [-1, -2, 0, 2, 1],
            [-2, -4, 0, 4, 2],
            [0, 0, 0, 0, 0],
            [2, 4, 0, -4, -2],
            [1, 2, 0, -2, -1],
        ],
        dtype=np.float64,
    )

    d1 = np.abs(g[:, 1:] - g[:, :-1])
    mean = float(g.mean())
    log = cv2.Laplacian(cv2.GaussianBlur(g, (9, 9), 1.2), cv2.CV_64F, ksize=9)

    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    f = np.fft.fftshift(np.fft.fft2(g))
    mag = np.abs(f)
    cy, cx = h // 2, w // 2
    rh, rw = max(1, h // 20), max(1, w // 20)
    low = mag[cy - rh : cy + rh, cx - rw : cx + rw].sum()
    hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    mlx = np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]], dtype=np.float64)
    mly = np.array([[0, -1, 0], [0, 2, 0], [0, -1, 0]], dtype=np.float64)
    ml = np.abs(cv2.filter2D(g, cv2.CV_64F, mlx)) + np.abs(cv2.filter2D(g, cv2.CV_64F, mly))

    return {
        "fft_high_freq_ratio": float((mag.sum() - low) / (mag.sum() + 1e-12)),
        "Sobel2ndOrder5x5": _chris_norm(_grad_energy(g, so5x, so5x.T), n),
        "grad_mean": float(np.mean(np.sqrt(gx * gx + gy * gy))),
        "Laplacian3x3": _chris_norm(_kernel_energy(g, lap3), n),
        "ThresholdGradient": _chris_norm(float(np.sum(d1)), n),
        "Sobel2ndOrder3x3": _chris_norm(_grad_energy(g, so3x, so3y), n),
        "roi_std": float(g.std()),
        "Vollath5": _chris_norm(float(np.sum(g[:-1, :] * g[1:, :]) - n * mean * mean), n),
        "LaplacianOfGaussian": _chris_norm(float(np.sum(log * log)), n),
        "Sobel2ndOrder3x3Cross": _chris_norm(_kernel_energy(g, cross3), n),
        "entropy": float(-(p * np.log2(p)).sum()),
        "modified_laplacian": float(ml.mean()),
        "Sobel2ndOrder5x5Cross": _chris_norm(_kernel_energy(g, cross5), n),
    }


def extract_features(image: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Return (vector, FEATURE_NAMES) for one ROI."""
    values = _feature_values(_as_gray_u8(image))
    vec = np.asarray([values[name] for name in FEATURE_NAMES], dtype=np.float32)
    return vec, list(FEATURE_NAMES)


def extract_dataset_features(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """Extract features for each row in a sample CSV (`path`, `group`)."""
    X_list = []
    keep_rows = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="features"):
        path = Path(str(row["path"]))
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  WARNING: could not read {path}")
            continue
        vec, _ = extract_features(img)
        if not np.isfinite(vec).all():
            print(f"  WARNING: non-finite features for {path}")
            continue
        X_list.append(vec)
        keep_rows.append(idx)

    if not X_list:
        raise RuntimeError("no features extracted")

    meta = df.loc[keep_rows].reset_index(drop=True)
    X = np.vstack(X_list)
    y_codes, _ = pd.factorize(meta["group"], sort=True)
    return X, y_codes.astype(np.int32), list(FEATURE_NAMES), meta


def save_feature_bundle(
    out_path: Path,
    X: np.ndarray,
    y_codes: np.ndarray,
    feature_names: list[str],
    meta: pd.DataFrame,
) -> None:
    """Write a compressed NPZ plus a small sidecar JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X,
        y_codes=y_codes,
        feature_names=np.array(feature_names),
        groups=meta["group"].to_numpy(dtype=str),
        paths=meta["path"].to_numpy(dtype=str),
        rating_raw=meta["rating_raw"].to_numpy(dtype=np.float32),
    )
    out_path.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "feature_names": feature_names,
                "group_counts": meta["group"].value_counts().sort_index().to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def consolidate_split_features(stem: str, feature_dir: Path) -> Path:
    """Merge `{stem}_{train,val,test}.npz` into `feature_dir/features.npz`."""
    parts = []
    for role in ("train", "val", "test"):
        path = feature_dir / f"{stem}_{role}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        d = np.load(path, allow_pickle=True)
        n = int(d["X"].shape[0])
        parts.append(
            {
                "X": d["X"].astype(np.float32),
                "groups": d["groups"].astype(str),
                "paths": d["paths"].astype(str),
                "rating_raw": d["rating_raw"].astype(np.float32),
                "split": np.full(n, role, dtype=object),
                "feature_names": d["feature_names"],
            }
        )

    out = feature_dir / "features.npz"
    out.parent.mkdir(parents=True, exist_ok=True)

    ref_names = [str(x) for x in parts[0]["feature_names"]]
    for role, part in zip(("train", "val", "test"), parts):
        names = [str(x) for x in part["feature_names"]]
        if names != ref_names:
            raise ValueError(
                f"feature_names mismatch: train={ref_names!r} vs {role}={names!r}"
            )
        if part["X"].shape[1] != len(ref_names):
            raise ValueError(
                f"{role} width {part['X'].shape[1]} != len(feature_names) {len(ref_names)}"
            )

    np.savez_compressed(
        out,
        X=np.vstack([p["X"] for p in parts]),
        groups=np.concatenate([p["groups"] for p in parts]),
        paths=np.concatenate([p["paths"] for p in parts]),
        rating_raw=np.concatenate([p["rating_raw"] for p in parts]),
        split=np.concatenate([p["split"] for p in parts]).astype(str),
        feature_names=np.array(ref_names, dtype=object),
    )
    return out


def parse_args() -> argparse.Namespace:
    """CLI for per-split extract or --consolidate."""
    p = argparse.ArgumentParser(description="Extract the locked 13 features for split CSVs")
    p.add_argument("--stem", default="train_sample")
    p.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    p.add_argument("--splits", default="train,val,test")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--consolidate", action="store_true")
    return p.parse_args()


def main() -> None:
    """Extract per-split NPZs, or merge them into features.npz."""
    args = parse_args()
    if args.consolidate:
        out = consolidate_split_features(args.stem, args.out_dir)
        print(f"saved {out.resolve()}")
        return

    split_paths = split_paths_for_stem(args.split_dir, args.stem)
    for role in [s.strip() for s in args.splits.split(",") if s.strip()]:
        df = load_dataset(split_paths[role])
        if args.limit is not None:
            df = df.head(args.limit).copy()
        X, y_codes, names, meta = extract_dataset_features(df)
        out = args.out_dir / f"{args.stem}_{role}.npz"
        save_feature_bundle(out, X, y_codes, names, meta)
        print(f"{role}: X={X.shape} -> {out.resolve()}")


if __name__ == "__main__":
    main()
