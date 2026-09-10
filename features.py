"""
features.py — wide focus/sharpness feature vector for ROI crops.

One feature set for now ("wide"):
  - all 23 Chris FOCUS CompLIB measures (OpenCV/NumPy port)
  - plus modern extras (Laplacian variance, Tenengrad, FFT ratio,
    ROI stats, entropy, modified Laplacian, BRISQUE-MSCN proxy,
    wavelet detail energy, ...)

Later we will rank importance and narrow down.

Training uses the locked top-15 subset in modeling.SELECTED_FEATURES
(not the full WIDE_NAMES list).

Chris values use his post-norm: v /= (w*h); if v > 0: v = sqrt(v).
Not bit-identical to C++, but same operators/intent.

BRISQUE note: opencv-headless has no QualityBRISQUE; we use an MSCN
natural-scene statistic proxy (core of BRISQUE) as `brisque_mscn_std`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pywt
from tqdm import tqdm

from dataset import DEFAULT_SPLIT_DIR, load_dataset, split_paths_for_stem

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_FEATURE_DIR = REPO_ROOT / "artifacts" / "features"

# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

CHRIS23_NAMES = [
    "1stOrder3x3",
    "Roberts3x3",
    "Prewitt3x3",
    "Scharr3x3",
    "Sobel3x3",
    "Sobel5x5",
    "Laplacian3x3",
    "Laplacian5x5",
    "Sobel2ndOrder3x3",
    "Sobel2ndOrder5x5",
    "Brenner",
    "ThresholdGradient",
    "SquaredGradient",
    "MMHistogram",
    "MasonGreenHistogram",
    "ThresholdTotal",
    "Power",
    "Vollath4",
    "Vollath5",
    "Sobel2ndOrder3x3Cross",
    "Sobel2ndOrder5x5Cross",
    "FirstDerivGaussian",
    "LaplacianOfGaussian",
]

MODERN_EXTRA_NAMES = [
    "var_laplacian",
    "tenengrad",
    "fft_high_freq_ratio",
    "grad_mean",
    "roi_mean",
    "roi_std",
    "entropy",
    "modified_laplacian",
    "brisque_mscn_std",
    "wavelet_detail_energy",
    "canny_edge_density",
]

WIDE_NAMES = CHRIS23_NAMES + MODERN_EXTRA_NAMES

FEATURE_SETS = {
    "wide": WIDE_NAMES,  # only set we use for training experiments for now
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_gray_u8(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("image is None")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image


def _chris_norm(v: float, n_pixels: int) -> float:
    v = float(v) / float(n_pixels)
    if v > 0.0:
        v = float(np.sqrt(v))
    return v


def _grad_energy(gray: np.ndarray, kx: np.ndarray, ky: np.ndarray) -> float:
    gx = cv2.filter2D(gray, cv2.CV_64F, kx)
    gy = cv2.filter2D(gray, cv2.CV_64F, ky)
    return float(np.sum(gx * gx + gy * gy))


def _kernel_energy(gray: np.ndarray, k: np.ndarray) -> float:
    r = cv2.filter2D(gray, cv2.CV_64F, k)
    return float(np.sum(r * r))


# ---------------------------------------------------------------------------
# Chris-23
# ---------------------------------------------------------------------------

def _chris23_values(gray_u8: np.ndarray) -> list[float]:
    g = gray_u8.astype(np.float64)
    h, w = g.shape
    n = h * w

    first = _grad_energy(
        g,
        np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], dtype=np.float64),
        np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]], dtype=np.float64),
    )
    roberts = _grad_energy(
        g,
        np.array([[0, 0, 0], [0, 1, 0], [-1, 0, 0]], dtype=np.float64),
        np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64),
    )
    prewitt = _grad_energy(
        g,
        np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64),
        np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64),
    )
    scharr = _grad_energy(
        g,
        np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float64),
        np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=np.float64),
    )
    sobel3 = _grad_energy(
        g,
        np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64),
        np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64),
    )
    sobel5 = _grad_energy(
        g,
        np.array(
            [
                [-1, -2, 0, 2, 1],
                [-4, -8, 0, 8, 4],
                [-6, -12, 0, 12, 6],
                [-4, -8, 0, 8, 4],
                [-1, -2, 0, 2, 1],
            ],
            dtype=np.float64,
        ),
        np.array(
            [
                [-1, -4, -6, -4, -1],
                [-2, -8, -12, -8, -2],
                [0, 0, 0, 0, 0],
                [2, 8, 12, 8, 2],
                [1, 4, 6, 4, 1],
            ],
            dtype=np.float64,
        ),
    )
    lap3 = _kernel_energy(
        g, np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)
    )
    lap5 = _kernel_energy(
        g,
        np.array(
            [
                [0, 0, -1, 0, 0],
                [0, -1, -2, -1, 0],
                [-1, -2, 16, -2, -1],
                [0, -1, -2, -1, 0],
                [0, 0, -1, 0, 0],
            ],
            dtype=np.float64,
        ),
    )
    sobel2_3 = _grad_energy(
        g,
        np.array([[1, 2, 1], [-2, -4, -2], [1, 2, 1]], dtype=np.float64),
        np.array([[1, -2, 1], [2, -4, 2], [1, -2, 1]], dtype=np.float64),
    )
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
    sobel2_5 = _grad_energy(g, so5x, so5x.T)

    d2 = np.abs(g[:, 2:] - g[:, :-2])
    brenner = float(np.sum(d2 * d2))
    d1 = np.abs(g[:, 1:] - g[:, :-1])
    thr_grad = float(np.sum(d1))
    sq_grad = float(np.sum(d1 * d1))

    hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    mm_hist = float(np.sum(np.arange(128, 256) * hist[128:256]))

    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    delta = np.abs(gx) + np.abs(gy)
    thr = int(np.clip((delta * g).sum() / (delta.sum() + 1e-12), 0, 255))
    bins = np.arange(256)
    mg_hist = float(np.sum(hist[thr + 1 :] * (bins[thr + 1 :] - thr)))

    thr_total = float(g.sum())
    power = float(np.sum(g * g))
    vollath4 = float(np.sum(g[:-1, :] * g[1:, :]) - np.sum(g[:-2, :] * g[2:, :]))
    mean = float(g.mean())
    vollath5 = float(np.sum(g[:-1, :] * g[1:, :]) - n * mean * mean)

    sobel_cross3 = _kernel_energy(
        g, np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]], dtype=np.float64)
    )
    sobel_cross5 = _kernel_energy(
        g,
        np.array(
            [
                [-1, -2, 0, 2, 1],
                [-2, -4, 0, 4, 2],
                [0, 0, 0, 0, 0],
                [2, 4, 0, -4, -2],
                [1, 2, 0, -2, -1],
            ],
            dtype=np.float64,
        ),
    )

    blur = cv2.GaussianBlur(g, (7, 7), 0.8)
    fdg = _grad_energy(
        blur,
        np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64),
        np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64),
    )
    log = cv2.Laplacian(cv2.GaussianBlur(g, (9, 9), 1.2), cv2.CV_64F, ksize=9)
    log_e = float(np.sum(log * log))

    raw = [
        first,
        roberts,
        prewitt,
        scharr,
        sobel3,
        sobel5,
        lap3,
        lap5,
        sobel2_3,
        sobel2_5,
        brenner,
        thr_grad,
        sq_grad,
        mm_hist,
        mg_hist,
        thr_total,
        power,
        vollath4,
        vollath5,
        sobel_cross3,
        sobel_cross5,
        fdg,
        log_e,
    ]
    return [_chris_norm(v, n) for v in raw]


# ---------------------------------------------------------------------------
# Modern extras
# ---------------------------------------------------------------------------

def _modern_extra_values(gray_u8: np.ndarray) -> list[float]:
    g = gray_u8.astype(np.float64)

    lap = cv2.Laplacian(g, cv2.CV_64F)
    var_laplacian = float(lap.var())

    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx * gx + gy * gy))
    grad_mean = float(np.mean(np.sqrt(gx * gx + gy * gy)))

    f = np.fft.fftshift(np.fft.fft2(g))
    mag = np.abs(f)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    rh, rw = max(1, h // 20), max(1, w // 20)
    low = mag[cy - rh : cy + rh, cx - rw : cx + rw].sum()
    fft_high_freq_ratio = float((mag.sum() - low) / (mag.sum() + 1e-12))

    roi_mean = float(g.mean())
    roi_std = float(g.std())

    hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum())

    kx = np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]], dtype=np.float64)
    ky = np.array([[0, -1, 0], [0, 2, 0], [0, -1, 0]], dtype=np.float64)
    ml = np.abs(cv2.filter2D(g, cv2.CV_64F, kx)) + np.abs(cv2.filter2D(g, cv2.CV_64F, ky))
    modified_laplacian = float(ml.mean())

    # BRISQUE-style MSCN coefficient spread (proxy; lower often "more natural")
    gf = g
    mu = cv2.GaussianBlur(gf, (7, 7), 7.0 / 6.0)
    sigma = np.sqrt(
        np.abs(cv2.GaussianBlur(gf * gf, (7, 7), 7.0 / 6.0) - mu * mu)
    )
    mscn = (gf - mu) / (sigma + 1.0)
    brisque_mscn_std = float(mscn.std())

    coeffs = pywt.wavedec2(g, wavelet="db1", level=2)
    detail_energy = 0.0
    total_energy = float(np.sum(coeffs[0] ** 2))
    for detail in coeffs[1:]:
        for arr in detail:
            e = float(np.sum(np.asarray(arr) ** 2))
            detail_energy += e
            total_energy += e
    wavelet_detail_energy = float(detail_energy / (total_energy + 1e-12))

    edges = cv2.Canny(gray_u8, 100, 200)
    canny_edge_density = float(edges.mean() / 255.0)

    return [
        var_laplacian,
        tenengrad,
        fft_high_freq_ratio,
        grad_mean,
        roi_mean,
        roi_std,
        entropy,
        modified_laplacian,
        brisque_mscn_std,
        wavelet_detail_energy,
        canny_edge_density,
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(
    image: np.ndarray,
    set_name: str = "wide",
) -> tuple[np.ndarray, list[str]]:
    """Return (feature_vector float32, feature_names)."""
    if set_name not in FEATURE_SETS:
        raise KeyError(
            f"unknown feature set {set_name!r}; available: {list(FEATURE_SETS)}"
        )

    gray = _as_gray_u8(image)
    vals = _chris23_values(gray) + _modern_extra_values(gray)
    return np.asarray(vals, dtype=np.float32), list(WIDE_NAMES)


def extract_dataset_features(
    df: pd.DataFrame,
    set_name: str = "wide",
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    X_list = []
    keep_rows = []
    names: list[str] | None = None

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"features:{set_name}"):
        path = Path(str(row["path"]))
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  WARNING: could not read {path}")
            continue
        vec, names = extract_features(img, set_name=set_name)
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
    assert names is not None
    return X, y_codes.astype(np.int32), names, meta


def save_feature_bundle(
    out_path: Path,
    X: np.ndarray,
    y_codes: np.ndarray,
    feature_names: list[str],
    meta: pd.DataFrame,
    set_name: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X,
        y_codes=y_codes,
        feature_names=np.array(feature_names),
        groups=meta["group"].to_numpy(dtype=str),
        paths=meta["path"].to_numpy(dtype=str),
        rating_raw=meta["rating_raw"].to_numpy(dtype=np.float32),
        set_name=np.array(set_name),
    )
    side = out_path.with_suffix(".meta.json")
    side.write_text(
        json.dumps(
            {
                "set_name": set_name,
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "feature_names": feature_names,
                "chris23": CHRIS23_NAMES,
                "modern_extras": MODERN_EXTRA_NAMES,
                "group_counts": meta["group"].value_counts().sort_index().to_dict(),
                "npz": str(out_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def consolidate_split_features(
    stem: str,
    set_name: str,
    feature_dir: Path,
    out_path: Path | None = None,
) -> Path:
    """Merge per-split NPZs into artifacts/features/<set>/features.npz for training."""
    out = out_path or (feature_dir / set_name / "features.npz")
    parts = []
    for role in ("train", "val", "test"):
        path = feature_dir / f"{stem}_{set_name}_{role}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        d = np.load(path, allow_pickle=True)
        n = int(d["X"].shape[0])
        print(f"{role}: X={d['X'].shape}  file={path}")
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

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        X=np.vstack([p["X"] for p in parts]),
        groups=np.concatenate([p["groups"] for p in parts]),
        paths=np.concatenate([p["paths"] for p in parts]),
        rating_raw=np.concatenate([p["rating_raw"] for p in parts]),
        split=np.concatenate([p["split"] for p in parts]).astype(str),
        feature_names=parts[0]["feature_names"],
    )
    loaded = np.load(out, allow_pickle=True)
    counts = dict(zip(*np.unique(loaded["split"], return_counts=True)))
    print(f"saved {out.resolve()}  X={loaded['X'].shape}  splits={counts}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Extract wide focus features for split CSVs")
    p.add_argument("--set", dest="set_name", default="wide", choices=list(FEATURE_SETS))
    p.add_argument("--stem", default="train_sample_five")
    p.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    p.add_argument("--splits", default="train,val,test")
    p.add_argument("--limit", type=int, default=None, help="Smoke-test row cap per split")
    p.add_argument(
        "--consolidate",
        action="store_true",
        help="Merge existing per-split NPZs into <out-dir>/<set>/features.npz (no extract)",
    )
    args = p.parse_args()

    if args.consolidate:
        consolidate_split_features(args.stem, args.set_name, args.out_dir)
        return

    roles = [s.strip() for s in args.splits.split(",") if s.strip()]
    split_paths = split_paths_for_stem(args.split_dir, args.stem)

    print(f"Wide feature count: {len(WIDE_NAMES)} "
          f"(chris23={len(CHRIS23_NAMES)} + modern={len(MODERN_EXTRA_NAMES)})")

    for role in roles:
        csv_path = split_paths[role]
        print(f"\n=== {args.set_name} / {role} ===")
        print(f"  csv: {csv_path}")
        df = load_dataset(csv_path)
        if args.limit is not None:
            df = df.head(args.limit).copy()
            print(f"  limited to {len(df)} rows")

        X, y_codes, names, meta = extract_dataset_features(df, args.set_name)
        out = args.out_dir / f"{args.stem}_{args.set_name}_{role}.npz"
        save_feature_bundle(out, X, y_codes, names, meta, args.set_name)
        print(f"  saved X={X.shape} -> {out.resolve()}")


if __name__ == "__main__":
    main()
