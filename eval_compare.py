"""
eval_compare.py — prep old Determinator eval frames and score with 9-class model.

Pipeline:
  1. Load full-frame images (jpg/png/tif/tiff; optional camera RAW via rawpy)
  2. Write full-frame TIFFs to <out>/raw/  (for visual inspection)
  3. Cut five 1024x1024 ROIs matching libDeterminator centers (UL, UR, LL, LR, C)
  4. Write ROI TIFFs to <out>/roi/
  5. Score each ROI with artifacts/models/xgboost_identity9.joblib (0.5-step bins)

ROI geometry matches Determinator::namedROIToXYCoords + 1024x1024 box centered
on that point (see libDeterminator determinatortypes.h / Determinator.cpp).

Examples:
  # drop full-frame images into artifacts/eval_compare/input/ then:
  python eval_compare.py
  python eval_compare.py --limit 5
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
from tqdm import tqdm

from features import extract_features
from modeling import _score_for_label

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = REPO_ROOT / "artifacts" / "models" / "xgboost_identity9.joblib"
DEFAULT_OUT = REPO_ROOT / "artifacts" / "eval_compare"
DEFAULT_INPUT = DEFAULT_OUT / "input"

ROI_SIZE = 1024
ROI_NAMES = ("UL", "UR", "LL", "LR", "C")

# OpenCV-readable full frames
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
# Camera RAW (needs rawpy)
CAMERA_RAW_EXTS = {".dng", ".nef", ".cr2", ".arw", ".orf", ".rw2"}


def roi_center(image_width: int, image_height: int, name: str) -> tuple[int, int]:
    """
    Center pixel of named ROI — mirrors Determinator::namedROIToXYCoords
    (including the +1 legacy offset). Do not simplify the integer math.
    """
    w = image_width // 3
    h = image_height // 3
    if name == "UL":
        x, y = w // 2, h // 2
    elif name == "UR":
        x, y = (w * 5) // 2, h // 2
    elif name == "LR":
        x, y = (w * 5) // 2, (h * 5) // 2
    elif name == "LL":
        x, y = w // 2, (h * 5) // 2
    elif name == "C":
        x, y = (w * 3) // 2, (h * 3) // 2
    else:
        raise KeyError(f"unknown ROI {name!r}")
    return x + 1, y + 1


def crop_roi(img: np.ndarray, name: str, size: int = ROI_SIZE) -> np.ndarray | None:
    """Return size×size crop centered on named ROI, or None if out of bounds."""
    h, w = img.shape[:2]
    cx, cy = roi_center(w, h, name)
    half = size // 2
    tlx = cx - half
    tly = cy - half
    if tlx < 0 or tly < 0 or tlx + size > w or tly + size > h:
        return None
    return img[tly : tly + size, tlx : tlx + size].copy()


def load_full_frame(path: Path) -> np.ndarray | None:
    """Load BGR uint8 image suitable for ROI cutting / feature extract."""
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        return _to_bgr_u8(img)

    if ext in CAMERA_RAW_EXTS:
        try:
            import rawpy  # type: ignore
        except ImportError:
            print(f"  WARNING: rawpy not installed; cannot read {path.name}")
            return None
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if ext == ".raw":
        print(
            f"  WARNING: packed .raw needs known width/height/Bayer pattern; "
            f"skipping {path.name}"
        )
        return None

    return None


def _to_bgr_u8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.shape[2] == 3:
        return img
    raise ValueError(f"unsupported channel count: {img.shape}")


def write_tiff(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise RuntimeError(f"failed to write {path}")


def iter_input_files(input_dir: Path) -> list[Path]:
    files: list[Path] = []
    for p in sorted(input_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() in IMAGE_EXTS | CAMERA_RAW_EXTS | {".raw"}:
            files.append(p)
    return files


def load_identity9(model_path: Path) -> dict[str, Any]:
    bundle = joblib.load(model_path)
    feature_names = list(bundle["feature_names"])
    # Prefer locked SELECTED_FEATURES order if present in bundle
    selected = list(bundle.get("selected_features") or feature_names)
    return {
        "model": bundle["model"],
        "label_encoder": bundle["label_encoder"],
        "feature_names": feature_names,
        "selected_features": selected,
    }


def score_roi(
    img: np.ndarray,
    bundle: dict[str, Any],
) -> dict[str, Any]:
    vec, names = extract_features(img, set_name="wide")
    name_to_idx = {n: i for i, n in enumerate(names)}
    selected = bundle["selected_features"]
    missing = [n for n in selected if n not in name_to_idx]
    if missing:
        raise KeyError(f"model features missing from wide extract: {missing}")
    x = vec[[name_to_idx[n] for n in selected]].astype(np.float32).reshape(1, -1)
    code = int(bundle["model"].predict(x)[0])
    label = str(bundle["label_encoder"].inverse_transform([code])[0])
    proba = None
    if hasattr(bundle["model"], "predict_proba"):
        p = bundle["model"].predict_proba(x)[0]
        proba = {str(c): float(v) for c, v in zip(bundle["label_encoder"].classes_, p)}
    return {
        "pred_group": label,
        "pred_score": float(_score_for_label(label)),
        "proba": proba,
    }


def process_image(
    src: Path,
    input_root: Path,
    raw_dir: Path,
    roi_dir: Path,
    bundle: dict[str, Any],
) -> list[dict[str, Any]]:
    img = load_full_frame(src)
    if img is None:
        print(f"  SKIP unreadable: {src}")
        return []

    h, w = img.shape[:2]
    if w < ROI_SIZE or h < ROI_SIZE:
        print(f"  SKIP too small ({w}x{h}): {src}")
        return []

    rel = src.relative_to(input_root) if src.is_relative_to(input_root) else Path(src.name)
    stem = rel.with_suffix("").as_posix().replace("/", "__")

    full_tiff = raw_dir / f"{stem}.tif"
    write_tiff(full_tiff, img)

    rows: list[dict[str, Any]] = []
    for name in ROI_NAMES:
        crop = crop_roi(img, name, ROI_SIZE)
        if crop is None:
            print(f"  WARNING: ROI {name} OOB for {src.name} ({w}x{h})")
            continue
        roi_path = roi_dir / f"{stem}__{name}.tif"
        write_tiff(roi_path, crop)
        pred = score_roi(crop, bundle)
        cx, cy = roi_center(w, h, name)
        rows.append(
            {
                "source": str(src),
                "full_tiff": str(full_tiff),
                "roi_tiff": str(roi_path),
                "roi": name,
                "image_wh": f"{w}x{h}",
                "roi_center_xy": f"{cx},{cy}",
                "pred_group": pred["pred_group"],
                "pred_score": pred["pred_score"],
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert old-eval frames → raw/ROI TIFFs and score with 9-class model"
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
        help="Folder of old Determinator eval images (default: artifacts/eval_compare/input)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="Output root (creates raw/ and roi/ subfolders)",
    )
    p.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="9-class (0.5-step) joblib model",
    )
    p.add_argument("--limit", type=int, default=None, help="Max images to process")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    out_dir = args.out_dir.resolve()
    raw_dir = out_dir / "raw"
    roi_dir = out_dir / "roi"
    raw_dir.mkdir(parents=True, exist_ok=True)
    roi_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        raise SystemExit(f"input-dir not found: {input_dir}")

    bundle = load_identity9(args.model.resolve())
    files = iter_input_files(input_dir)
    if args.limit is not None:
        files = files[: args.limit]

    print(f"input:  {input_dir}")
    print(f"raw/:   {raw_dir}")
    print(f"roi/:   {roi_dir}")
    print(f"model:  {args.model.resolve()}")
    print(f"classes: {list(bundle['label_encoder'].classes_)}")
    print(f"images: {len(files)}")
    print(f"ROI size: {ROI_SIZE}x{ROI_SIZE}  names: {', '.join(ROI_NAMES)}")

    all_rows: list[dict[str, Any]] = []
    for src in tqdm(files, desc="eval_compare"):
        all_rows.extend(process_image(src, input_dir, raw_dir, roi_dir, bundle))

    scores_path = out_dir / "scores_identity9.csv"
    fieldnames = [
        "source",
        "full_tiff",
        "roi_tiff",
        "roi",
        "image_wh",
        "roi_center_xy",
        "pred_group",
        "pred_score",
    ]
    with scores_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    # Per-image summary (mean of ROI scores) for easy old-vs-new compare later
    by_src: dict[str, list[float]] = {}
    for row in all_rows:
        by_src.setdefault(row["source"], []).append(float(row["pred_score"]))
    summary = [
        {
            "source": src,
            "n_rois": len(scores),
            "mean_pred_score": float(np.mean(scores)) if scores else None,
            "min_pred_score": float(np.min(scores)) if scores else None,
            "max_pred_score": float(np.max(scores)) if scores else None,
        }
        for src, scores in by_src.items()
    ]
    summary_path = out_dir / "scores_identity9_by_image.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["source", "n_rois", "mean_pred_score", "min_pred_score", "max_pred_score"],
        )
        w.writeheader()
        w.writerows(summary)

    meta = {
        "model": str(args.model.resolve()),
        "classes": [str(c) for c in bundle["label_encoder"].classes_],
        "roi_size": ROI_SIZE,
        "roi_names": list(ROI_NAMES),
        "n_source_images": len(files),
        "n_roi_scores": len(all_rows),
        "raw_dir": str(raw_dir),
        "roi_dir": str(roi_dir),
        "scores_csv": str(scores_path),
        "summary_csv": str(summary_path),
        "note": (
            "pred_group is 9-bin identity (1.0..5.0 step 0.5). "
            "ROI centers match libDeterminator namedROIToXYCoords."
        ),
    }
    meta_path = out_dir / "eval_compare_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nWrote {len(all_rows)} ROI scores")
    print(f"  scores  -> {scores_path}")
    print(f"  by-image -> {summary_path}")
    print(f"  meta    -> {meta_path}")
    print(f"  inspect raw TIFFs in {raw_dir}")
    print(f"  inspect ROI TIFFs in {roi_dir}")


if __name__ == "__main__":
    main()
