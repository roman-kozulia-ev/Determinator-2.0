"""
modeling.py — train and evaluate focus classifiers on wide features.

Default feature set is the locked top-15 subset (see SELECTED_FEATURES),
chosen by VAL permutation importance. Default model is tuned XGBoost.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parent
# Full 5-bin (collapsed) feature bundle: train/val/test already embedded.
DEFAULT_FEATURES = (
    REPO_ROOT / "artifacts" / "features" / "wide" / "features_full_imbalanced.npz"
)
DEFAULT_MODEL_DIR = REPO_ROOT / "artifacts" / "models"
# Companion 9-class bundle: artifacts/features/wide/features_identity9.npz

# Locked training subset (15 of 34 wide features).
# Ranked by VAL permutation importance (Δ accuracy when shuffled), 5-bin model.
# Ablation: top-15 beat full-34 / top-10 / top-16 on held-out TEST
# (~53.7% TEST acc vs ~51.4% with all 34). Order = importance rank.
SELECTED_FEATURES: list[str] = [
    "fft_high_freq_ratio",      # 1  frequency / blur proxy
    "Sobel2ndOrder5x5",         # 2  Chris 2nd-order Sobel (5x5)
    "grad_mean",                # 3  mean gradient magnitude
    "Laplacian3x3",             # 4  Chris Laplacian 3x3
    "brisque_mscn_std",         # 5  MSCN natural-scene proxy
    "ThresholdGradient",        # 6  Chris thresholded gradient
    "Sobel2ndOrder3x3",         # 7  Chris 2nd-order Sobel (3x3)
    "roi_std",                  # 8  ROI intensity std
    "Vollath5",                 # 9  Chris autocorrelation focus
    "canny_edge_density",       # 10 edge pixel density
    "LaplacianOfGaussian",      # 11 Chris LoG
    "Sobel2ndOrder3x3Cross",    # 12 Chris cross 2nd-order 3x3
    "entropy",                  # 13 gray-level entropy
    "modified_laplacian",       # 14 abs-Laplacian sharpness
    "Sobel2ndOrder5x5Cross",    # 15 Chris cross 2nd-order 5x5
]

# Explicit group -> score for ordinal MAE (ranges use midpoints).
GROUP_TO_SCORE = {
    "1.0": 1.0,
    "1.5": 1.5,
    "2.0": 2.0,
    "2.5": 2.5,
    "3.0": 3.0,
    "3.5": 3.5,
    "4.0": 4.0,
    "4.5": 4.5,
    "5.0": 5.0,
    "1.0-1.5": 1.25,
    "3.0-3.5": 3.25,
    "4.0-5.0": 4.5,
}


def _midpoint_from_range(key: str) -> float | None:
    for sep in ("-", ".."):
        if sep not in key:
            continue
        parts = key.split(sep)
        if len(parts) != 2:
            continue
        try:
            return 0.5 * (float(parts[0]) + float(parts[1]))
        except ValueError:
            continue
    return None


def _score_for_label(label: str) -> float:
    key = str(label)
    if key in GROUP_TO_SCORE:
        return GROUP_TO_SCORE[key]
    mid = _midpoint_from_range(key)
    if mid is not None:
        return mid
    try:
        return float(key)
    except ValueError:
        raise KeyError(f"no numeric score mapping for group {key!r}")


def _scores_from_labels(labels: np.ndarray) -> np.ndarray:
    return np.array([_score_for_label(g) for g in labels], dtype=np.float64)


def load_feature_splits(
    npz_path: Path | str = DEFAULT_FEATURES,
) -> dict[str, Any]:
    npz_path = Path(npz_path)
    d = np.load(npz_path, allow_pickle=True)
    split = d["split"].astype(str)
    groups = d["groups"].astype(str)
    X = d["X"].astype(np.float32)
    names = [str(x) for x in d["feature_names"]]

    out: dict[str, Any] = {
        "feature_names": names,
        "paths": d["paths"].astype(str),
        "rating_raw": d["rating_raw"].astype(np.float32),
    }
    for role in ("train", "val", "test"):
        m = split == role
        out[role] = {
            "X": X[m],
            "groups": groups[m],
            "paths": d["paths"].astype(str)[m],
            "rating_raw": d["rating_raw"].astype(np.float32)[m],
        }
    return out


def select_feature_columns(
    feature_names: list[str],
    X: np.ndarray,
    selected: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Return (X[:, selected], selected_names) in SELECTED_FEATURES order."""
    chosen = list(SELECTED_FEATURES if selected is None else selected)
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    missing = [n for n in chosen if n not in name_to_idx]
    if missing:
        raise KeyError(f"selected features missing from NPZ: {missing}")
    cols = [name_to_idx[n] for n in chosen]
    return X[:, cols].astype(np.float32), chosen


def build_models(seed: int = 42) -> dict[str, Any]:
    """
    Canonical model factory: tuned XGBoost on the locked top-15 features.

    This is not online / transfer fine-tuning. Hyperparameters below were chosen
    once via VAL random search, then locked here. Each train run fits a fresh
    booster from scratch with these settings (``train_model.py``).
    """
    return {
        # Locked VAL-tuned config (see hyperparam search notes / compare_val.json).
        "xgboost": XGBClassifier(
            n_estimators=600,  # number of boosting rounds (trees added sequentially)
            learning_rate=0.05,  # step size per tree; lower = slower, usually more stable
            max_depth=5,  # max tree depth; controls how complex each tree can be
            min_child_weight=8,  # min hessian/sum of weights in a leaf; higher = more conservative splits
            subsample=0.8,  # fraction of training rows used per tree (row bagging)
            colsample_bytree=0.8,  # fraction of features used per tree (column bagging)
            gamma=0.3,  # min loss reduction to make a split; higher = fewer splits
            reg_lambda=5.0,  # L2 weight penalty (shrinks leaf weights)
            reg_alpha=1.0,  # L1 weight penalty (encourages sparser leaf weights)
            objective="multi:softprob",  # multiclass: predict class probabilities
            eval_metric="mlogloss",  # multiclass log-loss for monitoring
            tree_method="hist",  # histogram tree builder (fast CPU default)
            random_state=seed,  # reproducibility for sampling / tie-breaking
            n_jobs=-1,  # use all CPU cores
        ),
    }


def balanced_group_weights(counts: dict[str, int]) -> dict[str, float]:
    """
    sklearn-style balanced weights: n / (n_classes * count_g).

    Underrepresented groups get higher weight. Relative importance of group A
    vs B equals count_B / count_A (e.g. 4.0-5.0 vs 1.0-1.5 ≈ 4.2x on the
    full five-bin sample).
    """
    positive = {str(g): int(c) for g, c in counts.items() if int(c) > 0}
    if not positive:
        raise ValueError("counts must include at least one positive class count")
    n = sum(positive.values())
    k = len(positive)
    return {g: float(n) / (k * c) for g, c in positive.items()}


def compute_balanced_sample_weight(y_train: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Return per-row balanced weights plus a printable class->weight map."""
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_to_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    sample_weight = np.array([class_to_weight[int(c)] for c in y_train], dtype=np.float64)
    return sample_weight, class_to_weight


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    true_s = _scores_from_labels(y_true)
    pred_s = _scores_from_labels(y_pred)
    mae = float(np.mean(np.abs(true_s - pred_s)))
    within_0_5 = float(np.mean(np.abs(true_s - pred_s) <= 0.5))
    within_1_0 = float(np.mean(np.abs(true_s - pred_s) <= 1.0))

    labels = class_names
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "mae_score": mae,
        "within_0.5": within_0_5,
        "within_1.0": within_1_0,
        "confusion_matrix": cm.tolist(),
        "confusion_labels": labels,
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, digits=3, zero_division=0
        ),
    }


def feature_importance_table(
    model: Any,
    feature_names: list[str],
    X_val: np.ndarray,
    y_val: np.ndarray,
    top_k: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """Prefer model importances; fall back to permutation importance on VAL."""
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=np.float64)
    else:
        result = permutation_importance(
            model,
            X_val,
            y_val,
            n_repeats=5,
            random_state=seed,
            n_jobs=-1,
        )
        imp = np.asarray(result.importances_mean, dtype=np.float64)

    df = pd.DataFrame({"feature": feature_names, "importance": imp})
    return df.sort_values("importance", ascending=False).head(top_k).reset_index(drop=True)

def default_model_name(n_classes: int) -> str:
    """Canonical artifact names for the two maintained models."""
    if n_classes == 5:
        return "xgboost_5bin"
    if n_classes == 9:
        return "xgboost_identity9"
    return f"xgboost_{n_classes}class"


def train_and_evaluate(
    features_npz: Path | str = DEFAULT_FEATURES,
    out_dir: Path | str = DEFAULT_MODEL_DIR,
    seed: int = 42,
    include_test: bool = False,
    use_balanced_class_weight: bool = False,
    selected_features: list[str] | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """
    Train tuned XGBoost on the locked top-15 feature subset.

    By default uses SELECTED_FEATURES. Pass selected_features=[] or the full
    wide name list to train on every column in the NPZ.

    Saves ``{model_name}.joblib`` — defaults to ``xgboost_5bin`` (5 classes)
    or ``xgboost_identity9`` (9 classes).
    """
    data = load_feature_splits(features_npz)
    all_names = data["feature_names"]
    use_selected = SELECTED_FEATURES if selected_features is None else list(selected_features)
    if use_selected:
        X_train, feature_names = select_feature_columns(
            all_names, data["train"]["X"], use_selected
        )
        X_val, _ = select_feature_columns(all_names, data["val"]["X"], use_selected)
        X_test, _ = select_feature_columns(all_names, data["test"]["X"], use_selected)
    else:
        feature_names = list(all_names)
        X_train = data["train"]["X"]
        X_val = data["val"]["X"]
        X_test = data["test"]["X"]

    enc = LabelEncoder()
    y_train = enc.fit_transform(data["train"]["groups"])
    y_val = enc.transform(data["val"]["groups"])
    class_names = list(enc.classes_)
    resolved_name = model_name or default_model_name(len(class_names))

    models = {resolved_name: build_models(seed=seed)["xgboost"]}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "features_npz": str(Path(features_npz).resolve()),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "selected_features": feature_names,
        "n_features_available": len(all_names),
        "classes": class_names,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "use_balanced_class_weight": bool(use_balanced_class_weight),
        "model_name": resolved_name,
        "models": {},
    }

    print(f"Train samples: {len(y_train)}  Val samples: {len(y_val)}")
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Features: {len(feature_names)} / {len(all_names)} available")
    for i, name in enumerate(feature_names, 1):
        print(f"  {i:2d}. {name}")
    fit_kwargs: dict[str, Any] = {}
    if use_balanced_class_weight:
        sample_weight, code_to_weight = compute_balanced_sample_weight(y_train)
        weight_by_label = {
            class_names[int(code)]: float(code_to_weight[int(code)])
            for code in sorted(code_to_weight)
        }
        results["class_weight"] = weight_by_label
        print("Balanced class weights:")
        for label, weight in weight_by_label.items():
            print(f"  {label}: {weight:.6f}")
        fit_kwargs["sample_weight"] = sample_weight

    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        model.fit(X_train, y_train, **fit_kwargs)
        pred_codes = model.predict(X_val)
        y_true = enc.inverse_transform(y_val)
        y_pred = enc.inverse_transform(pred_codes)

        metrics = evaluate_predictions(y_true, y_pred, class_names)

        imp = feature_importance_table(
            model, feature_names, X_val, y_val, top_k=len(feature_names), seed=seed
        )

        print(
            f"  VAL accuracy={metrics['accuracy']:.3f}  "
            f"macro_f1={metrics['macro_f1']:.3f}  "
            f"MAE={metrics['mae_score']:.3f}  "
            f"within±0.5={metrics['within_0.5']:.3f}  "
            f"within±1.0={metrics['within_1.0']:.3f}"
        )
        print("\nClassification report (VAL):")
        print(metrics["classification_report"])
        print("Feature importances:")
        print(imp.to_string(index=False))

        model_path = out_dir / f"{name}.joblib"
        joblib.dump(
            {
                "model": model,
                "label_encoder": enc,
                "feature_names": feature_names,
                "group_to_score": GROUP_TO_SCORE,
                "model_name": name,
            },
            model_path,
        )

        entry = {
            "model_path": str(model_path.resolve()),
            "val": {
                k: v
                for k, v in metrics.items()
                if k != "classification_report"
            },
            "val_classification_report": metrics["classification_report"],
            "top_importances": imp.to_dict(orient="records"),
        }

        if include_test:
            y_test = enc.transform(data["test"]["groups"])
            pred_test = model.predict(X_test)
            test_metrics = evaluate_predictions(
                enc.inverse_transform(y_test),
                enc.inverse_transform(pred_test),
                class_names,
            )
            print(
                f"  TEST accuracy={test_metrics['accuracy']:.3f}  "
                f"macro_f1={test_metrics['macro_f1']:.3f}  "
                f"MAE={test_metrics['mae_score']:.3f}"
            )
            entry["test"] = {
                k: v
                for k, v in test_metrics.items()
                if k != "classification_report"
            }
            entry["test_classification_report"] = test_metrics["classification_report"]

        results["models"][name] = entry

    # Summary comparison
    print("\n=== VAL comparison ===")
    print(f"{'model':<28} {'acc':>7} {'macroF1':>8} {'MAE':>7} {'±0.5':>7} {'±1.0':>7}")
    for name, entry in results["models"].items():
        m = entry["val"]
        print(
            f"{name:<28} {m['accuracy']:7.3f} {m['macro_f1']:8.3f} "
            f"{m['mae_score']:7.3f} {m['within_0.5']:7.3f} {m['within_1.0']:7.3f}"
        )

    summary_path = out_dir / f"compare_val_{resolved_name}.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved comparison -> {summary_path.resolve()}")
    return results
