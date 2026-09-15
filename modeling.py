"""
modeling.py — train the shipped 9-class XGBoost model and export C++ trees.

Fits on the locked 13 features (`features.FEATURE_NAMES`). Production inference
is C++; this module only trains and writes artifacts/models/export/*.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

from features import FEATURE_NAMES as SELECTED_FEATURES

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_FEATURES = REPO_ROOT / "artifacts" / "features" / "features.npz"
DEFAULT_MODEL_DIR = REPO_ROOT / "artifacts" / "models"
SHIPPED_MODEL_NAME = "xgboost_9_class_13_feat"

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
}


def _score_for_label(label: str) -> float:
    """Map a Haddock class label to its numeric score."""
    key = str(label)
    if key in GROUP_TO_SCORE:
        return GROUP_TO_SCORE[key]
    return float(key)


def _scores_from_labels(labels: np.ndarray) -> np.ndarray:
    """Vectorize `_score_for_label` for ordinal MAE."""
    return np.array([_score_for_label(g) for g in labels], dtype=np.float64)


def load_feature_splits(npz_path: Path | str = DEFAULT_FEATURES) -> dict[str, Any]:
    """Load a consolidated NPZ and split arrays by the embedded `split` column."""
    npz_path = Path(npz_path)
    d = np.load(npz_path, allow_pickle=True)
    split = d["split"].astype(str)
    groups = d["groups"].astype(str)
    X = d["X"].astype(np.float32)
    names = [str(x) for x in d["feature_names"]]
    out: dict[str, Any] = {"feature_names": names}
    for role in ("train", "val", "test"):
        m = split == role
        out[role] = {"X": X[m], "groups": groups[m]}
    return out


def select_feature_columns(
    feature_names: list[str],
    X: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Take NPZ columns in `SELECTED_FEATURES` order."""
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    missing = [n for n in SELECTED_FEATURES if n not in name_to_idx]
    if missing:
        raise KeyError(f"selected features missing from NPZ: {missing}")
    cols = [name_to_idx[n] for n in SELECTED_FEATURES]
    return X[:, cols].astype(np.float32), list(SELECTED_FEATURES)


def build_model(seed: int = 42) -> XGBClassifier:
    """Return the locked XGBClassifier (fresh fit each train; not fine-tuning)."""
    return XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=8,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.3,
        reg_lambda=5.0,
        reg_alpha=1.0,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )


def compute_balanced_sample_weight(
    y_train: np.ndarray,
) -> tuple[np.ndarray, dict[int, float]]:
    """Per-row balanced weights and the class-code → weight map."""
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_to_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    sample_weight = np.array([class_to_weight[int(c)] for c in y_train], dtype=np.float64)
    return sample_weight, class_to_weight


def assert_nine_haddock_classes(data: dict[str, Any]) -> None:
    """Require the NPZ labels match the nine Haddock classes in GROUP_TO_SCORE.

    Train must contain every class so a model named *_9_class_* is honest.
    Val/test may omit rare bins but must not introduce unknown labels.
    """
    expected = set(GROUP_TO_SCORE)
    for role in ("train", "val", "test"):
        labels = {str(g) for g in data[role]["groups"]}
        unknown = labels - expected
        if unknown:
            raise ValueError(
                f"{role} has unknown Haddock labels {sorted(unknown)}; "
                f"expected subset of {sorted(expected)}"
            )
    train_labels = {str(g) for g in data["train"]["groups"]}
    missing = expected - train_labels
    if missing:
        raise ValueError(
            f"train is missing Haddock classes {sorted(missing)}; "
            f"refusing to fit/export a 9-class model"
        )


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict[str, float]:
    """Accuracy, 9-class macro-F1, and ordinal MAE on Haddock scores."""
    true_s = _scores_from_labels(y_true)
    pred_s = _scores_from_labels(y_pred)
    delta = np.abs(true_s - pred_s)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=class_names,
                average="macro",
                zero_division=0,
            )
        ),
        "mae_score": float(np.mean(delta)),
        "within_0.5": float(np.mean(delta <= 0.5)),
        "within_1.0": float(np.mean(delta <= 1.0)),
    }


def train_and_evaluate(
    features_npz: Path | str = DEFAULT_FEATURES,
    out_dir: Path | str = DEFAULT_MODEL_DIR,
    seed: int = 42,
    include_test: bool = False,
    use_balanced_class_weight: bool = True,
    model_name: str = SHIPPED_MODEL_NAME,
) -> Path:
    """Fit XGBoost, write `{model_name}.joblib`, and export C++ JSON trees."""
    data = load_feature_splits(features_npz)
    assert_nine_haddock_classes(data)

    names = data["feature_names"]
    X_train, feature_names = select_feature_columns(names, data["train"]["X"])
    X_val, _ = select_feature_columns(names, data["val"]["X"])
    X_test, _ = select_feature_columns(names, data["test"]["X"])

    # Fit on the canonical nine labels so encoder order matches GROUP_TO_SCORE.
    class_names = list(GROUP_TO_SCORE.keys())
    enc = LabelEncoder()
    enc.fit(class_names)
    y_train = enc.transform(data["train"]["groups"])
    y_val = enc.transform(data["val"]["groups"])

    fit_kwargs: dict[str, Any] = {}
    if use_balanced_class_weight:
        sample_weight, _ = compute_balanced_sample_weight(y_train)
        fit_kwargs["sample_weight"] = sample_weight

    model = build_model(seed=seed)
    model.fit(X_train, y_train, **fit_kwargs)

    y_true = enc.inverse_transform(y_val)
    y_pred = enc.inverse_transform(model.predict(X_val))
    val = evaluate_predictions(y_true, y_pred, class_names)
    print(
        f"VAL acc={val['accuracy']:.3f}  macro_f1={val['macro_f1']:.3f}  "
        f"MAE={val['mae_score']:.3f}  ±0.5={val['within_0.5']:.3f}  "
        f"±1.0={val['within_1.0']:.3f}"
    )

    if include_test:
        y_test = enc.transform(data["test"]["groups"])
        test = evaluate_predictions(
            enc.inverse_transform(y_test),
            enc.inverse_transform(model.predict(X_test)),
            class_names,
        )
        print(
            f"TEST acc={test['accuracy']:.3f}  macro_f1={test['macro_f1']:.3f}  "
            f"MAE={test['mae_score']:.3f}"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{model_name}.joblib"
    joblib.dump(
        {
            "model": model,
            "label_encoder": enc,
            "feature_names": feature_names,
            "group_to_score": GROUP_TO_SCORE,
            "model_name": model_name,
        },
        model_path,
    )
    export_trees(
        model,
        feature_names=feature_names,
        class_names=class_names,
        out_dir=out_dir / "export",
        stem=model_name,
        source_joblib=model_path,
    )
    print(f"saved {model_path}")
    return model_path


def export_trees(
    model: XGBClassifier,
    feature_names: list[str],
    class_names: list[str],
    out_dir: Path,
    stem: str,
    source_joblib: Path,
) -> tuple[Path, Path]:
    """Write C++ booster JSON + sidecar meta next to the joblib."""
    if feature_names != list(SELECTED_FEATURES):
        print(
            "WARNING: export feature_names differ from FEATURE_NAMES.\n"
            f"  export: {feature_names}\n"
            f"  locked: {list(SELECTED_FEATURES)}"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_json = out_dir / f"{stem}.json"
    meta_json = out_dir / f"{stem}.meta.json"
    booster = model.get_booster()
    booster.save_model(str(model_json))

    try:
        source = Path(source_joblib).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        source = str(Path(source_joblib).resolve())

    meta = {
        "source_joblib": source,
        "model_name": stem,
        "booster_format": "xgboost_json",
        "n_features": len(feature_names),
        "feature_names": list(feature_names),
        "n_classes": len(class_names),
        "classes": list(class_names),
        "class_scores": [_score_for_label(c) for c in class_names],
        "objective": booster.attr("objective") or getattr(model, "objective", None),
        "notes": (
            "C++: load .json with XGBoosterLoadModel; feed features in "
            "feature_names order; argmax of multi:softprob -> classes[i] / "
            "class_scores[i] as Haddock score."
        ),
    }
    meta_json.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"exported {model_json}")
    print(f"exported {meta_json}")
    return model_json, meta_json
