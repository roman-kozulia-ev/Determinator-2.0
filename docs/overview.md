# Determinator 2.0

## Objective

Assess **image quality at the source**—on the aircraft—so soft or out-of-focus captures can be flagged and retaken while the flight opportunity still exists, rather than after landing or downstream processing.

## Ground truth

Sharpness labels are **Haddock scores**: a human nine-level ordinal scale from **1.0 (sharp) to 5.0 (soft)** in steps of 0.5, matching the focusM folder layout (`focusM/1.0` … `focusM/5.0`). Lower is sharper. The model is trained and measured against these ratings.

## Architecture

Determinator 2.0 replaces the original hand-designed **voting sets** with a tabular classifier. This repo **trains** that classifier and **exports** it. It does not run on the aircraft.

**Production feature extraction is C++.** Onboard libDeterminator cuts ROIs, computes the 13 features, and scores them with the exported trees. `features.py` is only a Python training port of those operators (same names and order; not bit-identical to C++).

**This repo**

1. Extract the 13 features in Python from labeled focusM ROIs (`features.py`) so the model can be trained.
2. Train XGBoost on those vectors (`features.FEATURE_NAMES` order).
3. Predict one of **nine Haddock classes** (`multi:softprob`).
4. Export booster trees as JSON for the XGBoost C API.

**Onboard C++**

```
full frame
    → cut 5 ROIs (1024×1024; UL, UR, LL, LR, C)
    → 13 features (same names / order as training)
    → load exported JSON trees
    → Haddock score 1.0 … 5.0
```

## Shipped artifacts

The C++ deliverable is the export pair. The joblib is only for Python retrain / inspect.

| File | Role |
|------|------|
| `artifacts/models/export/xgboost_9_class_13_feat.json` | Trees for C++ (`XGBoosterLoadModel`) |
| `artifacts/models/export/xgboost_9_class_13_feat.meta.json` | Feature order, class labels, Haddock scores |
| `artifacts/models/xgboost_9_class_13_feat.joblib` | Python bundle (not used by C++) |

Feature NPZs, the focusM path catalog, and evaluation outputs are local rebuild artifacts and are not in the repo.

## Locked features

Order is the model input layout. Do not reorder without re-exporting the C++ trees.

1. `fft_high_freq_ratio`
2. `Sobel2ndOrder5x5`
3. `grad_mean`
4. `Laplacian3x3`
5. `ThresholdGradient`
6. `Sobel2ndOrder3x3`
7. `roi_std`
8. `Vollath5`
9. `LaplacianOfGaussian`
10. `Sobel2ndOrder3x3Cross`
11. `entropy`
12. `modified_laplacian`
13. `Sobel2ndOrder5x5Cross`

## C++ inference

In production, C++ computes the 13 features (this repo does not). Then load `xgboost_9_class_13_feat.json` with `XGBoosterLoadModel`, pass those features in the order above, take `argmax` of `multi:softprob`, and map that index through `classes` / `class_scores` in the sidecar `.meta.json` to a Haddock score.

## Retrain

Feature caches and the focusM catalog are not shipped. Given a labeled CSV (`path`, `group`, `rating_raw` with the nine Haddock scores):

```bash
python train_model.py --make-splits --csv artifacts/train_sample.csv
python features.py
python features.py --consolidate
python train_model.py --include-test
```
