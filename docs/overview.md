# Determinator 2.0

## Objective

The goal of Determinator is to assess **image quality at the source**—on the aircraft—so that soft or out-of-focus captures can be flagged early and retaken while the flight opportunity still exists, rather than discovering quality problems only after landing or downstream processing.

## Ground truth: Haddock score

To evaluate sharpness, we use a human quality scale: the **Haddock score**. It is Marissa Haddock’s visual rating of image sharpness on a nine-level ordinal scale from **1.0 (sharp) to 5.0 (soft)** in steps of 0.5 — matching the focusM folder layout (`focusM/1.0` … `focusM/5.0`). Lower scores are sharper; higher scores are softer. These labels are the reference that the automated system is trained and measured against.

## Approach

Determinator 2.0 is a **machine-learning pipeline** that predicts Haddock-score quality labels from engineered image-quality (focus / sharpness) features. For each region of interest we compute a fixed numeric feature vector and train a classifier to map those features to a quality bin. The aim is a reliable, maintainable predictor that can support early quality decisions in the capture workflow.

## Background and motivation

The original Determinator computed many classical edge and focus operators on image windows and combined them with hand-designed **voting sets**, because no single metric reliably predicted Haddock ratings across varied scene content. That system was limited by severe class imbalance across ratings, content-dependent metric behavior, and opaque voting rules that were hard to improve systematically.

Determinator 2.0 keeps the same core idea—numeric focus features from regions of interest—but **replaces voting with a tabular classifier**. Features come from a Python port of Chris’s CompLIB focus operators, plus supplemental sharpness, frequency, and quality measures. A gradient-boosted model predicts collapsed Haddock bins. Contiguous ordinal class collapse (locked 5-bin partition) addresses sparse high ratings; balanced class weights counteract uneven sample counts. Improvement then comes from data, feature selection, and model tuning rather than from editing voting tables.
