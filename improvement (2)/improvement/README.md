# Improved RSNA Intracranial Hemorrhage Detection Pipeline

An upgraded version of the baseline NB01–NB07 pipeline, implementing all reviewer-requested improvements while maintaining the same modular separation pattern.

## What Changed vs. Baseline

| Feature        | Baseline (NB01–NB07)   | Improved (NB00–NB05)                              |
| -------------- | ---------------------- | ------------------------------------------------- |
| Architecture   | EfficientNet-B0 (5.3M) | EfficientNet-B4 (19M)                             |
| Input          | 2D, 3 channels         | **2.5D** (prev+center+next), 9 channels           |
| Formulation    | Binary (any vs none)   | **Multi-label** (any + 5 subtypes)                |
| Validation     | Single train/val split | **5-fold GroupKFold** (no patient leakage)        |
| Training       | Standard BCE           | Weighted BCE + label smoothing + **MixUp/CutMix** |
| Hard examples  | —                      | **Hard negative mining** (top-500, 3× boost)      |
| Calibration    | —                      | **Temperature scaling + isotonic regression**     |
| Aggregation    | —                      | **Patient-level** (max, mean, noisy-or, top-k)    |
| Explainability | Grad-CAM (2D)          | Grad-CAM (**2.5D**, per-subtype)                  |

## Notebook Structure

```
improvement/
├── 00_preprocess_metadata.ipynb   CPU     2.5D adjacency + GroupKFold splits
├── 01_training.ipynb              GPU     Full training with session chaining
├── 02_ablation.ipynb              GPU     5 controlled experiments
├── 03_gradcam.ipynb               GPU     TP/FP/FN/TN heatmaps + occlusion
├── 04_calibration.ipynb           GPU     ECE + temp scaling + isotonic + triage
├── 05_report.ipynb                GPU     Comparison, patient aggregation, ROCs
├── kaggle_guide.md                        Step-by-step Kaggle instructions
└── README.md                              This file
```

### NB00 — Preprocess Metadata (CPU)

- Parses RSNA competition CSV labels
- Reads DICOM headers (position/series) to establish slice ordering
- Builds 2.5D adjacency (prev/next slice per image)
- Creates 5-fold GroupKFold splits (grouped by patient)
- **Output**: `manifest_2_5d.csv`

### NB01 — Training (GPU, multi-session)

- **Session chaining** with `PREV_CHECKPOINT_DIR` (same pattern as baseline NB03)
- Full **lock system**: checkpoint save/restore (model + optimizer + scheduler + scaler + patience + early_stopped flag)
- **NaN divergence guard**: aborts after 5 consecutive NaN losses
- **Early stopping** with patience=5 on validation AUC
- **Discriminative LR**: backbone at 0.1× the head learning rate
- **Backbone freezing**: first 2 epochs train head only (warmup)
- 2.5D dataset loading from existing NB02 NPY cache
- **Output**: `best_model_fold{k}.pth`, `training_metrics.csv`, `session_state.json`

### NB02 — Ablation (GPU)

- 5 experiments on 15% subset, 3 epochs each:
  1. 2D vs 2.5D
  2. B0 vs B4
  3. Binary vs Multi-label
  4. Minimal vs Full augmentation
  5. Hard neg mining ON/OFF
- **Output**: `ablation_results.csv`, bar chart

### NB03 — Grad-CAM (GPU)

- Grad-CAM heatmaps for TP / FP / FN / TN categories
- Per-subtype activation maps
- Occlusion sensitivity analysis
- **Output**: PNG visualizations

### NB04 — Calibration (GPU)

- Collects out-of-fold predictions from all 5 folds
- Expected Calibration Error (ECE) computation
- Temperature scaling (parametric)
- Isotonic regression (non-parametric)
- Reliability diagrams (before/after)
- 3-band triage: HIGH ≥0.7, MEDIUM 0.3–0.7, LOW <0.3
- **Output**: `calibration_params.json`, `oof_predictions.csv`

### NB05 — Report (GPU)

- Head-to-head comparison: baseline vs improved
- Per-subtype ROC curves
- Patient-level aggregation comparison (max, mean, noisy-or, top-k)
- Clinical case reports with calibrated confidence
- Confusion matrix
- Deployment recommendations
- **Output**: CSV tables, ROC plots, clinical reports

## Quick Start

See [kaggle_guide.md](kaggle_guide.md) for detailed Kaggle instructions.

**TL;DR:**

1. Run NB00 (CPU, 10 min)
2. Run NB01 across 3–4 GPU sessions (update `PREV_CHECKPOINT_DIR` each time)
3. Run NB02–NB05 (1 GPU session each)

## Key Design Decisions

- **Reuses NB02 cache**: The 2.5D approach stacks existing 3-channel NPY files. No need to reprocess the 458 GB DICOM dataset.
- **Session chaining**: Matches the exact pattern from the baseline NB03, with full checkpoint save/restore.
- **GroupKFold**: Groups by `patient_id` to prevent data leakage across folds.
- **Multi-label**: Training on all 6 targets simultaneously provides auxiliary gradient signal that improves `any` prediction.
