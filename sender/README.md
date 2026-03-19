# AI-Assisted Intracranial Hemorrhage Screening (B4-Centric Project)

## Overview

This repository contains an end-to-end project for intracranial hemorrhage (ICH) screening from non-contrast head CT DICOM images.

The current final project position is focused on:

- EfficientNet-B4 single-model inference (Fold 4)
- Clinical screening assistance (not autonomous diagnosis)
- Confidence-aware triage outputs
- Explainability support via heatmaps
- Structured report generation

This README is a complete working project guide for now.
A polished final report and image-backed validation chapter will be added after additional outputs are shared.

## Full Model Reports

- reports/B4_Performance_Report.md
- reports/B0_Performance_Report.md

These are the primary long-form report documents for teacher-facing submission.

---

## Objectives

- Build a practical DICOM-to-decision AI pipeline for ICH screening.
- Improve detection quality using 2.5D local slice context.
- Improve confidence reliability using probability calibration.
- Provide explainability and triage recommendations for safer usage.
- Support reproducible baseline-vs-improved model comparison.

---

## Repository Structure

- compare_inference_models.py
  - Main comparison/evaluation runner for B4 and B0 pipelines.

- download_imp/
  - Improved inference pipeline (B4, 2.5D, calibration, report generation).
  - Includes model weights and calibration assets.

- webapp/
  - Baseline inference path (B0-oriented pipeline).

- comparison_runs/
  - Saved outputs from comparison executions.

- report_tables/
  - CSV tables prepared for report charts and tables.

- B4_Model_Project_Report.txt
  - Long-form teacher-facing report draft.

- rsna-intracranial-hemorrhage-detection/
  - RSNA dataset workspace used for labeled comparisons.

---

## Core Pipeline

### 1) DICOM Processing

- Reads CT DICOM slices.
- Preserves ordering and constructs adjacent slice context.

### 2) CT Windowing

- Brain window
- Subdural window
- Soft tissue context window

### 3) 2.5D Input Construction

- Previous + current + next slice context.
- 3 windows per slice -> 9-channel tensor.

### 4) Model Inference

- B4 model path in download_imp with fold selection support.
- Fold4 is used as the final single-model position in this project narrative.

### 5) Calibration

- Isotonic calibration when available.
- Temperature scaling fallback.

### 6) Output Layer

- Probability, threshold decision, confidence band, triage recommendation.
- Explainability artifact path in reports when heatmaps are enabled.

---

## Current Validation Snapshot (500 Labeled Samples)

Source: RSNA-format labels from stage_2_train.csv with image-level any label extraction.

### B4 Fold4 (Final Position)

- ROC-AUC: 0.95974
- PR-AUC: 0.84902
- Brier: 0.05450
- ECE (10-bin): 0.01764
- Sensitivity: 0.87952
- Specificity: 0.90647
- PPV: 0.65179
- NPV: 0.97423
- Runtime (CPU context, 500 samples): 241.93 seconds

### B0 Baseline (Context Comparison)

- ROC-AUC: 0.94067
- PR-AUC: 0.79040
- Brier: 0.13153
- ECE (10-bin): 0.19315
- Runtime (CPU context, 500 samples): 59.56 seconds

Interpretation:

- B4 fold4 improves discrimination and calibration over B0.
- B4 has higher runtime cost.
- Project conclusion remains screening-assistive with mandatory human oversight.

---

## Report-Ready Tables Already Prepared

Use these CSV files directly in Word/Excel charts:

- report_tables/table_main_metrics_b4_vs_b0.csv
- report_tables/table_confusion_components.csv
- report_tables/table_runtime_quality_context.csv

## Dataset Subset Utility (Random 500 Train Samples)

Use this script when you need a smaller train subset with labels preserved:

- sample_rsna_train_subset.py

It creates:

- a sampled stage_2_train folder with copied DICOMs
- selected_ids.csv
- selected_labels_any.csv
- selected_labels_long.csv

---

## How to Run

## 1) Activate Environment (PowerShell)

```powershell
& d:\major8thsem\.venv\Scripts\Activate.ps1
```

## 2) Run Comparison Pipeline

```powershell
python compare_inference_models.py
```

Depending on script defaults/flags, this can run:

- B4 fold4 only
- B4 ensemble only
- both modes (for analysis)

## 3) Check Outputs

Generated artifacts are written in:

- comparison_runs/<run_timestamp>/

Common outputs include:

- comparison_summary.json
- comparison_summary_fold4.json
- comparison_summary_ensemble.json
- comparison_merged.csv
- top10_probability_gaps.csv
- disagreement_cases.csv
- review_priority_cases.csv
- adjudication_template.csv
- weak_labels_consensus.csv

---

## Explainability Notes

- B4 inference path supports Grad-CAM-style overlays.
- Heatmaps can be disabled for speed benchmarking.
- Enable heatmaps when generating qualitative evidence for submission.

---

## Clinical Safety Position

This project is a decision-support tool for screening and triage prioritization.
It is not a diagnostic replacement.

Required policy:

- Human-in-the-loop review is mandatory.
- Low-confidence and positive cases should be prioritized for radiologist review.
- All outputs must be interpreted with clinical context.

---

## Current Limitations

- Labeled validation shown here is sample-based (500) and not external multi-center validation.
- Domain shift across scanners/protocols remains a known risk.
- False negatives can still occur.
- Subtype-level validation details will be expanded in the final report package.

---

## Upcoming Additions (When You Share Final Outputs)

The following sections are ready to be updated quickly:

### A) Validation Evidence Pack

- ROC curves
- PR curves
- Calibration curves
- Confusion matrix plots
- Case-level disagreement analysis highlights

### B) Qualitative Image Section

- Input DICOM slice examples
- Heatmap overlays
- Correct vs incorrect prediction examples
- Confidence band examples

### C) Final Document Integration

- Full polished report chapter links
- Figure numbering and captions
- Final conclusions tied to all output evidence

---

## Inspiration Reference

- He B, Xu Z, Zhou D, Zhang L. Deep multiscale convolutional feature learning for intracranial hemorrhage classification and weakly supervised localization. Heliyon. 2024;10(9):e30270. doi:10.1016/j.heliyon.2024.e30270.

---

## Quick Summary

- Final narrative model: B4 Fold4 single-model.
- B4 currently shows better quality metrics than B0 on the 500 labeled sample.
- Calibration and reporting improvements are major strengths of this implementation.
- Final image-rich validation chapter will be appended as soon as new outputs are shared.
