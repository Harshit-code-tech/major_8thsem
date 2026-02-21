# Kaggle Guide — Improved RSNA ICH Pipeline

Step-by-step instructions for running the 6 improved notebooks on Kaggle free tier.

---

## Prerequisites

You must have already run and committed the **baseline notebooks** (NB01–NB07).  
The improved pipeline reuses the **NB02 NPY cache** — no need to reprocess the 458 GB dataset.

---

## Notebook Overview

| Notebook                     | GPU?   | Est. Time          | Inputs                          | Outputs                             |
| ---------------------------- | ------ | ------------------ | ------------------------------- | ----------------------------------- |
| **00 – Preprocess Metadata** | ❌ CPU | ~10 min            | Competition CSV + DICOM headers | `manifest_2_5d.csv`                 |
| **01 – Training**            | ✅ GPU | 3–4 sessions × 12h | NB00 + NB02 cache               | `best_model_fold{k}.pth`, metrics   |
| **02 – Ablation**            | ✅ GPU | ~3h                | NB00 + NB02 cache + NB01 model  | `ablation_results.csv`, chart       |
| **03 – Grad-CAM**            | ✅ GPU | ~30 min            | NB01 model + NB00 + NB02 cache  | heatmap PNGs                        |
| **04 – Calibration**         | ✅ GPU | ~30 min            | NB01 all fold models + NB00     | `calibration_params.json`, OOF CSVs |
| **05 – Report**              | ✅ GPU | ~20 min            | NB04 OOF + NB01 models          | comparison tables, ROC curves       |

---

## Step-by-Step

### Step 1: NB00 — Preprocess Metadata (CPU, ~10 min)

1. Create a new Kaggle notebook
2. Upload `00_preprocess_metadata.ipynb`
3. **Add Data Sources** (click "+ Add Data" in sidebar):
   - `rsna-intracranial-hemorrhage-detection` (the competition dataset)
4. **Accelerator**: None (CPU)
5. Run all cells
6. **Save & Run All** → Commit when done

**Outputs to confirm:**

- `manifest_2_5d.csv` (~5 MB)
- `health_check_nb00.json` (should say PASS)

---

### Step 2: NB01 — Training, Session 1 (GPU, ~12h)

1. Create a new notebook
2. Upload `01_training.ipynb`
3. **Add Data Sources**:
   - Your committed **NB00** output
   - Your committed **NB02** (baseline preprocessing) output
4. **Accelerator**: GPU T4 × 2 (or T4)
5. **In the CONFIG cell**, verify:
   ```python
   PREV_CHECKPOINT_DIR = None  # First session
   ```
6. Update paths if your notebook names differ:
   ```python
   NB00_DIR = Path('/kaggle/input/notebooks/<your-username>/00-preprocess-metadata')
   NB02_DIR = Path('/kaggle/input/notebooks/<your-username>/nb02eda')
   ```
7. Run all cells
8. **Save & Run All** → Commit

**What happens:** Trains fold 0 (and possibly fold 1) for up to 5 epochs each.

---

### Step 3: NB01 — Training, Session 2+ (GPU, ~12h each)

1. **Copy** the NB01 notebook (or create new)
2. **Add Data Sources**:
   - NB00 output
   - NB02 output
   - **Previous NB01 session output** (the committed version from session 1)
3. **Update CONFIG**:
   ```python
   PREV_CHECKPOINT_DIR = '/kaggle/input/notebooks/<your-username>/01-training/'
   ```
4. Run all cells → Commit

**Repeat** until `session_state.json` shows all 5 folds completed.  
Typically **3–4 sessions** total (early stopping may reduce this).

**How to check progress:**  
Look at `session_state.json` in the output. It shows:

```json
{
  "completed_folds": [0, 1, 2],
  "current_fold": 3,
  "current_fold_epoch": 3
}
```

---

### Step 4: NB02 — Ablation (GPU, ~3h)

1. Create new notebook with `02_ablation.ipynb`
2. **Add Data Sources**: NB00 + NB02 + NB01 (final session)
3. Accelerator: GPU
4. Run all → Commit

**Output:** `ablation_results.csv`, `ablation_chart.png`

---

### Step 5: NB03 — Grad-CAM (GPU, ~30 min)

1. Create new notebook with `03_gradcam.ipynb`
2. **Add Data Sources**: NB00 + NB02 + NB01 (final session)
3. Accelerator: GPU
4. Run all → Commit

**Output:** `gradcam_tp_fp_fn_tn.png`, `gradcam_per_subtype.png`, `occlusion_sensitivity.png`

---

### Step 6: NB04 — Calibration (GPU, ~30 min)

1. Create new notebook with `04_calibration.ipynb`
2. **Add Data Sources**: NB00 + NB02 + NB01 (final session)
3. Accelerator: GPU
4. Run all → Commit

**Output:** `calibration_params.json`, `oof_predictions.csv`, reliability diagrams

---

### Step 7: NB05 — Report (GPU, ~20 min)

1. Create new notebook with `05_report.ipynb`
2. **Add Data Sources**:
   - NB00 output
   - NB01 (final session)
   - NB04 (calibration) output
   - Baseline NB03 output (for comparison)
3. Accelerator: GPU
4. Run all → Commit

**Output:** `baseline_vs_improved.csv`, ROC curves, confusion matrices, clinical reports

---

## Path Reference

When adding a committed notebook as a data source, Kaggle mounts it at:

```
/kaggle/input/notebooks/<your-username>/<notebook-slug>/
```

Check the actual mount point by running:

```python
import os
print(os.listdir('/kaggle/input'))
```

Then update the `*_DIR` path variables in each notebook accordingly.

---

## Storage Budget

| Item                        | Size       |
| --------------------------- | ---------- |
| NB02 NPY cache (shared)     | ~4–5 GB    |
| Manifest CSV                | ~5 MB      |
| Per-fold model (~75 MB × 5) | ~375 MB    |
| Checkpoint (temp)           | ~150 MB    |
| Metrics/plots               | ~10 MB     |
| **Total working**           | **< 1 GB** |

Working storage limit on Kaggle: **20 GB** — plenty of headroom.

---

## Troubleshooting

| Problem                             | Fix                                                         |
| ----------------------------------- | ----------------------------------------------------------- |
| `NB00 output not found`             | Check the data source slug matches the path in CONFIG       |
| `OOM (out of memory)`               | Reduce `BATCH_SIZE` to 8, or `ACCUM_STEPS` to 8             |
| `NaN divergence`                    | Reduce `BASE_LR` to 1e-4, or increase `GRAD_CLIP`           |
| `Session timeout`                   | Normal! Commit and resume with `PREV_CHECKPOINT_DIR`        |
| `All folds done but models missing` | Check the final session's output for `best_model_fold*.pth` |

---

## Session Planning

| Session | Duration | GPU? | What runs                              |
| ------- | -------- | ---- | -------------------------------------- |
| 1       | ~10 min  | No   | NB00 preprocessing                     |
| 2       | ~12h     | Yes  | NB01 training (folds 0–1)              |
| 3       | ~12h     | Yes  | NB01 training (folds 2–3)              |
| 4       | ~12h     | Yes  | NB01 training (fold 4 + any remaining) |
| 5       | ~4h      | Yes  | NB02 ablation + NB03 gradcam           |
| 6       | ~1h      | Yes  | NB04 calibration + NB05 report         |

**Total GPU hours: ~40h** (within Kaggle free tier's 30h/week if spread across 2 weeks)
