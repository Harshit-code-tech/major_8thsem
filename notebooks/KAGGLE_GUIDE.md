# Kaggle Execution Guide
## AI-Assisted Intracranial Hemorrhage Detection

This document is your step-by-step operational guide for running all notebooks
on Kaggle's free tier.  Read it fully before you run anything.

### Key design decisions
- **Patient-level split** (NB02): Uses `GroupShuffleSplit` on DICOM `PatientID` to
  prevent data leakage between train/val sets.
- **Dataset-specific normalization** (NB02→NB03+): Computes actual mean/std on
  cached NPY arrays instead of using ImageNet defaults.
- **NaN divergence guard** (NB03): Training aborts early if loss becomes NaN/Inf.
- **Health-check cells**: Every notebook ends with an automated validation cell
  that saves `health_check_nbXX.json` and raises on failure.

---

## Notebook Execution Order

```
01_eda.ipynb              ← Run first, no GPU needed
02_preprocess_cache.ipynb ← Must commit output (Save & Run All)
                            Now extracts PatientID + computes norm stats
03_train_session.ipynb    ← Run multiple times (session chaining)
04_ablations.ipynb        ← After training is complete
05_gradcam.ipynb          ← Requires final model
06_calibration.ipynb      ← Requires final model (commit output)
07_report.ipynb           ← Requires model + calibration params
```

---

## Step 1 — One-time setup

### 1.1 Accept the competition dataset
Go to:
```
https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/data
```
Click **"I Understand and Accept"** to agree to competition rules.
The dataset is then available at:
```
/kaggle/input/rsna-intracranial-hemorrhage-detection/
```

### 1.2 Create a new Kaggle notebook
- **New Notebook** → enable **GPU** accelerator (Settings → Accelerator → GPU T4 or P100)
- Paste / upload your notebook file

---

## Step 2 — Run Notebook 01 (EDA)

- Add dataset: `rsna-intracranial-hemorrhage-detection`
- Run interactively (no commit needed)
- Review class distribution, windowing effects, DICOM metadata

---

## Step 3 — Run Notebook 02 (Preprocessing Cache)

**This is the most important notebook to get right.**

### What it does (in order)
1. Loads CSV, parses labels
2. **Extracts PatientID** from all DICOM headers (~10-20 min, metadata-only read)
3. **Patient-level train/val split** using `GroupShuffleSplit` (no slice leakage)
4. Converts all DICOMs → 3-channel windowed NPY arrays (parallel, resume-safe)
5. **Computes dataset normalization stats** on cached NPY arrays
6. Saves `manifest.csv` (with `patient_id` column), `normalization_stats.json`
7. Runs health-check validations

### 3.1 Configuration
In the config cell, set:
```python
SUBSET_FRAC = 1.0   # full dataset; use 0.1 to test pipeline first
```

### 3.2 Commit (do not just run interactively)
```
Top right → Save Version → Save & Run All (Background Execution)
```
- This runs in the background.  You can close your browser.
- Check progress at: My Work → Notebooks → Your notebook → Output tab
- Takes ~3-5 hours for full dataset

### 3.3 Find your output name
After the run completes, go to:
```
Notebook → Output tab → note the name (e.g., "ich-preprocess-cache")
```
This name goes into subsequent notebooks as `CACHE_INPUT_DIR`.

### 3.4 Disk budget
- **Pilot** (10%, ~75K images × ~0.19 MB) ≈ **14 GB** — fits under the 19.5 GB commit limit
- **Full** (750K images) ≈ **143 GB** — too large for a single commit; keep `SUBSET_FRAC = 0.1` for this project

---

## Step 4 — Training (Session Chaining)

This is the multi-session workflow. **Do this carefully.**

### Session 1 (first run, no previous checkpoint)

```python
# In 03_train_session.ipynb config cell:
PREV_CHECKPOINT_DIR  = None       # ← first run
N_EPOCHS_THIS_SESSION = 5
CACHE_INPUT_DIR = '/kaggle/input/ich-preprocess-cache'  # ← from Step 3
```

**Add data**: Click "Add Data" → "Notebook Outputs" → search for your NB02 output name.

**Commit**: Save Version → Save & Run All

After completion, note this session's output name, e.g. `ich-train-session-1`.

---

### Session 2 (resume from Session 1)

Create a **new version** of the same notebook (or duplicate it).

```python
# In config cell:
PREV_CHECKPOINT_DIR  = '/kaggle/input/ich-train-session-1'  # ← Session 1 output
N_EPOCHS_THIS_SESSION = 5
```

**Add data**:
1. NB02 cache output (same as before)
2. NB03 Session 1 output (`ich-train-session-1`)

**Commit**: Save Version → Save & Run All

---

### Sessions 3, 4... (repeat until TOTAL_EPOCHS complete)

Each session:
1. Set `PREV_CHECKPOINT_DIR` to the **previous session's** output
2. Add both NB02 cache + previous session output as data
3. Commit

### Example schedule (20 epochs total)

| Session | Epochs trained | `PREV_CHECKPOINT_DIR` |
|---------|---------------|----------------------|
| 1 | 0-4 | `None` |
| 2 | 5-9 | `/kaggle/input/ich-train-session-1` |
| 3 | 10-14 | `/kaggle/input/ich-train-session-2` |
| 4 | 15-19 | `/kaggle/input/ich-train-session-3` |

---

## Step 5 — Ablations (Notebook 04)

After all training sessions are complete:

```python
CACHE_INPUT_DIR = '/kaggle/input/ich-preprocess-cache'
```

Add NB02 cache as data.  Run with GPU.  No commit needed (results saved to working dir).

---

## Step 6 — Grad-CAM (Notebook 05)

```python
MODEL_PATH  = '/kaggle/input/ich-train-session-4/best_model.pth'  # ← last session
CHECKPOINT  = '/kaggle/input/ich-train-session-4/checkpoint.pth'
NPY_CACHE_DIR = '/kaggle/input/ich-preprocess-cache/cache'
```

Add both NB02 cache + final NB03 session as data.

---

## Step 7 — Calibration (Notebook 06)

Same data inputs as NB05.  Commit to save `calibration_params.json` as output.

Note the commit name, e.g. `ich-calibration`.

---

## Step 8 — Report Generator (Notebook 07)

```python
NPY_CACHE_DIR     = '/kaggle/input/ich-preprocess-cache/cache'
MODEL_PATH        = '/kaggle/input/ich-train-session-4/best_model.pth'
CHECKPOINT_PATH   = '/kaggle/input/ich-train-session-4/checkpoint.pth'
CALIB_PARAMS_PATH = '/kaggle/input/ich-calibration/calibration_params.json'
```

Add 3 datasets: NB02 cache, final NB03 session, NB06 calibration output.

---

## Path Reference Card

When you add a notebook output as data, Kaggle mounts it at:
```
/kaggle/input/<notebook-output-name>/
```

The output name is the **slug** of the notebook title (lowercase, hyphens).
You can see the exact path by running:
```python
import os
for d in os.listdir('/kaggle/input'):
    print(d)
```

---

## Optimisation Tips

### Memory
- If OOM: reduce `BATCH_SIZE` from 16 → 8, or enable gradient checkpointing
- EfficientNet-B0 at batch 16, 256×256 uses ~6 GB VRAM (fine on P100/T4)

### Speed
- NB02 uses `NUM_WORKERS=4` for parallel DICOM conversion — this is important
- The `skip-if-exists` logic in NB02 means you can safely re-run if interrupted

### Reproducibility
- Every notebook sets `seed_everything(42)` — same seed → same train/val split
- Never change the seed between sessions

### Session time estimation (rough)
| Notebook | GPU | Time |
|----------|-----|------|
| NB01 EDA | No  | 15 min |
| NB02 Cache (full) | No | 3-5 hours |
| NB03 Train (5 epochs, ~75k train images) | Yes | 3-5 hours |
| NB04 Ablations (4 experiments, 3 epochs) | Yes | 3-4 hours |
| NB05 Grad-CAM | Yes | 30 min |
| NB06 Calibration | Yes | 20 min |
| NB07 Report | Yes | 20 min |

---

## Common Errors

### `FileNotFoundError: .dcm not found`
The RSNA competition dataset path may differ.  Check with:
```python
import os
print(os.listdir('/kaggle/input/rsna-intracranial-hemorrhage-detection'))
```
Update `TRAIN_DIR` in NB02 to match the actual subfolder name.

### `RuntimeError: CUDA out of memory`
Reduce `BATCH_SIZE` to 8 (NB03 config cell).

### `KeyError: 'subtype'` in NB01/NB02
The CSV parse logic splits on the **last** underscore.  If the competition CSV
format has changed, adjust:
```python
raw[['image_id', 'subtype']] = raw['ID'].str.rsplit('_', n=1, expand=True)
```

### NB03 starts from epoch 0 despite checkpoint
Verify `PREV_CHECKPOINT_DIR` is correct and the dataset was added to the notebook.
Run:
```python
import os
print(os.path.exists(f'{PREV_CHECKPOINT_DIR}/checkpoint.pth'))
```

---

## Output Files Per Notebook

| Notebook | Key outputs |
|----------|-------------|
| NB01 | `label_distribution.png`, `cooccurrence.png`, `windowing_comparison.png`, `health_check_nb01.json` |
| NB02 | `cache/*.npy`, `manifest.csv` (with `patient_id`), `normalization_stats.json`, `health_check_nb02.json` |
| NB03 | `checkpoint.pth`, `best_model.pth`, `training_metrics.csv`, `learning_curves.png`, `roc_curve.png`, `health_check_nb03.json` |
| NB04 | `ablation_results.csv`, `ablation_comparison.png`, `health_check_nb04.json` |
| NB05 | `gradcam_tp.png`, `gradcam_fn.png`, `gradcam_fp.png`, `occlusion_sanity_check.png`, `health_check_nb05.json` |
| NB06 | `calibration_params.json`, `isotonic_regressor.pkl`, `reliability_diagrams.png`, `confidence_bands.png`, `health_check_nb06.json` |
| NB07 | `reports/*.json`, `reports/*_gradcam.png`, `report_summary.csv`, `health_check_nb07.json` |

### Health Check Files
Every notebook saves a `health_check_nbXX.json` in `/kaggle/working/`. If any
health check fails, the cell raises a `RuntimeError` — you'll see it in the
notebook output. After a committed run, check the health JSON to confirm success
without scrolling through images.
