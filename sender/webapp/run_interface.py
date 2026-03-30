"""
Standalone Inference Script — ICH Screening from DICOM
======================================================
Reads raw DICOM CT brain slices, applies the trained EfficientNet-B0
screening model with temperature-scaled calibration, and generates:
  • Per-image JSON clinical reports (fixed schema)
  • Grad-CAM overlay PNGs
  • Summary CSV

No command-line arguments — all paths are configured in the CONFIG
section below.

Requirements:
  pip install torch torchvision pydicom opencv-python-headless numpy pandas matplotlib

Usage:
    python run_interface.py
"""

import os
import json
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm

# Try importing pydicom — needed for DICOM input mode
try:
    import pydicom
    import pydicom.multival
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

# ══════════════════════════════════════════════════════════════════════════
#  CONFIG — edit these paths before running
# ══════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / 'modal'

MODEL_PATH          = MODEL_DIR / 'best_model.pth'
CALIB_PARAMS_PATH   = MODEL_DIR / 'calibration_params.json'
NORM_STATS_PATH     = MODEL_DIR / 'normalization_stats.json'

# Input — DICOM folder (set this to the folder containing .dcm files)
DICOM_INPUT_DIR     = SCRIPT_DIR / 'dicom_inputs'

# Manifest — optional, for evaluation with ground truth labels
# Set to None to skip (real-world mode: no labels available)
MANIFEST_PATH       = MODEL_DIR / 'manifest.csv'

# Output
OUTPUT_DIR          = SCRIPT_DIR / 'outputs'

# Model architecture
ARCH       = 'efficientnet_b0'
IMG_SIZE   = 256
SEED       = 42
GENERATE_HEATMAPS = True

# CT windows: (window_center, window_width) — must match training
WINDOWS = [
    (40,   80),    # brain
    (75,  215),    # subdural
    (40,  380),    # soft tissue
]

# ══════════════════════════════════════════════════════════════════════════
#  FIXED VOCABULARY — clinical safety (same as NB07)
# ══════════════════════════════════════════════════════════════════════════

OUTCOME_POSITIVE = 'Hemorrhage indicator detected'
OUTCOME_NEGATIVE = 'No hemorrhage indicator detected'

BAND_LABELS = {
    'HIGH':   'High confidence',
    'MEDIUM': 'Moderate confidence',
    'LOW':    'Low confidence',
}

TRIAGE_ACTIONS = {
    ('POSITIVE', 'HIGH'):   'Urgent radiologist review recommended',
    ('POSITIVE', 'MEDIUM'): 'Prioritised radiologist review recommended',
    ('POSITIVE', 'LOW'):    'Radiologist review recommended — low confidence',
    ('NEGATIVE', 'HIGH'):   'Standard workflow — no urgent action',
    ('NEGATIVE', 'MEDIUM'): 'Standard workflow — manual review if clinically indicated',
    ('NEGATIVE', 'LOW'):    'Manual review recommended — model uncertainty high',
}

DISCLAIMER = (
    'This report is produced by an AI-assisted screening tool and does NOT '
    'constitute a medical diagnosis. All screening findings must be reviewed '
    'and confirmed by a qualified, licensed medical professional before any '
    'clinical decision is made. The system is intended solely as a '
    'decision-support aid in a screening workflow and is not cleared for '
    'standalone diagnostic use.'
)


# ══════════════════════════════════════════════════════════════════════════
#  DICOM PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════

def _to_scalar(val):
    """Safely extract scalar from DICOM field (may be list or MultiValue)."""
    if isinstance(val, (list, pydicom.multival.MultiValue)):
        return float(val[0])
    return float(val)


def apply_window(img_hu: np.ndarray, wc: float, ww: float) -> np.ndarray:
    """Apply a single CT window to HU image → [0, 1]."""
    lo = wc - ww / 2
    hi = wc + ww / 2
    return np.clip((img_hu - lo) / (hi - lo), 0.0, 1.0)


def dicom_to_rgb(dcm_path: str, size: int = 256) -> np.ndarray:
    """
    Read one DICOM file, apply 3 CT windows, stack as (H, W, 3) uint8 [0,255].
    Returns the 3-channel image ready for model input.
    """
    dcm = pydicom.dcmread(str(dcm_path))
    img = dcm.pixel_array.astype(np.float32)

    # Rescale to Hounsfield Units
    slope = _to_scalar(getattr(dcm, 'RescaleSlope', 1))
    inter = _to_scalar(getattr(dcm, 'RescaleIntercept', 0))
    img = img * slope + inter

    channels = []
    for wc, ww in WINDOWS:
        ch = apply_window(img, wc, ww)
        ch = cv2.resize(ch, (size, size), interpolation=cv2.INTER_AREA)
        channels.append(ch)

    img_3ch = (np.stack(channels, axis=-1) * 255).astype(np.uint8)
    return img_3ch


# ══════════════════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════════════════

def build_model(arch: str) -> nn.Module:
    if arch == 'efficientnet_b0':
        m = models.efficientnet_b0(weights=None)
        m.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(m.classifier[1].in_features, 1),
        )
    elif arch == 'resnet50':
        m = models.resnet50(weights=None)
        m.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(m.fc.in_features, 1),
        )
    else:
        raise ValueError(f'Unknown architecture: {arch}')
    return m


# ══════════════════════════════════════════════════════════════════════════
#  GRAD-CAM
# ══════════════════════════════════════════════════════════════════════════

class GradCAM:
    def __init__(self, model, arch):
        self.model = model
        self.activations = None
        self.gradients = None
        target = model.features[-1] if arch == 'efficientnet_b0' else model.layer4[-1]
        self._fh = target.register_forward_hook(
            lambda m, i, o: setattr(self, 'activations', o.detach())
        )
        self._bh = target.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'gradients', go[0].detach())
        )

    def remove(self):
        self._fh.remove()
        self._bh.remove()

    def generate(self, img_tensor):
        self.model.zero_grad()
        t = img_tensor.clone().requires_grad_(True)
        self.model(t).squeeze().backward()
        w = self.gradients.squeeze().mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((w * self.activations.squeeze()).sum(dim=0)).cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def make_overlay(orig_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend Grad-CAM heatmap onto original RGB image."""
    cam_r = cv2.resize(cam, (orig_rgb.shape[1], orig_rgb.shape[0]))
    heat = (mpl_cm.jet(cam_r)[:, :, :3] * 255).astype(np.uint8)
    return (alpha * heat + (1 - alpha) * orig_rgb).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════
#  INFERENCE + REPORT
# ══════════════════════════════════════════════════════════════════════════

def infer_single(img_rgb: np.ndarray, model, grad_cam_obj, transform, device,
                 temperature: float) -> dict:
    """Run forward pass + Grad-CAM on a single preprocessed image."""
    t = transform(img_rgb).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logit = model(t).squeeze().cpu().item()

    raw_prob = float(torch.sigmoid(torch.tensor(logit)).item())
    cal_prob = float(torch.sigmoid(torch.tensor(logit / temperature)).item())

    cam = None
    if grad_cam_obj is not None:
        # Grad-CAM needs gradients
        model.train()
        cam = grad_cam_obj.generate(t)
        model.eval()

    return {'logit': logit, 'raw_prob': raw_prob, 'cal_prob': cal_prob, 'cam': cam}


def build_report(image_id: str, inference: dict, calib_cfg: dict,
                 reports_dir: Path, img_rgb: np.ndarray,
                 true_label=None) -> dict:
    """
    Build a structured screening report with fixed vocabulary.
    Never outputs free-form clinical text.
    """
    cal_prob = inference['cal_prob']
    base_thr = calib_cfg['calibrated_threshold']
    high_thr = calib_cfg['high_threshold']
    low_thr  = calib_cfg['low_threshold']

    # Band assignment
    if cal_prob >= high_thr:
        band = 'HIGH'
    elif cal_prob >= low_thr:
        band = 'MEDIUM'
    else:
        band = 'LOW'

    # Outcome
    is_positive = cal_prob >= base_thr
    outcome_str = OUTCOME_POSITIVE if is_positive else OUTCOME_NEGATIVE
    outcome_key = 'POSITIVE' if is_positive else 'NEGATIVE'
    triage_action = TRIAGE_ACTIONS[(outcome_key, band)]

    cam_save_path = ''
    if inference.get('cam') is not None:
        overlay = make_overlay(img_rgb, inference['cam'])
        cam_save_path = str(reports_dir / f'{image_id}_gradcam.png')
        cv2.imwrite(cam_save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    report = {
        'report_id': f'RPT_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_{image_id[-8:]}',
        'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'image_id': image_id,
        'ground_truth_label': int(true_label) if true_label is not None else 'N/A',
        'screening_module': {
            'version': '1.0',
            'architecture': ARCH,
            'calibration_method': calib_cfg.get('method', 'temperature'),
        },
        'prediction': {
            'screening_outcome': outcome_str,
            'raw_probability': round(inference['raw_prob'], 4),
            'calibrated_probability': round(cal_prob, 4),
            'confidence_band': band,
            'confidence_band_label': BAND_LABELS[band],
            'decision_threshold': round(base_thr, 4),
        },
        'triage': {
            'action': triage_action,
            'urgency': 'URGENT' if (is_positive and band == 'HIGH') else 'STANDARD',
        },
        'explainability': {
            'method': 'Gradient-weighted Class Activation Mapping (Grad-CAM)' if cam_save_path else 'Disabled',
            'heatmap_path': cam_save_path,
            'note': (
                'Highlighted regions indicate areas with greatest influence on the '
                'screening decision. These are not confirmed anatomical findings.'
                if cam_save_path else
                'Heatmap generation disabled for this run.'
            ),
        },
        'disclaimer': DISCLAIMER,
    }
    return report


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 65)
    print('  ICH SCREENING — Standalone Inference')
    print('=' * 65)

    # ── Validate required files ──────────────────────────────────────────
    for name, path in [('Model', MODEL_PATH), ('Calibration', CALIB_PARAMS_PATH)]:
        if not path.exists():
            print(f'ERROR: {name} file not found: {path}')
            return

    if not HAS_PYDICOM:
        print('ERROR: pydicom is not installed. Run: pip install pydicom')
        return

    if not DICOM_INPUT_DIR.exists():
        print(f'ERROR: DICOM input folder not found: {DICOM_INPUT_DIR}')
        print(f'  Create this folder and place .dcm files inside it.')
        return

    # ── Device ───────────────────────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'\n  Device           : {device}')

    # ── Load normalization stats ─────────────────────────────────────────
    if NORM_STATS_PATH.exists():
        with open(NORM_STATS_PATH) as f:
            norm = json.load(f)
        mean = norm['mean']
        std  = norm['std']
        print(f'  Normalization    : dataset stats (mean={mean}, std={std})')
    else:
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        print(f'  Normalization    : ImageNet defaults')

    # ── Load calibration ─────────────────────────────────────────────────
    with open(CALIB_PARAMS_PATH) as f:
        calib_cfg = json.load(f)
    temperature = calib_cfg['temperature']
    print(f'  Calibration      : T={temperature:.4f}')
    print(f'  Threshold        : {calib_cfg["calibrated_threshold"]:.4f}')

    # ── Load model ───────────────────────────────────────────────────────
    model = build_model(ARCH)
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device))
    model = model.to(device)
    model.eval()
    print(f'  Model            : {ARCH} loaded')

    # ── Grad-CAM ─────────────────────────────────────────────────────────
    grad_cam_obj = GradCAM(model, ARCH) if GENERATE_HEATMAPS else None
    transform = T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    # ── Discover DICOM files ─────────────────────────────────────────────
    dcm_files = sorted(DICOM_INPUT_DIR.glob('*.dcm'))
    if not dcm_files:
        print(f'\nERROR: No .dcm files found in {DICOM_INPUT_DIR}')
        return
    print(f'  DICOM files found: {len(dcm_files)}')

    # ── Load manifest (optional, for labels) ─────────────────────────────
    label_map = {}
    if MANIFEST_PATH and Path(MANIFEST_PATH).exists():
        manifest = pd.read_csv(MANIFEST_PATH)
        if 'image_id' in manifest.columns and 'any' in manifest.columns:
            label_map = dict(zip(manifest['image_id'], manifest['any']))
            print(f'  Manifest loaded  : {len(label_map)} entries (labels available)')
        else:
            print(f'  Manifest loaded  : columns missing "image_id"/"any" — ignoring labels')
    else:
        print(f'  Manifest         : not found — running without ground truth')

    # ── Output directory ─────────────────────────────────────────────────
    reports_dir = OUTPUT_DIR / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)
    print(f'  Output directory : {OUTPUT_DIR}')

    # ── Run inference ────────────────────────────────────────────────────
    print(f'\n{"─" * 65}')
    print(f'  Processing {len(dcm_files)} DICOM files...')
    print(f'{"─" * 65}\n')

    all_reports = []
    summary_rows = []

    for i, dcm_path in enumerate(dcm_files, 1):
        image_id = dcm_path.stem  # e.g., ID_abc123def

        # Preprocess DICOM → 3-channel windowed image
        try:
            img_rgb = dicom_to_rgb(str(dcm_path), size=IMG_SIZE)
        except Exception as e:
            print(f'  [{i}/{len(dcm_files)}] SKIP {image_id}: {e}')
            continue

        # Get ground truth label if available
        true_label = label_map.get(image_id, None)

        # Inference
        inf = infer_single(img_rgb, model, grad_cam_obj, transform, device, temperature)

        # Build report
        rep = build_report(image_id, inf, calib_cfg, reports_dir, img_rgb, true_label)
        all_reports.append(rep)

        # Save individual JSON report
        report_path = reports_dir / f'{image_id}_report.json'
        with open(report_path, 'w') as f:
            json.dump(rep, f, indent=2)

        # Summary row
        pred = rep['prediction']
        summary_rows.append({
            'image_id': image_id,
            'true_label': int(true_label) if true_label is not None else '',
            'screening_outcome': pred['screening_outcome'],
            'raw_prob': pred['raw_probability'],
            'cal_prob': pred['calibrated_probability'],
            'confidence_band': pred['confidence_band'],
            'triage_action': rep['triage']['action'],
            'urgency': rep['triage']['urgency'],
        })

        # Progress
        status = '[+] POS' if 'detected' in pred['screening_outcome'] and 'No' not in pred['screening_outcome'] else '[-] NEG'
        band = pred['confidence_band']
        gt_str = f' (GT={int(true_label)})' if true_label is not None else ''
        print(f'  [{i}/{len(dcm_files)}] {image_id}  →  {status}  cal={pred["calibrated_probability"]:.4f}  band={band}{gt_str}')

    # ── Save summary CSV ─────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = OUTPUT_DIR / 'report_summary.csv'
    summary_df.to_csv(summary_csv_path, index=False)

    # ── Cleanup ──────────────────────────────────────────────────────────
    if grad_cam_obj is not None:
        grad_cam_obj.remove()

    # ── Print summary ────────────────────────────────────────────────────
    n_pos = sum(1 for r in all_reports
                if 'detected' in r['prediction']['screening_outcome']
                and 'No' not in r['prediction']['screening_outcome'])
    n_neg = len(all_reports) - n_pos
    n_urgent = sum(1 for r in all_reports if r['triage']['urgency'] == 'URGENT')

    print(f'\n{"═" * 65}')
    print(f'  INFERENCE COMPLETE')
    print(f'{"═" * 65}')
    print(f'  Total processed  : {len(all_reports)}')
    print(f'  Positive (flagged): {n_pos}')
    print(f'  Negative          : {n_neg}')
    print(f'  Urgent escalations: {n_urgent}')
    print(f'\n  Outputs:')
    print(f'    JSON reports    : {reports_dir}/')
    print(f'    Grad-CAM PNGs   : {reports_dir}/')
    print(f'    Summary CSV     : {summary_csv_path}')

    # ── Evaluation metrics (only if labels available) ────────────────────
    if label_map and any(r['ground_truth_label'] != 'N/A' for r in all_reports):
        labeled = [(r['prediction']['calibrated_probability'], r['ground_truth_label'])
                   for r in all_reports if r['ground_truth_label'] != 'N/A']
        if len(set(gt for _, gt in labeled)) > 1:
            from sklearn.metrics import roc_auc_score, confusion_matrix
            probs = [p for p, _ in labeled]
            labels = [int(g) for _, g in labeled]
            preds = [1 if p >= calib_cfg['calibrated_threshold'] else 0 for p in probs]

            auc = roc_auc_score(labels, probs)
            cm = confusion_matrix(labels, preds)
            tp, fn, fp, tn = cm[1, 1], cm[1, 0], cm[0, 1], cm[0, 0]
            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            ppv  = tp / max(tp + fp, 1)

            print(f'\n  Evaluation (labeled subset: {len(labeled)} images):')
            print(f'    AUC             : {auc:.5f}')
            print(f'    Sensitivity     : {sens:.4f}')
            print(f'    Specificity     : {spec:.4f}')
            print(f'    PPV             : {ppv:.4f}')
            print(f'    TP={tp}  FP={fp}  FN={fn}  TN={tn}')

    print(f'\n{"═" * 65}')
    print(f'  Done.')
    print(f'{"═" * 65}')


if __name__ == '__main__':
    main()
