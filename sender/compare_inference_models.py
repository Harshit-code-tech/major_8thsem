"""
Quick comparison runner for:
- download_imp/run_inference.py (EfficientNet-B4 ensemble)
- webapp/run_interface.py (EfficientNet-B0)

This script does NOT modify either inference file.
It loads each script as a module, overrides input/output paths in memory,
runs both on the same sampled DICOM subset, and writes short comparison outputs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from types import ModuleType

import pandas as pd

try:
    from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score

    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

ROOT = Path(__file__).resolve().parent
DOWNLOAD_SCRIPT = ROOT / "download_imp" / "run_inference.py"
WEBAPP_SCRIPT = ROOT / "webapp" / "run_interface.py"
RSNA_ROOT = ROOT / "rsna-intracranial-hemorrhage-detection"
DEFAULT_DICOM_DIR = RSNA_ROOT / "stage_2_train"
DEFAULT_LABELS_CSV = RSNA_ROOT / "stage_2_train.csv"
DEFAULT_SAMPLE_SIZE = 500
MAX_SAMPLE_SIZE = 500


def load_module(module_name: str, file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def sample_dicoms(
    src_dir: Path,
    sample_size: int,
    seed: int,
    dst_dir: Path,
    allowed_image_ids: set[str] | None = None,
) -> list[Path]:
    all_dicoms = sorted(src_dir.glob("*.dcm"))
    if allowed_image_ids is not None:
        all_dicoms = [p for p in all_dicoms if p.stem in allowed_image_ids]
    if not all_dicoms:
        raise RuntimeError(f"No .dcm files found in {src_dir}")

    k = min(sample_size, len(all_dicoms))
    rng = random.Random(seed)
    chosen = sorted(rng.sample(all_dicoms, k=k), key=lambda p: p.name)

    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in chosen:
        shutil.copy2(p, dst_dir / p.name)

    return chosen


def load_any_label_map(labels_csv: Path) -> dict[str, int]:
    if not labels_csv.exists():
        return {}

    # Format A: image_id, any
    try:
        head = pd.read_csv(labels_csv, nrows=5)
        cols = set(head.columns)
    except Exception:
        return {}

    if {"image_id", "any"}.issubset(cols):
        df = pd.read_csv(labels_csv, usecols=["image_id", "any"])
        df["any"] = pd.to_numeric(df["any"], errors="coerce")
        df = df[df["any"].isin([0, 1])]
        return {str(k): int(v) for k, v in zip(df["image_id"], df["any"])}

    # Format B (RSNA stage_2_train.csv): ID, Label where ID is ID_xxx_subtype
    if {"ID", "Label"}.issubset(cols):
        label_map: dict[str, int] = {}
        for chunk in pd.read_csv(labels_csv, usecols=["ID", "Label"], chunksize=1_000_000):
            parts = chunk["ID"].astype(str).str.rsplit("_", n=1, expand=True)
            if parts.shape[1] != 2:
                continue
            chunk = chunk.copy()
            chunk["image_id"] = parts[0]
            chunk["subtype"] = parts[1]
            any_rows = chunk[chunk["subtype"] == "any"].copy()
            any_rows["Label"] = pd.to_numeric(any_rows["Label"], errors="coerce")
            any_rows = any_rows[any_rows["Label"].isin([0, 1])]
            label_map.update({str(k): int(v) for k, v in zip(any_rows["image_id"], any_rows["Label"])})
        return label_map

    return {}


def run_download_imp(
    sample_dir: Path,
    out_dir: Path,
    b4_fold_selection: str = "ensemble",
    b4_generate_heatmaps: bool = True,
) -> tuple[pd.DataFrame, float]:
    mod = load_module("download_imp_run_inference_cmp", DOWNLOAD_SCRIPT)

    mod.DICOM_INPUT_DIR = sample_dir
    mod.OUTPUT_DIR = out_dir
    mod.FOLD_SELECTION = b4_fold_selection
    if hasattr(mod, "GENERATE_HEATMAPS"):
        mod.GENERATE_HEATMAPS = bool(b4_generate_heatmaps)
    default_manifest = mod.SCRIPT_DIR / "manifest.csv"
    mod.MANIFEST_PATH = default_manifest if default_manifest.exists() else Path("__missing_manifest__.csv")

    t0 = time.time()
    mod.main()
    elapsed = time.time() - t0

    csv_path = out_dir / "slice_predictions.csv"
    if not csv_path.exists():
        raise RuntimeError(f"Expected output missing: {csv_path}")

    df = pd.read_csv(csv_path)
    needed = {"image_id", "cal_any", "pred_any"}
    missing = needed.difference(df.columns)
    if missing:
        raise RuntimeError(f"download_imp output missing columns: {sorted(missing)}")

    cols = ["image_id", "cal_any", "pred_any"]
    if "true_any" in df.columns:
        cols.append("true_any")
    out = df[cols].copy()
    out = out.rename(columns={"cal_any": "prob_b4", "pred_any": "pred_b4"})
    out["pred_b4"] = out["pred_b4"].astype(int)
    if "true_any" in out.columns:
        out = out.rename(columns={"true_any": "label_b4"})
        out["label_b4"] = pd.to_numeric(out["label_b4"], errors="coerce")
    return out, elapsed


def run_webapp(sample_dir: Path, out_dir: Path, b0_generate_heatmaps: bool = True) -> tuple[pd.DataFrame, float]:
    mod = load_module("webapp_run_interface_cmp", WEBAPP_SCRIPT)

    mod.DICOM_INPUT_DIR = sample_dir
    mod.OUTPUT_DIR = out_dir
    if hasattr(mod, "GENERATE_HEATMAPS"):
        mod.GENERATE_HEATMAPS = bool(b0_generate_heatmaps)
    default_manifest = mod.MODEL_DIR / "manifest.csv"
    mod.MANIFEST_PATH = default_manifest if default_manifest.exists() else None

    t0 = time.time()
    mod.main()
    elapsed = time.time() - t0

    csv_path = out_dir / "report_summary.csv"
    if not csv_path.exists():
        raise RuntimeError(f"Expected output missing: {csv_path}")

    df = pd.read_csv(csv_path)
    needed = {"image_id", "cal_prob", "screening_outcome"}
    missing = needed.difference(df.columns)
    if missing:
        raise RuntimeError(f"webapp output missing columns: {sorted(missing)}")

    cols = ["image_id", "cal_prob", "screening_outcome"]
    if "true_label" in df.columns:
        cols.append("true_label")
    out = df[cols].copy()
    out["pred_b0"] = (~out["screening_outcome"].str.contains("No hemorrhage", case=False, na=False)).astype(int)
    out = out.rename(columns={"cal_prob": "prob_b0"})
    if "true_label" in out.columns:
        out = out.rename(columns={"true_label": "label_b0"})
        out["label_b0"] = pd.to_numeric(out["label_b0"], errors="coerce")
        out = out[["image_id", "prob_b0", "pred_b0", "label_b0"]]
    else:
        out = out[["image_id", "prob_b0", "pred_b0"]]
    return out, elapsed


def _binary_ece(labels: pd.Series, probs: pd.Series, n_bins: int = 10) -> float:
    y = labels.to_numpy(dtype=float)
    p = probs.to_numpy(dtype=float)
    edges = pd.Series(pd.interval_range(start=0.0, end=1.0, periods=n_bins))
    ece = 0.0
    n = len(y)
    if n == 0:
        return float("nan")
    for iv in edges:
        lo = float(iv.left)
        hi = float(iv.right)
        if hi >= 1.0:
            m = (p >= lo) & (p <= hi)
        else:
            m = (p >= lo) & (p < hi)
        if not m.any():
            continue
        acc = float(y[m].mean())
        conf = float(p[m].mean())
        ece += float(m.mean()) * abs(acc - conf)
    return float(ece)


def _metrics_at_threshold(labels: pd.Series, probs: pd.Series, threshold: float) -> dict:
    y = labels.to_numpy(dtype=int)
    p = probs.to_numpy(dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),
    }


def build_label_metrics(
    merged: pd.DataFrame,
    b4_threshold: float,
    b0_threshold: float,
    labels_csv: Path | None = None,
) -> dict | None:
    if not HAS_SKLEARN:
        return {"status": "skipped", "reason": "scikit-learn is not available"}

    labels = None

    external_labels_loaded = False
    if labels_csv is not None and labels_csv.exists():
        try:
            label_map = load_any_label_map(labels_csv)
            if label_map:
                labels = merged["image_id"].map(label_map)
                external_labels_loaded = True
        except Exception:
            labels = None

    if (not external_labels_loaded) and "label_b4" in merged.columns and merged["label_b4"].notna().any():
        labels = pd.to_numeric(merged["label_b4"], errors="coerce")
    elif (not external_labels_loaded) and "label_b0" in merged.columns and merged["label_b0"].notna().any():
        labels = pd.to_numeric(merged["label_b0"], errors="coerce")

    if labels is None:
        return {"status": "skipped", "reason": "no labels available in model outputs"}

    eval_df = merged.copy()
    eval_df["label"] = labels
    eval_df = eval_df[eval_df["label"].isin([0, 1])].copy()
    if eval_df.empty:
        return {"status": "skipped", "reason": "labels are present but none are valid binary labels"}

    y = eval_df["label"].astype(int)
    if y.nunique() < 2:
        return {
            "status": "skipped",
            "reason": "only one class present in labeled subset; AUC/PR-AUC undefined",
            "n_labeled": int(len(eval_df)),
            "label_positive_rate": float(y.mean()),
        }

    out = {
        "status": "ok",
        "n_labeled": int(len(eval_df)),
        "label_positive_rate": round(float(y.mean()), 4),
        "b4": {
            "roc_auc": round(float(roc_auc_score(y, eval_df["prob_b4"])), 5),
            "pr_auc": round(float(average_precision_score(y, eval_df["prob_b4"])), 5),
            "brier": round(float(brier_score_loss(y, eval_df["prob_b4"])), 5),
            "ece_10bin": round(float(_binary_ece(y, eval_df["prob_b4"], n_bins=10)), 5),
            "threshold_metrics": _metrics_at_threshold(y, eval_df["prob_b4"], b4_threshold),
        },
        "b0": {
            "roc_auc": round(float(roc_auc_score(y, eval_df["prob_b0"])), 5),
            "pr_auc": round(float(average_precision_score(y, eval_df["prob_b0"])), 5),
            "brier": round(float(brier_score_loss(y, eval_df["prob_b0"])), 5),
            "ece_10bin": round(float(_binary_ece(y, eval_df["prob_b0"], n_bins=10)), 5),
            "threshold_metrics": _metrics_at_threshold(y, eval_df["prob_b0"], b0_threshold),
        },
    }
    return out


def build_summary(merged: pd.DataFrame, t_b4: float, t_b0: float) -> dict:
    if merged.empty:
        raise RuntimeError("No overlapping image_ids found between model outputs")

    summary = {
        "n_compared": int(len(merged)),
        "runtime_seconds": {
            "download_imp_b4": round(t_b4, 2),
            "webapp_b0": round(t_b0, 2),
        },
        "positive_rate": {
            "download_imp_b4": round(float(merged["pred_b4"].mean()), 4),
            "webapp_b0": round(float(merged["pred_b0"].mean()), 4),
        },
        "prediction_agreement": round(float((merged["pred_b4"] == merged["pred_b0"]).mean()), 4),
        "mean_abs_probability_gap": round(float((merged["prob_b4"] - merged["prob_b0"]).abs().mean()), 4),
    }
    return summary


def build_unlabeled_evidence(
    merged: pd.DataFrame,
    b4_threshold: float,
    b0_threshold: float,
    uncertainty_margin: float = 0.05,
    large_gap_threshold: float = 0.30,
) -> tuple[dict, pd.DataFrame]:
    if merged.empty:
        return {"status": "skipped", "reason": "empty merged dataframe"}, merged.copy()

    df = merged.copy()
    df["prob_gap_signed"] = df["prob_b4"] - df["prob_b0"]
    df["prob_gap_abs"] = df["prob_gap_signed"].abs()
    df["pred_disagree"] = df["pred_b4"] != df["pred_b0"]
    df["b4_uncertain"] = (df["prob_b4"] - b4_threshold).abs() <= uncertainty_margin
    df["b0_uncertain"] = (df["prob_b0"] - b0_threshold).abs() <= uncertainty_margin
    df["any_uncertain"] = df["b4_uncertain"] | df["b0_uncertain"]
    df["large_gap"] = df["prob_gap_abs"] >= large_gap_threshold
    df["review_priority"] = df["pred_disagree"] | df["any_uncertain"] | df["large_gap"]
    df["b4_only_positive"] = (df["pred_b4"] == 1) & (df["pred_b0"] == 0)
    df["b0_only_positive"] = (df["pred_b4"] == 0) & (df["pred_b0"] == 1)

    n = len(df)
    n_disagree = int(df["pred_disagree"].sum())
    n_review = int(df["review_priority"].sum())
    n_b4_only = int(df["b4_only_positive"].sum())
    n_b0_only = int(df["b0_only_positive"].sum())

    tendency = "balanced"
    if n_b0_only > n_b4_only:
        tendency = "b0_more_aggressive"
    elif n_b4_only > n_b0_only:
        tendency = "b4_more_aggressive"

    evidence = {
        "status": "ok",
        "thresholds": {
            "b4": round(float(b4_threshold), 6),
            "b0": round(float(b0_threshold), 6),
            "uncertainty_margin": round(float(uncertainty_margin), 4),
            "large_gap_threshold": round(float(large_gap_threshold), 4),
        },
        "counts": {
            "n_compared": int(n),
            "prediction_disagreement": n_disagree,
            "prediction_disagreement_rate": round(float(n_disagree / max(n, 1)), 4),
            "review_priority_cases": n_review,
            "review_priority_rate": round(float(n_review / max(n, 1)), 4),
            "b4_only_positive": n_b4_only,
            "b0_only_positive": n_b0_only,
            "large_probability_gap_cases": int(df["large_gap"].sum()),
            "b4_uncertain_cases": int(df["b4_uncertain"].sum()),
            "b0_uncertain_cases": int(df["b0_uncertain"].sum()),
        },
        "tendency": tendency,
        "probability_shift": {
            "mean_prob_b4": round(float(df["prob_b4"].mean()), 4),
            "mean_prob_b0": round(float(df["prob_b0"].mean()), 4),
            "mean_signed_gap_b4_minus_b0": round(float(df["prob_gap_signed"].mean()), 4),
            "mean_abs_gap": round(float(df["prob_gap_abs"].mean()), 4),
        },
    }

    return evidence, df


def build_adjudication_sheet(evidence_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "image_id",
        "prob_b4",
        "pred_b4",
        "prob_b0",
        "pred_b0",
        "prob_gap_abs",
        "pred_disagree",
        "large_gap",
        "any_uncertain",
    ]
    out = evidence_df[evidence_df["review_priority"]][cols].copy()
    out = out.sort_values(["pred_disagree", "prob_gap_abs"], ascending=[False, False]).reset_index(drop=True)
    out["review_reason"] = out.apply(
        lambda r: "disagreement"
        if bool(r["pred_disagree"])
        else ("large_gap" if bool(r["large_gap"]) else "near_threshold"),
        axis=1,
    )
    out["radiologist_label_any"] = ""
    out["winner_model"] = ""  # expected values: b4, b0, tie
    out["reviewer_name"] = ""
    out["review_notes"] = ""
    return out


def build_weak_labels_sheet(
    evidence_df: pd.DataFrame,
    b4_threshold: float,
    b0_threshold: float,
    min_margin: float = 0.20,
) -> pd.DataFrame:
    df = evidence_df.copy()
    same_pred = df["pred_b4"] == df["pred_b0"]
    b4_margin = (df["prob_b4"] - b4_threshold).abs()
    b0_margin = (df["prob_b0"] - b0_threshold).abs()
    strong = (b4_margin >= min_margin) & (b0_margin >= min_margin)
    pick = df[same_pred & strong].copy()

    out = pd.DataFrame()
    out["image_id"] = pick["image_id"]
    out["weak_label_any"] = pick["pred_b4"].astype(int)
    out["prob_b4"] = pick["prob_b4"]
    out["prob_b0"] = pick["prob_b0"]
    out["min_margin_from_threshold"] = (
        pd.concat([b4_margin.loc[pick.index], b0_margin.loc[pick.index]], axis=1).min(axis=1)
    )
    out["label_type"] = "weak_consensus"
    out["warning"] = "Not ground truth. Use only as provisional label for review."
    out = out.sort_values("min_margin_from_threshold", ascending=False).reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare B4 and B0 inference on a short sampled DICOM set")
    parser.add_argument("--dicom-dir", type=Path, default=DEFAULT_DICOM_DIR, help="Source DICOM folder")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="How many DICOM files to compare")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "comparison_runs", help="Base output directory")
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=DEFAULT_LABELS_CSV,
        help='Labels CSV: either (image_id, any) or RSNA long format (ID, Label)',
    )
    parser.add_argument("--uncertainty-margin", type=float, default=0.05, help="Near-threshold zone for uncertainty review")
    parser.add_argument("--large-gap-threshold", type=float, default=0.30, help="Absolute probability-gap threshold for review")
    parser.add_argument(
        "--b4-fold-selection",
        type=str,
        default="both",
        help='B4 fold selection: "both", "ensemble", or fold id string 0..4',
    )
    parser.add_argument(
        "--disable-heatmaps",
        action="store_true",
        help="Disable Grad-CAM/heatmap generation during comparison to speed up runtime",
    )
    args = parser.parse_args()

    sample_size = max(1, min(int(args.sample_size), MAX_SAMPLE_SIZE))
    if sample_size != int(args.sample_size):
        print(f"Sample size capped to {MAX_SAMPLE_SIZE} (requested {args.sample_size})")

    mode_req = str(args.b4_fold_selection).strip().lower()
    if mode_req == "both":
        b4_modes: list[tuple[str, str]] = [("ensemble", "ensemble"), ("4", "fold4")]
    elif mode_req == "ensemble":
        b4_modes = [("ensemble", "ensemble")]
    elif mode_req.isdigit() and int(mode_req) in [0, 1, 2, 3, 4]:
        b4_modes = [(mode_req, f"fold{mode_req}")]
    else:
        raise ValueError('Invalid --b4-fold-selection. Use "both", "ensemble", or 0..4.')

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.out_dir / f"run_{run_tag}"
    sample_dir = run_root / "sample_dicoms"
    out_b0 = run_root / "webapp_outputs"

    run_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Short Comparison: download_imp(B4) vs webapp(B0)")
    print("=" * 72)
    print(f"B4 fold selection         : {args.b4_fold_selection}")
    print(f"Heatmaps enabled          : {not args.disable_heatmaps}")
    print(f"Sample size (effective)   : {sample_size}")

    label_map = load_any_label_map(args.labels_csv) if args.labels_csv is not None else {}
    if label_map:
        print(f"Loaded labels             : {len(label_map)} image-level labels")
    else:
        print("Loaded labels             : none (label metrics may be skipped)")

    chosen = sample_dicoms(
        args.dicom_dir,
        sample_size,
        args.seed,
        sample_dir,
        allowed_image_ids=set(label_map.keys()) if label_map else None,
    )
    print(f"Sampled {len(chosen)} DICOM files into: {sample_dir}")

    print("\nRunning webapp/run_interface.py ...")
    b0_df, t_b0 = run_webapp(sample_dir, out_b0, b0_generate_heatmaps=not args.disable_heatmaps)

    b4_thresh = 0.5
    try:
        with open(ROOT / "download_imp" / "calibration_params.json", "r", encoding="utf-8") as f:
            b4_cfg = json.load(f)
        b4_thresh = float(b4_cfg.get("threshold_at_spec90", 0.5))
    except Exception:
        pass

    b0_thresh = 0.5
    try:
        with open(ROOT / "webapp" / "modal" / "calibration_params.json", "r", encoding="utf-8") as f:
            b0_cfg = json.load(f)
        b0_thresh = float(b0_cfg.get("calibrated_threshold", 0.5))
    except Exception:
        pass

    all_mode_summaries: dict[str, dict] = {}
    multi_mode = len(b4_modes) > 1

    for fold_selection, mode_tag in b4_modes:
        print(f"\nRunning download_imp/run_inference.py ... (mode={mode_tag})")
        out_b4 = run_root / (f"download_imp_outputs_{mode_tag}" if multi_mode else "download_imp_outputs")
        b4_df, t_b4 = run_download_imp(
            sample_dir,
            out_dir=out_b4,
            b4_fold_selection=fold_selection,
            b4_generate_heatmaps=not args.disable_heatmaps,
        )

        merged = b4_df.merge(b0_df, on="image_id", how="inner")
        merged = merged.sort_values("image_id").reset_index(drop=True)
        merged["abs_prob_gap"] = (merged["prob_b4"] - merged["prob_b0"]).abs()

        summary = build_summary(merged, t_b4, t_b0)
        summary["b4_mode"] = mode_tag

        label_metrics = build_label_metrics(
            merged,
            b4_threshold=b4_thresh,
            b0_threshold=b0_thresh,
            labels_csv=args.labels_csv,
        )
        summary["label_evaluation"] = label_metrics

        unlabeled_evidence, evidence_df = build_unlabeled_evidence(
            merged,
            b4_threshold=b4_thresh,
            b0_threshold=b0_thresh,
            uncertainty_margin=float(args.uncertainty_margin),
            large_gap_threshold=float(args.large_gap_threshold),
        )
        summary["unlabeled_evidence"] = unlabeled_evidence

        suffix = f"_{mode_tag}" if multi_mode else ""
        merged_csv = run_root / f"comparison_merged{suffix}.csv"
        top_gap_csv = run_root / f"top10_probability_gaps{suffix}.csv"
        disagreement_csv = run_root / f"disagreement_cases{suffix}.csv"
        review_priority_csv = run_root / f"review_priority_cases{suffix}.csv"
        adjudication_csv = run_root / f"adjudication_template{suffix}.csv"
        weak_labels_csv = run_root / f"weak_labels_consensus{suffix}.csv"
        summary_json = run_root / f"comparison_summary{suffix}.json"

        merged.to_csv(merged_csv, index=False)
        merged.sort_values("abs_prob_gap", ascending=False).head(10).to_csv(top_gap_csv, index=False)
        evidence_df[evidence_df["pred_disagree"] | evidence_df["large_gap"]].sort_values(
            ["pred_disagree", "prob_gap_abs"], ascending=[False, False]
        ).to_csv(disagreement_csv, index=False)
        evidence_df[evidence_df["review_priority"]].sort_values(
            ["pred_disagree", "any_uncertain", "prob_gap_abs"], ascending=[False, False, False]
        ).to_csv(review_priority_csv, index=False)

        adjudication_df = build_adjudication_sheet(evidence_df)
        adjudication_df.to_csv(adjudication_csv, index=False)

        weak_labels_df = build_weak_labels_sheet(
            evidence_df,
            b4_threshold=b4_thresh,
            b0_threshold=b0_thresh,
            min_margin=0.20,
        )
        weak_labels_df.to_csv(weak_labels_csv, index=False)

        if isinstance(summary.get("unlabeled_evidence"), dict):
            summary["unlabeled_evidence"]["adjudication_rows"] = int(len(adjudication_df))
            summary["unlabeled_evidence"]["weak_consensus_labels"] = int(len(weak_labels_df))

        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        all_mode_summaries[mode_tag] = summary

        print("\n" + "-" * 72)
        print(f"SHORT SUMMARY (mode={mode_tag})")
        print("-" * 72)
        print(f"Compared samples           : {summary['n_compared']}")
        print(f"B4 runtime (sec)          : {summary['runtime_seconds']['download_imp_b4']}")
        print(f"B0 runtime (sec)          : {summary['runtime_seconds']['webapp_b0']}")
        print(f"B4 positive rate          : {summary['positive_rate']['download_imp_b4']}")
        print(f"B0 positive rate          : {summary['positive_rate']['webapp_b0']}")
        print(f"Prediction agreement      : {summary['prediction_agreement']}")
        print(f"Mean |prob_B4 - prob_B0|  : {summary['mean_abs_probability_gap']}")
        if isinstance(label_metrics, dict):
            if label_metrics.get("status") == "ok":
                print("\nLabeled evaluation:")
                print(f"  n_labeled             : {label_metrics['n_labeled']}")
                print(f"  B4 ROC-AUC / PR-AUC   : {label_metrics['b4']['roc_auc']} / {label_metrics['b4']['pr_auc']}")
                print(f"  B0 ROC-AUC / PR-AUC   : {label_metrics['b0']['roc_auc']} / {label_metrics['b0']['pr_auc']}")
                print(f"  B4 Brier / ECE        : {label_metrics['b4']['brier']} / {label_metrics['b4']['ece_10bin']}")
                print(f"  B0 Brier / ECE        : {label_metrics['b0']['brier']} / {label_metrics['b0']['ece_10bin']}")
            else:
                print(f"\nLabeled evaluation skipped: {label_metrics.get('reason', 'unknown reason')}")

        if isinstance(unlabeled_evidence, dict) and unlabeled_evidence.get("status") == "ok":
            c = unlabeled_evidence["counts"]
            print("\nUnlabeled operational evidence:")
            print(f"  Review-priority cases    : {c['review_priority_cases']} ({c['review_priority_rate']})")
            print(f"  Disagreement cases       : {c['prediction_disagreement']} ({c['prediction_disagreement_rate']})")
            print(f"  B4-only positives        : {c['b4_only_positive']}")
            print(f"  B0-only positives        : {c['b0_only_positive']}")
            print(f"  Decision tendency        : {unlabeled_evidence['tendency']}")

        print("\nOutputs:")
        print(f"  Summary JSON            : {summary_json}")
        print(f"  Merged CSV              : {merged_csv}")
        print(f"  Top-10 gaps CSV         : {top_gap_csv}")
        print(f"  Disagreement CSV        : {disagreement_csv}")
        print(f"  Review-priority CSV     : {review_priority_csv}")
        print(f"  Adjudication template   : {adjudication_csv}")
        print(f"  Weak labels CSV         : {weak_labels_csv}")

    if multi_mode:
        combined_summary = {
            "n_compared_requested": int(sample_size),
            "seed": int(args.seed),
            "modes": all_mode_summaries,
        }
        combined_summary_json = run_root / "comparison_summary.json"
        with open(combined_summary_json, "w", encoding="utf-8") as f:
            json.dump(combined_summary, f, indent=2)
        print(f"\nCombined summary JSON      : {combined_summary_json}")


if __name__ == "__main__":
    main()
