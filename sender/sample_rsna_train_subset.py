#!/usr/bin/env python3
"""Create a random RSNA train subset and label-safe CSV files.

This script selects random DICOM files from stage_2_train and writes:
1) Copied DICOM files for the subset
2) selected_ids.csv
3) selected_labels_any.csv (binary 0/1 per image)
4) selected_labels_long.csv (all subtype rows for selected IDs)

Example:
python sample_rsna_train_subset.py \
  --dataset-root d:/major8thsem/rsna-intracranial-hemorrhage-detection \
  --sample-size 500 \
  --seed 42 \
  --output-dir d:/major8thsem/rsna_subset_500
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create random RSNA train subset with labels.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("d:/major8thsem/rsna-intracranial-hemorrhage-detection"),
        help="RSNA dataset root containing stage_2_train and stage_2_train.csv",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Number of random train DICOM files to select",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("d:/major8thsem/rsna_subset_500"),
        help="Output directory for sampled data and CSV files",
    )
    return parser.parse_args()


def get_image_id_from_label_id(label_id: str) -> str:
    # stage_2_train.csv stores IDs like ID_xxx_any; keep prefix before last underscore.
    return label_id.rsplit("_", 1)[0]


def list_train_dicoms(train_dir: Path) -> list[Path]:
    return sorted(train_dir.glob("*.dcm"))


def write_selected_ids_csv(selected_dicoms: Iterable[Path], output_csv: Path) -> None:
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "dicom_filename"])
        for dcm_path in selected_dicoms:
            writer.writerow([dcm_path.stem, dcm_path.name])


def copy_selected_dicoms(selected_dicoms: Iterable[Path], destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for dcm_path in selected_dicoms:
        shutil.copy2(dcm_path, destination_dir / dcm_path.name)


def write_label_subsets(label_csv: Path, selected_ids: set[str], out_any_csv: Path, out_long_csv: Path) -> tuple[int, int]:
    any_rows = 0
    long_rows = 0

    with label_csv.open("r", newline="", encoding="utf-8") as src, \
        out_any_csv.open("w", newline="", encoding="utf-8") as any_f, \
        out_long_csv.open("w", newline="", encoding="utf-8") as long_f:

        reader = csv.DictReader(src)
        any_writer = csv.writer(any_f)
        long_writer = csv.writer(long_f)

        any_writer.writerow(["image_id", "label_any"])
        long_writer.writerow(["ID", "Label"])

        for row in reader:
            label_id = row["ID"]
            label_value = row["Label"]
            image_id = get_image_id_from_label_id(label_id)

            if image_id not in selected_ids:
                continue

            long_writer.writerow([label_id, label_value])
            long_rows += 1

            if label_id.endswith("_any"):
                any_writer.writerow([image_id, label_value])
                any_rows += 1

    return any_rows, long_rows


def main() -> None:
    args = parse_args()

    dataset_root = args.dataset_root
    train_dir = dataset_root / "stage_2_train"
    label_csv = dataset_root / "stage_2_train.csv"

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not label_csv.exists():
        raise FileNotFoundError(f"Label CSV not found: {label_csv}")

    all_dicoms = list_train_dicoms(train_dir)
    if not all_dicoms:
        raise RuntimeError(f"No DICOM files found in: {train_dir}")

    sample_size = min(args.sample_size, len(all_dicoms))
    random.seed(args.seed)
    selected_dicoms = random.sample(all_dicoms, sample_size)
    selected_ids = {p.stem for p in selected_dicoms}

    output_dir = args.output_dir
    sampled_train_dir = output_dir / "stage_2_train"
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_ids_csv = output_dir / "selected_ids.csv"
    selected_any_csv = output_dir / "selected_labels_any.csv"
    selected_long_csv = output_dir / "selected_labels_long.csv"

    write_selected_ids_csv(selected_dicoms, selected_ids_csv)
    copy_selected_dicoms(selected_dicoms, sampled_train_dir)
    any_rows, long_rows = write_label_subsets(label_csv, selected_ids, selected_any_csv, selected_long_csv)

    print("Subset creation complete")
    print(f"Dataset root        : {dataset_root}")
    print(f"Requested sample    : {args.sample_size}")
    print(f"Selected sample     : {sample_size}")
    print(f"Seed                : {args.seed}")
    print(f"Copied DICOMs       : {sampled_train_dir}")
    print(f"Selected IDs CSV    : {selected_ids_csv}")
    print(f"Any-label CSV       : {selected_any_csv} (rows={any_rows})")
    print(f"Long-label CSV      : {selected_long_csv} (rows={long_rows})")

    if any_rows != sample_size:
        print(
            "WARNING: number of _any labels does not match sampled IDs. "
            "Please verify label integrity."
        )


if __name__ == "__main__":
    main()
