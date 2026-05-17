"""
Infer and validate a raw Pascal-Context-459 -> PC-59 remapping.

This script learns the mapping directly from paired masks:
- raw masks stored as MATLAB files with a `LabelMap` array
- reference PC-59 masks stored as PNGs

It splits overlapping stems into train/holdout sets, infers the mapping on the
train portion, then validates that mapping on the holdout portion.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat


REF_LABEL_CAPACITY = 256


@dataclass(frozen=True)
class Paths:
    dataset_root: Path
    raw_dir: Path
    ref_dir: Path
    raw_labels: Path
    ref_labels: Path
    output_dir: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Infer a raw-459 to PC-59 mask remapping from paired masks."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--raw-dir", default="459_annotations")
    parser.add_argument("--ref-dir", default="59_context_labels")
    parser.add_argument("--raw-labels", default="459_labels.txt")
    parser.add_argument("--ref-labels", default="59_labels.txt")
    parser.add_argument("--output-dir", default="pc59_mapping_analysis")
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260324)
    parser.add_argument("--max-mismatch-examples", type=int, default=50)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _parse_label_file(path: Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        idx_text, name = line.split(":", 1)
        labels[int(idx_text.strip())] = name.strip()
    return labels


def _load_raw_mask(path: Path) -> np.ndarray:
    data = loadmat(path)
    if "LabelMap" not in data:
        raise KeyError(f"Missing LabelMap in {path}")
    return np.asarray(data["LabelMap"], dtype=np.int32)


def _load_ref_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.int32)


def _collect_overlap(raw_dir: Path, ref_dir: Path) -> list[str]:
    raw_stems = {path.stem for path in raw_dir.glob("*.mat")}
    ref_stems = {path.stem for path in ref_dir.glob("*.png")}
    return sorted(raw_stems & ref_stems)


def _split_stems(stems: list[str], holdout_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    if not 0.0 < holdout_ratio < 1.0:
        raise ValueError(f"holdout_ratio must be in (0, 1), got {holdout_ratio}")
    shuffled = list(stems)
    random.Random(seed).shuffle(shuffled)
    holdout_count = max(1, int(round(len(shuffled) * holdout_ratio)))
    holdout = sorted(shuffled[:holdout_count])
    train = sorted(shuffled[holdout_count:])
    return train, holdout


def _accumulate_confusion(
    stems: list[str],
    raw_dir: Path,
    ref_dir: Path,
    raw_label_max: int,
) -> np.ndarray:
    confusion = np.zeros((raw_label_max + 1, REF_LABEL_CAPACITY), dtype=np.int64)
    for stem in stems:
        raw = _load_raw_mask(raw_dir / f"{stem}.mat")
        ref = _load_ref_mask(ref_dir / f"{stem}.png")
        if raw.shape != ref.shape:
            raise ValueError(
                f"Shape mismatch for {stem}: raw={raw.shape}, ref={ref.shape}"
            )
        if raw.max(initial=0) > raw_label_max:
            raise ValueError(f"Observed raw label > configured max in {stem}")
        if ref.max(initial=0) >= REF_LABEL_CAPACITY:
            raise ValueError(f"Observed ref label >= {REF_LABEL_CAPACITY} in {stem}")
        code = raw.ravel() * REF_LABEL_CAPACITY + ref.ravel()
        counts = np.bincount(code, minlength=(raw_label_max + 1) * REF_LABEL_CAPACITY)
        confusion += counts.reshape(raw_label_max + 1, REF_LABEL_CAPACITY)
    return confusion


def _derive_mapping(
    confusion: np.ndarray,
    raw_labels: dict[int, str],
    ref_labels: dict[int, str],
) -> tuple[np.ndarray, list[dict], list[dict]]:
    mapping = np.zeros(confusion.shape[0], dtype=np.uint8)
    rows: list[dict] = []
    low_confidence: list[dict] = []

    for raw_id in range(confusion.shape[0]):
        total = int(confusion[raw_id].sum())
        if total == 0:
            continue

        best_ref = int(np.argmax(confusion[raw_id]))
        best_count = int(confusion[raw_id, best_ref])
        purity = float(best_count / total)
        mapping[raw_id] = best_ref

        top_targets = np.argsort(confusion[raw_id])[-5:][::-1]
        row = {
            "raw_id": raw_id,
            "raw_name": raw_labels.get(raw_id),
            "mapped_ref_id": best_ref,
            "mapped_ref_name": ref_labels.get(best_ref, "background_or_ignore" if best_ref == 0 else None),
            "total_pixels": total,
            "matched_pixels": best_count,
            "purity": purity,
            "top_targets": [
                {
                    "ref_id": int(ref_id),
                    "ref_name": ref_labels.get(int(ref_id), "background_or_ignore" if int(ref_id) == 0 else None),
                    "pixels": int(confusion[raw_id, int(ref_id)]),
                }
                for ref_id in top_targets
                if int(confusion[raw_id, int(ref_id)]) > 0
            ],
        }
        rows.append(row)
        if purity < 0.95:
            low_confidence.append(row)

    rows.sort(key=lambda item: item["raw_id"])
    low_confidence.sort(key=lambda item: (item["purity"], item["raw_id"]))
    return mapping, rows, low_confidence


def _convert_mask(raw: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    converted = np.zeros(raw.shape, dtype=np.uint8)
    valid = raw < mapping.shape[0]
    converted[valid] = mapping[raw[valid]]
    return converted


def _compute_iou(confusion: np.ndarray, class_id: int) -> float | None:
    tp = float(confusion[class_id, class_id])
    fp = float(confusion[:, class_id].sum() - tp)
    fn = float(confusion[class_id, :].sum() - tp)
    denom = tp + fp + fn
    if denom == 0.0:
        return None
    return tp / denom


def _validate_mapping(
    stems: list[str],
    raw_dir: Path,
    ref_dir: Path,
    mapping: np.ndarray,
    max_examples: int,
) -> dict:
    holdout_confusion = np.zeros((REF_LABEL_CAPACITY, REF_LABEL_CAPACITY), dtype=np.int64)
    exact_match_count = 0
    mismatch_examples: list[dict] = []
    unseen_raw_ids: set[int] = set()

    for stem in stems:
        raw = _load_raw_mask(raw_dir / f"{stem}.mat")
        ref = _load_ref_mask(ref_dir / f"{stem}.png")
        converted = _convert_mask(raw, mapping)
        unseen = np.unique(raw[raw >= mapping.shape[0]])
        unseen_raw_ids.update(int(value) for value in unseen)

        if converted.shape != ref.shape:
            raise ValueError(
                f"Shape mismatch while validating {stem}: converted={converted.shape}, ref={ref.shape}"
            )

        exact = np.array_equal(converted, ref)
        if exact:
            exact_match_count += 1
        elif len(mismatch_examples) < max_examples:
            mismatch_examples.append(
                {
                    "stem": stem,
                    "ref_unique": [int(v) for v in np.unique(ref)],
                    "converted_unique": [int(v) for v in np.unique(converted)],
                    "diff_pixels": int(np.count_nonzero(converted != ref)),
                }
            )

        code = ref.ravel() * REF_LABEL_CAPACITY + converted.ravel()
        counts = np.bincount(code, minlength=REF_LABEL_CAPACITY * REF_LABEL_CAPACITY)
        holdout_confusion += counts.reshape(REF_LABEL_CAPACITY, REF_LABEL_CAPACITY)

    total_pixels = int(holdout_confusion.sum())
    correct_pixels = int(np.trace(holdout_confusion))
    pixel_accuracy = float(correct_pixels / total_pixels) if total_pixels else 0.0

    per_class_iou: dict[str, float] = {}
    iou_values: list[float] = []
    for class_id in range(1, 60):
        iou = _compute_iou(holdout_confusion, class_id)
        if iou is None:
            continue
        per_class_iou[str(class_id)] = iou
        iou_values.append(iou)

    return {
        "holdout_images": len(stems),
        "exact_match_images": exact_match_count,
        "exact_match_rate": float(exact_match_count / len(stems)) if stems else 0.0,
        "pixel_accuracy": pixel_accuracy,
        "mean_iou_1_to_59": float(sum(iou_values) / len(iou_values)) if iou_values else 0.0,
        "iou_classes_present": len(iou_values),
        "per_class_iou": per_class_iou,
        "unseen_raw_ids_in_holdout": sorted(unseen_raw_ids),
        "mismatch_examples": mismatch_examples,
    }


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    dataset_root = Path(args.dataset_root).resolve()
    cwd = Path.cwd()
    output_dir = (cwd / args.output_dir).resolve() if not os.path.isabs(args.output_dir) else Path(args.output_dir).resolve()
    paths = Paths(
        dataset_root=dataset_root,
        raw_dir=dataset_root / args.raw_dir,
        ref_dir=dataset_root / args.ref_dir,
        raw_labels=dataset_root / args.raw_labels,
        ref_labels=dataset_root / args.ref_labels,
        output_dir=output_dir,
    )

    raw_labels = _parse_label_file(paths.raw_labels)
    ref_labels = _parse_label_file(paths.ref_labels)
    overlapping_stems = _collect_overlap(paths.raw_dir, paths.ref_dir)
    if not overlapping_stems:
        raise FileNotFoundError("No overlapping raw/ref mask stems found.")

    train_stems, holdout_stems = _split_stems(
        overlapping_stems,
        holdout_ratio=args.holdout_ratio,
        seed=args.seed,
    )
    raw_label_max = max(raw_labels)
    confusion = _accumulate_confusion(
        stems=train_stems,
        raw_dir=paths.raw_dir,
        ref_dir=paths.ref_dir,
        raw_label_max=raw_label_max,
    )
    mapping, mapping_rows, low_confidence = _derive_mapping(
        confusion=confusion,
        raw_labels=raw_labels,
        ref_labels=ref_labels,
    )
    validation = _validate_mapping(
        stems=holdout_stems,
        raw_dir=paths.raw_dir,
        ref_dir=paths.ref_dir,
        mapping=mapping,
        max_examples=args.max_mismatch_examples,
    )

    summary = {
        "dataset_root": str(paths.dataset_root),
        "raw_dir": str(paths.raw_dir),
        "ref_dir": str(paths.ref_dir),
        "output_dir": str(paths.output_dir),
        "seed": args.seed,
        "holdout_ratio": args.holdout_ratio,
        "overlapping_pairs": len(overlapping_stems),
        "train_pairs": len(train_stems),
        "holdout_pairs": len(holdout_stems),
        "mapped_raw_ids": len(mapping_rows),
        "low_confidence_raw_ids": len(low_confidence),
        "validation": validation,
    }

    mapping_json = {
        "mapping": {
            str(row["raw_id"]): row["mapped_ref_id"]
            for row in mapping_rows
        }
    }

    _write_json(paths.output_dir / "summary.json", summary)
    _write_json(paths.output_dir / "mapping.json", mapping_json)
    _write_json(paths.output_dir / "mapping_detailed.json", mapping_rows)
    _write_json(paths.output_dir / "low_confidence_mapping.json", low_confidence)
    return summary


def main(argv: list[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
