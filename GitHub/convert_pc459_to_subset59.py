"""
Convert Pascal-Context-459 MATLAB masks into a custom 59-class PNG subset.

Conversion rule:
- keep only the classes listed in 59_labels.txt
- remap them to contiguous ids 1..59 based on their order in 59_labels.txt
- map every other raw class to 0
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert Pascal-Context-459 .mat masks into a subset-59 PNG mask set."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--raw-dir", default="459_annotations")
    parser.add_argument("--raw-labels", default="459_labels.txt")
    parser.add_argument("--keep-labels", default="59_labels.txt")
    parser.add_argument("--output-dir", default="subset59_from_459")
    parser.add_argument("--summary-name", default="conversion_summary.json")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _parse_label_file(path: Path) -> list[tuple[int, str]]:
    items: list[tuple[int, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^(\d+)\s*[:\-]?\s*(.+)$", line)
        if not match:
            raise ValueError(f"Could not parse label line in {path}: {line!r}")
        items.append((int(match.group(1)), match.group(2).strip()))
    return items


def _load_raw_mask(path: Path) -> np.ndarray:
    data = loadmat(path)
    if "LabelMap" not in data:
        raise KeyError(f"Missing LabelMap in {path}")
    return np.asarray(data["LabelMap"], dtype=np.int32)


def _build_mapping(
    raw_labels: list[tuple[int, str]],
    keep_labels: list[tuple[int, str]],
) -> tuple[np.ndarray, dict[str, int], list[dict[str, int | str]]]:
    raw_dict = dict(raw_labels)
    keep_name_to_id = {
        name: new_id for new_id, (_, name) in enumerate(keep_labels, start=1)
    }
    raw_label_max = max(raw_dict)
    mapping = np.zeros(raw_label_max + 1, dtype=np.uint8)
    mapped_rows: list[dict[str, int | str]] = []

    for raw_id, raw_name in raw_labels:
        new_id = keep_name_to_id.get(raw_name, 0)
        mapping[raw_id] = new_id
        mapped_rows.append(
            {
                "raw_id": raw_id,
                "raw_name": raw_name,
                "subset59_id": int(new_id),
            }
        )

    return mapping, keep_name_to_id, mapped_rows


def _convert_mask(raw: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    converted = np.zeros(raw.shape, dtype=np.uint8)
    valid = (raw >= 0) & (raw < mapping.shape[0])
    converted[valid] = mapping[raw[valid]]
    return converted


def run(args: argparse.Namespace) -> dict:
    dataset_root = Path(args.dataset_root).resolve()
    raw_dir = dataset_root / args.raw_dir
    raw_labels_path = dataset_root / args.raw_labels
    keep_labels_path = dataset_root / args.keep_labels
    output_dir = dataset_root / args.output_dir

    raw_labels = _parse_label_file(raw_labels_path)
    keep_labels = _parse_label_file(keep_labels_path)
    mapping, keep_name_to_id, mapped_rows = _build_mapping(raw_labels, keep_labels)

    if len(keep_name_to_id) != len(keep_labels):
        raise ValueError("Duplicate class names found in keep-label list.")

    raw_paths = sorted(raw_dir.glob("*.mat"))
    if not raw_paths:
        raise FileNotFoundError(f"No .mat files found in {raw_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    overwritten = 0
    max_output_label = 0
    nonzero_pixels = 0
    total_pixels = 0

    for raw_path in raw_paths:
        out_path = output_dir / f"{raw_path.stem}.png"
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"{out_path} already exists. Re-run with --overwrite to replace it."
            )

        raw_mask = _load_raw_mask(raw_path)
        converted = _convert_mask(raw_mask, mapping)

        if out_path.exists():
            overwritten += 1
        Image.fromarray(converted, mode="L").save(out_path)
        written += 1

        max_output_label = max(max_output_label, int(converted.max(initial=0)))
        nonzero_pixels += int(np.count_nonzero(converted))
        total_pixels += int(converted.size)

    summary = {
        "dataset_root": str(dataset_root),
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "raw_label_count": len(raw_labels),
        "kept_class_count": len(keep_labels),
        "kept_classes": [
            {"subset59_id": new_id, "name": name}
            for new_id, (_, name) in enumerate(keep_labels, start=1)
        ],
        "converted_masks": written,
        "overwritten_masks": overwritten,
        "max_output_label": max_output_label,
        "nonzero_pixel_fraction": float(nonzero_pixels / total_pixels) if total_pixels else 0.0,
        "mapping_preview": mapped_rows[: min(len(mapped_rows), 80)],
        "rule": {
            "kept_labels_source": str(keep_labels_path),
            "dropped_labels_target": 0,
            "kept_labels_remapped_to": "1..59 in 59_labels.txt order",
        },
    }

    (output_dir / args.summary_name).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (output_dir / "mapping.json").write_text(
        json.dumps(
            {
                "mapping": {
                    str(row["raw_id"]): row["subset59_id"] for row in mapped_rows
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
