from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter

from tqdm import tqdm

from segrag.data.coco import get_image_filename


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the exact train/reference images used by an existing raw feature bank."
    )
    parser.add_argument(
        "--dataset-root",
        nargs="+",
        required=True,
        help="One or more dataset roots to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files that already exist under reference_imgs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and report the reference image set without copying files.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def resolve_train_ann_file(dataset_root: str) -> str:
    candidates = [
        os.path.join(dataset_root, "train.json"),
        os.path.join(dataset_root, "train", "lvis_v1_train.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"No supported train annotation JSON found under {dataset_root}")


def load_images_by_id(train_ann_file: str) -> dict[int, dict]:
    with open(train_ann_file, "r") as handle:
        data = json.load(handle)
    return {int(img["id"]): img for img in data.get("images", [])}


def collect_reference_image_ids(raw_feature_bank_dir: str) -> set[int]:
    if not os.path.isdir(raw_feature_bank_dir):
        raise FileNotFoundError(f"Raw feature bank directory not found: {raw_feature_bank_dir}")
    image_ids: set[int] = set()
    class_dirs = sorted(
        entry
        for entry in os.listdir(raw_feature_bank_dir)
        if os.path.isdir(os.path.join(raw_feature_bank_dir, entry)) and not entry.startswith("_")
    )
    for class_name in tqdm(class_dirs, desc="Scan raw bank", leave=False):
        class_dir = os.path.join(raw_feature_bank_dir, class_name)
        for fname in os.listdir(class_dir):
            if not fname.endswith(".pt"):
                continue
            prefix = fname.split("_", 1)[0]
            try:
                image_ids.add(int(prefix))
            except ValueError:
                continue
    return image_ids


def resolve_image_root(dataset_root: str, images_by_id: dict[int, dict], reference_image_ids: set[int]) -> str:
    if not reference_image_ids:
        raise ValueError(f"No reference image ids found for {dataset_root}")
    sample_ids = sorted(reference_image_ids)[: min(10, len(reference_image_ids))]
    sample_names = [get_image_filename(images_by_id[image_id]) for image_id in sample_ids if image_id in images_by_id]
    if not sample_names:
        raise KeyError(f"None of the sampled image ids were present in {dataset_root}")

    candidate_roots = [
        os.path.join(dataset_root, "JPEGImages"),
        os.path.join(dataset_root, "train", "images"),
        os.path.join(dataset_root, "images"),
        os.path.join(dataset_root, "train"),
        os.path.join(dataset_root, "leftImg8bit_trainvaltest", "leftImg8bit", "train"),
        os.path.join(dataset_root, "leftImg8bit", "train"),
    ]
    for candidate in candidate_roots:
        if not os.path.isdir(candidate):
            continue
        if all(_candidate_contains(candidate, sample_name) for sample_name in sample_names[:3]):
            return candidate

    wanted_basenames = set(os.path.basename(name) for name in sample_names)
    basename_hits: dict[str, list[str]] = {basename: [] for basename in wanted_basenames}
    for root, _, files in os.walk(dataset_root):
        matched = wanted_basenames.intersection(files)
        for basename in matched:
            basename_hits[basename].append(os.path.join(root, basename))
        if all(basename_hits[basename] for basename in wanted_basenames):
            break

    resolved_roots = []
    for sample_name in sample_names:
        basename = os.path.basename(sample_name)
        hits = basename_hits.get(basename, [])
        if not hits:
            continue
        found_path = hits[0]
        if os.path.sep in sample_name:
            suffix = sample_name.replace("/", os.sep)
            if found_path.endswith(suffix):
                resolved_roots.append(found_path[: -len(suffix)].rstrip(os.sep))
        else:
            resolved_roots.append(os.path.dirname(found_path))

    if resolved_roots:
        winner, _ = Counter(resolved_roots).most_common(1)[0]
        if os.path.isdir(winner):
            return winner

    raise FileNotFoundError(f"Could not resolve a train image root under {dataset_root}")


def _candidate_contains(candidate_root: str, sample_name: str) -> bool:
    rel_path = sample_name.replace("/", os.sep)
    full_path = os.path.join(candidate_root, rel_path)
    if os.path.isfile(full_path):
        return True
    return os.path.isfile(os.path.join(candidate_root, os.path.basename(sample_name)))


def resolve_source_image_path(image_root: str, image_info: dict) -> tuple[str, str]:
    rel_name = get_image_filename(image_info)
    rel_path = rel_name.replace("/", os.sep)
    direct = os.path.join(image_root, rel_path)
    if os.path.isfile(direct):
        return direct, rel_path
    basename = os.path.basename(rel_path)
    fallback = os.path.join(image_root, basename)
    if os.path.isfile(fallback):
        return fallback, basename
    raise FileNotFoundError(f"Could not resolve source image for {rel_name} under {image_root}")


def export_dataset_reference_images(dataset_root: str, overwrite: bool, dry_run: bool) -> dict:
    train_ann_file = resolve_train_ann_file(dataset_root)
    images_by_id = load_images_by_id(train_ann_file)
    raw_feature_bank_dir = os.path.join(dataset_root, "feature_bank_dinov3_vitl16_1536")
    reference_image_ids = collect_reference_image_ids(raw_feature_bank_dir)
    image_root = resolve_image_root(dataset_root, images_by_id, reference_image_ids)
    output_dir = os.path.join(dataset_root, "reference_imgs")
    manifest_path = os.path.join(dataset_root, "reference_imgs_manifest.json")

    copied = 0
    skipped_existing = 0
    missing_ids: list[int] = []
    missing_files: list[dict] = []
    copied_items = []
    image_ids_sorted = sorted(reference_image_ids)
    for image_id in tqdm(image_ids_sorted, desc=f"Copy {os.path.basename(dataset_root)}", leave=False):
        image_info = images_by_id.get(image_id)
        if image_info is None:
            missing_ids.append(image_id)
            continue
        try:
            src_path, rel_output_path = resolve_source_image_path(image_root, image_info)
        except FileNotFoundError:
            missing_files.append({"image_id": image_id, "image_name": get_image_filename(image_info)})
            continue

        dst_path = os.path.join(output_dir, rel_output_path)
        copied_items.append(
            {
                "image_id": image_id,
                "source_path": src_path,
                "relative_output_path": rel_output_path,
            }
        )
        if dry_run:
            continue
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        if os.path.exists(dst_path) and not overwrite:
            skipped_existing += 1
            continue
        shutil.copy2(src_path, dst_path)
        copied += 1

    manifest = {
        "dataset_root": dataset_root,
        "train_ann_file": train_ann_file,
        "raw_feature_bank_dir": raw_feature_bank_dir,
        "resolved_image_root": image_root,
        "reference_images_total": len(reference_image_ids),
        "copied": copied,
        "skipped_existing": skipped_existing,
        "missing_ids": missing_ids,
        "missing_files": missing_files,
        "items": copied_items,
    }
    if not dry_run:
        with open(manifest_path, "w") as handle:
            json.dump(manifest, handle, indent=2)
    return manifest


def run(args: argparse.Namespace) -> dict:
    datasets = {}
    for dataset_root in args.dataset_root:
        datasets[dataset_root] = export_dataset_reference_images(
            dataset_root=dataset_root,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    return {
        "overwrite": args.overwrite,
        "dry_run": args.dry_run,
        "datasets": datasets,
    }


def main(argv: list[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
