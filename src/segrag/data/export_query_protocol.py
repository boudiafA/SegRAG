from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone

from tqdm import tqdm

from segrag.data.coco import get_category_name, get_image_filename


PROTOCOL_VERSION = "query-protocol-v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export fixed query manifests and query protocol files for support-based semantic segmentation benchmarking."
    )
    parser.add_argument(
        "--dataset-root",
        nargs="+",
        required=True,
        help="One or more dataset roots to process.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def load_json(path: str) -> dict:
    with open(path, "r") as handle:
        return json.load(handle)


def dump_json_atomic(path: str, payload: dict) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, path)


def resolve_query_ann_file(dataset_root: str) -> str:
    candidates = [
        os.path.join(dataset_root, "val.json"),
        os.path.join(dataset_root, "test.json"),
        os.path.join(dataset_root, "val", "lvis_v1_val.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"No supported query annotation JSON found under {dataset_root}")


def load_annotations(ann_file: str) -> tuple[dict[int, dict], dict[int, dict], dict[tuple[int, int], list[dict]]]:
    data = load_json(ann_file)
    images = {int(img["id"]): img for img in data.get("images", [])}
    categories = {int(cat["id"]): cat for cat in data.get("categories", [])}
    anns_by_image_and_class: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for ann in data.get("annotations", []):
        anns_by_image_and_class[(int(ann["image_id"]), int(ann["category_id"]))].append(ann)
    return images, categories, dict(anns_by_image_and_class)


def infer_split_name(query_ann_file: str) -> str:
    lower = query_ann_file.lower()
    if "val" in lower:
        return "val"
    if "test" in lower:
        return "test"
    return "eval"


def _candidate_contains(candidate_root: str, sample_name: str) -> bool:
    rel_path = sample_name.replace("/", os.sep)
    full_path = os.path.join(candidate_root, rel_path)
    if os.path.isfile(full_path):
        return True
    return os.path.isfile(os.path.join(candidate_root, os.path.basename(sample_name)))


def resolve_query_image_root(dataset_root: str, images: dict[int, dict]) -> str:
    sample_names = [get_image_filename(img) for _, img in sorted(images.items())[: min(10, len(images))]]
    candidates = [
        os.path.join(dataset_root, "val", "images"),
        os.path.join(dataset_root, "test", "images"),
        os.path.join(dataset_root, "leftImg8bit_trainvaltest", "leftImg8bit", "val"),
        os.path.join(dataset_root, "leftImg8bit", "val"),
        os.path.join(dataset_root, "JPEGImages"),
        dataset_root,
    ]
    for candidate in candidates:
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
        counts = defaultdict(int)
        for root in resolved_roots:
            counts[root] += 1
        winner = max(counts.items(), key=lambda item: item[1])[0]
        if os.path.isdir(winner):
            return winner
    raise FileNotFoundError(f"Could not resolve query image root under {dataset_root}")


def query_rule_for_dataset(dataset_root: str) -> dict:
    name = os.path.basename(dataset_root).lower()
    if "lvis" in name:
        return {
            "mask_type": "semantic_union_binary_from_annotations",
            "description": "Query ground truth for class c is the union of all LVIS instances of class c in the query image.",
        }
    return {
        "mask_type": "semantic_union_binary_from_annotations",
        "description": "Query ground truth for class c is the union of all annotations of class c in the query image.",
    }


def build_query_manifest(dataset_root: str) -> dict:
    query_ann_file = resolve_query_ann_file(dataset_root)
    images, categories, anns_by_image_and_class = load_annotations(query_ann_file)
    query_image_root = resolve_query_image_root(dataset_root, images)
    query_image_root_rel = os.path.relpath(query_image_root, dataset_root).replace(os.sep, "/")
    split = infer_split_name(query_ann_file)

    items = []
    per_class_counts = {}
    for cat_id in tqdm(sorted(categories), desc=f"Build query {os.path.basename(dataset_root)}"):
        class_name = get_category_name(categories, cat_id)
        count = 0
        for image_id in sorted(images):
            anns = anns_by_image_and_class.get((image_id, cat_id), [])
            if not anns:
                continue
            image_info = images[image_id]
            items.append(
                {
                    "dataset": os.path.basename(dataset_root),
                    "split": split,
                    "class_id": cat_id,
                    "class_name": class_name,
                    "image_id": image_id,
                    "image_path": os.path.join(query_image_root_rel, get_image_filename(image_info).replace("/", os.sep)).replace(os.sep, "/"),
                    "source_image_name": get_image_filename(image_info),
                    "ann_ids": [int(ann["id"]) for ann in anns if "id" in ann],
                    "mask_type": "semantic_union_binary_from_annotations",
                }
            )
            count += 1
        per_class_counts[class_name] = count

    query_manifest_path = os.path.join(dataset_root, "query_manifest.json")
    query_protocol_path = os.path.join(dataset_root, "query_protocol.json")
    payload = {"items": items}
    dump_json_atomic(query_manifest_path, payload)

    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "generator": {
            "script": "segrag.data.export_query_protocol",
            "selection_seed": None,
            "selection_mode": "deterministic",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "dataset": os.path.basename(dataset_root),
        "dataset_root": dataset_root,
        "query_ann_file": query_ann_file,
        "query_split": split,
        "query_image_root": query_image_root,
        "query_manifest": query_manifest_path,
        "same_query_set_for_all_shots": True,
        "support_query_exclusion_rule": {
            "support_query_overlap_allowed": False,
            "enforcement": (
                "Support examples come from the train split and queries come from the official evaluation split, "
                "so a query image cannot also be a support image for the same class."
            ),
            "same_image_multiple_classes_allowed_in_support": True,
        },
        "query_semantic_rule": query_rule_for_dataset(dataset_root),
        "per_class_query_counts": [
            {
                "class_id": cat_id,
                "class_name": get_category_name(categories, cat_id),
                "query_count": per_class_counts.get(get_category_name(categories, cat_id), 0),
            }
            for cat_id in sorted(categories)
        ],
        "query_manifest_format": {
            "dataset": "dataset name",
            "split": "fixed evaluation split",
            "class_id": "dataset-specific eval class id",
            "class_name": "class label",
            "image_id": "query image id",
            "image_path": "relative path from dataset root to query image",
            "ann_ids": "annotation ids that should be unioned to form the binary semantic query mask for the class",
            "mask_type": "semantic_union_binary_from_annotations",
        },
    }
    dump_json_atomic(query_protocol_path, protocol)
    return {
        "dataset_root": dataset_root,
        "query_manifest_path": query_manifest_path,
        "query_protocol_path": query_protocol_path,
        "query_examples": len(items),
        "class_count": len(categories),
    }


def run(args: argparse.Namespace) -> dict:
    datasets = {}
    for dataset_root in args.dataset_root:
        datasets[dataset_root] = build_query_manifest(dataset_root)
    return {"datasets": datasets}


def main(argv: list[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
