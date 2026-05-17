from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm

from segrag.data.coco import decode_segmentation_to_mask, get_category_name, get_image_filename


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export support masks and manifests from the exact images used by the raw feature bank."
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
        help="Overwrite existing support masks and manifests.",
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


def resolve_query_ann_file(dataset_root: str) -> str | None:
    candidates = [
        os.path.join(dataset_root, "val.json"),
        os.path.join(dataset_root, "test.json"),
        os.path.join(dataset_root, "val", "lvis_v1_val.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def load_annotations(train_ann_file: str) -> tuple[dict[int, dict], dict[int, dict], dict[tuple[int, int], list[dict]]]:
    with open(train_ann_file, "r") as handle:
        data = json.load(handle)
    images = {int(img["id"]): img for img in data.get("images", [])}
    categories = {int(cat["id"]): cat for cat in data.get("categories", [])}
    anns_by_image_and_class: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for ann in data.get("annotations", []):
        anns_by_image_and_class[(int(ann["image_id"]), int(ann["category_id"]))].append(ann)
    return images, categories, dict(anns_by_image_and_class)


def load_reference_manifest(dataset_root: str) -> dict[int, dict]:
    manifest_path = os.path.join(dataset_root, "reference_imgs_manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"Reference image manifest not found at {manifest_path}. Run export_reference_images.py first."
        )
    with open(manifest_path, "r") as handle:
        manifest = json.load(handle)
    items = manifest.get("items", [])
    return {int(item["image_id"]): item for item in items}


def collect_reference_image_ids_by_class(raw_feature_bank_dir: str, categories: dict[int, dict]) -> dict[int, set[int]]:
    if not os.path.isdir(raw_feature_bank_dir):
        raise FileNotFoundError(f"Raw feature bank directory not found: {raw_feature_bank_dir}")
    category_name_to_id = {get_category_name(categories, cid): cid for cid in categories}
    image_ids_by_class: dict[int, set[int]] = defaultdict(set)
    class_dirs = sorted(
        entry
        for entry in os.listdir(raw_feature_bank_dir)
        if os.path.isdir(os.path.join(raw_feature_bank_dir, entry)) and not entry.startswith("_")
    )
    for class_name in tqdm(class_dirs, desc="Scan raw bank", leave=False):
        cat_id = category_name_to_id.get(class_name)
        if cat_id is None:
            continue
        class_dir = os.path.join(raw_feature_bank_dir, class_name)
        for fname in os.listdir(class_dir):
            if not fname.endswith(".pt"):
                continue
            prefix = fname.split("_", 1)[0]
            try:
                image_ids_by_class[cat_id].add(int(prefix))
            except ValueError:
                continue
    return {cat_id: ids for cat_id, ids in image_ids_by_class.items()}


def dataset_ignore_policy(dataset_root: str) -> dict:
    name = os.path.basename(dataset_root).lower()
    if "ade20k_150" in name:
        return {
            "ignore_label": 0,
            "description": "Background/unlabelled pixels are 0 and ignored. Support masks are binary per class.",
        }
    if "cityscapes" in name:
        return {
            "ignore_label": 0,
            "description": "Exported train.json already uses the 19 eval classes. Background/unmapped pixels are 0 and ignored.",
        }
    if "pc-59" in name or "pc59" in name:
        return {
            "ignore_label": 0,
            "description": "Background pixels are 0 and ignored. Support masks are binary per class.",
        }
    if "lvis" in name:
        return {
            "ignore_label": 0,
            "description": "Support masks are semantic union masks over all LVIS instances of the target class in the support image.",
        }
    return {
        "ignore_label": 0,
        "description": "Background/ignored pixels are represented by 0 in exported binary masks.",
    }


def build_union_mask(height: int, width: int, anns: list[dict]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in anns:
        ann_mask = decode_segmentation_to_mask(ann["segmentation"], height, width)
        mask = np.maximum(mask, ann_mask.astype(np.uint8))
    return mask


def safe_mask_filename(image_info: dict) -> str:
    image_name = get_image_filename(image_info).replace("/", "__")
    stem, _ = os.path.splitext(image_name)
    return f"{stem}.png"


def export_dataset_support_protocol(dataset_root: str, overwrite: bool) -> dict:
    train_ann_file = resolve_train_ann_file(dataset_root)
    query_ann_file = resolve_query_ann_file(dataset_root)
    raw_feature_bank_dir = os.path.join(dataset_root, "feature_bank_dinov3_vitl16_1536")
    images, categories, anns_by_image_and_class = load_annotations(train_ann_file)
    reference_images = load_reference_manifest(dataset_root)
    reference_image_ids_by_class = collect_reference_image_ids_by_class(raw_feature_bank_dir, categories)

    support_root = os.path.join(dataset_root, "reference_masks")
    support_manifest_path = os.path.join(dataset_root, "support_manifest.json")
    support_protocol_path = os.path.join(dataset_root, "support_protocol.json")

    support_items = []
    support_counts = {}
    class_entries = sorted(reference_image_ids_by_class.items(), key=lambda item: get_category_name(categories, item[0]))
    for cat_id, image_ids in tqdm(class_entries, desc=f"Export support {os.path.basename(dataset_root)}"):
        class_name = get_category_name(categories, cat_id)
        class_mask_dir = os.path.join(support_root, class_name)
        os.makedirs(class_mask_dir, exist_ok=True)
        support_count = 0
        for image_id in sorted(image_ids):
            image_info = images.get(image_id)
            if image_info is None:
                continue
            anns = anns_by_image_and_class.get((image_id, cat_id), [])
            if not anns:
                continue
            ref_info = reference_images.get(image_id)
            if ref_info is None:
                continue
            mask = build_union_mask(int(image_info["height"]), int(image_info["width"]), anns)
            rel_mask_path = os.path.join("reference_masks", class_name, safe_mask_filename(image_info))
            abs_mask_path = os.path.join(dataset_root, rel_mask_path)
            if overwrite or not os.path.isfile(abs_mask_path):
                Image.fromarray(mask * 255, mode="L").save(abs_mask_path)

            rel_image_path = os.path.join("reference_imgs", ref_info["relative_output_path"])
            support_items.append(
                {
                    "dataset": os.path.basename(dataset_root),
                    "split": "train",
                    "class_id": cat_id,
                    "class_name": class_name,
                    "image_id": image_id,
                    "image_path": rel_image_path,
                    "mask_path": rel_mask_path,
                    "ann_ids": [int(ann["id"]) for ann in anns if "id" in ann],
                    "mask_type": "semantic_union_binary",
                    "source_image_name": get_image_filename(image_info),
                }
            )
            support_count += 1
        support_counts[class_name] = support_count

    protocol = {
        "dataset": os.path.basename(dataset_root),
        "dataset_root": dataset_root,
        "train_ann_file": train_ann_file,
        "query_ann_file": query_ann_file,
        "raw_feature_bank_dir": raw_feature_bank_dir,
        "support_image_dir": os.path.join(dataset_root, "reference_imgs"),
        "support_mask_dir": support_root,
        "support_manifest": support_manifest_path,
        "class_vocabulary": [
            {
                "class_id": cid,
                "class_name": get_category_name(categories, cid),
                "support_count": support_counts.get(get_category_name(categories, cid), 0),
            }
            for cid in sorted(categories)
        ],
        "ignore_policy": dataset_ignore_policy(dataset_root),
        "metric_guidance": {
            "recommended_metrics": ["per-class IoU", "mIoU", "Dice", "precision", "recall"],
            "notes": "Support masks are binary per class. Query evaluation should follow the dataset's semantic segmentation protocol.",
        },
        "support_manifest_format": {
            "dataset": "dataset name",
            "class_id": "dataset-specific eval class id",
            "class_name": "class label",
            "image_id": "training image id used in the raw bank for this class",
            "image_path": "relative path from dataset root to support RGB image",
            "mask_path": "relative path from dataset root to class-specific binary support mask",
            "split": "train",
            "ann_ids": "annotation ids that were unioned to make the support mask",
        },
    }

    with open(support_manifest_path, "w") as handle:
        json.dump({"items": support_items}, handle, indent=2)
    with open(support_protocol_path, "w") as handle:
        json.dump(protocol, handle, indent=2)

    return {
        "dataset_root": dataset_root,
        "support_examples": len(support_items),
        "support_manifest_path": support_manifest_path,
        "support_protocol_path": support_protocol_path,
        "support_mask_dir": support_root,
        "class_count": len(categories),
    }


def run(args: argparse.Namespace) -> dict:
    datasets = {}
    for dataset_root in args.dataset_root:
        datasets[dataset_root] = export_dataset_support_protocol(dataset_root=dataset_root, overwrite=args.overwrite)
    return {
        "overwrite": args.overwrite,
        "datasets": datasets,
    }


def main(argv: list[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
