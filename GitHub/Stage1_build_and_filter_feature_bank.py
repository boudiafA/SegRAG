"""
Stage 1: build a raw DINOv3 feature bank, then filter it intra-class.

This stage merges the old Stage 1a and 1b entrypoints so the public workflow
has a single bank-preparation command.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2
from tqdm import tqdm

from GitHub.build_feature_bank_utils import (
    FeatureBankPaths,
    _count_existing_features_per_class,
    _iter_batches,
    _merge_resume_state,
    _rebuild_resume_state_from_disk,
    decode_segmentation_to_mask,
    extract_dense_features_batched,
    get_category_name,
    get_image_filename,
    load_dinov3_model,
    load_lvis_annotations,
    resize_transform,
)
from GitHub.intra_class_filter_impl import run_filter
from GitHub.resume import load_json, save_json_atomic


DEFAULT_DATASET_ROOT = "."
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
_RESIZE_TRANSFORMS: dict[int, v2.Compose] = {}


def _parse_optional_threshold(value: str) -> float | None:
    lowered = value.strip().lower()
    if lowered in {"none", "null"}:
        return None
    return float(value)


def _parse_optional_int(value: str) -> int | None:
    lowered = value.strip().lower()
    if lowered in {"none", "null"}:
        return None
    return int(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 1: build a raw feature bank from train.json, then filter it intra-class."
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--train-ann-file", default=None, help="COCO-style training annotation JSON.")
    parser.add_argument("--image-dir", default=None, help="Root directory used to resolve image paths.")
    parser.add_argument("--raw-feature-bank-dir", default=None, help="Stage 1a output directory.")
    parser.add_argument("--filtered-feature-bank-dir", default=None, help="Stage 1b output directory.")
    parser.add_argument("--skip-build", action="store_true", help="Skip raw feature-bank creation and only run filtering.")
    parser.add_argument("--skip-filter", action="store_true", help="Skip intra-class filtering after the raw bank is ready.")
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--image-size", type=int, default=1536)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--model-name", default="dinov3_vitl16")
    parser.add_argument("--repo-path", default="./")
    parser.add_argument("--weights-path", default="./weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    parser.add_argument("--mask-coverage-threshold", type=float, default=0.90)
    parser.add_argument("--features-per-class-threshold", type=int, default=None)
    parser.add_argument("--max-images-per-class", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--scan-workers", type=int, default=8)
    parser.add_argument("--checkpoint-name", default="_build_feature_bank_resume.json")

    parser.add_argument("--selection-mode", default="top-k-images")
    parser.add_argument("--max-source-images", type=int, default=100)
    parser.add_argument(
        "--top-k-features",
        type=_parse_optional_int,
        default=10000,
        help="Maximum kept features per class after filtering. Use `none` to disable the cap.",
    )
    parser.add_argument(
        "--keep-threshold",
        type=_parse_optional_threshold,
        default=0.8,
        help="Filtering threshold for Stage 1b. Use `none` to keep all features and scores.",
    )
    parser.add_argument("--min-matches", type=int, default=3)
    parser.add_argument("--min-keep-ratio", type=float, default=0.30)
    parser.add_argument("--filter-mode", default="hard", choices=["hard", "reweight"])
    parser.add_argument("--target-image-limit", type=int, default=100)
    parser.add_argument("--target-references", type=int, default=None)
    parser.add_argument("--query-chunk", type=int, default=256)
    parser.add_argument("--target-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--sim-floor", type=float, default=0.0)
    parser.add_argument("--early-accept", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _resolve_stage1_paths(args: argparse.Namespace) -> argparse.Namespace:
    args.train_ann_file = args.train_ann_file or os.path.join(args.dataset_root, "train.json")
    args.image_dir = args.image_dir or args.dataset_root
    args.raw_feature_bank_dir = args.raw_feature_bank_dir or os.path.join(
        args.dataset_root,
        "feature_bank_dinov3_vitl16_1536",
    )
    if hasattr(args, "filtered_feature_bank_dir"):
        args.filtered_feature_bank_dir = args.filtered_feature_bank_dir or os.path.join(
            args.dataset_root,
            "feature_bank_dinov3_vitl16_intra_class_filtered_1536",
        )
    return args


def _make_paths(args: argparse.Namespace) -> FeatureBankPaths:
    return FeatureBankPaths(
        train_root=args.image_dir,
        val_root=args.image_dir,
        image_size=args.image_size,
        patch_size=args.patch_size,
        model_name=args.model_name,
        repo_path=args.repo_path,
        weights_path=args.weights_path,
        mask_coverage_threshold=args.mask_coverage_threshold,
    )


def _resize_image_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    transform = _RESIZE_TRANSFORMS.get(image_size)
    if transform is None:
        transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((image_size, image_size), interpolation=v2.InterpolationMode.BICUBIC),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        )
        _RESIZE_TRANSFORMS[image_size] = transform
    return transform(image)


def _load_batch_record(job: tuple[int, dict, str, int]) -> dict[str, object]:
    image_id, img_info, img_path, image_size = job
    try:
        with Image.open(img_path) as image:
            rgb = image.convert("RGB")
            img_tensor = _resize_image_tensor(rgb, image_size)
        return {"image_id": image_id, "img_info": img_info, "img_tensor": img_tensor, "ok": True}
    except Exception:
        return {"image_id": image_id, "img_info": img_info, "img_tensor": None, "ok": False}


def _scan_image_ids_per_class(output_dir: str, categories: dict) -> dict[str, set[int]]:
    image_ids_per_class: dict[str, set[int]] = {}
    for cat_id in categories:
        cat_name = get_category_name(categories, cat_id)
        cat_dir = os.path.join(output_dir, cat_name)
        image_ids = set()
        if os.path.isdir(cat_dir):
            for fname in os.listdir(cat_dir):
                if not fname.endswith(".pt"):
                    continue
                stem = os.path.splitext(fname)[0]
                try:
                    image_id = int(stem.split("_", 1)[0])
                except Exception:
                    continue
                image_ids.add(image_id)
        image_ids_per_class[cat_name] = image_ids
    return image_ids_per_class


def run_build(args: argparse.Namespace) -> dict:
    os.makedirs(args.raw_feature_bank_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = _make_paths(args)
    images_dict, anns_by_image, categories = load_lvis_annotations(args.train_ann_file)
    for cat_id in categories:
        os.makedirs(os.path.join(args.raw_feature_bank_dir, get_category_name(categories, cat_id)), exist_ok=True)

    checkpoint_path = os.path.join(args.raw_feature_bank_dir, args.checkpoint_name)
    state = load_json(
        checkpoint_path,
        {
            "processed_image_ids": [],
            "saved_files": 0,
            "features_per_class": {},
            "images_per_class": {},
            "started_at": time.time(),
        },
    ) if args.resume else {
        "processed_image_ids": [],
        "saved_files": 0,
        "features_per_class": {},
        "images_per_class": {},
        "started_at": time.time(),
    }
    scanned_state = _rebuild_resume_state_from_disk(
        args.raw_feature_bank_dir,
        anns_by_image,
        categories,
        scan_workers=args.scan_workers,
    )
    state = _merge_resume_state(state, scanned_state)
    save_json_atomic(checkpoint_path, state)

    processed_image_ids = set(state.get("processed_image_ids", []))
    use_threshold = args.features_per_class_threshold is not None
    if use_threshold:
        features_per_class = defaultdict(
            int,
            state.get("features_per_class") or _count_existing_features_per_class(args.raw_feature_bank_dir, categories),
        )
    else:
        features_per_class = defaultdict(int)
    image_ids_per_class = _scan_image_ids_per_class(args.raw_feature_bank_dir, categories)

    patch_quant_filter = torch.nn.Conv2d(
        1,
        1,
        args.patch_size,
        stride=args.patch_size,
        bias=False,
    )
    patch_quant_filter.weight.data.fill_(1.0 / (args.patch_size * args.patch_size))
    patch_quant_filter = patch_quant_filter.to(device)

    model = load_dinov3_model(paths, device)
    image_ids = sorted(anns_by_image.keys())
    total_saved = int(state.get("saved_files", 0))
    pending_image_ids = []
    for image_id in image_ids:
        if image_id in processed_image_ids:
            continue
        if use_threshold and all(
            features_per_class.get(get_category_name(categories, cid), 0) >= args.features_per_class_threshold
            for cid in categories
        ):
            break
        if args.max_images_per_class is not None and all(
            len(image_ids_per_class.get(get_category_name(categories, cid), set())) >= args.max_images_per_class
            for cid in categories
        ):
            break
        pending_image_ids.append(image_id)

    batch_size = max(1, args.batch_size)
    total_batches = (len(pending_image_ids) + batch_size - 1) // batch_size
    for batch_index, image_batch in enumerate(
        tqdm(_iter_batches(pending_image_ids, batch_size), total=total_batches, desc="Build raw bank"),
        start=1,
    ):
        batch_records = []
        load_jobs = []
        for image_id in image_batch:
            img_info = images_dict[image_id]
            img_path = os.path.join(args.image_dir, get_image_filename(img_info))
            if not os.path.isfile(img_path):
                processed_image_ids.add(image_id)
                continue

            image_cat_ids = {ann["category_id"] for ann in anns_by_image[image_id]}
            if use_threshold and all(
                features_per_class.get(get_category_name(categories, cid), 0) >= args.features_per_class_threshold
                for cid in image_cat_ids
            ):
                processed_image_ids.add(image_id)
                continue
            if args.max_images_per_class is not None and all(
                len(image_ids_per_class.get(get_category_name(categories, cid), set())) >= args.max_images_per_class
                for cid in image_cat_ids
            ):
                processed_image_ids.add(image_id)
                continue

            load_jobs.append((image_id, img_info, img_path, args.image_size))

        if load_jobs:
            loader_workers = min(len(load_jobs), max(1, args.scan_workers))
            with ThreadPoolExecutor(max_workers=loader_workers) as executor:
                for record in executor.map(_load_batch_record, load_jobs):
                    if record["ok"]:
                        batch_records.append(record)
                    else:
                        processed_image_ids.add(int(record["image_id"]))

        if not batch_records:
            save_json_atomic(
                checkpoint_path,
                {
                    "processed_image_ids": sorted(processed_image_ids),
                    "saved_files": total_saved,
                    "features_per_class": dict(features_per_class),
                    "images_per_class": {k: len(v) for k, v in image_ids_per_class.items()},
                    "started_at": state.get("started_at"),
                    "updated_at": time.time(),
                },
            )
            continue

        img_batch_tensor = torch.stack([record["img_tensor"] for record in batch_records], dim=0)
        if device == "cuda":
            img_batch_tensor = img_batch_tensor.pin_memory()
        img_batch_tensor = img_batch_tensor.to(device, non_blocking=(device == "cuda"))
        try:
            batch_features = extract_dense_features_batched(model, img_batch_tensor)
        except Exception:
            for record in batch_records:
                processed_image_ids.add(record["image_id"])
            save_json_atomic(
                checkpoint_path,
                {
                    "processed_image_ids": sorted(processed_image_ids),
                    "saved_files": total_saved,
                    "features_per_class": dict(features_per_class),
                    "images_per_class": {k: len(v) for k, v in image_ids_per_class.items()},
                    "started_at": state.get("started_at"),
                    "updated_at": time.time(),
                },
            )
            del img_batch_tensor
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            continue

        for batch_index, record in enumerate(batch_records):
            image_id = record["image_id"]
            img_info = record["img_info"]
            full_features = batch_features[batch_index]
            orig_h, orig_w = img_info["height"], img_info["width"]
            class_saved_in_image: set[str] = set()

            for ann in anns_by_image[image_id]:
                cat_id = ann["category_id"]
                ann_id = ann["id"]
                cat_name = get_category_name(categories, cat_id)
                save_path = os.path.join(args.raw_feature_bank_dir, cat_name, f"{image_id}_{ann_id}.pt")

                if os.path.exists(save_path):
                    continue
                if use_threshold and features_per_class.get(cat_name, 0) >= args.features_per_class_threshold:
                    continue
                if (
                    args.max_images_per_class is not None
                    and image_id not in image_ids_per_class.get(cat_name, set())
                    and len(image_ids_per_class.get(cat_name, set())) >= args.max_images_per_class
                ):
                    continue

                try:
                    binary_mask_np = decode_segmentation_to_mask(ann["segmentation"], orig_h, orig_w)
                except Exception:
                    continue
                if binary_mask_np.sum() == 0:
                    continue

                mask_resized = Image.fromarray(binary_mask_np.astype("uint8") * 255, mode="L").resize(
                    (args.image_size, args.image_size),
                    Image.NEAREST,
                )
                mask_array = np.array(mask_resized, dtype=np.float32) / 255.0
                mask_tensor = torch.from_numpy(mask_array).to(device).unsqueeze(0).unsqueeze(0)
                mask_grid = patch_quant_filter(mask_tensor).squeeze()
                valid_patches = mask_grid > args.mask_coverage_threshold
                if not valid_patches.any():
                    continue

                object_features = F.normalize(full_features[valid_patches], dim=-1)
                torch.save(object_features.cpu(), save_path)
                total_saved += 1
                if use_threshold:
                    features_per_class[cat_name] += int(object_features.shape[0])
                class_saved_in_image.add(cat_name)

            for cat_name in class_saved_in_image:
                image_ids_per_class.setdefault(cat_name, set()).add(image_id)

            processed_image_ids.add(image_id)

        save_json_atomic(
            checkpoint_path,
            {
                "processed_image_ids": sorted(processed_image_ids),
                "saved_files": total_saved,
                "features_per_class": dict(features_per_class),
                "images_per_class": {k: len(v) for k, v in image_ids_per_class.items()},
                "started_at": state.get("started_at"),
                "updated_at": time.time(),
            },
        )
        del img_batch_tensor, batch_features
        if batch_index % 50 == 0:
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    return {
        "output_dir": args.raw_feature_bank_dir,
        "checkpoint_path": checkpoint_path,
        "saved_files": total_saved,
        "processed_images": len(processed_image_ids),
        "images_per_class": {k: len(v) for k, v in image_ids_per_class.items()},
    }


def run(args: argparse.Namespace) -> dict:
    args = _resolve_stage1_paths(args)
    build_result = None
    filter_result = None

    if not args.skip_build:
        build_result = run_build(args)

    if not args.skip_filter:
        filter_result = run_filter(
            input_dir=args.raw_feature_bank_dir,
            output_dir=args.filtered_feature_bank_dir,
            train_ann_file=args.train_ann_file,
            image_dir=args.image_dir,
            keep_threshold=args.keep_threshold,
            min_matches=args.min_matches,
            min_keep_ratio=args.min_keep_ratio,
            filter_mode=args.filter_mode,
            query_chunk=args.query_chunk,
            sim_floor=args.sim_floor,
            target_batch_size=args.target_batch_size,
            target_image_limit=args.target_image_limit,
            target_references=args.target_references,
            num_workers=args.num_workers,
            early_accept=args.early_accept,
            selection_mode=args.selection_mode,
            max_source_images=args.max_source_images,
            top_k_features=args.top_k_features,
            resume=args.resume,
        )

    return {
        "config": {
            "dataset_root": args.dataset_root,
            "train_ann_file": args.train_ann_file,
            "image_dir": args.image_dir,
            "raw_feature_bank_dir": args.raw_feature_bank_dir,
            "filtered_feature_bank_dir": args.filtered_feature_bank_dir,
            "resume": args.resume,
            "skip_build": args.skip_build,
            "skip_filter": args.skip_filter,
        },
        "build": build_result,
        "filter": filter_result,
    }


def main(argv: list[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
