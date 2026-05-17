"""
Resumable feature-matching prompt-cache generator.

This module precomputes and stores the final positive prompts for the three
feature-matching methods so later evaluation and SAM3 steps can reuse them
without recomputing dense retrieval on every run.

The implementation deliberately reuses the stable matcher/generator classes
from the current codebase while exposing a cleaner CLI and shared resume logic.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from GitHub.common import COMPARE_ALL, METHODS, default_dataset_paths, validate_method
from GitHub.feature_matching_backends import (
    DATALOADER_NUM_WORKERS,
    MAX_REFERENCES,
    PREFETCH_QUEUE_SIZE,
    SUPPRESSION_MARGIN,
    AbsoluteFeatureBankMatcher,
    DINOv3FeatureExtractor,
    DualThresholdPromptGenerator,
    LVIS,
    LVISImageRecord,
    LVISPrefetchDataset,
    RelativeFeatureBankMatcher,
    collate_records,
)
from GitHub.prompt_cache import (
    build_prompt_cache_dir,
    ensure_prompt_cache_dir,
    load_prompt_cache,
    save_prompt_cache,
)
from GitHub.resume import load_json, save_json_atomic


def parse_args():
    """
    Precompute and cache the final positive prompts for one feature-matching
    method or all methods.
    """
    parser = argparse.ArgumentParser(description="Precompute resumable feature-matching prompt caches.")
    parser.add_argument("--method", type=str, default=COMPARE_ALL)
    parser.add_argument("--dataset-root", type=str, default=".")
    parser.add_argument("--annotation-file", type=str, default=None)
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--feature-bank-dir", type=str, default=None)
    parser.add_argument("--filtered-bank-dir", type=str, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-references", type=int, default=MAX_REFERENCES)
    parser.add_argument("--num-points", type=int, default=10)
    parser.add_argument("--sim-threshold", type=float, default=0.8)
    parser.add_argument("--peak-threshold", type=float, default=0.2)
    parser.add_argument("--min-peak-distance", type=int, default=10)
    parser.add_argument("--suppression-margin", type=float, default=SUPPRESSION_MARGIN)
    parser.add_argument("--no-suppression", action="store_true")
    parser.add_argument("--loose-threshold", type=float, default=0.8)
    parser.add_argument("--min-component-size", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _resolve_paths(args):
    defaults = default_dataset_paths(args.dataset_root)
    args.annotation_file = args.annotation_file or defaults.annotation_file
    args.image_dir = args.image_dir or defaults.image_dir
    args.feature_bank_dir = args.feature_bank_dir or defaults.raw_feature_bank_dir
    args.filtered_bank_dir = args.filtered_bank_dir or defaults.filtered_feature_bank_dir


def _discover_records(lvis_api, image_dir: str, max_images: int | None):
    img_ids = lvis_api.get_img_ids()
    if max_images is not None:
        img_ids = img_ids[:max_images]
    records = []
    for img_id in tqdm(img_ids, desc="Scanning annotations"):
        img_info = lvis_api.load_imgs([img_id])[0]
        file_name = (img_info.get("file_name", "") or "").strip()
        if not file_name:
            coco_url = (img_info.get("coco_url", "") or "").strip()
            file_name = os.path.basename(coco_url)
        if not file_name:
            continue
        jpg_path = os.path.join(image_dir, file_name)
        if not os.path.isfile(jpg_path):
            continue
        ann_ids = lvis_api.get_ann_ids(img_ids=[img_id])
        anns = lvis_api.load_anns(ann_ids)
        if not anns:
            continue
        cats_in_img = set(ann["category_id"] for ann in anns)
        not_exhaustive = set(img_info.get("not_exhaustive_category_ids", []))
        valid_cats = cats_in_img - not_exhaustive
        if not valid_cats:
            continue
        records.append(LVISImageRecord(img_id, img_info, file_name, jpg_path, anns, valid_cats))
    return records


def _build_matchers(args, dino_extractor):
    abs_matcher = AbsoluteFeatureBankMatcher(
        args.filtered_bank_dir,
        dino_extractor.device,
        max_references=args.max_references,
    )
    supp_bank = None
    if not args.no_suppression:
        supp_path = os.path.join(args.filtered_bank_dir, "_suppression_bank.pt")
        if os.path.exists(supp_path):
            supp_bank = torch.load(supp_path, map_location="cpu", weights_only=False)
    rel_matcher = RelativeFeatureBankMatcher(
        args.filtered_bank_dir,
        dino_extractor.device,
        max_references=args.max_references,
        suppression_bank=supp_bank,
    )
    hyb_generator = DualThresholdPromptGenerator(
        args.filtered_bank_dir,
        dino_extractor.device,
        max_references=args.max_references,
    )
    return abs_matcher, rel_matcher, hyb_generator


def _cache_dir_for_method(args, method: str) -> str:
    if method == "absolute_similarity":
        config = {
            "name": "absolute_similarity",
            "feature_bank_dir": None,
            "filtered_bank_dir": args.filtered_bank_dir,
            "num_points": args.num_points,
            "sim_threshold": args.sim_threshold,
            "max_references": args.max_references,
        }
    elif method == "relative_similarity":
        config = {
            "name": "relative_similarity",
            "feature_bank_dir": None,
            "filtered_bank_dir": args.filtered_bank_dir,
            "peak_threshold": args.peak_threshold,
            "min_peak_distance": args.min_peak_distance,
            "suppression_margin": args.suppression_margin,
            "no_suppression": args.no_suppression,
            "max_references": args.max_references,
        }
    else:
        config = {
            "name": "hybrid",
            "filtered_bank_dir": args.filtered_bank_dir,
            "loose_threshold": args.loose_threshold,
            "min_peak_distance": args.min_peak_distance,
            "min_component_size": args.min_component_size,
            "max_references": args.max_references,
        }
    cache_dir = build_prompt_cache_dir(args.dataset_root, method, config)
    ensure_prompt_cache_dir(cache_dir, {"method": method, "config": config})
    return cache_dir


def _compute_absolute(abs_matcher, dino_features, class_name, original_size, args):
    coords, labels, scores = abs_matcher.match_points(
        dino_features,
        class_name,
        args.num_points,
        original_size,
        args.sim_threshold,
    )
    return {"filtered": {class_name: {"coords": coords, "labels": labels, "scores": scores}}}


def _compute_relative(rel_matcher, dino_features, class_name, original_size, args):
    coords, labels, scores = rel_matcher.match_points(
        dino_features,
        class_name,
        original_size,
        peak_threshold=args.peak_threshold,
        min_peak_distance=args.min_peak_distance,
        suppression_margin=args.suppression_margin,
    )
    return {"filtered": {class_name: {"coords": coords, "labels": labels, "scores": scores}}}


def _compute_hybrid(hyb_generator, dino_features, class_name, original_size, args):
    prompts, _ = hyb_generator.generate_prompts(
        query_features=dino_features,
        cat_name=class_name,
        original_size=original_size,
        loose_threshold=args.loose_threshold,
        min_peak_distance=args.min_peak_distance,
        min_component_size=args.min_component_size,
    )
    return {"text_filtered": {class_name: prompts or []}}


def _save_class_payload(cache_dir: str, file_name: str, class_id: int, payload_key: str, value):
    existing = load_prompt_cache(cache_dir, file_name) or {}
    bucket = existing.setdefault(payload_key, {})
    bucket[str(class_id)] = value
    save_prompt_cache(cache_dir, file_name, existing)


def _run_method(method: str, args) -> dict:
    cache_dir = _cache_dir_for_method(args, method)
    checkpoint_path = os.path.join(cache_dir, "_resume_state.json")
    state = load_json(checkpoint_path, {"processed_image_ids": []}) if args.resume else {"processed_image_ids": []}
    processed = set(state.get("processed_image_ids", []))

    lvis_api = LVIS(args.annotation_file)
    cats = lvis_api.load_cats(lvis_api.get_cat_ids())
    classes = {cat["id"]: cat["name"] for cat in cats}
    records = _discover_records(lvis_api, args.image_dir, args.max_images)
    dataset = LVISPrefetchDataset(records)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=DATALOADER_NUM_WORKERS,
        collate_fn=collate_records,
        prefetch_factor=PREFETCH_QUEUE_SIZE if DATALOADER_NUM_WORKERS > 0 else None,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=DATALOADER_NUM_WORKERS > 0,
    )
    dino_extractor = DINOv3FeatureExtractor(annotation_file=args.annotation_file)
    abs_matcher, rel_matcher, hyb_generator = _build_matchers(args, dino_extractor)

    for batch in tqdm(loader, total=len(records), desc=f"Caching {method}"):
        rec = batch[0]
        if rec.img_id in processed:
            continue
        dino_features = dino_extractor.extract(rec.pil_image, cache_key=rec.file_name)
        original_size = (rec.img_info["height"], rec.img_info["width"])
        for class_id in rec.valid_cats:
            class_name = classes[class_id]
            if method == "absolute_similarity":
                coords, labels, scores = abs_matcher.match_points(
                    dino_features, class_name, args.num_points, original_size, args.sim_threshold
                )
                _save_class_payload(
                    cache_dir,
                    rec.file_name,
                    class_id,
                    "filtered",
                    {"coords": coords, "labels": labels, "scores": scores},
                )
            elif method == "relative_similarity":
                coords, labels, scores = rel_matcher.match_points(
                    dino_features,
                    class_name,
                    original_size,
                    peak_threshold=args.peak_threshold,
                    min_peak_distance=args.min_peak_distance,
                    suppression_margin=args.suppression_margin,
                )
                _save_class_payload(
                    cache_dir,
                    rec.file_name,
                    class_id,
                    "filtered",
                    {"coords": coords, "labels": labels, "scores": scores},
                )
            else:
                prompts, _ = hyb_generator.generate_prompts(
                    query_features=dino_features,
                    cat_name=class_name,
                    original_size=original_size,
                    loose_threshold=args.loose_threshold,
                    min_peak_distance=args.min_peak_distance,
                    min_component_size=args.min_component_size,
                )
                _save_class_payload(
                    cache_dir,
                    rec.file_name,
                    class_id,
                    "text_filtered",
                    prompts or [],
                )
        processed.add(rec.img_id)
        save_json_atomic(checkpoint_path, {"processed_image_ids": sorted(processed)})

    return {"method": method, "cache_dir": cache_dir, "processed_images": len(processed)}


def run(args):
    _resolve_paths(args)
    method = validate_method(args.method, allow_compare_all=True)
    methods = METHODS if method == COMPARE_ALL else (method,)
    summary = {}
    for name in methods:
        summary[name] = _run_method(name, args)
    return summary


def main():
    args = parse_args()
    result = run(args)
    print(result)


if __name__ == "__main__":
    main()
