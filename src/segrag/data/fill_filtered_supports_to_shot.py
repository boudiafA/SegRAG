"""
Fill filtered support-image availability up to a target shot count when possible.

This script preserves the existing Stage 1 framework as closely as practical:
- frozen class-local target image set (same first `target_image_limit` train images)
- identical feature extraction settings
- identical per-feature scoring logic
- exact adaptive-q75 re-filtering for affected classes only

It writes an overlay filtered bank containing only affected classes. The effective
filtered bank is:
    base_filtered_bank_dir + overlay_dir overrides

Typical use:
    python -m segrag.data.fill_filtered_supports_to_shot \
      --dataset-root /path/to/dataset \
      --target-shot 20 \
      --resume
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from segrag.data.coco import (
    FeatureBankPaths,
    decode_segmentation_to_mask,
    extract_dense_features_batched,
    get_category_name,
    get_image_filename,
    load_dinov3_model,
    load_lvis_annotations,
    resize_transform,
)
from segrag.modeling.iccd import (
    _apply_adaptive_top_k_features,
    _score_sidecar_path,
    _tp_sidecar_path,
    _fp_sidecar_path,
    _limit_target_image_ids,
    _split_scores_by_file,
    build_target_worklist,
    load_scored_class_entries,
    parse_source_image_id,
    prepare_target_batches,
    save_class_outputs_from_disk,
)
from segrag.utils.resume import load_json, save_json_atomic


DEFAULT_TARGET_SHOT = 20
DEFAULT_TARGET_IMAGE_LIMIT = 100
DEFAULT_SCORE_KEEP_THRESHOLD = 0.6
DEFAULT_TOP_K_FEATURES = 10000
DEFAULT_IMAGE_SIZE = 1536
DEFAULT_PATCH_SIZE = 16
DEFAULT_MASK_COVERAGE_THRESHOLD = 0.90
DEFAULT_QUERY_CHUNK = 256
DEFAULT_TARGET_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 8
DEFAULT_SIM_FLOOR = 0.0
DEFAULT_MIN_MATCHES = 3
ACCEPTANCE_LOGIC_VERSION = "v2_fixed_0.6_gate_then_adaptive_q75"


@dataclass
class DatasetPaths:
    dataset_root: str
    train_ann_file: str
    image_dir: str
    raw_feature_bank_dir: str
    scored_feature_bank_dir: str | None
    filtered_feature_bank_dir: str
    overlay_filtered_bank_dir: str
    cache_dir: str
    state_path: str
    report_path: str
    support_manifest_path: str
    support_shots_path: str
    filter_report_path: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Class-local augmentation to fill filtered support-image availability up to a target shot count."
    )
    parser.add_argument("--dataset-root", nargs="+", required=True)
    parser.add_argument("--target-shot", type=int, default=DEFAULT_TARGET_SHOT)
    parser.add_argument("--target-image-limit", type=int, default=DEFAULT_TARGET_IMAGE_LIMIT)
    parser.add_argument("--score-keep-threshold", type=float, default=DEFAULT_SCORE_KEEP_THRESHOLD)
    parser.add_argument("--top-k-features", type=int, default=DEFAULT_TOP_K_FEATURES)
    parser.add_argument("--query-chunk", type=int, default=DEFAULT_QUERY_CHUNK)
    parser.add_argument("--target-batch-size", type=int, default=DEFAULT_TARGET_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--sim-floor", type=float, default=DEFAULT_SIM_FLOOR)
    parser.add_argument("--min-matches", type=int, default=DEFAULT_MIN_MATCHES)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--mask-coverage-threshold", type=float, default=DEFAULT_MASK_COVERAGE_THRESHOLD)
    parser.add_argument("--repo-path", default="./")
    parser.add_argument("--weights-path", default="./weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    parser.add_argument("--model-name", default="dinov3_vitl16")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-classes", type=int, default=None, help="Optional cap for testing.")
    parser.add_argument(
        "--refilter-every",
        type=int,
        default=5,
        help="Run exact class-local adaptive refilter after this many newly accepted candidate images.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _find_existing_dir(candidates: list[str]) -> str | None:
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return None


def _resolve_train_ann_file(dataset_root: str) -> str:
    candidates = [
        os.path.join(dataset_root, "train.json"),
        os.path.join(dataset_root, "train", "lvis_v1_train.json"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"Could not resolve train annotation file under {dataset_root}")


def _resolve_train_image_root(dataset_root: str, train_ann_file: str) -> str:
    with open(train_ann_file, "r") as handle:
        data = json.load(handle)
    sample_name = data["images"][0].get("file_name") or get_image_filename(data["images"][0])
    candidates = [
        os.path.join(dataset_root, "train", "images"),
        os.path.join(dataset_root, "JPEGImages"),
        os.path.join(dataset_root, "leftImg8bit_trainvaltest", "leftImg8bit", "train"),
        os.path.join(dataset_root, "leftImg8bit", "train"),
        os.path.join(
            dataset_root,
            "downloads",
            "extracted",
            "d8a517a010564481cae92dbe8423d064c7e963f395639c107697aa5e8a9b1bc3",
            "ADEChallengeData2016",
            "images",
            "training",
        ),
    ]
    for root in candidates:
        if os.path.isfile(os.path.join(root, sample_name)):
            return root
    for root, _dirs, files in os.walk(dataset_root):
        if os.path.basename(sample_name) in files:
            candidate = root
            if os.path.isfile(os.path.join(candidate, sample_name)):
                return candidate
    raise FileNotFoundError(f"Could not resolve train image root for sample '{sample_name}' under {dataset_root}")


def resolve_dataset_paths(dataset_root: str, target_shot: int) -> DatasetPaths:
    support_shots_path = os.path.join(dataset_root, "support_shots.json")
    support_manifest_path = os.path.join(dataset_root, "support_manifest.json")
    if not os.path.isfile(support_shots_path):
        raise FileNotFoundError(f"Missing support shots file: {support_shots_path}")
    if not os.path.isfile(support_manifest_path):
        raise FileNotFoundError(f"Missing support manifest file: {support_manifest_path}")

    with open(support_shots_path, "r") as handle:
        support_shots = json.load(handle)
    filtered_bank_dir = support_shots.get("filtered_bank_dir_used")
    if not filtered_bank_dir or not os.path.isdir(filtered_bank_dir):
        filtered_bank_dir = _find_existing_dir(
            [
                os.path.join(dataset_root, "feature_bank_adaptive_q75_from_thr060"),
                os.path.join(dataset_root, "train", "feature_bank_adaptive_q75_from_thr060"),
            ]
        )
    if filtered_bank_dir is None:
        raise FileNotFoundError(f"Could not resolve filtered bank directory for {dataset_root}")

    raw_feature_bank_dir = _find_existing_dir(
        [
            os.path.join(dataset_root, "feature_bank_dinov3_vitl16_1536"),
            os.path.join(dataset_root, "train", "feature_bank_dinov3_vitl16_1536"),
        ]
    )
    if raw_feature_bank_dir is None:
        raise FileNotFoundError(f"Could not resolve raw feature bank directory for {dataset_root}")

    scored_feature_bank_dir = _find_existing_dir(
        [
            os.path.join(dataset_root, "feature_bank_dinov3_vitl16_1536_scored_thr060"),
            os.path.join(dataset_root, "train", "feature_bank_dinov3_vitl16_1536_scored_thr060"),
        ]
    )

    train_ann_file = _resolve_train_ann_file(dataset_root)
    image_dir = _resolve_train_image_root(dataset_root, train_ann_file)
    work_root = os.path.join(dataset_root, f"_fill_filtered_to_{target_shot}")
    overlay_filtered_bank_dir = os.path.join(work_root, "filtered_bank_overlay")
    cache_dir = os.path.join(work_root, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(overlay_filtered_bank_dir, exist_ok=True)

    return DatasetPaths(
        dataset_root=dataset_root,
        train_ann_file=train_ann_file,
        image_dir=image_dir,
        raw_feature_bank_dir=raw_feature_bank_dir,
        scored_feature_bank_dir=scored_feature_bank_dir,
        filtered_feature_bank_dir=filtered_bank_dir,
        overlay_filtered_bank_dir=overlay_filtered_bank_dir,
        cache_dir=cache_dir,
        state_path=os.path.join(work_root, "fill_to_shot_state.json"),
        report_path=os.path.join(work_root, "fill_to_shot_report.json"),
        support_manifest_path=support_manifest_path,
        support_shots_path=support_shots_path,
        filter_report_path=os.path.join(filtered_bank_dir, "intra_class_filter_report.json"),
    )


def _load_support_items(path: str) -> list[dict]:
    with open(path, "r") as handle:
        data = json.load(handle)
    return data["items"] if isinstance(data, dict) and "items" in data else data


def _build_image_ids_by_cat(anns_by_image: dict[int, list[dict]]) -> dict[int, list[int]]:
    image_ids_by_cat: dict[int, set[int]] = defaultdict(set)
    for image_id, anns in anns_by_image.items():
        for ann in anns:
            image_ids_by_cat[ann["category_id"]].add(image_id)
    return {cat_id: sorted(image_ids) for cat_id, image_ids in image_ids_by_cat.items()}


def _parse_filtered_image_ids(cat_dir: str) -> set[int]:
    image_ids: set[int] = set()
    if not os.path.isdir(cat_dir):
        return image_ids
    for fname in os.listdir(cat_dir):
        if not fname.endswith(".pt"):
            continue
        try:
            image_ids.add(parse_source_image_id(fname))
        except Exception:
            continue
    return image_ids


def _filtered_floor_score(cat_dir: str) -> float | None:
    if not os.path.isdir(cat_dir):
        return None
    mins: list[float] = []
    for fname in os.listdir(cat_dir):
        if not fname.endswith(".pt"):
            continue
        score_path = _score_sidecar_path(cat_dir, fname)
        if not os.path.isfile(score_path):
            continue
        scores = np.load(score_path)
        if scores.size:
            mins.append(float(scores.min()))
    return min(mins) if mins else None


def _current_class_floor(
    base_filtered_cat_dir: str,
    overlay_filtered_cat_dir: str,
    current_filtered_report: dict,
) -> float:
    overlay_floor = _filtered_floor_score(overlay_filtered_cat_dir)
    if overlay_floor is not None:
        return overlay_floor
    base_floor = _filtered_floor_score(base_filtered_cat_dir)
    if base_floor is not None:
        return base_floor
    adaptive_threshold = current_filtered_report.get("adaptive_threshold")
    if adaptive_threshold is not None:
        return float(adaptive_threshold)
    return 0.65


def _save_scored_entries_from_memory(
    out_cat_dir: str,
    source_entries: list[tuple[str, torch.Tensor]],
    scores_by_file: dict[str, torch.Tensor],
    good_by_file: dict[str, torch.Tensor],
    bad_by_file: dict[str, torch.Tensor],
    keep_threshold: float,
) -> tuple[int, int]:
    os.makedirs(out_cat_dir, exist_ok=True)
    files_saved = 0
    vectors_saved = 0
    for fname, feats in source_entries:
        if fname not in scores_by_file:
            continue
        scores = scores_by_file[fname]
        keep_mask = scores >= keep_threshold
        if not keep_mask.any():
            continue
        kept_feats = feats[keep_mask].detach().cpu()
        torch.save(kept_feats, os.path.join(out_cat_dir, fname))
        np.save(_score_sidecar_path(out_cat_dir, fname), scores[keep_mask].detach().cpu().numpy().astype(np.float32, copy=False))
        np.save(_tp_sidecar_path(out_cat_dir, fname), good_by_file[fname][keep_mask].detach().cpu().numpy().astype(np.int32, copy=False))
        np.save(_fp_sidecar_path(out_cat_dir, fname), bad_by_file[fname][keep_mask].detach().cpu().numpy().astype(np.int32, copy=False))
        files_saved += 1
        vectors_saved += int(kept_feats.shape[0])
    return files_saved, vectors_saved


def _merge_scored_entries(cat_dirs: list[str]):
    file_refs: list[tuple[str, str]] = []
    scores_by_file: dict[str, torch.Tensor] = {}
    good_by_file: dict[str, torch.Tensor | None] = {}
    bad_by_file: dict[str, torch.Tensor | None] = {}
    seen: set[str] = set()
    for cat_dir in cat_dirs:
        if not cat_dir or not os.path.isdir(cat_dir):
            continue
        refs, scores, good, bad = load_scored_class_entries(cat_dir)
        for fname, path in refs:
            if fname in seen:
                continue
            seen.add(fname)
            file_refs.append((fname, path))
            scores_by_file[fname] = scores[fname]
            good_by_file[fname] = good[fname]
            bad_by_file[fname] = bad[fname]
    return file_refs, scores_by_file, good_by_file, bad_by_file


def _score_source_entries_with_prepared_targets(
    source_entries: list[tuple[str, torch.Tensor]],
    source_image_id: int,
    prepared_target_batches: list[list[dict]],
    device: str,
    query_chunk: int,
    sim_floor: float,
):
    if not source_entries:
        return {}, {}, {}
    source_feats = torch.cat([feats for _fname, feats in source_entries], dim=0)
    good = torch.zeros(source_feats.shape[0], dtype=torch.int32)
    bad = torch.zeros(source_feats.shape[0], dtype=torch.int32)
    source_feats_dev = source_feats.to(device)
    for prepared_targets in prepared_target_batches:
        for target_data in prepared_targets:
            if target_data["image_id"] == source_image_id:
                continue
            grid_dev = target_data["grid"].to(device)
            mask_dev = target_data["mask"].to(device)
            for start in range(0, source_feats.shape[0], query_chunk):
                end = min(start + query_chunk, source_feats.shape[0])
                query_dev = source_feats_dev[start:end]
                sims = query_dev @ grid_dev.T
                best_sims, best_idx = sims.max(dim=1)
                valid = best_sims >= sim_floor
                landed_inside = mask_dev[best_idx]
                good[start:end] += (valid & landed_inside).to(torch.int32).cpu()
                bad[start:end] += (valid & ~landed_inside).to(torch.int32).cpu()
                del query_dev, sims, best_sims, best_idx, valid, landed_inside
            del grid_dev, mask_dev
    scores = torch.full((source_feats.shape[0],), -1.0, dtype=torch.float32)
    total = good + bad
    any_match_mask = total >= 1
    scores[any_match_mask] = good[any_match_mask].float() / total[any_match_mask].float()
    scores_by_file: dict[str, torch.Tensor] = {}
    good_by_file: dict[str, torch.Tensor] = {}
    bad_by_file: dict[str, torch.Tensor] = {}
    offset = 0
    for fname, feats in source_entries:
        n = feats.shape[0]
        scores_by_file[fname] = scores[offset: offset + n]
        good_by_file[fname] = good[offset: offset + n]
        bad_by_file[fname] = bad[offset: offset + n]
        offset += n
    del source_feats_dev, source_feats, good, bad, scores
    return scores_by_file, good_by_file, bad_by_file


def _extract_candidate_source_entries(
    image_id: int,
    cat_id: int,
    images_dict: dict[int, dict],
    anns_by_image: dict[int, list[dict]],
    image_dir: str,
    image_size: int,
    patch_size: int,
    mask_coverage_threshold: float,
    model,
    device: str,
) -> list[tuple[str, torch.Tensor]]:
    img_info = images_dict[image_id]
    img_path = os.path.join(image_dir, get_image_filename(img_info))
    if not os.path.isfile(img_path):
        return []
    class_anns = [ann for ann in anns_by_image.get(image_id, []) if ann["category_id"] == cat_id]
    if not class_anns:
        return []
    try:
        with Image.open(img_path) as image:
            rgb = image.convert("RGB")
            img_tensor = resize_transform(rgb, image_size).unsqueeze(0).to(device)
    except Exception:
        return []

    patch_quant_filter = torch.nn.Conv2d(1, 1, patch_size, stride=patch_size, bias=False)
    patch_quant_filter.weight.data.fill_(1.0 / (patch_size * patch_size))
    patch_quant_filter = patch_quant_filter.to(device)

    try:
        full_features = extract_dense_features_batched(model, img_tensor)[0]
    finally:
        del img_tensor

    source_entries: list[tuple[str, torch.Tensor]] = []
    orig_h, orig_w = img_info["height"], img_info["width"]
    for ann in class_anns:
        try:
            binary_mask_np = decode_segmentation_to_mask(ann["segmentation"], orig_h, orig_w)
        except Exception:
            continue
        if binary_mask_np.sum() == 0:
            continue
        mask_resized = Image.fromarray(binary_mask_np.astype("uint8") * 255, mode="L").resize(
            (image_size, image_size),
            Image.NEAREST,
        )
        mask_array = np.array(mask_resized, dtype=np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_array).to(device).unsqueeze(0).unsqueeze(0)
        mask_grid = patch_quant_filter(mask_tensor).squeeze()
        valid_patches = mask_grid > mask_coverage_threshold
        if not valid_patches.any():
            continue
        object_features = F.normalize(full_features[valid_patches], dim=-1).detach().cpu()
        source_entries.append((f"{image_id}_{ann['id']}.pt", object_features))

    del full_features, patch_quant_filter
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return source_entries


def _candidate_image_metrics(
    scores_by_file: dict[str, torch.Tensor],
    good_by_file: dict[str, torch.Tensor],
    bad_by_file: dict[str, torch.Tensor],
    min_matches: int,
    acceptance_threshold: float,
) -> dict[str, float | int | bool]:
    if not scores_by_file:
        return {
            "accepted": False,
            "passing_features": 0,
            "max_passing_score": None,
            "mean_passing_score": None,
            "median_passing_score": None,
            "total_scored_features": 0,
        }
    scores = torch.cat([scores_by_file[fname] for fname in sorted(scores_by_file)])
    good = torch.cat([good_by_file[fname] for fname in sorted(scores_by_file)])
    bad = torch.cat([bad_by_file[fname] for fname in sorted(scores_by_file)])
    total = good + bad
    scored_mask = total >= min_matches
    scores_for_filter = scores.clone()
    scores_for_filter[~scored_mask] = -1.0
    passing = scores_for_filter >= acceptance_threshold
    passing_scores = scores_for_filter[passing]
    accepted = bool(passing.any().item())
    return {
        "accepted": accepted,
        "passing_features": int(passing.sum().item()),
        "max_passing_score": float(passing_scores.max().item()) if passing_scores.numel() else None,
        "mean_passing_score": float(passing_scores.mean().item()) if passing_scores.numel() else None,
        "median_passing_score": float(passing_scores.median().item()) if passing_scores.numel() else None,
        "total_scored_features": int(scored_mask.sum().item()),
    }


def _effective_filtered_image_count(cat_dir: str) -> int:
    return len(_parse_filtered_image_ids(cat_dir))


def _ensure_base_scored_class_dir(
    class_name: str,
    cat_id: int,
    raw_class_dir: str,
    scored_class_dir: str | None,
    cache_class_dir: str,
    target_worklist: list[tuple[int, str, dict, list[dict]]],
    prepared_target_batches: list[list[dict]],
    model,
    device: str,
    args: argparse.Namespace,
) -> str:
    if scored_class_dir and os.path.isdir(scored_class_dir) and any(f.endswith(".pt") for f in os.listdir(scored_class_dir)):
        return scored_class_dir

    rebuilt_dir = os.path.join(cache_class_dir, "base_scored")
    if os.path.isdir(rebuilt_dir) and any(f.endswith(".pt") for f in os.listdir(rebuilt_dir)):
        return rebuilt_dir

    os.makedirs(rebuilt_dir, exist_ok=True)
    grouped_bank = defaultdict(list)
    if not os.path.isdir(raw_class_dir):
        return rebuilt_dir
    for fname in sorted(f for f in os.listdir(raw_class_dir) if f.endswith(".pt")):
        grouped_bank[parse_source_image_id(fname)].append((fname, os.path.join(raw_class_dir, fname)))
    source_ids = sorted(grouped_bank)
    pbar = tqdm(source_ids, desc=f"  Rebuild base {class_name}", leave=False)
    for source_image_id in pbar:
        source_entries_paths = []
        for fname, path in grouped_bank[source_image_id]:
            try:
                feats = torch.load(path, map_location="cpu", weights_only=True).float()
            except Exception:
                continue
            if feats.ndim != 2 or feats.shape[0] == 0:
                continue
            source_entries_paths.append((fname, F.normalize(feats, dim=-1)))
        if not source_entries_paths:
            continue
        scores_by_file, good_by_file, bad_by_file = _score_source_entries_with_prepared_targets(
            source_entries=source_entries_paths,
            source_image_id=source_image_id,
            prepared_target_batches=prepared_target_batches,
            device=device,
            query_chunk=args.query_chunk,
            sim_floor=args.sim_floor,
        )
        _save_scored_entries_from_memory(
            out_cat_dir=rebuilt_dir,
            source_entries=source_entries_paths,
            scores_by_file=scores_by_file,
            good_by_file=good_by_file,
            bad_by_file=bad_by_file,
            keep_threshold=args.score_keep_threshold,
        )
        del source_entries_paths, scores_by_file, good_by_file, bad_by_file
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    pbar.close()
    return rebuilt_dir


def _refilter_class_overlay(
    base_scored_class_dir: str,
    accepted_scored_class_dir: str,
    overlay_class_dir: str,
    top_k_features: int,
) -> tuple[int, dict]:
    file_refs, scores_by_file, good_by_file, bad_by_file = _merge_scored_entries([base_scored_class_dir, accepted_scored_class_dir])
    if not file_refs:
        if os.path.isdir(overlay_class_dir):
            shutil.rmtree(overlay_class_dir)
        os.makedirs(overlay_class_dir, exist_ok=True)
        return 0, {
            "adaptive_q75": None,
            "adaptive_threshold": None,
            "survivors_before_top_k": 0,
            "kept_features": 0,
        }

    scores = torch.cat([scores_by_file[fname] for fname, _path in file_refs])
    keep_mask, q75_value, adaptive_threshold, survivors_before_top_k = _apply_adaptive_top_k_features(
        scores_for_filter=scores,
        top_k_features=top_k_features,
    )
    keep_by_file = _split_scores_by_file([(fname, scores_by_file[fname]) for fname, _path in file_refs], keep_mask)
    if os.path.isdir(overlay_class_dir):
        shutil.rmtree(overlay_class_dir)
    os.makedirs(overlay_class_dir, exist_ok=True)
    save_class_outputs_from_disk(
        out_cat_dir=overlay_class_dir,
        file_refs=file_refs,
        keep_by_file=keep_by_file,
        scores_by_file=scores_by_file,
    )
    filtered_image_count = _effective_filtered_image_count(overlay_class_dir)
    return filtered_image_count, {
        "adaptive_q75": q75_value,
        "adaptive_threshold": adaptive_threshold,
        "survivors_before_top_k": survivors_before_top_k,
        "kept_features": int(keep_mask.sum().item()),
    }


def _normalize_state(state: dict) -> dict:
    state.setdefault("datasets", {})
    return state


def _normalize_report(report: dict) -> dict:
    report.setdefault("datasets", {})
    return report


def run_for_dataset(paths: DatasetPaths, args: argparse.Namespace, state: dict, report: dict) -> dict:
    with open(paths.support_shots_path, "r") as handle:
        support_shots = json.load(handle)
    with open(paths.filter_report_path, "r") as handle:
        filtered_report = json.load(handle)
    support_items = _load_support_items(paths.support_manifest_path)
    support_image_ids_by_class: dict[str, set[int]] = defaultdict(set)
    for item in support_items:
        support_image_ids_by_class[item["class_name"]].add(int(item["image_id"]))

    images_dict, anns_by_image, categories = load_lvis_annotations(paths.train_ann_file)
    image_ids_by_cat = _build_image_ids_by_cat(anns_by_image)
    cat_id_by_name = {cat["name"]: cat_id for cat_id, cat in categories.items()}
    per_class_summary = support_shots["per_class_summary"]
    deficit_rows = [row for row in per_class_summary if int(row["filtered_available"]) < args.target_shot]
    if args.max_classes is not None:
        deficit_rows = deficit_rows[: args.max_classes]

    dataset_state = state["datasets"].setdefault(paths.dataset_root, {"classes": {}})
    dataset_report = report["datasets"].setdefault(
        paths.dataset_root,
        {
            "dataset_root": paths.dataset_root,
            "train_ann_file": paths.train_ann_file,
            "image_dir": paths.image_dir,
            "base_filtered_bank_dir": paths.filtered_feature_bank_dir,
            "overlay_filtered_bank_dir": paths.overlay_filtered_bank_dir,
            "target_shot": args.target_shot,
            "classes": {},
        },
    )
    save_json_atomic(paths.state_path, state)
    save_json_atomic(paths.report_path, report)

    print(
        f"[fill20] dataset={paths.dataset_root} deficits={len(deficit_rows)} target_shot={args.target_shot}",
        flush=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_dinov3_model(
        FeatureBankPaths(
            train_root=paths.image_dir,
            val_root=paths.image_dir,
            image_size=args.image_size,
            patch_size=args.patch_size,
            model_name=args.model_name,
            repo_path=args.repo_path,
            weights_path=args.weights_path,
            mask_coverage_threshold=args.mask_coverage_threshold,
        ),
        device=device,
    )

    class_pbar = tqdm(deficit_rows, desc=f"Fill to {args.target_shot} ({os.path.basename(paths.dataset_root)})")
    for row in class_pbar:
        class_name = row["class_name"]
        cat_id = row["class_id"]
        if row["filtered_available"] >= args.target_shot:
            continue

        class_state = dataset_state["classes"].setdefault(
            class_name,
            {
                "processed_candidate_ids": [],
                "accepted_candidate_ids": [],
                "status": "pending",
            },
        )
        if class_state.get("acceptance_logic_version") != ACCEPTANCE_LOGIC_VERSION:
            cache_class_dir = os.path.join(paths.cache_dir, class_name)
            accepted_scored_class_dir = os.path.join(cache_class_dir, "accepted_scored")
            overlay_class_dir = os.path.join(paths.overlay_filtered_bank_dir, class_name)
            if os.path.isdir(accepted_scored_class_dir):
                shutil.rmtree(accepted_scored_class_dir)
            if os.path.isdir(overlay_class_dir):
                shutil.rmtree(overlay_class_dir)
            class_state.clear()
            class_state.update(
                {
                    "processed_candidate_ids": [],
                    "accepted_candidate_ids": [],
                    "status": "pending",
                    "acceptance_logic_version": ACCEPTANCE_LOGIC_VERSION,
                }
            )
            save_json_atomic(paths.state_path, state)
        candidate_total = class_state.get("candidate_total")
        processed_count = len(class_state.get("processed_candidate_ids", []))
        if (
            args.resume
            and candidate_total is not None
            and processed_count >= int(candidate_total)
            and class_state.get("status") not in {"done", "exhausted", "insufficient_targets"}
        ):
            class_state["status"] = (
                "done"
                if int(class_state.get("current_effective_filtered_available", row["filtered_available"])) >= args.target_shot
                else "exhausted"
            )
            save_json_atomic(paths.state_path, state)
        if args.resume and class_state.get("status") in {"done", "exhausted", "insufficient_targets"}:
            continue

        base_filtered_cat_dir = os.path.join(paths.filtered_feature_bank_dir, class_name)
        overlay_class_dir = os.path.join(paths.overlay_filtered_bank_dir, class_name)
        cache_class_dir = os.path.join(paths.cache_dir, class_name)
        accepted_scored_class_dir = os.path.join(cache_class_dir, "accepted_scored")
        os.makedirs(accepted_scored_class_dir, exist_ok=True)
        os.makedirs(cache_class_dir, exist_ok=True)

        current_effective = _effective_filtered_image_count(overlay_class_dir) if os.path.isdir(overlay_class_dir) and any(f.endswith(".pt") for f in os.listdir(overlay_class_dir)) else int(row["filtered_available"])
        if current_effective >= args.target_shot:
            class_state["status"] = "done"
            save_json_atomic(paths.state_path, state)
            continue

        all_train_image_ids = image_ids_by_cat.get(cat_id, [])
        target_ids = _limit_target_image_ids(all_train_image_ids, args.target_image_limit)
        target_worklist = build_target_worklist(
            image_dir=paths.image_dir,
            images=images_dict,
            anns_by_image=anns_by_image,
            cat_id=cat_id,
            image_ids=target_ids,
        )
        if len(target_worklist) < 2:
            class_state["status"] = "insufficient_targets"
            dataset_report["classes"][class_name] = {
                "status": "insufficient_targets",
                "filtered_available_before": int(row["filtered_available"]),
                "filtered_available_after": int(current_effective),
            }
            save_json_atomic(paths.state_path, state)
            save_json_atomic(paths.report_path, report)
            continue

        existing_support_ids = support_image_ids_by_class.get(class_name, set())
        candidate_ids = [image_id for image_id in all_train_image_ids if image_id not in existing_support_ids]
        print(
            f"[fill20] class={class_name} start current={current_effective}/{args.target_shot} "
            f"targets={len(target_worklist)} candidates={len(candidate_ids)}",
            flush=True,
        )
        class_state["status"] = "preparing_targets"
        class_state["current_effective_filtered_available"] = current_effective
        class_state["candidate_total"] = len(candidate_ids)
        class_state["acceptance_logic_version"] = ACCEPTANCE_LOGIC_VERSION
        save_json_atomic(paths.state_path, state)

        prepared_target_batches = prepare_target_batches(
            model=model,
            device=device,
            target_worklist=target_worklist,
            target_batch_size=args.target_batch_size,
            num_workers=args.num_workers,
            class_name=class_name,
        )

        base_scored_class_dir = _ensure_base_scored_class_dir(
            class_name=class_name,
            cat_id=cat_id,
            raw_class_dir=os.path.join(paths.raw_feature_bank_dir, class_name),
            scored_class_dir=(os.path.join(paths.scored_feature_bank_dir, class_name) if paths.scored_feature_bank_dir else None),
            cache_class_dir=cache_class_dir,
            target_worklist=target_worklist,
            prepared_target_batches=prepared_target_batches,
            model=model,
            device=device,
            args=args,
        )

        current_floor = _current_class_floor(
            base_filtered_cat_dir=base_filtered_cat_dir,
            overlay_filtered_cat_dir=overlay_class_dir,
            current_filtered_report=filtered_report.get("per_class", {}).get(class_name, {}),
        )
        processed_candidate_ids = set(class_state.get("processed_candidate_ids", []))
        accepted_candidate_ids = set(class_state.get("accepted_candidate_ids", []))

        accepted_since_refilter = 0
        cand_pbar = tqdm(candidate_ids, desc=f"  Candidates {class_name}", leave=False)
        for candidate_image_id in cand_pbar:
            if candidate_image_id in processed_candidate_ids:
                continue
            if current_effective >= args.target_shot:
                break

            source_entries = _extract_candidate_source_entries(
                image_id=candidate_image_id,
                cat_id=cat_id,
                images_dict=images_dict,
                anns_by_image=anns_by_image,
                image_dir=paths.image_dir,
                image_size=args.image_size,
                patch_size=args.patch_size,
                mask_coverage_threshold=args.mask_coverage_threshold,
                model=model,
                device=device,
            )
            if not source_entries:
                processed_candidate_ids.add(candidate_image_id)
                class_state["processed_candidate_ids"] = sorted(processed_candidate_ids)
                save_json_atomic(paths.state_path, state)
                continue

            scores_by_file, good_by_file, bad_by_file = _score_source_entries_with_prepared_targets(
                source_entries=source_entries,
                source_image_id=candidate_image_id,
                prepared_target_batches=prepared_target_batches,
                device=device,
                query_chunk=args.query_chunk,
                sim_floor=args.sim_floor,
            )
            metrics = _candidate_image_metrics(
                scores_by_file=scores_by_file,
                good_by_file=good_by_file,
                bad_by_file=bad_by_file,
                min_matches=args.min_matches,
                acceptance_threshold=args.score_keep_threshold,
            )
            processed_candidate_ids.add(candidate_image_id)
            if metrics["accepted"]:
                _save_scored_entries_from_memory(
                    out_cat_dir=accepted_scored_class_dir,
                    source_entries=source_entries,
                    scores_by_file=scores_by_file,
                    good_by_file=good_by_file,
                    bad_by_file=bad_by_file,
                    keep_threshold=args.score_keep_threshold,
                )
                accepted_candidate_ids.add(candidate_image_id)
                accepted_since_refilter += 1

            class_state["processed_candidate_ids"] = sorted(processed_candidate_ids)
            class_state["accepted_candidate_ids"] = sorted(accepted_candidate_ids)
            class_state["status"] = "running"

            cand_pbar.set_postfix(
                current=current_effective,
                target=args.target_shot,
                accepted=len(accepted_candidate_ids),
                floor=f"{current_floor:.3f}",
            )
            save_json_atomic(paths.state_path, state)

            should_refilter = False
            if metrics["accepted"] and accepted_since_refilter >= max(1, args.refilter_every):
                should_refilter = True
            if metrics["accepted"] and (current_effective + accepted_since_refilter) >= args.target_shot:
                should_refilter = True

            if should_refilter:
                current_effective, refilter_meta = _refilter_class_overlay(
                    base_scored_class_dir=base_scored_class_dir,
                    accepted_scored_class_dir=accepted_scored_class_dir,
                    overlay_class_dir=overlay_class_dir,
                    top_k_features=args.top_k_features,
                )
                current_floor = _current_class_floor(
                    base_filtered_cat_dir=base_filtered_cat_dir,
                    overlay_filtered_cat_dir=overlay_class_dir,
                    current_filtered_report=filtered_report.get("per_class", {}).get(class_name, {}),
                )
                class_state["current_effective_filtered_available"] = current_effective
                class_state["last_refilter"] = refilter_meta
                accepted_since_refilter = 0
                print(
                    f"[fill20] class={class_name} refilter current={current_effective}/{args.target_shot} "
                    f"accepted={len(accepted_candidate_ids)} floor={current_floor:.4f}",
                    flush=True,
                )
                save_json_atomic(paths.state_path, state)
                save_json_atomic(
                    paths.report_path,
                    report,
                )

            del source_entries, scores_by_file, good_by_file, bad_by_file
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        cand_pbar.close()

        if accepted_since_refilter > 0:
            current_effective, refilter_meta = _refilter_class_overlay(
                base_scored_class_dir=base_scored_class_dir,
                accepted_scored_class_dir=accepted_scored_class_dir,
                overlay_class_dir=overlay_class_dir,
                top_k_features=args.top_k_features,
            )
            current_floor = _current_class_floor(
                base_filtered_cat_dir=base_filtered_cat_dir,
                overlay_filtered_cat_dir=overlay_class_dir,
                current_filtered_report=filtered_report.get("per_class", {}).get(class_name, {}),
            )
            class_state["current_effective_filtered_available"] = current_effective
            class_state["last_refilter"] = refilter_meta

        class_state["status"] = "done" if current_effective >= args.target_shot else "exhausted"
        print(
            f"[fill20] class={class_name} done status={class_state['status']} current={current_effective}/{args.target_shot} "
            f"accepted={len(accepted_candidate_ids)} processed={len(processed_candidate_ids)}",
            flush=True,
        )
        dataset_report["classes"][class_name] = {
            "status": class_state["status"],
            "filtered_available_before": int(row["filtered_available"]),
            "filtered_available_after": int(current_effective),
            "accepted_candidate_images": len(accepted_candidate_ids),
            "processed_candidate_images": len(processed_candidate_ids),
            "current_floor": current_floor,
            "acceptance_threshold": args.score_keep_threshold,
            "total_train_images_with_class": len(all_train_image_ids),
            "support_pool_images_before": len(existing_support_ids),
            "frozen_target_image_count": len(target_worklist),
        }
        save_json_atomic(paths.state_path, state)
        save_json_atomic(paths.report_path, report)
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    class_pbar.close()
    return report


def run(args: argparse.Namespace) -> dict:
    state = {"datasets": {}}
    report = {"datasets": {}}

    # Multi-dataset runs use per-dataset state/report files; keep in-memory mirrors simple.
    for dataset_root in args.dataset_root:
        paths = resolve_dataset_paths(dataset_root, args.target_shot)
        dataset_state = load_json(paths.state_path, {"datasets": {}}) if args.resume else {"datasets": {}}
        dataset_report = load_json(paths.report_path, {"datasets": {}}) if args.resume else {"datasets": {}}
        dataset_state = _normalize_state(dataset_state)
        dataset_report = _normalize_report(dataset_report)
        run_for_dataset(paths=paths, args=args, state=dataset_state, report=dataset_report)
        save_json_atomic(paths.state_path, dataset_state)
        save_json_atomic(paths.report_path, dataset_report)
        state["datasets"][dataset_root] = dataset_state["datasets"].get(dataset_root, {})
        report["datasets"][dataset_root] = dataset_report["datasets"].get(dataset_root, {})

    return report


def main(argv: list[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
