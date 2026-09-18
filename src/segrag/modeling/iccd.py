"""
Native SegRAG implementation of intra-class feature-bank filtering.

This module preserves the current optimized behavior:
- resumable per-class filtering
- target-batch preparation once per class
- DINOv3 dense feature extraction matching the current bank build
- exact shared source/target support sets with self-image exclusion
"""

from __future__ import annotations

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
from pycocotools import mask as mask_utils
from torchvision.transforms import v2
from tqdm import tqdm

from segrag.utils.resume import load_json, save_json_atomic
from segrag.utils.paths import resolve_dinov3_repo_path, resolve_dinov3_weights_path


DEFAULT_DATASET_ROOT = "."
DEFAULT_INPUT_DIR = os.path.join(DEFAULT_DATASET_ROOT, "feature_bank_dinov3_vitl16_1536")
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_DATASET_ROOT, "feature_bank_dinov3_vitl16_intra_class_filtered_1536")
DEFAULT_TRAIN_ANN = os.path.join(DEFAULT_DATASET_ROOT, "train.json")
DEFAULT_IMAGE_DIR = DEFAULT_DATASET_ROOT
DINOV3_REPO_PATH = "./"
WEIGHTS_PATH = "./weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
MODEL_NAME = "dinov3_vitl16"
IMAGE_SIZE = 1536
PATCH_SIZE = 16
MASK_COVERAGE_THRESHOLD = 0.50
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

_PATCH_AVG = torch.nn.Conv2d(1, 1, PATCH_SIZE, stride=PATCH_SIZE, bias=False)
_PATCH_AVG.weight.data.fill_(1.0 / (PATCH_SIZE * PATCH_SIZE))
_PATCH_AVG.requires_grad_(False)


def resize_transform(image: Image.Image, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((image_size, image_size), interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )
    return transform(image)


def load_model(device: str):
    repo_path = resolve_dinov3_repo_path(DINOV3_REPO_PATH)
    weights_path = resolve_dinov3_weights_path(WEIGHTS_PATH, repo_path=repo_path)
    model = torch.hub.load(
        repo_or_dir=repo_path,
        model=MODEL_NAME,
        source="local",
        weights=weights_path,
    )
    return model.to(device).eval()


@torch.inference_mode()
def extract_dense_features_batched(model, img_tensors: torch.Tensor) -> torch.Tensor:
    feats = model.get_intermediate_layers(
        img_tensors,
        n=1,
        reshape=True,
        norm=True,
        return_class_token=False,
    )
    return feats[0].permute(0, 2, 3, 1).contiguous()


def load_annotations(ann_file: str):
    with open(ann_file, "r") as handle:
        data = json.load(handle)
    images = {img["id"]: img for img in data["images"]}
    categories = {cat["id"]: cat for cat in data["categories"]}
    anns_by_image = defaultdict(list)
    image_ids_by_cat = defaultdict(set)
    for ann in data["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)
        image_ids_by_cat[ann["category_id"]].add(ann["image_id"])
    return images, dict(anns_by_image), categories, {k: sorted(v) for k, v in image_ids_by_cat.items()}


def get_image_filename(img_info: dict) -> str:
    if "file_name" in img_info:
        return img_info["file_name"]
    if "coco_url" in img_info:
        return img_info["coco_url"].split("/")[-1]
    raise KeyError(f"No 'file_name' or 'coco_url' in image info: {list(img_info.keys())}")


def get_category_name(categories: dict, cat_id: int) -> str:
    return categories.get(cat_id, {}).get("name", f"cat_{cat_id}")


def decode_segmentation_to_mask(segmentation, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(segmentation, dict):
        if isinstance(segmentation["counts"], list):
            rle = mask_utils.frPyObjects(segmentation, height, width)
        else:
            rle = segmentation
    else:
        raise ValueError(f"Unknown segmentation format: {type(segmentation)}")
    return mask_utils.decode(rle)


def build_mask_grid(anns: list[dict], height: int, width: int) -> torch.Tensor:
    union_mask = np.zeros((height, width), dtype=np.uint8)
    for ann in anns:
        try:
            mask = decode_segmentation_to_mask(ann["segmentation"], height, width)
        except Exception:
            continue
        union_mask = np.logical_or(union_mask, mask).astype(np.uint8)
    mask_pil = Image.fromarray(union_mask * 255, mode="L")
    mask_resized = mask_pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
    mask_np = np.array(mask_resized, dtype=np.float32) / 255.0
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
    with torch.inference_mode():
        mask_grid = _PATCH_AVG(mask_tensor).squeeze(0).squeeze(0)
    return (mask_grid > MASK_COVERAGE_THRESHOLD).reshape(-1)


def parse_source_image_id(fname: str) -> int:
    stem = os.path.splitext(fname)[0]
    return int(stem.split("_", 1)[0])


def index_class_bank_grouped(cat_dir: str):
    grouped = defaultdict(list)
    for fname in sorted(f for f in os.listdir(cat_dir) if f.endswith(".pt")):
        grouped[parse_source_image_id(fname)].append((fname, os.path.join(cat_dir, fname)))
    return dict(grouped)


def load_source_entries(file_refs: list[tuple[str, str]]):
    loaded = []
    for fname, path in file_refs:
        try:
            feats = torch.load(path, map_location="cpu", weights_only=True).float()
        except Exception:
            continue
        if feats.ndim != 2 or feats.shape[0] == 0:
            continue
        loaded.append((fname, F.normalize(feats, dim=-1)))
    return loaded


def build_target_worklist(image_dir: str, images: dict, anns_by_image: dict, cat_id: int, image_ids: list[int]):
    worklist = []
    for image_id in image_ids:
        img_info = images[image_id]
        class_anns = [ann for ann in anns_by_image.get(image_id, []) if ann["category_id"] == cat_id]
        if not class_anns:
            continue
        img_path = os.path.join(image_dir, get_image_filename(img_info))
        if not os.path.exists(img_path):
            continue
        worklist.append((image_id, img_path, img_info, class_anns))
    return worklist


def prepare_target_batch(model, device: str, batch: list[tuple[int, str, dict, list[dict]]], num_workers: int):
    def _prepare_target_item(item):
        image_id, img_path, img_info, class_anns = item
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            return None
        return (
            image_id,
            img_info,
            resize_transform(image),
            build_mask_grid(class_anns, img_info["height"], img_info["width"]).cpu(),
        )

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        prepared = list(executor.map(_prepare_target_item, batch))
    prepared = [item for item in prepared if item is not None]
    img_tensors = [item[2] for item in prepared]
    if not img_tensors:
        return []
    batch_tensor = torch.stack(img_tensors, dim=0).to(device)
    batch_features = extract_dense_features_batched(model, batch_tensor)
    prepared_targets = []
    for idx, (image_id, _img_info, _img_tensor, mask_grid) in enumerate(prepared):
        prepared_targets.append(
            {
                "image_id": image_id,
                "grid": F.normalize(batch_features[idx].reshape(-1, batch_features.shape[-1]), dim=-1).cpu(),
                "mask": mask_grid.bool().cpu(),
            }
        )
    del batch_tensor, batch_features
    return prepared_targets


def prepare_target_batches(model, device: str, target_worklist: list[tuple[int, str, dict, list[dict]]], target_batch_size: int, num_workers: int, class_name: str | None = None):
    desc = f"  Score {class_name}" if class_name else "  Score targets"
    prepared_batches = []
    for batch_start in tqdm(range(0, len(target_worklist), target_batch_size), desc=desc, leave=False):
        batch = target_worklist[batch_start: batch_start + target_batch_size]
        prepared_targets = prepare_target_batch(model=model, device=device, batch=batch, num_workers=num_workers)
        if prepared_targets:
            prepared_batches.append(prepared_targets)
    return prepared_batches


def _shared_support_image_ids(
    *,
    grouped_bank: dict[int, list[tuple[str, str]]],
    annotated_image_ids: list[int],
    support_image_limit: int | None,
    class_name: str,
) -> list[int]:
    source_ids = sorted(grouped_bank)
    selected_ids = list(
        annotated_image_ids
        if support_image_limit is None
        else annotated_image_ids[:support_image_limit]
    )
    if source_ids != selected_ids:
        source_only = sorted(set(source_ids) - set(selected_ids))
        annotation_only = sorted(set(selected_ids) - set(source_ids))
        raise RuntimeError(
            f"ICCD support-set mismatch for class {class_name!r}: the raw bank and "
            "support annotation must contain exactly the same selected images. "
            f"bank_only={source_only[:10]}, annotation_only={annotation_only[:10]}. "
            "Rebuild the raw bank from the exact N-shot support annotation."
        )
    return selected_ids


def _split_scores_by_file(file_entries: list[tuple[str, torch.Tensor]], flat_keep_mask: torch.Tensor):
    keep_by_file = {}
    offset = 0
    for fname, feats in file_entries:
        n = feats.shape[0]
        keep_by_file[fname] = flat_keep_mask[offset: offset + n]
        offset += n
    return keep_by_file


def _apply_top_k_features(scores_for_filter: torch.Tensor, keep_threshold: float | None, top_k_features: int | None) -> torch.Tensor:
    if keep_threshold is None:
        return torch.ones_like(scores_for_filter, dtype=torch.bool)
    passing_idx = (scores_for_filter >= keep_threshold).nonzero(as_tuple=True)[0]
    keep_mask = torch.zeros_like(scores_for_filter, dtype=torch.bool)
    if passing_idx.numel() == 0:
        return keep_mask
    if top_k_features is None:
        keep_mask[passing_idx] = True
        return keep_mask
    if top_k_features <= 0:
        return keep_mask
    if passing_idx.numel() <= top_k_features:
        keep_mask[passing_idx] = True
        return keep_mask
    passing_scores = scores_for_filter[passing_idx]
    _, top_idx = passing_scores.topk(top_k_features)
    keep_mask[passing_idx[top_idx]] = True
    return keep_mask


def _single_reference_keep_mask(num_features: int, top_k_features: int | None) -> torch.Tensor:
    keep_mask = torch.zeros(num_features, dtype=torch.bool)
    keep_count = num_features
    if top_k_features is not None and top_k_features > 0:
        keep_count = min(keep_count, top_k_features)
    keep_mask[:keep_count] = True
    return keep_mask


def _apply_adaptive_top_k_features(
    scores_for_filter: torch.Tensor,
    top_k_features: int | None,
) -> tuple[torch.Tensor, float | None, float | None, int]:
    keep_mask = torch.zeros_like(scores_for_filter, dtype=torch.bool)
    scored_mask = scores_for_filter >= 0
    scored_values = scores_for_filter[scored_mask]
    if scored_values.numel() == 0:
        return keep_mask, None, None, 0

    scored_values_float = scored_values.float()
    q75 = float(torch.quantile(scored_values_float, 0.75).item())
    adaptive_threshold = float(np.clip(q75 * 0.90, 0.65, 0.82))

    passing_idx = (scores_for_filter >= adaptive_threshold).nonzero(as_tuple=True)[0]
    survivors_before_top_k = int(passing_idx.numel())
    if passing_idx.numel() == 0:
        return keep_mask, q75, adaptive_threshold, survivors_before_top_k

    if top_k_features is None or top_k_features <= 0 or passing_idx.numel() <= top_k_features:
        keep_mask[passing_idx] = True
        return keep_mask, q75, adaptive_threshold, survivors_before_top_k

    passing_scores = scores_for_filter[passing_idx]
    _, top_idx = passing_scores.topk(top_k_features)
    keep_mask[passing_idx[top_idx]] = True
    return keep_mask, q75, adaptive_threshold, survivors_before_top_k


def _score_sidecar_path(out_cat_dir: str, fname: str) -> str:
    stem, _ = os.path.splitext(fname)
    return os.path.join(out_cat_dir, f"{stem}.scores.npy")


def _tp_sidecar_path(out_cat_dir: str, fname: str) -> str:
    stem, _ = os.path.splitext(fname)
    return os.path.join(out_cat_dir, f"{stem}.tp.npy")


def _fp_sidecar_path(out_cat_dir: str, fname: str) -> str:
    stem, _ = os.path.splitext(fname)
    return os.path.join(out_cat_dir, f"{stem}.fp.npy")


def save_class_outputs_from_disk(
    out_cat_dir: str,
    file_refs: list[tuple[str, str]],
    keep_by_file: dict[str, torch.Tensor],
    scores_by_file: dict[str, torch.Tensor],
):
    os.makedirs(out_cat_dir, exist_ok=True)
    files_saved = 0
    vectors_saved = 0
    for fname, path in file_refs:
        keep_mask = keep_by_file.get(fname)
        if keep_mask is None or not keep_mask.any():
            continue
        feats = torch.load(path, map_location="cpu", weights_only=True).float()
        kept = feats[keep_mask]
        torch.save(kept, os.path.join(out_cat_dir, fname))
        file_scores = scores_by_file.get(fname)
        if file_scores is not None:
            kept_scores = file_scores[keep_mask].detach().cpu().numpy().astype(np.float32, copy=False)
            np.save(_score_sidecar_path(out_cat_dir, fname), kept_scores)
        files_saved += 1
        vectors_saved += int(kept.shape[0])
    return files_saved, vectors_saved


def save_scored_bank_from_disk(
    out_cat_dir: str,
    file_refs: list[tuple[str, str]],
    keep_by_file: dict[str, torch.Tensor],
    scores_by_file: dict[str, torch.Tensor],
    good_by_file: dict[str, torch.Tensor],
    bad_by_file: dict[str, torch.Tensor],
):
    os.makedirs(out_cat_dir, exist_ok=True)
    files_saved = 0
    vectors_saved = 0
    for fname, path in file_refs:
        keep_mask = keep_by_file.get(fname)
        if keep_mask is None or not keep_mask.any():
            continue
        feats = torch.load(path, map_location="cpu", weights_only=True).float()
        kept = feats[keep_mask]
        torch.save(kept, os.path.join(out_cat_dir, fname))
        np.save(
            _score_sidecar_path(out_cat_dir, fname),
            scores_by_file[fname][keep_mask].detach().cpu().numpy().astype(np.float32, copy=False),
        )
        np.save(
            _tp_sidecar_path(out_cat_dir, fname),
            good_by_file[fname][keep_mask].detach().cpu().numpy().astype(np.int32, copy=False),
        )
        np.save(
            _fp_sidecar_path(out_cat_dir, fname),
            bad_by_file[fname][keep_mask].detach().cpu().numpy().astype(np.int32, copy=False),
        )
        files_saved += 1
        vectors_saved += int(kept.shape[0])
    return files_saved, vectors_saved


def load_scored_class_entries(cat_dir: str):
    file_refs: list[tuple[str, str]] = []
    scores_by_file: dict[str, torch.Tensor] = {}
    good_by_file: dict[str, torch.Tensor | None] = {}
    bad_by_file: dict[str, torch.Tensor | None] = {}
    for fname in sorted(f for f in os.listdir(cat_dir) if f.endswith(".pt")):
        feat_path = os.path.join(cat_dir, fname)
        score_path = _score_sidecar_path(cat_dir, fname)
        tp_path = _tp_sidecar_path(cat_dir, fname)
        fp_path = _fp_sidecar_path(cat_dir, fname)
        if not os.path.exists(score_path):
            continue
        scores = torch.from_numpy(np.load(score_path)).float()
        if scores.ndim != 1:
            continue

        good = None
        bad = None
        if os.path.exists(tp_path) and os.path.exists(fp_path):
            good = torch.from_numpy(np.load(tp_path)).to(torch.int32)
            bad = torch.from_numpy(np.load(fp_path)).to(torch.int32)
            if good.ndim != 1 or bad.ndim != 1:
                continue
            if not (len(scores) == len(good) == len(bad)):
                continue
        file_refs.append((fname, feat_path))
        scores_by_file[fname] = scores
        good_by_file[fname] = good
        bad_by_file[fname] = bad
    return file_refs, scores_by_file, good_by_file, bad_by_file


def _accumulate_previous_class_stats(report: dict, cat_name: str, totals: dict[str, int]) -> None:
    prev = report["per_class"].get(cat_name, {})
    totals["in"] += prev.get("total_features", 0)
    totals["out"] += prev.get("kept_features", 0)
    totals["tp"] += prev.get("tp", 0) or 0
    totals["fp"] += prev.get("fp", 0) or 0
    totals["scored"] += prev.get("scored_features", 0)


def score_single_reference_class(
    grouped_bank: dict[int, list[tuple[str, str]]],
    device: str,
    query_chunk: int,
    min_matches: int,
    class_name: str | None = None,
):
    """
    Fallback score for one-shot banks.

    Cross-image ICCD cannot be evaluated when the bank has only one source
    image. Retain the occupancy-gated foreground descriptors without inventing
    a within-image reliability estimate. Unit scores and synthetic match counts
    are bookkeeping values for the shared scored-bank format only.
    """
    source_ids = sorted(grouped_bank)
    file_refs: list[tuple[str, str]] = []
    for image_id in source_ids:
        file_refs.extend(grouped_bank[image_id])
    if len(source_ids) != 1:
        raise ValueError("single-reference fallback requires exactly one source image.")

    source_entries = load_source_entries(file_refs)
    scores_by_file: dict[str, torch.Tensor] = {}
    good_by_file: dict[str, torch.Tensor] = {}
    bad_by_file: dict[str, torch.Tensor] = {}
    if not source_entries:
        return file_refs, scores_by_file, good_by_file, bad_by_file

    del device, query_chunk, class_name
    source_feats = torch.cat([feats for _, feats in source_entries], dim=0)
    scores = torch.ones(source_feats.shape[0], dtype=torch.float32)
    good = torch.zeros(source_feats.shape[0], dtype=torch.int32)
    bad = torch.zeros(source_feats.shape[0], dtype=torch.int32)
    good.fill_(max(1, min_matches))

    offset = 0
    for fname, feats in source_entries:
        n = feats.shape[0]
        scores_by_file[fname] = scores[offset: offset + n]
        good_by_file[fname] = good[offset: offset + n]
        bad_by_file[fname] = bad[offset: offset + n]
        offset += n
    del source_feats, scores, good, bad, source_entries
    gc.collect()
    return file_refs, scores_by_file, good_by_file, bad_by_file


def score_class(
    grouped_bank: dict[int, list[tuple[str, str]]],
    target_worklist: list[tuple[int, str, dict, list[dict]]],
    model,
    device: str,
    query_chunk: int,
    sim_floor: float,
    target_batch_size: int,
    num_workers: int,
    min_matches: int,
    class_name: str | None = None,
):
    source_ids = sorted(grouped_bank)
    if len(source_ids) == 1:
        return score_single_reference_class(
            grouped_bank=grouped_bank,
            device=device,
            query_chunk=query_chunk,
            min_matches=min_matches,
            class_name=class_name,
        )

    file_refs = []
    scores_by_file = {}
    good_by_file = {}
    bad_by_file = {}
    for image_id in source_ids:
        file_refs.extend(grouped_bank[image_id])

    prepared_target_batches = prepare_target_batches(
        model=model,
        device=device,
        target_worklist=target_worklist,
        target_batch_size=target_batch_size,
        num_workers=num_workers,
        class_name=class_name,
    )
    source_pbar = tqdm(source_ids, desc=f"  Source {class_name}" if class_name else "  Source images", leave=False)
    for source_image_id in source_pbar:
        source_entries = load_source_entries(grouped_bank[source_image_id])
        if not source_entries:
            continue
        source_feats = torch.cat([feats for _, feats in source_entries], dim=0)
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
        total = good + bad
        scores = torch.full((source_feats.shape[0],), -1.0, dtype=torch.float32)
        any_match_mask = total >= 1
        scores[any_match_mask] = good[any_match_mask].float() / total[any_match_mask].float()
        offset = 0
        for fname, feats in source_entries:
            n = feats.shape[0]
            scores_by_file[fname] = scores[offset: offset + n]
            good_by_file[fname] = good[offset: offset + n]
            bad_by_file[fname] = bad[offset: offset + n]
            offset += n
        del source_feats_dev, source_feats, good, bad, scores, source_entries
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    source_pbar.close()
    return file_refs, scores_by_file, good_by_file, bad_by_file


def run_score_bank(
    input_dir: str,
    output_dir: str,
    train_ann_file: str,
    image_dir: str,
    keep_threshold: float | None,
    min_matches: int,
    query_chunk: int,
    sim_floor: float,
    target_batch_size: int,
    support_image_limit: int | None,
    num_workers: int,
    selection_mode: str,
    top_k_features: int | None,
    resume: bool,
):
    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    images, anns_by_image, categories, image_ids_by_cat = load_annotations(train_ann_file)
    class_dirs = sorted(e for e in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, e)) and not e.startswith("_"))
    report_path = os.path.join(output_dir, "scored_feature_bank_report.json")
    report = load_json(
        report_path,
        {
            "config": {
                "input_dir": input_dir,
                "output_dir": output_dir,
                "train_ann_file": train_ann_file,
                "image_dir": image_dir,
                "score_keep_threshold": keep_threshold,
                "min_matches": min_matches,
                "query_chunk": query_chunk,
                "sim_floor": sim_floor,
                "target_batch_size": target_batch_size,
                "support_image_limit": support_image_limit,
                "num_workers": num_workers,
                "selection_mode": selection_mode,
                "top_k_features": top_k_features,
            },
            "per_class": {},
            "summary": {},
        },
    )
    os.makedirs(output_dir, exist_ok=True)
    totals = {"in": 0, "out": 0, "tp": 0, "fp": 0, "scored": 0}
    skipped = 0

    for cat_name in tqdm(class_dirs, desc="Classes"):
        out_cat_dir = os.path.join(output_dir, cat_name)
        if resume and os.path.isdir(out_cat_dir) and any(f.endswith(".pt") for f in os.listdir(out_cat_dir)):
            skipped += 1
            _accumulate_previous_class_stats(report, cat_name, totals)
            continue
        if cat_name in report["per_class"] and resume:
            skipped += 1
            _accumulate_previous_class_stats(report, cat_name, totals)
            continue

        cat_id = next((cid for cid, cat in categories.items() if get_category_name(categories, cid) == cat_name), None)
        if cat_id is None:
            continue
        grouped_bank = index_class_bank_grouped(os.path.join(input_dir, cat_name))
        if not grouped_bank:
            continue
        support_image_ids = _shared_support_image_ids(
            grouped_bank=grouped_bank,
            annotated_image_ids=image_ids_by_cat.get(cat_id, []),
            support_image_limit=support_image_limit,
            class_name=cat_name,
        )
        support_bank = {image_id: grouped_bank[image_id] for image_id in support_image_ids}
        target_worklist = build_target_worklist(
            image_dir=image_dir,
            images=images,
            anns_by_image=anns_by_image,
            cat_id=cat_id,
            image_ids=support_image_ids,
        )
        single_reference_fallback = len(support_bank) == 1
        if len(target_worklist) < 2 and not single_reference_fallback:
            continue

        file_refs, scores_by_file, good_by_file, bad_by_file = score_class(
            grouped_bank=support_bank,
            target_worklist=target_worklist,
            model=model,
            device=device,
            query_chunk=query_chunk,
            sim_floor=sim_floor,
            target_batch_size=target_batch_size,
            num_workers=num_workers,
            min_matches=min_matches,
            class_name=cat_name,
        )

        file_entries = [(fname, scores_by_file[fname]) for fname, _ in file_refs if fname in scores_by_file]
        if not file_entries:
            continue
        scores = torch.cat([scores_by_file[fname] for fname, _ in file_refs if fname in scores_by_file])
        good = torch.cat([good_by_file[fname] for fname, _ in file_refs if fname in good_by_file])
        bad = torch.cat([bad_by_file[fname] for fname, _ in file_refs if fname in bad_by_file])
        total_matches = good + bad
        scored_mask = total_matches >= min_matches
        scores_for_filter = scores.clone()
        scores_for_filter[~scored_mask] = -1.0

        if keep_threshold is None:
            keep_mask = torch.ones_like(scores_for_filter, dtype=torch.bool)
        else:
            keep_mask = _apply_top_k_features(scores_for_filter, keep_threshold, top_k_features)
        keep_by_file = _split_scores_by_file(file_entries, keep_mask)
        files_saved, vectors_saved = save_scored_bank_from_disk(
            out_cat_dir=out_cat_dir,
            file_refs=file_refs,
            keep_by_file=keep_by_file,
            scores_by_file=scores_by_file,
            good_by_file=good_by_file,
            bad_by_file=bad_by_file,
        )

        kept_tp = int(good[keep_mask].sum().item())
        kept_fp = int(bad[keep_mask].sum().item())
        kept_scores = scores_for_filter[keep_mask & (scores_for_filter >= 0)]
        total_features = int(scores_for_filter.shape[0])
        kept_features = int(keep_mask.sum().item())

        report["per_class"][cat_name] = {
            "status": "scored_bank",
            "total_features": total_features,
            "kept_features": kept_features,
            "removed_features": total_features - kept_features,
            "removal_pct": round(100.0 * (total_features - kept_features) / max(total_features, 1), 1),
            "tp": kept_tp,
            "fp": kept_fp,
            "precision": (kept_tp / (kept_tp + kept_fp)) if (kept_tp + kept_fp) > 0 else 0.0,
            "scored_features": int(scored_mask.sum().item()),
            "unscored_features": int((~scored_mask).sum().item()),
            "score_mean": round(float(kept_scores.mean().item()), 4) if kept_scores.numel() else None,
            "score_median": round(float(kept_scores.median().item()), 4) if kept_scores.numel() else None,
            "score_min": round(float(kept_scores.min().item()), 4) if kept_scores.numel() else None,
            "score_q75": round(float(torch.quantile(kept_scores.float(), 0.75).item()), 4) if kept_scores.numel() else None,
            "score_max": round(float(kept_scores.max().item()), 4) if kept_scores.numel() else None,
            "files_saved": files_saved,
            "vectors_saved": vectors_saved,
            "score_keep_threshold": keep_threshold,
            "per_feature_scores_saved": True,
            "per_feature_tp_saved": True,
            "per_feature_fp_saved": True,
            "score_sidecar_suffix": ".scores.npy",
            "tp_sidecar_suffix": ".tp.npy",
            "fp_sidecar_suffix": ".fp.npy",
            "target_images_used": len(target_worklist),
            "single_reference_fallback": single_reference_fallback,
            "score_type": "single_reference_raw_occupancy" if single_reference_fallback else "cross_image_retrieval_precision",
            "match_counts_are_synthetic": single_reference_fallback,
            "selection_mode": selection_mode,
            "support_images_used": len(support_image_ids),
            "support_image_ids": support_image_ids,
            "top_k_features": top_k_features,
        }
        save_json_atomic(report_path, report)

        totals["in"] += total_features
        totals["out"] += kept_features
        totals["tp"] += kept_tp
        totals["fp"] += kept_fp
        totals["scored"] += int(scored_mask.sum().item())
        del grouped_bank, target_worklist, file_refs, file_entries, scores_by_file, good_by_file, bad_by_file, scores, good, bad, scores_for_filter, keep_mask
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    report["summary"] = {
        "classes_processed": len(report["per_class"]),
        "classes_skipped": skipped,
        "total_features_in": totals["in"],
        "total_features_out": totals["out"],
        "total_removed": totals["in"] - totals["out"],
        "removal_pct": round(100.0 * (totals["in"] - totals["out"]) / max(totals["in"], 1), 2) if totals["in"] else 0.0,
        "tp": totals["tp"],
        "fp": totals["fp"],
        "precision": (totals["tp"] / (totals["tp"] + totals["fp"])) if (totals["tp"] + totals["fp"]) > 0 else 0.0,
        "scored_features": totals["scored"],
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    save_json_atomic(report_path, report)
    return report


def _build_keep_mask_from_scored_bank(
    method: str,
    scores_for_filter: torch.Tensor,
    keep_threshold: float | None,
    top_k_features: int | None,
):
    meta: dict[str, object] = {}
    if method == "fixed":
        if keep_threshold is None:
            keep_mask = torch.ones_like(scores_for_filter, dtype=torch.bool)
        else:
            keep_mask = _apply_top_k_features(scores_for_filter, keep_threshold, top_k_features)
        meta["status"] = "fixed"
        return keep_mask, meta
    if method == "adaptive_q75":
        keep_mask, q75_value, adaptive_threshold, survivors_before_top_k = _apply_adaptive_top_k_features(
            scores_for_filter=scores_for_filter,
            top_k_features=top_k_features,
        )
        meta.update(
            {
                "status": "adaptive_q75_topk",
                "adaptive_q75": q75_value,
                "adaptive_threshold": adaptive_threshold,
                "survivors_before_top_k": survivors_before_top_k,
            }
        )
        return keep_mask, meta
    raise ValueError(f"Unknown filtering method: {method}")


def run_filter_from_scored_bank(
    input_dir: str,
    output_dir: str,
    method: str,
    keep_threshold: float | None,
    top_k_features: int | None,
    resume: bool,
):
    if method not in {"fixed", "adaptive_q75"}:
        raise ValueError(f"Unknown filtering method: {method}")
    t_start = time.time()
    class_dirs = sorted(e for e in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, e)) and not e.startswith("_"))
    report_path = os.path.join(output_dir, "intra_class_filter_report.json")
    report = load_json(
        report_path,
        {
            "config": {
                "input_dir": input_dir,
                "output_dir": output_dir,
                "method": method,
                "keep_threshold": keep_threshold,
                "top_k_features": top_k_features,
            },
            "per_class": {},
            "summary": {},
        },
    )
    os.makedirs(output_dir, exist_ok=True)
    scored_report = load_json(os.path.join(input_dir, "scored_feature_bank_report.json"), {})
    scored_report_by_class = scored_report.get("per_class", {}) if isinstance(scored_report, dict) else {}
    totals = {"in": 0, "out": 0, "tp": 0, "fp": 0, "scored": 0}
    skipped = 0

    for cat_name in tqdm(class_dirs, desc="Classes"):
        out_cat_dir = os.path.join(output_dir, cat_name)
        if resume and os.path.isdir(out_cat_dir) and any(f.endswith(".pt") for f in os.listdir(out_cat_dir)):
            skipped += 1
            _accumulate_previous_class_stats(report, cat_name, totals)
            continue
        if cat_name in report["per_class"] and resume:
            skipped += 1
            _accumulate_previous_class_stats(report, cat_name, totals)
            continue

        in_cat_dir = os.path.join(input_dir, cat_name)
        if not os.path.isdir(in_cat_dir):
            continue
        scored_meta = scored_report_by_class.get(cat_name, {})
        single_reference_fallback = bool(scored_meta.get("single_reference_fallback", False))
        file_refs, scores_by_file, good_by_file, bad_by_file = load_scored_class_entries(in_cat_dir)
        if not file_refs:
            continue

        scores = torch.cat([scores_by_file[fname] for fname, _ in file_refs if fname in scores_by_file])
        has_match_counts = all(good_by_file.get(fname) is not None and bad_by_file.get(fname) is not None for fname, _ in file_refs)
        good = (
            torch.cat([good_by_file[fname] for fname, _ in file_refs if good_by_file.get(fname) is not None])
            if has_match_counts
            else None
        )
        bad = (
            torch.cat([bad_by_file[fname] for fname, _ in file_refs if bad_by_file.get(fname) is not None])
            if has_match_counts
            else None
        )
        scores_for_filter = scores.clone()
        if single_reference_fallback:
            keep_mask = _single_reference_keep_mask(len(scores_for_filter), top_k_features)
            meta = {"status": "single_reference_raw_occupancy"}
        else:
            keep_mask, meta = _build_keep_mask_from_scored_bank(
                method=method,
                scores_for_filter=scores_for_filter,
                keep_threshold=keep_threshold,
                top_k_features=top_k_features,
            )
        keep_by_file = _split_scores_by_file([(fname, scores_by_file[fname]) for fname, _ in file_refs], keep_mask)
        files_saved, vectors_saved = save_class_outputs_from_disk(out_cat_dir, file_refs, keep_by_file, scores_by_file)

        kept_tp = int(good[keep_mask].sum().item()) if has_match_counts and good is not None else None
        kept_fp = int(bad[keep_mask].sum().item()) if has_match_counts and bad is not None else None
        scored_values = scores_for_filter[scores_for_filter >= 0]
        if scored_values.numel():
            scored_values_float = scored_values.float()
            score_std = round(float(scored_values_float.std(unbiased=False).item()), 4)
            score_q25 = round(float(torch.quantile(scored_values_float, 0.25).item()), 4)
            score_q75 = round(float(torch.quantile(scored_values_float, 0.75).item()), 4)
        else:
            score_std = None
            score_q25 = None
            score_q75 = None
        total_features = int(scores_for_filter.shape[0])
        kept_features = int(keep_mask.sum().item())

        class_report = {
            "status": meta.get("status", method),
            "total_features": total_features,
            "kept_features": kept_features,
            "removed_features": total_features - kept_features,
            "removal_pct": round(100.0 * (total_features - kept_features) / max(total_features, 1), 1),
            "tp": kept_tp,
            "fp": kept_fp,
            "precision": (
                (kept_tp / (kept_tp + kept_fp))
                if kept_tp is not None and kept_fp is not None and (kept_tp + kept_fp) > 0
                else None
            ),
            "scored_features": total_features,
            "unscored_features": 0,
            "score_mean": round(float(scored_values.mean().item()), 4) if scored_values.numel() else None,
            "score_median": round(float(scored_values.median().item()), 4) if scored_values.numel() else None,
            "score_std": score_std,
            "score_min": round(float(scored_values.min().item()), 4) if scored_values.numel() else None,
            "score_q25": score_q25,
            "score_q75": score_q75,
            "score_max": round(float(scored_values.max().item()), 4) if scored_values.numel() else None,
            "files_saved": files_saved,
            "vectors_saved": vectors_saved,
            "per_feature_scores_saved": True,
            "per_feature_match_counts_saved": has_match_counts,
            "single_reference_fallback": single_reference_fallback,
            "score_type": scored_meta.get(
                "score_type",
                "single_reference_raw_occupancy" if single_reference_fallback else "cross_image_retrieval_precision",
            ),
            "match_counts_are_synthetic": bool(scored_meta.get("match_counts_are_synthetic", False)),
            "score_sidecar_suffix": ".scores.npy",
            "top_k_features": top_k_features,
        }
        if "adaptive_q75" in meta:
            class_report["adaptive_q75"] = round(float(meta["adaptive_q75"]), 4) if meta["adaptive_q75"] is not None else None
        if "adaptive_threshold" in meta:
            class_report["adaptive_threshold"] = round(float(meta["adaptive_threshold"]), 4) if meta["adaptive_threshold"] is not None else None
        if "survivors_before_top_k" in meta:
            class_report["survivors_before_top_k"] = int(meta["survivors_before_top_k"])
        report["per_class"][cat_name] = class_report
        save_json_atomic(report_path, report)

        totals["in"] += total_features
        totals["out"] += kept_features
        totals["tp"] += kept_tp or 0
        totals["fp"] += kept_fp or 0
        totals["scored"] += total_features
        del file_refs, scores_by_file, good_by_file, bad_by_file, scores, good, bad, scores_for_filter, keep_mask

    report["summary"] = {
        "classes_processed": len(report["per_class"]),
        "classes_skipped": skipped,
        "total_features_in": totals["in"],
        "total_features_out": totals["out"],
        "total_removed": totals["in"] - totals["out"],
        "removal_pct": round(100.0 * (totals["in"] - totals["out"]) / max(totals["in"], 1), 2) if totals["in"] else 0.0,
        "tp": totals["tp"],
        "fp": totals["fp"],
        "precision": (totals["tp"] / (totals["tp"] + totals["fp"])) if (totals["tp"] + totals["fp"]) > 0 else None,
        "scored_features": totals["scored"],
        "elapsed_seconds": round(time.time() - t_start, 1),
        "method": method,
    }
    save_json_atomic(report_path, report)
    return report
