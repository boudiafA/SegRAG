"""
Native SegRAG implementation of intra-class feature-bank filtering.

This module preserves the current optimized behavior:
- resumable per-class filtering
- target-batch preparation once per class
- DINOv3 dense feature extraction matching the current bank build
- support for `target-references` and `top-k-images` selection modes
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


def _limit_target_image_ids(image_ids: list[int], target_image_limit: int | None) -> list[int]:
    return list(image_ids if target_image_limit is None else image_ids[:target_image_limit])


def _split_scores_by_file(file_entries: list[tuple[str, torch.Tensor]], flat_keep_mask: torch.Tensor):
    keep_by_file = {}
    offset = 0
    for fname, feats in file_entries:
        n = feats.shape[0]
        keep_by_file[fname] = flat_keep_mask[offset: offset + n]
        offset += n
    return keep_by_file


def _compute_keep_mask(scores: torch.Tensor, keep_threshold: float | None, min_keep_ratio: float):
    if keep_threshold is None:
        return torch.ones_like(scores, dtype=torch.bool), False
    scored_mask = scores >= 0
    keep_mask = (scores >= keep_threshold) | (~scored_mask)
    min_required = max(1, int(min_keep_ratio * scores.shape[0]))
    if int(keep_mask.sum().item()) < min_required:
        rank_scores = scores.clone()
        rank_scores[~scored_mask] = 0.5
        _, top_idx = rank_scores.topk(min_required)
        keep_mask = torch.zeros_like(keep_mask)
        keep_mask[top_idx] = True
        floor_applied = True
    else:
        floor_applied = False
    return keep_mask, floor_applied


def _apply_target_references_cap(keep_mask: torch.Tensor, scores_for_filter: torch.Tensor, target_references: int | None):
    if target_references is None:
        return keep_mask, False
    kept_count = int(keep_mask.sum().item())
    if kept_count <= target_references:
        return keep_mask, False
    capped_keep = torch.zeros_like(keep_mask)
    kept_indices = keep_mask.nonzero(as_tuple=True)[0]
    kept_scores = scores_for_filter[kept_indices].clone()
    kept_scores[kept_scores < 0] = 0.5
    _, top_idx = kept_scores.topk(target_references)
    capped_keep[kept_indices[top_idx]] = True
    return capped_keep, True


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


def _estimate_cluster_count(num_features: int) -> int:
    if num_features < 200:
        return 1
    estimate = int(round(np.sqrt(num_features / 250.0)))
    estimate = max(2, estimate)
    estimate = min(estimate, 8)
    estimate = min(estimate, max(1, num_features // 25))
    return max(1, estimate)


def _load_flat_feature_bank(file_refs: list[tuple[str, str]]) -> torch.Tensor:
    tensors = []
    for _fname, path in file_refs:
        feats = torch.load(path, map_location="cpu", weights_only=True).float()
        tensors.append(feats)
    return torch.cat(tensors, dim=0) if tensors else torch.empty((0, 0), dtype=torch.float32)


def _apply_clustered_adaptive_top_k_features(
    features_for_filter: torch.Tensor,
    scores_for_filter: torch.Tensor,
    top_k_features: int | None,
    n_clusters: int | str,
    min_cluster_size: int,
) -> tuple[torch.Tensor, dict]:
    from sklearn.cluster import KMeans

    keep_mask = torch.zeros_like(scores_for_filter, dtype=torch.bool)
    scored_idx = (scores_for_filter >= 0).nonzero(as_tuple=True)[0]
    if scored_idx.numel() == 0:
        return keep_mask, {
            "estimated_k": 0,
            "used_k": 0,
            "discarded_small_clusters": 0,
            "kept_clusters": 0,
            "cluster_thresholds": [],
            "survivors_before_top_k": 0,
        }

    scored_features = features_for_filter[scored_idx].cpu().numpy().astype(np.float32, copy=False)
    scored_scores = scores_for_filter[scored_idx].cpu().numpy().astype(np.float32, copy=False)
    estimated_k = _estimate_cluster_count(len(scored_scores))
    used_k = estimated_k if n_clusters == "auto" else int(n_clusters)
    used_k = max(1, min(used_k, len(scored_scores)))

    kmeans = KMeans(n_clusters=used_k, random_state=42, n_init="auto")
    cluster_labels = kmeans.fit_predict(scored_features)

    kept_scored_mask = np.zeros(len(scored_scores), dtype=bool)
    cluster_thresholds: list[float] = []
    discarded_small_clusters = 0
    kept_clusters = 0

    for cluster_id in range(used_k):
        cluster_mask = cluster_labels == cluster_id
        cluster_size = int(cluster_mask.sum())
        if cluster_size < min_cluster_size:
            discarded_small_clusters += 1
            continue

        cluster_scores = scored_scores[cluster_mask]
        cluster_threshold = float(np.clip(np.percentile(cluster_scores, 75) * 0.90, 0.65, 0.82))
        cluster_thresholds.append(cluster_threshold)
        quality_mask = cluster_scores >= cluster_threshold

        cluster_indices = np.nonzero(cluster_mask)[0]
        kept_scored_mask[cluster_indices[quality_mask]] = True
        kept_clusters += 1

    survivors_before_top_k = int(kept_scored_mask.sum())
    if survivors_before_top_k > 0 and top_k_features is not None and top_k_features > 0 and survivors_before_top_k > top_k_features:
        kept_indices = np.nonzero(kept_scored_mask)[0]
        kept_scores = scored_scores[kept_indices]
        top_idx = np.argsort(kept_scores)[::-1][:top_k_features]
        limited_mask = np.zeros_like(kept_scored_mask)
        limited_mask[kept_indices[top_idx]] = True
        kept_scored_mask = limited_mask

    keep_mask[scored_idx] = torch.from_numpy(kept_scored_mask)
    return keep_mask, {
        "estimated_k": estimated_k,
        "used_k": used_k,
        "discarded_small_clusters": discarded_small_clusters,
        "kept_clusters": kept_clusters,
        "cluster_thresholds": cluster_thresholds,
        "survivors_before_top_k": survivors_before_top_k,
    }


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
    image. In that case, score each foreground descriptor by its maximum cosine
    similarity to any other foreground descriptor from the same source image.
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

    source_feats = torch.cat([feats for _, feats in source_entries], dim=0)
    scores = torch.full((source_feats.shape[0],), -1.0, dtype=torch.float32)
    if source_feats.shape[0] > 1:
        feats_dev = source_feats.to(device)
        pbar = tqdm(
            range(0, source_feats.shape[0], query_chunk),
            desc=f"  Fallback {class_name}" if class_name else "  Fallback source",
            leave=False,
        )
        for start in pbar:
            end = min(start + query_chunk, source_feats.shape[0])
            sims = feats_dev[start:end] @ feats_dev.T
            rows = torch.arange(end - start, device=device)
            cols = torch.arange(start, end, device=device)
            sims[rows, cols] = -float("inf")
            best_sims = sims.max(dim=1).values
            scores[start:end] = best_sims.float().cpu()
            del sims, best_sims, rows, cols
        pbar.close()
        del feats_dev

    valid = scores >= 0
    good = torch.zeros(source_feats.shape[0], dtype=torch.int32)
    bad = torch.zeros(source_feats.shape[0], dtype=torch.int32)
    # These synthetic counts let the existing filtering path treat non-negative
    # fallback scores as scored vectors while preserving the score values.
    good[valid] = max(1, min_matches)

    offset = 0
    for fname, feats in source_entries:
        n = feats.shape[0]
        scores_by_file[fname] = scores[offset: offset + n]
        good_by_file[fname] = good[offset: offset + n]
        bad_by_file[fname] = bad[offset: offset + n]
        offset += n
    del source_feats, scores, good, bad, source_entries
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
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
    keep_threshold: float | None,
    min_matches: int,
    target_references: int | None,
    early_accept: bool,
    selection_mode: str,
    max_source_images: int | None,
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
    accepted_so_far = 0
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
    if selection_mode == "top-k-images" and max_source_images is not None:
        source_ids = source_ids[:max_source_images]
    source_pbar = tqdm(source_ids, desc=f"  Source {class_name}" if class_name else "  Source images", leave=False)
    for source_image_id in source_pbar:
        if selection_mode == "target-references" and early_accept and target_references is not None and accepted_so_far >= target_references:
            break
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
        if selection_mode == "target-references" and early_accept and target_references is not None:
            if keep_threshold is None:
                accepted_mask = total >= min_matches
            else:
                accepted_mask = (total >= min_matches) & (scores >= keep_threshold)
            remaining = target_references - accepted_so_far
            if remaining <= 0:
                break
            if int(accepted_mask.sum().item()) > remaining:
                accepted_indices = accepted_mask.nonzero(as_tuple=True)[0][:remaining]
                limited_mask = torch.zeros_like(accepted_mask)
                limited_mask[accepted_indices] = True
                accepted_mask = limited_mask
            accepted_so_far += int(accepted_mask.sum().item())
            reject_mask = ~accepted_mask
            scores[reject_mask] = 0.0
            good[reject_mask] = 0
            bad[reject_mask] = min_matches
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


def run_filter(
    input_dir: str,
    output_dir: str,
    train_ann_file: str,
    image_dir: str,
    keep_threshold: float | None,
    min_matches: int,
    min_keep_ratio: float,
    filter_mode: str,
    query_chunk: int,
    sim_floor: float,
    target_batch_size: int,
    target_image_limit: int | None,
    target_references: int | None,
    num_workers: int,
    early_accept: bool,
    selection_mode: str,
    max_source_images: int | None,
    top_k_features: int | None,
    resume: bool,
):
    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    images, anns_by_image, categories, image_ids_by_cat = load_annotations(train_ann_file)
    class_dirs = sorted(e for e in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, e)) and not e.startswith("_"))
    report_path = os.path.join(output_dir, "intra_class_filter_report.json")
    report = load_json(
        report_path,
        {
            "config": {
                "input_dir": input_dir,
                "output_dir": output_dir,
                "train_ann_file": train_ann_file,
                "image_dir": image_dir,
                "keep_threshold": keep_threshold,
                "min_matches": min_matches,
                "min_keep_ratio": min_keep_ratio,
                "filter_mode": filter_mode,
                "query_chunk": query_chunk,
                "sim_floor": sim_floor,
                "target_batch_size": target_batch_size,
                "target_image_limit": target_image_limit,
                "target_references": target_references,
                "num_workers": num_workers,
                "early_accept": early_accept,
                "selection_mode": selection_mode,
                "max_source_images": max_source_images,
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
        limited_target_ids = _limit_target_image_ids(image_ids_by_cat.get(cat_id, []), target_image_limit)
        target_worklist = build_target_worklist(
            image_dir=image_dir,
            images=images,
            anns_by_image=anns_by_image,
            cat_id=cat_id,
            image_ids=limited_target_ids,
        )
        single_reference_fallback = len(grouped_bank) == 1
        if len(target_worklist) < 2 and not single_reference_fallback:
            continue

        file_refs, scores_by_file, good_by_file, bad_by_file = score_class(
            grouped_bank=grouped_bank,
            target_worklist=target_worklist,
            model=model,
            device=device,
            query_chunk=query_chunk,
            sim_floor=sim_floor,
            target_batch_size=target_batch_size,
            num_workers=num_workers,
            keep_threshold=keep_threshold,
            min_matches=min_matches,
            target_references=target_references,
            early_accept=early_accept,
            selection_mode=selection_mode,
            max_source_images=max_source_images,
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

        capped_by_target_refs = False
        if filter_mode == "reweight":
            os.makedirs(out_cat_dir, exist_ok=True)
            for fname, path in file_refs:
                feats = torch.load(path, map_location="cpu", weights_only=True).float()
                torch.save(feats, os.path.join(out_cat_dir, fname))
                file_scores = scores_by_file.get(fname)
                if file_scores is not None:
                    np.save(
                        _score_sidecar_path(out_cat_dir, fname),
                        file_scores.detach().cpu().numpy().astype(np.float32, copy=False),
                    )
            weights_dir = os.path.join(output_dir, "_weights")
            os.makedirs(weights_dir, exist_ok=True)
            weight_tensor = scores_for_filter.clone()
            weight_tensor[weight_tensor < 0] = 0.5
            torch.save(weight_tensor.clamp(0.0, 1.0), os.path.join(weights_dir, f"{cat_name}.pt"))
            keep_mask = torch.ones_like(scores_for_filter, dtype=torch.bool)
            floor_applied = False
            files_saved = len(file_refs)
            vectors_saved = int(sum(scores_by_file[fname].shape[0] for fname, _ in file_refs if fname in scores_by_file))
        else:
            if keep_threshold is None:
                keep_mask = torch.ones_like(scores_for_filter, dtype=torch.bool)
                floor_applied = False
            if selection_mode == "top-k-images":
                if keep_threshold is not None:
                    keep_mask = _apply_top_k_features(scores_for_filter, keep_threshold, top_k_features)
                    floor_applied = False
            else:
                if keep_threshold is not None:
                    keep_mask, floor_applied = _compute_keep_mask(scores_for_filter, keep_threshold, min_keep_ratio)
                    keep_mask, capped_by_target_refs = _apply_target_references_cap(keep_mask, scores_for_filter, target_references)
            keep_by_file = _split_scores_by_file(file_entries, keep_mask)
            files_saved, vectors_saved = save_class_outputs_from_disk(out_cat_dir, file_refs, keep_by_file, scores_by_file)
        if filter_mode == "reweight":
            capped_by_target_refs = False

        kept_tp = int(good[keep_mask].sum().item())
        kept_fp = int(bad[keep_mask].sum().item())
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

        report["per_class"][cat_name] = {
            "status": (
                "reweight"
                if filter_mode == "reweight"
                else (
                    "threshold_disabled"
                    if keep_threshold is None
                    else (
                    "top_k_images"
                    if selection_mode == "top-k-images"
                    else (
                        "early_accept"
                        if early_accept and target_references is not None and kept_features <= target_references
                        else ("target_ref_cap" if capped_by_target_refs else ("floor_applied" if floor_applied else "filtered"))
                    )
                    )
                )
            ),
            "total_features": total_features,
            "kept_features": kept_features,
            "removed_features": total_features - kept_features,
            "removal_pct": round(100.0 * (total_features - kept_features) / max(total_features, 1), 1),
            "tp": kept_tp,
            "fp": kept_fp,
            "precision": (kept_tp / (kept_tp + kept_fp)) if (kept_tp + kept_fp) > 0 else 0.0,
            "scored_features": int(scored_mask.sum().item()),
            "unscored_features": int((~scored_mask).sum().item()),
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
            "score_sidecar_suffix": ".scores.npy",
            "target_images_used": len(target_worklist),
            "single_reference_fallback": single_reference_fallback,
            "score_type": "within_image_max_similarity" if single_reference_fallback else "cross_image_retrieval_precision",
            "match_counts_are_synthetic": single_reference_fallback,
            "target_references_cap": target_references,
            "selection_mode": selection_mode,
            "max_source_images": max_source_images,
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
    target_image_limit: int | None,
    num_workers: int,
    selection_mode: str,
    max_source_images: int | None,
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
                "target_image_limit": target_image_limit,
                "num_workers": num_workers,
                "selection_mode": selection_mode,
                "max_source_images": max_source_images,
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
        limited_target_ids = _limit_target_image_ids(image_ids_by_cat.get(cat_id, []), target_image_limit)
        target_worklist = build_target_worklist(
            image_dir=image_dir,
            images=images,
            anns_by_image=anns_by_image,
            cat_id=cat_id,
            image_ids=limited_target_ids,
        )
        single_reference_fallback = len(grouped_bank) == 1
        if len(target_worklist) < 2 and not single_reference_fallback:
            continue

        file_refs, scores_by_file, good_by_file, bad_by_file = score_class(
            grouped_bank=grouped_bank,
            target_worklist=target_worklist,
            model=model,
            device=device,
            query_chunk=query_chunk,
            sim_floor=sim_floor,
            target_batch_size=target_batch_size,
            num_workers=num_workers,
            keep_threshold=keep_threshold,
            min_matches=min_matches,
            target_references=None,
            early_accept=False,
            selection_mode=selection_mode,
            max_source_images=max_source_images,
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
            "score_type": "within_image_max_similarity" if single_reference_fallback else "cross_image_retrieval_precision",
            "match_counts_are_synthetic": single_reference_fallback,
            "selection_mode": selection_mode,
            "max_source_images": max_source_images,
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


def run_filter_adaptive(
    input_dir: str,
    output_dir: str,
    train_ann_file: str,
    image_dir: str,
    min_matches: int,
    query_chunk: int,
    sim_floor: float,
    target_batch_size: int,
    target_image_limit: int | None,
    num_workers: int,
    selection_mode: str,
    max_source_images: int | None,
    top_k_features: int | None,
    resume: bool,
):
    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    images, anns_by_image, categories, image_ids_by_cat = load_annotations(train_ann_file)
    class_dirs = sorted(e for e in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, e)) and not e.startswith("_"))
    report_path = os.path.join(output_dir, "intra_class_filter_report.json")
    report = load_json(
        report_path,
        {
            "config": {
                "input_dir": input_dir,
                "output_dir": output_dir,
                "train_ann_file": train_ann_file,
                "image_dir": image_dir,
                "filter_strategy": "adaptive_q75_topk",
                "adaptive_formula": "clip(q75 * 0.90, 0.65, 0.82)",
                "min_matches": min_matches,
                "query_chunk": query_chunk,
                "sim_floor": sim_floor,
                "target_batch_size": target_batch_size,
                "target_image_limit": target_image_limit,
                "num_workers": num_workers,
                "selection_mode": selection_mode,
                "max_source_images": max_source_images,
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
        limited_target_ids = _limit_target_image_ids(image_ids_by_cat.get(cat_id, []), target_image_limit)
        target_worklist = build_target_worklist(
            image_dir=image_dir,
            images=images,
            anns_by_image=anns_by_image,
            cat_id=cat_id,
            image_ids=limited_target_ids,
        )
        single_reference_fallback = len(grouped_bank) == 1
        if len(target_worklist) < 2 and not single_reference_fallback:
            continue

        file_refs, scores_by_file, good_by_file, bad_by_file = score_class(
            grouped_bank=grouped_bank,
            target_worklist=target_worklist,
            model=model,
            device=device,
            query_chunk=query_chunk,
            sim_floor=sim_floor,
            target_batch_size=target_batch_size,
            num_workers=num_workers,
            keep_threshold=None,
            min_matches=min_matches,
            target_references=None,
            early_accept=False,
            selection_mode=selection_mode,
            max_source_images=max_source_images,
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

        keep_mask, q75_value, adaptive_threshold, survivors_before_top_k = _apply_adaptive_top_k_features(
            scores_for_filter=scores_for_filter,
            top_k_features=top_k_features,
        )
        keep_by_file = _split_scores_by_file(file_entries, keep_mask)
        files_saved, vectors_saved = save_class_outputs_from_disk(out_cat_dir, file_refs, keep_by_file, scores_by_file)

        kept_tp = int(good[keep_mask].sum().item())
        kept_fp = int(bad[keep_mask].sum().item())
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

        report["per_class"][cat_name] = {
            "status": "adaptive_q75_topk",
            "total_features": total_features,
            "kept_features": kept_features,
            "removed_features": total_features - kept_features,
            "removal_pct": round(100.0 * (total_features - kept_features) / max(total_features, 1), 1),
            "tp": kept_tp,
            "fp": kept_fp,
            "precision": (kept_tp / (kept_tp + kept_fp)) if (kept_tp + kept_fp) > 0 else 0.0,
            "scored_features": int(scored_mask.sum().item()),
            "unscored_features": int((~scored_mask).sum().item()),
            "score_mean": round(float(scored_values.mean().item()), 4) if scored_values.numel() else None,
            "score_median": round(float(scored_values.median().item()), 4) if scored_values.numel() else None,
            "score_std": score_std,
            "score_min": round(float(scored_values.min().item()), 4) if scored_values.numel() else None,
            "score_q25": score_q25,
            "score_q75": score_q75,
            "score_max": round(float(scored_values.max().item()), 4) if scored_values.numel() else None,
            "adaptive_q75": round(q75_value, 4) if q75_value is not None else None,
            "adaptive_threshold": round(adaptive_threshold, 4) if adaptive_threshold is not None else None,
            "survivors_before_top_k": survivors_before_top_k,
            "files_saved": files_saved,
            "vectors_saved": vectors_saved,
            "per_feature_scores_saved": True,
            "score_sidecar_suffix": ".scores.npy",
            "target_images_used": len(target_worklist),
            "single_reference_fallback": single_reference_fallback,
            "score_type": "within_image_max_similarity" if single_reference_fallback else "cross_image_retrieval_precision",
            "match_counts_are_synthetic": single_reference_fallback,
            "selection_mode": selection_mode,
            "max_source_images": max_source_images,
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
        "filter_strategy": "adaptive_q75_topk",
    }
    save_json_atomic(report_path, report)
    return report


def run_filter_clustered_adaptive(
    input_dir: str,
    output_dir: str,
    train_ann_file: str,
    image_dir: str,
    min_matches: int,
    query_chunk: int,
    sim_floor: float,
    target_batch_size: int,
    target_image_limit: int | None,
    num_workers: int,
    selection_mode: str,
    max_source_images: int | None,
    top_k_features: int | None,
    n_clusters: int | str,
    min_cluster_size: int,
    resume: bool,
):
    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    images, anns_by_image, categories, image_ids_by_cat = load_annotations(train_ann_file)
    class_dirs = sorted(e for e in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, e)) and not e.startswith("_"))
    report_path = os.path.join(output_dir, "intra_class_filter_report.json")
    report = load_json(
        report_path,
        {
            "config": {
                "input_dir": input_dir,
                "output_dir": output_dir,
                "train_ann_file": train_ann_file,
                "image_dir": image_dir,
                "filter_strategy": "clustered_adaptive_q75_topk",
                "adaptive_formula": "cluster_threshold = clip(cluster_q75 * 0.90, 0.65, 0.82)",
                "cluster_strategy": "kmeans_on_scored_features",
                "n_clusters": n_clusters,
                "min_cluster_size": min_cluster_size,
                "min_matches": min_matches,
                "query_chunk": query_chunk,
                "sim_floor": sim_floor,
                "target_batch_size": target_batch_size,
                "target_image_limit": target_image_limit,
                "num_workers": num_workers,
                "selection_mode": selection_mode,
                "max_source_images": max_source_images,
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
        limited_target_ids = _limit_target_image_ids(image_ids_by_cat.get(cat_id, []), target_image_limit)
        target_worklist = build_target_worklist(
            image_dir=image_dir,
            images=images,
            anns_by_image=anns_by_image,
            cat_id=cat_id,
            image_ids=limited_target_ids,
        )
        single_reference_fallback = len(grouped_bank) == 1
        if len(target_worklist) < 2 and not single_reference_fallback:
            continue

        file_refs, scores_by_file, good_by_file, bad_by_file = score_class(
            grouped_bank=grouped_bank,
            target_worklist=target_worklist,
            model=model,
            device=device,
            query_chunk=query_chunk,
            sim_floor=sim_floor,
            target_batch_size=target_batch_size,
            num_workers=num_workers,
            keep_threshold=None,
            min_matches=min_matches,
            target_references=None,
            early_accept=False,
            selection_mode=selection_mode,
            max_source_images=max_source_images,
            class_name=cat_name,
        )

        file_entries = [(fname, scores_by_file[fname]) for fname, _ in file_refs if fname in scores_by_file]
        if not file_entries:
            continue
        features_for_filter = _load_flat_feature_bank(file_refs)
        scores = torch.cat([scores_by_file[fname] for fname, _ in file_refs if fname in scores_by_file])
        good = torch.cat([good_by_file[fname] for fname, _ in file_refs if fname in good_by_file])
        bad = torch.cat([bad_by_file[fname] for fname, _ in file_refs if fname in bad_by_file])
        total_matches = good + bad
        scored_mask = total_matches >= min_matches
        scores_for_filter = scores.clone()
        scores_for_filter[~scored_mask] = -1.0

        keep_mask, cluster_meta = _apply_clustered_adaptive_top_k_features(
            features_for_filter=features_for_filter,
            scores_for_filter=scores_for_filter,
            top_k_features=top_k_features,
            n_clusters=n_clusters,
            min_cluster_size=min_cluster_size,
        )
        keep_by_file = _split_scores_by_file(file_entries, keep_mask)
        files_saved, vectors_saved = save_class_outputs_from_disk(out_cat_dir, file_refs, keep_by_file, scores_by_file)

        kept_tp = int(good[keep_mask].sum().item())
        kept_fp = int(bad[keep_mask].sum().item())
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
        cluster_thresholds = cluster_meta["cluster_thresholds"]
        total_features = int(scores_for_filter.shape[0])
        kept_features = int(keep_mask.sum().item())

        report["per_class"][cat_name] = {
            "status": "clustered_adaptive_q75_topk",
            "total_features": total_features,
            "kept_features": kept_features,
            "removed_features": total_features - kept_features,
            "removal_pct": round(100.0 * (total_features - kept_features) / max(total_features, 1), 1),
            "tp": kept_tp,
            "fp": kept_fp,
            "precision": (kept_tp / (kept_tp + kept_fp)) if (kept_tp + kept_fp) > 0 else 0.0,
            "scored_features": int(scored_mask.sum().item()),
            "unscored_features": int((~scored_mask).sum().item()),
            "score_mean": round(float(scored_values.mean().item()), 4) if scored_values.numel() else None,
            "score_median": round(float(scored_values.median().item()), 4) if scored_values.numel() else None,
            "score_std": score_std,
            "score_min": round(float(scored_values.min().item()), 4) if scored_values.numel() else None,
            "score_q25": score_q25,
            "score_q75": score_q75,
            "score_max": round(float(scored_values.max().item()), 4) if scored_values.numel() else None,
            "estimated_k": cluster_meta["estimated_k"],
            "used_k": cluster_meta["used_k"],
            "min_cluster_size": min_cluster_size,
            "kept_clusters": cluster_meta["kept_clusters"],
            "discarded_small_clusters": cluster_meta["discarded_small_clusters"],
            "cluster_threshold_min": round(min(cluster_thresholds), 4) if cluster_thresholds else None,
            "cluster_threshold_mean": round(float(np.mean(cluster_thresholds)), 4) if cluster_thresholds else None,
            "cluster_threshold_max": round(max(cluster_thresholds), 4) if cluster_thresholds else None,
            "survivors_before_top_k": cluster_meta["survivors_before_top_k"],
            "files_saved": files_saved,
            "vectors_saved": vectors_saved,
            "per_feature_scores_saved": True,
            "score_sidecar_suffix": ".scores.npy",
            "target_images_used": len(target_worklist),
            "single_reference_fallback": single_reference_fallback,
            "score_type": "within_image_max_similarity" if single_reference_fallback else "cross_image_retrieval_precision",
            "match_counts_are_synthetic": single_reference_fallback,
            "selection_mode": selection_mode,
            "max_source_images": max_source_images,
            "top_k_features": top_k_features,
        }
        save_json_atomic(report_path, report)

        totals["in"] += total_features
        totals["out"] += kept_features
        totals["tp"] += kept_tp
        totals["fp"] += kept_fp
        totals["scored"] += int(scored_mask.sum().item())
        del grouped_bank, target_worklist, file_refs, file_entries, features_for_filter, scores_by_file, good_by_file, bad_by_file, scores, good, bad, scores_for_filter, keep_mask
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
        "filter_strategy": "clustered_adaptive_q75_topk",
    }
    save_json_atomic(report_path, report)
    return report


def _build_keep_mask_from_scored_bank(
    method: str,
    file_refs: list[tuple[str, str]],
    scores_for_filter: torch.Tensor,
    features_for_filter: torch.Tensor | None,
    keep_threshold: float | None,
    top_k_features: int | None,
    n_clusters: int | str,
    min_cluster_size: int,
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
    if method == "clustered_adaptive_q75":
        if features_for_filter is None:
            raise ValueError("features_for_filter is required for clustered adaptive filtering.")
        keep_mask, cluster_meta = _apply_clustered_adaptive_top_k_features(
            features_for_filter=features_for_filter,
            scores_for_filter=scores_for_filter,
            top_k_features=top_k_features,
            n_clusters=n_clusters,
            min_cluster_size=min_cluster_size,
        )
        cluster_meta["status"] = "clustered_adaptive_q75_topk"
        return keep_mask, cluster_meta
    raise ValueError(f"Unknown filtering method: {method}")


def run_filter_from_scored_bank(
    input_dir: str,
    output_dir: str,
    method: str,
    keep_threshold: float | None,
    top_k_features: int | None,
    n_clusters: int | str,
    min_cluster_size: int,
    resume: bool,
):
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
                "n_clusters": n_clusters,
                "min_cluster_size": min_cluster_size,
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
        features_for_filter = None
        if method == "clustered_adaptive_q75":
            features_for_filter = _load_flat_feature_bank(file_refs)

        keep_mask, meta = _build_keep_mask_from_scored_bank(
            method=method,
            file_refs=file_refs,
            scores_for_filter=scores_for_filter,
            features_for_filter=features_for_filter,
            keep_threshold=keep_threshold,
            top_k_features=top_k_features,
            n_clusters=n_clusters,
            min_cluster_size=min_cluster_size,
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
                "within_image_max_similarity" if single_reference_fallback else "cross_image_retrieval_precision",
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
        if method == "clustered_adaptive_q75":
            cluster_thresholds = meta.get("cluster_thresholds", [])
            class_report.update(
                {
                    "estimated_k": int(meta.get("estimated_k", 0)),
                    "used_k": int(meta.get("used_k", 0)),
                    "min_cluster_size": min_cluster_size,
                    "kept_clusters": int(meta.get("kept_clusters", 0)),
                    "discarded_small_clusters": int(meta.get("discarded_small_clusters", 0)),
                    "cluster_threshold_min": round(min(cluster_thresholds), 4) if cluster_thresholds else None,
                    "cluster_threshold_mean": round(float(np.mean(cluster_thresholds)), 4) if cluster_thresholds else None,
                    "cluster_threshold_max": round(max(cluster_thresholds), 4) if cluster_thresholds else None,
                }
            )
        report["per_class"][cat_name] = class_report
        save_json_atomic(report_path, report)

        totals["in"] += total_features
        totals["out"] += kept_features
        totals["tp"] += kept_tp or 0
        totals["fp"] += kept_fp or 0
        totals["scored"] += total_features
        del file_refs, scores_by_file, good_by_file, bad_by_file, scores, good, bad, scores_for_filter, keep_mask, features_for_filter

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
