"""
Native SegRAG implementation of SAM3 image-model segmentation evaluation.

This module supports:
- `text_only` image-model prompting via `set_image -> set_text_prompt`
- cached point-only prompting for absolute, relative, and hybrid methods
- resumable evaluation with JSON checkpoints
- optional export of predicted masks as COCO-style RLE JSON
"""

from __future__ import annotations

import gc
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader
from tqdm import tqdm

from segrag.utils.common import default_dataset_paths
from segrag.modeling.feature_matching import (
    DATALOADER_NUM_WORKERS,
    PREFETCH_QUEUE_SIZE,
    LVIS,
    LVISImageRecord,
    LVISPrefetchDataset,
    align_mask,
    collate_records,
    format_prompt,
)
from segrag.utils.prompt_cache import build_prompt_cache_dir, load_prompt_cache
from segrag.utils.metrics import MaskStageEvaluator
from segrag.utils.resume import load_json, save_json_atomic
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def _build_records(lvis_api, img_ids: list[int], image_dir: str) -> list[LVISImageRecord]:
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


def _prepare_gt(lvis_api, anns: list[dict], valid_cats: set[int], original_size: tuple[int, int]) -> tuple[dict[int, np.ndarray], dict[int, list[np.ndarray]]]:
    gt_union = {}
    gt_instances = {}
    for class_id in valid_cats:
        class_anns = [ann for ann in anns if ann["category_id"] == class_id]
        union_mask = np.zeros(original_size, dtype=np.uint8)
        instance_masks = []
        for ann in class_anns:
            mask = lvis_api.ann_to_mask(ann).astype(np.uint8)
            if mask.sum() == 0:
                continue
            union_mask = np.logical_or(union_mask, mask).astype(np.uint8)
            instance_masks.append(mask)
        gt_union[class_id] = union_mask
        gt_instances[class_id] = instance_masks
    return gt_union, gt_instances


def _deserialize_point_tuple(payload: dict | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not payload:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
        )
    return (
        np.asarray(payload.get("coords", []), dtype=np.float32).reshape(-1, 2),
        np.asarray(payload.get("labels", []), dtype=np.int32),
        np.asarray(payload.get("scores", []), dtype=np.float32),
    )


def _resolve_paths(args):
    defaults = default_dataset_paths(args.dataset_root)
    args.annotation_file = args.annotation_file or defaults.annotation_file
    args.image_dir = args.image_dir or defaults.image_dir
    args.feature_bank_dir = args.feature_bank_dir or defaults.raw_feature_bank_dir
    args.filtered_bank_dir = args.filtered_bank_dir or defaults.filtered_feature_bank_dir


def _resolve_prompt_cache_dir(args) -> str:
    if args.feature_matching_approach == "text_only":
        return ""
    if args.prompt_cache_dir:
        return args.prompt_cache_dir
    if args.feature_matching_approach == "absolute_similarity":
        config = {
            "name": "absolute_similarity",
            "feature_bank_dir": None,
            "filtered_bank_dir": args.filtered_bank_dir,
            "num_points": args.num_points,
            "sim_threshold": args.sim_threshold,
            "max_references": args.max_references,
        }
    elif args.feature_matching_approach == "relative_similarity":
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
    return build_prompt_cache_dir(args.dataset_root, args.feature_matching_approach, config)


def _load_cached_prompts(cache_payload: dict | None, approach: str, class_id: int, hybrid_threshold: float):
    if cache_payload is None:
        return None
    if approach in ("absolute_similarity", "relative_similarity"):
        return _deserialize_point_tuple(cache_payload.get("filtered", {}).get(str(class_id)))
    prompts = cache_payload.get("text_filtered", {}).get(str(class_id))
    if prompts is None:
        return None
    accepted = [p for p in prompts if float(p.get("score", 0.0)) >= hybrid_threshold]
    coords = np.asarray([[p["x"], p["y"]] for p in accepted], dtype=np.float32).reshape(-1, 2)
    labels = np.ones(len(coords), dtype=np.int32)
    scores = np.asarray([p["score"] for p in accepted], dtype=np.float32)
    return coords, labels, scores


def _encode_mask(mask: np.ndarray) -> dict:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {
        "size": [int(encoded["size"][0]), int(encoded["size"][1])],
        "counts": encoded["counts"].decode("utf-8"),
    }


def _normalize_mask(mask, original_size: tuple[int, int]) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    if mask.ndim == 4:
        mask = mask.squeeze(0)
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    mask = (mask > 0).astype(np.uint8)
    return align_mask(mask, original_size).astype(np.uint8)


class Sam3ImageEvaluator:
    def __init__(self):
        print("Loading SAM3 image model...")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.model = build_sam3_image_model(enable_inst_interactivity=True)
        self.processor = Sam3Processor(self.model, confidence_threshold=0.4)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("SAM3 image model loaded.")

    def predict_text_only(self, pil_image: Image.Image, text_prompt: str, original_size: tuple[int, int]) -> tuple[np.ndarray, float]:
        autocast_enabled = self.device == "cuda"
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            try:
                state = self.processor.set_image(pil_image)
                self.processor.reset_all_prompts(state)
                state = self.processor.set_text_prompt(prompt=text_prompt, state=state)
            except Exception as exc:
                print(f"[SAM3 image] Text-only prediction error: {exc}")
                return np.zeros(original_size, dtype=np.uint8), 0.0
        masks = state.get("masks")
        scores = state.get("scores")
        if masks is None:
            return np.zeros(original_size, dtype=np.uint8), 0.0
        if isinstance(masks, torch.Tensor):
            masks = masks.detach().float().cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().float().cpu().numpy()
        masks = np.asarray(masks)
        scores = np.asarray(scores) if scores is not None else np.empty((0,), dtype=np.float32)
        if masks.ndim == 4:
            masks = masks[:, 0, ...]
        elif masks.ndim == 2:
            masks = masks[None, ...]
        if masks.size == 0:
            return np.zeros(original_size, dtype=np.uint8), 0.0
        combined = np.zeros(original_size, dtype=np.uint8)
        best_score = float(scores.max()) if scores.size else 0.0
        for idx in range(masks.shape[0]):
            combined = np.logical_or(combined, _normalize_mask(masks[idx], original_size)).astype(np.uint8)
        return combined, best_score

    def predict_points(self, pil_image: Image.Image, point_coords: np.ndarray, point_labels: np.ndarray, original_size: tuple[int, int], max_points_per_class: int | None = None):
        if len(point_coords) == 0:
            return [], []
        if max_points_per_class is not None and max_points_per_class > 0:
            point_coords = point_coords[:max_points_per_class]
            point_labels = point_labels[:max_points_per_class]
        masks_out: list[np.ndarray] = []
        scores_out: list[float] = []
        autocast_enabled = self.device == "cuda"
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            state = self.processor.set_image(pil_image)
            self.processor.reset_all_prompts(state)
            batched_coords = point_coords.astype(np.float32)[:, None, :]
            batched_labels = point_labels.astype(np.int32)[:, None]
            try:
                masks, scores, _ = self.model.predict_inst(
                    state,
                    point_coords=batched_coords,
                    point_labels=batched_labels,
                    multimask_output=False,
                    return_logits=False,
                )
            except Exception as exc:
                print(f"[SAM3 image] Point-only prediction error: {exc}")
                return [], []
            if masks is None:
                return [], []
            masks = np.asarray(masks)
            scores = np.asarray(scores)
            if masks.ndim == 2:
                masks = masks[None, None, ...]
            elif masks.ndim == 3:
                masks = masks[:, None, ...]
            for idx in range(masks.shape[0]):
                pred_mask = _normalize_mask(masks[idx], original_size)
                if pred_mask.any():
                    masks_out.append(pred_mask)
                    scores_out.append(float(scores[idx][0]) if scores.ndim == 2 else float(scores[idx]))
        return masks_out, scores_out


def _resume_paths(output_dir: str) -> tuple[str, str]:
    return (
        os.path.join(output_dir, "checkpoint.json"),
        os.path.join(output_dir, "predicted_masks_resume.jsonl"),
    )


def _load_saved_masks_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _append_saved_masks_jsonl(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def run(args):
    _resolve_paths(args)
    prompt_cache_dir = _resolve_prompt_cache_dir(args)
    if args.feature_matching_approach != "text_only" and not os.path.isdir(prompt_cache_dir):
        raise FileNotFoundError(
            f"Prompt cache directory not found: {prompt_cache_dir}\n"
            "Build it first with segrag.stages.cache_prompts using the same method settings."
        )
    if args.output_dir is None:
        if args.feature_matching_approach == "text_only":
            args.output_dir = os.path.join(args.dataset_root, "evaluation_results_text_only_sam3")
        else:
            args.output_dir = os.path.join(
                args.dataset_root,
                f"evaluation_results_points_only_sam3_{args.feature_matching_approach}",
            )
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint_path, masks_resume_path = _resume_paths(args.output_dir)
    defaults = {
        "processed_image_ids": [],
        "evaluator": None,
        "missing_prompt_cache_by_category": {},
        "skipped_prediction_errors": 0,
        "started_at": time.time(),
    }
    state = load_json(checkpoint_path, defaults) if getattr(args, "resume", False) else defaults
    processed_image_ids = set(state.get("processed_image_ids", []))

    lvis_api = LVIS(args.annotation_file)
    cats = lvis_api.load_cats(lvis_api.get_cat_ids())
    classes = {cat["id"]: cat["name"] for cat in cats}
    evaluator = MaskStageEvaluator.from_json(classes, state.get("evaluator"))
    missing_prompt_cache = defaultdict(int, state.get("missing_prompt_cache_by_category", {}))
    skipped_prediction_errors = int(state.get("skipped_prediction_errors", 0))
    saved_masks = _load_saved_masks_jsonl(masks_resume_path) if args.save_mask_json else None

    img_ids = lvis_api.get_img_ids()
    if args.max_images is not None:
        img_ids = img_ids[:args.max_images]
    records = _build_records(lvis_api, img_ids, args.image_dir)
    if not records:
        raise ValueError("No valid evaluation records found.")

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

    sam_model = Sam3ImageEvaluator()
    started_at = float(state.get("started_at", time.time()))
    for image_idx, batch in enumerate(tqdm(loader, total=len(records), desc="Evaluating"), start=1):
        rec = batch[0]
        if rec.img_id in processed_image_ids:
            continue
        pil_image = rec.pil_image
        if pil_image is None:
            continue
        original_size = (rec.img_info["height"], rec.img_info["width"])
        gt_union_by_class, gt_instances_by_class = _prepare_gt(
            lvis_api=lvis_api,
            anns=rec.anns,
            valid_cats=rec.valid_cats,
            original_size=original_size,
        )
        cache_payload = None
        if args.feature_matching_approach != "text_only":
            cache_payload = load_prompt_cache(prompt_cache_dir, rec.file_name)
        image_saved_rows = []

        for class_id in rec.valid_cats:
            gt_instances = gt_instances_by_class[class_id]
            if not gt_instances:
                continue
            if args.feature_matching_approach == "text_only":
                pred_mask, pred_score = sam_model.predict_text_only(
                    pil_image=pil_image,
                    text_prompt=format_prompt(classes[class_id]),
                    original_size=original_size,
                )
                pred_masks = [pred_mask] if pred_mask.any() else []
                accepted_scores = [pred_score] if pred_masks else []
            else:
                cached = _load_cached_prompts(
                    cache_payload=cache_payload,
                    approach=args.feature_matching_approach,
                    class_id=class_id,
                    hybrid_threshold=args.hybrid_validation_threshold,
                )
                if cached is None:
                    missing_prompt_cache[classes[class_id]] += 1
                    continue
                point_coords, point_labels, point_scores = cached
                if len(point_coords) == 0:
                    evaluator.update(class_id, [], [], gt_union_by_class[class_id], gt_instances)
                    continue
                pred_masks, sam_scores = sam_model.predict_points(
                    pil_image=pil_image,
                    point_coords=point_coords,
                    point_labels=point_labels,
                    original_size=original_size,
                    max_points_per_class=args.max_points_per_class,
                )
                if not pred_masks and len(point_coords) > 0:
                    skipped_prediction_errors += 1
                accepted_scores = [
                    float(point_scores[idx]) * float(sam_scores[idx])
                    for idx in range(min(len(pred_masks), len(point_scores), len(sam_scores)))
                ]
                if len(accepted_scores) < len(pred_masks):
                    accepted_scores.extend([0.0] * (len(pred_masks) - len(accepted_scores)))

            if saved_masks is not None:
                for mask_idx, pred_mask in enumerate(pred_masks):
                    row = {
                        "image_id": int(rec.img_id),
                        "file_name": rec.file_name,
                        "category_id": int(class_id),
                        "category": classes[class_id],
                        "score": float(accepted_scores[mask_idx]) if mask_idx < len(accepted_scores) else 0.0,
                        "method": args.feature_matching_approach,
                        "segmentation": _encode_mask(pred_mask),
                    }
                    image_saved_rows.append(row)
                    saved_masks.append(row)

            evaluator.update(
                class_id=class_id,
                accepted_masks=pred_masks,
                accepted_scores=accepted_scores,
                gt_union=gt_union_by_class[class_id],
                gt_instances=gt_instances,
            )

        if args.save_mask_json:
            _append_saved_masks_jsonl(masks_resume_path, image_saved_rows)

        processed_image_ids.add(rec.img_id)
        save_json_atomic(
            checkpoint_path,
            {
                "processed_image_ids": sorted(processed_image_ids),
                "evaluator": evaluator.to_json(),
                "missing_prompt_cache_by_category": dict(sorted(missing_prompt_cache.items())),
                "skipped_prediction_errors": skipped_prediction_errors,
                "started_at": started_at,
            },
        )

        if args.cleanup_every > 0 and image_idx % args.cleanup_every == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elapsed = time.time() - started_at
    results = {
        "config": {
            "feature_matching_approach": args.feature_matching_approach,
            "dataset_root": args.dataset_root,
            "annotation_file": args.annotation_file,
            "image_dir": args.image_dir,
            "filtered_bank_dir": args.filtered_bank_dir,
            "prompt_cache_dir": prompt_cache_dir,
            "max_images": args.max_images,
            "max_references": args.max_references,
            "max_points_per_class": args.max_points_per_class,
            "num_points": args.num_points,
            "sim_threshold": args.sim_threshold,
            "peak_threshold": args.peak_threshold,
            "min_peak_distance": args.min_peak_distance,
            "suppression_margin": args.suppression_margin,
            "no_suppression": args.no_suppression,
            "loose_threshold": args.loose_threshold,
            "min_component_size": args.min_component_size,
            "hybrid_validation_threshold": args.hybrid_validation_threshold,
            "cleanup_every": args.cleanup_every,
            "elapsed_seconds": elapsed,
            "images_evaluated": len(processed_image_ids),
            "resume": bool(getattr(args, "resume", False)),
        },
        "global": evaluator.global_metrics(),
        "per_category": evaluator.per_class(),
        "missing_prompt_cache_by_category": dict(sorted(missing_prompt_cache.items())),
        "skipped_prediction_errors": skipped_prediction_errors,
    }

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as handle:
        json.dump(results, handle, indent=2)
    if args.save_mask_json:
        masks_path = os.path.join(args.output_dir, "predicted_masks.json")
        with open(masks_path, "w") as handle:
            json.dump(saved_masks, handle, indent=2)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    if args.save_mask_json and os.path.exists(masks_resume_path):
        os.remove(masks_resume_path)

    print(json.dumps(results["global"], indent=2))
    print(f"Results saved: {results_path}")
    if args.save_mask_json:
        print(f"Predicted masks saved: {os.path.join(args.output_dir, 'predicted_masks.json')}")
    return results
