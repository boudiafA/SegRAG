"""
Native GitHub implementation of SAM3 image-model evaluation with combined text
and cached point prompts.

This stage uses SAM3's grounding branch directly:
- text prompt enters the language backbone
- cached point prompts are appended to the grounding Prompt object
- the fused prompt sequence is re-run through grounding to produce masks

The evaluator follows the same checkpointing, metrics, and JSON export pattern
used by the existing SAM3 stages in this package.
"""

from __future__ import annotations

import gc
import json
import os
import time
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from GitHub.feature_matching_backends import (
    DATALOADER_NUM_WORKERS,
    PREFETCH_QUEUE_SIZE,
    LVIS,
    LVISPrefetchDataset,
    collate_records,
    format_prompt,
)
from GitHub.metrics import MaskStageEvaluator
from GitHub.prompt_cache import load_prompt_cache
from GitHub.resume import load_json, save_json_atomic
from GitHub.sam3_points_only_impl import (
    Sam3ImageEvaluator,
    _build_records,
    _encode_mask,
    _load_cached_prompts,
    _normalize_mask,
    _prepare_gt,
    _resolve_paths,
    _resolve_prompt_cache_dir,
)


def _clone_for_grounding(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_for_grounding(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_for_grounding(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_for_grounding(item) for item in value)
    clone = getattr(value, "clone", None)
    if callable(clone):
        try:
            return clone()
        except TypeError:
            pass
    try:
        return deepcopy(value)
    except Exception:
        return value


class Sam3TextAndPointsEvaluator(Sam3ImageEvaluator):
    def _extract_grounding_predictions(
        self,
        state: dict,
        original_size: tuple[int, int],
    ) -> tuple[list[np.ndarray], list[float]]:
        masks = state.get("masks")
        scores = state.get("scores")
        if masks is None:
            return [], []

        if isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().float().cpu().numpy()

        masks = np.asarray(masks)
        scores = np.asarray(scores) if scores is not None else np.empty((0,), dtype=np.float32)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)
        elif masks.ndim == 2:
            masks = masks[None, ...]

        accepted_masks: list[np.ndarray] = []
        accepted_scores: list[float] = []
        for idx in range(masks.shape[0]):
            pred_mask = _normalize_mask(masks[idx], original_size)
            if pred_mask.any():
                accepted_masks.append(pred_mask)
                accepted_scores.append(float(scores[idx]) if idx < len(scores) else 0.0)
        return accepted_masks, accepted_scores

    def predict_text_and_points(
        self,
        image_state: dict,
        text_prompt: str,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        original_size: tuple[int, int],
        max_points_per_class: int | None = None,
    ) -> tuple[list[np.ndarray], list[float], dict[str, bool]]:
        if max_points_per_class is not None and max_points_per_class > 0:
            point_coords = point_coords[:max_points_per_class]
            point_labels = point_labels[:max_points_per_class]

        autocast_enabled = self.device == "cuda"
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            # `set_image()` returns inference-mode tensors inside `state`.
            # Reusing them across classes and then mixing in new text/point inputs
            # can trigger autograd/inference-mode conflicts in SAM3 grounding.
            # Work on a per-class cloned state instead of mutating the shared base.
            state = _clone_for_grounding(image_state)
            self.processor.reset_all_prompts(state)

            if len(point_coords) == 0:
                state = self.processor.set_text_prompt(prompt=text_prompt, state=state)
                text_only_masks, text_only_scores = self._extract_grounding_predictions(state, original_size)
                return text_only_masks, text_only_scores, {
                    "used_text_only_fallback": True,
                    "combined_prediction_error": False,
                }

            try:
                with torch.no_grad():
                    text_outputs = self.model.backbone.forward_text([text_prompt], device=self.device)
                    state["backbone_out"].update(_clone_for_grounding(text_outputs))

                    points_norm = np.asarray(point_coords, dtype=np.float32).reshape(-1, 2).copy()
                    points_norm[:, 0] /= max(1, original_size[1])
                    points_norm[:, 1] /= max(1, original_size[0])
                    points_norm = np.clip(points_norm, 0.0, 1.0)

                    labels = np.asarray(point_labels, dtype=np.int64).reshape(-1)
                    points_tensor = torch.as_tensor(points_norm, device=self.device, dtype=torch.float32).view(-1, 1, 2)
                    labels_tensor = torch.as_tensor(labels, device=self.device, dtype=torch.long).view(-1, 1)

                    if "geometric_prompt" not in state:
                        state["geometric_prompt"] = self.model._get_dummy_prompt()
                    state["geometric_prompt"] = _clone_for_grounding(state["geometric_prompt"])
                    state["geometric_prompt"].append_points(points_tensor, labels_tensor)
                    state = self.processor._forward_grounding(state)
                accepted_masks, accepted_scores = self._extract_grounding_predictions(state, original_size)
                return accepted_masks, accepted_scores, {
                    "used_text_only_fallback": False,
                    "combined_prediction_error": False,
                }
            except Exception as exc:
                print(f"[SAM3 image] Text+point grounding error: {exc}")
                self.processor.reset_all_prompts(state)
                state = self.processor.set_text_prompt(prompt=text_prompt, state=state)
                text_only_masks, text_only_scores = self._extract_grounding_predictions(state, original_size)
                return text_only_masks, text_only_scores, {
                    "used_text_only_fallback": True,
                    "combined_prediction_error": True,
                }


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
    num_workers = max(0, int(getattr(args, "num_workers", DATALOADER_NUM_WORKERS)))
    prefetch_factor = max(1, int(getattr(args, "prefetch_factor", PREFETCH_QUEUE_SIZE)))
    prompt_cache_dir = _resolve_prompt_cache_dir(args)
    if not os.path.isdir(prompt_cache_dir):
        raise FileNotFoundError(
            f"Prompt cache directory not found: {prompt_cache_dir}\n"
            "Build it first with GitHub/Stage2_cache_feature_matching.py using the same method settings."
        )
    if args.output_dir is None:
        args.output_dir = os.path.join(
            args.dataset_root,
            f"evaluation_results_text_and_points_sam3_{args.feature_matching_approach}",
        )
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint_path, masks_resume_path = _resume_paths(args.output_dir)
    defaults = {
        "processed_image_ids": [],
        "evaluator": None,
        "missing_prompt_cache_by_category": {},
        "text_only_fallbacks": 0,
        "combined_prediction_errors": 0,
        "started_at": time.time(),
    }
    state = load_json(checkpoint_path, defaults) if getattr(args, "resume", False) else defaults
    processed_image_ids = set(state.get("processed_image_ids", []))

    lvis_api = LVIS(args.annotation_file)
    cats = lvis_api.load_cats(lvis_api.get_cat_ids())
    classes = {cat["id"]: cat["name"] for cat in cats}
    evaluator = MaskStageEvaluator.from_json(classes, state.get("evaluator"))
    missing_prompt_cache = defaultdict(int, state.get("missing_prompt_cache_by_category", {}))
    text_only_fallbacks = int(state.get("text_only_fallbacks", 0))
    combined_prediction_errors = int(state.get("combined_prediction_errors", 0))
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
        num_workers=num_workers,
        collate_fn=collate_records,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    sam_model = Sam3TextAndPointsEvaluator()
    started_at = float(state.get("started_at", time.time()))
    for image_idx, batch in enumerate(tqdm(loader, total=len(records), desc="Evaluating"), start=1):
        rec = batch[0]
        if rec.img_id in processed_image_ids:
            continue
        pil_image = rec.pil_image
        if pil_image is None:
            continue

        original_size = (rec.img_info["height"], rec.img_info["width"])
        image_state = sam_model.processor.set_image(pil_image)
        gt_union_by_class, gt_instances_by_class = _prepare_gt(
            lvis_api=lvis_api,
            anns=rec.anns,
            valid_cats=rec.valid_cats,
            original_size=original_size,
        )
        cache_payload = load_prompt_cache(prompt_cache_dir, rec.file_name)
        image_saved_rows = []

        for class_id in rec.valid_cats:
            gt_instances = gt_instances_by_class[class_id]
            if not gt_instances:
                continue

            cached = _load_cached_prompts(
                cache_payload=cache_payload,
                approach=args.feature_matching_approach,
                class_id=class_id,
                hybrid_threshold=args.hybrid_validation_threshold,
            )
            if cached is None:
                missing_prompt_cache[classes[class_id]] += 1
                point_coords = np.empty((0, 2), dtype=np.float32)
                point_labels = np.empty((0,), dtype=np.int32)
            else:
                point_coords, point_labels, _ = cached

            pred_masks, accepted_scores, meta = sam_model.predict_text_and_points(
                image_state=image_state,
                text_prompt=format_prompt(classes[class_id]),
                point_coords=point_coords,
                point_labels=point_labels,
                original_size=original_size,
                max_points_per_class=args.max_points_per_class,
            )

            text_only_fallbacks += int(meta["used_text_only_fallback"])
            combined_prediction_errors += int(meta["combined_prediction_error"])

            if saved_masks is not None:
                for mask_idx, pred_mask in enumerate(pred_masks):
                    row = {
                        "image_id": int(rec.img_id),
                        "file_name": rec.file_name,
                        "category_id": int(class_id),
                        "category": classes[class_id],
                        "score": float(accepted_scores[mask_idx]) if mask_idx < len(accepted_scores) else 0.0,
                        "method": args.feature_matching_approach,
                        "prompt_mode": "text_and_points",
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
                "text_only_fallbacks": text_only_fallbacks,
                "combined_prediction_errors": combined_prediction_errors,
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
            "num_workers": num_workers,
            "prefetch_factor": prefetch_factor,
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
        "text_only_fallbacks": text_only_fallbacks,
        "combined_prediction_errors": combined_prediction_errors,
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
