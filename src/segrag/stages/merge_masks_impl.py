"""
Native SegRAG implementation for merging text-only and point-only SAM3 masks.

Supported merge strategies:
- union
- intersection
- confidence_weighted
- nms_instances

The evaluator is resumable:
- processed image ids are checkpointed
- per-strategy metric state is checkpointed
- partial merged mask exports are written as JSONL and finalized at the end
"""

from __future__ import annotations

import json
import os
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from pycocotools import mask as mask_utils
from tqdm import tqdm

from segrag.modeling.feature_matching import LVIS
from segrag.utils.metrics import MaskStageEvaluator, compute_iou
from segrag.utils.resume import load_json, save_json_atomic

_WORKER_LVIS = None
_WORKER_TEXT_PREDS = None
_WORKER_POINT_PREDS = None
_WORKER_CLASSES = None
_WORKER_NMS_IOU_THRESHOLD = None
_WORKER_STRATEGY_NAMES = None


def _load_predictions(path: str) -> dict[tuple[int, int], list[dict]]:
    with open(path, "r") as handle:
        data = json.load(handle)
    by_key = defaultdict(list)
    for row in data:
        by_key[(int(row["image_id"]), int(row["category_id"]))].append(row)
    return by_key


def _decode_prediction_mask(segmentation: dict) -> np.ndarray:
    return mask_utils.decode(segmentation).astype(np.uint8)


def _encode_mask(mask: np.ndarray) -> dict:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {
        "size": [int(encoded["size"][0]), int(encoded["size"][1])],
        "counts": encoded["counts"].decode("utf-8"),
    }


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


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    mask = mask.astype(bool)
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = []
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    ys, xs = np.where(mask)
    for y0, x0 in zip(ys, xs):
        if visited[y0, x0]:
            continue
        queue = deque([(y0, x0)])
        visited[y0, x0] = True
        comp = np.zeros_like(mask, dtype=np.uint8)
        while queue:
            y, x = queue.popleft()
            comp[y, x] = 1
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if visited[ny, nx] or not mask[ny, nx]:
                    continue
                visited[ny, nx] = True
                queue.append((ny, nx))
        components.append(comp)
    return components


def _union_mask(preds: list[dict]) -> tuple[np.ndarray | None, float]:
    if not preds:
        return None, 0.0
    mask = None
    best_score = 0.0
    for pred in preds:
        pred_mask = _decode_prediction_mask(pred["segmentation"])
        mask = pred_mask if mask is None else np.logical_or(mask, pred_mask).astype(np.uint8)
        best_score = max(best_score, float(pred.get("score", 0.0)))
    return mask, best_score


def _build_conf_map(preds: list[dict], shape: tuple[int, int]) -> np.ndarray:
    conf = np.zeros(shape, dtype=np.float32)
    for pred in preds:
        pred_mask = _decode_prediction_mask(pred["segmentation"]).astype(bool)
        score = float(pred.get("score", 0.0))
        conf[pred_mask] = np.maximum(conf[pred_mask], score)
    return conf


def _merge_union(text_preds: list[dict], point_preds: list[dict]) -> tuple[list[np.ndarray], list[float]]:
    text_mask, text_score = _union_mask(text_preds)
    point_mask, point_score = _union_mask(point_preds)
    if text_mask is None and point_mask is None:
        return [], []
    if text_mask is None:
        return [point_mask], [point_score]
    if point_mask is None:
        return [text_mask], [text_score]
    return [np.logical_or(text_mask, point_mask).astype(np.uint8)], [max(text_score, point_score)]


def _merge_intersection(text_preds: list[dict], point_preds: list[dict]) -> tuple[list[np.ndarray], list[float]]:
    text_mask, text_score = _union_mask(text_preds)
    point_mask, point_score = _union_mask(point_preds)
    if text_mask is None or point_mask is None:
        return [], []
    merged = np.logical_and(text_mask, point_mask).astype(np.uint8)
    if not merged.any():
        return [], []
    return [merged], [min(text_score, point_score)]


def _merge_confidence_weighted(text_preds: list[dict], point_preds: list[dict]) -> tuple[list[np.ndarray], list[float]]:
    text_mask, text_score = _union_mask(text_preds)
    point_mask, point_score = _union_mask(point_preds)
    if text_mask is None and point_mask is None:
        return [], []
    if text_mask is None:
        return [point_mask], [point_score]
    if point_mask is None:
        return [text_mask], [text_score]
    union = np.logical_or(text_mask, point_mask).astype(np.uint8)
    text_conf = _build_conf_map(text_preds, union.shape)
    point_conf = _build_conf_map(point_preds, union.shape)
    merged = np.zeros_like(union, dtype=np.uint8)
    component_scores = []
    for comp in _connected_components(union):
        text_comp_score = float(text_conf[comp.astype(bool)].max()) if np.any(comp & text_mask) else 0.0
        point_comp_score = float(point_conf[comp.astype(bool)].max()) if np.any(comp & point_mask) else 0.0
        if point_comp_score > text_comp_score:
            chosen = np.logical_and(comp, point_mask).astype(np.uint8)
            component_scores.append(point_comp_score)
        else:
            chosen = np.logical_and(comp, text_mask).astype(np.uint8)
            component_scores.append(text_comp_score)
        merged = np.logical_or(merged, chosen).astype(np.uint8)
    if not merged.any():
        return [], []
    return [merged], [max(component_scores) if component_scores else max(text_score, point_score)]


def _merge_nms_instances(text_preds: list[dict], point_preds: list[dict], iou_threshold: float) -> tuple[list[np.ndarray], list[float]]:
    candidates = []
    for pred in text_preds + point_preds:
        candidates.append({"mask": _decode_prediction_mask(pred["segmentation"]), "score": float(pred.get("score", 0.0))})
    candidates.sort(key=lambda item: item["score"], reverse=True)
    kept = []
    for cand in candidates:
        if not cand["mask"].any():
            continue
        suppress = False
        for kept_item in kept:
            if compute_iou(cand["mask"], kept_item["mask"]) > iou_threshold:
                suppress = True
                break
        if not suppress:
            kept.append(cand)
    return [item["mask"] for item in kept], [item["score"] for item in kept]


def _worker_init(
    annotation_file: str,
    text_preds: dict,
    point_preds: dict,
    classes: dict[int, str],
    nms_iou_threshold: float,
    strategy_names: list[str],
) -> None:
    global _WORKER_LVIS, _WORKER_TEXT_PREDS, _WORKER_POINT_PREDS, _WORKER_CLASSES, _WORKER_NMS_IOU_THRESHOLD, _WORKER_STRATEGY_NAMES
    _WORKER_LVIS = LVIS(annotation_file)
    _WORKER_TEXT_PREDS = text_preds
    _WORKER_POINT_PREDS = point_preds
    _WORKER_CLASSES = classes
    _WORKER_NMS_IOU_THRESHOLD = nms_iou_threshold
    _WORKER_STRATEGY_NAMES = strategy_names


def _process_image(img_id: int) -> dict | None:
    lvis_api = _WORKER_LVIS
    img_info = lvis_api.load_imgs([img_id])[0]
    ann_ids = lvis_api.get_ann_ids(img_ids=[img_id])
    anns = lvis_api.load_anns(ann_ids)
    if not anns:
        return None
    cats_in_img = set(ann["category_id"] for ann in anns)
    not_exhaustive = set(img_info.get("not_exhaustive_category_ids", []))
    valid_cats = cats_in_img - not_exhaustive
    if not valid_cats:
        return None
    original_size = (img_info["height"], img_info["width"])
    gt_union_by_class, gt_instances_by_class = _prepare_gt(lvis_api, anns, valid_cats, original_size)
    mergers = {
        "union": lambda t, p: _merge_union(t, p),
        "intersection": lambda t, p: _merge_intersection(t, p),
        "confidence_weighted": lambda t, p: _merge_confidence_weighted(t, p),
        "nms_instances": lambda t, p: _merge_nms_instances(t, p, _WORKER_NMS_IOU_THRESHOLD),
    }
    mergers = {name: mergers[name] for name in _WORKER_STRATEGY_NAMES}
    strategy_updates = {name: [] for name in mergers}
    saved_rows = {name: [] for name in mergers}
    for class_id in valid_cats:
        gt_union = gt_union_by_class[class_id]
        gt_instances = gt_instances_by_class[class_id]
        t_preds = _WORKER_TEXT_PREDS.get((img_id, class_id), [])
        p_preds = _WORKER_POINT_PREDS.get((img_id, class_id), [])
        for name, merger in mergers.items():
            merged_masks, merged_scores = merger(t_preds, p_preds)
            strategy_updates[name].append(
                {
                    "class_id": int(class_id),
                    "accepted_masks": merged_masks,
                    "accepted_scores": merged_scores,
                    "gt_union": gt_union,
                    "gt_instances": gt_instances,
                }
            )
            for idx, mask in enumerate(merged_masks):
                saved_rows[name].append(
                    {
                        "image_id": int(img_id),
                        "file_name": img_info.get("file_name", ""),
                        "category_id": int(class_id),
                        "category": _WORKER_CLASSES[class_id],
                        "score": float(merged_scores[idx]) if idx < len(merged_scores) else 0.0,
                        "merge_strategy": name,
                        "segmentation": _encode_mask(mask),
                    }
                )
    return {"updates": strategy_updates, "saved_rows": saved_rows, "img_id": int(img_id)}


def _resume_paths(output_dir: str) -> tuple[str, str]:
    checkpoint_path = os.path.join(output_dir, "checkpoint.json")
    jsonl_paths = {
        "union": os.path.join(output_dir, "predicted_masks_union.resume.jsonl"),
        "intersection": os.path.join(output_dir, "predicted_masks_intersection.resume.jsonl"),
        "confidence_weighted": os.path.join(output_dir, "predicted_masks_confidence_weighted.resume.jsonl"),
        "nms_instances": os.path.join(output_dir, "predicted_masks_nms_instances.resume.jsonl"),
    }
    return checkpoint_path, jsonl_paths


def _load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _append_jsonl(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _jsonl_to_json_array(src_path: str, dst_path: str) -> None:
    with open(dst_path, "w") as dst:
        dst.write("[\n")
        first = True
        if os.path.exists(src_path):
            with open(src_path, "r") as src:
                for line in src:
                    line = line.strip()
                    if not line:
                        continue
                    if not first:
                        dst.write(",\n")
                    dst.write(line)
                    first = False
        dst.write("\n]\n")


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    text_preds = _load_predictions(args.text_mask_json)
    point_preds = _load_predictions(args.point_mask_json)

    lvis_api = LVIS(args.annotation_file)
    cats = lvis_api.load_cats(lvis_api.get_cat_ids())
    classes = {cat["id"]: cat["name"] for cat in cats}
    strategy_names = list(getattr(args, "strategies", None) or ["union", "intersection", "confidence_weighted", "nms_instances"])

    checkpoint_path, jsonl_paths = _resume_paths(args.output_dir)
    jsonl_paths = {name: jsonl_paths[name] for name in strategy_names}
    checkpoint = load_json(
        checkpoint_path,
        {
            "processed_image_ids": [],
            "evaluators": {name: None for name in strategy_names},
        },
    ) if getattr(args, "resume", False) else {
        "processed_image_ids": [],
        "evaluators": {name: None for name in strategy_names},
    }
    processed_image_ids = set(checkpoint.get("processed_image_ids", []))
    evaluators = {
        name: MaskStageEvaluator.from_json(classes, checkpoint.get("evaluators", {}).get(name))
        for name in strategy_names
    }

    img_ids = lvis_api.get_img_ids()
    remaining_img_ids = [img_id for img_id in img_ids if img_id not in processed_image_ids]

    if args.num_workers <= 1:
        _worker_init(args.annotation_file, text_preds, point_preds, classes, args.nms_iou_threshold, strategy_names)
        iterator = map(_process_image, remaining_img_ids)
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=_worker_init,
            initargs=(args.annotation_file, text_preds, point_preds, classes, args.nms_iou_threshold, strategy_names),
        )
        iterator = executor.map(_process_image, remaining_img_ids, chunksize=8)

    try:
        for result in tqdm(iterator, total=len(remaining_img_ids), desc="Evaluating merged masks"):
            if result is None:
                continue
            for name in strategy_names:
                for update in result["updates"][name]:
                    evaluators[name].update(
                        class_id=update["class_id"],
                        accepted_masks=update["accepted_masks"],
                        accepted_scores=update["accepted_scores"],
                        gt_union=update["gt_union"],
                        gt_instances=update["gt_instances"],
                    )
                _append_jsonl(jsonl_paths[name], result["saved_rows"][name])
            processed_image_ids.add(result["img_id"])
            save_json_atomic(
                checkpoint_path,
                {
                    "processed_image_ids": sorted(processed_image_ids),
                    "evaluators": {name: evaluators[name].to_json() for name in strategy_names},
                },
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    results = {
        "config": {
            "dataset_root": args.dataset_root,
            "annotation_file": args.annotation_file,
            "text_mask_json": args.text_mask_json,
            "point_mask_json": args.point_mask_json,
            "nms_iou_threshold": args.nms_iou_threshold,
            "num_workers": args.num_workers,
            "resume": bool(getattr(args, "resume", False)),
        },
        "strategies": {
            name: {"global": evaluators[name].global_metrics(), "per_category": evaluators[name].per_class()}
            for name in strategy_names
        },
    }

    results_path = os.path.join(args.output_dir, "results_merged.json")
    with open(results_path, "w") as handle:
        json.dump(results, handle, indent=2)
    for name in strategy_names:
        final_path = os.path.join(args.output_dir, f"predicted_masks_{name}.json")
        _jsonl_to_json_array(jsonl_paths[name], final_path)
        if os.path.exists(jsonl_paths[name]):
            os.remove(jsonl_paths[name])
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print(json.dumps({name: block["global"] for name, block in results["strategies"].items()}, indent=2))
    print(f"Results saved: {results_path}")
    return results
