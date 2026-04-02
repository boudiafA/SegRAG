"""
Shared segmentation metrics for the GitHub-facing workflows.

These helpers are intentionally JSON-serializable so long-running jobs can
resume without rebuilding evaluator state from scratch.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = float(np.logical_and(mask_a, mask_b).sum())
    union = float(np.logical_or(mask_a, mask_b).sum())
    return inter / union if union > 0 else 0.0


def compute_ap(records: list[tuple[float, int]], gt_count: int) -> float:
    if gt_count <= 0 or not records:
        return 0.0
    ordered = sorted(records, key=lambda item: item[0], reverse=True)
    tp = np.array([item[1] for item in ordered], dtype=np.float32)
    fp = 1.0 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / gt_count
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([1.0], precisions, [0.0]))
    for idx in range(len(precisions) - 2, -1, -1):
        precisions[idx] = max(precisions[idx], precisions[idx + 1])
    return float(np.sum((recalls[1:] - recalls[:-1]) * precisions[1:]))


class MaskStageEvaluator:
    def __init__(self, classes: dict[int, str], ap_iou_threshold: float = 0.5):
        self.classes = classes
        self.ap_iou_threshold = ap_iou_threshold
        self.intersection = defaultdict(int)
        self.union = defaultdict(int)
        self.pixel_tp = defaultdict(int)
        self.pixel_fp = defaultdict(int)
        self.pixel_fn = defaultdict(int)
        self.pred_records = defaultdict(list)
        self.gt_count = defaultdict(int)

    def update(
        self,
        class_id: int,
        accepted_masks: list[np.ndarray],
        accepted_scores: list[float],
        gt_union: np.ndarray,
        gt_instances: list[np.ndarray],
    ):
        pred_union = np.zeros(gt_union.shape, dtype=bool)
        for pred_mask in accepted_masks:
            pred_union |= np.asarray(pred_mask).astype(bool)

        gt_union_bool = np.asarray(gt_union).astype(bool)
        tp_pixels = int(np.logical_and(pred_union, gt_union_bool).sum())
        union_pixels = int(np.logical_or(pred_union, gt_union_bool).sum())

        self.intersection[class_id] += tp_pixels
        self.union[class_id] += union_pixels
        self.pixel_tp[class_id] += tp_pixels
        self.pixel_fp[class_id] += int(np.logical_and(pred_union, ~gt_union_bool).sum())
        self.pixel_fn[class_id] += int(np.logical_and(~pred_union, gt_union_bool).sum())

        self.gt_count[class_id] += len(gt_instances)
        matched_gt = [False] * len(gt_instances)
        for score, pred_mask in sorted(zip(accepted_scores, accepted_masks), key=lambda item: item[0], reverse=True):
            best_iou = 0.0
            best_idx = -1
            for idx, gt_mask in enumerate(gt_instances):
                if matched_gt[idx]:
                    continue
                iou = compute_iou(pred_mask.astype(bool), gt_mask.astype(bool))
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            is_tp = int(best_idx >= 0 and best_iou >= self.ap_iou_threshold)
            if is_tp:
                matched_gt[best_idx] = True
            self.pred_records[class_id].append((float(score), is_tp))

    def _pixel_metrics(self, tp: int, fp: int, fn: int) -> tuple[float, float]:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return precision, recall

    def class_metrics(self, class_id: int) -> dict:
        precision, recall = self._pixel_metrics(
            self.pixel_tp[class_id],
            self.pixel_fp[class_id],
            self.pixel_fn[class_id],
        )
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        iou = self.intersection[class_id] / self.union[class_id] if self.union[class_id] > 0 else 0.0
        ap50 = compute_ap(self.pred_records[class_id], self.gt_count[class_id])
        return {
            "IoU": iou,
            "F1": f1,
            "PixelPrecision": precision,
            "PixelRecall": recall,
            "AP50": ap50,
            "gt_instances": self.gt_count[class_id],
            "predictions": len(self.pred_records[class_id]),
        }

    def global_metrics(self) -> dict:
        total_intersection = sum(self.intersection.values())
        total_union = sum(self.union.values())
        total_tp = sum(self.pixel_tp.values())
        total_fp = sum(self.pixel_fp.values())
        total_fn = sum(self.pixel_fn.values())
        precision, recall = self._pixel_metrics(total_tp, total_fp, total_fn)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        valid_class_ids = [cid for cid in self.classes if self.gt_count[cid] > 0]
        map50 = float(np.mean([compute_ap(self.pred_records[cid], self.gt_count[cid]) for cid in valid_class_ids])) if valid_class_ids else 0.0
        miou = float(
            np.mean(
                [
                    self.intersection[cid] / self.union[cid] if self.union[cid] > 0 else 0.0
                    for cid in valid_class_ids
                ]
            )
        ) if valid_class_ids else 0.0
        return {
            "IoU": total_intersection / total_union if total_union > 0 else 0.0,
            "mIoU": miou,
            "F1": f1,
            "PixelPrecision": precision,
            "PixelRecall": recall,
            "mAP50": map50,
            "pixel_tp": total_tp,
            "pixel_fp": total_fp,
            "pixel_fn": total_fn,
        }

    def per_class(self) -> list[dict]:
        rows = []
        for class_id, name in self.classes.items():
            if self.gt_count[class_id] <= 0:
                continue
            row = self.class_metrics(class_id)
            row["category"] = name
            rows.append(row)
        rows.sort(key=lambda item: item["gt_instances"], reverse=True)
        return rows

    def to_json(self) -> dict:
        return {
            "intersection": dict(self.intersection),
            "union": dict(self.union),
            "pixel_tp": dict(self.pixel_tp),
            "pixel_fp": dict(self.pixel_fp),
            "pixel_fn": dict(self.pixel_fn),
            "gt_count": dict(self.gt_count),
            "pred_records": {str(k): v for k, v in self.pred_records.items()},
        }

    @classmethod
    def from_json(cls, classes: dict[int, str], payload: dict | None):
        evaluator = cls(classes)
        if not payload:
            return evaluator
        for key in ("intersection", "union", "pixel_tp", "pixel_fp", "pixel_fn", "gt_count"):
            store = getattr(evaluator, key)
            for class_id, value in payload.get(key, {}).items():
                store[int(class_id)] = int(value)
        for class_id, records in payload.get("pred_records", {}).items():
            evaluator.pred_records[int(class_id)] = [(float(score), int(is_tp)) for score, is_tp in records]
        return evaluator
