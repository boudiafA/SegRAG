"""
Shared constants and small helpers for the clean SegRAG workflows.

This module is intentionally small and dependency-light so entrypoints can
reuse the same task names, defaults, and validation without repeating code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


METHODS = ("absolute_similarity", "relative_similarity", "hybrid")
COMPARE_ALL = "compare_all"


@dataclass(frozen=True)
class DatasetPaths:
    dataset_root: str
    annotation_file: str
    image_dir: str
    raw_feature_bank_dir: str
    scored_feature_bank_dir: str
    filtered_feature_bank_dir: str


def default_dataset_paths(dataset_root: str) -> DatasetPaths:
    return DatasetPaths(
        dataset_root=dataset_root,
        annotation_file=os.path.join(dataset_root, "test.json"),
        image_dir=dataset_root,
        raw_feature_bank_dir=os.path.join(dataset_root, "feature_bank_dinov3_vitl16_1536"),
        scored_feature_bank_dir=os.path.join(
            dataset_root,
            "feature_bank_dinov3_vitl16_1536_scored_thr060",
        ),
        filtered_feature_bank_dir=os.path.join(
            dataset_root,
            "feature_bank_adaptive_q75_from_thr060",
        ),
    )


def validate_method(method: str, allow_compare_all: bool = False) -> str:
    if method in METHODS:
        return method
    if allow_compare_all and method == COMPARE_ALL:
        return method
    allowed = list(METHODS) + ([COMPARE_ALL] if allow_compare_all else [])
    raise ValueError(f"Unsupported method '{method}'. Expected one of: {allowed}")
