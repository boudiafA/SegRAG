from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch


def build_feature_cache_dir(
    *,
    annotation_file: str,
    image_size: int,
    model_name: str,
    cache_root: str | None = None,
) -> str:
    ann_path = Path(annotation_file)
    if cache_root is None:
        cache_root = str(ann_path.parent)
    ann_stem = ann_path.stem
    safe_model = model_name.replace("/", "_")
    return os.path.join(cache_root, f"_feature_cache_{safe_model}_{image_size}_{ann_stem}")


def normalize_cache_key(cache_key: str) -> str:
    return str(cache_key).replace("\\", "/").lstrip("./")


def cache_path_for_key(cache_dir: str, cache_key: str) -> str:
    normalized = normalize_cache_key(cache_key)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    prefix = normalized.replace("/", "__").replace(" ", "_")
    prefix = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in prefix)
    prefix = prefix[:80] if prefix else "image"
    return os.path.join(cache_dir, f"{prefix}_{digest}.pt")


class FeatureCache:
    def __init__(self, cache_dir: str | None):
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

    def load(self, cache_key: str) -> torch.Tensor | None:
        if self.cache_dir is None:
            return None
        path = cache_path_for_key(self.cache_dir, cache_key)
        if not os.path.exists(path):
            return None
        try:
            return torch.load(path, map_location="cpu", weights_only=True).float()
        except Exception:
            return None

    def save(self, cache_key: str, features: torch.Tensor) -> str | None:
        if self.cache_dir is None:
            return None
        path = cache_path_for_key(self.cache_dir, cache_key)
        tmp_path = f"{path}.tmp"
        torch.save(features.detach().cpu().float(), tmp_path)
        os.replace(tmp_path, path)
        return path
