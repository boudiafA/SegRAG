from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

import torch


def _slug(value: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not slug:
        slug = "cache"
    return slug[:max_len]


def _fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def build_prompt_cache_dir(
    dataset_root: str,
    approach: str,
    config: dict[str, Any],
) -> str:
    fingerprint = _fingerprint(config)
    run_name = config.get("name", approach)
    return os.path.join(
        dataset_root,
        "_prompt_cache",
        _slug(approach),
        f"{_slug(run_name)}_{fingerprint}",
    )


def ensure_prompt_cache_dir(cache_dir: str, metadata: dict[str, Any]) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    meta_path = os.path.join(cache_dir, "metadata.json")
    if not os.path.exists(meta_path):
        with open(meta_path, "w") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)


def prompt_cache_path(cache_dir: str, image_key: str) -> str:
    base = os.path.basename(image_key) or "image"
    stem, _ = os.path.splitext(base)
    digest = hashlib.sha1(image_key.encode("utf-8")).hexdigest()[:12]
    return os.path.join(cache_dir, f"{_slug(stem)}_{digest}.pt")


def load_prompt_cache(cache_dir: str, image_key: str) -> dict[str, Any] | None:
    path = prompt_cache_path(cache_dir, image_key)
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def save_prompt_cache(cache_dir: str, image_key: str, payload: dict[str, Any]) -> None:
    path = prompt_cache_path(cache_dir, image_key)
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
