"""
Small JSON resume helpers for the GitHub-facing scripts.

The helpers deliberately keep the schema simple:
- `processed_image_ids` for image-level resumable tasks
- `processed_methods` for orchestration tasks
"""

from __future__ import annotations

import json
import os
from typing import Any


def load_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r") as handle:
        return json.load(handle)


def save_json_atomic(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp_path, path)
