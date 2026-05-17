"""
Small JSON resume helpers for the SegRAG scripts.

The helpers deliberately keep the schema simple:
- `processed_image_ids` for image-level resumable tasks
- `processed_methods` for orchestration tasks
"""

from __future__ import annotations

import json
import os
import time
from typing import Any


def load_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r") as handle:
        return json.load(handle)


def save_json_atomic(path: str, payload: Any) -> None:
    parent = os.path.dirname(path)
    tmp_path = f"{path}.{os.getpid()}.tmp"
    last_error: OSError | None = None
    for attempt in range(3):
        try:
            os.makedirs(parent, exist_ok=True)
            with open(tmp_path, "w") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
            os.replace(tmp_path, path)
            return
        except OSError as exc:
            last_error = exc
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            time.sleep(0.1 * (attempt + 1))
    raise last_error
