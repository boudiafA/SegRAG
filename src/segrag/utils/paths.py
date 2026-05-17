from __future__ import annotations

import os


DEFAULT_WEIGHTS_BASENAME = "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"


def _has_dinov3_hub(repo_path: str) -> bool:
    return os.path.isfile(os.path.join(repo_path, "hubconf.py"))


def resolve_dinov3_repo_path(repo_path: str | None) -> str:
    env_path = os.environ.get("DINOV3_REPO_PATH")
    candidates = []
    if repo_path:
        candidates.append(repo_path)
    if env_path:
        candidates.append(env_path)
    candidates.extend(
        [
            os.getcwd(),
            os.path.join(os.getcwd(), "dinov3"),
            os.path.abspath(os.path.join(os.getcwd(), "..", "dinov3")),
        ]
    )
    seen: set[str] = set()
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)
        if _has_dinov3_hub(candidate):
            return candidate
    checked = ", ".join(seen)
    raise FileNotFoundError(
        "Could not resolve the local DINOv3 repository. "
        "Pass --repo-path, set DINOV3_REPO_PATH, or place dinov3 next to this repo. "
        f"Checked: {checked}"
    )


def resolve_dinov3_weights_path(weights_path: str | None, repo_path: str | None = None) -> str:
    env_path = os.environ.get("DINOV3_WEIGHTS_PATH")
    repo_root = resolve_dinov3_repo_path(repo_path)
    basename = os.path.basename(weights_path) if weights_path else DEFAULT_WEIGHTS_BASENAME
    candidates = []
    if weights_path:
        candidates.append(weights_path)
    if env_path:
        candidates.append(env_path)
    candidates.extend(
        [
            os.path.join(os.getcwd(), "weights", basename),
            os.path.join(repo_root, "weights", basename),
            os.path.abspath(os.path.join(os.getcwd(), "..", "weights", basename)),
        ]
    )
    seen: set[str] = set()
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.exists(candidate):
            return candidate
    checked = ", ".join(seen)
    raise FileNotFoundError(
        "Could not resolve the DINOv3 weights checkpoint. "
        "Pass --weights-path, set DINOV3_WEIGHTS_PATH, or place the checkpoint in ./weights or ../dinov3/weights. "
        f"Checked: {checked}"
    )
