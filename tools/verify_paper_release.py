#!/usr/bin/env python3
"""Verify frozen paper manifests, configs, results, and optional model weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dinov3-weights", type=Path)
    args = parser.parse_args()

    checksums = load_json(ROOT / "splits" / "standard" / "checksums.json")
    expected_results = load_json(ROOT / "reproducibility" / "table4_results.json")
    software = load_json(ROOT / "reproducibility" / "software.json")
    failures: list[str] = []
    checked: list[dict[str, str]] = []

    for dataset in checksums["datasets"]:
        for kind in ("support_manifest", "query_manifest"):
            path = ROOT / dataset[kind]
            actual = sha256_file(path)
            expected = dataset[f"{kind}_sha256"]
            checked.append({"path": str(path.relative_to(ROOT)), "sha256": actual})
            if actual != expected:
                failures.append(f"{path}: expected {expected}, got {actual}")

        config_name = {
            "PC-59": "pc59.yaml",
            "Cityscapes": "cityscapes.yaml",
            "ADE20K_150": "ade20k150.yaml",
            "LVIS": "lvis.yaml",
        }[dataset["dataset"]]
        with (ROOT / "configs" / "paper" / config_name).open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        for shot in ("1", "5"):
            archived = expected_results["results"][dataset["dataset"]][shot]["mIoU"]
            if float(config["reported_miou"][shot]) != float(archived):
                failures.append(f"{config_name}: archived {shot}-shot mIoU does not match config")

    if args.dinov3_weights:
        path = args.dinov3_weights.expanduser().resolve()
        actual = sha256_file(path)
        expected = software["dinov3"]["checkpoint_sha256"]
        checked.append({"path": str(path), "sha256": actual})
        if actual != expected:
            failures.append(f"{path}: expected {expected}, got {actual}")

    report = {"status": "failed" if failures else "ok", "checked": checked, "failures": failures}
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
