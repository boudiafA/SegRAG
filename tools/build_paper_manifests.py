#!/usr/bin/env python3
"""Build compact, path-independent paper manifests from the archived protocols."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROTOCOL_VERSION = "segrag-table4-v1"
DATASETS = {
    "pc59": "PC-59",
    "cityscapes": "Cityscapes",
    "ade20k150": "ADE20K_150",
    "lvis": "LVIS",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_compact(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, separators=(",", ":"))
        handle.write("\n")


def build_dataset(source_root: Path, output_root: Path, key: str, source_name: str) -> dict:
    source_dir = source_root / source_name
    support_source = source_dir / "support_shots.json"
    query_source = source_dir / "query_manifest.json"
    support = load_json(support_source)
    query = load_json(query_source)

    support_output = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": support["dataset"],
        "source_manifest_sha256": sha256_file(support_source),
        "selection_policy": support.get("selection_policy"),
        "support_query_rule": support.get("support_query_rule"),
        "shots": [1, 5],
        "valid_classes_by_shot": {
            shot: support["valid_classes_by_shot"][shot]
            for shot in ("1", "5")
        },
        "shot_items": {
            shot: support["shot_items"][shot]
            for shot in ("1", "5")
        },
    }
    query_output = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": support["dataset"],
        "source_manifest_sha256": sha256_file(query_source),
        "items": query["items"],
    }

    destination = output_root / key
    support_path = destination / "supports_1_5shot.json"
    query_path = destination / "queries.json"
    save_compact(support_path, support_output)
    save_compact(query_path, query_output)
    one_valid_ids = {
        int(row["class_id"])
        for row in support_output["valid_classes_by_shot"]["1"]
    }
    five_valid_ids = {
        int(row["class_id"])
        for row in support_output["valid_classes_by_shot"]["5"]
    }
    one_rows = [
        row for row in support_output["shot_items"]["1"]
        if int(row["class_id"]) in one_valid_ids
    ]
    five_rows = [
        row for row in support_output["shot_items"]["5"]
        if int(row["class_id"]) in five_valid_ids
    ]
    return {
        "dataset": support["dataset"],
        "support_manifest": str(support_path.relative_to(output_root.parent.parent)),
        "support_manifest_sha256": sha256_file(support_path),
        "support_source_sha256": support_output["source_manifest_sha256"],
        "query_manifest": str(query_path.relative_to(output_root.parent.parent)),
        "query_manifest_sha256": sha256_file(query_path),
        "query_source_sha256": query_output["source_manifest_sha256"],
        "one_shot_classes": len(one_valid_ids),
        "one_shot_items": len(one_rows),
        "five_shot_classes": len(five_valid_ids),
        "five_shot_items": len(five_rows),
        "query_items": len(query_output["items"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", default="splits/standard")
    args = parser.parse_args()

    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    reports = [
        build_dataset(source_root, output_root, key, source_name)
        for key, source_name in DATASETS.items()
    ]
    save_compact(
        output_root / "checksums.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "datasets": reports,
        },
    )
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
