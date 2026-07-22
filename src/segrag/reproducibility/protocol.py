from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from segrag.utils.resume import save_json_atomic


PAPER_PROTOCOL_VERSION = "segrag-table4-v1"
SUPPORTED_SHOTS = (1, 5)


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def load_paper_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Expected a mapping in {config_path}")
    if config.get("protocol_version") != PAPER_PROTOCOL_VERSION:
        raise ValueError(
            f"Unsupported protocol version {config.get('protocol_version')!r}; "
            f"expected {PAPER_PROTOCOL_VERSION!r}."
        )
    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(config_path.parent)
    return config


def resolve_config_path(config: dict[str, Any], value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config["_config_dir"]) / path
    return str(path.resolve())


def resolve_dataset_path(dataset_root: str, value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(dataset_root) / path
    return str(path.resolve())


def selected_supports(manifest: dict[str, Any], shot: int) -> list[dict[str, Any]]:
    if shot not in SUPPORTED_SHOTS:
        raise ValueError(f"Paper reproduction supports shots {SUPPORTED_SHOTS}, got {shot}.")
    shot_key = str(shot)
    items = manifest.get("shot_items", {}).get(shot_key)
    if not isinstance(items, list):
        raise KeyError(f"Shot {shot} is missing from the support manifest.")

    valid_rows = manifest.get("valid_classes_by_shot", {}).get(shot_key)
    if valid_rows is None:
        return items
    valid_ids = {int(row["class_id"]) for row in valid_rows}
    return [row for row in items if int(row["class_id"]) in valid_ids]


def validate_support_manifest(manifest: dict[str, Any], dataset: str, shot: int) -> dict[str, Any]:
    if manifest.get("protocol_version") != PAPER_PROTOCOL_VERSION:
        raise ValueError("Support manifest does not use the frozen paper protocol.")
    if manifest.get("dataset") != dataset:
        raise ValueError(
            f"Support manifest dataset {manifest.get('dataset')!r} does not match {dataset!r}."
        )

    rows = selected_supports(manifest, shot)
    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[int(row["class_id"])].append(row)

    bad_counts = {
        class_id: len(class_rows)
        for class_id, class_rows in by_class.items()
        if len(class_rows) != shot
    }
    expected_ranks = set(range(1, shot + 1))
    bad_ranks = {
        class_id: sorted(int(row["rank_within_class"]) for row in class_rows)
        for class_id, class_rows in by_class.items()
        if {int(row["rank_within_class"]) for row in class_rows} != expected_ranks
    }
    duplicate_images = {
        class_id: [int(row["image_id"]) for row in class_rows]
        for class_id, class_rows in by_class.items()
        if len({int(row["image_id"]) for row in class_rows}) != len(class_rows)
    }
    if bad_counts or bad_ranks or duplicate_images:
        raise ValueError(
            f"Invalid {shot}-shot support manifest: bad_counts={bad_counts}, "
            f"bad_ranks={dict(list(bad_ranks.items())[:10])}, "
            f"duplicate_images={dict(list(duplicate_images.items())[:10])}"
        )
    return {
        "shot": shot,
        "classes": len(by_class),
        "items": len(rows),
        "nested_rank1": all(any(int(row["rank_within_class"]) == 1 for row in values) for values in by_class.values()),
    }


def query_pairs(annotation: dict[str, Any]) -> set[tuple[int, int]]:
    return {
        (int(row["image_id"]), int(row["category_id"]))
        for row in annotation.get("annotations", [])
    }


def validate_query_manifest(manifest: dict[str, Any], dataset: str, annotation: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("protocol_version") != PAPER_PROTOCOL_VERSION:
        raise ValueError("Query manifest does not use the frozen paper protocol.")
    if manifest.get("dataset") != dataset:
        raise ValueError(
            f"Query manifest dataset {manifest.get('dataset')!r} does not match {dataset!r}."
        )
    expected = query_pairs(annotation)
    actual = {
        (int(row["image_id"]), int(row["class_id"]))
        for row in manifest.get("items", [])
    }
    duplicate_pairs = len(manifest.get("items", [])) - len(actual)
    missing = expected - actual
    extra = actual - expected
    if missing or extra or duplicate_pairs:
        raise ValueError(
            "Query manifest does not match the evaluation annotation pairs: "
            f"missing={len(missing)}, extra={len(extra)}, duplicates={duplicate_pairs}."
        )
    return {
        "annotation_pairs": len(expected),
        "manifest_pairs": len(actual),
        "matches": True,
    }


def build_exact_support_annotations(
    train_annotation: dict[str, Any],
    support_rows: list[dict[str, Any]],
    output_path: str | None,
) -> dict[str, Any]:
    images_by_id = {int(row["id"]): dict(row) for row in train_annotation.get("images", [])}
    categories_by_id = {int(row["id"]): row for row in train_annotation.get("categories", [])}
    annotations_by_pair: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in train_annotation.get("annotations", []):
        annotations_by_pair[(int(row["image_id"]), int(row["category_id"]))].append(row)

    selected_images: dict[int, dict[str, Any]] = {}
    selected_annotations: dict[int, dict[str, Any]] = {}
    selected_category_ids: set[int] = set()
    missing_pairs: list[tuple[int, int]] = []

    for support in support_rows:
        image_id = int(support["image_id"])
        class_id = int(support["class_id"])
        matching_annotations = annotations_by_pair.get((image_id, class_id), [])
        if not matching_annotations:
            missing_pairs.append((image_id, class_id))
            continue
        if image_id not in images_by_id:
            raise KeyError(f"Support image {image_id} is missing from the training annotations.")

        image = dict(images_by_id[image_id])
        support_path = str(support.get("image_path", "")).replace("\\", "/")
        if support_path.startswith("reference_imgs/"):
            image["file_name"] = support_path[len("reference_imgs/"):]
        selected_images[image_id] = image
        selected_category_ids.add(class_id)
        for annotation in matching_annotations:
            selected_annotations[int(annotation["id"])] = annotation

    if missing_pairs:
        raise KeyError(
            f"Selected support pairs are absent from the training annotations; "
            f"count={len(missing_pairs)}, first={missing_pairs[0]}."
        )

    subset = {
        **{
            key: value
            for key, value in train_annotation.items()
            if key not in {"images", "annotations", "categories"}
        },
        "images": [selected_images[key] for key in sorted(selected_images)],
        "annotations": [selected_annotations[key] for key in sorted(selected_annotations)],
        "categories": [categories_by_id[key] for key in sorted(selected_category_ids)],
    }
    if output_path is not None:
        save_json_atomic(output_path, subset)
    return subset


def protocol_fingerprint(paths: dict[str, str]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in sorted(paths.items())}
