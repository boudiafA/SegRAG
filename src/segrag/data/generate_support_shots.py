from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np


DEFAULT_SHOTS = [1, 5, 20, 50, 100]
PROTOCOL_VERSION = "support-shots-v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic support-shot lists per class from prepared support manifests."
    )
    parser.add_argument(
        "--dataset-root",
        nargs="+",
        required=True,
        help="One or more dataset roots to process.",
    )
    parser.add_argument(
        "--shots",
        nargs="+",
        type=int,
        default=DEFAULT_SHOTS,
        help="Shot counts to prepare. Defaults to 1 5 20 50 100.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def load_json(path: str) -> dict:
    with open(path, "r") as handle:
        return json.load(handle)


def dump_json_atomic(path: str, payload: dict) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, path)


def resolve_filtered_bank_dir(dataset_root: str) -> str | None:
    candidates = [
        os.path.join(dataset_root, "feature_bank_adaptive_q75_from_thr060"),
        os.path.join(dataset_root, "train", "feature_bank_adaptive_q75_from_thr060"),
        os.path.join(dataset_root, "feature_bank_dinov3_vitl16_intra_class_filtered_1536_imgcap100"),
        os.path.join(dataset_root, "train", "feature_bank_dinov3_vitl16_intra_class_filtered_1536_imgcap100"),
        os.path.join(dataset_root, "feature_bank_dinov3_vitl16_intra_class_filtered_1536_imgcap30"),
        os.path.join(dataset_root, "train", "feature_bank_dinov3_vitl16_intra_class_filtered_1536_imgcap30"),
        os.path.join(dataset_root, "feature_bank_dinov3_vitl16_intra_class_filtered_1536"),
        os.path.join(dataset_root, "train", "feature_bank_dinov3_vitl16_intra_class_filtered_1536"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return None


def load_support_manifest(dataset_root: str) -> list[dict]:
    manifest_path = os.path.join(dataset_root, "support_manifest.json")
    data = load_json(manifest_path)
    return data["items"]


def load_support_protocol(dataset_root: str) -> dict:
    return load_json(os.path.join(dataset_root, "support_protocol.json"))


def build_support_index(support_items: list[dict]) -> dict[int, dict[int, dict]]:
    by_class: dict[int, dict[int, dict]] = defaultdict(dict)
    for item in support_items:
        by_class[int(item["class_id"])][int(item["image_id"])] = item
    return {cid: dict(items) for cid, items in by_class.items()}


def _parse_image_id_from_filename(fname: str) -> int | None:
    prefix = fname.split("_", 1)[0]
    try:
        return int(prefix)
    except ValueError:
        return None


def collect_filtered_image_quality(filtered_bank_dir: str) -> dict[str, dict[int, dict]]:
    if filtered_bank_dir is None:
        return {}
    quality_by_class: dict[str, dict[int, dict]] = {}
    class_dirs = sorted(
        entry
        for entry in os.listdir(filtered_bank_dir)
        if os.path.isdir(os.path.join(filtered_bank_dir, entry)) and not entry.startswith("_")
    )
    for class_name in class_dirs:
        class_dir = os.path.join(filtered_bank_dir, class_name)
        per_image_scores: dict[int, list[np.ndarray]] = defaultdict(list)
        per_image_vectors: dict[int, int] = defaultdict(int)
        for fname in os.listdir(class_dir):
            if fname.endswith(".pt"):
                image_id = _parse_image_id_from_filename(fname)
                if image_id is None:
                    continue
                per_image_vectors[image_id] += 1
                continue
            if not fname.endswith(".scores.npy"):
                continue
            image_id = _parse_image_id_from_filename(fname)
            if image_id is None:
                continue
            scores = np.load(os.path.join(class_dir, fname))
            per_image_scores[image_id].append(scores)
        class_quality = {}
        for image_id in sorted(per_image_vectors):
            chunks = per_image_scores.get(image_id, [])
            all_scores = np.concatenate(chunks, axis=0) if chunks else np.empty((0,), dtype=np.float32)
            class_quality[image_id] = {
                "image_id": image_id,
                "has_scores": bool(all_scores.size),
                "max_score": float(all_scores.max()) if all_scores.size else None,
                "mean_score": float(all_scores.mean()) if all_scores.size else None,
                "median_score": float(np.median(all_scores)) if all_scores.size else None,
                "feature_count": per_image_vectors[image_id],
            }
        quality_by_class[class_name] = class_quality
    return quality_by_class


def order_supports_for_class(
    class_name: str,
    supports_by_image: dict[int, dict],
    filtered_quality_by_image: dict[int, dict],
) -> tuple[list[dict], dict]:
    filtered_candidates = []
    fallback_candidates = []
    for image_id, support in supports_by_image.items():
        quality = filtered_quality_by_image.get(image_id)
        if quality is not None:
            filtered_candidates.append(
                {
                    "image_id": image_id,
                    "class_name": class_name,
                    "source": "filtered",
                    "quality": quality,
                    "support": support,
                }
            )
        else:
            fallback_candidates.append(
                {
                    "image_id": image_id,
                    "class_name": class_name,
                    "source": "prefilter_fallback",
                    "quality": None,
                    "support": support,
                }
            )

    filtered_candidates.sort(
        key=lambda item: (
            0 if item["quality"]["has_scores"] else 1,
            -(item["quality"]["max_score"] if item["quality"]["max_score"] is not None else float("-inf")),
            -(item["quality"]["mean_score"] if item["quality"]["mean_score"] is not None else float("-inf")),
            -(item["quality"]["median_score"] if item["quality"]["median_score"] is not None else float("-inf")),
            -item["quality"]["feature_count"],
            item["image_id"],
        )
    )
    fallback_candidates.sort(key=lambda item: item["image_id"])
    ordered = filtered_candidates + fallback_candidates
    stats = {
        "filtered_available": len(filtered_candidates),
        "fallback_available": len(fallback_candidates),
        "total_available": len(ordered),
    }
    return ordered, stats


def build_shot_protocol_for_dataset(dataset_root: str, shots: list[int]) -> dict:
    shots = sorted(set(int(shot) for shot in shots))
    support_items = load_support_manifest(dataset_root)
    protocol = load_support_protocol(dataset_root)
    support_index = build_support_index(support_items)
    filtered_bank_dir = resolve_filtered_bank_dir(dataset_root)
    filtered_quality_by_class = collect_filtered_image_quality(filtered_bank_dir)

    shot_items: dict[str, list[dict]] = {str(shot): [] for shot in shots}
    per_class_summary = []
    valid_classes_by_shot: dict[str, list[dict]] = {str(shot): [] for shot in shots}

    for class_row in protocol["class_vocabulary"]:
        class_id = int(class_row["class_id"])
        class_name = str(class_row["class_name"])
        supports_by_image = support_index.get(class_id, {})
        ordered_supports, stats = order_supports_for_class(
            class_name=class_name,
            supports_by_image=supports_by_image,
            filtered_quality_by_image=filtered_quality_by_class.get(class_name, {}),
        )

        shot_counts = {}
        for shot in shots:
            selected = ordered_supports[: min(shot, len(ordered_supports))]
            shot_counts[str(shot)] = len(selected)
            if len(ordered_supports) >= shot:
                valid_classes_by_shot[str(shot)].append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                    }
                )
            for rank, entry in enumerate(selected, start=1):
                support = entry["support"]
                shot_items[str(shot)].append(
                    {
                        "dataset": os.path.basename(dataset_root),
                        "shot": shot,
                        "rank_within_class": rank,
                        "class_id": class_id,
                        "class_name": class_name,
                        "image_id": int(support["image_id"]),
                        "image_path": support["image_path"],
                        "mask_path": support["mask_path"],
                        "split": support["split"],
                        "selection_source": entry["source"],
                        "quality": entry["quality"],
                    }
                )

        per_class_summary.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "filtered_bank_dir": filtered_bank_dir,
                "filtered_available": stats["filtered_available"],
                "fallback_available": stats["fallback_available"],
                "total_available": stats["total_available"],
                "shot_counts": shot_counts,
                "is_valid_for_shots": {
                    str(shot): stats["total_available"] >= shot for shot in shots
                },
            }
        )

    output = {
        "protocol_version": PROTOCOL_VERSION,
        "generator": {
            "script": "segrag.data.generate_support_shots",
            "selection_seed": None,
            "selection_mode": "deterministic",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "dataset": os.path.basename(dataset_root),
        "dataset_root": dataset_root,
        "filtered_bank_dir_used": filtered_bank_dir,
        "shots": shots,
        "shot_availability_policy": {
            "rule": "evaluate_only_classes_with_at_least_k_total_supports",
            "description": (
                "For k-shot evaluation, a class is valid only if the total support pool contains at least k support "
                "examples. Classes with fewer than k total supports are excluded from k-shot mIoU. "
                "Filtered-first ordering only determines which supports are chosen within the valid pool."
            ),
        },
        "selection_policy": {
            "description": (
                "For each class, rank support images from the current filtered bank first, "
                "using image-level quality derived from surviving feature scores. "
                "Order filtered candidates by max_score, mean_score, median_score, feature_count, image_id. "
                "If a requested shot count exceeds the filtered-image pool, fill the remainder from the pre-filter "
                "support pool in ascending image_id order."
            ),
            "low_shot_bias": "1-shot and 5-shot are prefixes of the same highest-quality filtered-first ordering.",
            "fairness_note": "Shot counts are defined on original support images. Filtering preference is only used to avoid recomputing the current method.",
        },
        "support_query_rule": {
            "query_split_used_for_all_shots": True,
            "support_query_overlap_allowed": False,
            "support_query_overlap_enforcement": (
                "Support examples come from the train split and queries come from the official eval split "
                "(val/test depending on the dataset export), so support/query overlap is disallowed by split."
            ),
            "same_image_multiple_classes_allowed": True,
            "same_image_multiple_classes_note": (
                "A training image may appear as support for multiple classes, each with its own class-specific binary mask."
            ),
        },
        "valid_classes_by_shot": valid_classes_by_shot,
        "per_class_summary": per_class_summary,
        "shot_items": shot_items,
    }

    output_path = os.path.join(dataset_root, "support_shots.json")
    dump_json_atomic(output_path, output)
    return {
        "dataset_root": dataset_root,
        "output_path": output_path,
        "filtered_bank_dir_used": filtered_bank_dir,
        "classes": len(per_class_summary),
        "shots": shots,
    }


def run(args: argparse.Namespace) -> dict:
    datasets = {}
    for dataset_root in args.dataset_root:
        datasets[dataset_root] = build_shot_protocol_for_dataset(dataset_root, args.shots)
    return {
        "shots": sorted(set(args.shots)),
        "datasets": datasets,
    }


def main(argv: list[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
