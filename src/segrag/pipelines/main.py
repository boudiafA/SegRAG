"""
High-level main pipeline for the SegRAG workflow.

This entrypoint is meant to be simpler than the stage-oriented scripts:
- it takes one dataset root
- auto-resolves train/validation annotations and image directories
- exposes a small set of practical knobs
- runs only the stages needed for the selected segmentation mode
- writes feature banks, caches, and evaluation outputs under the same dataset root

Preferred dataset layout
`dataset_root/`
- `train.json` or `lvis_v1_train.json`
- `val.json` or `test.json` or `lvis_v1_val.json`
- image files, or `train/images` and `val/images`

Also supported
- `dataset_root/train/lvis_v1_train.json`
- `dataset_root/val/lvis_v1_val.json`
- `dataset_root/train/images`
- `dataset_root/val/images`
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

from segrag.stages import filter_bank as stage1_filter
from segrag.stages import score_bank as stage1_score
from segrag.stages import cache_prompts as stage2
from segrag.stages import evaluate_sam3 as stage4


SEGMENTATION_METHODS = ("text-only", "point-only", "text-and-point", "all")


@dataclass(frozen=True)
class DatasetLayout:
    dataset_root: str
    train_ann_file: str
    val_ann_file: str
    train_image_dir: str
    val_image_dir: str

    @property
    def raw_feature_bank_dir(self) -> str:
        return os.path.join(self.dataset_root, "feature_bank_dinov3_vitl16_1536")

    @property
    def scored_feature_bank_dir(self) -> str:
        return os.path.join(self.dataset_root, "feature_bank_dinov3_vitl16_1536_scored_thr060")

    @property
    def filtered_feature_bank_dir(self) -> str:
        return os.path.join(
            self.dataset_root,
            "feature_bank_adaptive_q75_from_thr060",
        )


def _parse_optional_threshold(value: str) -> float | None:
    lowered = value.strip().lower()
    if lowered in {"none", "null"}:
        return None
    return float(value)


def _parse_optional_int(value: str) -> int | None:
    lowered = value.strip().lower()
    if lowered in {"none", "null"}:
        return None
    return int(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the main segmentation pipeline from one dataset root."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Dataset root containing train/val JSON annotations and image directories.",
    )
    parser.add_argument(
        "--segmentation-method",
        choices=SEGMENTATION_METHODS,
        required=True,
        help="Segmentation mode to run: text-only, point-only, text-and-point, or all.",
    )
    parser.add_argument(
        "--feature-matching-method",
        default="hybrid",
        choices=("absolute_similarity", "relative_similarity", "hybrid"),
        help="Feature matching method used by point-based segmentation modes.",
    )
    parser.add_argument(
        "--reference-images-per-class",
        type=int,
        default=30,
        help="Maximum number of source images per class used while building the raw bank.",
    )
    parser.add_argument(
        "--raw-bank-batch-size",
        type=int,
        default=4,
        help="Batch size used during Stage 1 raw-bank DINOv3 feature extraction.",
    )
    parser.add_argument(
        "--feature-top-k",
        type=_parse_optional_int,
        default=10000,
        help="Top-k features kept after the default adaptive_q75 Stage 1 filtering pass. Use `none` to disable the cap.",
    )
    parser.add_argument(
        "--feature-accuracy-threshold",
        type=_parse_optional_threshold,
        default=0.8,
        help="Similarity threshold used by Stage 2 prompt generation and Stage 4 point validation. It does not change the default adaptive_q75 Stage 1 filter.",
    )
    parser.add_argument(
        "--max-references",
        type=int,
        default=10000,
        help="Maximum number of cached feature-bank references used by Stage 2 and Stage 4.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap on validation images for Stage 2 and Stage 4.",
    )
    parser.add_argument(
        "--save-mask-json",
        action="store_true",
        help="Export predicted masks JSON from Stage 4.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save a summary JSON for this main pipeline run.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _first_existing(paths: list[str]) -> str | None:
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def _resolve_dataset_layout(dataset_root: str) -> DatasetLayout:
    train_ann = _first_existing(
        [
            os.path.join(dataset_root, "train.json"),
            os.path.join(dataset_root, "lvis_v1_train.json"),
            os.path.join(dataset_root, "train", "train.json"),
            os.path.join(dataset_root, "train", "lvis_v1_train.json"),
        ]
    )
    val_ann = _first_existing(
        [
            os.path.join(dataset_root, "val.json"),
            os.path.join(dataset_root, "test.json"),
            os.path.join(dataset_root, "lvis_v1_val.json"),
            os.path.join(dataset_root, "val", "val.json"),
            os.path.join(dataset_root, "val", "test.json"),
            os.path.join(dataset_root, "val", "lvis_v1_val.json"),
        ]
    )
    if train_ann is None:
        raise FileNotFoundError(
            f"Could not find a training annotation JSON under {dataset_root}. "
            "Expected train.json, lvis_v1_train.json, or a train/ subdirectory variant."
        )
    if val_ann is None:
        raise FileNotFoundError(
            f"Could not find a validation annotation JSON under {dataset_root}. "
            "Expected val.json, test.json, lvis_v1_val.json, or a val/ subdirectory variant."
        )

    train_image_dir = _first_existing(
        [
            os.path.join(dataset_root, "train", "images"),
            os.path.join(dataset_root, "images", "train"),
            os.path.join(dataset_root, "images"),
            dataset_root,
        ]
    )
    val_image_dir = _first_existing(
        [
            os.path.join(dataset_root, "val", "images"),
            os.path.join(dataset_root, "test", "images"),
            os.path.join(dataset_root, "images", "val"),
            os.path.join(dataset_root, "images"),
            dataset_root,
        ]
    )
    if train_image_dir is None or not os.path.isdir(train_image_dir):
        raise FileNotFoundError(f"Could not resolve a training image directory under {dataset_root}.")
    if val_image_dir is None or not os.path.isdir(val_image_dir):
        raise FileNotFoundError(f"Could not resolve a validation image directory under {dataset_root}.")

    return DatasetLayout(
        dataset_root=dataset_root,
        train_ann_file=train_ann,
        val_ann_file=val_ann,
        train_image_dir=train_image_dir,
        val_image_dir=val_image_dir,
    )


def _run_stage1(args: argparse.Namespace, layout: DatasetLayout) -> dict:
    score_args = argparse.Namespace(
        dataset_root=layout.dataset_root,
        train_ann_file=layout.train_ann_file,
        image_dir=layout.train_image_dir,
        raw_feature_bank_dir=layout.raw_feature_bank_dir,
        scored_feature_bank_dir=layout.scored_feature_bank_dir,
        skip_build=False,
        resume=args.resume,
        image_size=1536,
        patch_size=16,
        model_name="dinov3_vitl16",
        repo_path="./",
        weights_path="./weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        mask_coverage_threshold=0.90,
        features_per_class_threshold=None,
        max_images_per_class=args.reference_images_per_class,
        batch_size=args.raw_bank_batch_size,
        scan_workers=8,
        checkpoint_name="_build_feature_bank_resume.json",
        selection_mode="top-k-images",
        max_source_images=args.reference_images_per_class,
        top_k_features=None,
        keep_threshold=0.6,
        min_matches=3,
        target_image_limit=100,
        query_chunk=256,
        target_batch_size=16,
        num_workers=8,
        sim_floor=0.0,
    )
    filter_args = argparse.Namespace(
        scored_feature_bank_dir=layout.scored_feature_bank_dir,
        filtered_feature_bank_dir=layout.filtered_feature_bank_dir,
        method="adaptive_q75",
        keep_threshold=None,
        top_k_features=args.feature_top_k,
        n_clusters="auto",
        min_cluster_size=5,
        resume=args.resume,
    )
    return {
        "score": stage1_score.run(score_args),
        "filter": stage1_filter.run(filter_args),
    }


def _run_stage2(args: argparse.Namespace, layout: DatasetLayout) -> dict:
    stage_args = argparse.Namespace(
        method=args.feature_matching_method,
        dataset_root=layout.dataset_root,
        annotation_file=layout.val_ann_file,
        image_dir=layout.val_image_dir,
        feature_bank_dir=None,
        filtered_bank_dir=layout.filtered_feature_bank_dir,
        max_images=args.max_images,
        max_references=args.max_references,
        num_points=10,
        sim_threshold=args.feature_accuracy_threshold,
        peak_threshold=0.2,
        min_peak_distance=10,
        suppression_margin=0.0,
        no_suppression=False,
        loose_threshold=args.feature_accuracy_threshold,
        min_component_size=4,
        resume=args.resume,
    )
    return stage2.run(stage_args)


def _run_stage4_once(args: argparse.Namespace, layout: DatasetLayout, segmentation_method: str) -> dict:
    mode_map = {
        "text-only": "text_prompt",
        "point-only": "point_prompt",
        "text-and-point": "text_and_point",
    }
    stage_args = argparse.Namespace(
        prompt_mode=mode_map[segmentation_method],
        feature_matching_method=args.feature_matching_method,
        dataset_root=layout.dataset_root,
        annotation_file=layout.val_ann_file,
        image_dir=layout.val_image_dir,
        feature_bank_dir=None,
        filtered_bank_dir=layout.filtered_feature_bank_dir,
        prompt_cache_dir=None,
        output_dir=None,
        max_images=args.max_images,
        max_references=args.max_references,
        max_points_per_class=None,
        num_points=10,
        sim_threshold=args.feature_accuracy_threshold,
        peak_threshold=0.2,
        min_peak_distance=10,
        suppression_margin=0.0,
        no_suppression=False,
        loose_threshold=args.feature_accuracy_threshold,
        min_component_size=4,
        hybrid_validation_threshold=args.feature_accuracy_threshold,
        cleanup_every=50,
        num_workers=8,
        prefetch_factor=16,
        save_mask_json=args.save_mask_json,
        resume=args.resume,
    )
    return stage4.run(stage_args)


def run(args: argparse.Namespace) -> dict:
    layout = _resolve_dataset_layout(args.dataset_root)
    print("Resolved dataset layout:")
    print(f"  train annotations : {layout.train_ann_file}")
    print(f"  val annotations   : {layout.val_ann_file}")
    print(f"  train images      : {layout.train_image_dir}")
    print(f"  val images        : {layout.val_image_dir}")
    print(f"  raw feature bank  : {layout.raw_feature_bank_dir}")
    print(f"  scored bank       : {layout.scored_feature_bank_dir}")
    print(f"  filtered bank     : {layout.filtered_feature_bank_dir}")
    results: dict[str, object] = {
        "config": {
            **vars(args),
            "resolved_train_ann_file": layout.train_ann_file,
            "resolved_val_ann_file": layout.val_ann_file,
            "resolved_train_image_dir": layout.train_image_dir,
            "resolved_val_image_dir": layout.val_image_dir,
            "resolved_raw_feature_bank_dir": layout.raw_feature_bank_dir,
            "resolved_scored_feature_bank_dir": layout.scored_feature_bank_dir,
            "resolved_filtered_feature_bank_dir": layout.filtered_feature_bank_dir,
            "default_stage1_score_threshold": 0.6,
            "default_stage1_score_top_k": None,
            "default_stage1_filter_method": "adaptive_q75",
        },
        "stages": {},
    }

    if args.segmentation_method in ("point-only", "text-and-point", "all"):
        print("Running Stage 1: build raw bank, score features, then apply adaptive_q75 filtering")
        results["stages"]["stage1"] = _run_stage1(args, layout)
        print("Running Stage 2: cache feature matching prompts")
        results["stages"]["stage2"] = _run_stage2(args, layout)
    else:
        print("Skipping Stage 1 and Stage 2 for text-only segmentation")
        results["stages"]["stage1"] = {"skipped": True, "reason": "text-only segmentation does not use the feature bank"}
        results["stages"]["stage2"] = {"skipped": True, "reason": "text-only segmentation does not use cached prompts"}

    if args.segmentation_method == "all":
        print("Running Stage 4: text-only segmentation")
        text_only = _run_stage4_once(args, layout, "text-only")
        print("Running Stage 4: point-only segmentation")
        point_only = _run_stage4_once(args, layout, "point-only")
        print("Running Stage 4: text-and-point segmentation")
        text_and_point = _run_stage4_once(args, layout, "text-and-point")
        results["stages"]["stage4"] = {
            "text-only": text_only,
            "point-only": point_only,
            "text-and-point": text_and_point,
        }
    else:
        print(f"Running Stage 4: {args.segmentation_method}")
        results["stages"]["stage4"] = _run_stage4_once(args, layout, args.segmentation_method)

    if args.output_json:
        out_dir = os.path.dirname(args.output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w") as handle:
            json.dump(results, handle, indent=2, default=str)

    return results


def main(argv: list[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
