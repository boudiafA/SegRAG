"""
Stage 0 runner: execute the full GitHub-facing pipeline stage by stage.

Pipeline overview
1. Stage 0 (optional): convert PASCAL-5i / VOC masks into `train.json` and `test.json`.
2. Stage 1: build the raw DINOv3 feature bank from `train.json`, score it once at 0.6, then apply the default `adaptive_q75` filter.
3. Stage 2: cache reusable prompts for feature matching.
4. Stage 3: evaluate feature matching without SAM3.
5. Stage 4: run SAM3 with `text_prompt`, `point_prompt`, or `text_and_point`.
6. Stage 5: merge text-only and point-only SAM3 masks and evaluate the merged result.

Expected dataset folder structure
`dataset_root/`
- `train.json`
- `test.json`
- image files referenced by `file_name` in the JSONs
- `feature_bank_dinov3_vitl16_1536/` created by Stage 1
- `feature_bank_dinov3_vitl16_1536_scored_thr060/` created by Stage 1
- `feature_bank_adaptive_q75_from_thr060/` created by Stage 1

Expected annotation file format
- COCO-style JSON with top-level `images`, `annotations`, and `categories`.
- `images` entries should include `id`, `file_name`, `width`, and `height`.
- `annotations` entries should include `id`, `image_id`, `category_id`, `segmentation`, `bbox`, `area`, and `iscrowd`.
- `segmentation` may be COCO RLE or polygons. The provided Stage 0 script writes compressed RLE.

Notes
- Stage 1 reads `train.json`.
- Stages 2 to 5 read `test.json` by default.
- Stage 5 expects a Stage 4 text-only run and at least one Stage 4 point-only run.
"""

from __future__ import annotations

import argparse
import json
import os

from GitHub import Stage0_prepare_pascal5i_annotations as stage0_prepare
from GitHub import Stage1_filter_scored_feature_bank as stage1_filter
from GitHub import Stage1_score_feature_bank as stage1_score
from GitHub import Stage2_cache_feature_matching as stage2
from GitHub import Stage3_evaluate_feature_matching as stage3
from GitHub import Stage4_evaluate_sam3 as stage4
from GitHub import Stage5_evaluate_merged_sam3_masks as stage5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full GitHub-facing pipeline from Stage 1 to Stage 5.")
    parser.add_argument("--dataset-root", default=".")
    parser.add_argument("--image-dir", default=None, help="Fallback image dir used when train/eval image dirs are not set.")
    parser.add_argument("--train-image-dir", default=None)
    parser.add_argument("--eval-image-dir", default=None)
    parser.add_argument("--train-ann-file", default=None)
    parser.add_argument("--test-ann-file", default=None)
    parser.add_argument("--start-stage", type=int, default=1)
    parser.add_argument("--end-stage", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-stage3", action="store_true")

    parser.add_argument(
        "--prepare-pascal5i-root",
        default=None,
        help="Optional VOC/PASCAL root. If provided and Stage 0 is within range, train/test JSONs are generated first.",
    )

    parser.add_argument("--stage2-method", default="compare_all")
    parser.add_argument("--stage3-method", default="compare_all")
    parser.add_argument(
        "--stage4-modes",
        nargs="+",
        default=["text_prompt", "point_prompt", "text_and_point"],
        choices=stage4.PROMPT_MODES,
    )
    parser.add_argument(
        "--feature-matching-methods",
        nargs="+",
        default=["hybrid"],
        help="Feature-matching methods used for Stage 4 point-based modes and Stage 5.",
    )
    parser.add_argument("--save-mask-json", action="store_true")
    parser.add_argument("--stage5-strategies", nargs="+", default=None)
    parser.add_argument("--stage1-max-images-per-class", type=int, default=None)

    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for Stage 2-4 evaluation images.")
    parser.add_argument("--max-references", type=int, default=None)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _run_stage0(args: argparse.Namespace) -> dict:
    if not args.prepare_pascal5i_root:
        return {"skipped": True, "reason": "prepare_pascal5i_root_not_provided"}
    stage_args = argparse.Namespace(
        dataset_root=args.prepare_pascal5i_root,
        train_output=args.train_ann_file,
        test_output=args.test_ann_file,
    )
    return stage0_prepare.run(stage_args)


def _run_stage1(args: argparse.Namespace) -> dict:
    score_args = argparse.Namespace(
        dataset_root=args.dataset_root,
        train_ann_file=args.train_ann_file,
        image_dir=args.train_image_dir or args.image_dir,
        raw_feature_bank_dir=None,
        scored_feature_bank_dir=None,
        skip_build=False,
        resume=args.resume,
        image_size=1536,
        patch_size=16,
        model_name="dinov3_vitl16",
        repo_path="./",
        weights_path="./weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        mask_coverage_threshold=0.90,
        features_per_class_threshold=None,
        max_images_per_class=args.stage1_max_images_per_class,
        batch_size=4,
        scan_workers=8,
        checkpoint_name="_build_feature_bank_resume.json",
        selection_mode="top-k-images",
        max_source_images=args.stage1_max_images_per_class or 100,
        top_k_features=None,
        keep_threshold=0.6,
        min_matches=3,
        target_image_limit=100,
        query_chunk=256,
        target_batch_size=16,
        num_workers=8,
        sim_floor=0.0,
    )
    score_result = stage1_score.run(score_args)
    filter_result = stage1_filter.run(
        argparse.Namespace(
            scored_feature_bank_dir=score_result["config"]["scored_feature_bank_dir"],
            filtered_feature_bank_dir=os.path.join(
                args.dataset_root,
                "feature_bank_adaptive_q75_from_thr060",
            ),
            method="adaptive_q75",
            keep_threshold=None,
            top_k_features=10000,
            n_clusters="auto",
            min_cluster_size=5,
            resume=args.resume,
        )
    )
    return {"score": score_result, "filter": filter_result}


def _run_stage2(args: argparse.Namespace) -> dict:
    stage_args = argparse.Namespace(
        method=args.stage2_method,
        dataset_root=args.dataset_root,
        annotation_file=args.test_ann_file,
        image_dir=args.eval_image_dir or args.image_dir,
        feature_bank_dir=None,
        filtered_bank_dir=None,
        max_images=args.max_images,
        max_references=args.max_references,
        num_points=10,
        sim_threshold=0.8,
        peak_threshold=0.2,
        min_peak_distance=10,
        suppression_margin=0.0,
        no_suppression=False,
        loose_threshold=0.8,
        min_component_size=4,
        resume=args.resume,
    )
    return stage2.run(stage_args)


def _run_stage3(args: argparse.Namespace) -> dict:
    stage_args = argparse.Namespace(
        method=args.stage3_method,
        dataset_root=args.dataset_root,
        feature_bank_dir=None,
        annotation_file=args.test_ann_file,
        image_dir=args.eval_image_dir or args.image_dir,
        absolute_threshold="0.8",
        relative_threshold=0.2,
        hybrid_threshold=0.8,
        max_images=args.max_images,
        max_references=args.max_references,
        min_peak_distance=10,
        min_component_size=4,
        suppression_margin=0.0,
        no_suppression=False,
        output_json=None,
        resume=args.resume,
    )
    return stage3.run(stage_args)


def _run_stage4(args: argparse.Namespace) -> list[dict]:
    results = []
    for prompt_mode in args.stage4_modes:
        if prompt_mode == "text_prompt":
            stage_args = argparse.Namespace(
                prompt_mode="text_prompt",
                feature_matching_method="hybrid",
                dataset_root=args.dataset_root,
                annotation_file=args.test_ann_file,
                image_dir=args.eval_image_dir or args.image_dir,
                feature_bank_dir=None,
                filtered_bank_dir=None,
                prompt_cache_dir=None,
                output_dir=None,
                max_images=args.max_images,
                max_references=args.max_references,
                max_points_per_class=None,
                num_points=10,
                sim_threshold=0.8,
                peak_threshold=0.2,
                min_peak_distance=10,
                suppression_margin=0.0,
                no_suppression=False,
                loose_threshold=0.8,
                min_component_size=4,
                hybrid_validation_threshold=0.8,
                cleanup_every=50,
                num_workers=8,
                prefetch_factor=16,
                save_mask_json=args.save_mask_json,
                resume=args.resume,
            )
            results.append({"prompt_mode": prompt_mode, "method": None, "result": stage4.run(stage_args)})
            continue

        for method in args.feature_matching_methods:
            stage_args = argparse.Namespace(
                prompt_mode=prompt_mode,
                feature_matching_method=method,
                dataset_root=args.dataset_root,
                annotation_file=args.test_ann_file,
                image_dir=args.eval_image_dir or args.image_dir,
                feature_bank_dir=None,
                filtered_bank_dir=None,
                prompt_cache_dir=None,
                output_dir=None,
                max_images=args.max_images,
                max_references=args.max_references,
                max_points_per_class=None,
                num_points=10,
                sim_threshold=0.8,
                peak_threshold=0.2,
                min_peak_distance=10,
                suppression_margin=0.0,
                no_suppression=False,
                loose_threshold=0.8,
                min_component_size=4,
                hybrid_validation_threshold=0.8,
                cleanup_every=50,
                num_workers=8,
                prefetch_factor=16,
                save_mask_json=args.save_mask_json,
                resume=args.resume,
            )
            results.append({"prompt_mode": prompt_mode, "method": method, "result": stage4.run(stage_args)})
    return results


def _run_stage5(args: argparse.Namespace) -> list[dict]:
    results = []
    for method in args.feature_matching_methods:
        stage_args = argparse.Namespace(
            method=method,
            strategies=args.stage5_strategies,
            dataset_root=args.dataset_root,
            annotation_file=args.test_ann_file,
            text_mask_json=None,
            point_mask_json=None,
            output_dir=None,
            nms_iou_threshold=0.5,
            num_workers=8,
            resume=args.resume,
        )
        results.append({"method": method, "result": stage5.run(stage_args)})
    return results


def run(args: argparse.Namespace) -> dict:
    results: dict[str, object] = {"config": vars(args).copy(), "stages": {}}

    if args.start_stage <= 0 <= args.end_stage:
        results["stages"]["stage0"] = _run_stage0(args)
    if args.start_stage <= 1 <= args.end_stage:
        results["stages"]["stage1"] = _run_stage1(args)
    if args.start_stage <= 2 <= args.end_stage:
        results["stages"]["stage2"] = _run_stage2(args)
    if args.start_stage <= 3 <= args.end_stage and not args.skip_stage3:
        results["stages"]["stage3"] = _run_stage3(args)
    if args.start_stage <= 4 <= args.end_stage:
        results["stages"]["stage4"] = _run_stage4(args)
    if args.start_stage <= 5 <= args.end_stage:
        results["stages"]["stage5"] = _run_stage5(args)

    return results


def main(argv: list[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
