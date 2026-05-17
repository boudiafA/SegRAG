"""
Stage 5: merge text-only and point-only SAM3 masks and evaluate the merged output.
"""

from __future__ import annotations

import argparse
import json
import os

from segrag.utils.common import validate_method
from segrag.stages.merge_masks_impl import run as run_impl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 5: merge text-only and point-only SAM3 masks and evaluate the merged results.")
    parser.add_argument("--method", type=str, default="hybrid")
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=None,
        choices=["union", "intersection", "confidence_weighted", "nms_instances"],
    )
    parser.add_argument("--dataset-root", type=str, default=".")
    parser.add_argument("--annotation-file", type=str, default=None)
    parser.add_argument("--text-mask-json", type=str, default=None)
    parser.add_argument("--point-mask-json", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    method = validate_method(args.method)
    if args.annotation_file is None:
        args.annotation_file = os.path.join(args.dataset_root, "test.json")
    if args.text_mask_json is None:
        args.text_mask_json = os.path.join(
            args.dataset_root,
            "evaluation_results_text_only_sam3",
            "predicted_masks.json",
        )
    if args.point_mask_json is None:
        args.point_mask_json = os.path.join(
            args.dataset_root,
            f"evaluation_results_points_only_sam3_{method}",
            "predicted_masks.json",
        )
    if args.output_dir is None:
        args.output_dir = os.path.join(
            args.dataset_root,
            f"evaluation_results_points_only_sam3_merged_{method}",
        )
    return run_impl(args)


def main(argv: list[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
