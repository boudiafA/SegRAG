"""
Stage 4: run SAM3 with one of three prompt modes.

Supported prompt modes:
- `text_prompt`: class text only
- `point_prompt`: cached points only
- `text_and_point`: fused class text plus cached points
"""

from __future__ import annotations

import argparse
import json

from GitHub.common import default_dataset_paths, validate_method
from GitHub.sam3_points_only_impl import run as run_points_impl
from GitHub.sam3_text_and_points_impl import run as run_text_and_points_impl


PROMPT_MODES = ("text_prompt", "point_prompt", "text_and_point")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 4: run SAM3 with text prompts, point prompts, or fused text+point prompts.")
    parser.add_argument("--prompt-mode", type=str, default="point_prompt", choices=PROMPT_MODES)
    parser.add_argument("--feature-matching-method", type=str, default="hybrid")
    parser.add_argument("--dataset-root", type=str, default=".")
    parser.add_argument("--annotation-file", type=str, default=None)
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--feature-bank-dir", type=str, default=None)
    parser.add_argument("--filtered-bank-dir", type=str, default=None)
    parser.add_argument("--prompt-cache-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-references", type=int, default=None)
    parser.add_argument("--max-points-per-class", type=int, default=None)
    parser.add_argument("--num-points", type=int, default=10)
    parser.add_argument("--sim-threshold", type=float, default=0.8)
    parser.add_argument("--peak-threshold", type=float, default=0.2)
    parser.add_argument("--min-peak-distance", type=int, default=10)
    parser.add_argument("--suppression-margin", type=float, default=0.0)
    parser.add_argument("--no-suppression", action="store_true")
    parser.add_argument("--loose-threshold", type=float, default=0.8)
    parser.add_argument("--min-component-size", type=int, default=4)
    parser.add_argument("--hybrid-validation-threshold", type=float, default=0.8)
    parser.add_argument("--cleanup-every", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=16)
    parser.add_argument("--save-mask-json", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _resolve_mode(args: argparse.Namespace) -> argparse.Namespace:
    defaults = default_dataset_paths(args.dataset_root)
    args.annotation_file = args.annotation_file or defaults.annotation_file
    args.image_dir = args.image_dir or defaults.image_dir
    args.filtered_bank_dir = args.filtered_bank_dir or defaults.filtered_feature_bank_dir

    if args.prompt_mode == "text_prompt":
        args.feature_matching_approach = "text_only"
    else:
        validate_method(args.feature_matching_method)
        args.feature_matching_approach = args.feature_matching_method
    return args


def run(args: argparse.Namespace) -> dict:
    args = _resolve_mode(args)
    if args.prompt_mode in ("text_prompt", "point_prompt"):
        return run_points_impl(args)
    return run_text_and_points_impl(args)


def main(argv: list[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
