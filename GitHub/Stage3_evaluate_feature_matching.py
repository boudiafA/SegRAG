"""
Stage 3: evaluate feature matching before SAM3 segmentation.
"""

from __future__ import annotations

import argparse
import json
import os

from GitHub.common import COMPARE_ALL, default_dataset_paths, validate_method
from GitHub.non_sam_eval_impl import run as run_impl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 3: evaluate non-SAM feature matching for one method or all methods.")
    parser.add_argument("--method", type=str, default=COMPARE_ALL)
    parser.add_argument("--dataset-root", type=str, default=".")
    parser.add_argument("--feature-bank-dir", type=str, default=None)
    parser.add_argument("--annotation-file", type=str, default=None)
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--absolute-threshold", type=str, default="0.8")
    parser.add_argument("--relative-threshold", type=float, default=0.2)
    parser.add_argument("--hybrid-threshold", type=float, default=0.8)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-references", type=int, default=None)
    parser.add_argument("--min-peak-distance", type=int, default=10)
    parser.add_argument("--min-component-size", type=int, default=4)
    parser.add_argument("--suppression-margin", type=float, default=0.0)
    parser.add_argument("--no-suppression", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    validate_method(args.method, allow_compare_all=True)
    defaults = default_dataset_paths(args.dataset_root)
    args.annotation_file = args.annotation_file or defaults.annotation_file
    args.image_dir = args.image_dir or defaults.image_dir
    args.feature_bank_dir = args.feature_bank_dir or defaults.filtered_feature_bank_dir
    results = run_impl(args)
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w") as handle:
            json.dump(results, handle, indent=2, default=str)
    return results


def main(argv: list[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
