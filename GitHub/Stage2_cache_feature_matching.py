"""
Stage 2: precompute reusable prompt caches for feature matching.
"""

from __future__ import annotations

import argparse
import json

from GitHub.cache_feature_matching import run as run_impl
from GitHub.common import COMPARE_ALL
from GitHub.feature_matching_backends import MAX_REFERENCES, SUPPRESSION_MARGIN


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 2: precompute resumable feature-matching prompt caches.")
    parser.add_argument("--method", type=str, default=COMPARE_ALL)
    parser.add_argument("--dataset-root", type=str, default=".")
    parser.add_argument("--annotation-file", type=str, default=None)
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--feature-bank-dir", type=str, default=None)
    parser.add_argument("--filtered-bank-dir", type=str, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-references", type=int, default=MAX_REFERENCES)
    parser.add_argument("--num-points", type=int, default=10)
    parser.add_argument("--sim-threshold", type=float, default=0.8)
    parser.add_argument("--peak-threshold", type=float, default=0.2)
    parser.add_argument("--min-peak-distance", type=int, default=10)
    parser.add_argument("--suppression-margin", type=float, default=SUPPRESSION_MARGIN)
    parser.add_argument("--no-suppression", action="store_true")
    parser.add_argument("--loose-threshold", type=float, default=0.8)
    parser.add_argument("--min-component-size", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    return run_impl(args)


def main(argv: list[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
