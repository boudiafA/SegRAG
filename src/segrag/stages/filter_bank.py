"""
Stage 1 filtering from a reusable scored bank.

Supported methods:
- `fixed`
- `adaptive_q75`
- `clustered_adaptive_q75`
"""

from __future__ import annotations

import argparse
import json

from segrag.stages.build_bank import _parse_optional_int, _parse_optional_threshold
from segrag.modeling.iccd import run_filter_from_scored_bank


METHODS = ("fixed", "adaptive_q75", "clustered_adaptive_q75")


def _parse_cluster_count(value: str) -> int | str:
    lowered = value.strip().lower()
    if lowered == "auto":
        return "auto"
    return int(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 1 filtering from a reusable scored bank."
    )
    parser.add_argument("--scored-feature-bank-dir", required=True, help="Input scored bank directory.")
    parser.add_argument("--filtered-feature-bank-dir", required=True, help="Output filtered bank directory.")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument(
        "--keep-threshold",
        type=_parse_optional_threshold,
        default=0.8,
        help="Used only by `fixed`. Use `none` to keep everything in the scored bank.",
    )
    parser.add_argument(
        "--top-k-features",
        type=_parse_optional_int,
        default=10000,
        help="Maximum kept features per class. Use `none` to disable the cap.",
    )
    parser.add_argument("--n-clusters", type=_parse_cluster_count, default="auto")
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    result = run_filter_from_scored_bank(
        input_dir=args.scored_feature_bank_dir,
        output_dir=args.filtered_feature_bank_dir,
        method=args.method,
        keep_threshold=args.keep_threshold,
        top_k_features=args.top_k_features,
        n_clusters=args.n_clusters,
        min_cluster_size=args.min_cluster_size,
        resume=args.resume,
    )
    return {
        "config": {
            "scored_feature_bank_dir": args.scored_feature_bank_dir,
            "filtered_feature_bank_dir": args.filtered_feature_bank_dir,
            "method": args.method,
            "keep_threshold": args.keep_threshold,
            "top_k_features": args.top_k_features,
            "n_clusters": args.n_clusters,
            "min_cluster_size": args.min_cluster_size,
            "resume": args.resume,
        },
        "filter": result,
    }


def main(argv: list[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
