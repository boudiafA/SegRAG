"""
Stage 1 v3: build a raw DINOv3 feature bank, then filter it with clustered
adaptive thresholding.

Clustered adaptive filter:
- cluster scored DINOv3 features per class with KMeans
- discard clusters smaller than `min_cluster_size`
- per cluster: threshold = clip(cluster_q75 * 0.90, 0.65, 0.82)
- keep scores >= threshold inside each cluster
- cap survivors to top_k_features across clusters combined
"""

from __future__ import annotations

import argparse
import json

from segrag.stages.build_bank import (
    _parse_optional_int,
    _parse_optional_threshold,
    _resolve_stage1_paths,
    run_build,
)
from segrag.modeling.iccd import run_filter_clustered_adaptive


def _parse_cluster_count(value: str) -> int | str:
    lowered = value.strip().lower()
    if lowered == "auto":
        return "auto"
    return int(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 1 v3: build a raw feature bank, then filter it with clustered adaptive q75 thresholding."
    )
    parser.add_argument("--dataset-root", default=".")
    parser.add_argument("--train-ann-file", default=None, help="COCO-style training annotation JSON.")
    parser.add_argument("--image-dir", default=None, help="Root directory used to resolve image paths.")
    parser.add_argument("--raw-feature-bank-dir", default=None, help="Stage 1a output directory.")
    parser.add_argument("--filtered-feature-bank-dir", default=None, help="Stage 1b output directory.")
    parser.add_argument("--skip-build", action="store_true", help="Skip raw feature-bank creation and only run filtering.")
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--image-size", type=int, default=1536)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--model-name", default="dinov3_vitl16")
    parser.add_argument("--repo-path", default="./")
    parser.add_argument("--weights-path", default="./weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    parser.add_argument("--mask-coverage-threshold", type=float, default=0.90)
    parser.add_argument("--features-per-class-threshold", type=int, default=None)
    parser.add_argument("--max-images-per-class", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--scan-workers", type=int, default=8)
    parser.add_argument("--checkpoint-name", default="_build_feature_bank_resume.json")

    parser.add_argument("--selection-mode", default="top-k-images")
    parser.add_argument("--max-source-images", type=int, default=100)
    parser.add_argument(
        "--top-k-features",
        type=_parse_optional_int,
        default=10000,
        help="Maximum kept features per class after filtering. Use `none` to disable the cap.",
    )
    parser.add_argument(
        "--keep-threshold",
        type=_parse_optional_threshold,
        default=0.8,
        help="Accepted for CLI compatibility but ignored by v3 clustered adaptive filtering.",
    )
    parser.add_argument("--min-matches", type=int, default=3)
    parser.add_argument("--target-image-limit", type=int, default=100)
    parser.add_argument("--query-chunk", type=int, default=256)
    parser.add_argument("--target-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--sim-floor", type=float, default=0.0)
    parser.add_argument("--n-clusters", type=_parse_cluster_count, default="auto")
    parser.add_argument("--min-cluster-size", type=int, default=5)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    args = _resolve_stage1_paths(args)
    build_result = None
    filter_result = None

    if not args.skip_build:
        build_result = run_build(args)

    filter_result = run_filter_clustered_adaptive(
        input_dir=args.raw_feature_bank_dir,
        output_dir=args.filtered_feature_bank_dir,
        train_ann_file=args.train_ann_file,
        image_dir=args.image_dir,
        min_matches=args.min_matches,
        query_chunk=args.query_chunk,
        sim_floor=args.sim_floor,
        target_batch_size=args.target_batch_size,
        target_image_limit=args.target_image_limit,
        num_workers=args.num_workers,
        selection_mode=args.selection_mode,
        max_source_images=args.max_source_images,
        top_k_features=args.top_k_features,
        n_clusters=args.n_clusters,
        min_cluster_size=args.min_cluster_size,
        resume=args.resume,
    )

    return {
        "config": {
            "dataset_root": args.dataset_root,
            "train_ann_file": args.train_ann_file,
            "image_dir": args.image_dir,
            "raw_feature_bank_dir": args.raw_feature_bank_dir,
            "filtered_feature_bank_dir": args.filtered_feature_bank_dir,
            "resume": args.resume,
            "skip_build": args.skip_build,
            "filter_strategy": "clustered_adaptive_q75_topk",
            "adaptive_formula": "cluster_threshold = clip(cluster_q75 * 0.90, 0.65, 0.82)",
            "n_clusters": args.n_clusters,
            "min_cluster_size": args.min_cluster_size,
        },
        "build": build_result,
        "filter": filter_result,
    }


def main(argv: list[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
