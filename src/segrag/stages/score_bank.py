"""
Stage 1 scoring: build a raw DINOv3 feature bank if needed, then score features
once and save a reusable scored bank.

The scored bank stores:
- feature tensors (`.pt`)
- per-feature score sidecars (`.scores.npy`)
- per-feature tp / fp sidecars (`.tp.npy`, `.fp.npy`)

Default behavior keeps all scored features with score >= 0.6 and applies no
top-k cap so later filtering experiments can reuse the scored bank.
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
from segrag.modeling.iccd import run_score_bank


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 1 scoring: build a raw feature bank, then score and save a reusable scored bank."
    )
    parser.add_argument("--dataset-root", default=".")
    parser.add_argument("--train-ann-file", default=None, help="COCO-style training annotation JSON.")
    parser.add_argument("--image-dir", default=None, help="Root directory used to resolve image paths.")
    parser.add_argument("--raw-feature-bank-dir", default=None, help="Stage 1a output directory.")
    parser.add_argument("--scored-feature-bank-dir", default=None, help="Stage 1 scoring output directory.")
    parser.add_argument("--skip-build", action="store_true", help="Skip raw feature-bank creation and only run scoring.")
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
        default=None,
        help="Maximum scored features kept per class. Default is `none` for no cap.",
    )
    parser.add_argument(
        "--keep-threshold",
        type=_parse_optional_threshold,
        default=0.6,
        help="Minimum score to keep in the scored bank. Default is 0.6.",
    )
    parser.add_argument("--min-matches", type=int, default=3)
    parser.add_argument("--target-image-limit", type=int, default=100)
    parser.add_argument("--query-chunk", type=int, default=256)
    parser.add_argument("--target-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--sim-floor", type=float, default=0.0)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    args = _resolve_stage1_paths(args)
    if args.scored_feature_bank_dir is None:
        args.scored_feature_bank_dir = f"{args.raw_feature_bank_dir}_scored"

    build_result = None
    score_result = None

    if not args.skip_build:
        build_result = run_build(args)

    score_result = run_score_bank(
        input_dir=args.raw_feature_bank_dir,
        output_dir=args.scored_feature_bank_dir,
        train_ann_file=args.train_ann_file,
        image_dir=args.image_dir,
        keep_threshold=args.keep_threshold,
        min_matches=args.min_matches,
        query_chunk=args.query_chunk,
        sim_floor=args.sim_floor,
        target_batch_size=args.target_batch_size,
        target_image_limit=args.target_image_limit,
        num_workers=args.num_workers,
        selection_mode=args.selection_mode,
        max_source_images=args.max_source_images,
        top_k_features=args.top_k_features,
        resume=args.resume,
    )

    return {
        "config": {
            "dataset_root": args.dataset_root,
            "train_ann_file": args.train_ann_file,
            "image_dir": args.image_dir,
            "raw_feature_bank_dir": args.raw_feature_bank_dir,
            "scored_feature_bank_dir": args.scored_feature_bank_dir,
            "resume": args.resume,
            "skip_build": args.skip_build,
        },
        "build": build_result,
        "score": score_result,
    }


def main(argv: list[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
