from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from segrag.reproducibility.protocol import (
    SUPPORTED_SHOTS,
    build_exact_support_annotations,
    load_json,
    load_paper_config,
    protocol_fingerprint,
    resolve_config_path,
    resolve_dataset_path,
    selected_supports,
    sha256_file,
    validate_query_manifest,
    validate_support_manifest,
)
from segrag.utils.resume import save_json_atomic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce the frozen SegRAG one-shot/five-shot paper protocol."
    )
    parser.add_argument("--config", required=True, help="Dataset YAML from configs/paper/.")
    parser.add_argument("--dataset-root", required=True, help="Prepared dataset root.")
    parser.add_argument("--shot", required=True, type=int, choices=SUPPORTED_SHOTS)
    parser.add_argument("--output-root", required=True, help="New directory for all banks, caches, and predictions.")
    parser.add_argument("--dinov3-repo", required=True, help="Official DINOv3 repository checkout.")
    parser.add_argument("--dinov3-weights", required=True, help="DINOv3 ViT-L/16 checkpoint.")
    state = parser.add_mutually_exclusive_group()
    state.add_argument("--resume", action="store_true")
    state.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Stop after constructing the exact-shot bank.")
    parser.add_argument("--validate-only", action="store_true", help="Validate paths and manifests without loading models.")
    parser.add_argument("--max-images", type=int, default=None, help="Smoke-test cap; omit for reported results.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--save-mask-json", action="store_true")
    parser.add_argument(
        "--strict-score-check",
        action="store_true",
        help="Fail when the completed mIoU differs from the archived score beyond the configured tolerance.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _git_commit() -> str | None:
    try:
        root = Path(__file__).resolve().parents[3]
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _resolve_paths(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, str]:
    dataset_root = str(Path(args.dataset_root).expanduser().resolve())
    path_config = config["paths"]
    paths = {
        "dataset_root": dataset_root,
        "train_annotations": resolve_dataset_path(dataset_root, path_config["train_annotations"]),
        "query_annotations": resolve_dataset_path(dataset_root, path_config["query_annotations"]),
        "support_images": resolve_dataset_path(dataset_root, path_config["support_images"]),
        "query_images": resolve_dataset_path(dataset_root, path_config["query_images"]),
        "support_manifest": resolve_config_path(config, path_config["support_manifest"]),
        "query_manifest": resolve_config_path(config, path_config["query_manifest"]),
        "dinov3_repo": str(Path(args.dinov3_repo).expanduser().resolve()),
        "dinov3_weights": str(Path(args.dinov3_weights).expanduser().resolve()),
        "output_root": str(Path(args.output_root).expanduser().resolve()),
    }
    return paths


def _require_paths(paths: dict[str, str]) -> None:
    file_keys = (
        "train_annotations",
        "query_annotations",
        "support_manifest",
        "query_manifest",
        "dinov3_weights",
    )
    dir_keys = ("dataset_root", "support_images", "query_images", "dinov3_repo")
    missing = [paths[key] for key in file_keys if not os.path.isfile(paths[key])]
    missing.extend(paths[key] for key in dir_keys if not os.path.isdir(paths[key]))
    if missing:
        raise FileNotFoundError("Required paper-reproduction paths are missing:\n" + "\n".join(missing))
    if not os.path.isfile(os.path.join(paths["dinov3_repo"], "hubconf.py")):
        raise FileNotFoundError(f"Not a DINOv3 repository: {paths['dinov3_repo']}")


def _validate_image_files(
    paths: dict[str, str],
    support_subset: dict[str, Any],
    query_manifest: dict[str, Any],
) -> dict[str, Any]:
    missing_support = [
        str(row.get("file_name", ""))
        for row in support_subset.get("images", [])
        if not os.path.isfile(os.path.join(paths["support_images"], str(row.get("file_name", ""))))
    ]
    missing_query = [
        str(row.get("image_path", ""))
        for row in query_manifest.get("items", [])
        if not os.path.isfile(os.path.join(paths["dataset_root"], str(row.get("image_path", ""))))
    ]
    if missing_support or missing_query:
        raise FileNotFoundError(
            "Frozen protocol references missing images: "
            f"support={len(missing_support)}, query={len(missing_query)}, "
            f"first_support={missing_support[:1]}, first_query={missing_query[:1]}"
        )
    return {
        "support_images": len(support_subset.get("images", [])),
        "query_records": len(query_manifest.get("items", [])),
        "missing_support_images": 0,
        "missing_query_images": 0,
    }


def _workspace(paths: dict[str, str], dataset: str, shot: int) -> dict[str, str]:
    root = os.path.join(paths["output_root"], dataset, f"{shot}shot")
    return {
        "root": root,
        "support_annotations": os.path.join(root, "train_support_exact.json"),
        "raw_bank": os.path.join(root, "feature_bank_dinov3_vitl16_1536"),
        "scored_bank": os.path.join(root, "feature_bank_dinov3_vitl16_1536_scored_thr060"),
        "filtered_bank": os.path.join(root, "feature_bank_adaptive_q75_from_thr060"),
        "output": os.path.join(root, "evaluation_results_text_and_points_sam3_hybrid"),
        "run_metadata": os.path.join(root, "run_metadata.json"),
        "summary": os.path.join(root, "paper_run_summary.json"),
    }


def _reset_workspace(workspace: dict[str, str]) -> None:
    if os.path.isdir(workspace["root"]):
        shutil.rmtree(workspace["root"])
    os.makedirs(workspace["root"], exist_ok=True)


def _prepare_workspace(args: argparse.Namespace, workspace: dict[str, str]) -> None:
    root = Path(workspace["root"])
    if args.overwrite:
        _reset_workspace(workspace)
        return
    if root.is_dir() and any(root.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Paper workspace is not empty: {root}. Use --resume to continue the same "
            "protocol or --overwrite to rebuild it."
        )
    root.mkdir(parents=True, exist_ok=True)


def _validate_resume_metadata(metadata_path: str, metadata: dict[str, Any]) -> None:
    if not os.path.isfile(metadata_path):
        return
    previous = load_json(metadata_path)
    fields = ("protocol_version", "dataset", "shot", "fingerprints")
    mismatches = [field for field in fields if previous.get(field) != metadata.get(field)]
    if mismatches:
        raise ValueError(
            "Refusing to resume a workspace created by a different frozen protocol; "
            f"mismatched fields: {', '.join(mismatches)}. Use a new output root or --overwrite."
        )


def _score_verification(config: dict[str, Any], shot: int, evaluation: dict[str, Any]) -> dict[str, Any]:
    global_metrics = evaluation.get("text_and_point", {}).get("global", {})
    observed = global_metrics.get("mIoU")
    if observed is None:
        raise RuntimeError("The completed evaluator did not return global.mIoU.")
    expected = float(config["reported_miou"][str(shot)])
    tolerance = float(config.get("reproduction_tolerance", 5e-4))
    absolute_error = abs(float(observed) - expected)
    return {
        "observed_miou": float(observed),
        "expected_miou": expected,
        "absolute_error": absolute_error,
        "tolerance": tolerance,
        "within_tolerance": absolute_error <= tolerance,
    }


def _materialize_direct_one_shot_bank(raw_bank: str, filtered_bank: str) -> dict[str, Any]:
    if os.path.exists(filtered_bank):
        shutil.rmtree(filtered_bank)
    linked = 0
    per_class: dict[str, int] = {}
    for source in sorted(Path(raw_bank).glob("*/*.pt")):
        destination_dir = Path(filtered_bank) / source.parent.name
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / source.name
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        linked += 1
        per_class[source.parent.name] = per_class.get(source.parent.name, 0) + 1
    if linked == 0:
        raise RuntimeError("The one-shot raw feature bank is empty.")
    report = {
        "mode": "single_reference_raw_occupancy",
        "iccd_scoring": "not_defined_for_one_reference",
        "source_raw_bank": raw_bank,
        "filtered_bank": filtered_bank,
        "feature_files": linked,
        "per_class": dict(sorted(per_class.items())),
    }
    save_json_atomic(os.path.join(filtered_bank, "_single_reference_report.json"), report)
    return report


def _bank_feature_counts(bank_dir: str) -> dict[str, int]:
    return {
        class_dir.name: sum(1 for path in class_dir.glob("*.pt") if path.is_file())
        for class_dir in sorted(Path(bank_dir).iterdir())
        if class_dir.is_dir()
    }


def _build_bank(
    args: argparse.Namespace,
    config: dict[str, Any],
    paths: dict[str, str],
    workspace: dict[str, str],
    required_class_names: set[str],
) -> dict[str, Any]:
    from segrag.stages import build_bank as stage1_build
    from segrag.stages import filter_bank as stage1_filter
    from segrag.stages import score_bank as stage1_score

    bank = config["bank"]
    build_args = argparse.Namespace(
        dataset_root=workspace["root"],
        train_ann_file=workspace["support_annotations"],
        image_dir=paths["support_images"],
        raw_feature_bank_dir=workspace["raw_bank"],
        filtered_feature_bank_dir=None,
        skip_build=False,
        skip_filter=True,
        resume=args.resume,
        image_size=int(bank["image_size"]),
        patch_size=int(bank["patch_size"]),
        model_name=str(bank["model_name"]),
        repo_path=paths["dinov3_repo"],
        weights_path=paths["dinov3_weights"],
        mask_coverage_threshold=float(bank["foreground_occupancy_threshold"]),
        features_per_class_threshold=None,
        max_images_per_class=None,
        batch_size=int(bank["build_batch_size"]),
        scan_workers=int(args.num_workers),
        checkpoint_name="_build_feature_bank_resume.json",
        selection_mode="top-k-images",
        max_source_images=int(bank["max_source_images"]),
        top_k_features=None,
        keep_threshold=float(bank["score_keep_threshold"]),
        min_matches=int(bank["min_matches"]),
        min_keep_ratio=0.30,
        filter_mode="hard",
        target_image_limit=int(bank["target_image_limit"]),
        target_references=None,
        query_chunk=int(bank["query_chunk"]),
        target_batch_size=int(bank["target_batch_size"]),
        num_workers=int(args.num_workers),
        sim_floor=float(bank["similarity_floor"]),
        early_accept=False,
    )
    build_result = stage1_build.run_build(build_args)
    raw_counts = _bank_feature_counts(workspace["raw_bank"])
    missing_classes = sorted(required_class_names - {name for name, count in raw_counts.items() if count})
    if missing_classes:
        raise RuntimeError(
            "Selected supports produced no usable foreground descriptor for: "
            + ", ".join(missing_classes)
        )
    if args.shot == 1:
        return {
            "build": build_result,
            "raw_feature_files_by_class": raw_counts,
            "single_reference": _materialize_direct_one_shot_bank(
                workspace["raw_bank"], workspace["filtered_bank"]
            ),
        }

    score_args = argparse.Namespace(
        **{
            key: value
            for key, value in vars(build_args).items()
            if key not in {
                "filtered_feature_bank_dir",
                "skip_build",
                "skip_filter",
                "top_k_features",
                "min_keep_ratio",
                "filter_mode",
                "target_references",
                "early_accept",
            }
        },
        scored_feature_bank_dir=workspace["scored_bank"],
        skip_build=True,
        top_k_features=None,
    )
    score_result = stage1_score.run(score_args)
    filter_result = stage1_filter.run(
        argparse.Namespace(
            scored_feature_bank_dir=workspace["scored_bank"],
            filtered_feature_bank_dir=workspace["filtered_bank"],
            method="adaptive_q75",
            keep_threshold=None,
            top_k_features=int(bank["top_k_features"]),
            n_clusters="auto",
            min_cluster_size=5,
            resume=args.resume,
        )
    )
    return {
        "build": build_result,
        "raw_feature_files_by_class": raw_counts,
        "score": score_result,
        "filter": filter_result,
    }


def _run_query_evaluation(
    args: argparse.Namespace,
    config: dict[str, Any],
    paths: dict[str, str],
    workspace: dict[str, str],
) -> dict[str, Any]:
    from segrag.stages import cache_prompts as stage2
    from segrag.stages import evaluate_sam3 as stage4

    tsg = config["tsg"]
    stage2_args = argparse.Namespace(
        method="hybrid",
        dataset_root=workspace["root"],
        annotation_file=paths["query_annotations"],
        image_dir=paths["query_images"],
        feature_bank_dir=None,
        filtered_bank_dir=workspace["filtered_bank"],
        max_images=args.max_images,
        max_references=int(tsg["max_references"]),
        num_points=int(tsg["num_points"]),
        sim_threshold=float(tsg["similarity_threshold"]),
        peak_threshold=float(tsg["peak_threshold"]),
        min_peak_distance=int(tsg["min_peak_distance"]),
        suppression_margin=0.0,
        no_suppression=False,
        loose_threshold=float(tsg["loose_threshold"]),
        min_component_size=int(tsg["min_component_size"]),
        resume=args.resume,
    )
    stage2_result = stage2.run(stage2_args)
    stage4_result = stage4.run(
        argparse.Namespace(
            prompt_mode="text_and_point",
            feature_matching_method="hybrid",
            dataset_root=workspace["root"],
            annotation_file=paths["query_annotations"],
            image_dir=paths["query_images"],
            feature_bank_dir=None,
            filtered_bank_dir=workspace["filtered_bank"],
            prompt_cache_dir=None,
            output_dir=workspace["output"],
            max_images=args.max_images,
            max_references=int(tsg["max_references"]),
            max_points_per_class=None,
            num_points=int(tsg["num_points"]),
            sim_threshold=float(tsg["similarity_threshold"]),
            peak_threshold=float(tsg["peak_threshold"]),
            min_peak_distance=int(tsg["min_peak_distance"]),
            suppression_margin=0.0,
            no_suppression=False,
            loose_threshold=float(tsg["loose_threshold"]),
            min_component_size=int(tsg["min_component_size"]),
            hybrid_validation_threshold=float(tsg["validation_threshold"]),
            cleanup_every=50,
            num_workers=int(args.num_workers),
            prefetch_factor=16,
            save_mask_json=args.save_mask_json,
            resume=args.resume,
        )
    )
    return {"prompt_cache": stage2_result, "text_and_point": stage4_result}


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_paper_config(args.config)
    paths = _resolve_paths(args, config)
    _require_paths(paths)
    dataset = str(config["dataset"])
    workspace = _workspace(paths, dataset, args.shot)

    support_manifest = load_json(paths["support_manifest"])
    query_manifest = load_json(paths["query_manifest"])
    train_annotation = load_json(paths["train_annotations"])
    query_annotation = load_json(paths["query_annotations"])
    support_validation = validate_support_manifest(support_manifest, dataset, args.shot)
    query_validation = validate_query_manifest(query_manifest, dataset, query_annotation)
    del query_annotation

    if not args.validate_only:
        _prepare_workspace(args, workspace)
    support_rows = selected_supports(support_manifest, args.shot)
    support_subset = build_exact_support_annotations(
        train_annotation,
        support_rows,
        None if args.validate_only else workspace["support_annotations"],
    )
    del train_annotation
    image_validation = _validate_image_files(paths, support_subset, query_manifest)

    validation = {
        "dataset": dataset,
        "shot": args.shot,
        "support": support_validation,
        "query": query_validation,
        "images": image_validation,
        "paths": paths,
    }
    if args.validate_only:
        print(json.dumps(validation, indent=2))
        return validation

    if args.max_images is not None:
        print("WARNING: --max-images creates a smoke run and cannot reproduce a reported table score.")

    os.environ["DINOV3_REPO_PATH"] = paths["dinov3_repo"]
    os.environ["DINOV3_WEIGHTS_PATH"] = paths["dinov3_weights"]

    import torch

    fingerprints = protocol_fingerprint(
        {
            "config": config["_config_path"],
            "support_manifest": paths["support_manifest"],
            "query_manifest": paths["query_manifest"],
            "dinov3_weights": paths["dinov3_weights"],
        }
    )
    metadata = {
        "protocol_version": config["protocol_version"],
        "dataset": dataset,
        "shot": args.shot,
        "git_commit": _git_commit(),
        "command": sys.argv,
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
        "fingerprints": fingerprints,
        "support_images": len(support_subset.get("images", [])),
        "support_annotations": len(support_subset.get("annotations", [])),
        "support_classes": len(support_subset.get("categories", [])),
        "query_validation": query_validation,
        "image_validation": image_validation,
    }
    if args.resume:
        _validate_resume_metadata(workspace["run_metadata"], metadata)
    save_json_atomic(workspace["run_metadata"], metadata)

    bank_result = _build_bank(
        args,
        config,
        paths,
        workspace,
        {str(row["name"]) for row in support_subset.get("categories", [])},
    )
    strict_error: str | None = None
    if args.prepare_only:
        summary = {"metadata": metadata, "bank": bank_result, "status": "prepared"}
    else:
        evaluation = _run_query_evaluation(args, config, paths, workspace)
        score_check = _score_verification(config, args.shot, evaluation)
        summary = {
            "metadata": metadata,
            "bank": bank_result,
            "evaluation": evaluation,
            "score_verification": score_check,
            "status": "complete",
        }
        if not score_check["within_tolerance"]:
            message = (
                "Completed mIoU is outside the archived reproduction tolerance: "
                f"observed={score_check['observed_miou']:.8f}, "
                f"expected={score_check['expected_miou']:.8f}, "
                f"tolerance={score_check['tolerance']:.8f}."
            )
            if args.strict_score_check:
                strict_error = message
            else:
                print(f"WARNING: {message}", file=sys.stderr)
    save_json_atomic(workspace["summary"], summary)
    print(json.dumps(summary, indent=2, default=str))
    if strict_error is not None:
        raise RuntimeError(strict_error)
    return summary


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
