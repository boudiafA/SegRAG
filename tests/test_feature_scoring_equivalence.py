"""
Experiment script to compare Stage 1 feature scoring in two modes:

1. Reference one-to-one scoring
   This is the current implementation in `segrag.modeling.iccd.score_class`,
   where each source image is matched against target images one at a time.

2. Batched one-to-many scoring
   This preserves the same scoring rule but batches multiple target images in a
   single tensor op. For each source feature and each target image, it still
   takes the best-matching patch in that target image and votes good/bad based
   on whether that patch lands inside the target class mask.

The script is meant for equivalence testing, not pipeline execution.
It reports whether scores / TP / FP counts match for the chosen classes.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time

import torch
from tqdm import tqdm

from segrag.modeling.iccd import (
    _limit_target_image_ids,
    build_target_worklist,
    get_category_name,
    index_class_bank_grouped,
    load_annotations,
    load_model,
    load_source_entries,
    prepare_target_batches,
    score_class,
)


def _parse_optional_int(value: str) -> int | None:
    lowered = value.strip().lower()
    if lowered in {"none", "null"}:
        return None
    return int(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare one-to-one and one-to-many Stage 1 feature scoring."
    )
    parser.add_argument("--dataset-root", required=True, help="Dataset root. Example: /path/to/PC-59")
    parser.add_argument("--train-ann-file", default=None, help="Training annotation JSON. Defaults to dataset_root/train.json")
    parser.add_argument("--image-dir", default=None, help="Training image directory. For PC59 this is usually dataset_root/JPEGImages")
    parser.add_argument("--raw-feature-bank-dir", default=None, help="Raw feature bank directory. Defaults to dataset_root/feature_bank_dinov3_vitl16_1536")
    parser.add_argument("--class-name", default=None, help="Single class to test. If omitted, uses the first available class.")
    parser.add_argument("--max-classes", type=int, default=1, help="Maximum number of classes to test when --class-name is not given.")
    parser.add_argument("--max-source-images", type=int, default=100)
    parser.add_argument("--target-image-limit", type=int, default=100)
    parser.add_argument("--query-chunk", type=int, default=256)
    parser.add_argument("--target-batch-size", type=int, default=16, help="Batch size used while preparing target image features.")
    parser.add_argument(
        "--many-target-batch-size",
        type=_parse_optional_int,
        default=None,
        help="How many prepared target images to compare at once in one-to-many mode. Use `none` for all prepared targets at once.",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--sim-floor", type=float, default=0.0)
    parser.add_argument(
        "--selection-mode",
        choices=["top-k-images"],
        default="top-k-images",
        help="Equivalence test is scoped to the maintained top-k-images path.",
    )
    parser.add_argument(
        "--keep-threshold",
        type=float,
        default=0.6,
        help="Optional threshold used only for an extra filtered-set comparison in the report.",
    )
    parser.add_argument(
        "--skip-one-to-one",
        action="store_true",
        help="Skip the reference scorer and run only one-to-many timing. Comparison fields will be omitted.",
    )
    parser.add_argument("--output-json", default=None, help="Optional path to save the comparison report.")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    args.train_ann_file = args.train_ann_file or os.path.join(args.dataset_root, "train.json")
    if args.image_dir is None:
        candidates = [
            os.path.join(args.dataset_root, "JPEGImages"),
            os.path.join(args.dataset_root, "train", "images"),
            os.path.join(args.dataset_root, "images"),
            args.dataset_root,
        ]
        args.image_dir = next((path for path in candidates if os.path.isdir(path)), args.dataset_root)
    args.raw_feature_bank_dir = args.raw_feature_bank_dir or os.path.join(
        args.dataset_root,
        "feature_bank_dinov3_vitl16_1536",
    )
    return args


def _flatten_target_batches(prepared_target_batches: list[list[dict]]) -> list[dict]:
    flat_targets: list[dict] = []
    for batch in prepared_target_batches:
        flat_targets.extend(batch)
    return flat_targets


def score_class_one_to_many(
    grouped_bank: dict[int, list[tuple[str, str]]],
    target_worklist: list[tuple[int, str, dict, list[dict]]],
    model,
    device: str,
    query_chunk: int,
    sim_floor: float,
    target_batch_size: int,
    many_target_batch_size: int | None,
    num_workers: int,
    selection_mode: str,
    max_source_images: int | None,
    class_name: str | None = None,
):
    file_refs: list[tuple[str, str]] = []
    scores_by_file: dict[str, torch.Tensor] = {}
    good_by_file: dict[str, torch.Tensor] = {}
    bad_by_file: dict[str, torch.Tensor] = {}
    for image_id in sorted(grouped_bank):
        file_refs.extend(grouped_bank[image_id])

    with torch.inference_mode():
        prepared_target_batches = prepare_target_batches(
            model=model,
            device=device,
            target_worklist=target_worklist,
            target_batch_size=target_batch_size,
            num_workers=num_workers,
            class_name=class_name,
        )
    flat_targets = _flatten_target_batches(prepared_target_batches)
    target_image_ids = [target["image_id"] for target in flat_targets]
    if flat_targets:
        grids_dev_all = torch.stack([target["grid"] for target in flat_targets], dim=0).transpose(1, 2).contiguous().to(device)
        masks_dev_all = torch.stack([target["mask"] for target in flat_targets], dim=0).to(device=device, dtype=torch.bool)
    else:
        grids_dev_all = None
        masks_dev_all = None
    source_ids = sorted(grouped_bank)
    if selection_mode == "top-k-images" and max_source_images is not None:
        source_ids = source_ids[:max_source_images]

    effective_many_batch = many_target_batch_size
    if effective_many_batch is None:
        effective_many_batch = len(flat_targets) if flat_targets else 1
    effective_many_batch = max(1, int(effective_many_batch))
    active_chunk_tensors_by_source: dict[int, list[torch.Tensor]] = {}
    for source_image_id in source_ids:
        active_indices = [idx for idx, image_id in enumerate(target_image_ids) if image_id != source_image_id]
        chunk_tensors = []
        for start_target in range(0, len(active_indices), effective_many_batch):
            chunk_indices = active_indices[start_target: start_target + effective_many_batch]
            if chunk_indices:
                chunk_tensors.append(torch.as_tensor(chunk_indices, device=device, dtype=torch.long))
        active_chunk_tensors_by_source[source_image_id] = chunk_tensors

    source_pbar = tqdm(
        source_ids,
        desc=f"  Source-many {class_name}" if class_name else "  Source-many images",
        leave=False,
    )
    for source_image_id in source_pbar:
        source_entries = load_source_entries(grouped_bank[source_image_id])
        if not source_entries:
            continue

        source_feats = torch.cat([feats for _, feats in source_entries], dim=0)
        source_feats_dev = source_feats.to(device)
        good_dev = torch.zeros(source_feats.shape[0], dtype=torch.int32, device=device)
        bad_dev = torch.zeros(source_feats.shape[0], dtype=torch.int32, device=device)

        for chunk_index_tensor in active_chunk_tensors_by_source[source_image_id]:
            grids_dev = grids_dev_all.index_select(0, chunk_index_tensor)
            masks_dev = masks_dev_all.index_select(0, chunk_index_tensor)

            for start in range(0, source_feats.shape[0], query_chunk):
                end = min(start + query_chunk, source_feats.shape[0])
                query_dev = source_feats_dev[start:end]
                sims = torch.matmul(query_dev.unsqueeze(0), grids_dev).permute(1, 0, 2)
                best_sims, best_idx = sims.max(dim=2)
                valid = best_sims >= sim_floor
                landed_inside = masks_dev.gather(1, best_idx.transpose(0, 1)).transpose(0, 1)
                good_dev[start:end] += (valid & landed_inside).to(torch.int32).sum(dim=1)
                bad_dev[start:end] += (valid & ~landed_inside).to(torch.int32).sum(dim=1)
                del query_dev, sims, best_sims, best_idx, valid, landed_inside

            del chunk_index_tensor, grids_dev, masks_dev

        good = good_dev.cpu()
        bad = bad_dev.cpu()
        total = good + bad
        scores = torch.full((source_feats.shape[0],), -1.0, dtype=torch.float32)
        any_match_mask = total >= 1
        scores[any_match_mask] = good[any_match_mask].float() / total[any_match_mask].float()

        offset = 0
        for fname, feats in source_entries:
            n = feats.shape[0]
            scores_by_file[fname] = scores[offset: offset + n]
            good_by_file[fname] = good[offset: offset + n]
            bad_by_file[fname] = bad[offset: offset + n]
            offset += n

        del source_feats_dev, source_feats, good_dev, bad_dev, good, bad, scores, source_entries
        if (source_pbar.n + 1) % 100 == 0:
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    source_pbar.close()
    del active_chunk_tensors_by_source
    del grids_dev_all, masks_dev_all
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return {
        "file_refs": file_refs,
        "scores_by_file": scores_by_file,
        "good_by_file": good_by_file,
        "bad_by_file": bad_by_file,
        "meta": {
            "prepared_target_images": len(flat_targets),
            "effective_many_target_batch_size": effective_many_batch,
            "source_images_scored": len(source_ids),
        },
    }


def _compare_class_results(
    class_name: str,
    one_to_one: dict,
    one_to_many: dict,
    keep_threshold: float | None,
) -> dict:
    ref_files = sorted(fname for fname, _ in one_to_one["file_refs"])
    many_files = sorted(fname for fname, _ in one_to_many["file_refs"])
    file_set_match = ref_files == many_files

    score_exact = True
    good_exact = True
    bad_exact = True
    keep_mask_exact = True
    max_score_abs_diff = 0.0
    mismatched_score_files: list[str] = []
    mismatched_good_files: list[str] = []
    mismatched_bad_files: list[str] = []
    mismatched_keep_mask_files: list[str] = []
    total_vectors = 0

    for fname in sorted(set(ref_files) & set(many_files)):
        ref_scores = one_to_one["scores_by_file"][fname]
        many_scores = one_to_many["scores_by_file"][fname]
        ref_good = one_to_one["good_by_file"][fname]
        many_good = one_to_many["good_by_file"][fname]
        ref_bad = one_to_one["bad_by_file"][fname]
        many_bad = one_to_many["bad_by_file"][fname]

        total_vectors += int(ref_scores.numel())

        if not torch.equal(ref_good, many_good):
            good_exact = False
            mismatched_good_files.append(fname)
        if not torch.equal(ref_bad, many_bad):
            bad_exact = False
            mismatched_bad_files.append(fname)
        if not torch.allclose(ref_scores, many_scores, atol=1e-6, rtol=1e-6):
            score_exact = False
            mismatched_score_files.append(fname)
        if ref_scores.numel() > 0:
            max_score_abs_diff = max(
                max_score_abs_diff,
                float((ref_scores - many_scores).abs().max().item()),
            )

        if keep_threshold is not None:
            ref_keep = ref_scores >= keep_threshold
            many_keep = many_scores >= keep_threshold
            if not torch.equal(ref_keep, many_keep):
                keep_mask_exact = False
                mismatched_keep_mask_files.append(fname)

    return {
        "category": class_name,
        "file_set_match": file_set_match,
        "score_exact": score_exact,
        "good_exact": good_exact,
        "bad_exact": bad_exact,
        "keep_mask_exact": keep_mask_exact if keep_threshold is not None else None,
        "max_score_abs_diff": max_score_abs_diff,
        "total_vectors_compared": total_vectors,
        "mismatched_score_files": mismatched_score_files,
        "mismatched_good_files": mismatched_good_files,
        "mismatched_bad_files": mismatched_bad_files,
        "mismatched_keep_mask_files": mismatched_keep_mask_files if keep_threshold is not None else None,
        "one_to_many_meta": one_to_many["meta"],
    }


def _select_class_names(raw_feature_bank_dir: str, requested_class: str | None, max_classes: int) -> list[str]:
    class_dirs = sorted(
        entry
        for entry in os.listdir(raw_feature_bank_dir)
        if os.path.isdir(os.path.join(raw_feature_bank_dir, entry)) and not entry.startswith("_")
    )
    if requested_class is not None:
        if requested_class not in class_dirs:
            raise FileNotFoundError(
                f"Class '{requested_class}' was not found in the raw bank at {raw_feature_bank_dir}."
            )
        return [requested_class]
    if not class_dirs:
        raise FileNotFoundError(f"No class directories found under {raw_feature_bank_dir}.")
    return class_dirs[: max(1, max_classes)]


def run(args: argparse.Namespace) -> dict:
    args = _resolve_paths(args)
    if not os.path.isdir(args.raw_feature_bank_dir):
        raise FileNotFoundError(f"Raw feature bank directory not found: {args.raw_feature_bank_dir}")
    if not os.path.isfile(args.train_ann_file):
        raise FileNotFoundError(f"Training annotation JSON not found: {args.train_ann_file}")
    if not os.path.isdir(args.image_dir):
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")

    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    images, anns_by_image, categories, image_ids_by_cat = load_annotations(args.train_ann_file)
    class_names = _select_class_names(args.raw_feature_bank_dir, args.class_name, args.max_classes)

    class_reports = []
    total_one_to_one_seconds = 0.0
    total_one_to_many_seconds = 0.0
    for class_name in class_names:
        cat_id = next(
            (cid for cid, cat in categories.items() if get_category_name(categories, cid) == class_name),
            None,
        )
        if cat_id is None:
            continue

        grouped_bank = index_class_bank_grouped(os.path.join(args.raw_feature_bank_dir, class_name))
        limited_target_ids = _limit_target_image_ids(image_ids_by_cat.get(cat_id, []), args.target_image_limit)
        target_worklist = build_target_worklist(
            image_dir=args.image_dir,
            images=images,
            anns_by_image=anns_by_image,
            cat_id=cat_id,
            image_ids=limited_target_ids,
        )
        if len(target_worklist) < 2:
            class_reports.append(
                {
                    "category": class_name,
                    "skipped": True,
                    "reason": "fewer than 2 target images available",
                }
            )
            continue

        one_to_one = None
        one_to_one_seconds = None
        if not args.skip_one_to_one:
            t_one_to_one = time.time()
            one_to_one_refs, one_to_one_scores, one_to_one_good, one_to_one_bad = score_class(
                grouped_bank=grouped_bank,
                target_worklist=target_worklist,
                model=model,
                device=device,
                query_chunk=args.query_chunk,
                sim_floor=args.sim_floor,
                target_batch_size=args.target_batch_size,
                num_workers=args.num_workers,
                keep_threshold=None,
                min_matches=1,
                target_references=None,
                early_accept=False,
                selection_mode=args.selection_mode,
                max_source_images=args.max_source_images,
                class_name=class_name,
            )
            one_to_one_seconds = time.time() - t_one_to_one
            total_one_to_one_seconds += one_to_one_seconds
            one_to_one = {
                "file_refs": one_to_one_refs,
                "scores_by_file": one_to_one_scores,
                "good_by_file": one_to_one_good,
                "bad_by_file": one_to_one_bad,
            }

        t_one_to_many = time.time()
        one_to_many = score_class_one_to_many(
            grouped_bank=grouped_bank,
            target_worklist=target_worklist,
            model=model,
            device=device,
            query_chunk=args.query_chunk,
            sim_floor=args.sim_floor,
            target_batch_size=args.target_batch_size,
            many_target_batch_size=args.many_target_batch_size,
            num_workers=args.num_workers,
            selection_mode=args.selection_mode,
            max_source_images=args.max_source_images,
            class_name=class_name,
        )
        one_to_many_seconds = time.time() - t_one_to_many

        total_one_to_many_seconds += one_to_many_seconds

        if one_to_one is None:
            class_report = {
                "category": class_name,
                "comparison_skipped": True,
                "one_to_many_meta": one_to_many["meta"],
            }
        else:
            class_report = _compare_class_results(
                class_name=class_name,
                one_to_one=one_to_one,
                one_to_many=one_to_many,
                keep_threshold=args.keep_threshold,
            )
        class_report["timing"] = {
            "one_to_one_seconds": round(one_to_one_seconds, 3) if one_to_one_seconds is not None else None,
            "one_to_many_seconds": round(one_to_many_seconds, 3),
            "speedup_one_to_many_vs_one_to_one": (
                round(one_to_one_seconds / one_to_many_seconds, 4)
                if one_to_one_seconds is not None and one_to_many_seconds > 0
                else None
            ),
        }
        class_reports.append(class_report)

    compared = [row for row in class_reports if not row.get("skipped")]
    comparable_rows = [row for row in compared if not row.get("comparison_skipped")]
    all_equivalent = (
        all(
            row["file_set_match"]
            and row["score_exact"]
            and row["good_exact"]
            and row["bad_exact"]
            and (row["keep_mask_exact"] in (None, True))
            for row in comparable_rows
        )
        if comparable_rows
        else None
    )

    result = {
        "config": {
            "dataset_root": args.dataset_root,
            "train_ann_file": args.train_ann_file,
            "image_dir": args.image_dir,
            "raw_feature_bank_dir": args.raw_feature_bank_dir,
            "class_name": args.class_name,
            "max_classes": args.max_classes,
            "max_source_images": args.max_source_images,
            "target_image_limit": args.target_image_limit,
            "query_chunk": args.query_chunk,
            "target_batch_size": args.target_batch_size,
            "many_target_batch_size": args.many_target_batch_size,
            "num_workers": args.num_workers,
            "sim_floor": args.sim_floor,
            "selection_mode": args.selection_mode,
            "keep_threshold": args.keep_threshold,
            "skip_one_to_one": args.skip_one_to_one,
            "device": device,
        },
        "summary": {
            "classes_requested": len(class_names),
            "classes_compared": len(compared),
            "classes_with_comparison": len(comparable_rows),
            "all_equivalent": all_equivalent,
            "one_to_one_total_seconds": round(total_one_to_one_seconds, 3) if not args.skip_one_to_one else None,
            "one_to_many_total_seconds": round(total_one_to_many_seconds, 3),
            "speedup_one_to_many_vs_one_to_one": (
                round(total_one_to_one_seconds / total_one_to_many_seconds, 4)
                if (not args.skip_one_to_one) and total_one_to_many_seconds > 0
                else None
            ),
            "elapsed_seconds": round(time.time() - t_start, 1),
        },
        "per_class": class_reports,
    }

    if args.output_json:
        output_dir = os.path.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, "w") as handle:
            json.dump(result, handle, indent=2)

    return result


def main(argv: list[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
