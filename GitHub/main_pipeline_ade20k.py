"""
ADE20K adapters for the GitHub-facing main pipeline.

Supported local dataset roots:

1. `1aurent/ADE20K` HuggingFace parquet download
   - `data/train-*.parquet`
   - `data/validation-*.parquet`

2. `zhoubolei/scene_parse_150` / ADE20K-150 cache
   - local HuggingFace cache under the dataset root
   - or extracted `ADEChallengeData2016/` images and annotations

For either format this script exports:
- `train.json`
- `val.json`

Then it runs the regular segmentation pipeline while keeping feature banks,
prompt caches, and evaluation outputs under the same `dataset_root`.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
from dataclasses import dataclass
from glob import glob

import numpy as np
from PIL import Image
from tqdm import tqdm

SEGMENTATION_METHODS = ("text-only", "point-only", "text-and-point", "all")
DATASET_FORMATS = ("auto", "parquet", "ade20k_150")
EXPORT_SUMMARY_NAME = "_ade20k_export_summary.json"

ADE20K_150_CLASSES = [
    "background",
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed",
    "windowpane", "grass", "cabinet", "sidewalk", "person", "earth",
    "door", "table", "mountain", "plant", "curtain", "chair", "car",
    "water", "painting", "sofa", "shelf", "house", "sea", "mirror",
    "rug", "field", "armchair", "seat", "fence", "desk", "rock",
    "wardrobe", "lamp", "bathtub", "railing", "cushion", "base",
    "box", "column", "signboard", "chest of drawers", "counter", "sand",
    "sink", "skyscraper", "fireplace", "refrigerator", "grandstand",
    "path", "stairs", "runway", "case", "pool table", "pillow",
    "screen door", "stairway", "river", "bridge", "bookcase", "blind",
    "coffee table", "toilet", "flower", "book", "hill", "bench",
    "countertop", "stove", "palm", "kitchen island", "computer",
    "swivel chair", "boat", "bar", "arcade machine", "hovel", "bus",
    "towel", "light", "truck", "tower", "chandelier", "awning",
    "streetlight", "booth", "television receiver", "airplane", "dirt track",
    "apparel", "pole", "land", "bannister", "escalator", "ottoman",
    "bottle", "buffet", "poster", "stage", "van", "ship", "fountain",
    "conveyer belt", "canopy", "washer", "plaything", "swimming pool",
    "stool", "barrel", "basket", "waterfall", "tent", "bag", "minibike",
    "cradle", "oven", "ball", "food", "step", "tank", "trade name",
    "microwave", "pot", "animal", "bicycle", "lake", "dishwasher",
    "screen", "blanket", "sculpture", "hood", "sconce", "vase",
    "traffic light", "tray", "ashcan", "fan", "pier", "crt screen",
    "plate", "monitor", "bulletin board", "shower", "radiator", "glass",
    "clock", "flag",
]


@dataclass(frozen=True)
class ExportPaths:
    dataset_root: str

    @property
    def train_json(self) -> str:
        return os.path.join(self.dataset_root, "train.json")

    @property
    def val_json(self) -> str:
        return os.path.join(self.dataset_root, "val.json")

    @property
    def summary_json(self) -> str:
        return os.path.join(self.dataset_root, EXPORT_SUMMARY_NAME)


@dataclass(frozen=True)
class ExportLayout:
    dataset_root: str
    dataset_format: str
    train_ann_file: str
    val_ann_file: str
    train_image_dir: str
    val_image_dir: str


def _parse_optional_threshold(value: str) -> float | None:
    lowered = value.strip().lower()
    if lowered in {"none", "null"}:
        return None
    return float(value)


def _parse_optional_int(value: str) -> int | None:
    lowered = value.strip().lower()
    if lowered in {"none", "null"}:
        return None
    return int(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export ADE20K into a COCO-style view, then run the segmentation pipeline."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Local ADE20K dataset root.",
    )
    parser.add_argument(
        "--dataset-format",
        choices=DATASET_FORMATS,
        default="auto",
        help="Dataset format to use. `auto` detects parquet vs ADE20K-150 cache layout.",
    )
    parser.add_argument(
        "--segmentation-method",
        choices=SEGMENTATION_METHODS,
        required=True,
        help="Segmentation mode to run after export: text-only, point-only, text-and-point, or all.",
    )
    parser.add_argument(
        "--feature-matching-method",
        default="hybrid",
        choices=("absolute_similarity", "relative_similarity", "hybrid"),
        help="Feature matching method used by point-based segmentation modes.",
    )
    parser.add_argument(
        "--reference-images-per-class",
        type=int,
        default=30,
        help="Maximum number of source images per class used while building the raw bank.",
    )
    parser.add_argument(
        "--raw-bank-batch-size",
        type=int,
        default=4,
        help="Batch size used during Stage 1 raw-bank DINOv3 feature extraction.",
    )
    parser.add_argument(
        "--feature-top-k",
        type=_parse_optional_int,
        default=10000,
        help="Top-k features kept after the default adaptive_q75 Stage 1 filtering pass. Use `none` to disable the cap.",
    )
    parser.add_argument(
        "--feature-accuracy-threshold",
        type=_parse_optional_threshold,
        default=0.8,
        help="Similarity threshold used by Stage 2 prompt generation and Stage 4 point validation. It does not change the default adaptive_q75 Stage 1 filter.",
    )
    parser.add_argument(
        "--max-references",
        type=int,
        default=10000,
        help="Maximum number of cached feature-bank references used by Stage 2 and Stage 4.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap on validation images for Stage 2 and Stage 4.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only export the local COCO-style ADE20K view and skip pipeline execution.",
    )
    parser.add_argument(
        "--force-rebuild-export",
        action="store_true",
        help="Rebuild train.json and val.json even if they already exist.",
    )
    parser.add_argument(
        "--save-mask-json",
        action="store_true",
        help="Export predicted masks JSON from Stage 4.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save a combined export + pipeline summary JSON.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _mask_utils():
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "GitHub.main_pipeline_ade20k requires pycocotools. "
            "Install it with `pip install pycocotools`."
        ) from exc
    return mask_utils


def _parquet_module():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Parquet ADE20K export requires pyarrow. Install it with `pip install pyarrow`."
        ) from exc
    return pq


def _sanitize_name(name: str) -> str:
    name = re.sub(r"\s+", " ", str(name).strip())
    name = name.replace("/", " or ").replace("\\", " ")
    return name or "unknown"


def _raw_feature_bank_dir(dataset_root: str) -> str:
    return os.path.join(dataset_root, "feature_bank_dinov3_vitl16_1536")


def _scored_feature_bank_dir(dataset_root: str) -> str:
    return os.path.join(dataset_root, "feature_bank_dinov3_vitl16_1536_scored_thr060")


def _filtered_feature_bank_dir(dataset_root: str) -> str:
    return os.path.join(dataset_root, "feature_bank_adaptive_q75_from_thr060")


def _detect_dataset_format(dataset_root: str, forced: str) -> str:
    if forced != "auto":
        return forced
    if glob(os.path.join(dataset_root, "data", "train-*.parquet")):
        return "parquet"
    if _find_ade20k150_root(dataset_root) is not None:
        return "ade20k_150"
    raise FileNotFoundError(
        f"Could not detect a supported ADE20K layout under {dataset_root}. "
        "Expected parquet shards or an extracted ADEChallengeData2016 tree."
    )


def _can_reuse_export(paths: ExportPaths) -> bool:
    return (
        os.path.isfile(paths.summary_json)
        and os.path.isfile(paths.train_json)
        and os.path.isfile(paths.val_json)
    )


def _write_summary(paths: ExportPaths, summary: dict) -> None:
    with open(paths.summary_json, "w") as handle:
        json.dump(summary, handle, indent=2, default=str)


def _load_summary(paths: ExportPaths) -> dict:
    with open(paths.summary_json, "r") as handle:
        return json.load(handle)


def _find_ade20k150_root(dataset_root: str) -> str | None:
    candidates = [
        os.path.join(dataset_root, "ADEChallengeData2016"),
    ]
    candidates.extend(
        glob(os.path.join(dataset_root, "downloads", "extracted", "*", "ADEChallengeData2016"))
    )
    for candidate in candidates:
        if not os.path.isdir(candidate):
            continue
        required = [
            os.path.join(candidate, "images", "training"),
            os.path.join(candidate, "images", "validation"),
            os.path.join(candidate, "annotations", "training"),
            os.path.join(candidate, "annotations", "validation"),
        ]
        if all(os.path.isdir(path) for path in required):
            return candidate
    return None


def _ade20k150_categories() -> list[dict]:
    return [
        {"id": class_id, "name": _sanitize_name(ADE20K_150_CLASSES[class_id])}
        for class_id in range(1, len(ADE20K_150_CLASSES))
    ]


def _list_image_pairs(image_dir: str, annotation_dir: str) -> tuple[list[tuple[str, str, str]], dict]:
    valid_exts = (".jpg", ".jpeg", ".png")
    image_by_stem: dict[str, str] = {}
    for fname in os.listdir(image_dir):
        lower = fname.lower()
        if lower.endswith(valid_exts):
            image_by_stem[os.path.splitext(fname)[0]] = fname

    pairs: list[tuple[str, str, str]] = []
    missing_images = 0
    for ann_name in sorted(os.listdir(annotation_dir)):
        if not ann_name.lower().endswith(".png"):
            continue
        stem = os.path.splitext(ann_name)[0]
        image_name = image_by_stem.get(stem)
        if image_name is None:
            missing_images += 1
            continue
        pairs.append(
            (
                os.path.join(image_dir, image_name),
                os.path.join(annotation_dir, ann_name),
                image_name,
            )
        )
    return pairs, {"missing_images_for_masks": missing_images}


def _encode_binary_mask(binary_mask: np.ndarray) -> tuple[dict, float, list[float]] | None:
    mask_utils = _mask_utils()
    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)
    if int(binary_mask.sum()) == 0:
        return None
    encoded = mask_utils.encode(np.asfortranarray(binary_mask))
    area = float(mask_utils.area(encoded))
    if area <= 0.0:
        return None
    bbox = [float(v) for v in mask_utils.toBbox(encoded).tolist()]
    segmentation = {
        "size": [int(encoded["size"][0]), int(encoded["size"][1])],
        "counts": encoded["counts"].decode("utf-8"),
    }
    return segmentation, area, bbox


def _write_json_from_rows(output_json: str, images: list[dict], annotations: list[dict], categories: list[dict]) -> None:
    with open(output_json, "w") as handle:
        json.dump(
            {
                "images": images,
                "annotations": annotations,
                "categories": categories,
            },
            handle,
            separators=(",", ":"),
        )


def _export_ade20k150_split(
    split_name: str,
    image_dir: str,
    annotation_dir: str,
    output_json: str,
    categories: list[dict],
) -> dict:
    pairs, pair_stats = _list_image_pairs(image_dir, annotation_dir)
    print(
        f"Preparing ADE20K-150 {split_name} split from {image_dir} "
        f"with {len(pairs)} image/mask pairs."
    )

    images: list[dict] = []
    annotations: list[dict] = []
    next_ann_id = 1
    images_without_foreground = 0
    ignored_background_only = 0

    iterator = tqdm(pairs, desc=f"ADE20K-150 export [{split_name}]", total=len(pairs))
    for image_id, (image_path, mask_path, image_name) in enumerate(iterator, start=1):
        with Image.open(image_path) as image:
            width, height = image.size
        with Image.open(mask_path) as mask_image:
            mask = np.array(mask_image, dtype=np.uint8)

        images.append(
            {
                "id": image_id,
                "width": int(width),
                "height": int(height),
                "file_name": image_name,
                "not_exhaustive_category_ids": [],
            }
        )

        class_ids = [int(class_id) for class_id in np.unique(mask) if 1 <= int(class_id) <= 150]
        if not class_ids:
            images_without_foreground += 1
            if np.all(mask == 0):
                ignored_background_only += 1
            continue

        for class_id in class_ids:
            encoded = _encode_binary_mask(mask == class_id)
            if encoded is None:
                continue
            segmentation, area, bbox = encoded
            annotations.append(
                {
                    "id": next_ann_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "segmentation": segmentation,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                }
            )
            next_ann_id += 1

    _write_json_from_rows(output_json, images, annotations, categories)
    return {
        "output_json": output_json,
        "image_dir": image_dir,
        "annotation_dir": annotation_dir,
        "images": len(images),
        "annotations": len(annotations),
        "images_without_foreground": images_without_foreground,
        "background_only_images": ignored_background_only,
        **pair_stats,
    }


def export_ade20k150(dataset_root: str, force_rebuild_export: bool = False) -> ExportLayout:
    paths = ExportPaths(dataset_root=dataset_root)
    source_root = _find_ade20k150_root(dataset_root)
    if source_root is None:
        raise FileNotFoundError(
            f"Could not find extracted ADEChallengeData2016 under {dataset_root}."
        )

    train_image_dir = os.path.join(source_root, "images", "training")
    val_image_dir = os.path.join(source_root, "images", "validation")
    train_annotation_dir = os.path.join(source_root, "annotations", "training")
    val_annotation_dir = os.path.join(source_root, "annotations", "validation")

    if not force_rebuild_export and _can_reuse_export(paths):
        summary = _load_summary(paths)
        summary["reused_existing_export"] = True
        _write_summary(paths, summary)
        print(f"Reusing existing ADE20K-150 export at {dataset_root}")
        return ExportLayout(
            dataset_root=dataset_root,
            dataset_format="ade20k_150",
            train_ann_file=paths.train_json,
            val_ann_file=paths.val_json,
            train_image_dir=train_image_dir,
            val_image_dir=val_image_dir,
        )

    categories = _ade20k150_categories()
    print(f"Exporting ADE20K-150 with {len(categories)} foreground classes. Background class 0 is ignored.")
    train_summary = _export_ade20k150_split(
        split_name="train",
        image_dir=train_image_dir,
        annotation_dir=train_annotation_dir,
        output_json=paths.train_json,
        categories=categories,
    )
    val_summary = _export_ade20k150_split(
        split_name="validation",
        image_dir=val_image_dir,
        annotation_dir=val_annotation_dir,
        output_json=paths.val_json,
        categories=categories,
    )

    summary = {
        "dataset_root": dataset_root,
        "dataset_format": "ade20k_150",
        "reused_existing_export": False,
        "source_root": source_root,
        "categories": {"count": len(categories)},
        "export": {
            "train": train_summary,
            "validation": val_summary,
        },
    }
    _write_summary(paths, summary)
    print(f"ADE20K-150 export complete at {dataset_root}")
    return ExportLayout(
        dataset_root=dataset_root,
        dataset_format="ade20k_150",
        train_ann_file=paths.train_json,
        val_ann_file=paths.val_json,
        train_image_dir=train_image_dir,
        val_image_dir=val_image_dir,
    )


def _extract_raw_bytes(cell) -> bytes | None:
    if isinstance(cell, dict):
        for key in ("bytes", "data"):
            raw = cell.get(key)
            if isinstance(raw, (bytes, bytearray)):
                return bytes(raw)
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    return None


def _decode_image_cell(cell) -> Image.Image:
    if isinstance(cell, Image.Image):
        return cell.convert("RGB")
    raw = _extract_raw_bytes(cell)
    if raw is None:
        raise ValueError(f"Unsupported ADE20K image cell type: {type(cell)!r}")
    with Image.open(io.BytesIO(raw)) as image:
        return image.convert("RGB")


def _norm_text(value) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def _object_category_name(obj: dict) -> str:
    for key in ("name", "raw_name"):
        value = _norm_text(obj.get(key))
        if value:
            return value
    return ""


def _discover_parquet_files(dataset_root: str) -> dict[str, list[str]]:
    data_dir = os.path.join(dataset_root, "data")
    train_files = sorted(glob(os.path.join(data_dir, "train-*.parquet")))
    val_files = sorted(glob(os.path.join(data_dir, "validation-*.parquet")))
    if not train_files:
        raise FileNotFoundError(f"No train parquet shards found under {data_dir}")
    if not val_files:
        raise FileNotFoundError(f"No validation parquet shards found under {data_dir}")
    return {"train": train_files, "validation": val_files}


def _iter_parquet_rows(parquet_paths: list[str], columns: list[str]):
    pq = _parquet_module()
    for parquet_path in parquet_paths:
        parquet_file = pq.ParquetFile(parquet_path)
        for row_group_idx in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(row_group_idx, columns=columns)
            for row in table.to_pylist():
                yield row


def _count_parquet_rows(parquet_paths: list[str]) -> int:
    pq = _parquet_module()
    return sum(int(pq.ParquetFile(path).metadata.num_rows) for path in parquet_paths)


def _relative_parquet_image_path(row: dict) -> str:
    filename = _norm_text(row.get("filename"))
    folder = _norm_text(row.get("folder"))
    if not filename:
        source = row.get("source") or {}
        filename = _norm_text(source.get("filename"))
        folder = folder or _norm_text(source.get("folder"))
    if not filename:
        raise ValueError("ADE20K parquet row is missing filename/source.filename")
    filename = os.path.basename(filename.replace("\\", "/"))
    folder = folder.replace("\\", "/").strip("/")
    return f"{folder}/{filename}" if folder else filename


def _polygon_from_object(obj: dict, width: int, height: int) -> list[list[float]] | None:
    polygon = obj.get("polygon") or {}
    xs = polygon.get("x") or []
    ys = polygon.get("y") or []
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    flat: list[float] = []
    for x, y in zip(xs, ys):
        flat.extend(
            [
                min(max(float(x), 0.0), max(0.0, float(width - 1))),
                min(max(float(y), 0.0), max(0.0, float(height - 1))),
            ]
        )
    if len(flat) < 6:
        return None
    return [flat]


def _annotation_from_polygon(
    ann_id: int,
    image_id: int,
    category_id: int,
    segmentation: list[list[float]],
    width: int,
    height: int,
) -> dict | None:
    mask_utils = _mask_utils()
    try:
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
        area = float(mask_utils.area(rle))
        bbox = [float(v) for v in mask_utils.toBbox(rle).tolist()]
    except Exception:
        return None
    if area <= 0.0:
        return None
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": segmentation,
        "bbox": bbox,
        "area": area,
        "iscrowd": 0,
    }


def _build_parquet_category_mapping(shards: dict[str, list[str]]) -> tuple[dict[str, int], list[dict]]:
    original_names: set[str] = set()
    for split_name in ("train", "validation"):
        iterator = _iter_parquet_rows(shards[split_name], columns=["objects"])
        total_rows = _count_parquet_rows(shards[split_name])
        for row in tqdm(iterator, total=total_rows, desc=f"ADE20K parquet category scan [{split_name}]"):
            for obj in row.get("objects") or []:
                name = _object_category_name(obj)
                if name:
                    original_names.add(name)

    original_to_id: dict[str, int] = {}
    categories: list[dict] = []
    used_names: set[str] = set()
    for next_id, original_name in enumerate(sorted(original_names), start=1):
        export_name = _sanitize_name(original_name)
        if export_name in used_names:
            suffix = 2
            candidate = f"{export_name} ({suffix})"
            while candidate in used_names:
                suffix += 1
                candidate = f"{export_name} ({suffix})"
            export_name = candidate
        used_names.add(export_name)
        original_to_id[original_name] = next_id
        categories.append({"id": next_id, "name": export_name, "original_name": original_name})
    return original_to_id, categories


def _write_parquet_image(cell, dst_path: str) -> None:
    raw = _extract_raw_bytes(cell)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if raw is not None:
        with open(dst_path, "wb") as handle:
            handle.write(raw)
        return
    image = _decode_image_cell(cell)
    image.save(dst_path)


def _export_parquet_split(
    split_name: str,
    parquet_paths: list[str],
    output_json: str,
    image_root: str,
    categories: list[dict],
    category_ids: dict[str, int],
) -> dict:
    os.makedirs(image_root, exist_ok=True)
    images: list[dict] = []
    annotations: list[dict] = []
    next_ann_id = 1
    skipped_objects = 0

    iterator = _iter_parquet_rows(parquet_paths, columns=["image", "filename", "folder", "source", "objects"])
    total_rows = _count_parquet_rows(parquet_paths)
    for image_id, row in enumerate(tqdm(iterator, total=total_rows, desc=f"ADE20K parquet export [{split_name}]"), start=1):
        image = _decode_image_cell(row["image"])
        width, height = image.size
        rel_path = _relative_parquet_image_path(row)
        dst_path = os.path.join(image_root, rel_path)
        if not os.path.exists(dst_path):
            _write_parquet_image(row["image"], dst_path)

        images.append(
            {
                "id": image_id,
                "width": int(width),
                "height": int(height),
                "file_name": rel_path,
                "not_exhaustive_category_ids": [],
            }
        )

        for obj in row.get("objects") or []:
            category_name = _object_category_name(obj)
            category_id = category_ids.get(category_name)
            segmentation = _polygon_from_object(obj, width, height)
            if category_id is None or segmentation is None:
                skipped_objects += 1
                continue
            annotation = _annotation_from_polygon(
                ann_id=next_ann_id,
                image_id=image_id,
                category_id=category_id,
                segmentation=segmentation,
                width=width,
                height=height,
            )
            if annotation is None:
                skipped_objects += 1
                continue
            annotations.append(annotation)
            next_ann_id += 1

    _write_json_from_rows(output_json, images, annotations, categories)
    return {
        "output_json": output_json,
        "image_root": image_root,
        "images": len(images),
        "annotations": len(annotations),
        "skipped_objects": skipped_objects,
    }


def export_parquet_ade20k(dataset_root: str, force_rebuild_export: bool = False) -> ExportLayout:
    paths = ExportPaths(dataset_root=dataset_root)
    train_image_dir = os.path.join(dataset_root, "train", "images")
    val_image_dir = os.path.join(dataset_root, "val", "images")

    if not force_rebuild_export and _can_reuse_export(paths):
        summary = _load_summary(paths)
        summary["reused_existing_export"] = True
        _write_summary(paths, summary)
        print(f"Reusing existing ADE20K parquet export at {dataset_root}")
        return ExportLayout(
            dataset_root=dataset_root,
            dataset_format="parquet",
            train_ann_file=paths.train_json,
            val_ann_file=paths.val_json,
            train_image_dir=train_image_dir,
            val_image_dir=val_image_dir,
        )

    shards = _discover_parquet_files(dataset_root)
    print(f"Discovered {len(shards['train'])} train parquet shards and {len(shards['validation'])} validation parquet shards.")
    print("Building ADE20K parquet category mapping...")
    category_ids, categories = _build_parquet_category_mapping(shards)
    print(f"Resolved {len(categories)} categories. Exporting train/val COCO-style files...")

    train_summary = _export_parquet_split(
        split_name="train",
        parquet_paths=shards["train"],
        output_json=paths.train_json,
        image_root=train_image_dir,
        categories=categories,
        category_ids=category_ids,
    )
    val_summary = _export_parquet_split(
        split_name="validation",
        parquet_paths=shards["validation"],
        output_json=paths.val_json,
        image_root=val_image_dir,
        categories=categories,
        category_ids=category_ids,
    )

    summary = {
        "dataset_root": dataset_root,
        "dataset_format": "parquet",
        "reused_existing_export": False,
        "categories": {"count": len(categories)},
        "export": {"train": train_summary, "validation": val_summary},
    }
    _write_summary(paths, summary)
    print(f"ADE20K parquet export complete at {dataset_root}")
    return ExportLayout(
        dataset_root=dataset_root,
        dataset_format="parquet",
        train_ann_file=paths.train_json,
        val_ann_file=paths.val_json,
        train_image_dir=train_image_dir,
        val_image_dir=val_image_dir,
    )


def export_ade20k(dataset_root: str, dataset_format: str, force_rebuild_export: bool = False) -> ExportLayout:
    resolved_format = _detect_dataset_format(dataset_root, dataset_format)
    if resolved_format == "ade20k_150":
        return export_ade20k150(dataset_root, force_rebuild_export=force_rebuild_export)
    return export_parquet_ade20k(dataset_root, force_rebuild_export=force_rebuild_export)


def _run_pipeline(args: argparse.Namespace, layout: ExportLayout) -> dict:
    from GitHub import main_pipeline

    pipeline_args = argparse.Namespace(
        dataset_root=layout.dataset_root,
        segmentation_method=args.segmentation_method,
        feature_matching_method=args.feature_matching_method,
        reference_images_per_class=args.reference_images_per_class,
        raw_bank_batch_size=args.raw_bank_batch_size,
        feature_top_k=args.feature_top_k,
        feature_accuracy_threshold=args.feature_accuracy_threshold,
        max_references=args.max_references,
        max_images=args.max_images,
        save_mask_json=args.save_mask_json,
        resume=args.resume,
        output_json=None,
    )

    stage_layout = main_pipeline.DatasetLayout(
        dataset_root=layout.dataset_root,
        train_ann_file=layout.train_ann_file,
        val_ann_file=layout.val_ann_file,
        train_image_dir=layout.train_image_dir,
        val_image_dir=layout.val_image_dir,
    )

    print("Resolved pipeline layout:")
    print(f"  train annotations : {layout.train_ann_file}")
    print(f"  val annotations   : {layout.val_ann_file}")
    print(f"  train images      : {layout.train_image_dir}")
    print(f"  val images        : {layout.val_image_dir}")
    print(f"  raw feature bank  : {_raw_feature_bank_dir(layout.dataset_root)}")
    print(f"  scored bank       : {_scored_feature_bank_dir(layout.dataset_root)}")
    print(f"  filtered bank     : {_filtered_feature_bank_dir(layout.dataset_root)}")

    results: dict[str, object] = {
        "config": {
            **vars(pipeline_args),
            "resolved_train_ann_file": stage_layout.train_ann_file,
            "resolved_val_ann_file": stage_layout.val_ann_file,
            "resolved_train_image_dir": stage_layout.train_image_dir,
            "resolved_val_image_dir": stage_layout.val_image_dir,
            "resolved_raw_feature_bank_dir": stage_layout.raw_feature_bank_dir,
            "resolved_scored_feature_bank_dir": stage_layout.scored_feature_bank_dir,
            "resolved_filtered_feature_bank_dir": stage_layout.filtered_feature_bank_dir,
            "dataset_format": layout.dataset_format,
        },
        "stages": {},
    }

    if args.segmentation_method in ("point-only", "text-and-point", "all"):
        print("Running Stage 1: build raw bank, score features, then apply adaptive_q75 filtering")
        results["stages"]["stage1"] = main_pipeline._run_stage1(pipeline_args, stage_layout)
        print("Running Stage 2: cache feature matching prompts")
        results["stages"]["stage2"] = main_pipeline._run_stage2(pipeline_args, stage_layout)
    else:
        print("Skipping Stage 1 and Stage 2 for text-only segmentation")
        results["stages"]["stage1"] = {"skipped": True, "reason": "text-only segmentation does not use the feature bank"}
        results["stages"]["stage2"] = {"skipped": True, "reason": "text-only segmentation does not use cached prompts"}

    if args.segmentation_method == "all":
        print("Running Stage 4: text-only segmentation")
        text_only = main_pipeline._run_stage4_once(pipeline_args, stage_layout, "text-only")
        print("Running Stage 4: point-only segmentation")
        point_only = main_pipeline._run_stage4_once(pipeline_args, stage_layout, "point-only")
        print("Running Stage 4: text-and-point segmentation")
        text_and_point = main_pipeline._run_stage4_once(pipeline_args, stage_layout, "text-and-point")
        results["stages"]["stage4"] = {
            "text-only": text_only,
            "point-only": point_only,
            "text-and-point": text_and_point,
        }
    else:
        print(f"Running Stage 4: {args.segmentation_method}")
        results["stages"]["stage4"] = main_pipeline._run_stage4_once(
            pipeline_args,
            stage_layout,
            args.segmentation_method,
        )
    return results


def _print_final_evaluation_summary(pipeline_result: dict) -> None:
    stage4 = pipeline_result.get("stages", {}).get("stage4")
    if not isinstance(stage4, dict):
        return
    label_map = {
        "text-only": "text-only",
        "point-only": "points-only (hybrid)",
        "text-and-point": "text-and-points (hybrid points)",
        "stage4": "stage4",
    }

    def print_one(method_name: str, payload: dict) -> None:
        global_metrics = payload.get("global")
        if not isinstance(global_metrics, dict):
            return
        images_evaluated = payload.get("config", {}).get("images_evaluated", "n/a")
        map50 = global_metrics.get("mAP50", global_metrics.get("AP50", 0.0))
        print(
            f"- {label_map.get(method_name, method_name)}: "
            f"IoU={global_metrics.get('IoU', 0.0):.4f} | "
            f"mIoU={global_metrics.get('mIoU', 0.0):.4f} | "
            f"F1={global_metrics.get('F1', 0.0):.4f} | "
            f"PixelPrecision={global_metrics.get('PixelPrecision', 0.0):.4f} | "
            f"PixelRecall={global_metrics.get('PixelRecall', 0.0):.4f} | "
            f"mAP50={map50:.4f} | "
            f"images={images_evaluated}"
        )

    print("Final evaluation summary:")
    if "global" in stage4:
        print_one("stage4", stage4)
        return
    for method_name in ("text-only", "point-only", "text-and-point"):
        payload = stage4.get(method_name)
        if isinstance(payload, dict):
            print_one(method_name, payload)


def run(args: argparse.Namespace) -> dict:
    print(f"Starting ADE20K pipeline at {args.dataset_root}")
    layout = export_ade20k(
        dataset_root=args.dataset_root,
        dataset_format=args.dataset_format,
        force_rebuild_export=args.force_rebuild_export,
    )
    result: dict[str, object] = {
        "export": _load_summary(ExportPaths(args.dataset_root)),
    }

    if args.prepare_only:
        print("Export completed. Skipping segmentation because --prepare-only was used.")
        result["pipeline"] = {"skipped": True, "reason": "--prepare-only was used"}
    else:
        result["pipeline"] = _run_pipeline(args, layout)
        _print_final_evaluation_summary(result["pipeline"])

    if args.output_json:
        out_dir = os.path.dirname(args.output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w") as handle:
            json.dump(result, handle, indent=2, default=str)
    return result


def main(argv: list[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
