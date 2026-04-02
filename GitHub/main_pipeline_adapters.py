"""
Global adapter-based dataset pipeline.

This entrypoint keeps the stage execution logic in one place and lets each
dataset family provide only the export/normalization adapter it needs.

Current adapters:
- `ade20k_parquet` : 1aurent/ADE20K HuggingFace parquet download
- `ade20k_150`     : zhoubolei/scene_parse_150 / ADEChallengeData2016 cache
- `ade20k_847`     : ADE20K A-847 built from the raw HuggingFace Arrow cache
- `cityscapes`     : raw Cityscapes leftImg8bit/gtFine download
- `lvis`           : existing LVIS-style train/val JSON + image layout
- `agri_segmentation`: existing COCO-style train/test JSON + image layout
- `pc59`           : Pascal Context 59 mask export with predefined train/val splits
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import os
from glob import glob

import numpy as np
from PIL import Image
from tqdm import tqdm

from GitHub import main_pipeline_ade20k as ade20k


ADAPTERS = ("auto", "ade20k_parquet", "ade20k_150", "ade20k_847", "cityscapes", "lvis", "agri_segmentation", "pc59")
A847_LOCAL_METADATA_CANDIDATES = ()
A847_OBJECTS_URL = (
    "https://raw.githubusercontent.com/CSAILVision/ADE20K"
    "/main/dataset/ADE20K_2021_17_01/objects.txt"
)
CITYSCAPES_CLASSES = [
    (7, "road"),
    (8, "sidewalk"),
    (11, "building"),
    (12, "wall"),
    (13, "fence"),
    (17, "pole"),
    (19, "traffic light"),
    (20, "traffic sign"),
    (21, "vegetation"),
    (22, "terrain"),
    (23, "sky"),
    (24, "person"),
    (25, "rider"),
    (26, "car"),
    (27, "truck"),
    (28, "bus"),
    (31, "train"),
    (32, "motorcycle"),
    (33, "bicycle"),
]


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
        description="Run the segmentation pipeline through a dataset adapter."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--adapter", choices=ADAPTERS, default="auto")
    parser.add_argument(
        "--segmentation-method",
        choices=ade20k.SEGMENTATION_METHODS,
        required=True,
    )
    parser.add_argument(
        "--feature-matching-method",
        default="hybrid",
        choices=("absolute_similarity", "relative_similarity", "hybrid"),
    )
    parser.add_argument("--reference-images-per-class", type=int, default=30)
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
    parser.add_argument("--max-references", type=int, default=10000)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force-rebuild-export", action="store_true")
    parser.add_argument("--save-mask-json", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--a847-class-list",
        default=None,
        help="Optional local path to the official A-847 objects.txt file.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _pyarrow():
    import pyarrow as pa
    import pyarrow.ipc as ipc

    return pa, ipc


def _auto_adapter(dataset_root: str) -> str:
    if glob(os.path.join(dataset_root, "data", "train-*.parquet")):
        return "ade20k_parquet"
    if ade20k._find_ade20k150_root(dataset_root) is not None:
        return "ade20k_150"
    if _find_a847_arrow_root(dataset_root) is not None:
        return "ade20k_847"
    if _find_cityscapes_roots(dataset_root) is not None:
        return "cityscapes"
    if _find_lvis_root(dataset_root):
        return "lvis"
    if _find_agri_segmentation_root(dataset_root):
        return "agri_segmentation"
    if _find_pc59_root(dataset_root):
        return "pc59"
    raise FileNotFoundError(
        f"Could not detect a supported adapter layout under {dataset_root}."
    )


def _resolve_adapter(adapter: str, dataset_root: str) -> str:
    return _auto_adapter(dataset_root) if adapter == "auto" else adapter


def _decode_pil_cell(cell) -> Image.Image:
    if isinstance(cell, Image.Image):
        return cell.copy()
    raw = ade20k._extract_raw_bytes(cell)
    if raw is None:
        raise ValueError(f"Unsupported image cell type: {type(cell)!r}")
    with Image.open(io.BytesIO(raw)) as image:
        return image.copy()


def _write_image_cell(cell, dst_path: str) -> None:
    raw = ade20k._extract_raw_bytes(cell)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if raw is not None:
        with open(dst_path, "wb") as handle:
            handle.write(raw)
        return
    _decode_pil_cell(cell).save(dst_path)


def _find_a847_arrow_root(dataset_root: str) -> str | None:
    candidates = glob(
        os.path.join(
            dataset_root,
            "hf_cache",
            "1aurent___ade20_k",
            "default",
            "0.0.0",
            "*",
        )
    )
    for candidate in candidates:
        if glob(os.path.join(candidate, "ade20_k-train-*.arrow")):
            return candidate
    return None


def _find_cityscapes_roots(dataset_root: str) -> tuple[str, str] | None:
    candidates = [
        (
            os.path.join(dataset_root, "leftImg8bit_trainvaltest", "leftImg8bit"),
            os.path.join(dataset_root, "gtFine_trainvaltest", "gtFine"),
        ),
        (
            os.path.join(dataset_root, "leftImg8bit"),
            os.path.join(dataset_root, "gtFine"),
        ),
    ]
    for image_root, annotation_root in candidates:
        required = [
            os.path.join(image_root, "train"),
            os.path.join(image_root, "val"),
            os.path.join(annotation_root, "train"),
            os.path.join(annotation_root, "val"),
        ]
        if all(os.path.isdir(path) for path in required):
            return image_root, annotation_root
    return None


def _find_lvis_root(dataset_root: str) -> bool:
    candidates = [
        (
            os.path.join(dataset_root, "train", "lvis_v1_train.json"),
            os.path.join(dataset_root, "val", "lvis_v1_val.json"),
        ),
        (
            os.path.join(dataset_root, "lvis_v1_train.json"),
            os.path.join(dataset_root, "lvis_v1_val.json"),
        ),
    ]
    return any(os.path.isfile(train_json) and os.path.isfile(val_json) for train_json, val_json in candidates)


def _find_agri_segmentation_root(dataset_root: str) -> bool:
    required = [
        os.path.join(dataset_root, "train.json"),
        os.path.join(dataset_root, "test.json"),
    ]
    return all(os.path.isfile(path) for path in required)


def _find_pc59_root(dataset_root: str) -> bool:
    required = [
        os.path.join(dataset_root, "JPEGImages"),
        os.path.join(dataset_root, "subset59_from_459"),
        os.path.join(dataset_root, "ImageSets", "pascal_context_train.txt"),
        os.path.join(dataset_root, "ImageSets", "pascal_context_val.txt"),
        os.path.join(dataset_root, "59_labels.txt"),
    ]
    return all(os.path.exists(path) for path in required)


def _iter_arrow_rows(arrow_paths: list[str]):
    pa, ipc = _pyarrow()
    for arrow_path in arrow_paths:
        reader = ipc.open_stream(pa.memory_map(arrow_path, "r"))
        for batch in reader:
            for row in batch.to_pylist():
                yield row


def _ensure_a847_class_list(dataset_root: str, explicit_path: str | None) -> str:
    if explicit_path:
        if not os.path.isfile(explicit_path):
            raise FileNotFoundError(f"A-847 class list not found: {explicit_path}")
        return explicit_path

    local_path = os.path.join(dataset_root, "objects_847.txt")
    if os.path.isfile(local_path):
        return local_path

    try:
        import requests
    except ImportError as exc:
        raise ImportError(
            "ADE20K-847 export requires requests if objects_847.txt is not already present. "
            "Install it or place objects_847.txt under the dataset root."
        ) from exc

    print(f"Downloading A-847 class list to {local_path}")
    response = requests.get(A847_OBJECTS_URL, timeout=30)
    response.raise_for_status()
    with open(local_path, "w", encoding="utf-8") as handle:
        handle.write(response.text)
    return local_path


def _load_a847_mapping(objects_path: str) -> tuple[dict[str, int], list[dict]]:
    with open(objects_path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle.readlines() if line.strip()]

    name_to_idx: dict[str, int] = {}
    categories: list[dict] = []
    for idx, line in enumerate(lines, start=1):
        name = line.split("\t")[-1].strip().lower()
        name_to_idx[name] = idx
        categories.append({"id": idx, "name": ade20k._sanitize_name(name)})
    return name_to_idx, categories


def _load_a847_mapping_from_local_metadata() -> tuple[dict[str, int], list[dict]] | None:
    for metadata_path in A847_LOCAL_METADATA_CANDIDATES:
        if not os.path.isfile(metadata_path):
            continue
        with open(metadata_path, "r", encoding="utf-8") as handle:
            module_ast = ast.parse(handle.read(), filename=metadata_path)
        for node in module_ast.body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "ADE20K_SEM_SEG_FULL_CATEGORIES":
                    categories_raw = ast.literal_eval(node.value)
                    name_to_idx: dict[str, int] = {}
                    categories: list[dict] = []
                    for item in categories_raw:
                        class_id = int(item["trainId"]) + 1
                        name = str(item["name"]).strip().lower()
                        name_to_idx[name] = class_id
                        categories.append({"id": class_id, "name": ade20k._sanitize_name(name)})
                    return name_to_idx, categories
    return None


def _a847_lookup(obj: dict, name_to_idx: dict[str, int]) -> int:
    for key in ("name", "raw_name"):
        value = str(obj.get(key, "")).strip().lower()
        if value in name_to_idx:
            return name_to_idx[value]
    return 0


def _binary_mask_from_instance(cell) -> np.ndarray:
    arr = np.asarray(_decode_pil_cell(cell))
    if arr.ndim == 3:
        return np.any(arr > 0, axis=-1)
    return arr > 0


def _build_a847_flat_mask(row: dict, height: int, width: int, name_to_idx: dict[str, int]) -> tuple[np.ndarray, bool]:
    flat_mask = np.zeros((height, width), dtype=np.uint16)
    objects = row.get("objects") or []
    instances = row.get("instances") or []
    if not objects or not instances:
        return flat_mask, False

    mismatched = len(objects) != len(instances)
    paired = sorted(
        zip(objects, instances),
        key=lambda item: item[0].get("depth_ordering_rank", 9999),
    )
    for obj, instance_cell in paired:
        class_id = _a847_lookup(obj, name_to_idx)
        if class_id == 0:
            continue
        belongs = _binary_mask_from_instance(instance_cell)
        flat_mask[belongs] = class_id
    return flat_mask, mismatched


def _encode_binary_mask(binary_mask: np.ndarray) -> tuple[dict, float, list[float]] | None:
    mask_utils = ade20k._mask_utils()
    binary_mask = np.asarray(binary_mask).astype(np.uint8)
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


def _cityscapes_categories() -> tuple[dict[int, int], list[dict]]:
    label_to_category_id: dict[int, int] = {}
    categories: list[dict] = []
    for category_id, (label_id, name) in enumerate(CITYSCAPES_CLASSES, start=1):
        label_to_category_id[label_id] = category_id
        categories.append(
            {
                "id": category_id,
                "name": ade20k._sanitize_name(name),
                "cityscapes_label_id": label_id,
            }
        )
    return label_to_category_id, categories


def _pc59_categories(labels_path: str) -> list[dict]:
    categories: list[dict] = []
    with open(labels_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if ":" in line:
                class_id_str, name = line.split(":", 1)
                class_id = int(class_id_str.strip())
            else:
                class_id = len(categories) + 1
                name = line
            if class_id <= 0:
                continue
            categories.append({"id": class_id, "name": ade20k._sanitize_name(name.strip())})
    return categories


def _list_pc59_pairs(
    split_file: str,
    image_dir: str,
    mask_dir: str,
) -> tuple[list[tuple[str, str, str]], dict]:
    pairs: list[tuple[str, str, str]] = []
    missing_images = 0
    missing_masks = 0
    with open(split_file, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    for line in lines:
        parts = line.split()
        image_rel = parts[0]
        stem = os.path.splitext(os.path.basename(image_rel))[0]
        image_path = os.path.join(image_dir, f"{stem}.jpg")
        if not os.path.isfile(image_path):
            image_path = os.path.join(image_dir, f"{stem}.png")
        mask_path = os.path.join(mask_dir, f"{stem}.png")
        if not os.path.isfile(image_path):
            missing_images += 1
            continue
        if not os.path.isfile(mask_path):
            missing_masks += 1
            continue
        pairs.append((image_path, mask_path, os.path.basename(image_path)))
    return pairs, {
        "split_entries": len(lines),
        "missing_images": missing_images,
        "missing_masks": missing_masks,
    }


def _list_cityscapes_pairs(image_split_dir: str, annotation_split_dir: str) -> tuple[list[tuple[str, str, str]], dict]:
    pairs: list[tuple[str, str, str]] = []
    missing_images = 0
    pattern = os.path.join(annotation_split_dir, "*", "*_gtFine_labelIds.png")
    for mask_path in sorted(glob(pattern)):
        rel_mask_path = os.path.relpath(mask_path, annotation_split_dir)
        city_name = os.path.dirname(rel_mask_path)
        mask_name = os.path.basename(mask_path)
        stem = mask_name[: -len("_gtFine_labelIds.png")]
        rel_image_path = os.path.join(city_name, f"{stem}_leftImg8bit.png")
        image_path = os.path.join(image_split_dir, rel_image_path)
        if not os.path.isfile(image_path):
            missing_images += 1
            continue
        pairs.append((image_path, mask_path, rel_image_path.replace(os.sep, "/")))
    return pairs, {"missing_images_for_masks": missing_images}


def _remap_cityscapes_mask(mask: np.ndarray, label_to_category_id: dict[int, int]) -> np.ndarray:
    remapped = np.zeros(mask.shape, dtype=np.uint8)
    for label_id, category_id in label_to_category_id.items():
        remapped[mask == label_id] = category_id
    return remapped


def _export_cityscapes_split(
    split_name: str,
    image_dir: str,
    annotation_dir: str,
    output_json: str,
    categories: list[dict],
    label_to_category_id: dict[int, int],
) -> dict:
    pairs, pair_stats = _list_cityscapes_pairs(image_dir, annotation_dir)
    print(
        f"Preparing Cityscapes {split_name} split from {image_dir} "
        f"with {len(pairs)} image/mask pairs."
    )

    images: list[dict] = []
    annotations: list[dict] = []
    next_ann_id = 1
    images_without_foreground = 0
    ignored_only_images = 0

    iterator = tqdm(pairs, desc=f"Cityscapes export [{split_name}]", total=len(pairs))
    for image_id, (image_path, mask_path, rel_image_path) in enumerate(iterator, start=1):
        with Image.open(image_path) as image:
            width, height = image.size
        with Image.open(mask_path) as mask_image:
            raw_mask = np.array(mask_image, dtype=np.int32)
        mask = _remap_cityscapes_mask(raw_mask, label_to_category_id)

        images.append(
            {
                "id": image_id,
                "width": int(width),
                "height": int(height),
                "file_name": rel_image_path,
                "not_exhaustive_category_ids": [],
            }
        )

        class_ids = [int(class_id) for class_id in np.unique(mask) if 1 <= int(class_id) <= len(categories)]
        if not class_ids:
            images_without_foreground += 1
            if np.all(mask == 0):
                ignored_only_images += 1
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

    ade20k._write_json_from_rows(output_json, images, annotations, categories)
    return {
        "output_json": output_json,
        "image_dir": image_dir,
        "annotation_dir": annotation_dir,
        "images": len(images),
        "annotations": len(annotations),
        "images_without_foreground": images_without_foreground,
        "ignored_only_images": ignored_only_images,
        **pair_stats,
    }


def _export_pc59_split(
    split_name: str,
    split_file: str,
    image_dir: str,
    mask_dir: str,
    output_json: str,
    categories: list[dict],
) -> dict:
    pairs, pair_stats = _list_pc59_pairs(split_file, image_dir, mask_dir)
    print(
        f"Preparing PC59 {split_name} split from {split_file} "
        f"with {len(pairs)} image/mask pairs."
    )

    max_class_id = max(category["id"] for category in categories) if categories else 0
    images: list[dict] = []
    annotations: list[dict] = []
    next_ann_id = 1
    images_without_foreground = 0
    background_only_images = 0

    iterator = tqdm(pairs, desc=f"PC59 export [{split_name}]", total=len(pairs))
    for image_id, (image_path, mask_path, rel_image_path) in enumerate(iterator, start=1):
        with Image.open(image_path) as image:
            width, height = image.size
        with Image.open(mask_path) as mask_image:
            mask = np.array(mask_image, dtype=np.int32)

        images.append(
            {
                "id": image_id,
                "width": int(width),
                "height": int(height),
                "file_name": rel_image_path,
                "not_exhaustive_category_ids": [],
            }
        )

        class_ids = [int(class_id) for class_id in np.unique(mask) if 1 <= int(class_id) <= max_class_id]
        if not class_ids:
            images_without_foreground += 1
            if np.all(mask == 0):
                background_only_images += 1
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

    ade20k._write_json_from_rows(output_json, images, annotations, categories)
    return {
        "output_json": output_json,
        "split_file": split_file,
        "image_dir": image_dir,
        "mask_dir": mask_dir,
        "images": len(images),
        "annotations": len(annotations),
        "images_without_foreground": images_without_foreground,
        "background_only_images": background_only_images,
        **pair_stats,
    }


def _export_a847_split(
    split_name: str,
    arrow_paths: list[str],
    image_root: str,
    output_json: str,
    categories: list[dict],
    name_to_idx: dict[str, int],
) -> dict:
    os.makedirs(image_root, exist_ok=True)
    images: list[dict] = []
    annotations: list[dict] = []
    ann_id = 1
    images_without_foreground = 0
    background_only_images = 0
    instance_length_mismatches = 0

    pbar = tqdm(desc=f"ADE20K-847 export [{split_name}]", unit="image")
    for image_id, row in enumerate(_iter_arrow_rows(arrow_paths), start=1):
        image = _decode_pil_cell(row["image"]).convert("RGB")
        width, height = image.size
        file_name = os.path.basename(str(row.get("filename") or ""))
        dst_path = os.path.join(image_root, file_name)
        if not os.path.exists(dst_path):
            _write_image_cell(row["image"], dst_path)

        flat_mask, mismatched = _build_a847_flat_mask(row, height, width, name_to_idx)
        if mismatched:
            instance_length_mismatches += 1

        images.append(
            {
                "id": image_id,
                "width": int(width),
                "height": int(height),
                "file_name": file_name,
                "not_exhaustive_category_ids": [],
            }
        )

        class_ids = [int(cid) for cid in np.unique(flat_mask) if 1 <= int(cid) <= len(categories)]
        if not class_ids:
            images_without_foreground += 1
            if np.all(flat_mask == 0):
                background_only_images += 1
            pbar.update(1)
            continue

        for class_id in class_ids:
            encoded = _encode_binary_mask(flat_mask == class_id)
            if encoded is None:
                continue
            segmentation, area, bbox = encoded
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "segmentation": segmentation,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
        pbar.update(1)
    pbar.close()

    ade20k._write_json_from_rows(output_json, images, annotations, categories)
    return {
        "output_json": output_json,
        "image_root": image_root,
        "images": len(images),
        "annotations": len(annotations),
        "images_without_foreground": images_without_foreground,
        "background_only_images": background_only_images,
        "instance_length_mismatches": instance_length_mismatches,
    }


def export_ade20k_847(dataset_root: str, force_rebuild_export: bool, class_list_path: str | None) -> ade20k.ExportLayout:
    paths = ade20k.ExportPaths(dataset_root=dataset_root)
    train_image_dir = os.path.join(dataset_root, "train", "images")
    val_image_dir = os.path.join(dataset_root, "val", "images")

    if not force_rebuild_export and ade20k._can_reuse_export(paths):
        summary = ade20k._load_summary(paths)
        summary["reused_existing_export"] = True
        ade20k._write_summary(paths, summary)
        print(f"Reusing existing ADE20K-847 export at {dataset_root}")
        return ade20k.ExportLayout(
            dataset_root=dataset_root,
            dataset_format="ade20k_847",
            train_ann_file=paths.train_json,
            val_ann_file=paths.val_json,
            train_image_dir=train_image_dir,
            val_image_dir=val_image_dir,
        )

    arrow_root = _find_a847_arrow_root(dataset_root)
    if arrow_root is None:
        raise FileNotFoundError(
            f"Could not find the raw ADE20K-847 Arrow cache under {dataset_root}."
        )

    mapping = _load_a847_mapping_from_local_metadata()
    objects_path = None
    if mapping is not None:
        name_to_idx, categories = mapping
        print("Using local ADE20K-847 metadata discovered from the configured adapter metadata candidates.")
    else:
        objects_path = _ensure_a847_class_list(dataset_root, class_list_path)
        name_to_idx, categories = _load_a847_mapping(objects_path)
    train_arrows = sorted(glob(os.path.join(arrow_root, "ade20_k-train-*.arrow")))
    val_arrows = sorted(glob(os.path.join(arrow_root, "ade20_k-validation-*.arrow")))
    if not train_arrows or not val_arrows:
        raise FileNotFoundError(f"Missing train/validation Arrow shards under {arrow_root}")

    print(f"Exporting ADE20K-847 with {len(categories)} foreground classes. Background class 0 is ignored.")
    train_summary = _export_a847_split(
        split_name="train",
        arrow_paths=train_arrows,
        image_root=train_image_dir,
        output_json=paths.train_json,
        categories=categories,
        name_to_idx=name_to_idx,
    )
    val_summary = _export_a847_split(
        split_name="validation",
        arrow_paths=val_arrows,
        image_root=val_image_dir,
        output_json=paths.val_json,
        categories=categories,
        name_to_idx=name_to_idx,
    )

    summary = {
        "dataset_root": dataset_root,
        "dataset_format": "ade20k_847",
        "reused_existing_export": False,
        "arrow_root": arrow_root,
        "objects_847_txt": objects_path,
        "local_metadata_source": None,
        "categories": {"count": len(categories)},
        "export": {
            "train": train_summary,
            "validation": val_summary,
        },
    }
    ade20k._write_summary(paths, summary)
    print(f"ADE20K-847 export complete at {dataset_root}")
    return ade20k.ExportLayout(
        dataset_root=dataset_root,
        dataset_format="ade20k_847",
        train_ann_file=paths.train_json,
        val_ann_file=paths.val_json,
        train_image_dir=train_image_dir,
        val_image_dir=val_image_dir,
    )


def export_cityscapes(dataset_root: str, force_rebuild_export: bool) -> ade20k.ExportLayout:
    paths = ade20k.ExportPaths(dataset_root=dataset_root)
    roots = _find_cityscapes_roots(dataset_root)
    if roots is None:
        raise FileNotFoundError(
            f"Could not find a supported raw Cityscapes layout under {dataset_root}."
        )
    image_root, annotation_root = roots
    train_image_dir = os.path.join(image_root, "train")
    val_image_dir = os.path.join(image_root, "val")
    train_annotation_dir = os.path.join(annotation_root, "train")
    val_annotation_dir = os.path.join(annotation_root, "val")

    if not force_rebuild_export and ade20k._can_reuse_export(paths):
        summary = ade20k._load_summary(paths)
        summary["reused_existing_export"] = True
        ade20k._write_summary(paths, summary)
        print(f"Reusing existing Cityscapes export at {dataset_root}")
        return ade20k.ExportLayout(
            dataset_root=dataset_root,
            dataset_format="cityscapes",
            train_ann_file=paths.train_json,
            val_ann_file=paths.val_json,
            train_image_dir=train_image_dir,
            val_image_dir=val_image_dir,
        )

    label_to_category_id, categories = _cityscapes_categories()
    print(
        f"Exporting Cityscapes with {len(categories)} foreground classes. "
        "Void and unlabeled ids are ignored."
    )
    train_summary = _export_cityscapes_split(
        split_name="train",
        image_dir=train_image_dir,
        annotation_dir=train_annotation_dir,
        output_json=paths.train_json,
        categories=categories,
        label_to_category_id=label_to_category_id,
    )
    val_summary = _export_cityscapes_split(
        split_name="validation",
        image_dir=val_image_dir,
        annotation_dir=val_annotation_dir,
        output_json=paths.val_json,
        categories=categories,
        label_to_category_id=label_to_category_id,
    )

    summary = {
        "dataset_root": dataset_root,
        "dataset_format": "cityscapes",
        "reused_existing_export": False,
        "source": {
            "image_root": image_root,
            "annotation_root": annotation_root,
        },
        "categories": {"count": len(categories)},
        "export": {
            "train": train_summary,
            "validation": val_summary,
        },
    }
    ade20k._write_summary(paths, summary)
    print(f"Cityscapes export complete at {dataset_root}")
    return ade20k.ExportLayout(
        dataset_root=dataset_root,
        dataset_format="cityscapes",
        train_ann_file=paths.train_json,
        val_ann_file=paths.val_json,
        train_image_dir=train_image_dir,
        val_image_dir=val_image_dir,
    )


def export_lvis(dataset_root: str, force_rebuild_export: bool) -> ade20k.ExportLayout:
    from GitHub import main_pipeline

    paths = ade20k.ExportPaths(dataset_root=dataset_root)
    layout = main_pipeline._resolve_dataset_layout(dataset_root)

    if not force_rebuild_export and os.path.isfile(paths.summary_json):
        summary = ade20k._load_summary(paths)
        summary["reused_existing_export"] = True
        ade20k._write_summary(paths, summary)
        print(f"Reusing existing LVIS adapter summary at {dataset_root}")
    else:
        summary = {
            "dataset_root": dataset_root,
            "dataset_format": "lvis",
            "reused_existing_export": False,
            "source": {
                "train_ann_file": layout.train_ann_file,
                "val_ann_file": layout.val_ann_file,
                "train_image_dir": layout.train_image_dir,
                "val_image_dir": layout.val_image_dir,
            },
            "export": {
                "train": {
                    "output_json": layout.train_ann_file,
                    "image_dir": layout.train_image_dir,
                },
                "validation": {
                    "output_json": layout.val_ann_file,
                    "image_dir": layout.val_image_dir,
                },
            },
        }
        ade20k._write_summary(paths, summary)
        print(f"LVIS adapter summary written at {dataset_root}")

    return ade20k.ExportLayout(
        dataset_root=dataset_root,
        dataset_format="lvis",
        train_ann_file=layout.train_ann_file,
        val_ann_file=layout.val_ann_file,
        train_image_dir=layout.train_image_dir,
        val_image_dir=layout.val_image_dir,
    )


def export_agri_segmentation(dataset_root: str, force_rebuild_export: bool) -> ade20k.ExportLayout:
    from GitHub import main_pipeline

    paths = ade20k.ExportPaths(dataset_root=dataset_root)
    layout = main_pipeline._resolve_dataset_layout(dataset_root)

    if not force_rebuild_export and os.path.isfile(paths.summary_json):
        summary = ade20k._load_summary(paths)
        summary["reused_existing_export"] = True
        ade20k._write_summary(paths, summary)
        print(f"Reusing existing agri_segmentation adapter summary at {dataset_root}")
    else:
        summary = {
            "dataset_root": dataset_root,
            "dataset_format": "agri_segmentation",
            "reused_existing_export": False,
            "source": {
                "train_ann_file": layout.train_ann_file,
                "val_ann_file": layout.val_ann_file,
                "train_image_dir": layout.train_image_dir,
                "val_image_dir": layout.val_image_dir,
            },
            "export": {
                "train": {
                    "output_json": layout.train_ann_file,
                    "image_dir": layout.train_image_dir,
                },
                "validation": {
                    "output_json": layout.val_ann_file,
                    "image_dir": layout.val_image_dir,
                },
            },
        }
        ade20k._write_summary(paths, summary)
        print(f"agri_segmentation adapter summary written at {dataset_root}")

    return ade20k.ExportLayout(
        dataset_root=dataset_root,
        dataset_format="agri_segmentation",
        train_ann_file=layout.train_ann_file,
        val_ann_file=layout.val_ann_file,
        train_image_dir=layout.train_image_dir,
        val_image_dir=layout.val_image_dir,
    )


def export_pc59(dataset_root: str, force_rebuild_export: bool) -> ade20k.ExportLayout:
    paths = ade20k.ExportPaths(dataset_root=dataset_root)
    if not _find_pc59_root(dataset_root):
        raise FileNotFoundError(
            f"Could not find a supported PC59 layout under {dataset_root}."
        )

    image_dir = os.path.join(dataset_root, "JPEGImages")
    mask_dir = os.path.join(dataset_root, "subset59_from_459")
    train_split = os.path.join(dataset_root, "ImageSets", "pascal_context_train.txt")
    val_split = os.path.join(dataset_root, "ImageSets", "pascal_context_val.txt")
    labels_path = os.path.join(dataset_root, "59_labels.txt")

    if not force_rebuild_export and ade20k._can_reuse_export(paths):
        summary = ade20k._load_summary(paths)
        summary["reused_existing_export"] = True
        ade20k._write_summary(paths, summary)
        print(f"Reusing existing PC59 export at {dataset_root}")
        return ade20k.ExportLayout(
            dataset_root=dataset_root,
            dataset_format="pc59",
            train_ann_file=paths.train_json,
            val_ann_file=paths.val_json,
            train_image_dir=image_dir,
            val_image_dir=image_dir,
        )

    categories = _pc59_categories(labels_path)
    print(
        f"Exporting PC59 with {len(categories)} foreground classes. "
        "Background class 0 is ignored."
    )
    train_summary = _export_pc59_split(
        split_name="train",
        split_file=train_split,
        image_dir=image_dir,
        mask_dir=mask_dir,
        output_json=paths.train_json,
        categories=categories,
    )
    val_summary = _export_pc59_split(
        split_name="validation",
        split_file=val_split,
        image_dir=image_dir,
        mask_dir=mask_dir,
        output_json=paths.val_json,
        categories=categories,
    )

    summary = {
        "dataset_root": dataset_root,
        "dataset_format": "pc59",
        "reused_existing_export": False,
        "source": {
            "image_dir": image_dir,
            "mask_dir": mask_dir,
            "train_split": train_split,
            "val_split": val_split,
            "labels_path": labels_path,
        },
        "categories": {"count": len(categories)},
        "export": {
            "train": train_summary,
            "validation": val_summary,
        },
    }
    ade20k._write_summary(paths, summary)
    print(f"PC59 export complete at {dataset_root}")
    return ade20k.ExportLayout(
        dataset_root=dataset_root,
        dataset_format="pc59",
        train_ann_file=paths.train_json,
        val_ann_file=paths.val_json,
        train_image_dir=image_dir,
        val_image_dir=image_dir,
    )


def _export_with_adapter(args: argparse.Namespace) -> ade20k.ExportLayout:
    adapter = _resolve_adapter(args.adapter, args.dataset_root)
    if adapter == "ade20k_parquet":
        return ade20k.export_ade20k(
            dataset_root=args.dataset_root,
            dataset_format="parquet",
            force_rebuild_export=args.force_rebuild_export,
        )
    if adapter == "ade20k_150":
        return ade20k.export_ade20k(
            dataset_root=args.dataset_root,
            dataset_format="ade20k_150",
            force_rebuild_export=args.force_rebuild_export,
        )
    if adapter == "ade20k_847":
        return export_ade20k_847(
            dataset_root=args.dataset_root,
            force_rebuild_export=args.force_rebuild_export,
            class_list_path=args.a847_class_list,
        )
    if adapter == "cityscapes":
        return export_cityscapes(
            dataset_root=args.dataset_root,
            force_rebuild_export=args.force_rebuild_export,
        )
    if adapter == "lvis":
        return export_lvis(
            dataset_root=args.dataset_root,
            force_rebuild_export=args.force_rebuild_export,
        )
    if adapter == "agri_segmentation":
        return export_agri_segmentation(
            dataset_root=args.dataset_root,
            force_rebuild_export=args.force_rebuild_export,
        )
    if adapter == "pc59":
        return export_pc59(
            dataset_root=args.dataset_root,
            force_rebuild_export=args.force_rebuild_export,
        )
    raise ValueError(f"Unsupported adapter: {adapter}")


def run(args: argparse.Namespace) -> dict:
    print(f"Starting adapter-based pipeline at {args.dataset_root}")
    layout = _export_with_adapter(args)
    result: dict[str, object] = {
        "export": ade20k._load_summary(ade20k.ExportPaths(args.dataset_root)),
    }

    if args.prepare_only:
        print("Export completed. Skipping segmentation because --prepare-only was used.")
        result["pipeline"] = {"skipped": True, "reason": "--prepare-only was used"}
    else:
        result["pipeline"] = ade20k._run_pipeline(args, layout)
        ade20k._print_final_evaluation_summary(result["pipeline"])

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
