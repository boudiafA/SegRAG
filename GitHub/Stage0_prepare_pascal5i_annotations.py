"""
Stage 0: prepare COCO-style train/test annotations for PASCAL-5i / VOC.

This optional preprocessing step converts VOC semantic masks into the COCO-like
JSON format expected by the rest of the GitHub-facing pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm


VOC_CATEGORIES = [
    (1, "aeroplane"),
    (2, "bicycle"),
    (3, "bird"),
    (4, "boat"),
    (5, "bottle"),
    (6, "bus"),
    (7, "car"),
    (8, "cat"),
    (9, "chair"),
    (10, "cow"),
    (11, "diningtable"),
    (12, "dog"),
    (13, "horse"),
    (14, "motorbike"),
    (15, "person"),
    (16, "pottedplant"),
    (17, "sheep"),
    (18, "sofa"),
    (19, "train"),
    (20, "tvmonitor"),
]


@dataclass(frozen=True)
class PascalPaths:
    dataset_root: str

    @property
    def jpeg_dir(self) -> str:
        return os.path.join(self.dataset_root, "JPEGImages")

    @property
    def seg_dir(self) -> str:
        return os.path.join(self.dataset_root, "SegmentationClass")

    @property
    def split_dir(self) -> str:
        return os.path.join(self.dataset_root, "ImageSets", "Segmentation")

    @property
    def train_split(self) -> str:
        return os.path.join(self.split_dir, "train.txt")

    @property
    def val_split(self) -> str:
        return os.path.join(self.split_dir, "val.txt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 0: create train.json and test.json for PASCAL-5i.")
    parser.add_argument("--dataset-root", required=True, help="Root directory of the PASCAL-5i / VOC dataset.")
    parser.add_argument("--train-output", default=None, help="Optional output path for train.json.")
    parser.add_argument("--test-output", default=None, help="Optional output path for test.json.")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _load_split_ids(split_file: str) -> list[str]:
    with open(split_file, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _encode_binary_mask(binary_mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def _bbox_from_mask(binary_mask: np.ndarray) -> list[float]:
    ys, xs = np.where(binary_mask)
    x0 = int(xs.min())
    x1 = int(xs.max())
    y0 = int(ys.min())
    y1 = int(ys.max())
    return [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]


def _build_dataset(paths: PascalPaths, image_ids: list[str], split_name: str) -> dict:
    categories = [{"id": cid, "name": name} for cid, name in VOC_CATEGORIES]
    images = []
    annotations = []
    ann_id = 1

    for image_index, image_stem in enumerate(tqdm(image_ids, desc=f"Build {split_name}"), start=1):
        image_path = os.path.join(paths.jpeg_dir, f"{image_stem}.jpg")
        mask_path = os.path.join(paths.seg_dir, f"{image_stem}.png")
        if not os.path.isfile(image_path) or not os.path.isfile(mask_path):
            continue

        with Image.open(image_path) as image:
            width, height = image.size
        with Image.open(mask_path) as mask_image:
            mask = np.array(mask_image, dtype=np.uint8)

        images.append(
            {
                "id": image_index,
                "file_name": os.path.join("JPEGImages", f"{image_stem}.jpg"),
                "width": width,
                "height": height,
            }
        )

        present_class_ids = sorted(int(v) for v in np.unique(mask) if v not in (0, 255))
        for class_id in present_class_ids:
            binary_mask = mask == class_id
            if not np.any(binary_mask):
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_index,
                    "category_id": class_id,
                    "segmentation": _encode_binary_mask(binary_mask),
                    "area": float(binary_mask.sum()),
                    "bbox": _bbox_from_mask(binary_mask),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    return {
        "info": {
            "description": "PASCAL-5i semantic segmentation converted to COCO-style JSON",
            "split": split_name,
        },
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def run(args: argparse.Namespace) -> dict:
    paths = PascalPaths(dataset_root=args.dataset_root)
    train_output = args.train_output or os.path.join(args.dataset_root, "train.json")
    test_output = args.test_output or os.path.join(args.dataset_root, "test.json")

    train_ids = _load_split_ids(paths.train_split)
    val_ids = _load_split_ids(paths.val_split)

    train_data = _build_dataset(paths, train_ids, split_name="train")
    test_data = _build_dataset(paths, val_ids, split_name="test")

    with open(train_output, "w", encoding="utf-8") as handle:
        json.dump(train_data, handle)
    with open(test_output, "w", encoding="utf-8") as handle:
        json.dump(test_data, handle)

    return {
        "train_output": train_output,
        "train_images": len(train_data["images"]),
        "train_annotations": len(train_data["annotations"]),
        "test_output": test_output,
        "test_images": len(test_data["images"]),
        "test_annotations": len(test_data["annotations"]),
    }


def main(argv: list[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
