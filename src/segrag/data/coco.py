from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from torchvision.transforms import v2
from tqdm import tqdm

from segrag.utils.paths import resolve_dinov3_repo_path, resolve_dinov3_weights_path


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


@dataclass
class FeatureBankPaths:
    train_root: str
    val_root: str
    image_size: int
    patch_size: int
    model_name: str
    repo_path: str
    weights_path: str
    mask_coverage_threshold: float


def resize_transform(image: Image.Image, image_size: int) -> torch.Tensor:
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((image_size, image_size), interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )
    return transform(image)


def load_dinov3_model(paths: FeatureBankPaths, device: str):
    repo_path = resolve_dinov3_repo_path(paths.repo_path)
    weights_path = resolve_dinov3_weights_path(paths.weights_path, repo_path=repo_path)
    model = torch.hub.load(
        repo_or_dir=repo_path,
        model=paths.model_name,
        source="local",
        weights=weights_path,
    )
    return model.to(device).eval()


def extract_dense_features_batched(model, img_tensors: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        feats = model.get_intermediate_layers(
            img_tensors,
            n=1,
            reshape=True,
            norm=True,
            return_class_token=False,
        )
        return feats[0].permute(0, 2, 3, 1)


def get_image_filename(img_info: dict) -> str:
    if "file_name" in img_info:
        return img_info["file_name"]
    if "coco_url" in img_info:
        return img_info["coco_url"].split("/")[-1]
    raise KeyError(f"No image filename field in {list(img_info.keys())}")


def load_lvis_annotations(ann_file: str):
    with open(ann_file, "r") as handle:
        data = json.load(handle)
    images_dict = {img["id"]: img for img in data["images"]}
    categories = {cat["id"]: cat for cat in data["categories"]}
    anns_by_image = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)
    return images_dict, dict(anns_by_image), categories


def decode_segmentation_to_mask(segmentation, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(segmentation, dict):
        if isinstance(segmentation["counts"], list):
            rle = mask_utils.frPyObjects(segmentation, height, width)
        else:
            rle = segmentation
    else:
        raise ValueError(f"Unknown segmentation format: {type(segmentation)}")
    return mask_utils.decode(rle)


def get_category_name(categories: dict, cat_id: int) -> str:
    return categories.get(cat_id, {}).get("name", f"cat_{cat_id}")


def _count_existing_features_per_class(bank_dir: str, categories: dict) -> dict:
    counts = {}
    for cat_id in categories:
        cat_name = get_category_name(categories, cat_id)
        cat_dir = os.path.join(bank_dir, cat_name)
        total = 0
        if os.path.isdir(cat_dir):
            for fname in os.listdir(cat_dir):
                if fname.endswith(".pt"):
                    try:
                        tensor = torch.load(
                            os.path.join(cat_dir, fname),
                            map_location="cpu",
                            weights_only=True,
                        )
                    except Exception:
                        continue
                    total += int(tensor.shape[0])
        counts[cat_name] = total
    return counts


def _rebuild_resume_state_from_disk(
    output_dir: str,
    anns_by_image: dict,
    categories: dict,
    scan_workers: int = 8,
):
    processed_image_ids = set()
    features_per_class = defaultdict(int)
    saved_files = 0

    def inspect_image(image_id: int):
        local_processed = True
        local_saved_files = 0
        local_features = defaultdict(int)
        for ann in anns_by_image[image_id]:
            cat_name = get_category_name(categories, ann["category_id"])
            save_path = os.path.join(output_dir, cat_name, f"{image_id}_{ann['id']}.pt")
            if not os.path.exists(save_path):
                local_processed = False
                continue
            try:
                tensor = torch.load(save_path, map_location="cpu", weights_only=True)
            except Exception:
                local_processed = False
                continue
            local_saved_files += 1
            local_features[cat_name] += int(tensor.shape[0])
        return image_id, local_processed, local_saved_files, dict(local_features)

    image_ids = sorted(anns_by_image.keys())
    if scan_workers <= 1:
        iterator = (inspect_image(image_id) for image_id in image_ids)
        for image_id, image_complete, image_saved_files, image_features in tqdm(
            iterator,
            total=len(image_ids),
            desc="Scan existing bank",
        ):
            saved_files += image_saved_files
            for cat_name, count in image_features.items():
                features_per_class[cat_name] += count
            if image_complete:
                processed_image_ids.add(image_id)
    else:
        with ThreadPoolExecutor(max_workers=scan_workers) as executor:
            futures = [executor.submit(inspect_image, image_id) for image_id in image_ids]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Scan existing bank"):
                image_id, image_complete, image_saved_files, image_features = future.result()
                saved_files += image_saved_files
                for cat_name, count in image_features.items():
                    features_per_class[cat_name] += count
                if image_complete:
                    processed_image_ids.add(image_id)

    return {
        "processed_image_ids": sorted(processed_image_ids),
        "saved_files": saved_files,
        "features_per_class": dict(features_per_class),
        "updated_at": time.time(),
    }


def _merge_resume_state(checkpoint_state: dict, scanned_state: dict) -> dict:
    merged_processed = sorted(
        set(checkpoint_state.get("processed_image_ids", []))
        | set(scanned_state.get("processed_image_ids", []))
    )

    merged_features = defaultdict(int)
    for source in (checkpoint_state.get("features_per_class", {}), scanned_state.get("features_per_class", {})):
        for cat_name, count in source.items():
            merged_features[cat_name] = max(merged_features[cat_name], int(count))

    return {
        "processed_image_ids": merged_processed,
        "saved_files": max(
            int(checkpoint_state.get("saved_files", 0)),
            int(scanned_state.get("saved_files", 0)),
        ),
        "features_per_class": dict(merged_features),
        "started_at": checkpoint_state.get("started_at", time.time()),
        "updated_at": scanned_state.get("updated_at", time.time()),
    }


def _iter_batches(items: list[int], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]
