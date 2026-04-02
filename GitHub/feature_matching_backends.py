from __future__ import annotations

import gc
import os
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from GitHub.feature_cache import FeatureCache, build_feature_cache_dir
from GitHub.runtime_paths import resolve_dinov3_repo_path, resolve_dinov3_weights_path

try:
    from lvis import LVIS
except ImportError as exc:  # pragma: no cover
    raise ImportError("lvis is required. Install with `pip install lvis`.") from exc

try:
    from scipy.ndimage import maximum_filter
    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SCIPY_AVAILABLE = False


MAX_REFERENCES = 10000
SUPPRESSION_MARGIN = 0.0
DATALOADER_NUM_WORKERS = 4
PREFETCH_QUEUE_SIZE = 8
IMAGE_SIZE = 1536
MODEL_NAME = "dinov3_vitl16"
PATCH_SIZE = 16
DINOV3_REPO_PATH = "./"
WEIGHTS_PATH = "./weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
PEAK_THRESHOLD = 0.05
MIN_PEAK_DISTANCE = 3
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _a_or_an(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def format_prompt(class_name: str) -> str:
    pretty = class_name.replace("_", " ")
    return f"{_a_or_an(pretty)} {pretty}"


def _build_image_transform(image_size: int = IMAGE_SIZE) -> v2.Compose:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((image_size, image_size), interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
        ]
    )


_IMAGE_TRANSFORM = _build_image_transform()


def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    if tensor.is_floating_point() and tensor.dtype != torch.float32:
        tensor = tensor.float()
    return tensor.detach().cpu().numpy()


def align_mask(pred_mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if pred_mask.shape == target_shape:
        return pred_mask
    try:
        import cv2

        return cv2.resize(
            pred_mask.astype(np.uint8),
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    except ImportError:
        tensor = torch.as_tensor(pred_mask.astype(np.float32))[None, None]
        resized = F.interpolate(tensor, size=target_shape, mode="nearest")
        return resized[0, 0].numpy().astype(np.uint8)


class LVISImageRecord:
    __slots__ = ("img_id", "img_info", "file_name", "jpg_path", "anns", "valid_cats", "pil_image")

    def __init__(self, img_id, img_info, file_name, jpg_path, anns, valid_cats):
        self.img_id = img_id
        self.img_info = img_info
        self.file_name = file_name
        self.jpg_path = jpg_path
        self.anns = anns
        self.valid_cats = valid_cats
        self.pil_image = None


class LVISPrefetchDataset(Dataset):
    def __init__(self, records: list):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        rec.pil_image = Image.open(rec.jpg_path).convert("RGB")
        return rec


def collate_records(batch):
    return batch


class DINOv3FeatureExtractor:
    def __init__(
        self,
        device: str | None = None,
        cache_dir: str | None = None,
        annotation_file: str = "test.json",
        image_size: int = IMAGE_SIZE,
        model_name: str = MODEL_NAME,
        repo_path: str = DINOV3_REPO_PATH,
        weights_path: str = WEIGHTS_PATH,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache = FeatureCache(
            cache_dir
            or build_feature_cache_dir(
                annotation_file=annotation_file,
                image_size=image_size,
                model_name=model_name,
            )
        )
        print(f"Loading DINOv3 model ({model_name}) onto {self.device} ...")
        repo_path = resolve_dinov3_repo_path(repo_path)
        weights_path = resolve_dinov3_weights_path(weights_path, repo_path=repo_path)
        self.model = torch.hub.load(
            repo_or_dir=repo_path,
            model=model_name,
            source="local",
            weights=weights_path,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def extract(self, image: Image.Image, cache_key: str | None = None) -> torch.Tensor:
        if cache_key is not None:
            cached = self.cache.load(cache_key)
            if cached is not None:
                return cached
        tensor = _IMAGE_TRANSFORM(image).unsqueeze(0).to(self.device)
        feats = self.model.get_intermediate_layers(
            tensor, n=1, reshape=True, return_class_token=False, norm=True
        )
        result = feats[0][0].permute(1, 2, 0).float().cpu()
        del tensor, feats
        if cache_key is not None:
            self.cache.save(cache_key, result)
        return result


def _compute_best_sim(query_flat: torch.Tensor, bank_dev: torch.Tensor) -> torch.Tensor:
    n_patches, n_bank = query_flat.shape[0], bank_dev.shape[0]
    chunk_floats = 512 * 1024 * 1024
    if n_patches * n_bank > chunk_floats:
        chunk_size = max(1, chunk_floats // n_patches)
        best_sim = torch.full((n_patches,), -1.0, device=query_flat.device, dtype=query_flat.dtype)
        for start in range(0, n_bank, chunk_size):
            end = min(start + chunk_size, n_bank)
            chunk_max, _ = (query_flat @ bank_dev[start:end].T).max(dim=1)
            best_sim = torch.maximum(best_sim, chunk_max)
    else:
        best_sim, _ = (query_flat @ bank_dev.T).max(dim=1)
    return best_sim


def find_peaks(heatmap_2d: np.ndarray, min_distance: int = 3, abs_threshold: float = 0.05) -> list[tuple[int, int, float]]:
    if _SCIPY_AVAILABLE:
        size = 2 * min_distance + 1
        local_max = maximum_filter(heatmap_2d, size=size)
        is_peak = (heatmap_2d == local_max) & (heatmap_2d > abs_threshold)
        rows, cols = np.where(is_peak)
        scores = heatmap_2d[rows, cols]
    else:
        rows, cols = np.where(heatmap_2d > abs_threshold)
        scores = heatmap_2d[rows, cols]
    order = np.argsort(-scores)
    return [(int(rows[i]), int(cols[i]), float(scores[i])) for i in order]


class AbsoluteFeatureBankMatcher:
    def __init__(self, feature_bank_dir: str, device: str, max_references: int | None = None):
        self.feature_bank_dir = feature_bank_dir
        self.device = device
        self.max_references = max_references
        self._cache: dict[str, torch.Tensor] = {}
        self._missing_cats: set[str] = set()
        if not os.path.isdir(feature_bank_dir):
            raise FileNotFoundError(f"Feature bank not found: {feature_bank_dir}")

    def _load_bank(self, cat_name: str) -> torch.Tensor | None:
        if cat_name in self._cache:
            return self._cache[cat_name]
        if cat_name in self._missing_cats:
            return None
        cat_dir = os.path.join(self.feature_bank_dir, cat_name)
        if not os.path.isdir(cat_dir):
            self._missing_cats.add(cat_name)
            return None
        files = sorted(f for f in os.listdir(cat_dir) if f.endswith(".pt"))
        if not files:
            self._missing_cats.add(cat_name)
            return None
        all_feats = []
        total = 0
        for fname in files:
            feats = torch.load(os.path.join(cat_dir, fname), map_location="cpu", weights_only=True)
            if self.max_references is not None:
                remaining = self.max_references - total
                if remaining <= 0:
                    break
                if feats.shape[0] > remaining:
                    feats = feats[:remaining]
            all_feats.append(feats)
            total += feats.shape[0]
        if not all_feats:
            self._missing_cats.add(cat_name)
            return None
        bank = torch.cat(all_feats, dim=0).to(self.device).float()
        bank = F.normalize(bank, dim=-1)
        self._cache[cat_name] = bank
        return bank

    def match_points(
        self,
        query_features: torch.Tensor,
        cat_name: str,
        num_points: int,
        original_size: tuple[int, int],
        similarity_threshold: float = 0.8,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        bank = self._load_bank(cat_name)
        if bank is None:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float32),
            )
        grid_h, grid_w, dim = query_features.shape
        orig_h, orig_w = original_size
        query_flat = F.normalize(query_features.reshape(-1, dim).float(), dim=-1).to(self.device)
        best_sim = _compute_best_sim(query_flat, bank).float()
        above_thresh = best_sim > similarity_threshold
        if not above_thresh.any():
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float32),
            )
        valid_indices = torch.where(above_thresh)[0]
        valid_sims = best_sim[valid_indices]
        sorted_order = torch.argsort(valid_sims, descending=True)
        n_take = min(num_points, len(sorted_order))
        top_indices = valid_indices[sorted_order[:n_take]]
        top_sims = valid_sims[sorted_order[:n_take]]
        rows = (top_indices // grid_w).cpu().numpy()
        cols = (top_indices % grid_w).cpu().numpy()
        patch_h = orig_h / grid_h
        patch_w = orig_w / grid_w
        coords = np.stack([(cols + 0.5) * patch_w, (rows + 0.5) * patch_h], axis=1).astype(np.float32)
        labels = np.ones(len(coords), dtype=np.int32)
        return coords, labels, top_sims.cpu().numpy().astype(np.float32)


class RelativeFeatureBankMatcher:
    def __init__(
        self,
        feature_bank_dir: str,
        device: str,
        max_references: int | None = None,
        suppression_bank: dict | None = None,
    ):
        self.feature_bank_dir = feature_bank_dir
        self.device = device
        self.max_references = max_references
        self.suppression_bank = suppression_bank
        self._cache: dict[str, torch.Tensor] = {}
        self._missing_cats: set[str] = set()
        if not os.path.isdir(feature_bank_dir):
            raise FileNotFoundError(f"Feature bank not found: {feature_bank_dir}")

    def _load_bank(self, cat_name: str) -> torch.Tensor | None:
        if cat_name in self._cache:
            return self._cache[cat_name]
        if cat_name in self._missing_cats:
            return None
        cat_dir = os.path.join(self.feature_bank_dir, cat_name)
        if not os.path.isdir(cat_dir):
            self._missing_cats.add(cat_name)
            return None
        proto_path = os.path.join(cat_dir, "prototypes.pt")
        if os.path.exists(proto_path):
            feats = torch.load(proto_path, map_location="cpu", weights_only=True).float()
            if self.max_references and feats.shape[0] > self.max_references:
                feats = feats[:self.max_references]
            self._cache[cat_name] = F.normalize(feats, dim=-1)
            return self._cache[cat_name]
        files = sorted(f for f in os.listdir(cat_dir) if f.endswith(".pt"))
        if not files:
            self._missing_cats.add(cat_name)
            return None
        all_feats = []
        total = 0
        for fname in files:
            feats = torch.load(os.path.join(cat_dir, fname), map_location="cpu", weights_only=True)
            if self.max_references:
                remaining = self.max_references - total
                if remaining <= 0:
                    break
                if feats.shape[0] > remaining:
                    feats = feats[:remaining]
            all_feats.append(feats)
            total += feats.shape[0]
        if not all_feats:
            self._missing_cats.add(cat_name)
            return None
        bank = F.normalize(torch.cat(all_feats, dim=0).float(), dim=-1)
        self._cache[cat_name] = bank
        return bank

    def match_points(
        self,
        query_features: torch.Tensor,
        cat_name: str,
        original_size: tuple[int, int],
        peak_threshold: float = PEAK_THRESHOLD,
        min_peak_distance: int = MIN_PEAK_DISTANCE,
        suppression_margin: float = SUPPRESSION_MARGIN,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        empty = (
            np.empty((0, 2), np.float32),
            np.empty(0, np.int32),
            np.empty(0, np.float32),
        )
        bank = self._load_bank(cat_name)
        if bank is None:
            return empty
        grid_h, grid_w, dim = query_features.shape
        orig_h, orig_w = original_size
        query_flat = F.normalize(query_features.reshape(-1, dim).float(), dim=-1).to(self.device)
        bank_dev = bank.to(self.device, non_blocking=True)
        target_sim = _compute_best_sim(query_flat, bank_dev)
        del bank_dev
        confidence = target_sim.clone()
        if self.suppression_bank is not None:
            other_parts = [v for k, v in self.suppression_bank.items() if k != cat_name]
            if other_parts:
                other_bank = torch.cat(other_parts, dim=0).float().to(self.device, non_blocking=True)
                other_sim = _compute_best_sim(query_flat, other_bank)
                confidence = target_sim - other_sim - suppression_margin
                del other_bank, other_sim
        del query_flat
        conf_2d = _tensor_to_numpy(confidence).reshape(grid_h, grid_w)
        peaks = find_peaks(conf_2d, min_distance=min_peak_distance, abs_threshold=peak_threshold)
        if not peaks:
            return empty
        rows = np.array([p[0] for p in peaks])
        cols = np.array([p[1] for p in peaks])
        scores = np.array([p[2] for p in peaks], dtype=np.float32)
        patch_h = orig_h / grid_h
        patch_w = orig_w / grid_w
        coords = np.stack([(cols + 0.5) * patch_w, (rows + 0.5) * patch_h], axis=1).astype(np.float32)
        labels = np.ones(len(coords), dtype=np.int32)
        del target_sim, confidence
        return coords, labels, scores


def _discover_bank_classes(feature_bank_dir: str) -> list[str]:
    if not os.path.isdir(feature_bank_dir):
        raise FileNotFoundError(f"Feature bank directory not found: {feature_bank_dir}")
    bank_classes = []
    for entry in sorted(os.listdir(feature_bank_dir)):
        cat_dir = os.path.join(feature_bank_dir, entry)
        if not os.path.isdir(cat_dir):
            continue
        try:
            has_pt = any(name.endswith(".pt") for name in os.listdir(cat_dir))
        except OSError:
            continue
        if has_pt:
            bank_classes.append(entry)
    return bank_classes


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    try:
        from scipy.ndimage import label as scipy_label

        structure = np.ones((3, 3), dtype=np.uint8)
        return scipy_label(mask.astype(np.uint8), structure=structure)
    except ImportError:
        labels = np.zeros(mask.shape, dtype=np.int32)
        current = 0
        rows, cols = mask.shape
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
        for r in range(rows):
            for c in range(cols):
                if not mask[r, c] or labels[r, c] != 0:
                    continue
                current += 1
                stack = [(r, c)]
                labels[r, c] = current
                while stack:
                    cr, cc = stack.pop()
                    for dr, dc in neighbors:
                        nr, nc = cr + dr, cc + dc
                        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                            continue
                        if not mask[nr, nc] or labels[nr, nc] != 0:
                            continue
                        labels[nr, nc] = current
                        stack.append((nr, nc))
        return labels, current


def _nms_points(candidates: list[tuple[int, int, float]], min_distance: int) -> list[tuple[int, int, float]]:
    kept: list[tuple[int, int, float]] = []
    min_dist_sq = float(min_distance * min_distance)
    for row, col, score in candidates:
        too_close = False
        for kept_row, kept_col, _ in kept:
            if (row - kept_row) ** 2 + (col - kept_col) ** 2 < min_dist_sq:
                too_close = True
                break
        if not too_close:
            kept.append((row, col, score))
    return kept


def _extract_component_peaks(
    sim_map: np.ndarray,
    component_mask: np.ndarray,
    min_peak_distance: int,
) -> list[tuple[int, int, float]]:
    masked_scores = np.where(component_mask, sim_map, -np.inf)
    peaks = find_peaks(masked_scores, min_distance=min_peak_distance, abs_threshold=-np.inf)
    candidates = [(r, c, score) for r, c, score in peaks if component_mask[r, c] and np.isfinite(score)]
    candidates.sort(key=lambda item: item[2], reverse=True)
    kept = _nms_points(candidates, min_peak_distance)
    if kept:
        return kept
    flat_idx = int(np.argmax(masked_scores))
    row, col = np.unravel_index(flat_idx, sim_map.shape)
    if component_mask[row, col]:
        return [(int(row), int(col), float(sim_map[row, col]))]
    return []


class DualThresholdPromptGenerator:
    def __init__(self, feature_bank_dir: str, device: str, max_references: int | None = None):
        self.feature_bank_dir = feature_bank_dir
        self.device = device
        self.max_references = max_references
        self._cache: dict[str, torch.Tensor] = {}
        self._missing_cats: set[str] = set()
        bank_classes = _discover_bank_classes(feature_bank_dir)
        if not bank_classes:
            raise ValueError(
                f"No class feature-bank subdirectories with .pt files were found in: {feature_bank_dir}"
            )

    def _load_bank(self, cat_name: str) -> torch.Tensor | None:
        if cat_name in self._cache:
            return self._cache[cat_name]
        if cat_name in self._missing_cats:
            return None
        cat_dir = os.path.join(self.feature_bank_dir, cat_name)
        if not os.path.isdir(cat_dir):
            self._missing_cats.add(cat_name)
            return None
        proto_path = os.path.join(cat_dir, "prototypes.pt")
        if os.path.exists(proto_path):
            feats = torch.load(proto_path, map_location="cpu", weights_only=True).float()
            if self.max_references is not None and feats.shape[0] > self.max_references:
                feats = feats[:self.max_references]
            self._cache[cat_name] = F.normalize(feats, dim=-1)
            return self._cache[cat_name]
        all_feats = []
        total = 0
        for fname in sorted(f for f in os.listdir(cat_dir) if f.endswith(".pt")):
            feats = torch.load(os.path.join(cat_dir, fname), map_location="cpu", weights_only=True).float()
            if self.max_references is not None:
                remaining = self.max_references - total
                if remaining <= 0:
                    break
                if feats.shape[0] > remaining:
                    feats = feats[:remaining]
            all_feats.append(feats)
            total += feats.shape[0]
        if not all_feats:
            self._missing_cats.add(cat_name)
            return None
        bank = F.normalize(torch.cat(all_feats, dim=0), dim=-1)
        self._cache[cat_name] = bank
        return bank

    def compute_similarity_map(
        self,
        query_features: torch.Tensor,
        bank: torch.Tensor | None = None,
        cat_name: str | None = None,
    ) -> np.ndarray | None:
        if bank is None:
            if cat_name is None:
                raise ValueError("Either bank or cat_name must be provided.")
            bank = self._load_bank(cat_name)
        if bank is None:
            return None
        grid_h, grid_w, dim = query_features.shape
        query_flat = F.normalize(query_features.reshape(-1, dim).float(), dim=-1).to(self.device)
        bank_dev = bank.to(self.device, non_blocking=True)
        best_sim = _compute_best_sim(query_flat, bank_dev)
        sim_map = _tensor_to_numpy(best_sim).reshape(grid_h, grid_w)
        del query_flat, bank_dev, best_sim
        return sim_map

    def generate_prompts_from_sim_map(
        self,
        sim_map: np.ndarray,
        original_size: tuple[int, int],
        loose_threshold: float,
        min_peak_distance: int,
        min_component_size: int,
    ) -> list[dict]:
        grid_h, grid_w = sim_map.shape
        orig_h, orig_w = original_size
        loose_mask = sim_map >= loose_threshold
        labels, num_labels = _label_components(loose_mask)
        prompts = []
        patch_h = orig_h / grid_h
        patch_w = orig_w / grid_w
        for component_id in range(1, num_labels + 1):
            component_mask = labels == component_id
            component_size = int(component_mask.sum())
            if component_size < min_component_size:
                continue
            peaks = _extract_component_peaks(
                sim_map=sim_map,
                component_mask=component_mask,
                min_peak_distance=min_peak_distance,
            )
            for row, col, score in peaks:
                prompts.append(
                    {
                        "row": row,
                        "col": col,
                        "x": float((col + 0.5) * patch_w),
                        "y": float((row + 0.5) * patch_h),
                        "score": float(score),
                    }
                )
        prompts.sort(key=lambda item: item["score"], reverse=True)
        return prompts

    def generate_prompts(
        self,
        query_features: torch.Tensor,
        cat_name: str,
        original_size: tuple[int, int],
        loose_threshold: float,
        min_peak_distance: int,
        min_component_size: int,
    ) -> tuple[list[dict], np.ndarray] | tuple[None, None]:
        bank = self._load_bank(cat_name)
        if bank is None:
            return None, None
        sim_map = self.compute_similarity_map(query_features, bank, cat_name)
        prompts = self.generate_prompts_from_sim_map(
            sim_map=sim_map,
            original_size=original_size,
            loose_threshold=loose_threshold,
            min_peak_distance=min_peak_distance,
            min_component_size=min_component_size,
        )
        return prompts, sim_map

    def clear_cache(self):
        self._cache.clear()
        self._missing_cats.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
