"""
=============================================================================
SCRIPT 2 — HOW TO ACCESS ADE20K A-847
=============================================================================
Loads the dataset built by download_ade20k_847.py from disk.

Is it the same as A-150?
  YES — structurally identical:
    image      → PIL Image, mode=RGB
    annotation → PIL Image, mode=I  (32-bit int), pixel values 0–847
                 (0 = background/unlisted, 1–847 = A-847 class index)
    scene      → str  (scene category name, e.g. "street")
    filename   → str  (original ADE filename)

  DIFFERENCES vs A-150:
    • 847 classes instead of 150
    • Annotation mode is "I" (32-bit int) instead of "L" (8-bit uint),
      because values go up to 847 which doesn't fit in uint8
    • Slightly more pixels labelled as background (0) because ~2200 rare
      ADE20K object names are not in the A-847 vocabulary and are ignored
    • No scene_category integer field — scene is stored as a string

Install:
    pip install datasets Pillow numpy matplotlib torch torchvision
=============================================================================
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from datasets import load_from_disk


# =============================================================================
# SECTION 1 — Load the processed dataset
# =============================================================================

PROCESSED_DIR = "./ADE20K_847_processed"   # output of download_ade20k_847.py

print("=" * 70)
print("SECTION 1 — Loading processed A-847 dataset from disk")
print("=" * 70)

ds = load_from_disk(PROCESSED_DIR)
print(f"\n{ds}")

train_ds = ds["train"]
val_ds   = ds["validation"]

print(f"\n[Train]      {len(train_ds)} examples")
print(f"[Validation] {len(val_ds)} examples")
print(f"[Features]   {train_ds.features}")


# =============================================================================
# SECTION 2 — Load the A-847 class name mapping
# =============================================================================

MAPPING_PATH = os.path.join(PROCESSED_DIR, "a847_class_names.json")

with open(MAPPING_PATH, "r") as f:
    _raw = json.load(f)

# JSON keys are always strings — convert back to int
A847_IDX_TO_NAME = {int(k): v for k, v in _raw.items()}
A847_IDX_TO_NAME[0] = "background"

print("\n" + "=" * 70)
print("SECTION 2 — A-847 class names")
print("=" * 70)
print(f"\n[Total classes (incl. background)] {len(A847_IDX_TO_NAME)}")
print(f"[First 10]:")
for i in range(0, 11):
    print(f"  {i:3d}: {A847_IDX_TO_NAME.get(i, '?')}")


# =============================================================================
# SECTION 3 — Inspect a single example
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3 — Single example inspection")
print("=" * 70)

sample = train_ds[0]

image      = sample["image"]        # PIL Image RGB
annotation = sample["annotation"]   # PIL Image mode=I  (32-bit int)
scene      = sample["scene"]        # str
filename   = sample["filename"]     # str

print(f"\n[Image]      mode={image.mode}, size={image.size}")
print(f"[Annotation] mode={annotation.mode}, size={annotation.size}")
print(f"[Scene]      {scene}")
print(f"[Filename]   {filename}")

# Convert annotation to numpy
mask = np.array(annotation)   # shape (H, W), dtype int32, values 0-847
print(f"\n[Mask numpy] shape={mask.shape}, dtype={mask.dtype}")
print(f"[Unique class IDs in this image]: {np.unique(mask).tolist()}")

print("\n[Classes present]:")
for idx in np.unique(mask):
    print(f"  {idx:3d}: {A847_IDX_TO_NAME.get(idx, 'unknown')}")


# =============================================================================
# SECTION 4 — Comparison with A-150
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4 — A-847 vs A-150 quick comparison")
print("=" * 70)
print("""
  Feature            A-150                    A-847
  ─────────────────────────────────────────────────────────────────
  HuggingFace repo   zhoubolei/scene_parse150  (built from 1aurent/ADE20K)
  Load method        load_dataset()            load_from_disk()
  image              PIL RGB                   PIL RGB          <- same
  annotation         PIL mode=L  (uint8)       PIL mode=I (int32)
  annotation values  0-150                     0-847
  scene label        integer index             string
  num train images   20,210                    ~25,000
  num val images      2,000                    ~2,000
  background pixels  very few                  more  (rare classes -> 0)
""")


# =============================================================================
# SECTION 5 — Visualisation
# =============================================================================
print("=" * 70)
print("SECTION 5 — Visualisation")
print("=" * 70)

def colorize_mask(mask_array: np.ndarray, n_classes: int = 848) -> np.ndarray:
    """Map class indices to distinct RGB colours."""
    cmap    = plt.get_cmap("gist_ncar", n_classes)
    colored = cmap(mask_array % n_classes)[:, :, :3]
    return (colored * 255).astype(np.uint8)


def show_samples(dataset, indices: list, save_path: str = None):
    n   = len(indices)
    fig = plt.figure(figsize=(14, 4 * n))
    gs  = gridspec.GridSpec(n, 3, figure=fig, wspace=0.05, hspace=0.35)

    for row, idx in enumerate(indices):
        s         = dataset[idx]
        img       = s["image"]
        mask      = np.array(s["annotation"])
        scene_str = s["scene"] or "unknown"
        n_cls     = len(np.unique(mask)) - 1   # exclude background

        cls_color = colorize_mask(mask)
        overlay   = Image.blend(
            img.resize((mask.shape[1], mask.shape[0])).convert("RGB"),
            Image.fromarray(cls_color),
            alpha=0.45
        )

        for col, (im, title) in enumerate([
            (img,                        f"RGB  [{scene_str}]"),
            (Image.fromarray(cls_color), f"A-847 mask ({n_cls} classes)"),
            (overlay,                    "Overlay (45% blend)"),
        ]):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(im)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

    fig.suptitle("ADE20K A-847  — image / annotation / overlay",
                 fontsize=13, fontweight="bold")
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  Saved to {save_path}")
    else:
        plt.show()


show_samples(train_ds, indices=[0, 1, 2, 3], save_path="ade20k_847_preview.png")


# =============================================================================
# SECTION 6 — Batch iteration  (identical pattern to A-150)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 6 — Batch iteration")
print("=" * 70)

for i, batch in enumerate(train_ds.iter(batch_size=8)):
    imgs   = batch["image"]        # list of 8 PIL Images
    masks  = batch["annotation"]   # list of 8 PIL Images (mode=I)
    scenes = batch["scene"]
    print(f"  batch {i}: {len(imgs)} images | scenes={scenes}")
    if i >= 2:
        break


# =============================================================================
# SECTION 7 — Filter and map  (identical pattern to A-150)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 7 — Filter and map")
print("=" * 70)

# Filter: keep only landscape images
landscape_ds = train_ds.filter(
    lambda ex: ex["image"].width >= ex["image"].height
)
print(f"\nLandscape images: {len(landscape_ds)} / {len(train_ds)}")

# Map: resize everything to 512x512
def resize_512(example):
    example["image"]      = example["image"].resize((512, 512), Image.BILINEAR)
    example["annotation"] = example["annotation"].resize((512, 512), Image.NEAREST)
    return example

resized_ds = train_ds.map(resize_512)
print(f"Resized dataset (lazy): {resized_ds}")


# =============================================================================
# SECTION 8 — Statistics
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 8 — Statistics on first 200 training images")
print("=" * 70)

class_pixel_counts = np.zeros(848, dtype=np.int64)
widths, heights    = [], []

for i in range(200):
    s = train_ds[i]
    widths.append(s["image"].width)
    heights.append(s["image"].height)
    m = np.array(s["annotation"]).astype(np.int32)
    for cls in range(848):
        class_pixel_counts[cls] += int(np.sum(m == cls))

print(f"  Image widths  — min:{min(widths)}  max:{max(widths)}  mean:{np.mean(widths):.0f}")
print(f"  Image heights — min:{min(heights)}  max:{max(heights)}  mean:{np.mean(heights):.0f}")

top10 = np.argsort(class_pixel_counts[1:])[::-1][:10] + 1   # skip background
print(f"\n  Top 10 A-847 classes by pixel count (first 200 images):")
for rank, cls in enumerate(top10, 1):
    print(f"    {rank:2d}. [{cls:3d}] {A847_IDX_TO_NAME.get(cls,'?'):<35s} "
          f"{class_pixel_counts[cls]:,} px")


# =============================================================================
# SECTION 9 — Export images and masks to disk
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 9 — Export first 10 images to disk")
print("=" * 70)

OUT_DIR = "./ade20k_847_extracted"
os.makedirs(os.path.join(OUT_DIR, "images"),      exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "annotations"), exist_ok=True)

for i in range(min(10, len(train_ds))):
    s    = train_ds[i]
    stem = f"{i:05d}"
    s["image"].save(os.path.join(OUT_DIR, "images", f"{stem}.jpg"), quality=95)
    # Save as 16-bit PNG (values 0-847 fit in uint16)
    mask_u16 = np.array(s["annotation"]).astype(np.uint16)
    Image.fromarray(mask_u16).save(
        os.path.join(OUT_DIR, "annotations", f"{stem}.png")
    )

print(f"  Saved 10 image/mask pairs to '{OUT_DIR}'")
print(f"  Note: annotation PNGs are 16-bit (values 0-847)")


# =============================================================================
# SECTION 10 — PyTorch Dataset wrapper  (identical pattern to A-150)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 10 — PyTorch Dataset wrapper")
print("=" * 70)

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset, DataLoader
    from torchvision import transforms

    class ADE20K847Dataset(TorchDataset):
        """
        Returns:
            image_tensor : Float32 (3, H, W)  ImageNet-normalised
            mask_tensor  : Int64   (H, W)      class indices 0-847
        """
        def __init__(self, hf_split, img_size=(512, 512)):
            self.ds       = hf_split
            self.img_size = img_size
            self.img_tf   = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std =[0.229, 0.224, 0.225]),
            ])

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            s          = self.ds[idx]
            img_tensor = self.img_tf(s["image"].convert("RGB"))
            mask_resized = s["annotation"].resize(self.img_size, Image.NEAREST)
            mask_tensor  = torch.from_numpy(
                np.array(mask_resized).astype(np.int32)
            ).long()
            return img_tensor, mask_tensor

    torch_train = ADE20K847Dataset(train_ds, img_size=(512, 512))
    loader      = DataLoader(torch_train, batch_size=4, shuffle=True, num_workers=0)
    imgs, masks = next(iter(loader))

    print(f"\n  Image tensor shape : {imgs.shape}   dtype={imgs.dtype}")
    print(f"  Mask  tensor shape : {masks.shape}  dtype={masks.dtype}")
    print(f"  Unique class IDs   : {masks.unique().tolist()}")

except ImportError:
    print("  (PyTorch not installed — skipping Section 10)")


print("\n" + "=" * 70)
print("ALL DONE — ADE20K A-847 dataset fully accessible.")
print("=" * 70)
