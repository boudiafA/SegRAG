# Custom COCO-Style Dataset

SegRAG expects semantic segmentation data as COCO/LVIS-style annotations.

Minimal layout:

```text
dataset_root/
  train.json
  val.json
  images/
    image_0001.jpg
    image_0002.jpg
```

Each annotation JSON must contain top-level `images`, `annotations`, and
`categories` arrays. Each annotation should include `image_id`, `category_id`,
`segmentation`, `bbox`, and `area`.

Run:

```bash
python scripts/run_pipeline.py \
  --dataset-root /path/to/dataset_root \
  --segmentation-method text-and-point \
  --reference-images-per-class 20 \
  --resume
```

For datasets that need conversion first, use:

```bash
python scripts/run_adapters.py \
  --dataset-root /path/to/raw_dataset \
  --adapter auto \
  --segmentation-method text-and-point \
  --reference-images-per-class 20 \
  --resume
```
