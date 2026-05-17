# PC-59 Quick Start

This example runs the full SegRAG text+point pipeline on a COCO-style PC-59
export with a fixed number of reference images per class.

```bash
python scripts/run_pipeline.py \
  --dataset-root /path/to/pc59_20shot \
  --segmentation-method text-and-point \
  --reference-images-per-class 20 \
  --feature-matching-method hybrid \
  --resume \
  --save-mask-json
```

Expected files under `dataset-root`:

```text
train.json
val.json
images/
```

Generated outputs stay inside the dataset root unless explicit output paths are
provided. Feature banks and prompt caches are resumable.
