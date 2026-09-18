# PC-59 Quick Start

Use the fixed support/query manifests for the paper's PC-59 results:

```bash
python scripts/evaluate_paper.py \
  --config configs/paper/pc59.yaml \
  --dataset-root /path/to/PC-59 \
  --shot 5 \
  --output-root /path/to/new/paper_outputs \
  --dinov3-repo /path/to/dinov3 \
  --dinov3-weights /path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --resume \
  --save-mask-json
```

Expected files under `dataset-root`:

```text
train.json
val.json
reference_imgs/
JPEGImages/
```

First add `--validate-only` to verify the files without loading models. Use
`--shot 1` for the one-shot result. Banks, prompts, masks, and resume metadata
stay under the new output root, not in the dataset directory.

See [the full reproduction guide](../docs/PAPER_REPRODUCTION.md). The generic
runner in [the custom dataset example](custom_coco_dataset.md) uses first-N
support selection and is not a substitute for this benchmark protocol.
