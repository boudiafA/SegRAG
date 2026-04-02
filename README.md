# SegRAG

Standalone v1 release of the maintained SegRAG pipeline.

This repo contains only the current supported path:
- Stage 1 raw-bank build
- Stage 1 scoring at `0.6`
- Stage 1 adaptive `q75` filtering
- Stage 2 hybrid feature-matching cache
- Stage 4 SAM3 evaluation in `text-only`, `point-only`, and `text-and-point`
- dataset adapters for `ADE20K`, `ADE20K_150`, `ADE20K_847`, `Cityscapes`, `LVIS`, `PC59`, and `agri_segmentation`

It does not include visualization code, Stage 5 mask merging, or older abandoned evaluation branches.

## Repository Layout

```text
SegRAG/
  GitHub/
    main_pipeline.py
    main_pipeline_adapters.py
    main_pipeline_ade20k.py
    ...
```

The Python package name is currently `GitHub` for compatibility with the existing commands.

## External Dependencies

This repo is standalone except for:
- a local `dinov3` checkout
- an installed `sam3` package

Recommended layout:

```text
~/RAG-SAM/
  SegRAG/
  dinov3/
  sam3/
```

The code will try to resolve DINOv3 automatically from:
- `--repo-path`
- `DINOV3_REPO_PATH`
- `./dinov3`
- `../dinov3`

It will try to resolve the DINOv3 checkpoint from:
- `--weights-path`
- `DINOV3_WEIGHTS_PATH`
- `./weights/`
- `../dinov3/weights/`

## Installation

Create an environment and install PyTorch separately first.

Then install the non-PyTorch requirements:

```bash
pip install -r requirements.txt
```

Install `sam3` from the sibling repository:

```bash
cd ../sam3
pip install -e .
```

Return to `SegRAG` before running commands:

```bash
cd ../SegRAG
```

## Main Entrypoints

Generic dataset root:

```bash
python -m GitHub.main_pipeline \
  --dataset-root /path/to/dataset \
  --segmentation-method text-and-point \
  --reference-images-per-class 30 \
  --feature-top-k 10000 \
  --feature-matching-method hybrid \
  --save-mask-json
```

Adapter-based datasets:

```bash
python -m GitHub.main_pipeline_adapters \
  --dataset-root /path/to/dataset \
  --adapter auto \
  --segmentation-method text-and-point \
  --reference-images-per-class 30 \
  --feature-top-k 10000 \
  --feature-matching-method hybrid \
  --save-mask-json
```

ADE20K variants:

```bash
python -m GitHub.main_pipeline_ade20k \
  --dataset-root /path/to/ADE20K_or_ADE20K_150 \
  --segmentation-method all \
  --reference-images-per-class 30 \
  --feature-top-k 10000 \
  --feature-matching-method hybrid \
  --save-mask-json
```

## Default Pipeline Behavior

For point-based runs, the default maintained path is:

1. Build the raw feature bank under `feature_bank_dinov3_vitl16_1536/`
2. Score all features once at threshold `0.6` with no top-k cap under `feature_bank_dinov3_vitl16_1536_scored_thr060/`
3. Filter that scored bank with `adaptive_q75` into `feature_bank_adaptive_q75_from_thr060/`
4. Build hybrid prompt caches under `_prompt_cache/`
5. Run SAM3 segmentation

## Dataset Adapters

Available adapters:
- `auto`
- `ade20k_parquet`
- `ade20k_150`
- `ade20k_847`
- `cityscapes`
- `lvis`
- `agri_segmentation`
- `pc59`

## Notes

- `text-only` uses only SAM3 text prompts
- `text-and-point` uses the adaptive-q75 filtered bank with hybrid point prompts by default
- `--raw-bank-batch-size` controls Stage 1 raw DINOv3 extraction batch size
- outputs are written under the dataset root
