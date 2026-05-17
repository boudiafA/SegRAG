# SegRAG

This repository contains the public SegRAG workflow for retrieval-augmented
semantic segmentation with DINOv3 features and SAM3 mask decoding.

Design goals:
- one stage-oriented public script per workflow step
- shared utilities for paths, method names, caching, and resume state
- no repeated argument routing across scripts
- a simple main runner that can execute the full pipeline end to end

Primary entrypoint:
- `python -m GitHub.main_pipeline ...`
- `python -m GitHub.main_pipeline_adapters ...` for adapter-based dataset exports
- `python -m GitHub.main_pipeline_ade20k ...` for ADE20K dataset variants that need export first

Repository layout expected by this workflow:
- `SegRAG/` for this repository
- `dinov3/` for the local DINOv3 repository
- `sam3/` for the local SAM3 repository

The code imports this repository's `GitHub` Python package, a local DINOv3
checkout, and an installed local SAM3 package. The default path resolver checks
the current working directory, `./dinov3`, and `../dinov3`; you can also set
`DINOV3_REPO_PATH` and `DINOV3_WEIGHTS_PATH`.

## Named Method Steps

This workflow contains two named method steps that are useful to reference in
papers, reports, and future code changes.

### Intra-Class Cohesion Distillation (ICCD)

ICCD is the feature-bank quality filtering step used during Stage 1. It happens
after raw DINOv3 feature extraction and before the final filtered bank is reused
by inference.

Pipeline location:
- `GitHub.main_pipeline._run_stage1`
- `GitHub.Stage1_score_feature_bank`
- `GitHub.Stage1_filter_scored_feature_bank`
- `GitHub.intra_class_filter_impl`

Implementation flow:
- build the raw bank under `feature_bank_dinov3_vitl16_1536/`
- score each candidate feature by matching it against same-class target masks
- save a reusable scored bank under `feature_bank_dinov3_vitl16_1536_scored_thr060/`
- filter the scored bank with `adaptive_q75` into `feature_bank_adaptive_q75_from_thr060/`

The core scoring function is `score_class` in
`GitHub/intra_class_filter_impl.py`. It tracks same-class matches that land
inside target masks as positive evidence and matches outside the target mask as
negative evidence. The default filtering rule is implemented by
`_apply_adaptive_top_k_features`, which computes the class-level 75th percentile
score and derives an adaptive keep threshold before applying the optional
top-k cap.

### Topographic Similarity Grounding (TSG)

TSG is the inference-time feature-matching refinement step. It happens after
dense DINOv3 feature matching on a query image and before the selected matches
are passed to SAM3 as point prompts.

Pipeline location:
- `GitHub.main_pipeline._run_stage2`
- `GitHub.Stage2_cache_feature_matching`
- `GitHub.cache_feature_matching`
- `GitHub.feature_matching_backends`
- `GitHub.sam3_points_only_impl`
- `GitHub.sam3_text_and_points_impl`

Implementation flow:
- load the ICCD-filtered feature bank
- compute class-specific similarity maps for query images
- keep only high-quality spatial matches using thresholding, connected
  components, peak extraction, and point suppression/NMS
- cache the final prompts under `_prompt_cache/`
- reuse the cached prompts during Stage 4 SAM3 evaluation

The strongest TSG path is the `hybrid` feature-matching method implemented by
`DualThresholdPromptGenerator` in `GitHub/feature_matching_backends.py`. It
turns dense similarity maps into spatial components, removes small components,
extracts local peaks, and sorts accepted prompts by score. During Stage 4,
`_load_cached_prompts` applies the final `hybrid_validation_threshold` before
the prompts are sent to SAM3.

## Expected Data Layout

The preferred way to use this package is to give `GitHub.main_pipeline` a single dataset root.

Supported dataset layouts:

### Layout A: flat root

```text
dataset_root/
  train.json
  test.json
  images/
    ...
```

or

```text
dataset_root/
  train.json
  val.json
  ...
```

### Layout B: split train/val subfolders

```text
dataset_root/
  train/
    lvis_v1_train.json
    images/
      ...
  val/
    lvis_v1_val.json
    images/
      ...
```

This second layout is common for LVIS and is supported directly.

### Layout C: ADE20K adapters

```text
dataset_root/
  either:
    data/
      train-00000-of-xxxxx.parquet
      ...
      validation-00000-of-xxxxx.parquet

  or:
    downloads/
      extracted/
        .../ADEChallengeData2016/
          images/training/
          images/validation/
          annotations/training/
          annotations/validation/
```

For this layout, use `GitHub.main_pipeline_ade20k`. It exports a local
COCO-style view under the same dataset root before running the regular stages.

For `zhoubolei/scene_parse_150`, the adapter uses the extracted
`ADEChallengeData2016` images and label masks directly, exports semantic
annotations, and ignores class `0` (`background`).

Outputs are written under the same dataset root:

```text
dataset_root/
  train.json
  val.json
  feature_bank_dinov3_vitl16_1536/
  feature_bank_dinov3_vitl16_1536_scored_thr060/
  feature_bank_adaptive_q75_from_thr060/
  _prompt_cache/
  evaluation_results_...
```

## Expected Annotation Format

Training and validation annotation files should be COCO/LVIS-style JSONs with:
- top-level `images`
- top-level `annotations`
- top-level `categories`

Expected `images` fields:
- `id`
- `width`
- `height`
- either `file_name` or `coco_url`

Expected `annotations` fields:
- `id`
- `image_id`
- `category_id`
- `segmentation`
- `bbox`
- `area`

Notes:
- `segmentation` can be polygons or COCO RLE
- if `file_name` is missing, the code falls back to the basename of `coco_url`
- Stage 1 uses the training annotation file
- Stage 2 and Stage 4 use the validation annotation file

## Installation

### 1. Clone the two repositories

```bash
mkdir -p ~/RAG-SAM
cd ~/RAG-SAM

git clone https://github.com/boudiafA/SegRAG.git SegRAG
git clone https://github.com/facebookresearch/dinov3.git dinov3
git clone https://github.com/facebookresearch/sam3.git sam3
```

If you already created the repositories in these exact paths, keep that layout.

### 2. Create and activate a Python environment

Python 3.11 is the safest choice for this workflow.

```bash
conda create -n rag-sam python=3.11 -y
conda activate rag-sam
```

### 3. Install PyTorch first

Install a PyTorch build that matches your CUDA version from the official PyTorch instructions.

Example for CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

If you are using CPU only, install the CPU build from the PyTorch website instead.

### 4. Install SegRAG and DINOv3-side dependencies

From inside the SegRAG repository:

```bash
cd ~/RAG-SAM/SegRAG
pip install -r requirements.txt
```

These extra packages are needed by the SegRAG workflow and the SAM3 integration.

If you want to use the ADE20K adapter with parquet shards, also install:

```bash
pip install pyarrow
```

### 5. Install SAM3 from the sibling repository

```bash
cd ~/RAG-SAM/sam3
pip install -e .
```

This is required because the Stage 4 code imports `sam3.model_builder` and `sam3.model.sam3_image_processor`.

### 6. Return to the SegRAG repository before running the pipeline

```bash
cd ~/RAG-SAM/SegRAG
```

Run the stage scripts from the SegRAG repository root so local imports like
`GitHub.*`, the sibling DINOv3 checkout, and the installed `sam3` package
resolve correctly.

## Checkpoints and weights

### DINOv3 checkpoint

Stage 1 expects the DINOv3 ViT-L/16 checkpoint at:

```text
~/RAG-SAM/dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

If you place it elsewhere, pass a custom `--weights-path` or set
`DINOV3_WEIGHTS_PATH`.

### SAM3 checkpoint

The SAM3 image model is built through the local `sam3` package. By default, SAM3 can download its image-model checkpoint from Hugging Face automatically on first use. If you prefer a fully offline setup, install the checkpoint separately and modify the SAM3-side loading path accordingly.

## Minimal setup check

After installation, these should succeed from `~/RAG-SAM/SegRAG`:

```bash
python -c "import torch; print(torch.__version__)"
python -c "import sam3; print('sam3 import ok')"
python -c "from GitHub.Stage4_evaluate_sam3 import PROMPT_MODES; print(PROMPT_MODES)"
python -c "from GitHub.main_pipeline import SEGMENTATION_METHODS; print(SEGMENTATION_METHODS)"
```

## Quick Start

The recommended entrypoint is `GitHub.main_pipeline`.

By default, `main_pipeline` now uses this Stage 1 flow for point-based runs:
- build the raw bank under `feature_bank_dinov3_vitl16_1536/`
- score all features once with threshold `0.6` and no top-k cap under `feature_bank_dinov3_vitl16_1536_scored_thr060/`
- filter that scored bank with `adaptive_q75` into `feature_bank_adaptive_q75_from_thr060/`

This adaptive-q75 filtered bank is the default bank used by Stage 2, Stage 3, and Stage 4 when no explicit filtered-bank path is provided.

The examples below assume:
- you are in `~/RAG-SAM/SegRAG`
- the DINOv3 checkpoint exists at `../dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`
- your dataset root follows one of the supported layouts above

### Optional: Stage 0 for PASCAL-5i / VOC

If your dataset is still in VOC mask format, create `train.json` and `test.json` first:

```bash
python -m GitHub.Stage0_prepare_pascal5i_annotations \
  --dataset-root /path/to/VOCdevkit/VOC2012
```

### Main pipeline: text-only segmentation

```bash
python -m GitHub.main_pipeline \
  --dataset-root /path/to/dataset \
  --segmentation-method text-only \
  --save-mask-json \
  --resume
```

### Main pipeline: point-only segmentation

```bash
python -m GitHub.main_pipeline \
  --dataset-root /path/to/dataset \
  --segmentation-method point-only \
  --reference-images-per-class 30 \
  --feature-top-k 10000 \
  --feature-matching-method hybrid \
  --save-mask-json \
  --resume
```

### Main pipeline: text-and-point segmentation

```bash
python -m GitHub.main_pipeline \
  --dataset-root /path/to/dataset \
  --segmentation-method text-and-point \
  --reference-images-per-class 30 \
  --raw-bank-batch-size 4 \
  --feature-top-k 10000 \
  --feature-matching-method hybrid \
  --save-mask-json \
  --resume
```

### Main pipeline: run all three segmentation modes

```bash
python -m GitHub.main_pipeline \
  --dataset-root /path/to/dataset \
  --segmentation-method all \
  --reference-images-per-class 30 \
  --feature-top-k 10000 \
  --feature-matching-method hybrid \
  --save-mask-json \
  --resume
```

Behavior:
- `text-only` runs only Stage 4
- `point-only` runs Stage 1, Stage 2, and Stage 4
- `text-and-point` runs Stage 1, Stage 2, and Stage 4
- `all` runs Stage 1 and Stage 2 once, then runs all three Stage 4 segmentation modes
- `--raw-bank-batch-size` controls the Stage 1 DINOv3 extraction batch size and defaults to `4`

All outputs are written under the same `dataset_root` passed to `main_pipeline.py`.

### ADE20K: export and run

Use `GitHub.main_pipeline_ade20k` when the dataset root is either:
- the HuggingFace parquet ADE20K download
- the `zhoubolei/scene_parse_150` / ADE20K-150 cache with extracted `ADEChallengeData2016`

This entrypoint:
- auto-detects the ADE20K variant by default
- exports `train.json` and `val.json`
- for ADE20K-150, converts each semantic mask into one annotation per foreground class and ignores background `0`
- then runs the same stage sequence used by the standard entrypoint

Example:

```bash
python -m GitHub.main_pipeline_ade20k \
  --dataset-root /path/to/ADE20K_or_ADE20K_150 \
  --segmentation-method all \
  --reference-images-per-class 30 \
  --feature-top-k 10000 \
  --feature-matching-method hybrid \
  --save-mask-json \
  --resume
```

Useful ADE20K-specific options:
- `--dataset-format auto|parquet|ade20k_150`
- `--prepare-only` exports `train.json`, `val.json`, and the images but skips segmentation
- `--force-rebuild-export` rebuilds the exported JSON/image view even if it already exists

Assumptions used by the ADE20K adapter:
- parquet ADE20K derives categories from object names and exports polygon segmentations
- ADE20K-150 uses the official 150 semantic labels and skips `background`
- all exported files, feature banks, caches, and evaluation outputs stay under the same `dataset_root`

### Global Adapter Pipeline

`GitHub.main_pipeline_adapters` is the generic entrypoint for datasets that
need an export/normalization adapter before running Stage 1/2/4.

Current adapters:
- `ade20k_parquet`
- `ade20k_150`
- `ade20k_847`
- `cityscapes`
- `lvis`
- `agri_segmentation`
- `pc59`

Example:

```bash
python -m GitHub.main_pipeline_adapters \
  --dataset-root /path/to/dataset \
  --adapter auto \
  --segmentation-method all \
  --reference-images-per-class 30 \
  --feature-top-k 10000 \
  --feature-matching-method hybrid \
  --save-mask-json \
  --resume
```

For ADE20K-847:
- the adapter uses the raw HuggingFace Arrow cache under `hf_cache/`
- it builds one semantic mask from the per-object instance masks
- it ignores class `0`
- it saves exported images under `train/images/` and `val/images/`
- if `objects_847.txt` is missing, it will try to download it once into the dataset root

For Cityscapes:
- the adapter reads `leftImg8bit` and `gtFine` directly
- it remaps the standard semantic label ids to the 19 train classes
- it exports `train.json` and `val.json` under the same dataset root

For LVIS:
- the adapter is a thin wrapper around an existing LVIS-style layout
- it reuses `train/lvis_v1_train.json` and `val/lvis_v1_val.json` when present
- it does not rewrite images or annotations

For `agri_segmentation`:
- the adapter is a thin wrapper around an existing COCO-style dataset root
- it expects `train.json` and `test.json` or `val.json`
- it does not rewrite images or annotations

For PC59:
- the adapter reads `JPEGImages/`, `subset59_from_459/`, and the predefined train/val split text files
- it exports `train.json` and `val.json`
- it ignores background class `0`

### Stage-oriented entrypoints

If you want full manual control, the stage-oriented scripts are still available:
- `Stage1_build_and_filter_feature_bank.py`
- `Stage1_score_feature_bank.py`
- `Stage1_filter_scored_feature_bank.py`
- `Stage2_cache_feature_matching.py`
- `Stage3_evaluate_feature_matching.py`
- `Stage4_evaluate_sam3.py`
- `Stage5_evaluate_merged_sam3_masks.py`

## Outputs by Stage

### Stage 0

Creates COCO-style annotation files:
- `train.json`
- `test.json`

### Stage 1

Creates three bank directories by default:
- `feature_bank_dinov3_vitl16_1536/`
- `feature_bank_dinov3_vitl16_1536_scored_thr060/`
- `feature_bank_adaptive_q75_from_thr060/`

The scored bank stores reusable sidecars for later filtering experiments:
- `*.scores.npy`
- `*.tp.npy`
- `*.fp.npy`

The adaptive filtered bank is the default bank reused by Stages 2 to 4.

Also writes resume/report files inside those directories, including:
- `_build_feature_bank_resume.json`
- `intra_class_filter_report.json`

### Stage 2

Creates prompt-cache directories under the dataset root for the selected feature-matching methods.

Also writes temporary resume state under:
- `_github_prompt_cache_state/`

The cached prompt data is later reused by Stage 3 and Stage 4.

### Stage 3

Writes feature-matching evaluation results as JSON if `--output-json` is provided.

When resume mode is used, temporary checkpoints are written under:
- `_github_non_sam_eval/`

### Stage 4

Creates SAM3 evaluation result folders under the dataset root.

Typical outputs include:
- `evaluation_results_text_only_sam3/`
- `evaluation_results_points_only_sam3_<method>/`
- `evaluation_results_text_and_points_sam3_<method>/`

Inside each result folder, the main outputs are:
- `results.json`
- `predicted_masks.json` when `--save-mask-json` is enabled

### Stage 5

Creates merged-mask evaluation folders under the dataset root:
- `evaluation_results_points_only_sam3_merged_<method>/`

Inside each folder, the main outputs are:
- `results_merged.json`
- `predicted_masks_union.json`
- `predicted_masks_intersection.json`
- `predicted_masks_confidence_weighted.json`
- `predicted_masks_nms_instances.json`

Public entry points:
- `main_pipeline.py`
- `Stage0_prepare_pascal5i_annotations.py`
- `Stage0_run_pipeline.py`
- `Stage1_build_and_filter_feature_bank.py`
- `Stage1_score_feature_bank.py`
- `Stage1_filter_scored_feature_bank.py`
- `Stage2_cache_feature_matching.py`
- `Stage3_evaluate_feature_matching.py`
- `Stage4_evaluate_sam3.py`
- `Stage5_evaluate_merged_sam3_masks.py`

Pipeline summary:
- Stage 0: optional PASCAL-5i / VOC to COCO-style JSON conversion
- Stage 1: raw feature-bank build + reusable scoring + ICCD adaptive_q75 filtering
- Stage 2: TSG feature-matching prompt cache generation
- Stage 3: non-SAM feature-matching evaluation
- Stage 4: SAM3 evaluation with `text_prompt`, `point_prompt`, or `text_and_point`
- Stage 5: merge text-only and point-only SAM3 masks

Core modules:
- `common.py`
- `resume.py`
- `metrics.py`
- `cache_feature_matching.py`
- `intra_class_filter_impl.py`
- `non_sam_eval_impl.py`
- `sam3_points_only_impl.py`
- `sam3_text_and_points_impl.py`
- `merge_masks_impl.py`
