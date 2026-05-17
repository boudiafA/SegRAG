# SegRAG

SegRAG is a retrieval-augmented semantic segmentation pipeline that combines
DINOv3 visual feature banks with SAM3 mask decoding. The repository is arranged
as a clean paper-code release: importable source lives under `src/segrag`,
reproducible command-line entrypoints live under `scripts`, and dataset-specific
run templates live under `configs`.

## Method Overview

SegRAG has two main stages beyond the SAM3 text-only baseline.

**Intra-Class Cohesion Distillation (ICCD).** Stage 1 builds a raw DINOv3
feature bank from reference images, scores same-class features against annotated
masks, and filters the bank with the default adaptive `q75` rule. The filtered
bank is saved once and reused during evaluation.

**Topographic Similarity Grounding (TSG).** Stage 2 matches the filtered bank
against each query image, converts dense similarity maps into spatially coherent
point prompts, and caches those prompts. Stage 4 sends class text and/or cached
points to SAM3 and evaluates the resulting masks.

## Repository Layout

```text
SegRAG/
  src/segrag/
    data/        dataset export and shot-list utilities
    modeling/    DINOv3 matching, ICCD, SAM3 prompt execution
    pipelines/   high-level end-to-end runners
    stages/      stage-level wrappers and evaluation code
    utils/       paths, metrics, cache, and resume helpers
  scripts/       stable CLI entrypoints
  configs/       dataset run templates
  examples/      short runnable examples
  tests/         regression and equivalence checks
```

## Installation

The tested layout keeps SegRAG next to local DINOv3 and SAM3 checkouts:

```bash
mkdir -p ~/RAG-SAM
cd ~/RAG-SAM
git clone https://github.com/boudiafA/SegRAG.git SegRAG
git clone https://github.com/facebookresearch/dinov3.git dinov3
git clone https://github.com/facebookresearch/sam3.git sam3
```

Create the environment:

```bash
cd ~/RAG-SAM/SegRAG
conda env create -f environment.yml
conda activate segrag
pip install -e .
```

Install SAM3 according to the official SAM3 repository instructions. SegRAG
loads DINOv3 from the sibling `../dinov3` checkout by default. Override paths
when needed:

```bash
export DINOV3_REPO_PATH=/path/to/dinov3
export DINOV3_WEIGHTS_PATH=/path/to/dinov3_vitl16_pretrain_lvd1689m.pth
```

## Data Format

The main runner expects a COCO/LVIS-style dataset root:

```text
dataset_root/
  train.json
  val.json
  images/
```

`train.json` supplies reference images and masks for the feature bank. `val.json`
or `test.json` supplies query images and ground truth masks for evaluation.
Each JSON must contain top-level `images`, `annotations`, and `categories`.
Polygon and COCO RLE segmentations are supported.

LVIS-style split folders are also detected:

```text
dataset_root/
  train/lvis_v1_train.json
  train/images/
  val/lvis_v1_val.json
  val/images/
```

ADE20K and other exported layouts can be converted through `scripts/run_ade20k.py`
or `scripts/run_adapters.py`.

## Quick Start

Run the full text+point pipeline on a COCO-style dataset:

```bash
python scripts/run_pipeline.py \
  --dataset-root /path/to/dataset_root \
  --segmentation-method text-and-point \
  --reference-images-per-class 20 \
  --feature-matching-method hybrid \
  --resume \
  --save-mask-json
```

Run the SAM3 text-only baseline on the same validation split:

```bash
python scripts/evaluate_sam3.py \
  --dataset-root /path/to/dataset_root \
  --prompt-mode text_prompt \
  --resume \
  --save-mask-json
```

Run ADE20K-150 after exporting it to the internal COCO-style view:

```bash
python scripts/run_ade20k.py \
  --dataset-root /path/to/ade20k_root \
  --dataset-format ade20k_150 \
  --segmentation-method text-and-point \
  --reference-images-per-class 20 \
  --resume
```

Run an adapter-supported dataset:

```bash
python scripts/run_adapters.py \
  --dataset-root /path/to/raw_dataset \
  --adapter auto \
  --segmentation-method text-and-point \
  --reference-images-per-class 20 \
  --resume
```

## Outputs

By default, generated artifacts are written under the dataset root:

```text
feature_bank_dinov3_vitl16_1536/
feature_bank_dinov3_vitl16_1536_scored_thr060/
feature_bank_adaptive_q75_from_thr060/
_prompt_cache/
evaluation_results_*/
```

Use separate dataset/output roots for different shot settings to avoid mixing
feature banks or prompt caches from different protocols.

## Entry Points

After `pip install -e .`, these console commands are available:

```bash
segrag-run
segrag-run-adapters
segrag-run-ade20k
segrag-evaluate-sam3
segrag-generate-support-shots
segrag-prepare-pascal5i
```

The equivalent direct scripts are under `scripts/`.

## Development Checks

```bash
PYTHONPATH=src python -m compileall -q src scripts tests
PYTHONPATH=src python scripts/run_pipeline.py --help
PYTHONPATH=src python scripts/evaluate_sam3.py --help
```

## Citation

The citation entry will be added after paper metadata is finalized.
