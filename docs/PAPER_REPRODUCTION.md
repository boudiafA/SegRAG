# Paper Reproduction

This document reproduces the one-shot and five-shot SegRAG rows on PC-59,
Cityscapes, ADE20K-150, and LVIS. The frozen entrypoint is
`scripts/evaluate_paper.py`; the generic `scripts/run_pipeline.py` is intended
for new datasets and is not the normative Table 4 protocol.

## Frozen Method

Both settings use DINOv3 ViT-L/16 at 1536 x 1536, foreground-patch occupancy
of 0.90, SAM 3 joint text-and-point prompting, and the complete fixed query
manifest.

The selected `N` support images per class are the complete labelled set
available to the bank stage. They provide both source descriptors and ICCD
scoring targets. Self-image comparisons are excluded, so every source
descriptor is evaluated against exactly the other `N-1` selected images. No
separate target-image or calibration-image allowance is used.

For one-shot, DINOv3 foreground descriptors from the rank-1 support image are
used directly after the 0.90 occupancy gate. Cross-image ICCD scoring requires
another reference image and is therefore not defined for this setting. The
paper runner deliberately does not call the newer within-image fallback in the
generic ICCD module.

For five-shot, ICCD scores each support descriptor through cross-image
same-class retrieval. The scored bank first keeps scores >= 0.60, then applies
the class-adaptive threshold `clip(0.90 * q75, 0.65, 0.82)` and retains at most
10,000 descriptors per class. TSG uses similarity, loose, and validation
thresholds of 0.80, a minimum connected-component size of 4, minimum peak
distance of 10, and at most 10 points.

## Expected Results

All values are mIoU (%). Exact machine-readable metrics are in
`reproducibility/table4_results.json`.

| Dataset | 1-shot | 5-shot |
|---|---:|---:|
| PC-59 | 65.91 | 66.77 |
| Cityscapes | 66.35 | 67.25 |
| ADE20K-150 | 53.52 | 54.77 |
| LVIS | 56.71 | 58.84 |

The runner records observed and expected mIoU and checks an absolute tolerance
of 0.0005 on the fractional scale (0.05 percentage points). Add
`--strict-score-check` to turn a mismatch into a non-zero exit.

## Dependencies

Install SegRAG, official DINOv3, and official SAM 3 as described in the main
README. The exact tested repositories, checkpoint hashes, Hugging Face SAM 3
revision, and package versions are recorded in
`reproducibility/software.json`. Verify the release and DINOv3 checkpoint:

```bash
conda env create -f environment.paper.yml
conda activate segrag-paper
pip install -e /path/to/sam3 --no-deps
pip install -e . --no-deps
```

`--no-deps` preserves the tested numerical stack already installed by the
paper environment while registering both local packages.

Verify the release and DINOv3 checkpoint:

```bash
python tools/verify_paper_release.py \
  --dinov3-weights /path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

SAM 3 is loaded by its official `build_sam3_image_model()` function from the
gated `facebook/sam3` Hugging Face repository. Authenticate with an account
that has accepted the model terms before running evaluation.

## Dataset Layout

Each config in `configs/paper/` defines its exact annotation and image paths.
Prepare the official data in the following layouts, or edit only the relative
path fields while preserving the frozen manifests and model parameters:

```text
PC-59/                 Cityscapes/
  train.json             train.json
  val.json               val.json
  reference_imgs/        reference_imgs/
  JPEGImages/            leftImg8bit_trainvaltest/leftImg8bit/val/

ADE20K_150/            LVIS/
  train.json             train/lvis_v1_train.json
  val.json               val/lvis_v1_val.json
  reference_imgs/        reference_imgs/
  downloads/.../         val/images/
```

The evaluator reconstructs a new exact-shot training annotation from the
released support IDs. It never reads an old feature bank, score sidecar, prompt
cache, or prediction directory.

## Validate First

Validation checks the support counts, exact query image-class pairs, annotation
links, and every referenced image without loading either model:

```bash
python scripts/evaluate_paper.py \
  --config configs/paper/pc59.yaml \
  --dataset-root /data/PC-59 \
  --shot 5 \
  --output-root /scratch/segrag-paper \
  --dinov3-repo /opt/dinov3 \
  --dinov3-weights /models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --validate-only
```

## Run

Remove `--validate-only` for a full clean run:

```bash
python scripts/evaluate_paper.py \
  --config configs/paper/pc59.yaml \
  --dataset-root /data/PC-59 \
  --shot 5 \
  --output-root /scratch/segrag-paper \
  --dinov3-repo /opt/dinov3 \
  --dinov3-weights /models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --num-workers 8 \
  --save-mask-json \
  --strict-score-check
```

Use `pc59.yaml`, `cityscapes.yaml`, `ade20k150.yaml`, or `lvis.yaml`, and set
`--shot 1` or `--shot 5`. Use a new `--output-root` for paper reproduction.
`--max-images` is smoke-test-only and cannot produce a reported score.

Each dataset/shot workspace contains its reconstructed support annotations,
raw/scored/filtered banks, prompt cache, predictions, run metadata, file
fingerprints, and final summary. An interrupted run may be continued with
`--resume`. Resume refuses metadata from another config, manifest, checkpoint,
dataset, or shot. Use `--overwrite` only when deliberately rebuilding that
workspace from scratch.
