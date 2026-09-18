# Public Release Audit

Date: 2026-09-18. Base: `543715e2c741900b938dcb89357934a1c6958b82`
(`paper-release-1.1.0`). Package version after cleanup: `1.2.0`.

## Supported Workflows

| Entry point | Purpose |
|---|---|
| `scripts/evaluate_paper.py` | Fixed one/five-shot standard benchmark manifests and archived score checks |
| `scripts/run_pipeline.py` | Strict shared-N support construction/scoring on new COCO/LVIS-style data |
| `scripts/run_adapters.py`, `scripts/run_ade20k.py` | Data conversion followed by the same generic pipeline |
| `scripts/evaluate_sam3.py` | Text-only, point-only, or joint prompting with explicit artifacts |
| `scripts/generate_support_shots.py`, `scripts/prepare_pascal5i.py` | Support-list and dataset preparation |

The main inference path remains DINOv3 foreground descriptors, cross-image
ICCD for multiple supports, class-adaptive q75 filtering, hybrid TSG, and
simultaneous SAM 3 text-and-point grounding. One-shot uses occupancy-gated raw
descriptors without within-image scoring. Shared-N means at most N selected
supports, each scored against the other selected supports only.

## Removed From The Active Tree

- Versioned one-pass bank builders and the KMeans/clustered-ICCD alternative.
- Post-hoc mask-union/merging stages and the obsolete Stage 0-5 orchestrator.
- The orphan non-SAM evaluation branch and unused exploratory dataset viewers.
- YAML templates that no CLI consumed, and private reviewer-response prose.
- An experimental batched-scoring benchmark stored under `tests/` that contained
  no collected regression tests. Maintained behavior now has CPU regression tests.
- Early-accept/reference-budget scoring shortcuts unused by the paper path.
- Superseded component and shot-sweep tables in the README.

Dataset conversion/export tools, fixed-threshold sensitivity filtering, and
alternative point matchers remain deliberate utilities, not replacements for
the paper's adaptive-q75/hybrid defaults.

All removed tracked content is recoverable from the base tag. Earlier code
also remains in `archive/legacy-main` and `legacy-main-a08bc00`. No Git history,
research-workspace outputs, checkpoints, or manuscript files are deleted.

## Reproduction Safeguards

- SAM 3 now uses the already documented fixed checkpoint revision and verifies
  its SHA-256; an explicit local checkpoint receives the same check.
- Paper resume metadata fingerprints actual SegRAG source and annotation
  contents as well as config, manifests, and DINOv3 weights.
- Paper resume rejects changed smoke-image limits, changed mask-export mode,
  or nonempty output directories without provenance metadata.
- SciPy is an explicit dependency; missing SciPy must not silently substitute
  a different local-peak selection algorithm.
- The paper environment uses the recorded PyTorch/CUDA wheel versions rather
  than requesting unavailable newer PyTorch builds from the old Conda channel.

These are packaging, fail-fast, and provenance changes. No ICCD score formula,
active threshold, TSG default, SAM 3 decoding operation, or metric formula is
changed. Source cleanup is not evidence of a fresh full-score reproduction.

## Validation And Limits

Validation completed in the existing Python 3.12 SAM 3 environment:

- 41 CPU tests pass, including protocol checks, synthetic bank filtering,
  TSG connectivity and uncapped prompt counts, mocked simultaneous SAM 3
  grounding, and checkpoint pin/hash handling.
- All seven public source-checkout CLI entrypoints pass `--help`, including
  when invoked outside the repository without an editable install.
- Source, scripts, tests, and tools compile; `git diff --check` passes.
- All eight standard support/query manifest hashes and the archived score
  configuration checks pass. The local DINOv3 checkpoint SHA-256 also matches.
- Thirteen core definitions retain identical syntax trees to the base release;
  synthetic one-shot and five-shot scores and TP/FP votes also match the
  pre-cleanup scoring function exactly after removing unused shortcut branches.
- A `segrag-1.2.0` wheel builds successfully without bundling checkpoints or data.
- The old working checkout's pre-existing edits remain untouched.

GPU inference was not run during this cleanup: the execution sandbox reports
GPU access blocked by the operating system. SAM 3 predictions in the regression
tests are mocked, not fresh pretrained-model outputs. The Conda environment
file has not been installed into a new environment as part of these checks.

This cleanup does not run the full dataset evaluations or reproduce every
historical ablation. The README reports the accepted manuscript's scores;
separate rerun measurements are not substituted into those tables.
The known AgML result discrepancy is documented in
[result provenance](RESULT_PROVENANCE.md). Keep that caveat with the release;
do not promise exact reproduction of the historical 59.24% by the strict runner.
