# Result Provenance

## Reporting Convention

The accepted Round 2 manuscript is the source for the README's reported-score
tables, including AgML 59.24% mIoU and its published per-class values. Separate
rerun artifacts remain unchanged and are labelled as separate measurements.
Following the paper for presentation does not establish numerical equivalence
between those artifacts and the reported results.

## Standard Benchmarks

The fixed one-shot and five-shot protocols for ADE20K-150, Cityscapes, PC-59,
and LVIS are in `splits/standard/` and `configs/paper/`. Full-precision archived
metrics are in `reproducibility/table4_results.json`. Their values and manifest
contents have not been changed by the release cleanup. Use `evaluate_paper.py`,
not the generic first-N runner, to reproduce these rows.

Archived results are reference measurements, not a claim that this cleanup
reran every benchmark. Numerical parity still requires the documented data,
checkpoints, software, and full evaluations. `--strict-score-check` checks the
observed mIoU against the archived value.

The README explicitly marks literature numbers. GF-SAM's LVIS paper scores use
a different protocol and are context, not a matched local SegRAG comparison.

## AgML Largest-Reference Discrepancy

The accepted Round 2 manuscript reports **59.24%** AgML mIoU. The saved corrected
strict up-to-30-shot artifact instead contains **58.4508608598833%**. These are
different measurements; the release does not claim that the corrected strict
runner reproduces 59.24%.

The strict result has been exported without machine-specific absolute paths to
[`agml_strict_up_to_30shot.json`](../reproducibility/agml_strict_up_to_30shot.json).
It retains the original artifact's SHA-256, full-precision metrics, per-class
scores, settings, and error counts. The source is the completed August 30 run
`job3_corrected_up_to_30shot_v2`, with zero joint-prediction errors and no missing
prompt caches.

| Measurement | mIoU (%) | Interpretation |
|---|---:|---|
| Accepted manuscript, largest-reference row | 59.24 | Historical reported value; not a verified strict-run target |
| Corrected strict up-to-30-shot saved run | 58.45 | Same N supports for building/scoring; self-image excluded |
| Controlled seed-17 five-shot full system | 59.29 | Separate support selection and shot count |

For the strict largest-reference run, ten classes use 30 supports and sugarbeet
weed uses 17, giving 317 support-class entries and 2,183 query-class pairs.
Tomato is 61.96% in this artifact, not the historical 67.80%. Neither value is
silently substituted into the accepted manuscript by this cleanup.

The split manifests describe image membership, not an assertion that every
historical result used the same bank-scoring implementation. The accepted
figures are retained as qualitative paper assets; they are not regenerated
predictions from this cleaned release.

## Component Ablations

The README uses the final controlled seed-17 five-shot component experiment,
not the superseded mixed-protocol component table or historical shot sweep.
All six rows share the 2,183-pair query set. Raw+TSG is 46.69% mIoU, not the
older 44.40%. Text+point means simultaneous grounding, never post-hoc merging.

The maintained public entrypoints implement the main SegRAG evaluation and
text/point prompt modes. They do not yet constitute a standalone reproduction
harness for every repeated-seed, raw/no-TSG, or timing experiment in the paper.
Those result summaries must not be mistaken for runnable ablation commands.
