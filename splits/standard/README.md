# Frozen Standard-Benchmark Protocol

These manifests freeze the image identifiers used for the one-shot and
five-shot results reported by SegRAG. They contain identifiers and relative
paths only; benchmark images and annotations must be obtained from their
original sources.

| Dataset | 1-shot classes/items | 5-shot classes/items | Unique 5-shot images | Query images | Image-class query pairs |
|---|---:|---:|---:|---:|---:|
| PC-59 | 59/59 | 59/295 | 262 | 5,105 | 25,975 |
| Cityscapes | 19/19 | 19/95 | 56 | 500 | 6,005 |
| ADE20K-150 | 150/150 | 150/750 | 685 | 2,000 | 16,909 |
| LVIS | 1,203/1,203 | 1,015/5,075 | 4,065 | 19,626 | 70,139 |

Support examples are selected exclusively from each benchmark's training
split. Queries use the full exported validation split represented by the
corresponding `queries.json`; support/query overlap is therefore disallowed by
construction. A training image may support more than one class, with a
different binary mask for each class.

The one-shot item is rank 1 of the same deterministic class ordering used by
five-shot. Ranking prefers images whose foreground descriptors survived the
archived ICCD bank, then orders them by archived feature-quality statistics and
image ID. The fixed IDs, rather than an external cache, are the normative
protocol.

LVIS has 1,203 categories in the training annotations, 1,035 categories in the
fixed query annotations, and 1,015 categories with at least five valid support
images. The five-shot evaluator still processes the full fixed query set;
classes without a five-reference bank follow the model's text-only fallback.

`checksums.json` records the SHA-256 of every released compact manifest and of
the original archived manifest from which it was generated. Run
`python tools/verify_paper_release.py` before evaluation to detect drift.
