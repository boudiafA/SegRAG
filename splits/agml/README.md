# AgML Split Manifest

This directory documents the exact AgML image identifiers used for the agricultural SegRAG evaluation. The evaluation used a 30-shot reference setting per class where possible. For `sugarbeet_weed`, only 17 training images containing the class are available in the local AgML train split, so all 17 are used. Query evaluation uses every test image containing the target class; no query subsampling is applied.

The detailed image-level manifests are:

- `agml_30shot_support_manifest.json`: reference/support image IDs, relative file names, sizes, and annotation IDs per class.
- `agml_query_manifest.json`: evaluation/query image IDs, relative file names, sizes, and annotation IDs per class.
- `agml_class_summary.csv`: compact per-class support/evaluation counts.

Selection rule: for each class, support images are the first 30 sorted training image IDs containing that class after filtering to the selected AgML classes. Numeric image IDs are sorted numerically, with lexical fallback for non-numeric IDs. ICCD filters patch descriptors inside the selected support images but does not change the support image ID manifest.

| Class | Category ID | Reference images | Reference anns. | Evaluation images | Evaluation anns. |
|---|---:|---:|---:|---:|---:|
| apple | 1 | 30/30 | 1111 | 134 | 3821 |
| bean leaf | 2 | 30/30 | 4237 | 478 | 66238 |
| bell pepper | 4 | 30/30 | 139 | 204 | 701 |
| carrot | 5 | 30/30 | 87 | 12 | 38 |
| cauliflower | 7 | 30/30 | 76 | 297 | 830 |
| flower | 8 | 30/30 | 1614 | 39 | 1390 |
| grape | 10 | 30/30 | 122 | 107 | 346 |
| rice | 11 | 30/30 | 985 | 45 | 2445 |
| sugarbeet weed | 14 | 17/30 | 1352 | 25 | 1955 |
| tomato | 15 | 30/30 | 146 | 89 | 550 |
| weed | 16 | 30/30 | 213 | 753 | 3102 |
