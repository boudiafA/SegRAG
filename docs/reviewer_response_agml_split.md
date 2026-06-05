# AgML Split Documentation

Reviewer concern: the exact AgML split manifest and image identifiers were not present in the manuscript or editor letter.

Response text to use: We now explicitly document the AgML split used in the agricultural domain generalisation experiment. SegRAG was evaluated with a 30-shot reference setting per class where available. The support/reference images are selected from `train.json` as the first sorted image IDs containing the target class, and the evaluation set uses all `test.json` images containing the target class. The exact image identifiers, relative file names, image sizes, and annotation IDs are provided in the GitHub repository under `splits/agml/`.

Per-class split sizes:

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

Note: `sugarbeet_weed` has only 17 available training images containing the target class, so the experiment uses all available reference images for that class.
