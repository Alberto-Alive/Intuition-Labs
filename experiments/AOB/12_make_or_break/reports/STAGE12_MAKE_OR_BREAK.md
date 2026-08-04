# Stage 12 Make-Or-Break

## Source

- Requested preset: `stage5_final`
- Actual preset: `stage5_final`
- Device: `cuda`

## Prior Hypothesis: Avenue Residual

- Mean shared ratio: `0.6999`
- Avenue probe accuracy: `0.7052`
- Family-centred avenue probe accuracy: `0.2749`
- Shuffled avenue probe accuracy: `0.2716`
- Duplicate prompt residual norm: `0.00000000`
- Avenue Sub-experiment B weighted residual vs wrong avenue: `0.3611` vs `0.6356`
- Contrast: the avenue-residual hypothesis showed a family confound and did not identify a clean avenue-specific communication primitive.

## SUB-EXPERIMENT A — FAMILY DECOMPOSITION

- Family shared component exists: `NO`
- Family probe accuracy above 0.70: `YES`
- Cross-family probe on residuals near chance: `YES`
- Decomposition is clean: `NO / CONFOUNDED`

- Mean family shared ratio: `0.3797`
- Family probe accuracy: `0.8667`
- Cross-family probe on residuals: `0.2694`
- Example probe within family: `0.0004`
- Example probe chance within family: `0.0002`
- Within-family shuffle example probe: `0.0002`
- Cross-family shuffle probe accuracy: `0.2667`
- Duplicate prompt residual norm: `0.00000000`

Family size distribution:

- test_real_import_restore_0: `2048`
- test_real_import_restore_4: `2048`
- test_real_import_restore_6: `6144`
- train_real_import_restore_0: `8192`
- train_real_import_restore_4: `8192`
- train_real_import_restore_6: `4096`

- Families below 10 examples: `[]`
- Single-example families excluded from ratio calculation: `[]`
- Pairwise family-mean cosine mean/max/min off diagonal: `{'max_off_diagonal': 0.9999158978462219, 'mean_off_diagonal': 0.1685871439985931, 'min_off_diagonal': -0.06727595627307892}`
- Shared-ratio vs full-accuracy correlation: `0.9998366995870881`

## SUB-EXPERIMENT B — FAMILY RESIDUAL INJECTION

- Target set non-empty: `YES`
- Same family injection beats different family injection: `NO`
- Same family same avenue ≈ same family different avenue: `YES`
- Example residual alone weaker than family mean: `NO`
- Within-family shuffle ≈ same family injection: `YES`
- Controls passed: `NO`

Target set size per seed:

- Seed `30`: `0`
- Seed `31`: `768`
- Seed `32`: `0`
- Seed `33`: `0`
- Seed `34`: `0`
- Seed `35`: `0`
- Seed `36`: `0`
- Seed `37`: `456`
- Seed `38`: `0`
- Seed `39`: `0`

| condition | mean acc | weighted acc |
|---|---:|---:|
| baseline | 0.0000 | 0.0000 |
| different_family | 0.0000 | 0.0000 |
| example_residual_only | 0.0254 | 0.0319 |
| noise | 0.0039 | 0.0049 |
| same_family_different_avenue | 0.0000 | 0.0000 |
| same_family_same_avenue | 0.0000 | 0.0000 |
| within_family_shuffled | 0.0000 | 0.0000 |

| seed | baseline | same family same avenue | same family different avenue | different family | example residual only | within-family shuffled | noise |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 31 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0508 | 0.0000 | 0.0078 |
| 37 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## OVERALL VERDICT

- Redesigned hypothesis supported: `NO`
- What the result means in one sentence: The redesigned family-shared decomposition did not clear the structural gate.
- Recommended next step: Resolve the family confound or strengthen the family-mean intervention before making a communication-primitive claim.

## INTERNAL STRUCTURE DISCOVERY

Clustering summary:

- Optimal K: `3`
- Method that produced most stable clusters: `pca10_kmeans@K=2`
- Selected split agreement with PCA K-means: `0.3497`
- Raw vs PCA agreement by K: `{'2': 1.0, '3': 0.3497186027429692, '4': 0.8375449572006739, '5': 0.21873727535469736, '6': 0.9441644711510564, '7': 0.38181990338938077, '8': 0.645698282633294}`
- Cluster sizes: `{'0': 3356, '1': 34461, '2': 3143}`
- Pairwise cluster mean cosine similarities: `{'mean_off_diagonal': -0.05183397978544235, 'max_off_diagonal': 0.09361419826745987, 'min_off_diagonal': -0.1790887415409088}`

Per-cluster characterisation:

- Cluster 0: size `1024` example-seed cases / `3356` rows, families `test_real_import_restore_6:3356 (1.00)`, avenues `0:1024 (0.31), 2:1024 (0.31), 3:1024 (0.31), 1:284 (0.08)`, full `0.5498`, best single `0.7148`, gap `-0.1650`, difficulty `mixed`
- Cluster 1: size `8197` example-seed cases / `34461` rows, families `test_real_import_restore_6:18077 (0.52), test_real_import_restore_0:8192 (0.24), test_real_import_restore_4:8192 (0.24)`, avenues `3:9140 (0.27), 1:8937 (0.26), 0:8192 (0.24), 2:8192 (0.24)`, full `1.0000`, best single `0.8764`, gap `0.1236`, difficulty `uniformly easy`
- Cluster 2: size `1019` example-seed cases / `3143` rows, families `test_real_import_restore_6:3143 (1.00)`, avenues `0:1024 (0.33), 2:1024 (0.33), 1:1019 (0.32), 3:76 (0.02)`, full `1.0000`, best single `1.0000`, gap `0.0000`, difficulty `uniformly easy`

Stability:

- Mean per-example cluster stability across seeds: `0.8005`
- Stable (above 0.80): `YES`

Difficulty confound check:

- Correlation of cluster identity with accuracy: `full=0.6514`, `best_single=0.1924`, `gap=0.2740`
- Is clustering purely difficulty-stratified: `NO`

Target set alignment:

- Target set examples concentrate in specific clusters: `NO`
- If yes, which clusters: `[]`

Interpretation:

- What do the clusters appear to represent: A coarse latent grouping is real, but the elbow-selected refinement is method-sensitive and it does not isolate the target-set cases.
- Are they a viable unit of analysis for the communication hypothesis: `UNCLEAR`

Recommended next step: Use the stable 2-cluster split as a coarse probe or move to layerwise representations before redesigning Sub-experiments A and B.
