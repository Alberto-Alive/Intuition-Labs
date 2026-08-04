# Attack Matrix

| Attack family | Goal | Primary datasets | Metrics |
|---|---|---|---|
| Membership inference | infer whether a protected unit was in the source data | NIST, MIMIC-IV demo, TCGA | AUC, TPR@FPR, precision |
| Attribute inference | infer a hidden attribute of a protected unit | NIST, MIMIC-IV demo, TCGA | accuracy, balanced accuracy, calibration |
| Subgroup presence | detect whether a subgroup exists in the data | NIST, TCGA | attack success rate |
| Reconstruction / inversion | recover sensitive properties from outputs | NIST, TCGA | reconstruction error, match rate |
| Linkage | link DIGIT outputs to auxiliary external records | MIMIC-IV demo, TCGA | linkage precision / recall |
| Adaptive repeated-query | reduce uncertainty by steering queries | all | breach rate vs query count |
| Composition stress | estimate cumulative leakage over many queries | all | success vs budget curve |
