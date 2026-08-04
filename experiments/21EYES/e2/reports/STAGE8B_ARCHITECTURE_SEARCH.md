# Stage 8B Architecture Search

Decision: STAGE8B_HYPOTHESIS_NOT_SUPPORTED_ON_DEV

Explicit statement: NO_STAGE8C_CLAIM. Stage 8B is dev-only architecture search and does not validate a final 10x claim.

## Baseline And Target

- Stage 8A matched monolithic baseline capacity: 0
- Target capacity: 0
- Stage 8B target N: 8
- Budget signature: c2db67c343ab2249

## Search Summary

- Evaluated configs: 50
- Promoted configs: 10
- Finalist rows: 1
- Selected config: latent_r8_a8_035251
- Selected dev capacity: 0
- Selected dev capacity ratio: 0.0

## Ranked Rows

| Phase | Config | Capacity | Ratio | Max Accuracy | Compute | Controls |
|---|---|---:|---:|---:|---:|---|
| stage8b_finalist | latent_r8_a8_035251 | 0 | 0.0 | 0.1458 | 434176.0 | tracked |
| stage8b_promoted | latent_r8_a8_197133 | 0 | 0.0 | 0.1562 | 851968.0 | tracked |
| stage8b_promoted | latent_r4_a1_658127 | 0 | 0.0 | 0.1528 | 68608.0 | tracked |
| stage8b_promoted | latent_r8_a8_035251 | 0 | 0.0 | 0.1493 | 434176.0 | tracked |
| stage8b_promoted | latent_r4_a8_301861 | 0 | 0.0 | 0.1458 | 294912.0 | tracked |
| stage8b_promoted | latent_r16_a8_706261 | 0 | 0.0 | 0.1458 | 1130496.0 | tracked |
| stage8b_promoted | latent_r4_a8_273590 | 0 | 0.0 | 0.1319 | 434176.0 | tracked |
| stage8b_promoted | latent_r2_a2_251045 | 0 | 0.0 | 0.1250 | 68608.0 | tracked |
| stage8b_promoted | latent_r4_a2_817255 | 0 | 0.0 | 0.1215 | 120832.0 | tracked |
| stage8b_promoted | latent_r2_a4_189077 | 0 | 0.0 | 0.1181 | 86016.0 | tracked |
| stage8b_promoted | latent_r4_a2_857859 | 0 | 0.0 | 0.1146 | 120832.0 | tracked |
| stage8b_small | latent_r16_a8_536110 | 0 | 0.0 | 0.2188 | 1187840.0 | tracked |
| stage8b_small | latent_r4_a8_301861 | 0 | 0.0 | 0.2188 | 286720.0 | tracked |
| stage8b_small | latent_single_role_ablation | 0 | 0.0 | 0.1979 | 77824.0 | tracked |
| stage8b_small | latent_r8_a8_197133 | 0 | 0.0 | 0.1979 | 843776.0 | tracked |
| stage8b_small | latent_r2_a1_978147 | 0 | 0.0 | 0.1875 | 34304.0 | tracked |
| stage8b_small | latent_r4_a2_857859 | 0 | 0.0 | 0.1875 | 112640.0 | tracked |
| stage8b_small | latent_r2_a4_579363 | 0 | 0.0 | 0.1771 | 60416.0 | tracked |
| stage8b_small | latent_r2_a8_765752 | 0 | 0.0 | 0.1771 | 112640.0 | tracked |
| stage8b_small | latent_r4_a1_658127 | 0 | 0.0 | 0.1771 | 60416.0 | tracked |
| stage8b_small | latent_r2_a1_069577 | 0 | 0.0 | 0.1771 | 21248.0 | tracked |
| stage8b_small | latent_r8_a8_035251 | 0 | 0.0 | 0.1771 | 425984.0 | tracked |
| stage8b_small | latent_r16_a8_892670 | 0 | 0.0 | 0.1771 | 1122304.0 | tracked |
| stage8b_small | latent_r16_a8_706261 | 0 | 0.0 | 0.1771 | 1122304.0 | tracked |
| stage8b_small | latent_r16_a4_826871 | 0 | 0.0 | 0.1771 | 450560.0 | tracked |
| stage8b_small | latent_r4_a1_784310 | 0 | 0.0 | 0.1771 | 34304.0 | tracked |
| stage8b_small | latent_r8_a2_379836 | 0 | 0.0 | 0.1771 | 131072.0 | tracked |
| stage8b_small | latent_r8_a2_953938 | 0 | 0.0 | 0.1667 | 229376.0 | tracked |
| stage8b_small | latent_r2_a2_862696 | 0 | 0.0 | 0.1667 | 43008.0 | tracked |
| stage8b_small | latent_r2_a2_044352 | 0 | 0.0 | 0.1667 | 34304.0 | tracked |
| stage8b_small | latent_r16_a4_402820 | 0 | 0.0 | 0.1667 | 794624.0 | tracked |
| stage8b_small | latent_r4_a2_817255 | 0 | 0.0 | 0.1667 | 112640.0 | tracked |
| stage8b_small | latent_r2_a4_376704 | 0 | 0.0 | 0.1667 | 63488.0 | tracked |
| stage8b_small | latent_r4_a2_candidate_task_mixture | 0 | 0.0 | 0.1562 | 77824.0 | tracked |
| stage8b_small | latent_r8_a4_topk_semantic | 0 | 0.0 | 0.1562 | 286720.0 | tracked |
| stage8b_small | latent_r4_a8_overlap_sparsemax | 0 | 0.0 | 0.1562 | 286720.0 | tracked |
| stage8b_small | latent_single_avenue_ablation | 0 | 0.0 | 0.1562 | 77824.0 | tracked |
| stage8b_small | latent_r8_a2_122824 | 0 | 0.0 | 0.1562 | 112640.0 | tracked |
| stage8b_small | latent_r2_a4_189077 | 0 | 0.0 | 0.1562 | 77824.0 | tracked |
| stage8b_small | latent_r2_a2_251045 | 0 | 0.0 | 0.1562 | 60416.0 | tracked |
| stage8b_small | latent_r16_a8_913740 | 0 | 0.0 | 0.1562 | 1122304.0 | tracked |
| stage8b_small | latent_r16_a4_373873 | 0 | 0.0 | 0.1562 | 991232.0 | tracked |
| stage8b_small | latent_r16_a2_hierarchical | 0 | 0.0 | 0.1458 | 286720.0 | tracked |
| stage8b_small | latent_r8_a8_two_stage | 0 | 0.0 | 0.1458 | 565248.0 | tracked |
| stage8b_small | latent_r8_a8_836695 | 0 | 0.0 | 0.1458 | 425984.0 | tracked |
| stage8b_small | latent_r4_a2_607854 | 0 | 0.0 | 0.1458 | 60416.0 | tracked |
| stage8b_small | latent_r4_a2_886491 | 0 | 0.0 | 0.1458 | 112640.0 | tracked |
| stage8b_small | latent_r8_a4_263121 | 0 | 0.0 | 0.1458 | 425984.0 | tracked |
| stage8b_small | latent_r8_a1_037397 | 0 | 0.0 | 0.1458 | 112640.0 | tracked |
| stage8b_small | latent_r16_a8_502358 | 0 | 0.0 | 0.1458 | 843776.0 | tracked |
| stage8b_small | latent_r4_a8_273590 | 0 | 0.0 | 0.1458 | 425984.0 | tracked |
| stage8b_small | latent_r4_a1_155610 | 0 | 0.0 | 0.1458 | 43008.0 | tracked |
| stage8b_small | latent_r2_a8_435328 | 0 | 0.0 | 0.1458 | 155648.0 | tracked |
| stage8b_small | latent_r8_a8_711693 | 0 | 0.0 | 0.1354 | 892928.0 | tracked |
| stage8b_small | latent_r4_a1_027993 | 0 | 0.0 | 0.1354 | 34304.0 | tracked |
| stage8b_small | latent_r2_a2_220805 | 0 | 0.0 | 0.1354 | 45056.0 | tracked |
| stage8b_small | latent_r2_a2_271819 | 0 | 0.0 | 0.1354 | 35840.0 | tracked |
| stage8b_small | latent_r16_a2_417284 | 0 | 0.0 | 0.1354 | 286720.0 | tracked |
| stage8b_small | latent_r16_a2_627363 | 0 | 0.0 | 0.1354 | 286720.0 | tracked |
| stage8b_small | latent_r2_a8_322835 | 0 | 0.0 | 0.1250 | 131072.0 | tracked |
| stage8b_small | latent_r16_a1_207660 | 0 | 0.0 | 0.1250 | 155648.0 | tracked |

## Artifacts

- Search database: `results\stage8_search_database.jsonl`
- Promoted configs: `results\stage8_promoted_configs.json`
- Finalists: `results\stage8_finalist_results.json`
- Frozen best config: `results\stage8_best_config.yaml`
- Freeze manifest: `results\stage8_config_freeze_manifest.json`
- Controls audit: `results\stage8_stage8b_controls_audit.jsonl`
- Compute audit: `results\stage8_stage8b_compute_audit.json`
- Capacity curves: `results\stage8_stage8b_capacity_curves.csv`

Failed configs and zero-capacity configs are retained in the JSONL database.
