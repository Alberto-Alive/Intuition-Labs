# Attack Runbook

Run attacks in this order:
0. all-in-one entrypoint (`python3 experiments/DIGIT/Validation/scripts/run_all_comparative.py`)
1. comparative MIA (`python3 -m Validation.runners.run_comparative_mia`)
2. query averaging (`python3 -m Validation.runners.run_query_averaging`)
3. differencing (`python3 -m Validation.runners.run_differencing_attack`)
4. attribute inference (`python3 -m Validation.runners.run_attribute_inference`)
5. composition stress (`python3 -m Validation.runners.run_composition_stress`)
6. reconstruction / inversion (`python3 -m Validation.runners.run_reconstruction_attack`)
7. canary detection (`python3 -m Validation.runners.run_canary_detection`)
8. linkage (`python3 -m Validation.runners.run_linkage_attack`)
9. model extraction (`python3 -m Validation.runners.run_model_extraction`)
10. gradient optimization (`python3 -m Validation.runners.run_gradient_optimization`)
11. auxiliary amplification (`python3 -m Validation.runners.run_auxiliary_amplification`)

For every run, record:
- DIGIT version / commit
- dataset snapshot and split
- threat-model version
- attacker knowledge setting
- query budget
- evaluation mode (`mechanism` or `system`)
- seed
- hardware / runtime

Every attack result must be reported for:
- raw baseline
- DIGIT
- DP baseline
- DIGIT ablation if relevant
