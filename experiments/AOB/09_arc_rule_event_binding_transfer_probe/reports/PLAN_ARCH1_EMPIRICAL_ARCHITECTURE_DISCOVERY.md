# PLAN-ARCH-1 Empirical Architecture Discovery

## Scope

- Bounded architecture search for latent candidate-plan evaluation on synthetic gridworld candidate-plan verification.
- This is not final validation, autonomous planning, ARC solving, world modeling, or general agentic AI.
- Exact frozen same-architecture comparators, shortcut baselines, mismatch controls, candidate-order remap, role-order remap, and hidden-state shuffle controls are required.
- Final validation was not launched.

## Leaderboard

| rank | phase | variant | trainable | frozen | delta | success | shortcut | controls |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | cheap | Q_candidate_self_attention | 0.6562 | 0.4062 | 0.2500 | 0.6562 | 0.6250 | `False` |
| 2 | cheap | Q_bidirectional_candidate_evidence | 0.6250 | 0.4688 | 0.1562 | 0.6250 | 0.6250 | `False` |
| 3 | cheap | P1_rollout_no_final | 0.5625 | 0.4688 | 0.0938 | 0.5625 | 0.6250 | `False` |

## Failure Taxonomy

```json
{
  "control_failure": 3,
  "shortcut_baseline_explains_result": 2
}
```

## Medium Gates

```json
{}
```

## Required Answers

1. Which architecture variant beats P1_rollout_no_final? `none`.
2. Which mechanism mattered most? `candidate_query`.
3. Did any variant improve trainable-frozen delta? `True best=Q_candidate_self_attention delta=0.2500`.
4. Did any variant improve absolute selected-plan success? `Q_candidate_self_attention success=0.6562`.
5. Did any variant reduce frozen performance while increasing trainable performance? `Q_candidate_self_attention`.
6. Did any variant fail because frozen was suspiciously high? `False`.
7. Did pyramid become valid after controls? `not_run`.
8. Did stacking help or overfit? `not_run`.
9. Did multi-avenue clones help? `not_run`.
10. Which variants failed and why? `{'control_failure': 3, 'shortcut_baseline_explains_result': 2}`.
11. Is there a medium-ready variant? `None`.
12. Should the winning mechanism be transferred back to real-code or ARC-hybrid? `only after medium gates pass; current smoke results are architecture-search diagnostics`.

## Claim Boundary

- Do not claim autonomous planning.
- Do not claim world modeling.
- Do not claim general agentic AI.
- Allowed claim only if final gates eventually pass: A searched variant of shared-weight latent candidate evaluation improves controlled candidate-plan ranking over the locked P1 baseline and exact frozen comparator while passing shortcut, mismatch, shuffle, leakage, and invariance controls.
