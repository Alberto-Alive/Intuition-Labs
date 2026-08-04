# Stage 6 Integrated Multi-Perspective Search

## Scope

- Fixed benchmark: Stage 4 schema-aware `balanced_categories_v3`.
- Dataset labels: unchanged.
- Oracle metadata/gold candidates: not exposed to model-facing records.
- Frozen comparator: exact same architecture, same masks, same fit seed; shared encoder weights frozen.
- Final seeds `[50..59]` are reserved unless `--phase final` is explicitly run.

## Variant Plan

| variant | family | coordinator | layers | description |
|---|---|---|---:|---|
| candidate_token_direct_lr3e4_clip1 | stage4_clone_baseline | candidate_token_cross_attention | 1 | Proven Stage 4 clone baseline: candidate queries attend directly over per-role token states. |
| integrated_bridge_late1_lr3e4 | hybrid_clone_to_integrated_bridge | integrated_bridge_late | 1 | Clone token states feed a semi-integrated masked bridge; candidates attend full role tokens plus role summaries. |
| integrated_bridge_late2_lr3e4 | hybrid_clone_to_integrated_bridge | integrated_bridge_late | 2 | Two candidate-conditioned integrated bridge blocks after separated role encoding. |
| integrated_summary_late1_lr3e4 | semi_integrated_cross_view | integrated_summary_late | 1 | Role token streams are summarized, then candidates query a compact cross-view summary layer. |
| integrated_router_late1_lr3e4 | routing_token_architecture | integrated_router_late | 1 | Learned router tokens aggregate role token states; candidates query routers only. |
| integrated_router_plus_tokens_late1_lr3e4 | routing_token_architecture | integrated_router_plus_tokens | 1 | Router tokens and full role token states are both visible to candidate queries. |
| integrated_candidate_guided2_lr3e4 | candidate_guided_evidence_processing | integrated_candidate_guided | 2 | Candidate hypotheses are present while summaries are refined and query role evidence across two blocks. |
| integrated_early_mixing1_lr3e4 | masked_cross_perspective_transformer | integrated_early_mixing | 1 | Role summaries mix early before candidate-conditioned querying; included to test leakage risk. |
| integrated_late_mixing_only2_lr3e4 | masked_cross_perspective_transformer | integrated_late_mixing_only | 2 | No early cross-view update; candidates perform all cross-role coordination late. |

## Phase 1 Cheap Screen

| variant | seeds | wins | mean train | mean frozen | mean delta | controls | masks | mean-gate pass |
|---|---:|---:|---:|---:|---:|---|---|---|
| integrated_early_mixing1_lr3e4 | 3 | 3 | 0.8060 | 0.1758 | 0.6302 | `True` | `True` | `True` |
| integrated_candidate_guided2_lr3e4 | 3 | 2 | 0.5599 | 0.1211 | 0.4388 | `True` | `True` | `True` |
| integrated_router_plus_tokens_late1_lr3e4 | 3 | 2 | 0.5169 | 0.1784 | 0.3385 | `True` | `True` | `True` |
| integrated_summary_late1_lr3e4 | 3 | 3 | 0.9297 | 0.1641 | 0.7656 | `False` | `True` | `False` |
| candidate_token_direct_lr3e4_clip1 | 3 | 3 | 0.7357 | 0.1680 | 0.5677 | `False` | `True` | `False` |
| integrated_router_late1_lr3e4 | 3 | 1 | 0.4258 | 0.1589 | 0.2669 | `True` | `True` | `False` |
| integrated_late_mixing_only2_lr3e4 | 3 | 1 | 0.3385 | 0.1510 | 0.1875 | `False` | `True` | `False` |
| integrated_bridge_late2_lr3e4 | 3 | 1 | 0.2539 | 0.1458 | 0.1081 | `True` | `True` | `False` |
| integrated_bridge_late1_lr3e4 | 3 | 0 | 0.1628 | 0.1628 | 0.0000 | `True` | `True` | `False` |

## Phase 2 Medium Validation

- Not run: `blocked by Stage 6 Phase 1 audit: no strict eligible variants and/or baseline regression not clean`

## Phase 3 Final Validation

- Not run: `not requested; reserved seeds [50..59]`

## Required Questions

1. Can the clone architecture be folded into a more integrated one-pass architecture? `not established; Phase 1 audit blocked integrated promotion`
2. Did integration preserve the scientific proof? `not preserved under strict audit; mean-gate pass is diagnostic only`
3. Did any variant beat or match the Stage 4 clone baseline? `not established under strict audit`
4. Did any variant improve weak seeds? `not established under strict audit`
5. Did any variant reduce compute/memory/latency? `yes: ['integrated_bridge_late2_lr3e4', 'integrated_candidate_guided2_lr3e4', 'integrated_early_mixing1_lr3e4', 'integrated_late_mixing_only2_lr3e4', 'integrated_router_late1_lr3e4', 'integrated_router_plus_tokens_late1_lr3e4', 'integrated_summary_late1_lr3e4']`
6. Did candidate-guided processing help? `mean dev delta 0.4388`
7. Did early cross-view mixing help or create leakage? `mean dev delta 0.6302; mean-gate pass=True; strict per-seed row failures=2/3`
8. Did attention masks enforce the intended information barriers? `yes`
9. Did controls remain near chance? `strict audit failed; see reports/STAGE6_PHASE1_AUDIT_AND_BASELINE_REGRESSION.md`
10. Ready for full final validation, or keep clone mainline? `blocked by Phase 1 audit; keep clone mainline and rerun Phase 1 with clean baseline regression before medium validation`

## Conservative Interpretation

- Final validation ran: `False`
- Final gates passed: `False`
- Success claim: `none unless final_passed is true`
- No general breakthrough is claimed unless Phase 3 final gates pass.
