# Stage 11G Action-Query Token Relay Combined Read

## Question

Can the Stage 11F Q/A relay be made more GPT-native by moving the action-specific answer signal into explicit Actor action-query tokens rather than adding a direct action-logit bias?

## Architecture Added

Implemented in `code/experiment11_gridworld_latent_attention.py`.

- `ACTION_QUERY_BASE..ACTION_QUERY_BASE+3` action-query tokens are inserted after the current state prefix.
- Private-view roles remain the same as Stage 11F.
- In token Q/A modes, Planner/Reader/Critic answers update the Actor's four action-query token states.
- Pure token variants score actions from the four final action-query token states.
- Hybrid token variants add action-token scores to the normal Actor state action head.
- Decision-token variants append a decision token after the action-query tokens; later causal layers let that decision token read the updated action-query tokens, and the normal action head scores the decision token.

This tests whether the communication can live inside the causal token stream rather than as a post-hoc action bias.

## Variants Tried

- `private_view_action_tokens_no_path`
- `private_view_qa_token_soft`
- `private_view_qa_token_hard`
- `private_view_qa_token_hard_aux`
- `private_view_qa_token_hard_aux_late`
- `private_view_qa_token_hard_aux_every`
- `full_view_qa_token_hard_aux`
- `private_view_action_tokens_hybrid_no_path`
- `private_view_qa_token_hard_aux_hybrid`
- `private_view_qa_token_hard_aux_hybrid_late`
- `private_view_qa_token_hard_aux_hybrid_every`
- `full_view_qa_token_hard_aux_hybrid`
- `private_view_action_tokens_decision_no_path`
- `private_view_qa_token_hard_aux_decision`
- `private_view_qa_token_hard_aux_decision_every`
- `full_view_qa_token_hard_aux_decision`

Runs:

- `results/stage11g_action_token_tiny_smoke`
- `results/stage11g_action_token_screen`
- `results/stage11g_action_token_retest16`
- `results/stage11g_action_token_hybrid_screen`
- `results/stage11g_action_token_decision_screen`
- `results/stage11g_action_token_decision_retest16`

## Main Results

### Pure Action-Token Retest, 16 Epochs

| method | rollout mean | raw rollout |
|---|---:|---|
| `single_actor` / `role_clones_no_path` | 0.3542 | 0.2969, 0.3750, 0.3906 |
| `private_view_no_path` | 0.1146 | 0.1406, 0.0781, 0.1250 |
| `private_view_cross_causal_mean` | 0.2969 | 0.2656, 0.3281, 0.2969 |
| `private_view_qa_hard_aux_bias` | 0.2240 | 0.2656, 0.2344, 0.1719 |
| `private_view_action_tokens_no_path` | 0.1302 | 0.1719, 0.1562, 0.0625 |
| `private_view_qa_token_hard` | 0.1042 | 0.1719, 0.0156, 0.1250 |
| `private_view_qa_token_hard_aux` | 0.1146 | 0.1562, 0.0938, 0.0938 |
| `private_view_qa_token_hard_aux_every` | 0.1458 | 0.1875, 0.1094, 0.1406 |

### Decision-Token Retest, 16 Epochs

| method | rollout mean | raw rollout |
|---|---:|---|
| `single_actor` / `role_clones_no_path` | 0.3542 | 0.2969, 0.3750, 0.3906 |
| `private_view_no_path` | 0.1146 | 0.1406, 0.0781, 0.1250 |
| `private_view_qa_hard_aux_bias` | 0.2240 | 0.2656, 0.2344, 0.1719 |
| `private_view_action_tokens_decision_no_path` | 0.1250 | 0.1562, 0.1094, 0.1094 |
| `private_view_qa_token_hard_aux_decision` | 0.1510 | 0.1406, 0.1562, 0.1562 |
| `full_view_qa_token_hard_aux_decision` | 0.3594 | 0.3281, 0.4688, 0.2812 |

## Mechanism Checks

The Stage 11G token relays did not show convincing reliance.

For `private_view_qa_token_hard_aux_decision`, mean rollout was `0.1510`:

| ablation | rollout mean | delta |
|---|---:|---:|
| `action_answer_shuffle` | 0.1302 | -0.0208 |
| `cross_task_kv_shuffle` | 0.1406 | -0.0104 |
| `mask_actor_peer_reads` | 0.1562 | +0.0052 |
| `remove_path` | 0.1615 | +0.0104 |
| `role_kv_shuffle` | 0.1510 | +0.0000 |

For `private_view_qa_token_hard_aux_every`, mean rollout was `0.1458`:

| ablation | rollout mean | delta |
|---|---:|---:|
| `action_answer_shuffle` | 0.1406 | -0.0052 |
| `cross_task_kv_shuffle` | 0.1510 | +0.0052 |
| `mask_actor_peer_reads` | 0.1875 | +0.0417 |
| `remove_path` | 0.1875 | +0.0417 |
| `role_kv_shuffle` | 0.1406 | -0.0052 |

These ablations fail the mechanism test. If removing the path or masking peer reads does not hurt, the token relay is not being used robustly.

## Conclusion

Stage 11G explored the cleaner GPT-token formulation, but it is not the active lead.

- Full-view action-token methods can learn, so the token insertion itself is not broken.
- Private-view pure token, hybrid token, and decision-token Q/A relays did not recover the Stage 11F dependence signal.
- The best decision-token private Q/A result, `0.1510`, barely beat its token no-path control, `0.1250`, and stayed far below Stage 11F's direct-bias Q/A result from the 5-seed read, `0.2875`.
- Mechanism ablations were weak or backwards.

Current active lead remains Stage 11F:

```text
private role views
  + action-conditioned specialist answers
  + direct action-specific integration
```

The next architecture should not simply add more GPT tokens. It should preserve action-specific routing but make it less hand-wired, for example by using a small differentiable action board/head that consumes the four answer contexts, or by moving to a task where the answer labels are exact sufficient statistics for the action decision.
