# Stage 11F Private-View Q/A Relay Combined 5-Seed Read

## Question

Can the AOB architecture be changed so agents rely on each other for success, rather than hoping same-input role clones spontaneously specialize?

Stage 11F tested the architectural answer suggested by the "twenty questions" analogy: give roles different private views, force the Actor to ask action-conditioned questions, and route only constrained answers back into the Actor before action emission.

## Architecture Added

Implemented in `code/experiment11_gridworld_latent_attention.py`.

- Private role views:
  - Planner sees goal/history/current, but not obstacle cells.
  - Reader sees obstacle cells/current/history, but not goal.
  - Critic sees obstacle cells/current/history, but not goal.
  - Actor sees current/history, but not goal or obstacle cells.
- Q/A relay:
  - Actor forms one latent question per candidate action.
  - Planner/Reader/Critic answer each action query with `YES`, `NO`, `IRRELEVANT`, or `UNKNOWN`.
  - Hard variants use straight-through Gumbel answers during training and argmax answers at eval.
  - Auxiliary-answer variants train Planner toward goal-direction answers and Reader/Critic toward map-validity answers.
- Bias-preserving Q/A:
  - The first relay averaged action-specific answers into one hidden update and mostly failed.
  - The stronger relay keeps a per-action answer context and adds a learned action-logit bias before action selection.

## Variants Tried

- `private_view_no_path`
- `private_view_cross_causal_mean`
- `private_view_qa_soft`
- `private_view_qa_hard`
- `private_view_qa_hard_aux`
- `private_view_qa_hard_aux_late`
- `private_view_qa_hard_aux_every`
- `private_view_qa_soft_bias`
- `private_view_qa_hard_aux_bias`
- `private_view_qa_hard_aux_bias_late`
- `private_view_qa_hard_aux_bias_every`
- `full_view_qa_hard_aux`

Three screens were run:

- Tiny smoke: `results/stage11f_private_qa_tiny_smoke`
- 8-epoch screens: `results/stage11f_private_qa_screen`, `results/stage11f_private_qa_bias_screen`
- 16-epoch key-method retest plus two extra seeds: `results/stage11f_private_qa_retest16`, `results/stage11f_private_qa_retest16_extra_seeds`

## Combined 5-Seed Result

The combined 5-seed read uses seeds `101..105` for the key 16-epoch methods.

| method | seeds | rollout success mean | rollout std | raw rollout |
|---|---:|---:|---:|---|
| `single_actor` | 5 | 0.3750 | 0.0576 | 0.3281, 0.4219, 0.2969, 0.3750, 0.4531 |
| `role_clones_no_path` | 5 | 0.3750 | 0.0576 | 0.3281, 0.4219, 0.2969, 0.3750, 0.4531 |
| `private_view_no_path` | 5 | 0.1125 | 0.0182 | 0.0938, 0.1094, 0.1406, 0.1250, 0.0938 |
| `private_view_cross_causal_mean` | 5 | 0.2812 | 0.0625 | 0.2500, 0.2969, 0.2031, 0.2656, 0.3906 |
| `private_view_qa_hard_aux_bias` | 5 | 0.2875 | 0.0449 | 0.3125, 0.3594, 0.2344, 0.2500, 0.2812 |

## Mechanism Ablations

For `private_view_qa_hard_aux_bias`, mean rollout success was `0.2875`. Eval-time ablations:

| ablation | rollout mean | delta |
|---|---:|---:|
| `action_answer_shuffle` | 0.0469 | -0.2406 |
| `cross_task_kv_shuffle` | 0.0969 | -0.1906 |
| `remove_path` | 0.1187 | -0.1687 |
| `mask_actor_peer_reads` | 0.1219 | -0.1656 |
| `role_kv_shuffle` | 0.2156 | -0.0719 |

For `private_view_cross_causal_mean`, mean rollout success was `0.2812`. Eval-time ablations:

| ablation | rollout mean | delta |
|---|---:|---:|
| `cross_task_kv_shuffle` | 0.0625 | -0.2188 |
| `mask_actor_peer_reads` | 0.0875 | -0.1938 |
| `remove_path` | 0.0969 | -0.1844 |
| `role_kv_shuffle` | 0.2125 | -0.0688 |

## What This Shows

- Private-view masking successfully created dependence. The Actor-blind no-path baseline collapsed from full-view `0.3750` to `0.1125`.
- Peer communication recovered much of the missing performance. Continuous private peer reads reached `0.2812`; the best constrained Q/A relay reached `0.2875`.
- The fixed Q/A relay only worked after preserving action identity into the logits. The hidden-update-only Q/A relay did not beat private no-path reliably.
- The best Q/A relay has real mechanism evidence: removing the path, masking Actor peer reads, shuffling cross-task answers, and especially shuffling action answers all sharply reduced rollout success.
- This is the first AOB branch in this folder where the architecture actually made agents rely on each other for success under ablation.

## What This Does Not Show

- It does not beat the full-view single-agent or full-view no-path controls. This is a task-redesign branch, not a rescue of the original same-observation role-clone claim.
- The best Q/A variant uses auxiliary answer supervision and an action-logit bias. That is valid architecture pressure toward cooperation, but it is not spontaneous emergent specialization.
- The Q/A advantage over continuous private peer read is tiny at 5 seeds (`+0.0063`). The robust claim is dependence versus private no-path, not Q/A superiority over continuous communication.
- The backbone is still a small randomly initialized GPT-style decoder, not a pretrained causal decoder.

## Current Conclusion

The architectural solution for AOB is not "more attention among identical agents." It is:

```text
private role views
  + action-conditioned latent questions
  + low-bandwidth specialist answers
  + Actor-only final action
  + ablations that destroy the answer path
```

This creates the specialization pressure the earlier stages lacked. The next version should replace the direct action-logit bias with explicit action query tokens inside the GPT stream, or test a cleaner compositional task where Planner and Map evidence are both necessary on most episodes.
