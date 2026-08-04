# E24 Summary

## 1. Question

Does a learned scalar gate — included in the gradient optimizer alongside all
semantic parameters — allow the familiarity attention pathway to open and
contribute to withheld-quadrant extrapolation when the model determines it is
useful?

## 2. Method

E24 used the same three-model structure as E23: a matched baseline MLP, the
E24 dual-parameter model with experience-updated prototype memory, and a
learned-memory ablation with gradient-trained slots. The only change from E23
was the addition of one learned scalar gate per layer, initialised to −5.0,
modulating the familiarity attention contribution as a sigmoid-gated residual
added to the semantic output. Both gates were included in the same Adam
optimizer as all semantic parameters, giving the model full freedom to open
the familiarity pathway if it provided a gradient signal.

## 3. Primary Result

FAIL. E24 mean withheld macro accuracy 0.9363 versus baseline 0.9471. Mean
delta −0.0107 in the wrong direction. E24 marginally outperforms the ablation
— mean delta +0.0053 — but this is within seed-to-seed noise (population std
0.0137) and does not constitute a meaningful separation.

## 4. Gate Values

Gates remained nearly closed across all five seeds for both models. Final
sigmoid values ranged from 0.011 to 0.042, compared to the initialisation
value of 0.0067. Layer 1 was consistently more open than layer 2 in every
seed for both models — mean layer 1 minus layer 2 gap of +0.866 raw for E24
and +0.810 for the ablation. The model correctly learned that the familiarity
pathway had nothing useful to contribute to this task and held the gates
nearly shut throughout training.

## 5. Per-Combination Breakdown

The three failing combinations — (1,4), (3,4), (3,5) — failed equally across
all three arms. These failures are caused by compositional coverage limits in
the training data, not by the familiarity system. Familiarity neither helped
nor hurt at the boundary combinations. The (3,5) combination sat at mean
accuracy 0.600 for every arm. Columns b=6 and b=7 were at ceiling for every
arm throughout.

## 6. Conclusion Across E17–E24

Every attempt to use familiarity to improve learning or generalisation on
synthetic compositional tasks has failed. Not because familiarity is wrong but
because the tasks are cleanly learnable by semantic weights alone. The
familiarity system has nothing to add when the semantic system can solve the
task without help.

## 7. What Is Proven and What Is Not

Proven — the familiarity signal is real, graded, intrinsic, and updates at
inference. Not proven — familiarity improving learning, generalisation, or
compositional reasoning on tasks the semantic system can solve alone.

## 8. Preregistered Next Investigation

The relationship between familiarity density and the Lottery Ticket
Hypothesis. Whether high familiarity prototype regions correspond to winning
ticket weights — the weights that survive pruning — and whether familiarity
density during training predicts which weights will be in the winning ticket
before full training is complete.
