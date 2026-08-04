# 21EYES Experiment 6: Activation-Conditioned Inter-Layer Routing

## Hypothesis

The experiment tests whether a transformer layer can produce a causal routing signal from its hidden activations, and whether the next layer can use that signal as a soft filtering prior over its own key/value field.

Positive evidence requires the `inter_layer_route_gate` variant to improve accuracy under clutter and longer context while showing non-trivial learned gates. Accuracy alone is not enough: the route gate must beat parameter-matched and random-gate controls, rank useful evidence above distractors, move alpha away from zero, and specialize across layers.

## Why This Is Not KV Eviction or KV Sharing

This experiment does not remove KV entries, compress KV entries, share KV tensors across layers, or pass previous-layer values into later layers. Every attention layer still computes its own Q, K, and V from its own current hidden stream.

The only cross-layer signal is a learned routing descriptor from layer `l` that biases layer `l+1` attention logits before softmax. The value payload still comes from `V_next`, not from the previous layer.

## Architecture

Baseline attention:

```text
base_logits = Q_next @ K_next.T / sqrt(d_head)
weights = softmax(causal_mask(base_logits))
out = weights @ V_next
```

Inter-layer routed attention:

```text
base_logits = Q_next @ K_next.T / sqrt(d_head)
r_prev = route(h_prev_layer)
U_next = hidden_next @ W_U
gate_logits = r_prev @ U_next.T / sqrt(d_route)
gate = sigmoid(gate_logits + gate_bias)
routed_logits = base_logits + alpha * log(gate + eps)
weights = softmax(causal_mask(routed_logits))
out = weights @ V_next
```

Implementation details:

- `alpha` is a learned scalar per layer, initialized to `0.0`.
- `gate_bias` is initialized to `5.0`, so gates start near open.
- Layer 0 has no previous-layer route in `inter_layer_route_gate`.
- The causal mask is applied to the routed logits, and future gate entries are set open before the log-bias is added.
- The model never receives diagnostic relevant-token positions during training.

## Variants

- `baseline`: normal decoder-only causal transformer.
- `param_matched`: baseline plus non-routing MLP adapters with roughly comparable extra parameters.
- `same_layer_u_gate`: same-layer learned gate using current-layer hidden descriptors.
- `inter_layer_route_gate`: main proposed previous-layer activation-conditioned route gate.
- `prev_attn_reuse`: adds the previous layer attention distribution as a detached soft prior.
- `random_gate`: same route-gate architecture as the proposed model, but route and U projection parameters are frozen randomly.

## Synthetic Tasks

The generator creates query-answer language-model examples with randomized query positions and post-query clutter. The answer token is used only as the training target at the query `?` position; it is not inserted as a visible answer token.

Tasks:

- Password overwrite: valid `SET key value` facts mixed with target-key invalid facts and unrelated valid facts.
- Variable shadowing: repeated assignments to a variable mixed with fake or irrelevant assignments.
- Multi-hop lookup: two-hop `A -> B`, `B -> value` chains mixed with fake links and unrelated links.
- Matched distractors: real and fake keys with overwrite-like distractors.

Validation uses new random seeds, held-out key/value combinations, longer lengths, changed distractor counts, changed overwrite depths, and different evidence/query positions. Diagnostic useful and distractor positions are returned only for evaluation metrics.

## How To Run

Activate the intended environment first:

```bash
conda activate tnt
cd experiments/21EYES/e6
bash run_all.sh
```

Single variant:

```bash
python train.py --config configs/inter_layer_route_gate.yaml
```

Fast smoke run:

```bash
python train.py --config configs/inter_layer_route_gate.yaml --device cpu --max-steps 2 --eval-examples 8 --batch-size 2
python eval.py --report
```

The default configs use one seed (`seed: 0`) to keep the initial 6-variant run practical on a single 16 GB GPU. For stronger evidence, rerun each config with `--seed 1` and `--seed 2`, then regenerate the report:

```bash
python train.py --config configs/inter_layer_route_gate.yaml --seed 1
python train.py --config configs/inter_layer_route_gate.yaml --seed 2
python eval.py --report
```

## Outputs

Each run writes:

- `results/<variant>/<seed>/metrics.jsonl`
- `results/<variant>/<seed>/final_metrics.json`
- `results/<variant>/<seed>/config.yaml`
- `results/<variant>/<seed>/model_summary.txt`

The report writer creates:

- `results/report.md`
- `plots/accuracy_by_variant.png`
- `plots/accuracy_vs_seq_len.png`
- `plots/accuracy_vs_distractors.png`
- `plots/gate_mean_by_layer.png`
- `plots/gate_entropy_by_layer.png`
- `plots/attention_entropy_by_layer.png`
- `plots/alpha_by_layer.png`

## Interpreting Results

Positive evidence:

- `inter_layer_route_gate` beats `baseline`, `param_matched`, `same_layer_u_gate`, and `random_gate`.
- Gains are largest in high-distractor or longer-context profiles.
- Useful evidence positions have better gate rank than distractor positions.
- Alpha moves away from zero.
- Gate mean decreases from the near-open initialization.
- Deeper layers show lower attention entropy or clearer gate specialization.

Negative evidence:

- Alpha stays near zero.
- Gates stay open near one.
- Gains disappear against `param_matched`.
- `random_gate` performs similarly.
- Improvements appear only in-distribution and collapse at longer lengths.
- Useful-token gate rank is no better than distractor-token gate rank.

The conclusion in `results/report.md` is intentionally conservative. If any required control is missing, the report marks the result inconclusive rather than inferring support from partial runs.
