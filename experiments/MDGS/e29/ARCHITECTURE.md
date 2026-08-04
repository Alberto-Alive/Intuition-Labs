# E29 Architecture Note

`e29` is a strict uncertainty-geometry test.

Decision path:
- The encoder emits one anchor latent `a` per example.
- A conditional latent diffusion model is trained around `a`.
- The readout samples a denoised cloud `z^(1) ... z^(K)` from that diffusion model.
- The public decision uses only cloud geometry:
  - `mu = mean(z_k)`
  - `var = per-dimension variance(z_k)`
  - `radius = mean(||z_k - mu||^2)`
  - `agreement = mean cosine(z_k, mu)`
- The outcome head is:
  - `m = linear(mu)`
  - `u = softplus(alpha * radius + beta * mean(var) - gamma * abs(m) + b)`
  - `success = m - u`
  - `uncertain = u`
  - `failure = -m - u`

Removed bypasses from `e10`:
- No `executor_features` path in the model.
- No `shared_joint` concat of encoder and executor states.
- No `path_evidence_head` feeding public logits.
- No `stability_cert_head`, `support_cert_head`, or `success_guard_head` in the decision path.
- No outcome head that concatenates trajectory or pattern probabilities.
- No `head_mode` rescue variants such as `mlp`, `linear_evidence_only`, or `ordered_threshold`.
- No prototype-family support, conflict, margin, or anchor-switch features feeding outcome logits.
- No raw anchor concatenation into the final readout.
- No early bucketization inside the model forward path.

Optional diagnostics:
- `order`, `boundary`, and `support` probes can be enabled.
- Those probes are disconnected from the public decision logits.

New modules:
- `extrapolation/config.py`
- `extrapolation/schema.py`
- `extrapolation/data/ground_truth.py`
- `extrapolation/data/trace_dataset.py`
- `extrapolation/data/vocabulary.py`
- `extrapolation/models/diffusion.py`
- `extrapolation/models/encoder.py`
- `extrapolation/models/readout.py`
- `extrapolation/models/digit.py`
- `extrapolation/losses.py`
- `extrapolation/metrics.py`
- `extrapolation/trace_pipeline.py`
- `scripts/run_experiment.py`
- `tests/test_e29_geometry_invariants.py`

Assumptions:
- `e29` reads its own local `trace_cache` and does not import the `e10` trace pipeline.
- The current standalone entrypoint expects `e29/trace_cache/{train,val,test}.jsonl` to exist.
- Trace inputs supervise only the diffusion denoising loss; they do not enter the decision logits.
- Reverse sampling is deterministic given the initial cloud noise, which keeps the invariants reproducible.
