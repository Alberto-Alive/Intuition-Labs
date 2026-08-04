"""Invariant tests for the E29 strict uncertainty-geometry architecture."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

from experiments.DIGIT.Extrapolation.e29.extrapolation.config import E29Config
from experiments.DIGIT.Extrapolation.e29.extrapolation.metrics import bucketize_outcome_logits
from experiments.DIGIT.Extrapolation.e29.extrapolation.schema import TRACE_INPUT_DIM
from experiments.DIGIT.Extrapolation.e29.extrapolation.models.digit import E29Model
from experiments.DIGIT.Extrapolation.e29.extrapolation.models.readout import (
    CloudGeometryReadout,
    CloudGeometryStats,
)


def _make_config(**overrides) -> E29Config:
    config = E29Config(
        device="cpu",
        d_model=16,
        nhead=4,
        num_encoder_layers=1,
        d_ff=32,
        dropout=0.0,
        num_diffusion_steps=3,
        cloud_num_samples=4,
        diffusion_hidden_dim=32,
        trace_target_hidden_dim=24,
        diffusion_dropout=0.0,
        enable_diagnostic_probes=False,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _synthetic_queries(config: E29Config, batch_size: int) -> torch.Tensor:
    columns = [
        torch.randint(low=0, high=vocab_size, size=(batch_size,), dtype=torch.long)
        for vocab_size in config.field_vocab_sizes
    ]
    return torch.stack(columns, dim=1)


def _synthetic_trace_inputs(batch_size: int) -> torch.Tensor:
    return torch.randn(batch_size, TRACE_INPUT_DIM)


def test_forward_pass_smoke_has_expected_shapes() -> None:
    torch.manual_seed(0)
    config = _make_config()
    model = E29Model(config)
    batch_size = 3
    queries = _synthetic_queries(config, batch_size=batch_size)
    trace_inputs = _synthetic_trace_inputs(batch_size=batch_size)

    outputs = model(queries, trace_inputs=trace_inputs)

    assert outputs.anchor.shape == (batch_size, config.d_model)
    assert outputs.cloud_samples.shape == (config.cloud_num_samples, batch_size, config.d_model)
    assert outputs.cloud_stats.mu.shape == (batch_size, config.d_model)
    assert outputs.cloud_stats.var.shape == (batch_size, config.d_model)
    assert outputs.cloud_stats.radius.shape == (batch_size,)
    assert outputs.cloud_stats.agreement.shape == (batch_size,)
    assert outputs.outcome_logits.shape == (batch_size, 3)
    assert outputs.diffusion_denoise_loss.ndim == 0


def test_final_logits_are_computable_from_cloud_stats_only() -> None:
    torch.manual_seed(1)
    config = _make_config()
    model = E29Model(config)
    queries = _synthetic_queries(config, batch_size=2)
    trace_inputs = _synthetic_trace_inputs(batch_size=2)
    outputs = model(queries, trace_inputs=trace_inputs)

    recomputed = model.readout(outputs.cloud_stats)
    assert torch.allclose(outputs.outcome_logits, recomputed.outcome_logits, atol=1e-6)
    assert torch.allclose(outputs.margin, recomputed.margin, atol=1e-6)
    assert torch.allclose(outputs.uncertainty, recomputed.uncertainty, atol=1e-6)


def test_no_raw_trace_executor_or_probe_tensor_is_wired_into_final_logits() -> None:
    torch.manual_seed(2)
    config = _make_config(enable_diagnostic_probes=True)
    model = E29Model(config)
    queries = _synthetic_queries(config, batch_size=2)
    fixed_sampling_noise = torch.randn(config.cloud_num_samples, 2, config.d_model)
    fixed_training_noise = torch.randn(2, config.d_model)
    fixed_timesteps = torch.tensor([1, 3], dtype=torch.long)
    trace_inputs_a = torch.randn(2, TRACE_INPUT_DIM)
    trace_inputs_b = torch.randn(2, TRACE_INPUT_DIM) * 9.0

    outputs_a = model(
        queries,
        trace_inputs=trace_inputs_a,
        sampling_noise=fixed_sampling_noise,
        diffusion_training_noise=fixed_training_noise,
        diffusion_training_timesteps=fixed_timesteps,
    )
    outputs_b = model(
        queries,
        trace_inputs=trace_inputs_b,
        sampling_noise=fixed_sampling_noise,
        diffusion_training_noise=fixed_training_noise,
        diffusion_training_timesteps=fixed_timesteps,
    )

    assert torch.allclose(outputs_a.outcome_logits, outputs_b.outcome_logits, atol=1e-6)
    assert torch.allclose(outputs_a.margin, outputs_b.margin, atol=1e-6)
    assert torch.allclose(outputs_a.uncertainty, outputs_b.uncertainty, atol=1e-6)
    assert outputs_a.diffusion_denoise_loss.item() != outputs_b.diffusion_denoise_loss.item()
    assert set(outputs_a.decision_inputs) == {"mu", "radius", "mean_var", "agreement", "margin_abs"}
    readout_signature = inspect.signature(model.readout.forward)
    assert list(readout_signature.parameters) == ["stats"]
    assert not hasattr(model, "executor")


def test_increasing_synthetic_radius_increases_uncertainty_logit() -> None:
    torch.manual_seed(3)
    readout = CloudGeometryReadout(latent_dim=4)
    mu = torch.tensor([[0.2, -0.1, 0.3, -0.4]], dtype=torch.float32)
    var = torch.zeros_like(mu)

    low_radius_stats = CloudGeometryStats(
        mu=mu,
        var=var,
        radius=torch.tensor([0.05]),
        agreement=torch.tensor([0.9]),
        mean_var=torch.tensor([0.0]),
    )
    high_radius_stats = CloudGeometryStats(
        mu=mu,
        var=var,
        radius=torch.tensor([1.50]),
        agreement=torch.tensor([0.9]),
        mean_var=torch.tensor([0.0]),
    )

    low = readout(low_radius_stats)
    high = readout(high_radius_stats)
    assert high.uncertainty.item() > low.uncertainty.item()
    assert high.outcome_logits[0, 1].item() > low.outcome_logits[0, 1].item()


def test_bucketization_is_late_only() -> None:
    torch.manual_seed(4)
    config = _make_config()
    model = E29Model(config)
    queries = _synthetic_queries(config, batch_size=2)
    outputs = model(queries, trace_inputs=_synthetic_trace_inputs(batch_size=2))

    assert not hasattr(outputs, "predicted_outcome")
    assert not hasattr(outputs, "outcome_discrete")
    bucket_ids = bucketize_outcome_logits(outputs.outcome_logits)
    assert bucket_ids.shape == (2,)
    assert bucket_ids.dtype == torch.long


def test_probe_heads_are_disconnected_from_final_decision_logits() -> None:
    torch.manual_seed(5)
    config = _make_config(enable_diagnostic_probes=True)
    model = E29Model(config)
    queries = _synthetic_queries(config, batch_size=2)
    trace_inputs = _synthetic_trace_inputs(batch_size=2)
    fixed_sampling_noise = torch.randn(config.cloud_num_samples, 2, config.d_model)
    fixed_training_noise = torch.randn(2, config.d_model)
    fixed_timesteps = torch.tensor([2, 1], dtype=torch.long)

    before = model(
        queries,
        trace_inputs=trace_inputs,
        sampling_noise=fixed_sampling_noise,
        diffusion_training_noise=fixed_training_noise,
        diffusion_training_timesteps=fixed_timesteps,
    )

    with torch.no_grad():
        for parameter in model.probe_heads.parameters():
            parameter.add_(torch.randn_like(parameter) * 50.0)

    after = model(
        queries,
        trace_inputs=trace_inputs,
        sampling_noise=fixed_sampling_noise,
        diffusion_training_noise=fixed_training_noise,
        diffusion_training_timesteps=fixed_timesteps,
    )

    assert torch.allclose(before.outcome_logits, after.outcome_logits, atol=1e-6)
    assert torch.allclose(before.margin, after.margin, atol=1e-6)
    assert torch.allclose(before.uncertainty, after.uncertainty, atol=1e-6)
    assert before.diagnostic_probes is not None
    assert after.diagnostic_probes is not None
    assert not torch.allclose(before.diagnostic_probes.order, after.diagnostic_probes.order)


def main() -> None:
    test_forward_pass_smoke_has_expected_shapes()
    test_final_logits_are_computable_from_cloud_stats_only()
    test_no_raw_trace_executor_or_probe_tensor_is_wired_into_final_logits()
    test_increasing_synthetic_radius_increases_uncertainty_logit()
    test_bucketization_is_late_only()
    test_probe_heads_are_disconnected_from_final_decision_logits()
    print("All E29 geometry invariants passed.")


if __name__ == "__main__":
    main()
