"""Smoke tests for the E32 disturbance-profile integration."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

from experiments.DIGIT.Extrapolation.e32.extrapolation.config import E32Config
from experiments.DIGIT.Extrapolation.e32.extrapolation.data.ground_truth import OUTCOME_LABELS
from experiments.DIGIT.Extrapolation.e32.extrapolation.losses import E32Loss
from experiments.DIGIT.Extrapolation.e32.extrapolation.models.digit import E32Model
from experiments.DIGIT.Extrapolation.e32.extrapolation.models.readout import (
    bucketize_commitment_decisions,
)
from experiments.DIGIT.Extrapolation.e32.extrapolation.schema import TRACE_INPUT_DIM
from experiments.DIGIT.Extrapolation.e32.scripts.run_experiment import _evaluate


def _make_queries(config: E32Config, batch_size: int) -> torch.Tensor:
    columns = [
        torch.randint(0, vocab_size, (batch_size,), dtype=torch.long)
        for vocab_size in config.field_vocab_sizes
    ]
    return torch.stack(columns, dim=1)


def test_disturbance_profile_smoke_pass() -> None:
    torch.manual_seed(0)
    config = E32Config(
        device="cpu",
        d_model=16,
        nhead=4,
        num_encoder_layers=1,
        d_ff=32,
        dropout=0.0,
        num_diffusion_steps=3,
        num_cloud_samples=2,
        diffusion_hidden_dim=32,
        diffusion_target_hidden_dim=24,
        diffusion_dropout=0.0,
        point_reader_mode="identity",
        point_reader_hidden_dim=16,
        point_reader_dropout=0.0,
        batch_size=2,
        enable_path_uncertainty=False,
        enable_diagnostic_probes=False,
        log_pre_reader_geometry=False,
        enable_disturbance_profile=True,
        enable_boundary_probe=True,
        enable_feature_mask_probe=True,
        enable_learned_probe=False,
        detach_profile_from_base=True,
        profile_risk_input_mode="axis_scores",
        enable_support_novelty=True,
        profile_support_k=1,
        profile_support_max_items=4,
        enable_profile_normalization=True,
        use_disturbance_risk_in_commitment_gate=True,
        lambda_profile_risk=100.0,
        profile_risk_temperature=1.0,
        target_commit_rate_min=0.20,
        target_commit_rate_max=0.80,
        disturbance_num_samples=2,
        disturbance_noise_std=0.05,
        disturbance_feature_mask_prob=0.15,
        disturbance_boundary_epsilon=0.10,
        disturbance_profile_hidden_dim=32,
        disturbance_profile_dropout=0.0,
        disturbance_learned_hidden_dim=16,
        profile_loss_weight=0.1,
        risk_loss_weight=1.0,
        risk_target_mode="incorrect_only",
        tau_uncertain_init=-1000.0,
        commit_risk_threshold=0.0,
    )

    batch_size = 2
    queries = _make_queries(config, batch_size)
    trace_inputs = torch.randn(batch_size, TRACE_INPUT_DIM)
    evidence_targets = torch.zeros(batch_size, 5, dtype=torch.float32)
    prim_targets = torch.zeros(batch_size, 4, dtype=torch.long)
    prim_targets[:, 3] = torch.tensor([0, 2], dtype=torch.long)
    target_ids = torch.arange(batch_size, dtype=torch.long)

    model = E32Model(config)
    _ = model(queries, trace_inputs=trace_inputs)
    assert model.disturbance_profile is not None
    with torch.no_grad():
        final_linear = model.disturbance_profile.aggregator.net[-1]
        assert isinstance(final_linear, torch.nn.Linear)
        final_linear.weight.zero_()
        final_linear.bias.fill_(10.0)

    outputs = model(queries, trace_inputs=trace_inputs)
    criterion = E32Loss(config)
    losses = criterion(outputs, prim_targets, evidence_targets)

    assert outputs.clean_logits is not None
    assert outputs.clean_logits.shape == (batch_size, len(OUTCOME_LABELS))
    assert outputs.risk_logit is not None
    assert outputs.risk_logit.shape == (batch_size,)
    assert outputs.risk_prob is not None
    assert outputs.risk_prob.shape == (batch_size,)
    assert outputs.calibrated_risk is not None
    assert outputs.calibrated_risk.shape == (batch_size,)
    assert outputs.disturbance_profile is not None
    assert outputs.disturbance_profile.profile.shape[0] == batch_size
    assert len(outputs.disturbance_profile.feature_names) == outputs.disturbance_profile.profile.shape[1]
    assert outputs.disturbance_profile.normalized_profile.shape == outputs.disturbance_profile.profile.shape
    assert len(outputs.disturbance_profile.feature_specs) == len(outputs.disturbance_profile.feature_names)
    assert "support/memory_available" in outputs.disturbance_profile.feature_names
    assert "support_novelty" in outputs.disturbance_profile.axis_names
    assert outputs.disturbance_profile.axis_scores.shape == (
        batch_size,
        len(outputs.disturbance_profile.axis_names),
    )
    assert outputs.disturbance_profile.risk_profile.shape == outputs.disturbance_profile.axis_scores.shape
    assert outputs.disturbance_profile.risk_profile_names == outputs.disturbance_profile.axis_names
    assert outputs.final_commitment_score is not None
    assert outputs.final_commitment_logit is not None
    assert outputs.final_commitment_threshold is not None
    assert torch.all(outputs.calibrated_risk > 0.5)

    model.zero_grad(set_to_none=True)
    losses["risk"].backward(retain_graph=True)
    profile_grads = []
    base_grads = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("disturbance_profile."):
            profile_grads.append(parameter.grad)
        else:
            base_grads.append(parameter.grad)
    assert any(grad is not None for grad in profile_grads), "Expected disturbance-profile grads."
    assert all(grad is None for grad in base_grads), "Risk loss should not backprop into the base model."
    model.zero_grad(set_to_none=True)

    model.disturbance_profile.fit_support_memory(
        outputs.anchor.detach(),
        prim_targets[:, 3],
        torch.tensor([0.0, 1.0], dtype=outputs.anchor.dtype),
    )
    outputs_with_support = model(queries, trace_inputs=trace_inputs)
    assert outputs_with_support.disturbance_profile is not None
    memory_available_idx = outputs_with_support.disturbance_profile.feature_names.index(
        "support/memory_available"
    )
    assert torch.all(
        outputs_with_support.disturbance_profile.profile[:, memory_available_idx] == 1
    )

    base_gate = bucketize_commitment_decisions(
        outputs.decision_inputs["base_global_margin"],
        outputs.decision_inputs["base_commitment_score"],
        outputs.decision_inputs["base_commitment_threshold"],
    )
    final_gate = bucketize_commitment_decisions(
        outputs.decision_inputs["global_margin"],
        outputs.decision_inputs["final_commitment_score"],
        outputs.decision_inputs["final_commitment_threshold"],
    )
    assert torch.equal(final_gate, outputs.commit)
    assert not torch.equal(base_gate, final_gate)
    assert torch.isfinite(losses["total"])

    losses["total"].backward()
    trainable_grads = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert trainable_grads, "Expected at least one trainable parameter to receive a gradient."

    lazy_model = E32Model(config)
    lazy_status_before = lazy_model.lazy_module_status()
    assert lazy_status_before["uninitialized_parameters"]
    lazy_status_after = lazy_model.materialize_lazy_modules(device=torch.device("cpu"))
    assert not lazy_status_after["uninitialized_parameters"]
    assert not lazy_status_after["uninitialized_buffers"]
    restored_state = copy.deepcopy(lazy_model.state_dict())
    lazy_model.load_state_dict(restored_state)

    dataset = TensorDataset(
        queries,
        trace_inputs,
        evidence_targets,
        prim_targets,
        target_ids,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    records = [{"is_correct": bool(idx == 0)} for idx in range(batch_size)]
    eval_metrics, _ = _evaluate(model, criterion, loader, torch.device("cpu"), records, config)

    sweep = eval_metrics["risk_gate_threshold_sweep"]
    assert len(sweep["rows"]) == 9
    for row in sweep["rows"]:
        assert "coverage" in row
        assert "committed_accuracy" in row
        assert "success_recall" in row
        assert "failure_recall" in row
        assert "uncertain_recall" in row
        assert "outcome_macro_f1" in row
    success_diag = eval_metrics["success_recovery_diagnostics"]
    assert "blockers" in success_diag
    assert "dominant_reason" in success_diag
    disturbance = eval_metrics["disturbance_profile"]
    assert "calibrated_risk_mean" in disturbance
    assert "threshold_sweep" in disturbance
    assert "selected_threshold" in disturbance["threshold_sweep"]
    assert "axis_score_means" in disturbance
    assert "support_novelty" in disturbance["axis_score_means"]
    assert "profile_state" in disturbance
    assert disturbance["profile_state"]["support_memory_size"] == batch_size
    assert "uncertainty_axis_diagnostics" in eval_metrics
