"""Diagnostics tests for the E31 cooperative witness/path experiment."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

from experiments.DIGIT.Extrapolation.e31.extrapolation.config import E31Config
from experiments.DIGIT.Extrapolation.e31.extrapolation.data.ground_truth import OUTCOME_LABELS
from experiments.DIGIT.Extrapolation.e31.extrapolation.data.trace_dataset import (
    TraceWitnessDataset,
    load_trace_records,
    trace_collate_fn,
)
from experiments.DIGIT.Extrapolation.e31.extrapolation.data.vocabulary import Vocabulary
from experiments.DIGIT.Extrapolation.e31.extrapolation.losses import E31Loss
from experiments.DIGIT.Extrapolation.e31.extrapolation.models.digit import E31Model
from experiments.DIGIT.Extrapolation.e31.extrapolation.models.path_uncertainty import (
    LatentPathUncertainty,
)
from experiments.DIGIT.Extrapolation.e31.extrapolation.models.point_reader import (
    IdentityPointReader,
    LinearScalarMarginReader,
    LinearSharedProjPointReader,
)
from experiments.DIGIT.Extrapolation.e31.extrapolation.models.readout import (
    WitnessAgreementReadout,
    WitnessSetStats,
    bucketize_hard_commitment_decisions,
    bucketize_outcome_logits,
)
from experiments.DIGIT.Extrapolation.e31.scripts.run_experiment import (
    _build_report_highlights,
    _evaluate,
    _primary_metric_value,
    _resolve_output_dir,
    _summary_diagnostic_sections,
)


def _make_config(**overrides) -> E31Config:
    config = E31Config(
        device="cpu",
        d_model=16,
        nhead=4,
        num_encoder_layers=1,
        d_ff=32,
        dropout=0.0,
        num_diffusion_steps=3,
        num_cloud_samples=4,
        diffusion_hidden_dim=32,
        diffusion_target_hidden_dim=24,
        diffusion_dropout=0.0,
        point_reader_mode="identity",
        point_reader_hidden_dim=16,
        point_reader_dropout=0.0,
        batch_size=2,
        enable_diagnostic_probes=False,
        log_pre_reader_geometry=True,
        log_grouped_geometry=True,
        log_weighting_diagnostics=True,
        gate_mode="soft_commitment_gate",
        use_hard_eval_gate=True,
        commitment_scale_init=1.0,
        tau_uncertain_init=0.0,
        enable_path_uncertainty=True,
        path_uncertainty_embedding_dim=8,
        path_uncertainty_hidden_dim=16,
        path_consistency_loss_weight=0.05,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _synthetic_queries(config: E31Config, batch_size: int) -> torch.Tensor:
    columns = [
        torch.randint(low=0, high=vocab_size, size=(batch_size,), dtype=torch.long)
        for vocab_size in config.field_vocab_sizes
    ]
    return torch.stack(columns, dim=1)


def _synthetic_trace_inputs(batch_size: int) -> torch.Tensor:
    return torch.randn(batch_size, 19)


def _make_eval_setup(config: E31Config, limit: int = 2):
    trace_dir = Path(__file__).resolve().parents[2] / "e29" / "trace_cache"
    records = load_trace_records(trace_dir / "train.jsonl")[:limit]
    dataset = TraceWitnessDataset(records, vocab=Vocabulary(), config=config)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=trace_collate_fn,
    )
    class_weights = dataset.get_class_weights(
        power=config.trace_class_weight_power,
        clip=config.trace_class_weight_clip,
    )["outcome"]
    return records, loader, class_weights


def _make_pairwise(num_samples: int, offdiag_value: float) -> torch.Tensor:
    pairwise = torch.full((1, num_samples, num_samples), float(offdiag_value), dtype=torch.float32)
    diag = torch.arange(num_samples)
    pairwise[:, diag, diag] = 1.0
    return pairwise


def _make_stats(margins: list[float], offdiag_value: float = 1.0) -> WitnessSetStats:
    margins_t = torch.tensor([margins], dtype=torch.float32)
    num_samples = margins_t.shape[-1]
    weights = torch.full_like(margins_t, 1.0 / float(num_samples))
    pairwise_agreement = _make_pairwise(num_samples, offdiag_value)
    if num_samples > 1:
        coherence_scores = (pairwise_agreement.sum(dim=-1) - 1.0) / float(num_samples - 1)
        offdiag_mask = ~torch.eye(num_samples, dtype=torch.bool)
        pairwise_values = pairwise_agreement[:, offdiag_mask].reshape(1, -1)
        pairwise_mean = pairwise_values.mean(dim=-1)
        pairwise_std = pairwise_values.std(dim=-1, unbiased=False)
    else:
        coherence_scores = torch.zeros_like(margins_t)
        pairwise_mean = torch.ones(1)
        pairwise_std = torch.zeros(1)
    global_margin = (weights * margins_t).sum(dim=-1)
    centered = margins_t - global_margin.unsqueeze(-1)
    vote_disagreement = (weights * centered.pow(2)).sum(dim=-1)
    witness_margin_mean = margins_t.mean(dim=-1)
    witness_margin_std = margins_t.std(dim=-1, unbiased=False)
    witness_weight_entropy = torch.zeros_like(global_margin)
    if num_samples > 1:
        witness_weight_entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=-1) / math.log(num_samples)
    witness_weight_kl_uniform = torch.zeros_like(global_margin)
    return WitnessSetStats(
        witness_margins=margins_t,
        pairwise_agreement=pairwise_agreement,
        coherence_scores=coherence_scores,
        witness_weights=weights,
        global_margin=global_margin,
        vote_disagreement=vote_disagreement,
        geometric_disagreement=1.0 - pairwise_mean,
        margin_abs=global_margin.abs(),
        witness_margin_mean=witness_margin_mean,
        witness_margin_std=witness_margin_std,
        witness_weight_entropy=witness_weight_entropy,
        witness_weight_max=weights.max(dim=-1).values,
        witness_weight_kl_uniform=witness_weight_kl_uniform,
        pairwise_agreement_mean=pairwise_mean,
        pairwise_agreement_std=pairwise_std,
    )


def _assert_grouped_summary(summary: dict[str, object], expected_count: int) -> None:
    assert "overall" in summary
    assert "by_true_class" in summary
    assert "by_predicted_class" in summary
    assert "by_correctness" in summary
    assert summary["overall"]["count"] == expected_count
    assert set(summary["by_true_class"]) == set(OUTCOME_LABELS)
    assert set(summary["by_predicted_class"]) == set(OUTCOME_LABELS)
    assert set(summary["by_correctness"]) == {"correct", "incorrect"}


def test_point_reader_modes_are_concrete() -> None:
    torch.manual_seed(0)
    identity_config = _make_config(point_reader_mode="identity")
    scalar_config = _make_config(point_reader_mode="linear_scalar_margin")
    shared_config = _make_config(point_reader_mode="linear_shared_proj")

    identity_model = E31Model(identity_config)
    scalar_model = E31Model(scalar_config)
    shared_model = E31Model(shared_config)

    sample = torch.randn(4, 2, identity_config.d_model)

    assert isinstance(identity_model.point_reader, IdentityPointReader)
    assert sum(parameter.numel() for parameter in identity_model.point_reader.parameters()) == 0
    assert torch.equal(identity_model.point_reader(sample), sample)

    assert isinstance(scalar_model.point_reader, LinearScalarMarginReader)
    assert sum(isinstance(module, nn.Linear) for module in scalar_model.point_reader.modules()) == 1
    scalar_output = scalar_model.point_reader(sample)
    assert scalar_output.shape == (4, 2, 1)

    assert isinstance(shared_model.point_reader, LinearSharedProjPointReader)
    assert sum(isinstance(module, nn.Linear) for module in shared_model.point_reader.modules()) == 1
    shared_output = shared_model.point_reader(sample)
    assert shared_output.shape == (4, 2, identity_config.d_model)


def test_path_uncertainty_encodes_two_view_trajectory_stability() -> None:
    torch.manual_seed(7)
    module = LatentPathUncertainty(
        latent_dim=5,
        embedding_dim=4,
        hidden_dim=6,
        contrastive_temperature=0.2,
    )
    trajectory = torch.randn(4, 3, 2, 5)
    same_result = module(trajectory, trajectory)
    different_result = module(trajectory, torch.randn_like(trajectory))

    assert same_result.per_witness_embedding_a.shape == (3, 2, 4)
    assert same_result.per_witness_path_disagreement.shape == (2, 3)
    assert same_result.path_disagreement.shape == (2,)
    assert same_result.path_disagreement.abs().max().item() < 1e-5
    assert torch.isfinite(same_result.contrastive_loss)
    assert torch.all(different_result.path_disagreement >= 0.0)
    assert torch.all(different_result.path_disagreement <= 2.0)


def test_cooperative_readout_raises_uncertainty_and_lowers_commitment_for_unstable_paths() -> None:
    readout = WitnessAgreementReadout(
        latent_dim=4,
        use_path_disagreement=True,
        use_method_conflict=True,
        use_path_commitment_penalty=True,
    )
    stable_stats = _make_stats([10.0, 10.5, 11.0], offdiag_value=0.95)
    unstable_stats = _make_stats([10.0, 10.5, 11.0], offdiag_value=0.95)
    stable_stats.path_disagreement = torch.tensor([0.0], dtype=torch.float32)
    unstable_stats.path_disagreement = torch.tensor([1.0], dtype=torch.float32)

    stable = readout(stable_stats)
    unstable = readout(unstable_stats)

    assert unstable.uncertainty.item() > stable.uncertainty.item()
    assert (
        unstable.decision_inputs["commitment_score"].item()
        < stable.decision_inputs["commitment_score"].item()
    )
    assert unstable.decision_inputs["method_conflict"].item() > stable.decision_inputs["method_conflict"].item()


def test_path_aware_witness_weights_downweight_unstable_paths() -> None:
    torch.manual_seed(8)
    readout = WitnessAgreementReadout(
        latent_dim=4,
        use_path_aware_witness_weights=True,
        path_weight_eta_init=2.0,
    )
    witness_samples = torch.randn(2, 1, 4)
    path_disagreement = torch.tensor([[0.0, 2.0]], dtype=torch.float32)
    stats = readout.summarize(
        witness_samples,
        per_witness_path_disagreement=path_disagreement,
        path_disagreement=torch.tensor([1.0], dtype=torch.float32),
    )

    assert stats.witness_weights[0, 0].item() > stats.witness_weights[0, 1].item()


def test_path_signed_evidence_head_emits_supervised_directional_signals() -> None:
    torch.manual_seed(9)
    config = _make_config(enable_path_signed_evidence=True)
    model = E31Model(config)
    criterion = E31Loss(config)
    batch_size = 2
    queries = _synthetic_queries(config, batch_size)
    trace_inputs = _synthetic_trace_inputs(batch_size)
    prim_targets = torch.zeros(batch_size, 4, dtype=torch.long)
    prim_targets[:, 3] = torch.tensor([0, 2], dtype=torch.long)

    outputs = model(queries, trace_inputs=trace_inputs)
    losses = criterion(outputs, prim_targets)

    assert model.readout.use_path_disagreement
    assert model.readout.use_method_conflict
    assert model.readout.use_path_commitment_penalty
    assert model.readout.use_path_aware_witness_weights
    assert outputs.path_outcome_logits is not None
    assert outputs.path_outcome_logits.shape == (batch_size, len(OUTCOME_LABELS))
    assert outputs.path_failure_logit is not None
    assert outputs.path_failure_logit.shape == (batch_size,)
    assert outputs.base_outcome_logits.shape == outputs.outcome_logits.shape
    assert "path_signed_margin" in outputs.decision_inputs
    assert "path_failure_gate_logit" in outputs.decision_inputs
    assert "path_failure_gate_probability" in outputs.decision_inputs
    assert "path_failure_gate_signal" in outputs.decision_inputs
    assert "base_global_margin" in outputs.decision_inputs
    assert outputs.decision_inputs["global_margin"].shape == (batch_size,)
    assert torch.any(outputs.decision_inputs["path_failure_gate_signal"] > 0)
    assert torch.isfinite(losses["path_outcome"])
    assert torch.isfinite(losses["path_failure"])


def test_path_failure_gate_only_pulls_the_margin_down() -> None:
    torch.manual_seed(10)
    config = _make_config(enable_path_signed_evidence=True)
    model = E31Model(config)
    queries = _synthetic_queries(config, 3)
    trace_inputs = _synthetic_trace_inputs(3)

    outputs = model(queries, trace_inputs=trace_inputs)

    assert outputs.path_failure_logit is not None
    assert torch.all(outputs.decision_inputs["path_failure_gate_signal"] >= 0)
    assert torch.all(outputs.decision_inputs["path_signed_margin"] >= 0)
    assert torch.all(outputs.decision_inputs["path_signed_margin_delta"] <= 1e-6)


def test_forward_pass_emits_reader_geometry_diversity_and_margin_diagnostics() -> None:
    torch.manual_seed(1)
    config = _make_config()
    records, loader, class_weights = _make_eval_setup(config)
    model = E31Model(config)
    criterion = E31Loss(config, outcome_class_weights=class_weights)

    metrics, bundle = _evaluate(model, criterion, loader, torch.device("cpu"), records, config)

    assert metrics["prediction_mode"] == "hard_commitment_gate"
    assert "pre_reader_witness_geometry" in metrics
    assert "post_reader_witness_geometry" in metrics
    assert "diversity_retention" in metrics
    assert "margin_channel_diagnostics" in metrics
    assert "failure_channel_diagnostics" in metrics
    assert "success_channel_diagnostics" in metrics
    assert "path_uncertainty_diagnostics" in metrics
    assert "decision_comparison" in metrics
    assert "decision_delta_diagnostics" in metrics
    assert "signal_forensics" in metrics
    assert "base_signal_forensics" in metrics
    assert "weighting_diagnostics" in metrics
    assert "decision_geometry" in metrics
    assert "commitment_score" in bundle
    assert "tau_uncertain" in bundle
    assert "path_disagreement" in bundle
    assert "per_witness_path_disagreement" in bundle
    assert "base_outcome_logits" in bundle
    assert "base_global_margin" in bundle

    expected_count = len(records)
    for section_name in [
        "pre_reader_witness_geometry",
        "post_reader_witness_geometry",
        "diversity_retention",
        "decision_geometry",
        "margin_channel_diagnostics",
        "path_uncertainty_diagnostics",
        "decision_delta_diagnostics",
        "weighting_diagnostics",
    ]:
        for metric_name, summary in metrics[section_name].items():
            _assert_grouped_summary(summary, expected_count)

    pre_reader = metrics["pre_reader_witness_geometry"]
    assert "pairwise_cosine_agreement" in pre_reader
    assert "mean_squared_distance_to_anchor" in pre_reader
    assert "principal_singular_value_ratio" in pre_reader

    post_reader = metrics["post_reader_witness_geometry"]
    assert "pairwise_cosine_agreement" in post_reader
    assert "witness_weight_entropy" in post_reader
    assert "witness_weight_kl_uniform" in post_reader
    assert "witness_margin_range" in post_reader
    assert "weighted_minus_unweighted_margin" in post_reader

    diversity_retention = metrics["diversity_retention"]
    assert "reader_agreement_gain" in diversity_retention
    assert "reader_margin_std_ratio" in diversity_retention
    assert "reader_weighting_gain" in diversity_retention

    margin_channel = metrics["margin_channel_diagnostics"]
    assert "margin_positive_fraction" in margin_channel
    assert "margin_negative_fraction" in margin_channel
    assert "margin_near_zero_fraction" in margin_channel
    assert "witness_positive_count" in margin_channel
    assert "witness_negative_count" in margin_channel
    assert "witness_sign_majority_fraction" in margin_channel
    assert "witness_sign_disagreement_rate" in margin_channel
    assert "mixed_sign_witness_fraction" in margin_channel
    assert "weighted_margin_sign_flip_rate" in margin_channel
    assert "weighted_minus_unweighted_margin" in margin_channel

    failure_channel = metrics["failure_channel_diagnostics"]
    for metric_name, summary in failure_channel.items():
        assert "overall" in summary, metric_name
    assert "margin_M" in failure_channel
    assert "commitment_score" in failure_channel
    assert "uncertainty_U" in failure_channel
    assert "witness_positive_count" in failure_channel
    assert "witness_negative_count" in failure_channel
    assert "witness_sign_disagreement_rate" in failure_channel
    assert "mixed_sign_witness_fraction" in failure_channel
    assert "true_failure_margin_negative_fraction" in failure_channel
    assert "true_failure_margin_positive_fraction" in failure_channel
    assert "true_failure_commitment_below_tau_fraction" in failure_channel
    assert "true_failure_predicted_success_fraction" in failure_channel
    assert "true_failure_predicted_uncertain_fraction" in failure_channel
    assert "true_failure_predicted_failure_fraction" in failure_channel

    success_channel = metrics["success_channel_diagnostics"]
    for metric_name, summary in success_channel.items():
        assert "overall" in summary, metric_name
    assert "true_success_margin_negative_fraction" in success_channel
    assert "true_success_margin_positive_fraction" in success_channel
    assert "true_success_commitment_below_tau_fraction" in success_channel
    assert "true_success_predicted_success_fraction" in success_channel
    assert "true_success_predicted_uncertain_fraction" in success_channel
    assert "true_success_predicted_failure_fraction" in success_channel

    weighting_diagnostics = metrics["weighting_diagnostics"]
    assert "coherence_score_spread_before_softmax" in weighting_diagnostics
    assert "witness_weight_spread_after_softmax" in weighting_diagnostics
    assert "coherence_margin_corr" in weighting_diagnostics
    assert "coherence_abs_margin_corr" in weighting_diagnostics
    assert "weighted_minus_unweighted_margin" in weighting_diagnostics
    assert "weighted_margin_sign_flip_rate" in weighting_diagnostics

    path_diagnostics = metrics["path_uncertainty_diagnostics"]
    assert "path_disagreement" in path_diagnostics
    assert "path_similarity" in path_diagnostics
    assert "path_confidence" in path_diagnostics
    assert "endpoint_confidence" in path_diagnostics
    assert "method_conflict" in path_diagnostics
    assert "per_witness_path_disagreement_mean" in path_diagnostics
    assert "path_commitment_penalty" in path_diagnostics
    assert "path_disagreement_error_corr" in metrics
    assert "endpoint_path_conflict_error_corr" in metrics

    decision_geometry = metrics["decision_geometry"]
    assert "margin_M" in decision_geometry
    assert "base_margin_M" in decision_geometry
    assert "final_minus_base_margin" in decision_geometry
    assert "vote_disagreement_D" in decision_geometry
    assert "geometric_disagreement_G" in decision_geometry
    assert "uncertainty_U" in decision_geometry
    assert "success_minus_uncertain" in decision_geometry
    assert "failure_minus_uncertain" in decision_geometry
    assert "margin_positive_fraction" in decision_geometry
    assert "margin_negative_fraction" in decision_geometry
    assert "margin_near_zero_fraction" in decision_geometry
    assert "witness_sign_disagreement_rate" in decision_geometry
    assert "mixed_sign_witness_fraction" in decision_geometry
    assert "commitment_score" in decision_geometry
    assert "commitment_logit" in decision_geometry
    assert "path_disagreement" in decision_geometry
    assert "method_conflict" in decision_geometry
    assert "base_hard_gate" in metrics["decision_comparison"]
    assert "final_hard_gate" in metrics["decision_comparison"]
    assert "base_soft_logits" in metrics["decision_comparison"]
    assert "final_soft_logits" in metrics["decision_comparison"]


def test_hard_eval_gate_routes_low_commitment_to_uncertain() -> None:
    global_margin = torch.tensor([1.5], dtype=torch.float32)
    commitment_score = torch.tensor([-0.25], dtype=torch.float32)
    tau_uncertain = torch.tensor([0.0], dtype=torch.float32)

    prediction = bucketize_hard_commitment_decisions(global_margin, commitment_score, tau_uncertain)

    assert prediction.item() == 1


def test_hard_eval_gate_routes_high_commitment_by_margin_sign() -> None:
    tau_uncertain = torch.tensor([0.0], dtype=torch.float32)
    success_prediction = bucketize_hard_commitment_decisions(
        torch.tensor([0.8], dtype=torch.float32),
        torch.tensor([0.7], dtype=torch.float32),
        tau_uncertain,
    )
    failure_prediction = bucketize_hard_commitment_decisions(
        torch.tensor([-0.8], dtype=torch.float32),
        torch.tensor([0.7], dtype=torch.float32),
        tau_uncertain,
    )

    assert success_prediction.item() == 0
    assert failure_prediction.item() == 2


def test_balanced_margin_primary_penalizes_zero_success_regimes() -> None:
    base_metrics = {
        "outcome_macro_f1": 0.72,
        "failure_recall": 0.40,
        "success_recall": 0.35,
        "outcome_share": {
            "SUCCESS_LIKELY": 0.20,
            "UNCERTAIN": 0.50,
            "FAILURE_LIKELY": 0.30,
        },
        "collapse_diagnostics": {
            "dominant_class_share": 0.50,
        },
        "margin_channel_diagnostics": {
            "margin_positive_fraction": {"overall": {"mean": 0.40}},
        },
    }
    zero_success = {
        **base_metrics,
        "outcome_share": {
            "SUCCESS_LIKELY": 0.00,
            "UNCERTAIN": 0.50,
            "FAILURE_LIKELY": 0.50,
        },
        "collapse_diagnostics": {
            "dominant_class_share": 0.50,
        },
        "margin_channel_diagnostics": {
            "margin_positive_fraction": {"overall": {"mean": 0.0}},
        },
    }
    zero_margin = {
        **base_metrics,
        "margin_channel_diagnostics": {
            "margin_positive_fraction": {"overall": {"mean": 0.0}},
        },
    }

    base_score = _primary_metric_value(base_metrics, "balanced_margin_primary")
    zero_success_score = _primary_metric_value(zero_success, "balanced_margin_primary")
    zero_margin_score = _primary_metric_value(zero_margin, "balanced_margin_primary")

    assert zero_success_score < base_score
    assert zero_margin_score < base_score


def test_summary_sections_include_failure_breakdown_fields() -> None:
    fake_metrics = {
        "decision_geometry": {"ok": True},
        "pre_reader_witness_geometry": {"ok": True},
        "post_reader_witness_geometry": {"ok": True},
        "diversity_retention": {"ok": True},
        "margin_channel_diagnostics": {"ok": True},
        "failure_channel_diagnostics": {
            "true_failure_margin_negative_fraction": {"overall": {"mean": 0.75}},
            "true_failure_predicted_failure_fraction": {"overall": {"mean": 0.10}},
            "true_failure_predicted_uncertain_fraction": {"overall": {"mean": 0.80}},
            "true_failure_predicted_success_fraction": {"overall": {"mean": 0.10}},
        },
        "success_channel_diagnostics": {
            "true_success_margin_positive_fraction": {"overall": {"mean": 0.60}},
        },
        "weighting_diagnostics": {"ok": True},
    }

    summary_sections = _summary_diagnostic_sections(fake_metrics)

    assert "failure_channel_diagnostics" in summary_sections
    assert "success_channel_diagnostics" in summary_sections
    assert "true_failure_margin_negative_fraction" in summary_sections["failure_channel_diagnostics"]
    assert "true_failure_predicted_failure_fraction" in summary_sections["failure_channel_diagnostics"]
    assert "true_failure_predicted_uncertain_fraction" in summary_sections["failure_channel_diagnostics"]
    assert "true_failure_predicted_success_fraction" in summary_sections["failure_channel_diagnostics"]


def test_report_highlights_include_failure_breakdown_fields() -> None:
    fake_metrics = {
        "prediction_mode": "hard_commitment_gate",
        "class_shares": {
            "SUCCESS_LIKELY": 0.2,
            "UNCERTAIN": 0.5,
            "FAILURE_LIKELY": 0.3,
        },
        "failure_recall": 0.4,
        "success_recall": 0.6,
        "uncertainty_error_corr": -0.1,
        "pre_reader_witness_geometry": {
            "pairwise_cosine_agreement": {"overall": {"mean": 0.2}},
            "witness_latent_variance": {"overall": {"mean": 2.0}},
        },
        "post_reader_witness_geometry": {
            "pairwise_cosine_agreement": {"overall": {"mean": 0.3}},
            "witness_margin_std": {"overall": {"mean": 0.4}},
            "witness_weight_entropy": {"overall": {"mean": 0.9}},
            "witness_weight_kl_uniform": {"overall": {"mean": 0.1}},
        },
        "diversity_retention": {
            "reader_agreement_gain": {"overall": {"mean": 0.1}},
        },
        "decision_geometry": {
            "commitment_score": {"overall": {"mean": -0.2}},
        },
        "margin_channel_diagnostics": {
            "margin_positive_fraction": {"overall": {"mean": 0.4}},
            "margin_negative_fraction": {"overall": {"mean": 0.6}},
            "margin_near_zero_fraction": {"overall": {"mean": 0.1}},
            "witness_positive_count": {"overall": {"mean": 5.0}},
            "witness_negative_count": {"overall": {"mean": 3.0}},
            "weighted_margin_sign_flip_rate": {"overall": {"mean": 0.0}},
            "weighted_minus_unweighted_margin": {"overall": {"mean": 0.0}},
        },
        "failure_channel_diagnostics": {
            "true_failure_margin_negative_fraction": {"overall": {"mean": 0.75}},
            "true_failure_margin_positive_fraction": {"overall": {"mean": 0.25}},
            "true_failure_margin_near_zero_fraction": {"overall": {"mean": 0.05}},
            "true_failure_commitment_below_tau_fraction": {"overall": {"mean": 0.8}},
            "true_failure_predicted_success_fraction": {"overall": {"mean": 0.1}},
            "true_failure_predicted_uncertain_fraction": {"overall": {"mean": 0.8}},
            "true_failure_predicted_failure_fraction": {"overall": {"mean": 0.1}},
        },
        "success_channel_diagnostics": {
            "true_success_margin_negative_fraction": {"overall": {"mean": 0.2}},
            "true_success_margin_positive_fraction": {"overall": {"mean": 0.8}},
            "true_success_margin_near_zero_fraction": {"overall": {"mean": 0.05}},
            "true_success_commitment_below_tau_fraction": {"overall": {"mean": 0.2}},
            "true_success_predicted_success_fraction": {"overall": {"mean": 0.7}},
            "true_success_predicted_uncertain_fraction": {"overall": {"mean": 0.2}},
            "true_success_predicted_failure_fraction": {"overall": {"mean": 0.1}},
        },
        "weighting_diagnostics": {
            "coherence_margin_corr": {"overall": {"mean": 0.0}},
            "coherence_abs_margin_corr": {"overall": {"mean": 0.0}},
        },
    }

    report_highlights = _build_report_highlights(fake_metrics)

    assert report_highlights["true_failure_margin_negative_fraction_mean"] == 0.75
    assert report_highlights["true_failure_predicted_failure_fraction_mean"] == 0.1
    assert report_highlights["true_failure_predicted_uncertain_fraction_mean"] == 0.8
    assert report_highlights["true_failure_predicted_success_fraction_mean"] == 0.1
    assert report_highlights["true_success_margin_positive_fraction_mean"] == 0.8
    assert report_highlights["true_success_predicted_success_fraction_mean"] == 0.7


def test_soft_and_hard_gate_paths_are_distinct() -> None:
    torch.manual_seed(4)
    config_soft = _make_config(use_hard_eval_gate=False)
    config_hard = _make_config(use_hard_eval_gate=True)
    records, loader, class_weights = _make_eval_setup(config_soft)
    criterion_soft = E31Loss(config_soft, outcome_class_weights=class_weights)
    criterion_hard = E31Loss(config_hard, outcome_class_weights=class_weights)

    soft_metrics, _ = _evaluate(
        E31Model(config_soft), criterion_soft, loader, torch.device("cpu"), records, config_soft
    )
    hard_metrics, _ = _evaluate(
        E31Model(config_hard), criterion_hard, loader, torch.device("cpu"), records, config_hard
    )

    assert soft_metrics["prediction_mode"] == "soft_logits"
    assert hard_metrics["prediction_mode"] == "hard_commitment_gate"
    assert soft_metrics["prediction_mode"] != hard_metrics["prediction_mode"]
    assert "Hard evaluation gate" in bucketize_hard_commitment_decisions.__doc__ or "hard evaluation gate" in bucketize_hard_commitment_decisions.__doc__.lower()
    assert "Late-only" in bucketize_outcome_logits.__doc__


def test_explicit_repo_relative_output_dir_does_not_duplicate_variant_prefix() -> None:
    config = _make_config()
    explicit = "experiments/DIGIT/Extrapolation/e31/experiments/results/path_gate_seed_123"
    resolved = _resolve_output_dir(config, seed=123, explicit_output_dir=explicit)

    assert "experiments/DIGIT/Extrapolation/e31/experiments/DIGIT/Extrapolation/e31" not in str(
        resolved
    )
    assert str(resolved).endswith(explicit)


def test_bare_output_dir_name_is_placed_under_results_root() -> None:
    config = _make_config()
    resolved = _resolve_output_dir(config, seed=123, explicit_output_dir="path_gate_seed_123")

    assert str(resolved).endswith("experiments/results/path_gate_seed_123")
