"""Smoke test for the DIGIT Extrapolation E10 package."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from extrapolation.config import Config
from extrapolation.executor_schema import EXECUTOR_FEATURE_DIM, TRACE_INPUT_DIM, UNCERTAINTY_FEATURE_DIM
from extrapolation.data.adult_loader import load_adult_data
from extrapolation.data.dataset import AdultEntropyDataset, collate_fn
from extrapolation.data.ground_truth import (
    ATTENTION_PATTERN_LABELS,
    CONFIDENCE_LABELS,
    GroundTruthComputer,
    OUTCOME_LABELS,
    TRAJECTORY_SHAPE_LABELS,
)
from extrapolation.data.private_store import PrivateAdultDataset
from extrapolation.data.query_generator import QueryGenerator
from extrapolation.data.trace_dataset import TraceEntropyDataset
from extrapolation.data.vocabulary import Vocabulary
from extrapolation.losses import DIGITLoss
from extrapolation.models.baselines import BaselineA, BaselineB
from extrapolation.models.digit import DIGITModel


def _assert_labeler_consistency() -> None:
    gt = GroundTruthComputer(private_data=None, min_group_size=10, income_margin=0.10)

    decisive = gt.derive_from_stats(
        {
            "n": 240,
            "income_rate": 0.45,
            "support_ratio": 0.12,
            "variance": 0.18,
            "specificity": 4,
            "abs_margin": 0.20,
        }
    )
    assert TRAJECTORY_SHAPE_LABELS[decisive["trajectory_shape"]] == "DECREASING"
    assert ATTENTION_PATTERN_LABELS[decisive["attention_pattern"]] == "FOCUSED"
    assert CONFIDENCE_LABELS[decisive["confidence"]] == "HIGH"
    assert OUTCOME_LABELS[decisive["outcome"]] == "SUCCESS_LIKELY"

    volatile = gt.derive_from_stats(
        {
            "n": 8,
            "income_rate": 0.50,
            "support_ratio": 0.002,
            "variance": 0.25,
            "specificity": 7,
            "abs_margin": 0.02,
        }
    )
    assert TRAJECTORY_SHAPE_LABELS[volatile["trajectory_shape"]] == "VOLATILE"
    assert ATTENTION_PATTERN_LABELS[volatile["attention_pattern"]] == "DIFFUSE"
    assert CONFIDENCE_LABELS[volatile["confidence"]] == "LOW"
    assert OUTCOME_LABELS[volatile["outcome"]] == "FAILURE_LIKELY"

    borderline = gt.derive_from_stats(
        {
            "n": 40,
            "income_rate": 0.31,
            "support_ratio": 0.03,
            "variance": 0.21,
            "specificity": 3,
            "abs_margin": 0.03,
        }
    )
    assert TRAJECTORY_SHAPE_LABELS[borderline["trajectory_shape"]] == "INCREASING"
    assert ATTENTION_PATTERN_LABELS[borderline["attention_pattern"]] in {"MIXED", "DIFFUSE"}
    assert CONFIDENCE_LABELS[borderline["confidence"]] in {"LOW", "MEDIUM"}
    assert OUTCOME_LABELS[borderline["outcome"]] == "FAILURE_LIKELY"


print("All imports OK")

_assert_labeler_consistency()
print("Labeler self-consistency: OK")

config = Config()
vocab = Vocabulary()
assert config.trace_input_dim == TRACE_INPUT_DIM
assert config.executor_feature_dim == EXECUTOR_FEATURE_DIM
assert config.uncertainty_feature_dim == UNCERTAINTY_FEATURE_DIM

print(f"Vocab size: {len(vocab)}")
print(f"Max bits per query: {config.max_bits_per_query:.6f}")
assert math.isclose(config.max_bits_per_query, math.log2(108), rel_tol=0.0, abs_tol=1e-9)

device = torch.device(config.device if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("\nLoading Adult Income data...")
cache_dir = str(Path(__file__).parent.parent / "data_cache")
splits = load_adult_data(cache_dir=cache_dir, split_seed=config.data_split_seed)
print(f"Train: {len(splits.train)}, Val: {len(splits.val)}, Test: {len(splits.test)}")

private = PrivateAdultDataset(splits.train, splits.categorical_maps)
print(f"Private records: {private.num_records}")

print("\nGenerating queries...")
generator = QueryGenerator(private, min_group_size=config.min_group_size)
queries = generator.generate_queries(100, seed=42)
print(f"Generated {len(queries)} queries")

print("\nBuilding dataset...")
dataset = AdultEntropyDataset(
    private_data=private,
    num_queries=config.num_train_queries,
    vocab=vocab,
    config=config,
    seed=42,
)
print(f"Label distribution: {dataset.get_label_distribution()}")

loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
batch_queries, prim_targets, target_ids = next(iter(loader))
print(
    f"Batch: queries={tuple(batch_queries.shape)}, "
    f"targets={tuple(target_ids.shape)}, "
    f"prims={tuple(prim_targets.shape)}"
)

print("\nTesting DIGIT E10 model...")
model = DIGITModel(config, vocab).to(device)
assert not hasattr(model.bottleneck, "direct_outcome_head"), "clean-latent outcome shortcut should not exist"
assert hasattr(model.bottleneck, "path_evidence_head"), "E10 should decode shared evidence per diffusion path"
assert hasattr(model.bottleneck, "stability_cert_head"), "E10 should expose a cooperative stability head"
assert hasattr(model.bottleneck, "support_cert_head"), "E10 should expose a cooperative support head"
assert hasattr(model.bottleneck, "success_guard_head"), "E10 should expose a cooperative success guard head"
assert hasattr(model.bottleneck, "prototype_bank"), "E10 should expose a diffusion-native prototype bank"
assert not hasattr(model.bottleneck, "path_outcome_decoder"), "E10 should not keep a separate path outcome decoder"
assert not hasattr(model.bottleneck, "trust_head"), "E10 should not keep a split certainty head"
assert not hasattr(model.bottleneck, "fragility_head"), "E10 should not keep a split fragility head"
batch_queries = batch_queries.to(device)
prim_targets = prim_targets.to(device)
target_ids = target_ids.to(device)

out = model(batch_queries, private, target_ids, bottleneck_mode="gumbel", tau=1.0)
prim = out["primitives"]
print(f"Decoder logits: {tuple(out['decoder_logits'].shape)}")
print(f"Trajectory logits: {tuple(prim.trajectory_shape_logits.shape)}")
print(f"Confidence logits: {tuple(prim.confidence_logits.shape)}")
print(f"Outcome logits: {tuple(prim.outcome_logits.shape)}")
print(f"Executor features: {tuple(out['executor_features'].shape)}")

assert prim.trajectory_shape_logits.shape[-1] == config.num_trajectory_classes
assert prim.attention_pattern_logits.shape[-1] == config.num_pattern_classes
assert prim.confidence_logits.shape[-1] == config.num_confidence_classes
assert prim.outcome_logits.shape[-1] == config.num_outcome_classes
assert out["executor_features"].shape[-1] == config.executor_feature_dim
assert prim.anchor_state.shape[-1] == config.anchor_state_dim
assert not hasattr(prim, "path_outcome_logits"), "path_outcome_logits shortcut field should not exist"
assert torch.isfinite(prim.commitment_depth).all(), "commitment_depth has non-finite values"
assert torch.isfinite(prim.anchor_conflict).all(), "anchor_conflict has non-finite values"
assert torch.isfinite(prim.cert_risk).all(), "cert_risk has non-finite values"
assert torch.isfinite(prim.certainty_score).all(), "certainty_score has non-finite values"
assert torch.isfinite(prim.fragility_risk).all(), "fragility_risk has non-finite values"
assert torch.isfinite(prim.fragility_score).all(), "fragility_score has non-finite values"
assert torch.isfinite(prim.effective_certainty).all(), "effective_certainty has non-finite values"
assert torch.isfinite(prim.decisiveness_score).all(), "decisiveness_score has non-finite values"
assert torch.isfinite(prim.prototype_top_support).all(), "prototype_top_support has non-finite values"
assert torch.isfinite(prim.prototype_total_support).all(), "prototype_total_support has non-finite values"
assert torch.isfinite(prim.prototype_family_margin).all(), "prototype_family_margin has non-finite values"
assert torch.isfinite(prim.prototype_overlap).all(), "prototype_overlap has non-finite values"
assert torch.isfinite(prim.prototype_family_switch_rate).all(), "prototype_family_switch_rate has non-finite values"
assert torch.isfinite(prim.prototype_anchor_switch_rate).all(), "prototype_anchor_switch_rate has non-finite values"
assert torch.isfinite(prim.prototype_nearest_anchor_distance).all(), "prototype_nearest_anchor_distance has non-finite values"
assert torch.isfinite(prim.path_order_score_paths).all(), "path_order_score_paths has non-finite values"
assert torch.isfinite(prim.path_boundary_paths).all(), "path_boundary_paths has non-finite values"
assert torch.isfinite(prim.path_support_paths).all(), "path_support_paths has non-finite values"
assert torch.isfinite(prim.path_outcome_prob_paths).all(), "path_outcome_prob_paths has non-finite values"
assert prim.prototype_family_support_paths.shape == (config.num_random_paths, batch_queries.size(0), 3)
assert prim.prototype_family_distribution_paths.shape == (config.num_random_paths, batch_queries.size(0), 3)
assert prim.path_order_score_paths.shape == (config.num_random_paths, batch_queries.size(0))
assert prim.path_boundary_paths.shape == (config.num_random_paths, batch_queries.size(0))
assert prim.path_support_paths.shape == (config.num_random_paths, batch_queries.size(0))
assert prim.path_outcome_prob_paths.shape == (config.num_random_paths, batch_queries.size(0), 3)
assert torch.allclose(
    prim.prototype_family_distribution_paths.sum(dim=-1),
    torch.ones_like(prim.prototype_family_distribution_paths[..., 0]),
    atol=1e-5,
), "prototype_family_distribution_paths must sum to 1"
assert torch.allclose(
    prim.path_outcome_prob_paths.sum(dim=-1),
    torch.ones_like(prim.path_outcome_prob_paths[..., 0]),
    atol=1e-5,
), "path_outcome_prob_paths must sum to 1"
assert prim.anchor_assignment_probs.shape[2] == config.anchors_per_family
assert hasattr(prim, "prototype_pull_loss"), "prototype_pull_loss field missing"
assert hasattr(prim, "prototype_usage_loss"), "prototype_usage_loss field missing"
assert hasattr(prim, "prototype_repulsion_loss"), "prototype_repulsion_loss field missing"
assert torch.isfinite(prim.prototype_pull_loss), "prototype_pull_loss is non-finite"
assert torch.isfinite(prim.prototype_usage_loss), "prototype_usage_loss is non-finite"
assert torch.isfinite(prim.prototype_repulsion_loss), "prototype_repulsion_loss is non-finite"

# E10-specific auxiliary losses
assert hasattr(prim, "diffusion_denoise_loss"), "diffusion_denoise_loss field missing"
assert torch.isfinite(prim.diffusion_denoise_loss), "diffusion_denoise_loss is non-finite"
print(f"Diffusion denoising loss: {prim.diffusion_denoise_loss.item():.4f}")
print(f"Prototype pull loss: {prim.prototype_pull_loss.item():.4f}")
print(f"Prototype usage loss: {prim.prototype_usage_loss.item():.4f}")
print(f"Prototype repulsion loss: {prim.prototype_repulsion_loss.item():.4f}")

# Check certainty / basin stability fields
assert torch.all(prim.stability_cert >= 0.0) and torch.all(prim.stability_cert <= 1.0), "stability_cert out of [0,1]"
assert torch.all(prim.anchor_conflict >= 0.0) and torch.all(prim.anchor_conflict <= 1.0), "anchor_conflict out of [0,1]"
assert torch.all(prim.certainty_score >= 0.0) and torch.all(prim.certainty_score <= 1.0), "certainty_score out of [0,1]"
assert torch.all(prim.fragility_risk >= 0.0) and torch.all(prim.fragility_risk <= 1.0), "fragility_risk out of [0,1]"
assert torch.all(prim.effective_certainty >= 0.0) and torch.all(prim.effective_certainty <= 1.0), "effective_certainty out of [0,1]"
assert torch.all(prim.prototype_total_support >= 0.0) and torch.all(prim.prototype_total_support <= 1.0), "prototype_total_support out of [0,1]"
assert torch.all(prim.prototype_overlap >= 0.0) and torch.all(prim.prototype_overlap <= 1.0), "prototype_overlap out of [0,1]"
assert torch.all(prim.prototype_nearest_anchor_distance >= 0.0) and torch.all(prim.prototype_nearest_anchor_distance <= 1.0), "prototype_nearest_anchor_distance out of [0,1]"
assert prim.prototype_order_score_paths.shape == (config.num_random_paths, batch_queries.size(0)), "prototype_order_score_paths shape mismatch"
assert prim.path_reconstruction_energy.shape == (config.num_random_paths, batch_queries.size(0)), "path_reconstruction_energy shape mismatch"
assert torch.allclose(
    prim.success_base,
    prim.path_outcome_prob_paths[..., 0].mean(dim=0),
    atol=1e-5,
), "success_base must equal mean path success evidence"
assert torch.allclose(
    prim.success_core,
    prim.success_base,
    atol=1e-5,
), "success_core should equal ungated success_base in E10"
assert torch.allclose(
    prim.success_proposal,
    prim.success_base,
    atol=1e-5,
), "success_proposal should stay equal to success_base in the simplified success chain"
assert torch.allclose(prim.certainty_score, 1.0 - prim.cert_risk, atol=1e-5), "certainty_score must equal 1 - cert_risk"
assert torch.allclose(
    prim.effective_certainty,
    prim.stability_cert * prim.support_cert,
    atol=1e-5,
), "effective_certainty must equal stability_cert * support_cert"
assert torch.allclose(
    prim.fragility_score,
    1.0 - prim.fragility_risk,
    atol=1e-5,
), "fragility_score must equal 1 - fragility_risk"
assert torch.allclose(prim.decisiveness_score, 1.0 - prim.ambiguity_cert, atol=1e-5), "decisiveness_score must equal 1 - ambiguity_cert"
assert torch.allclose(
    prim.commitment_depth,
    prim.success_guard * prim.support_cert * prim.decisiveness_score,
    atol=1e-5,
), "commitment_depth must equal success_guard * support_cert * decisiveness_score"
assert torch.allclose(
    prim.success_cert,
    prim.success_base * prim.success_guard,
    atol=1e-5,
), "success_cert should apply only a single guard to success_base"
assert torch.allclose(
    prim.prototype_order_score,
    prim.prototype_family_distribution_paths[..., 0].mean(dim=0) - prim.prototype_family_distribution_paths[..., 1].mean(dim=0),
    atol=1e-5,
), "prototype_order_score must match success-minus-failure path score"
assert torch.all(prim.success_cert <= prim.success_guard + 1e-5), "success_cert must be guarded by success_guard"
assert torch.all(prim.success_proposal >= prim.success_cert - 1e-5), "success_proposal should dominate success_cert"
print("Shared evidence invariants: OK")

synthetic_perturbed_executor = out["executor_features"] * 0.95
perturbed_prim = model.bottleneck(
    z_q=out["z_q"],
    executor_features=synthetic_perturbed_executor,
    mode="soft",
    tau=1.0,
)
aux_losses = model.bottleneck.compute_perturbation_aux_losses(
    z_q=out["z_q"],
    executor_features=out["executor_features"],
    perturbed_executor_features=synthetic_perturbed_executor,
    primitives=prim,
    perturbed_primitives=perturbed_prim,
    outcome_targets=prim_targets[:, 3],
)
assert torch.isfinite(aux_losses["perturbation_rank"]), "perturbation_rank is non-finite"
assert torch.isfinite(aux_losses["perturbation_proximity"]), "perturbation_proximity is non-finite"
assert torch.isfinite(aux_losses["basin_order"]), "basin_order is non-finite"
assert torch.isfinite(aux_losses["ordered_geometry"]), "ordered_geometry is non-finite"
assert torch.isfinite(aux_losses["fragility_target_loss"]), "fragility_target_loss is non-finite"
assert torch.isfinite(aux_losses["fragility_rank"]), "fragility_rank is non-finite"
print(
    "Aux losses: "
    f"basin_order={aux_losses['basin_order'].item():.4f}, "
    f"ordered_geometry={aux_losses['ordered_geometry'].item():.4f}, "
    f"rank={aux_losses['perturbation_rank'].item():.4f}, "
    f"proximity={aux_losses['perturbation_proximity'].item():.4f}, "
    f"fragility_target={aux_losses['fragility_target_loss'].item():.4f}, "
    f"fragility_rank={aux_losses['fragility_rank'].item():.4f}"
)

criterion = DIGITLoss(
    config,
    pad_idx=vocab.pad_idx,
    class_weights=dataset.get_class_weights(),
).to(device)
losses = criterion(
    out["decoder_logits"],
    prim,
    target_ids,
    prim_targets,
    out["executor_features"],
    evidence_targets=None,
)
print("Losses: " + ", ".join(f"{k}={v.item():.4f}" for k, v in losses.items()))
assert "diffusion_denoise" in losses, "diffusion_denoise loss key missing"
assert "prototype_pull" in losses, "prototype_pull loss key missing"
assert "prototype_usage" in losses, "prototype_usage loss key missing"
assert "prototype_repulsion" in losses, "prototype_repulsion loss key missing"
assert "family_supervision" in losses, "family_supervision loss key missing"
assert "basin_order" in losses, "basin_order loss key missing"
assert "ordered_geometry" in losses, "ordered_geometry loss key missing"
assert "fragility_target" in losses, "fragility_target loss key missing"
assert "fragility_rank" in losses, "fragility_rank loss key missing"

gen = model.generate(batch_queries, private)
gen_prim = gen["primitives"]
traj_idx = gen_prim.trajectory_shape_discrete.argmax(dim=-1)
conf_idx = gen_prim.confidence_discrete.argmax(dim=-1)
outcome_idx = gen_prim.outcome_discrete.argmax(dim=-1)
text = vocab.decode(gen["token_ids"][0].cpu().tolist())
print(
    "Generated sample: "
    f"trajectory={TRAJECTORY_SHAPE_LABELS[traj_idx[0].item()]}, "
    f"confidence={CONFIDENCE_LABELS[conf_idx[0].item()]}, "
    f"outcome={OUTCOME_LABELS[outcome_idx[0].item()]}"
)
print(f"Generated text: {text}")

trace_dir = Path(__file__).parent.parent / config.trace_dir
if (trace_dir / "train.jsonl").exists():
    print("\nTesting trace-driven uncertainty path...")
    trace_dataset = TraceEntropyDataset.from_jsonl(trace_dir / "train.jsonl", vocab=vocab, config=config)
    trace_query, trace_inputs, trace_evidence, trace_prims, trace_target = trace_dataset[0]
    assert trace_inputs.shape[0] == config.trace_input_dim
    assert trace_evidence.shape[0] == 5

    trace_out = model(
        trace_query.unsqueeze(0).to(device),
        trace_inputs.unsqueeze(0).to(device),
        trace_target.unsqueeze(0).to(device),
        bottleneck_mode="hard",
        tau=1.0,
    )
    assert trace_out["executor_features"].shape[-1] == config.executor_feature_dim
    print("Trace path: OK")

print("\nTesting baselines...")
baseline_a = BaselineA(config, vocab).to(device)
out_a = baseline_a(batch_queries, private)
print(f"Baseline A outcome: {OUTCOME_LABELS[out_a['outcomes'][0].item()]}")

baseline_b = BaselineB(config, vocab).to(device)
out_b = baseline_b(batch_queries, private)
print(f"Baseline B text: {out_b['texts'][0]}")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nParameters: {total_params:,} total, {trainable_params:,} trainable")

for mode in ["hard", "gumbel", "straight_through", "soft"]:
    out = model(batch_queries, private, target_ids, bottleneck_mode=mode, tau=1.0)
    print(f"Bottleneck mode '{mode}': OK")

print("\nAll E10 tests PASSED")
