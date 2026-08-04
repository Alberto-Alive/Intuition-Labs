"""Smoke test for the DIGIT Extrapolation E6 package."""

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

print("\nTesting DIGIT E6 model...")
model = DIGITModel(config, vocab).to(device)
assert not hasattr(model.bottleneck, "direct_outcome_head"), "clean-latent outcome shortcut should not exist"
assert model.bottleneck.trust_head.STATS_DIM == 5, "trust head should consume certainty stats only"
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
assert torch.isfinite(prim.decisiveness_score).all(), "decisiveness_score has non-finite values"

# E6-specific: diffusion denoising loss
assert hasattr(prim, "diffusion_denoise_loss"), "diffusion_denoise_loss field missing"
assert torch.isfinite(prim.diffusion_denoise_loss), "diffusion_denoise_loss is non-finite"
print(f"Diffusion denoising loss: {prim.diffusion_denoise_loss.item():.4f}")

# Check certainty / basin stability fields
assert torch.all(prim.stability_cert >= 0.0) and torch.all(prim.stability_cert <= 1.0), "stability_cert out of [0,1]"
assert torch.all(prim.anchor_conflict >= 0.0) and torch.all(prim.anchor_conflict <= 1.0), "anchor_conflict out of [0,1]"
outcome_probs = torch.softmax(prim.outcome_logits, dim=-1)
assert torch.allclose(outcome_probs[:, 0], prim.success_cert, atol=1e-5), "success_cert must equal diffusion outcome prob"
assert torch.allclose(outcome_probs[:, 1], prim.ambiguity_cert, atol=1e-5), "ambiguity_cert must equal diffusion outcome prob"
assert torch.allclose(outcome_probs[:, 2], prim.failure_cert, atol=1e-5), "failure_cert must equal diffusion outcome prob"
assert torch.allclose(prim.certainty_score, 1.0 - prim.cert_risk, atol=1e-5), "certainty_score must equal 1 - cert_risk"
assert torch.allclose(prim.decisiveness_score, 1.0 - prim.ambiguity_cert, atol=1e-5), "decisiveness_score must equal 1 - ambiguity_cert"
assert torch.allclose(
    prim.commitment_depth,
    prim.certainty_score * prim.decisiveness_score,
    atol=1e-5,
), "commitment_depth must equal certainty_score * decisiveness_score"
print("Diffusion-specific invariants: OK")

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

print("\nAll E6 tests PASSED")
