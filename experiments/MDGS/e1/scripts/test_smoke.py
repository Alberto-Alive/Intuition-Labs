"""Smoke test for the DIGIT Extrapolation package."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from extrapolation.config import Config
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

print(f"Vocab size: {len(vocab)}")
print(f"Max bits per query: {config.max_bits_per_query:.6f}")
assert math.isclose(config.max_bits_per_query, math.log2(108), rel_tol=0.0, abs_tol=1e-9)

device = torch.device(config.device if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("\nLoading Adult Income data...")
cache_dir = str(Path(__file__).parent.parent / "data_cache")
splits = load_adult_data(cache_dir=cache_dir, split_seed=config.data_split_seed)
print(f"Train: {len(splits.train)}, Val: {len(splits.val)}, Test: {len(splits.test)}")
print(f"Income >50K rate: {splits.train['label'].mean():.3f}")

private = PrivateAdultDataset(splits.train, splits.categorical_maps)
print(f"Private records: {private.num_records}")

print("\nGenerating queries...")
generator = QueryGenerator(private, min_group_size=config.min_group_size)
queries = generator.generate_queries(100, seed=42)
print(f"Generated {len(queries)} queries, shape: {queries[0].shape}")

gt_computer = GroundTruthComputer(
    private_data=private,
    min_group_size=config.min_group_size,
    income_margin=config.income_margin,
)
gt = gt_computer.compute(queries[0])
print(
    "Sample GT: "
    f"trajectory={TRAJECTORY_SHAPE_LABELS[gt['trajectory_shape']]}, "
    f"pattern={ATTENTION_PATTERN_LABELS[gt['attention_pattern']]}, "
    f"confidence={CONFIDENCE_LABELS[gt['confidence']]}, "
    f"outcome={OUTCOME_LABELS[gt['outcome']]}, "
    f"n={gt['stats']['n']}, "
    f"mean_entropy={sum(gt['entropy_trajectory']) / len(gt['entropy_trajectory']):.3f}"
)
print(f"Entropy trajectory: {gt['entropy_trajectory']}")
print(f"Response: {gt['response_text']}")

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

print("\nTesting DIGIT model...")
model = DIGITModel(config, vocab).to(device)
batch_queries = batch_queries.to(device)
prim_targets = prim_targets.to(device)
target_ids = target_ids.to(device)

out = model(batch_queries, private, target_ids, bottleneck_mode="gumbel", tau=1.0)
print(f"Decoder logits: {tuple(out['decoder_logits'].shape)}")
print(f"Trajectory logits: {tuple(out['primitives'].trajectory_shape_logits.shape)}")
print(f"Pattern logits: {tuple(out['primitives'].attention_pattern_logits.shape)}")
print(f"Confidence logits: {tuple(out['primitives'].confidence_logits.shape)}")
print(f"Outcome logits: {tuple(out['primitives'].outcome_logits.shape)}")
print(f"Executor features: {tuple(out['executor_features'].shape)}")

assert out["primitives"].trajectory_shape_logits.shape[-1] == config.num_trajectory_classes
assert out["primitives"].attention_pattern_logits.shape[-1] == config.num_pattern_classes
assert out["primitives"].confidence_logits.shape[-1] == config.num_confidence_classes
assert out["primitives"].outcome_logits.shape[-1] == config.num_outcome_classes
assert out["executor_features"].shape[-1] == config.executor_feature_dim

criterion = DIGITLoss(
    config,
    pad_idx=vocab.pad_idx,
    class_weights=dataset.get_class_weights(),
).to(device)
losses = criterion(
    out["decoder_logits"],
    out["primitives"],
    target_ids,
    prim_targets,
    out["executor_features"],
)
print("Losses: " + ", ".join(f"{k}={v.item():.4f}" for k, v in losses.items()))

gen = model.generate(batch_queries, private)
text = vocab.decode(gen["token_ids"][0].cpu().tolist())
pred_idx = model.bottleneck.get_primitive_indices(gen["primitives"])
print(
    "Generated sample: "
    f"trajectory={TRAJECTORY_SHAPE_LABELS[pred_idx['trajectory_shape'][0].item()]}, "
    f"pattern={ATTENTION_PATTERN_LABELS[pred_idx['attention_pattern'][0].item()]}, "
    f"confidence={CONFIDENCE_LABELS[pred_idx['confidence'][0].item()]}, "
    f"outcome={OUTCOME_LABELS[pred_idx['outcome'][0].item()]}"
)
print(f"Generated text: {text}")

print("\nTesting baselines...")
baseline_a = BaselineA(config, vocab).to(device)
out_a = baseline_a(batch_queries, private)
print(f"Baseline A outcome: {OUTCOME_LABELS[out_a['outcomes'][0].item()]}")
print(f"Baseline A text: {out_a['texts'][0]}")

baseline_b = BaselineB(config, vocab).to(device)
out_b = baseline_b(batch_queries, private)
print(f"Baseline B text: {out_b['texts'][0]}")

total_params = sum(param.numel() for param in model.parameters())
trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
print(f"\nParameters: {total_params:,} total, {trainable_params:,} trainable")

for mode in ["hard", "gumbel", "straight_through", "soft"]:
    out = model(batch_queries, private, target_ids, bottleneck_mode=mode, tau=1.0)
    print(f"Bottleneck mode '{mode}': OK")

print("\nAll tests PASSED")
