"""Smoke test: verify all modules import and forward pass works."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from poc.config import Config
from poc.data.vocabulary import Vocabulary
from poc.data.adult_loader import load_adult_data
from poc.data.private_store import PrivateAdultDataset
from poc.data.query_generator import QueryGenerator
from poc.data.ground_truth import GroundTruthComputer, ANSWER_LABELS
from poc.data.dataset import AdultIntuitionDataset, collate_fn
from poc.models.digit import DIGITModel
from poc.models.baselines import BaselineA, BaselineB
from poc.losses import DIGITLoss
from torch.utils.data import DataLoader

print("All imports OK")

config = Config()
vocab = Vocabulary()
print(f"Vocab size: {len(vocab)}")
print(f"Max bits per query: {config.max_bits_per_query:.2f}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load data
print("\nLoading Adult Income data...")
cache_dir = str(Path(__file__).parent.parent / "data_cache")
splits = load_adult_data(cache_dir=cache_dir, split_seed=0)
print(f"Train: {len(splits.train)}, Val: {len(splits.val)}, Test: {len(splits.test)}")
print(f"Income >50K rate: {splits.train['label'].mean():.3f}")

# Private dataset
private = PrivateAdultDataset(splits.train, splits.categorical_maps)
print(f"Private records: {private.num_records}")

# Generate queries
print("\nGenerating queries...")
generator = QueryGenerator(private, min_group_size=config.min_group_size)
queries = generator.generate_queries(100, seed=42)
print(f"Generated {len(queries)} queries, shape: {queries[0].shape}")

# Ground truth
gt_computer = GroundTruthComputer(private, min_group_size=config.min_group_size)
gt = gt_computer.compute(queries[0])
print(f"Sample GT: answer={ANSWER_LABELS[gt['answer']]}, "
      f"support={gt['support']}, n={gt['stats']['n']}, "
      f"income_rate={gt['stats']['income_rate']:.3f}")
print(f"Response: {gt['response_text']}")

# Dataset
print("\nBuilding dataset...")
dataset = AdultIntuitionDataset(
    private_data=private, num_queries=64,
    vocab=vocab, config=config, seed=42,
)
print(f"Label distribution: {dataset.get_label_distribution()}")

loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
batch_queries, prim_targets, target_ids = next(iter(loader))
print(f"Batch: queries={batch_queries.shape}, targets={target_ids.shape}, "
      f"prims={prim_targets.shape}")

# Model forward pass
print("\nTesting DIGIT model...")
model = DIGITModel(config, vocab).to(device)
batch_queries = batch_queries.to(device)
prim_targets = prim_targets.to(device)
target_ids = target_ids.to(device)

out = model(batch_queries, private, target_ids, bottleneck_mode="gumbel", tau=1.0)
print(f"Decoder logits: {out['decoder_logits'].shape}")
print(f"Answer logits: {out['primitives'].answer_logits.shape}")
print(f"Executor features: {out['executor_features'].shape}")

# Loss
criterion = DIGITLoss(config, pad_idx=vocab.pad_idx).to(device)
losses = criterion(
    out["decoder_logits"], out["primitives"],
    target_ids, prim_targets, out["executor_features"],
)
print("Losses: " + ", ".join(f"{k}={v.item():.4f}" for k, v in losses.items()))

# Generate
gen = model.generate(batch_queries, private)
text = vocab.decode(gen["token_ids"][0].cpu().tolist())
print(f"Generated: {text}")

# Baselines
print("\nTesting baselines...")
ba = BaselineA(config, vocab).to(device)
out_a = ba(batch_queries, private)
print(f"Baseline A: {out_a['texts'][0]}")

bb = BaselineB(config, vocab).to(device)
out_b = bb(batch_queries, private)
print(f"Baseline B: {out_b['texts'][0]}")

# Parameters
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nParameters: {total:,} total, {trainable:,} trainable")

# All bottleneck modes
for mode in ["hard", "gumbel", "straight_through", "soft"]:
    out = model(batch_queries, private, target_ids, bottleneck_mode=mode, tau=1.0)
    print(f"Bottleneck mode '{mode}': OK")

print("\nAll tests PASSED")
