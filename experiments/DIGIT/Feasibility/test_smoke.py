"""Smoke test: verify all modules import and forward pass works."""
import torch
from intuition.config import Config
from intuition.data.vocabulary import Vocabulary
from intuition.data.dataset import IntuitionDataset, PrivateDataset, collate_fn
from intuition.models import (
    QueryEncoder, SafePrivateExecutor, PrimitiveBottleneckHead,
    IntuitionDecoder, IntuitionModel, BaselineA, BaselineB,
)
from intuition.losses import IntuitionLoss
from torch.utils.data import DataLoader

print("All imports OK")

config = Config()
vocab = Vocabulary()
print(f"Vocab size: {len(vocab)}")
print(f"Executor feature dim: {config.executor_feature_dim}")
print(f"Total primitive classes: {config.total_primitive_classes}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

private_data = PrivateDataset(num_records=100, num_features=8)
dataset = IntuitionDataset(num_samples=64, num_query_fields=8, vocab=vocab, seed=42)
loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
queries, group_ids, prim_targets, target_ids = next(iter(loader))
print(f"Batch: queries={queries.shape}, targets={target_ids.shape}, prims={prim_targets.shape}")

model = IntuitionModel(config, vocab).to(device)
queries = queries.to(device)
group_ids = group_ids.to(device)
prim_targets = prim_targets.to(device)
target_ids = target_ids.to(device)

out = model(queries, group_ids, private_data, target_ids, bottleneck_mode="gumbel", tau=1.0)
print(f"Decoder logits: {out['decoder_logits'].shape}")
print(f"Answer logits: {out['primitives'].answer_logits.shape}")

criterion = IntuitionLoss(config, pad_idx=vocab.pad_idx).to(device)
losses = criterion(out["decoder_logits"], out["primitives"], target_ids, prim_targets, out["executor_features"])
print("Losses: " + ", ".join(f"{k}={v.item():.4f}" for k, v in losses.items()))

gen = model.generate(queries, group_ids, private_data)
text = vocab.decode(gen["token_ids"][0].cpu().tolist())
print(f"Generated: {text}")

ba = BaselineA(config, vocab).to(device)
out_a = ba(queries, group_ids, private_data)
print(f"Baseline A: {out_a['texts'][0]}")

bb = BaselineB(config, vocab).to(device)
out_b = bb(queries, group_ids, private_data)
print(f"Baseline B: {out_b['texts'][0]}")

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {total:,} total, {trainable:,} trainable")

# Test all bottleneck modes
for mode in ["hard", "gumbel", "straight_through", "soft"]:
    out = model(queries, group_ids, private_data, target_ids, bottleneck_mode=mode, tau=1.0)
    print(f"Bottleneck mode '{mode}': OK")

print("\nAll tests PASSED")
