"""Monotonicity smoke test for the E3 partial-monotone outcome head."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT_DIR = Path(__file__).resolve().parents[2]
VARIANT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(VARIANT_DIR))

from eval_common import audit_monotonicity
from extrapolation.config import Config
from extrapolation.data.trace_dataset import TraceEntropyDataset, trace_collate_fn
from extrapolation.data.vocabulary import Vocabulary
from extrapolation.models.digit import DIGITModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    config = Config()
    vocab = Vocabulary()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    trace_dir = VARIANT_DIR / config.trace_dir
    dataset = TraceEntropyDataset.from_jsonl(trace_dir / f"{args.split}.jsonl", vocab=vocab, config=config)
    if args.limit and args.limit < len(dataset):
        dataset = Subset(dataset, range(int(args.limit)))

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=trace_collate_fn,
    )

    model = DIGITModel(config, vocab).to(device)
    metadata_path = trace_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            model.set_trace_metadata(json.load(f))

    audit = audit_monotonicity(model, loader, device)
    keys = [
        "monotonicity_violation_rate",
        "all_supported_monotonicity_violation_rate",
        "nonfailure_monotonicity_violation_rate",
        "all_supported_nonfailure_monotonicity_violation_rate",
        "stability_cert_violation_rate",
        "support_cert_violation_rate",
        "success_cert_violation_rate",
        "failure_cert_violation_rate",
    ]

    print("E3 monotonicity smoke test")
    print(f"Device: {device}")
    print(f"Trace split: {args.split}")
    print(f"Examples: {audit['num_examples']}")
    print(f"Supported checks: {', '.join(audit['supported_checks'])}")
    for key in keys:
        print(f"{key}: {audit[key]:.8f}")

    failing = [key for key in keys if float(audit[key]) > float(args.tolerance)]
    if failing:
        joined = ", ".join(failing)
        raise AssertionError(f"Monotonicity smoke test failed for: {joined}")

    print("Partial-monotone outcome head: OK")


if __name__ == "__main__":
    main()
