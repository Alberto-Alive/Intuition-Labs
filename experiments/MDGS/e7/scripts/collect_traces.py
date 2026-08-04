"""Collect real attention-entropy traces for DIGIT Extrapolation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from extrapolation.config import Config
from extrapolation.trace_pipeline import ensure_trace_corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--pattern-threshold-override", type=float, default=None)
    args = parser.parse_args()

    config = Config()
    config.trace_pattern_threshold_override = args.pattern_threshold_override
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    trace_dir = Path(__file__).parent.parent / config.trace_dir

    info = ensure_trace_corpus(
        config=config,
        root_dir=trace_dir,
        device=device,
        rebuild=args.force,
    )

    print("Trace collection ready")
    print(f"Device: {device}")
    print(f"Trace dir: {info['trace_dir']}")
    print(f"Label version: {info.get('label_version', 'unknown')}")
    print(f"Corpus fingerprint: {info.get('corpus_fingerprint')}")
    print(f"Pattern z-threshold: {info.get('pattern_z_threshold')}")
    print(f"Pattern threshold mode: {info.get('pattern_threshold_mode')}")
    print(
        f"Records train/val/test: "
        f"{info['num_train_records']}/{info['num_val_records']}/{info['num_test_records']}"
    )
    print(
        "Failure shares train/val/test: "
        f"{info.get('train_failure_share', 0.0):.3f}/"
        f"{info.get('val_failure_share', 0.0):.3f}/"
        f"{info.get('test_failure_share', 0.0):.3f}"
    )
    print(f"Train hard-mined cases: {info.get('train_hard_case_count', 0)}")


if __name__ == "__main__":
    main()
