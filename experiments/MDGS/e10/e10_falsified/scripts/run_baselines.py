"""Run simple baselines on the collected trace corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.DIGIT.Extrapolation.e10.e10_falsified.extrapolation.config import Config
from experiments.DIGIT.Extrapolation.e10.e10_falsified.extrapolation.data.trace_dataset import load_trace_records
from experiments.DIGIT.Extrapolation.e10.e10_falsified.extrapolation.trace_pipeline import ensure_trace_corpus, evaluate_trace_baselines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-traces", action="store_true")
    parser.add_argument("--pattern-threshold-override", type=float, default=None)
    args = parser.parse_args()

    config = Config()
    config.trace_pattern_threshold_override = args.pattern_threshold_override
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    trace_dir = Path(__file__).parent.parent / config.trace_dir

    ensure_trace_corpus(
        config=config,
        root_dir=trace_dir,
        device=device,
        rebuild=args.rebuild_traces,
        strict_existing=not args.rebuild_traces,
    )

    train_records = load_trace_records(trace_dir / "train.jsonl")
    test_records = load_trace_records(trace_dir / "test.jsonl")
    results = evaluate_trace_baselines(train_records, test_records, config=config)

    metadata_path = trace_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        metadata["mlp_converged"] = results["mlp"]["outcome"]["converged"]
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    print("Trace baseline evaluation")
    print(f"Trace dir: {trace_dir}")
    print(f"Corpus fingerprint: {metadata.get('corpus_fingerprint')}")
    print(
        "Entropy threshold: "
        f"outcome_acc={results['entropy_threshold']['outcome_accuracy']:.3f}, "
        f"outcome_macro_f1={results['entropy_threshold']['outcome_macro_f1']:.3f}"
    )
    print(
        "Logistic regression outcome: "
        f"acc={results['logistic_regression']['outcome']['accuracy']:.3f}, "
        f"macro_f1={results['logistic_regression']['outcome']['macro_f1']:.3f}"
    )
    print(
        "MLP outcome: "
        f"acc={results['mlp']['outcome']['accuracy']:.3f}, "
        f"macro_f1={results['mlp']['outcome']['macro_f1']:.3f}, "
        f"converged={results['mlp']['outcome']['converged']}, "
        f"n_iter={results['mlp']['outcome']['n_iter']}"
    )


if __name__ == "__main__":
    main()
