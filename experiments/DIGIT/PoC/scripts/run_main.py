"""Main experiment entry point.

Runs the full DIGIT evaluation pipeline:
1. Load and split Adult Income data
2. Evaluate baselines (A, B)
3. Train DIGIT across 5 seeds with early stopping
4. Evaluate on test set
5. Run DP baseline comparison
6. Run privacy attacks
7. Compute significance tests
8. Generate figures and tables
"""

from __future__ import annotations

import os
import sys
import json
import logging
from pathlib import Path

import torch
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from poc.config import Config
from poc.data.adult_loader import load_adult_data
from poc.data.private_store import PrivateAdultDataset
from poc.data.vocabulary import Vocabulary
from poc.data.dataset import AdultIntuitionDataset, collate_fn
from poc.data.ground_truth import ANSWER_LABELS, SUPPORT_LABELS
from poc.models.digit import DIGITModel
from poc.models.dp_baseline import DPLaplaceBaseline
from poc.train import (
    train_single_seed, evaluate_baselines, evaluate_on_test,
)
from poc.evaluation.metrics import compute_metrics, aggregate_across_seeds
from poc.privacy.membership_inference import run_membership_inference_attack
from poc.privacy.attribute_inference import run_attribute_inference_attack
from poc.privacy.privacy_metrics import aggregate_privacy_results
from poc.analysis.significance_tests import (
    run_all_significance_tests, apply_bonferroni,
)
from poc.analysis.plotting import (
    plot_training_curves, plot_system_comparison,
    plot_privacy_utility_tradeoff, plot_confusion_matrix, plot_ablation,
)
from poc.analysis.results_formatter import (
    format_main_results_table, format_privacy_table, format_dp_comparison_table,
)

logger = logging.getLogger(__name__)


def main(config_path: str = None, output_dir: str = "outputs"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    if config_path:
        config = Config.from_yaml(config_path)
    else:
        default_config = str(Path(__file__).parent.parent / "configs" / "default.yaml")
        if os.path.exists(default_config):
            config = Config.from_yaml(default_config)
        else:
            config = Config()

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Max bits per query: {config.max_bits_per_query:.2f}")

    # ═══════════════════════════════════════════════════════════════
    # 1. Load data
    # ═══════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Loading UCI Adult Income dataset")
    logger.info("=" * 60)

    cache_dir = str(Path(__file__).parent.parent / "data_cache")
    splits = load_adult_data(cache_dir=cache_dir, split_seed=config.data_split_seed)

    vocab = Vocabulary()
    logger.info(f"Vocabulary size: {len(vocab)}")

    # Private datasets for each split
    private_train = PrivateAdultDataset(splits.train, splits.categorical_maps)
    private_val = PrivateAdultDataset(splits.val, splits.categorical_maps)
    private_test = PrivateAdultDataset(splits.test, splits.categorical_maps)

    logger.info(f"Private train: {private_train.num_records} records, "
                f"income rate: {private_train.overall_income_rate:.3f}")

    # ═══════════════════════════════════════════════════════════════
    # 2. Baselines
    # ═══════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Evaluating baselines")
    logger.info("=" * 60)

    # Generate test queries once (shared across all evaluations)
    test_dataset = AdultIntuitionDataset(
        private_data=private_test, num_queries=config.num_test_queries,
        vocab=vocab, config=config, seed=config.data_split_seed + 100,
    )
    logger.info(f"Test dataset label distribution: {test_dataset.get_label_distribution()}")

    baseline_results = evaluate_baselines(
        config, vocab, test_dataset, private_test, device
    )
    logger.info(f"Baseline A — 3-class acc: {baseline_results['baseline_a']['accuracy_3class']:.3f}")
    logger.info(f"Baseline B — answer acc: {baseline_results['baseline_b']['answer_accuracy']:.3f}")
    logger.info(f"Baseline B — support acc: {baseline_results['baseline_b']['support_accuracy']:.3f}")

    # ═══════════════════════════════════════════════════════════════
    # 3. Train DIGIT across seeds
    # ═══════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info(f"Training DIGIT across {len(config.seeds)} seeds")
    logger.info("=" * 60)

    all_histories = []
    all_seed_results = []
    all_test_results = []

    for seed in config.seeds:
        logger.info(f"\n--- Seed {seed} ---")

        train_dataset = AdultIntuitionDataset(
            private_data=private_train, num_queries=config.num_train_queries,
            vocab=vocab, config=config, seed=seed,
        )
        val_dataset = AdultIntuitionDataset(
            private_data=private_val, num_queries=config.num_val_queries,
            vocab=vocab, config=config, seed=seed + 1000,
        )

        seed_result = train_single_seed(
            config, vocab, train_dataset, val_dataset,
            private_train, device, seed, output_dir,
        )
        all_seed_results.append(seed_result)
        all_histories.append(seed_result["history"])

        # Evaluate on test set
        model_path = os.path.join(output_dir, f"seed_{seed}", "best_model.pt")
        test_result = evaluate_on_test(
            config, vocab, test_dataset, private_test,
            model_path, device,
        )
        all_test_results.append(test_result)

        logger.info(
            f"Seed {seed} test — answer acc: {test_result['accuracy']['answer']:.3f}, "
            f"leakage: {test_result['accuracy']['leakage']:.4f}"
        )

    # ═══════════════════════════════════════════════════════════════
    # 4. DP Baseline comparison
    # ═══════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Running DP Laplace baseline")
    logger.info("=" * 60)

    dp_results = []
    for eps in config.dp_epsilons:
        dp = DPLaplaceBaseline(
            private_data=private_test, epsilon=eps,
            min_group_size=config.min_group_size,
            income_margin=config.income_margin,
        )
        dp_eval = dp.evaluate(test_dataset.queries, test_dataset.ground_truths)
        dp_results.append(dp_eval)
        logger.info(f"DP (eps={eps}) — answer acc: {dp_eval['answer_accuracy']:.3f}")

    # ═══════════════════════════════════════════════════════════════
    # 5. Privacy attacks
    # ═══════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Running privacy attacks")
    logger.info("=" * 60)

    # Use holdout (test split records) as non-members
    holdout_store = PrivateAdultDataset(splits.test, splits.categorical_maps)

    all_mia_results = []
    all_aia_results = []

    for seed in config.seeds:
        model_path = os.path.join(output_dir, f"seed_{seed}", "best_model.pt")
        model = DIGITModel(config, vocab).to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

        # Membership inference
        mia = run_membership_inference_attack(
            model=model,
            private_data=private_train,
            holdout_features=holdout_store.features,
            holdout_labels=holdout_store.labels,
            device=device,
            num_samples=2000,
        )
        all_mia_results.append(mia)

        # Attribute inference
        aia = run_attribute_inference_attack(
            model=model,
            private_data=private_train,
            target_features=holdout_store.features,
            target_labels=holdout_store.labels,
            device=device,
            num_samples=2000,
        )
        all_aia_results.append(aia)

    privacy_agg = aggregate_privacy_results(all_mia_results, all_aia_results)
    logger.info(f"MIA AUC (mean): {privacy_agg.get('mia_best_auc_mean', 'N/A')}")
    logger.info(f"AIA Advantage (mean): {privacy_agg.get('aia_auc_advantage_mean', 'N/A')}")

    # ═══════════════════════════════════════════════════════════════
    # 6. Aggregate results and significance tests
    # ═══════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Computing aggregate results and significance tests")
    logger.info("=" * 60)

    # Aggregate test metrics across seeds
    digit_answer_accs = [r["accuracy"]["answer"] for r in all_test_results]
    digit_support_accs = [r["accuracy"]["support"] for r in all_test_results]
    digit_leakages = [r["accuracy"]["leakage"] for r in all_test_results]

    baseline_b_acc = baseline_results["baseline_b"]["answer_accuracy"]
    baseline_b_accs = [baseline_b_acc] * len(config.seeds)  # constant

    sig_tests = [
        run_all_significance_tests(
            digit_answer_accs, baseline_b_accs,
            metric_name="DIGIT vs Baseline B (answer accuracy)",
        ),
    ]
    sig_tests = apply_bonferroni(sig_tests)

    # ═══════════════════════════════════════════════════════════════
    # 7. Generate figures
    # ═══════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Generating figures")
    logger.info("=" * 60)

    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Training curves
    plot_training_curves(all_histories, config.seeds, figures_dir)

    # Confusion matrix (from first seed)
    if all_test_results:
        cm = compute_metrics(
            all_test_results[0]["predictions"],
            all_test_results[0]["targets"],
            num_classes=config.num_answer_classes,
            class_names=ANSWER_LABELS[:config.num_answer_classes],
        )
        plot_confusion_matrix(
            np.array(cm["confusion_matrix"]),
            ANSWER_LABELS[:config.num_answer_classes],
            figures_dir,
        )

    # ═══════════════════════════════════════════════════════════════
    # 8. Save all results
    # ═══════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("Saving results")
    logger.info("=" * 60)

    all_results = {
        "config": {
            "max_bits_per_query": config.max_bits_per_query,
            "num_answer_classes": config.num_answer_classes,
            "num_support_classes": config.num_support_classes,
            "num_confidence_classes": config.num_confidence_classes,
            "num_risk_classes": config.num_risk_classes,
            "seeds": config.seeds,
        },
        "baselines": baseline_results,
        "digit": {
            "answer_accuracy": {
                "mean": float(np.mean(digit_answer_accs)),
                "std": float(np.std(digit_answer_accs)),
                "per_seed": digit_answer_accs,
            },
            "support_accuracy": {
                "mean": float(np.mean(digit_support_accs)),
                "std": float(np.std(digit_support_accs)),
                "per_seed": digit_support_accs,
            },
            "leakage": {
                "mean": float(np.mean(digit_leakages)),
                "std": float(np.std(digit_leakages)),
                "per_seed": digit_leakages,
            },
        },
        "dp_baselines": dp_results,
        "privacy": privacy_agg,
        "significance_tests": sig_tests,
        "sample_outputs": all_seed_results[0]["sample_outputs"] if all_seed_results else [],
    }

    with open(os.path.join(output_dir, "experiment_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # LaTeX tables
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    with open(os.path.join(tables_dir, "main_results.tex"), "w") as f:
        f.write(format_main_results_table({
            "Baseline A": {"answer_accuracy": baseline_results["baseline_a"]["accuracy_3class"]},
            "Baseline B": baseline_results["baseline_b"],
            "DIGIT": all_results["digit"],
        }))

    with open(os.path.join(tables_dir, "privacy.tex"), "w") as f:
        f.write(format_privacy_table({"DIGIT": privacy_agg}))

    with open(os.path.join(tables_dir, "dp_comparison.tex"), "w") as f:
        f.write(format_dp_comparison_table(dp_results, {
            "answer_accuracy": float(np.mean(digit_answer_accs)),
            "mia_auc": privacy_agg.get("mia_best_auc_mean", 0.5),
            "max_bits": config.max_bits_per_query,
        }))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Baseline A (3-class): {baseline_results['baseline_a']['accuracy_3class']:.3f}")
    logger.info(f"Baseline B (full):    {baseline_results['baseline_b']['answer_accuracy']:.3f}")
    logger.info(f"DIGIT (mean±std):     {np.mean(digit_answer_accs):.3f} ± {np.std(digit_answer_accs):.3f}")
    logger.info(f"Leakage (mean±std):   {np.mean(digit_leakages):.4f} ± {np.std(digit_leakages):.4f}")
    logger.info(f"MIA AUC (mean):       {privacy_agg.get('mia_best_auc_mean', 'N/A')}")
    logger.info(f"AIA Advantage (mean): {privacy_agg.get('aia_auc_advantage_mean', 'N/A')}")
    logger.info(f"\nResults saved to {output_dir}/")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DIGIT PoC — Adult Income Experiment")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()
    main(config_path=args.config, output_dir=args.output_dir)
