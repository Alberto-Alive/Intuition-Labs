"""Run bottleneck size ablation study."""

from __future__ import annotations

import os
import sys
import json
import logging
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poc.config import Config
from poc.data.adult_loader import load_adult_data
from poc.data.private_store import PrivateAdultDataset
from poc.data.vocabulary import Vocabulary
from poc.data.dataset import AdultIntuitionDataset
from poc.train import train_single_seed, evaluate_on_test
from poc.analysis.plotting import plot_ablation

logger = logging.getLogger(__name__)


def main(output_dir: str = "outputs_ablation"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    configs_dir = Path(__file__).parent.parent / "configs"
    base_config_path = str(configs_dir / "default.yaml")

    ablation_configs = {
        "small": str(configs_dir / "ablation_bottleneck_small.yaml"),
        "default": base_config_path,
        "large": str(configs_dir / "ablation_bottleneck_large.yaml"),
    }

    os.makedirs(output_dir, exist_ok=True)

    # Load data once
    base_config = Config.from_yaml(base_config_path)
    device = torch.device(base_config.device if torch.cuda.is_available() else "cpu")
    cache_dir = str(Path(__file__).parent.parent / "data_cache")
    splits = load_adult_data(cache_dir=cache_dir, split_seed=base_config.data_split_seed)
    vocab = Vocabulary()

    private_train = PrivateAdultDataset(splits.train, splits.categorical_maps)
    private_val = PrivateAdultDataset(splits.val, splits.categorical_maps)
    private_test = PrivateAdultDataset(splits.test, splits.categorical_maps)

    ablation_results = {}

    for name, config_path in ablation_configs.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Ablation: {name}")
        logger.info(f"{'='*60}")

        if name == "default":
            config = Config.from_yaml(config_path)
        else:
            config = Config.from_yaml(config_path, base_path=base_config_path)

        logger.info(f"Bottleneck: {config.num_answer_classes}x{config.num_support_classes}"
                     f"x{config.num_confidence_classes}x{config.num_risk_classes} "
                     f"= {config.max_bits_per_query:.2f} bits")

        abl_dir = os.path.join(output_dir, name)
        seed_accs = []
        seed_leakages = []

        test_dataset = AdultIntuitionDataset(
            private_data=private_test, num_queries=config.num_test_queries,
            vocab=vocab, config=config, seed=config.data_split_seed + 100,
        )

        # Run 5 seeds
        for seed in config.seeds:
            train_dataset = AdultIntuitionDataset(
                private_data=private_train, num_queries=config.num_train_queries,
                vocab=vocab, config=config, seed=seed,
            )
            val_dataset = AdultIntuitionDataset(
                private_data=private_val, num_queries=config.num_val_queries,
                vocab=vocab, config=config, seed=seed + 1000,
            )

            train_single_seed(
                config, vocab, train_dataset, val_dataset,
                private_train, device, seed, abl_dir,
            )

            model_path = os.path.join(abl_dir, f"seed_{seed}", "best_model.pt")
            test_result = evaluate_on_test(
                config, vocab, test_dataset, private_test,
                model_path, device,
            )
            seed_accs.append(test_result["accuracy"]["answer"])
            seed_leakages.append(test_result["accuracy"]["leakage"])

        ablation_results[name] = {
            "max_bits": config.max_bits_per_query,
            "answer_accuracy_mean": float(np.mean(seed_accs)),
            "answer_accuracy_std": float(np.std(seed_accs)),
            "leakage_mean": float(np.mean(seed_leakages)),
            "leakage_std": float(np.std(seed_leakages)),
            "per_seed_accuracy": seed_accs,
            "per_seed_leakage": seed_leakages,
        }

    # Save and plot
    with open(os.path.join(output_dir, "ablation_results.json"), "w") as f:
        json.dump(ablation_results, f, indent=2)

    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    plot_ablation(ablation_results, figures_dir)

    logger.info("\nAblation Summary:")
    for name, r in ablation_results.items():
        logger.info(f"  {name} ({r['max_bits']:.2f} bits): "
                     f"acc={r['answer_accuracy_mean']:.3f}±{r['answer_accuracy_std']:.3f}, "
                     f"leak={r['leakage_mean']:.4f}±{r['leakage_std']:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="outputs_ablation")
    args = parser.parse_args()
    main(output_dir=args.output_dir)
