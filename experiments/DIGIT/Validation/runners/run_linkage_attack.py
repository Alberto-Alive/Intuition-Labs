"""Comparative linkage runner."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from Validation.attacks.linkage_attack_eval import _overlapping_datasets, run_linkage_attack
from Validation.comparative.systems import build_system
from Validation.runners.comparative_common import (
    add_common_args,
    device_from_arg,
    ensure_required_digit_checkpoints,
    prepare_bundle,
    resolved_seeds,
    resolve_modes,
    write_attack_seed_result,
    write_summary_json,
)
from Validation.comparative.specs import AttackRunConfig
from Validation.comparative.systems import comparative_system_key, disambiguate_digit_variants, variant_name_for_mode
from Validation.comparative.stats import summarize_per_seed_metrics
from Validation.runners.comparative_common import default_system_specs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Comparative linkage attack")
    add_common_args(parser, "Comparative linkage")
    parser.add_argument("--per-record-budget", type=int, default=20)
    args = parser.parse_args(argv)

    seeds, seed_mode = resolved_seeds(args)
    modes = resolve_modes(args.modes)
    specs = default_system_specs(
        include_raw=not args.skip_raw,
        include_digit=not args.skip_digit,
        include_stability_gated_digit=args.add_stability_gated_digit,
        include_specificity_gated_digit=args.add_specificity_gated_digit,
        include_dp=not args.skip_dp,
        digit_variants=args.digit_variants,
        dp_epsilons=args.dp_epsilons,
    )
    label_digit_variants = disambiguate_digit_variants(specs)
    device = device_from_arg(args.device)
    results_root = Path(args.results_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    summary = {}

    for dataset_name in args.datasets:
        bundle = prepare_bundle(dataset_name, args.data_cache, args.split_seed)
        ensure_required_digit_checkpoints(
            bundle=bundle,
            dataset_name=dataset_name,
            system_specs=specs,
            modes=modes,
            seeds=seeds,
            ckpt_dir=ckpt_dir,
            device=device,
            train_missing=args.train,
        )
        ds_summary = {}
        for mode in modes:
            mode_summary = {}
            left_ds, right_ds, shared = _overlapping_datasets(bundle.train, bundle.attack_spec.linkage_overlap_fraction, seed=0)
            for spec in specs:
                system_key = comparative_system_key(spec, disambiguate_digit=label_digit_variants)
                per_seed = []
                for seed in seeds:
                    left_system, _, left_extra = build_system(
                        system_spec=spec,
                        private_data=left_ds,
                        mode=mode,
                        seed=seed,
                        ckpt_dir=ckpt_dir,
                        dataset_name=dataset_name,
                        device=device,
                    )
                    right_system, _, _ = build_system(
                        system_spec=spec,
                        private_data=right_ds,
                        mode=mode,
                        seed=seed,
                        ckpt_dir=ckpt_dir,
                        dataset_name=dataset_name,
                        device=device,
                    )
                    result = run_linkage_attack(
                        left_system,
                        right_system,
                        left_ds,
                        right_ds,
                        shared,
                        seed=seed,
                        query_budget=args.per_record_budget,
                    )
                    run_cfg = AttackRunConfig(
                        dataset=dataset_name,
                        phase="phase1_privacy",
                        task="linkage_attack",
                        mode=mode,
                        variant=variant_name_for_mode(mode, args.variant_setting),
                        seeds=seeds,
                        query_budget=args.per_record_budget,
                        results_root=results_root,
                        seed_mode=seed_mode,
                    )
                    write_attack_seed_result(
                        run_cfg=run_cfg,
                        system_spec=spec,
                        system_key=system_key,
                        seed=seed,
                        result=result,
                        config_extra=left_extra,
                    )
                    per_seed.append(result)
                mode_summary[system_key] = {
                    "per_seed": per_seed,
                    **summarize_per_seed_metrics(per_seed, ["precision", "recall", "top1_accuracy", "query_count", "cumulative_epsilon"]),
                }
            ds_summary[mode.value] = mode_summary
        summary[dataset_name] = ds_summary

    write_summary_json(results_root, "phase1_privacy", "linkage_attack_summary.json", summary)


if __name__ == "__main__":
    main()
