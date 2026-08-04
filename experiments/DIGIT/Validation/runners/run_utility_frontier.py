"""Fair comparative privacy-utility frontier runner."""
from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

from Validation.comparative.specs import AttackMode, AttackRunConfig
from Validation.comparative.systems import (
    comparative_system_key,
    disambiguate_digit_variants,
    variant_name_for_mode,
)
from Validation.comparative.utility import (
    evaluate_fair_utility,
    render_utility_summary_markdown,
    utility_comparison_fields,
)
from Validation.runners.comparative_common import (
    add_common_args,
    aggregate_attack_results,
    device_from_arg,
    default_system_specs,
    resolved_seeds,
    resolve_modes,
    run_single_seed_attack,
    write_attack_seed_result,
    write_summary_json,
)
from Validation.utility.fair import build_fair_utility_bundle
from Validation.utility.protocol import require_fair_automated_utility

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_METRIC_NAMES = [
    "primitive_accuracy_mean",
    "answer_accuracy",
    "support_accuracy",
    "confidence_accuracy",
    "risk_accuracy",
    "auroc",
    "auprc",
    "f1",
    "accuracy",
    "balanced_accuracy",
    "brier",
    "macro_f1",
    "log_loss",
    "rmse",
    "pearson",
    "spearman",
    "ndcg",
    "utility_feature_dim",
    "utility_subproblem_count",
]


def _write_summary_markdown(results_root: Path, payload: dict) -> Path:
    path = results_root / "phase2_utility" / "privacy_utility_frontier_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_utility_summary_markdown(payload), encoding="utf-8")
    logger.info("Summary → %s", path)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Comparative fair utility frontier")
    add_common_args(parser, "Comparative fair utility frontier")
    parser.set_defaults(
        digit_variants=["a2_s2_c2_r2", "a3_s2_c2_r2", "a4_s3_c2_r2", "a6_s4_c3_r3"],
        modes=[AttackMode.SYSTEM.value],
    )
    parser.add_argument("--internal-queries", type=int, default=1000)
    parser.add_argument("--queries-per-record", type=int, default=8)
    args = parser.parse_args(argv)

    seeds, seed_mode = resolved_seeds(args)
    modes = list(resolve_modes(args.modes))
    device = device_from_arg(args.device)
    results_root = Path(args.results_dir)
    ckpt_dir = Path(args.checkpoint_dir)

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

    summary: dict = {}
    for dataset_name in args.datasets:
        protocol = require_fair_automated_utility(dataset_name)
        fair_bundle = build_fair_utility_bundle(
            dataset_name,
            data_cache=args.data_cache,
            split_seed=args.split_seed,
        )
        dataset_summary = {}
        for mode in modes:
            systems_summary = {}
            for system_spec in specs:
                system_key = comparative_system_key(
                    system_spec,
                    disambiguate_digit=label_digit_variants,
                )
                per_seed_results = []
                for seed in seeds:
                    logger.info(
                        "Task=%s dataset=%s mode=%s system=%s seed=%s evaluate_fair_utility",
                        "privacy_utility_frontier",
                        dataset_name,
                        mode.value,
                        system_key,
                        seed,
                    )
                    result, stdout_log = run_single_seed_attack(
                        runner=evaluate_fair_utility,
                        runner_kwargs={
                            "bundle": fair_bundle,
                            "system_spec": system_spec,
                            "mode": mode,
                            "seed": seed,
                            "ckpt_dir": ckpt_dir,
                            "device": device,
                            "train_missing": args.train,
                            "internal_queries": args.internal_queries,
                            "queries_per_record": args.queries_per_record,
                        },
                    )
                    config_extra = result.pop("_config_extra", {})
                    run_cfg_variant = variant_name_for_mode(mode, getattr(args, "variant_setting", "default"))
                    run_cfg = AttackRunConfig(
                        dataset=dataset_name,
                        phase="phase2_utility",
                        task="privacy_utility_frontier",
                        mode=mode,
                        variant=run_cfg_variant,
                        seeds=seeds,
                        query_budget=args.query_budget,
                        results_root=results_root,
                        seed_mode=seed_mode,
                    )
                    write_attack_seed_result(
                        run_cfg=run_cfg,
                        system_spec=system_spec,
                        system_key=system_key,
                        seed=seed,
                        result=result,
                        config_extra=config_extra,
                        stdout_log=stdout_log,
                    )
                    per_seed_results.append(result)
                systems_summary[system_key] = aggregate_attack_results(
                    per_seed_results,
                    metric_names=_METRIC_NAMES,
                    seed=0,
                )

            dataset_summary[mode.value] = {
                "protocol": dataclasses.asdict(protocol),
                "systems": systems_summary,
                "comparisons": utility_comparison_fields(protocol, systems_summary),
            }
        summary[dataset_name] = dataset_summary

    write_summary_json(results_root, "phase2_utility", "privacy_utility_frontier_summary.json", summary)
    _write_summary_markdown(results_root, summary)


if __name__ == "__main__":
    main()
