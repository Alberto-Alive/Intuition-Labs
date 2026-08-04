"""Shared runner utilities for comparative attacks."""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

_digit_root = Path(__file__).resolve().parents[2]
_poc_path = str(_digit_root / "PoC")
if _poc_path not in sys.path:
    sys.path.insert(0, _poc_path)

from Validation.comparative.registry import (
    ComparativeDatasetBundle,
    get_supported_dataset_names,
    load_dataset_bundle,
)
from Validation.comparative.specs import AttackMode, AttackRunConfig, SystemSpec
from Validation.comparative.stats import summarize_curve_events, summarize_per_seed_metrics
from Validation.comparative.systems import (
    DEFAULT_DP_EPSILONS,
    build_system,
    comparative_system_key,
    disambiguate_digit_variants,
    ensure_digit_checkpoints,
    system_dir_name,
    variant_name_for_mode,
)
from Validation.harness.output_writer import RESULTS_ROOT, write_seed_run

logger = logging.getLogger(__name__)


def add_common_args(parser: argparse.ArgumentParser, task_name: str) -> argparse.ArgumentParser:
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["nist_genomics", "tcga"],
        choices=get_supported_dataset_names(),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[AttackMode.MECHANISM.value, AttackMode.SYSTEM.value],
        choices=[m.value for m in AttackMode],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--reduced-seeds", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--data-cache", type=str, default="data_cache")
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_ROOT))
    parser.add_argument("--query-budget", type=int, default=1000)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--variant-setting", type=str, default="default")
    parser.add_argument("--digit-variants", nargs="+", default=["a6_s4_c3_r3"])
    parser.add_argument("--add-stability-gated-digit", action="store_true")
    parser.add_argument("--add-specificity-gated-digit", action="store_true")
    parser.add_argument("--dp-epsilons", nargs="+", type=float, default=DEFAULT_DP_EPSILONS)
    parser.add_argument("--skip-raw", action="store_true")
    parser.add_argument("--skip-digit", action="store_true")
    parser.add_argument("--skip-dp", action="store_true")
    parser.description = task_name
    return parser


def resolved_seeds(args) -> Tuple[List[int], str]:
    if getattr(args, "reduced_seeds", False):
        return [0, 1, 2], "reduced"
    return list(args.seeds), "full"


def resolve_modes(mode_names: Sequence[str]) -> List[AttackMode]:
    return [AttackMode(name) for name in mode_names]


def default_system_specs(
    *,
    include_raw: bool = True,
    include_digit: bool = True,
    include_stability_gated_digit: bool = False,
    include_specificity_gated_digit: bool = False,
    include_dp: bool = True,
    dp_epsilons: Optional[Sequence[float]] = None,
    digit_variants: Optional[Sequence[str]] = None,
) -> List[SystemSpec]:
    specs: List[SystemSpec] = []
    if include_raw:
        specs.append(SystemSpec(family="raw"))
    if include_digit:
        variants = list(digit_variants or ["a6_s4_c3_r3"])
        for variant in variants:
            specs.append(SystemSpec(family="digit", digit_variant=variant))
    if include_stability_gated_digit:
        variants = list(digit_variants or ["a6_s4_c3_r3"])
        for variant in variants:
            specs.append(SystemSpec(family="digit_stability_gate", digit_variant=variant))
    if include_specificity_gated_digit:
        variants = list(digit_variants or ["a6_s4_c3_r3"])
        for variant in variants:
            specs.append(SystemSpec(family="digit_specificity_gate", digit_variant=variant))
    if include_dp:
        for eps in list(dp_epsilons or DEFAULT_DP_EPSILONS):
            specs.append(SystemSpec(family="dp_laplace", epsilon=float(eps)))
    return specs


def device_from_arg(device_arg: str) -> torch.device:
    return torch.device(device_arg or ("cuda" if torch.cuda.is_available() else "cpu"))


def prepare_bundle(
    dataset_name: str,
    data_cache: str,
    split_seed: int,
) -> ComparativeDatasetBundle:
    return load_dataset_bundle(dataset_name, data_cache=data_cache, split_seed=split_seed)


def ensure_required_digit_checkpoints(
    bundle: ComparativeDatasetBundle,
    dataset_name: str,
    system_specs: Sequence[SystemSpec],
    modes: Sequence[AttackMode],
    seeds: Sequence[int],
    ckpt_dir: Path,
    device: torch.device,
    train_missing: bool,
) -> None:
    variants = sorted(
        {
            spec.digit_variant
            for spec in system_specs
            if spec.family in {"digit", "digit_stability_gate"}
        }
    )
    for digit_variant in variants:
        for mode in modes:
            ensure_digit_checkpoints(
                dataset_name=dataset_name,
                train_data=bundle.train,
                val_data=bundle.val,
                mode=mode,
                digit_variant=digit_variant,
                seeds=list(seeds),
                ckpt_dir=ckpt_dir,
                device=device,
                force_train=train_missing,
            )


def _capture_logs() -> Tuple[io.StringIO, logging.Handler]:
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(handler)
    return buf, handler


def _release_logs(buf: io.StringIO, handler: logging.Handler) -> str:
    logging.getLogger().removeHandler(handler)
    return buf.getvalue()


def _checkpoint_dataset_name(dataset_name: str, bundle: ComparativeDatasetBundle) -> str:
    # Use the comparative registry key for checkpoint paths so training and loading
    # do not diverge when a loader exposes a more specific internal dataset name.
    return bundle.attack_spec.dataset_name or dataset_name


def _system_builder(
    system_spec: SystemSpec,
    mode: AttackMode,
    bundle: ComparativeDatasetBundle,
    dataset_name: str,
    seed: int,
    ckpt_dir: Path,
    device: torch.device,
):
    system, factory, extra = build_system(
        system_spec=system_spec,
        private_data=bundle.train,
        mode=mode,
        seed=seed,
        ckpt_dir=ckpt_dir,
        dataset_name=_checkpoint_dataset_name(dataset_name, bundle),
        device=device,
    )
    rebuild = (lambda ds, _s=system: _s) if factory is None else factory
    return {"system": system, "factory": factory, "rebuild": rebuild, "config_extra": extra}


def write_attack_seed_result(
    *,
    run_cfg: AttackRunConfig,
    system_spec: SystemSpec,
    system_key: str,
    seed: int,
    result: Dict[str, Any],
    config_extra: Optional[Dict[str, Any]] = None,
    stdout_log: str = "",
) -> Path:
    events = list(result.get("events", []))
    predictions = list(result.get("predictions", []))
    reserved = {"events", "predictions"}
    raw_metrics = {k: v for k, v in result.items() if k not in reserved}
    metrics: Dict[str, Any] = {}
    for key, value in raw_metrics.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                if isinstance(subvalue, (int, float, str, bool)) or subvalue is None:
                    metrics[f"{key}_{subkey}"] = subvalue
        else:
            metrics[key] = value
    metrics.setdefault("pass_fail_threshold", run_cfg.pass_fail_threshold)
    metrics.setdefault("attacker_knowledge", run_cfg.attacker_knowledge)
    metrics.setdefault("query_budget", run_cfg.query_budget)

    extra = {
        "mode": run_cfg.mode.value,
        "query_budget": run_cfg.query_budget,
        "pass_fail_threshold": run_cfg.pass_fail_threshold,
        "attacker_knowledge": run_cfg.attacker_knowledge,
        **(config_extra or {}),
    }
    return write_seed_run(
        phase=run_cfg.phase,
        dataset=run_cfg.dataset,
        task=run_cfg.task,
        system_key=system_key,
        variant=run_cfg.variant,
        seed=seed,
        seed_mode=run_cfg.seed_mode,
        metrics=metrics,
        predictions=predictions,
        events=events,
        config_extra=extra,
        results_root=run_cfg.results_root,
        stdout_log=stdout_log,
    )


def aggregate_attack_results(
    per_seed_results: List[Dict[str, Any]],
    metric_names: Iterable[str],
    seed: int = 0,
) -> Dict[str, Any]:
    agg = {"per_seed": per_seed_results}
    agg.update(summarize_per_seed_metrics(per_seed_results, metric_names=metric_names, seed=seed))
    if per_seed_results and "events" in per_seed_results[0]:
        if per_seed_results[0]["events"]:
            first_event = per_seed_results[0]["events"][0]
            x_key = "query_count" if "query_count" in first_event else next(iter(first_event))
            y_candidates = [k for k in first_event.keys() if k != x_key and isinstance(first_event[k], (int, float))]
            if y_candidates:
                agg["curve_summary"] = summarize_curve_events(
                    [list(r.get("events", [])) for r in per_seed_results],
                    x_key=x_key,
                    y_key=y_candidates[0],
                    seed=seed,
                )
    return agg


def run_single_seed_attack(
    *,
    runner: Callable[..., Dict[str, Any]],
    runner_kwargs: Dict[str, Any],
) -> Tuple[Dict[str, Any], str]:
    buf, handler = _capture_logs()
    try:
        result = runner(**runner_kwargs)
    finally:
        stdout_log = _release_logs(buf, handler)
    return result, stdout_log


def summary_json_path(results_root: Path, phase: str, filename: str) -> Path:
    return results_root / phase / filename


def write_summary_json(results_root: Path, phase: str, filename: str, payload: Dict[str, Any]) -> Path:
    path = summary_json_path(results_root, phase, filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Summary → %s", path)
    return path


def execute_attack_grid(
    *,
    args,
    task: str,
    phase: str,
    metric_names: Sequence[str],
    per_seed_runner: Callable[..., Dict[str, Any]],
    per_seed_runner_name: str = "",
    system_specs: Optional[Sequence[SystemSpec]] = None,
    modes: Optional[Sequence[AttackMode]] = None,
) -> Dict[str, Any]:
    seeds, seed_mode = resolved_seeds(args)
    modes = list(modes or resolve_modes(args.modes))
    device = device_from_arg(args.device)
    results_root = Path(args.results_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    specs = list(
        system_specs
        or default_system_specs(
            include_raw=not args.skip_raw,
            include_digit=not args.skip_digit,
            include_stability_gated_digit=args.add_stability_gated_digit,
            include_specificity_gated_digit=args.add_specificity_gated_digit,
            include_dp=not args.skip_dp,
            digit_variants=args.digit_variants,
            dp_epsilons=args.dp_epsilons,
        )
    )
    label_digit_variants = disambiguate_digit_variants(specs)

    summary: Dict[str, Any] = {}
    for dataset_name in args.datasets:
        bundle = prepare_bundle(dataset_name, args.data_cache, args.split_seed)
        checkpoint_dataset_name = _checkpoint_dataset_name(dataset_name, bundle)
        ensure_required_digit_checkpoints(
            bundle=bundle,
            dataset_name=checkpoint_dataset_name,
            system_specs=specs,
            modes=modes,
            seeds=seeds,
            ckpt_dir=ckpt_dir,
            device=device,
            train_missing=args.train,
        )

        ds_summary: Dict[str, Any] = {}
        for mode in modes:
            mode_summary: Dict[str, Any] = {}
            for system_spec in specs:
                system_key = comparative_system_key(
                    system_spec,
                    disambiguate_digit=label_digit_variants,
                )
                per_seed_results: List[Dict[str, Any]] = []
                for seed in seeds:
                    builder = _system_builder(
                        system_spec,
                        mode,
                        bundle,
                        checkpoint_dataset_name,
                        seed,
                        ckpt_dir,
                        device,
                    )
                    logger.info(
                        "Task=%s dataset=%s mode=%s system=%s seed=%s %s",
                        task,
                        dataset_name,
                        mode.value,
                        system_key,
                        seed,
                        per_seed_runner_name,
                    )
                    result, stdout_log = run_single_seed_attack(
                        runner=per_seed_runner,
                        runner_kwargs={
                            "bundle": bundle,
                            "system_spec": system_spec,
                            "mode": mode,
                            "seed": seed,
                            "builder": builder,
                            "args": args,
                        },
                    )
                    run_cfg = AttackRunConfig(
                        dataset=dataset_name,
                        phase=phase,
                        task=task,
                        mode=mode,
                        variant=variant_name_for_mode(
                            mode,
                            result.get("_variant_setting", getattr(args, "variant_setting", "default")),
                        ),
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
                        config_extra=builder["config_extra"],
                        stdout_log=stdout_log,
                    )
                    per_seed_results.append(result)
                mode_summary[system_key] = aggregate_attack_results(
                    per_seed_results,
                    metric_names=metric_names,
                    seed=0,
                )
            ds_summary[mode.value] = mode_summary
        summary[dataset_name] = ds_summary

    write_summary_json(Path(args.results_dir), phase, f"{task}_summary.json", summary)
    return summary
