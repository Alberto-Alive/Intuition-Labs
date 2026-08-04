from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping

from src.experiments.stage8_architecture_registry import architecture_config_from_dict, default_latent_config
from src.experiments.stage8_capacity_evaluator import Stage8EvaluationConfig, dump_json
from src.experiments.stage8_search_scheduler import (
    BASELINE_CAPACITY_PATH,
    BEST_CONFIG_PATH,
    FINALIST_RESULTS_PATH,
    FINAL_VALIDATION_RESULTS_PATH,
    PROMOTED_CONFIGS_PATH,
    SEARCH_DATABASE_PATH,
    STAGE8A_CONTROLS_AUDIT_PATH,
    STAGE8A_REPORT_PATH,
    STAGE8A_RESULTS_PATH,
    CONFIG_FREEZE_MANIFEST_PATH,
    STAGE8B_CAPACITY_CURVES_PATH,
    STAGE8B_COMPUTE_AUDIT_PATH,
    STAGE8B_CONTROLS_AUDIT_PATH,
    STAGE8B_REPORT_PATH,
    Stage8SearchBudget,
    run_stage8a_baseline_validation,
    run_stage8b_architecture_search,
    run_stage8c_final_validation,
)


REPORT_SEARCH_PATH = Path("reports/STAGE8_LATENT_ATTENTION_CAPACITY_SEARCH.md")
REPORT_FINAL_PATH = Path("reports/STAGE8_FINAL_VALIDATION.md")
REPORT_OVERVIEW_PATH = Path("reports/STAGE8_LATENT_ATTENTION_CAPACITY.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 8 latent attention capacity search.")
    parser.add_argument("--stage", choices=("8a", "8b", "8c", "all", "smoke"), default="smoke")
    parser.add_argument("--max-configs", type=int, default=50)
    parser.add_argument("--max-promoted", type=int, default=8)
    parser.add_argument("--max-finalists", type=int, default=2)
    parser.add_argument("--max-n", type=int, default=128)
    parser.add_argument("--max-wall-clock-seconds", type=float, default=1800.0)
    parser.add_argument("--train-examples", type=int, default=192)
    parser.add_argument("--eval-examples", type=int, default=96)
    parser.add_argument("--final-examples", type=int, default=128)
    parser.add_argument("--stage8a-seeds", default=None, help="Comma-separated seeds for Stage 8A, for example 0,1,2.")
    parser.add_argument("--no-controls", action="store_true")
    parser.add_argument("--stage8c-config", default=None, help="Optional JSON/YAML-like config path; defaults to Stage 8B selected config.")
    args = parser.parse_args()

    if args.stage == "smoke":
        budget = Stage8SearchBudget(
            max_architecture_configs=3,
            max_promoted_configs=1,
            max_finalists=1,
            max_wall_clock_seconds=min(float(args.max_wall_clock_seconds), 120.0),
            max_n=min(int(args.max_n), 16),
            stage8a_seeds=(0,),
            stage8b_small_seeds=(0,),
            stage8b_promoted_seeds=(0,),
            stage8b_finalist_seeds=(0,),
            stage8c_seeds=(10,),
            train_examples=min(int(args.train_examples), 24),
            eval_examples=min(int(args.eval_examples), 24),
            final_examples=min(int(args.final_examples), 24),
            run_controls=not bool(args.no_controls),
        )
    else:
        stage8a_seeds = _parse_seed_tuple(args.stage8a_seeds, default=(0, 1, 2))
        budget = Stage8SearchBudget(
            max_architecture_configs=int(args.max_configs),
            max_promoted_configs=int(args.max_promoted),
            max_finalists=int(args.max_finalists),
            max_wall_clock_seconds=float(args.max_wall_clock_seconds),
            max_n=int(args.max_n),
            train_examples=int(args.train_examples),
            eval_examples=int(args.eval_examples),
            final_examples=int(args.final_examples),
            stage8a_seeds=stage8a_seeds,
            run_controls=not bool(args.no_controls),
        )
    eval_config = Stage8EvaluationConfig(
        train_examples=budget.train_examples,
        eval_examples=budget.eval_examples,
        final_examples=budget.final_examples,
    )

    outputs: Dict[str, object] = {"started_at_utc": _now(), "stage": args.stage, "budget": asdict(budget)}
    baseline_payload: Mapping[str, object] | None = None
    if args.stage in {"8a", "all", "smoke"}:
        baseline_payload = run_stage8a_baseline_validation(budget, eval_config)
        outputs["stage8a"] = baseline_payload
    elif BASELINE_CAPACITY_PATH.exists():
        baseline_payload = json.loads(BASELINE_CAPACITY_PATH.read_text(encoding="utf-8"))

    if args.stage in {"8b", "all", "smoke"}:
        search_payload = run_stage8b_architecture_search(budget, eval_config, baseline_payload)
        outputs["stage8b"] = search_payload

    if args.stage in {"8c", "all"}:
        config = _load_stage8c_config(args.stage8c_config)
        final_payload = run_stage8c_final_validation(config, budget, eval_config, baseline_payload)
        outputs["stage8c"] = final_payload
    elif not FINAL_VALIDATION_RESULTS_PATH.exists():
        placeholder = {
            "stage": "8C",
            "status": "not_run",
            "decision": "NO_STAGE8C_CLAIM",
            "note": "Stage 8C has not been run. Do not claim target achievement.",
        }
        dump_json(FINAL_VALIDATION_RESULTS_PATH, placeholder)

    _write_reports(outputs)
    print(
        "stage8: wrote {baseline}, {db}, {promoted}, {finalists}, {final_validation}, {report}".format(
            baseline=BASELINE_CAPACITY_PATH,
            db=SEARCH_DATABASE_PATH,
            promoted=PROMOTED_CONFIGS_PATH,
            finalists=FINALIST_RESULTS_PATH,
            final_validation=FINAL_VALIDATION_RESULTS_PATH,
            report=REPORT_SEARCH_PATH,
        )
    )
    if args.stage == "8a":
        print(
            "stage8a: wrote {results}, {controls}, {report}".format(
                results=STAGE8A_RESULTS_PATH,
                controls=STAGE8A_CONTROLS_AUDIT_PATH,
                report=STAGE8A_REPORT_PATH,
            )
        )
    if args.stage == "8b":
        print(
            "stage8b: wrote {controls}, {compute}, {curves}, {manifest}, {report}".format(
                controls=STAGE8B_CONTROLS_AUDIT_PATH,
                compute=STAGE8B_COMPUTE_AUDIT_PATH,
                curves=STAGE8B_CAPACITY_CURVES_PATH,
                manifest=CONFIG_FREEZE_MANIFEST_PATH,
                report=STAGE8B_REPORT_PATH,
            )
        )


def _load_stage8c_config(path: str | None):
    if path:
        text = Path(path).read_text(encoding="utf-8")
        if path.endswith(".json"):
            return architecture_config_from_dict(json.loads(text))
        values: Dict[str, object] = {}
        for line in text.splitlines():
            if not line or line.strip().startswith("#") or ":" not in line:
                continue
            key, raw = line.split(":", 1)
            key = key.strip()
            if key in {"stage8b_capacity_C", "stage8b_capacity_ratio_vs_baseline"}:
                continue
            values[key] = json.loads(raw.strip())
        return architecture_config_from_dict(values)
    if FINALIST_RESULTS_PATH.exists():
        payload = json.loads(FINALIST_RESULTS_PATH.read_text(encoding="utf-8"))
        selected = payload.get("selected_final_architecture")
        if isinstance(selected, Mapping) and isinstance(selected.get("architecture_parameters"), Mapping):
            return architecture_config_from_dict(dict(selected["architecture_parameters"]))
    if BEST_CONFIG_PATH.exists():
        return _load_stage8c_config(str(BEST_CONFIG_PATH))
    return default_latent_config()


def _parse_seed_tuple(raw: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if not raw:
        return default
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    return values or default


def _write_reports(outputs: Mapping[str, object]) -> None:
    REPORT_SEARCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    search = outputs.get("stage8b") if isinstance(outputs.get("stage8b"), Mapping) else {}
    baseline = outputs.get("stage8a") if isinstance(outputs.get("stage8a"), Mapping) else {}
    final = outputs.get("stage8c") if isinstance(outputs.get("stage8c"), Mapping) else {}
    lines = [
        "# Stage 8 Latent Attention Capacity Search",
        "",
        f"Updated: {_now()}",
        "",
        "This report is conservative by construction. Stage 8A/8B results are benchmark and dev-search outputs only; no final capacity claim is made until Stage 8C held-out gates pass.",
        "",
        "## Artifacts",
        "",
        f"- Baseline capacity: `{BASELINE_CAPACITY_PATH}`",
        f"- Search database: `{SEARCH_DATABASE_PATH}`",
        f"- Promoted configs: `{PROMOTED_CONFIGS_PATH}`",
        f"- Finalists: `{FINALIST_RESULTS_PATH}`",
        f"- Frozen best config: `{BEST_CONFIG_PATH}`",
        "",
        "## Current Status",
        "",
        f"- Stage 8A best monolithic capacity: {_capacity_text(baseline.get('best_monolithic_capacity') if isinstance(baseline, Mapping) else None)}",
        f"- Stage 8B selected architecture: {_selected_text(search)}",
        f"- Stage 8B status: {search.get('status', 'not_run') if isinstance(search, Mapping) else 'not_run'}",
        "",
        "## Decision",
        "",
        f"- Current decision output: {final.get('decision', 'NO_STAGE8C_CLAIM') if isinstance(final, Mapping) else 'NO_STAGE8C_CLAIM'}",
    ]
    REPORT_SEARCH_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    final_lines = [
        "# Stage 8 Final Validation",
        "",
        f"Updated: {_now()}",
        "",
        "Stage 8C must use frozen architecture and hyperparameters selected before final held-out evaluation.",
        "",
        f"Decision: {final.get('decision', 'not_run') if isinstance(final, Mapping) else 'not_run'}",
        f"Capacity ratio: {final.get('capacity_ratio', 'not_run') if isinstance(final, Mapping) else 'not_run'}",
        f"Critical controls pass: {final.get('critical_controls_pass', 'not_run') if isinstance(final, Mapping) else 'not_run'}",
        "",
        "No stronger claim should be made from Stage 8A or Stage 8B outputs.",
    ]
    REPORT_FINAL_PATH.write_text("\n".join(final_lines) + "\n", encoding="utf-8")
    REPORT_OVERVIEW_PATH.write_text("\n".join(lines + ["", "See also:", f"- `{REPORT_FINAL_PATH}`"]) + "\n", encoding="utf-8")


def _capacity_text(row) -> str:
    if not isinstance(row, Mapping):
        return "not_run"
    return f"{row.get('capacity', 0)} ({row.get('architecture_name', 'unknown')})"


def _selected_text(search) -> str:
    if not isinstance(search, Mapping):
        return "not_run"
    selected = search.get("selected_final_architecture")
    if not isinstance(selected, Mapping):
        return "none"
    return f"{selected.get('architecture_name')} capacity={selected.get('capacity_C')} ratio={selected.get('capacity_ratio_vs_baseline')}"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    main()
