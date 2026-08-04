from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import torch

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from src.experiments import e5_1_state_stability as e


PROBE_PATH = e.RESULTS_DIR / "e5_1_faithful_state_attention_probe.json"
REPORT_PATH = e.REPORTS_DIR / "E5_1_FAITHFUL_STATE_ATTENTION_PROBE.md"


def _load_json(path: Path) -> Mapping[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _variant_rows(rows: Sequence[Mapping[str, object]], variant: str, *, control: str = "none") -> List[Mapping[str, object]]:
    return [row for row in rows if row.get("variant") == variant and row.get("control", "none") == control]


def _length_mean(rows: Sequence[Mapping[str, object]], variant: str, length: int) -> float:
    vals = [float(row["accuracy"]) for row in rows if row.get("variant") == variant and row.get("control", "none") == "none" and int(row["eval_length"]) == length]
    return e._mean(vals)


def _metric_mean(rows: Sequence[Mapping[str, object]], variant: str, key: str) -> float:
    vals = [float(row[key]) for row in _variant_rows(rows, variant) if row.get(key) is not None]
    return e._mean(vals)


def _control_degradation(control_rows: Sequence[Mapping[str, object]], variant: str, control: str) -> float:
    vals = [float(row.get("degradation", 0.0)) for row in control_rows if row.get("variant") == variant and row.get("control") == control]
    return e._mean(vals)


def _run_probe_controls(
    trained: Mapping[str, torch.nn.Module],
    *,
    budget: e.E5Budget,
    teacher: e.FittedFullContextQKVTeacher,
    device: torch.device,
    controls: Sequence[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for model_id, model in trained.items():
        seed = int(model_id.rsplit("_", 1)[-1])
        variant = model_id.rsplit("_seed_", 1)[0]
        base = e.evaluate_model(
            model,
            model_id=model_id,
            variant=variant,
            seed=seed,
            seq_len=64,
            n_examples=budget.eval_examples,
            device=device,
            teacher=teacher,
        )
        for control in controls:
            row = e.evaluate_model(
                model,
                model_id=model_id,
                variant=variant,
                seed=seed,
                seq_len=64,
                n_examples=budget.eval_examples,
                device=device,
                teacher=teacher,
                control=control,
            )
            row["degradation"] = float(base["accuracy"]) - float(row["accuracy"])
            row["passes"] = float(row.get("memory_dependent_accuracy") or 0.0) <= float(base.get("memory_dependent_accuracy") or 0.0) - 0.05
            rows.append(row)
    return rows


def _write_report(
    *,
    result: Mapping[str, object],
    baseline_result: Mapping[str, object],
    control_rows: Sequence[Mapping[str, object]],
    variants: Sequence[str],
    preflight: Mapping[str, object],
) -> None:
    rows = result.get("rows", [])  # type: ignore[assignment]
    capacities = result.get("capacities", {})  # type: ignore[assignment]
    baseline_rows = baseline_result.get("rows", []) if baseline_result else []  # type: ignore[assignment]
    post64 = e._variant_mean(baseline_rows, "post_attention_residual_memory", 64) if baseline_rows else 0.0
    current64 = e._variant_mean(baseline_rows, "current_only_no_state", 64) if baseline_rows else 0.0

    lines = [
        "# E5.1 Faithful State-Attention Probe",
        "",
        f"- CUDA: {preflight.get('cuda_available')} on {preflight.get('gpu_name')}",
        f"- Variants: {', '.join(variants)}",
        f"- Steps: {result.get('steps')}; seeds: {result.get('seeds')}; eval examples: {result.get('eval_examples')}",
        f"- Baseline N64: current-only {current64:.4f}; post-attention residual {post64:.4f}",
        "",
        "## Summary",
        "",
        "| variant | C | N64 | N128 | N256 | N512 | rank | cosine | geom mean | geom active | state-KV | route shuf | slot perm | slot drop |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in variants:
        lines.append(
            "| {variant} | {cap} | {n64:.4f} | {n128:.4f} | {n256:.4f} | {n512:.4f} | {rank:.2f} | {cos:.3f} | {gd:.3f} | {ga:.3f} | {skv:.0f} | {route:.4f} | {slot:.4f} | {drop:.4f} |".format(
                variant=variant,
                cap=int(capacities.get(variant, 0)),
                n64=_length_mean(rows, variant, 64),
                n128=_length_mean(rows, variant, 128),
                n256=_length_mean(rows, variant, 256),
                n512=_length_mean(rows, variant, 512),
                rank=_metric_mean(rows, variant, "state_effective_rank"),
                cos=_metric_mean(rows, variant, "pairwise_slot_cosine_mean"),
                gd=_metric_mean(rows, variant, "geometry_gate_density_mean"),
                ga=_metric_mean(rows, variant, "geometry_gate_active_fraction"),
                skv=_metric_mean(rows, variant, "fixed_state_kv_token_count"),
                route=_control_degradation(control_rows, variant, "route_shuffle"),
                slot=_control_degradation(control_rows, variant, "slot_permutation"),
                drop=_control_degradation(control_rows, variant, "slot_dropout_eval"),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `state-KV` is the mean number of fixed rolling-state slots appended as K/V at the final step.",
            "- `geom mean` is average sigmoid gate mass over learned possibility vectors; `geom active` is the fraction above 0.5.",
            "- Positive shuffle/drop degradation means the corresponding organization mattered for the answer.",
            "- A variant can use the state strongly while still failing slot organization if route/slot controls do not hurt.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> str:
    preflight = e.cuda_preflight_status(device=args.device, allow_cpu_smoke=False)
    if not preflight["passes"]:
        raise RuntimeError(e.DECISION_CUDA_REQUIRED)
    device = e.assert_training_device(args.device, allow_cpu_smoke=False)
    e.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    e.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    unknown = [variant for variant in variants if variant not in e.VARIANT_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")

    budget = e.E5Budget(
        train_lengths=(8, 16, 32),
        eval_lengths=tuple(int(x) for x in args.eval_lengths.split(",")),
        extrapolation_lengths=(),
        seeds=tuple(int(x) for x in args.seeds.split(",")),
        teacher_eval_examples=args.eval_examples,
        eval_examples=args.eval_examples,
        train_batch_size=args.batch_size,
        baseline_steps=0,
        student_steps=args.steps,
        randomized_label_steps=0,
        learning_rate=args.learning_rate,
        d_model=args.d_model,
        n_layers=1,
        n_heads=4,
        num_state_slots=args.num_state_slots,
        state_dim=args.d_model,
        curriculum=True,
        state_ablation_margin=0.55,
        organized_writes=True,
    )

    teacher = e.FittedFullContextQKVTeacher(d_model=budget.d_model).to(device)
    start = time.perf_counter()
    result, trained = e.train_variant_set(
        teacher,
        budget=budget,
        device=device,
        variants=variants,
        steps=args.steps,
        seeds=budget.seeds,
        output_path=PROBE_PATH,
        tag="faithful_state_attention_probe",
    )
    controls = tuple(item.strip() for item in args.controls.split(",") if item.strip())
    control_rows = _run_probe_controls(trained, budget=budget, teacher=teacher, device=device, controls=controls)
    baseline_result = _load_json(e.BASELINES_PATH)

    payload = dict(result)
    payload["preflight"] = preflight
    payload["control_rows"] = control_rows
    payload["controls"] = list(controls)
    payload["eval_examples"] = args.eval_examples
    payload["wall_clock_time"] = round(time.perf_counter() - start, 3)
    e._dump_json(PROBE_PATH, payload)
    _write_report(
        result=payload,
        baseline_result=baseline_result,
        control_rows=control_rows,
        variants=variants,
        preflight=preflight,
    )
    print(PROBE_PATH)
    print(REPORT_PATH)
    return str(PROBE_PATH)


def main() -> str:
    parser = argparse.ArgumentParser(description="Focused E5.1 probe for faithful rolling-state attention variants.")
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    parser.add_argument("--variants", default="R2_slot_dropout_retention,G1_state_kv_meanless,G2_state_kv_cross_update,G3_state_kv_strict_topk,G4_state_kv_rank_pressure")
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--eval-examples", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-state-slots", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--eval-lengths", default="32,64,128,256,512")
    parser.add_argument("--controls", default="zeroed_state_at_query,shuffled_state_across_batch,route_shuffle,slot_permutation,slot_dropout_eval,state_reset_every_step")
    return run(parser.parse_args())


if __name__ == "__main__":
    main()
