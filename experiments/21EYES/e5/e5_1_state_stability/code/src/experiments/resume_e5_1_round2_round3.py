from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from src.experiments import e5_1_state_stability as e


def _load(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _merged_ranked_variants(*results: Mapping[str, object]) -> Sequence[str]:
    rows = []
    for result in results:
        rows.extend(result.get("rows", []))  # type: ignore[arg-type]
    variants = sorted({str(row["variant"]) for row in rows})
    merged = {
        "rows": rows,
        "capacities": {variant: e.capacity_from_rows([row for row in rows if row["variant"] == variant]) for variant in variants},
    }
    return e._rank_variants(merged)


def run(args: argparse.Namespace) -> str:
    preflight = e.write_preflight(device=args.device, allow_cpu_smoke=False)
    if not preflight["passes"]:
        print(e.DECISION_CUDA_REQUIRED)
        return e.DECISION_CUDA_REQUIRED

    device = e.assert_training_device(args.device, allow_cpu_smoke=False)
    replay_result = _load(e.REPLAY_PATH)
    round1_result = _load(e.ROUND1_PATH)
    baseline_result = _load(e.BASELINES_PATH)
    teacher_result = _load(e.TEACHER_PATH)

    budget = e.E5Budget(
        train_lengths=(8, 16, 32),
        eval_lengths=(32, 64, 128, 256, 512),
        extrapolation_lengths=(),
        seeds=(0, 1, 2),
        teacher_eval_examples=args.eval_examples,
        eval_examples=args.eval_examples,
        train_batch_size=args.batch_size,
        baseline_steps=0,
        student_steps=args.round2_steps,
        randomized_label_steps=80,
        learning_rate=2.0e-3,
        d_model=128,
        n_layers=1,
        n_heads=4,
        num_state_slots=32,
        state_dim=128,
        curriculum=True,
        state_ablation_margin=0.55,
        organized_writes=True,
    )
    teacher = e.FittedFullContextQKVTeacher(d_model=budget.d_model).to(device)

    round2_variants = [name for name, config in e.VARIANT_CONFIGS.items() if config.family == "round2"][: args.round2_limit]
    round2_budget = e.replace(budget, seeds=tuple(int(x) for x in args.round2_seeds.split(",")))
    round2_result, _round2_trained = e.train_variant_set(
        teacher,
        budget=round2_budget,
        device=device,
        variants=round2_variants,
        steps=args.round2_steps,
        seeds=round2_budget.seeds,
        output_path=e.ROUND2_PATH,
        tag="round2_combined_resume",
    )

    if args.round2_top_steps > args.round2_steps:
        top2 = e._rank_variants(round2_result)[:2]
        round2_top_result, _ = e.train_variant_set(
            teacher,
            budget=round2_budget,
            device=device,
            variants=top2,
            steps=args.round2_top_steps,
            seeds=round2_budget.seeds,
            output_path=e.ROUND2_PATH,
            tag="round2_top2_resume",
        )
        round2_top_result["round2_320"] = round2_result
        e._dump_json(e.ROUND2_PATH, round2_top_result)
        round2_for_selection = round2_top_result
    else:
        round2_for_selection = round2_result

    top3 = list(_merged_ranked_variants(round1_result, round2_result))[:3]
    round3_budget = e.replace(budget, seeds=tuple(int(x) for x in args.round3_seeds.split(",")), student_steps=args.round3_steps)
    round3_result, round3_trained = e.train_variant_set(
        teacher,
        budget=round3_budget,
        device=device,
        variants=top3,
        steps=args.round3_steps,
        seeds=round3_budget.seeds,
        output_path=e.ROUND3_PATH,
        tag="round3_final_stability_resume",
    )

    capacities = round3_result["capacities"]  # type: ignore[index]
    best_variant = max(
        capacities,
        key=lambda variant: (
            capacities[variant],
            e._length_min(round3_result["rows"], variant, 128),  # type: ignore[index]
            e._length_mean(round3_result["rows"], variant, 256),  # type: ignore[index]
            e._length_mean(round3_result["rows"], variant, 512),  # type: ignore[index]
        ),
    )
    top_ids = e._top_student_ids(round3_result, round3_trained, limit=3)
    control_result = e.run_controls(
        round3_trained,
        top_ids,
        teacher,
        budget=round3_budget,
        device=device,
        baseline_rows=baseline_result["rows"],  # type: ignore[arg-type]
        student_rows=round3_result["rows"],  # type: ignore[arg-type]
    )
    decision = e.decide(
        teacher_result=teacher_result,
        baseline_result=baseline_result,
        replay_result=replay_result,
        student_result=round3_result,
        control_result=control_result,
        extrapolation_rows=(),
    )

    best64 = e._variant_mean(round3_result["rows"], best_variant, 64)  # type: ignore[arg-type]
    best_trained_id = next((model_id for model_id in round3_trained if model_id.startswith(f"{best_variant}_seed_")), None)
    parameter_count = e.count_parameters(round3_trained[best_trained_id]) if best_trained_id else 0
    state_audit, memory_audit = e.write_state_and_memory_audits(
        budget=round3_budget,
        student_rows=round3_result["rows"],  # type: ignore[arg-type]
        control_result=control_result,
        best_variant=best_variant,
        parameter_count=parameter_count,
    )
    all_rows = []
    all_rows.extend(teacher_result.get("rows", []))  # type: ignore[arg-type]
    all_rows.extend(baseline_result.get("rows", []))  # type: ignore[arg-type]
    all_rows.extend(replay_result.get("rows", []))  # type: ignore[arg-type]
    all_rows.extend(round1_result.get("rows", []))  # type: ignore[arg-type]
    all_rows.extend(round2_for_selection.get("rows", []))  # type: ignore[arg-type]
    all_rows.extend(round3_result.get("rows", []))  # type: ignore[arg-type]
    all_rows.extend(control_result.get("rows", []))  # type: ignore[arg-type]
    e.write_capacity_curves(all_rows)
    e.write_best_config(
        budget=round3_budget,
        decision=decision,
        best_variant=best_variant,
        best_capacity=int(capacities[best_variant]),  # type: ignore[index]
        best64=best64,
    )
    e.write_seed_stability([replay_result, round1_result, round2_for_selection, round3_result])
    e.write_report(
        decision=decision,
        preflight=preflight,
        replay_result=replay_result,
        baseline_result=baseline_result,
        final_result=round3_result,
        control_result=control_result,
        state_audit=state_audit,
        memory_audit=memory_audit,
        best_variant=best_variant,
        replay_matched=e.replay_matches_prior_probe(replay_result),
    )
    print(decision)
    return decision


def main() -> str:
    parser = argparse.ArgumentParser(description="Resume E5.1 from completed Round 1 and run Round 2/3.")
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    parser.add_argument("--eval-examples", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--round2-limit", type=int, default=6)
    parser.add_argument("--round2-steps", type=int, default=320)
    parser.add_argument("--round2-top-steps", type=int, default=320)
    parser.add_argument("--round2-seeds", default="0,1,2,3,4")
    parser.add_argument("--round3-steps", type=int, default=320)
    parser.add_argument("--round3-seeds", default="0,1,2,3,4")
    return run(parser.parse_args())


if __name__ == "__main__":
    main()
