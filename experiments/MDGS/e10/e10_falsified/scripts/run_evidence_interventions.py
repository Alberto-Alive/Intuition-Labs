"""No-retrain falsification harness for the E10 shared evidence state."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
VARIANT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(VARIANT_DIR))

from experiments.DIGIT.Extrapolation.e10.e10_falsified.extrapolation.config import Config
from experiments.DIGIT.Extrapolation.e10.e10_falsified.extrapolation.falsification import (
    SimpleVocabulary,
    build_loader,
    build_model,
    load_trace_metadata,
    load_trace_split,
    set_global_seed,
    slice_records,
    write_json,
)


INTERVENTIONS = {
    "boundary_up": {
        "description": "Increase boundary mass and expect more ambiguity / less commitment.",
        "evidence_override": {"delta_boundary": 0.25, "delta_order": 0.0, "delta_support": 0.0, "apply_to": "all_paths"},
        "expected_signs": {
            "ambiguity_cert": +1,
            "success_guard": -1,
            "success_cert": -1,
            "commitment_depth": -1,
            "support_cert": -1,
            "success_prob": -1,
            "uncertain_prob": +1,
        },
    },
    "support_down": {
        "description": "Decrease support and expect weaker support / weaker commitment.",
        "evidence_override": {"delta_support": -0.25, "delta_order": 0.0, "delta_boundary": 0.0, "apply_to": "all_paths"},
        "expected_signs": {
            "support_cert": -1,
            "success_guard": -1,
            "success_cert": -1,
            "commitment_depth": -1,
            "ambiguity_cert": +1,
            "success_prob": -1,
            "failure_prob": +1,
        },
    },
    "order_failure": {
        "description": "Push order toward failure and expect success to drop.",
        "evidence_override": {"delta_order": -0.35, "delta_boundary": 0.0, "delta_support": 0.0, "apply_to": "all_paths"},
        "expected_signs": {
            "success_base": -1,
            "failure_cert": +1,
            "success_guard": -1,
            "success_cert": -1,
            "commitment_depth": -1,
            "success_prob": -1,
            "failure_prob": +1,
        },
    },
}


def _sign_rate(delta: torch.Tensor, expected_sign: int) -> float:
    if expected_sign > 0:
        return float((delta > 0).float().mean().item())
    return float((delta < 0).float().mean().item())


def _evaluate_intervention(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    evidence_override: dict,
    expected_signs: dict[str, int],
) -> dict[str, object]:
    deltas: dict[str, list[torch.Tensor]] = {
        "success_guard": [],
        "success_cert": [],
        "ambiguity_cert": [],
        "support_cert": [],
        "commitment_depth": [],
        "success_base": [],
        "failure_cert": [],
        "success_prob": [],
        "uncertain_prob": [],
        "failure_prob": [],
        "predicted_class_flip": [],
        "outcome_logits": [],
        "outcome_probs": [],
    }
    sign_hits: dict[str, list[float]] = {key: [] for key in expected_signs}

    model.eval()
    with torch.no_grad():
        for batch in loader:
            queries, trace_inputs, _, _, target_ids = batch
            queries = queries.to(device)
            trace_inputs = trace_inputs.to(device)
            target_ids = target_ids.to(device)

            base_out = model(
                queries,
                trace_inputs,
                target_ids,
                bottleneck_mode="hard",
                tau=1.0,
                skip_decoder=True,
            )
            diffusion_state = base_out.get("diffusion_state")
            perturbed_out = model(
                queries,
                trace_inputs,
                target_ids,
                bottleneck_mode="hard",
                tau=1.0,
                skip_decoder=True,
                evidence_override=evidence_override,
                diffusion_override=diffusion_state,
            )

            base = base_out["primitives"]
            perturbed = perturbed_out["primitives"]
            base_probs = torch.softmax(base.outcome_logits, dim=-1)
            perturbed_probs = torch.softmax(perturbed.outcome_logits, dim=-1)

            for name in ["success_guard", "success_cert", "ambiguity_cert", "support_cert", "commitment_depth", "success_base", "failure_cert"]:
                delta = getattr(perturbed, name) - getattr(base, name)
                deltas[name].append(delta.detach().cpu())
                if name in expected_signs:
                    sign_hits[name].append(_sign_rate(delta, expected_signs[name]))

            prob_delta = perturbed_probs - base_probs
            for idx, name in enumerate(["success_prob", "uncertain_prob", "failure_prob"]):
                delta = prob_delta[:, idx]
                deltas[name].append(delta.detach().cpu())
                if name in expected_signs:
                    sign_hits[name].append(_sign_rate(delta, expected_signs[name]))

            logits_delta = perturbed.outcome_logits - base.outcome_logits
            deltas["outcome_logits"].append(logits_delta.detach().cpu())
            deltas["outcome_probs"].append(prob_delta.detach().cpu())
            flip_rate = (perturbed_probs.argmax(dim=-1) != base_probs.argmax(dim=-1)).float().cpu()
            deltas["predicted_class_flip"].append(flip_rate)

    summary = {
        "mean_delta": {
            name: float(torch.cat(values, dim=0).mean().item())
            for name, values in deltas.items()
            if values and name not in {"outcome_logits", "outcome_probs", "predicted_class_flip"}
        },
        "mean_outcome_logit_delta": torch.cat(deltas["outcome_logits"], dim=0).mean(dim=0).tolist(),
        "mean_outcome_prob_delta": torch.cat(deltas["outcome_probs"], dim=0).mean(dim=0).tolist(),
        "predicted_class_flip_rate": float(torch.cat(deltas["predicted_class_flip"], dim=0).mean().item()),
        "sign_consistency": {
            name: float(sum(values) / max(len(values), 1))
            for name, values in sign_hits.items()
        },
    }
    summary["overall_sign_consistency"] = float(
        sum(summary["sign_consistency"].values()) / max(len(summary["sign_consistency"]), 1)
    )
    return summary


def _conclusion(intervention_summaries: dict[str, dict[str, object]]) -> str:
    if not intervention_summaries:
        return "evidence mostly descriptive"
    target_rates = [float(payload.get("overall_sign_consistency", 0.0)) for payload in intervention_summaries.values()]
    flip_rates = [float(payload.get("predicted_class_flip_rate", 0.0)) for payload in intervention_summaries.values()]
    if min(target_rates) >= 0.60 and max(flip_rates) >= 0.10:
        return "evidence looks causal"
    return "evidence mostly descriptive"


def _format_table(rows: list[dict[str, object]]) -> str:
    headers = [
        "intervention",
        "sign_consistency",
        "flip_rate",
        "guard_delta",
        "succ_cert_delta",
        "ambiguity_delta",
        "support_delta",
        "commitment_delta",
        "succ_logit_delta",
    ]
    lines = [
        "# E10 Evidence Intervention Audit",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["intervention"]),
                    f"{float(row['overall_sign_consistency']):.3f}",
                    f"{float(row['predicted_class_flip_rate']):.3f}",
                    f"{float(row['mean_delta']['success_guard']):.4f}",
                    f"{float(row['mean_delta']['success_cert']):.4f}",
                    f"{float(row['mean_delta']['ambiguity_cert']):.4f}",
                    f"{float(row['mean_delta']['support_cert']):.4f}",
                    f"{float(row['mean_delta']['commitment_depth']):.4f}",
                    f"{float(row['mean_outcome_logit_delta'][0]):.4f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--trace-dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--apply-to", type=str, default="all_paths")
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--path-index", type=int, default=None)
    parser.add_argument("--delta-boundary", type=float, default=0.25)
    parser.add_argument("--delta-support", type=float, default=-0.25)
    parser.add_argument("--delta-order", type=float, default=-0.35)
    args = parser.parse_args()

    config = Config()
    trace_dir = Path(args.trace_dir).expanduser().resolve() if args.trace_dir else (VARIANT_DIR / config.trace_dir).resolve()
    device = torch.device(
        args.device if args.device is not None else (config.device if torch.cuda.is_available() else "cpu")
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (Path(__file__).resolve().parents[2] / "results" / "e10" / "evidence_interventions").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed)
    metadata = load_trace_metadata(trace_dir)
    vocab = SimpleVocabulary()
    records = slice_records(load_trace_split(trace_dir, args.split), limit=args.limit, offset=args.offset)
    dataset, loader = build_loader(
        records,
        vocab,
        max_output_len=config.max_output_len,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    del dataset

    model = build_model(
        config,
        vocab,
        checkpoint_path=args.checkpoint,
        head_mode="mlp",
        trace_metadata=metadata,
        device=device,
        strict=True,
    )

    intervention_summaries: dict[str, dict[str, object]] = {}
    for name, spec in INTERVENTIONS.items():
        override = dict(spec["evidence_override"])
        if name == "boundary_up":
            override["delta_boundary"] = float(args.delta_boundary)
        elif name == "support_down":
            override["delta_support"] = float(args.delta_support)
        elif name == "order_failure":
            override["delta_order"] = float(args.delta_order)
        override["apply_to"] = args.apply_to
        if args.apply_to == "topk_paths":
            override["topk_paths_k"] = int(args.topk)
        if args.apply_to == "random_path" and args.path_index is not None:
            override["path_index"] = int(args.path_index)
        summary = _evaluate_intervention(
            model,
            loader,
            device,
            override,
            dict(spec["expected_signs"]),
        )
        summary["intervention"] = name
        summary["description"] = spec["description"]
        summary["evidence_override"] = override
        intervention_summaries[name] = summary

    conclusion = _conclusion(intervention_summaries)
    rows = list(intervention_summaries.values())
    markdown = _format_table(rows) + "\nConclusion: " + conclusion + "\n"
    (output_dir / "summary.md").write_text(markdown)

    payload = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "trace_dir": str(trace_dir),
        "split": args.split,
        "limit": args.limit,
        "offset": args.offset,
        "seed": args.seed,
        "apply_to": args.apply_to,
        "topk": args.topk,
        "path_index": args.path_index,
        "conclusion": conclusion,
        "interventions": intervention_summaries,
    }
    write_json(output_dir / "summary.json", payload)
    print(markdown.rstrip())
    print(conclusion)


if __name__ == "__main__":
    main()
