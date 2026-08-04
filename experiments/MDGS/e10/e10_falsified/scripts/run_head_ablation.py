"""Head-mode ablation for the E10 final confidence/outcome readouts."""

from __future__ import annotations

import argparse
import copy
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast

ROOT_DIR = Path(__file__).resolve().parents[2]
VARIANT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(VARIANT_DIR))

from eval_common import audit_monotonicity
from experiments.DIGIT.Extrapolation.e10.e10_falsified.extrapolation.config import Config
from experiments.DIGIT.Extrapolation.e10.e10_falsified.extrapolation.falsification import (
    SimpleVocabulary,
    active_head_modules,
    build_loader,
    build_model,
    class_weights_from_records,
    commitment_monotonicity_audit,
    compute_model_threshold_sensitivity,
    evaluate_model,
    load_trace_metadata,
    load_trace_split,
    set_global_seed,
    slice_records,
    write_json,
)


HEAD_MODES = ["mlp", "linear_evidence_only", "ordered_threshold"]


def _head_modules(model: torch.nn.Module, head_mode: str) -> list[torch.nn.Module]:
    return active_head_modules(model, head_mode)


def _train_head_mode(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    train_records,
    val_records,
    device: torch.device,
    *,
    head_mode: str,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, object]:
    model = model.to(device)
    head_modules = _head_modules(model, head_mode)
    for module in head_modules:
        module.train()
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError(f"No trainable parameters found for head_mode={head_mode!r}")

    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)
    confidence_weights = class_weights_from_records(train_records, "confidence", num_classes=3).to(device)
    outcome_weights = class_weights_from_records(train_records, "outcome", num_classes=3).to(device)
    confidence_ce = torch.nn.CrossEntropyLoss(weight=confidence_weights)
    outcome_ce = torch.nn.CrossEntropyLoss(weight=outcome_weights)

    best_state = copy.deepcopy(model.state_dict())
    best_score = float("-inf")
    best_epoch = 0
    history: list[dict[str, float]] = []
    patience_counter = 0

    for epoch in range(epochs):
        model.eval()
        for module in head_modules:
            module.train()
        epoch_loss = 0.0
        epoch_items = 0
        for batch in train_loader:
            queries, trace_inputs, _, prim_targets, target_ids = batch
            queries = queries.to(device)
            trace_inputs = trace_inputs.to(device)
            prim_targets = prim_targets.to(device)
            target_ids = target_ids.to(device)

            optimizer.zero_grad(set_to_none=True)
            amp_context = autocast("cuda", enabled=use_amp) if use_amp else nullcontext()
            with amp_context:
                out = model(
                    queries,
                    trace_inputs,
                    target_ids,
                    bottleneck_mode="hard",
                    tau=1.0,
                    skip_decoder=True,
                )
                prim = out["primitives"]
                loss = confidence_ce(prim.confidence_logits, prim_targets[:, 2]) + outcome_ce(
                    prim.outcome_logits, prim_targets[:, 3]
                )

            if not torch.isfinite(loss):
                continue

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

            batch_size = int(queries.size(0))
            epoch_loss += float(loss.item()) * batch_size
            epoch_items += batch_size

        train_loss = epoch_loss / max(epoch_items, 1)
        val_metrics = evaluate_model(model, val_loader, device, records=val_records)
        val_score = (
            float(val_metrics.get("outcome_macro_f1", 0.0))
            + 0.50 * float(val_metrics.get("failure_recall", 0.0))
            - 0.50 * float(val_metrics.get("unsafe_success_rate_correctness", 0.0))
        )
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(train_loss),
                "val_outcome_macro_f1": float(val_metrics.get("outcome_macro_f1", 0.0)),
                "val_failure_recall": float(val_metrics.get("failure_recall", 0.0)),
                "val_unsafe_success_rate_correctness": float(
                    val_metrics.get("unsafe_success_rate_correctness", 0.0)
                ),
                "val_score": float(val_score),
            }
        )
        if val_score > best_score:
            best_score = float(val_score)
            best_epoch = int(epoch + 1)
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    return {
        "best_state": best_state,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "history": history,
    }


def _summary_row(
    mode: str,
    pre_metrics: dict[str, float],
    post_metrics: dict[str, float],
    monotonicity: dict[str, float],
    commitment_monotonicity: dict[str, float],
    threshold_swing: float,
    train_result: dict[str, object],
) -> dict[str, object]:
    return {
        "mode": mode,
        "best_epoch": int(train_result["best_epoch"]),
        "best_score": float(train_result["best_score"]),
        "pre": pre_metrics,
        "post": post_metrics,
        "monotonicity": monotonicity,
        "commitment_monotonicity": commitment_monotonicity,
        "max_outcome_share_swing": float(threshold_swing),
    }


def _recommend(rows: list[dict[str, object]]) -> dict[str, object]:
    return min(
        rows,
        key=lambda row: (
            float(row["post"].get("unsafe_success_rate_correctness", float("inf"))),
            -float(row["post"].get("outcome_macro_f1", float("-inf"))),
            -float(row["post"].get("failure_recall", float("-inf"))),
            float(row["monotonicity"].get("monotonicity_violation_rate", float("inf"))),
            float(row["commitment_monotonicity"].get("commitment_monotonicity_violation_rate", float("inf"))),
        ),
    )


def _format_markdown(rows: list[dict[str, object]], recommended: dict[str, object]) -> str:
    headers = [
        "mode",
        "pre_f1",
        "post_f1",
        "post_fail_rec",
        "post_unsafe",
        "post_succ_prec",
        "mono_v",
        "commit_v",
        "swing",
    ]
    lines = [
        "# E10 Head Ablation",
        "",
        f"Recommended mode: `{recommended['mode']}`",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        post = row["post"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["mode"]),
                    f"{float(row['pre'].get('outcome_macro_f1', 0.0)):.3f}",
                    f"{float(post.get('outcome_macro_f1', 0.0)):.3f}",
                    f"{float(post.get('failure_recall', 0.0)):.3f}",
                    f"{float(post.get('unsafe_success_rate_correctness', 0.0)):.3f}",
                    f"{float(post.get('success_precision_vs_is_correct', 0.0)):.3f}",
                    f"{float(row['monotonicity'].get('monotonicity_violation_rate', 0.0)):.3f}",
                    f"{float(row['commitment_monotonicity'].get('commitment_monotonicity_violation_rate', 0.0)):.3f}",
                    f"{float(row['max_outcome_share_swing']):.3f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--trace-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--modes", type=str, default=",".join(HEAD_MODES))
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--val-offset", type=int, default=0)
    parser.add_argument("--test-offset", type=int, default=0)
    args = parser.parse_args()

    config = Config()
    trace_dir = Path(args.trace_dir).expanduser().resolve() if args.trace_dir else (VARIANT_DIR / config.trace_dir).resolve()
    device = torch.device(
        args.device if args.device is not None else (config.device if torch.cuda.is_available() else "cpu")
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (Path(__file__).resolve().parents[2] / "results" / "e10" / "head_ablation").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed)
    metadata = load_trace_metadata(trace_dir)
    vocab = SimpleVocabulary()

    train_records = slice_records(load_trace_split(trace_dir, "train"), limit=args.train_limit, offset=args.train_offset)
    val_records = slice_records(load_trace_split(trace_dir, "val"), limit=args.val_limit, offset=args.val_offset)
    test_records = slice_records(load_trace_split(trace_dir, "test"), limit=args.test_limit, offset=args.test_offset)

    train_dataset, train_loader = build_loader(
        train_records,
        vocab,
        max_output_len=config.max_output_len,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    val_dataset, val_loader = build_loader(
        val_records,
        vocab,
        max_output_len=config.max_output_len,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    test_dataset, test_loader = build_loader(
        test_records,
        vocab,
        max_output_len=config.max_output_len,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    del train_dataset, val_dataset, test_dataset

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    rows: list[dict[str, object]] = []

    for mode in modes:
        set_global_seed(args.seed)
        strict = mode == "mlp"
        model = build_model(
            config,
            vocab,
            checkpoint_path=args.checkpoint,
            head_mode=mode,
            trace_metadata=metadata,
            device=device,
            strict=strict,
        )
        pre_metrics = evaluate_model(model, test_loader, device, records=test_records)
        train_result = _train_head_mode(
            model,
            train_loader,
            val_loader,
            train_records,
            val_records,
            device,
            head_mode=mode,
            epochs=args.epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            weight_decay=config.weight_decay,
        )
        model.load_state_dict(train_result["best_state"])
        post_metrics, bundle = evaluate_model(
            model,
            test_loader,
            device,
            records=test_records,
            collect_bundle=True,
        )
        monotonicity = audit_monotonicity(model, test_loader, device)
        commitment = commitment_monotonicity_audit(model, test_loader, device)
        threshold = compute_model_threshold_sensitivity(bundle["probs"]["outcome"])
        row = _summary_row(
            mode,
            pre_metrics,
            post_metrics,
            monotonicity,
            commitment,
            float(threshold["max_outcome_share_swing"]),
            train_result,
        )
        row["threshold_sensitivity"] = threshold
        rows.append(row)

        mode_dir = output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": train_result["best_state"],
                "mode": mode,
                "seed": args.seed,
                "best_epoch": train_result["best_epoch"],
                "best_score": train_result["best_score"],
                "post_metrics": post_metrics,
            },
            mode_dir / "best_state.pt",
        )
        write_json(mode_dir / "metrics.json", row)

    recommended = _recommend(rows)
    markdown = _format_markdown(rows, recommended)
    (output_dir / "summary.md").write_text(markdown)
    summary = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "trace_dir": str(trace_dir),
        "seed": args.seed,
        "modes": modes,
        "recommended_mode": recommended["mode"],
        "rows": rows,
    }
    write_json(output_dir / "summary.json", summary)
    print(markdown.rstrip())
    print(f"Recommended mode: {recommended['mode']}")


if __name__ == "__main__":
    main()
