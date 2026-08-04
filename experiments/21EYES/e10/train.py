from __future__ import annotations

import argparse
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.nn import functional as F

from data import SyntheticBatcher
from eval import append_jsonl, evaluate_controls, evaluate_model, save_json, write_report
from model import (
    BLOCK_SPARSE_INTERWEAVE_VARIANTS,
    CROSSING_VARIANTS,
    MATRIX_INTERWOVEN_VARIANTS,
    build_model,
    model_summary,
    parameter_count,
)


E10_ROOT = Path(__file__).resolve().parent


try:
    import yaml
except ImportError as exc:  # pragma: no cover
    yaml = None
    YAML_IMPORT_ERROR = exc
else:
    YAML_IMPORT_ERROR = None


def _load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read configs/*.yaml") from YAML_IMPORT_ERROR
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config {path} did not parse to a mapping")
    return payload


def _dump_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _autocast(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    return nullcontext()


def _resolve_batch_size(value: Any, *, device: torch.device, train: bool, four_stream: bool) -> int:
    if value is None or str(value).lower() == "auto":
        if device.type == "cuda":
            if four_stream:
                return 32 if train else 24
            return 64 if train else 32
        return 4 if four_stream else 8
    return int(value)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(config)
    cfg.setdefault("model", {})
    cfg.setdefault("data", {})
    cfg.setdefault("training", {})
    cfg.setdefault("loss", {})
    cfg.setdefault("notes", {})
    variant = str(cfg.get("variant", cfg["model"].get("variant", "baseline_transformer")))
    cfg["variant"] = variant
    cfg["model"]["variant"] = variant
    cfg["model"].setdefault("max_seq_len", max(int(v) for v in cfg["data"].get("seq_len_eval", [256])))
    cfg.setdefault("seed", 0)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.max_steps is not None:
        cfg["training"]["max_steps"] = int(args.max_steps)
    if args.device is not None:
        cfg["training"]["device"] = args.device
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = int(args.batch_size)
    if args.eval_examples is not None:
        cfg["data"]["eval_examples"] = int(args.eval_examples)
        for profile in cfg["data"].get("eval_profiles", {}).values():
            profile["examples"] = int(args.eval_examples)
    return cfg


def _loss_for_batch(model: torch.nn.Module, batch, device: torch.device, amp_enabled: bool, config: Dict[str, Any]):
    loss_cfg = config.get("loss", {})
    variant = str(config.get("variant", "baseline_transformer"))
    with _autocast(device, amp_enabled):
        out = model(batch.input_ids, return_diagnostics=False)
        logits = out["logits"]
        task_loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), batch.labels.view(-1), ignore_index=-100)
        aux = out.get("aux_losses", {})
        align_loss = aux.get("avenue_alignment_loss", task_loss.new_zeros(()))
        div_loss = aux.get("avenue_diversity_loss", task_loss.new_zeros(()))
        orth_loss = aux.get("crossing_orthogonality_loss", task_loss.new_zeros(()))
        lambda_align = float(loss_cfg.get("lambda_align", 0.0)) if variant == "e8_best_reference" else 0.0
        lambda_div = float(loss_cfg.get("lambda_div", 0.0))
        lambda_orth = float(loss_cfg.get("lambda_orth", 0.0)) if variant in CROSSING_VARIANTS else 0.0
        if variant in {"baseline_transformer", "param_matched_baseline"}:
            lambda_div = 0.0
        loss = task_loss + lambda_align * align_loss + lambda_div * div_loss + lambda_orth * orth_loss
    return loss, {
        "task_loss": task_loss.detach(),
        "avenue_alignment_loss": align_loss.detach(),
        "avenue_diversity_loss": div_loss.detach(),
        "crossing_orthogonality_loss": orth_loss.detach(),
        "total_loss": loss.detach(),
    }


def run_training(config_path: Path, args: argparse.Namespace) -> Path:
    config = _prepare_config(_load_yaml(config_path), args)
    seed = int(config.get("seed", 0))
    variant = str(config["variant"])
    _set_seed(seed)

    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    requested_device = str(train_cfg.get("device", "cuda"))
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU for smoke/debug execution.")
        requested_device = "cpu"
    device = torch.device(requested_device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats(device)

    four_stream = (
        variant in CROSSING_VARIANTS
        or variant in MATRIX_INTERWOVEN_VARIANTS
        or variant in BLOCK_SPARSE_INTERWEAVE_VARIANTS
    )
    batch_size = _resolve_batch_size(data_cfg.get("batch_size", "auto"), device=device, train=True, four_stream=four_stream)
    eval_batch_size = _resolve_batch_size(
        data_cfg.get("eval_batch_size", "auto"), device=device, train=False, four_stream=four_stream
    )
    data_cfg["batch_size"] = batch_size
    data_cfg["eval_batch_size"] = eval_batch_size
    config["data"] = data_cfg

    model = build_model(config["model"]).to(device)
    total_params = parameter_count(model)
    trainable_params = parameter_count(model, trainable_only=True)
    output_root = Path(args.output_root).resolve() if args.output_root is not None else E10_ROOT / "results"
    run_dir = output_root / variant / str(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    _dump_yaml(run_dir / "config.yaml", config)
    (run_dir / "model_summary.txt").write_text(model_summary(model), encoding="utf-8")

    train_batcher = SyntheticBatcher(
        seed=seed,
        split="train",
        seq_len=int(data_cfg.get("seq_len_train", 128)),
        distractor_range=tuple(int(v) for v in data_cfg.get("train_distractor_range", [4, 14])),
        overwrite_range=tuple(int(v) for v in data_cfg.get("train_overwrite_range", [1, 4])),
    )
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(train_cfg.get("learning_rate", 3.0e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        betas=tuple(train_cfg.get("betas", [0.9, 0.95])),
    )
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except TypeError:  # pragma: no cover
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    max_steps = int(train_cfg.get("max_steps", 2500))
    eval_every = int(train_cfg.get("eval_every", 250))
    log_every = int(train_cfg.get("log_every", 50))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    print(
        f"Training {variant} seed={seed} on {device} for {max_steps} steps "
        f"(batch={batch_size}, eval_batch={eval_batch_size}, params={total_params})."
    )
    model.train()
    ema_loss: Optional[float] = None
    ema_task: Optional[float] = None
    train_examples = 0
    train_tokens = 0
    start = time.perf_counter()
    last_eval: Optional[Dict[str, Any]] = None

    for step in range(1, max_steps + 1):
        batch = train_batcher.sample(batch_size).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, loss_parts = _loss_for_batch(model, batch, device, amp_enabled, config)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        loss_value = float(loss_parts["total_loss"].float().cpu())
        task_value = float(loss_parts["task_loss"].float().cpu())
        ema_loss = loss_value if ema_loss is None else 0.95 * ema_loss + 0.05 * loss_value
        ema_task = task_value if ema_task is None else 0.95 * ema_task + 0.05 * task_value
        train_examples += batch_size
        train_tokens += batch_size * int(data_cfg.get("seq_len_train", 128))

        if step % log_every == 0:
            elapsed = max(time.perf_counter() - start, 1.0e-9)
            append_jsonl(
                metrics_path,
                {
                    "type": "train",
                    "step": step,
                    "train_loss": loss_value,
                    "train_task_loss": task_value,
                    "train_loss_ema": ema_loss,
                    "train_task_loss_ema": ema_task,
                    "avenue_alignment_loss": float(loss_parts["avenue_alignment_loss"].float().cpu()),
                    "avenue_diversity_loss": float(loss_parts["avenue_diversity_loss"].float().cpu()),
                    "crossing_orthogonality_loss": float(loss_parts["crossing_orthogonality_loss"].float().cpu()),
                    "examples_per_sec": train_examples / elapsed,
                    "tokens_per_sec": train_tokens / elapsed,
                },
            )

        if step % eval_every == 0 or step == max_steps:
            elapsed = max(time.perf_counter() - start, 1.0e-9)
            eval_metrics = evaluate_model(model, config, device, seed_offset=100_000 + step * 17)
            last_eval = eval_metrics
            append_jsonl(
                metrics_path,
                {
                    "type": "eval",
                    "step": step,
                    "train_loss_ema": ema_loss,
                    "train_task_loss_ema": ema_task,
                    "train_examples": train_examples,
                    "train_tokens": train_tokens,
                    "train_examples_per_sec": train_examples / elapsed,
                    "train_tokens_per_sec": train_tokens / elapsed,
                    "eval": eval_metrics,
                },
            )
            model.train()
            print(
                f"step {step:05d} train_loss_ema={ema_loss:.4f} "
                f"task_loss_ema={ema_task:.4f} eval_acc={eval_metrics['accuracy']:.3f} "
                f"eval_loss={eval_metrics['loss']:.4f}"
            )

    elapsed = max(time.perf_counter() - start, 1.0e-9)
    if last_eval is None:
        last_eval = evaluate_model(model, config, device, seed_offset=999_000)
    controls = evaluate_controls(model, config, device, seed_offset=700_000)
    peak_memory_mb = None
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        last_eval["peak_gpu_memory_mb"] = max(peak_memory_mb, float(last_eval["peak_gpu_memory_mb"] or 0.0))

    final_metrics = {
        "variant": variant,
        "seed": seed,
        "config_path": str(config_path),
        "parameter_count": total_params,
        "trainable_parameter_count": trainable_params,
        "device": str(device),
        "max_steps": max_steps,
        "train_loss_ema": ema_loss,
        "train_task_loss_ema": ema_task,
        "train_examples": train_examples,
        "train_tokens": train_tokens,
        "train_examples_per_sec": train_examples / elapsed,
        "train_tokens_per_sec": train_tokens / elapsed,
        "wall_clock_train_seconds": elapsed,
        "peak_gpu_memory_mb": peak_memory_mb,
        "eval": last_eval,
        "controls": controls,
        "custom_variant_rationale": config.get("notes", {}).get("custom_rationale"),
        "attention_kernel_summary": model.attention_kernel_summary() if hasattr(model, "attention_kernel_summary") else None,
    }
    save_json(run_dir / "final_metrics.json", final_metrics)
    if output_root == E10_ROOT / "results":
        write_report(E10_ROOT)
    print(f"Wrote {run_dir / 'final_metrics.json'}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 21EYES Experiment 10 scout variant")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-examples", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = E10_ROOT / config_path
    run_training(config_path, args)


if __name__ == "__main__":
    main()
