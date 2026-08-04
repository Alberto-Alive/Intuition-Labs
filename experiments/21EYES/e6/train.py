from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.nn import functional as F

from data import SyntheticBatcher
from eval import append_jsonl, evaluate_model, save_json, write_report
from model import build_model, model_summary, parameter_count


E6_ROOT = Path(__file__).resolve().parent


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
        return
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _autocast(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    return nullcontext()


def _resolve_batch_size(value: Any, *, device: torch.device, train: bool) -> int:
    if value is None or str(value).lower() == "auto":
        if device.type == "cuda":
            return 64 if train else 32
        return 8 if train else 8
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
    variant = str(cfg.get("variant", cfg["model"].get("variant", "baseline")))
    cfg["variant"] = variant
    cfg["model"]["variant"] = variant
    cfg["model"].setdefault("max_seq_len", max(int(v) for v in cfg["data"].get("seq_len_eval", [512])))
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


def _loss_for_batch(model: torch.nn.Module, batch, device: torch.device, amp_enabled: bool):
    with _autocast(device, amp_enabled):
        out = model(batch.input_ids)
        logits = out["logits"]
        loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            batch.labels.view(-1),
            ignore_index=-100,
        )
    return loss, out


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

    batch_size = _resolve_batch_size(data_cfg.get("batch_size", "auto"), device=device, train=True)
    eval_batch_size = _resolve_batch_size(data_cfg.get("eval_batch_size", "auto"), device=device, train=False)
    data_cfg["batch_size"] = batch_size
    data_cfg["eval_batch_size"] = eval_batch_size
    config["data"] = data_cfg

    model = build_model(config["model"]).to(device)
    total_params = parameter_count(model)
    trainable_params = parameter_count(model, trainable_only=True)
    output_root = Path(args.output_root).resolve() if args.output_root is not None else E6_ROOT / "results"
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
    except TypeError:  # pragma: no cover - older PyTorch fallback.
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    max_steps = int(train_cfg.get("max_steps", 5000))
    eval_every = int(train_cfg.get("eval_every", 250))
    log_every = int(train_cfg.get("log_every", 50))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    print(
        f"Training {variant} seed={seed} on {device} for {max_steps} steps "
        f"(batch={batch_size}, eval_batch={eval_batch_size}, params={total_params})."
    )

    model.train()
    ema_loss: Optional[float] = None
    train_examples = 0
    train_tokens = 0
    start = time.perf_counter()
    last_eval: Optional[Dict[str, Any]] = None

    for step in range(1, max_steps + 1):
        batch = train_batcher.sample(batch_size).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, _out = _loss_for_batch(model, batch, device, amp_enabled)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        loss_value = float(loss.detach().float().cpu())
        ema_loss = loss_value if ema_loss is None else 0.95 * ema_loss + 0.05 * loss_value
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
                    "train_loss_ema": ema_loss,
                    "examples_per_sec": train_examples / elapsed,
                    "tokens_per_sec": train_tokens / elapsed,
                    "alpha": model.alpha_values() if hasattr(model, "alpha_values") else None,
                    "beta": model.beta_values() if hasattr(model, "beta_values") else None,
                },
            )

        if step % eval_every == 0 or step == max_steps:
            elapsed = max(time.perf_counter() - start, 1.0e-9)
            eval_metrics = evaluate_model(model, config, device, seed_offset=100_000 + step * 17)
            last_eval = eval_metrics
            record = {
                "type": "eval",
                "step": step,
                "train_loss_ema": ema_loss,
                "train_examples": train_examples,
                "train_tokens": train_tokens,
                "train_examples_per_sec": train_examples / elapsed,
                "train_tokens_per_sec": train_tokens / elapsed,
                "eval": eval_metrics,
            }
            append_jsonl(metrics_path, record)
            model.train()
            print(
                f"step {step:05d} train_loss_ema={ema_loss:.4f} "
                f"eval_acc={eval_metrics['accuracy']:.3f} eval_loss={eval_metrics['loss']:.4f}"
            )

    elapsed = max(time.perf_counter() - start, 1.0e-9)
    if last_eval is None:
        last_eval = evaluate_model(model, config, device, seed_offset=999_000)
    peak_memory_mb = None
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        last_eval["peak_gpu_memory_mb"] = max(
            peak_memory_mb,
            float(last_eval["peak_gpu_memory_mb"] or 0.0),
        )

    final_metrics = {
        "variant": variant,
        "seed": seed,
        "config_path": str(config_path),
        "parameter_count": total_params,
        "trainable_parameter_count": trainable_params,
        "device": str(device),
        "max_steps": max_steps,
        "train_loss_ema": ema_loss,
        "train_examples": train_examples,
        "train_tokens": train_tokens,
        "train_examples_per_sec": train_examples / elapsed,
        "train_tokens_per_sec": train_tokens / elapsed,
        "wall_clock_train_seconds": elapsed,
        "peak_gpu_memory_mb": peak_memory_mb,
        "eval": last_eval,
        "alpha": model.alpha_values() if hasattr(model, "alpha_values") else None,
        "beta": model.beta_values() if hasattr(model, "beta_values") else None,
        "attention_kernel_summary": model.attention_kernel_summary() if hasattr(model, "attention_kernel_summary") else None,
    }
    save_json(run_dir / "final_metrics.json", final_metrics)
    if output_root == E6_ROOT / "results":
        write_report(E6_ROOT)
    print(f"Wrote {run_dir / 'final_metrics.json'}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 21EYES Experiment 6 variant")
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
        config_path = E6_ROOT / config_path
    run_training(config_path, args)


if __name__ == "__main__":
    main()
