from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.nn import functional as F

from data import TASKS, SyntheticBatch, SyntheticBatcher


E7_ROOT = Path(__file__).resolve().parent
SLOT_CONTROLS = ("slot_shuffle", "slot_zero", "slot_random", "hidden_state_shuffle")


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _jsonable(value.detach().cpu().item())
        return [_jsonable(v) for v in value.detach().cpu().tolist()]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")


def _autocast(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    from contextlib import nullcontext

    return nullcontext()


def _bucket_distractors(value: int) -> str:
    if value <= 4:
        return "0-4"
    if value <= 8:
        return "5-8"
    if value <= 16:
        return "9-16"
    return "17+"


def _bucket_overwrites(value: int) -> str:
    if value <= 1:
        return "1"
    if value == 2:
        return "2"
    if value == 3:
        return "3"
    return "4+"


def _add_group(groups: DefaultDict[str, List[int]], key: str, correct: int, count: int = 1) -> None:
    groups[str(key)][0] += int(correct)
    groups[str(key)][1] += int(count)


def _finish_groups(groups: DefaultDict[str, List[int]]) -> Dict[str, float]:
    return {
        key: float(correct) / float(count) if count else 0.0
        for key, (correct, count) in sorted(groups.items(), key=lambda item: item[0])
    }


def _default_eval_profiles(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    data_cfg = config.get("data", {})
    profiles_cfg = data_cfg.get("eval_profiles", {})
    seq_len_train = int(data_cfg.get("seq_len_train", 128))
    seq_len_eval = [int(v) for v in data_cfg.get("seq_len_eval", [seq_len_train, 256, 512])]
    base_distractors = tuple(data_cfg.get("eval_distractor_range", data_cfg.get("train_distractor_range", [4, 14])))
    base_overwrites = tuple(data_cfg.get("eval_overwrite_range", data_cfg.get("train_overwrite_range", [1, 4])))
    eval_examples = int(data_cfg.get("eval_examples", 256))
    profiles: List[Dict[str, Any]] = []

    if "in_distribution" in profiles_cfg:
        profile = dict(profiles_cfg["in_distribution"])
        profile.setdefault("name", "in_distribution")
        profiles.append(profile)
    else:
        profiles.append(
            {
                "name": "in_distribution",
                "seq_len": seq_len_train,
                "distractor_range": list(base_distractors),
                "overwrite_range": list(base_overwrites),
                "examples": eval_examples,
            }
        )

    for seq_len in seq_len_eval:
        if seq_len == seq_len_train:
            continue
        profiles.append(
            {
                "name": f"seq_len_{seq_len}",
                "seq_len": seq_len,
                "distractor_range": list(base_distractors),
                "overwrite_range": list(base_overwrites),
                "examples": eval_examples,
            }
        )

    for name, default in (
        (
            "high_distractor",
            {
                "name": "high_distractor",
                "seq_len": seq_len_train,
                "distractor_range": [18, 30],
                "overwrite_range": list(base_overwrites),
                "examples": eval_examples,
            },
        ),
        (
            "many_overwrite",
            {
                "name": "many_overwrite",
                "seq_len": seq_len_train,
                "distractor_range": list(base_distractors),
                "overwrite_range": [5, 8],
                "examples": eval_examples,
            },
        ),
    ):
        if name in profiles_cfg:
            profile = dict(profiles_cfg[name])
            profile.setdefault("name", name)
            profiles.append(profile)
        else:
            profiles.append(default)
    return profiles


def _extract_layer_metrics(layer_accum: List[Dict[str, float]], denom: int) -> Dict[str, List[Optional[float]]]:
    names = [
        "attention_entropy",
        "effective_attended_tokens",
        "sdpa_used",
        "slot_norm",
        "slot_diversity",
        "token_to_slot_entropy",
        "slot_read_entropy",
        "token_to_slot_max",
        "slot_read_max",
        "slot_utilization",
        "slot_to_token_entropy",
        "stable_norm",
        "innovation_norm",
        "innovation_stable_ratio",
    ]
    out: Dict[str, List[Optional[float]]] = {name: [] for name in names}
    for accum in layer_accum:
        for name in names:
            out[name].append(accum.get(name, None) / max(1, denom) if name in accum else None)
    return out


def _update_layer_accum(layer_accum: List[Dict[str, float]], layer_stats: Sequence[Dict[str, torch.Tensor]], weight: int) -> None:
    for idx, stats in enumerate(layer_stats):
        accum = layer_accum[idx]
        for name, value in stats.items():
            accum[name] = accum.get(name, 0.0) + float(value.detach().float().cpu()) * int(weight)


def _auc_for_positions(scores: torch.Tensor, useful_positions: torch.Tensor, distractor_positions: torch.Tensor) -> Optional[float]:
    useful = useful_positions[useful_positions >= 0].clamp(max=scores.numel() - 1)
    distractors = distractor_positions[distractor_positions >= 0].clamp(max=scores.numel() - 1)
    if useful.numel() == 0 or distractors.numel() == 0:
        return None
    useful_scores = scores[useful].float().view(-1, 1)
    distractor_scores = scores[distractors].float().view(1, -1)
    wins = (useful_scores > distractor_scores).float()
    ties = (useful_scores == distractor_scores).float() * 0.5
    return float((wins + ties).mean().detach().cpu())


def _topk_position_recall(scores: torch.Tensor, positions: torch.Tensor, k: int) -> Optional[float]:
    valid = positions[positions >= 0].clamp(max=scores.numel() - 1)
    if valid.numel() == 0:
        return None
    top = torch.topk(scores, k=min(int(k), scores.numel()), dim=-1).indices
    hits = (valid.view(-1, 1) == top.view(1, -1)).any(dim=-1).float()
    return float(hits.mean().detach().cpu())


def _slot_topk_recall(update_weights: torch.Tensor, query_weights: torch.Tensor, positions: torch.Tensor, k: int) -> Optional[float]:
    valid = positions[positions >= 0].clamp(max=update_weights.shape[0] - 1)
    if valid.numel() == 0:
        return None
    useful_slots = update_weights[valid].argmax(dim=-1)
    top_slots = torch.topk(query_weights, k=min(int(k), query_weights.numel()), dim=-1).indices
    hits = (useful_slots.view(-1, 1) == top_slots.view(1, -1)).any(dim=-1).float()
    return float(hits.mean().detach().cpu())


def _update_slot_evidence_diagnostics(
    rank_accum: List[Dict[str, float]],
    slot_attn: Optional[Sequence[Optional[Dict[str, torch.Tensor]]]],
    batch: SyntheticBatch,
) -> None:
    if not slot_attn:
        return
    rel = batch.relevant_positions
    dist = batch.distractor_positions
    queries = batch.query_positions
    for layer_idx, diag in enumerate(slot_attn):
        if diag is None:
            continue
        accum = rank_accum[layer_idx]
        update = diag["token_to_slot"].detach().float()
        read = diag["slot_read"].detach().float()
        for row in range(update.shape[0]):
            q = int(queries[row].item())
            if q < 0:
                continue
            q = min(q, update.shape[1] - 1)
            query_slots = read[row, q]
            token_scores = (update[row, : q + 1] * query_slots.view(1, -1)).sum(dim=-1)
            useful = rel[row][rel[row] >= 0].clamp(max=q)
            distractors = dist[row][dist[row] >= 0].clamp(max=q)
            auc = _auc_for_positions(token_scores, useful, distractors)
            if auc is not None:
                accum["slot_auc_sum"] += auc
                accum["slot_auc_count"] += 1.0
            for k in (1, 3, 5, 10):
                pos_recall = _topk_position_recall(token_scores, useful, k)
                if pos_recall is not None:
                    accum[f"top{k}_position_recall_sum"] += pos_recall
                    accum[f"top{k}_position_recall_count"] += 1.0
                slot_recall = _slot_topk_recall(update[row, : q + 1], query_slots, useful, k)
                if slot_recall is not None:
                    accum[f"top{k}_slot_recall_sum"] += slot_recall
                    accum[f"top{k}_slot_recall_count"] += 1.0
            if useful.numel() > 0:
                accum["useful_concentration_sum"] += float(update[row, useful].amax(dim=-1).mean().cpu())
                accum["useful_concentration_count"] += 1.0
            if distractors.numel() > 0:
                accum["distractor_concentration_sum"] += float(update[row, distractors].amax(dim=-1).mean().cpu())
                accum["distractor_concentration_count"] += 1.0


def _finish_slot_evidence(rank_accum: List[Dict[str, float]]) -> Dict[str, List[Optional[float]]]:
    out: Dict[str, List[Optional[float]]] = {
        "useful_vs_distractor_slot_auc": [],
        "useful_vs_distractor_auc": [],
        "useful_evidence_slot_concentration": [],
        "distractor_slot_concentration": [],
    }
    for k in (1, 3, 5, 10):
        out[f"useful_top{k}_position_recall"] = []
        out[f"useful_top{k}_slot_recall"] = []
        out[f"useful_top{k}_recall"] = []
    for accum in rank_accum:
        auc_count = accum.get("slot_auc_count", 0.0)
        auc = accum["slot_auc_sum"] / auc_count if auc_count else None
        out["useful_vs_distractor_slot_auc"].append(auc)
        out["useful_vs_distractor_auc"].append(auc)
        uc = accum.get("useful_concentration_count", 0.0)
        dc = accum.get("distractor_concentration_count", 0.0)
        out["useful_evidence_slot_concentration"].append(accum["useful_concentration_sum"] / uc if uc else None)
        out["distractor_slot_concentration"].append(accum["distractor_concentration_sum"] / dc if dc else None)
        for k in (1, 3, 5, 10):
            pc = accum.get(f"top{k}_position_recall_count", 0.0)
            sc = accum.get(f"top{k}_slot_recall_count", 0.0)
            pos_val = accum[f"top{k}_position_recall_sum"] / pc if pc else None
            slot_val = accum[f"top{k}_slot_recall_sum"] / sc if sc else None
            out[f"useful_top{k}_position_recall"].append(pos_val)
            out[f"useful_top{k}_slot_recall"].append(slot_val)
            out[f"useful_top{k}_recall"].append(slot_val)
    return out


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    config: Dict[str, Any],
    device: torch.device,
    *,
    seed_offset: int = 100_000,
    profiles: Optional[Sequence[Dict[str, Any]]] = None,
    control: Optional[str] = None,
) -> Dict[str, Any]:
    model.eval()
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    eval_batch_size = int(data_cfg.get("eval_batch_size", data_cfg.get("batch_size", 32)))
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    profiles = list(profiles or _default_eval_profiles(config))
    n_layers = int(config.get("model", {}).get("n_layers", 4))

    total_loss = 0.0
    total_examples = 0
    total_correct = 0
    total_tokens = 0
    groups_task: DefaultDict[str, List[int]] = defaultdict(lambda: [0, 0])
    groups_len: DefaultDict[str, List[int]] = defaultdict(lambda: [0, 0])
    groups_dist: DefaultDict[str, List[int]] = defaultdict(lambda: [0, 0])
    groups_ow: DefaultDict[str, List[int]] = defaultdict(lambda: [0, 0])
    scenarios: Dict[str, Dict[str, Any]] = {}
    layer_accum = [defaultdict(float) for _ in range(n_layers)]
    rank_accum = [defaultdict(float) for _ in range(n_layers)]
    global_diag_accum: DefaultDict[str, float] = defaultdict(float)
    global_diag_count = 0
    align_pair_accum: List[float] = []
    align_pair_count: List[int] = []
    cross_pair_accum: List[float] = []
    cross_pair_count: List[int] = []

    start = time.perf_counter()
    for profile_idx, profile in enumerate(profiles):
        name = str(profile["name"])
        seq_len = int(profile.get("seq_len", data_cfg.get("seq_len_train", 128)))
        examples = int(profile.get("examples", data_cfg.get("eval_examples", 256)))
        distractor_range = tuple(int(v) for v in profile.get("distractor_range", data_cfg.get("eval_distractor_range", [4, 14])))
        overwrite_range = tuple(int(v) for v in profile.get("overwrite_range", data_cfg.get("eval_overwrite_range", [1, 4])))
        batcher = SyntheticBatcher(
            seed=int(config.get("seed", 0)) + seed_offset + profile_idx * 10_000,
            split=str(profile.get("split", "val")),
            seq_len=seq_len,
            distractor_range=distractor_range,
            overwrite_range=overwrite_range,
            tasks=profile.get("tasks", TASKS),
        )

        scenario_loss = 0.0
        scenario_count = 0
        scenario_correct = 0
        remaining = examples
        while remaining > 0:
            bsz = min(eval_batch_size, remaining)
            batch = batcher.sample(bsz).to(device)
            with _autocast(device, amp_enabled):
                out = model(batch.input_ids, return_diagnostics=True, control=control)
                logits = out["logits"]
                loss = F.cross_entropy(
                    logits.view(-1, logits.shape[-1]),
                    batch.labels.view(-1),
                    ignore_index=-100,
                )
            query_logits = logits[torch.arange(bsz, device=device), batch.query_positions]
            pred = query_logits.argmax(dim=-1)
            correct_mask = pred.eq(batch.target_ids)
            correct = int(correct_mask.sum().item())
            loss_value = float(loss.detach().float().cpu())

            scenario_loss += loss_value * bsz
            scenario_correct += correct
            scenario_count += bsz
            total_loss += loss_value * bsz
            total_examples += bsz
            total_correct += correct
            total_tokens += bsz * seq_len

            _update_layer_accum(layer_accum, out["layer_stats"], bsz)
            _update_slot_evidence_diagnostics(rank_accum, out.get("slot_attn"), batch)
            diag = out.get("diagnostics", {})
            for key in ("cross_layer_slot_cosine", "predicted_actual_slot_cosine"):
                if key in diag:
                    global_diag_accum[key] += float(diag[key].detach().float().cpu()) * bsz
            for key, accum, counts in (
                ("predicted_actual_slot_cosine_by_pair", align_pair_accum, align_pair_count),
                ("cross_layer_slot_cosine_by_pair", cross_pair_accum, cross_pair_count),
            ):
                values = diag.get(key, [])
                while len(accum) < len(values):
                    accum.append(0.0)
                    counts.append(0)
                for idx, value in enumerate(values):
                    accum[idx] += float(value.detach().float().cpu()) * bsz
                    counts[idx] += bsz
            global_diag_count += bsz

            for i in range(bsz):
                c = int(correct_mask[i].item())
                _add_group(groups_task, batch.tasks[i], c)
                _add_group(groups_len, str(seq_len), c)
                _add_group(groups_dist, _bucket_distractors(int(batch.distractor_counts[i].item())), c)
                _add_group(groups_ow, _bucket_overwrites(int(batch.overwrite_depths[i].item())), c)
            remaining -= bsz

        scenarios[name] = {
            "loss": scenario_loss / max(1, scenario_count),
            "accuracy": scenario_correct / max(1, scenario_count),
            "examples": scenario_count,
            "seq_len": seq_len,
            "distractor_range": list(distractor_range),
            "overwrite_range": list(overwrite_range),
        }

    elapsed = max(time.perf_counter() - start, 1.0e-9)
    layers = _extract_layer_metrics(layer_accum, total_examples)
    layers.update(_finish_slot_evidence(rank_accum))
    peak_memory_mb = None
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)

    slot_diag = {
        "cross_layer_slot_cosine": global_diag_accum["cross_layer_slot_cosine"] / max(1, global_diag_count),
        "predicted_actual_slot_cosine": global_diag_accum["predicted_actual_slot_cosine"] / max(1, global_diag_count),
        "predicted_actual_slot_cosine_by_pair": [
            accum / max(1, count) for accum, count in zip(align_pair_accum, align_pair_count)
        ],
        "cross_layer_slot_cosine_by_pair": [
            accum / max(1, count) for accum, count in zip(cross_pair_accum, cross_pair_count)
        ],
    }

    return {
        "loss": total_loss / max(1, total_examples),
        "accuracy": total_correct / max(1, total_examples),
        "held_out_template_accuracy": total_correct / max(1, total_examples),
        "examples": total_examples,
        "tokens": total_tokens,
        "examples_per_sec": total_examples / elapsed,
        "tokens_per_sec": total_tokens / elapsed,
        "wall_clock_eval_seconds": elapsed,
        "accuracy_by_task": _finish_groups(groups_task),
        "accuracy_by_seq_len": _finish_groups(groups_len),
        "accuracy_by_distractor_count": _finish_groups(groups_dist),
        "accuracy_by_overwrite_depth": _finish_groups(groups_ow),
        "scenarios": scenarios,
        "layers": layers,
        "slot_diagnostics": slot_diag,
        "peak_gpu_memory_mb": peak_memory_mb,
        "control": control,
    }


@torch.no_grad()
def evaluate_slot_controls(
    model: torch.nn.Module,
    config: Dict[str, Any],
    device: torch.device,
    *,
    seed_offset: int = 900_000,
) -> Dict[str, Any]:
    if not bool(getattr(model, "has_slots", False)):
        return {}
    data_cfg = config.get("data", {})
    examples = int(data_cfg.get("control_examples", min(128, int(data_cfg.get("eval_examples", 256)))))
    profile = {
        "name": "slot_control_in_distribution",
        "seq_len": int(data_cfg.get("seq_len_train", 128)),
        "distractor_range": list(data_cfg.get("eval_distractor_range", data_cfg.get("train_distractor_range", [4, 14]))),
        "overwrite_range": list(data_cfg.get("eval_overwrite_range", data_cfg.get("train_overwrite_range", [1, 4]))),
        "examples": examples,
    }
    base = evaluate_model(model, config, device, seed_offset=seed_offset, profiles=[profile], control=None)
    controls: Dict[str, Dict[str, Any]] = {}
    for idx, control in enumerate(SLOT_CONTROLS):
        metrics = evaluate_model(
            model,
            config,
            device,
            seed_offset=seed_offset + idx * 777,
            profiles=[profile],
            control=control,
        )
        controls[control] = {
            "accuracy": metrics["accuracy"],
            "loss": metrics["loss"],
            "drop": base["accuracy"] - metrics["accuracy"],
            "examples": metrics["examples"],
            "tokens_per_sec": metrics["tokens_per_sec"],
        }
    return {
        "base_accuracy": base["accuracy"],
        "base_loss": base["loss"],
        "examples": examples,
        "controls": controls,
    }


def _load_runs(results_dir: Path) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for path in sorted(results_dir.glob("*/*/final_metrics.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        payload["_path"] = str(path)
        runs.append(payload)
    return runs


def _scenario_acc(run: Dict[str, Any], name: str) -> Optional[float]:
    try:
        return float(run["eval"]["scenarios"][name]["accuracy"])
    except KeyError:
        return None


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _plot_bar(path: Path, labels: Sequence[str], values: Sequence[float], title: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(labels, values, color="#3f7f93")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_lines(path: Path, series: Dict[str, Dict[str, float]], title: str, xlabel: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for label, values in series.items():
        keys = list(values.keys())
        vals = [values[k] for k in keys]
        ax.plot(keys, vals, marker="o", label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=20)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_plots(runs: Sequence[Dict[str, Any]], plots_dir: Path) -> List[str]:
    if not runs:
        return []
    written: List[str] = []
    labels = [f"{run.get('variant')}/s{run.get('seed')}" for run in runs]
    try:
        _plot_bar(
            plots_dir / "accuracy_by_variant.png",
            labels,
            [float(run.get("eval", {}).get("accuracy", 0.0)) for run in runs],
            "Accuracy by Variant",
            "accuracy",
        )
        written.append(str(plots_dir / "accuracy_by_variant.png"))

        _plot_lines(
            plots_dir / "accuracy_by_task.png",
            {labels[i]: runs[i].get("eval", {}).get("accuracy_by_task", {}) for i in range(len(runs))},
            "Accuracy by Task",
            "task",
            "accuracy",
        )
        written.append(str(plots_dir / "accuracy_by_task.png"))

        _plot_lines(
            plots_dir / "accuracy_vs_seq_len.png",
            {labels[i]: runs[i].get("eval", {}).get("accuracy_by_seq_len", {}) for i in range(len(runs))},
            "Accuracy vs Sequence Length",
            "seq_len",
            "accuracy",
        )
        written.append(str(plots_dir / "accuracy_vs_seq_len.png"))

        _plot_lines(
            plots_dir / "accuracy_vs_distractors.png",
            {labels[i]: runs[i].get("eval", {}).get("accuracy_by_distractor_count", {}) for i in range(len(runs))},
            "Accuracy vs Distractor Count",
            "distractor bucket",
            "accuracy",
        )
        written.append(str(plots_dir / "accuracy_vs_distractors.png"))

        _plot_lines(
            plots_dir / "accuracy_vs_overwrite_depth.png",
            {labels[i]: runs[i].get("eval", {}).get("accuracy_by_overwrite_depth", {}) for i in range(len(runs))},
            "Accuracy vs Overwrite Depth",
            "overwrite depth bucket",
            "accuracy",
        )
        written.append(str(plots_dir / "accuracy_vs_overwrite_depth.png"))

        for metric, filename, title in (
            ("slot_diversity", "slot_diversity_by_layer.png", "Slot Diversity by Layer"),
            ("slot_read_entropy", "slot_entropy_by_layer.png", "Slot Read Entropy by Layer"),
            ("attention_entropy", "attention_entropy_by_layer.png", "Attention Entropy by Layer"),
            ("innovation_stable_ratio", "innovation_stable_ratio.png", "Innovation / Stable Ratio"),
        ):
            series: Dict[str, Dict[str, float]] = {}
            for label, run in zip(labels, runs):
                values = run.get("eval", {}).get("layers", {}).get(metric, [])
                series[label] = {str(i): float(v) for i, v in enumerate(values) if v is not None}
            _plot_lines(plots_dir / filename, series, title, "layer", metric)
            written.append(str(plots_dir / filename))

        align_series: Dict[str, Dict[str, float]] = {}
        for label, run in zip(labels, runs):
            values = run.get("eval", {}).get("slot_diagnostics", {}).get("predicted_actual_slot_cosine_by_pair", [])
            align_series[label] = {f"{i}->{i+1}": float(v) for i, v in enumerate(values)}
        _plot_lines(plots_dir / "slot_alignment_by_layer.png", align_series, "Predicted vs Actual Slot Alignment", "layer pair", "cosine")
        written.append(str(plots_dir / "slot_alignment_by_layer.png"))

        drops: Dict[str, float] = {}
        for run in runs:
            variant = f"{run.get('variant')}/s{run.get('seed')}"
            controls = run.get("slot_controls", {}).get("controls", {})
            if controls:
                drops[variant] = sum(float(item.get("drop", 0.0)) for item in controls.values()) / max(1, len(controls))
        _plot_bar(plots_dir / "slot_control_drops.png", list(drops.keys()), list(drops.values()), "Mean Slot Control Drop", "accuracy drop")
        written.append(str(plots_dir / "slot_control_drops.png"))

        _plot_bar(
            plots_dir / "throughput_by_variant.png",
            labels,
            [float(run.get("train_tokens_per_sec") or 0.0) for run in runs],
            "Training Throughput by Variant",
            "tokens/sec",
        )
        written.append(str(plots_dir / "throughput_by_variant.png"))

        _plot_bar(
            plots_dir / "gpu_memory_by_variant.png",
            labels,
            [float(run.get("peak_gpu_memory_mb") or run.get("eval", {}).get("peak_gpu_memory_mb") or 0.0) for run in runs],
            "Peak GPU Memory by Variant",
            "MB",
        )
        written.append(str(plots_dir / "gpu_memory_by_variant.png"))
    except Exception as exc:  # pragma: no cover
        written.append(f"plotting_failed: {exc}")
    return written


def _conclusion(runs: Sequence[Dict[str, Any]]) -> str:
    by_variant = {str(run.get("variant")): run for run in runs}
    required = [
        "baseline_transformer",
        "param_matched_baseline",
        "memory_slot_attention",
        "slot_alignment_loss",
        "predictive_slot_residual",
        "mirrored_slot_attention",
    ]
    if any(name not in by_variant for name in required):
        return "inconclusive: not all required E7 variants have completed runs"
    baseline = by_variant["baseline_transformer"]
    param = by_variant["param_matched_baseline"]
    memory = by_variant["memory_slot_attention"]
    align = by_variant["slot_alignment_loss"]
    pred = by_variant["predictive_slot_residual"]

    base_acc = float(baseline.get("eval", {}).get("accuracy", 0.0))
    param_acc = float(param.get("eval", {}).get("accuracy", 0.0))
    memory_acc = float(memory.get("eval", {}).get("accuracy", 0.0))
    align_acc = float(align.get("eval", {}).get("accuracy", 0.0))
    pred_acc = float(pred.get("eval", {}).get("accuracy", 0.0))
    pred_long = _mean(
        value.get("accuracy")
        for key, value in pred.get("eval", {}).get("scenarios", {}).items()
        if key.startswith("seq_len_") or key in {"high_distractor", "many_overwrite"}
    )
    memory_long = _mean(
        value.get("accuracy")
        for key, value in memory.get("eval", {}).get("scenarios", {}).items()
        if key.startswith("seq_len_") or key in {"high_distractor", "many_overwrite"}
    )
    pred_ratio = _mean(pred.get("eval", {}).get("layers", {}).get("innovation_stable_ratio", []))
    pred_align = pred.get("eval", {}).get("slot_diagnostics", {}).get("predicted_actual_slot_cosine")
    memory_div = _mean(memory.get("eval", {}).get("layers", {}).get("slot_diversity", []))
    control_drops = [
        float(item.get("drop", 0.0))
        for item in pred.get("slot_controls", {}).get("controls", {}).values()
    ]
    controls_hurt = bool(control_drops) and sum(control_drops) / len(control_drops) > 0.02

    if (
        memory_acc > base_acc
        and memory_acc > param_acc
        and align_acc >= memory_acc
        and (pred_acc >= memory_acc or (pred_long is not None and memory_long is not None and pred_long > memory_long))
        and controls_hurt
        and memory_div is not None
        and memory_div > 0.01
        and pred_align is not None
        and float(pred_align) > 0.05
        and pred_ratio is not None
        and pred_ratio > 0.05
    ):
        return "supported: slots beat both controls, alignment/predictive variants improve or preserve quality, controls degrade accuracy, slots remain non-collapsed, and innovation stays nonzero"
    if memory_acc > max(base_acc, param_acc) and controls_hurt:
        return "weakly supported: memory slots beat the controls and slot ablations hurt, but alignment or predictive residual evidence is incomplete"
    if param_acc >= memory_acc and param_acc >= align_acc and param_acc >= pred_acc:
        return "not supported: the parameter-matched baseline equals or beats the slot variants"
    return "inconclusive: slot quality or diagnostics do not cleanly satisfy the support criteria"


def write_report(
    root: Path = E7_ROOT,
    *,
    results_dir: Optional[Path] = None,
    plots_dir: Optional[Path] = None,
    report_path: Optional[Path] = None,
    title: str = "Experiment 7 Report",
) -> Path:
    results_dir = results_dir or (root / "results")
    plots_dir = plots_dir or (root / "plots")
    report_path = report_path or (results_dir / "report.md")
    runs = _load_runs(results_dir)
    written_plots = write_plots(runs, plots_dir)

    lines: List[str] = [f"# {title}", ""]
    if not runs:
        lines.extend(
            [
                "No completed runs were found under `results/<variant>/<seed>/final_metrics.json`.",
                "",
                "## Conclusion",
                "",
                "inconclusive: no completed runs yet",
                "",
            ]
        )
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path

    seeds = sorted({str(run.get("seed")) for run in runs})
    if len(seeds) == 1:
        lines.extend(
            [
                "## Run Scope",
                "",
                f"Single-seed run only (`seed={seeds[0]}`). Treat results as preliminary unless effect sizes are large and controls agree.",
                "",
            ]
        )

    lines.extend(["## Summary Table", ""])
    lines.append(
        "| Variant | Seed | Params | Acc | In-dist | Long mean | High distractor | Many overwrite | Slot div | Slot entropy | Align cos | Innov/stable | Control drop | GPU MB | Train tok/s | Eval tok/s |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for run in runs:
        eval_metrics = run.get("eval", {})
        layers = eval_metrics.get("layers", {})
        long_accs = [
            value["accuracy"]
            for key, value in eval_metrics.get("scenarios", {}).items()
            if key.startswith("seq_len_")
        ]
        controls = run.get("slot_controls", {}).get("controls", {})
        control_drop = _mean(item.get("drop") for item in controls.values()) if controls else None
        lines.append(
            "| {variant} | {seed} | {params} | {acc} | {ind} | {long} | {high} | {ow} | {div} | {ent} | {align} | {ratio} | {drop} | {gpu} | {traintps} | {evaltps} |".format(
                variant=run.get("variant"),
                seed=run.get("seed"),
                params=run.get("parameter_count", "NA"),
                acc=_fmt(eval_metrics.get("accuracy")),
                ind=_fmt(_scenario_acc(run, "in_distribution")),
                long=_fmt(_mean(long_accs)),
                high=_fmt(_scenario_acc(run, "high_distractor")),
                ow=_fmt(_scenario_acc(run, "many_overwrite")),
                div=_fmt(_mean(layers.get("slot_diversity", []))),
                ent=_fmt(_mean(layers.get("slot_read_entropy", []))),
                align=_fmt(eval_metrics.get("slot_diagnostics", {}).get("predicted_actual_slot_cosine")),
                ratio=_fmt(_mean(layers.get("innovation_stable_ratio", []))),
                drop=_fmt(control_drop),
                gpu=_fmt(run.get("peak_gpu_memory_mb") or eval_metrics.get("peak_gpu_memory_mb"), 1),
                traintps=_fmt(run.get("train_tokens_per_sec"), 1),
                evaltps=_fmt(eval_metrics.get("tokens_per_sec"), 1),
            )
        )

    for heading, key in (
        ("Accuracy by Task", "accuracy_by_task"),
        ("Accuracy by Sequence Length", "accuracy_by_seq_len"),
        ("Accuracy by Distractor Count", "accuracy_by_distractor_count"),
        ("Accuracy by Overwrite Depth", "accuracy_by_overwrite_depth"),
    ):
        lines.extend(["", f"## {heading}", ""])
        for run in runs:
            lines.append(f"- `{run.get('variant')}` seed `{run.get('seed')}`: {run.get('eval', {}).get(key, {})}")

    lines.extend(["", "## Slot Diagnostics", ""])
    for run in runs:
        layers = run.get("eval", {}).get("layers", {})
        slot_diag = run.get("eval", {}).get("slot_diagnostics", {})
        lines.append(
            f"- `{run.get('variant')}` seed `{run.get('seed')}` "
            f"diversity={layers.get('slot_diversity')} "
            f"read_entropy={layers.get('slot_read_entropy')} "
            f"utilization={layers.get('slot_utilization')} "
            f"slot_norm={layers.get('slot_norm')} "
            f"cross_layer_cosine={slot_diag.get('cross_layer_slot_cosine')} "
            f"predicted_actual={slot_diag.get('predicted_actual_slot_cosine')} "
            f"predicted_actual_by_pair={slot_diag.get('predicted_actual_slot_cosine_by_pair')} "
            f"innovation_stable_ratio={layers.get('innovation_stable_ratio')}"
        )

    lines.extend(["", "## Slot Control Results", ""])
    for run in runs:
        controls = run.get("slot_controls", {})
        lines.append(f"- `{run.get('variant')}` seed `{run.get('seed')}`: {controls if controls else 'not applicable'}")

    lines.extend(["", "## Evidence Diagnostics", ""])
    for run in runs:
        layers = run.get("eval", {}).get("layers", {})
        lines.append(
            f"- `{run.get('variant')}` seed `{run.get('seed')}` "
            f"useful_vs_distractor_slot_auc={layers.get('useful_vs_distractor_slot_auc')} "
            f"useful_slot_top1={layers.get('useful_top1_slot_recall')} "
            f"useful_slot_top3={layers.get('useful_top3_slot_recall')} "
            f"useful_slot_top5={layers.get('useful_top5_slot_recall')} "
            f"useful_slot_top10={layers.get('useful_top10_slot_recall')} "
            f"useful_concentration={layers.get('useful_evidence_slot_concentration')} "
            f"distractor_concentration={layers.get('distractor_slot_concentration')}"
        )

    lines.extend(["", "## Attention Diagnostics", ""])
    for run in runs:
        layers = run.get("eval", {}).get("layers", {})
        lines.append(
            f"- `{run.get('variant')}` seed `{run.get('seed')}` "
            f"attention_entropy={layers.get('attention_entropy')} "
            f"effective_tokens={layers.get('effective_attended_tokens')} "
            f"token_to_slot_entropy={layers.get('token_to_slot_entropy')} "
            f"slot_to_token_entropy={layers.get('slot_to_token_entropy')}"
        )

    lines.extend(["", "## Performance", ""])
    for run in runs:
        eval_metrics = run.get("eval", {})
        lines.append(
            f"- `{run.get('variant')}` seed `{run.get('seed')}`: "
            f"params={run.get('parameter_count')}, trainable={run.get('trainable_parameter_count')}, "
            f"peak_gpu_memory_mb={_fmt(run.get('peak_gpu_memory_mb') or eval_metrics.get('peak_gpu_memory_mb'), 1)}, "
            f"train_tokens_per_sec={_fmt(run.get('train_tokens_per_sec'), 1)}, "
            f"eval_tokens_per_sec={_fmt(eval_metrics.get('tokens_per_sec'), 1)}, "
            f"train_seconds={_fmt(run.get('wall_clock_train_seconds'), 1)}, "
            f"eval_seconds={_fmt(eval_metrics.get('wall_clock_eval_seconds'), 1)}, "
            f"attention_kernel={run.get('attention_kernel_summary')}"
        )

    conclusion = _conclusion(runs)
    lines.extend(["", "## Conclusion", "", conclusion, ""])
    lines.extend(
        [
            "## Failure Analysis",
            "",
            "Interpret failures against the critical controls: if parameter matching removes gains, if slot controls do not reduce accuracy, if slot diversity is near zero, or if predictive innovation collapses toward zero, the slot-state mediation lemma is not supported by this run.",
            "",
            "## Next Recommended Experiment",
            "",
            "If E7 is positive, rerun the six-variant matrix with seeds 1 and 2 and sweep `lambda_align` over 0.01, 0.05, and 0.1. If E7 is negative, isolate whether causal per-token slots are too weak by testing chunk-causal prefix slots with the same anti-cheat constraints.",
            "",
            "## Plots",
            "",
        ]
    )
    for item in written_plots:
        lines.append(f"- `{item}`")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation/report utilities for 21EYES E7")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--plots-dir", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--title", type=str, default="Experiment 7 Report")
    args = parser.parse_args()
    if not args.report:
        parser.error("No action requested. Use --report.")
    results_dir = args.results_dir
    plots_dir = args.plots_dir
    report_path = args.report_path
    if results_dir is not None and not results_dir.is_absolute():
        results_dir = E7_ROOT / results_dir
    if plots_dir is not None and not plots_dir.is_absolute():
        plots_dir = E7_ROOT / plots_dir
    if report_path is not None and not report_path.is_absolute():
        report_path = E7_ROOT / report_path
    path = write_report(E7_ROOT, results_dir=results_dir, plots_dir=plots_dir, report_path=report_path, title=args.title)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
