from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence

import torch
from torch.nn import functional as F

from data import TASKS, SyntheticBatch, SyntheticBatcher


E8_ROOT = Path(__file__).resolve().parent
AVENUE_CONTROLS = (
    "avenue_zero",
    "avenue_shuffle",
    "avenue_random",
    "avenue_order_shuffle",
    "hidden_state_shuffle",
)


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
    seq_len_eval = [int(v) for v in data_cfg.get("seq_len_eval", [seq_len_train, 256])]
    base_distractors = tuple(data_cfg.get("eval_distractor_range", data_cfg.get("train_distractor_range", [4, 14])))
    base_overwrites = tuple(data_cfg.get("eval_overwrite_range", data_cfg.get("train_overwrite_range", [1, 4])))
    eval_examples = int(data_cfg.get("eval_examples", 128))
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
        "avenue_gate_entropy",
        "avenue_output_norm",
        "pairwise_avenue_cosine",
        "avenue_diversity",
        "avenue_scale",
        "avenue_usage_0",
        "avenue_usage_1",
        "avenue_usage_2",
        "avenue_usage_3",
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


def _topk_avenue_recall(gates: torch.Tensor, query_gates: torch.Tensor, positions: torch.Tensor, k: int) -> Optional[float]:
    valid = positions[positions >= 0].clamp(max=gates.shape[0] - 1)
    if valid.numel() == 0:
        return None
    useful_avenues = gates[valid].argmax(dim=-1)
    top_avenues = torch.topk(query_gates, k=min(int(k), query_gates.numel()), dim=-1).indices
    hits = (useful_avenues.view(-1, 1) == top_avenues.view(1, -1)).any(dim=-1).float()
    return float(hits.mean().detach().cpu())


def _update_avenue_evidence_diagnostics(
    rank_accum: List[Dict[str, float]],
    avenue_attn: Optional[Sequence[Optional[Dict[str, torch.Tensor]]]],
    batch: SyntheticBatch,
) -> None:
    if not avenue_attn:
        return
    rel = batch.relevant_positions
    dist = batch.distractor_positions
    queries = batch.query_positions
    for layer_idx, diag in enumerate(avenue_attn):
        if diag is None:
            continue
        accum = rank_accum[layer_idx]
        gates = diag["gates"].detach().float()
        for row in range(gates.shape[0]):
            q = min(int(queries[row].item()), gates.shape[1] - 1)
            if q < 0:
                continue
            query_gate = gates[row, q]
            token_scores = (gates[row, : q + 1] * query_gate.view(1, -1)).sum(dim=-1)
            useful = rel[row][rel[row] >= 0].clamp(max=q)
            distractors = dist[row][dist[row] >= 0].clamp(max=q)
            auc = _auc_for_positions(token_scores, useful, distractors)
            if auc is not None:
                accum["avenue_auc_sum"] += auc
                accum["avenue_auc_count"] += 1.0
            for k in (1, 3):
                pos_recall = _topk_position_recall(token_scores, useful, k)
                if pos_recall is not None:
                    accum[f"top{k}_position_recall_sum"] += pos_recall
                    accum[f"top{k}_position_recall_count"] += 1.0
                avenue_recall = _topk_avenue_recall(gates[row, : q + 1], query_gate, useful, k)
                if avenue_recall is not None:
                    accum[f"top{k}_avenue_recall_sum"] += avenue_recall
                    accum[f"top{k}_avenue_recall_count"] += 1.0
            if useful.numel() > 0:
                accum["useful_concentration_sum"] += float(gates[row, useful].amax(dim=-1).mean().cpu())
                accum["useful_concentration_count"] += 1.0
                useful_mean = gates[row, useful].mean(dim=0)
                accum["query_useful_gate_cosine_sum"] += float(F.cosine_similarity(query_gate, useful_mean, dim=0).cpu())
                accum["query_useful_gate_cosine_count"] += 1.0
            if distractors.numel() > 0:
                accum["distractor_concentration_sum"] += float(gates[row, distractors].amax(dim=-1).mean().cpu())
                accum["distractor_concentration_count"] += 1.0


def _finish_evidence(rank_accum: List[Dict[str, float]]) -> Dict[str, List[Optional[float]]]:
    out: Dict[str, List[Optional[float]]] = {
        "useful_vs_distractor_avenue_auc": [],
        "useful_evidence_avenue_concentration": [],
        "distractor_avenue_concentration": [],
        "query_useful_gate_cosine": [],
    }
    for k in (1, 3):
        out[f"useful_top{k}_position_recall"] = []
        out[f"useful_top{k}_avenue_recall"] = []
    for accum in rank_accum:
        ac = accum.get("avenue_auc_count", 0.0)
        out["useful_vs_distractor_avenue_auc"].append(accum["avenue_auc_sum"] / ac if ac else None)
        uc = accum.get("useful_concentration_count", 0.0)
        dc = accum.get("distractor_concentration_count", 0.0)
        qc = accum.get("query_useful_gate_cosine_count", 0.0)
        out["useful_evidence_avenue_concentration"].append(accum["useful_concentration_sum"] / uc if uc else None)
        out["distractor_avenue_concentration"].append(accum["distractor_concentration_sum"] / dc if dc else None)
        out["query_useful_gate_cosine"].append(accum["query_useful_gate_cosine_sum"] / qc if qc else None)
        for k in (1, 3):
            pc = accum.get(f"top{k}_position_recall_count", 0.0)
            ac2 = accum.get(f"top{k}_avenue_recall_count", 0.0)
            out[f"useful_top{k}_position_recall"].append(accum[f"top{k}_position_recall_sum"] / pc if pc else None)
            out[f"useful_top{k}_avenue_recall"].append(accum[f"top{k}_avenue_recall_sum"] / ac2 if ac2 else None)
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
    diag_accum: DefaultDict[str, float] = defaultdict(float)
    diag_count = 0
    align_pair_accum: List[float] = []
    align_pair_count: List[int] = []

    start = time.perf_counter()
    for profile_idx, profile in enumerate(profiles):
        name = str(profile["name"])
        seq_len = int(profile.get("seq_len", data_cfg.get("seq_len_train", 128)))
        examples = int(profile.get("examples", data_cfg.get("eval_examples", 128)))
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
                loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), batch.labels.view(-1), ignore_index=-100)
            query_logits = logits[torch.arange(bsz, device=device), batch.query_positions]
            pred = query_logits.argmax(dim=-1)
            correct_mask = pred.eq(batch.target_ids)
            correct = int(correct_mask.sum().item())
            loss_value = float(loss.detach().float().cpu())

            scenario_loss += loss_value * bsz
            scenario_correct += correct
            scenario_count += bsz
            total_loss += loss_value * bsz
            total_correct += correct
            total_examples += bsz
            total_tokens += bsz * seq_len

            _update_layer_accum(layer_accum, out["layer_stats"], bsz)
            _update_avenue_evidence_diagnostics(rank_accum, out.get("avenue_attn"), batch)
            diag = out.get("diagnostics", {})
            if "predicted_actual_avenue_cosine" in diag:
                diag_accum["predicted_actual_avenue_cosine"] += float(diag["predicted_actual_avenue_cosine"].detach().float().cpu()) * bsz
            values = diag.get("predicted_actual_avenue_cosine_by_pair", [])
            while len(align_pair_accum) < len(values):
                align_pair_accum.append(0.0)
                align_pair_count.append(0)
            for idx, value in enumerate(values):
                align_pair_accum[idx] += float(value.detach().float().cpu()) * bsz
                align_pair_count[idx] += bsz
            diag_count += bsz

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
    layers.update(_finish_evidence(rank_accum))
    peak_memory_mb = None
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)

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
        "avenue_diagnostics": {
            "predicted_actual_avenue_cosine": diag_accum["predicted_actual_avenue_cosine"] / max(1, diag_count),
            "predicted_actual_avenue_cosine_by_pair": [
                accum / max(1, count) for accum, count in zip(align_pair_accum, align_pair_count)
            ],
        },
        "peak_gpu_memory_mb": peak_memory_mb,
        "control": control,
    }


@torch.no_grad()
def evaluate_avenue_controls(
    model: torch.nn.Module,
    config: Dict[str, Any],
    device: torch.device,
    *,
    seed_offset: int = 800_000,
) -> Dict[str, Any]:
    if not bool(getattr(model, "has_avenues", False)):
        return {}
    data_cfg = config.get("data", {})
    examples = int(data_cfg.get("control_examples", min(96, int(data_cfg.get("eval_examples", 128)))))
    profile = {
        "name": "control_in_distribution",
        "seq_len": int(data_cfg.get("seq_len_train", 128)),
        "distractor_range": list(data_cfg.get("eval_distractor_range", data_cfg.get("train_distractor_range", [4, 14]))),
        "overwrite_range": list(data_cfg.get("eval_overwrite_range", data_cfg.get("train_overwrite_range", [1, 4]))),
        "examples": examples,
    }
    base = evaluate_model(model, config, device, seed_offset=seed_offset, profiles=[profile], control=None)
    controls: Dict[str, Dict[str, Any]] = {}
    for control in AVENUE_CONTROLS:
        metrics = evaluate_model(model, config, device, seed_offset=seed_offset, profiles=[profile], control=control)
        controls[control] = {
            "accuracy": metrics["accuracy"],
            "loss": metrics["loss"],
            "drop": base["accuracy"] - metrics["accuracy"],
            "examples": metrics["examples"],
            "tokens_per_sec": metrics["tokens_per_sec"],
        }

    single: Dict[str, Dict[str, Any]] = {}
    leave_one: Dict[str, Dict[str, Any]] = {}
    num_avenues = int(getattr(model, "num_avenues", config.get("model", {}).get("num_avenues", 1)))
    for idx in range(num_avenues):
        control = f"single_avenue_{idx}"
        metrics = evaluate_model(model, config, device, seed_offset=seed_offset, profiles=[profile], control=control)
        single[control] = {
            "accuracy": metrics["accuracy"],
            "loss": metrics["loss"],
            "drop": base["accuracy"] - metrics["accuracy"],
        }
        if num_avenues > 1:
            leave_control = f"leave_one_{idx}"
            leave_metrics = evaluate_model(model, config, device, seed_offset=seed_offset, profiles=[profile], control=leave_control)
            leave_one[leave_control] = {
                "accuracy": leave_metrics["accuracy"],
                "loss": leave_metrics["loss"],
                "drop": base["accuracy"] - leave_metrics["accuracy"],
            }
    max_single = max((item["accuracy"] for item in single.values()), default=None)
    return {
        "base_accuracy": base["accuracy"],
        "base_loss": base["loss"],
        "examples": examples,
        "controls": controls,
        "single_avenue": single,
        "leave_one_avenue_out": leave_one,
        "max_single_avenue_accuracy": max_single,
        "full_minus_max_single_avenue": None if max_single is None else base["accuracy"] - max_single,
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
    ax.bar(labels, values, color="#496f5d")
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
        for filename, key, title in (
            ("accuracy_by_task.png", "accuracy_by_task", "Accuracy by Task"),
            ("accuracy_vs_seq_len.png", "accuracy_by_seq_len", "Accuracy vs Sequence Length"),
            ("accuracy_vs_distractors.png", "accuracy_by_distractor_count", "Accuracy vs Distractor Count"),
            ("accuracy_vs_overwrite_depth.png", "accuracy_by_overwrite_depth", "Accuracy vs Overwrite Depth"),
        ):
            _plot_lines(
                plots_dir / filename,
                {labels[i]: runs[i].get("eval", {}).get(key, {}) for i in range(len(runs))},
                title,
                key,
                "accuracy",
            )
            written.append(str(plots_dir / filename))
        for metric, filename, title in (
            ("avenue_gate_entropy", "avenue_entropy_by_layer.png", "Avenue Gate Entropy by Layer"),
            ("avenue_diversity", "avenue_diversity_by_layer.png", "Avenue Diversity by Layer"),
        ):
            series: Dict[str, Dict[str, float]] = {}
            for label, run in zip(labels, runs):
                values = run.get("eval", {}).get("layers", {}).get(metric, [])
                series[label] = {str(i): float(v) for i, v in enumerate(values) if v is not None}
            _plot_lines(plots_dir / filename, series, title, "layer", metric)
            written.append(str(plots_dir / filename))
        usage_series: Dict[str, Dict[str, float]] = {}
        for label, run in zip(labels, runs):
            layers = run.get("eval", {}).get("layers", {})
            usage_series[label] = {
                f"a{i}": _mean(layers.get(f"avenue_usage_{i}", [])) or 0.0
                for i in range(4)
            }
        _plot_lines(plots_dir / "avenue_usage_by_layer.png", usage_series, "Mean Avenue Usage", "avenue", "usage")
        written.append(str(plots_dir / "avenue_usage_by_layer.png"))
        loo = {
            label: float(_mean(item.get("drop") for item in run.get("avenue_controls", {}).get("leave_one_avenue_out", {}).values()) or 0.0)
            for label, run in zip(labels, runs)
        }
        _plot_bar(plots_dir / "leave_one_avenue_out.png", list(loo.keys()), list(loo.values()), "Mean Leave-One-Avenue-Out Drop", "accuracy drop")
        written.append(str(plots_dir / "leave_one_avenue_out.png"))
        single_gap = {
            label: float(run.get("avenue_controls", {}).get("full_minus_max_single_avenue") or 0.0)
            for label, run in zip(labels, runs)
        }
        _plot_bar(plots_dir / "single_avenue_vs_full.png", list(single_gap.keys()), list(single_gap.values()), "Full Minus Max Single Avenue", "accuracy gap")
        written.append(str(plots_dir / "single_avenue_vs_full.png"))
        drops = {
            label: float(_mean(item.get("drop") for item in run.get("avenue_controls", {}).get("controls", {}).values()) or 0.0)
            for label, run in zip(labels, runs)
        }
        _plot_bar(plots_dir / "control_drops.png", list(drops.keys()), list(drops.values()), "Mean Avenue Control Drop", "accuracy drop")
        written.append(str(plots_dir / "control_drops.png"))
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
        "single_avenue",
        "four_avenue_sum",
        "four_avenue_gated",
        "four_avenue_dropout05",
        "four_avenue_leave_one_out",
        "mirrored_four_avenue",
    ]
    if any(name not in by_variant for name in required):
        return "inconclusive: not all required E8 scout variants have completed runs"
    baseline = float(by_variant["baseline_transformer"].get("eval", {}).get("accuracy", 0.0))
    param = float(by_variant["param_matched_baseline"].get("eval", {}).get("accuracy", 0.0))
    promising: List[str] = []
    weak: List[str] = []
    for name, run in by_variant.items():
        if name in {"baseline_transformer", "param_matched_baseline", "single_avenue"}:
            continue
        acc = float(run.get("eval", {}).get("accuracy", 0.0))
        controls = run.get("avenue_controls", {})
        control_drop = _mean(item.get("drop") for item in controls.get("controls", {}).values()) or 0.0
        single_gap = controls.get("full_minus_max_single_avenue")
        entropy = _mean(run.get("eval", {}).get("layers", {}).get("avenue_gate_entropy", []))
        diversity = _mean(run.get("eval", {}).get("layers", {}).get("avenue_diversity", []))
        if (
            acc > baseline
            and acc > param
            and control_drop > 0.02
            and single_gap is not None
            and float(single_gap) > 0.02
            and entropy is not None
            and entropy > 0.5
            and diversity is not None
            and diversity > 0.05
        ):
            promising.append(name)
        elif acc > baseline and acc > param:
            weak.append(name)
    if promising:
        return f"promising: {', '.join(promising)} beat both controls and passed the main avenue-use checks"
    if weak:
        return f"weakly promising: {', '.join(weak)} beat both controls, but controls or single-avenue diagnostics are incomplete"
    best_avenue = max(
        (run for run in runs if str(run.get("variant")) not in {"baseline_transformer", "param_matched_baseline"}),
        key=lambda item: float(item.get("eval", {}).get("accuracy", 0.0)),
    )
    if param >= float(best_avenue.get("eval", {}).get("accuracy", 0.0)):
        return "not promising: the parameter-matched baseline equals or beats all avenue variants"
    return "inconclusive: no avenue variant cleanly satisfies the scout criteria"


def write_report(
    root: Path = E8_ROOT,
    *,
    results_dir: Optional[Path] = None,
    plots_dir: Optional[Path] = None,
    report_path: Optional[Path] = None,
    title: str = "Experiment 8 Report",
) -> Path:
    results_dir = results_dir or (root / "results")
    plots_dir = plots_dir or (root / "plots")
    report_path = report_path or (results_dir / "report.md")
    runs = _load_runs(results_dir)
    written_plots = write_plots(runs, plots_dir)

    lines: List[str] = [f"# {title}", ""]
    lines.extend(["## Run Scope", ""])
    if runs:
        seeds = sorted({str(run.get("seed")) for run in runs})
        scope = f"Single-seed scout (`seed={seeds[0]}`)." if len(seeds) == 1 else f"Seeds: {', '.join(seeds)}."
        lines.append(scope + " This is an exploratory screen, not a proof run.")
    else:
        lines.append("No completed runs found.")
    lines.append("")

    lines.extend(["## Summary Table", ""])
    lines.append(
        "| Variant | Seed | Params | Acc | In-dist | Long | High distractor | Many overwrite | Gate entropy | Usage max | Diversity | Single gap | Control drop | GPU MB | Train tok/s | Eval tok/s |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for run in runs:
        layers = run.get("eval", {}).get("layers", {})
        usages = [_mean(layers.get(f"avenue_usage_{i}", [])) for i in range(4)]
        usage_max = max([float(v) for v in usages if v is not None], default=None)
        controls = run.get("avenue_controls", {})
        control_drop = _mean(item.get("drop") for item in controls.get("controls", {}).values()) if controls else None
        long_acc = _scenario_acc(run, "seq_len_256")
        lines.append(
            "| {variant} | {seed} | {params} | {acc} | {ind} | {long} | {high} | {ow} | {entropy} | {usage} | {div} | {gap} | {drop} | {gpu} | {traintps} | {evaltps} |".format(
                variant=run.get("variant"),
                seed=run.get("seed"),
                params=run.get("parameter_count", "NA"),
                acc=_fmt(run.get("eval", {}).get("accuracy")),
                ind=_fmt(_scenario_acc(run, "in_distribution")),
                long=_fmt(long_acc),
                high=_fmt(_scenario_acc(run, "high_distractor")),
                ow=_fmt(_scenario_acc(run, "many_overwrite")),
                entropy=_fmt(_mean(layers.get("avenue_gate_entropy", []))),
                usage=_fmt(usage_max),
                div=_fmt(_mean(layers.get("avenue_diversity", []))),
                gap=_fmt(controls.get("full_minus_max_single_avenue") if controls else None),
                drop=_fmt(control_drop),
                gpu=_fmt(run.get("peak_gpu_memory_mb") or run.get("eval", {}).get("peak_gpu_memory_mb"), 1),
                traintps=_fmt(run.get("train_tokens_per_sec"), 1),
                evaltps=_fmt(run.get("eval", {}).get("tokens_per_sec"), 1),
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

    lines.extend(["", "## Avenue Diagnostics", ""])
    for run in runs:
        layers = run.get("eval", {}).get("layers", {})
        lines.append(
            f"- `{run.get('variant')}` seed `{run.get('seed')}` "
            f"gate_entropy={layers.get('avenue_gate_entropy')} "
            f"usage={[layers.get(f'avenue_usage_{i}') for i in range(4)]} "
            f"output_norm={layers.get('avenue_output_norm')} "
            f"pairwise_cosine={layers.get('pairwise_avenue_cosine')} "
            f"diversity={layers.get('avenue_diversity')} "
            f"alignment={run.get('eval', {}).get('avenue_diagnostics', {})}"
        )

    lines.extend(["", "## Single-Avenue Analysis", ""])
    for run in runs:
        controls = run.get("avenue_controls", {})
        lines.append(
            f"- `{run.get('variant')}` seed `{run.get('seed')}` "
            f"full={controls.get('base_accuracy') if controls else 'NA'} "
            f"single={controls.get('single_avenue') if controls else 'NA'} "
            f"max_single={controls.get('max_single_avenue_accuracy') if controls else 'NA'} "
            f"full_minus_max={controls.get('full_minus_max_single_avenue') if controls else 'NA'} "
            f"leave_one={controls.get('leave_one_avenue_out') if controls else 'NA'}"
        )

    lines.extend(["", "## Control Results", ""])
    for run in runs:
        controls = run.get("avenue_controls", {})
        lines.append(f"- `{run.get('variant')}` seed `{run.get('seed')}`: {controls.get('controls') if controls else 'not applicable'}")

    lines.extend(["", "## Evidence Diagnostics", ""])
    for run in runs:
        layers = run.get("eval", {}).get("layers", {})
        lines.append(
            f"- `{run.get('variant')}` seed `{run.get('seed')}` "
            f"auc={layers.get('useful_vs_distractor_avenue_auc')} "
            f"top1_pos={layers.get('useful_top1_position_recall')} "
            f"top3_pos={layers.get('useful_top3_position_recall')} "
            f"top1_avenue={layers.get('useful_top1_avenue_recall')} "
            f"top3_avenue={layers.get('useful_top3_avenue_recall')} "
            f"useful_concentration={layers.get('useful_evidence_avenue_concentration')} "
            f"distractor_concentration={layers.get('distractor_avenue_concentration')} "
            f"query_useful_gate_cosine={layers.get('query_useful_gate_cosine')}"
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

    lines.extend(["", "## Conclusion", "", _conclusion(runs), ""])
    lines.extend(
        [
            "## Failure Analysis",
            "",
            "Avenue gains should be discounted if parameter matching wins, controls improve or do not reduce accuracy, max single-avenue accuracy matches the full model, usage collapses to one avenue, or pairwise avenue cosine indicates identical pathways.",
            "",
            "## Next Recommended Experiment",
            "",
            "Scale only variants that beat both controls and show positive full-minus-single-avenue gaps. For the next pass, rerun seeds 0-2 and add an explicit avenue-diversity sweep with `lambda_div` in {0.001, 0.01, 0.05}.",
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
    parser = argparse.ArgumentParser(description="Evaluation/report utilities for 21EYES E8")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--plots-dir", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--title", type=str, default="Experiment 8 Report")
    args = parser.parse_args()
    if not args.report:
        parser.error("No action requested. Use --report.")
    results_dir = args.results_dir
    plots_dir = args.plots_dir
    report_path = args.report_path
    if results_dir is not None and not results_dir.is_absolute():
        results_dir = E8_ROOT / results_dir
    if plots_dir is not None and not plots_dir.is_absolute():
        plots_dir = E8_ROOT / plots_dir
    if report_path is not None and not report_path.is_absolute():
        report_path = E8_ROOT / report_path
    path = write_report(E8_ROOT, results_dir=results_dir, plots_dir=plots_dir, report_path=report_path, title=args.title)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
