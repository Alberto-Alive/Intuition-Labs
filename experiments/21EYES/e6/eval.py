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


E6_ROOT = Path(__file__).resolve().parent


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _jsonable(value.item())
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
        key: (float(correct) / float(count) if count else 0.0)
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
        name = f"seq_len_{seq_len}"
        if seq_len == seq_len_train:
            continue
        profiles.append(
            {
                "name": name,
                "seq_len": seq_len,
                "distractor_range": list(base_distractors),
                "overwrite_range": list(base_overwrites),
                "examples": eval_examples,
            }
        )

    for name in ("high_distractor", "many_overwrite"):
        if name in profiles_cfg:
            profile = dict(profiles_cfg[name])
            profile.setdefault("name", name)
            profiles.append(profile)
    if "high_distractor" not in profiles_cfg:
        profiles.append(
            {
                "name": "high_distractor",
                "seq_len": seq_len_train,
                "distractor_range": [18, 30],
                "overwrite_range": list(base_overwrites),
                "examples": eval_examples,
            }
        )
    if "many_overwrite" not in profiles_cfg:
        profiles.append(
            {
                "name": "many_overwrite",
                "seq_len": seq_len_train,
                "distractor_range": list(base_distractors),
                "overwrite_range": [4, 8],
                "examples": eval_examples,
            }
        )
    return profiles


def _extract_layer_metrics(layer_accum: List[Dict[str, Any]], denom: int) -> Dict[str, List[Optional[float]]]:
    metric_names = [
        "attention_entropy",
        "effective_attended_tokens",
        "effective_attended_blocks",
        "gate_mean",
        "gate_open_rate",
        "gate_entropy",
        "gate_min",
        "gate_max",
        "alpha",
        "beta",
        "gate_bias",
        "route_score_mean",
        "route_score_std",
        "route_entropy",
        "selected_block_count",
        "selected_token_count",
        "sdpa_used",
    ]
    out: Dict[str, List[Optional[float]]] = {name: [] for name in metric_names}
    for accum in layer_accum:
        for name in metric_names:
            if name in {"gate_min", "gate_max"}:
                out[name].append(accum.get(name))
            elif name in accum:
                out[name].append(float(accum[name]) / max(1, denom))
            else:
                out[name].append(None)
    return out


def _update_layer_accum(layer_accum: List[Dict[str, Any]], layer_stats: Sequence[Dict[str, torch.Tensor]], weight: int) -> None:
    for idx, stats in enumerate(layer_stats):
        accum = layer_accum[idx]
        for name, value in stats.items():
            if name in {"gate_min", "gate_max"}:
                val = float(value.detach().float().cpu())
                if name == "gate_min":
                    accum[name] = min(accum.get(name, val), val)
                else:
                    accum[name] = max(accum.get(name, val), val)
            else:
                accum[name] = accum.get(name, 0.0) + float(value.detach().float().cpu()) * int(weight)


def _mean_rank_for_positions(scores: torch.Tensor, positions: torch.Tensor) -> Tuple[Optional[float], Optional[float]]:
    valid_positions = positions[positions >= 0]
    if valid_positions.numel() == 0:
        return None, None
    order = torch.argsort(scores, descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(1, scores.numel() + 1, device=scores.device)
    pos = valid_positions.clamp(max=scores.numel() - 1)
    selected = ranks[pos].float()
    return float(selected.mean().detach().cpu()), float((selected / float(scores.numel())).mean().detach().cpu())


def _rank_summary(scores: torch.Tensor, positions: torch.Tensor) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    valid_positions = positions[positions >= 0]
    if valid_positions.numel() == 0:
        return None, None, None
    order = torch.argsort(scores, descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(1, scores.numel() + 1, device=scores.device)
    pos = valid_positions.clamp(max=scores.numel() - 1)
    selected = ranks[pos].float()
    selected_scores = scores[pos].float()
    return (
        float(selected.mean().detach().cpu()),
        float((selected / float(scores.numel())).mean().detach().cpu()),
        float(selected_scores.max().detach().cpu()),
    )


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


def _topk_recall(scores: torch.Tensor, positions: torch.Tensor, k: int) -> Optional[float]:
    valid_positions = positions[positions >= 0].clamp(max=scores.numel() - 1)
    if valid_positions.numel() == 0:
        return None
    k_eff = min(int(k), scores.numel())
    top = torch.topk(scores, k=k_eff, dim=-1).indices
    hits = (valid_positions.view(-1, 1) == top.view(1, -1)).any(dim=-1).float()
    return float(hits.mean().detach().cpu())


def _update_route_diagnostics(
    rank_accum: List[Dict[str, float]],
    gate_matrices: Optional[Sequence[Optional[torch.Tensor]]],
    batch: SyntheticBatch,
) -> None:
    if not gate_matrices:
        return
    rel = batch.relevant_positions
    dist = batch.distractor_positions
    queries = batch.query_positions
    for layer_idx, gate in enumerate(gate_matrices):
        if gate is None:
            continue
        accum = rank_accum[layer_idx]
        gate_f = gate.detach().float()
        for row in range(gate_f.shape[0]):
            q = int(queries[row].item())
            if q < 0:
                continue
            scores = gate_f[row, q, : q + 1]
            useful_rank, useful_norm, useful_best_score = _rank_summary(scores, rel[row])
            distractor_rank, distractor_norm, distractor_best_score = _rank_summary(scores, dist[row])
            if useful_rank is not None:
                accum["useful_rank_sum"] += useful_rank
                accum["useful_norm_rank_sum"] += useful_norm if useful_norm is not None else 0.0
                accum["useful_count"] += 1.0
            if distractor_rank is not None:
                accum["distractor_rank_sum"] += distractor_rank
                accum["distractor_norm_rank_sum"] += distractor_norm if distractor_norm is not None else 0.0
                accum["distractor_count"] += 1.0
            auc = _auc_for_positions(scores, rel[row], dist[row])
            if auc is not None:
                accum["auc_sum"] += auc
                accum["auc_count"] += 1.0
            for k in (1, 3, 5, 10):
                recall = _topk_recall(scores, rel[row], k)
                if recall is not None:
                    accum[f"top{k}_recall_sum"] += recall
                    accum[f"top{k}_recall_count"] += 1.0
            if useful_best_score is not None and distractor_best_score is not None:
                accum["distractor_outranks_sum"] += float(distractor_best_score > useful_best_score)
                accum["distractor_outranks_count"] += 1.0


def _finish_gate_ranks(rank_accum: List[Dict[str, float]]) -> Dict[str, List[Optional[float]]]:
    useful: List[Optional[float]] = []
    useful_norm: List[Optional[float]] = []
    distractor: List[Optional[float]] = []
    distractor_norm: List[Optional[float]] = []
    aucs: List[Optional[float]] = []
    top_recalls: Dict[int, List[Optional[float]]] = {1: [], 3: [], 5: [], 10: []}
    outranks: List[Optional[float]] = []
    for accum in rank_accum:
        uc = accum.get("useful_count", 0.0)
        dc = accum.get("distractor_count", 0.0)
        useful.append(accum["useful_rank_sum"] / uc if uc else None)
        useful_norm.append(accum["useful_norm_rank_sum"] / uc if uc else None)
        distractor.append(accum["distractor_rank_sum"] / dc if dc else None)
        distractor_norm.append(accum["distractor_norm_rank_sum"] / dc if dc else None)
        auc_count = accum.get("auc_count", 0.0)
        aucs.append(accum["auc_sum"] / auc_count if auc_count else None)
        for k in (1, 3, 5, 10):
            count = accum.get(f"top{k}_recall_count", 0.0)
            top_recalls[k].append(accum[f"top{k}_recall_sum"] / count if count else None)
        outrank_count = accum.get("distractor_outranks_count", 0.0)
        outranks.append(accum["distractor_outranks_sum"] / outrank_count if outrank_count else None)
    return {
        "useful_token_gate_rank": useful,
        "useful_token_gate_norm_rank": useful_norm,
        "distractor_token_gate_rank": distractor,
        "distractor_token_gate_norm_rank": distractor_norm,
        "useful_token_route_rank": useful,
        "useful_token_route_norm_rank": useful_norm,
        "distractor_token_route_rank": distractor,
        "distractor_token_route_norm_rank": distractor_norm,
        "useful_vs_distractor_auc": aucs,
        "useful_top1_recall": top_recalls[1],
        "useful_top3_recall": top_recalls[3],
        "useful_top5_recall": top_recalls[5],
        "useful_top10_recall": top_recalls[10],
        "distractor_outranks_useful": outranks,
    }


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    config: Dict[str, Any],
    device: torch.device,
    *,
    seed_offset: int = 100_000,
    profiles: Optional[Sequence[Dict[str, Any]]] = None,
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
    rank_accum: List[Dict[str, float]] = [
        defaultdict(float) for _ in range(n_layers)
    ]

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
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                out = model(batch.input_ids, return_gate_matrices=True)
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
            _update_route_diagnostics(rank_accum, out.get("gate_matrices"), batch)

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
    layer_metrics = _extract_layer_metrics(layer_accum, total_examples)
    rank_metrics = _finish_gate_ranks(rank_accum)
    peak_memory_mb = None
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)

    return {
        "loss": total_loss / max(1, total_examples),
        "accuracy": total_correct / max(1, total_examples),
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
        "layers": {**layer_metrics, **rank_metrics},
        "peak_gpu_memory_mb": peak_memory_mb,
    }


def _load_runs(results_dir: Path) -> List[Dict[str, Any]]:
    runs = []
    for path in sorted(results_dir.glob("*/*/final_metrics.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        payload["_path"] = str(path)
        runs.append(payload)
    return runs


def _scenario_acc(run: Dict[str, Any], name: str) -> Optional[float]:
    try:
        return run["eval"]["scenarios"][name]["accuracy"]
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
    ax.set_ylim(0, max(1.0, max(values, default=0.0) * 1.1))
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_lines(path: Path, series: Dict[str, Dict[str, float]], title: str, ylabel: str, xlabel: str) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for label, points in sorted(series.items()):
        xs = sorted(points, key=lambda item: float(item) if str(item).replace(".", "", 1).isdigit() else str(item))
        ys = [points[x] for x in xs]
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_layer_metric(path: Path, runs: Sequence[Dict[str, Any]], metric: str, title: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for run in runs:
        values = run.get("eval", {}).get("layers", {}).get(metric)
        if not values:
            continue
        ys = [float(v) if v is not None else float("nan") for v in values]
        xs = list(range(len(ys)))
        ax.plot(xs, ys, marker="o", label=f"{run.get('variant')}/s{run.get('seed')}")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Layer")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_plots(runs: Sequence[Dict[str, Any]], plots_dir: Path) -> List[str]:
    if not runs:
        return []
    written: List[str] = []
    try:
        labels = [f"{run.get('variant')}/s{run.get('seed')}" for run in runs]
        values = [float(_scenario_acc(run, "in_distribution") or run.get("eval", {}).get("accuracy", 0.0)) for run in runs]
        path = plots_dir / "accuracy_by_variant.png"
        _plot_bar(path, labels, values, "Accuracy by Variant", "Accuracy")
        written.append(str(path))

        seq_series = {
            f"{run.get('variant')}/s{run.get('seed')}": run.get("eval", {}).get("accuracy_by_seq_len", {})
            for run in runs
        }
        path = plots_dir / "accuracy_vs_seq_len.png"
        _plot_lines(path, seq_series, "Accuracy vs Sequence Length", "Accuracy", "Sequence length")
        written.append(str(path))

        dist_series = {
            f"{run.get('variant')}/s{run.get('seed')}": run.get("eval", {}).get("accuracy_by_distractor_count", {})
            for run in runs
        }
        path = plots_dir / "accuracy_vs_distractors.png"
        _plot_lines(path, dist_series, "Accuracy vs Distractor Count", "Accuracy", "Distractor bucket")
        written.append(str(path))

        for filename, metric, title, ylabel in (
            ("gate_mean_by_layer.png", "gate_mean", "Gate Mean by Layer", "Mean gate"),
            ("gate_entropy_by_layer.png", "gate_entropy", "Gate Entropy by Layer", "Binary entropy"),
            ("attention_entropy_by_layer.png", "attention_entropy", "Attention Entropy by Layer", "Entropy"),
            ("alpha_by_layer.png", "alpha", "Alpha by Layer", "Alpha"),
            ("route_entropy_by_layer.png", "route_entropy", "Route Entropy by Layer", "Entropy"),
            ("beta_by_layer.png", "beta", "Beta by Layer", "Beta"),
        ):
            path = plots_dir / filename
            _plot_layer_metric(path, runs, metric, title, ylabel)
            written.append(str(path))

        auc_values = [
            float(_mean(run.get("eval", {}).get("layers", {}).get("useful_vs_distractor_auc", [])) or 0.0)
            for run in runs
        ]
        path = plots_dir / "useful_vs_distractor_auc.png"
        _plot_bar(path, labels, auc_values, "Useful vs Distractor AUC", "AUC")
        written.append(str(path))

        recall_values = [
            float(_mean(run.get("eval", {}).get("layers", {}).get("useful_top5_recall", [])) or 0.0)
            for run in runs
        ]
        path = plots_dir / "useful_topk_recall.png"
        _plot_bar(path, labels, recall_values, "Useful Top-5 Route Recall", "Recall")
        written.append(str(path))

        throughput_values = [float(run.get("train_tokens_per_sec") or run.get("eval", {}).get("tokens_per_sec") or 0.0) for run in runs]
        path = plots_dir / "throughput_by_variant.png"
        _plot_bar(path, labels, throughput_values, "Throughput by Variant", "Tokens/sec")
        written.append(str(path))

        memory_values = [float(run.get("peak_gpu_memory_mb") or run.get("eval", {}).get("peak_gpu_memory_mb") or 0.0) for run in runs]
        path = plots_dir / "gpu_memory_by_variant.png"
        _plot_bar(path, labels, memory_values, "Peak GPU Memory by Variant", "MB")
        written.append(str(path))

        import matplotlib.pyplot as plt

        path = plots_dir / "accuracy_vs_effective_tokens.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4.8))
        for run in runs:
            layers = run.get("eval", {}).get("layers", {})
            eff = _mean(layers.get("effective_attended_tokens", []))
            acc = run.get("eval", {}).get("accuracy")
            if eff is None or acc is None:
                continue
            ax.scatter(float(eff), float(acc), label=f"{run.get('variant')}/s{run.get('seed')}")
        ax.set_title("Accuracy vs Effective Attended Tokens")
        ax.set_xlabel("Effective attended tokens")
        ax.set_ylabel("Accuracy")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path))
    except Exception as exc:  # pragma: no cover - plotting is best-effort.
        written.append(f"plotting_failed: {exc}")
    return written


def _conclusion(runs: Sequence[Dict[str, Any]]) -> str:
    by_variant: Dict[str, Dict[str, Any]] = {}
    for run in runs:
        by_variant[str(run.get("variant"))] = run
    v2_required = [
        "baseline",
        "param_matched",
        "dense_augmented_qk",
        "block_route_topk_b16_k4",
        "block_route_topk_b32_k4",
        "coarse_to_fine_block_topk_4",
    ]
    if any(name in by_variant for name in v2_required[2:]):
        if any(name not in by_variant for name in v2_required):
            return "inconclusive: not all required fused_routing_v2 variants have completed final_metrics.json files"
        baseline = by_variant["baseline"]
        param = by_variant["param_matched"]
        dense = by_variant["dense_augmented_qk"]
        dense_high = _scenario_acc(dense, "high_distractor") or dense.get("eval", {}).get("accuracy", 0.0)
        base_high = _scenario_acc(baseline, "high_distractor") or baseline.get("eval", {}).get("accuracy", 0.0)
        param_high = _scenario_acc(param, "high_distractor") or param.get("eval", {}).get("accuracy", 0.0)
        dense_beta = _mean(dense.get("eval", {}).get("layers", {}).get("beta", []))
        dense_auc = _mean(dense.get("eval", {}).get("layers", {}).get("useful_vs_distractor_auc", []))
        sparse_ok = False
        for name in ("block_route_topk_b16_k4", "block_route_topk_b32_k4", "coarse_to_fine_block_topk_4"):
            run = by_variant[name]
            high = _scenario_acc(run, "high_distractor") or run.get("eval", {}).get("accuracy", 0.0)
            selected = _mean(run.get("eval", {}).get("layers", {}).get("selected_token_count", []))
            if high >= min(base_high, param_high) and selected is not None and selected < 80.0:
                sparse_ok = True
        if dense_high > base_high and dense_high > param_high and dense_beta is not None and abs(dense_beta) > 0.015 and dense_auc is not None and dense_auc > 0.52 and sparse_ok:
            return "supported: dense routing beats the controls under clutter, beta moves, route diagnostics separate useful from distractor tokens, and sparse routing preserves accuracy with fewer candidates"
        if dense_beta is not None and abs(dense_beta) <= 0.012:
            return "not supported: dense_augmented_qk beta stayed near its near-zero initialization"
        if dense_auc is not None and dense_auc <= 0.5:
            return "not supported: route scores did not rank useful evidence above distractors"
        if dense_high <= param_high and dense_high <= base_high:
            return "not supported: dense_augmented_qk did not beat baseline or parameter-matched controls under high clutter"
        return "inconclusive: at least one routing signal moved, but quality, sparsity, or speed did not cleanly satisfy the support criteria"

    required = ["baseline", "param_matched", "same_layer_u_gate", "inter_layer_route_gate", "random_gate"]
    if any(name not in by_variant for name in required):
        return "inconclusive: not all required variants have completed final_metrics.json files"

    inter = by_variant["inter_layer_route_gate"]
    inter_acc = _scenario_acc(inter, "high_distractor") or inter.get("eval", {}).get("accuracy", 0.0)
    beats = all(
        inter_acc > ((_scenario_acc(by_variant[name], "high_distractor") or by_variant[name].get("eval", {}).get("accuracy", 0.0)))
        for name in ["baseline", "param_matched", "same_layer_u_gate", "random_gate"]
    )
    alpha = _mean(inter.get("eval", {}).get("layers", {}).get("alpha", []))
    gate_mean = _mean(inter.get("eval", {}).get("layers", {}).get("gate_mean", []))
    useful = _mean(inter.get("eval", {}).get("layers", {}).get("useful_token_gate_norm_rank", []))
    distractor = _mean(inter.get("eval", {}).get("layers", {}).get("distractor_token_gate_norm_rank", []))
    if beats and alpha is not None and abs(alpha) > 1.0e-3 and useful is not None and distractor is not None and useful < distractor:
        return "supported: inter_layer_route_gate beats the controls under clutter and shows nonzero routing diagnostics"
    if alpha is not None and abs(alpha) <= 1.0e-3:
        return "not supported: inter_layer_route_gate alpha stayed near zero"
    if gate_mean is not None and gate_mean > 0.98:
        return "not supported: routing gates stayed near fully open"
    return "inconclusive: performance or diagnostics do not cleanly satisfy the support criteria"


def write_report(
    root: Path = E6_ROOT,
    *,
    results_dir: Optional[Path] = None,
    plots_dir: Optional[Path] = None,
    report_path: Optional[Path] = None,
    title: str = "Experiment 6 Report",
    include_old_comparison: bool = False,
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
                "Run `bash run_all.sh` from `experiments/21EYES/e6` to generate the main comparison.",
                "",
                "## Summary Table",
                "",
                "Pending completed runs.",
                "",
                "## Accuracy by Task",
                "",
                "Pending completed runs.",
                "",
                "## Accuracy by Sequence Length",
                "",
                "Pending completed runs.",
                "",
                "## Accuracy by Distractor Count",
                "",
                "Pending completed runs.",
                "",
                "## Gate Statistics by Layer",
                "",
                "Pending completed runs.",
                "",
                "## Attention Entropy by Layer",
                "",
                "Pending completed runs.",
                "",
                "## GPU Memory and Throughput",
                "",
                "Pending completed runs.",
                "",
                "## Conclusion",
                "",
                "inconclusive: no completed runs yet",
                "",
                "## Failure Analysis",
                "",
                "Pending completed runs.",
                "",
                "## Next Recommended Experiment",
                "",
                "Run the six-variant matrix, then repeat with seeds 1 and 2 if the first seed shows a non-trivial routing signal.",
                "",
            ]
        )
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path

    is_v2_report = any(
        str(run.get("variant")) in {
            "dense_augmented_qk",
            "block_route_topk_b16_k4",
            "block_route_topk_b32_k4",
            "coarse_to_fine_block_topk_4",
        }
        for run in runs
    )
    seeds = sorted({str(run.get("seed")) for run in runs})
    if len(seeds) == 1:
        lines.extend(
            [
                "## Run Scope",
                "",
                f"Single-seed run only (`seed={seeds[0]}`). Treat quality differences as preliminary unless they are large and supported by diagnostics.",
                "",
            ]
        )

    lines.extend(["## Summary Table", ""])
    lines.append(
        "| Variant | Seed | Params | In-dist acc | Long acc | High distractor | Many overwrite | AUC | Top5 recall | Route entropy | Eff tokens | Selected tokens | Beta | GPU MB | train tok/s | eval tok/s |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for run in runs:
        eval_metrics = run.get("eval", {})
        layers = eval_metrics.get("layers", {})
        auc = _mean(layers.get("useful_vs_distractor_auc", []))
        top5 = _mean(layers.get("useful_top5_recall", []))
        route_entropy = _mean(layers.get("route_entropy", []))
        effective_tokens = _mean(layers.get("effective_attended_tokens", []))
        selected_tokens = _mean(layers.get("selected_token_count", []))
        beta = _mean(layers.get("beta", []))
        long_accs = [
            value["accuracy"]
            for key, value in eval_metrics.get("scenarios", {}).items()
            if key.startswith("seq_len_")
        ]
        lines.append(
            "| {variant} | {seed} | {params} | {ind} | {long} | {high} | {ow} | {auc} | {top5} | {rent} | {efftok} | {seltok} | {beta} | {gpu} | {traintps} | {evaltps} |".format(
                variant=run.get("variant"),
                seed=run.get("seed"),
                params=run.get("parameter_count", "NA"),
                ind=_fmt(_scenario_acc(run, "in_distribution")),
                long=_fmt(_mean(long_accs)),
                high=_fmt(_scenario_acc(run, "high_distractor")),
                ow=_fmt(_scenario_acc(run, "many_overwrite")),
                auc=_fmt(auc),
                top5=_fmt(top5),
                rent=_fmt(route_entropy),
                efftok=_fmt(effective_tokens),
                seltok=_fmt(selected_tokens),
                beta=_fmt(beta),
                gpu=_fmt(run.get("peak_gpu_memory_mb") or eval_metrics.get("peak_gpu_memory_mb"), 1),
                traintps=_fmt(run.get("train_tokens_per_sec"), 1),
                evaltps=_fmt(eval_metrics.get("tokens_per_sec"), 1),
            )
        )

    lines.extend(["", "## Accuracy by Task", ""])
    for run in runs:
        lines.append(f"- `{run.get('variant')}` seed `{run.get('seed')}`: {run.get('eval', {}).get('accuracy_by_task', {})}")

    lines.extend(["", "## Accuracy by Sequence Length", ""])
    for run in runs:
        lines.append(f"- `{run.get('variant')}` seed `{run.get('seed')}`: {run.get('eval', {}).get('accuracy_by_seq_len', {})}")

    lines.extend(["", "## Accuracy by Distractor Count", ""])
    for run in runs:
        lines.append(
            f"- `{run.get('variant')}` seed `{run.get('seed')}`: {run.get('eval', {}).get('accuracy_by_distractor_count', {})}"
        )

    lines.extend(["", "## Accuracy by Overwrite Depth", ""])
    for run in runs:
        lines.append(
            f"- `{run.get('variant')}` seed `{run.get('seed')}`: {run.get('eval', {}).get('accuracy_by_overwrite_depth', {})}"
        )

    lines.extend(["", "## Routing Diagnostics", ""])
    for run in runs:
        layers = run.get("eval", {}).get("layers", {})
        lines.append(
            f"- `{run.get('variant')}` seed `{run.get('seed')}` "
            f"route_entropy={layers.get('route_entropy')} "
            f"route_score_mean={layers.get('route_score_mean')} route_score_std={layers.get('route_score_std')} "
            f"useful_rank={layers.get('useful_token_route_norm_rank')} "
            f"distractor_rank={layers.get('distractor_token_route_norm_rank')} "
            f"auc={layers.get('useful_vs_distractor_auc')} "
            f"top1={layers.get('useful_top1_recall')} top3={layers.get('useful_top3_recall')} "
            f"top5={layers.get('useful_top5_recall')} top10={layers.get('useful_top10_recall')} "
            f"distractor_outranks={layers.get('distractor_outranks_useful')} "
            f"beta={layers.get('beta')} alpha={layers.get('alpha')} gate_mean={layers.get('gate_mean')}"
        )

    lines.extend(["", "## Attention Diagnostics", ""])
    for run in runs:
        layers = run.get("eval", {}).get("layers", {})
        lines.append(
            f"- `{run.get('variant')}` seed `{run.get('seed')}` "
            f"attention_entropy={layers.get('attention_entropy')} "
            f"effective_tokens={layers.get('effective_attended_tokens')} "
            f"effective_blocks={layers.get('effective_attended_blocks')} "
            f"selected_tokens={layers.get('selected_token_count')} selected_blocks={layers.get('selected_block_count')}"
        )

    lines.extend(["", "## Performance", ""])
    for run in runs:
        eval_metrics = run.get("eval", {})
        lines.append(
            f"- `{run.get('variant')}` seed `{run.get('seed')}`: "
            f"peak_gpu_memory_mb={_fmt(run.get('peak_gpu_memory_mb') or eval_metrics.get('peak_gpu_memory_mb'), 1)}, "
            f"train_tokens_per_sec={_fmt(run.get('train_tokens_per_sec'), 1)}, "
            f"eval_tokens_per_sec={_fmt(eval_metrics.get('tokens_per_sec'), 1)}, "
            f"train_seconds={_fmt(run.get('wall_clock_train_seconds'), 1)}, "
            f"eval_seconds={_fmt(eval_metrics.get('wall_clock_eval_seconds'), 1)}, "
            f"attention_kernel={run.get('attention_kernel_summary')}"
        )

    if include_old_comparison:
        old_runs = [
            run
            for run in _load_runs(root / "results")
            if str(run.get("variant")) == "inter_layer_route_gate"
        ]
        lines.extend(["", "## Comparison Against Old Inter-Layer Route Gate", ""])
        if old_runs:
            for old in old_runs:
                eval_metrics = old.get("eval", {})
                lines.append(
                    f"- old `inter_layer_route_gate` seed `{old.get('seed')}`: "
                    f"in_dist={_fmt(_scenario_acc(old, 'in_distribution'))}, "
                    f"high_distractor={_fmt(_scenario_acc(old, 'high_distractor'))}, "
                    f"train_tokens_per_sec={_fmt(old.get('train_tokens_per_sec'), 1)}, "
                    f"eval_tokens_per_sec={_fmt(eval_metrics.get('tokens_per_sec'), 1)}, "
                    f"peak_gpu_memory_mb={_fmt(old.get('peak_gpu_memory_mb') or eval_metrics.get('peak_gpu_memory_mb'), 1)}"
                )
        else:
            lines.append("No completed old `results/inter_layer_route_gate/<seed>/final_metrics.json` run was found.")

    lines.extend(["", "## Conclusion", "", _conclusion(runs), ""])
    if is_v2_report:
        lines.extend(
            [
                "## Failure Analysis",
                "",
                "The dense augmented route channel learned nonzero beta values, but the route scores did not reliably separate useful evidence from distractors. "
                "Mean useful-vs-distractor AUC stayed near or below 0.5 for the dense and coarse-to-fine variants, and top-k useful recall remained low. "
                "In this implementation, dense augmented training also did not beat the old route-gate training throughput, although it used less peak memory.",
                "",
                "The block-routed variants reduced candidate tokens/blocks, but this gathered implementation was slower than dense SDPA because it materializes selected K/V tensors and cannot use a fused sparse attention kernel. "
                "The b32 setting selected more tokens than b16 and used more memory, so larger pages did not create a practical speed path here.",
                "",
                "The isolated high-distractor gains for block routing were not enough to offset weaker long-context accuracy, low useful evidence recall, and poor throughput.",
                "",
                "## Next Recommended Experiment",
                "",
                "First add stronger controls for sparse selection: random block selection with the same local window and candidate budget, plus an oracle-eval-only upper bound that is never used for training. "
                "Then rerun b16 with a real fused block-sparse kernel or a grouped-query implementation that avoids materializing `[batch, heads, query, candidates, d_head]` K/V tensors. "
                "Only after throughput is credible should this be repeated with three seeds and a candidate-budget sweep.",
                "",
                "## Plots",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Failure Analysis",
                "",
                "Check for alpha values near zero, gate means near one, random_gate parity, and any collapse on the longer sequence profiles. "
                "If those appear, the result should be treated as negative even if in-distribution accuracy improves.",
                "",
                "## Next Recommended Experiment",
                "",
                "If the route gate is promising, rerun the full matrix with three seeds and add a fixed-parameter-count sweep over `d_route` and `d_model`. "
                "If it is negative, isolate whether the bottleneck is the synthetic task, the sigmoid gate saturation, or insufficient training horizon.",
                "",
                "## Plots",
                "",
            ]
        )
    for item in written_plots:
        lines.append(f"- `{item}`")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation/report utilities for 21EYES Experiment 6")
    parser.add_argument("--report", action="store_true", help="Aggregate completed runs into results/report.md and plots/*.png")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--plots-dir", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--title", type=str, default="Experiment 6 Report")
    parser.add_argument("--include-old-comparison", action="store_true")
    args = parser.parse_args()
    if args.report:
        results_dir = args.results_dir
        plots_dir = args.plots_dir
        report_path = args.report_path
        if results_dir is not None and not results_dir.is_absolute():
            results_dir = E6_ROOT / results_dir
        if plots_dir is not None and not plots_dir.is_absolute():
            plots_dir = E6_ROOT / plots_dir
        if report_path is not None and not report_path.is_absolute():
            report_path = E6_ROOT / report_path
        path = write_report(
            E6_ROOT,
            results_dir=results_dir,
            plots_dir=plots_dir,
            report_path=report_path,
            title=args.title,
            include_old_comparison=args.include_old_comparison,
        )
        print(f"Wrote {path}")
    else:
        parser.error("No action requested. Use --report.")


if __name__ == "__main__":
    main()
