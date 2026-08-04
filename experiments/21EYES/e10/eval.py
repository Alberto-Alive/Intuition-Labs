from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence

import torch
from torch.nn import functional as F

from data import TASKS, SyntheticBatcher


E10_ROOT = Path(__file__).resolve().parent
AVENUE_CONTROLS = (
    "avenue_zero",
    "avenue_random",
    "avenue_order_shuffle",
    "avenue_shuffle",
    "hidden_state_shuffle",
)
CROSSING_CONTROLS = (
    "crossing_zero",
    "crossing_shuffle",
    "crossing_random",
    "crossing_identity",
)
INTERWEAVE_CONTROLS = (
    "interweave_zero",
    "interweave_shuffle",
    "interweave_random",
)
PATTERN_CONTROLS = (
    "pattern_zero",
    "pattern_shuffle",
    "pattern_random",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().float().cpu().item()
        return value.detach().float().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")


def _autocast(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
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


def _add_group(groups: DefaultDict[str, List[int]], key: str, correct: int) -> None:
    groups[key][0] += int(correct)
    groups[key][1] += 1


def _finish_groups(groups: DefaultDict[str, List[int]]) -> Dict[str, float]:
    return {key: values[0] / max(1, values[1]) for key, values in sorted(groups.items())}


def _default_eval_profiles(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    data_cfg = config.get("data", {})
    profiles: List[Dict[str, Any]] = []
    raw_profiles = data_cfg.get("eval_profiles", {})
    if isinstance(raw_profiles, dict):
        for name, raw in raw_profiles.items():
            profile = dict(raw)
            profile["name"] = str(name)
            profile.setdefault("split", "val")
            profiles.append(profile)
    if not profiles:
        profiles.append(
            {
                "name": "in_distribution",
                "split": "val",
                "seq_len": int(data_cfg.get("seq_len_train", 128)),
                "distractor_range": list(data_cfg.get("eval_distractor_range", [6, 18])),
                "overwrite_range": list(data_cfg.get("eval_overwrite_range", [2, 5])),
                "examples": int(data_cfg.get("eval_examples", 128)),
            }
        )
    seq_lens = [int(value) for value in data_cfg.get("seq_len_eval", [data_cfg.get("seq_len_train", 128)])]
    train_len = int(data_cfg.get("seq_len_train", seq_lens[0]))
    long_lens = [value for value in seq_lens if value > train_len]
    if long_lens and not any(str(profile.get("name")) == "long" for profile in profiles):
        profiles.append(
            {
                "name": "long",
                "split": "val",
                "seq_len": max(long_lens),
                "distractor_range": list(data_cfg.get("eval_distractor_range", [6, 18])),
                "overwrite_range": list(data_cfg.get("eval_overwrite_range", [2, 5])),
                "examples": int(data_cfg.get("eval_examples", 128)),
            }
        )
    return profiles


def _update_layer_accum(
    layer_accum: List[DefaultDict[str, float]],
    layer_stats: Sequence[Dict[str, torch.Tensor]],
    weight: int,
) -> None:
    for layer_idx, stats in enumerate(layer_stats):
        for key, value in stats.items():
            if not isinstance(value, torch.Tensor) or value.numel() != 1:
                continue
            layer_accum[layer_idx][key] += float(value.detach().float().cpu()) * weight
            layer_accum[layer_idx][f"{key}__count"] += weight


def _extract_layer_metrics(layer_accum: List[DefaultDict[str, float]]) -> Dict[str, List[Optional[float]]]:
    keys = sorted(
        {
            key
            for layer in layer_accum
            for key in layer
            if not key.endswith("__count")
        }
    )
    finished: Dict[str, List[Optional[float]]] = {key: [] for key in keys}
    for key in keys:
        for layer in layer_accum:
            count = layer.get(f"{key}__count", 0.0)
            finished[key].append(None if count <= 0 else layer[key] / count)
    return finished


def _valid_positions(positions: torch.Tensor, query_pos: int) -> torch.Tensor:
    return positions[(positions >= 0) & (positions <= int(query_pos))]


def _auc_for_positions(scores: torch.Tensor, useful: torch.Tensor, distractors: torch.Tensor) -> Optional[float]:
    if useful.numel() == 0 or distractors.numel() == 0:
        return None
    useful_scores = scores.index_select(0, useful.long())
    distractor_scores = scores.index_select(0, distractors.long())
    comparisons = useful_scores[:, None] - distractor_scores[None, :]
    return float(((comparisons > 0).float() + 0.5 * (comparisons == 0).float()).mean().cpu())


def _topk_recall(scores: torch.Tensor, useful: torch.Tensor, query_pos: int, k: int) -> Optional[float]:
    if useful.numel() == 0:
        return None
    visible = scores[: int(query_pos) + 1]
    if visible.numel() == 0:
        return None
    topk = visible.topk(min(int(k), visible.numel())).indices
    return float(torch.isin(useful.long(), topk.long()).any().float().cpu())


def _score_concentration(scores: torch.Tensor, positions: torch.Tensor, query_pos: int) -> Optional[float]:
    if positions.numel() == 0:
        return None
    visible = scores[: int(query_pos) + 1].clamp_min(0.0)
    denom = visible.sum().clamp_min(1.0e-8)
    return float(scores.index_select(0, positions.long()).clamp_min(0.0).sum().div(denom).cpu())


def _query_useful_cosine(context: torch.Tensor, query_pos: int, useful: torch.Tensor) -> Optional[float]:
    if useful.numel() == 0:
        return None
    query = context[int(query_pos)].view(1, -1)
    evidence = context.index_select(0, useful.long())
    return float(F.cosine_similarity(query.float(), evidence.float(), dim=-1).mean().cpu())


def _update_rank_metric(accum: DefaultDict[str, float], key: str, value: Optional[float]) -> None:
    if value is None:
        return
    accum[key] += float(value)
    accum[f"{key}__count"] += 1.0


def _update_crossing_evidence(
    rank_accum: List[DefaultDict[str, float]],
    crossing_diags: Optional[Sequence[Optional[Dict[str, torch.Tensor]]]],
    batch,
) -> None:
    if not crossing_diags:
        return
    for layer_idx, diag in enumerate(crossing_diags):
        if not diag:
            continue
        scores = diag["score"]
        before = diag["before_context"]
        after = diag["after_context"]
        for example_idx in range(scores.shape[0]):
            query_pos = int(batch.query_positions[example_idx].item())
            useful = _valid_positions(batch.relevant_positions[example_idx], query_pos)
            distractors = _valid_positions(batch.distractor_positions[example_idx], query_pos)
            score = scores[example_idx]
            _update_rank_metric(
                rank_accum[layer_idx],
                "useful_vs_distractor_crossing_auc",
                _auc_for_positions(score, useful, distractors),
            )
            _update_rank_metric(rank_accum[layer_idx], "useful_top1_recall", _topk_recall(score, useful, query_pos, 1))
            _update_rank_metric(rank_accum[layer_idx], "useful_top3_recall", _topk_recall(score, useful, query_pos, 3))
            _update_rank_metric(
                rank_accum[layer_idx],
                "useful_evidence_concentration",
                _score_concentration(score, useful, query_pos),
            )
            _update_rank_metric(
                rank_accum[layer_idx],
                "distractor_concentration",
                _score_concentration(score, distractors, query_pos),
            )
            before_cos = _query_useful_cosine(before[example_idx], query_pos, useful)
            after_cos = _query_useful_cosine(after[example_idx], query_pos, useful)
            _update_rank_metric(rank_accum[layer_idx], "query_useful_cosine_before_crossing", before_cos)
            _update_rank_metric(rank_accum[layer_idx], "query_useful_cosine_after_crossing", after_cos)
            if before_cos is not None and after_cos is not None:
                _update_rank_metric(rank_accum[layer_idx], "query_useful_cosine_delta", after_cos - before_cos)


def _finish_evidence(rank_accum: List[DefaultDict[str, float]]) -> Dict[str, List[Optional[float]]]:
    keys = sorted(
        {
            key
            for layer in rank_accum
            for key in layer
            if not key.endswith("__count")
        }
    )
    result: Dict[str, List[Optional[float]]] = {key: [] for key in keys}
    for key in keys:
        for layer in rank_accum:
            count = layer.get(f"{key}__count", 0.0)
            result[key].append(None if count <= 0 else layer[key] / count)
    return result


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    config: Dict[str, Any],
    device: torch.device,
    *,
    seed_offset: int = 100_000,
    profiles: Optional[Sequence[Dict[str, Any]]] = None,
    control: Optional[str] = None,
    collect_diagnostics: bool = True,
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
    layer_accum: List[DefaultDict[str, float]] = [defaultdict(float) for _ in range(n_layers)]
    rank_accum: List[DefaultDict[str, float]] = [defaultdict(float) for _ in range(n_layers)]
    diag_accum: DefaultDict[str, float] = defaultdict(float)
    diag_count = 0
    align_pair_accum: List[float] = []
    align_pair_count: List[int] = []
    scenarios: Dict[str, Dict[str, Any]] = {}
    start = time.perf_counter()

    for profile_idx, profile in enumerate(profiles):
        name = str(profile["name"])
        seq_len = int(profile.get("seq_len", data_cfg.get("seq_len_train", 128)))
        examples = int(profile.get("examples", data_cfg.get("eval_examples", 128)))
        distractor_range = tuple(
            int(value) for value in profile.get("distractor_range", data_cfg.get("eval_distractor_range", [6, 18]))
        )
        overwrite_range = tuple(
            int(value) for value in profile.get("overwrite_range", data_cfg.get("eval_overwrite_range", [2, 5]))
        )
        batcher = SyntheticBatcher(
            seed=int(config.get("seed", 0)) + seed_offset + profile_idx * 10_000,
            split=str(profile.get("split", "val")),
            seq_len=seq_len,
            distractor_range=distractor_range,
            overwrite_range=overwrite_range,
            tasks=profile.get("tasks", TASKS),
        )
        scenario_loss = 0.0
        scenario_examples = 0
        scenario_correct = 0
        remaining = examples
        while remaining > 0:
            bsz = min(eval_batch_size, remaining)
            batch = batcher.sample(bsz).to(device)
            with _autocast(device, amp_enabled):
                out = model(batch.input_ids, return_diagnostics=collect_diagnostics, control=control)
                logits = out["logits"]
                loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), batch.labels.view(-1), ignore_index=-100)
            query_logits = logits[torch.arange(bsz, device=device), batch.query_positions]
            pred = query_logits.argmax(dim=-1)
            correct_mask = pred.eq(batch.target_ids)
            correct = int(correct_mask.sum().item())
            loss_value = float(loss.detach().float().cpu())

            scenario_loss += loss_value * bsz
            scenario_examples += bsz
            scenario_correct += correct
            total_loss += loss_value * bsz
            total_examples += bsz
            total_correct += correct
            total_tokens += bsz * seq_len
            _update_layer_accum(layer_accum, out.get("layer_stats", []), bsz)
            if collect_diagnostics:
                _update_crossing_evidence(rank_accum, out.get("crossing_diags"), batch)
            diag = out.get("diagnostics", {})
            if "predicted_actual_avenue_cosine" in diag:
                diag_accum["predicted_actual_avenue_cosine"] += float(
                    diag["predicted_actual_avenue_cosine"].detach().float().cpu()
                ) * bsz
            values = diag.get("predicted_actual_avenue_cosine_by_pair", [])
            while len(align_pair_accum) < len(values):
                align_pair_accum.append(0.0)
                align_pair_count.append(0)
            for idx, value in enumerate(values):
                align_pair_accum[idx] += float(value.detach().float().cpu()) * bsz
                align_pair_count[idx] += bsz
            diag_count += bsz
            for idx in range(bsz):
                c = int(correct_mask[idx].item())
                _add_group(groups_task, batch.tasks[idx], c)
                _add_group(groups_len, str(seq_len), c)
                _add_group(groups_dist, _bucket_distractors(int(batch.distractor_counts[idx].item())), c)
                _add_group(groups_ow, _bucket_overwrites(int(batch.overwrite_depths[idx].item())), c)
            remaining -= bsz

        scenarios[name] = {
            "loss": scenario_loss / max(1, scenario_examples),
            "accuracy": scenario_correct / max(1, scenario_examples),
            "examples": scenario_examples,
            "seq_len": seq_len,
            "distractor_range": list(distractor_range),
            "overwrite_range": list(overwrite_range),
        }

    elapsed = max(time.perf_counter() - start, 1.0e-9)
    layers = _extract_layer_metrics(layer_accum)
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
def evaluate_controls(
    model: torch.nn.Module,
    config: Dict[str, Any],
    device: torch.device,
    *,
    seed_offset: int = 800_000,
) -> Dict[str, Any]:
    has_avenues = bool(getattr(model, "has_avenues", False))
    has_patterns = bool(getattr(model, "has_patterns", False))
    if not has_avenues and not has_patterns:
        return {}
    data_cfg = config.get("data", {})
    examples = int(data_cfg.get("control_examples", min(96, int(data_cfg.get("eval_examples", 128)))))
    profile = {
        "name": "control_in_distribution",
        "seq_len": int(data_cfg.get("seq_len_train", 128)),
        "distractor_range": list(data_cfg.get("eval_distractor_range", [6, 18])),
        "overwrite_range": list(data_cfg.get("eval_overwrite_range", [2, 5])),
        "examples": examples,
    }
    base = evaluate_model(
        model,
        config,
        device,
        seed_offset=seed_offset,
        profiles=[profile],
        control=None,
        collect_diagnostics=False,
    )

    def _control_metrics(control: str) -> Dict[str, Any]:
        metrics = evaluate_model(
            model,
            config,
            device,
            seed_offset=seed_offset,
            profiles=[profile],
            control=control,
            collect_diagnostics=False,
        )
        return {
            "accuracy": metrics["accuracy"],
            "loss": metrics["loss"],
            "drop": base["accuracy"] - metrics["accuracy"],
            "examples": metrics["examples"],
            "tokens_per_sec": metrics["tokens_per_sec"],
        }

    avenue_controls = {control: _control_metrics(control) for control in AVENUE_CONTROLS} if has_avenues else {}
    crossing_controls: Dict[str, Dict[str, Any]] = {}
    if bool(getattr(model, "has_crossings", False)):
        for control in CROSSING_CONTROLS:
            crossing_controls[control] = _control_metrics(control)
        for idx in range(int(config.get("model", {}).get("n_layers", 4))):
            control = f"crossing_point_zero_{idx}"
            crossing_controls[control] = _control_metrics(control)
    interweave_controls: Dict[str, Dict[str, Any]] = {}
    if bool(getattr(model, "has_interweaves", False)):
        for control in INTERWEAVE_CONTROLS:
            interweave_controls[control] = _control_metrics(control)
        for idx in range(int(config.get("model", {}).get("n_layers", 4))):
            control = f"interweave_point_zero_{idx}"
            interweave_controls[control] = _control_metrics(control)
    pattern_controls: Dict[str, Dict[str, Any]] = {}
    if has_patterns:
        for control in PATTERN_CONTROLS:
            pattern_controls[control] = _control_metrics(control)
        for idx in range(int(config.get("model", {}).get("n_layers", 4))):
            control = f"pattern_point_zero_{idx}"
            pattern_controls[control] = _control_metrics(control)

    single: Dict[str, Dict[str, Any]] = {}
    leave_one: Dict[str, Dict[str, Any]] = {}
    num_avenues = int(getattr(model, "num_avenues", config.get("model", {}).get("num_avenues", 1)))
    if has_avenues:
        for idx in range(num_avenues):
            control = f"single_avenue_{idx}"
            single[control] = _control_metrics(control)
            if num_avenues > 1:
                leave_control = f"leave_one_{idx}"
                leave_one[leave_control] = _control_metrics(leave_control)
    max_single = max((item["accuracy"] for item in single.values()), default=None)
    return {
        "base_accuracy": base["accuracy"],
        "base_loss": base["loss"],
        "examples": examples,
        "avenue_controls": avenue_controls,
        "crossing_controls": crossing_controls,
        "interweave_controls": interweave_controls,
        "pattern_controls": pattern_controls,
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
    vals = [float(value) for value in values if value is not None]
    return sum(vals) / len(vals) if vals else None


def _mean_layer(run: Dict[str, Any], key: str) -> Optional[float]:
    return _mean(run.get("eval", {}).get("layers", {}).get(key, []))


def _control_drop(run: Dict[str, Any], family: str, control: str) -> Optional[float]:
    try:
        return float(run["controls"][family][control]["drop"])
    except KeyError:
        return None


def _plot_bar(path: Path, labels: Sequence[str], values: Sequence[float], title: str, ylabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(labels, values, color="#336b87")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=28)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_lines(path: Path, series: Dict[str, Dict[str, float]], title: str, xlabel: str, ylabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    plotted = False
    for label, values in series.items():
        if not values:
            continue
        keys = list(values.keys())
        ax.plot(keys, [values[key] for key in keys], marker="o", label=label)
        plotted = True
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=18)
    if plotted:
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
                {label: {str(key2): float(value) for key2, value in run["eval"].get(key, {}).items()} for label, run in zip(labels, runs)},
                title,
                key,
                "accuracy",
            )
            written.append(str(plots_dir / filename))
        single_gap = {
            label: float(run.get("controls", {}).get("full_minus_max_single_avenue") or 0.0)
            for label, run in zip(labels, runs)
        }
        _plot_bar(
            plots_dir / "full_vs_max_single_avenue.png",
            list(single_gap.keys()),
            list(single_gap.values()),
            "Full Minus Max Single Avenue",
            "accuracy gap",
        )
        written.append(str(plots_dir / "full_vs_max_single_avenue.png"))
        leave_drop = {
            label: float(
                _mean(item.get("drop") for item in run.get("controls", {}).get("leave_one_avenue_out", {}).values())
                or 0.0
            )
            for label, run in zip(labels, runs)
        }
        _plot_bar(
            plots_dir / "leave_one_avenue_drop.png",
            list(leave_drop.keys()),
            list(leave_drop.values()),
            "Mean Leave-One-Avenue-Out Drop",
            "accuracy drop",
        )
        written.append(str(plots_dir / "leave_one_avenue_drop.png"))
        crossing_drop = {
            label: float(
                _mean(item.get("drop") for item in run.get("controls", {}).get("crossing_controls", {}).values()) or 0.0
            )
            for label, run in zip(labels, runs)
        }
        _plot_bar(
            plots_dir / "crossing_control_drops.png",
            list(crossing_drop.keys()),
            list(crossing_drop.values()),
            "Mean Crossing Control Drop",
            "accuracy drop",
        )
        written.append(str(plots_dir / "crossing_control_drops.png"))
        avenue_drop = {
            label: float(
                _mean(item.get("drop") for item in run.get("controls", {}).get("avenue_controls", {}).values()) or 0.0
            )
            for label, run in zip(labels, runs)
        }
        _plot_bar(
            plots_dir / "avenue_control_drops.png",
            list(avenue_drop.keys()),
            list(avenue_drop.values()),
            "Mean Avenue Control Drop",
            "accuracy drop",
        )
        written.append(str(plots_dir / "avenue_control_drops.png"))
        for filename, key, title, ylabel in (
            ("crossing_gamma_by_layer.png", "crossing_gamma", "Crossing Gamma by Layer", "gamma"),
            (
                "crossing_contribution_norm.png",
                "crossing_contribution_norm",
                "Crossing Contribution Norm by Layer",
                "norm",
            ),
        ):
            _plot_lines(
                plots_dir / filename,
                {
                    label: {
                        str(idx): float(value)
                        for idx, value in enumerate(run.get("eval", {}).get("layers", {}).get(key, []))
                        if value is not None
                    }
                    for label, run in zip(labels, runs)
                },
                title,
                "layer",
                ylabel,
            )
            written.append(str(plots_dir / filename))
        _plot_lines(
            plots_dir / "avenue_diversity_before_after_crossing.png",
            {
                label: {
                    "before": float(_mean_layer(run, "avenue_diversity_before_crossing") or 0.0),
                    "after": float(_mean_layer(run, "avenue_diversity_after_crossing") or 0.0),
                }
                for label, run in zip(labels, runs)
            },
            "Avenue Diversity Before and After Crossing",
            "crossing state",
            "diversity",
        )
        written.append(str(plots_dir / "avenue_diversity_before_after_crossing.png"))
        _plot_lines(
            plots_dir / "pairwise_avenue_cosine_before_after.png",
            {
                label: {
                    "before": float(_mean_layer(run, "pairwise_avenue_cosine_before_crossing") or 0.0),
                    "after": float(_mean_layer(run, "pairwise_avenue_cosine_after_crossing") or 0.0),
                }
                for label, run in zip(labels, runs)
            },
            "Pairwise Avenue Cosine Before and After Crossing",
            "crossing state",
            "mean cosine",
        )
        written.append(str(plots_dir / "pairwise_avenue_cosine_before_after.png"))
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
            [
                float(run.get("peak_gpu_memory_mb") or run.get("eval", {}).get("peak_gpu_memory_mb") or 0.0)
                for run in runs
            ],
            "Peak GPU Memory by Variant",
            "MB",
        )
        written.append(str(plots_dir / "gpu_memory_by_variant.png"))
    except Exception as exc:  # pragma: no cover
        written.append(f"plotting_failed: {exc}")
    return written


def _conclusion(runs: Sequence[Dict[str, Any]]) -> str:
    by_variant = {str(run.get("variant")): run for run in runs}
    required = {
        "baseline_transformer",
        "param_matched_baseline",
        "single_avenue_reference",
        "e8_best_reference",
        "shared_subspace_crossing",
        "gated_subspace_crossing",
        "superposition_crossing",
        "rotation_crossing",
        "pairwise_crossing",
        "crossing_every_2_layers",
        "crossing_every_layer",
        "scout_best_custom",
    }
    if not required.issubset(by_variant):
        return "inconclusive: not all required E10 scout variants have completed runs"
    baseline = float(by_variant["baseline_transformer"]["eval"].get("accuracy", 0.0))
    param = float(by_variant["param_matched_baseline"]["eval"].get("accuracy", 0.0))
    e8 = float(by_variant["e8_best_reference"]["eval"].get("accuracy", 0.0))
    promising: List[str] = []
    weak: List[str] = []
    for name, run in by_variant.items():
        if name not in required or name in {
            "baseline_transformer",
            "param_matched_baseline",
            "single_avenue_reference",
            "e8_best_reference",
        }:
            continue
        acc = float(run["eval"].get("accuracy", 0.0))
        gap = run.get("controls", {}).get("full_minus_max_single_avenue")
        crossing_controls = run.get("controls", {}).get("crossing_controls", {})
        mandatory_drop = _mean(
            crossing_controls.get(control, {}).get("drop")
            for control in ("crossing_zero", "crossing_shuffle", "crossing_random")
        ) or 0.0
        identity_drop = crossing_controls.get("crossing_identity", {}).get("drop", 0.0)
        diversity = _mean_layer(run, "avenue_diversity_after_crossing") or 0.0
        contribution = _mean_layer(run, "crossing_contribution_ratio") or 0.0
        if (
            acc > baseline
            and acc > param
            and acc > e8
            and gap is not None
            and float(gap) > 0.0
            and mandatory_drop > 0.0
            and float(identity_drop) > 0.0
            and diversity > 0.05
            and contribution > 1.0e-4
        ):
            promising.append(name)
        elif acc > max(baseline, param, e8):
            weak.append(name)
    if promising:
        return f"promising: {', '.join(promising)} beat the required references and passed the primary crossing-use checks"
    if weak:
        return f"weakly promising: {', '.join(weak)} beat the required references, but controls or cooperation checks are weak"
    best_crossing = max(
        (run for run in runs if str(run.get("variant")) not in {
            "baseline_transformer",
            "param_matched_baseline",
            "single_avenue_reference",
            "e8_best_reference",
        }),
        key=lambda item: float(item.get("eval", {}).get("accuracy", 0.0)),
    )
    if float(best_crossing.get("eval", {}).get("accuracy", 0.0)) <= max(baseline, param, e8):
        return "not promising: every crossing variant is at or below a required reference on this scout"
    return "inconclusive: crossing variants did not satisfy the scout criteria cleanly"


def _summary_rows(runs: Sequence[Dict[str, Any]]) -> List[str]:
    rows: List[str] = []
    for run in runs:
        controls = run.get("controls", {})
        crossing_controls = controls.get("crossing_controls", {})
        interweave_controls = controls.get("interweave_controls", {})
        pattern_controls = controls.get("pattern_controls", {})
        rows.append(
            "| "
            + " | ".join(
                [
                    str(run.get("variant")),
                    str(run.get("seed")),
                    str(run.get("parameter_count")),
                    _fmt(float(run.get("train_loss_ema")) if run.get("train_loss_ema") is not None else None),
                    _fmt(float(run.get("eval", {}).get("loss")) if run.get("eval", {}).get("loss") is not None else None),
                    _fmt(float(run.get("eval", {}).get("accuracy")) if run.get("eval", {}).get("accuracy") is not None else None),
                    _fmt(_scenario_acc(run, "in_distribution")),
                    _fmt(_scenario_acc(run, "long")),
                    _fmt(_scenario_acc(run, "high_distractor")),
                    _fmt(_scenario_acc(run, "many_overwrite")),
                    _fmt(controls.get("full_minus_max_single_avenue")),
                    _fmt(_mean_layer(run, "crossing_gamma")),
                    _fmt(_mean_layer(run, "crossing_contribution_ratio")),
                    _fmt(crossing_controls.get("crossing_zero", {}).get("drop")),
                    _fmt(crossing_controls.get("crossing_identity", {}).get("drop")),
                    _fmt(_mean_layer(run, "block_sparse_delta_ratio")),
                    _fmt(interweave_controls.get("interweave_zero", {}).get("drop")),
                    _fmt(_mean_layer(run, "pattern_attention_delta_ratio")),
                    _fmt(pattern_controls.get("pattern_zero", {}).get("drop")),
                    _fmt(run.get("peak_gpu_memory_mb") or run.get("eval", {}).get("peak_gpu_memory_mb")),
                    _fmt(run.get("train_tokens_per_sec"), 1),
                ]
            )
            + " |"
        )
    return rows


def write_report(
    root: Path = E10_ROOT,
    *,
    results_dir: Optional[Path] = None,
    plots_dir: Optional[Path] = None,
    report_path: Optional[Path] = None,
) -> Path:
    results_dir = results_dir or (root / "results")
    plots_dir = plots_dir or (root / "plots")
    report_path = report_path or (results_dir / "report.md")
    runs = _load_runs(results_dir)
    plots = write_plots(runs, plots_dir)
    lines: List[str] = ["# Experiment 10 Report", "", "## Run Scope", ""]
    if runs:
        seeds = sorted({str(run.get("seed")) for run in runs})
        scope = f"Single-seed scout (`seed={seeds[0]}`)." if len(seeds) == 1 else f"Seeds: {', '.join(seeds)}."
        lines.append(scope + " This is an exploratory architecture screen, not a proof run.")
    else:
        lines.append("No completed runs found.")
    lines.extend(
        [
            "",
            "E10 tests activation-space crossing: existing avenue activations enter a shared feature subspace, mix per token across avenues, and return by residual projection. It does not use E9 explicit junction tokens or a junction memory object.",
            "",
            "## Summary Table",
            "",
            "| Variant | Seed | Params | Train loss | Val loss | Acc | In-dist | Long | High distractor | Many overwrite | Full-max single | Gamma | Contribution ratio | Crossing zero drop | Identity drop | Interweave ratio | Interweave zero drop | Pattern ratio | Pattern zero drop | GPU MB | Train tok/s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(_summary_rows(runs))

    for title, key in (
        ("Accuracy by Task", "accuracy_by_task"),
        ("Accuracy by Sequence Length", "accuracy_by_seq_len"),
        ("Accuracy by Distractor Count", "accuracy_by_distractor_count"),
        ("Accuracy by Overwrite Depth", "accuracy_by_overwrite_depth"),
    ):
        lines.extend(["", f"## {title}", ""])
        for run in runs:
            lines.append(f"- `{run.get('variant')}` seed `{run.get('seed')}`: {run.get('eval', {}).get(key, {})}")

    lines.extend(["", "## Cooperation Diagnostics", ""])
    for run in runs:
        controls = run.get("controls", {})
        leave_one = controls.get("leave_one_avenue_out", {})
        lines.append(
            f"- `{run.get('variant')}` seed `{run.get('seed')}`: full={_fmt(controls.get('base_accuracy'))}, "
            f"max_single={_fmt(controls.get('max_single_avenue_accuracy'))}, "
            f"full_minus_max_single={_fmt(controls.get('full_minus_max_single_avenue'))}, "
            f"leave_one_drops={[item.get('drop') for item in leave_one.values()]}"
        )

    lines.extend(["", "## Crossing Diagnostics", ""])
    for run in runs:
        layers = run.get("eval", {}).get("layers", {})
        crossing_controls = run.get("controls", {}).get("crossing_controls", {})
        mix_matrix = [
            [
                _mean(layers.get(f"crossing_mix_{target}_{source}", []))
                for source in range(4)
            ]
            for target in range(4)
        ]
        source_usage = [_mean(layers.get(f"crossing_source_usage_{idx}", [])) for idx in range(4)]
        lines.append(
            f"- `{run.get('variant')}` seed `{run.get('seed')}`: gamma={layers.get('crossing_gamma', [])}, "
            f"contribution_norm={layers.get('crossing_contribution_norm', [])}, "
            f"contribution_ratio={layers.get('crossing_contribution_ratio', [])}, "
            f"controls={crossing_controls}, identity_drop={_fmt(_control_drop(run, 'crossing_controls', 'crossing_identity'))}, "
            f"diversity_before={layers.get('avenue_diversity_before_crossing', [])}, "
            f"diversity_after={layers.get('avenue_diversity_after_crossing', [])}, "
            f"pairwise_cos_before={layers.get('pairwise_avenue_cosine_before_crossing', [])}, "
            f"pairwise_cos_after={layers.get('pairwise_avenue_cosine_after_crossing', [])}, "
            f"gate_entropy={layers.get('crossing_gate_entropy', [])}, source_usage={source_usage}, mean_mix={mix_matrix}"
        )

    lines.extend(["", "## Avenue Diagnostics", ""])
    for run in runs:
        layers = run.get("eval", {}).get("layers", {})
        lines.append(
            f"- `{run.get('variant')}` seed `{run.get('seed')}`: output_norm={layers.get('avenue_output_norm', [])}, "
            f"pairwise_cosine={layers.get('pairwise_avenue_cosine', [])}, diversity={layers.get('avenue_diversity', [])}, "
            f"usage={[layers.get(f'avenue_usage_{idx}', []) for idx in range(4)]}, "
            f"avenue_controls={run.get('controls', {}).get('avenue_controls', {})}"
        )

    lines.extend(["", "## Block-Sparse Interweave Diagnostics", ""])
    for run in runs:
        layers = run.get("eval", {}).get("layers", {})
        controls = run.get("controls", {}).get("interweave_controls", {})
        if not layers.get("block_sparse_interweave_applied") and not controls:
            continue
        lines.append(
            f"- `{run.get('variant')}` seed `{run.get('seed')}`: "
            f"applied={layers.get('block_sparse_interweave_applied', [])}, "
            f"alpha_mean={layers.get('block_sparse_alpha_mean', [])}, "
            f"delta_norm={layers.get('block_sparse_delta_norm', [])}, "
            f"delta_ratio={layers.get('block_sparse_delta_ratio', [])}, "
            f"source_contribution_norm={layers.get('block_sparse_source_contribution_norm', [])}, "
            f"controls={controls}"
        )

    lines.extend(["", "## Activation Pattern Attention Diagnostics", ""])
    for run in runs:
        layers = run.get("eval", {}).get("layers", {})
        controls = run.get("controls", {}).get("pattern_controls", {})
        if not layers.get("pattern_attention_applied") and not controls:
            continue
        lines.append(
            f"- `{run.get('variant')}` seed `{run.get('seed')}`: "
            f"applied={layers.get('pattern_attention_applied', [])}, "
            f"gamma={layers.get('pattern_attention_gamma', [])}, "
            f"entropy={layers.get('pattern_attention_entropy', [])}, "
            f"delta_norm={layers.get('pattern_attention_delta_norm', [])}, "
            f"delta_ratio={layers.get('pattern_attention_delta_ratio', [])}, "
            f"controls={controls}"
        )

    lines.extend(["", "## Evidence Diagnostics", ""])
    for run in runs:
        layers = run.get("eval", {}).get("layers", {})
        lines.append(
            f"- `{run.get('variant')}` seed `{run.get('seed')}`: useful_vs_distractor_crossing_auc={layers.get('useful_vs_distractor_crossing_auc', [])}, "
            f"useful_concentration={layers.get('useful_evidence_concentration', [])}, "
            f"distractor_concentration={layers.get('distractor_concentration', [])}, "
            f"query_useful_before={layers.get('query_useful_cosine_before_crossing', [])}, "
            f"query_useful_after={layers.get('query_useful_cosine_after_crossing', [])}, "
            f"query_useful_delta={layers.get('query_useful_cosine_delta', [])}, "
            f"useful_top1={layers.get('useful_top1_recall', [])}, useful_top3={layers.get('useful_top3_recall', [])}"
        )

    lines.extend(["", "## Performance", ""])
    for run in runs:
        lines.append(
            f"- `{run.get('variant')}` seed `{run.get('seed')}`: params={run.get('parameter_count')}, "
            f"trainable={run.get('trainable_parameter_count')}, peak_gpu_memory_mb={_fmt(run.get('peak_gpu_memory_mb'))}, "
            f"train_tokens_per_sec={_fmt(run.get('train_tokens_per_sec'), 1)}, "
            f"eval_tokens_per_sec={_fmt(run.get('eval', {}).get('tokens_per_sec'), 1)}, "
            f"train_seconds={_fmt(run.get('wall_clock_train_seconds'), 1)}, "
            f"eval_seconds={_fmt(run.get('eval', {}).get('wall_clock_eval_seconds'), 1)}, "
            f"attention_kernel={run.get('attention_kernel_summary')}"
        )

    custom = next((run for run in runs if run.get("variant") == "scout_best_custom"), None)
    lines.extend(["", "## What Custom Variant Was Tried and Why", ""])
    if custom is None:
        lines.append("`scout_best_custom` has not completed yet.")
    else:
        lines.append(str(custom.get("custom_variant_rationale") or "No custom rationale was saved in the config."))

    conclusion = _conclusion(runs)
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            conclusion,
            "",
            "## Failure Analysis",
            "",
            "Crossing should be discounted when a required reference wins, full evaluation does not beat the best single avenue, crossing zero/shuffle/random or identity controls do not reduce accuracy, gamma or contribution ratios stay near zero, or crossing reduces avenue diversity into collapse. Long-context, high-distractor, and overwrite slices matter more than easy in-distribution wins.",
            "",
            "## Next Recommended Experiment",
            "",
            "Scale only crossing operators that beat the baseline, parameter-matched baseline, and E8 reference while retaining positive cooperation gaps and nonzero crossing-control drops. If this scout is weak, narrow the next pass to the strongest simple shared operator, sweep crossing placement and gamma, and add only one anti-collapse or leave-one training intervention at a time.",
            "",
            "## Plots",
            "",
        ]
    )
    lines.extend([f"- `{path}`" for path in plots])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate/report 21EYES Experiment 10 runs")
    parser.add_argument("--report", action="store_true", help="Write plots and results/report.md from completed runs")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--plots-dir", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    args = parser.parse_args()
    if args.report:
        path = write_report(
            E10_ROOT,
            results_dir=args.results_dir,
            plots_dir=args.plots_dir,
            report_path=args.report_path,
        )
        print(f"Wrote {path}")
    else:
        parser.error("use --report")


if __name__ == "__main__":
    main()
