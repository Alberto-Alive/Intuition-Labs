from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
from functools import lru_cache
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

TOL = 1e-7

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def stable_int(*parts: object) -> int:
    h = hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@lru_cache(maxsize=4096)
def orthogonal_matrix(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    return q


def append_to_basis(q: np.ndarray, v: np.ndarray, tol: float = TOL) -> Tuple[np.ndarray, float]:
    residual = v - q @ (q.T @ v) if q.shape[1] else v.copy()
    norm = float(np.linalg.norm(residual))
    if norm <= tol:
        return q, 0.0
    return np.column_stack([q, residual / norm]), norm


def residual_norms(vectors: np.ndarray, q: np.ndarray) -> np.ndarray:
    if q.shape[1] == 0:
        return np.linalg.norm(vectors, axis=1)
    projection = vectors @ q
    residual = vectors - projection @ q.T
    return np.linalg.norm(residual, axis=1)


@dataclass(frozen=True)
class Task:
    task_id: str
    split: str
    input_noise: float
    rank: int
    n_agents: int
    anchors: np.ndarray          # [rank, ambient]
    items: np.ndarray            # [items, ambient]
    targets: np.ndarray          # [items, rank], non-negative and L2 normalised
    scores: np.ndarray           # [items]


@dataclass
class CoordinationResult:
    seed: int
    task_id: str
    split: str
    input_noise: float
    protocol: str
    n_agents: int
    solved: bool
    rounds: int
    lower_bound: int
    ratio_to_lower_bound: float
    final_rank: int
    submissions: int
    duplicate_submissions: int
    duplicate_fraction: float


def make_task(
    seed: int,
    task_index: int,
    split: str,
    n_agents: int,
    rank: int = 24,
    ambient: int = 40,
    input_noise: float = 0.0,
    aliases: int = 6,
) -> Task:
    # The ID split shares the transform used in training. Each OOD task receives an unseen transform.
    if split == "id":
        transform_seed = stable_int("train-transform", seed) % (2**32)
    elif split == "ood":
        transform_seed = stable_int("ood-transform", seed, task_index) % (2**32)
    else:
        raise ValueError(split)

    rng = np.random.default_rng(stable_int("task", seed, task_index, split, input_noise) % (2**32))
    rotation = orthogonal_matrix(ambient, int(transform_seed))
    goal_basis = rotation[:, :rank]
    nuisance_basis = rotation[:, rank:]

    # Goal tokens are task-specific references. Their order defines the shared goal-relative coordinates.
    anchors = goal_basis.T.copy()
    if input_noise:
        anchors += input_noise * rng.normal(size=anchors.shape)
    anchors /= np.maximum(np.linalg.norm(anchors, axis=1, keepdims=True), TOL)

    items: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    scores: List[float] = []
    priority = rng.uniform(0.7, 1.3, size=rank)

    # Multiple surface forms of each latent work direction.
    for j in range(rank):
        target = np.zeros(rank, dtype=float)
        target[j] = 1.0
        for a in range(aliases):
            nuisance = rng.normal(size=ambient - rank)
            nuisance /= max(float(np.linalg.norm(nuisance)), TOL)
            # Small semantic contamination makes nearest-anchor mapping non-trivial.
            other = int(rng.integers(rank - 1))
            if other >= j:
                other += 1
            x = (
                goal_basis[:, j]
                + rng.uniform(0.05, 0.22) * goal_basis[:, other]
                + rng.uniform(0.25, 0.65) * (nuisance_basis @ nuisance)
            )
            if input_noise:
                x += (0.35 * input_noise) * rng.normal(size=ambient)
            x /= max(float(np.linalg.norm(x)), TOL)
            items.append(x)
            targets.append(target)
            scores.append(float(priority[j] + 0.02 * rng.random() - 0.002 * a))

    # Compositional work items cover two goal-relative directions.
    for _ in range(rank):
        i, j = rng.choice(rank, size=2, replace=False)
        coeff = float(rng.uniform(0.35, 0.9))
        target = np.zeros(rank, dtype=float)
        target[i] = 1.0
        target[j] = coeff
        target /= np.linalg.norm(target)
        nuisance = rng.normal(size=ambient - rank)
        nuisance /= max(float(np.linalg.norm(nuisance)), TOL)
        x = goal_basis @ target + rng.uniform(0.20, 0.55) * (nuisance_basis @ nuisance)
        if input_noise:
            x += (0.35 * input_noise) * rng.normal(size=ambient)
        x /= max(float(np.linalg.norm(x)), TOL)
        items.append(x)
        targets.append(target)
        scores.append(float(max(priority[i], priority[j]) + 0.02 * rng.random()))

    # Pure nuisance avenues: novel in raw coordinates but irrelevant to this goal.
    for _ in range(2 * rank):
        nuisance = rng.normal(size=ambient - rank)
        nuisance /= max(float(np.linalg.norm(nuisance)), TOL)
        x = nuisance_basis @ nuisance
        if input_noise:
            x += (0.35 * input_noise) * rng.normal(size=ambient)
        x /= max(float(np.linalg.norm(x)), TOL)
        items.append(x)
        targets.append(np.zeros(rank, dtype=float))
        scores.append(float(1.35 + 0.10 * rng.random()))

    return Task(
        task_id=f"{split}-n{input_noise:.2f}-a{n_agents}-s{seed}-t{task_index}",
        split=split,
        input_noise=input_noise,
        rank=rank,
        n_agents=n_agents,
        anchors=np.asarray(anchors, dtype=np.float32),
        items=np.asarray(items, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.float32),
        scores=np.asarray(scores, dtype=np.float32),
    )


class StaticMap(nn.Module):
    def __init__(self, ambient: int, rank: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ambient, 96), nn.ReLU(),
            nn.Linear(96, 96), nn.ReLU(),
            nn.Linear(96, rank),
        )

    def forward(self, items: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        return self.net(items)


class PooledGoalMap(nn.Module):
    def __init__(self, ambient: int, rank: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ambient * 2, 128), nn.ReLU(),
            nn.Linear(128, 96), nn.ReLU(),
            nn.Linear(96, rank),
        )

    def forward(self, items: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        pooled = anchors.mean(dim=1, keepdim=True).expand(-1, items.shape[1], -1)
        return self.net(torch.cat([items, pooled], dim=-1))


class RelationalPairMap(nn.Module):
    """Goal-relative pair comparisons without task-set context."""

    def __init__(self):
        super().__init__()
        self.pair = nn.Sequential(
            nn.Linear(2, 24), nn.Tanh(),
            nn.Linear(24, 1),
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, items: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        sim = torch.einsum("bmd,bkd->bmk", items, anchors)
        feat = torch.stack([sim, sim.square()], dim=-1)
        return self.pair(feat).squeeze(-1) * self.temperature.exp().clamp(max=30.0)


class RelationalTaskMap(nn.Module):
    """Goal-relative map that refines goal tokens from the arrangement of task items.

    Initial item-to-goal attention assigns items softly to goal directions. Those
    assignments form task-specific prototypes, which denoise/re-express the goal
    coordinates before the final work map is produced.
    """

    def __init__(self):
        super().__init__()
        self.assignment_temperature = nn.Parameter(torch.tensor(1.0))
        self.refine_logit = nn.Parameter(torch.tensor(0.0))
        self.pair = nn.Sequential(
            nn.Linear(3, 32), nn.Tanh(),
            nn.Linear(32, 1),
        )
        self.output_temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, items: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        sim0 = torch.einsum("bmd,bkd->bmk", items, anchors)
        assignment = torch.softmax(sim0 * self.assignment_temperature.exp().clamp(max=30.0), dim=-1)
        denom = assignment.sum(dim=1).clamp_min(1e-6)
        prototypes = torch.einsum("bmk,bmd->bkd", assignment, items) / denom[..., None]
        prototypes = torch.nn.functional.normalize(prototypes, dim=-1)
        mix = torch.sigmoid(self.refine_logit)
        refined = torch.nn.functional.normalize((1.0 - mix) * anchors + mix * prototypes, dim=-1)
        sim1 = torch.einsum("bmd,bkd->bmk", items, refined)
        feat = torch.stack([sim0, sim1, sim1.square()], dim=-1)
        return self.pair(feat).squeeze(-1) * self.output_temperature.exp().clamp(max=30.0)


def soft_target_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    mass = targets.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(targets, 1.0 / targets.shape[-1])
    probs = torch.where(mass > 1e-8, targets / mass.clamp_min(1e-8), uniform)
    return -(probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def task_batch(seed: int, step: int, batch_size: int, rank: int, ambient: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    noise_levels = (0.0, 0.04, 0.08, 0.12, 0.18)
    tasks = [make_task(seed, step * batch_size + i, "id", 1, rank, ambient, input_noise=noise_levels[(step + i) % len(noise_levels)]) for i in range(batch_size)]
    return (
        torch.tensor(np.stack([t.items for t in tasks])),
        torch.tensor(np.stack([t.anchors for t in tasks])),
        torch.tensor(np.stack([t.targets for t in tasks])),
    )


def train_models(seed: int, rank: int, ambient: int, steps: int, batch_size: int) -> Dict[str, nn.Module]:
    set_seed(seed)
    models: Dict[str, nn.Module] = {
        "static_map": StaticMap(ambient, rank),
        "pooled_goal_map": PooledGoalMap(ambient, rank),
        "relational_pair_map": RelationalPairMap(),
        "relational_task_map": RelationalTaskMap(),
    }
    opts = {name: torch.optim.Adam(model.parameters(), lr=2e-3) for name, model in models.items()}

    for step in range(steps):
        items, anchors, targets = task_batch(seed, step, batch_size, rank, ambient)
        for name, model in models.items():
            model.train()
            opts[name].zero_grad(set_to_none=True)
            logits = model(items, anchors)
            loss = soft_target_loss(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opts[name].step()
    for model in models.values():
        model.eval()
    return models


def sparse_embedding_from_logits(logits: np.ndarray, top_k: int = 2) -> np.ndarray:
    # Sparse work coordinates prevent tiny logit differences from creating false full rank.
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    out = np.zeros_like(probs)
    idx = np.argpartition(probs, -top_k, axis=1)[:, -top_k:]
    rows = np.arange(len(probs))[:, None]
    out[rows, idx] = probs[rows, idx]
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), TOL)
    # Uniform predictions represent no goal-relative avenue and therefore near-zero work.
    confidence = np.maximum(probs.max(axis=1) - (1.0 / probs.shape[1]), 0.0)
    return out * confidence[:, None]


def predict_embeddings(task: Task, protocol: str, models: Dict[str, nn.Module]) -> np.ndarray:
    if protocol == "oracle_previous":
        return task.targets.astype(float)
    if protocol == "raw_previous":
        return task.items.astype(float)
    if protocol in models:
        with torch.no_grad():
            items = torch.tensor(task.items[None, ...])
            anchors = torch.tensor(task.anchors[None, ...])
            logits = models[protocol](items, anchors)[0].cpu().numpy()
        return sparse_embedding_from_logits(logits, top_k=2)
    raise KeyError(protocol)


def sequential_select(embeddings: np.ndarray, scores: np.ndarray, available: np.ndarray, basis: np.ndarray, n_agents: int) -> List[int]:
    provisional = basis.copy()
    remaining = available.copy()
    selected: List[int] = []
    for _ in range(n_agents):
        if len(remaining) == 0:
            break
        novelty = residual_norms(embeddings[remaining], provisional)
        utility = novelty + 1e-4 * scores[remaining]
        pos = int(np.argmax(utility))
        if novelty[pos] <= 1e-6:
            break
        idx = int(remaining[pos])
        selected.append(idx)
        provisional, _ = append_to_basis(provisional, embeddings[idx], tol=1e-6)
        remaining = np.delete(remaining, pos)
    return selected


def run_coordination(task: Task, protocol: str, models: Dict[str, nn.Module], seed: int, max_rounds: int = 200, predicted_override: np.ndarray | None = None) -> CoordinationResult:
    predicted = predicted_override if predicted_override is not None else predict_embeddings(task, protocol, models)
    available_mask = np.ones(len(task.items), dtype=bool)
    estimated_basis = np.zeros((predicted.shape[1], 0), dtype=float)
    true_basis = np.zeros((task.rank, 0), dtype=float)
    rounds = 0
    submissions = 0
    duplicates = 0

    while true_basis.shape[1] < task.rank and rounds < max_rounds and available_mask.any():
        rounds += 1
        available = np.flatnonzero(available_mask)
        selected = sequential_select(predicted, task.scores, available, estimated_basis, task.n_agents)
        if not selected:
            # External task residual says unfinished: reopen the map by taking the most goal-relevant unused items.
            # This avoids silently treating an internally saturated map as success.
            selected = [int(i) for i in available[np.argsort(task.scores[available])[::-1][: task.n_agents]]]
        if not selected:
            break
        for idx in selected:
            available_mask[idx] = False
            submissions += 1
            estimated_basis, _ = append_to_basis(estimated_basis, predicted[idx], tol=1e-6)
            old = true_basis.shape[1]
            true_basis, _ = append_to_basis(true_basis, task.targets[idx], tol=1e-6)
            if true_basis.shape[1] == old:
                duplicates += 1

    lower = math.ceil(task.rank / task.n_agents)
    solved = true_basis.shape[1] == task.rank
    return CoordinationResult(
        seed=seed,
        task_id=task.task_id,
        split=task.split,
        input_noise=task.input_noise,
        protocol=protocol,
        n_agents=task.n_agents,
        solved=solved,
        rounds=rounds,
        lower_bound=lower,
        ratio_to_lower_bound=rounds / lower,
        final_rank=true_basis.shape[1],
        submissions=submissions,
        duplicate_submissions=duplicates,
        duplicate_fraction=duplicates / submissions if submissions else 0.0,
    )


def representation_metrics(task: Task, protocol: str, models: Dict[str, nn.Module], predicted_override: np.ndarray | None = None) -> Dict[str, float]:
    pred = predicted_override if predicted_override is not None else predict_embeddings(task, protocol, models)
    if pred.shape[1] != task.rank:
        return {"map_cosine": math.nan, "top1_accuracy": math.nan}
    relevant = np.linalg.norm(task.targets, axis=1) > 1e-8
    cosine = np.sum(pred[relevant] * task.targets[relevant], axis=1) / (
        np.maximum(np.linalg.norm(pred[relevant], axis=1), TOL) * np.maximum(np.linalg.norm(task.targets[relevant], axis=1), TOL)
    )
    alias_mask = np.count_nonzero(task.targets > 1e-8, axis=1) == 1
    acc = np.mean(np.argmax(pred[alias_mask], axis=1) == np.argmax(task.targets[alias_mask], axis=1))
    return {"map_cosine": float(np.mean(cosine)), "top1_accuracy": float(acc)}


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["split", "input_noise", "n_agents", "protocol"], as_index=False)
        .agg(
            tasks=("solved", "size"),
            solve_rate=("solved", "mean"),
            mean_rounds=("rounds", "mean"),
            median_rounds=("rounds", "median"),
            mean_ratio_to_lower_bound=("ratio_to_lower_bound", "mean"),
            mean_duplicate_fraction=("duplicate_fraction", "mean"),
            mean_map_cosine=("map_cosine", "mean"),
            mean_top1_accuracy=("top1_accuracy", "mean"),
        )
    )


def plots(summary: pd.DataFrame, out: Path) -> None:
    d = summary[(summary.split == "ood") & (summary.input_noise == 0.20) & (summary.n_agents == 16)]
    plt.figure(figsize=(9, 6))
    x = np.arange(len(d))
    plt.bar(x, d.mean_rounds)
    plt.xticks(x, d.protocol, rotation=25, ha="right")
    plt.ylabel("Mean rounds")
    plt.title("OOD goal-relative task mapping, 16 agents")
    plt.tight_layout()
    plt.savefig(out / "ood_16_agent_rounds.png", dpi=180)
    plt.close()

    d = summary[(summary.split == "ood") & (summary.n_agents == 16)]
    plt.figure(figsize=(9, 6))
    for protocol, g in d.groupby("protocol"):
        g = g.sort_values("input_noise")
        plt.plot(g.input_noise, g.solve_rate, marker="o", label=protocol)
    plt.xlabel("Input/goal-token noise")
    plt.ylabel("Solve rate")
    plt.title("OOD robustness of learned task maps")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "ood_noise_solve_rate.png", dpi=180)
    plt.close()

    d = summary[(summary.split == "ood") & (summary.input_noise == 0.20)]
    plt.figure(figsize=(9, 6))
    for protocol, g in d.groupby("protocol"):
        g = g.sort_values("n_agents")
        speedup = g[g.n_agents == 1].mean_rounds.iloc[0] / g.mean_rounds
        plt.plot(g.n_agents, speedup, marker="o", label=protocol)
    ns = sorted(d.n_agents.unique())
    plt.plot(ns, ns, linestyle="--", label="ideal linear")
    plt.xscale("log", base=2)
    plt.yscale("log", base=2)
    plt.xlabel("Agents")
    plt.ylabel("Speedup vs one agent")
    plt.title("Strong scaling on unseen task geometry")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "ood_strong_scaling.png", dpi=180)
    plt.close()


def paired_bootstrap(task_df: pd.DataFrame, a: str, b: str, split: str, noise: float, agents: int, seed: int = 0) -> Tuple[float, float, float]:
    x = task_df[(task_df.protocol == a) & (task_df.split == split) & (task_df.input_noise == noise) & (task_df.n_agents == agents)]
    y = task_df[(task_df.protocol == b) & (task_df.split == split) & (task_df.input_noise == noise) & (task_df.n_agents == agents)]
    merged = x[["task_id", "rounds"]].merge(y[["task_id", "rounds"]], on="task_id", suffixes=("_a", "_b"))
    delta = (merged.rounds_a - merged.rounds_b).to_numpy()
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(5000)])
    return float(delta.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def write_results(summary: pd.DataFrame, task_df: pd.DataFrame, out: Path, args: argparse.Namespace) -> None:
    key = summary[(summary.split == "ood") & (summary.input_noise == 0.20) & (summary.n_agents == 16)].sort_values("mean_rounds")
    delta_static = paired_bootstrap(task_df, "static_map", "relational_task_map", "ood", 0.20, 16)
    delta_pooled = paired_bootstrap(task_df, "pooled_goal_map", "relational_task_map", "ood", 0.20, 16)
    delta_pair = paired_bootstrap(task_df, "relational_pair_map", "relational_task_map", "ood", 0.20, 16)
    delta_raw = paired_bootstrap(task_df, "raw_previous", "relational_task_map", "ood", 0.20, 16)
    lines = [
        "# Goal-relative relational task-map experiment",
        "",
        "## Question",
        "",
        "Does mapping every task part relative to a shared goal and the arrangement of the other parts recover scalable division of labour on unseen task geometries better than a fixed learned work representation?",
        "",
        "## Critical comparison",
        "",
        "The prior oracle work-subspace router is a rank-theoretic upper bound and cannot be beaten in rounds. This experiment tests the missing step: inferring the work geometry from raw task tokens. Static and pooled-goal networks are trained on one task coordinate system; the relational map uses shared item-to-goal comparisons and a task-level support statistic, making it invariant to a common rotation and equivariant to goal-token order.",
        "",
        "## Key OOD condition",
        "",
        "All OOD tasks use unseen orthogonal task geometries. Input/goal-token noise is 0.20 and the team has 16 agents.",
        "",
        key.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"Static map minus relational map rounds: {delta_static[0]:.2f}, 95% paired bootstrap interval [{delta_static[1]:.2f}, {delta_static[2]:.2f}].",
        f"Pooled-goal map minus relational map rounds: {delta_pooled[0]:.2f}, 95% interval [{delta_pooled[1]:.2f}, {delta_pooled[2]:.2f}].",
        f"Context-free relational pair map minus contextual task map at noise 0.20: {delta_pair[0]:.2f}, 95% interval [{delta_pair[1]:.2f}, {delta_pair[2]:.2f}].",
        f"Previous raw-space projection minus contextual task map: {delta_raw[0]:.2f}, 95% interval [{delta_raw[1]:.2f}, {delta_raw[2]:.2f}].",
        "",
        "## Interpretation",
        "",
        "A positive result means the relational coordinate trick recovers the abstract work space on transformations never seen during training, allowing the same projection-and-reservation coordinator to keep assigning non-overlapping work. It does not show that a generic transformer will spontaneously discover this map without the relational/invariance bias.",
        "",
        "## Reproduction",
        "",
        f"`python run_taskmap_experiment.py --train-steps {args.train_steps} --tasks-per-seed {args.tasks_per_seed} --seeds {args.seeds}`",
        "",
        "## Full summary",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
    ]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tasks-per-seed", type=int, default=20)
    parser.add_argument("--rank", type=int, default=24)
    parser.add_argument("--ambient", type=int, default=72)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    protocols = ["oracle_previous", "raw_previous", "static_map", "pooled_goal_map", "relational_pair_map", "relational_task_map"]
    rows: List[Dict[str, object]] = []
    training_log: List[Dict[str, object]] = []

    for seed in range(args.seed_offset, args.seed_offset + args.seeds):
        models = train_models(seed, args.rank, args.ambient, args.train_steps, args.batch_size)
        training_log.append({"seed": seed, "models": list(models)})
        for split in ("id", "ood"):
            for noise in (0.0, 0.10, 0.20, 0.35):
                for task_index in range(args.tasks_per_seed):
                    base_task = make_task(seed, task_index, split, 1, args.rank, args.ambient, noise)
                    predictions = {p: predict_embeddings(base_task, p, models) for p in protocols}
                    metrics_by_protocol = {p: representation_metrics(base_task, p, models, predictions[p]) for p in protocols}
                    for n_agents in (1, 2, 4, 8, 16):
                        task = replace(
                            base_task,
                            n_agents=n_agents,
                            task_id=f"{split}-n{noise:.2f}-a{n_agents}-s{seed}-t{task_index}",
                        )
                        for protocol in protocols:
                            result = run_coordination(task, protocol, models, seed, predicted_override=predictions[protocol])
                            rows.append(asdict(result) | metrics_by_protocol[protocol])

    task_df = pd.DataFrame(rows)
    task_df.to_csv(args.output_dir / "task_level_results.csv", index=False)
    summary = summarize(task_df)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    plots(summary, args.output_dir)
    write_results(summary, task_df, args.output_dir, args)

    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "args": vars(args) | {"output_dir": str(args.output_dir)},
        "training_log": training_log,
    }
    (args.output_dir / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
