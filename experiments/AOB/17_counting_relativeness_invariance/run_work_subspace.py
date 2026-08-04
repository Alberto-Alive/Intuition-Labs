from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TOL = 1e-8


@dataclass(frozen=True)
class WorkTask:
    task_id: str
    regime: str
    n_agents: int
    goal_rank: int
    nuisance_dim: int
    vectors: np.ndarray          # [items, ambient]
    goal_coordinates: np.ndarray  # [items, goal_rank]
    scores: np.ndarray           # [items]
    semantic_groups: np.ndarray  # [items], canonical source direction


@dataclass
class RunResult:
    seed: int
    task_id: str
    regime: str
    protocol: str
    n_agents: int
    goal_rank: int
    solved: bool
    rounds: int
    lower_bound: int
    speed_ratio_to_lower_bound: float
    useful_rank_gain: int
    duplicate_submissions: int
    irrelevant_submissions: int
    submissions: int
    duplicate_fraction: float
    irrelevant_fraction: float
    final_rank: int


def stable_int(*parts: object) -> int:
    h = hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def orthonormal_columns(x: np.ndarray, tol: float = TOL) -> np.ndarray:
    if x.size == 0:
        return np.zeros((x.shape[0], 0), dtype=float)
    q, r = np.linalg.qr(x, mode="reduced")
    diag = np.abs(np.diag(r))
    keep = diag > tol
    return q[:, keep]


def append_to_basis(q: np.ndarray, v: np.ndarray, tol: float = TOL) -> Tuple[np.ndarray, float]:
    residual = v - q @ (q.T @ v) if q.shape[1] else v.copy()
    norm = float(np.linalg.norm(residual))
    if norm <= tol:
        return q, 0.0
    residual /= norm
    return np.column_stack([q, residual]), norm


def residual_norms(vectors: np.ndarray, q: np.ndarray) -> np.ndarray:
    if q.shape[1] == 0:
        return np.linalg.norm(vectors, axis=1)
    projections = vectors @ q
    residuals = vectors - projections @ q.T
    return np.linalg.norm(residuals, axis=1)


def make_task(seed: int, task_index: int, regime: str, n_agents: int, rotate: bool = True) -> WorkTask:
    # Strong scaling uses exactly the same underlying task for every team size.
    key_agents = 1 if regime == "strong" else n_agents
    rng = np.random.default_rng(stable_int("content", seed, task_index, regime, key_agents) % (2**32))
    rotation_rng = np.random.default_rng(stable_int("rotation", seed, task_index, regime, key_agents) % (2**32))
    py_rng = random.Random(stable_int("scores", seed, task_index, regime, key_agents))

    goal_rank = 48 if regime == "strong" else 4 * n_agents
    nuisance_dim = max(24, goal_rank // 3)
    ambient_dim = goal_rank + nuisance_dim

    # Goal basis in ambient coordinates. Rotating both data and basis tests coordinate invariance.
    if rotate:
        m = rotation_rng.normal(size=(ambient_dim, ambient_dim))
        ambient_q, _ = np.linalg.qr(m)
        goal_basis = ambient_q[:, :goal_rank]
        nuisance_basis = ambient_q[:, goal_rank:]
    else:
        goal_basis = np.eye(ambient_dim)[:, :goal_rank]
        nuisance_basis = np.eye(ambient_dim)[:, goal_rank:]

    vectors: List[np.ndarray] = []
    goal_coords: List[np.ndarray] = []
    scores: List[float] = []
    groups: List[int] = []

    # Each latent work direction has several superficially different aliases.
    # They are distinct item IDs but identical after projection into goal space.
    aliases_per_direction = 6
    direction_priority = rng.uniform(0.5, 1.5, size=goal_rank)
    for i in range(goal_rank):
        e = np.zeros(goal_rank)
        e[i] = 1.0
        for alias in range(aliases_per_direction):
            nuisance = rng.normal(size=nuisance_dim)
            nuisance /= max(np.linalg.norm(nuisance), TOL)
            nuisance_scale = rng.uniform(0.35, 1.25)
            w = goal_basis @ e + nuisance_scale * (nuisance_basis @ nuisance)
            w /= max(np.linalg.norm(w), TOL)
            vectors.append(w)
            goal_coords.append(e)
            # Aliases of one latent direction cluster in score, causing exact-ID routers to collide semantically.
            scores.append(float(direction_priority[i] + 0.03 * py_rng.random() - 0.002 * alias))
            groups.append(i)

    # Add compositional work items. They can be partly new and partly redundant.
    for _ in range(2 * goal_rank):
        i, j = rng.choice(goal_rank, size=2, replace=False)
        coeff = rng.uniform(0.35, 1.0)
        g = np.zeros(goal_rank)
        g[i] = 1.0
        g[j] = coeff
        g /= np.linalg.norm(g)
        nuisance = rng.normal(size=nuisance_dim)
        nuisance /= max(np.linalg.norm(nuisance), TOL)
        w = goal_basis @ g + rng.uniform(0.2, 1.0) * (nuisance_basis @ nuisance)
        w /= max(np.linalg.norm(w), TOL)
        vectors.append(w)
        goal_coords.append(g)
        scores.append(float(max(direction_priority[i], direction_priority[j]) + 0.02 * py_rng.random()))
        groups.append(int(min(i, j)))

    return WorkTask(
        task_id=f"{regime}-a{n_agents}-s{seed}-t{task_index}",
        regime=regime,
        n_agents=n_agents,
        goal_rank=goal_rank,
        nuisance_dim=nuisance_dim,
        vectors=np.asarray(vectors, dtype=float),
        goal_coordinates=np.asarray(goal_coords, dtype=float),
        scores=np.asarray(scores, dtype=float),
        semantic_groups=np.asarray(groups, dtype=int),
    )


class Runtime:
    def __init__(self, task: WorkTask):
        self.task = task
        self.available = np.ones(len(task.vectors), dtype=bool)
        self.goal_basis = np.zeros((task.goal_rank, 0), dtype=float)
        self.raw_basis = np.zeros((task.vectors.shape[1], 0), dtype=float)
        self.round = 0
        self.useful_rank_gain = 0
        self.duplicate_submissions = 0
        self.irrelevant_submissions = 0
        self.submissions = 0

    @property
    def rank(self) -> int:
        return self.goal_basis.shape[1]

    def solved(self) -> bool:
        return self.rank >= self.task.goal_rank

    def submit(self, assignments: Sequence[int | None]) -> None:
        self.round += 1
        seen_ids: set[int] = set()
        for idx in assignments:
            if idx is None:
                continue
            self.submissions += 1
            if idx in seen_ids or not self.available[idx]:
                self.duplicate_submissions += 1
                continue
            seen_ids.add(idx)
            self.available[idx] = False

            old_rank = self.rank
            self.goal_basis, goal_novelty = append_to_basis(self.goal_basis, self.task.goal_coordinates[idx])
            self.raw_basis, _ = append_to_basis(self.raw_basis, self.task.vectors[idx])
            gain = self.rank - old_rank
            if gain:
                self.useful_rank_gain += gain
            elif goal_novelty <= TOL:
                self.duplicate_submissions += 1
            else:
                self.irrelevant_submissions += 1


def available_indices(runtime: Runtime) -> np.ndarray:
    return np.flatnonzero(runtime.available)


def assign_clone_collapse(task: WorkTask, runtime: Runtime, n_agents: int, rng: np.random.Generator) -> List[int | None]:
    avail = available_indices(runtime)
    if len(avail) == 0:
        return [None] * n_agents
    novelty = residual_norms(task.goal_coordinates[avail], runtime.goal_basis)
    utility = novelty + 1e-4 * task.scores[avail]
    idx = int(avail[np.argmax(utility)])
    return [idx] * n_agents


def assign_exact_id_router(task: WorkTask, runtime: Runtime, n_agents: int, rng: np.random.Generator) -> List[int | None]:
    avail = available_indices(runtime)
    if len(avail) == 0:
        return [None] * n_agents
    order = avail[np.argsort(task.scores[avail])[::-1]]
    chosen = [int(x) for x in order[:n_agents]]
    return chosen + [None] * (n_agents - len(chosen))


def assign_random_unique(task: WorkTask, runtime: Runtime, n_agents: int, rng: np.random.Generator) -> List[int | None]:
    avail = available_indices(runtime)
    if len(avail) == 0:
        return [None] * n_agents
    take = min(n_agents, len(avail))
    chosen = rng.choice(avail, size=take, replace=False)
    return [int(x) for x in chosen] + [None] * (n_agents - take)


def sequential_projection_selection(
    vectors: np.ndarray,
    scores: np.ndarray,
    available: np.ndarray,
    initial_basis: np.ndarray,
    n_agents: int,
) -> List[int | None]:
    provisional = initial_basis.copy()
    remaining = available.copy()
    selected: List[int] = []
    for _ in range(n_agents):
        if len(remaining) == 0:
            break
        novelty = residual_norms(vectors[remaining], provisional)
        # Novelty is primary; score only breaks near-equal cases.
        utility = novelty + 1e-4 * scores[remaining]
        pos = int(np.argmax(utility))
        if novelty[pos] <= TOL:
            break
        idx = int(remaining[pos])
        selected.append(idx)
        provisional, _ = append_to_basis(provisional, vectors[idx])
        remaining = np.delete(remaining, pos)
    return selected + [None] * (n_agents - len(selected))


def assign_raw_projection(task: WorkTask, runtime: Runtime, n_agents: int, rng: np.random.Generator) -> List[int | None]:
    return sequential_projection_selection(
        task.vectors,
        task.scores,
        available_indices(runtime),
        runtime.raw_basis,
        n_agents,
    )


def assign_goal_projection(task: WorkTask, runtime: Runtime, n_agents: int, rng: np.random.Generator) -> List[int | None]:
    return sequential_projection_selection(
        task.goal_coordinates,
        task.scores,
        available_indices(runtime),
        runtime.goal_basis,
        n_agents,
    )


ASSIGNERS: Dict[str, Callable] = {
    "clone_collapse": assign_clone_collapse,
    "exact_id_router": assign_exact_id_router,
    "random_unique_router": assign_random_unique,
    "raw_projection_router": assign_raw_projection,
    "goal_projection_router": assign_goal_projection,
}


def run_protocol(task: WorkTask, protocol: str, seed: int, max_rounds: int) -> RunResult:
    runtime = Runtime(task)
    rng = np.random.default_rng(stable_int(seed, task.task_id, protocol) % (2**32))
    while not runtime.solved() and runtime.round < max_rounds and runtime.available.any():
        assignments = ASSIGNERS[protocol](task, runtime, task.n_agents, rng)
        if all(x is None for x in assignments):
            break
        runtime.submit(assignments)

    lower_bound = math.ceil(task.goal_rank / task.n_agents)
    solved = runtime.solved()
    return RunResult(
        seed=seed,
        task_id=task.task_id,
        regime=task.regime,
        protocol=protocol,
        n_agents=task.n_agents,
        goal_rank=task.goal_rank,
        solved=solved,
        rounds=runtime.round,
        lower_bound=lower_bound,
        speed_ratio_to_lower_bound=(runtime.round / lower_bound if lower_bound else 1.0),
        useful_rank_gain=runtime.useful_rank_gain,
        duplicate_submissions=runtime.duplicate_submissions,
        irrelevant_submissions=runtime.irrelevant_submissions,
        submissions=runtime.submissions,
        duplicate_fraction=(runtime.duplicate_submissions / runtime.submissions if runtime.submissions else 0.0),
        irrelevant_fraction=(runtime.irrelevant_submissions / runtime.submissions if runtime.submissions else 0.0),
        final_rank=runtime.rank,
    )


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (regime, n_agents, protocol), g in df.groupby(["regime", "n_agents", "protocol"], sort=True):
        solved = g[g.solved]
        rows.append({
            "regime": regime,
            "n_agents": n_agents,
            "protocol": protocol,
            "tasks": len(g),
            "solve_rate": g.solved.mean(),
            "mean_rounds": solved.rounds.mean() if len(solved) else math.nan,
            "median_rounds": solved.rounds.median() if len(solved) else math.nan,
            "mean_lower_bound": g.lower_bound.mean(),
            "mean_ratio_to_lower_bound": solved.speed_ratio_to_lower_bound.mean() if len(solved) else math.nan,
            "mean_duplicate_fraction": g.duplicate_fraction.mean(),
            "mean_irrelevant_fraction": g.irrelevant_fraction.mean(),
        })
    summary = pd.DataFrame(rows)
    strong = summary[summary.regime == "strong"]
    one = strong[strong.n_agents == 1].set_index("protocol").mean_rounds.to_dict()
    summary["speedup_vs_1_agent"] = [
        one.get(r.protocol, math.nan) / r.mean_rounds
        if r.regime == "strong" and r.mean_rounds and not math.isnan(r.mean_rounds)
        else math.nan
        for r in summary.itertuples()
    ]
    summary["parallel_efficiency"] = summary.speedup_vs_1_agent / summary.n_agents
    return summary


def make_plots(summary: pd.DataFrame, out: Path) -> None:
    strong = summary[summary.regime == "strong"]
    plt.figure(figsize=(9, 6))
    for protocol, g in strong.groupby("protocol"):
        g = g.sort_values("n_agents")
        plt.plot(g.n_agents, g.speedup_vs_1_agent, marker="o", label=protocol)
    ns = sorted(strong.n_agents.unique())
    plt.plot(ns, ns, linestyle="--", label="ideal linear")
    plt.xscale("log", base=2)
    plt.yscale("log", base=2)
    plt.xlabel("Agents")
    plt.ylabel("Speedup vs 1 agent")
    plt.title("Strong scaling with abstract duplicate-work projection")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "strong_scaling_projection.png", dpi=180)
    plt.close()

    weak = summary[summary.regime == "weak"]
    plt.figure(figsize=(9, 6))
    for protocol, g in weak.groupby("protocol"):
        g = g.sort_values("n_agents")
        plt.plot(g.n_agents, g.mean_rounds, marker="o", label=protocol)
    plt.xscale("log", base=2)
    plt.xlabel("Agents (goal rank grows proportionally)")
    plt.ylabel("Mean rounds")
    plt.title("Weak scaling of collective work-span completion")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "weak_scaling_projection.png", dpi=180)
    plt.close()

    d = strong[strong.n_agents == 16].sort_values("mean_rounds")
    plt.figure(figsize=(9, 6))
    x = np.arange(len(d))
    plt.bar(x, d.mean_rounds)
    plt.xticks(x, d.protocol, rotation=30, ha="right")
    plt.ylabel("Mean rounds")
    plt.title("16-agent fixed-workload comparison")
    plt.tight_layout()
    plt.savefig(out / "protocol_comparison_16_agents.png", dpi=180)
    plt.close()


def write_report(summary: pd.DataFrame, out: Path, args: argparse.Namespace) -> None:
    gp = summary[summary.protocol == "goal_projection_router"].sort_values(["regime", "n_agents"])
    lines = [
        "# Abstract duplicate-work projection experiment",
        "",
        "## Question",
        "",
        "Can completed collective work be represented as an abstract object—a linear subspace—and can agents project candidate work against it to contribute only the goal-relevant remainder?",
        "",
        "## Operationalization",
        "",
        "Each task contains a hidden goal space of independent work dimensions. Candidate work items include multiple superficially distinct aliases of the same latent contribution, compositional mixtures, and nuisance components irrelevant to the goal. The shared completed-work object is the span of accepted goal-relevant contributions.",
        "",
        "For candidate work vector w, goal projector P_G, and completed-work projector P_D, the useful remainder is:",
        "",
        "`r = (I - P_D) P_G w`",
        "",
        "The projection router sequentially reserves candidates whose residual increases the provisional work span, making same-round assignments non-overlapping in the abstract—not merely by item ID.",
        "",
        "## Protocols",
        "",
        "- clone_collapse: all clones submit the same highest-scored item;",
        "- exact_id_router: reserves distinct item IDs but cannot recognise semantic aliases;",
        "- random_unique_router: reserves random distinct IDs;",
        "- raw_projection_router: projects in the full ambient space and can mistake nuisance novelty for useful novelty;",
        "- goal_projection_router: projects only goal-relevant work against the completed-work span.",
        "",
        "## Pre-registered predictions",
        "",
        "1. Goal-relative projection should approach the rank-theoretic lower bound of ceil(goal_rank / agents).",
        "2. Exact-ID de-duplication should underperform because different item IDs can encode the same work.",
        "3. Raw projection should underperform because irrelevant nuisance directions appear novel.",
        "4. Results for goal projection should be invariant under a global orthogonal coordinate rotation.",
        "5. Weak-scaling rounds should remain approximately constant as goal rank and agent count grow together.",
        "",
        "## Aggregated results",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Goal-projection results",
        "",
    ]
    for row in gp.itertuples():
        if row.regime == "strong":
            lines.append(
                f"- Strong scaling, {row.n_agents} agents: {row.mean_rounds:.2f} rounds, "
                f"{row.speedup_vs_1_agent:.2f}x speedup, {row.parallel_efficiency:.3f} efficiency, "
                f"{row.mean_duplicate_fraction:.4f} duplicate fraction."
            )
        else:
            lines.append(
                f"- Weak scaling, {row.n_agents} agents for rank {4*row.n_agents}: "
                f"{row.mean_rounds:.2f} rounds, {row.mean_ratio_to_lower_bound:.3f}x lower bound."
            )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "A positive result demonstrates that abstract duplicate work can be represented and removed mechanically when a useful work geometry is available. It does not demonstrate that a neural model will discover that geometry or projector without supervision.",
        "",
        "The next neural experiment should learn P_G and the work embeddings while preserving the projection-and-reservation algorithm, then test whether scaling and permutation/rotation invariance survive.",
        "",
        "## Reproduction",
        "",
        f"`python run_work_subspace.py --seeds {args.seeds} --tasks-per-seed {args.tasks_per_seed}`",
    ])
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--tasks-per-seed", type=int, default=20)
    parser.add_argument("--max-rounds", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[RunResult] = []
    for seed in range(args.seeds):
        for regime in ("strong", "weak"):
            for n_agents in (1, 2, 4, 8, 16):
                for task_index in range(args.tasks_per_seed):
                    task = make_task(seed, task_index, regime, n_agents, rotate=True)
                    for protocol in ASSIGNERS:
                        rows.append(run_protocol(task, protocol, seed, args.max_rounds))

    df = pd.DataFrame(asdict(x) for x in rows)
    df.to_csv(args.output_dir / "task_level_results.csv", index=False)
    summary = summarize(df)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    make_plots(summary, args.output_dir)
    write_report(summary, args.output_dir, args)

    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": __import__("matplotlib").__version__,
        "args": vars(args) | {"output_dir": str(args.output_dir)},
    }
    (args.output_dir / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
