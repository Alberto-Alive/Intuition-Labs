"""Experiment A: controlled FQC mechanism test on the Stage 18 world.

Stage 18's coordinator (`sequential_select`) is a *centralized* projection/
reservation scheduler: within a round it assigns distinct work sequentially,
updating a provisional basis between picks. That is the controlled-world analog
of externally forced distinct allocation.

FQC removes the scheduler. Every clone proposes in parallel from the same
shared public state (the quotient basis of accepted contributions); the only
symmetry breaker is private randomness. Duplicates are quotiented after the
fact and the updated residual map redirects clones in the next round.

Protocols
  centralized baselines (Stage 18 verbatim, imported):
    oracle_reservation    reservation over ground-truth work coordinates
    taskmap_reservation   reservation over the learned relational task map
    raw_reservation       reservation over raw item vectors
  decentralized FQC:
    fqc_taskmap           FQC over relational task-map embeddings
    fqc_raw               FQC over raw item vectors (no learned map)
    fqc_oracle_embed      diagnostic ceiling: FQC over ground-truth embeddings
                          (separates map error from mechanism error; not a
                          claimable protocol because it sees the decomposition)
  controls:
    fqc_taskmap_no_quotient   no shared covered-work state; propose by priority
    fqc_taskmap_no_reassign   residual map frozen after the first update
    fqc_taskmap_shuffled      quotient state built from permuted embeddings
    random_antidup            uniform proposals over never-performed items

FQC never receives the ground-truth decomposition (except the labeled
diagnostic). Duplicates are always measured against the ground-truth basis,
exactly as in Stage 18, so rows are directly comparable.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = CODE_DIR.parent
STAGE18_PATH = (
    EXPERIMENT_ROOT.parent / "18_aob_goal_relative_taskmap_experiment" / "run_taskmap_experiment.py"
)

sys.path.insert(0, str(CODE_DIR))
from fqc_core import FunctionalQuotientState, propose_stochastic  # noqa: E402


def _load_stage18():
    spec = importlib.util.spec_from_file_location("stage18_taskmap", STAGE18_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["stage18_taskmap"] = module
    spec.loader.exec_module(module)
    return module


S18 = _load_stage18()

BASELINE_PROTOCOLS = {
    "oracle_reservation": "oracle_previous",
    "taskmap_reservation": "relational_task_map",
    "raw_reservation": "raw_previous",
}

FQC_VARIANTS: Dict[str, Dict[str, object]] = {
    "fqc_taskmap": {"embed": "relational_task_map"},
    "fqc_raw": {"embed": "raw_previous"},
    "fqc_oracle_embed": {"embed": "oracle_previous"},
    "fqc_taskmap_no_quotient": {"embed": "relational_task_map", "no_state": True},
    "fqc_taskmap_no_reassign": {"embed": "relational_task_map", "no_reassign": True},
    "fqc_taskmap_shuffled": {"embed": "relational_task_map", "shuffle": True},
    "random_antidup": {"embed": "relational_task_map", "random_policy": True},
}

NOVELTY_TOL = 1e-6


def run_fqc_coordination(
    task,
    predicted: np.ndarray,
    seed: int,
    n_agents: int,
    variant: Dict[str, object],
    beta: float = 25.0,
    view_noise: float = 0.0,
    equivalence_cos: float = 0.95,
    redundancy_theta: float = 0.3,
    max_rounds: int = 200,
) -> Dict[str, object]:
    no_state = bool(variant.get("no_state", False))
    no_reassign = bool(variant.get("no_reassign", False))
    shuffle = bool(variant.get("shuffle", False))
    random_policy = bool(variant.get("random_policy", False))

    rng_shared = np.random.default_rng(S18.stable_int("fqc-shared", seed, task.task_id) % (2**32))
    coord_embeddings = predicted.astype(float)
    if shuffle:
        # Break the item<->embedding correspondence the quotient relies on.
        perm = rng_shared.permutation(len(predicted))
        coord_embeddings = coord_embeddings[perm]

    agent_rngs = [
        np.random.default_rng(S18.stable_int("fqc-agent", seed, task.task_id, i) % (2**32))
        for i in range(n_agents)
    ]

    state = FunctionalQuotientState(
        dim=predicted.shape[1], equivalence_cos=equivalence_cos, novelty_tol=NOVELTY_TOL
    )

    available_mask = np.ones(len(task.items), dtype=bool)
    true_basis = np.zeros((task.rank, 0), dtype=float)
    rounds = 0
    submissions = 0
    duplicates = 0
    collisions = 0
    idle_agent_rounds = 0
    evaluations = 0
    outcome_counts = {"new": 0, "duplicate": 0, "update": 0, "reject": 0}

    def novelty_against(basis: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        if basis.shape[1] == 0:
            return np.linalg.norm(vectors, axis=1)
        proj = vectors @ basis
        return np.linalg.norm(vectors - proj @ basis.T, axis=1)

    def sample_intent(rng, avail: np.ndarray, basis: np.ndarray, exclude: set[int], allow_reopen: bool) -> int | None:
        """One agent's independent pick from the shared residual map."""
        nonlocal evaluations, idle_agent_rounds
        usable = np.array([i for i in avail if i not in exclude], dtype=int)
        if len(usable) == 0:
            idle_agent_rounds += 1
            return None
        evaluations += len(usable)
        if random_policy:
            return int(rng.choice(usable))
        view = coord_embeddings[usable]
        if view_noise > 0.0:
            view = view + view_noise * rng.normal(size=view.shape)
        if no_state:
            # No shared covered-work state: follow apparent priority only.
            logits = beta * (task.scores[usable] - task.scores[usable].max())
            probs = np.exp(logits)
            probs /= probs.sum()
            return int(rng.choice(usable, p=probs))
        novelty = novelty_against(basis, view)
        # An item is a viable contribution only when its unresolved component is
        # a meaningful fraction of the item itself -- the same relative-novelty
        # rule the quotient uses, so agents do not intend mostly-covered work.
        norms = np.linalg.norm(view, axis=1)
        viable_mask = novelty > np.maximum(redundancy_theta * norms, NOVELTY_TOL)
        if not viable_mask.any():
            if not allow_reopen:
                idle_agent_rounds += 1
                return None
            # Residual map says everything is resolved but the external task is
            # still unresolved: reopen by apparent priority (Stage 18's reopen
            # rule, sampled with private randomness).
            logits = beta * (task.scores[usable] - task.scores[usable].max())
            probs = np.exp(logits)
            probs /= probs.sum()
            return int(rng.choice(usable, p=probs))
        # Stand-down: when unresolved work is scarcer than the team, surplus
        # agents verify instead of piling on. Uses only public state plus the
        # agent's own view.
        viable = int(viable_mask.sum())
        if viable < n_agents and rng.random() > viable / n_agents:
            idle_agent_rounds += 1
            return None
        pos = propose_stochastic(
            rng, novelty, task.scores[usable], beta=beta, novelty_tol=NOVELTY_TOL, viable=viable_mask
        )
        if pos is None:
            idle_agent_rounds += 1
            return None
        return int(usable[pos])

    while true_basis.shape[1] < task.rank and rounds < max_rounds and available_mask.any():
        rounds += 1
        avail = np.flatnonzero(available_mask)

        # Phase 1: parallel intents from the shared state (private rng only).
        intents = [
            (idx, sample_intent(rng, avail, state.basis, set(), allow_reopen=True))
            for idx, rng in enumerate(agent_rngs)
        ]

        # Phase 2: one public intent-quotient pass. Intents equivalent to an
        # accepted class or to an earlier accepted intent this round are
        # redundant; their agents are reassigned once toward the remaining
        # residual (or idle when nothing novel remains).
        performed: List[int] = []
        redundant_agents: List[int] = []
        if no_state or random_policy:
            performed = [item for _, item in intents if item is not None]
        else:
            provisional = state.basis.copy()
            accepted_items: set[int] = set()
            order = rng_shared.permutation(len(intents))
            for k in order:
                agent_idx, item = intents[int(k)]
                if item is None:
                    continue
                vec = coord_embeddings[item]
                evaluations += provisional.shape[1] + 1
                if item in accepted_items:
                    redundant_agents.append(agent_idx)
                    continue
                # Redundant when the unresolved component of the contribution
                # (relative to accepted classes plus this round's accepted
                # intents) is a small fraction of the contribution itself.
                nov = float(novelty_against(provisional, vec[None, :])[0])
                vec_norm = float(np.linalg.norm(vec))
                if nov <= max(NOVELTY_TOL, redundancy_theta * vec_norm):
                    redundant_agents.append(agent_idx)
                    continue
                accepted_items.add(item)
                performed.append(item)
                provisional, _ = S18.append_to_basis(provisional, vec, tol=NOVELTY_TOL)

            if no_reassign:
                # Quotient detects redundancy but agents are not redirected.
                idle_agent_rounds += len(redundant_agents)
            else:
                exclude = set(accepted_items)
                for agent_idx in redundant_agents:
                    item = sample_intent(agent_rngs[agent_idx], avail, provisional, exclude, allow_reopen=False)
                    if item is None:
                        continue
                    performed.append(item)
                    exclude.add(item)
                    provisional, _ = S18.append_to_basis(
                        provisional, coord_embeddings[item], tol=NOVELTY_TOL
                    )

            if not performed:
                # The quotient rejected every intent (map falsely saturated but
                # the task is externally unresolved): perform the distinct
                # reopened intents directly rather than stalling.
                seen: set[int] = set()
                for _, item in intents:
                    if item is not None and item not in seen:
                        seen.add(item)
                        performed.append(item)

        if not performed:
            break

        # Phase 3: perform the work, quotient the results, measure truthfully.
        done_this_round: set[int] = set()
        for item in performed:
            submissions += 1
            if item in done_this_round or not available_mask[item]:
                collisions += 1
                duplicates += 1
                outcome = state.submit(coord_embeddings[item], utility=float(task.scores[item]))
                evaluations += max(1, state.class_count)
                outcome_counts[outcome] += 1
                continue
            done_this_round.add(item)
            available_mask[item] = False
            outcome = state.submit(coord_embeddings[item], utility=float(task.scores[item]))
            evaluations += max(1, state.class_count)
            outcome_counts[outcome] += 1
            old_rank = true_basis.shape[1]
            true_basis, _ = S18.append_to_basis(true_basis, task.targets[item], tol=NOVELTY_TOL)
            if true_basis.shape[1] == old_rank:
                duplicates += 1

    lower = math.ceil(task.rank / n_agents)
    agent_rounds = rounds * n_agents
    return {
        "seed": seed,
        "task_id": task.task_id,
        "split": task.split,
        "input_noise": task.input_noise,
        "n_agents": n_agents,
        "solved": bool(true_basis.shape[1] == task.rank),
        "rounds": rounds,
        "lower_bound": lower,
        "ratio_to_lower_bound": rounds / lower,
        "final_rank": int(true_basis.shape[1]),
        "submissions": submissions,
        "duplicate_submissions": duplicates,
        "duplicate_fraction": duplicates / submissions if submissions else 0.0,
        "unique_useful_rate": (submissions - duplicates) / submissions if submissions else 0.0,
        "quotient_class_count": state.class_count,
        "quotient_collapsed": outcome_counts["duplicate"] + outcome_counts["update"],
        "quotient_rejected": outcome_counts["reject"],
        "collisions": collisions,
        "idle_agent_rounds": idle_agent_rounds,
        "agent_utilization": (submissions - duplicates) / agent_rounds if agent_rounds else 0.0,
        "coordination_evaluations": evaluations,
    }


def baseline_row(task, protocol: str, models, seed: int, predicted: np.ndarray) -> Dict[str, object]:
    result = S18.run_coordination(task, BASELINE_PROTOCOLS[protocol], models, seed, predicted_override=predicted)
    row = {
        "seed": result.seed,
        "task_id": result.task_id,
        "split": result.split,
        "input_noise": result.input_noise,
        "n_agents": result.n_agents,
        "solved": result.solved,
        "rounds": result.rounds,
        "lower_bound": result.lower_bound,
        "ratio_to_lower_bound": result.ratio_to_lower_bound,
        "final_rank": result.final_rank,
        "submissions": result.submissions,
        "duplicate_submissions": result.duplicate_submissions,
        "duplicate_fraction": result.duplicate_fraction,
        "unique_useful_rate": (
            (result.submissions - result.duplicate_submissions) / result.submissions
            if result.submissions
            else 0.0
        ),
        "quotient_class_count": math.nan,
        "quotient_collapsed": math.nan,
        "quotient_rejected": math.nan,
        "collisions": math.nan,
        "idle_agent_rounds": math.nan,
        "agent_utilization": (
            (result.submissions - result.duplicate_submissions) / (result.rounds * result.n_agents)
            if result.rounds
            else 0.0
        ),
        "coordination_evaluations": math.nan,
    }
    return row


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["split", "input_noise", "n_agents", "protocol"], as_index=False)
        .agg(
            tasks=("solved", "size"),
            solve_rate=("solved", "mean"),
            mean_rounds=("rounds", "mean"),
            mean_ratio_to_lower_bound=("ratio_to_lower_bound", "mean"),
            mean_duplicate_fraction=("duplicate_fraction", "mean"),
            mean_unique_useful_rate=("unique_useful_rate", "mean"),
            mean_quotient_classes=("quotient_class_count", "mean"),
            mean_collisions=("collisions", "mean"),
            mean_agent_utilization=("agent_utilization", "mean"),
            mean_evaluations=("coordination_evaluations", "mean"),
        )
    )


def marginal_utility(df: pd.DataFrame) -> pd.DataFrame:
    """Extra solved rank per agent-round when doubling the team, from mean rounds."""
    rows = []
    for (split, noise, protocol), group in df.groupby(["split", "input_noise", "protocol"]):
        g = group.groupby("n_agents", as_index=False).agg(mean_rounds=("rounds", "mean"))
        g = g.sort_values("n_agents")
        agents = list(g.n_agents)
        for a, b in zip(agents, agents[1:]):
            ra = float(g[g.n_agents == a].mean_rounds.iloc[0])
            rb = float(g[g.n_agents == b].mean_rounds.iloc[0])
            rows.append(
                {
                    "split": split,
                    "input_noise": noise,
                    "protocol": protocol,
                    "from_agents": a,
                    "to_agents": b,
                    "speedup": ra / rb if rb else math.nan,
                    "marginal_speedup_positive": bool(ra / rb > 1.0) if rb else False,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tasks-per-seed", type=int, default=12)
    parser.add_argument("--rank", type=int, default=24)
    parser.add_argument("--ambient", type=int, default=72)
    parser.add_argument("--beta", type=float, default=25.0)
    parser.add_argument("--view-noise", type=float, default=0.0)
    parser.add_argument("--equivalence-cos", type=float, default=0.95)
    parser.add_argument("--redundancy-theta", type=float, default=0.3)
    parser.add_argument("--noises", default="0.0,0.2")
    parser.add_argument("--agents", default="1,2,4,8,16,32,64")
    parser.add_argument("--tag", default="full")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_ROOT / "results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    noises = [float(v) for v in args.noises.split(",")]
    agent_counts = [int(v) for v in args.agents.split(",")]

    rows: List[Dict[str, object]] = []
    for seed in range(args.seed_offset, args.seed_offset + args.seeds):
        models = S18.train_models(seed, args.rank, args.ambient, args.train_steps, args.batch_size)
        for split in ("id", "ood"):
            for noise in noises:
                for task_index in range(args.tasks_per_seed):
                    base_task = S18.make_task(seed, task_index, split, 1, args.rank, args.ambient, noise)
                    predictions = {
                        name: S18.predict_embeddings(base_task, name, models)
                        for name in ("oracle_previous", "raw_previous", "relational_task_map")
                    }
                    for n_agents in agent_counts:
                        task = replace(
                            base_task,
                            n_agents=n_agents,
                            task_id=f"{split}-n{noise:.2f}-a{n_agents}-s{seed}-t{task_index}",
                        )
                        for protocol in BASELINE_PROTOCOLS:
                            row = baseline_row(
                                task, protocol, models, seed, predictions[BASELINE_PROTOCOLS[protocol]]
                            )
                            row["protocol"] = protocol
                            rows.append(row)
                        for name, variant in FQC_VARIANTS.items():
                            row = run_fqc_coordination(
                                task,
                                predictions[str(variant["embed"])],
                                seed,
                                n_agents,
                                variant,
                                beta=args.beta,
                                view_noise=args.view_noise,
                                equivalence_cos=args.equivalence_cos,
                                redundancy_theta=args.redundancy_theta,
                            )
                            row["protocol"] = name
                            rows.append(row)
        print(f"seed {seed} done ({len(rows)} rows)", flush=True)

    task_df = pd.DataFrame(rows)
    task_df.to_csv(args.output_dir / f"experiment_a_task_level_{args.tag}.csv", index=False)
    summary = summarize(task_df)
    summary.to_csv(args.output_dir / f"experiment_a_summary_{args.tag}.csv", index=False)
    marginal = marginal_utility(task_df)
    marginal.to_csv(args.output_dir / f"experiment_a_marginal_{args.tag}.csv", index=False)

    payload = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "summary": summary.to_dict(orient="records"),
        "marginal_utility": marginal.to_dict(orient="records"),
    }
    (args.output_dir / f"experiment_a_results_{args.tag}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    headline = summary[(summary.split == "ood") & (summary.input_noise == 0.2)]
    if len(headline):
        print(headline.to_string(index=False))


if __name__ == "__main__":
    main()
