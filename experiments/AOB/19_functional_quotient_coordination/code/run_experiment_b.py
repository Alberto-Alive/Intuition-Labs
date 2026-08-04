"""Experiment B: decisive real-task FQC test on the Stage 16 HotpotQA setup.

Stage 16's best variant (`scratch_continuous_distinct_multi_support`) relies on
`learned_distinct` routing: a no-replacement constraint that guarantees every
clone reads a different context window, so full evidence coverage holds by
construction. This experiment removes that guarantee and asks whether
Functional Quotient Coordination recovers (or beats) it with *learned, dynamic*
non-duplication.

Routing modes (existing Stage 16 modes are inherited unchanged):
    learned            independent clones; natural overlap (baseline 1)
    learned_distinct   Stage 16 forced no-replacement (baseline 2 / upper control)
    oracle_support     oracle distinct support coverage (upper control)
    fqc                parallel picks -> content quotient -> residual reassignment
    fqc_no_quotient    equivalence collapse disabled (must match `learned`)
    fqc_no_reassign    redundancy detected but clones are not redirected
    fqc_shuffled       quotient similarities computed on shuffled window contents
    fqc_random_reassign redundant clones redirected to a uniformly random window
    fqc_lexical        quotient similarity from token-overlap (Jaccard) only

FQC modes receive NO ground-truth decomposition: `support_windows` and
`answer_window` are never read on the fqc code paths. Equivalence is judged on
position-free window content embeddings; redirection maximizes question
relevance minus coverage of already-accepted contributions, so surplus clones
drift toward relevant-but-uncovered evidence or verification of high-value
windows rather than piling onto one window.

A duplicated-evidence stress variant tiles each context out of w/2 unique
windows so index-distinctness no longer implies content-distinctness.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.nn import functional as F

CODE_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = CODE_DIR.parent
STAGE16_PATH = (
    EXPERIMENT_ROOT.parent
    / "16_exploratory_combination_search"
    / "code"
    / "src"
    / "experiments"
    / "run_stage16_exploratory_combination_search.py"
)


def _load_stage16():
    spec = importlib.util.spec_from_file_location("stage16_combination", STAGE16_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["stage16_combination"] = module
    spec.loader.exec_module(module)
    return module


S16 = _load_stage16()

FQC_MODES = {
    "fqc",
    "fqc_no_quotient",
    "fqc_no_reassign",
    "fqc_shuffled",
    "fqc_random_reassign",
    "fqc_lexical",
}

FQC_EQUIVALENCE_TAU = 0.9
FQC_COVERAGE_MARGIN = 0.5
FQC_COVERAGE_LAMBDA = 4.0
FQC_ROUNDS = 6
# Private symmetry-breaking noise on re-picks: without it, every redundant
# clone re-picks the same argmax window at eval and duplication only halves
# per quotient round (the Experiment A design uses private randomness for the
# same reason).
FQC_REASSIGN_JITTER = 0.05


class FQCRoutedModel(S16.QuestionRoutedMultiHopQACloneModel):
    """Stage 16 clone model with functional-quotient routing modes added.

    Non-FQC routing modes defer to the unmodified Stage 16 implementation.
    """

    def _build_clone_inputs(self, input_ids, attention_mask, support_windows, answer_window):
        # Position-free window content summaries: the public objects the
        # quotient compares. Same content at different positions must compare
        # as equivalent, so position embeddings are excluded.
        context_ids = input_ids[:, self.question_segment_length :]
        context_mask = attention_mask[:, self.question_segment_length :]
        batch = input_ids.shape[0]
        tokens = self.token_embedding(context_ids) * context_mask.unsqueeze(-1).to(
            dtype=self.token_embedding.weight.dtype
        )
        window_tokens = tokens.view(batch, self.num_windows, self.window_size, -1)
        counts = (
            context_mask.view(batch, self.num_windows, self.window_size)
            .sum(dim=2, keepdim=True)
            .clamp(min=1.0)
        )
        content = window_tokens.sum(dim=2) / counts
        self._fqc_content = F.normalize(content, dim=-1)
        self._fqc_context_ids = context_ids
        return super()._build_clone_inputs(input_ids, attention_mask, support_windows, answer_window)

    def _fqc_window_similarity(self) -> torch.Tensor:
        """Pairwise window equivalence similarity [batch, w, w]."""
        mode = self.config.routing_mode
        if mode == "fqc_lexical":
            ids = self._fqc_context_ids.view(-1, self.num_windows, self.window_size)
            batch = ids.shape[0]
            sim = torch.zeros(batch, self.num_windows, self.num_windows, device=ids.device)
            for row in range(batch):
                sets = [set(ids[row, w].tolist()) - {0} for w in range(self.num_windows)]
                for a in range(self.num_windows):
                    for b in range(a, self.num_windows):
                        union = len(sets[a] | sets[b])
                        jac = len(sets[a] & sets[b]) / union if union else 0.0
                        sim[row, a, b] = jac
                        sim[row, b, a] = jac
            return sim
        content = self._fqc_content.detach()
        if mode == "fqc_shuffled":
            # Break the window<->content correspondence the quotient relies on.
            perm = torch.randperm(self.num_windows, device=content.device)
            content = content[:, perm, :]
        return torch.einsum("bwh,bvh->bwv", content, content)

    def _route_windows(self, question_hidden, question_mask, window_hidden, window_mask, support_windows, answer_window):
        if self.config.routing_mode not in FQC_MODES:
            return super()._route_windows(
                question_hidden, question_mask, window_hidden, window_mask, support_windows, answer_window
            )
        # NOTE: support_windows / answer_window are intentionally unused here.
        mode = self.config.routing_mode
        device = question_hidden.device
        batch = question_hidden.shape[0]
        n = self.config.num_clones
        w = self.num_windows

        question_summary = self._question_summary(question_hidden, question_mask)
        window_summary = self._window_summaries(window_hidden, window_mask)
        clone_ids = torch.arange(n, dtype=torch.long, device=device)
        clone_query = self.router_question_proj(question_summary).unsqueeze(1) + self.router_clone_identity(
            clone_ids
        ).view(1, n, -1)
        clone_query = F.normalize(torch.tanh(clone_query), dim=-1)
        window_key = F.normalize(torch.tanh(self.router_window_proj(window_summary)), dim=-1)
        score_logits = torch.einsum("bnh,bwh->bnw", clone_query, window_key)

        def hard_pick(logits: torch.Tensor) -> torch.Tensor:
            if self.training:
                return F.gumbel_softmax(
                    logits, tau=float(self.config.router_temperature), hard=True, dim=-1
                )
            idx = logits.argmax(dim=-1)
            return F.one_hot(idx, num_classes=w).to(dtype=logits.dtype)

        # Phase 1: fully independent parallel picks (natural overlap allowed).
        route_weights = hard_pick(score_logits)

        if mode != "fqc_no_quotient":
            sim = self._fqc_window_similarity().to(dtype=route_weights.dtype)
            for _ in range(FQC_ROUNDS):
                # Quotient: clone c's contribution is redundant when another
                # clone with higher pick-priority holds an equivalent window.
                pick_sim = torch.einsum("bnw,bwv,bmv->bnm", route_weights, sim, route_weights)
                priority = (route_weights * score_logits).sum(dim=-1)
                higher = priority.unsqueeze(1) < priority.unsqueeze(2)
                tie = priority.unsqueeze(1) == priority.unsqueeze(2)
                earlier = (
                    torch.arange(n, device=device).view(1, 1, n)
                    < torch.arange(n, device=device).view(1, n, 1)
                )
                dominated = higher | (tie & earlier)
                redundant = ((pick_sim >= FQC_EQUIVALENCE_TAU) & dominated).any(dim=2)
                if mode == "fqc_no_reassign" or not redundant.any():
                    break
                # Residual map: coverage of each window by accepted contributions.
                accepted = route_weights * (~redundant).unsqueeze(-1).to(dtype=route_weights.dtype)
                coverage = torch.einsum("bnw,bwv->bnv", accepted, sim).max(dim=1).values
                penalty = FQC_COVERAGE_LAMBDA * torch.clamp(coverage - FQC_COVERAGE_MARGIN, min=0.0)
                if mode == "fqc_random_reassign":
                    reassign_logits = torch.rand(batch, n, w, device=device) * 2.0 - 1.0
                else:
                    jitter = FQC_REASSIGN_JITTER * torch.randn(batch, n, w, device=device)
                    reassign_logits = score_logits - penalty.detach().unsqueeze(1) + jitter
                new_weights = hard_pick(reassign_logits)
                route_weights = torch.where(redundant.unsqueeze(-1), new_weights, route_weights)

        selected_windows = route_weights.argmax(dim=-1)
        return route_weights, selected_windows, score_logits


def duplicated_evidence_transform(
    rows: Sequence, window_size: int, num_windows: int, seed: int
) -> List:
    """Rebuild each context by tiling w/2 unique windows twice.

    Index-distinct routing then wastes clones on content copies; content-level
    quotienting should not. Answer and support windows are kept in the unique
    set; support/answer indices are remapped to every copy position.
    """
    rng = np.random.default_rng(seed)
    unique_count = max(1, num_windows // 2)
    out = []
    for row in rows:
        windows = [
            row.context_tokens[i * window_size : (i + 1) * window_size]
            for i in range(num_windows)
        ]
        must_keep = sorted({row.answer_window} | set(row.support_windows))
        must_keep = [wdx for wdx in must_keep if wdx is not None and 0 <= wdx < num_windows]
        if len(must_keep) > unique_count:
            continue  # cannot preserve all evidence in the unique set
        others = [i for i in range(num_windows) if i not in must_keep]
        keep = must_keep + others[: unique_count - len(must_keep)]
        tiled = (keep * ((num_windows // len(keep)) + 1))[:num_windows]
        order = rng.permutation(num_windows)
        placed = [tiled[i] for i in order]

        new_context: List[str] = []
        for src in placed:
            new_context.extend(windows[src])
        positions_of = {src: [i for i, s in enumerate(placed) if s == src] for src in keep}
        answer_positions = positions_of[row.answer_window]
        answer_offset = row.answer_start - row.answer_window * window_size
        first = min(answer_positions)
        new_start = first * window_size + answer_offset
        new_end = new_start + (row.answer_end - row.answer_start)
        new_supports = sorted(
            pos for src in set(row.support_windows) & set(keep) for pos in positions_of[src]
        )
        out.append(
            replace(
                row,
                context_tokens=new_context,
                answer_start=new_start,
                answer_end=new_end,
                answer_window=first,
                support_windows=new_supports,
            )
        )
    return out


@torch.no_grad()
def fqc_eval_extras(model, loader, device) -> Dict[str, object]:
    """Duplicate-work metrics measured on the test set (any routing mode)."""
    model.eval()
    unique_index_fracs: List[float] = []
    unique_content_fracs: List[float] = []
    content_dup_fracs: List[float] = []
    answer_hits: List[float] = []
    support_cov: List[float] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        answer_window = batch["answer_window"].to(device)
        clone_hidden, clone_mask, selected_windows, _logits = model._build_clone_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            support_windows=batch["support_windows"],
            answer_window=answer_window,
        )
        content = model._fqc_content
        sim = torch.einsum("bwh,bvh->bwv", content, content)
        n = model.config.num_clones
        for row in range(input_ids.shape[0]):
            picks = [int(v) for v in selected_windows[row].tolist()]
            unique_index_fracs.append(len(set(picks)) / n)
            # Content classes among picks: greedy clustering at the same tau.
            reps: List[int] = []
            dup_pairs = 0
            total_pairs = 0
            for pick in picks:
                if any(float(sim[row, pick, rep]) >= FQC_EQUIVALENCE_TAU for rep in reps):
                    dup_pairs += 1
                else:
                    reps.append(pick)
                total_pairs += 1
            unique_content_fracs.append(len(reps) / n)
            content_dup_fracs.append(dup_pairs / total_pairs if total_pairs else 0.0)
            aw = int(answer_window[row].item())
            if aw >= 0:
                answer_hits.append(1.0 if aw in picks else 0.0)
            support = [int(v) for v in batch["support_windows"][row]]
            if support:
                support_cov.append(len(set(support) & set(picks)) / len(set(support)))
    return {
        "mean_unique_index_fraction": float(np.mean(unique_index_fracs)),
        "mean_unique_content_fraction": float(np.mean(unique_content_fracs)),
        "mean_content_duplicate_fraction": float(np.mean(content_dup_fracs)),
        "answer_window_hit_rate": float(np.mean(answer_hits)) if answer_hits else 0.0,
        "mean_support_coverage": float(np.mean(support_cov)) if support_cov else 0.0,
    }


def build_config(args, num_clones: int, seed: int, routing_mode: str) -> "S16.Stage15cConfig":
    return S16.Stage15cConfig(
        corpus=S16.CorpusConfig(
            n_train=args.n_train,
            n_dev=args.n_dev,
            n_test=args.n_test,
            vocab_size=args.vocab_size,
            question_max_length=args.question_max_length,
            window_size=args.window_size,
            answer_max_length=args.answer_max_length,
            example_filter="multi_support",
            mask_span_length=4,
            mask_strategy="uniform",
        ),
        model=S16.ModelConfig(
            num_clones=num_clones,
            routing_mode=routing_mode,
            communication_mode="continuous",
        ),
        training=S16.TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=seed,
            device=args.device,
        ),
        curriculum=S16.CurriculumConfig(
            use_mask_pretrain=False,
            use_support_warmup=args.support_warmup_epochs > 0,
            support_warmup_epochs=args.support_warmup_epochs,
        ),
    )


def run_single(args, prepared_splits, num_clones: int, seed: int, mode: str) -> Dict[str, object]:
    config = build_config(args, num_clones, seed, mode)
    device = torch.device(args.device)
    S16._set_seed(seed)
    vocab = S16._build_vocab(prepared_splits["train"], vocab_size=config.corpus.vocab_size)
    model = FQCRoutedModel(
        vocab_size=vocab.size,
        question_max_length=config.corpus.question_max_length,
        window_size=config.corpus.window_size,
        config=config.model,
    ).to(device)

    started = time.time()
    warmup_summary = None
    if args.support_warmup_epochs > 0:
        # Stage 16's support warmup: train-time-only routing supervision from
        # gold supports (same protocol as its learned_relative variant); the
        # router still consumes no ground truth at test time.
        context_length = config.corpus.window_size * num_clones
        warm_train = S16.HotpotQADataset(
            prepared_splits["train"],
            vocab=vocab,
            question_max_length=config.corpus.question_max_length,
            context_length=context_length,
            window_size=config.corpus.window_size,
        )
        warm_dev = S16.HotpotQADataset(
            prepared_splits["dev"],
            vocab=vocab,
            question_max_length=config.corpus.question_max_length,
            context_length=context_length,
            window_size=config.corpus.window_size,
        )
        warmup = S16._run_support_warmup(
            model=model,
            train_loader=S16._make_loader(warm_train, batch_size=config.training.batch_size, shuffle=True, seed=seed),
            dev_loader=S16._make_loader(warm_dev, batch_size=config.training.batch_size, shuffle=False, seed=seed + 1),
            config=config,
            device=device,
        )
        model = warmup["model"]
        warmup_summary = {k: v for k, v in warmup.items() if k != "model"}
    result = S16.train_on_prepared_splits(prepared_splits, config, initial_model=model)
    elapsed = time.time() - started

    context_length = config.corpus.window_size * num_clones
    test_dataset = S16.HotpotQADataset(
        prepared_splits["test"],
        vocab=vocab,
        question_max_length=config.corpus.question_max_length,
        context_length=context_length,
        window_size=config.corpus.window_size,
    )
    test_loader = S16._make_loader(
        test_dataset, batch_size=config.training.batch_size, shuffle=False, seed=seed + 2
    )
    extras = fqc_eval_extras(model, test_loader, device)

    final_test = result["final_test"]
    return {
        "mode": mode if args.support_warmup_epochs == 0 else f"{mode}+warm{args.support_warmup_epochs}",
        "num_clones": num_clones,
        "seed": seed,
        "duplicated_evidence": bool(args.duplicated_evidence),
        "support_warmup": warmup_summary,
        "train_seconds": round(elapsed, 1),
        "best_epoch": result["best_epoch"],
        "test_f1": final_test["f1"],
        "test_em": final_test["exact_match"],
        "best_single_clone_f1": final_test["best_single_clone_f1"],
        "joint_beats_single": final_test["joint_beats_single"],
        "per_clone_f1": final_test["per_clone_f1"],
        "clone_cosine": final_test["mean_off_diagonal_clone_cosine"],
        "routing": final_test["routing"],
        "fqc_extras": extras,
    }


def main() -> None:
    global FQC_ROUNDS, FQC_REASSIGN_JITTER
    parser = argparse.ArgumentParser(description="Experiment B: FQC on Stage 16 HotpotQA.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--scales", default="4,8,16")
    parser.add_argument(
        "--modes",
        default="learned,learned_distinct,oracle_support,fqc,fqc_no_quotient,"
        "fqc_no_reassign,fqc_shuffled,fqc_random_reassign,fqc_lexical",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-train", type=int, default=256)
    parser.add_argument("--n-dev", type=int, default=64)
    parser.add_argument("--n-test", type=int, default=64)
    parser.add_argument("--question-max-length", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--answer-max-length", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--duplicated-evidence", action="store_true")
    parser.add_argument("--support-warmup-epochs", type=int, default=0)
    parser.add_argument("--fqc-rounds", type=int, default=FQC_ROUNDS)
    parser.add_argument("--fqc-jitter", type=float, default=FQC_REASSIGN_JITTER)
    parser.add_argument("--tag", default="screen")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_ROOT / "results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    FQC_ROUNDS = args.fqc_rounds
    FQC_REASSIGN_JITTER = args.fqc_jitter

    seeds = [int(v) for v in args.seeds.split(",")]
    scales = [int(v) for v in args.scales.split(",")]
    modes = [v.strip() for v in args.modes.split(",") if v.strip()]

    out_path = args.output_dir / f"experiment_b_runs_{args.tag}.jsonl"
    for num_clones in scales:
        for seed in seeds:
            corpus = build_config(args, num_clones, seed, "learned").corpus
            prepared_splits, stats = S16._load_hotpot_splits(corpus, num_clones=num_clones, seed=seed)
            if args.duplicated_evidence:
                prepared_splits = {
                    split: duplicated_evidence_transform(
                        rows, args.window_size, num_clones, seed=seed
                    )
                    for split, rows in prepared_splits.items()
                }
            for mode in modes:
                row = run_single(args, prepared_splits, num_clones, seed, mode)
                row["preprocessing"] = stats
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                print(
                    f"[{args.tag}] N={num_clones} seed={seed} mode={mode}: "
                    f"F1={row['test_f1']:.4f} EM={row['test_em']:.4f} "
                    f"joint_beats_single={row['joint_beats_single']} "
                    f"uniq_content={row['fqc_extras']['mean_unique_content_fraction']:.3f} "
                    f"ans_hit={row['fqc_extras']['answer_window_hit_rate']:.3f} "
                    f"({row['train_seconds']}s)",
                    flush=True,
                )


if __name__ == "__main__":
    main()
