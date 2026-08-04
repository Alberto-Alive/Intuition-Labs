from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from src.datasets.swe_patch_selection_dataset import (
    PatchCandidate,
    PatchSelectionExample,
    STAGE6_NUM_CANDIDATES,
    load_patch_selection_jsonl,
    validate_patch_selection_examples,
    write_patch_selection_jsonl,
)


DEFAULT_OUTPUT = Path("results/stage6_candidate_pools.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 6 SWE-style fixed candidate pools.")
    parser.add_argument("--input-jsonl", default=None, help="Existing SWE-style candidate-pool JSONL to normalize.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--synthetic-smoke", action="store_true", help="Create a clearly marked local synthetic smoke pool.")
    parser.add_argument("--n-tasks", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-gold-diagnostic", action="store_true")
    args = parser.parse_args()

    if args.input_jsonl:
        examples = load_patch_selection_jsonl(args.input_jsonl)
    elif args.synthetic_smoke:
        examples = build_synthetic_stage6_candidate_pool(n_tasks=args.n_tasks, seed=args.seed)
    else:
        raise SystemExit("provide --input-jsonl or --synthetic-smoke")
    audit = validate_patch_selection_examples(examples, allow_gold_diagnostic=bool(args.allow_gold_diagnostic))
    if not bool(audit["passes"]):
        raise SystemExit(json.dumps(audit, indent=2, sort_keys=True))
    write_patch_selection_jsonl(args.output, examples)
    print(f"stage6 candidate pool builder: wrote {len(examples)} examples to {args.output}")


def build_synthetic_stage6_candidate_pool(
    n_tasks: int,
    seed: int,
    dataset_name: str = "synthetic_swe_style_stage6_smoke",
) -> List[PatchSelectionExample]:
    """Create deterministic SWE-style pools for pipeline tests, not for claims."""

    rng = np.random.default_rng(seed + 600_006)
    examples: List[PatchSelectionExample] = []
    task_families = ("api_contract", "edge_case", "state_regression", "dependency_update")
    repos = ("synthetic/repo_alpha", "synthetic/repo_beta", "synthetic/repo_gamma", "synthetic/repo_delta")
    for index in range(int(n_tasks)):
        family = task_families[index % len(task_families)]
        repo = repos[index % len(repos)]
        split = _split_for_index(index, n_tasks)
        issue_id = f"SYN-{seed}-{index:04d}"
        required_token = f"mode_{(index * 3 + seed) % 7}"
        file_stem = f"module_{index % 11}"
        issue_text = (
            f"{family} failure in {file_stem}: the resolver must preserve {required_token} "
            "when applying the issue-resolution patch. Choose a patch, do not generate one."
        )
        failing = (
            f"pytest synthetic_tests/test_{file_stem}.py::test_{family}_{required_token} fails with "
            f"AssertionError: expected {required_token} to be returned after the edge case."
        )
        contexts = (
            f"def resolve(value, mode):\n    # surrounding code requires mode token {required_token}\n    return value\n",
            f"related callgraph: handler -> resolve -> validator; validator accepts {required_token}.",
        )
        pass_slots = _pass_slots(index, rng)
        candidates = []
        labels = []
        for candidate_index in range(STAGE6_NUM_CANDIDATES):
            generator = f"generator_{candidate_index // 2}"
            sample = candidate_index % 2
            passes = candidate_index in pass_slots
            token = required_token if passes else f"mode_{(index + candidate_index + seed + 1) % 7}"
            visible = _visible_result_for_candidate(passes, candidate_index, index)
            diff = _synthetic_diff(
                file_stem=file_stem,
                family=family,
                token=token,
                task_index=index,
                candidate_index=candidate_index,
                sample=sample,
                include_test=(candidate_index + index) % 3 == 0,
                wide_patch=(candidate_index + seed) % 5 == 0,
            )
            candidates.append(
                PatchCandidate(
                    candidate_id=f"{issue_id}-cand-{candidate_index}",
                    candidate_source_agent=generator,
                    candidate_diff=diff,
                    candidate_visible_test_result=visible,
                    metadata={"synthetic_pass_source": "not_model_visible", "sample_index": sample},
                )
            )
            labels.append(1 if passes else 0)
        examples.append(
            PatchSelectionExample(
                id=f"{dataset_name}-{seed}-{index:04d}",
                dataset_name=dataset_name,
                repo=repo,
                issue_id=issue_id,
                issue_text=issue_text,
                failing_test_summary=failing,
                retrieved_contexts=contexts,
                candidates=tuple(candidates),
                labels_pass_fail=tuple(labels),
                split=split,
                metadata={
                    "task_family": family,
                    "candidate_pool_version": "synthetic_stage6_smoke_v1",
                    "visible_test_results_allowed": True,
                    "gold_patch_included": False,
                    "candidate_pool_source": "deterministic_local_synthetic_smoke",
                    "publishable_claim_allowed": False,
                    "dependency_callgraph_related_file_evidence": (
                        f"{file_stem}.resolve is called by service_{index % 5}; related validator expects {required_token}."
                    ),
                },
            )
        )
    return examples


def load_or_build_candidate_pool(
    candidate_pool_path: Path,
    synthetic_if_missing: bool,
    n_tasks: int,
    seed: int,
) -> List[PatchSelectionExample]:
    if candidate_pool_path.exists() and candidate_pool_path.stat().st_size > 0:
        return load_patch_selection_jsonl(candidate_pool_path)
    if not synthetic_if_missing:
        return []
    examples = build_synthetic_stage6_candidate_pool(n_tasks=n_tasks, seed=seed)
    write_patch_selection_jsonl(candidate_pool_path, examples)
    return examples


def _split_for_index(index: int, n_tasks: int) -> str:
    train_cut = max(1, int(round(n_tasks * 0.6)))
    dev_cut = max(train_cut + 1, int(round(n_tasks * 0.8)))
    if index < train_cut:
        return "train"
    if index < dev_cut:
        return "dev"
    return "test"


def _pass_slots(index: int, rng: np.random.Generator) -> Sequence[int]:
    if index % 10 == 0:
        return ()
    first = int((index * 5 + 1) % STAGE6_NUM_CANDIDATES)
    if index % 6 == 0:
        second = int((first + 3 + int(rng.integers(0, 2))) % STAGE6_NUM_CANDIDATES)
        if second != first:
            return (first, second)
    return (first,)


def _visible_result_for_candidate(passes: bool, candidate_index: int, task_index: int) -> str:
    if passes and (candidate_index + task_index) % 4 == 0:
        return "visible_subset_passed"
    if not passes and (candidate_index + task_index) % 5 == 0:
        return "visible_subset_failed"
    return "not_run"


def _synthetic_diff(
    file_stem: str,
    family: str,
    token: str,
    task_index: int,
    candidate_index: int,
    sample: int,
    include_test: bool,
    wide_patch: bool,
) -> str:
    extra = ""
    if wide_patch:
        extra = "\n+    audit_log = f'candidate_{candidate_index}_{family}'\n+    _ = audit_log.lower()\n"
    test_block = ""
    if include_test:
        test_block = (
            f"\ndiff --git a/tests/test_{file_stem}.py b/tests/test_{file_stem}.py\n"
            f"--- a/tests/test_{file_stem}.py\n"
            f"+++ b/tests/test_{file_stem}.py\n"
            "@@\n"
            f"+def test_candidate_{candidate_index}_{sample}():\n"
            f"+    assert '{token}'.startswith('mode_')\n"
        )
    return (
        f"diff --git a/src/{file_stem}.py b/src/{file_stem}.py\n"
        f"--- a/src/{file_stem}.py\n"
        f"+++ b/src/{file_stem}.py\n"
        "@@\n"
        " def resolve(value, mode):\n"
        f"-    return value\n"
        f"+    selected_mode = '{token}'\n"
        f"+    candidate_marker = 'task_{task_index}_candidate_{candidate_index}_{sample}'\n"
        f"+    _ = candidate_marker\n"
        f"+    if mode == selected_mode:\n"
        f"+        return selected_mode\n"
        f"+    return value\n"
        f"{extra}"
        f"{test_block}"
    )


if __name__ == "__main__":
    main()
