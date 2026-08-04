from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from src.datasets.multiview_code_patch_selection import (
    Candidate as MultiViewCandidate,
    MultiViewTaskExample,
    View,
)


STAGE6_NUM_CANDIDATES = 8
STAGE6_ROLE_NAMES = (
    "issue_and_failing_test_evidence",
    "candidate_diff_evidence",
    "retrieved_source_context",
    "dependency_callgraph_related_file_evidence",
)
STAGE6_AVENUE_NAMES = (
    "local_syntax_api_compatibility",
    "behavioral_test_failure_compatibility",
    "surrounding_code_consistency",
    "regression_security_risk_compatibility",
)
STAGE6_ATTRIBUTE_VALUE_PAIRS = (
    ("syntax_api_compact_patch", "syntax_api_large_or_wide_patch"),
    ("behavior_no_visible_pass_signal", "behavior_visible_pass_signal"),
    ("surrounding_single_file_patch", "surrounding_multi_file_patch"),
    ("risk_no_test_or_security_touch", "risk_test_or_security_touch"),
)
STAGE6_CONTROL_NAMES = (
    "none",
    "randomized_labels",
    "candidate_only",
    "issue_only",
    "patch_only",
    "context_only",
    "view_masked_candidates_visible",
    "evidence_only_no_candidates",
    "candidate_evidence_mismatch",
    "cross_task_view_bundle_shuffle",
    "hidden_states_shuffled_across_examples",
    "schema_template_only",
    "physical_order_shuffled_roles_avenues_preserved",
    "candidate_order_shuffled_with_label_remap",
)


@dataclass(frozen=True)
class PatchCandidate:
    candidate_id: str
    candidate_source_agent: str
    candidate_diff: str
    candidate_visible_test_result: str | None = None
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def patch_hash(self) -> str:
        return stable_patch_hash(self.candidate_diff)


@dataclass(frozen=True)
class PatchSelectionExample:
    id: str
    dataset_name: str
    repo: str
    issue_id: str
    issue_text: str
    failing_test_summary: str
    retrieved_contexts: Tuple[str, ...]
    candidates: Tuple[PatchCandidate, ...]
    labels_pass_fail: Tuple[int, ...]
    split: str
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def oracle_pass_at_8(self) -> bool:
        return any(int(value) == 1 for value in self.labels_pass_fail)


def stable_patch_hash(diff: str) -> str:
    normalized = "\n".join(line.rstrip() for line in str(diff).strip().splitlines())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def issue_text_hash(text: str) -> str:
    normalized = normalize_issue_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def normalize_issue_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9_]+", str(text).lower()))


def load_patch_selection_jsonl(path: str | Path) -> List[PatchSelectionExample]:
    rows: List[PatchSelectionExample] = []
    source = Path(path)
    if not source.exists():
        return rows
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(patch_selection_example_from_record(json.loads(line)))
    return rows


def write_patch_selection_jsonl(path: str | Path, examples: Sequence[PatchSelectionExample]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(json.dumps(patch_selection_example_to_record(example), sort_keys=True) for example in examples),
        encoding="utf-8",
    )


def patch_selection_example_from_record(record: Dict[str, object]) -> PatchSelectionExample:
    raw_candidates = list(record.get("candidates", []))
    if not raw_candidates and "candidate_diff" in record:
        diffs = list(record.get("candidate_diff", []))
        agents = list(record.get("candidate_source_agent", []))
        visible = list(record.get("candidate_visible_test_result", []))
        raw_candidates = [
            {
                "candidate_id": f"{record.get('id', 'example')}-cand-{index}",
                "candidate_source_agent": agents[index] if index < len(agents) else "unknown_agent",
                "candidate_diff": diff,
                "candidate_visible_test_result": visible[index] if index < len(visible) else None,
            }
            for index, diff in enumerate(diffs)
        ]
    candidates = tuple(
        PatchCandidate(
            candidate_id=str(row.get("candidate_id", f"{record.get('id', 'example')}-cand-{index}")),
            candidate_source_agent=str(row.get("candidate_source_agent", row.get("source_agent", "unknown_agent"))),
            candidate_diff=str(row.get("candidate_diff", row.get("diff", ""))),
            candidate_visible_test_result=(
                None
                if row.get("candidate_visible_test_result", row.get("visible_test_result")) is None
                else str(row.get("candidate_visible_test_result", row.get("visible_test_result")))
            ),
            metadata=dict(row.get("metadata", {})) if isinstance(row.get("metadata", {}), dict) else {},
        )
        for index, row in enumerate(raw_candidates)
        if isinstance(row, dict)
    )
    labels = tuple(int(value) for value in record.get("labels_pass_fail", []))
    return PatchSelectionExample(
        id=str(record.get("id", "")),
        dataset_name=str(record.get("dataset_name", "unknown_swe_style_dataset")),
        repo=str(record.get("repo", "")),
        issue_id=str(record.get("issue_id", "")),
        issue_text=str(record.get("issue_text", "")),
        failing_test_summary=str(record.get("failing_test_summary", "")),
        retrieved_contexts=tuple(str(value) for value in record.get("retrieved_contexts", [])),
        candidates=candidates,
        labels_pass_fail=labels,
        split=str(record.get("split", "unassigned")),
        metadata=dict(record.get("metadata", {})) if isinstance(record.get("metadata", {}), dict) else {},
    )


def patch_selection_example_to_record(example: PatchSelectionExample) -> Dict[str, object]:
    return {
        "id": example.id,
        "dataset_name": example.dataset_name,
        "repo": example.repo,
        "issue_id": example.issue_id,
        "issue_text": example.issue_text,
        "failing_test_summary": example.failing_test_summary,
        "retrieved_contexts": list(example.retrieved_contexts),
        "candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "candidate_source_agent": candidate.candidate_source_agent,
                "candidate_diff": candidate.candidate_diff,
                "candidate_visible_test_result": candidate.candidate_visible_test_result,
                "patch_hash": candidate.patch_hash,
                "metadata": dict(candidate.metadata),
            }
            for candidate in example.candidates
        ],
        "labels_pass_fail": [int(value) for value in example.labels_pass_fail],
        "split": example.split,
        "metadata": dict(example.metadata),
    }


def validate_patch_selection_examples(
    examples: Sequence[PatchSelectionExample],
    k: int = STAGE6_NUM_CANDIDATES,
    allow_gold_diagnostic: bool = False,
) -> Dict[str, object]:
    errors: List[str] = []
    duplicate_rows = duplicate_candidate_patch_hash_audit(examples)
    for example in examples:
        if len(example.candidates) != k:
            errors.append(f"{example.id}: expected {k} candidates, found {len(example.candidates)}")
        if len(example.labels_pass_fail) != k:
            errors.append(f"{example.id}: expected {k} pass/fail labels, found {len(example.labels_pass_fail)}")
        if any(int(value) not in {0, 1} for value in example.labels_pass_fail):
            errors.append(f"{example.id}: labels_pass_fail must be binary")
        if not allow_gold_diagnostic and bool(example.metadata.get("gold_patch_included", False)):
            errors.append(f"{example.id}: gold patch is included outside a diagnostic run")
    return {
        "examples_checked": len(examples),
        "candidate_count": k,
        "passes": not errors and bool(duplicate_rows.get("passes", False)),
        "errors": errors,
        "duplicate_candidate_patch_hash_audit": duplicate_rows,
    }


def patch_selection_to_multiview(example: PatchSelectionExample) -> MultiViewTaskExample:
    labels = [int(value) for value in example.labels_pass_fail]
    label = next((index for index, value in enumerate(labels) if value == 1), 0)
    candidates = tuple(
        MultiViewCandidate(
            candidate_id=candidate.candidate_id,
            text=_candidate_text(candidate),
            source_path=example.repo,
            patch_hash=candidate.patch_hash,
            attributes=_candidate_attributes(candidate),
        )
        for candidate in example.candidates
    )
    metadata = {
        "dataset_source": "stage6_swe_patch_selection_fixed_candidate_pool",
        "dataset_name": example.dataset_name,
        "repo": example.repo,
        "issue_id": example.issue_id,
        "num_candidates": len(example.candidates),
        "n_views": len(STAGE6_ROLE_NAMES),
        "candidate_representation": "candidate_diff_text",
        "generator_version": str(example.metadata.get("candidate_pool_version", "stage6_pool_v1")),
        "attribute_value_pairs": [list(pair) for pair in STAGE6_ATTRIBUTE_VALUE_PAIRS],
        "stage6_task_family": str(example.metadata.get("task_family", "unknown")),
        "stage6_visible_test_results_allowed": bool(example.metadata.get("visible_test_results_allowed", True)),
    }
    oracle_metadata = {
        **metadata,
        "labels_pass_fail": [int(value) for value in example.labels_pass_fail],
        "oracle_pass_at_8": bool(example.oracle_pass_at_8),
        "candidate_patch_hashes": [candidate.patch_hash for candidate in example.candidates],
        "candidate_source_agents": [candidate.candidate_source_agent for candidate in example.candidates],
        "edited_files": sorted({path for candidate in example.candidates for path in edited_files_from_diff(candidate.candidate_diff)}),
        "issue_text_hash": issue_text_hash(example.issue_text),
    }
    return MultiViewTaskExample(
        id=example.id,
        views=_stage6_views(example),
        candidates=candidates,
        label=int(label),
        metadata=metadata,
        oracle_metadata=oracle_metadata,
    )


def patch_selection_to_multiview_many(examples: Sequence[PatchSelectionExample]) -> List[MultiViewTaskExample]:
    return [patch_selection_to_multiview(example) for example in examples]


def label_matrix(examples: Sequence[PatchSelectionExample]) -> np.ndarray:
    if not examples:
        return np.zeros((0, STAGE6_NUM_CANDIDATES), dtype=np.float32)
    return np.asarray([[int(value) for value in example.labels_pass_fail] for example in examples], dtype=np.float32)


def first_positive_labels(examples: Sequence[PatchSelectionExample]) -> np.ndarray:
    labels = []
    for example in examples:
        labels.append(next((index for index, value in enumerate(example.labels_pass_fail) if int(value) == 1), 0))
    return np.asarray(labels, dtype=np.int64)


def randomized_label_matrix(
    examples: Sequence[PatchSelectionExample],
    seed: int,
    num_candidates: int = STAGE6_NUM_CANDIDATES,
) -> np.ndarray:
    rng = np.random.default_rng(seed + 607_001)
    rows = np.zeros((len(examples), num_candidates), dtype=np.float32)
    base = np.tile(np.arange(num_candidates, dtype=np.int64), int(np.ceil(max(1, len(examples)) / num_candidates)))[: len(examples)]
    rng.shuffle(base)
    for row, index in enumerate(base):
        rows[row, int(index)] = 1.0
    true = label_matrix(examples)
    if len(rows) > 1 and rows.shape == true.shape and np.array_equal(rows, true):
        rows = np.roll(rows, 1, axis=1)
    return rows


def apply_stage6_control(
    examples: Sequence[PatchSelectionExample],
    condition: str,
    seed: int,
) -> List[PatchSelectionExample]:
    if condition not in STAGE6_CONTROL_NAMES:
        raise ValueError(f"unknown Stage 6 control condition: {condition}")
    if condition in {"none", "hidden_states_shuffled_across_examples", "physical_order_shuffled_roles_avenues_preserved"}:
        return list(examples)
    if condition == "randomized_labels":
        randomized = randomized_label_matrix(examples, seed)
        return [
            replace(example, labels_pass_fail=tuple(int(value) for value in randomized[index].tolist()))
            for index, example in enumerate(examples)
        ]
    if condition == "candidate_order_shuffled_with_label_remap":
        rng = np.random.default_rng(seed + 901_113)
        out = []
        for example in examples:
            perm = _non_identity_permutation(len(example.candidates), rng)
            out.append(
                replace(
                    example,
                    candidates=tuple(example.candidates[int(index)] for index in perm),
                    labels_pass_fail=tuple(int(example.labels_pass_fail[int(index)]) for index in perm),
                    metadata={**example.metadata, "stage6_candidate_order_control": [int(value) for value in perm.tolist()]},
                )
            )
        return out
    if condition == "candidate_only":
        return [
            replace(
                example,
                issue_text="Issue text hidden for candidate-only control.",
                failing_test_summary="Failing-test evidence hidden for candidate-only control.",
                retrieved_contexts=("Source context hidden for candidate-only control.",),
            )
            for example in examples
        ]
    if condition == "issue_only":
        return [_hide_candidates(replace(example, retrieved_contexts=("Source context hidden for issue-only control.",)), condition) for example in examples]
    if condition == "patch_only":
        return [
            replace(
                example,
                issue_text="Issue text hidden for patch-only control.",
                failing_test_summary="Failing-test evidence hidden for patch-only control.",
                retrieved_contexts=("Source context hidden for patch-only control.",),
            )
            for example in examples
        ]
    if condition == "context_only":
        return [
            _hide_candidates(
                replace(
                    example,
                    issue_text="Issue text hidden for context-only control.",
                    failing_test_summary="Failing-test evidence hidden for context-only control.",
                ),
                condition,
            )
            for example in examples
        ]
    if condition == "view_masked_candidates_visible":
        return [
            replace(
                example,
                issue_text="Issue and test evidence masked; candidate patches remain visible.",
                failing_test_summary="Failing-test evidence masked; candidate patches remain visible.",
                retrieved_contexts=("Retrieved context masked; candidate patches remain visible.",),
            )
            for example in examples
        ]
    if condition == "evidence_only_no_candidates":
        return [_hide_candidates(example, condition) for example in examples]
    if condition == "schema_template_only":
        return [
            _hide_candidates(
                replace(
                    example,
                    issue_text="Stage 6 role 0 schema template: issue plus failing-test evidence fields only.",
                    failing_test_summary="Stage 6 role 0 schema template: traceback and failing assertion fields only.",
                    retrieved_contexts=(
                        "Stage 6 role 2 schema template: retrieved source context fields only.",
                        "Stage 6 role 3 schema template: dependency and related-file fields only.",
                    ),
                ),
                condition,
            )
            for example in examples
        ]
    if condition in {"candidate_evidence_mismatch", "cross_task_view_bundle_shuffle"}:
        rng = np.random.default_rng(seed + 503_331)
        perm = _non_identity_permutation(len(examples), rng)
        out = []
        for index, example in enumerate(examples):
            donor = examples[int(perm[index])]
            out.append(
                replace(
                    example,
                    issue_text=donor.issue_text,
                    failing_test_summary=donor.failing_test_summary,
                    retrieved_contexts=donor.retrieved_contexts,
                    metadata={**example.metadata, f"stage6_{condition}_donor_id": donor.id},
                )
            )
        return out
    raise ValueError(f"unhandled Stage 6 control condition: {condition}")


def stage6_split_leakage_audit(
    benchmark: str,
    seed: int,
    splits: Dict[str, Sequence[PatchSelectionExample]],
) -> Dict[str, object]:
    ids = {name: {example.id for example in rows} for name, rows in splits.items()}
    repo_issue = {name: {(example.repo, example.issue_id) for example in rows} for name, rows in splits.items()}
    patch_hashes = {
        name: {candidate.patch_hash for example in rows for candidate in example.candidates}
        for name, rows in splits.items()
    }
    files = {
        name: {f"{example.repo}:{path}" for example in rows for candidate in example.candidates for path in edited_files_from_diff(candidate.candidate_diff)}
        for name, rows in splits.items()
    }
    issue_hashes = {name: {issue_text_hash(example.issue_text) for example in rows} for name, rows in splits.items()}
    near_dupes = _near_duplicate_issue_overlaps(splits)
    return {
        "benchmark": benchmark,
        "seed": int(seed),
        "type": "stage6_swe_split_leakage",
        "train_dev_id_overlap": len(ids.get("train", set()) & ids.get("dev", set())),
        "train_test_id_overlap": len(ids.get("train", set()) & ids.get("test", set())),
        "dev_test_id_overlap": len(ids.get("dev", set()) & ids.get("test", set())),
        "train_dev_repo_issue_overlap": len(repo_issue.get("train", set()) & repo_issue.get("dev", set())),
        "train_test_repo_issue_overlap": len(repo_issue.get("train", set()) & repo_issue.get("test", set())),
        "dev_test_repo_issue_overlap": len(repo_issue.get("dev", set()) & repo_issue.get("test", set())),
        "train_dev_candidate_hash_overlap": len(patch_hashes.get("train", set()) & patch_hashes.get("dev", set())),
        "train_test_candidate_hash_overlap": len(patch_hashes.get("train", set()) & patch_hashes.get("test", set())),
        "dev_test_candidate_hash_overlap": len(patch_hashes.get("dev", set()) & patch_hashes.get("test", set())),
        "train_dev_file_path_overlap": len(files.get("train", set()) & files.get("dev", set())),
        "train_test_file_path_overlap": len(files.get("train", set()) & files.get("test", set())),
        "dev_test_file_path_overlap": len(files.get("dev", set()) & files.get("test", set())),
        "train_dev_issue_text_hash_overlap": len(issue_hashes.get("train", set()) & issue_hashes.get("dev", set())),
        "train_test_issue_text_hash_overlap": len(issue_hashes.get("train", set()) & issue_hashes.get("test", set())),
        "dev_test_issue_text_hash_overlap": len(issue_hashes.get("dev", set()) & issue_hashes.get("test", set())),
        "near_duplicate_issue_text_overlaps": near_dupes,
        "candidate_patch_hash_leakage_audit_passes": not (
            patch_hashes.get("train", set()) & patch_hashes.get("dev", set())
            or patch_hashes.get("train", set()) & patch_hashes.get("test", set())
            or patch_hashes.get("dev", set()) & patch_hashes.get("test", set())
        ),
        "passes": not (
            ids.get("train", set()) & ids.get("dev", set())
            or ids.get("train", set()) & ids.get("test", set())
            or ids.get("dev", set()) & ids.get("test", set())
            or repo_issue.get("train", set()) & repo_issue.get("dev", set())
            or repo_issue.get("train", set()) & repo_issue.get("test", set())
            or repo_issue.get("dev", set()) & repo_issue.get("test", set())
            or patch_hashes.get("train", set()) & patch_hashes.get("dev", set())
            or patch_hashes.get("train", set()) & patch_hashes.get("test", set())
            or patch_hashes.get("dev", set()) & patch_hashes.get("test", set())
            or issue_hashes.get("train", set()) & issue_hashes.get("dev", set())
            or issue_hashes.get("train", set()) & issue_hashes.get("test", set())
            or issue_hashes.get("dev", set()) & issue_hashes.get("test", set())
            or near_dupes
        ),
    }


def duplicate_candidate_patch_hash_audit(examples: Sequence[PatchSelectionExample]) -> Dict[str, object]:
    rows = []
    for example in examples:
        hashes = [candidate.patch_hash for candidate in example.candidates]
        duplicates = sorted({value for value in hashes if hashes.count(value) > 1})
        if duplicates:
            rows.append({"id": example.id, "duplicate_patch_hashes": duplicates})
    return {"examples_checked": len(examples), "duplicates": rows, "passes": not rows}


def stage6_output_leakage_audit(
    benchmark: str,
    seed: int,
    examples: Sequence[PatchSelectionExample],
) -> Dict[str, object]:
    forbidden = (
        r"\blabels_pass_fail\b",
        r"\boracle[_ -]?pass\b",
        r"\bhidden[_ -]?label\b",
        r"\bbenchmark[_ -]?pass\b",
        r"\bgold[_ -]?patch\b",
        r"\bselected[_ -]?label\b",
        r"\bpass_fail_label\b",
    )
    visible_texts: List[str] = []
    for example in examples:
        visible_texts.extend(view.text for view in _stage6_views(example))
        visible_texts.extend(_candidate_text(candidate) for candidate in example.candidates)
    hits = []
    for pattern in forbidden:
        regex = re.compile(pattern, flags=re.IGNORECASE)
        if any(regex.search(text) for text in visible_texts):
            hits.append(pattern)
    return {
        "benchmark": benchmark,
        "seed": int(seed),
        "type": "stage6_output_leakage",
        "examples_checked": len(examples),
        "visible_text_or_candidate_forbidden_pattern_hits": hits,
        "hidden_pass_fail_labels_excluded_from_visible_text": not hits,
        "passes": not hits,
    }


def stage6_dataset_summary(splits: Dict[str, Sequence[PatchSelectionExample]]) -> Dict[str, object]:
    all_examples = [example for rows in splits.values() for example in rows]
    oracle = [example.oracle_pass_at_8 for example in all_examples]
    return {
        "dataset_names": sorted({example.dataset_name for example in all_examples}),
        "split_sizes": {split: len(rows) for split, rows in splits.items()},
        "num_candidates": STAGE6_NUM_CANDIDATES,
        "num_roles": len(STAGE6_ROLE_NAMES),
        "num_avenues": len(STAGE6_AVENUE_NAMES),
        "repos": sorted({example.repo for example in all_examples}),
        "oracle_pass_at_8": float(np.mean(oracle)) if oracle else 0.0,
        "oracle_empty_tasks": int(sum(not value for value in oracle)),
        "candidate_source_agents": sorted({candidate.candidate_source_agent for example in all_examples for candidate in example.candidates}),
        "publishable_proof_dataset": bool(
            all_examples and all(bool(example.metadata.get("publishable_proof_dataset", False)) for example in all_examples)
        ),
        "selector_only_no_patch_generation_claim": True,
    }


def edited_files_from_diff(diff: str) -> List[str]:
    files = []
    for line in str(diff).splitlines():
        if line.startswith("+++ b/"):
            files.append(line[6:].strip())
        elif line.startswith("--- a/"):
            files.append(line[6:].strip())
        elif line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4 and parts[3].startswith("b/"):
                files.append(parts[3][2:])
    return sorted({path for path in files if path and path != "/dev/null"})


def _stage6_views(example: PatchSelectionExample) -> Tuple[View, ...]:
    dependency = str(
        example.metadata.get(
            "dependency_callgraph_related_file_evidence",
            example.metadata.get("dependency_context", "No dependency or callgraph artifact supplied."),
        )
    )
    diff_summary = _candidate_diff_inventory(example)
    contexts = "\n\n".join(example.retrieved_contexts) if example.retrieved_contexts else "No retrieved source context supplied."
    rows = (
        (
            "Role 0: issue plus failing test / traceback evidence.\n"
            f"Repository: {example.repo}\nIssue: {example.issue_text}\nFailing test summary:\n{example.failing_test_summary}"
        ),
        (
            "Role 1: candidate diff evidence.\n"
            "Patch bodies are available in the candidate block. Non-oracle candidate inventory:\n"
            f"{diff_summary}"
        ),
        (
            "Role 2: retrieved source context around edited files.\n"
            f"{contexts}"
        ),
        (
            "Role 3: dependency, callgraph, and related-file evidence.\n"
            f"{dependency}"
        ),
    )
    return tuple(
        View(
            role=STAGE6_ROLE_NAMES[index],
            text=text,
            source_path=example.repo,
            source_type=f"stage6_role_{index}",
            allowed_visibility="agent_private_partial_view",
        )
        for index, text in enumerate(rows)
    )


def _candidate_text(candidate: PatchCandidate) -> str:
    visible = candidate.candidate_visible_test_result
    visible_text = "not supplied" if visible is None else str(visible)
    return (
        f"source_agent: {candidate.candidate_source_agent}\n"
        f"visible_test_result: {visible_text}\n"
        "diff:\n"
        f"{candidate.candidate_diff}"
    )


def _candidate_attributes(candidate: PatchCandidate) -> Tuple[str, str, str, str]:
    diff = str(candidate.candidate_diff)
    files = edited_files_from_diff(diff)
    added = sum(1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++"))
    deleted = sum(1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---"))
    visible = str(candidate.candidate_visible_test_result or "").lower()
    touches_tests = any("test" in path.lower() for path in files) or re.search(r"\b(assert|pytest|unittest)\b", diff, re.IGNORECASE)
    touches_security = re.search(r"\b(auth|token|password|permission|encrypt|decrypt|sql|xss|csrf|path traversal)\b", diff, re.IGNORECASE)
    return (
        STAGE6_ATTRIBUTE_VALUE_PAIRS[0][1] if added + deleted > 40 else STAGE6_ATTRIBUTE_VALUE_PAIRS[0][0],
        STAGE6_ATTRIBUTE_VALUE_PAIRS[1][1] if "pass" in visible and "fail" not in visible else STAGE6_ATTRIBUTE_VALUE_PAIRS[1][0],
        STAGE6_ATTRIBUTE_VALUE_PAIRS[2][1] if len(files) > 1 else STAGE6_ATTRIBUTE_VALUE_PAIRS[2][0],
        STAGE6_ATTRIBUTE_VALUE_PAIRS[3][1] if touches_tests or touches_security else STAGE6_ATTRIBUTE_VALUE_PAIRS[3][0],
    )


def _candidate_diff_inventory(example: PatchSelectionExample) -> str:
    rows = []
    for index, candidate in enumerate(example.candidates):
        files = ", ".join(edited_files_from_diff(candidate.candidate_diff)[:6]) or "unknown files"
        added = sum(1 for line in candidate.candidate_diff.splitlines() if line.startswith("+") and not line.startswith("+++"))
        deleted = sum(1 for line in candidate.candidate_diff.splitlines() if line.startswith("-") and not line.startswith("---"))
        rows.append(
            f"candidate {index}: source_agent={candidate.candidate_source_agent}; files={files}; added={added}; deleted={deleted}; "
            f"visible_test_result={candidate.candidate_visible_test_result or 'not supplied'}"
        )
    return "\n".join(rows)


def _hide_candidates(example: PatchSelectionExample, condition: str) -> PatchSelectionExample:
    candidates = tuple(
        replace(
            candidate,
            candidate_diff=f"Patch candidate hidden for {condition} control.",
            candidate_visible_test_result=None,
            metadata={**candidate.metadata, "stage6_hidden_for_control": condition},
        )
        for candidate in example.candidates
    )
    return replace(example, candidates=candidates)


def _near_duplicate_issue_overlaps(splits: Dict[str, Sequence[PatchSelectionExample]]) -> List[Dict[str, object]]:
    names = sorted(splits)
    out = []
    token_sets = {
        name: [(example.id, set(normalize_issue_text(example.issue_text).split())) for example in rows]
        for name, rows in splits.items()
    }
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            for left_id, left_tokens in token_sets[left]:
                for right_id, right_tokens in token_sets[right]:
                    denom = max(1, len(left_tokens | right_tokens))
                    score = len(left_tokens & right_tokens) / float(denom)
                    if score >= 0.92 and left_tokens and right_tokens:
                        out.append({"left_split": left, "right_split": right, "left_id": left_id, "right_id": right_id, "jaccard": score})
    return out


def _non_identity_permutation(n: int, rng: np.random.Generator) -> np.ndarray:
    perm = rng.permutation(n)
    if n > 1 and np.array_equal(perm, np.arange(n)):
        perm = np.roll(perm, 1)
    return perm.astype(np.int64, copy=False)
