from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from src.datasets.multiview_code_patch_selection import Candidate as MultiViewCandidate
from src.datasets.multiview_code_patch_selection import MultiViewTaskExample, View


STAGE7_NUM_CANDIDATES = 8
STAGE7_ROLE_NAMES = (
    "user_goal_and_conversation_evidence",
    "policy_rule_evidence",
    "candidate_trajectory_evidence",
    "tool_state_evidence",
)
STAGE7_AVENUE_NAMES = (
    "goal_completion_compatibility",
    "policy_compliance",
    "tool_call_correctness_and_argument_consistency",
    "final_response_state_consistency_and_risk",
)
STAGE7_ATTRIBUTE_VALUE_PAIRS = (
    ("tool_count_short", "tool_count_long"),
    ("no_repeated_tool_calls", "has_repeated_tool_calls"),
    ("no_visible_tool_errors", "has_visible_tool_errors"),
    ("final_response_does_not_claim_completion", "final_response_claims_completion"),
)
STAGE7_CONTROL_NAMES = (
    "none",
    "randomized_labels",
    "candidate_order_shuffled_with_label_remap",
    "physical_order_shuffled_roles_avenues_preserved",
    "candidate_only",
    "user_goal_only",
    "policy_only",
    "trajectory_only",
    "tool_observations_only",
    "evidence_only_no_candidate",
    "candidate_evidence_mismatch",
    "cross_task_user_goal_shuffle",
    "cross_task_policy_shuffle",
    "cross_task_tool_observation_shuffle",
    "hidden_states_shuffled_across_examples",
    "schema_template_only",
    "generator_identity_only",
)

FORBIDDEN_SELECTOR_VISIBLE_PATTERNS = (
    r"\bofficial[_ -]?success\b",
    r"\bofficial[_ -]?eval",
    r"\bevaluator[_ -]?verdict\b",
    r"\bfinal[_ -]?verdict\b",
    r"\bpass[_ -]?fail[_ -]?label\b",
    r"\bsuccess[_ -]?label\b",
    r"\boracle[_ -]?pass\b",
    r"\bhidden[_ -]?goal[_ -]?state\b",
    r"\bpost[_ -]?hoc\b",
    r"\bdatabase[_ -]?diff\b",
    r"\bcandidate[_ -]?rank\b",
    r"\bgenerator[_ -]?name\b",
    r"\bgenerator[_ -]?config\b",
)


@dataclass(frozen=True)
class SelectorVisibleTrajectory:
    user_messages: Tuple[str, ...] = ()
    assistant_messages: Tuple[str, ...] = ()
    tool_calls: Tuple[Dict[str, object], ...] = ()
    tool_observations_sanitized: Tuple[str, ...] = ()
    final_response: str = ""

    def to_record(self) -> Dict[str, object]:
        return {
            "user_messages": list(self.user_messages),
            "assistant_messages": list(self.assistant_messages),
            "tool_calls": [dict(row) for row in self.tool_calls],
            "tool_observations_sanitized": list(self.tool_observations_sanitized),
            "final_response": self.final_response,
        }

    @classmethod
    def from_record(cls, record: Dict[str, object] | None) -> "SelectorVisibleTrajectory":
        row = record if isinstance(record, dict) else {}
        return cls(
            user_messages=tuple(str(value) for value in row.get("user_messages", []) if value is not None),
            assistant_messages=tuple(str(value) for value in row.get("assistant_messages", []) if value is not None),
            tool_calls=tuple(_safe_dict(value) for value in row.get("tool_calls", []) if isinstance(value, dict)),
            tool_observations_sanitized=tuple(
                str(value) for value in row.get("tool_observations_sanitized", []) if value is not None
            ),
            final_response=str(row.get("final_response", "")),
        )


@dataclass(frozen=True)
class TauBenchTrajectoryCandidate:
    task_id: str
    domain: str
    candidate_id: str
    candidate_order_index: int
    generator_name: str
    generator_config: str
    seed: int
    raw_trajectory_path: str
    selector_visible_trajectory: SelectorVisibleTrajectory
    official_success: bool | None
    official_eval_metadata_path: str
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def trajectory_hash(self) -> str:
        return stable_trajectory_hash(self.selector_visible_trajectory.to_record())


@dataclass(frozen=True)
class TauBenchTrajectorySelectionExample:
    id: str
    task_id: str
    domain: str
    candidates: Tuple[TauBenchTrajectoryCandidate, ...]
    split: str
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def labels_pass_fail(self) -> Tuple[int, ...]:
        return tuple(1 if candidate.official_success is True else 0 for candidate in self.candidates)

    @property
    def oracle_pass_at_8(self) -> bool:
        return any(self.labels_pass_fail)


def stable_trajectory_hash(selector_visible_trajectory: Dict[str, object]) -> str:
    text = json.dumps(selector_visible_trajectory, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def task_text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9_]+", str(text).lower()))


def taubench_candidate_from_record(record: Dict[str, object]) -> TauBenchTrajectoryCandidate:
    success = record.get("official_success", None)
    if success is None:
        parsed_success: bool | None = None
    elif isinstance(success, bool):
        parsed_success = bool(success)
    else:
        parsed_success = str(success).strip().lower() in {"1", "true", "yes", "success", "passed"}
    return TauBenchTrajectoryCandidate(
        task_id=str(record.get("task_id", "")),
        domain=str(record.get("domain", "")),
        candidate_id=str(record.get("candidate_id", "")),
        candidate_order_index=int(record.get("candidate_order_index", 0)),
        generator_name=str(record.get("generator_name", "unknown_generator")),
        generator_config=_jsonish_string(record.get("generator_config", "")),
        seed=int(record.get("seed", 0)),
        raw_trajectory_path=str(record.get("raw_trajectory_path", "")),
        selector_visible_trajectory=SelectorVisibleTrajectory.from_record(
            record.get("selector_visible_trajectory") if isinstance(record.get("selector_visible_trajectory"), dict) else {}
        ),
        official_success=parsed_success,
        official_eval_metadata_path=str(record.get("official_eval_metadata_path", "")),
        metadata=dict(record.get("metadata", {})) if isinstance(record.get("metadata", {}), dict) else {},
    )


def taubench_candidate_to_record(candidate: TauBenchTrajectoryCandidate) -> Dict[str, object]:
    return {
        "task_id": candidate.task_id,
        "domain": candidate.domain,
        "candidate_id": candidate.candidate_id,
        "candidate_order_index": int(candidate.candidate_order_index),
        "generator_name": candidate.generator_name,
        "generator_config": candidate.generator_config,
        "seed": int(candidate.seed),
        "raw_trajectory_path": candidate.raw_trajectory_path,
        "selector_visible_trajectory": candidate.selector_visible_trajectory.to_record(),
        "official_success": candidate.official_success,
        "official_eval_metadata_path": candidate.official_eval_metadata_path,
        "trajectory_hash": candidate.trajectory_hash,
        "metadata": dict(candidate.metadata),
    }


def load_taubench_candidates_jsonl(path: str | Path) -> List[TauBenchTrajectoryCandidate]:
    source = Path(path)
    if not source.exists():
        return []
    rows: List[TauBenchTrajectoryCandidate] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(taubench_candidate_from_record(json.loads(line)))
    return rows


def write_taubench_candidates_jsonl(path: str | Path, candidates: Sequence[TauBenchTrajectoryCandidate]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(json.dumps(taubench_candidate_to_record(candidate), sort_keys=True) for candidate in candidates),
        encoding="utf-8",
    )


def build_taubench_selection_examples(
    candidates: Sequence[TauBenchTrajectoryCandidate],
    splits: Dict[str, str] | None = None,
) -> List[TauBenchTrajectorySelectionExample]:
    by_task: Dict[Tuple[str, str], List[TauBenchTrajectoryCandidate]] = {}
    for candidate in candidates:
        by_task.setdefault((candidate.domain, candidate.task_id), []).append(candidate)
    examples: List[TauBenchTrajectorySelectionExample] = []
    for (domain, task_id), rows in sorted(by_task.items()):
        ordered = tuple(sorted(rows, key=lambda row: (int(row.candidate_order_index), row.candidate_id)))
        first_meta = dict(ordered[0].metadata) if ordered else {}
        split = str(first_meta.get("split", "unassigned"))
        if splits and f"{domain}:{task_id}" in splits:
            split = splits[f"{domain}:{task_id}"]
        examples.append(
            TauBenchTrajectorySelectionExample(
                id=f"{domain}-{task_id}",
                task_id=task_id,
                domain=domain,
                candidates=ordered,
                split=split,
                metadata={
                    "domain": domain,
                    "task_id": task_id,
                    "policy_text": str(first_meta.get("policy_text", "")),
                    "task_type": str(first_meta.get("task_type", "unknown")),
                    "public_task_request": str(first_meta.get("public_task_request", "")),
                    "candidate_order_randomized": bool(first_meta.get("candidate_order_randomized", False)),
                    "candidate_order_seed": first_meta.get("candidate_order_seed"),
                    "tau_bench_task_split_name": str(first_meta.get("tau_bench_task_split_name", "base")),
                    "tau_bench_task_set_name": str(first_meta.get("tau_bench_task_set_name", "default")),
                    "selector_visible_excludes_generator_identity": True,
                },
            )
        )
    return examples


def load_taubench_selection_examples(
    path: str | Path,
    splits: Dict[str, str] | None = None,
) -> List[TauBenchTrajectorySelectionExample]:
    return build_taubench_selection_examples(load_taubench_candidates_jsonl(path), splits=splits)


def validate_taubench_selection_examples(
    examples: Sequence[TauBenchTrajectorySelectionExample],
    k: int = STAGE7_NUM_CANDIDATES,
    require_labels: bool = True,
) -> Dict[str, object]:
    errors: List[str] = []
    for example in examples:
        if len(example.candidates) != k:
            errors.append(f"{example.id}: expected K={k}, found {len(example.candidates)}")
        if sorted(candidate.candidate_order_index for candidate in example.candidates) != list(range(len(example.candidates))):
            errors.append(f"{example.id}: candidate_order_index must be a 0..K-1 permutation")
        if require_labels and any(candidate.official_success is None for candidate in example.candidates):
            errors.append(f"{example.id}: missing official_success labels")
        if len({candidate.candidate_id for candidate in example.candidates}) != len(example.candidates):
            errors.append(f"{example.id}: duplicate candidate_id")
        if any(candidate.domain != example.domain or candidate.task_id != example.task_id for candidate in example.candidates):
            errors.append(f"{example.id}: mixed task/domain candidates")
    leakage = stage7_output_leakage_audit("stage7_taubench_trajectory_selector", 0, examples)
    duplicates = duplicate_trajectory_hash_audit(examples)
    return {
        "examples_checked": len(examples),
        "candidate_count": k,
        "passes": not errors and bool(leakage.get("passes", False)) and bool(duplicates.get("passes", False)),
        "errors": errors,
        "output_leakage_audit": leakage,
        "duplicate_trajectory_hash_audit": duplicates,
    }


def taubench_to_multiview(example: TauBenchTrajectorySelectionExample) -> MultiViewTaskExample:
    labels = [int(value) for value in example.labels_pass_fail]
    label = next((index for index, value in enumerate(labels) if value == 1), 0)
    candidates = tuple(
        MultiViewCandidate(
            candidate_id=candidate.candidate_id,
            text=trajectory_candidate_text(candidate),
            source_path=example.domain,
            patch_hash=candidate.trajectory_hash,
            attributes=trajectory_candidate_attributes(example, candidate),
        )
        for candidate in example.candidates
    )
    metadata = {
        "dataset_source": "stage7_taubench_fixed_trajectory_pool",
        "domain": example.domain,
        "task_id": example.task_id,
        "num_candidates": len(example.candidates),
        "n_views": len(STAGE7_ROLE_NAMES),
        "candidate_representation": "selector_visible_tool_use_trajectory",
        "generator_version": "generator_identity_blinded_from_selector_visible_text",
        "attribute_value_pairs": [list(pair) for pair in STAGE7_ATTRIBUTE_VALUE_PAIRS],
        "stage7_task_type": str(example.metadata.get("task_type", "unknown")),
        "selector_only_no_generation_claim": True,
    }
    oracle_metadata = {
        **metadata,
        "labels_pass_fail": [int(value) for value in example.labels_pass_fail],
        "oracle_pass_at_8": bool(example.oracle_pass_at_8),
        "candidate_trajectory_hashes": [candidate.trajectory_hash for candidate in example.candidates],
        "candidate_generator_names": [candidate.generator_name for candidate in example.candidates],
        "task_text_hash": task_text_hash(_user_goal_text(example)),
    }
    return MultiViewTaskExample(
        id=example.id,
        views=stage7_views(example),
        candidates=candidates,
        label=int(label),
        metadata=metadata,
        oracle_metadata=oracle_metadata,
    )


def taubench_to_multiview_many(examples: Sequence[TauBenchTrajectorySelectionExample]) -> List[MultiViewTaskExample]:
    return [taubench_to_multiview(example) for example in examples]


def taubench_label_matrix(examples: Sequence[TauBenchTrajectorySelectionExample]) -> np.ndarray:
    if not examples:
        return np.zeros((0, STAGE7_NUM_CANDIDATES), dtype=np.float32)
    return np.asarray([[int(value) for value in example.labels_pass_fail] for example in examples], dtype=np.float32)


def randomized_label_matrix(
    examples: Sequence[TauBenchTrajectorySelectionExample],
    seed: int,
    num_candidates: int = STAGE7_NUM_CANDIDATES,
) -> np.ndarray:
    rng = np.random.default_rng(seed + 707_001)
    rows = np.zeros((len(examples), num_candidates), dtype=np.float32)
    base = np.tile(np.arange(num_candidates, dtype=np.int64), int(np.ceil(max(1, len(examples)) / num_candidates)))[: len(examples)]
    rng.shuffle(base)
    for row, index in enumerate(base):
        rows[row, int(index)] = 1.0
    true = taubench_label_matrix(examples)
    if len(rows) > 1 and rows.shape == true.shape and np.array_equal(rows, true):
        rows = np.roll(rows, 1, axis=1)
    return rows


def apply_stage7_control(
    examples: Sequence[TauBenchTrajectorySelectionExample],
    condition: str,
    seed: int,
) -> List[TauBenchTrajectorySelectionExample]:
    if condition not in STAGE7_CONTROL_NAMES:
        raise ValueError(f"unknown Stage 7 control condition: {condition}")
    if condition in {"none", "hidden_states_shuffled_across_examples", "physical_order_shuffled_roles_avenues_preserved"}:
        return list(examples)
    if condition == "randomized_labels":
        randomized = randomized_label_matrix(examples, seed)
        out = []
        for row, example in enumerate(examples):
            candidates = tuple(
                replace(candidate, official_success=bool(randomized[row, index] > 0.0))
                for index, candidate in enumerate(example.candidates)
            )
            out.append(replace(example, candidates=candidates, metadata={**example.metadata, "stage7_control": condition}))
        return out
    if condition == "candidate_order_shuffled_with_label_remap":
        rng = np.random.default_rng(seed + 901_713)
        out = []
        for example in examples:
            perm = _non_identity_permutation(len(example.candidates), rng)
            shuffled = []
            for new_index, old_index in enumerate(perm.tolist()):
                old = example.candidates[int(old_index)]
                shuffled.append(replace(old, candidate_order_index=int(new_index)))
            out.append(
                replace(
                    example,
                    candidates=tuple(shuffled),
                    metadata={**example.metadata, "stage7_candidate_order_control": [int(value) for value in perm.tolist()]},
                )
            )
        return out
    if condition == "candidate_only":
        return [_with_example_evidence_hidden(example, condition, keep_policy=False, keep_user=False, keep_tool=False) for example in examples]
    if condition == "trajectory_only":
        return [
            replace(
                _with_example_evidence_hidden(example, condition, keep_policy=False, keep_user=False, keep_tool=False),
                candidates=tuple(_candidate_trajectory_only(candidate, condition) for candidate in example.candidates),
            )
            for example in examples
        ]
    if condition == "tool_observations_only":
        return [
            replace(
                _with_example_evidence_hidden(example, condition, keep_policy=False, keep_user=False, keep_tool=True),
                candidates=tuple(_candidate_tool_observations_only(candidate, condition) for candidate in example.candidates),
            )
            for example in examples
        ]
    if condition == "user_goal_only":
        return [_with_hidden_candidates(_with_example_evidence_hidden(example, condition, keep_policy=False, keep_user=True, keep_tool=False), condition) for example in examples]
    if condition == "policy_only":
        return [_with_hidden_candidates(_with_example_evidence_hidden(example, condition, keep_policy=True, keep_user=False, keep_tool=False), condition) for example in examples]
    if condition == "evidence_only_no_candidate":
        return [_with_hidden_candidates(_with_example_evidence_hidden(example, condition, keep_policy=True, keep_user=True, keep_tool=True), condition) for example in examples]
    if condition == "schema_template_only":
        return [_schema_template_only(example) for example in examples]
    if condition == "generator_identity_only":
        return [_generator_identity_only(example) for example in examples]
    if condition in {
        "candidate_evidence_mismatch",
        "cross_task_user_goal_shuffle",
        "cross_task_policy_shuffle",
        "cross_task_tool_observation_shuffle",
    }:
        rng = np.random.default_rng(seed + 503_731)
        perm = _non_identity_permutation(len(examples), rng)
        out = []
        for index, example in enumerate(examples):
            donor = examples[int(perm[index])]
            if condition == "candidate_evidence_mismatch":
                out.append(_mismatch_evidence(example, donor, condition))
            elif condition == "cross_task_user_goal_shuffle":
                out.append(_shuffle_user_goal(example, donor, condition))
            elif condition == "cross_task_policy_shuffle":
                out.append(
                    replace(
                        example,
                        metadata={**example.metadata, "policy_text": str(donor.metadata.get("policy_text", "")), f"stage7_{condition}_donor_id": donor.id},
                    )
                )
            else:
                out.append(_shuffle_tool_observations(example, donor, condition))
        return out
    raise ValueError(f"unhandled Stage 7 control condition: {condition}")


def stage7_views(example: TauBenchTrajectorySelectionExample) -> Tuple[View, ...]:
    policy = str(example.metadata.get("policy_text", "")).strip() or "No domain policy text was supplied with this task record."
    task_type = str(example.metadata.get("task_type", "unknown"))
    rows = (
        (
            "Role 0: user goal and conversation evidence.\n"
            f"Domain: {example.domain}\nTask ID: {example.task_id}\nTask type: {task_type}\n"
            f"User/task request evidence:\n{_user_goal_text(example)}"
        ),
        (
            "Role 1: policy and rule evidence.\n"
            "Use only public domain policy text and rules available to the agent.\n"
            f"{policy}"
        ),
        (
            "Role 2: candidate trajectory evidence.\n"
            "Candidate trajectories are complete tool-use attempts for the same task. Generator identity is blinded.\n"
            f"{_trajectory_inventory(example)}"
        ),
        (
            "Role 3: tool/state evidence.\n"
            "Use only tool observations shown during each trajectory; no evaluator verdicts or hidden database diffs are visible.\n"
            f"{_tool_observation_inventory(example)}"
        ),
    )
    return tuple(
        View(
            role=STAGE7_ROLE_NAMES[index],
            text=text,
            source_path=example.domain,
            source_type=f"stage7_role_{index}",
            allowed_visibility="agent_private_partial_view",
        )
        for index, text in enumerate(rows)
    )


def trajectory_candidate_text(candidate: TauBenchTrajectoryCandidate) -> str:
    trajectory = candidate.selector_visible_trajectory
    return "\n".join(
        [
            f"candidate_id: {candidate.candidate_id}",
            "generator_identity: blinded",
            "user_messages:",
            _format_lines(trajectory.user_messages),
            "assistant_messages:",
            _format_lines(trajectory.assistant_messages),
            "tool_calls:",
            _json_lines(trajectory.tool_calls),
            "tool_observations_sanitized:",
            _format_lines(trajectory.tool_observations_sanitized),
            "final_response:",
            trajectory.final_response,
            "static_trajectory_features:",
            json.dumps(trajectory_static_features(candidate, user_goal=""), sort_keys=True),
        ]
    )


def trajectory_static_feature_vector(
    example: TauBenchTrajectorySelectionExample,
    candidate: TauBenchTrajectoryCandidate,
) -> np.ndarray:
    features = trajectory_static_features(candidate, user_goal=_user_goal_text(example))
    keys = (
        "num_tool_calls",
        "repeated_tool_calls",
        "invalid_tool_calls",
        "policy_mentions",
        "final_response_length",
        "tool_error_count",
        "final_response_claims_completion",
        "user_goal_entity_overlap",
        "conversation_length",
    )
    values = []
    for key in keys:
        value = float(features.get(key, 0.0))
        if key in {"num_tool_calls", "final_response_length", "conversation_length"}:
            value = float(np.log1p(max(0.0, value)))
        values.append(value)
    return np.asarray(values, dtype=np.float32)


def trajectory_static_features(candidate: TauBenchTrajectoryCandidate, user_goal: str = "") -> Dict[str, float]:
    trajectory = candidate.selector_visible_trajectory
    tool_names = [_tool_name(call) for call in trajectory.tool_calls]
    tool_counter = Counter(tool_names)
    observations_text = "\n".join(trajectory.tool_observations_sanitized).lower()
    calls_text = "\n".join(json.dumps(call, sort_keys=True) for call in trajectory.tool_calls).lower()
    final = trajectory.final_response.lower()
    policy_mentions = len(re.findall(r"\b(policy|required|allowed|refund|cancel|change|verify|confirm|return|exchange)\b", final + "\n" + calls_text))
    error_count = len(re.findall(r"\b(error|failed|invalid|not found|exception|denied|unauthorized)\b", observations_text + "\n" + calls_text))
    goal_tokens = {token for token in _tokens(user_goal) if len(token) >= 4}
    final_tokens = set(_tokens(final))
    overlap = len(goal_tokens & final_tokens) / float(max(1, len(goal_tokens))) if goal_tokens else 0.0
    return {
        "num_tool_calls": float(len(trajectory.tool_calls)),
        "repeated_tool_calls": float(sum(1 for count in tool_counter.values() if count > 1)),
        "invalid_tool_calls": float(len(re.findall(r"\b(invalid_tool|unknown_tool|schema_error|bad_request)\b", calls_text))),
        "policy_mentions": float(policy_mentions),
        "final_response_length": float(len(_tokens(final))),
        "tool_error_count": float(error_count),
        "final_response_claims_completion": float(
            bool(re.search(r"\b(done|completed|processed|cancelled|canceled|refunded|updated|changed|placed|scheduled)\b", final))
        ),
        "user_goal_entity_overlap": float(overlap),
        "conversation_length": float(len(trajectory.user_messages) + len(trajectory.assistant_messages) + len(trajectory.tool_calls)),
    }


def trajectory_candidate_attributes(
    example: TauBenchTrajectorySelectionExample,
    candidate: TauBenchTrajectoryCandidate,
) -> Tuple[str, str, str, str]:
    features = trajectory_static_features(candidate, user_goal=_user_goal_text(example))
    return (
        STAGE7_ATTRIBUTE_VALUE_PAIRS[0][1] if features["num_tool_calls"] >= 5 else STAGE7_ATTRIBUTE_VALUE_PAIRS[0][0],
        STAGE7_ATTRIBUTE_VALUE_PAIRS[1][1] if features["repeated_tool_calls"] > 0 else STAGE7_ATTRIBUTE_VALUE_PAIRS[1][0],
        STAGE7_ATTRIBUTE_VALUE_PAIRS[2][1] if features["tool_error_count"] > 0 else STAGE7_ATTRIBUTE_VALUE_PAIRS[2][0],
        STAGE7_ATTRIBUTE_VALUE_PAIRS[3][1] if features["final_response_claims_completion"] > 0 else STAGE7_ATTRIBUTE_VALUE_PAIRS[3][0],
    )


def stage7_output_leakage_audit(
    benchmark: str,
    seed: int,
    examples: Sequence[TauBenchTrajectorySelectionExample],
) -> Dict[str, object]:
    visible_texts: List[str] = []
    generator_hits = []
    for example in examples:
        visible_texts.extend(view.text for view in stage7_views(example))
        for candidate in example.candidates:
            text = trajectory_candidate_text(candidate)
            visible_texts.append(text)
            generator_tokens = [candidate.generator_name, candidate.generator_config]
            for token in generator_tokens:
                token_text = str(token).strip()
                if token_text and token_text.lower() not in {"unknown_generator", "{}", "none"} and token_text in text:
                    generator_hits.append({"candidate_id": candidate.candidate_id, "generator_token": token_text[:120]})
    pattern_hits = []
    for pattern in FORBIDDEN_SELECTOR_VISIBLE_PATTERNS:
        regex = re.compile(pattern, flags=re.IGNORECASE)
        if any(regex.search(text) for text in visible_texts):
            pattern_hits.append(pattern)
    return {
        "benchmark": benchmark,
        "seed": int(seed),
        "type": "stage7_output_leakage",
        "examples_checked": len(examples),
        "visible_text_forbidden_pattern_hits": pattern_hits,
        "generator_identity_visible_hits": generator_hits,
        "hidden_success_labels_excluded_from_visible_text": not pattern_hits,
        "generator_identity_blinded_from_selector_visible_text": not generator_hits,
        "passes": not pattern_hits and not generator_hits,
    }


def stage7_split_leakage_audit(
    benchmark: str,
    seed: int,
    splits: Dict[str, Sequence[TauBenchTrajectorySelectionExample]],
) -> Dict[str, object]:
    ids = {name: {example.id for example in rows} for name, rows in splits.items()}
    task_ids = {name: {(example.domain, example.task_id) for example in rows} for name, rows in splits.items()}
    hashes = {
        name: {candidate.trajectory_hash for example in rows for candidate in example.candidates}
        for name, rows in splits.items()
    }
    task_hashes = {name: {task_text_hash(_user_goal_text(example)) for example in rows} for name, rows in splits.items()}
    near_dupes = _near_duplicate_task_overlaps(splits)
    overlaps = {
        "train_dev_id_overlap": len(ids.get("train", set()) & ids.get("dev", set())),
        "train_test_id_overlap": len(ids.get("train", set()) & ids.get("test", set())),
        "dev_test_id_overlap": len(ids.get("dev", set()) & ids.get("test", set())),
        "train_dev_task_overlap": len(task_ids.get("train", set()) & task_ids.get("dev", set())),
        "train_test_task_overlap": len(task_ids.get("train", set()) & task_ids.get("test", set())),
        "dev_test_task_overlap": len(task_ids.get("dev", set()) & task_ids.get("test", set())),
        "train_dev_candidate_hash_overlap": len(hashes.get("train", set()) & hashes.get("dev", set())),
        "train_test_candidate_hash_overlap": len(hashes.get("train", set()) & hashes.get("test", set())),
        "dev_test_candidate_hash_overlap": len(hashes.get("dev", set()) & hashes.get("test", set())),
        "train_dev_task_text_hash_overlap": len(task_hashes.get("train", set()) & task_hashes.get("dev", set())),
        "train_test_task_text_hash_overlap": len(task_hashes.get("train", set()) & task_hashes.get("test", set())),
        "dev_test_task_text_hash_overlap": len(task_hashes.get("dev", set()) & task_hashes.get("test", set())),
    }
    return {
        "benchmark": benchmark,
        "seed": int(seed),
        "type": "stage7_taubench_split_leakage",
        **overlaps,
        "near_duplicate_task_text_overlaps": near_dupes,
        "candidate_trajectory_hash_leakage_audit_passes": not (
            hashes.get("train", set()) & hashes.get("dev", set())
            or hashes.get("train", set()) & hashes.get("test", set())
            or hashes.get("dev", set()) & hashes.get("test", set())
        ),
        "passes": not any(overlaps.values()) and not near_dupes,
    }


def duplicate_trajectory_hash_audit(examples: Sequence[TauBenchTrajectorySelectionExample]) -> Dict[str, object]:
    rows = []
    for example in examples:
        hashes = [candidate.trajectory_hash for candidate in example.candidates]
        duplicates = sorted({value for value in hashes if hashes.count(value) > 1})
        if duplicates:
            rows.append({"id": example.id, "duplicate_trajectory_hashes": duplicates})
    return {"examples_checked": len(examples), "duplicates": rows, "passes": not rows}


def near_duplicate_trajectory_audit(
    examples: Sequence[TauBenchTrajectorySelectionExample],
    threshold: float = 0.98,
) -> Dict[str, object]:
    rows = []
    for example in examples:
        texts = [normalize_text(trajectory_candidate_text(candidate)) for candidate in example.candidates]
        sets = [set(text.split()) for text in texts]
        for left in range(len(sets)):
            for right in range(left + 1, len(sets)):
                denom = max(1, len(sets[left] | sets[right]))
                score = len(sets[left] & sets[right]) / float(denom)
                if score >= threshold and sets[left] and sets[right]:
                    rows.append(
                        {
                            "id": example.id,
                            "left_candidate_id": example.candidates[left].candidate_id,
                            "right_candidate_id": example.candidates[right].candidate_id,
                            "jaccard": float(score),
                        }
                    )
    return {"examples_checked": len(examples), "threshold": float(threshold), "near_duplicates": rows, "passes": not rows}


def stage7_dataset_summary(splits: Dict[str, Sequence[TauBenchTrajectorySelectionExample]]) -> Dict[str, object]:
    all_examples = [example for rows in splits.values() for example in rows]
    oracle = [example.oracle_pass_at_8 for example in all_examples]
    return {
        "dataset_names": ["stage7_taubench_trajectory_selection"],
        "split_sizes": {split: len(rows) for split, rows in splits.items()},
        "num_candidates": STAGE7_NUM_CANDIDATES,
        "num_roles": len(STAGE7_ROLE_NAMES),
        "num_avenues": len(STAGE7_AVENUE_NAMES),
        "domains": sorted({example.domain for example in all_examples}),
        "oracle_pass_at_8": float(np.mean(oracle)) if oracle else 0.0,
        "oracle_empty_tasks": int(sum(not value for value in oracle)),
        "candidate_generators_audit_only": sorted(
            {candidate.generator_name for example in all_examples for candidate in example.candidates}
        ),
        "selector_visible_generator_identity_blinded": True,
        "selector_only_no_trajectory_generation_claim": True,
    }


def _user_goal_text(example: TauBenchTrajectorySelectionExample) -> str:
    public = str(example.metadata.get("public_task_request", "")).strip()
    messages: List[str] = []
    for candidate in example.candidates[:1]:
        messages.extend(candidate.selector_visible_trajectory.user_messages)
    joined = "\n".join(messages).strip()
    return "\n".join(value for value in (public, joined) if value).strip() or "No user goal text supplied."


def _trajectory_inventory(example: TauBenchTrajectorySelectionExample) -> str:
    rows = []
    for index, candidate in enumerate(example.candidates):
        features = trajectory_static_features(candidate, user_goal=_user_goal_text(example))
        rows.append(
            "candidate {index}: id={candidate_id}; tool_calls={tool_calls}; repeated_tool_calls={repeated}; "
            "tool_error_count={errors}; final_response_tokens={tokens}; generator_identity=blinded".format(
                index=index,
                candidate_id=candidate.candidate_id,
                tool_calls=int(features["num_tool_calls"]),
                repeated=int(features["repeated_tool_calls"]),
                errors=int(features["tool_error_count"]),
                tokens=int(features["final_response_length"]),
            )
        )
    return "\n".join(rows)


def _tool_observation_inventory(example: TauBenchTrajectorySelectionExample) -> str:
    rows = []
    for index, candidate in enumerate(example.candidates):
        observations = candidate.selector_visible_trajectory.tool_observations_sanitized
        summary = " | ".join(str(value).replace("\n", " ")[:240] for value in observations[:6])
        rows.append(f"candidate {index}: observations={summary or 'none supplied'}")
    return "\n".join(rows)


def _format_lines(values: Sequence[str]) -> str:
    if not values:
        return "- none supplied"
    return "\n".join(f"- {str(value)}" for value in values)


def _json_lines(values: Sequence[Dict[str, object]]) -> str:
    if not values:
        return "- none supplied"
    return "\n".join(f"- {json.dumps(value, sort_keys=True)}" for value in values)


def _tool_name(call: Dict[str, object]) -> str:
    for key in ("tool_name", "name", "function", "action"):
        value = call.get(key)
        if isinstance(value, dict):
            for child_key in ("name", "tool_name"):
                if child_key in value:
                    return str(value[child_key])
        if value not in (None, ""):
            return str(value)
    return "unknown_tool"


def _with_example_evidence_hidden(
    example: TauBenchTrajectorySelectionExample,
    condition: str,
    keep_policy: bool,
    keep_user: bool,
    keep_tool: bool,
) -> TauBenchTrajectorySelectionExample:
    metadata = dict(example.metadata)
    if not keep_policy:
        metadata["policy_text"] = f"Policy evidence hidden for {condition} control."
    if not keep_user:
        metadata["public_task_request"] = f"User goal hidden for {condition} control."
    if not keep_tool:
        metadata["tool_state_evidence_hidden"] = True
    metadata["stage7_control"] = condition
    return replace(example, metadata=metadata)


def _with_hidden_candidates(example: TauBenchTrajectorySelectionExample, condition: str) -> TauBenchTrajectorySelectionExample:
    return replace(example, candidates=tuple(_hidden_candidate(candidate, condition) for candidate in example.candidates))


def _hidden_candidate(candidate: TauBenchTrajectoryCandidate, condition: str) -> TauBenchTrajectoryCandidate:
    visible = SelectorVisibleTrajectory(
        user_messages=(f"Candidate trajectory hidden for {condition} control.",),
        assistant_messages=(f"Candidate trajectory hidden for {condition} control.",),
        tool_calls=(),
        tool_observations_sanitized=(f"Candidate tool observations hidden for {condition} control.",),
        final_response=f"Candidate final response hidden for {condition} control.",
    )
    return replace(candidate, selector_visible_trajectory=visible, metadata={**candidate.metadata, "stage7_hidden_for_control": condition})


def _candidate_trajectory_only(candidate: TauBenchTrajectoryCandidate, condition: str) -> TauBenchTrajectoryCandidate:
    old = candidate.selector_visible_trajectory
    visible = SelectorVisibleTrajectory(
        user_messages=(f"User goal hidden for {condition} control.",),
        assistant_messages=old.assistant_messages,
        tool_calls=old.tool_calls,
        tool_observations_sanitized=(f"Tool observations hidden for {condition} control.",),
        final_response=old.final_response,
    )
    return replace(candidate, selector_visible_trajectory=visible, metadata={**candidate.metadata, "stage7_control": condition})


def _candidate_tool_observations_only(candidate: TauBenchTrajectoryCandidate, condition: str) -> TauBenchTrajectoryCandidate:
    old = candidate.selector_visible_trajectory
    visible = SelectorVisibleTrajectory(
        user_messages=(f"User goal hidden for {condition} control.",),
        assistant_messages=(f"Assistant messages hidden for {condition} control.",),
        tool_calls=(),
        tool_observations_sanitized=old.tool_observations_sanitized,
        final_response=f"Final response hidden for {condition} control.",
    )
    return replace(candidate, selector_visible_trajectory=visible, metadata={**candidate.metadata, "stage7_control": condition})


def _schema_template_only(example: TauBenchTrajectorySelectionExample) -> TauBenchTrajectorySelectionExample:
    templated = _with_hidden_candidates(example, "schema_template_only")
    return replace(
        templated,
        metadata={
            **templated.metadata,
            "policy_text": "Role 1 schema template: public policy text field.",
            "public_task_request": "Role 0 schema template: user goal and conversation fields.",
            "stage7_control": "schema_template_only",
        },
    )


def _generator_identity_only(example: TauBenchTrajectorySelectionExample) -> TauBenchTrajectorySelectionExample:
    candidates = []
    for candidate in example.candidates:
        visible = SelectorVisibleTrajectory(
            user_messages=("Generator identity diagnostic only; not selector-visible in main runs.",),
            assistant_messages=(f"generator={candidate.generator_name}",),
            tool_calls=(),
            tool_observations_sanitized=(f"generator_config={candidate.generator_config}",),
            final_response="Generator identity diagnostic only.",
        )
        candidates.append(replace(candidate, selector_visible_trajectory=visible, metadata={**candidate.metadata, "audit_only": True}))
    return replace(example, candidates=tuple(candidates), metadata={**example.metadata, "stage7_control": "generator_identity_only"})


def _mismatch_evidence(
    example: TauBenchTrajectorySelectionExample,
    donor: TauBenchTrajectorySelectionExample,
    condition: str,
) -> TauBenchTrajectorySelectionExample:
    donor_user = _first_user_messages(donor)
    donor_obs = _candidate_observations_by_slot(donor)
    candidates = []
    for index, candidate in enumerate(example.candidates):
        old = candidate.selector_visible_trajectory
        visible = replace(
            old,
            user_messages=donor_user,
            tool_observations_sanitized=donor_obs[index % max(1, len(donor_obs))],
        )
        candidates.append(replace(candidate, selector_visible_trajectory=visible))
    return replace(
        example,
        candidates=tuple(candidates),
        metadata={
            **example.metadata,
            "policy_text": str(donor.metadata.get("policy_text", "")),
            "public_task_request": str(donor.metadata.get("public_task_request", "")),
            f"stage7_{condition}_donor_id": donor.id,
        },
    )


def _shuffle_user_goal(
    example: TauBenchTrajectorySelectionExample,
    donor: TauBenchTrajectorySelectionExample,
    condition: str,
) -> TauBenchTrajectorySelectionExample:
    donor_user = _first_user_messages(donor)
    candidates = []
    for candidate in example.candidates:
        old = candidate.selector_visible_trajectory
        candidates.append(replace(candidate, selector_visible_trajectory=replace(old, user_messages=donor_user)))
    return replace(
        example,
        candidates=tuple(candidates),
        metadata={
            **example.metadata,
            "public_task_request": str(donor.metadata.get("public_task_request", "")),
            f"stage7_{condition}_donor_id": donor.id,
        },
    )


def _shuffle_tool_observations(
    example: TauBenchTrajectorySelectionExample,
    donor: TauBenchTrajectorySelectionExample,
    condition: str,
) -> TauBenchTrajectorySelectionExample:
    donor_obs = _candidate_observations_by_slot(donor)
    candidates = []
    for index, candidate in enumerate(example.candidates):
        old = candidate.selector_visible_trajectory
        candidates.append(
            replace(
                candidate,
                selector_visible_trajectory=replace(old, tool_observations_sanitized=donor_obs[index % max(1, len(donor_obs))]),
            )
        )
    return replace(example, candidates=tuple(candidates), metadata={**example.metadata, f"stage7_{condition}_donor_id": donor.id})


def _first_user_messages(example: TauBenchTrajectorySelectionExample) -> Tuple[str, ...]:
    if not example.candidates:
        return ("No donor user messages supplied.",)
    values = example.candidates[0].selector_visible_trajectory.user_messages
    return values if values else ("No donor user messages supplied.",)


def _candidate_observations_by_slot(example: TauBenchTrajectorySelectionExample) -> List[Tuple[str, ...]]:
    rows = [candidate.selector_visible_trajectory.tool_observations_sanitized for candidate in example.candidates]
    return [row if row else ("No donor tool observations supplied.",) for row in rows] or [("No donor tool observations supplied.",)]


def _near_duplicate_task_overlaps(
    splits: Dict[str, Sequence[TauBenchTrajectorySelectionExample]],
    threshold: float = 0.92,
) -> List[Dict[str, object]]:
    names = sorted(splits)
    token_sets = {
        name: [(example.id, set(normalize_text(_user_goal_text(example)).split())) for example in rows]
        for name, rows in splits.items()
    }
    out = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            for left_id, left_tokens in token_sets[left]:
                for right_id, right_tokens in token_sets[right]:
                    denom = max(1, len(left_tokens | right_tokens))
                    score = len(left_tokens & right_tokens) / float(denom)
                    if score >= threshold and left_tokens and right_tokens:
                        out.append({"left_split": left, "right_split": right, "left_id": left_id, "right_id": right_id, "jaccard": score})
    return out


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", str(text).lower())


def _safe_dict(value: object) -> Dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    return {"value": str(value)}


def _jsonish_string(value: object) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return str(value)


def _non_identity_permutation(n: int, rng: np.random.Generator) -> np.ndarray:
    perm = rng.permutation(n)
    if n > 1 and np.array_equal(perm, np.arange(n)):
        perm = np.roll(perm, 1)
    return perm.astype(np.int64, copy=False)


def _asdict_no_none(value: object) -> Dict[str, object]:
    row = asdict(value) if hasattr(value, "__dataclass_fields__") else dict(value)  # type: ignore[arg-type]
    return {key: child for key, child in row.items() if child is not None}
