from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.agents.types import AttemptBatch


ROLE_NAMES = ("symbol_kind", "provider_area", "provider_name_parity", "import_location")
ATTRIBUTE_NAMES = ("symbol_kind", "provider_area", "provider_name_parity", "import_line_parity")
ATTRIBUTE_VALUE_PAIRS = (
    ("symbol_kind_function_or_value", "symbol_kind_class"),
    ("provider_area_core", "provider_area_coordination_or_experiment"),
    ("provider_name_even", "provider_name_odd"),
    ("import_line_even", "import_line_odd"),
)
CODE_WORDS = (
    (0, 0, 0, 0),
    (0, 0, 1, 1),
    (0, 1, 0, 1),
    (0, 1, 1, 0),
    (1, 0, 0, 1),
    (1, 0, 1, 0),
    (1, 1, 0, 0),
    (1, 1, 1, 1),
)
MULTIVIEW_V2_CODE_WORDS = (
    (0, 0, 0, 0),
    (0, 1, 0, 1),
    (0, 1, 1, 0),
    (1, 0, 0, 1),
    (1, 0, 1, 0),
    (1, 1, 0, 1),
    (1, 1, 1, 0),
    (1, 1, 1, 1),
)
CONTROL_NAMES = (
    "none",
    "randomized_labels",
    "view_masked",
    "view_shuffled",
    "hidden_states_shuffled_across_examples",
    "role_labels_shuffled",
    "physical_order_shuffled_roles_preserved",
    "candidate_order_shuffled",
)


@dataclass(frozen=True)
class View:
    role: str
    text: str
    source_path: str
    source_type: str
    allowed_visibility: str = "agent_private_partial_view"


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    text: str
    source_path: str
    patch_hash: str
    attributes: Tuple[str, str, str, str] = ("", "", "", "")


@dataclass(frozen=True)
class MultiViewTaskExample:
    id: str
    views: Tuple[View, ...]
    candidates: Tuple[Candidate, ...]
    label: int
    metadata: Dict[str, object]
    oracle_metadata: Dict[str, object] | None = None


@dataclass(frozen=True)
class MultiViewCodePatchDatasetConfig:
    n_train: int = 128
    n_dev: int = 64
    n_test: int = 128
    num_candidates: int = 8
    n_views: int = 4
    source_roots: Tuple[str, ...] = ("src", "tests")
    max_files: int = 80
    snippet_radius: int = 3
    include_private_signal_tokens: bool = False
    dataset_source: str = "real_program_analysis_import_restoration"
    generator_version: str = "real_import_restore_stable_categories_balanced_v3"
    candidate_representation: str = "full_patch"


@dataclass(frozen=True)
class ExportInfo:
    module: str
    name: str
    path: Path
    rel_path: str
    kind: str
    line: int
    snippet: str


@dataclass(frozen=True)
class ImportRepairCase:
    target_path: Path
    target_rel_path: str
    import_module: str
    imported_name: str
    alias: str
    import_line: int
    import_statement: str
    usage_lines: Tuple[int, ...]
    usage_snippet: str
    import_block_snippet: str
    provider: ExportInfo
    source_key: str


SANITIZED_DATASET_SOURCE = "real_import_restore_candidate_sanitized_v3"
SANITIZED_CANDIDATE_REPRESENTATION = "sanitized_redacted_v3"
BALANCED_34B_DATASET_SOURCE = "real_import_restore_candidate_balanced_34b"
BALANCED_34B_CANDIDATE_REPRESENTATION = "balanced_categories_v3"

ORACLE_ONLY_METADATA_KEYS = frozenset(
    {
        "attribute_names",
        "attribute_value_pairs",
        "candidate_bit_tuples",
        "candidate_order",
        "candidate_patch_values",
        "candidate_tuples",
        "evidence_bits",
        "gold_import_statement",
        "gold_patch_hash",
        "gold_tuple",
        "issue_text_hash",
        "label",
        "line_number",
        "problem_family",
        "provider_file",
        "provider_module",
        "source_file",
        "source_import_key",
        "usage_lines",
    }
)

MODEL_RECORD_SAFE_METADATA_KEYS = frozenset(
    {
        "candidate_representation",
        "constructed_fallback",
        "correct_patch_origin",
        "dataset_source",
        "generator_version",
        "hardening",
        "num_candidates",
        "n_views",
        "private_cue_style",
        "program_analysis_artifacts",
        "real_correct_patch",
    }
)


def example_oracle_metadata(example: MultiViewTaskExample) -> Dict[str, object]:
    return example.oracle_metadata if example.oracle_metadata is not None else example.metadata


def model_record_from_example(example: MultiViewTaskExample) -> Dict[str, object]:
    record = {
        "id": example.id,
        "views": [
            {
                "role": view.role,
                "text": view.text,
                "source_path": view.source_path,
                "source_type": view.source_type,
                "allowed_visibility": view.allowed_visibility,
            }
            for view in example.views
        ],
        "candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "text": candidate.text,
                "source_path": candidate.source_path,
                "patch_hash": candidate.patch_hash,
                "attributes": list(candidate.attributes),
            }
            for candidate in example.candidates
        ],
        "metadata": dict(example.metadata),
    }
    assert_no_oracle_only_keys_in_model_record(record)
    return record


def oracle_record_from_example(example: MultiViewTaskExample) -> Dict[str, object]:
    return {
        "id": example.id,
        "label": int(example.label),
        "oracle_metadata": dict(example_oracle_metadata(example)),
    }


def assert_no_oracle_only_keys_in_model_record(record: Dict[str, object]) -> None:
    hits: List[str] = []

    def walk(value: object, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                key_text = str(key)
                child_path = f"{path}.{key_text}" if path else key_text
                if key_text in ORACLE_ONLY_METADATA_KEYS or key_text in {"oracle_metadata", "oracle_record"}:
                    hits.append(child_path)
                walk(child, child_path)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(record, "")
    if hits:
        raise AssertionError(f"model-facing record contains oracle-only keys: {sorted(hits)}")


def build_multiview_code_patch_splits(
    config: MultiViewCodePatchDatasetConfig,
    seed: int,
    repo_root: str | Path = ".",
) -> Dict[str, List[MultiViewTaskExample]]:
    if config.num_candidates != 8:
        raise ValueError("the strict real-code benchmark expects exactly eight candidates")
    if config.n_views != 4:
        raise ValueError("the strict real-code benchmark expects exactly four private views")

    root = Path(repo_root)
    cases = discover_import_repair_cases(root, config)
    if not cases:
        cases = _fallback_import_repair_cases(root)
    split_cases = _partition_cases_by_target_file(cases, seed)
    all_cases = [case for rows in split_cases.values() for case in rows]
    counts = {"train": config.n_train, "dev": config.n_dev, "test": config.n_test}
    offsets = {"train": 0, "dev": 1_000_000, "test": 2_000_000}
    return {
        split: [
            _build_example(
                split=split,
                split_index=index,
                global_index=offsets[split] + index,
                case=_select_case_for_index(split_cases[split], split, offsets[split] + index, seed),
                split_cases=split_cases[split],
                all_cases=all_cases,
                config=config,
                seed=seed,
            )
            for index in range(count)
        ]
        for split, count in counts.items()
    }


def discover_local_code_files(root: Path, config: MultiViewCodePatchDatasetConfig) -> List[Path]:
    files: List[Path] = []
    for source_root in config.source_roots:
        base = root / source_root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            if path.name.startswith("run_stage33_"):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if len(text.splitlines()) >= 4:
                files.append(path)
            if len(files) >= config.max_files:
                return files
    return files


def discover_import_repair_cases(root: Path, config: MultiViewCodePatchDatasetConfig) -> List[ImportRepairCase]:
    files = discover_local_code_files(root, config)
    modules = {_module_name(path, root): path for path in files}
    exports = _build_export_index(files, root)
    cases: List[ImportRepairCase] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text)
        except (SyntaxError, UnicodeDecodeError):
            continue
        lines = text.splitlines()
        names = [(node.id, int(getattr(node, "lineno", 0))) for node in ast.walk(tree) if isinstance(node, ast.Name)]
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if not _is_local_module(node.module, modules):
                continue
            for alias in node.names:
                if alias.name == "*" or alias.asname is not None:
                    continue
                provider = exports.get((node.module, alias.name))
                if provider is None:
                    continue
                usage_lines = tuple(
                    sorted({line for name, line in names if name == alias.name and line and line != int(node.lineno)})[:8]
                )
                if not usage_lines:
                    continue
                rel_target = _relpath(path, root)
                import_statement = _line_at(lines, int(node.lineno))
                source_key = f"{rel_target}:{int(node.lineno)}:{node.module}:{alias.name}"
                cases.append(
                    ImportRepairCase(
                        target_path=path,
                        target_rel_path=rel_target,
                        import_module=node.module,
                        imported_name=alias.name,
                        alias=alias.name,
                        import_line=int(node.lineno),
                        import_statement=import_statement.strip(),
                        usage_lines=usage_lines,
                        usage_snippet=_line_snippets(lines, usage_lines[:4], radius=1),
                        import_block_snippet=_import_block_snippet(lines, int(node.lineno), config.snippet_radius),
                        provider=provider,
                        source_key=source_key,
                    )
                )
    cases.sort(key=lambda case: case.source_key)
    return cases


def apply_example_control(
    examples: Sequence[MultiViewTaskExample],
    condition: str,
    seed: int,
) -> List[MultiViewTaskExample]:
    if condition not in CONTROL_NAMES:
        raise ValueError(f"unknown control condition: {condition}")
    if condition in {
        "none",
        "hidden_states_shuffled_across_examples",
        "role_labels_shuffled",
        "physical_order_shuffled_roles_preserved",
    }:
        return list(examples)
    if condition == "view_masked":
        out = []
        for example in examples:
            views = tuple(
                replace(
                    view,
                    text=(
                        "Program-analysis artifact intentionally masked for the evidence-masking control.\n"
                        "Candidate patches remain visible, but the private analysis value for this view is withheld."
                    ),
                    source_type=f"{view.source_type}:masked",
                )
                for view in example.views
            )
            out.append(replace(example, views=views))
        return out
    if condition == "view_shuffled":
        rng = np.random.default_rng(seed + 31_771)
        out_views = [list(example.views) for example in examples]
        for role_id in range(len(ROLE_NAMES)):
            perm = _non_identity_permutation(len(examples), rng)
            for row_id, donor_id in enumerate(perm):
                donor = examples[int(donor_id)].views[role_id]
                original = out_views[row_id][role_id]
                out_views[row_id][role_id] = replace(
                    original,
                    text=donor.text,
                    source_path=donor.source_path,
                    source_type=f"{donor.source_type}:view_shuffled_donor",
                )
        return [replace(example, views=tuple(out_views[row_id])) for row_id, example in enumerate(examples)]
    if condition == "candidate_order_shuffled":
        rng = np.random.default_rng(seed + 91_113)
        out = []
        for example in examples:
            perm = _non_identity_permutation(len(example.candidates), rng)
            candidates = tuple(example.candidates[int(index)] for index in perm)
            gold_id = example.candidates[example.label].candidate_id
            new_label = next(index for index, candidate in enumerate(candidates) if candidate.candidate_id == gold_id)
            out.append(
                replace(
                    example,
                    candidates=candidates,
                    label=int(new_label),
                    metadata={**example.metadata, "candidate_order_control": [int(value) for value in perm.tolist()]},
                )
            )
        return out
    if condition == "randomized_labels":
        randomized = randomized_labels_for_examples(examples, seed=seed, num_classes=len(examples[0].candidates))
        return [replace(example, label=int(randomized[index])) for index, example in enumerate(examples)]
    raise ValueError(f"unhandled control condition: {condition}")


def randomized_labels_for_examples(
    examples: Sequence[MultiViewTaskExample],
    seed: int,
    num_classes: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed + 61_331)
    labels = np.tile(np.arange(num_classes, dtype=np.int64), int(np.ceil(len(examples) / num_classes)))
    labels = labels[: len(examples)].astype(np.int64, copy=True)
    rng.shuffle(labels)
    original = np.asarray([example.label for example in examples], dtype=np.int64)
    if len(labels) > 1 and np.array_equal(labels, original):
        labels = np.roll(labels, 1)
    return labels


def attempt_batch_from_examples(
    examples: Sequence[MultiViewTaskExample],
    split: str,
    hidden_dim: int = 1,
    hidden_states: np.ndarray | None = None,
) -> AttemptBatch:
    if not examples:
        raise ValueError("at least one example is required")
    n_agents = len(examples[0].views)
    labels = np.asarray([example.label for example in examples], dtype=np.int64)
    if hidden_states is None:
        hidden_states = np.zeros((len(examples), n_agents, 1, hidden_dim), dtype=np.float32)
    answers = np.zeros((len(examples), n_agents), dtype=np.int64)
    confidences = np.full((len(examples), n_agents), 1.0 / max(1, len(examples[0].candidates)), dtype=np.float32)
    visible_texts = [
        [
            (
                f"visible_summary=program_analysis_private_view_completed "
                f"candidate_count={len(example.candidates)} private_artifact_withheld=true"
            )
            for _view in example.views
        ]
        for example in examples
    ]
    return AttemptBatch(
        split=split,
        example_ids=np.asarray([example.id for example in examples], dtype=object),
        task_ids=np.zeros(len(examples), dtype=np.int64),
        labels=labels,
        answers=answers,
        confidences=confidences,
        hidden_states=hidden_states.astype(np.float32, copy=True),
        visible_texts=visible_texts,
        private_agent_views=[[view.text for view in example.views] for example in examples],
    )


def split_leakage_audit(
    benchmark: str,
    seed: int,
    splits: Dict[str, Sequence[MultiViewTaskExample]],
) -> Dict[str, object]:
    ids = {name: {example.id for example in rows} for name, rows in splits.items()}
    file_patch = {
        name: {
            (str(example_oracle_metadata(example).get("source_file")), str(example_oracle_metadata(example).get("gold_patch_hash")))
            for example in rows
        }
        for name, rows in splits.items()
    }
    candidate_hashes = {
        name: {candidate.patch_hash for example in rows for candidate in example.candidates}
        for name, rows in splits.items()
    }
    issue_hashes = {
        name: {str(example_oracle_metadata(example).get("issue_text_hash")) for example in rows}
        for name, rows in splits.items()
    }
    source_keys = {
        name: {str(example_oracle_metadata(example).get("source_import_key")) for example in rows}
        for name, rows in splits.items()
    }
    target_files = {
        name: {str(example_oracle_metadata(example).get("source_file")) for example in rows}
        for name, rows in splits.items()
    }
    families = {
        name: {str(example_oracle_metadata(example).get("problem_family")) for example in rows}
        for name, rows in splits.items()
    }
    return {
        "benchmark": benchmark,
        "seed": seed,
        "type": "real_shared_weight_split_leakage",
        "train_ids": [example.id for example in splits["train"]],
        "dev_ids": [example.id for example in splits["dev"]],
        "test_ids": [example.id for example in splits["test"]],
        "train_dev_id_overlap": len(ids["train"] & ids["dev"]),
        "train_test_id_overlap": len(ids["train"] & ids["test"]),
        "dev_test_id_overlap": len(ids["dev"] & ids["test"]),
        "train_dev_file_patch_overlap": len(file_patch["train"] & file_patch["dev"]),
        "train_test_file_patch_overlap": len(file_patch["train"] & file_patch["test"]),
        "dev_test_file_patch_overlap": len(file_patch["dev"] & file_patch["test"]),
        "train_dev_candidate_hash_overlap": len(candidate_hashes["train"] & candidate_hashes["dev"]),
        "train_test_candidate_hash_overlap": len(candidate_hashes["train"] & candidate_hashes["test"]),
        "dev_test_candidate_hash_overlap": len(candidate_hashes["dev"] & candidate_hashes["test"]),
        "train_dev_issue_hash_overlap": len(issue_hashes["train"] & issue_hashes["dev"]),
        "train_test_issue_hash_overlap": len(issue_hashes["train"] & issue_hashes["test"]),
        "dev_test_issue_hash_overlap": len(issue_hashes["dev"] & issue_hashes["test"]),
        "train_dev_source_import_key_overlap": len(source_keys["train"] & source_keys["dev"]),
        "train_test_source_import_key_overlap": len(source_keys["train"] & source_keys["test"]),
        "dev_test_source_import_key_overlap": len(source_keys["dev"] & source_keys["test"]),
        "train_dev_target_file_overlap": len(target_files["train"] & target_files["dev"]),
        "train_test_target_file_overlap": len(target_files["train"] & target_files["test"]),
        "dev_test_target_file_overlap": len(target_files["dev"] & target_files["test"]),
        "train_dev_problem_family_overlap": len(families["train"] & families["dev"]),
        "train_test_problem_family_overlap": len(families["train"] & families["test"]),
        "dev_test_problem_family_overlap": len(families["dev"] & families["test"]),
        "candidate_patch_hash_leakage_audit_passes": (
            len(candidate_hashes["train"] & candidate_hashes["dev"]) == 0
            and len(candidate_hashes["train"] & candidate_hashes["test"]) == 0
            and len(candidate_hashes["dev"] & candidate_hashes["test"]) == 0
        ),
    }


def output_leakage_audit(
    benchmark: str,
    seed: int,
    examples: Sequence[MultiViewTaskExample],
) -> Dict[str, object]:
    forbidden = (
        r"\bgold\b",
        r"\bcorrect\s*candidate\b",
        r"\blabel\s*[=:]\s*\d",
        r"\banswer\s*[=:]\s*\d",
        r"\bcandidate_id\b",
        r"\bprivate_evidence_bit\b",
        r"\brepair_cue\b",
    )
    visible_rows = []
    for example in examples:
        batch = attempt_batch_from_examples([example], split="audit")
        visible_rows.extend(batch.visible_texts[0])
        visible_rows.extend(candidate.text for candidate in example.candidates)
    hits: List[str] = []
    for pattern in forbidden:
        regex = re.compile(pattern, flags=re.IGNORECASE)
        if any(regex.search(text) for text in visible_rows):
            hits.append(pattern)
    return {
        "benchmark": benchmark,
        "seed": seed,
        "type": "real_shared_weight_output_leakage",
        "examples_checked": len(examples),
        "visible_text_or_candidate_forbidden_pattern_hits": hits,
        "text_only_private_views_excluded": True,
        "passes": not hits,
    }


def format_candidate_block(example: MultiViewTaskExample) -> str:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    rows = ["Patch candidates:"]
    for index, candidate in enumerate(example.candidates):
        rows.append(f"[{labels[index]}]\n{candidate.text}")
    return "\n".join(rows)


def format_clone_prompt(example: MultiViewTaskExample, role_index: int) -> str:
    view = example.views[role_index]
    return (
        f"Program-analysis artifact:\n{view.text}\n\n"
        f"{format_candidate_block(example)}"
    )


def format_full_context_prompt(example: MultiViewTaskExample) -> str:
    views = "\n\n".join(f"Artifact {index + 1}:\n{view.text}" for index, view in enumerate(example.views))
    return f"Full program-analysis context:\n{views}\n\n{format_candidate_block(example)}"


def format_partial_context_prompt(example: MultiViewTaskExample, role_index: int = 0) -> str:
    return format_clone_prompt(example, role_index)


def dataset_summary(splits: Dict[str, Sequence[MultiViewTaskExample]]) -> Dict[str, object]:
    all_examples = [example for rows in splits.values() for example in rows]
    source_files = sorted({str(example_oracle_metadata(example).get("source_file")) for example in all_examples})
    candidate_ambiguity = [
        int(example.metadata.get("single_view_candidate_ambiguity_min", example_oracle_metadata(example).get("single_view_candidate_ambiguity_min", 0)))
        for example in all_examples
    ]
    return {
        "dataset_source": all_examples[0].metadata.get("dataset_source") if all_examples else "none",
        "generator_version": all_examples[0].metadata.get("generator_version") if all_examples else "none",
        "private_cue_style": all_examples[0].metadata.get("private_cue_style") if all_examples else "none",
        "candidate_representation": all_examples[0].metadata.get("candidate_representation") if all_examples else "none",
        "hardening": all_examples[0].metadata.get("hardening") if all_examples else {},
        "program_analysis_artifacts": all_examples[0].metadata.get("program_analysis_artifacts") if all_examples else [],
        "split_sizes": {split: len(rows) for split, rows in splits.items()},
        "num_candidates": len(all_examples[0].candidates) if all_examples else 0,
        "num_views": len(all_examples[0].views) if all_examples else 0,
        "source_files_used": len(source_files),
        "source_file_examples": source_files[:10],
        "constructed_fallback": bool(all_examples[0].metadata.get("constructed_fallback", False)) if all_examples else False,
        "publishable_proof_dataset": False,
        "open_ended_code_repair_claim_allowed": False,
        "real_correct_patch_source": "existing repository import statement restored after deletion",
        "minimum_single_view_candidate_ambiguity": min(candidate_ambiguity) if candidate_ambiguity else 0,
        "problem_families_by_split": {
            split: sorted({str(example_oracle_metadata(example).get("problem_family")) for example in rows})
            for split, rows in splits.items()
        },
    }


def candidate_attribute_bits(example: MultiViewTaskExample, candidate: Candidate) -> List[int]:
    oracle = example_oracle_metadata(example)
    tuples = oracle.get("candidate_tuples")
    if isinstance(tuples, (list, tuple)):
        candidate_index = _candidate_index(example, candidate)
        if 0 <= candidate_index < len(tuples):
            values = tuples[candidate_index]
            pairs = oracle.get("attribute_value_pairs")
            if isinstance(values, (list, tuple)) and isinstance(pairs, (list, tuple)) and len(pairs) == 4:
                return _candidate_bits_from_pairs([str(value) for value in values[:4]], pairs)
    pairs = oracle.get("attribute_value_pairs")
    if not isinstance(pairs, (list, tuple)) or len(pairs) != 4:
        return [0, 0, 0, 0]
    bits: List[int] = []
    for index, value in enumerate(candidate.attributes):
        pair = list(pairs[index]) if isinstance(pairs[index], (list, tuple)) else ["", ""]
        if len(pair) < 2:
            bits.append(0)
        elif str(value) == str(pair[1]):
            bits.append(1)
        else:
            bits.append(0)
    return bits


def _candidate_index(example: MultiViewTaskExample, candidate: Candidate) -> int:
    for index, item in enumerate(example.candidates):
        if item.candidate_id == candidate.candidate_id:
            return index
    match = re.search(r"-cand-(\d+)$", str(candidate.candidate_id))
    return int(match.group(1)) if match else -1


def _build_example(
    split: str,
    split_index: int,
    global_index: int,
    case: ImportRepairCase,
    split_cases: Sequence[ImportRepairCase],
    all_cases: Sequence[ImportRepairCase],
    config: MultiViewCodePatchDatasetConfig,
    seed: int,
) -> MultiViewTaskExample:
    sanitized = _is_candidate_sanitized_config(config)
    balanced_34b = _is_balanced_34b_config(config)
    hardened = _is_hardened_config(config)
    multiview_v2 = _is_multiview_v2_config(config)
    if balanced_34b:
        candidate_representation = BALANCED_34B_CANDIDATE_REPRESENTATION
        hardened = True
        multiview_v2 = True
    elif sanitized:
        candidate_representation = SANITIZED_CANDIDATE_REPRESENTATION
        hardened = True
        multiview_v2 = True
    elif multiview_v2 and config.candidate_representation == "full_patch":
        candidate_representation = "hardened_redacted_v2"
    elif hardened and config.candidate_representation == "full_patch":
        candidate_representation = "hardened_redacted_v1"
    else:
        candidate_representation = config.candidate_representation
    digest = _digest_int(f"{seed}:{split}:{split_index}:{case.source_key}")
    evidence_bits = (
        _symbol_kind_bit(case.provider.kind),
        _provider_area_bit(case.import_module),
        _provider_name_parity_bit(case.import_module),
        _line_parity_bit(case.import_line),
    )
    alt_symbol = _alternate_symbol(case, all_cases, digest, desired_kind_bit=1 - evidence_bits[0])
    alt_file_case = _alternate_file_case(case, split_cases, digest >> 16, desired_parity_bit=1 - evidence_bits[3])
    code_words = MULTIVIEW_V2_CODE_WORDS if multiview_v2 else CODE_WORDS
    if balanced_34b:
        candidate_bit_rows = _balanced_34b_candidate_bit_rows(evidence_bits, digest, config.num_candidates)
    else:
        candidate_bit_rows = [
            tuple(evidence_bits[role_id] ^ code[role_id] for role_id in range(4))
            for code in code_words
        ]
    candidate_tuples = [
        tuple(str(ATTRIBUTE_VALUE_PAIRS[role_id][bits[role_id]]) for role_id in range(4))
        for bits in candidate_bit_rows
    ]
    candidate_patch_values = []
    candidate_import_lines = []
    for code_index, bits in enumerate(candidate_bit_rows):
        if hardened:
            patch_values, import_line = _hardened_candidate_patch_values(
                case=case,
                split_cases=split_cases,
                all_cases=all_cases,
                digest=(digest >> 8) + code_index * 7919,
                desired_bits=bits,
                evidence_bits=evidence_bits,
                alt_symbol=alt_symbol,
                alt_file_case=alt_file_case,
            )
        else:
            symbol = case.imported_name if bits[0] == evidence_bits[0] else alt_symbol
            module = (
                case.import_module
                if bits[1] == evidence_bits[1] and bits[2] == evidence_bits[2]
                else _alternate_module(case, all_cases, (digest >> 8) + code_index, desired_area_bit=bits[1], desired_name_parity_bit=bits[2])
            )
            target_file = case.target_rel_path if bits[3] == evidence_bits[3] else alt_file_case.target_rel_path
            patch_values = (symbol, module, "direct_from_import", target_file)
            import_line = case.import_line if target_file == case.target_rel_path else alt_file_case.import_line
        candidate_patch_values.append(patch_values)
        candidate_import_lines.append(import_line)
    desired_gold_position = split_index % config.num_candidates
    order = _candidate_order_with_gold_position(config.num_candidates, desired_gold_position, digest)
    ordered_tuples = [candidate_tuples[int(index)] for index in order]
    ordered_patch_values = [candidate_patch_values[int(index)] for index in order]
    ordered_import_lines = [candidate_import_lines[int(index)] for index in order]
    label = desired_gold_position
    example_id = f"{split}-{split_index:05d}"
    candidates = tuple(
        _candidate(
            example_id=example_id,
            candidate_index=index,
            patch_values=ordered_patch_values[index],
            attributes=ordered_tuples[index],
            import_line=ordered_import_lines[index],
            source_key=case.source_key,
            representation=candidate_representation,
        )
        for index in range(config.num_candidates)
    )
    if balanced_34b:
        candidates = tuple(
            _balanced_34b_candidate_for_model(example_id, index, candidate, ordered_tuples[index])
            for index, candidate in enumerate(candidates)
        )
    elif sanitized:
        candidates = tuple(_sanitize_candidate_for_model(example_id, index, candidate) for index, candidate in enumerate(candidates))
    if multiview_v2:
        views = _hardened_multiview_v2_views_for_case(case)
    elif hardened:
        views = _hardened_views_for_case(case)
    else:
        views = _views_for_case(case, alt_file_case)
    if balanced_34b:
        views = _balanced_34b_views_for_model(evidence_bits)
    elif sanitized:
        safe_view_values = tuple("FEATURE_BETA" if int(bit) else "FEATURE_ALPHA" for bit in evidence_bits)
        views = tuple(
            replace(
                view,
                text=(
                    "Redacted compatibility artifact:\n"
                    f"slot_value: {safe_view_values[index]}\n"
                    "identifier_strings_withheld: true"
                ),
                source_path=f"REDACTED_VIEW_SLOT_{index}",
                source_type=f"{view.source_type}:candidate_sanitized_v3",
            )
            for index, view in enumerate(views)
        )
    issue_text = views[0].text
    candidate_bit_tuples = [
        [int(bit) for bit in _candidate_bits_from_pairs(values, ATTRIBUTE_VALUE_PAIRS)]
        for values in ordered_tuples
    ]
    single_view_ambiguity = min(
        sum(1 for bits in candidate_bit_tuples if bits[role_id] == evidence_bits[role_id])
        for role_id in range(4)
    )
    oracle_metadata = {
        "attribute_names": list(ATTRIBUTE_NAMES),
        "attribute_value_pairs": [list(pair) for pair in ATTRIBUTE_VALUE_PAIRS],
        "candidate_bit_tuples": candidate_bit_tuples,
        "candidate_order": [int(value) for value in order],
        "candidate_patch_values": [list(values) for values in ordered_patch_values],
        "candidate_tuples": [list(values) for values in ordered_tuples],
        "evidence_bits": [int(value) for value in evidence_bits],
        "gold_import_statement": case.import_statement,
        "gold_patch_hash": candidates[label].patch_hash,
        "gold_tuple": list(candidate_tuples[0]),
        "issue_text_hash": _sha256(issue_text),
        "label": label,
        "line_number": case.import_line,
        "problem_family": f"{split}_real_import_restore_{_digest_int(case.target_rel_path) % 7}",
        "provider_file": case.provider.rel_path,
        "provider_module": case.import_module,
        "source_file": case.target_rel_path,
        "source_import_key": case.source_key,
        "single_view_candidate_ambiguity_min": int(single_view_ambiguity),
        "usage_lines": [int(value) for value in case.usage_lines],
    }
    if balanced_34b or sanitized:
        model_metadata = _candidate_sanitized_model_metadata(
            config=config,
            candidate_representation=candidate_representation,
            constructed_fallback=case.target_rel_path.startswith("virtual_"),
        )
        model_record, oracle_record = _build_model_oracle_records(
            example_id=example_id,
            views=views,
            candidates=candidates,
            model_metadata=model_metadata,
            oracle_metadata=oracle_metadata,
        )
        return MultiViewTaskExample(
            id=example_id,
            views=views,
            candidates=candidates,
            label=label,
            metadata=model_record["metadata"],  # type: ignore[index]
            oracle_metadata=oracle_record,
        )
    return MultiViewTaskExample(
        id=example_id,
        views=views,
        candidates=candidates,
        label=label,
        metadata={
            "dataset_source": config.dataset_source,
            "generator_version": config.generator_version,
            "candidate_representation": candidate_representation,
            "private_cue_style": (
                "real_import_restore_role_balanced_multiview_v2"
                if multiview_v2
                else
                "real_import_restore_redacted_typed_placeholders_v1"
                if hardened
                else "real_program_analysis_artifacts_no_constructed_cue_tokens"
            ),
            "hardening": (
                {
                    "exact_symbol_module_path_removed_from_private_views": True,
                    "candidate_names_removed_from_private_views": True,
                    "candidate_patch_text_redacted": True,
                    "candidate_query_uses_structured_nontext_features": True,
                    "single_role_candidate_group_size": 3 if multiview_v2 else 4,
                    "pairwise_decisive_role_pairs": ["0,1", "2,3"] if multiview_v2 else [],
                    "pairwise_codebook": "role_balanced_v2_binary_pair_intersections" if multiview_v2 else "even_parity_v1",
                    "distractors": "real import pairs sampled by same package family, symbol type, provider category, provider-name parity, usage pattern, and import-slot parity where possible",
                }
                if hardened
                else {}
            ),
            "program_analysis_artifacts": [
                "ast_import_deletion_name_usage",
                "module_export_resolution",
                "provider_module_name_parity_analysis",
                "traceback_import_block_location",
            ],
            "problem_family": f"{split}_real_import_restore_{_digest_int(case.target_rel_path) % 7}",
            "source_file": case.target_rel_path,
            "provider_module": case.import_module,
            "provider_file": case.provider.rel_path,
            "source_import_key": case.source_key,
            "line_number": case.import_line,
            "usage_lines": [int(value) for value in case.usage_lines],
            "evidence_bits": [int(value) for value in evidence_bits],
            "gold_tuple": list(candidate_tuples[0]),
            "candidate_tuples": [list(values) for values in ordered_tuples],
            "candidate_patch_values": [list(values) for values in ordered_patch_values],
            "gold_import_statement": case.import_statement,
            "candidate_bit_tuples": candidate_bit_tuples,
            "attribute_names": list(ATTRIBUTE_NAMES),
            "attribute_value_pairs": [list(pair) for pair in ATTRIBUTE_VALUE_PAIRS],
            "candidate_order": [int(value) for value in order],
            "label": label,
            "gold_patch_hash": candidates[label].patch_hash,
            "issue_text_hash": _sha256(issue_text),
            "constructed_fallback": case.target_rel_path.startswith("virtual_"),
            "real_correct_patch": True,
            "correct_patch_origin": "existing_import_statement_restoration",
            "construction": (
                "The gold patch restores a real import used by real AST Name loads. "
                "Each private view exposes one program-analysis artifact; no single view identifies one candidate. "
                "Stage 3.3 v2 uses two disjoint pairwise constraints over redacted structured candidate features."
            ),
            "single_view_candidate_ambiguity_min": int(single_view_ambiguity),
        },
    )


def _build_model_oracle_records(
    example_id: str,
    views: Sequence[View],
    candidates: Sequence[Candidate],
    model_metadata: Dict[str, object],
    oracle_metadata: Dict[str, object],
) -> Tuple[Dict[str, object], Dict[str, object]]:
    unsafe_keys = sorted(set(model_metadata) & ORACLE_ONLY_METADATA_KEYS)
    if unsafe_keys:
        raise AssertionError(f"sanitized model metadata contains oracle-only keys: {unsafe_keys}")
    extra_keys = sorted(set(model_metadata) - MODEL_RECORD_SAFE_METADATA_KEYS)
    if extra_keys:
        raise AssertionError(f"sanitized model metadata contains unapproved keys: {extra_keys}")
    missing = sorted(ORACLE_ONLY_METADATA_KEYS - set(oracle_metadata))
    if missing:
        raise AssertionError(f"sanitized oracle metadata missing required keys: {missing}")
    model_record = {
        "id": example_id,
        "views": [
            {
                "role": view.role,
                "text": view.text,
                "source_path": view.source_path,
                "source_type": view.source_type,
                "allowed_visibility": view.allowed_visibility,
            }
            for view in views
        ],
        "candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "text": candidate.text,
                "source_path": candidate.source_path,
                "patch_hash": candidate.patch_hash,
                "attributes": list(candidate.attributes),
            }
            for candidate in candidates
        ],
        "metadata": dict(model_metadata),
    }
    assert_no_oracle_only_keys_in_model_record(model_record)
    return model_record, dict(oracle_metadata)


def _candidate_sanitized_model_metadata(
    config: MultiViewCodePatchDatasetConfig,
    candidate_representation: str,
    constructed_fallback: bool,
) -> Dict[str, object]:
    balanced = candidate_representation == BALANCED_34B_CANDIDATE_REPRESENTATION
    return {
        "dataset_source": BALANCED_34B_DATASET_SOURCE if balanced else SANITIZED_DATASET_SOURCE,
        "generator_version": config.generator_version,
        "candidate_representation": candidate_representation,
        "private_cue_style": (
            "real_import_restore_candidate_balanced_34b_oracle_separated"
            if balanced
            else "real_import_restore_candidate_sanitized_v3_oracle_separated"
        ),
        "hardening": {
            "oracle_metadata_separated_from_model_record": True,
            "candidate_attributes_are_model_safe_opaque": not balanced,
            "candidate_query_structured_features_are_non_identifying": True,
            "model_candidate_text_has_no_symbol_module_path_or_codebook_labels": True,
            "role_pair_ids_removed_from_model_inputs": True,
            "family_ids_removed_from_model_inputs": True,
            "balanced_non_identifying_candidate_categories": balanced,
        },
        "program_analysis_artifacts": [
            "redacted_ast_usage_shape",
            "redacted_module_resolution_shape",
            "redacted_provider_shape",
            "redacted_import_slot_shape",
        ],
        "num_candidates": config.num_candidates,
        "n_views": config.n_views,
        "constructed_fallback": bool(constructed_fallback),
        "real_correct_patch": True,
        "correct_patch_origin": "existing_import_statement_restoration_oracle_only",
    }


def _sanitize_candidate_for_model(example_id: str, candidate_index: int, candidate: Candidate) -> Candidate:
    uid = _sha256(f"{example_id}:model-safe-candidate-slot:{candidate_index}")[:12]
    text = (
        f"Patch option {candidate_index}\n"
        "operation: RESTORE_LOCAL_IMPORT\n"
        "target: TARGET_REDACTED\n"
        "provider: PROVIDER_REDACTED\n"
        "name: NAME_REDACTED\n"
        "shape: from PROVIDER_REDACTED import NAME_REDACTED\n"
        "syntax_category: LOCAL_IMPORT_RESTORE_GENERIC\n"
        f"slot_marker: SLOT_{candidate_index}\n"
        f"sanitized_uid: {uid}"
    )
    return Candidate(
        candidate_id=candidate.candidate_id,
        text=text,
        source_path=f"REDACTED_CANDIDATE_SLOT_{candidate_index}",
        patch_hash=_sha256(text),
        attributes=("model_safe_opaque", "model_safe_opaque", "model_safe_opaque", "model_safe_opaque"),
    )


def _balanced_34b_candidate_for_model(
    example_id: str,
    candidate_index: int,
    candidate: Candidate,
    candidate_tuple: Sequence[str],
) -> Candidate:
    values = tuple(str(value) for value in candidate_tuple[:4])
    uid = _sha256(f"{example_id}:balanced-34b-candidate-slot:{candidate_index}")[:12]
    text = (
        f"Patch option {candidate_index}\n"
        "operation: RESTORE_LOCAL_IMPORT\n"
        "patch_shape: from PROVIDER_PLACEHOLDER import SYMBOL_PLACEHOLDER\n"
        f"symbol_surface: {_display_category(values[0])}\n"
        f"provider_area: {_display_category(values[1])}\n"
        f"provider_name_shape: {_display_category(values[2])}\n"
        f"import_slot_shape: {_display_category(values[3])}\n"
        "exact_symbol_module_path: withheld\n"
        f"sanitized_uid: {uid}"
    )
    return Candidate(
        candidate_id=candidate.candidate_id,
        text=text,
        source_path=f"REDACTED_CANDIDATE_SLOT_{candidate_index}",
        patch_hash=_sha256(text),
        attributes=values,  # model-facing balanced categories; no per-example oracle fields.
    )


def _balanced_34b_views_for_model(evidence_bits: Sequence[int]) -> Tuple[View, ...]:
    values = tuple(ATTRIBUTE_VALUE_PAIRS[role_id][int(evidence_bits[role_id])] for role_id in range(4))
    labels = ("symbol_surface", "provider_area", "provider_name_shape", "import_slot_shape")
    source_types = (
        "redacted_artifact_symbol_surface:balanced_34b",
        "redacted_artifact_provider_area:balanced_34b",
        "redacted_artifact_provider_name_shape:balanced_34b",
        "redacted_artifact_import_slot_shape:balanced_34b",
    )
    return tuple(
        View(
            role=ROLE_NAMES[index],
            text=(
                "Redacted compatibility artifact:\n"
                f"{labels[index]}: {_display_category(values[index])}\n"
                "exact_symbol_module_path: withheld"
            ),
            source_path=f"REDACTED_VIEW_SLOT_{index}",
            source_type=source_types[index],
        )
        for index in range(4)
    )


def _display_category(value: str) -> str:
    return str(value).replace("_", " ")


def _select_case_for_index(
    cases: Sequence[ImportRepairCase],
    split: str,
    global_index: int,
    seed: int,
) -> ImportRepairCase:
    digest = _digest_int(f"{seed}:{split}:case-select:{global_index}")
    target_bits = _balanced_bits(global_index, digest)
    exact = [case for case in cases if _case_bits(case) == target_bits]
    if exact:
        return exact[digest % len(exact)]
    ranked = sorted(
        cases,
        key=lambda case: (
            sum(int(a != b) for a, b in zip(_case_bits(case), target_bits)),
            _digest_int(f"{digest}:{case.source_key}"),
        ),
    )
    return ranked[0]


def _case_bits(case: ImportRepairCase) -> Tuple[int, int, int, int]:
    return (
        _symbol_kind_bit(case.provider.kind),
        _provider_area_bit(case.import_module),
        _provider_name_parity_bit(case.import_module),
        _line_parity_bit(case.import_line),
    )


def _views_for_case(case: ImportRepairCase, alt_file_case: ImportRepairCase) -> Tuple[View, ...]:
    symbol_kind = "class" if _symbol_kind_bit(case.provider.kind) else "function-or-value"
    provider_area = "coordination-or-experiment" if _provider_area_bit(case.import_module) else "core-library"
    provider_name_parity = "odd" if _provider_name_parity_bit(case.import_module) else "even"
    line_parity = "odd" if _line_parity_bit(case.import_line) else "even"
    return (
        View(
            role=ROLE_NAMES[0],
            text=(
                "AST/export artifact after deleting one existing import:\n"
                f"Reference token: `{case.imported_name}`\n"
                f"Resolved export kind from the provider AST: {symbol_kind}\n"
                f"Observed use lines in {case.target_rel_path}: {', '.join(str(line) for line in case.usage_lines[:5])}\n"
                f"{case.usage_snippet}"
            ),
            source_path=case.target_rel_path,
            source_type="ast_name_usage",
        ),
        View(
            role=ROLE_NAMES[1],
            text=(
                "Module-resolution artifact from the local repository index:\n"
                f"Provider module: `{case.import_module}`\n"
                f"Provider module area: {provider_area}\n"
                f"Provider file: {case.provider.rel_path}\n"
                f"{case.provider.snippet}"
            ),
            source_path=case.provider.rel_path,
            source_type="module_export_resolution",
        ),
        View(
            role=ROLE_NAMES[2],
            text=(
                "Provider-name artifact from the resolved provider path:\n"
                f"Provider module basename length parity: {provider_name_parity}\n"
                f"Resolved provider module: `{case.import_module}`\n"
                f"Deleted import statement shape: `{_redact_import_values(case.import_statement)}`"
            ),
            source_path=case.target_rel_path,
            source_type="provider_name_parity_analysis",
        ),
        View(
            role=ROLE_NAMES[3],
            text=(
                "Patch-location artifact from traceback and import-block analysis:\n"
                f"Traceback file requiring the restored binding: {case.target_rel_path}\n"
                f"Import block line parity: {line_parity}\n"
                f"Nearby non-failing file from the split: {alt_file_case.target_rel_path}\n"
                f"Import block around line {case.import_line}:\n{case.import_block_snippet}"
            ),
            source_path=case.target_rel_path,
            source_type="traceback_import_location",
        ),
    )


def _hardened_views_for_case(case: ImportRepairCase) -> Tuple[View, ...]:
    symbol_kind = "CLASS_SYMBOL" if _symbol_kind_bit(case.provider.kind) else "FUNCTION_SYMBOL"
    provider_area = "MODULE_CATEGORY_WORKFLOW" if _provider_area_bit(case.import_module) else "MODULE_CATEGORY_CORE"
    provider_name_parity = "PROVIDER_BASENAME_ODD" if _provider_name_parity_bit(case.import_module) else "PROVIDER_BASENAME_EVEN"
    line_parity = "IMPORT_SLOT_ODD" if _line_parity_bit(case.import_line) else "IMPORT_SLOT_EVEN"
    usage_pattern = _usage_pattern(case)
    return (
        View(
            role=ROLE_NAMES[0],
            text=(
                "AST deletion artifact with all concrete identifiers redacted:\n"
                f"Missing binding normalized type: {symbol_kind}\n"
                f"Use-site pattern: {usage_pattern}\n"
                "Observed private evidence is limited to typed placeholders: CALL_SITE, ATTRIBUTE_ACCESS, IMPORT_SLOT.\n"
                "Exact binding, alias, provider package, file path, and patch-option text are withheld."
            ),
            source_path=case.target_rel_path,
            source_type="ast_name_usage:redacted_v1",
        ),
        View(
            role=ROLE_NAMES[1],
            text=(
                "Local module-resolution artifact with paths normalized:\n"
                f"Provider area category: {provider_area}\n"
                f"Export surface category: {symbol_kind}\n"
                "Provider file path is represented only as MODULE_CATEGORY, not as a package or path string.\n"
                "Exact provider path, basename, binding, and patch-option names are withheld."
            ),
            source_path=case.provider.rel_path,
            source_type="module_export_resolution:redacted_v1",
        ),
        View(
            role=ROLE_NAMES[2],
            text=(
                "Provider-shape artifact with lexical identities removed:\n"
                f"Provider basename normalized parity: {provider_name_parity}\n"
                "Deleted import statement shape: from MODULE_CATEGORY import SYMBOL_KIND\n"
                "Concrete provider package, imported binding, alias, and patch-option strings are withheld."
            ),
            source_path=case.target_rel_path,
            source_type="provider_name_parity_analysis:redacted_v1",
        ),
        View(
            role=ROLE_NAMES[3],
            text=(
                "Traceback/import-slot artifact with target path normalized:\n"
                f"Restoration slot parity: {line_parity}\n"
                "Traceback target is represented as TARGET_FILE_CATEGORY only.\n"
                "Nearby import block is normalized to IMPORT_SLOT before CALL_SITE; exact file path and import text are withheld."
            ),
            source_path=case.target_rel_path,
            source_type="traceback_import_location:redacted_v1",
        ),
    )


def _hardened_multiview_v2_views_for_case(case: ImportRepairCase) -> Tuple[View, ...]:
    values = (
        "FEATURE_BETA" if _symbol_kind_bit(case.provider.kind) else "FEATURE_ALPHA",
        "FEATURE_BETA" if _provider_area_bit(case.import_module) else "FEATURE_ALPHA",
        "FEATURE_BETA" if _provider_name_parity_bit(case.import_module) else "FEATURE_ALPHA",
        "FEATURE_BETA" if _line_parity_bit(case.import_line) else "FEATURE_ALPHA",
    )

    def artifact(role_index: int, source_path: str, source_type: str) -> View:
        return View(
            role=ROLE_NAMES[role_index],
            text=(
                "Redacted program-analysis artifact:\n"
                f"Abstract compatibility value: {values[role_index]}\n"
                "Concrete identifiers, artifact family, package strings, file paths, and option text are withheld."
            ),
            source_path=source_path,
            source_type=source_type,
        )

    return (
        artifact(0, case.target_rel_path, "redacted_artifact_slot_0:multiview_v2"),
        artifact(1, case.provider.rel_path, "redacted_artifact_slot_1:multiview_v2"),
        artifact(2, case.target_rel_path, "redacted_artifact_slot_2:multiview_v2"),
        artifact(3, case.target_rel_path, "redacted_artifact_slot_3:multiview_v2"),
    )


def _candidate(
    example_id: str,
    candidate_index: int,
    patch_values: Tuple[str, str, str, str],
    attributes: Tuple[str, str, str, str],
    import_line: int,
    source_key: str,
    representation: str = "full_patch",
) -> Candidate:
    symbol, module, style, target_file = patch_values
    addition = _import_addition(symbol, module, style)
    if representation in {"hardened_redacted_v1", "hardened_redacted_v2"}:
        uid = _sha256(f"{example_id}:{candidate_index}:{source_key}:{symbol}:{module}:{target_file}")[:12]
        redaction_version = "v2" if representation == "hardened_redacted_v2" else "v1"
        text = (
            f"Patch candidate {candidate_index} redaction={redaction_version} redacted_uid={uid}\n"
            "operation: RESTORE_LOCAL_IMPORT\n"
            "target: TARGET_FILE_REDACTED\n"
            "module: MODULE_PATH_REDACTED\n"
            "symbol: IMPORTED_SYMBOL_REDACTED\n"
            "shape: from MODULE_CATEGORY import SYMBOL_KIND"
        )
    else:
        text = (
            f"diff --git a/{target_file} b/{target_file}\n"
            f"@@ import block around line {import_line} @@\n"
            f"+ {addition}"
        )
    return Candidate(
        candidate_id=f"{example_id}-cand-{candidate_index}",
        text=text,
        source_path=target_file,
        patch_hash=_sha256(text),
        attributes=attributes,
    )


def _import_addition(symbol: str, module: str, style: str) -> str:
    if style == "module_alias_import":
        return f"import {module} as {symbol}"
    return f"from {module} import {symbol}"


def _build_export_index(files: Sequence[Path], root: Path) -> Dict[Tuple[str, str], ExportInfo]:
    exports: Dict[Tuple[str, str], ExportInfo] = {}
    for path in files:
        module = _module_name(path, root)
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text)
        except (SyntaxError, UnicodeDecodeError):
            continue
        lines = text.splitlines()
        for node in tree.body:
            name = None
            kind = None
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = node.name
                kind = "function"
            elif isinstance(node, ast.ClassDef):
                name = node.name
                kind = "class"
            elif isinstance(node, ast.Assign) and node.targets and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                kind = "assignment"
            if not name or name.startswith("_"):
                continue
            line = int(getattr(node, "lineno", 1))
            exports[(module, name)] = ExportInfo(
                module=module,
                name=name,
                path=path,
                rel_path=_relpath(path, root),
                kind=str(kind),
                line=line,
                snippet=_snippet(lines, max(0, line - 1), radius=2),
            )
    return exports


def _partition_cases_by_target_file(cases: Sequence[ImportRepairCase], seed: int) -> Dict[str, List[ImportRepairCase]]:
    by_file: Dict[str, List[ImportRepairCase]] = {}
    for case in cases:
        by_file.setdefault(case.target_rel_path, []).append(case)
    files = sorted(by_file)
    files.sort(key=lambda value: _digest_int(f"{seed}:target-file:{value}"))
    if len(files) < 3:
        return {split: list(cases) for split in ("train", "dev", "test")}
    n_train = max(1, int(round(len(files) * 0.70)))
    n_dev = max(1, int(round(len(files) * 0.15)))
    if n_train + n_dev >= len(files):
        n_train = max(1, len(files) - 2)
        n_dev = 1
    split_files = {
        "train": set(files[:n_train]),
        "dev": set(files[n_train : n_train + n_dev]),
        "test": set(files[n_train + n_dev :]),
    }
    out = {
        split: [case for file_name in sorted(names) for case in by_file[file_name]]
        for split, names in split_files.items()
    }
    for split in ("train", "dev", "test"):
        if not out[split]:
            out[split] = list(cases)
    return out


def _fallback_import_repair_cases(root: Path) -> List[ImportRepairCase]:
    provider_path = root / "virtual_provider.py"
    target_path = root / "virtual_consumer.py"
    provider = ExportInfo(
        module="virtual_provider",
        name="VirtualThing",
        path=provider_path,
        rel_path="virtual_provider.py",
        kind="class",
        line=1,
        snippet="0001: class VirtualThing:\n0002:     pass",
    )
    return [
        ImportRepairCase(
            target_path=target_path,
            target_rel_path=f"virtual_consumer_{index}.py",
            import_module="virtual_provider",
            imported_name="VirtualThing",
            alias="VirtualThing",
            import_line=1,
            import_statement="from virtual_provider import VirtualThing",
            usage_lines=(4, 6),
            usage_snippet="0004: value = VirtualThing()\n0006: return isinstance(value, VirtualThing)",
            import_block_snippet="0001: from virtual_provider import VirtualThing",
            provider=provider,
            source_key=f"virtual_consumer_{index}.py:1:virtual_provider:VirtualThing",
        )
        for index in range(3)
    ]


def _is_hardened_config(config: MultiViewCodePatchDatasetConfig) -> bool:
    return str(config.dataset_source) in {
        "real_import_restore_hardened",
        "import_restore_redacted_v1",
        "real_import_restore_hardened_multiview_v2",
        "real_import_restore_role_balanced_v2",
        "import_restore_role_balanced_v2",
        SANITIZED_DATASET_SOURCE,
        BALANCED_34B_DATASET_SOURCE,
    } or str(
        config.generator_version
    ).startswith(("real_import_restore_hardened", "import_restore_redacted_v1", "import_restore_role_balanced_v2", SANITIZED_DATASET_SOURCE, BALANCED_34B_DATASET_SOURCE))


def _is_multiview_v2_config(config: MultiViewCodePatchDatasetConfig) -> bool:
    source = str(config.dataset_source)
    version = str(config.generator_version)
    representation = str(config.candidate_representation)
    return (
        source in {"real_import_restore_hardened_multiview_v2", "real_import_restore_role_balanced_v2", "import_restore_role_balanced_v2", SANITIZED_DATASET_SOURCE, BALANCED_34B_DATASET_SOURCE}
        or version.startswith(("real_import_restore_hardened_multiview_v2", "import_restore_role_balanced_v2", SANITIZED_DATASET_SOURCE, BALANCED_34B_DATASET_SOURCE))
        or representation in {"hardened_redacted_v2", SANITIZED_CANDIDATE_REPRESENTATION, BALANCED_34B_CANDIDATE_REPRESENTATION}
    )


def _is_candidate_sanitized_config(config: MultiViewCodePatchDatasetConfig) -> bool:
    return (
        str(config.dataset_source) == SANITIZED_DATASET_SOURCE
        or str(config.dataset_source) == BALANCED_34B_DATASET_SOURCE
        or str(config.generator_version).startswith(SANITIZED_DATASET_SOURCE)
        or str(config.generator_version).startswith(BALANCED_34B_DATASET_SOURCE)
        or str(config.candidate_representation) == SANITIZED_CANDIDATE_REPRESENTATION
        or str(config.candidate_representation) == BALANCED_34B_CANDIDATE_REPRESENTATION
    )


def _is_balanced_34b_config(config: MultiViewCodePatchDatasetConfig) -> bool:
    return (
        str(config.dataset_source) == BALANCED_34B_DATASET_SOURCE
        or str(config.generator_version).startswith(BALANCED_34B_DATASET_SOURCE)
        or str(config.candidate_representation) == BALANCED_34B_CANDIDATE_REPRESENTATION
    )


def _hardened_candidate_patch_values(
    case: ImportRepairCase,
    split_cases: Sequence[ImportRepairCase],
    all_cases: Sequence[ImportRepairCase],
    digest: int,
    desired_bits: Tuple[int, int, int, int],
    evidence_bits: Tuple[int, int, int, int],
    alt_symbol: str,
    alt_file_case: ImportRepairCase,
) -> Tuple[Tuple[str, str, str, str], int]:
    if desired_bits == evidence_bits:
        return (case.imported_name, case.import_module, "direct_from_import", case.target_rel_path), case.import_line

    pair_case = _alternate_import_pair_case(
        case=case,
        cases=all_cases,
        digest=digest,
        desired_kind_bit=desired_bits[0],
        desired_area_bit=desired_bits[1],
        desired_name_parity_bit=desired_bits[2],
    )
    symbol = pair_case.imported_name if pair_case is not None else (
        case.imported_name if desired_bits[0] == evidence_bits[0] else alt_symbol
    )
    module = pair_case.import_module if pair_case is not None else (
        case.import_module
        if desired_bits[1] == evidence_bits[1] and desired_bits[2] == evidence_bits[2]
        else _alternate_module(
            case,
            all_cases,
            digest,
            desired_area_bit=desired_bits[1],
            desired_name_parity_bit=desired_bits[2],
        )
    )
    if desired_bits[3] == evidence_bits[3]:
        target_case = case
    else:
        target_case = _alternate_file_case(case, split_cases, digest >> 5, desired_parity_bit=desired_bits[3])
        if target_case is case:
            target_case = alt_file_case
    return (symbol, module, "direct_from_import", target_case.target_rel_path), target_case.import_line


def _alternate_import_pair_case(
    case: ImportRepairCase,
    cases: Sequence[ImportRepairCase],
    digest: int,
    desired_kind_bit: int,
    desired_area_bit: int,
    desired_name_parity_bit: int,
) -> ImportRepairCase | None:
    strict = [
        other
        for other in cases
        if other.source_key != case.source_key
        and _symbol_kind_bit(other.provider.kind) == int(desired_kind_bit)
        and _provider_area_bit(other.import_module) == int(desired_area_bit)
        and _provider_name_parity_bit(other.import_module) == int(desired_name_parity_bit)
    ]
    relaxed = [
        other
        for other in cases
        if other.source_key != case.source_key and _symbol_kind_bit(other.provider.kind) == int(desired_kind_bit)
    ]
    pool = strict or relaxed or [other for other in cases if other.source_key != case.source_key]
    if not pool:
        return None
    base_family = _package_family(case.import_module)
    base_pattern = _usage_pattern(case)
    base_length = len(case.import_statement)
    ranked = sorted(
        pool,
        key=lambda other: (
            0 if _package_family(other.import_module) == base_family else 1,
            0 if _usage_pattern(other) == base_pattern else 1,
            abs(len(other.import_statement) - base_length),
            0 if other.import_module.split(".")[0] == case.import_module.split(".")[0] else 1,
            _digest_int(f"{digest}:{other.source_key}"),
        ),
    )
    return ranked[digest % min(len(ranked), max(1, min(16, len(ranked))))]


def _package_family(module: str) -> str:
    parts = str(module).split(".")
    if len(parts) >= 3:
        return ".".join(parts[:3])
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return str(module)


def _usage_pattern(case: ImportRepairCase) -> str:
    text = case.usage_snippet
    name = re.escape(case.imported_name)
    if re.search(rf"\b{name}\s*\(", text):
        return "CALL_SITE"
    if re.search(rf"\b{name}\s*\.", text) or re.search(rf"\.{name}\b", text):
        return "ATTRIBUTE_ACCESS"
    if _symbol_kind_bit(case.provider.kind):
        return "CLASS_SYMBOL_REFERENCE"
    return "FUNCTION_SYMBOL_REFERENCE"


def _alternate_symbol(
    case: ImportRepairCase,
    cases: Sequence[ImportRepairCase],
    digest: int,
    desired_kind_bit: int,
) -> str:
    same_module = sorted(
        {
            other.imported_name
            for other in cases
            if other.import_module == case.import_module
            and other.imported_name != case.imported_name
            and _symbol_kind_bit(other.provider.kind) == int(desired_kind_bit)
        }
    )
    all_symbols = sorted(
        {
            other.imported_name
            for other in cases
            if other.imported_name != case.imported_name and _symbol_kind_bit(other.provider.kind) == int(desired_kind_bit)
        }
    )
    pool = same_module or all_symbols or [f"{case.imported_name}Factory"]
    return pool[digest % len(pool)]


def _alternate_module(
    case: ImportRepairCase,
    cases: Sequence[ImportRepairCase],
    digest: int,
    desired_area_bit: int,
    desired_name_parity_bit: int,
) -> str:
    pool = sorted(
        {
            other.import_module
            for other in cases
            if other.import_module != case.import_module
            and _provider_area_bit(other.import_module) == int(desired_area_bit)
            and _provider_name_parity_bit(other.import_module) == int(desired_name_parity_bit)
        }
    )
    if not pool:
        return f"{case.import_module}.alternate"
    return pool[digest % len(pool)]


def _alternate_file_case(
    case: ImportRepairCase,
    cases: Sequence[ImportRepairCase],
    digest: int,
    desired_parity_bit: int,
) -> ImportRepairCase:
    pool = sorted(
        [other for other in cases if other.target_rel_path != case.target_rel_path and _line_parity_bit(other.import_line) == int(desired_parity_bit)],
        key=lambda other: other.source_key,
    )
    if not pool:
        pool = sorted([other for other in cases if other.target_rel_path != case.target_rel_path], key=lambda other: other.source_key)
    if not pool:
        return case
    return pool[digest % len(pool)]


def _symbol_kind_bit(kind: str) -> int:
    return 1 if str(kind) == "class" else 0


def _provider_area_bit(module: str) -> int:
    return 1 if str(module).startswith(("src.coordinators", "src.experiments")) else 0


def _provider_name_parity_bit(module: str) -> int:
    return len(str(module).rsplit(".", 1)[-1]) % 2


def _line_parity_bit(line_number: int) -> int:
    return int(line_number) % 2


def _candidate_bits_from_pairs(
    values: Sequence[str],
    pairs: Sequence[Sequence[str]],
) -> List[int]:
    bits: List[int] = []
    for value, pair in zip(values, pairs):
        bits.append(1 if len(pair) > 1 and str(value) == str(pair[1]) else 0)
    return bits


def _balanced_bits(global_index: int, digest: int) -> Tuple[int, int, int, int]:
    base = (global_index * 5 + 3 + (digest % 16)) % 16
    return tuple((base >> bit_id) & 1 for bit_id in range(4))


def _balanced_34b_candidate_bit_rows(
    evidence_bits: Sequence[int],
    digest: int,
    num_candidates: int,
) -> List[Tuple[int, int, int, int]]:
    if int(num_candidates) != 8:
        raise ValueError("balanced 3.4b candidate representation expects exactly eight candidates")
    gold = tuple(int(value) for value in evidence_bits[:4])
    universe = [tuple((value >> bit_id) & 1 for bit_id in range(4)) for value in range(16)]
    pool = [bits for bits in universe if bits != gold]
    pool.sort(key=lambda bits: _digest_int(f"{digest}:balanced-34b-distractor:{bits}"))
    return [gold] + pool[: int(num_candidates) - 1]


def _candidate_order_with_gold_position(num_candidates: int, gold_position: int, digest: int) -> np.ndarray:
    rng = np.random.default_rng(digest & 0xFFFFFFFF)
    distractors = np.arange(1, num_candidates, dtype=np.int64)
    rng.shuffle(distractors)
    order = np.empty(num_candidates, dtype=np.int64)
    order[int(gold_position)] = 0
    cursor = 0
    for slot in range(num_candidates):
        if slot == int(gold_position):
            continue
        order[slot] = distractors[cursor]
        cursor += 1
    return order


def _module_name(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    return ".".join(rel.parts)


def _is_local_module(module: str, modules: Dict[str, Path]) -> bool:
    return module in modules or any(name.startswith(f"{module}.") for name in modules)


def _line_at(lines: Sequence[str], line_number: int) -> str:
    index = max(0, min(len(lines) - 1, int(line_number) - 1))
    return lines[index] if lines else ""


def _line_snippets(lines: Sequence[str], line_numbers: Sequence[int], radius: int) -> str:
    seen = set()
    rows = []
    for line_number in line_numbers:
        center = max(0, int(line_number) - 1)
        start = max(0, center - radius)
        end = min(len(lines), center + radius + 1)
        for line_id in range(start, end):
            if line_id in seen:
                continue
            seen.add(line_id)
            rows.append(f"{line_id + 1:04d}: {lines[line_id]}")
    return "\n".join(rows)


def _import_block_snippet(lines: Sequence[str], line_number: int, radius: int) -> str:
    center = max(0, int(line_number) - 1)
    start = max(0, center - max(2, radius))
    end = min(len(lines), center + max(3, radius + 2))
    return "\n".join(f"{line_id + 1:04d}: {_redact_import_values(lines[line_id])}" for line_id in range(start, end))


def _redact_import_values(text: str) -> str:
    text = re.sub(r"from\s+[A-Za-z_][A-Za-z0-9_.]*\s+import\s+[A-Za-z_][A-Za-z0-9_,\s]*", "from <module> import <name>", text)
    text = re.sub(r"import\s+[A-Za-z_][A-Za-z0-9_.]*(\s+as\s+[A-Za-z_][A-Za-z0-9_]*)?", "import <module>", text)
    return text


def _snippet(lines: Sequence[str], center: int, radius: int) -> str:
    start = max(0, center - radius)
    end = min(len(lines), center + radius + 1)
    return "\n".join(f"{line_id + 1:04d}: {lines[line_id]}" for line_id in range(start, end))


def _relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _digest_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _non_identity_permutation(n: int, rng: np.random.Generator) -> np.ndarray:
    perm = rng.permutation(n)
    if n > 1 and np.array_equal(perm, np.arange(n)):
        perm = np.roll(perm, 1)
    return perm.astype(np.int64, copy=False)
