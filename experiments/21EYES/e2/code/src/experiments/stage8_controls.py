from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from dataclasses import replace
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from src.datasets.latent_attention_capacity_dataset import Stage8Example


REQUIRED_STAGE8_CONTROLS: Tuple[str, ...] = (
    "randomized_labels",
    "randomized_candidate_order_with_label_remap",
    "candidate_order_shuffle_with_label_remap",
    "randomized_evidence_block_order",
    "evidence_block_order_shuffle",
    "evidence_candidate_mismatch",
    "candidate_evidence_mismatch",
    "cross_task_evidence_shuffle",
    "cross_task_query_shuffle",
    "distractor_only",
    "candidate_only",
    "query_only",
    "evidence_only",
    "schema_template_only",
    "hidden_state_shuffle",
    "role_permutation",
    "avenue_permutation",
    "physical_token_order_shuffle",
    "evidence_position_randomization",
    "block_id_randomization",
    "memory_slot_permutation",
    "memory_shuffle_across_examples",
    "memory_disabled",
    "memory_gate_forced_closed",
    "memory_gate_forced_open",
)

DEGRADATION_CONTROLS: Tuple[str, ...] = (
    "randomized_labels",
    "evidence_candidate_mismatch",
    "candidate_evidence_mismatch",
    "cross_task_evidence_shuffle",
    "cross_task_query_shuffle",
    "distractor_only",
    "candidate_only",
    "query_only",
    "evidence_only",
    "schema_template_only",
    "hidden_state_shuffle",
    "memory_shuffle_across_examples",
    "memory_disabled",
    "memory_gate_forced_closed",
)

INVARIANCE_CONTROLS: Tuple[str, ...] = (
    "randomized_candidate_order_with_label_remap",
    "candidate_order_shuffle_with_label_remap",
    "randomized_evidence_block_order",
    "evidence_block_order_shuffle",
    "physical_token_order_shuffle",
    "evidence_position_randomization",
    "block_id_randomization",
    "role_permutation",
    "avenue_permutation",
    "memory_slot_permutation",
)


def apply_stage8_control(
    examples: Sequence[Stage8Example],
    control: str,
    seed: int,
) -> List[Stage8Example]:
    if control not in REQUIRED_STAGE8_CONTROLS:
        raise ValueError(f"unknown Stage 8 control: {control}")
    rng = random.Random(seed)
    if control == "randomized_labels":
        return [_with_random_label(example, rng, control) for example in examples]
    if control in {"randomized_candidate_order_with_label_remap", "candidate_order_shuffle_with_label_remap"}:
        return [_with_candidate_order_shuffle(example, rng, control) for example in examples]
    if control in {"randomized_evidence_block_order", "evidence_block_order_shuffle", "evidence_position_randomization", "block_id_randomization"}:
        return [_with_evidence_order_shuffle(example, rng, control) for example in examples]
    if control in {"evidence_candidate_mismatch", "candidate_evidence_mismatch"}:
        return _with_candidate_evidence_mismatch(examples, rng, control)
    if control == "cross_task_evidence_shuffle":
        return _with_cross_task_evidence_shuffle(examples, rng, control)
    if control == "cross_task_query_shuffle":
        return _with_cross_task_query_shuffle(examples, rng, control)
    if control == "distractor_only":
        return [_with_distractor_only(example, control) for example in examples]
    if control == "candidate_only":
        return [_with_candidate_only(example, control) for example in examples]
    if control == "query_only":
        return [_with_query_only(example, control) for example in examples]
    if control == "evidence_only":
        return [_with_evidence_only(example, control) for example in examples]
    if control == "schema_template_only":
        return [_with_schema_template_only(example, control) for example in examples]
    if control == "hidden_state_shuffle":
        return [_with_metadata_flag(example, control, hidden_state_shuffle=True) for example in examples]
    if control == "role_permutation":
        return [_with_metadata_flag(example, control, role_permutation=True) for example in examples]
    if control == "avenue_permutation":
        return [_with_metadata_flag(example, control, avenue_permutation=True) for example in examples]
    if control == "memory_slot_permutation":
        return [_with_metadata_flag(example, control, memory_slot_permutation=True) for example in examples]
    if control == "memory_shuffle_across_examples":
        return _with_memory_shuffle_across_examples(examples, rng, control)
    if control == "memory_disabled":
        return [_with_metadata_flag(example, control, memory_disabled=True) for example in examples]
    if control == "memory_gate_forced_closed":
        return [_with_metadata_flag(example, control, memory_gate_forced_closed=True) for example in examples]
    if control == "memory_gate_forced_open":
        return [_with_metadata_flag(example, control, memory_gate_forced_open=True) for example in examples]
    if control == "physical_token_order_shuffle":
        return [_with_physical_token_shuffle(example, rng, control) for example in examples]
    raise AssertionError(control)


def shortcut_audit(examples: Sequence[Stage8Example]) -> Dict[str, object]:
    """Check common shortcut channels before any Stage 8 claim is allowed."""
    failures: List[str] = []
    n = len(examples)
    labels = [example.label for example in examples if example.label >= 0]
    k = max((example.k_candidates for example in examples), default=8)
    label_hist = Counter(labels)
    max_label_fraction = max((count / max(1, len(labels)) for count in label_hist.values()), default=0.0)
    if n >= 32 and max_label_fraction > max(0.35, 2.5 / max(1, k)):
        failures.append("candidate index distribution is too skewed")

    template_label_max = _max_group_label_fraction(examples, lambda e: "|".join(e.template_ids))
    if n >= 48 and template_label_max > 0.55:
        failures.append("template id is predictive of label")

    namespace_label_max = _max_group_label_fraction(examples, lambda e: str(e.metadata.get("namespace", "")))
    if n >= 48 and namespace_label_max > 0.55:
        failures.append("entity namespace is predictive of label")

    relevant_positions = [
        index / max(1, example.n_blocks - 1)
        for example in examples
        if example.n_blocks > 1
        for index in example.relevant_block_indices
    ]
    relevant_position_mean = sum(relevant_positions) / max(1, len(relevant_positions))
    if n >= 48 and relevant_positions and (relevant_position_mean < 0.20 or relevant_position_mean > 0.80):
        failures.append("relevant evidence positions are skewed")

    length_by_label: Dict[int, List[int]] = defaultdict(list)
    for example in examples:
        if example.label >= 0:
            length_by_label[example.label].append(sum(len(block) for block in example.evidence_blocks))
    length_spread = 0.0
    if length_by_label:
        means = [sum(values) / len(values) for values in length_by_label.values() if values]
        if means:
            length_spread = max(means) - min(means)
            if n >= 128 and length_spread > 0.35 * max(1.0, sum(means) / len(means)):
                failures.append("evidence length is predictive of label")

    correct_text_markers = [
        example.example_id
        for example in examples
        if example.label >= 0 and _candidate_has_artifact(example.candidates[example.label])
    ]
    if correct_text_markers:
        failures.append("correct candidate text contains formatting artifact")

    n_blocks_by_split: Dict[str, set[int]] = defaultdict(set)
    for example in examples:
        n_blocks_by_split[example.split].add(example.n_blocks)
    if len(n_blocks_by_split) > 1 and any(len(values) == 1 for values in n_blocks_by_split.values()):
        split_values = {split: sorted(values) for split, values in n_blocks_by_split.items()}
    else:
        split_values = {split: sorted(values) for split, values in n_blocks_by_split.items()}

    return {
        "passes": not failures,
        "failures": failures,
        "num_examples": n,
        "candidate_index_histogram": [label_hist.get(i, 0) for i in range(k)],
        "max_candidate_index_fraction": max_label_fraction,
        "max_template_label_fraction": template_label_max,
        "max_namespace_label_fraction": namespace_label_max,
        "first_relevant_position_mean": relevant_position_mean,
        "evidence_length_mean_spread_by_label": length_spread,
        "correct_candidate_artifact_examples": correct_text_markers[:10],
        "n_blocks_by_split": split_values,
    }


def control_expectation(control: str) -> str:
    if control in DEGRADATION_CONTROLS:
        return "degrade"
    if control in INVARIANCE_CONTROLS:
        return "invariant"
    return "diagnostic"


def _with_random_label(example: Stage8Example, rng: random.Random, control: str) -> Stage8Example:
    label = rng.randrange(example.k_candidates)
    return replace(
        example,
        correct_indices=(label,),
        metadata={**dict(example.metadata), "control": control, "original_correct_indices": list(example.correct_indices)},
    )


def _with_candidate_order_shuffle(example: Stage8Example, rng: random.Random, control: str) -> Stage8Example:
    indexed = list(enumerate(example.candidates))
    rng.shuffle(indexed)
    original_to_new = {old: new for new, (old, _) in enumerate(indexed)}
    return replace(
        example,
        candidates=tuple(candidate for _, candidate in indexed),
        correct_indices=tuple(sorted(original_to_new[index] for index in example.correct_indices)),
        metadata={**dict(example.metadata), "control": control, "candidate_order_randomized": True},
    )


def _with_evidence_order_shuffle(example: Stage8Example, rng: random.Random, control: str) -> Stage8Example:
    indexed = list(enumerate(example.evidence_blocks))
    rng.shuffle(indexed)
    relevant = set(example.relevant_block_indices)
    return replace(
        example,
        evidence_blocks=tuple(block for _, block in indexed),
        relevant_block_indices=tuple(i for i, (old_index, _) in enumerate(indexed) if old_index in relevant),
        metadata={**dict(example.metadata), "control": control, "evidence_order_randomized": True},
    )


def _with_candidate_evidence_mismatch(
    examples: Sequence[Stage8Example],
    rng: random.Random,
    control: str,
) -> List[Stage8Example]:
    donors = list(examples)
    rng.shuffle(donors)
    output: List[Stage8Example] = []
    for index, example in enumerate(examples):
        donor = donors[index]
        if donor.example_id == example.example_id and len(donors) > 1:
            donor = donors[(index + 1) % len(donors)]
        output.append(
            replace(
                example,
                candidates=donor.candidates,
                correct_indices=(rng.randrange(example.k_candidates),),
                metadata={
                    **dict(example.metadata),
                    "control": control,
                    "candidate_donor_example_id": donor.example_id,
                    "original_correct_indices": list(example.correct_indices),
                },
            )
        )
    return output


def _with_cross_task_evidence_shuffle(
    examples: Sequence[Stage8Example],
    rng: random.Random,
    control: str,
) -> List[Stage8Example]:
    by_family: Dict[str, List[Stage8Example]] = defaultdict(list)
    for example in examples:
        by_family[example.task_family].append(example)
    output: List[Stage8Example] = []
    for example in examples:
        donor_pool = [candidate for family, rows in by_family.items() if family != example.task_family for candidate in rows]
        donor = rng.choice(donor_pool or list(examples))
        output.append(
            replace(
                example,
                evidence_blocks=donor.evidence_blocks,
                relevant_block_indices=(),
                metadata={**dict(example.metadata), "control": control, "evidence_donor_example_id": donor.example_id},
            )
        )
    return output


def _with_cross_task_query_shuffle(
    examples: Sequence[Stage8Example],
    rng: random.Random,
    control: str,
) -> List[Stage8Example]:
    output: List[Stage8Example] = []
    for example in examples:
        candidates = [row for row in examples if row.task_family != example.task_family]
        donor = rng.choice(candidates or list(examples))
        output.append(
            replace(
                example,
                query=donor.query,
                metadata={**dict(example.metadata), "control": control, "query_donor_example_id": donor.example_id},
            )
        )
    return output


def _with_memory_shuffle_across_examples(
    examples: Sequence[Stage8Example],
    rng: random.Random,
    control: str,
) -> List[Stage8Example]:
    donors = list(examples)
    rng.shuffle(donors)
    output: List[Stage8Example] = []
    for index, example in enumerate(examples):
        donor = donors[index]
        if donor.example_id == example.example_id and len(donors) > 1:
            donor = donors[(index + 1) % len(donors)]
        output.append(
            replace(
                example,
                evidence_blocks=donor.evidence_blocks,
                relevant_block_indices=(),
                metadata={
                    **dict(example.metadata),
                    "control": control,
                    "memory_donor_example_id": donor.example_id,
                    "memory_shuffle_across_examples": True,
                },
            )
        )
    return output


def _with_distractor_only(example: Stage8Example, control: str) -> Stage8Example:
    relevant = set(example.relevant_block_indices)
    answer = str(example.metadata.get("answer_token", ""))
    blocks = []
    for index, block in enumerate(example.evidence_blocks):
        if index not in relevant:
            blocks.append(block)
        else:
            blocks.append(block.replace(answer, "masked_stage8_value"))
    return replace(
        example,
        evidence_blocks=tuple(blocks),
        relevant_block_indices=(),
        metadata={**dict(example.metadata), "control": control, "answer_masked": True},
    )


def _with_candidate_only(example: Stage8Example, control: str) -> Stage8Example:
    return replace(
        example,
        evidence_blocks=(),
        relevant_block_indices=(),
        n_blocks=0,
        metadata={**dict(example.metadata), "control": control, "evidence_removed": True},
    )


def _with_query_only(example: Stage8Example, control: str) -> Stage8Example:
    generic = tuple(f"candidate slot {i}" for i in range(example.k_candidates))
    return replace(
        example,
        candidates=generic,
        evidence_blocks=(),
        relevant_block_indices=(),
        n_blocks=0,
        metadata={**dict(example.metadata), "control": control, "evidence_and_candidate_values_removed": True},
    )


def _with_evidence_only(example: Stage8Example, control: str) -> Stage8Example:
    generic = tuple(f"candidate slot {i}" for i in range(example.k_candidates))
    return replace(
        example,
        query="stage8 evidence only query mask",
        candidates=generic,
        metadata={**dict(example.metadata), "control": control, "query_and_candidate_values_removed": True},
    )


def _with_schema_template_only(example: Stage8Example, control: str) -> Stage8Example:
    return replace(
        example,
        query=_schema_mask(example.query),
        candidates=tuple(_schema_mask(candidate) for candidate in example.candidates),
        evidence_blocks=tuple(_schema_mask(block) for block in example.evidence_blocks),
        metadata={**dict(example.metadata), "control": control, "schema_masked": True},
    )


def _with_metadata_flag(example: Stage8Example, control: str, **flag: object) -> Stage8Example:
    return replace(example, metadata={**dict(example.metadata), "control": control, **flag})


def _with_physical_token_shuffle(example: Stage8Example, rng: random.Random, control: str) -> Stage8Example:
    return replace(
        example,
        query=_shuffle_words(example.query, rng),
        evidence_blocks=tuple(_shuffle_words(block, rng) for block in example.evidence_blocks),
        metadata={**dict(example.metadata), "control": control, "physical_token_order_shuffled": True},
    )


def _schema_mask(text: str) -> str:
    value = re.sub(r"\b(?:trn|dev|fin|xpl)_[a-z]+_[a-z]+_\d+_\d+\b", "<ID>", text)
    value = re.sub(r"\b\d+\b", "<NUM>", value)
    return value


def _shuffle_words(text: str, rng: random.Random) -> str:
    words = text.split()
    rng.shuffle(words)
    return " ".join(words)


def _max_group_label_fraction(examples: Sequence[Stage8Example], group_fn) -> float:
    grouped: Dict[str, Counter[int]] = defaultdict(Counter)
    for example in examples:
        if example.label >= 0:
            grouped[str(group_fn(example))][example.label] += 1
    fractions = []
    for hist in grouped.values():
        total = sum(hist.values())
        if total:
            fractions.append(max(hist.values()) / total)
    return max(fractions, default=0.0)


def _candidate_has_artifact(candidate: str) -> bool:
    lowered = candidate.lower()
    return any(marker in lowered for marker in ("correct", "gold", "answer=", "label="))
