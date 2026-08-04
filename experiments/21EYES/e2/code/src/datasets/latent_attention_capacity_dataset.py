from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


TASK_FAMILIES: Tuple[str, ...] = (
    "needle_binding",
    "multi_hop_binding",
    "constraint_satisfaction",
    "conflict_resolution",
    "compositional_role_evidence",
    "sparse_relevant_evidence",
)

SCALE_SCHEDULE: Tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512, 1024, 2048)


@dataclass(frozen=True)
class Stage8DatasetConfig:
    n_examples: int = 96
    n_blocks: int = 64
    k_candidates: int = 8
    split: str = "train"
    task_families: Tuple[str, ...] = TASK_FAMILIES
    min_chain_hops: int = 2
    max_chain_hops: int = 4
    min_sparse_relevant: int = 1
    max_sparse_relevant: int = 4
    distractor_similarity: str = "near_match"
    template_split: str | None = None


@dataclass(frozen=True)
class Stage8Example:
    example_id: str
    split: str
    task_family: str
    n_blocks: int
    query: str
    candidates: Tuple[str, ...]
    correct_indices: Tuple[int, ...]
    evidence_blocks: Tuple[str, ...]
    relevant_block_indices: Tuple[int, ...]
    template_ids: Tuple[str, ...]
    metadata: Mapping[str, object]

    @property
    def label(self) -> int:
        if not self.correct_indices:
            return -1
        return int(self.correct_indices[0])

    @property
    def k_candidates(self) -> int:
        return len(self.candidates)


def build_stage8_examples(config: Stage8DatasetConfig, seed: int) -> List[Stage8Example]:
    """Build one deterministic split of the Stage 8 capacity benchmark.

    The generator deliberately randomizes candidate order, evidence block order,
    entity/value namespaces, lexical templates, and distractor generators. Final
    validation can therefore use unseen template and namespace splits without
    reusing train/dev strings.
    """
    if config.k_candidates < 2:
        raise ValueError("k_candidates must be at least 2")
    if config.n_blocks < 1:
        raise ValueError("n_blocks must be positive")
    families = tuple(config.task_families)
    unknown = sorted(set(families) - set(TASK_FAMILIES))
    if unknown:
        raise ValueError(f"unknown Stage 8 task families: {unknown}")
    rng = random.Random(_stable_seed("stage8-dataset", seed, config.split, config.n_blocks, config.n_examples))
    examples: List[Stage8Example] = []
    for index in range(config.n_examples):
        family = families[index % len(families)]
        family_rng = random.Random(_stable_seed(seed, config.split, family, config.n_blocks, index))
        if family == "needle_binding":
            example = _build_needle_binding(config, family_rng, index)
        elif family == "multi_hop_binding":
            example = _build_multi_hop_binding(config, family_rng, index)
        elif family == "constraint_satisfaction":
            example = _build_constraint_satisfaction(config, family_rng, index)
        elif family == "conflict_resolution":
            example = _build_conflict_resolution(config, family_rng, index)
        elif family == "compositional_role_evidence":
            example = _build_compositional_role_evidence(config, family_rng, index)
        elif family == "sparse_relevant_evidence":
            example = _build_sparse_relevant_evidence(config, family_rng, index)
        else:  # pragma: no cover - guarded above.
            raise AssertionError(family)
        examples.append(example)
    rng.shuffle(examples)
    return [
        replace(example, example_id=f"{config.split}-N{config.n_blocks}-{i:05d}-{example.task_family}")
        for i, example in enumerate(examples)
    ]


def build_stage8_splits(
    n_blocks: int,
    seed: int,
    train_examples: int = 192,
    dev_examples: int = 96,
    final_examples: int = 128,
    k_candidates: int = 8,
    task_families: Sequence[str] = TASK_FAMILIES,
) -> Dict[str, List[Stage8Example]]:
    families = tuple(task_families)
    return {
        "train": build_stage8_examples(
            Stage8DatasetConfig(
                n_examples=train_examples,
                n_blocks=n_blocks,
                k_candidates=k_candidates,
                split="train",
                task_families=families,
                template_split="train",
            ),
            seed=seed,
        ),
        "dev": build_stage8_examples(
            Stage8DatasetConfig(
                n_examples=dev_examples,
                n_blocks=n_blocks,
                k_candidates=k_candidates,
                split="dev",
                task_families=families,
                template_split="dev",
            ),
            seed=seed + 10_000,
        ),
        "final": build_stage8_examples(
            Stage8DatasetConfig(
                n_examples=final_examples,
                n_blocks=n_blocks,
                k_candidates=k_candidates,
                split="final",
                task_families=families,
                template_split="final",
            ),
            seed=seed + 20_000,
        ),
    }


def example_to_dict(example: Stage8Example) -> Dict[str, object]:
    return {
        "example_id": example.example_id,
        "split": example.split,
        "task_family": example.task_family,
        "n_blocks": int(example.n_blocks),
        "query": example.query,
        "candidates": list(example.candidates),
        "correct_indices": list(example.correct_indices),
        "evidence_blocks": list(example.evidence_blocks),
        "relevant_block_indices": list(example.relevant_block_indices),
        "template_ids": list(example.template_ids),
        "metadata": dict(example.metadata),
    }


def example_from_dict(row: Mapping[str, object]) -> Stage8Example:
    return Stage8Example(
        example_id=str(row["example_id"]),
        split=str(row["split"]),
        task_family=str(row["task_family"]),
        n_blocks=int(row["n_blocks"]),
        query=str(row["query"]),
        candidates=tuple(str(value) for value in row["candidates"]),  # type: ignore[index]
        correct_indices=tuple(int(value) for value in row["correct_indices"]),  # type: ignore[index]
        evidence_blocks=tuple(str(value) for value in row["evidence_blocks"]),  # type: ignore[index]
        relevant_block_indices=tuple(int(value) for value in row["relevant_block_indices"]),  # type: ignore[index]
        template_ids=tuple(str(value) for value in row.get("template_ids", ())),  # type: ignore[union-attr]
        metadata=dict(row.get("metadata", {})),
    )


def write_stage8_jsonl(path: Path, examples: Sequence[Stage8Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example_to_dict(example), sort_keys=True) + "\n")


def load_stage8_jsonl(path: Path) -> List[Stage8Example]:
    with path.open("r", encoding="utf-8") as handle:
        return [example_from_dict(json.loads(line)) for line in handle if line.strip()]


def dataset_summary(examples: Sequence[Stage8Example]) -> Dict[str, object]:
    by_family: Dict[str, int] = {}
    by_split: Dict[str, int] = {}
    by_n: Dict[str, int] = {}
    candidate_hist = [0 for _ in range(max((example.k_candidates for example in examples), default=0))]
    for example in examples:
        by_family[example.task_family] = by_family.get(example.task_family, 0) + 1
        by_split[example.split] = by_split.get(example.split, 0) + 1
        by_n[str(example.n_blocks)] = by_n.get(str(example.n_blocks), 0) + 1
        if example.label >= 0:
            candidate_hist[example.label] += 1
    return {
        "num_examples": len(examples),
        "splits": by_split,
        "task_families": by_family,
        "n_blocks": by_n,
        "candidate_index_histogram": candidate_hist,
        "all_have_k8": all(example.k_candidates == 8 for example in examples),
        "all_have_randomized_order_flag": all(bool(example.metadata.get("candidate_order_randomized")) for example in examples),
    }


def validate_stage8_examples(examples: Sequence[Stage8Example]) -> Dict[str, object]:
    failures: List[str] = []
    seen_ids: set[str] = set()
    for example in examples:
        if example.example_id in seen_ids:
            failures.append(f"duplicate example_id {example.example_id}")
        seen_ids.add(example.example_id)
        if len(example.candidates) != 8:
            failures.append(f"{example.example_id}: expected K=8 candidates")
        if len(example.evidence_blocks) != example.n_blocks:
            failures.append(f"{example.example_id}: evidence block count mismatch")
        if not example.correct_indices:
            failures.append(f"{example.example_id}: no correct candidate")
        if any(index < 0 or index >= len(example.candidates) for index in example.correct_indices):
            failures.append(f"{example.example_id}: invalid correct index")
        if any(index < 0 or index >= len(example.evidence_blocks) for index in example.relevant_block_indices):
            failures.append(f"{example.example_id}: invalid relevant block index")
        if len(set(example.candidates)) != len(example.candidates):
            failures.append(f"{example.example_id}: duplicate candidate text")
        if "correct" in " ".join(example.candidates).lower():
            failures.append(f"{example.example_id}: candidate text contains correctness marker")
    summary = dataset_summary(examples)
    return {
        "passes": not failures,
        "failures": failures,
        "summary": summary,
    }


def _build_needle_binding(config: Stage8DatasetConfig, rng: random.Random, index: int) -> Stage8Example:
    family = "needle_binding"
    template = _template(config, family, rng)
    namespace = _namespace(config)
    entity = _name(namespace, "entity", index, rng)
    key = _name(namespace, "key", index, rng)
    value = _name(namespace, "value", index, rng)
    candidates, correct_indices = _candidate_list(config, rng, value, namespace, index)
    relevant = [
        template["evidence"].format(entity=entity, key=key, value=value, source=_name(namespace, "source", index, rng))
    ]
    distractors = []
    for slot in range(max(0, config.n_blocks - len(relevant))):
        d_entity = entity if slot % 3 == 0 else _name(namespace, "entity", index * 100 + slot + 1, rng)
        d_key = key if slot % 3 == 1 else _name(namespace, "key", index * 100 + slot + 1, rng)
        d_value = _name(namespace, "value", index * 100 + slot + 1, rng)
        distractors.append(
            template["distractor"].format(entity=d_entity, key=d_key, value=d_value, source=_name(namespace, "source", slot, rng))
        )
    evidence, relevant_indices = _shuffled_evidence(relevant, distractors, rng)
    return Stage8Example(
        example_id=f"pending-{family}-{index}",
        split=config.split,
        task_family=family,
        n_blocks=config.n_blocks,
        query=template["query"].format(entity=entity, key=key),
        candidates=tuple(template["candidate"].format(candidate=candidate) for candidate in candidates),
        correct_indices=correct_indices,
        evidence_blocks=evidence,
        relevant_block_indices=relevant_indices,
        template_ids=(template["id"],),
        metadata=_metadata(
            config,
            family,
            rng,
            target_entity=entity,
            target_key=key,
            answer_value=value,
            answer_token=value,
            relevant_count=len(relevant),
        ),
    )


def _build_multi_hop_binding(config: Stage8DatasetConfig, rng: random.Random, index: int) -> Stage8Example:
    family = "multi_hop_binding"
    template = _template(config, family, rng)
    namespace = _namespace(config)
    max_hops_for_n = max(1, config.n_blocks - 1)
    min_hops = min(config.min_chain_hops, max_hops_for_n)
    max_hops = max(min_hops, min(config.max_chain_hops, max_hops_for_n))
    hops = rng.randint(min_hops, max_hops)
    nodes = [_name(namespace, "node", index * 10 + i, rng) for i in range(hops + 1)]
    final_value = _name(namespace, "value", index, rng)
    candidates, correct_indices = _candidate_list(config, rng, final_value, namespace, index)
    relevant = []
    for hop in range(hops):
        relevant.append(template["link"].format(left=nodes[hop], right=nodes[hop + 1], relation=f"rel_{hop}"))
    relevant.append(template["final"].format(left=nodes[-1], value=final_value, relation=f"rel_{hops}"))
    distractors = []
    for slot in range(max(0, config.n_blocks - len(relevant))):
        left = nodes[slot % len(nodes)] if slot % 2 == 0 else _name(namespace, "node", index * 100 + slot, rng)
        right = _name(namespace, "node", index * 200 + slot, rng)
        value = _name(namespace, "value", index * 300 + slot, rng)
        if slot % 4 == 0:
            distractors.append(template["final"].format(left=left, value=value, relation=f"false_{slot % 5}"))
        else:
            distractors.append(template["link"].format(left=left, right=right, relation=f"false_{slot % 5}"))
    evidence, relevant_indices = _shuffled_evidence(relevant, distractors, rng)
    return Stage8Example(
        example_id=f"pending-{family}-{index}",
        split=config.split,
        task_family=family,
        n_blocks=config.n_blocks,
        query=template["query"].format(start=nodes[0], hops=hops),
        candidates=tuple(template["candidate"].format(candidate=candidate) for candidate in candidates),
        correct_indices=correct_indices,
        evidence_blocks=evidence,
        relevant_block_indices=relevant_indices,
        template_ids=(template["id"],),
        metadata=_metadata(
            config,
            family,
            rng,
            chain=list(nodes),
            chain_hops=hops,
            answer_value=final_value,
            answer_token=final_value,
            relevant_count=len(relevant),
        ),
    )


def _build_constraint_satisfaction(config: Stage8DatasetConfig, rng: random.Random, index: int) -> Stage8Example:
    family = "constraint_satisfaction"
    template = _template(config, family, rng)
    namespace = _namespace(config)
    items = [_name(namespace, "item", index * 10 + i, rng) for i in range(config.k_candidates)]
    correct_item = items[rng.randrange(len(items))]
    attr_names = ("color", "shape", "zone")
    attr_count = max(1, min(len(attr_names), config.n_blocks // 2))
    constraints = {
        attr: _name(namespace, attr, index, rng)
        for attr in attr_names[:attr_count]
    }
    relevant = [template["constraint"].format(attr=attr, value=value) for attr, value in constraints.items()]
    relevant.extend(template["fact"].format(item=correct_item, attr=attr, value=value) for attr, value in constraints.items())
    distractors = []
    for slot in range(max(0, config.n_blocks - len(relevant))):
        item = items[slot % len(items)]
        attr = list(constraints)[slot % len(constraints)]
        value = constraints[attr] if item != correct_item and slot % 2 == 0 else _name(namespace, attr, index * 100 + slot, rng)
        distractors.append(template["fact"].format(item=item, attr=attr, value=value))
    evidence, relevant_indices = _shuffled_evidence(relevant, distractors, rng)
    shuffled_items, correct_indices = _shuffle_candidates(items, correct_item, rng)
    return Stage8Example(
        example_id=f"pending-{family}-{index}",
        split=config.split,
        task_family=family,
        n_blocks=config.n_blocks,
        query=template["query"].format(goal=_name(namespace, "goal", index, rng)),
        candidates=tuple(template["candidate"].format(candidate=item) for item in shuffled_items),
        correct_indices=correct_indices,
        evidence_blocks=evidence,
        relevant_block_indices=relevant_indices,
        template_ids=(template["id"],),
        metadata=_metadata(
            config,
            family,
            rng,
            constraints=constraints,
            answer_item=correct_item,
            answer_token=correct_item,
            relevant_count=len(relevant),
        ),
    )


def _build_conflict_resolution(config: Stage8DatasetConfig, rng: random.Random, index: int) -> Stage8Example:
    family = "conflict_resolution"
    template = _template(config, family, rng)
    namespace = _namespace(config)
    entity = _name(namespace, "case", index, rng)
    source_count = max(1, min(config.k_candidates, max(1, config.n_blocks // 2)))
    sources = [_name(namespace, "source", index * 10 + i, rng) for i in range(source_count)]
    values = [_name(namespace, "value", index * 10 + i, rng) for i in range(source_count)]
    winning_source_index = rng.randrange(len(sources))
    priority_order = [winning_source_index] + [i for i in range(len(sources)) if i != winning_source_index]
    winning_source = sources[winning_source_index]
    correct_value = values[winning_source_index]
    relevant = [
        template["priority"].format(source=sources[source_index], rank=rank)
        for rank, source_index in enumerate(priority_order, start=1)
    ]
    relevant.extend(template["claim"].format(source=source, entity=entity, value=value) for source, value in zip(sources, values))
    distractors = []
    for slot in range(max(0, config.n_blocks - len(relevant))):
        source = _name(namespace, "source", index * 100 + slot, rng)
        value = _name(namespace, "value", index * 100 + slot, rng)
        distractors.append(template["claim"].format(source=source, entity=entity if slot % 3 == 0 else _name(namespace, "case", slot, rng), value=value))
    evidence, relevant_indices = _shuffled_evidence(relevant, distractors, rng)
    candidates, correct_indices = _candidate_list(config, rng, correct_value, namespace, index, pool=values)
    return Stage8Example(
        example_id=f"pending-{family}-{index}",
        split=config.split,
        task_family=family,
        n_blocks=config.n_blocks,
        query=template["query"].format(entity=entity, source=winning_source),
        candidates=tuple(template["candidate"].format(candidate=candidate) for candidate in candidates),
        correct_indices=correct_indices,
        evidence_blocks=evidence,
        relevant_block_indices=relevant_indices,
        template_ids=(template["id"],),
        metadata=_metadata(
            config,
            family,
            rng,
            target_entity=entity,
            winning_source=winning_source,
            answer_value=correct_value,
            answer_token=correct_value,
            relevant_count=len(relevant),
        ),
    )


def _build_compositional_role_evidence(config: Stage8DatasetConfig, rng: random.Random, index: int) -> Stage8Example:
    family = "compositional_role_evidence"
    template = _template(config, family, rng)
    namespace = _namespace(config)
    candidates = [_name(namespace, "action", index * 10 + i, rng) for i in range(config.k_candidates)]
    correct = candidates[rng.randrange(len(candidates))]
    goal = _name(namespace, "goal", index, rng)
    rule = _name(namespace, "rule", index, rng)
    local_fact = _name(namespace, "fact", index, rng)
    exception = _name(namespace, "exception", index, rng)
    all_relevant = [
        template["goal"].format(goal=goal, action=correct),
        template["rule"].format(rule=rule, action=correct),
        template["fact"].format(fact=local_fact, action=correct),
        template["exception"].format(exception=exception, action=_name(namespace, "action", index * 20 + 1, rng)),
    ]
    relevant = all_relevant[: max(1, min(len(all_relevant), config.n_blocks))]
    distractors = []
    roles = ("goal", "rule", "fact", "exception")
    for slot in range(max(0, config.n_blocks - len(relevant))):
        role = roles[slot % len(roles)]
        action = candidates[slot % len(candidates)]
        if action == correct:
            action = _name(namespace, "action", index * 100 + slot, rng)
        distractors.append(template[role].format(goal=goal, rule=rule, fact=local_fact, exception=exception, action=action))
    evidence, relevant_indices = _shuffled_evidence(relevant, distractors, rng)
    shuffled_candidates, correct_indices = _shuffle_candidates(candidates, correct, rng)
    return Stage8Example(
        example_id=f"pending-{family}-{index}",
        split=config.split,
        task_family=family,
        n_blocks=config.n_blocks,
        query=template["query"].format(goal=goal, rule=rule, fact=local_fact, exception=exception),
        candidates=tuple(template["candidate"].format(candidate=candidate) for candidate in shuffled_candidates),
        correct_indices=correct_indices,
        evidence_blocks=evidence,
        relevant_block_indices=relevant_indices,
        template_ids=(template["id"],),
        metadata=_metadata(
            config,
            family,
            rng,
            goal=goal,
            rule=rule,
            local_fact=local_fact,
            exception=exception,
            answer_action=correct,
            answer_token=correct,
            relevant_count=len(relevant),
        ),
    )


def _build_sparse_relevant_evidence(config: Stage8DatasetConfig, rng: random.Random, index: int) -> Stage8Example:
    family = "sparse_relevant_evidence"
    template = _template(config, family, rng)
    namespace = _namespace(config)
    max_relevant = max(1, min(config.max_sparse_relevant, config.n_blocks))
    min_relevant = max(1, min(config.min_sparse_relevant, max_relevant))
    relevant_count = rng.randint(min_relevant, max_relevant)
    value = _name(namespace, "value", index, rng)
    anchors = [_name(namespace, "anchor", index * 10 + i, rng) for i in range(relevant_count)]
    candidates, correct_indices = _candidate_list(config, rng, value, namespace, index)
    relevant = [
        template["signal"].format(anchor=anchor, value=value, clue=_name(namespace, "clue", index * 10 + i, rng))
        for i, anchor in enumerate(anchors)
    ]
    distractors = []
    for slot in range(max(0, config.n_blocks - len(relevant))):
        anchor = anchors[slot % len(anchors)] if slot % 4 == 0 else _name(namespace, "anchor", index * 100 + slot, rng)
        distractor_value = _name(namespace, "value", index * 200 + slot, rng)
        distractors.append(template["distractor"].format(anchor=anchor, value=distractor_value, clue=_name(namespace, "clue", slot, rng)))
    evidence, relevant_indices = _shuffled_evidence(relevant, distractors, rng)
    return Stage8Example(
        example_id=f"pending-{family}-{index}",
        split=config.split,
        task_family=family,
        n_blocks=config.n_blocks,
        query=template["query"].format(anchors=", ".join(anchors), count=relevant_count),
        candidates=tuple(template["candidate"].format(candidate=candidate) for candidate in candidates),
        correct_indices=correct_indices,
        evidence_blocks=evidence,
        relevant_block_indices=relevant_indices,
        template_ids=(template["id"],),
        metadata=_metadata(
            config,
            family,
            rng,
            anchors=anchors,
            answer_value=value,
            answer_token=value,
            relevant_count=len(relevant),
        ),
    )


def _candidate_list(
    config: Stage8DatasetConfig,
    rng: random.Random,
    correct_value: str,
    namespace: str,
    index: int,
    pool: Sequence[str] | None = None,
) -> Tuple[Tuple[str, ...], Tuple[int, ...]]:
    values = list(pool or [])
    if correct_value not in values:
        values.append(correct_value)
    slot = 0
    while len(values) < config.k_candidates:
        candidate = _name(namespace, "value", index * 1000 + slot, rng)
        if candidate not in values:
            values.append(candidate)
        slot += 1
    values = values[: config.k_candidates]
    if correct_value not in values:
        values[-1] = correct_value
    return _shuffle_candidates(values, correct_value, rng)


def _shuffle_candidates(candidates: Sequence[str], correct: str, rng: random.Random) -> Tuple[Tuple[str, ...], Tuple[int, ...]]:
    rows = list(candidates)
    rng.shuffle(rows)
    return tuple(rows), tuple(i for i, candidate in enumerate(rows) if candidate == correct)


def _shuffled_evidence(relevant: Sequence[str], distractors: Sequence[str], rng: random.Random) -> Tuple[Tuple[str, ...], Tuple[int, ...]]:
    rows = [(text, True) for text in relevant] + [(text, False) for text in distractors]
    rng.shuffle(rows)
    evidence = tuple(text for text, _ in rows)
    relevant_indices = tuple(i for i, (_, is_relevant) in enumerate(rows) if is_relevant)
    return evidence, relevant_indices


def _metadata(config: Stage8DatasetConfig, family: str, rng: random.Random, **values: object) -> Dict[str, object]:
    return {
        **values,
        "task_family": family,
        "split": config.split,
        "template_split": _template_split(config),
        "namespace": _namespace(config),
        "candidate_order_randomized": True,
        "evidence_order_randomized": True,
        "distractor_similarity": config.distractor_similarity,
        "block_id_randomization_token": _name(_namespace(config), "blockrand", rng.randrange(1_000_000), rng),
    }


def _template(config: Stage8DatasetConfig, family: str, rng: random.Random) -> Dict[str, str]:
    split = _template_split(config)
    choices = _TEMPLATES[split][family]
    return dict(rng.choice(choices))


def _template_split(config: Stage8DatasetConfig) -> str:
    split = config.template_split or config.split
    if split in _TEMPLATES:
        return split
    if split in {"test", "heldout", "stage8c"}:
        return "final"
    return "dev" if split.startswith("dev") else "train"


def _namespace(config: Stage8DatasetConfig) -> str:
    split = config.split
    if split in {"final", "stage8c", "heldout"}:
        return "fin"
    if split in {"extrapolation", "larger_n"}:
        return "xpl"
    if split == "dev":
        return "dev"
    return "trn"


def _name(namespace: str, kind: str, index: int, rng: random.Random) -> str:
    syllables = ("ka", "lo", "mi", "zen", "tor", "val", "rix", "sun", "pel", "dra", "nav", "cor")
    return f"{namespace}_{kind}_{syllables[rng.randrange(len(syllables))]}_{index}_{rng.randrange(10_000):04d}"


def _stable_seed(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") & 0x7FFFFFFF


_TEMPLATES: Dict[str, Dict[str, List[Dict[str, str]]]] = {
    "train": {
        "needle_binding": [
            {
                "id": "train_needle_0",
                "query": "Find the value bound to entity {entity} under key {key}.",
                "evidence": "Registry says entity {entity} has key {key} with value {value} from {source}.",
                "distractor": "Registry says entity {entity} has key {key} with value {value} from {source}.",
                "candidate": "candidate value {candidate}",
            },
            {
                "id": "train_needle_1",
                "query": "Which candidate matches the {key} slot for {entity}?",
                "evidence": "Binding record: {entity} :: {key} :: {value} :: {source}.",
                "distractor": "Binding record: {entity} :: {key} :: {value} :: {source}.",
                "candidate": "candidate value {candidate}",
            },
        ],
        "multi_hop_binding": [
            {
                "id": "train_hop_0",
                "query": "Starting at {start}, follow {hops} links and return the final value.",
                "link": "Hop fact states {left} maps through {relation} to {right}.",
                "final": "Terminal fact states {left} maps through {relation} to value {value}.",
                "candidate": "candidate value {candidate}",
            }
        ],
        "constraint_satisfaction": [
            {
                "id": "train_constraint_0",
                "query": "Select the item satisfying every listed constraint for goal {goal}.",
                "constraint": "Required constraint: attribute {attr} must equal {value}.",
                "fact": "Item fact: {item} has attribute {attr} equal to {value}.",
                "candidate": "candidate item {candidate}",
            }
        ],
        "conflict_resolution": [
            {
                "id": "train_conflict_0",
                "query": "For {entity}, apply the priority table and choose the winning value.",
                "priority": "Priority table: source {source} has rank {rank}.",
                "claim": "Claim table: source {source} reports {entity} as value {value}.",
                "candidate": "candidate value {candidate}",
            }
        ],
        "compositional_role_evidence": [
            {
                "id": "train_role_0",
                "query": "Choose the action matching goal {goal}, rule {rule}, fact {fact}, and exception {exception}.",
                "goal": "User-goal role: goal {goal} supports action {action}.",
                "rule": "Rule role: rule {rule} permits action {action}.",
                "fact": "Local-fact role: fact {fact} confirms action {action}.",
                "exception": "Exception role: exception {exception} blocks action {action}.",
                "candidate": "candidate action {candidate}",
            }
        ],
        "sparse_relevant_evidence": [
            {
                "id": "train_sparse_0",
                "query": "Use the {count} anchor clues {anchors} to choose the shared value.",
                "signal": "Sparse signal: anchor {anchor} with clue {clue} points to value {value}.",
                "distractor": "Sparse near-match: anchor {anchor} with clue {clue} points to value {value}.",
                "candidate": "candidate value {candidate}",
            }
        ],
    },
    "dev": {
        "needle_binding": [
            {
                "id": "dev_needle_0",
                "query": "Resolve key {key} for entity {entity}; pick the matching value.",
                "evidence": "Audit row from {source}: value {value} is attached to {entity} at key {key}.",
                "distractor": "Audit row from {source}: value {value} is attached to {entity} at key {key}.",
                "candidate": "candidate value {candidate}",
            }
        ],
        "multi_hop_binding": [
            {
                "id": "dev_hop_0",
                "query": "Trace {hops} mapping steps from {start} and report the terminal value.",
                "link": "Mapping edge {relation}: {left} continues to {right}.",
                "final": "Mapping edge {relation}: {left} terminates at value {value}.",
                "candidate": "candidate value {candidate}",
            }
        ],
        "constraint_satisfaction": [
            {
                "id": "dev_constraint_0",
                "query": "Pick the item that meets all constraints for {goal}.",
                "constraint": "Constraint ledger requires {attr} to be {value}.",
                "fact": "Item ledger lists {item} with {attr} set to {value}.",
                "candidate": "candidate item {candidate}",
            }
        ],
        "conflict_resolution": [
            {
                "id": "dev_conflict_0",
                "query": "Choose the top-priority value asserted for {entity}.",
                "priority": "Ordering note: {source} receives priority rank {rank}.",
                "claim": "Observation note: {source} assigns {entity} the value {value}.",
                "candidate": "candidate value {candidate}",
            }
        ],
        "compositional_role_evidence": [
            {
                "id": "dev_role_0",
                "query": "Select the action supported by {goal}, {rule}, {fact}, while avoiding {exception}.",
                "goal": "Goal-view record: {goal} favors action {action}.",
                "rule": "Rule-view record: {rule} authorizes action {action}.",
                "fact": "Fact-view record: {fact} verifies action {action}.",
                "exception": "Exception-view record: {exception} rejects action {action}.",
                "candidate": "candidate action {candidate}",
            }
        ],
        "sparse_relevant_evidence": [
            {
                "id": "dev_sparse_0",
                "query": "Find the value jointly indicated by anchors {anchors}.",
                "signal": "Needle clue {clue}: anchor {anchor} indicates value {value}.",
                "distractor": "Near clue {clue}: anchor {anchor} indicates value {value}.",
                "candidate": "candidate value {candidate}",
            }
        ],
    },
    "final": {
        "needle_binding": [
            {
                "id": "final_needle_0",
                "query": "Return the option whose value is recorded for {entity} and property {key}.",
                "evidence": "Heldout catalog entry from {source}: property {key} on {entity} stores {value}.",
                "distractor": "Heldout catalog entry from {source}: property {key} on {entity} stores {value}.",
                "candidate": "candidate value {candidate}",
            }
        ],
        "multi_hop_binding": [
            {
                "id": "final_hop_0",
                "query": "Advance from {start} across {hops} relations and choose the resulting value.",
                "link": "Heldout relation {relation} carries {left} onward to {right}.",
                "final": "Heldout relation {relation} carries {left} onward to terminal value {value}.",
                "candidate": "candidate value {candidate}",
            }
        ],
        "constraint_satisfaction": [
            {
                "id": "final_constraint_0",
                "query": "Identify the item satisfying the complete requirement set for {goal}.",
                "constraint": "Requirement sheet: field {attr} needs value {value}.",
                "fact": "Candidate sheet: item {item} has field {attr} value {value}.",
                "candidate": "candidate item {candidate}",
            }
        ],
        "conflict_resolution": [
            {
                "id": "final_conflict_0",
                "query": "Use source priority evidence for {entity}; select the reported value.",
                "priority": "Authority sheet: source {source} is priority {rank}.",
                "claim": "Authority claim: source {source} labels {entity} with value {value}.",
                "candidate": "candidate value {candidate}",
            }
        ],
        "compositional_role_evidence": [
            {
                "id": "final_role_0",
                "query": "Combine goal {goal}, rule {rule}, fact {fact}, and exception {exception} to select an action.",
                "goal": "Heldout goal lane: {goal} selects action {action}.",
                "rule": "Heldout rule lane: {rule} enables action {action}.",
                "fact": "Heldout fact lane: {fact} validates action {action}.",
                "exception": "Heldout exception lane: {exception} disallows action {action}.",
                "candidate": "candidate action {candidate}",
            }
        ],
        "sparse_relevant_evidence": [
            {
                "id": "final_sparse_0",
                "query": "From sparse anchors {anchors}, choose the only shared value.",
                "signal": "Heldout sparse marker {clue}: {anchor} supports value {value}.",
                "distractor": "Heldout sparse foil {clue}: {anchor} supports value {value}.",
                "candidate": "candidate value {candidate}",
            }
        ],
    },
}
