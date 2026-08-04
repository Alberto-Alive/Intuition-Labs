from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


PAD = 0
NOISE = 1
SET = 2
DISTRACTOR = 3
IGNORE = 4
QUERY = 5
QMARK = 6
ASSIGN = 7
EQ = 8
ASK = 9
ARROW = 10
REALSET = 11
FAKESET = 12
OVERWRITE = 13
FAKEOVERWRITE = 14
LINK = 15
FAKELINK = 16

KEY_OFFSET = 64
N_KEYS = 48
VALUE_OFFSET = 128
N_VALUES = 96
VOCAB_SIZE = VALUE_OFFSET + N_VALUES

TASKS: Tuple[str, ...] = (
    "password_overwrite",
    "variable_shadowing",
    "multi_hop_lookup",
    "matched_distractors",
)
TASK_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(TASKS)}


def key_token(key_id: int) -> int:
    return KEY_OFFSET + int(key_id % N_KEYS)


def value_token(value_id: int) -> int:
    return VALUE_OFFSET + int(value_id % N_VALUES)


def token_name(token_id: int) -> str:
    names = {
        PAD: "PAD",
        NOISE: "NOISE",
        SET: "SET",
        DISTRACTOR: "DISTRACTOR",
        IGNORE: "IGNORE",
        QUERY: "QUERY",
        QMARK: "?",
        ASSIGN: "ASSIGN",
        EQ: "=",
        ASK: "ASK",
        ARROW: "->",
        REALSET: "REALSET",
        FAKESET: "FAKESET",
        OVERWRITE: "OVERWRITE",
        FAKEOVERWRITE: "FAKEOVERWRITE",
        LINK: "LINK",
        FAKELINK: "FAKELINK",
    }
    if token_id in names:
        return names[token_id]
    if KEY_OFFSET <= token_id < KEY_OFFSET + N_KEYS:
        return f"k{token_id - KEY_OFFSET}"
    if VALUE_OFFSET <= token_id < VALUE_OFFSET + N_VALUES:
        return f"v{token_id - VALUE_OFFSET}"
    return f"tok{token_id}"


@dataclass(frozen=True)
class Statement:
    tokens: Tuple[int, ...]
    tag: str = "neutral"
    value_index: Optional[int] = None
    value_token_id: Optional[int] = None


@dataclass(frozen=True)
class SyntheticExample:
    input_ids: Tuple[int, ...]
    labels: Tuple[int, ...]
    task: str
    target_id: int
    query_pos: int
    relevant_positions: Tuple[int, ...]
    distractor_positions: Tuple[int, ...]
    distractor_count: int
    overwrite_depth: int
    seq_len: int

    def render(self) -> str:
        return " ".join(token_name(tok) for tok in self.input_ids)


@dataclass
class SyntheticBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    query_positions: torch.Tensor
    target_ids: torch.Tensor
    task_ids: torch.Tensor
    distractor_counts: torch.Tensor
    overwrite_depths: torch.Tensor
    relevant_positions: torch.Tensor
    distractor_positions: torch.Tensor
    tasks: Tuple[str, ...]
    examples: Tuple[SyntheticExample, ...]

    def to(self, device: torch.device | str) -> "SyntheticBatch":
        return SyntheticBatch(
            input_ids=self.input_ids.to(device),
            labels=self.labels.to(device),
            query_positions=self.query_positions.to(device),
            target_ids=self.target_ids.to(device),
            task_ids=self.task_ids.to(device),
            distractor_counts=self.distractor_counts.to(device),
            overwrite_depths=self.overwrite_depths.to(device),
            relevant_positions=self.relevant_positions.to(device),
            distractor_positions=self.distractor_positions.to(device),
            tasks=self.tasks,
            examples=self.examples,
        )


def is_heldout_pair(key_id: int, value_id: int) -> bool:
    return ((int(key_id) * 37 + int(value_id) * 17 + 11) % 7) == 0


def _rand_key(rng: random.Random, exclude: Iterable[int] = ()) -> int:
    excluded = set(exclude)
    candidates = [i for i in range(N_KEYS) if i not in excluded]
    return rng.choice(candidates)


def _rand_value_for_key(rng: random.Random, key_id: int, split: str) -> int:
    want_heldout = split != "train"
    for _ in range(512):
        value_id = rng.randrange(N_VALUES)
        if is_heldout_pair(key_id, value_id) == want_heldout:
            return value_id
    return rng.randrange(N_VALUES)


def _rand_any_value(rng: random.Random) -> int:
    return rng.randrange(N_VALUES)


def _noise_statement(rng: random.Random) -> Statement:
    shape = rng.randrange(4)
    if shape == 0:
        return Statement((NOISE,))
    if shape == 1:
        return Statement((NOISE, key_token(_rand_key(rng)), value_token(_rand_any_value(rng))))
    if shape == 2:
        return Statement((DISTRACTOR, key_token(_rand_key(rng)), value_token(_rand_any_value(rng))), "distractor", 2)
    return Statement((FAKESET, key_token(_rand_key(rng)), value_token(_rand_any_value(rng))), "distractor", 2)


def _suffix_noise(rng: random.Random) -> Statement:
    # Future tokens are never visible from the supervised query position, but
    # valid-looking future clutter prevents a fixed "query is final" shortcut.
    if rng.random() < 0.5:
        return _noise_statement(rng)
    key_id = _rand_key(rng)
    return Statement((SET, key_token(key_id), value_token(_rand_any_value(rng))), "future", 2)


def _password_statements(
    rng: random.Random,
    split: str,
    distractor_count: int,
    overwrite_depth: int,
) -> Tuple[List[Statement], int, int]:
    target_key = _rand_key(rng)
    statements: List[Statement] = []
    for _ in range(max(1, overwrite_depth)):
        val = _rand_value_for_key(rng, target_key, split)
        statements.append(
            Statement((SET, key_token(target_key), value_token(val)), "target_valid", 2, value_token(val))
        )
    for _ in range(distractor_count):
        mode = rng.randrange(4)
        if mode == 0:
            val = _rand_any_value(rng)
            statements.append(Statement((DISTRACTOR, key_token(target_key), value_token(val)), "distractor", 2))
        elif mode == 1:
            val = _rand_any_value(rng)
            statements.append(Statement((IGNORE, key_token(target_key), value_token(val)), "distractor", 2))
        else:
            other_key = _rand_key(rng, exclude=(target_key,))
            val = _rand_any_value(rng)
            statements.append(Statement((SET, key_token(other_key), value_token(val)), "distractor", 2))
    return statements, target_key, max(1, overwrite_depth)


def _shadow_statements(
    rng: random.Random,
    split: str,
    distractor_count: int,
    overwrite_depth: int,
) -> Tuple[List[Statement], int, int]:
    target_var = _rand_key(rng)
    statements: List[Statement] = []
    for _ in range(max(1, overwrite_depth)):
        val = _rand_value_for_key(rng, target_var, split)
        statements.append(
            Statement((ASSIGN, key_token(target_var), value_token(val)), "target_valid", 2, value_token(val))
        )
    for _ in range(distractor_count):
        mode = rng.randrange(3)
        if mode == 0:
            other_var = _rand_key(rng, exclude=(target_var,))
            statements.append(
                Statement((ASSIGN, key_token(other_var), value_token(_rand_any_value(rng))), "distractor", 2)
            )
        elif mode == 1:
            statements.append(
                Statement((FAKESET, key_token(target_var), value_token(_rand_any_value(rng))), "distractor", 2)
            )
        else:
            statements.append(_noise_statement(rng))
    return statements, target_var, max(1, overwrite_depth)


def _multihop_statements(
    rng: random.Random,
    split: str,
    distractor_count: int,
    overwrite_depth: int,
) -> Tuple[List[Statement], int, int, int]:
    start = _rand_key(rng)
    mid = _rand_key(rng, exclude=(start,))
    terminal_value = _rand_value_for_key(rng, start, split)
    statements: List[Statement] = [
        Statement((LINK, key_token(start), key_token(mid)), "chain_link", 2, key_token(mid)),
        Statement((LINK, key_token(mid), value_token(terminal_value)), "chain_value", 2, value_token(terminal_value)),
    ]
    excluded = {start, mid}
    for _ in range(distractor_count):
        mode = rng.randrange(4)
        if mode == 0:
            statements.append(
                Statement((FAKELINK, key_token(start), value_token(_rand_any_value(rng))), "distractor", 2)
            )
        else:
            src = _rand_key(rng, exclude=excluded if len(excluded) < N_KEYS - 1 else ())
            dst_key = _rand_key(rng, exclude=(src,))
            if mode == 1:
                statements.append(Statement((LINK, key_token(src), key_token(dst_key)), "distractor", 2))
            elif mode == 2:
                statements.append(
                    Statement((LINK, key_token(src), value_token(_rand_any_value(rng))), "distractor", 2)
                )
            else:
                statements.append(_noise_statement(rng))
    return statements, start, max(2, overwrite_depth), value_token(terminal_value)


def _matched_statements(
    rng: random.Random,
    split: str,
    distractor_count: int,
    overwrite_depth: int,
) -> Tuple[List[Statement], int, int]:
    real_key = _rand_key(rng)
    fake_key = _rand_key(rng, exclude=(real_key,))
    statements: List[Statement] = []
    first_val = _rand_value_for_key(rng, real_key, split)
    statements.append(Statement((REALSET, key_token(real_key), value_token(first_val)), "target_valid", 2, value_token(first_val)))
    for _ in range(max(0, overwrite_depth - 1)):
        val = _rand_value_for_key(rng, real_key, split)
        statements.append(
            Statement((OVERWRITE, key_token(real_key), value_token(val)), "target_valid", 2, value_token(val))
        )
    for _ in range(distractor_count):
        mode = rng.randrange(4)
        if mode == 0:
            statements.append(
                Statement((REALSET, key_token(fake_key), value_token(_rand_any_value(rng))), "distractor", 2)
            )
        elif mode == 1:
            statements.append(
                Statement((FAKEOVERWRITE, key_token(real_key), value_token(_rand_any_value(rng))), "distractor", 2)
            )
        elif mode == 2:
            statements.append(
                Statement((OVERWRITE, key_token(fake_key), value_token(_rand_any_value(rng))), "distractor", 2)
            )
        else:
            statements.append(_noise_statement(rng))
    return statements, real_key, max(1, overwrite_depth)


def _flatten_prefix(statements: Sequence[Statement]) -> Tuple[List[int], List[Tuple[str, int, Optional[int]]]]:
    tokens: List[int] = []
    tagged_positions: List[Tuple[str, int, Optional[int]]] = []
    for stmt in statements:
        start = len(tokens)
        tokens.extend(stmt.tokens)
        if stmt.value_index is not None:
            tagged_positions.append((stmt.tag, start + stmt.value_index, stmt.value_token_id))
    return tokens, tagged_positions


def _append_until(tokens: List[int], rng: random.Random, target_len: int, *, suffix: bool) -> None:
    while len(tokens) < target_len:
        stmt = _suffix_noise(rng) if suffix else _noise_statement(rng)
        remaining = target_len - len(tokens)
        tokens.extend(stmt.tokens[:remaining])


def generate_example(
    rng: random.Random,
    seq_len: int,
    split: str,
    *,
    task: Optional[str] = None,
    distractor_range: Tuple[int, int] = (4, 14),
    overwrite_range: Tuple[int, int] = (1, 4),
) -> SyntheticExample:
    if task is None:
        task = rng.choice(TASKS)
    if task not in TASK_TO_ID:
        raise ValueError(f"Unknown task '{task}'")
    if seq_len < 16:
        raise ValueError("seq_len must be at least 16 for the synthetic tasks")

    query_len = 3
    min_d, max_d = distractor_range
    min_o, max_o = overwrite_range
    min_d = max(0, int(min_d))
    max_d = max(min_d, int(max_d))
    min_o = max(1, int(min_o))
    max_o = max(min_o, int(max_o))

    for _attempt in range(64):
        suffix_budget = rng.randint(0, max(0, min(seq_len // 3, 48)))
        max_prefix_tokens = seq_len - query_len - suffix_budget
        overwrite_depth = rng.randint(min_o, max_o)
        distractor_count = rng.randint(min_d, max_d)
        # Keep retries cheap for short smoke tests while preserving the
        # configured high-clutter settings for normal 128+ token runs.
        max_stmt_count = max(2, max_prefix_tokens // 3 - overwrite_depth - 2)
        distractor_count = min(distractor_count, max_stmt_count)

        if task == "password_overwrite":
            statements, query_key, depth = _password_statements(rng, split, distractor_count, overwrite_depth)
            query_tokens = (QUERY, key_token(query_key), QMARK)
            fixed_answer: Optional[int] = None
        elif task == "variable_shadowing":
            statements, query_key, depth = _shadow_statements(rng, split, distractor_count, overwrite_depth)
            query_tokens = (ASK, key_token(query_key), QMARK)
            fixed_answer = None
        elif task == "multi_hop_lookup":
            statements, query_key, depth, answer = _multihop_statements(
                rng, split, distractor_count, overwrite_depth
            )
            query_tokens = (ASK, key_token(query_key), QMARK)
            fixed_answer = answer
        else:
            statements, query_key, depth = _matched_statements(rng, split, distractor_count, overwrite_depth)
            query_tokens = (ASK, key_token(query_key), QMARK)
            fixed_answer = None

        rng.shuffle(statements)
        prefix_tokens, tagged = _flatten_prefix(statements)
        if len(prefix_tokens) <= max_prefix_tokens:
            break
    else:
        raise RuntimeError("Failed to sample a fitting synthetic example")

    pre_noise_target = rng.randint(len(prefix_tokens), max_prefix_tokens)
    _append_until(prefix_tokens, rng, pre_noise_target, suffix=False)
    if rng.random() < 0.5:
        # Mix evidence and neutral clutter instead of always appending clutter
        # after the facts.
        prefix_statements = statements + [_noise_statement(rng) for _ in range(rng.randint(0, 3))]
        rng.shuffle(prefix_statements)
        prefix_tokens, tagged = _flatten_prefix(prefix_statements)
        if len(prefix_tokens) > max_prefix_tokens:
            prefix_tokens = prefix_tokens[:max_prefix_tokens]
            tagged = [item for item in tagged if item[1] < max_prefix_tokens]
        _append_until(prefix_tokens, rng, pre_noise_target, suffix=False)

    if fixed_answer is None:
        candidates = [(pos, tok) for tag, pos, tok in tagged if tag == "target_valid" and tok is not None]
        if not candidates:
            raise RuntimeError("Generated example has no valid target evidence")
        candidates.sort(key=lambda item: item[0])
        relevant_positions = (candidates[-1][0],)
        answer_token = int(candidates[-1][1])
    else:
        link_positions = [pos for tag, pos, _tok in tagged if tag in {"chain_link", "chain_value"}]
        relevant_positions = tuple(sorted(link_positions))
        answer_token = int(fixed_answer)

    distractor_positions = tuple(pos for tag, pos, _tok in tagged if tag == "distractor")

    input_ids = list(prefix_tokens)
    query_start = len(input_ids)
    input_ids.extend(query_tokens)
    query_pos = query_start + len(query_tokens) - 1
    suffix_target = seq_len
    _append_until(input_ids, rng, suffix_target, suffix=True)
    input_ids = input_ids[:seq_len]
    if len(input_ids) < seq_len:
        input_ids.extend([PAD] * (seq_len - len(input_ids)))

    labels = [-100] * seq_len
    labels[query_pos] = answer_token

    return SyntheticExample(
        input_ids=tuple(input_ids),
        labels=tuple(labels),
        task=task,
        target_id=answer_token,
        query_pos=query_pos,
        relevant_positions=tuple(pos for pos in relevant_positions if pos <= query_pos),
        distractor_positions=tuple(pos for pos in distractor_positions if pos <= query_pos),
        distractor_count=int(distractor_count),
        overwrite_depth=int(depth),
        seq_len=int(seq_len),
    )


def batch_examples(examples: Sequence[SyntheticExample]) -> SyntheticBatch:
    if not examples:
        raise ValueError("Cannot batch an empty example list")
    input_ids = torch.tensor([ex.input_ids for ex in examples], dtype=torch.long)
    labels = torch.tensor([ex.labels for ex in examples], dtype=torch.long)
    query_positions = torch.tensor([ex.query_pos for ex in examples], dtype=torch.long)
    target_ids = torch.tensor([ex.target_id for ex in examples], dtype=torch.long)
    task_ids = torch.tensor([TASK_TO_ID[ex.task] for ex in examples], dtype=torch.long)
    distractor_counts = torch.tensor([ex.distractor_count for ex in examples], dtype=torch.long)
    overwrite_depths = torch.tensor([ex.overwrite_depth for ex in examples], dtype=torch.long)

    max_rel = max(1, max(len(ex.relevant_positions) for ex in examples))
    max_dist = max(1, max(len(ex.distractor_positions) for ex in examples))
    relevant_positions = torch.full((len(examples), max_rel), -1, dtype=torch.long)
    distractor_positions = torch.full((len(examples), max_dist), -1, dtype=torch.long)
    for i, ex in enumerate(examples):
        if ex.relevant_positions:
            relevant_positions[i, : len(ex.relevant_positions)] = torch.tensor(ex.relevant_positions, dtype=torch.long)
        if ex.distractor_positions:
            distractor_positions[i, : len(ex.distractor_positions)] = torch.tensor(
                ex.distractor_positions, dtype=torch.long
            )

    return SyntheticBatch(
        input_ids=input_ids,
        labels=labels,
        query_positions=query_positions,
        target_ids=target_ids,
        task_ids=task_ids,
        distractor_counts=distractor_counts,
        overwrite_depths=overwrite_depths,
        relevant_positions=relevant_positions,
        distractor_positions=distractor_positions,
        tasks=tuple(ex.task for ex in examples),
        examples=tuple(examples),
    )


class SyntheticBatcher:
    def __init__(
        self,
        *,
        seed: int,
        split: str,
        seq_len: int,
        distractor_range: Tuple[int, int],
        overwrite_range: Tuple[int, int],
        tasks: Sequence[str] = TASKS,
    ) -> None:
        self.rng = random.Random(int(seed))
        self.split = split
        self.seq_len = int(seq_len)
        self.distractor_range = tuple(int(v) for v in distractor_range)
        self.overwrite_range = tuple(int(v) for v in overwrite_range)
        self.tasks = tuple(tasks)

    def sample(self, batch_size: int) -> SyntheticBatch:
        examples = [
            generate_example(
                self.rng,
                self.seq_len,
                self.split,
                task=self.rng.choice(self.tasks),
                distractor_range=self.distractor_range,
                overwrite_range=self.overwrite_range,
            )
            for _ in range(int(batch_size))
        ]
        return batch_examples(examples)


def make_batch(
    *,
    batch_size: int,
    seq_len: int,
    seed: int,
    split: str,
    distractor_range: Tuple[int, int],
    overwrite_range: Tuple[int, int],
    tasks: Sequence[str] = TASKS,
) -> SyntheticBatch:
    batcher = SyntheticBatcher(
        seed=seed,
        split=split,
        seq_len=seq_len,
        distractor_range=distractor_range,
        overwrite_range=overwrite_range,
        tasks=tasks,
    )
    return batcher.sample(batch_size)
