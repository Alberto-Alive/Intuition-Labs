from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import string
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from datasets import load_dataset
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


EXPERIMENT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_PATH = EXPERIMENT_ROOT / "results" / "stage15b_question_routed_multihop_qa_clone_collaboration_results.json"
REPORT_PATH = EXPERIMENT_ROOT / "reports" / "STAGE15B_QUESTION_ROUTED_MULTIHOP_QA_CLONE_COLLABORATION.md"
VARIANT_SUMMARY_PATH = (
    EXPERIMENT_ROOT / "reports" / "STAGE15B_QUESTION_ROUTED_MULTIHOP_QA_CLONE_COLLABORATION_VARIANT_SUMMARY.md"
)
STAGE15_REPORT_PATH = EXPERIMENT_ROOT / "reports" / "STAGE15_MULTIHOP_QA_CLONE_COLLABORATION.md"

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
UNK_TOKEN = "<unk>"
SEP_TOKEN = "<sep>"
SPECIAL_TOKENS = (PAD_TOKEN, BOS_TOKEN, UNK_TOKEN, SEP_TOKEN)
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|==|!=|<=|>=|[-+*/%(){}\[\].,:;?]")
NEG_INF = -1.0e4


@dataclass(frozen=True)
class CorpusConfig:
    n_train: int = 256
    n_dev: int = 64
    n_test: int = 64
    vocab_size: int = 8192
    question_max_length: int = 32
    window_size: int = 32
    answer_max_length: int = 16
    dataset_name: str = "hotpot_qa"
    dataset_config: str = "distractor"
    example_filter: str = "all"


@dataclass(frozen=True)
class ModelConfig:
    num_clones: int = 4
    hidden_size: int = 64
    num_layers: int = 3
    num_heads: int = 2
    ff_dim: int = 128
    dropout: float = 0.1
    identity_init_std: float = 0.02
    routing_mode: str = "learned"
    router_temperature: float = 1.0


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 0
    device: str = "cpu"
    early_stopping_patience: int = 6
    gradient_clip: float = 1.0
    use_bf16: bool = True


@dataclass(frozen=True)
class Stage15bConfig:
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


@dataclass(frozen=True)
class Vocabulary:
    stoi: Dict[str, int]
    itos: List[str]

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK_TOKEN]

    @property
    def sep_id(self) -> int:
        return self.stoi[SEP_TOKEN]

    @property
    def size(self) -> int:
        return len(self.itos)

    def encode_tokens(self, tokens: Sequence[str]) -> List[int]:
        return [self.stoi.get(token, self.unk_id) for token in tokens]


@dataclass(frozen=True)
class PreparedHotpotExample:
    question_tokens: List[str]
    context_tokens: List[str]
    answer_tokens: List[str]
    answer_text: str
    answer_start: int
    answer_end: int
    answer_window: int | None
    support_windows: List[int]


class HotpotQADataset(Dataset):
    def __init__(
        self,
        rows: Sequence[PreparedHotpotExample],
        vocab: Vocabulary,
        question_max_length: int,
        context_length: int,
        window_size: int,
    ) -> None:
        examples: List[Dict[str, object]] = []
        for row in rows:
            question_tokens = row.question_tokens[:question_max_length]
            question_ids = vocab.encode_tokens(question_tokens)
            question_pad = question_max_length - len(question_ids)
            question_segment = question_ids + [vocab.pad_id] * max(0, question_pad)

            context_tokens = row.context_tokens[:context_length]
            context_ids = vocab.encode_tokens(context_tokens)
            context_pad = context_length - len(context_ids)
            context_segment = context_ids + [vocab.pad_id] * max(0, context_pad)

            input_ids = question_segment + [vocab.sep_id] + context_segment
            attention_mask = (
                [1] * len(question_ids)
                + [0] * max(0, question_pad)
                + [1]
                + [1] * len(context_ids)
                + [0] * max(0, context_pad)
            )
            context_mask = [1] * len(context_ids) + [0] * max(0, context_pad)
            examples.append(
                {
                    "input_ids": torch.as_tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.as_tensor(attention_mask, dtype=torch.bool),
                    "context_mask": torch.as_tensor(context_mask, dtype=torch.bool),
                    "start_positions": torch.as_tensor(int(row.answer_start), dtype=torch.long),
                    "end_positions": torch.as_tensor(int(row.answer_end), dtype=torch.long),
                    "answer_text": row.answer_text,
                    "answer_window": -1 if row.answer_window is None else int(row.answer_window),
                    "support_windows": list(int(value) for value in row.support_windows),
                    "question_tokens": list(question_tokens),
                    "context_tokens": list(context_tokens),
                    "window_size": int(window_size),
                }
            )
        self.rows = examples
        self.question_max_length = int(question_max_length)
        self.context_length = int(context_length)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return self.rows[index]


def _collate_hotpot(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "context_mask": torch.stack([item["context_mask"] for item in batch]),
        "start_positions": torch.stack([item["start_positions"] for item in batch]),
        "end_positions": torch.stack([item["end_positions"] for item in batch]),
        "answer_text": [str(item["answer_text"]) for item in batch],
        "answer_window": torch.as_tensor([int(item["answer_window"]) for item in batch], dtype=torch.long),
        "support_windows": [list(int(value) for value in item["support_windows"]) for item in batch],
        "question_tokens": [list(item["question_tokens"]) for item in batch],
        "context_tokens": [list(item["context_tokens"]) for item in batch],
        "window_size": int(batch[0]["window_size"]),
    }


def _text_tokens(text: str) -> List[str]:
    return TOKEN_RE.findall(str(text).lower())


def _flatten_hotpot_context_with_spans(context: Dict[str, Sequence[Sequence[str]]]) -> tuple[List[str], List[Dict[str, int | str]]]:
    tokens: List[str] = []
    spans: List[Dict[str, int | str]] = []
    titles = list(context.get("title", []))
    sentences = list(context.get("sentences", []))
    for title, sentence_list in zip(titles, sentences):
        title_text = str(title).strip()
        title_tokens = _text_tokens(title_text)
        normalized_sentences = [str(sentence).strip() for sentence in sentence_list]
        added_title = False
        for sentence_index, sentence in enumerate(normalized_sentences):
            sentence_tokens = _text_tokens(sentence)
            if not sentence_tokens:
                continue
            start = len(tokens)
            if title_tokens and not added_title:
                tokens.extend(title_tokens)
                tokens.append(":")
                added_title = True
            tokens.extend(sentence_tokens)
            end = len(tokens)
            spans.append(
                {
                    "title": title_text.lower(),
                    "sentence_index": int(sentence_index),
                    "start": int(start),
                    "end": int(end),
                }
            )
    return tokens, spans


def _find_subsequence(haystack: Sequence[str], needle: Sequence[str]) -> int | None:
    if not needle or len(needle) > len(haystack):
        return None
    limit = len(haystack) - len(needle) + 1
    for start in range(limit):
        if list(haystack[start : start + len(needle)]) == list(needle):
            return start
    return None


def _extract_support_windows(
    raw_example: Dict[str, object],
    sentence_spans: Sequence[Dict[str, int | str]],
    context_length: int,
    window_size: int,
) -> List[int]:
    windows: set[int] = set()
    supporting = raw_example.get("supporting_facts", {})
    titles = list(supporting.get("title", [])) if isinstance(supporting, dict) else []
    sentence_ids = list(supporting.get("sent_id", [])) if isinstance(supporting, dict) else []
    span_lookup = {
        (str(span["title"]).lower(), int(span["sentence_index"])): span
        for span in sentence_spans
    }
    for title, sentence_index in zip(titles, sentence_ids):
        span = span_lookup.get((str(title).lower(), int(sentence_index)))
        if span is None:
            continue
        start = int(span["start"])
        end = min(int(span["end"]), int(context_length))
        if start >= end:
            continue
        first_window = start // window_size
        last_window = max(start, end - 1) // window_size
        for window_index in range(first_window, last_window + 1):
            windows.add(int(window_index))
    return sorted(windows)


def _prepare_hotpot_example(
    raw_example: Dict[str, object],
    question_max_length: int,
    context_length: int,
    window_size: int,
) -> PreparedHotpotExample | None:
    answer_text = str(raw_example.get("answer", "")).strip()
    if not answer_text:
        return None

    question_tokens = _text_tokens(str(raw_example.get("question", "")))[:question_max_length]
    context_tokens_full, sentence_spans = _flatten_hotpot_context_with_spans(raw_example.get("context", {}))  # type: ignore[arg-type]
    answer_tokens = _text_tokens(answer_text)
    if not answer_tokens:
        return None

    truncated_context = context_tokens_full[:context_length]
    start = _find_subsequence(truncated_context, answer_tokens)
    if start is None:
        return None
    end = start + len(answer_tokens) - 1
    if end >= context_length:
        return None

    answer_window = start // window_size
    if end >= (answer_window + 1) * window_size:
        answer_window = None
    support_windows = _extract_support_windows(
        raw_example=raw_example,
        sentence_spans=sentence_spans,
        context_length=context_length,
        window_size=window_size,
    )

    return PreparedHotpotExample(
        question_tokens=list(question_tokens),
        context_tokens=list(truncated_context),
        answer_tokens=list(answer_tokens),
        answer_text=answer_text,
        answer_start=int(start),
        answer_end=int(end),
        answer_window=answer_window,
        support_windows=support_windows,
    )


def _keep_prepared_example(row: PreparedHotpotExample, filter_mode: str) -> bool:
    if filter_mode == "all":
        return True
    if filter_mode == "answer_aligned":
        return row.answer_window is not None
    if filter_mode == "multi_support":
        return row.answer_window is not None and len(set(row.support_windows)) >= 2
    raise ValueError(f"unknown filter_mode={filter_mode}")


def _collect_prepared_examples(
    split,
    limit: int,
    question_max_length: int,
    context_length: int,
    window_size: int,
    filter_mode: str,
) -> tuple[List[PreparedHotpotExample], Dict[str, int]]:
    rows: List[PreparedHotpotExample] = []
    attempted = 0
    filtered = 0
    for example in split:
        attempted += 1
        prepared = _prepare_hotpot_example(
            example,
            question_max_length=question_max_length,
            context_length=context_length,
            window_size=window_size,
        )
        if prepared is None or not _keep_prepared_example(prepared, filter_mode=filter_mode):
            filtered += 1
            continue
        rows.append(prepared)
        if len(rows) >= limit:
            break
    if len(rows) < limit:
        raise RuntimeError(f"needed {limit} prepared rows, found {len(rows)} after scanning {attempted} examples")
    return rows, {"attempted": int(attempted), "kept": int(len(rows)), "filtered": int(filtered)}


def _load_hotpot_splits(config: CorpusConfig, num_clones: int, seed: int) -> tuple[Dict[str, List[PreparedHotpotExample]], Dict[str, object]]:
    dataset = load_dataset(config.dataset_name, config.dataset_config)
    context_length = int(config.window_size) * int(num_clones)

    train_split = dataset["train"].shuffle(seed=int(seed))
    validation_split = dataset["validation"].shuffle(seed=int(seed) + 1)

    train_rows, train_stats = _collect_prepared_examples(
        train_split,
        limit=int(config.n_train),
        question_max_length=int(config.question_max_length),
        context_length=context_length,
        window_size=int(config.window_size),
        filter_mode=str(config.example_filter),
    )
    validation_rows, val_stats = _collect_prepared_examples(
        validation_split,
        limit=int(config.n_dev) + int(config.n_test),
        question_max_length=int(config.question_max_length),
        context_length=context_length,
        window_size=int(config.window_size),
        filter_mode=str(config.example_filter),
    )
    dev_rows = validation_rows[: int(config.n_dev)]
    test_rows = validation_rows[int(config.n_dev) : int(config.n_dev) + int(config.n_test)]
    return (
        {"train": train_rows, "dev": dev_rows, "test": test_rows},
        {
            "train": train_stats,
            "validation": val_stats,
        },
    )


def _build_vocab(rows: Sequence[PreparedHotpotExample], vocab_size: int) -> Vocabulary:
    counts: Dict[str, int] = {}
    for row in rows:
        for token in row.question_tokens:
            counts[token] = counts.get(token, 0) + 1
        for token in row.context_tokens:
            counts[token] = counts.get(token, 0) + 1
        for token in row.answer_tokens:
            counts[token] = counts.get(token, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    budget = max(0, int(vocab_size) - len(SPECIAL_TOKENS))
    tokens = list(SPECIAL_TOKENS)
    tokens.extend(token for token, _count in ordered[:budget])
    stoi = {token: index for index, token in enumerate(tokens)}
    return Vocabulary(stoi=stoi, itos=tokens)


def _make_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        collate_fn=_collate_hotpot,
    )


class SparseLocalCloneSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_clones: int, dropout: float) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.num_clones = int(num_clones)
        self.head_dim = self.hidden_size // self.num_heads
        self.q_weight = nn.Parameter(torch.empty(self.num_clones, self.hidden_size, self.hidden_size))
        self.k_weight = nn.Parameter(torch.empty(self.num_clones, self.hidden_size, self.hidden_size))
        self.v_weight = nn.Parameter(torch.empty(self.num_clones, self.hidden_size, self.hidden_size))
        self.o_weight = nn.Parameter(torch.empty(self.num_clones, self.hidden_size, self.hidden_size))
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for param in (self.q_weight, self.k_weight, self.v_weight, self.o_weight):
            nn.init.xavier_uniform_(param)

    def forward(self, hidden: torch.Tensor, clone_visible_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, clones, tokens, _hidden = hidden.shape
        q = torch.einsum("bnth,nhd->bntd", hidden, self.q_weight)
        k = torch.einsum("bnth,nhd->bntd", hidden, self.k_weight)
        v = torch.einsum("bnth,nhd->bntd", hidden, self.v_weight)

        q = q.reshape(batch, clones, tokens, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k = k.reshape(batch, clones, tokens, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        v = v.reshape(batch, clones, tokens, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        key_mask = clone_visible_mask.view(batch, clones, 1, 1, tokens)
        scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores.float(), dim=-1).to(dtype=hidden.dtype)
        probs = self.attn_dropout(probs)
        query_mask = clone_visible_mask.view(batch, clones, 1, tokens, 1).to(dtype=hidden.dtype)
        probs = probs * query_mask

        attended = torch.matmul(probs, v).permute(0, 1, 3, 2, 4).reshape(batch, clones, tokens, self.hidden_size)
        attended = torch.einsum("bnth,nhd->bntd", attended, self.o_weight)
        attended = self.out_dropout(attended)
        attended = attended * clone_visible_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        return attended, probs


class SparseLocalCloneBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_clones: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SparseLocalCloneSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_clones=num_clones,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, hidden: torch.Tensor, clone_visible_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_input = self.norm1(hidden)
        attn_out, attn_probs = self.attn(attn_input, clone_visible_mask=clone_visible_mask)
        hidden = hidden + attn_out
        ff_input = self.norm2(hidden)
        ff_out = self.ff(ff_input)
        ff_out = ff_out * clone_visible_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        hidden = (hidden + ff_out) * clone_visible_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        return hidden, attn_probs


class CloneMetaAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float, init_std: float) -> None:
        super().__init__()
        self.global_query = nn.Parameter(torch.empty(hidden_size))
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        nn.init.normal_(self.global_query, mean=0.0, std=float(init_std))

    def forward(self, clone_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = clone_states.shape[0]
        query = self.global_query.view(1, 1, -1).expand(batch, 1, -1)
        meta_out, weights = self.attn(query, clone_states, clone_states, need_weights=True)
        return meta_out.squeeze(1), weights.squeeze(1)


class QuestionRoutedMultiHopQACloneModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        question_max_length: int,
        window_size: int,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.question_max_length = int(question_max_length)
        self.question_segment_length = int(question_max_length) + 1
        self.window_size = int(window_size)
        self.context_length = int(window_size) * int(config.num_clones)
        self.total_length = self.question_segment_length + self.context_length
        self.num_windows = int(config.num_clones)
        self.visible_tokens = self.question_segment_length + self.window_size

        self.token_embedding = nn.Embedding(vocab_size, config.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.total_length, config.hidden_size)
        self.clone_identity = nn.Embedding(config.num_clones, config.hidden_size)
        self.router_clone_identity = nn.Embedding(config.num_clones, config.hidden_size)
        self.router_question_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.router_window_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [
                SparseLocalCloneBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    num_clones=config.num_clones,
                    ff_dim=config.ff_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.meta_attention = CloneMetaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            dropout=config.dropout,
            init_std=config.identity_init_std,
        )
        self.start_query_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.end_query_proj = nn.Linear(config.hidden_size, config.hidden_size)
        nn.init.normal_(self.clone_identity.weight, mean=0.0, std=float(config.identity_init_std))
        nn.init.normal_(self.router_clone_identity.weight, mean=0.0, std=float(config.identity_init_std))

    def _build_window_bank(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, total_tokens = input_ids.shape
        if total_tokens != self.total_length:
            raise ValueError(f"expected total_length={self.total_length}, got {total_tokens}")
        device = input_ids.device
        positions = torch.arange(total_tokens, dtype=torch.long, device=device)
        token_hidden = self.token_embedding(input_ids) + self.position_embedding(positions).unsqueeze(0)

        question_hidden = token_hidden[:, : self.question_segment_length, :]
        question_mask = attention_mask[:, : self.question_segment_length]
        context_hidden = token_hidden[:, self.question_segment_length :, :]
        context_mask = attention_mask[:, self.question_segment_length :]

        window_hidden = context_hidden.view(batch, self.num_windows, self.window_size, self.config.hidden_size)
        window_mask = context_mask.view(batch, self.num_windows, self.window_size)
        return question_hidden, question_mask, window_hidden, window_mask

    def _question_summary(self, question_hidden: torch.Tensor, question_mask: torch.Tensor) -> torch.Tensor:
        weights = question_mask.unsqueeze(-1).to(dtype=question_hidden.dtype)
        totals = (question_hidden * weights).sum(dim=1)
        counts = weights.sum(dim=1).clamp(min=1.0)
        return totals / counts

    def _window_summaries(self, window_hidden: torch.Tensor, window_mask: torch.Tensor) -> torch.Tensor:
        weights = window_mask.unsqueeze(-1).to(dtype=window_hidden.dtype)
        totals = (window_hidden * weights).sum(dim=2)
        counts = weights.sum(dim=2).clamp(min=1.0)
        return totals / counts

    def _oracle_route_support(
        self,
        support_windows: Sequence[Sequence[int]],
        answer_window: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = len(support_windows)
        selected = torch.zeros(batch, self.config.num_clones, dtype=torch.long, device=device)
        for row_index, support in enumerate(support_windows):
            candidates = [int(value) for value in support if 0 <= int(value) < self.num_windows]
            if not candidates and int(answer_window[row_index].item()) >= 0:
                candidates = [int(answer_window[row_index].item())]
            if not candidates:
                candidates = [index % self.num_windows for index in range(self.config.num_clones)]
            for clone_index in range(self.config.num_clones):
                selected[row_index, clone_index] = int(candidates[clone_index % len(candidates)])
        one_hot = F.one_hot(selected, num_classes=self.num_windows).to(dtype=torch.float32)
        return one_hot, selected

    def _route_windows(
        self,
        question_hidden: torch.Tensor,
        question_mask: torch.Tensor,
        window_hidden: torch.Tensor,
        window_mask: torch.Tensor,
        support_windows: Sequence[Sequence[int]],
        answer_window: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = question_hidden.shape[0]
        device = question_hidden.device
        question_summary = self._question_summary(question_hidden, question_mask)
        window_summary = self._window_summaries(window_hidden, window_mask)

        if self.config.routing_mode == "oracle_support":
            route_weights, selected_windows = self._oracle_route_support(
                support_windows=support_windows,
                answer_window=answer_window,
                device=device,
            )
            score_logits = route_weights
            return route_weights.to(device=device), selected_windows, score_logits.to(device=device)

        clone_ids = torch.arange(self.config.num_clones, dtype=torch.long, device=device)
        clone_query = self.router_question_proj(question_summary).unsqueeze(1) + self.router_clone_identity(clone_ids).view(
            1, self.config.num_clones, -1
        )
        clone_query = F.normalize(torch.tanh(clone_query), dim=-1)
        window_key = F.normalize(torch.tanh(self.router_window_proj(window_summary)), dim=-1)
        score_logits = torch.einsum("bnh,bwh->bnw", clone_query, window_key)

        if self.config.routing_mode == "learned_distinct":
            available = torch.ones(batch, self.num_windows, dtype=torch.bool, device=device)
            chosen_weights = []
            chosen_indices = []
            for clone_index in range(self.config.num_clones):
                clone_logits = score_logits[:, clone_index, :].masked_fill(~available, NEG_INF)
                if self.training:
                    clone_weights = F.gumbel_softmax(
                        clone_logits,
                        tau=float(self.config.router_temperature),
                        hard=True,
                        dim=-1,
                    )
                    clone_index_tensor = clone_weights.argmax(dim=-1)
                else:
                    clone_index_tensor = clone_logits.argmax(dim=-1)
                    clone_weights = F.one_hot(clone_index_tensor, num_classes=self.num_windows).to(dtype=window_hidden.dtype)
                chosen_weights.append(clone_weights)
                chosen_indices.append(clone_index_tensor)
                available = available & ~clone_weights.to(dtype=torch.bool)
            route_weights = torch.stack(chosen_weights, dim=1)
            selected_windows = torch.stack(chosen_indices, dim=1)
            return route_weights.to(dtype=window_hidden.dtype), selected_windows, score_logits

        if self.training:
            route_weights = F.gumbel_softmax(
                score_logits,
                tau=float(self.config.router_temperature),
                hard=True,
                dim=-1,
            )
            selected_windows = route_weights.argmax(dim=-1)
        else:
            selected_windows = score_logits.argmax(dim=-1)
            route_weights = F.one_hot(selected_windows, num_classes=self.num_windows).to(dtype=window_hidden.dtype)
        return route_weights.to(dtype=window_hidden.dtype), selected_windows, score_logits

    def _build_clone_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        support_windows: Sequence[Sequence[int]],
        answer_window: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        question_hidden, question_mask, window_hidden, window_mask = self._build_window_bank(input_ids, attention_mask)
        route_weights, selected_windows, route_logits = self._route_windows(
            question_hidden=question_hidden,
            question_mask=question_mask,
            window_hidden=window_hidden,
            window_mask=window_mask,
            support_windows=support_windows,
            answer_window=answer_window,
        )

        expanded_windows = window_hidden.unsqueeze(1).expand(-1, self.config.num_clones, -1, -1, -1)
        expanded_masks = window_mask.unsqueeze(1).expand(-1, self.config.num_clones, -1, -1)
        selected_context_hidden = torch.einsum("bnw,bnwth->bnth", route_weights, expanded_windows)
        selected_context_mask = torch.einsum("bnw,bnwt->bnt", route_weights, expanded_masks.to(dtype=window_hidden.dtype)) > 0.5

        clone_question_hidden = question_hidden.unsqueeze(1).expand(-1, self.config.num_clones, -1, -1)
        clone_question_mask = question_mask.unsqueeze(1).expand(-1, self.config.num_clones, -1)
        clone_hidden = torch.cat([clone_question_hidden, selected_context_hidden], dim=2)
        clone_visible_mask = torch.cat([clone_question_mask, selected_context_mask], dim=2)

        clone_ids = torch.arange(self.config.num_clones, dtype=torch.long, device=input_ids.device)
        clone_hidden = clone_hidden + self.clone_identity(clone_ids).view(1, self.config.num_clones, 1, -1)
        clone_hidden = self.embedding_dropout(clone_hidden)
        clone_hidden = clone_hidden * clone_visible_mask.unsqueeze(-1).to(dtype=clone_hidden.dtype)
        return clone_hidden, clone_visible_mask, selected_windows, route_logits

    def pooled_clone_states(self, clone_hidden: torch.Tensor, clone_visible_mask: torch.Tensor) -> torch.Tensor:
        context_hidden = clone_hidden[:, :, self.question_segment_length :, :]
        context_mask = clone_visible_mask[:, :, self.question_segment_length :]
        weights = context_mask.unsqueeze(-1).to(dtype=clone_hidden.dtype)
        totals = (context_hidden * weights).sum(dim=2)
        counts = weights.sum(dim=2).clamp(min=1.0)
        return totals / counts

    def _scatter_local_logits(
        self,
        local_logits: torch.Tensor,
        selected_windows: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, clones, window_size = local_logits.shape
        min_value = NEG_INF
        positions = selected_windows.unsqueeze(-1) * self.window_size + torch.arange(
            window_size, dtype=torch.long, device=local_logits.device
        ).view(1, 1, window_size)
        expanded = torch.full(
            (batch, clones, self.context_length),
            fill_value=min_value,
            dtype=local_logits.dtype,
            device=local_logits.device,
        )
        for offset in range(window_size):
            expanded = expanded.scatter(2, positions[:, :, offset : offset + 1], local_logits[:, :, offset : offset + 1])
        expanded = expanded.masked_fill(~context_mask.unsqueeze(1), min_value)
        return expanded.max(dim=1).values

    def qa_logits_from_components(
        self,
        clone_hidden: torch.Tensor,
        clone_visible_mask: torch.Tensor,
        selected_windows: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        clone_states = self.pooled_clone_states(clone_hidden, clone_visible_mask)
        meta_state, meta_weights = self.meta_attention(clone_states)
        local_context_hidden = clone_hidden[:, :, self.question_segment_length :, :]
        start_query = F.normalize(self.start_query_proj(meta_state), dim=-1)
        end_query = F.normalize(self.end_query_proj(meta_state), dim=-1)
        local_start = torch.einsum("bnth,bh->bnt", local_context_hidden, start_query) / math.sqrt(float(self.config.hidden_size))
        local_end = torch.einsum("bnth,bh->bnt", local_context_hidden, end_query) / math.sqrt(float(self.config.hidden_size))
        start_logits = self._scatter_local_logits(local_start, selected_windows=selected_windows, context_mask=context_mask)
        end_logits = self._scatter_local_logits(local_end, selected_windows=selected_windows, context_mask=context_mask)
        min_value = NEG_INF
        start_logits = start_logits.masked_fill(~context_mask, min_value)
        end_logits = end_logits.masked_fill(~context_mask, min_value)
        return start_logits, end_logits, meta_weights, clone_states

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_mask: torch.Tensor,
        support_windows: Sequence[Sequence[int]],
        answer_window: torch.Tensor,
        return_clone_states: bool = False,
        return_attention: bool = False,
    ) -> Dict[str, object]:
        clone_hidden, clone_visible_mask, selected_windows, route_logits = self._build_clone_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            support_windows=support_windows,
            answer_window=answer_window,
        )
        attention_probs: List[torch.Tensor] = []
        for layer in self.layers:
            clone_hidden, probs = layer(clone_hidden, clone_visible_mask=clone_visible_mask)
            if return_attention:
                attention_probs.append(probs)
        clone_hidden = self.final_norm(clone_hidden)
        clone_hidden = clone_hidden * clone_visible_mask.unsqueeze(-1).to(dtype=clone_hidden.dtype)
        start_logits, end_logits, meta_weights, clone_states = self.qa_logits_from_components(
            clone_hidden=clone_hidden,
            clone_visible_mask=clone_visible_mask,
            selected_windows=selected_windows,
            context_mask=context_mask,
        )
        out: Dict[str, object] = {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "meta_attention_weights": meta_weights.detach().cpu(),
            "selected_windows": selected_windows.detach().cpu(),
            "route_logits": route_logits.detach().cpu(),
        }
        if return_clone_states:
            out["clone_states"] = clone_states
        if return_attention:
            out["attention_probs"] = attention_probs
            out["clone_visible_mask"] = clone_visible_mask
        return out


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _amp_context(device: torch.device, use_bf16: bool):
    if device.type == "cuda" and use_bf16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _qa_loss(start_logits: torch.Tensor, end_logits: torch.Tensor, start_positions: torch.Tensor, end_positions: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(start_logits, start_positions) + F.cross_entropy(end_logits, end_positions)


def _clone_attention_entropy(attention_probs: Sequence[torch.Tensor], clone_visible_mask: torch.Tensor) -> List[float]:
    if not attention_probs:
        return []
    clone_count = attention_probs[0].shape[1]
    totals = torch.zeros(clone_count, dtype=torch.float64, device=clone_visible_mask.device)
    counts = torch.zeros(clone_count, dtype=torch.float64, device=clone_visible_mask.device)
    query_mask = clone_visible_mask.to(dtype=torch.float64)
    for probs in attention_probs:
        entropy = -(probs.clamp_min(1e-9).float() * probs.clamp_min(1e-9).float().log()).sum(dim=-1)
        masked = entropy.to(dtype=torch.float64) * query_mask.view(query_mask.shape[0], query_mask.shape[1], 1, query_mask.shape[2])
        totals += masked.sum(dim=(0, 2, 3))
        counts += query_mask.sum(dim=(0, 2)).to(dtype=torch.float64) * probs.shape[2]
    values = totals / counts.clamp(min=1.0)
    return [float(value.detach().cpu().item()) for value in values]


def _pairwise_clone_cosine(clone_states: torch.Tensor) -> Dict[str, object]:
    normalized = F.normalize(clone_states.float(), dim=-1)
    cosine = torch.einsum("bnd,bmd->bnm", normalized, normalized).mean(dim=0)
    off_diag = []
    for row in range(cosine.shape[0]):
        for col in range(row + 1, cosine.shape[1]):
            off_diag.append(float(cosine[row, col].detach().cpu().item()))
    return {
        "mean_off_diagonal": float(np.mean(off_diag)) if off_diag else 1.0,
        "matrix": [[float(value) for value in line] for line in cosine.detach().cpu().tolist()],
    }


def _meta_attention_distribution(meta_weights: Sequence[torch.Tensor]) -> Dict[str, object]:
    if not meta_weights:
        return {"mean_weights": [], "std_weights": [], "mean_entropy": 0.0, "max_weight_mean": 0.0}
    merged = torch.cat(meta_weights, dim=0).float()
    entropy = -(merged.clamp_min(1e-9) * merged.clamp_min(1e-9).log()).sum(dim=-1)
    return {
        "mean_weights": [float(value) for value in merged.mean(dim=0).cpu().tolist()],
        "std_weights": [float(value) for value in merged.std(dim=0, unbiased=False).cpu().tolist()],
        "mean_entropy": float(entropy.mean().cpu().item()),
        "max_weight_mean": float(merged.max(dim=-1).values.mean().cpu().item()),
        "weight_balance_std": float(merged.mean(dim=0).std(unbiased=False).cpu().item()),
    }


def _normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _answer_f1(prediction: str, gold: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = {}
    for token in pred_tokens:
        common[token] = min(pred_tokens.count(token), gold_tokens.count(token))
    overlap = float(sum(common.values()))
    if overlap <= 0.0:
        return 0.0
    precision = overlap / max(1.0, float(len(pred_tokens)))
    recall = overlap / max(1.0, float(len(gold_tokens)))
    return 2.0 * precision * recall / max(1e-9, precision + recall)


def _exact_match(prediction: str, gold: str) -> float:
    return 1.0 if _normalize_answer(prediction) == _normalize_answer(gold) else 0.0


def _best_span(start_logits: torch.Tensor, end_logits: torch.Tensor, context_mask: torch.Tensor, max_answer_length: int) -> tuple[int, int]:
    length = start_logits.shape[0]
    scores = start_logits.unsqueeze(1) + end_logits.unsqueeze(0)
    invalid = torch.triu(torch.ones(length, length, dtype=torch.bool, device=scores.device), diagonal=int(max_answer_length))
    lower = torch.tril(torch.ones(length, length, dtype=torch.bool, device=scores.device), diagonal=-1)
    invalid = invalid | lower
    valid_positions = context_mask.to(dtype=torch.bool)
    invalid = invalid | ~valid_positions.view(length, 1) | ~valid_positions.view(1, length)
    scores = scores.masked_fill(invalid, NEG_INF)
    flat_index = int(scores.reshape(-1).argmax().detach().cpu().item())
    start = flat_index // length
    end = flat_index % length
    return int(start), int(end)


def _decode_context_span(context_tokens: Sequence[str], start: int, end: int) -> str:
    if start < 0 or end < start or start >= len(context_tokens):
        return ""
    end = min(end, len(context_tokens) - 1)
    return " ".join(context_tokens[start : end + 1]).strip()


def _answer_window_distribution(answer_windows: Sequence[int], num_clones: int) -> Dict[str, object]:
    counts = np.zeros(num_clones, dtype=np.float64)
    cross_boundary = 0
    for window in answer_windows:
        if window < 0:
            cross_boundary += 1
        else:
            counts[int(window)] += 1.0
    total = counts.sum()
    return {
        "mean_mass_per_clone": [float(value / max(1.0, total)) for value in counts],
        "cross_boundary_count": int(cross_boundary),
    }


def _routing_metrics(
    selected_windows: Sequence[torch.Tensor],
    answer_windows: Sequence[torch.Tensor],
    support_windows: Sequence[Sequence[Sequence[int]]],
    num_clones: int,
) -> Dict[str, object]:
    if not selected_windows:
        return {
            "selected_window_mean_mass": [0.0] * num_clones,
            "answer_window_hit_rate": 0.0,
            "support_window_hit_rate": 0.0,
        }
    merged = torch.cat(selected_windows, dim=0)
    counts = np.zeros(num_clones, dtype=np.float64)
    answer_hits = 0
    answer_total = 0
    support_hits = 0
    support_total = 0
    offset = 0
    merged_answer_windows = torch.cat(answer_windows, dim=0)
    flat_support = [item for batch in support_windows for item in batch]
    for row_index in range(merged.shape[0]):
        row = [int(value) for value in merged[row_index].tolist()]
        for value in row:
            counts[value] += 1.0
        answer_window = int(merged_answer_windows[row_index].item())
        if answer_window >= 0:
            answer_total += 1
            if answer_window in row:
                answer_hits += 1
        support = [int(value) for value in flat_support[row_index]]
        if support:
            support_total += 1
            if any(value in support for value in row):
                support_hits += 1
        offset += 1
    total = counts.sum()
    return {
        "selected_window_mean_mass": [float(value / max(1.0, total)) for value in counts],
        "answer_window_hit_rate": float(answer_hits / max(1, answer_total)),
        "support_window_hit_rate": float(support_hits / max(1, support_total)),
    }


def evaluate_model(
    model: QuestionRoutedMultiHopQACloneModel,
    loader: DataLoader,
    device: torch.device,
    answer_max_length: int,
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    batch_count = 0
    all_em: List[float] = []
    all_f1: List[float] = []
    clone_em: List[List[float]] = [[] for _ in range(model.config.num_clones)]
    clone_f1: List[List[float]] = [[] for _ in range(model.config.num_clones)]
    attention_entropy_sum = np.zeros(model.config.num_clones, dtype=np.float64)
    cosine_values: List[float] = []
    cosine_matrices: List[np.ndarray] = []
    meta_weight_batches: List[torch.Tensor] = []
    answer_windows: List[int] = []
    selected_window_batches: List[torch.Tensor] = []
    answer_window_batches: List[torch.Tensor] = []
    support_window_batches: List[Sequence[Sequence[int]]] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            context_mask = batch["context_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)
            answer_window = batch["answer_window"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                context_mask=context_mask,
                support_windows=batch["support_windows"],
                answer_window=answer_window,
                return_clone_states=True,
                return_attention=True,
            )
            start_logits = outputs["start_logits"]
            end_logits = outputs["end_logits"]
            clone_states = outputs["clone_states"]
            clone_visible_mask = outputs["clone_visible_mask"]
            attention_probs = outputs["attention_probs"]
            meta_weights = outputs["meta_attention_weights"]
            selected_windows = outputs["selected_windows"]
            clone_hidden, clone_mask, recomputed_selected_windows, _route_logits = model._build_clone_inputs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                support_windows=batch["support_windows"],
                answer_window=answer_window,
            )
            for layer in model.layers:
                clone_hidden, _ = layer(clone_hidden, clone_visible_mask=clone_mask)
            clone_hidden = model.final_norm(clone_hidden)
            clone_hidden = clone_hidden * clone_mask.unsqueeze(-1).to(dtype=clone_hidden.dtype)

            loss = _qa_loss(start_logits, end_logits, start_positions, end_positions)
            total_loss += float(loss.detach().cpu().item())
            batch_count += 1

            attention_entropy_sum += np.asarray(_clone_attention_entropy(attention_probs, clone_visible_mask), dtype=np.float64)
            cosine = _pairwise_clone_cosine(clone_states)
            cosine_values.append(float(cosine["mean_off_diagonal"]))
            cosine_matrices.append(np.asarray(cosine["matrix"], dtype=np.float64))
            meta_weight_batches.append(meta_weights)
            selected_window_batches.append(selected_windows)
            answer_window_batches.append(batch["answer_window"])
            support_window_batches.append(batch["support_windows"])
            answer_windows.extend(int(value) for value in batch["answer_window"].tolist())

            context_tokens_batch = batch["context_tokens"]
            answer_text_batch = batch["answer_text"]
            for row_index in range(input_ids.shape[0]):
                pred_start, pred_end = _best_span(
                    start_logits[row_index],
                    end_logits[row_index],
                    context_mask[row_index],
                    max_answer_length=answer_max_length,
                )
                prediction = _decode_context_span(context_tokens_batch[row_index], pred_start, pred_end)
                gold = answer_text_batch[row_index]
                all_em.append(_exact_match(prediction, gold))
                all_f1.append(_answer_f1(prediction, gold))

            for clone_index in range(model.config.num_clones):
                single_start, single_end, _single_meta, _single_states = model.qa_logits_from_components(
                    clone_hidden=clone_hidden[:, clone_index : clone_index + 1],
                    clone_visible_mask=clone_mask[:, clone_index : clone_index + 1],
                    selected_windows=recomputed_selected_windows[:, clone_index : clone_index + 1],
                    context_mask=context_mask,
                )
                for row_index in range(input_ids.shape[0]):
                    pred_start, pred_end = _best_span(
                        single_start[row_index],
                        single_end[row_index],
                        context_mask[row_index],
                        max_answer_length=answer_max_length,
                    )
                    prediction = _decode_context_span(context_tokens_batch[row_index], pred_start, pred_end)
                    gold = answer_text_batch[row_index]
                    clone_em[clone_index].append(_exact_match(prediction, gold))
                    clone_f1[clone_index].append(_answer_f1(prediction, gold))

    mean_cosine_matrix = np.mean(cosine_matrices, axis=0) if cosine_matrices else np.eye(model.config.num_clones, dtype=np.float64)
    per_clone_em = [float(np.mean(values)) if values else 0.0 for values in clone_em]
    per_clone_f1 = [float(np.mean(values)) if values else 0.0 for values in clone_f1]
    meta_distribution = _meta_attention_distribution(meta_weight_batches)
    routing = _routing_metrics(
        selected_windows=selected_window_batches,
        answer_windows=answer_window_batches,
        support_windows=support_window_batches,
        num_clones=model.config.num_clones,
    )
    return {
        "loss": float(total_loss / max(1, batch_count)),
        "exact_match": float(np.mean(all_em)) if all_em else 0.0,
        "f1": float(np.mean(all_f1)) if all_f1 else 0.0,
        "per_clone_exact_match": per_clone_em,
        "per_clone_f1": per_clone_f1,
        "best_single_clone_f1": float(max(per_clone_f1)) if per_clone_f1 else 0.0,
        "joint_beats_single": bool((float(np.mean(all_f1)) if all_f1 else 0.0) > (max(per_clone_f1) if per_clone_f1 else 0.0)),
        "mean_off_diagonal_clone_cosine": float(np.mean(cosine_values)) if cosine_values else 1.0,
        "clone_cosine_matrix": [[float(value) for value in row] for row in mean_cosine_matrix.tolist()],
        "attention_entropy_per_clone": [float(value) for value in (attention_entropy_sum / max(1, len(loader)))],
        "meta_attention_distribution": meta_distribution,
        "answer_window_coverage": _answer_window_distribution(answer_windows, num_clones=model.config.num_clones),
        "routing": routing,
    }


def train_on_prepared_splits(
    prepared_splits: Dict[str, Sequence[PreparedHotpotExample]],
    config: Stage15bConfig,
) -> Dict[str, object]:
    _set_seed(int(config.training.seed))
    device = torch.device(config.training.device)
    context_length = int(config.corpus.window_size) * int(config.model.num_clones)
    vocab = _build_vocab(prepared_splits["train"], vocab_size=int(config.corpus.vocab_size))

    train_dataset = HotpotQADataset(
        prepared_splits["train"],
        vocab=vocab,
        question_max_length=int(config.corpus.question_max_length),
        context_length=context_length,
        window_size=int(config.corpus.window_size),
    )
    dev_dataset = HotpotQADataset(
        prepared_splits["dev"],
        vocab=vocab,
        question_max_length=int(config.corpus.question_max_length),
        context_length=context_length,
        window_size=int(config.corpus.window_size),
    )
    test_dataset = HotpotQADataset(
        prepared_splits["test"],
        vocab=vocab,
        question_max_length=int(config.corpus.question_max_length),
        context_length=context_length,
        window_size=int(config.corpus.window_size),
    )

    train_loader = _make_loader(train_dataset, batch_size=int(config.training.batch_size), shuffle=True, seed=int(config.training.seed))
    dev_loader = _make_loader(dev_dataset, batch_size=int(config.training.batch_size), shuffle=False, seed=int(config.training.seed) + 1)
    test_loader = _make_loader(test_dataset, batch_size=int(config.training.batch_size), shuffle=False, seed=int(config.training.seed) + 2)

    model = QuestionRoutedMultiHopQACloneModel(
        vocab_size=vocab.size,
        question_max_length=int(config.corpus.question_max_length),
        window_size=int(config.corpus.window_size),
        config=config.model,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.training.lr),
        weight_decay=float(config.training.weight_decay),
    )

    history: List[Dict[str, object]] = []
    best_dev_f1 = -1.0
    best_epoch = 0
    best_state: Dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    for epoch in range(int(config.training.epochs)):
        model.train()
        loss_sum = 0.0
        batch_count = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            context_mask = batch["context_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)
            answer_window = batch["answer_window"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with _amp_context(device, bool(config.training.use_bf16)):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    context_mask=context_mask,
                    support_windows=batch["support_windows"],
                    answer_window=answer_window,
                )
                loss = _qa_loss(outputs["start_logits"], outputs["end_logits"], start_positions, end_positions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(config.training.gradient_clip))
            optimizer.step()
            loss_sum += float(loss.detach().cpu().item())
            batch_count += 1

        dev_metrics = evaluate_model(model, dev_loader, device=device, answer_max_length=int(config.corpus.answer_max_length))
        history.append(
            {
                "epoch": int(epoch + 1),
                "train_loss": float(loss_sum / max(1, batch_count)),
                "dev_exact_match": float(dev_metrics["exact_match"]),
                "dev_f1": float(dev_metrics["f1"]),
                "dev_mean_off_diagonal_clone_cosine": float(dev_metrics["mean_off_diagonal_clone_cosine"]),
                "dev_meta_weight_balance": float(dev_metrics["meta_attention_distribution"]["weight_balance_std"]),
                "dev_joint_beats_single": bool(dev_metrics["joint_beats_single"]),
                "dev_answer_window_hit_rate": float(dev_metrics["routing"]["answer_window_hit_rate"]),
                "dev_support_window_hit_rate": float(dev_metrics["routing"]["support_window_hit_rate"]),
            }
        )

        if float(dev_metrics["f1"]) > best_dev_f1:
            best_dev_f1 = float(dev_metrics["f1"])
            best_epoch = int(epoch + 1)
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(config.training.early_stopping_patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_dev = evaluate_model(model, dev_loader, device=device, answer_max_length=int(config.corpus.answer_max_length))
    final_test = evaluate_model(model, test_loader, device=device, answer_max_length=int(config.corpus.answer_max_length))
    return {
        "config": asdict(config),
        "context_length": int(context_length),
        "question_max_length": int(config.corpus.question_max_length),
        "total_length": int(context_length + config.corpus.question_max_length + 1),
        "window_size": int(config.corpus.window_size),
        "vocab_size": int(vocab.size),
        "dataset_sizes": {
            "train": len(train_dataset),
            "dev": len(dev_dataset),
            "test": len(test_dataset),
        },
        "best_epoch": int(best_epoch),
        "history": history,
        "final_dev": final_dev,
        "final_test": final_test,
    }


def _build_scale_config(base: Stage15bConfig, num_clones: int) -> Stage15bConfig:
    return Stage15bConfig(
        corpus=base.corpus,
        model=ModelConfig(
            num_clones=int(num_clones),
            hidden_size=base.model.hidden_size,
            num_layers=base.model.num_layers,
            num_heads=base.model.num_heads,
            ff_dim=base.model.ff_dim,
            dropout=base.model.dropout,
            identity_init_std=base.model.identity_init_std,
            routing_mode=base.model.routing_mode,
            router_temperature=base.model.router_temperature,
        ),
        training=base.training,
    )


def _format_scale_table(scale_results: Dict[str, Dict[str, object]]) -> List[str]:
    lines = [
        "N | seq_len | EM | F1 | clone_cosine | meta_weight_balance | joint_beats_single | answer_window_hit",
        "--- | ---: | ---: | ---: | ---: | ---: | --- | ---: ",
    ]
    for scale_key in sorted(scale_results.keys(), key=int):
        row = scale_results[scale_key]
        final_test = row["final_test"]
        lines.append(
            "{n} | {seq_len} | {em:.4f} | {f1:.4f} | {cosine:.6f} | {balance:.6f} | {beats} | {hit:.4f}".format(
                n=scale_key,
                seq_len=int(row["context_length"]),
                em=float(final_test["exact_match"]),
                f1=float(final_test["f1"]),
                cosine=float(final_test["mean_off_diagonal_clone_cosine"]),
                balance=float(final_test["meta_attention_distribution"]["weight_balance_std"]),
                beats=bool(final_test["joint_beats_single"]),
                hit=float(final_test["routing"]["answer_window_hit_rate"]),
            )
        )
    return lines


def write_report(results: Dict[str, object], report_path: Path) -> None:
    variant_name = str(results.get("variant_name", "unknown"))
    scales = dict(results.get("scales", {}))
    lines = [
        f"# Stage 15B Question-Routed Multihop QA Clone Collaboration ({variant_name})",
        "",
        "## Scale Comparison",
        "",
    ]
    lines.extend(_format_scale_table(scales))
    for scale_key in sorted(scales.keys(), key=int):
        row = scales[scale_key]
        final_test = row["final_test"]
        lines.extend(
            [
                "",
                f"## Scale N={scale_key}",
                f"- Context sequence length: `{row['context_length']}`",
                f"- Window size: `{row['window_size']}`",
                f"- Exact match: `{float(final_test['exact_match']):.4f}`",
                f"- F1: `{float(final_test['f1']):.4f}`",
                f"- Clone cosine: `{float(final_test['mean_off_diagonal_clone_cosine']):.6f}`",
                f"- Meta-attention mean weights: `{[round(float(v), 4) for v in final_test['meta_attention_distribution']['mean_weights']]}`",
                f"- Meta-attention weight balance std: `{float(final_test['meta_attention_distribution']['weight_balance_std']):.6f}`",
                f"- Routing selected-window mass: `{[round(float(v), 4) for v in final_test['routing']['selected_window_mean_mass']]}`",
                f"- Routing answer-window hit rate: `{float(final_test['routing']['answer_window_hit_rate']):.4f}`",
                f"- Routing support-window hit rate: `{float(final_test['routing']['support_window_hit_rate']):.4f}`",
                f"- Per-clone answer coverage: `{[round(float(v), 4) for v in final_test['answer_window_coverage']['mean_mass_per_clone']]}`",
                f"- Joint beats best single clone F1: `{bool(final_test['joint_beats_single'])}`",
                f"- Best single clone F1: `{float(final_test['best_single_clone_f1']):.4f}`",
                f"- Per-clone F1: `{[round(float(v), 4) for v in final_test['per_clone_f1']]}`",
            ]
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_variant_summary(variant_results: Sequence[Dict[str, object]], path: Path) -> None:
    lines = [
        "# Stage 15B Variant Summary",
        "",
        "variant | routing | filter | N | F1 | clone_cosine | joint_beats_single | answer_window_hit | support_window_hit",
        "--- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: ",
    ]
    for item in variant_results:
        result = item["result"]
        scale_key = str(item["screen_scale"])
        final_test = result["scales"][scale_key]["final_test"]
        config = result["base_config"]
        lines.append(
            "{variant} | {routing} | {filter_mode} | {n} | {f1:.4f} | {cosine:.6f} | {beats} | {answer_hit:.4f} | {support_hit:.4f}".format(
                variant=item["variant_name"],
                routing=config["model"]["routing_mode"],
                filter_mode=config["corpus"]["example_filter"],
                n=scale_key,
                f1=float(final_test["f1"]),
                cosine=float(final_test["mean_off_diagonal_clone_cosine"]),
                beats=bool(final_test["joint_beats_single"]),
                answer_hit=float(final_test["routing"]["answer_window_hit_rate"]),
                support_hit=float(final_test["routing"]["support_window_hit_rate"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(config: Stage15bConfig, scales: Sequence[int], variant_name: str) -> Dict[str, object]:
    scale_results: Dict[str, Dict[str, object]] = {}
    preprocessing: Dict[str, object] = {}
    for num_clones in scales:
        scale_config = _build_scale_config(config, num_clones=int(num_clones))
        prepared_splits, stats = _load_hotpot_splits(scale_config.corpus, num_clones=int(num_clones), seed=int(scale_config.training.seed))
        scale_results[str(num_clones)] = train_on_prepared_splits(prepared_splits, scale_config)
        preprocessing[str(num_clones)] = stats
    results = {
        "variant_name": variant_name,
        "base_config": asdict(config),
        "scales": scale_results,
        "preprocessing": preprocessing,
    }
    write_report(results, REPORT_PATH)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def _parse_scales(value: str) -> List[int]:
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def _build_variant_config(base: Stage15bConfig, *, routing_mode: str, filter_mode: str) -> Stage15bConfig:
    return Stage15bConfig(
        corpus=CorpusConfig(
            n_train=base.corpus.n_train,
            n_dev=base.corpus.n_dev,
            n_test=base.corpus.n_test,
            vocab_size=base.corpus.vocab_size,
            question_max_length=base.corpus.question_max_length,
            window_size=base.corpus.window_size,
            answer_max_length=base.corpus.answer_max_length,
            dataset_name=base.corpus.dataset_name,
            dataset_config=base.corpus.dataset_config,
            example_filter=filter_mode,
        ),
        model=ModelConfig(
            num_clones=base.model.num_clones,
            hidden_size=base.model.hidden_size,
            num_layers=base.model.num_layers,
            num_heads=base.model.num_heads,
            ff_dim=base.model.ff_dim,
            dropout=base.model.dropout,
            identity_init_std=base.model.identity_init_std,
            routing_mode=routing_mode,
            router_temperature=base.model.router_temperature,
        ),
        training=base.training,
    )


def run_variant_screen(
    base_config: Stage15bConfig,
    screen_scale: int,
) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    variants = [
        ("learned_all", _build_variant_config(base_config, routing_mode="learned", filter_mode="all")),
        ("learned_distinct_all", _build_variant_config(base_config, routing_mode="learned_distinct", filter_mode="all")),
        ("learned_multi_support", _build_variant_config(base_config, routing_mode="learned", filter_mode="multi_support")),
        ("learned_distinct_multi_support", _build_variant_config(base_config, routing_mode="learned_distinct", filter_mode="multi_support")),
        ("oracle_support_all", _build_variant_config(base_config, routing_mode="oracle_support", filter_mode="all")),
    ]
    results: List[Dict[str, object]] = []
    for variant_name, config in variants:
        result = run_experiment(config, scales=[int(screen_scale)], variant_name=variant_name)
        results.append({"variant_name": variant_name, "screen_scale": int(screen_scale), "result": result})
    _write_variant_summary(results, VARIANT_SUMMARY_PATH)

    def _score(item: Dict[str, object]) -> tuple[float, float, float]:
        final_test = item["result"]["scales"][str(screen_scale)]["final_test"]
        return (
            1.0 if bool(final_test["joint_beats_single"]) else 0.0,
            float(final_test["f1"]),
            float(final_test["routing"]["answer_window_hit_rate"]),
        )

    best = max(results, key=_score)
    return results, best


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 15B question-routed multihop QA clone collaboration.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-train", type=int, default=256)
    parser.add_argument("--n-dev", type=int, default=64)
    parser.add_argument("--n-test", type=int, default=64)
    parser.add_argument("--question-max-length", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--answer-max-length", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--scales", default="4")
    parser.add_argument("--routing-mode", default="learned")
    parser.add_argument("--example-filter", default="all")
    parser.add_argument("--router-temperature", type=float, default=1.0)
    parser.add_argument("--variant-name", default="manual")
    parser.add_argument("--screen", action="store_true")
    args = parser.parse_args()

    base_config = Stage15bConfig(
        corpus=CorpusConfig(
            n_train=int(args.n_train),
            n_dev=int(args.n_dev),
            n_test=int(args.n_test),
            vocab_size=int(args.vocab_size),
            question_max_length=int(args.question_max_length),
            window_size=int(args.window_size),
            answer_max_length=int(args.answer_max_length),
            example_filter=str(args.example_filter),
        ),
        model=ModelConfig(
            routing_mode=str(args.routing_mode),
            router_temperature=float(args.router_temperature),
        ),
        training=TrainingConfig(
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            seed=int(args.seed),
            device=str(args.device),
        ),
    )
    scales = _parse_scales(args.scales)

    if args.screen:
        screen_scale = int(scales[0])
        variant_results, best = run_variant_screen(base_config, screen_scale=screen_scale)
        best_name = str(best["variant_name"])
        best_config = _build_variant_config(
            base_config,
            routing_mode=str(best["result"]["base_config"]["model"]["routing_mode"]),
            filter_mode=str(best["result"]["base_config"]["corpus"]["example_filter"]),
        )
        final_results = run_experiment(best_config, scales=scales, variant_name=f"best_from_screen:{best_name}")
        print(
            "stage15b screen: wrote {results} and {report}; best_variant={variant}; N={n}; f1={f1:.4f}; joint_beats_single={beats}".format(
                results=RESULTS_PATH,
                report=REPORT_PATH,
                variant=best_name,
                n=scales[0],
                f1=float(final_results["scales"][str(scales[0])]["final_test"]["f1"]),
                beats=bool(final_results["scales"][str(scales[0])]["final_test"]["joint_beats_single"]),
            )
        )
        return

    results = run_experiment(base_config, scales=scales, variant_name=str(args.variant_name))
    first_scale = str(scales[0])
    print(
        "stage15b: wrote {results} and {report}; variant={variant}; N={n}; f1={f1:.4f}; joint_beats_single={beats}".format(
            results=RESULTS_PATH,
            report=REPORT_PATH,
            variant=args.variant_name,
            n=first_scale,
            f1=float(results["scales"][first_scale]["final_test"]["f1"]),
            beats=bool(results["scales"][first_scale]["final_test"]["joint_beats_single"]),
        )
    )


if __name__ == "__main__":
    main()
