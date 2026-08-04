from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


EXPERIMENT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_PATH = EXPERIMENT_ROOT / "results" / "stage14_sparse_local_clone_meta_attention_results.json"
REPORT_PATH = EXPERIMENT_ROOT / "reports" / "STAGE14_SPARSE_LOCAL_CLONE_META_ATTENTION.md"
STAGE13_CLONE_COSINE_BASELINE = 0.99997

STAGE10_CODE_ROOT = (
    Path(__file__).resolve().parents[4]
    / "10_complementarity_regularized_candidate_token_direct_coordination"
    / "code"
)
if str(STAGE10_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE10_CODE_ROOT))

from src.datasets.multiview_code_patch_selection import (  # type: ignore  # noqa: E402
    BALANCED_34B_CANDIDATE_REPRESENTATION,
    BALANCED_34B_DATASET_SOURCE,
    MultiViewCodePatchDatasetConfig,
    MultiViewTaskExample,
    build_multiview_code_patch_splits,
    dataset_summary,
)


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
UNK_TOKEN = "<unk>"
MASK_TOKEN = "<mask>"
SPECIAL_TOKENS = (PAD_TOKEN, BOS_TOKEN, UNK_TOKEN, MASK_TOKEN)
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|==|!=|<=|>=|[-+*/%(){}\[\].,:;]")


@dataclass(frozen=True)
class CorpusConfig:
    n_train: int = 512
    n_dev: int = 128
    n_test: int = 128
    num_candidates: int = 8
    n_views: int = 4
    max_files: int = 80
    snippet_radius: int = 3
    max_length: int = 64
    vocab_size: int = 4096
    mask_span_length: int = 4
    mask_strategy: str = "uniform"


@dataclass(frozen=True)
class ModelConfig:
    num_clones: int = 4
    hidden_size: int = 64
    num_layers: int = 3
    num_heads: int = 2
    ff_dim: int = 128
    dropout: float = 0.1
    identity_init_std: float = 0.02


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
class Stage14Config:
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
    def mask_id(self) -> int:
        return self.stoi[MASK_TOKEN]

    @property
    def size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        token_ids = [self.bos_id]
        token_ids.extend(self.stoi.get(token, self.unk_id) for token in _text_tokens(text))
        return token_ids


class MaskedSpanDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        vocab: Vocabulary,
        sequence_length: int,
        mask_span_length: int,
        mask_strategy: str,
        num_clones: int,
        seed: int,
    ) -> None:
        rows: List[Dict[str, torch.Tensor]] = []
        example_counter = 0
        for text in texts:
            token_ids = vocab.encode(text)
            if len(token_ids) < sequence_length:
                continue
            for start in range(0, len(token_ids) - sequence_length + 1):
                window = token_ids[start : start + sequence_length]
                mask_start = _mask_start_for_window(
                    sequence_length=sequence_length,
                    mask_span_length=mask_span_length,
                    strategy=mask_strategy,
                    num_clones=num_clones,
                    example_index=example_counter,
                    seed=seed,
                )
                target_ids = window[mask_start : mask_start + mask_span_length]
                masked_input = list(window)
                masked_input[mask_start : mask_start + mask_span_length] = [vocab.mask_id] * mask_span_length
                rows.append(
                    {
                        "input_ids": torch.as_tensor(masked_input, dtype=torch.long),
                        "target_ids": torch.as_tensor(target_ids, dtype=torch.long),
                        "attention_mask": torch.ones(sequence_length, dtype=torch.bool),
                        "mask_start": torch.as_tensor(mask_start, dtype=torch.long),
                    }
                )
                example_counter += 1
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.rows[index]


def _build_clone_window_mask(num_clones: int, sequence_length: int) -> torch.Tensor:
    if sequence_length % num_clones != 0:
        raise ValueError(
            f"sequence_length={sequence_length} must divide evenly across num_clones={num_clones}"
        )
    window_size = sequence_length // num_clones
    mask = torch.zeros(num_clones, sequence_length, dtype=torch.bool)
    for clone_index in range(num_clones):
        start = clone_index * window_size
        stop = start + window_size
        mask[clone_index, start:stop] = True
    return mask


def _mask_start_for_window(
    sequence_length: int,
    mask_span_length: int,
    strategy: str,
    num_clones: int,
    example_index: int,
    seed: int,
) -> int:
    max_start = sequence_length - mask_span_length
    if max_start < 0:
        raise ValueError(
            f"mask_span_length={mask_span_length} exceeds sequence_length={sequence_length}"
        )
    strategy_name = str(strategy).lower()
    if strategy_name == "center":
        return max(0, (sequence_length // 2) - (mask_span_length // 2))
    if strategy_name == "boundary":
        window_size = sequence_length // num_clones
        boundaries = []
        for boundary_index in range(1, num_clones):
            center = boundary_index * window_size
            start = max(0, min(max_start, center - (mask_span_length // 2)))
            boundaries.append(start)
        if not boundaries:
            return 0
        return boundaries[example_index % len(boundaries)]
    if strategy_name == "uniform":
        rng = random.Random((seed * 1_000_003) + example_index)
        return rng.randint(0, max_start)
    raise ValueError(f"unsupported mask strategy: {strategy}")


class SparseLocalCloneSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_clones: int,
        max_length: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.num_clones = int(num_clones)
        self.max_length = int(max_length)
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

    def forward(
        self,
        hidden: torch.Tensor,
        clone_token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, clones, tokens, _hidden = hidden.shape
        q = torch.einsum("bnth,nhd->bntd", hidden, self.q_weight)
        k = torch.einsum("bnth,nhd->bntd", hidden, self.k_weight)
        v = torch.einsum("bnth,nhd->bntd", hidden, self.v_weight)

        q = q.reshape(batch, clones, tokens, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k = k.reshape(batch, clones, tokens, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        v = v.reshape(batch, clones, tokens, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        key_mask = clone_token_mask.view(batch, clones, 1, 1, tokens)
        scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)

        probs = torch.softmax(scores.float(), dim=-1).to(dtype=hidden.dtype)
        probs = self.attn_dropout(probs)
        query_mask = clone_token_mask.view(batch, clones, 1, tokens, 1).to(dtype=hidden.dtype)
        probs = probs * query_mask

        attended = torch.matmul(probs, v).permute(0, 1, 3, 2, 4).reshape(batch, clones, tokens, self.hidden_size)
        attended = torch.einsum("bnth,nhd->bntd", attended, self.o_weight)
        attended = self.out_dropout(attended)
        attended = attended * clone_token_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        return attended, probs


class SparseLocalCloneBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_clones: int,
        max_length: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SparseLocalCloneSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_clones=num_clones,
            max_length=max_length,
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

    def forward(
        self,
        hidden: torch.Tensor,
        clone_token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_input = self.norm1(hidden)
        attn_out, attn_probs = self.attn(attn_input, clone_token_mask=clone_token_mask)
        hidden = hidden + attn_out
        ff_input = self.norm2(hidden)
        ff_out = self.ff(ff_input)
        ff_out = ff_out * clone_token_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        hidden = (hidden + ff_out) * clone_token_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        return hidden, attn_probs


class SpanMetaAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        span_length: int,
        dropout: float,
        init_std: float,
    ) -> None:
        super().__init__()
        self.span_queries = nn.Parameter(torch.empty(span_length, hidden_size))
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        nn.init.normal_(self.span_queries, mean=0.0, std=float(init_std))

    def forward(self, clone_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = clone_states.shape[0]
        query = self.span_queries.unsqueeze(0).expand(batch, -1, -1)
        meta_out, weights = self.attn(query, clone_states, clone_states, need_weights=True)
        return meta_out, weights


class MaskedSpanLocalCloneMetaAttentionLM(nn.Module):
    def __init__(self, vocab_size: int, max_length: int, span_length: int, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.max_length = int(max_length)
        self.span_length = int(span_length)
        self.window_mask = _build_clone_window_mask(config.num_clones, self.max_length)
        self.token_embedding = nn.Embedding(vocab_size, config.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, config.hidden_size)
        self.clone_identity = nn.Embedding(config.num_clones, config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [
                SparseLocalCloneBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    num_clones=config.num_clones,
                    max_length=max_length,
                    ff_dim=config.ff_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.meta_attention = SpanMetaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            span_length=self.span_length,
            dropout=config.dropout,
            init_std=config.identity_init_std,
        )
        self.lm_head = nn.Linear(config.hidden_size, vocab_size)
        nn.init.normal_(self.clone_identity.weight, mean=0.0, std=float(config.identity_init_std))

    def clone_token_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        window_mask = self.window_mask[:, : attention_mask.shape[1]].to(device=attention_mask.device)
        return attention_mask.unsqueeze(1) & window_mask.unsqueeze(0)

    def pooled_clone_states(
        self,
        clone_hidden: torch.Tensor,
        clone_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        weights = clone_token_mask.unsqueeze(-1).to(dtype=clone_hidden.dtype)
        totals = (clone_hidden * weights).sum(dim=2)
        counts = weights.sum(dim=2).clamp(min=1.0)
        return totals / counts

    def span_logits_from_clone_states(self, clone_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        meta_state, meta_weights = self.meta_attention(clone_states)
        logits = self.lm_head(meta_state)
        return logits, meta_weights

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_clone_states: bool = False,
        return_attention: bool = False,
    ) -> Dict[str, object]:
        batch, tokens = input_ids.shape
        device = input_ids.device
        positions = torch.arange(tokens, dtype=torch.long, device=device)
        token_hidden = self.token_embedding(input_ids) + self.position_embedding(positions).unsqueeze(0)
        clone_hidden = token_hidden.unsqueeze(1).expand(-1, self.config.num_clones, -1, -1).clone()
        clone_ids = torch.arange(self.config.num_clones, dtype=torch.long, device=device)
        clone_hidden = clone_hidden + self.clone_identity(clone_ids).view(1, self.config.num_clones, 1, -1)
        clone_token_mask = self.clone_token_mask(attention_mask)
        clone_hidden = self.embedding_dropout(clone_hidden)
        clone_hidden = clone_hidden * clone_token_mask.unsqueeze(-1).to(dtype=clone_hidden.dtype)

        attention_probs: List[torch.Tensor] = []
        for layer in self.layers:
            clone_hidden, probs = layer(clone_hidden, clone_token_mask=clone_token_mask)
            if return_attention:
                attention_probs.append(probs)
        clone_hidden = self.final_norm(clone_hidden)
        clone_hidden = clone_hidden * clone_token_mask.unsqueeze(-1).to(dtype=clone_hidden.dtype)
        pooled = self.pooled_clone_states(clone_hidden, clone_token_mask)
        logits, meta_weights = self.span_logits_from_clone_states(pooled)

        out: Dict[str, object] = {
            "joint_logits": logits,
            "meta_attention_weights": meta_weights.detach().cpu(),
        }
        if return_clone_states:
            out["clone_states"] = pooled
        if return_attention:
            out["attention_probs"] = attention_probs
            out["clone_token_mask"] = clone_token_mask
        return out


def _text_tokens(text: str) -> List[str]:
    return TOKEN_RE.findall(str(text).lower())


def _example_to_corpus_text(example: MultiViewTaskExample) -> str:
    parts: List[str] = []
    parts.extend(str(view.text).strip() for view in example.views if str(view.text).strip())
    parts.extend(str(candidate.text).strip() for candidate in example.candidates if str(candidate.text).strip())
    return "\n\n".join(parts)


def _build_corpus_splits(config: CorpusConfig, seed: int) -> tuple[Dict[str, List[str]], Dict[str, object]]:
    dataset_config = MultiViewCodePatchDatasetConfig(
        n_train=int(config.n_train),
        n_dev=int(config.n_dev),
        n_test=int(config.n_test),
        num_candidates=int(config.num_candidates),
        n_views=int(config.n_views),
        max_files=int(config.max_files),
        snippet_radius=int(config.snippet_radius),
        dataset_source=BALANCED_34B_DATASET_SOURCE,
        generator_version="real_import_restore_candidate_balanced_34b_stage14_sparse_local_clone_meta_attention_masked_span",
        candidate_representation=BALANCED_34B_CANDIDATE_REPRESENTATION,
    )
    splits = build_multiview_code_patch_splits(dataset_config, seed=int(seed), repo_root=Path("."))
    texts = {split: [_example_to_corpus_text(example) for example in rows] for split, rows in splits.items()}
    return texts, dataset_summary(splits)


def _build_vocab(texts: Sequence[str], vocab_size: int) -> Vocabulary:
    counts: Dict[str, int] = {}
    for text in texts:
        for token in _text_tokens(text):
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
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=bool(shuffle), generator=generator)


def _masked_span_loss(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_ids.reshape(-1),
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clone_attention_entropy(
    attention_probs: Sequence[torch.Tensor],
    clone_token_mask: torch.Tensor,
) -> List[float]:
    if not attention_probs:
        return []
    clone_count = attention_probs[0].shape[1]
    totals = torch.zeros(clone_count, dtype=torch.float64, device=clone_token_mask.device)
    counts = torch.zeros(clone_count, dtype=torch.float64, device=clone_token_mask.device)
    query_mask = clone_token_mask.to(dtype=torch.float64)
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


def _clone_window_coverage(
    attention_probs: Sequence[torch.Tensor],
    window_mask: torch.Tensor,
) -> Dict[str, object]:
    if not attention_probs:
        return {"mean_outside_window_mass": 0.0, "max_outside_window_mass": 0.0}
    base_mask = window_mask.to(dtype=torch.float32)
    outside_mass_values: List[float] = []
    for probs in attention_probs:
        tokens = probs.shape[-1]
        allowed = base_mask[:, :tokens].view(1, base_mask.shape[0], 1, 1, tokens).to(device=probs.device, dtype=probs.dtype)
        outside_mass = (probs * (1.0 - allowed)).sum(dim=-1)
        outside_mass_values.extend(float(value) for value in outside_mass.detach().cpu().reshape(-1).tolist())
    return {
        "mean_outside_window_mass": float(np.mean(outside_mass_values)) if outside_mass_values else 0.0,
        "max_outside_window_mass": float(np.max(outside_mass_values)) if outside_mass_values else 0.0,
    }


def _meta_attention_distribution(meta_weights: Sequence[torch.Tensor]) -> Dict[str, object]:
    if not meta_weights:
        return {
            "mean_weights": [],
            "std_weights": [],
            "mean_entropy": 0.0,
            "max_weight_mean": 0.0,
        }
    merged = torch.cat(meta_weights, dim=0).float()
    flattened = merged.reshape(-1, merged.shape[-1])
    entropy = -(flattened.clamp_min(1e-9) * flattened.clamp_min(1e-9).log()).sum(dim=-1)
    return {
        "mean_weights": [float(value) for value in flattened.mean(dim=0).cpu().tolist()],
        "std_weights": [float(value) for value in flattened.std(dim=0, unbiased=False).cpu().tolist()],
        "mean_entropy": float(entropy.mean().cpu().item()),
        "max_weight_mean": float(flattened.max(dim=-1).values.mean().cpu().item()),
    }


def _target_window_distribution(mask_starts: Sequence[torch.Tensor], window_size: int, num_clones: int) -> Dict[str, object]:
    if not mask_starts:
        return {"mean_mass_per_clone": [0.0] * num_clones}
    merged = torch.cat(mask_starts, dim=0).cpu().tolist()
    counts = np.zeros(num_clones, dtype=np.float64)
    for start in merged:
        clone_index = min(num_clones - 1, int(start) // int(window_size))
        counts[clone_index] += 1.0
    total = float(counts.sum())
    return {"mean_mass_per_clone": [float(value / max(1.0, total)) for value in counts]}


def _amp_context(device: torch.device, use_bf16: bool):
    if device.type == "cuda" and use_bf16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def evaluate_model(
    model: MaskedSpanLocalCloneMetaAttentionLM,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    joint_nll = 0.0
    token_count = 0
    clone_nll = torch.zeros(model.config.num_clones, dtype=torch.float64, device=device)
    clone_tokens = torch.zeros(model.config.num_clones, dtype=torch.float64, device=device)
    attention_entropy_sum = np.zeros(model.config.num_clones, dtype=np.float64)
    cosine_values: List[float] = []
    cosine_matrices: List[np.ndarray] = []
    meta_weight_batches: List[torch.Tensor] = []
    all_attention_probs: List[List[torch.Tensor]] | None = None
    clone_mask_batches: List[torch.Tensor] = []
    mask_start_batches: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_clone_states=True,
                return_attention=True,
            )
            joint_logits = outputs["joint_logits"]
            clone_states = outputs["clone_states"]
            attention_probs = outputs["attention_probs"]
            clone_token_mask = outputs["clone_token_mask"]
            meta_weights = outputs["meta_attention_weights"]

            joint_loss = F.cross_entropy(
                joint_logits.reshape(-1, joint_logits.shape[-1]),
                target_ids.reshape(-1),
                reduction="sum",
            )
            joint_nll += float(joint_loss.detach().cpu().item())
            token_count += int(target_ids.numel())

            for clone_index in range(model.config.num_clones):
                single_logits, _single_weights = model.span_logits_from_clone_states(clone_states[:, clone_index : clone_index + 1])
                clone_loss = F.cross_entropy(
                    single_logits.reshape(-1, single_logits.shape[-1]),
                    target_ids.reshape(-1),
                    reduction="sum",
                )
                clone_nll[clone_index] += clone_loss.detach()
                clone_tokens[clone_index] += float(target_ids.numel())

            attention_entropy_sum += np.asarray(_clone_attention_entropy(attention_probs, clone_token_mask), dtype=np.float64)
            cosine = _pairwise_clone_cosine(clone_states)
            cosine_values.append(float(cosine["mean_off_diagonal"]))
            cosine_matrices.append(np.asarray(cosine["matrix"], dtype=np.float64))
            meta_weight_batches.append(meta_weights)
            clone_mask_batches.append(clone_token_mask.detach().cpu())
            mask_start_batches.append(batch["mask_start"].detach().cpu())
            if all_attention_probs is None:
                all_attention_probs = [[] for _layer in attention_probs]
            for layer_index, probs in enumerate(attention_probs):
                all_attention_probs[layer_index].append(probs.detach().cpu())

    clone_perplexity = [
        math.exp(float(clone_nll[index].detach().cpu().item()) / max(1.0, float(clone_tokens[index].detach().cpu().item())))
        for index in range(model.config.num_clones)
    ]
    mean_cosine_matrix = np.mean(cosine_matrices, axis=0) if cosine_matrices else np.eye(model.config.num_clones, dtype=np.float64)
    attention_prob_tensors = (
        [torch.cat(layer_batches, dim=0) for layer_batches in all_attention_probs] if all_attention_probs is not None else []
    )
    meta_distribution = _meta_attention_distribution(meta_weight_batches)
    coverage = _clone_window_coverage(attention_prob_tensors, model.window_mask)
    target_distribution = _target_window_distribution(
        mask_start_batches,
        window_size=(model.max_length // model.config.num_clones),
        num_clones=model.config.num_clones,
    )
    joint_perplexity = math.exp(joint_nll / max(1, token_count))
    return {
        "joint_perplexity": float(joint_perplexity),
        "clone_perplexity": [float(value) for value in clone_perplexity],
        "mean_clone_perplexity": float(np.mean(clone_perplexity)) if clone_perplexity else float("nan"),
        "clone_perplexity_spread": float(max(clone_perplexity) - min(clone_perplexity)) if clone_perplexity else 0.0,
        "joint_beats_best_clone": bool(joint_perplexity < min(clone_perplexity)) if clone_perplexity else False,
        "mean_off_diagonal_clone_cosine": float(np.mean(cosine_values)) if cosine_values else 1.0,
        "clone_cosine_matrix": [[float(value) for value in row] for row in mean_cosine_matrix.tolist()],
        "attention_entropy_per_clone": [float(value) for value in (attention_entropy_sum / max(1, len(loader)))],
        "meta_attention_distribution": meta_distribution,
        "meta_attention_weights_last_batch": meta_weight_batches[-1].tolist() if meta_weight_batches else [],
        "clone_window_coverage": coverage,
        "clone_window_layout": [
            [int(index) for index, active in enumerate(row.tolist()) if active]
            for row in model.window_mask
        ],
        "target_window_distribution": target_distribution,
        "token_count": int(token_count),
    }


def train_on_text_splits(
    text_splits: Dict[str, Sequence[str]],
    config: Stage14Config,
) -> Dict[str, object]:
    _set_seed(int(config.training.seed))
    device = torch.device(config.training.device)

    vocab = _build_vocab(text_splits["train"], vocab_size=int(config.corpus.vocab_size))
    train_dataset = MaskedSpanDataset(
        text_splits["train"],
        vocab=vocab,
        sequence_length=int(config.corpus.max_length),
        mask_span_length=int(config.corpus.mask_span_length),
        mask_strategy=str(config.corpus.mask_strategy),
        num_clones=int(config.model.num_clones),
        seed=int(config.training.seed),
    )
    dev_dataset = MaskedSpanDataset(
        text_splits["dev"],
        vocab=vocab,
        sequence_length=int(config.corpus.max_length),
        mask_span_length=int(config.corpus.mask_span_length),
        mask_strategy=str(config.corpus.mask_strategy),
        num_clones=int(config.model.num_clones),
        seed=int(config.training.seed) + 1,
    )
    test_dataset = MaskedSpanDataset(
        text_splits["test"],
        vocab=vocab,
        sequence_length=int(config.corpus.max_length),
        mask_span_length=int(config.corpus.mask_span_length),
        mask_strategy=str(config.corpus.mask_strategy),
        num_clones=int(config.model.num_clones),
        seed=int(config.training.seed) + 2,
    )

    train_loader = _make_loader(train_dataset, batch_size=int(config.training.batch_size), shuffle=True, seed=int(config.training.seed))
    dev_loader = _make_loader(dev_dataset, batch_size=int(config.training.batch_size), shuffle=False, seed=int(config.training.seed) + 1)
    test_loader = _make_loader(test_dataset, batch_size=int(config.training.batch_size), shuffle=False, seed=int(config.training.seed) + 2)

    model = MaskedSpanLocalCloneMetaAttentionLM(
        vocab_size=vocab.size,
        max_length=int(config.corpus.max_length),
        span_length=int(config.corpus.mask_span_length),
        config=config.model,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.training.lr),
        weight_decay=float(config.training.weight_decay),
    )

    history: List[Dict[str, object]] = []
    best_dev_perplexity = float("inf")
    best_epoch = 0
    best_state: Dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    for epoch in range(int(config.training.epochs)):
        model.train()
        loss_sum = 0.0
        batch_count = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with _amp_context(device, bool(config.training.use_bf16)):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_clone_states=False,
                    return_attention=False,
                )
                loss = _masked_span_loss(outputs["joint_logits"], target_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(config.training.gradient_clip))
            optimizer.step()
            loss_sum += float(loss.detach().cpu().item())
            batch_count += 1

        dev_metrics = evaluate_model(model, dev_loader, device=device)
        history.append(
            {
                "epoch": int(epoch + 1),
                "train_loss": float(loss_sum / max(1, batch_count)),
                "train_perplexity": float(math.exp(loss_sum / max(1, batch_count))),
                "dev_joint_perplexity": float(dev_metrics["joint_perplexity"]),
                "dev_clone_perplexity": [float(value) for value in dev_metrics["clone_perplexity"]],
                "dev_clone_perplexity_spread": float(dev_metrics["clone_perplexity_spread"]),
                "dev_mean_off_diagonal_clone_cosine": float(dev_metrics["mean_off_diagonal_clone_cosine"]),
                "dev_attention_entropy_per_clone": [float(value) for value in dev_metrics["attention_entropy_per_clone"]],
                "dev_meta_attention_mean_weights": [
                    float(value) for value in dev_metrics["meta_attention_distribution"]["mean_weights"]
                ],
            }
        )

        if float(dev_metrics["joint_perplexity"]) < best_dev_perplexity:
            best_dev_perplexity = float(dev_metrics["joint_perplexity"])
            best_epoch = int(epoch + 1)
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(config.training.early_stopping_patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_dev = evaluate_model(model, dev_loader, device=device)
    final_test = evaluate_model(model, test_loader, device=device)
    return {
        "config": asdict(config),
        "vocab_size": int(vocab.size),
        "sequence_length": int(config.corpus.max_length),
        "mask_span_length": int(config.corpus.mask_span_length),
        "mask_strategy": str(config.corpus.mask_strategy),
        "dataset_sizes": {
            "train_sequences": len(train_dataset),
            "dev_sequences": len(dev_dataset),
            "test_sequences": len(test_dataset),
        },
        "window_size": int(config.corpus.max_length // config.model.num_clones),
        "best_epoch": int(best_epoch),
        "stopped_early": bool(best_epoch < int(config.training.epochs)),
        "history": history,
        "final_dev": final_dev,
        "final_test": final_test,
    }


def write_report(results: Dict[str, object], report_path: Path) -> None:
    history = list(results.get("history", []))
    final_test = dict(results.get("final_test", {}))
    meta_distribution = dict(final_test.get("meta_attention_distribution", {}))
    clone_cosine = float(final_test.get("mean_off_diagonal_clone_cosine", 1.0))
    delta_vs_stage13 = clone_cosine - STAGE13_CLONE_COSINE_BASELINE
    lines = [
        "# Stage 14 Sparse Local Clone Meta-Attention",
        "",
        "## Setup",
        f"- Mask strategy / span length: `{results['mask_strategy']}` / `{results['mask_span_length']}`",
        f"- Clones: `{results['config']['model']['num_clones']}`",
        f"- Hidden size / layers: `{results['config']['model']['hidden_size']}` / `{results['config']['model']['num_layers']}`",
        f"- Sequence length / window size: `{results['sequence_length']}` / `{results['window_size']}`",
        f"- Vocab size: `{results['vocab_size']}`",
        f"- Dataset sequences: `{results['dataset_sizes']}`",
        f"- Best epoch: `{results['best_epoch']}`",
        "",
        "## Final Held-Out Metrics",
        f"- Joint perplexity: `{final_test.get('joint_perplexity', float('nan')):.4f}`",
        f"- Per-clone perplexity: `{[round(float(value), 4) for value in final_test.get('clone_perplexity', [])]}`",
        f"- Joint beats best single clone: `{bool(final_test.get('joint_beats_best_clone', False))}`",
        f"- Mean off-diagonal clone cosine: `{clone_cosine:.6f}`",
        f"- Clone cosine dropped vs Stage 13 `0.99997`: `{clone_cosine < STAGE13_CLONE_COSINE_BASELINE}` (delta `{delta_vs_stage13:+.6f}`)",
        f"- Attention entropy per clone: `{[round(float(value), 4) for value in final_test.get('attention_entropy_per_clone', [])]}`",
        f"- Meta-attention mean weights: `{[round(float(value), 4) for value in meta_distribution.get('mean_weights', [])]}`",
        f"- Meta-attention std weights: `{[round(float(value), 4) for value in meta_distribution.get('std_weights', [])]}`",
        f"- Meta-attention mean entropy: `{float(meta_distribution.get('mean_entropy', 0.0)):.4f}`",
        f"- Meta-attention mean max weight: `{float(meta_distribution.get('max_weight_mean', 0.0)):.4f}`",
        f"- Target window distribution: `{[round(float(value), 4) for value in final_test.get('target_window_distribution', {}).get('mean_mass_per_clone', [])]}`",
        f"- Clone window outside-mass mean / max: `{float(final_test.get('clone_window_coverage', {}).get('mean_outside_window_mass', 0.0)):.6f}` / `{float(final_test.get('clone_window_coverage', {}).get('max_outside_window_mass', 0.0)):.6f}`",
        "",
        "## Fixed Windows",
    ]
    for clone_index, positions in enumerate(final_test.get("clone_window_layout", [])):
        if positions:
            lines.append(f"- Clone {clone_index}: `[{positions[0]}:{positions[-1] + 1}]`")
    lines.extend(
        [
            "",
            "## Training Trend",
        ]
    )
    if history:
        lines.extend(
            [
                f"- Initial dev joint perplexity: `{float(history[0]['dev_joint_perplexity']):.4f}`",
                f"- Final dev joint perplexity: `{float(history[-1]['dev_joint_perplexity']):.4f}`",
                f"- Initial clone perplexity spread: `{float(history[0]['dev_clone_perplexity_spread']):.4f}`",
                f"- Final clone perplexity spread: `{float(history[-1]['dev_clone_perplexity_spread']):.4f}`",
                f"- Initial mean off-diagonal clone cosine: `{float(history[0]['dev_mean_off_diagonal_clone_cosine']):.6f}`",
                f"- Final mean off-diagonal clone cosine: `{float(history[-1]['dev_mean_off_diagonal_clone_cosine']):.6f}`",
            ]
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(config: Stage14Config | None = None) -> Dict[str, object]:
    active = config or Stage14Config()
    text_splits, summary = _build_corpus_splits(active.corpus, seed=int(active.training.seed))
    results = train_on_text_splits(text_splits, active)
    results["corpus_summary"] = summary
    write_report(results, REPORT_PATH)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 14 sparse local-clone meta-attention masked-span variant.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-train", type=int, default=512)
    parser.add_argument("--n-dev", type=int, default=128)
    parser.add_argument("--n-test", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--mask-span-length", type=int, default=4)
    parser.add_argument("--mask-strategy", default="uniform", choices=["uniform", "boundary", "center"])
    args = parser.parse_args()

    config = Stage14Config(
        corpus=CorpusConfig(
            n_train=int(args.n_train),
            n_dev=int(args.n_dev),
            n_test=int(args.n_test),
            max_length=int(args.max_length),
            mask_span_length=int(args.mask_span_length),
            mask_strategy=str(args.mask_strategy),
        ),
        training=TrainingConfig(
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            seed=int(args.seed),
            device=str(args.device),
        ),
    )
    results = run_experiment(config)
    print(
        "stage14: wrote {results} and {report}; joint_ppl={ppl:.4f}; cosine={cosine:.6f}".format(
            results=RESULTS_PATH,
            report=REPORT_PATH,
            ppl=float(results["final_test"]["joint_perplexity"]),
            cosine=float(results["final_test"]["mean_off_diagonal_clone_cosine"]),
        )
    )


if __name__ == "__main__":
    main()
