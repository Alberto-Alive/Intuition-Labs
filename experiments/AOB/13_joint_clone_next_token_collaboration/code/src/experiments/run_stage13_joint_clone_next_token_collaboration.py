from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


EXPERIMENT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_PATH = EXPERIMENT_ROOT / "results" / "stage13_joint_clone_next_token_collaboration_results.json"
REPORT_PATH = EXPERIMENT_ROOT / "reports" / "STAGE13_JOINT_CLONE_NEXT_TOKEN_COLLABORATION.md"

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
SPECIAL_TOKENS = (PAD_TOKEN, BOS_TOKEN, UNK_TOKEN)
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
    max_length: int = 128
    vocab_size: int = 4096


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
    epochs: int = 8
    batch_size: int = 16
    lr: float = 0.001
    weight_decay: float = 0.0001
    seed: int = 0
    device: str = "cpu"


@dataclass(frozen=True)
class Stage13Config:
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
    def size(self) -> int:
        return len(self.itos)

    def encode(self, text: str, max_length: int) -> List[int]:
        token_ids = [self.bos_id]
        for token in _text_tokens(text)[:max_length]:
            token_ids.append(self.stoi.get(token, self.unk_id))
        return token_ids


class NextTokenDataset(Dataset):
    def __init__(self, texts: Sequence[str], vocab: Vocabulary, max_length: int) -> None:
        rows: List[Dict[str, torch.Tensor]] = []
        for text in texts:
            token_ids = vocab.encode(text, max_length=max_length)
            if len(token_ids) < 2:
                continue
            token_ids = token_ids[: max_length + 1]
            input_ids = token_ids[:-1]
            target_ids = token_ids[1:]
            mask = [1] * len(input_ids)
            pad = max_length - len(input_ids)
            if pad > 0:
                input_ids = input_ids + [vocab.pad_id] * pad
                target_ids = target_ids + [-100] * pad
                mask = mask + [0] * pad
            rows.append(
                {
                    "input_ids": torch.as_tensor(input_ids, dtype=torch.long),
                    "target_ids": torch.as_tensor(target_ids, dtype=torch.long),
                    "attention_mask": torch.as_tensor(mask, dtype=torch.bool),
                }
            )
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.rows[index]


class CollaborativeCloneSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_clones: int, max_length: int, dropout: float, bias_init_std: float) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.num_clones = int(num_clones)
        self.max_length = int(max_length)
        self.head_dim = self.hidden_size // self.num_heads
        self.conditioned_dim = self.hidden_size * max(1, self.num_clones)
        self.q_proj = nn.Linear(self.conditioned_dim, hidden_size)
        self.k_proj = nn.Linear(self.conditioned_dim, hidden_size)
        self.v_proj = nn.Linear(self.conditioned_dim, hidden_size)
        self.clone_attention_bias = nn.Parameter(
            torch.empty(self.num_clones, self.num_heads, self.max_length, self.max_length)
        )
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        nn.init.normal_(self.clone_attention_bias, mean=0.0, std=float(bias_init_std))

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        override_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conditioned = _concat_with_detached_other_clones(hidden)
        batch, clones, tokens, _dim = conditioned.shape
        q = self.q_proj(conditioned).reshape(batch, clones, tokens, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k = self.k_proj(conditioned).reshape(batch, clones, tokens, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        v = self.v_proj(conditioned).reshape(batch, clones, tokens, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        bias = self.clone_attention_bias if override_bias is None else override_bias
        scores = scores + bias[:, :, :tokens, :tokens].unsqueeze(0).to(dtype=scores.dtype)
        causal_mask = torch.triu(
            torch.ones(tokens, tokens, dtype=torch.bool, device=hidden.device),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask.view(1, 1, 1, tokens, tokens), torch.finfo(scores.dtype).min)
        key_mask = attention_mask.view(batch, 1, 1, 1, tokens)
        scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)

        probs = torch.softmax(scores.float(), dim=-1).to(dtype=hidden.dtype)
        probs = self.attn_dropout(probs)
        query_mask = attention_mask.view(batch, 1, 1, tokens, 1).to(dtype=hidden.dtype)
        probs = probs * query_mask

        attended = torch.matmul(probs, v).permute(0, 1, 3, 2, 4).reshape(batch, clones, tokens, self.hidden_size)
        attended = self.out_dropout(self.out_proj(attended))
        attended = attended * attention_mask.view(batch, 1, tokens, 1).to(dtype=hidden.dtype)
        return attended, probs


class CollaborativeCloneBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_clones: int,
        max_length: int,
        ff_dim: int,
        dropout: float,
        bias_init_std: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = CollaborativeCloneSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_clones=num_clones,
            max_length=max_length,
            dropout=dropout,
            bias_init_std=bias_init_std,
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
        attention_mask: torch.Tensor,
        override_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_input = self.norm1(hidden)
        attn_out, attn_probs = self.attn(attn_input, attention_mask=attention_mask, override_bias=override_bias)
        hidden = hidden + attn_out
        ff_input = self.norm2(hidden)
        ff_out = self.ff(ff_input)
        ff_out = ff_out * attention_mask.view(hidden.shape[0], 1, hidden.shape[2], 1).to(dtype=hidden.dtype)
        hidden = hidden + ff_out
        return hidden, attn_probs


class CollaborativeCloneLM(nn.Module):
    def __init__(self, vocab_size: int, max_length: int, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(vocab_size, config.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, config.hidden_size)
        self.clone_identity = nn.Embedding(config.num_clones, config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [
                CollaborativeCloneBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    num_clones=config.num_clones,
                    max_length=max_length,
                    ff_dim=config.ff_dim,
                    dropout=config.dropout,
                    bias_init_std=config.identity_init_std,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.clone_mix_logits = nn.Parameter(torch.zeros(config.num_clones))
        self.lm_head = nn.Linear(config.hidden_size, vocab_size)
        nn.init.normal_(self.clone_identity.weight, mean=0.0, std=float(config.identity_init_std))

    def joint_weights(self) -> torch.Tensor:
        return torch.softmax(self.clone_mix_logits, dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_clone_states: bool = False,
        return_attention: bool = False,
        override_attention_biases: Sequence[torch.Tensor] | None = None,
    ) -> Dict[str, object]:
        batch, tokens = input_ids.shape
        device = input_ids.device
        positions = torch.arange(tokens, dtype=torch.long, device=device)
        token_hidden = self.token_embedding(input_ids) + self.position_embedding(positions).unsqueeze(0)
        clone_hidden = token_hidden.unsqueeze(1).expand(-1, self.config.num_clones, -1, -1).clone()
        clone_ids = torch.arange(self.config.num_clones, dtype=torch.long, device=device)
        clone_hidden = clone_hidden + self.clone_identity(clone_ids).view(1, self.config.num_clones, 1, -1)
        clone_hidden = self.embedding_dropout(clone_hidden)
        clone_hidden = clone_hidden * attention_mask.view(batch, 1, tokens, 1).to(dtype=clone_hidden.dtype)

        attention_probs: List[torch.Tensor] = []
        for layer_index, layer in enumerate(self.layers):
            layer_bias = None
            if override_attention_biases is not None:
                layer_bias = override_attention_biases[layer_index]
            clone_hidden, probs = layer(clone_hidden, attention_mask=attention_mask, override_bias=layer_bias)
            if return_attention:
                attention_probs.append(probs)
        clone_hidden = self.final_norm(clone_hidden)
        clone_hidden = clone_hidden * attention_mask.view(batch, 1, tokens, 1).to(dtype=clone_hidden.dtype)

        weights = self.joint_weights()
        joint_state = torch.einsum("n,bntd->btd", weights, clone_hidden)
        joint_logits = self.lm_head(joint_state)
        out: Dict[str, object] = {"joint_logits": joint_logits, "joint_weights": weights.detach().cpu()}
        if return_clone_states:
            out["clone_states"] = clone_hidden
        if return_attention:
            out["attention_probs"] = attention_probs
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
        generator_version="real_import_restore_candidate_balanced_34b_stage13_joint_clone_lm",
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


def _joint_loss(joint_logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        joint_logits.reshape(-1, joint_logits.shape[-1]),
        target_ids.reshape(-1),
        ignore_index=-100,
    )


def _other_clone_concat(hidden: torch.Tensor) -> torch.Tensor:
    if hidden.dim() != 4:
        raise ValueError(f"expected [batch, clones, tokens, hidden], got {tuple(hidden.shape)}")
    batch, clones, tokens, hidden_dim = hidden.shape
    if clones == 1:
        return torch.zeros(batch, clones, tokens, 0, dtype=hidden.dtype, device=hidden.device)
    pieces = []
    for clone_index in range(clones):
        others = [hidden[:, other_index] for other_index in range(clones) if other_index != clone_index]
        pieces.append(torch.cat(others, dim=-1))
    return torch.stack(pieces, dim=1)


def _concat_with_detached_other_clones(hidden: torch.Tensor) -> torch.Tensor:
    return torch.cat([hidden, _other_clone_concat(hidden).detach()], dim=-1)


def _weighted_clone_sum(clone_states: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.einsum("n,bntd->btd", weights, clone_states)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clone_attention_entropy(attention_probs: Sequence[torch.Tensor], attention_mask: torch.Tensor) -> List[float]:
    if not attention_probs:
        return []
    clone_count = attention_probs[0].shape[1]
    totals = torch.zeros(clone_count, dtype=torch.float64, device=attention_mask.device)
    counts = torch.zeros(clone_count, dtype=torch.float64, device=attention_mask.device)
    query_mask = attention_mask.to(dtype=torch.float64)
    for probs in attention_probs:
        entropy = -(probs.clamp_min(1e-9).float() * probs.clamp_min(1e-9).float().log()).sum(dim=-1)
        masked = entropy.to(dtype=torch.float64) * query_mask.view(query_mask.shape[0], 1, 1, query_mask.shape[1])
        totals += masked.sum(dim=(0, 2, 3))
        counts += query_mask.sum().to(dtype=torch.float64) * probs.shape[2]
    values = totals / counts.clamp(min=1.0)
    return [float(value.detach().cpu().item()) for value in values]


def _safe_prob_kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = p.clamp_min(1e-9)
    q = q.clamp_min(1e-9)
    return (p * (p.log() - q.log())).sum(dim=-1)


def _symmetrized_attention_kl(attention_probs: Sequence[torch.Tensor], attention_mask: torch.Tensor) -> Dict[str, object]:
    if not attention_probs:
        return {"per_layer_mean": [], "overall_mean": 0.0, "clone_pair_matrix": []}
    clone_count = attention_probs[0].shape[1]
    pair_totals = torch.zeros(clone_count, clone_count, dtype=torch.float64, device=attention_mask.device)
    pair_counts = torch.zeros(clone_count, clone_count, dtype=torch.float64, device=attention_mask.device)
    per_layer_mean: List[float] = []
    query_mask = attention_mask.view(attention_mask.shape[0], 1, 1, attention_mask.shape[1]).to(dtype=torch.float64)
    for probs in attention_probs:
        layer_total = 0.0
        layer_count = 0
        for left in range(clone_count):
            for right in range(left + 1, clone_count):
                p = probs[:, left]
                q = probs[:, right]
                sym = 0.5 * (_safe_prob_kl(p, q) + _safe_prob_kl(q, p))
                masked = sym.to(dtype=torch.float64) * query_mask
                total = masked.sum()
                count = query_mask.sum() * probs.shape[2]
                pair_totals[left, right] += total
                pair_totals[right, left] += total
                pair_counts[left, right] += count
                pair_counts[right, left] += count
                layer_total += float(total.detach().cpu().item())
                layer_count += int(count.detach().cpu().item())
        per_layer_mean.append(layer_total / max(1, layer_count))
    matrix = torch.zeros_like(pair_totals)
    valid = pair_counts > 0
    matrix[valid] = pair_totals[valid] / pair_counts[valid]
    off_diag = []
    for row in range(clone_count):
        for col in range(row + 1, clone_count):
            off_diag.append(float(matrix[row, col].detach().cpu().item()))
    return {
        "per_layer_mean": [float(value) for value in per_layer_mean],
        "overall_mean": float(np.mean(off_diag)) if off_diag else 0.0,
        "clone_pair_matrix": [[float(value) for value in line] for line in matrix.detach().cpu().tolist()],
    }


def _mean_attention_distribution_shift(
    attention_probs: Sequence[torch.Tensor],
    zero_attention_probs: Sequence[torch.Tensor],
    attention_mask: torch.Tensor,
) -> Dict[str, object]:
    if not attention_probs or not zero_attention_probs:
        return {"per_layer_mean": [], "overall_mean": 0.0}
    query_mask = attention_mask.view(attention_mask.shape[0], 1, 1, attention_mask.shape[1]).to(dtype=torch.float64)
    per_layer = []
    for actual, zeroed in zip(attention_probs, zero_attention_probs):
        sym = 0.5 * (_safe_prob_kl(actual, zeroed) + _safe_prob_kl(zeroed, actual))
        masked = sym.to(dtype=torch.float64) * query_mask
        per_layer.append(float(masked.sum().detach().cpu().item() / max(1, int((query_mask.sum() * actual.shape[2]).detach().cpu().item()))))
    return {"per_layer_mean": [float(value) for value in per_layer], "overall_mean": float(np.mean(per_layer)) if per_layer else 0.0}


def _attention_bias_norms(model: CollaborativeCloneLM) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for layer_index, layer in enumerate(model.layers):
        bias = layer.attn.clone_attention_bias.detach().float()
        per_clone_l2 = bias.reshape(bias.shape[0], -1).norm(dim=1)
        per_clone_abs = bias.abs().reshape(bias.shape[0], -1).mean(dim=1)
        rows.append(
            {
                "layer": int(layer_index),
                "per_clone_l2_norm": [float(value) for value in per_clone_l2.cpu().tolist()],
                "per_clone_mean_abs": [float(value) for value in per_clone_abs.cpu().tolist()],
                "layer_l2_norm": float(bias.norm().cpu().item()),
                "layer_mean_abs": float(bias.abs().mean().cpu().item()),
            }
        )
    return rows


def _pairwise_clone_cosine(clone_states: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, object]:
    mask = attention_mask.unsqueeze(1).unsqueeze(-1).to(dtype=clone_states.dtype)
    pooled = (clone_states * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1.0)
    pooled = F.normalize(pooled.float(), dim=-1)
    cosine = torch.einsum("bnd,bmd->bnm", pooled, pooled).mean(dim=0)
    off_diag = []
    for row in range(cosine.shape[0]):
        for col in range(row + 1, cosine.shape[1]):
            off_diag.append(float(cosine[row, col].detach().cpu().item()))
    return {
        "mean_off_diagonal": float(np.mean(off_diag)) if off_diag else 1.0,
        "matrix": [[float(value) for value in line] for line in cosine.detach().cpu().tolist()],
    }


def evaluate_model(model: CollaborativeCloneLM, loader: DataLoader, pad_id: int, device: torch.device) -> Dict[str, object]:
    del pad_id
    model.eval()
    joint_nll = 0.0
    joint_tokens = 0
    clone_nll = torch.zeros(model.config.num_clones, dtype=torch.float64, device=device)
    clone_tokens = torch.zeros(model.config.num_clones, dtype=torch.float64, device=device)
    attention_entropy_sum = np.zeros(model.config.num_clones, dtype=np.float64)
    cosine_values: List[float] = []
    cosine_matrices: List[np.ndarray] = []
    joint_weights: List[List[float]] = []
    merged_attention_probs: List[List[torch.Tensor]] | None = None
    bias_zero_joint_nll = 0.0
    merged_zero_attention_probs: List[List[torch.Tensor]] | None = None
    all_attention_masks: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            all_attention_masks.append(attention_mask.detach().cpu())
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_clone_states=True,
                return_attention=True,
            )
            joint_logits = outputs["joint_logits"]
            clone_states = outputs["clone_states"]
            attention_probs = outputs["attention_probs"]
            joint_weights.append([float(value) for value in outputs["joint_weights"].tolist()])
            if merged_attention_probs is None:
                merged_attention_probs = [[] for _layer in attention_probs]
            for layer_index, probs in enumerate(attention_probs):
                merged_attention_probs[layer_index].append(probs.detach().cpu())

            joint_loss = F.cross_entropy(
                joint_logits.reshape(-1, joint_logits.shape[-1]),
                target_ids.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            valid = target_ids.ne(-100).sum().item()
            joint_nll += float(joint_loss.detach().cpu().item())
            joint_tokens += int(valid)

            for clone_index in range(model.config.num_clones):
                clone_logits = model.lm_head(clone_states[:, clone_index])
                clone_loss = F.cross_entropy(
                    clone_logits.reshape(-1, clone_logits.shape[-1]),
                    target_ids.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                clone_nll[clone_index] += clone_loss.detach()
                clone_tokens[clone_index] += float(valid)

            attention_entropy_sum += np.asarray(_clone_attention_entropy(attention_probs, attention_mask), dtype=np.float64)
            cosine = _pairwise_clone_cosine(clone_states, attention_mask)
            cosine_values.append(float(cosine["mean_off_diagonal"]))
            cosine_matrices.append(np.asarray(cosine["matrix"], dtype=np.float64))

            zero_biases = [torch.zeros_like(layer.attn.clone_attention_bias) for layer in model.layers]
            zero_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_clone_states=False,
                return_attention=True,
                override_attention_biases=zero_biases,
            )
            zero_logits = zero_outputs["joint_logits"]
            zero_loss = F.cross_entropy(
                zero_logits.reshape(-1, zero_logits.shape[-1]),
                target_ids.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            bias_zero_joint_nll += float(zero_loss.detach().cpu().item())
            if merged_zero_attention_probs is None:
                merged_zero_attention_probs = [[] for _layer in zero_outputs["attention_probs"]]
            for layer_index, probs in enumerate(zero_outputs["attention_probs"]):
                merged_zero_attention_probs[layer_index].append(probs.detach().cpu())

    joint_perplexity = math.exp(joint_nll / max(1, joint_tokens))
    clone_perplexity = [
        math.exp(float(clone_nll[index].detach().cpu().item()) / max(1.0, float(clone_tokens[index].detach().cpu().item())))
        for index in range(model.config.num_clones)
    ]
    mean_cosine_matrix = np.mean(cosine_matrices, axis=0) if cosine_matrices else np.eye(model.config.num_clones, dtype=np.float64)
    merged_mask = torch.cat(all_attention_masks, dim=0).to(device=device) if all_attention_masks else torch.ones(1, 1, dtype=torch.bool, device=device)
    attention_prob_tensors = (
        [torch.cat(layer_batches, dim=0).to(device=device) for layer_batches in merged_attention_probs]
        if merged_attention_probs is not None
        else []
    )
    zero_attention_prob_tensors = (
        [torch.cat(layer_batches, dim=0).to(device=device) for layer_batches in merged_zero_attention_probs]
        if merged_zero_attention_probs is not None
        else []
    )
    attention_kl = _symmetrized_attention_kl(
        attention_prob_tensors,
        merged_mask,
    )
    bias_zero_attention_kl = _symmetrized_attention_kl(
        zero_attention_prob_tensors,
        merged_mask,
    )
    bias_effect_kl = _mean_attention_distribution_shift(attention_prob_tensors, zero_attention_prob_tensors, merged_mask)
    return {
        "joint_perplexity": float(joint_perplexity),
        "clone_perplexity": [float(value) for value in clone_perplexity],
        "mean_clone_perplexity": float(np.mean(clone_perplexity)) if clone_perplexity else float("nan"),
        "clone_perplexity_spread": float(max(clone_perplexity) - min(clone_perplexity)) if clone_perplexity else 0.0,
        "joint_beats_best_clone": bool(joint_perplexity < min(clone_perplexity)) if clone_perplexity else False,
        "mean_off_diagonal_clone_cosine": float(np.mean(cosine_values)) if cosine_values else 1.0,
        "clone_cosine_matrix": [[float(value) for value in row] for row in mean_cosine_matrix.tolist()],
        "attention_entropy_per_clone": [float(value) for value in (attention_entropy_sum / max(1, len(loader)))],
        "pairwise_attention_kl": attention_kl,
        "bias_zero_pairwise_attention_kl": bias_zero_attention_kl,
        "bias_used_diagnostic": {
            "joint_perplexity_without_bias": float(math.exp(bias_zero_joint_nll / max(1, joint_tokens))),
            "joint_perplexity_delta_without_bias": float(math.exp(bias_zero_joint_nll / max(1, joint_tokens)) - joint_perplexity),
            "mean_attention_shift_kl_without_bias": bias_effect_kl,
        },
        "joint_weights": joint_weights[-1] if joint_weights else [],
        "token_count": int(joint_tokens),
    }


def train_on_text_splits(
    text_splits: Dict[str, Sequence[str]],
    config: Stage13Config,
) -> Dict[str, object]:
    _set_seed(int(config.training.seed))
    device = torch.device(config.training.device)

    vocab = _build_vocab(text_splits["train"], vocab_size=int(config.corpus.vocab_size))
    train_dataset = NextTokenDataset(text_splits["train"], vocab=vocab, max_length=int(config.corpus.max_length))
    dev_dataset = NextTokenDataset(text_splits["dev"], vocab=vocab, max_length=int(config.corpus.max_length))
    test_dataset = NextTokenDataset(text_splits["test"], vocab=vocab, max_length=int(config.corpus.max_length))

    train_loader = _make_loader(train_dataset, batch_size=int(config.training.batch_size), shuffle=True, seed=int(config.training.seed))
    dev_loader = _make_loader(dev_dataset, batch_size=int(config.training.batch_size), shuffle=False, seed=int(config.training.seed) + 1)
    test_loader = _make_loader(test_dataset, batch_size=int(config.training.batch_size), shuffle=False, seed=int(config.training.seed) + 2)

    model = CollaborativeCloneLM(vocab_size=vocab.size, max_length=int(config.corpus.max_length), config=config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.training.lr),
        weight_decay=float(config.training.weight_decay),
    )

    history: List[Dict[str, object]] = []
    for epoch in range(int(config.training.epochs)):
        model.train()
        loss_sum = 0.0
        token_sum = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_clone_states=False, return_attention=False)
            loss = _joint_loss(outputs["joint_logits"], target_ids)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            valid_tokens = int(target_ids.ne(-100).sum().item())
            loss_sum += float(loss.detach().cpu().item()) * valid_tokens
            token_sum += valid_tokens

        dev_metrics = evaluate_model(model, dev_loader, pad_id=vocab.pad_id, device=device)
        history.append(
            {
                "epoch": int(epoch + 1),
                "train_perplexity": float(math.exp(loss_sum / max(1, token_sum))),
                "dev_joint_perplexity": float(dev_metrics["joint_perplexity"]),
                "dev_clone_perplexity": [float(value) for value in dev_metrics["clone_perplexity"]],
                "dev_clone_perplexity_spread": float(dev_metrics["clone_perplexity_spread"]),
                "dev_mean_off_diagonal_clone_cosine": float(dev_metrics["mean_off_diagonal_clone_cosine"]),
                "dev_attention_entropy_per_clone": [float(value) for value in dev_metrics["attention_entropy_per_clone"]],
                "joint_weights": [float(value) for value in dev_metrics["joint_weights"]],
            }
        )

    final_dev = evaluate_model(model, dev_loader, pad_id=vocab.pad_id, device=device)
    final_test = evaluate_model(model, test_loader, pad_id=vocab.pad_id, device=device)
    return {
        "config": asdict(config),
        "vocab_size": int(vocab.size),
        "sequence_length": int(config.corpus.max_length),
        "dataset_sizes": {
            "train_sequences": len(train_dataset),
            "dev_sequences": len(dev_dataset),
            "test_sequences": len(test_dataset),
        },
        "attention_bias_norms": _attention_bias_norms(model),
        "history": history,
        "final_dev": final_dev,
        "final_test": final_test,
    }


def _history_delta(history: Sequence[Dict[str, object]], key: str) -> float | None:
    if len(history) < 2:
        return None
    start = history[0].get(key)
    end = history[-1].get(key)
    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        return float(end) - float(start)
    return None


def write_report(results: Dict[str, object], report_path: Path) -> None:
    history = list(results.get("history", []))
    final_test = dict(results.get("final_test", {}))
    bias_norms = list(results.get("attention_bias_norms", []))
    lines = [
        "# Stage 13 Joint Clone Next-Token Collaboration",
        "",
        "## Setup",
        f"- Clones: `{results['config']['model']['num_clones']}`",
        f"- Hidden size / layers: `{results['config']['model']['hidden_size']}` / `{results['config']['model']['num_layers']}`",
        f"- Sequence length: `{results['sequence_length']}`",
        f"- Vocab size: `{results['vocab_size']}`",
        f"- Dataset sequences: `{results['dataset_sizes']}`",
        "",
        "## Final Held-Out Metrics",
        f"- Joint perplexity: `{final_test.get('joint_perplexity'):.4f}`",
        f"- Per-clone perplexity: `{[round(float(value), 4) for value in final_test.get('clone_perplexity', [])]}`",
        f"- Joint beats best single clone: `{bool(final_test.get('joint_beats_best_clone', False))}`",
        f"- Mean off-diagonal clone cosine: `{float(final_test.get('mean_off_diagonal_clone_cosine', 1.0)):.4f}`",
        f"- Attention entropy per clone: `{[round(float(value), 4) for value in final_test.get('attention_entropy_per_clone', [])]}`",
        f"- Pairwise attention KL overall mean: `{float(final_test.get('pairwise_attention_kl', {}).get('overall_mean', 0.0)):.6f}`",
        f"- Bias-ablation attention shift KL: `{float(final_test.get('bias_used_diagnostic', {}).get('mean_attention_shift_kl_without_bias', {}).get('overall_mean', 0.0)):.6f}`",
        f"- Joint perplexity without learned bias: `{float(final_test.get('bias_used_diagnostic', {}).get('joint_perplexity_without_bias', 0.0)):.4f}`",
        f"- Joint weights: `{[round(float(value), 4) for value in final_test.get('joint_weights', [])]}`",
        "",
        "## Bias Norms",
    ]
    for row in bias_norms:
        lines.append(
            "- Layer {layer}: l2={l2:.4f}, mean_abs={mean_abs:.6f}, per_clone_l2={per_clone}".format(
                layer=int(row["layer"]),
                l2=float(row["layer_l2_norm"]),
                mean_abs=float(row["layer_mean_abs"]),
                per_clone=[round(float(value), 4) for value in row["per_clone_l2_norm"]],
            )
        )
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
                f"- Initial mean off-diagonal clone cosine: `{float(history[0]['dev_mean_off_diagonal_clone_cosine']):.4f}`",
                f"- Final mean off-diagonal clone cosine: `{float(history[-1]['dev_mean_off_diagonal_clone_cosine']):.4f}`",
            ]
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(config: Stage13Config | None = None) -> Dict[str, object]:
    active = config or Stage13Config()
    text_splits, summary = _build_corpus_splits(active.corpus, seed=int(active.training.seed))
    results = train_on_text_splits(text_splits, active)
    results["corpus_summary"] = summary
    write_report(results, REPORT_PATH)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 13 joint-clone next-token collaboration.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-train", type=int, default=512)
    parser.add_argument("--n-dev", type=int, default=128)
    parser.add_argument("--n-test", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    config = Stage13Config(
        corpus=CorpusConfig(
            n_train=int(args.n_train),
            n_dev=int(args.n_dev),
            n_test=int(args.n_test),
            max_length=int(args.max_length),
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
        "stage13: wrote {results} and {report}; joint_ppl={ppl:.4f}".format(
            results=RESULTS_PATH,
            report=REPORT_PATH,
            ppl=float(results["final_test"]["joint_perplexity"]),
        )
    )


if __name__ == "__main__":
    main()
