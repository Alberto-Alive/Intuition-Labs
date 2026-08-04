from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from src.agents.types import AttemptBatch
from src.coordinators.latent_coordination import LatentCoordinatorConfig, make_candidate_query_coordinator, make_latent_coordinator
from src.coordinators.mlp import MLPClassifier, MLPTrainingConfig
from src.coordinators.probes import HiddenStateOnlyLabelProbe
from src.datasets.multiview_code_patch_selection import (
    ATTRIBUTE_NAMES,
    ATTRIBUTE_VALUE_PAIRS,
    CONTROL_NAMES,
    MultiViewCodePatchDatasetConfig,
    MultiViewTaskExample,
    apply_example_control,
    attempt_batch_from_examples,
    build_multiview_code_patch_splits,
    dataset_summary,
    example_oracle_metadata,
    format_clone_prompt,
    format_candidate_block,
    format_full_context_prompt,
    format_partial_context_prompt,
    candidate_attribute_bits as oracle_candidate_attribute_bits,
    output_leakage_audit,
    randomized_labels_for_examples,
    split_leakage_audit,
)
from src.evaluation.audit import audit_event, summarize_test_access
from src.evaluation.metrics import evaluate_predictions
from src.evaluation.report import write_report


BENCHMARK = "real_shared_weight_latent_coordination"
PROPOSED_MESSAGE_METHOD = "trainable_shared_agent_active_msg_head_aux_candidate_query"
PROPOSED_FROZEN_MESSAGE_METHOD = "frozen_shared_agent_active_msg_head_aux_candidate_query"
EVIDENCE_TOKEN_UPPER_BOUND_METHOD = "trainable_shared_agent_evidence_token_diagnostic_upper_bound"
FROZEN_EVIDENCE_TOKEN_UPPER_BOUND_METHOD = "frozen_shared_agent_evidence_token_diagnostic_upper_bound"
ATTRIBUTE_VALUES = (
    ("guard_soft", "guard_strict"),
    ("patch_local", "patch_boundary"),
    ("contract_legacy", "contract_updated"),
    ("policy_default", "policy_explicit"),
)


@dataclass(frozen=True)
class SharedTransformerAgentConfig:
    agent_mode: str = "tiny_transformer"
    model_name_or_path: str = "EleutherAI/pythia-70m-deduped"
    local_files_only: bool = True
    allow_tiny_fallback: bool = True
    max_length: int = 128
    hidden_dim: int = 64
    tiny_vocab_size: int = 4096
    tiny_layers: int = 1
    tiny_heads: int = 2
    tiny_ff_dim: int = 128
    lora_rank: int = 4
    lora_alpha: float = 8.0
    lora_target_modules: Tuple[str, ...] = (
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    )
    adapter_hidden_dim: int = 64
    train_top_layers: int = 0
    gradient_checkpointing: bool = True


@dataclass(frozen=True)
class RealSharedWeightTrainingConfig:
    epochs: int = 4
    batch_size: int = 16
    lr: float = 0.001
    weight_decay: float = 0.0001
    patience: int = 3
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "none"
    gradient_clip_norm: float = 0.0


@dataclass(frozen=True)
class MessageChannelConfig:
    use_msg_token: bool = False
    msg_position: str = "prepend"
    readout_source: str = "msg"
    use_message_head: bool = False
    message_head_type: str = "mlp"
    message_dim: int = 64
    message_head_dropout: float = 0.0
    coordinator_family: str = "latent"
    use_private_cue_aux: bool = False
    aux_loss_weight: float = 0.05
    aux_warmup_epochs: int = 2
    aux_decay_epochs: int = 4
    active_message_layers: Tuple[int, ...] = (-1,)
    active_message_readout_type: str = "active_query"
    active_message_num_queries: int = 1
    active_message_heads: int = 2
    active_message_ff_dim: int = 128
    active_message_dropout: float = 0.0
    active_message_topk: int = 8
    residual_message_readout: bool = False
    message_variance_regularizer_weight: float = 0.0
    message_variance_regularizer_target: float = 0.01
    message_norm_regularizer_weight: float = 0.0
    message_norm_regularizer_target: float = 8.0
    canonicalize_role_order: bool = False
    num_avenues: int = 1
    avenue_prompt_mode: str = "single"
    avenue_dropout: float = 0.0
    canonicalize_avenue_order: bool = False


@dataclass
class FitResult:
    method: str
    condition: str
    agent: "SharedTransformerAgent"
    coordinator: nn.Module
    system: "SharedClonedAgentSystem"
    trainable_agent: bool
    param_count: int
    audit: Dict[str, object]
    history: List[Dict[str, float]]


@dataclass
class ContextFitResult:
    method: str
    agent: "SharedTransformerAgent"
    head: nn.Module
    param_count: int
    history: List[Dict[str, float]]
    prompt_mode: str


class HashTextTokenizer:
    def __init__(self, vocab_size: int, max_length: int) -> None:
        self.vocab_size = int(vocab_size)
        self.max_length = int(max_length)
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.msg_token_id = 2

    def encode_batch(self, texts: Sequence[str], device: torch.device) -> torch.Tensor:
        rows = []
        for text in texts:
            ids = [self._token_id(token) for token in _text_tokens(text)[: self.max_length]]
            if len(ids) < self.max_length:
                ids.extend([self.pad_token_id] * (self.max_length - len(ids)))
            rows.append(ids)
        return torch.as_tensor(rows, dtype=torch.long, device=device)

    def encode_batch_with_message(
        self,
        texts: Sequence[str],
        device: torch.device,
        msg_position: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if msg_position not in {"prepend", "append"}:
            raise ValueError(f"unknown [MSG] position: {msg_position}")
        rows = []
        positions = []
        text_capacity = max(0, self.max_length - 1)
        for text in texts:
            text_ids = [self._token_id(token) for token in _text_tokens(text)[:text_capacity]]
            if msg_position == "append":
                ids = text_ids + [self.msg_token_id]
                msg_index = len(text_ids)
            else:
                ids = [self.msg_token_id] + text_ids
                msg_index = 0
            if len(ids) < self.max_length:
                ids.extend([self.pad_token_id] * (self.max_length - len(ids)))
            rows.append(ids[: self.max_length])
            positions.append(min(msg_index, self.max_length - 1))
        return (
            torch.as_tensor(rows, dtype=torch.long, device=device),
            torch.as_tensor(positions, dtype=torch.long, device=device),
        )

    def _token_id(self, token: str) -> int:
        raw = __import__("hashlib").sha256(token.encode("utf-8")).hexdigest()
        return 3 + (int(raw[:8], 16) % max(1, self.vocab_size - 3))

    def evidence_token_ids(self) -> Tuple[int, int]:
        return (self._token_id("repair_cue_low"), self._token_id("repair_cue_high"))


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.scaling = float(alpha) / max(1, self.rank)
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.lora_a = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_b = nn.Linear(self.rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_in = x.to(dtype=self.lora_a.weight.dtype)
        lora_out = self.lora_b(self.lora_a(lora_in)).to(dtype=base_out.dtype)
        return base_out + lora_out * self.scaling


class SharedTransformerAgent(nn.Module):
    """One shared text encoder reused for every clone path."""

    def __init__(self, config: SharedTransformerAgentConfig) -> None:
        super().__init__()
        self.config = config
        self.mode = config.agent_mode
        self.max_length = int(config.max_length)
        self.tokenizer = None
        self.text_tokenizer: HashTextTokenizer | None = None
        self.pretrained_model: nn.Module | None = None
        self.tiny_embedding: nn.Embedding | None = None
        self.tiny_position: nn.Embedding | None = None
        self.tiny_encoder: nn.TransformerEncoder | None = None
        self.tiny_norm: nn.LayerNorm | None = None
        self.lora_modules_installed = 0
        self.pretrained_hidden_dim = config.hidden_dim
        self.msg_token = "[MSG]"
        self.pretrained_msg_token_id: int | None = None

        if self.mode == "pretrained_adapter":
            self._init_pretrained()
        elif self.mode == "tiny_transformer":
            self._init_tiny()
        else:
            raise ValueError(f"unknown agent mode: {self.mode}")

        self.activation_adapter = nn.Sequential(
            nn.LayerNorm(self.pretrained_hidden_dim),
            nn.Linear(self.pretrained_hidden_dim, config.adapter_hidden_dim),
            nn.GELU(),
            nn.Linear(config.adapter_hidden_dim, config.hidden_dim),
        )
        self._coordination_trainable_names = self._default_coordination_trainable_names()
        self.configure_trainable(True)

    @property
    def output_dim(self) -> int:
        return int(self.config.hidden_dim)

    def forward_texts(self, texts: Sequence[str]) -> torch.Tensor:
        if self.mode == "tiny_transformer":
            return self._forward_tiny(texts)
        return self._forward_pretrained(texts)

    def forward_texts_with_readouts(
        self,
        texts: Sequence[str],
        use_msg_token: bool = False,
        msg_position: str = "prepend",
        selected_layer_ids: Sequence[int] | None = None,
    ) -> Dict[str, torch.Tensor]:
        if self.mode == "tiny_transformer":
            return self._forward_tiny_readouts(
                texts,
                use_msg_token=use_msg_token,
                msg_position=msg_position,
                selected_layer_ids=selected_layer_ids,
            )
        return self._forward_pretrained_readouts(
            texts,
            use_msg_token=use_msg_token,
            msg_position=msg_position,
            selected_layer_ids=selected_layer_ids,
        )

    def configure_trainable(self, trainable: bool) -> None:
        for _name, parameter in self.named_parameters():
            parameter.requires_grad = False
        if not trainable:
            return
        allowed = set(self._coordination_trainable_names)
        for name, parameter in self.named_parameters():
            if name in allowed:
                parameter.requires_grad = True

    def coordination_parameter_items(self) -> List[Tuple[str, nn.Parameter]]:
        allowed = set(self._coordination_trainable_names)
        return [(name, parameter) for name, parameter in self.named_parameters() if name in allowed]

    def trainable_parameter_names(self) -> List[str]:
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

    def _init_tiny(self) -> None:
        self.text_tokenizer = HashTextTokenizer(self.config.tiny_vocab_size, self.max_length)
        self.pretrained_hidden_dim = self.config.hidden_dim
        self.tiny_embedding = nn.Embedding(self.config.tiny_vocab_size, self.config.hidden_dim, padding_idx=0)
        self.tiny_position = nn.Embedding(self.max_length, self.config.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_dim,
            nhead=_compatible_heads(self.config.hidden_dim, self.config.tiny_heads),
            dim_feedforward=self.config.tiny_ff_dim,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.tiny_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.config.tiny_layers)
        self.tiny_norm = nn.LayerNorm(self.config.hidden_dim)

    def _init_pretrained(self) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name_or_path,
                local_files_only=self.config.local_files_only,
            )
            self.pretrained_model = AutoModel.from_pretrained(
                self.config.model_name_or_path,
                local_files_only=self.config.local_files_only,
            )
            if self.config.gradient_checkpointing and hasattr(self.pretrained_model, "gradient_checkpointing_enable"):
                self.pretrained_model.gradient_checkpointing_enable()
            if getattr(self.tokenizer, "pad_token", None) is None:
                self.tokenizer.pad_token = getattr(self.tokenizer, "eos_token", None) or getattr(self.tokenizer, "unk_token", None)
            self._ensure_pretrained_msg_token()
            for parameter in self.pretrained_model.parameters():
                parameter.requires_grad = False
            self.lora_modules_installed = _install_lora_adapters(
                self.pretrained_model,
                target_names=tuple(self.config.lora_target_modules),
                rank=self.config.lora_rank,
                alpha=self.config.lora_alpha,
            )
            self._unfreeze_top_layers(self.config.train_top_layers)
            hidden_size = getattr(self.pretrained_model.config, "hidden_size", None)
            if hidden_size is None:
                hidden_size = getattr(self.pretrained_model.config, "d_model", None)
            self.pretrained_hidden_dim = int(hidden_size)
        except Exception:
            if not self.config.allow_tiny_fallback:
                raise
            self.mode = "tiny_transformer"
            self._init_tiny()

    def _unfreeze_top_layers(self, train_top_layers: int) -> None:
        if self.pretrained_model is None or train_top_layers <= 0:
            return
        layers = _find_transformer_layers(self.pretrained_model)
        for layer in layers[-int(train_top_layers) :]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    def _forward_tiny(self, texts: Sequence[str]) -> torch.Tensor:
        return self._forward_tiny_readouts(texts, use_msg_token=False, msg_position="prepend")["pooled"]

    def _forward_tiny_readouts(
        self,
        texts: Sequence[str],
        use_msg_token: bool,
        msg_position: str,
        selected_layer_ids: Sequence[int] | None = None,
    ) -> Dict[str, torch.Tensor]:
        if self.text_tokenizer is None or self.tiny_embedding is None or self.tiny_position is None or self.tiny_encoder is None or self.tiny_norm is None:
            raise RuntimeError("tiny transformer is not initialized")
        device = next(self.parameters()).device
        if use_msg_token:
            token_ids, msg_positions = self.text_tokenizer.encode_batch_with_message(texts, device, msg_position)
        else:
            token_ids = self.text_tokenizer.encode_batch(texts, device)
            msg_positions = torch.zeros(token_ids.shape[0], dtype=torch.long, device=device)
        positions = torch.arange(token_ids.shape[1], dtype=torch.long, device=device)
        hidden = self.tiny_embedding(token_ids) + self.tiny_position(positions).unsqueeze(0)
        pad_mask = token_ids.eq(self.text_tokenizer.pad_token_id)
        encoded = hidden
        layer_outputs: List[torch.Tensor] = []
        for layer in self.tiny_encoder.layers:
            if self.config.gradient_checkpointing and self.training and encoded.requires_grad:
                encoded = checkpoint(lambda value, layer=layer: layer(value, src_key_padding_mask=pad_mask), encoded, use_reentrant=False)
            else:
                encoded = layer(encoded, src_key_padding_mask=pad_mask)
            layer_outputs.append(self.tiny_norm(encoded))
        if not layer_outputs:
            layer_outputs.append(self.tiny_norm(encoded))
        encoded = layer_outputs[-1]
        lengths = (~pad_mask).sum(dim=1).clamp(min=1) - 1
        pooled = encoded[torch.arange(encoded.shape[0], device=device), lengths]
        msg = encoded[torch.arange(encoded.shape[0], device=device), msg_positions]
        cue_ids = self.text_tokenizer.evidence_token_ids()
        cue_mask = token_ids.eq(cue_ids[0]) | token_ids.eq(cue_ids[1])
        cue_counts = cue_mask.sum(dim=1).clamp(min=1).to(dtype=encoded.dtype)
        evidence = (encoded * cue_mask.unsqueeze(-1).to(dtype=encoded.dtype)).sum(dim=1) / cue_counts.unsqueeze(-1)
        evidence = torch.where(cue_mask.any(dim=1).view(-1, 1), evidence, msg)
        adapter_dtype = next(self.activation_adapter.parameters()).dtype
        selected_layers = _select_layer_tensors(layer_outputs, selected_layer_ids)
        token_states = torch.cat([self.activation_adapter(layer.to(dtype=adapter_dtype)) for layer in selected_layers], dim=1)
        token_mask = (~pad_mask).repeat(1, len(selected_layers))
        return {
            "pooled": self.activation_adapter(pooled.to(dtype=adapter_dtype)),
            "msg": self.activation_adapter(msg.to(dtype=adapter_dtype)),
            "evidence": self.activation_adapter(evidence.to(dtype=adapter_dtype)),
            "token_states": token_states,
            "token_mask": token_mask,
        }

    def _forward_pretrained(self, texts: Sequence[str]) -> torch.Tensor:
        return self._forward_pretrained_readouts(texts, use_msg_token=False, msg_position="prepend")["pooled"]

    def _forward_pretrained_readouts(
        self,
        texts: Sequence[str],
        use_msg_token: bool,
        msg_position: str,
        selected_layer_ids: Sequence[int] | None = None,
    ) -> Dict[str, torch.Tensor]:
        if self.tokenizer is None or self.pretrained_model is None:
            raise RuntimeError("pretrained model is not initialized")
        device = next(self.parameters()).device
        if use_msg_token:
            if msg_position == "append":
                model_texts = [f"{text}\n{self.msg_token}" for text in texts]
            elif msg_position == "prepend":
                model_texts = [f"{self.msg_token}\n{text}" for text in texts]
            else:
                raise ValueError(f"unknown [MSG] position: {msg_position}")
        else:
            model_texts = list(texts)
        encoded = self.tokenizer(
            model_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = self.pretrained_model(**encoded, output_hidden_states=True, return_dict=True)
        last_hidden = getattr(outputs, "last_hidden_state", None)
        if last_hidden is None:
            last_hidden = outputs.hidden_states[-1]
        mask = encoded.get("attention_mask")
        if mask is None:
            pooled = last_hidden[:, -1, :]
        else:
            lengths = mask.sum(dim=1).clamp(min=1) - 1
            pooled = last_hidden[torch.arange(last_hidden.shape[0], device=device), lengths]
        msg_positions = torch.zeros(last_hidden.shape[0], dtype=torch.long, device=device)
        if use_msg_token and self.pretrained_msg_token_id is not None and "input_ids" in encoded:
            msg_mask = encoded["input_ids"].eq(int(self.pretrained_msg_token_id))
            has_msg = msg_mask.any(dim=1)
            first_msg = msg_mask.float().argmax(dim=1).to(dtype=torch.long)
            fallback = lengths if mask is not None else torch.full_like(first_msg, last_hidden.shape[1] - 1)
            msg_positions = torch.where(has_msg, first_msg, fallback)
        elif mask is not None:
            msg_positions = lengths
        msg = last_hidden[torch.arange(last_hidden.shape[0], device=device), msg_positions]
        cue_mask = torch.zeros(last_hidden.shape[:2], dtype=torch.bool, device=device)
        if "input_ids" in encoded:
            cue_ids = self._pretrained_evidence_token_ids()
            if cue_ids:
                for cue_id in cue_ids:
                    cue_mask = cue_mask | encoded["input_ids"].eq(int(cue_id))
        cue_counts = cue_mask.sum(dim=1).clamp(min=1).to(dtype=last_hidden.dtype)
        evidence = (last_hidden * cue_mask.unsqueeze(-1).to(dtype=last_hidden.dtype)).sum(dim=1) / cue_counts.unsqueeze(-1)
        evidence = torch.where(cue_mask.any(dim=1).view(-1, 1), evidence, msg)
        adapter_dtype = next(self.activation_adapter.parameters()).dtype
        hidden_layers = list(getattr(outputs, "hidden_states", None) or [last_hidden])
        selected_layers = _select_layer_tensors(hidden_layers, selected_layer_ids)
        token_states = torch.cat([self.activation_adapter(layer.to(dtype=adapter_dtype)) for layer in selected_layers], dim=1)
        if mask is None:
            token_mask = torch.ones(last_hidden.shape[:2], dtype=torch.bool, device=device)
        else:
            token_mask = mask.bool()
        token_mask = token_mask.repeat(1, len(selected_layers))
        return {
            "pooled": self.activation_adapter(pooled.to(dtype=adapter_dtype)),
            "msg": self.activation_adapter(msg.to(dtype=adapter_dtype)),
            "evidence": self.activation_adapter(evidence.to(dtype=adapter_dtype)),
            "token_states": token_states,
            "token_mask": token_mask,
        }

    def _ensure_pretrained_msg_token(self) -> None:
        if self.tokenizer is None or self.pretrained_model is None:
            return
        added = 0
        try:
            vocab = self.tokenizer.get_vocab()
            if self.msg_token not in vocab:
                added = int(self.tokenizer.add_special_tokens({"additional_special_tokens": [self.msg_token]}))
            self.pretrained_msg_token_id = int(self.tokenizer.convert_tokens_to_ids(self.msg_token))
            if added and hasattr(self.pretrained_model, "resize_token_embeddings"):
                self.pretrained_model.resize_token_embeddings(len(self.tokenizer))
        except Exception:
            self.pretrained_msg_token_id = None

    def _pretrained_evidence_token_ids(self) -> List[int]:
        if self.tokenizer is None:
            return []
        ids: List[int] = []
        try:
            for text in ("REPAIR_CUE_LOW", "REPAIR_CUE_HIGH"):
                encoded = self.tokenizer(text, add_special_tokens=False)
                value = encoded.get("input_ids", [])
                if isinstance(value, list):
                    ids.extend(int(item) for item in value)
        except Exception:
            return []
        return sorted(set(ids))

    def _default_coordination_trainable_names(self) -> List[str]:
        if self.mode == "tiny_transformer":
            return [name for name, _parameter in self.named_parameters()]
        names = []
        for name, _parameter in self.named_parameters():
            if ".lora_a." in name or ".lora_b." in name or name.startswith("activation_adapter."):
                names.append(name)
        if not names:
            names = [name for name, _parameter in self.named_parameters() if name.startswith("activation_adapter.")]
        return names


class SharedMessageHead(nn.Module):
    def __init__(self, input_dim: int, message_dim: int, head_type: str = "mlp", dropout: float = 0.0) -> None:
        super().__init__()
        self.head_type = str(head_type)
        if self.head_type == "linear":
            self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, message_dim))
        elif self.head_type == "gated_mlp":
            self.norm = nn.LayerNorm(input_dim)
            self.value = nn.Sequential(nn.Linear(input_dim, message_dim), nn.GELU(), nn.Dropout(dropout))
            self.gate = nn.Sequential(nn.Linear(input_dim, message_dim), nn.Sigmoid())
            self.out = nn.Linear(message_dim, message_dim)
        elif self.head_type == "residual_mlp":
            self.input_projection = nn.Linear(input_dim, message_dim) if input_dim != message_dim else nn.Identity()
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, message_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(message_dim, message_dim),
            )
            self.out_norm = nn.LayerNorm(message_dim)
        elif self.head_type in {"mlp", "layernorm_mlp"}:
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, message_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(message_dim, message_dim),
            )
        else:
            raise ValueError(f"unknown message head type: {head_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.head_type == "gated_mlp":
            y = self.norm(x)
            return self.out(self.value(y) * self.gate(y))
        if self.head_type == "residual_mlp":
            return self.out_norm(self.input_projection(x) + self.net(x))
        return self.net(x)


class ActiveMessageReadout(nn.Module):
    """A learned per-role message query cross-attends over clone token states."""

    def __init__(
        self,
        input_dim: int,
        n_roles: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        readout_type: str = "active_query",
        num_queries: int = 1,
        topk: int = 8,
    ) -> None:
        super().__init__()
        heads = _compatible_heads(input_dim, num_heads)
        self.readout_type = str(readout_type)
        self.num_queries = max(1, int(num_queries))
        self.topk = max(1, int(topk))
        self.message_query = nn.Parameter(torch.zeros(1, self.num_queries, input_dim))
        self.role_embedding = nn.Embedding(n_roles, input_dim)
        self.query_norm = nn.LayerNorm(input_dim)
        self.token_norm = nn.LayerNorm(input_dim)
        self.ff_norm = nn.LayerNorm(input_dim)
        self.out_norm = nn.LayerNorm(input_dim)
        self.pool_scorer = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 1))
        self.attention = nn.MultiheadAttention(input_dim, heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, input_dim),
        )
        nn.init.normal_(self.message_query, mean=0.0, std=1.0 / max(1, input_dim) ** 0.5)

    def forward(
        self,
        token_states: torch.Tensor,
        token_mask: torch.Tensor,
        role_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch, roles, tokens, dim = token_states.shape
        flat_tokens = token_states.reshape(batch * roles, tokens, dim)
        flat_mask = token_mask.reshape(batch * roles, tokens).bool()
        flat_roles = role_ids.reshape(batch * roles).clamp(min=0, max=self.role_embedding.num_embeddings - 1)
        role = self.role_embedding(flat_roles)
        if self.readout_type in {"attention_pool", "topk_attention"}:
            scores = self.pool_scorer(flat_tokens + role.unsqueeze(1)).squeeze(-1)
            scores = scores.masked_fill(~flat_mask, -1e9)
            if self.readout_type == "topk_attention":
                k = min(self.topk, scores.shape[-1])
                top_values, _top_indices = torch.topk(scores, k=k, dim=-1)
                threshold = top_values[:, -1].unsqueeze(-1)
                scores = scores.masked_fill(scores < threshold, -1e9)
            weights = torch.softmax(scores, dim=-1)
            pooled = torch.sum(flat_tokens * weights.unsqueeze(-1), dim=1)
            return self.out_norm(pooled).reshape(batch, roles, dim)
        if self.readout_type == "multi_query_topk":
            query = self.message_query.expand(batch * roles, -1, -1) + role.unsqueeze(1)
            query = self.query_norm(query)
            keys = self.token_norm(flat_tokens)
            scores = torch.einsum("bqd,btd->bqt", query, keys) / max(1.0, float(dim) ** 0.5)
            scores = scores.masked_fill(~flat_mask.unsqueeze(1), -1e9)
            k = min(self.topk, scores.shape[-1])
            top_values, _top_indices = torch.topk(scores, k=k, dim=-1)
            threshold = top_values[:, :, -1].unsqueeze(-1)
            scores = scores.masked_fill(scores < threshold, -1e9)
            weights = torch.softmax(scores, dim=-1)
            attended = torch.einsum("bqt,btd->bqd", weights, flat_tokens)
            message = query + attended
            message = message + self.feed_forward(self.ff_norm(message))
            message = message.mean(dim=1)
            return self.out_norm(message).reshape(batch, roles, dim)
        if self.readout_type not in {"active_query", "multi_query"}:
            raise ValueError(f"unknown active message readout type: {self.readout_type}")
        query = self.message_query.expand(batch * roles, -1, -1)
        query = query + role.unsqueeze(1)
        attended, _weights = self.attention(
            query=self.query_norm(query),
            key=self.token_norm(flat_tokens),
            value=flat_tokens,
            key_padding_mask=~flat_mask,
            need_weights=False,
        )
        message = query + attended
        message = message + self.feed_forward(self.ff_norm(message))
        message = message.mean(dim=1)
        return self.out_norm(message).reshape(batch, roles, dim)


class SharedClonedAgentSystem(nn.Module):
    def __init__(
        self,
        shared_agent: SharedTransformerAgent,
        coordinator: nn.Module,
        n_roles: int,
        visible_explicit_evidence: bool = False,
        message_config: MessageChannelConfig | None = None,
    ) -> None:
        super().__init__()
        self.shared_agent = shared_agent
        self.coordinator = coordinator
        self.n_roles = int(n_roles)
        self.visible_explicit_evidence = bool(visible_explicit_evidence)
        self.message_config = message_config or MessageChannelConfig()
        self.active_message_readout: nn.Module | None = None
        self.message_head: nn.Module | None = None
        self.private_cue_head: nn.Module | None = None
        message_input_dim = shared_agent.output_dim
        if self.message_config.readout_source == "active_message":
            self.active_message_readout = ActiveMessageReadout(
                input_dim=message_input_dim,
                n_roles=self.n_roles,
                num_heads=int(self.message_config.active_message_heads),
                ff_dim=int(self.message_config.active_message_ff_dim),
                dropout=float(self.message_config.active_message_dropout),
                readout_type=self.message_config.active_message_readout_type,
                num_queries=int(self.message_config.active_message_num_queries),
                topk=int(self.message_config.active_message_topk),
            )
        if self.message_config.use_message_head:
            self.message_head = SharedMessageHead(
                message_input_dim,
                int(self.message_config.message_dim),
                head_type=self.message_config.message_head_type,
                dropout=float(self.message_config.message_head_dropout),
            )
            message_input_dim = int(self.message_config.message_dim)
        if self.message_config.use_private_cue_aux:
            self.private_cue_head = nn.Sequential(nn.LayerNorm(message_input_dim), nn.Linear(message_input_dim, 2))
        self.last_aux_logits: torch.Tensor | None = None

    @property
    def coordinator_input_dim(self) -> int:
        if self.message_config.use_message_head:
            return int(self.message_config.message_dim)
        return int(self.shared_agent.output_dim)

    def collect_clone_representations(
        self,
        examples: Sequence[MultiViewTaskExample],
        condition: str = "none",
        seed: int = 0,
        role_ids: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        avenue_count = max(1, int(self.message_config.num_avenues))
        role_pooled = []
        role_msg = []
        role_evidence = []
        role_tokens = []
        role_token_masks = []
        for role_index in range(self.n_roles):
            avenue_pooled = []
            avenue_msg = []
            avenue_evidence = []
            avenue_tokens = []
            avenue_token_masks = []
            for avenue_index in range(avenue_count):
                if self.visible_explicit_evidence:
                    texts = [_format_explicit_evidence_clone_prompt(example, role_index) for example in examples]
                else:
                    texts = [_format_avenue_clone_prompt(example, role_index, avenue_index, self.message_config) for example in examples]
                readouts = self.shared_agent.forward_texts_with_readouts(
                    texts,
                    use_msg_token=self.message_config.use_msg_token,
                    msg_position=self.message_config.msg_position,
                    selected_layer_ids=self.message_config.active_message_layers,
                )
                avenue_pooled.append(readouts["pooled"])
                avenue_msg.append(readouts["msg"])
                avenue_evidence.append(readouts.get("evidence", readouts["msg"]))
                avenue_tokens.append(readouts["token_states"])
                avenue_token_masks.append(readouts["token_mask"])
            role_pooled.append(torch.stack(avenue_pooled, dim=1))
            role_msg.append(torch.stack(avenue_msg, dim=1))
            role_evidence.append(torch.stack(avenue_evidence, dim=1))
            role_tokens.append(torch.stack(avenue_tokens, dim=1))
            role_token_masks.append(torch.stack(avenue_token_masks, dim=1))
        raw_pooled_full = torch.stack(role_pooled, dim=1)
        msg_full = torch.stack(role_msg, dim=1)
        evidence_full = torch.stack(role_evidence, dim=1)
        token_states_full = torch.stack(role_tokens, dim=1)
        token_mask_full = torch.stack(role_token_masks, dim=1)
        raw_pooled = raw_pooled_full.mean(dim=2)
        msg = msg_full.mean(dim=2)
        evidence = evidence_full.mean(dim=2)
        token_states = token_states_full[:, :, 0, :, :] if avenue_count == 1 else token_states_full
        token_mask = token_mask_full[:, :, 0, :] if avenue_count == 1 else token_mask_full
        if role_ids is None:
            role_ids = torch.arange(self.n_roles, dtype=torch.long, device=raw_pooled.device).view(1, self.n_roles)
            role_ids = role_ids.expand(raw_pooled.shape[0], self.n_roles)
        else:
            role_ids = role_ids.to(device=raw_pooled.device, dtype=torch.long)

        if condition == "hidden_states_shuffled_across_examples" and raw_pooled.shape[0] > 1:
            rng = np.random.default_rng(seed + 53_977)
            perm = _non_identity_permutation(raw_pooled.shape[0], rng)
            index = torch.as_tensor(perm, dtype=torch.long, device=raw_pooled.device)
            raw_pooled = raw_pooled.index_select(0, index)
            msg = msg.index_select(0, index)
            evidence = evidence.index_select(0, index)
            token_states = token_states.index_select(0, index)
            token_mask = token_mask.index_select(0, index)

        active_message = None
        if self.message_config.readout_source == "active_message":
            if int(self.message_config.num_avenues) > 1:
                raise ValueError("active_message readout is not implemented for multi-avenue inputs")
            if self.active_message_readout is None:
                raise RuntimeError("active_message readout requested but no active readout module was initialized")
            active_message = self.active_message_readout(token_states, token_mask, role_ids)
            coordinator_input = active_message + raw_pooled if self.message_config.residual_message_readout else active_message
        elif self.message_config.readout_source == "evidence_tokens":
            coordinator_input = evidence
        elif self.message_config.readout_source == "pooled":
            coordinator_input = raw_pooled
        elif self.message_config.readout_source == "msg":
            coordinator_input = msg if self.message_config.use_msg_token else raw_pooled
        else:
            raise ValueError(f"unknown message readout source: {self.message_config.readout_source}")
        if self.message_head is not None:
            coordinator_input = self.message_head(coordinator_input)
        return {
            "raw_pooled": raw_pooled,
            "msg": msg,
            "evidence": evidence,
            "active_message": active_message if active_message is not None else raw_pooled,
            "message": coordinator_input,
            "token_states": token_states,
            "token_mask": token_mask,
            "raw_pooled_avenues": raw_pooled_full,
            "msg_avenues": msg_full,
            "evidence_avenues": evidence_full,
            "avenue_ids": _avenue_ids(raw_pooled_full.shape[0], self.n_roles, avenue_count, raw_pooled.device),
        }

    def forward(
        self,
        examples: Sequence[MultiViewTaskExample],
        condition: str = "none",
        seed: int = 0,
        retain_activation_grad: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, object], torch.Tensor]:
        if not examples:
            raise ValueError("at least one example is required")
        device = next(self.parameters()).device
        base_role_ids = torch.arange(self.n_roles, dtype=torch.long, device=device).view(1, self.n_roles)
        base_role_ids = base_role_ids.expand(len(examples), self.n_roles)
        role_ids = base_role_ids
        rng = np.random.default_rng(seed + 53_977)
        if condition == "role_labels_shuffled":
            order = _role_permutation(len(examples), self.n_roles, rng, device)
            role_ids = base_role_ids.gather(1, order)

        readouts = self.collect_clone_representations(examples, condition=condition, seed=seed, role_ids=role_ids)
        clone_activations = readouts["message"]
        self.last_aux_logits = None
        if self.private_cue_head is not None:
            self.last_aux_logits = self.private_cue_head(clone_activations)
        if retain_activation_grad and clone_activations.requires_grad:
            clone_activations.retain_grad()

        if condition == "physical_order_shuffled_roles_preserved":
            order = _role_permutation(clone_activations.shape[0], self.n_roles, rng, clone_activations.device)
            clone_activations = _gather_role_axis(clone_activations, order)
            role_ids = role_ids.gather(1, order)
            readouts["raw_pooled"] = _gather_role_axis(readouts["raw_pooled"], order)
            readouts["msg"] = _gather_role_axis(readouts["msg"], order)
            readouts["evidence"] = _gather_role_axis(readouts["evidence"], order)
            readouts["active_message"] = _gather_role_axis(readouts["active_message"], order)
            readouts["message"] = clone_activations
            readouts["token_states"] = _gather_role_axis(readouts["token_states"], order)
            readouts["token_mask"] = _gather_role_axis(readouts["token_mask"], order)
            readouts["raw_pooled_avenues"] = _gather_role_axis(readouts["raw_pooled_avenues"], order)
            readouts["msg_avenues"] = _gather_role_axis(readouts["msg_avenues"], order)
            readouts["evidence_avenues"] = _gather_role_axis(readouts["evidence_avenues"], order)
            readouts["avenue_ids"] = _gather_role_axis(readouts["avenue_ids"], order)

            if readouts["token_states"].dim() == 5 and readouts["token_states"].shape[2] > 1:
                avenue_order = _avenue_permutation(
                    clone_activations.shape[0],
                    self.n_roles,
                    readouts["token_states"].shape[2],
                    rng,
                    clone_activations.device,
                )
                readouts["token_states"] = _gather_avenue_axis(readouts["token_states"], avenue_order)
                readouts["token_mask"] = _gather_avenue_axis(readouts["token_mask"], avenue_order)
                readouts["raw_pooled_avenues"] = _gather_avenue_axis(readouts["raw_pooled_avenues"], avenue_order)
                readouts["msg_avenues"] = _gather_avenue_axis(readouts["msg_avenues"], avenue_order)
                readouts["evidence_avenues"] = _gather_avenue_axis(readouts["evidence_avenues"], avenue_order)
                readouts["avenue_ids"] = _gather_avenue_axis(readouts["avenue_ids"], avenue_order)

        if self.message_config.canonicalize_role_order:
            order = torch.argsort(role_ids, dim=1)
            clone_activations = _gather_role_axis(clone_activations, order)
            role_ids = role_ids.gather(1, order)
            readouts["raw_pooled"] = _gather_role_axis(readouts["raw_pooled"], order)
            readouts["msg"] = _gather_role_axis(readouts["msg"], order)
            readouts["evidence"] = _gather_role_axis(readouts["evidence"], order)
            readouts["active_message"] = _gather_role_axis(readouts["active_message"], order)
            readouts["message"] = clone_activations
            readouts["token_states"] = _gather_role_axis(readouts["token_states"], order)
            readouts["token_mask"] = _gather_role_axis(readouts["token_mask"], order)
            readouts["raw_pooled_avenues"] = _gather_role_axis(readouts["raw_pooled_avenues"], order)
            readouts["msg_avenues"] = _gather_role_axis(readouts["msg_avenues"], order)
            readouts["evidence_avenues"] = _gather_role_axis(readouts["evidence_avenues"], order)
            readouts["avenue_ids"] = _gather_role_axis(readouts["avenue_ids"], order)

        if self.message_config.canonicalize_avenue_order and readouts["token_states"].dim() == 5:
            avenue_order = torch.argsort(readouts["avenue_ids"], dim=2)
            readouts["token_states"] = _gather_avenue_axis(readouts["token_states"], avenue_order)
            readouts["token_mask"] = _gather_avenue_axis(readouts["token_mask"], avenue_order)
            readouts["raw_pooled_avenues"] = _gather_avenue_axis(readouts["raw_pooled_avenues"], avenue_order)
            readouts["msg_avenues"] = _gather_avenue_axis(readouts["msg_avenues"], avenue_order)
            readouts["evidence_avenues"] = _gather_avenue_axis(readouts["evidence_avenues"], avenue_order)
            readouts["avenue_ids"] = _gather_avenue_axis(readouts["avenue_ids"], avenue_order)

        if self.training and float(self.message_config.avenue_dropout) > 0.0 and readouts["token_mask"].dim() == 4:
            readouts["token_mask"] = _apply_avenue_dropout(readouts["token_mask"], float(self.message_config.avenue_dropout))

        if readouts["token_mask"].dim() == 4 and str(condition).startswith("avenue_only_"):
            avenue_index = int(str(condition).rsplit("_", 1)[-1])
            if 0 <= avenue_index < readouts["token_mask"].shape[2]:
                avenue_mask = torch.zeros_like(readouts["token_mask"], dtype=torch.bool)
                avenue_mask[:, :, avenue_index, :] = readouts["token_mask"][:, :, avenue_index, :].bool()
                readouts["token_mask"] = avenue_mask

        if readouts["token_mask"].dim() == 4 and str(condition).startswith("avenue_masked_"):
            avenue_index = int(str(condition).rsplit("_", 1)[-1])
            if 0 <= avenue_index < readouts["token_mask"].shape[2]:
                avenue_mask = readouts["token_mask"].bool().clone()
                avenue_mask[:, :, avenue_index, :] = False
                if not avenue_mask.any(dim=(2, 3)).all():
                    avenue_mask[:, :, avenue_index, :] = readouts["token_mask"][:, :, avenue_index, :].bool()
                readouts["token_mask"] = avenue_mask

        if bool(getattr(self.coordinator, "requires_candidate_features", False)):
            candidate_features = _candidate_feature_tensor(examples, clone_activations.device)
            if bool(getattr(self.coordinator, "requires_token_states", False)):
                logits = self.coordinator(
                    clone_activations,
                    role_ids,
                    candidate_features,
                    readouts["token_states"],
                    readouts["token_mask"],
                    readouts.get("avenue_ids"),
                )
            else:
                logits = self.coordinator(clone_activations, role_ids, candidate_features)
        else:
            logits = self.coordinator(clone_activations, role_ids)
        audit = {
            "activation_requires_grad_before_coordinator": bool(clone_activations.requires_grad),
            "no_detach_between_clone_activations_and_loss": bool(clone_activations.grad_fn is not None or clone_activations.requires_grad),
            "shared_parameter_identity": shared_parameter_identity_check(self.shared_agent, self.n_roles)["pass"],
            "clone_activation_shape": list(clone_activations.shape),
            "message_config": asdict(self.message_config),
            "uses_msg_token": self.message_config.use_msg_token,
            "readout_source": self.message_config.readout_source,
            "uses_active_message_readout": self.message_config.readout_source == "active_message",
            "active_message_layers": [int(value) for value in _as_int_tuple(self.message_config.active_message_layers)],
            "uses_message_head": self.message_config.use_message_head,
            "message_head_type": self.message_config.message_head_type,
            "uses_private_cue_aux": self.message_config.use_private_cue_aux,
            "canonicalize_role_order": self.message_config.canonicalize_role_order,
            "num_avenues": self.message_config.num_avenues,
            "avenue_prompt_mode": self.message_config.avenue_prompt_mode,
            "avenue_dropout": self.message_config.avenue_dropout,
            "canonicalize_avenue_order": self.message_config.canonicalize_avenue_order,
            "coordinator_requires_candidate_features": bool(getattr(self.coordinator, "requires_candidate_features", False)),
            "coordinator_requires_token_states": bool(getattr(self.coordinator, "requires_token_states", False)),
        }
        return logits, audit, clone_activations


class TextOutputOnlyCoordinator:
    def __init__(self, num_classes: int, training: MLPTrainingConfig, seed: int, device: str, feature_dim: int = 256) -> None:
        self.num_classes = int(num_classes)
        self.training = training
        self.seed = int(seed)
        self.device = torch.device(device)
        self.feature_dim = int(feature_dim)
        self.model: MLPClassifier | None = None
        self.param_count = 0
        self.history: List[Dict[str, float]] = []

    def fit(self, train_examples: Sequence[MultiViewTaskExample], dev_examples: Sequence[MultiViewTaskExample]) -> None:
        torch.manual_seed(self.seed + 81_771)
        x_train = _text_output_features(train_examples, self.feature_dim)
        y_train = np.asarray([example.label for example in train_examples], dtype=np.int64)
        x_dev = _text_output_features(dev_examples, self.feature_dim)
        y_dev = np.asarray([example.label for example in dev_examples], dtype=np.int64)
        self.model = MLPClassifier(x_train.shape[1], self.training.hidden_dims, self.num_classes).to(self.device)
        self.param_count = int(sum(parameter.numel() for parameter in self.model.parameters()))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.training.lr, weight_decay=self.training.weight_decay)
        tx = torch.as_tensor(x_train, dtype=torch.float32, device=self.device)
        ty = torch.as_tensor(y_train, dtype=torch.long, device=self.device)
        dx = torch.as_tensor(x_dev, dtype=torch.float32, device=self.device)
        dy = torch.as_tensor(y_dev, dtype=torch.long, device=self.device)
        best_state = None
        best_dev = -1.0
        stale = 0
        rng = np.random.default_rng(self.seed + 13_513)
        for epoch in range(self.training.epochs):
            self.model.train()
            for batch_idx in _batches(rng.permutation(len(y_train)), self.training.batch_size):
                idx = torch.as_tensor(batch_idx, dtype=torch.long, device=self.device)
                loss = F.cross_entropy(self.model(tx.index_select(0, idx)), ty.index_select(0, idx))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            train_acc = _tensor_accuracy(self.model, tx, ty)
            dev_acc = _tensor_accuracy(self.model, dx, dy)
            self.history.append({"epoch": float(epoch + 1), "train_acc": train_acc, "dev_acc": dev_acc})
            if dev_acc > best_dev:
                best_dev = dev_acc
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= self.training.patience:
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

    def predict(self, examples: Sequence[MultiViewTaskExample]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("text-only coordinator has not been fit")
        x = torch.as_tensor(_text_output_features(examples, self.feature_dim), dtype=torch.float32, device=self.device)
        preds = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, x.shape[0], 2048):
                logits = self.model(x[start : start + 2048])
                preds.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
        return np.concatenate(preds).astype(np.int64)


class BagOfWordsDiagnosticBaseline:
    def __init__(
        self,
        method: str,
        num_classes: int,
        training: MLPTrainingConfig,
        seed: int,
        device: str,
        text_builder,
        feature_dim: int = 256,
    ) -> None:
        self.method = method
        self.num_classes = int(num_classes)
        self.training = training
        self.seed = int(seed)
        self.device = torch.device(device)
        self.text_builder = text_builder
        self.feature_dim = int(feature_dim)
        self.model: MLPClassifier | None = None
        self.param_count = 0
        self.history: List[Dict[str, float]] = []

    def fit(self, train_examples: Sequence[MultiViewTaskExample], dev_examples: Sequence[MultiViewTaskExample]) -> None:
        torch.manual_seed(self.seed + 91_001)
        x_train = _hashed_text_features([self.text_builder(example) for example in train_examples], self.feature_dim)
        y_train = np.asarray([example.label for example in train_examples], dtype=np.int64)
        x_dev = _hashed_text_features([self.text_builder(example) for example in dev_examples], self.feature_dim)
        y_dev = np.asarray([example.label for example in dev_examples], dtype=np.int64)
        self.model = MLPClassifier(x_train.shape[1], self.training.hidden_dims, self.num_classes).to(self.device)
        self.param_count = int(sum(parameter.numel() for parameter in self.model.parameters()))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.training.lr, weight_decay=self.training.weight_decay)
        tx = torch.as_tensor(x_train, dtype=torch.float32, device=self.device)
        ty = torch.as_tensor(y_train, dtype=torch.long, device=self.device)
        dx = torch.as_tensor(x_dev, dtype=torch.float32, device=self.device)
        dy = torch.as_tensor(y_dev, dtype=torch.long, device=self.device)
        best_state = None
        best_dev = -1.0
        stale = 0
        rng = np.random.default_rng(self.seed + 92_001)
        for epoch in range(self.training.epochs):
            self.model.train()
            for batch_idx in _batches(rng.permutation(len(y_train)), self.training.batch_size):
                idx = torch.as_tensor(batch_idx, dtype=torch.long, device=self.device)
                loss = F.cross_entropy(self.model(tx.index_select(0, idx)), ty.index_select(0, idx))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            train_acc = _tensor_accuracy(self.model, tx, ty)
            dev_acc = _tensor_accuracy(self.model, dx, dy)
            self.history.append({"epoch": float(epoch + 1), "train_acc": train_acc, "dev_acc": dev_acc})
            if dev_acc > best_dev:
                best_dev = dev_acc
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= self.training.patience:
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

    def predict(self, examples: Sequence[MultiViewTaskExample]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(f"{self.method} has not been fit")
        x = torch.as_tensor(
            _hashed_text_features([self.text_builder(example) for example in examples], self.feature_dim),
            dtype=torch.float32,
            device=self.device,
        )
        preds = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, x.shape[0], 2048):
                logits = self.model(x[start : start + 2048])
                preds.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
        return np.concatenate(preds).astype(np.int64)


class FixedPredictionBaseline:
    def __init__(self, method: str, prediction: int) -> None:
        self.method = method
        self.prediction = int(prediction)
        self.param_count = 0

    def predict(self, examples: Sequence[MultiViewTaskExample]) -> np.ndarray:
        return np.full(len(examples), self.prediction, dtype=np.int64)


class StructuredEvidenceNeuralControl:
    def __init__(
        self,
        method: str,
        num_classes: int,
        training: MLPTrainingConfig,
        seed: int,
        device: str,
        feature_mode: str,
    ) -> None:
        self.method = method
        self.num_classes = int(num_classes)
        self.training = training
        self.seed = int(seed)
        self.device = torch.device(device)
        self.feature_mode = feature_mode
        self.model: nn.Module | None = None
        self.param_count = 0
        self.history: List[Dict[str, float]] = []

    def fit(self, train_examples: Sequence[MultiViewTaskExample], dev_examples: Sequence[MultiViewTaskExample]) -> None:
        torch.manual_seed(self.seed + 101_001)
        x_train = _structured_candidate_features(train_examples, self.feature_mode)
        y_train = np.asarray([example.label for example in train_examples], dtype=np.int64)
        x_dev = _structured_candidate_features(dev_examples, self.feature_mode)
        y_dev = np.asarray([example.label for example in dev_examples], dtype=np.int64)
        self.model = CandidatewiseScoringModel(
            input_dim=x_train.shape[-1],
            hidden_dims=self.training.hidden_dims,
        ).to(self.device)
        self.param_count = int(sum(parameter.numel() for parameter in self.model.parameters()))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.training.lr, weight_decay=self.training.weight_decay)
        tx = torch.as_tensor(x_train, dtype=torch.float32, device=self.device)
        ty = torch.as_tensor(y_train, dtype=torch.long, device=self.device)
        dx = torch.as_tensor(x_dev, dtype=torch.float32, device=self.device)
        dy = torch.as_tensor(y_dev, dtype=torch.long, device=self.device)
        best_state = None
        best_dev = -1.0
        stale = 0
        rng = np.random.default_rng(self.seed + 102_001)
        for epoch in range(self.training.epochs):
            self.model.train()
            for batch_idx in _batches(rng.permutation(len(y_train)), self.training.batch_size):
                idx = torch.as_tensor(batch_idx, dtype=torch.long, device=self.device)
                loss = F.cross_entropy(self.model(tx.index_select(0, idx)), ty.index_select(0, idx))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            train_acc = _tensor_accuracy(self.model, tx, ty)
            dev_acc = _tensor_accuracy(self.model, dx, dy)
            self.history.append({"epoch": float(epoch + 1), "train_acc": train_acc, "dev_acc": dev_acc})
            if dev_acc > best_dev:
                best_dev = dev_acc
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= self.training.patience:
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

    def predict(self, examples: Sequence[MultiViewTaskExample]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(f"{self.method} has not been fit")
        x = torch.as_tensor(_structured_candidate_features(examples, self.feature_mode), dtype=torch.float32, device=self.device)
        preds = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, x.shape[0], 2048):
                logits = self.model(x[start : start + 2048])
                preds.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
        return np.concatenate(preds).astype(np.int64)


class CandidatewiseScoringModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Iterable[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        current = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current, int(hidden_dim)))
            layers.append(nn.GELU())
            current = int(hidden_dim)
        layers.append(nn.Linear(current, 1))
        self.scorer = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, candidates, dim = x.shape
        return self.scorer(x.reshape(batch * candidates, dim)).reshape(batch, candidates)


def run_experiment(config_path: str) -> Dict[str, object]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    previous_run_summary = _previous_run_summary(Path(str(config.get("output_path", "results/real_shared_weight_latent_coordination_results.json"))))
    seed_values = [int(value) for value in config.get("seeds", [0])]
    device = _resolve_device(str(config.get("device", "cpu")))
    if bool(config.get("deterministic", False)):
        torch.use_deterministic_algorithms(True, warn_only=True)

    dataset_config = _make_dataclass(MultiViewCodePatchDatasetConfig, config.get("dataset", {}))
    agent_config = _make_dataclass(SharedTransformerAgentConfig, config.get("agent", {}))
    coordinator_config_base = _make_dataclass(LatentCoordinatorConfig, config.get("coordinator", {}))
    training_config = _make_dataclass(RealSharedWeightTrainingConfig, config.get("training", {}))
    message_channel_base = _make_dataclass(MessageChannelConfig, config.get("message_channel", {}))
    baseline_training = _make_mlp_training_config(config.get("baseline_training", config.get("training", {})))
    controls = [str(value) for value in config.get("controls", list(CONTROL_NAMES))]
    coordinator_families = [str(value) for value in config.get("coordinator_families", [coordinator_config_base.family])]

    metrics: List[Dict[str, object]] = []
    diagnostics: List[Dict[str, object]] = []
    audit: List[Dict[str, object]] = []
    audit_log_rows: List[Dict[str, object]] = []
    summaries: Dict[str, object] = {}

    for seed in seed_values:
        print(f"seed={seed}: building real shared-weight local-code splits")
        splits = build_multiview_code_patch_splits(dataset_config, seed=seed, repo_root=Path("."))
        summaries[str(seed)] = dataset_summary(splits)
        audit.append(split_leakage_audit(BENCHMARK, seed, splits))
        audit.append(output_leakage_audit(BENCHMARK, seed, splits["train"] + splits["dev"] + splits["test"]))
        diagnostics.extend(_benchmark_validity_diagnostic_rows(splits, seed, dataset_config.num_candidates))
        event_index = 0

        canonical_trainable: FitResult | None = None
        canonical_frozen: FitResult | None = None
        latent_results: List[FitResult] = []
        for family_index, family in enumerate(coordinator_families):
            coordinator_config = _replace_dataclass(coordinator_config_base, family=family, input_dim=agent_config.hidden_dim)
            suffix = "" if family_index == 0 else f"_{family}"
            frozen_name = f"frozen_shared_agent_latent_coordinator{suffix}"
            trainable_name = f"trainable_shared_agent_latent_coordinator{suffix}"

            print(f"seed={seed}: fitting {frozen_name}")
            audit.append(audit_event(BENCHMARK, seed, event_index, "fit", frozen_name, "none", ("train", "dev")))
            event_index += 1
            frozen = fit_latent_system(
                train_examples=splits["train"],
                dev_examples=splits["dev"],
                agent_config=agent_config,
                coordinator_config=coordinator_config,
                training_config=training_config,
                num_classes=dataset_config.num_candidates,
                seed=seed + family_index * 1000,
                device=device,
                trainable_agent=False,
                method=frozen_name,
            )
            diagnostics.append(frozen.audit)
            audit_log_rows.append(frozen.audit)
            latent_results.append(frozen)

            print(f"seed={seed}: fitting {trainable_name}")
            audit.append(audit_event(BENCHMARK, seed, event_index, "fit", trainable_name, "none", ("train", "dev")))
            event_index += 1
            trainable = fit_latent_system(
                train_examples=splits["train"],
                dev_examples=splits["dev"],
                agent_config=agent_config,
                coordinator_config=coordinator_config,
                training_config=training_config,
                num_classes=dataset_config.num_candidates,
                seed=seed + 17 + family_index * 1000,
                device=device,
                trainable_agent=True,
                method=trainable_name,
            )
            diagnostics.append(trainable.audit)
            audit_log_rows.append(trainable.audit)
            latent_results.append(trainable)
            diagnostics.extend(_learning_curve_rows(frozen, seed))
            diagnostics.extend(_learning_curve_rows(trainable, seed))
            if family_index == 0:
                canonical_trainable = trainable
                canonical_frozen = frozen

        if canonical_trainable is None or canonical_frozen is None:
            raise RuntimeError("at least one coordinator family is required")

        proposed_trainable: FitResult | None = None
        proposed_frozen: FitResult | None = None
        if bool(config.get("run_message_channel_variants", True)):
            run_layer_token_sweep = bool(config.get("run_layer_token_sweep_variants", False))
            diagnostics.extend(
                _message_variant_descriptor_rows(
                    message_channel_base,
                    agent_config,
                    seed,
                    include_layer_token_sweep=run_layer_token_sweep,
                )
            )
            message_specs = _message_variant_specs(message_channel_base, agent_config)
            if run_layer_token_sweep:
                message_specs.extend(_layer_token_sweep_variant_specs(message_channel_base, agent_config))
            for variant_index, spec in enumerate(message_specs):
                method = str(spec["method"])
                print(f"seed={seed}: fitting {method}")
                audit.append(audit_event(BENCHMARK, seed, event_index, "fit", method, "none", ("train", "dev")))
                event_index += 1
                message_result = fit_latent_system(
                    train_examples=splits["train"],
                    dev_examples=splits["dev"],
                    agent_config=agent_config,
                    coordinator_config=coordinator_config_base,
                    training_config=training_config,
                    num_classes=dataset_config.num_candidates,
                    seed=seed + 30_000 + variant_index * 101,
                    device=device,
                    trainable_agent=bool(spec["trainable_agent"]),
                    method=method,
                    message_config=spec["message_config"],
                )
                diagnostics.append(message_result.audit)
                audit_log_rows.append(message_result.audit)
                latent_results.append(message_result)
                diagnostics.extend(_learning_curve_rows(message_result, seed))
                if method == PROPOSED_MESSAGE_METHOD:
                    proposed_trainable = message_result
                elif method == PROPOSED_FROZEN_MESSAGE_METHOD:
                    proposed_frozen = message_result

        print(f"seed={seed}: fitting randomized-label sanity condition")
        randomized_train = randomized_labels_for_examples(splits["train"], seed + 44_001, dataset_config.num_candidates)
        randomized_dev = randomized_labels_for_examples(splits["dev"], seed + 44_101, dataset_config.num_candidates)
        audit.append(audit_event(BENCHMARK, seed, event_index, "fit", "trainable_shared_agent_latent_coordinator", "randomized_labels", ("train", "dev")))
        event_index += 1
        randomized = fit_latent_system(
            train_examples=splits["train"],
            dev_examples=splits["dev"],
            agent_config=agent_config,
            coordinator_config=_replace_dataclass(coordinator_config_base, input_dim=agent_config.hidden_dim),
            training_config=training_config,
            num_classes=dataset_config.num_candidates,
            seed=seed + 23_000,
            device=device,
            trainable_agent=True,
            method="trainable_shared_agent_latent_coordinator",
            condition="randomized_labels",
            train_labels=randomized_train,
            dev_labels=randomized_dev,
        )
        diagnostics.append(randomized.audit)
        audit_log_rows.append(randomized.audit)

        randomized_message: FitResult | None = None
        if proposed_trainable is not None:
            print(f"seed={seed}: fitting randomized-label sanity condition for {PROPOSED_MESSAGE_METHOD}")
            audit.append(audit_event(BENCHMARK, seed, event_index, "fit", PROPOSED_MESSAGE_METHOD, "randomized_labels", ("train", "dev")))
            event_index += 1
            randomized_message = fit_latent_system(
                train_examples=splits["train"],
                dev_examples=splits["dev"],
                agent_config=agent_config,
                coordinator_config=coordinator_config_base,
                training_config=training_config,
                num_classes=dataset_config.num_candidates,
                seed=seed + 44_500,
                device=device,
                trainable_agent=True,
                method=PROPOSED_MESSAGE_METHOD,
                condition="randomized_labels",
                train_labels=randomized_train,
                dev_labels=randomized_dev,
                message_config=proposed_trainable.system.message_config,
            )
            diagnostics.append(randomized_message.audit)
            audit_log_rows.append(randomized_message.audit)
            diagnostics.extend(_learning_curve_rows(randomized_message, seed))

        print(f"seed={seed}: fitting explicit-evidence latent upper-bound baseline")
        explicit_latent = fit_latent_system(
            train_examples=splits["train"],
            dev_examples=splits["dev"],
            agent_config=agent_config,
            coordinator_config=_replace_dataclass(coordinator_config_base, input_dim=agent_config.hidden_dim),
            training_config=training_config,
            num_classes=dataset_config.num_candidates,
            seed=seed + 24_000,
            device=device,
            trainable_agent=True,
            method="trainable_shared_agent_latent_visible_explicit_evidence",
            condition="visible_explicit_evidence",
            visible_explicit_evidence=True,
        )
        diagnostics.append(explicit_latent.audit)
        audit_log_rows.append(explicit_latent.audit)
        diagnostics.extend(_learning_curve_rows(explicit_latent, seed))

        print(f"seed={seed}: fitting text-only, single-agent, and probe baselines")
        text_only = TextOutputOnlyCoordinator(
            num_classes=dataset_config.num_candidates,
            training=baseline_training,
            seed=seed + 8_000,
            device=device,
            feature_dim=int(config.get("text_feature_dim", 256)),
        )
        audit.append(audit_event(BENCHMARK, seed, event_index, "fit", "text_output_only_multi_agent_coordinator", "none", ("train", "dev")))
        event_index += 1
        text_only.fit(splits["train"], splits["dev"])

        train_labels = np.asarray([example.label for example in splits["train"]], dtype=np.int64)
        label_counts = np.bincount(train_labels, minlength=dataset_config.num_candidates)
        majority_prediction = int(np.argmax(label_counts))
        diagnostic_baselines: List[object] = [
            FixedPredictionBaseline("majority_class_baseline", majority_prediction),
            FixedPredictionBaseline("candidate_order_baseline", majority_prediction),
        ]
        patch_only = BagOfWordsDiagnosticBaseline(
            method="bag_of_words_candidate_patch_only",
            num_classes=dataset_config.num_candidates,
            training=baseline_training,
            seed=seed + 8_100,
            device=device,
            text_builder=lambda example: format_candidate_block(example),
            feature_dim=int(config.get("text_feature_dim", 256)),
        )
        patch_only.fit(splits["train"], splits["dev"])
        diagnostic_baselines.append(patch_only)
        partial_text = BagOfWordsDiagnosticBaseline(
            method="text_only_partial_view_baseline",
            num_classes=dataset_config.num_candidates,
            training=baseline_training,
            seed=seed + 8_200,
            device=device,
            text_builder=lambda example: format_clone_prompt(example, 0),
            feature_dim=int(config.get("text_feature_dim", 256)),
        )
        partial_text.fit(splits["train"], splits["dev"])
        diagnostic_baselines.append(partial_text)
        for role_index in range(len(splits["train"][0].views)):
            role_baseline = BagOfWordsDiagnosticBaseline(
                method=f"single_view_text_role_{role_index}",
                num_classes=dataset_config.num_candidates,
                training=baseline_training,
                seed=seed + 8_300 + role_index,
                device=device,
                text_builder=lambda example, role_index=role_index: format_clone_prompt(example, role_index),
                feature_dim=int(config.get("text_feature_dim", 256)),
            )
            role_baseline.fit(splits["train"], splits["dev"])
            diagnostic_baselines.append(role_baseline)
        masked_train = apply_example_control(splits["train"], "view_masked", seed=seed + 8_400)
        masked_dev = apply_example_control(splits["dev"], "view_masked", seed=seed + 8_401)
        masked_evidence = BagOfWordsDiagnosticBaseline(
            method="masked_evidence_text_baseline",
            num_classes=dataset_config.num_candidates,
            training=baseline_training,
            seed=seed + 8_500,
            device=device,
            text_builder=lambda example: "\n".join(view.text for view in example.views) + "\n" + format_candidate_block(example),
            feature_dim=int(config.get("text_feature_dim", 256)),
        )
        masked_evidence.fit(masked_train, masked_dev)
        diagnostic_baselines.append(masked_evidence)
        role_labels_only = BagOfWordsDiagnosticBaseline(
            method="role_labels_only_baseline",
            num_classes=dataset_config.num_candidates,
            training=baseline_training,
            seed=seed + 8_600,
            device=device,
            text_builder=lambda example: "\n".join(view.role for view in example.views),
            feature_dim=32,
        )
        role_labels_only.fit(splits["train"], splits["dev"])
        diagnostic_baselines.append(role_labels_only)

        positive_control_training = _make_mlp_training_config(config.get("positive_control_training", config.get("baseline_training", config.get("training", {}))))
        positive_controls: List[object] = []
        for offset, (method, mode) in enumerate(
            [
                ("explicit_evidence_neural_tuple_model", "tuple"),
                ("raw_structured_evidence_coordinator", "raw"),
            ]
        ):
            control = StructuredEvidenceNeuralControl(
                method=method,
                num_classes=dataset_config.num_candidates,
                training=positive_control_training,
                seed=seed + 12_000 + offset,
                device=device,
                feature_mode=mode,
            )
            control.fit(splits["train"], splits["dev"])
            positive_controls.append(control)

        large_agent_config = _replace_dataclass(
            agent_config,
            hidden_dim=int(config.get("large_full_context_hidden_dim", max(agent_config.hidden_dim * 2, 96))),
            adapter_hidden_dim=int(config.get("large_full_context_hidden_dim", max(agent_config.hidden_dim * 2, 96))),
            tiny_layers=int(config.get("large_full_context_layers", max(agent_config.tiny_layers + 1, 2))),
            tiny_heads=int(config.get("large_full_context_heads", max(agent_config.tiny_heads, 2))),
            tiny_ff_dim=int(config.get("large_full_context_ff_dim", max(agent_config.tiny_ff_dim * 2, 192))),
            tiny_vocab_size=int(config.get("large_full_context_vocab_size", max(agent_config.tiny_vocab_size, 4096))),
        )
        large_full_context = fit_context_baseline(
            train_examples=splits["train"],
            dev_examples=splits["dev"],
            agent_config=large_agent_config,
            training_config=training_config,
            num_classes=dataset_config.num_candidates,
            seed=seed + 13_000,
            device=device,
            method="larger_tiny_transformer_full_context",
            prompt_mode="full",
        )

        full_context = fit_context_baseline(
            train_examples=splits["train"],
            dev_examples=splits["dev"],
            agent_config=agent_config,
            training_config=training_config,
            num_classes=dataset_config.num_candidates,
            seed=seed + 9_000,
            device=device,
            method="single_agent_full_context",
            prompt_mode="full",
        )
        partial_context = fit_context_baseline(
            train_examples=splits["train"],
            dev_examples=splits["dev"],
            agent_config=agent_config,
            training_config=training_config,
            num_classes=dataset_config.num_candidates,
            seed=seed + 10_000,
            device=device,
            method="single_agent_partial_view",
            prompt_mode="partial",
        )

        frozen_hidden_batches = capture_hidden_attempt_batches(canonical_frozen.system, splits, agent_config.hidden_dim, device)
        hidden_probe = HiddenStateOnlyLabelProbe(
            num_classes=dataset_config.num_candidates,
            components=min(8, agent_config.hidden_dim),
            training=baseline_training,
            seed=seed + 11_000,
            device=device,
            pooling="mean",
        )
        hidden_probe.fit(frozen_hidden_batches["train"], frozen_hidden_batches["dev"])

        audit.append(audit_event(BENCHMARK, seed, event_index, "test_gate_opened", None, None, ()))
        event_index += 1

        for result in latent_results:
            if (
                result.method == "trainable_shared_agent_latent_coordinator"
                or bool(result.audit.get("uses_msg_token", False))
                or bool(result.audit.get("uses_active_message_readout", False))
            ):
                diagnostics.extend(
                    _message_mechanism_diagnostic_rows(
                        result,
                        splits,
                        seed=seed,
                        num_classes=dataset_config.num_candidates,
                        device=device,
                    )
                )

        for split_name in ("train", "dev", "test"):
            if split_name == "test":
                audit.append(audit_event(BENCHMARK, seed, event_index, "predict", "all_real_shared_weight_methods", "none", ("test",)))
                event_index += 1
            examples = splits[split_name]
            batch = attempt_batch_from_examples(examples, split=split_name, hidden_dim=agent_config.hidden_dim)
            for result in latent_results:
                predictions = predict_latent_system(result, examples, condition="none", seed=seed + _split_offset(split_name))
                metrics.append(
                    evaluate_predictions(
                        method=result.method,
                        condition="none",
                        predictions=predictions,
                        batch=batch,
                        seed=seed,
                        param_count=result.param_count,
                        benchmark=BENCHMARK,
                        metadata=_fit_metadata(result),
                    )
                )
            metrics.append(
                evaluate_predictions(
                    method=explicit_latent.method,
                    condition="visible_explicit_evidence",
                    predictions=predict_latent_system(explicit_latent, examples, condition="none", seed=seed + _split_offset(split_name)),
                    batch=batch,
                    seed=seed,
                    param_count=explicit_latent.param_count,
                    benchmark=BENCHMARK,
                    metadata={**_fit_metadata(explicit_latent), "upper_bound_visible_explicit_evidence": True},
                )
            )
            metrics.append(
                evaluate_predictions(
                    method="text_output_only_multi_agent_coordinator",
                    condition="none",
                    predictions=text_only.predict(examples),
                    batch=batch,
                    seed=seed,
                    param_count=text_only.param_count,
                    benchmark=BENCHMARK,
                    metadata={"uses_hidden_activations": False, "visible_outputs_only": True},
                )
            )
            for baseline in diagnostic_baselines:
                baseline_examples = examples
                condition = "none"
                if getattr(baseline, "method", "") == "masked_evidence_text_baseline":
                    baseline_examples = apply_example_control(examples, "view_masked", seed=seed + _split_offset(split_name))
                    condition = "view_masked"
                metrics.append(
                    evaluate_predictions(
                        method=getattr(baseline, "method"),
                        condition=condition,
                        predictions=baseline.predict(baseline_examples),
                        batch=attempt_batch_from_examples(baseline_examples, split=split_name, hidden_dim=agent_config.hidden_dim),
                        seed=seed,
                        param_count=int(getattr(baseline, "param_count", 0)),
                        benchmark=BENCHMARK,
                        metadata={"diagnostic_triviality_baseline": True, "num_classes": dataset_config.num_candidates},
                    )
                )
            for control in positive_controls:
                metrics.append(
                    evaluate_predictions(
                        method=getattr(control, "method"),
                        condition="explicit_evidence",
                        predictions=control.predict(examples),
                        batch=batch,
                        seed=seed,
                        param_count=int(getattr(control, "param_count", 0)),
                        benchmark=BENCHMARK,
                        metadata={"learned_positive_control": True},
                    )
                )
            metrics.append(
                evaluate_predictions(
                    method="single_agent_full_context",
                    condition="none",
                    predictions=predict_context_baseline(full_context, examples),
                    batch=batch,
                    seed=seed,
                    param_count=full_context.param_count,
                    benchmark=BENCHMARK,
                    metadata={"receives_all_views": True},
                )
            )
            metrics.append(
                evaluate_predictions(
                    method="larger_tiny_transformer_full_context",
                    condition="none",
                    predictions=predict_context_baseline(large_full_context, examples),
                    batch=batch,
                    seed=seed,
                    param_count=large_full_context.param_count,
                    benchmark=BENCHMARK,
                    metadata={"receives_all_views": True, "larger_tiny_transformer": True},
                )
            )
            metrics.append(
                evaluate_predictions(
                    method="single_agent_partial_view",
                    condition="none",
                    predictions=predict_context_baseline(partial_context, examples),
                    batch=batch,
                    seed=seed,
                    param_count=partial_context.param_count,
                    benchmark=BENCHMARK,
                    metadata={"receives_one_partial_view": True},
                )
            )
            metrics.append(
                evaluate_predictions(
                    method="explicit_evidence_oracle",
                    condition="none",
                    predictions=np.asarray([example.label for example in examples], dtype=np.int64),
                    batch=batch,
                    seed=seed,
                    param_count=0,
                    benchmark=BENCHMARK,
                    metadata={"uses_structured_gold_evidence": True, "oracle_upper_bound": True},
                )
            )
            hidden_batch = frozen_hidden_batches[split_name]
            metrics.append(
                evaluate_predictions(
                    method="coordinator_only_probe",
                    condition="none",
                    predictions=hidden_probe.predict(hidden_batch),
                    batch=hidden_batch,
                    seed=seed,
                    param_count=int(hidden_probe.param_count),
                    benchmark=BENCHMARK,
                    metadata={**hidden_probe.metadata(), "diagnostic_only": True, "shared_agent_trainable": False},
                )
            )

        for split_name in ("dev", "test"):
            base_examples = splits[split_name]
            control_targets = [proposed_trainable or canonical_trainable]
            for control_target in control_targets:
                for condition in controls:
                    if condition in {"none", "randomized_labels"}:
                        continue
                    controlled = apply_example_control(base_examples, condition=condition, seed=seed + _split_offset(split_name))
                    predictions = predict_latent_system(
                        control_target,
                        controlled,
                        condition=condition,
                        seed=seed + _split_offset(split_name),
                    )
                    control_batch = attempt_batch_from_examples(controlled, split=split_name, hidden_dim=agent_config.hidden_dim)
                    metrics.append(
                        evaluate_predictions(
                            method=control_target.method,
                            condition=condition,
                            predictions=predictions,
                            batch=control_batch,
                            seed=seed,
                            param_count=control_target.param_count,
                            benchmark=BENCHMARK,
                            metadata=_fit_metadata(control_target),
                        )
                    )
                    control_acc = float(np.mean(predictions.astype(np.int64) == control_batch.labels.astype(np.int64)))
                    if (
                        control_target.method == (proposed_trainable.method if proposed_trainable is not None else canonical_trainable.method)
                        and condition in {"view_masked", "view_shuffled", "hidden_states_shuffled_across_examples", "role_labels_shuffled"}
                    ):
                        diagnostics.append(
                            {
                                "benchmark": BENCHMARK,
                                "seed": seed,
                                "split": split_name,
                                "probe": "benchmark_validity_control_inspection",
                                "condition": condition,
                                "accuracy": control_acc,
                                "chance": 1.0 / dataset_config.num_candidates,
                                "correct_examples": _control_correct_examples(controlled, predictions, limit=5),
                            }
                        )
            randomized_predictions = predict_latent_system(randomized, base_examples, condition="none", seed=seed + _split_offset(split_name))
            metrics.append(
                evaluate_predictions(
                    method="trainable_shared_agent_latent_coordinator",
                    condition="randomized_labels",
                    predictions=randomized_predictions,
                    batch=attempt_batch_from_examples(base_examples, split=split_name, hidden_dim=agent_config.hidden_dim),
                    seed=seed,
                    param_count=randomized.param_count,
                    benchmark=BENCHMARK,
                    metadata={**_fit_metadata(randomized), "train_labels_randomized": True},
                )
            )
            if randomized_message is not None:
                randomized_message_predictions = predict_latent_system(randomized_message, base_examples, condition="none", seed=seed + _split_offset(split_name))
                metrics.append(
                    evaluate_predictions(
                        method=PROPOSED_MESSAGE_METHOD,
                        condition="randomized_labels",
                        predictions=randomized_message_predictions,
                        batch=attempt_batch_from_examples(base_examples, split=split_name, hidden_dim=agent_config.hidden_dim),
                        seed=seed,
                        param_count=randomized_message.param_count,
                        benchmark=BENCHMARK,
                        metadata={**_fit_metadata(randomized_message), "train_labels_randomized": True},
                    )
                )

    diagnostics.extend(_active_message_accuracy_comparison_rows(metrics, dataset_config.num_candidates))
    audit.append(summarize_test_access(audit))
    validation = _validation_summary(metrics, diagnostics, audit, num_classes=dataset_config.num_candidates)
    results = {
        "metadata": {
            "stage": "real_shared_weight_latent_coordination",
            "config_path": str(config_path),
            "config": config,
            "device_requested": config.get("device", "cpu"),
            "device_used": device,
            "agent_config": asdict(agent_config),
            "coordinator_config": asdict(coordinator_config_base),
            "message_channel_config": asdict(message_channel_base),
            "proposed_latent_method": PROPOSED_MESSAGE_METHOD,
            "proposed_frozen_method": PROPOSED_FROZEN_MESSAGE_METHOD,
            "training_config": asdict(training_config),
            "baseline_training_config": asdict(baseline_training),
            "dataset_config": asdict(dataset_config),
            "dataset_summary_by_seed": summaries,
            "previous_run_summary": previous_run_summary,
            "environment": _environment_summary(device),
        },
        "metrics": metrics,
        "diagnostics": diagnostics,
        "audit": audit,
        "validation": validation,
    }
    output_path = Path(str(config.get("output_path", "results/real_shared_weight_latent_coordination_results.json")))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    audit_path = Path(str(config.get("audit_log_path", "results/real_shared_weight_latent_coordination_audit.jsonl")))
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in audit_log_rows + audit), encoding="utf-8")
    report_path = Path(str(config.get("report_path", "reports/REPORT.md")))
    write_report(results, report_path)
    return results


def fit_latent_system(
    train_examples: Sequence[MultiViewTaskExample],
    dev_examples: Sequence[MultiViewTaskExample],
    agent_config: SharedTransformerAgentConfig,
    coordinator_config: LatentCoordinatorConfig,
    training_config: RealSharedWeightTrainingConfig,
    num_classes: int,
    seed: int,
    device: str,
    trainable_agent: bool,
    method: str,
    condition: str = "none",
    train_labels: np.ndarray | None = None,
    dev_labels: np.ndarray | None = None,
    visible_explicit_evidence: bool = False,
    message_config: MessageChannelConfig | None = None,
) -> FitResult:
    torch.manual_seed(seed + 19_007)
    np_rng = np.random.default_rng(seed + 29_001)
    message_config = message_config or MessageChannelConfig()
    agent = SharedTransformerAgent(agent_config).to(device)
    agent.configure_trainable(trainable_agent)
    coordinator_input_dim = int(message_config.message_dim) if message_config.use_message_head else int(agent.output_dim)
    effective_coordinator_config = _replace_dataclass(coordinator_config, input_dim=coordinator_input_dim)
    candidate_query_families = {
        "candidate_query_cross_attention",
        "candidate_token_cross_attention",
        "bilinear_candidate",
        "contrastive_candidate",
        "global_then_candidate",
        "two_round_message_passing",
    }
    if message_config.coordinator_family in candidate_query_families:
        family = "cross_attention" if message_config.coordinator_family == "candidate_query_cross_attention" else message_config.coordinator_family
        coordinator = make_candidate_query_coordinator(
            n_roles=len(train_examples[0].views),
            candidate_feature_dim=4,
            config=_replace_dataclass(effective_coordinator_config, family=family),
        ).to(device)
    elif message_config.coordinator_family == "latent":
        coordinator = make_latent_coordinator(
            n_roles=len(train_examples[0].views),
            num_classes=num_classes,
            config=effective_coordinator_config,
        ).to(device)
    else:
        raise ValueError(f"unknown message coordinator family: {message_config.coordinator_family}")
    system = SharedClonedAgentSystem(
        agent,
        coordinator,
        n_roles=len(train_examples[0].views),
        visible_explicit_evidence=visible_explicit_evidence,
        message_config=message_config,
    ).to(device)
    params = [parameter for parameter in system.parameters() if parameter.requires_grad]
    if not params:
        raise ValueError("no trainable parameters available for latent system")
    optimizer = torch.optim.AdamW(params, lr=training_config.lr, weight_decay=training_config.weight_decay)
    initial_agent = _flat_named_parameters(agent.coordination_parameter_items())
    initial_coordinator = _flat_module_parameters(coordinator)
    initial_active_message_readout = _flat_module_parameters(system.active_message_readout) if system.active_message_readout is not None else torch.zeros(0)
    initial_message_head = _flat_module_parameters(system.message_head) if system.message_head is not None else torch.zeros(0)
    initial_private_cue_head = _flat_module_parameters(system.private_cue_head) if system.private_cue_head is not None else torch.zeros(0)
    train_label_map = _label_map(train_examples, train_labels)
    dev_label_map = _label_map(dev_examples, dev_labels)
    history: List[Dict[str, float]] = []
    agent_grad_norms: List[float] = []
    coordinator_grad_norms: List[float] = []
    active_message_readout_grad_norms: List[float] = []
    message_head_grad_norms: List[float] = []
    private_cue_head_grad_norms: List[float] = []
    best_state = None
    best_dev = -1.0
    stale = 0
    first_backward: Dict[str, object] | None = None
    step = 0
    accumulation = max(1, int(training_config.gradient_accumulation_steps))

    for epoch in range(training_config.epochs):
        system.train()
        order = np_rng.permutation(len(train_examples))
        optimizer.zero_grad(set_to_none=True)
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        for batch_ids in _batches(order, training_config.batch_size):
            batch_examples = [train_examples[int(index)] for index in batch_ids]
            labels = _labels_tensor(batch_examples, train_label_map, device)
            with _autocast_context(device, training_config.mixed_precision):
                logits, forward_audit, clone_activations = system(
                    batch_examples,
                    condition="none",
                    seed=seed + epoch,
                    retain_activation_grad=first_backward is None,
                )
                main_loss = F.cross_entropy(logits, labels)
                aux_weight = _auxiliary_loss_weight(epoch, message_config)
                aux_loss = torch.zeros((), dtype=main_loss.dtype, device=main_loss.device)
                if message_config.use_private_cue_aux:
                    if system.last_aux_logits is None:
                        raise RuntimeError("private-cue auxiliary loss requested but no auxiliary logits were produced")
                    cue_targets = _private_cue_targets(batch_examples, logits.device)
                    aux_loss = F.cross_entropy(system.last_aux_logits.reshape(-1, 2), cue_targets.reshape(-1))
                regularization_loss = _message_regularization_loss(clone_activations, message_config)
                total_loss = main_loss + aux_weight * aux_loss + regularization_loss
                loss = total_loss / accumulation
            epoch_loss_sum += float(total_loss.detach().float().cpu().item()) * len(batch_examples)
            epoch_loss_count += len(batch_examples)
            loss.backward()
            agent_grad_norm = _grad_norm(parameter for _name, parameter in agent.coordination_parameter_items())
            coordinator_grad_norm = _grad_norm(coordinator.parameters())
            active_message_readout_grad_norm = _grad_norm(system.active_message_readout.parameters()) if system.active_message_readout is not None else 0.0
            message_head_grad_norm = _grad_norm(system.message_head.parameters()) if system.message_head is not None else 0.0
            private_cue_head_grad_norm = _grad_norm(system.private_cue_head.parameters()) if system.private_cue_head is not None else 0.0
            agent_grad_norms.append(agent_grad_norm)
            coordinator_grad_norms.append(coordinator_grad_norm)
            active_message_readout_grad_norms.append(active_message_readout_grad_norm)
            message_head_grad_norms.append(message_head_grad_norm)
            private_cue_head_grad_norms.append(private_cue_head_grad_norm)
            if first_backward is None:
                per_clone = []
                if clone_activations.grad is not None:
                    per_clone = [
                        float(clone_activations.grad[:, clone_id, :].detach().float().norm().cpu().item())
                        for clone_id in range(clone_activations.shape[1])
                    ]
                first_backward = {
                    **forward_audit,
                    "loss_backward_reaches_shared_agent": bool(agent_grad_norm > 0.0),
                    "loss_backward_reaches_active_message_readout": bool(active_message_readout_grad_norm > 0.0),
                    "loss_backward_reaches_message_head": bool(message_head_grad_norm > 0.0),
                    "loss_backward_reaches_private_cue_head": bool(private_cue_head_grad_norm > 0.0),
                    "per_clone_activation_grad_norms": per_clone,
                    "per_clone_gradient_contribution": bool(per_clone and all(value > 0.0 for value in per_clone)),
                }
            step += 1
            if step % accumulation == 0:
                if float(training_config.gradient_clip_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(params, float(training_config.gradient_clip_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if step % accumulation != 0:
            if float(training_config.gradient_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(params, float(training_config.gradient_clip_norm))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        train_acc = latent_accuracy(system, train_examples, None, device)
        dev_acc = latent_accuracy(system, dev_examples, None, device)
        dev_selection_acc = latent_accuracy(system, dev_examples, dev_label_map, device)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(epoch_loss_sum / max(1, epoch_loss_count)),
                "train_acc": train_acc,
                "dev_acc": dev_acc,
                "dev_selection_acc": dev_selection_acc,
                "aux_weight": _auxiliary_loss_weight(epoch, message_config),
            }
        )
        if dev_selection_acc > best_dev:
            best_dev = dev_selection_acc
            best_state = _system_checkpoint(system)
            stale = 0
        else:
            stale += 1
            if stale >= training_config.patience:
                break

    if best_state is not None:
        _load_system_checkpoint(system, best_state, device)
    agent_delta = _parameter_delta(agent.coordination_parameter_items(), initial_agent)
    coordinator_delta = _parameter_delta(list(coordinator.named_parameters()), initial_coordinator)
    active_message_readout_delta = (
        _parameter_delta(list(system.active_message_readout.named_parameters()), initial_active_message_readout)
        if system.active_message_readout is not None
        else 0.0
    )
    message_head_delta = _parameter_delta(list(system.message_head.named_parameters()), initial_message_head) if system.message_head is not None else 0.0
    private_cue_head_delta = (
        _parameter_delta(list(system.private_cue_head.named_parameters()), initial_private_cue_head)
        if system.private_cue_head is not None
        else 0.0
    )
    param_count = int(sum(parameter.numel() for parameter in params))
    first_backward = first_backward or {}
    audit = {
        "benchmark": BENCHMARK,
        "seed": seed,
        "split": "dev",
        "probe": "real_shared_weight_training_audit",
        "method": method,
        "condition": condition,
        "agent_trainable": trainable_agent,
        "visible_explicit_evidence": visible_explicit_evidence,
        "message_config": asdict(message_config),
        "uses_msg_token": message_config.use_msg_token,
        "msg_position": message_config.msg_position,
        "readout_source": message_config.readout_source,
        "uses_active_message_readout": message_config.readout_source == "active_message",
        "active_message_layers": [int(value) for value in _as_int_tuple(message_config.active_message_layers)],
        "active_message_heads": message_config.active_message_heads,
        "active_message_ff_dim": message_config.active_message_ff_dim,
        "active_message_readout_type": message_config.active_message_readout_type,
        "active_message_num_queries": message_config.active_message_num_queries,
        "active_message_topk": message_config.active_message_topk,
        "residual_message_readout": message_config.residual_message_readout,
        "uses_message_head": message_config.use_message_head,
        "message_head_type": message_config.message_head_type,
        "message_dim": message_config.message_dim,
        "message_coordinator_family": message_config.coordinator_family,
        "uses_private_cue_aux": message_config.use_private_cue_aux,
        "aux_loss_weight": message_config.aux_loss_weight,
        "agent_mode": agent.mode,
        "pretrained_model_name_or_path": agent_config.model_name_or_path if agent.mode == "pretrained_adapter" else None,
        "lora_modules_installed": agent.lora_modules_installed,
        "trainable_parameter_names": agent.trainable_parameter_names(),
        "coordination_parameter_names": [name for name, _parameter in agent.coordination_parameter_items()],
        "shared_parameter_identity": shared_parameter_identity_check(agent, len(train_examples[0].views))["pass"],
        "shared_parameter_count": shared_parameter_identity_check(agent, len(train_examples[0].views))["parameter_count"],
        "agent_grad_norm_mean": _mean(agent_grad_norms),
        "agent_grad_norm_std": _std(agent_grad_norms),
        "coordinator_grad_norm_mean": _mean(coordinator_grad_norms),
        "coordinator_grad_norm_std": _std(coordinator_grad_norms),
        "active_message_readout_grad_norm_mean": _mean(active_message_readout_grad_norms),
        "active_message_readout_grad_norm_std": _std(active_message_readout_grad_norms),
        "message_head_grad_norm_mean": _mean(message_head_grad_norms),
        "message_head_grad_norm_std": _std(message_head_grad_norms),
        "private_cue_head_grad_norm_mean": _mean(private_cue_head_grad_norms),
        "private_cue_head_grad_norm_std": _std(private_cue_head_grad_norms),
        "agent_parameter_delta": agent_delta,
        "coordinator_parameter_delta": coordinator_delta,
        "active_message_readout_parameter_delta": active_message_readout_delta,
        "message_head_parameter_delta": message_head_delta,
        "private_cue_head_parameter_delta": private_cue_head_delta,
        "activation_requires_grad_before_coordinator": bool(first_backward.get("activation_requires_grad_before_coordinator", False)),
        "clone_activation_requires_grad": bool(first_backward.get("activation_requires_grad_before_coordinator", False)),
        "no_detach_between_clone_activations_and_loss": bool(first_backward.get("no_detach_between_clone_activations_and_loss", False)),
        "loss_backward_reaches_shared_agent": bool(first_backward.get("loss_backward_reaches_shared_agent", False)),
        "loss_backward_reaches_active_message_readout": bool(first_backward.get("loss_backward_reaches_active_message_readout", False)),
        "loss_backward_reaches_message_head": bool(first_backward.get("loss_backward_reaches_message_head", False)),
        "loss_backward_reaches_private_cue_head": bool(first_backward.get("loss_backward_reaches_private_cue_head", False)),
        "per_clone_gradient_contribution": bool(first_backward.get("per_clone_gradient_contribution", False)),
        "per_clone_activation_grad_norms": first_backward.get("per_clone_activation_grad_norms", []),
        "epochs_run": len(history),
        "best_dev_selection_acc": best_dev,
        "batch_size": training_config.batch_size,
        "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
        "gradient_clip_norm": training_config.gradient_clip_norm,
        "mixed_precision": training_config.mixed_precision,
        "max_length": agent_config.max_length,
        "device": device,
        "cuda_max_memory_allocated": _cuda_max_memory(device),
        "param_count": param_count,
    }
    return FitResult(method, condition, agent, coordinator, system, trainable_agent, param_count, audit, history)


def fit_context_baseline(
    train_examples: Sequence[MultiViewTaskExample],
    dev_examples: Sequence[MultiViewTaskExample],
    agent_config: SharedTransformerAgentConfig,
    training_config: RealSharedWeightTrainingConfig,
    num_classes: int,
    seed: int,
    device: str,
    method: str,
    prompt_mode: str,
) -> ContextFitResult:
    torch.manual_seed(seed + 71_113)
    rng = np.random.default_rng(seed + 72_113)
    agent = SharedTransformerAgent(agent_config).to(device)
    agent.configure_trainable(True)
    head = nn.Linear(agent.output_dim, num_classes).to(device)
    params = [parameter for parameter in list(agent.parameters()) + list(head.parameters()) if parameter.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=training_config.lr, weight_decay=training_config.weight_decay)
    history: List[Dict[str, float]] = []
    best_state = None
    best_dev = -1.0
    stale = 0
    for epoch in range(training_config.epochs):
        agent.train()
        head.train()
        for batch_ids in _batches(rng.permutation(len(train_examples)), training_config.batch_size):
            batch_examples = [train_examples[int(index)] for index in batch_ids]
            logits = head(agent.forward_texts(_context_prompts(batch_examples, prompt_mode)))
            labels = torch.as_tensor([example.label for example in batch_examples], dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        train_acc = context_accuracy(agent, head, train_examples, prompt_mode, device)
        dev_acc = context_accuracy(agent, head, dev_examples, prompt_mode, device)
        history.append({"epoch": float(epoch + 1), "train_acc": train_acc, "dev_acc": dev_acc})
        if dev_acc > best_dev:
            best_dev = dev_acc
            best_state = {
                "agent": {key: value.detach().cpu().clone() for key, value in agent.state_dict().items()},
                "head": {key: value.detach().cpu().clone() for key, value in head.state_dict().items()},
            }
            stale = 0
        else:
            stale += 1
            if stale >= training_config.patience:
                break
    if best_state is not None:
        agent.load_state_dict(best_state["agent"])
        head.load_state_dict(best_state["head"])
        agent.to(device)
        head.to(device)
    param_count = int(sum(parameter.numel() for parameter in params))
    return ContextFitResult(method=method, agent=agent, head=head, param_count=param_count, history=history, prompt_mode=prompt_mode)


def predict_latent_system(
    result: FitResult,
    examples: Sequence[MultiViewTaskExample],
    condition: str,
    seed: int,
) -> np.ndarray:
    result.system.eval()
    preds = []
    with torch.no_grad():
        for batch in _example_batches(examples, 128):
            logits, _audit, _acts = result.system(batch, condition=condition, seed=seed)
            preds.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
    return np.concatenate(preds).astype(np.int64)


def latent_accuracy(
    system: SharedClonedAgentSystem,
    examples: Sequence[MultiViewTaskExample],
    label_map: Dict[str, int] | None,
    device: str,
) -> float:
    del device
    system.eval()
    values = []
    with torch.no_grad():
        for batch in _example_batches(examples, 128):
            logits, _audit, _acts = system(batch)
            labels = _labels_tensor(batch, label_map, logits.device)
            values.append((torch.argmax(logits, dim=1) == labels).float().mean().detach().cpu().item())
    return float(np.mean(values)) if values else 0.0


def predict_context_baseline(result: ContextFitResult, examples: Sequence[MultiViewTaskExample]) -> np.ndarray:
    result.agent.eval()
    result.head.eval()
    preds = []
    with torch.no_grad():
        for batch in _example_batches(examples, 128):
            logits = result.head(result.agent.forward_texts(_context_prompts(batch, result.prompt_mode)))
            preds.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
    return np.concatenate(preds).astype(np.int64)


def context_accuracy(
    agent: SharedTransformerAgent,
    head: nn.Module,
    examples: Sequence[MultiViewTaskExample],
    prompt_mode: str,
    device: str,
) -> float:
    del device
    agent.eval()
    head.eval()
    values = []
    with torch.no_grad():
        for batch in _example_batches(examples, 128):
            logits = head(agent.forward_texts(_context_prompts(batch, prompt_mode)))
            labels = torch.as_tensor([example.label for example in batch], dtype=torch.long, device=logits.device)
            values.append((torch.argmax(logits, dim=1) == labels).float().mean().detach().cpu().item())
    return float(np.mean(values)) if values else 0.0


def capture_hidden_attempt_batches(
    system: SharedClonedAgentSystem,
    splits: Dict[str, Sequence[MultiViewTaskExample]],
    hidden_dim: int,
    device: str,
) -> Dict[str, AttemptBatch]:
    del device
    out: Dict[str, AttemptBatch] = {}
    system.eval()
    with torch.no_grad():
        for split_name, examples in splits.items():
            rows = []
            for batch in _example_batches(examples, 128):
                role_outputs = []
                for role_index in range(system.n_roles):
                    role_outputs.append(system.shared_agent.forward_texts([format_clone_prompt(example, role_index) for example in batch]))
                rows.append(torch.stack(role_outputs, dim=1).detach().cpu().numpy().astype(np.float32))
            hidden = np.concatenate(rows, axis=0) if rows else np.zeros((0, system.n_roles, hidden_dim), dtype=np.float32)
            out[split_name] = attempt_batch_from_examples(
                examples,
                split=split_name,
                hidden_dim=hidden.shape[-1] if hidden.size else hidden_dim,
                hidden_states=hidden[:, :, None, :],
            )
    return out


def shared_parameter_identity_check(agent: SharedTransformerAgent, n_clones: int) -> Dict[str, object]:
    ids_by_clone = [[id(parameter) for _name, parameter in agent.named_parameters()] for _ in range(n_clones)]
    first = ids_by_clone[0] if ids_by_clone else []
    return {
        "pass": all(ids == first for ids in ids_by_clone[1:]),
        "parameter_count": len(first),
        "clone_parameter_ids": ids_by_clone,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real shared-weight latent coordination experiment.")
    parser.add_argument("--config", default="configs/real_shared_weight_latent_coordination_cpu_debug.json")
    args = parser.parse_args()
    run_experiment(args.config)


def _install_lora_adapters(model: nn.Module, target_names: Tuple[str, ...], rank: int, alpha: float) -> int:
    if rank <= 0:
        return 0
    installed = 0
    for child_name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and _is_lora_target(child_name, target_names):
            setattr(model, child_name, LoRALinear(child, rank=rank, alpha=alpha))
            installed += 1
        else:
            installed += _install_lora_adapters(child, target_names, rank, alpha)
    return installed


def _is_lora_target(module_name: str, target_names: Tuple[str, ...]) -> bool:
    return any(module_name == target or module_name.endswith(target) for target in target_names)


def _find_transformer_layers(model: nn.Module) -> List[nn.Module]:
    candidates = []
    for _name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) >= 2:
            candidates = list(module)
    return candidates


def _text_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|==|!=|<=|>=|[-+*/%(){}\[\].,:;]", text.lower())


def _text_output_features(examples: Sequence[MultiViewTaskExample], feature_dim: int) -> np.ndarray:
    texts = []
    for example in examples:
        batch = attempt_batch_from_examples([example], split="feature")
        visible = "\n".join(batch.visible_texts[0])
        candidate_text = "\n".join(candidate.text for candidate in example.candidates)
        texts.append(visible + "\n" + candidate_text)
    return _hashed_text_features(texts, feature_dim)


def _hashed_text_features(texts: Sequence[str], feature_dim: int) -> np.ndarray:
    features = np.zeros((len(texts), feature_dim), dtype=np.float32)
    for row_id, text in enumerate(texts):
        for token in _text_tokens(text):
            raw = __import__("hashlib").sha256(token.encode("utf-8")).hexdigest()
            index = int(raw[:8], 16) % feature_dim
            sign = 1.0 if int(raw[8:10], 16) % 2 == 0 else -1.0
            features[row_id, index] += sign
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1.0)


def _structured_candidate_features(
    examples: Sequence[MultiViewTaskExample],
    feature_mode: str,
) -> np.ndarray:
    rows: List[np.ndarray] = []
    for example in examples:
        oracle = example_oracle_metadata(example)
        evidence = np.asarray(oracle.get("evidence_bits", [0, 0, 0, 0]), dtype=np.float32)
        per_candidate = []
        for position, candidate in enumerate(example.candidates):
            candidate_bits = np.asarray(oracle_candidate_attribute_bits(example, candidate), dtype=np.float32)
            equality = (candidate_bits == evidence).astype(np.float32)
            mismatch = np.asarray([1.0 - equality.mean()], dtype=np.float32)
            position_features = np.zeros(len(example.candidates), dtype=np.float32)
            position_features[position] = 1.0
            if feature_mode == "raw":
                features = np.concatenate([evidence, candidate_bits, np.abs(evidence - candidate_bits), position_features])
            elif feature_mode == "tuple":
                features = np.concatenate([evidence, candidate_bits, equality, mismatch, position_features])
            else:
                raise ValueError(f"unknown structured feature mode: {feature_mode}")
            per_candidate.append(features)
        rows.append(np.stack(per_candidate).astype(np.float32))
    return np.stack(rows).astype(np.float32)


def _candidate_attribute_bits(example: MultiViewTaskExample, attributes: Sequence[str]) -> List[int]:
    pairs = example.metadata.get("attribute_value_pairs")
    if isinstance(pairs, (list, tuple)) and len(pairs) == 4:
        bits: List[int] = []
        for index, value in enumerate(attributes):
            pair = pairs[index]
            if isinstance(pair, (list, tuple)) and len(pair) > 1 and str(value) == str(pair[1]):
                bits.append(1)
            else:
                bits.append(0)
        return bits
    if len(attributes) >= 4:
        bits = []
        matched = False
        for index, value in enumerate(attributes[:4]):
            if index >= len(ATTRIBUTE_VALUE_PAIRS):
                bits.append(0)
                continue
            pair = ATTRIBUTE_VALUE_PAIRS[index]
            if str(value) == str(pair[1]):
                bits.append(1)
                matched = True
            else:
                bits.append(0)
                matched = matched or str(value) == str(pair[0])
        if matched:
            return bits
    fallback: List[int] = []
    for index, value in enumerate(attributes):
        low, high = ATTRIBUTE_VALUES[index]
        if value == high:
            fallback.append(1)
        else:
            fallback.append(0)
    return fallback


def _candidate_feature_tensor(
    examples: Sequence[MultiViewTaskExample],
    device: str | torch.device,
) -> torch.Tensor:
    rows = []
    for example in examples:
        rows.append([_candidate_attribute_bits(example, candidate.attributes) for candidate in example.candidates])
    return torch.as_tensor(rows, dtype=torch.float32, device=device)


def _private_cue_targets(
    examples: Sequence[MultiViewTaskExample],
    device: str | torch.device,
) -> torch.Tensor:
    rows = [example_oracle_metadata(example).get("evidence_bits", [0, 0, 0, 0]) for example in examples]
    return torch.as_tensor(rows, dtype=torch.long, device=device)


def _auxiliary_loss_weight(epoch: int, message_config: MessageChannelConfig) -> float:
    if not message_config.use_private_cue_aux:
        return 0.0
    base = float(message_config.aux_loss_weight)
    warmup = max(0, int(message_config.aux_warmup_epochs))
    if int(epoch) < warmup:
        return base
    decay_epochs = max(1, int(message_config.aux_decay_epochs))
    progress = min(1.0, float(int(epoch) - warmup + 1) / float(decay_epochs))
    return base * (1.0 - progress)


def _message_regularization_loss(messages: torch.Tensor, message_config: MessageChannelConfig) -> torch.Tensor:
    loss = torch.zeros((), dtype=messages.dtype, device=messages.device)
    variance_weight = float(message_config.message_variance_regularizer_weight)
    if variance_weight > 0.0 and messages.numel():
        variance = messages.float().reshape(messages.shape[0], -1).var(dim=0, unbiased=False).mean().to(dtype=messages.dtype)
        target = torch.as_tensor(float(message_config.message_variance_regularizer_target), dtype=messages.dtype, device=messages.device)
        loss = loss + variance_weight * F.relu(target - variance).pow(2)
    norm_weight = float(message_config.message_norm_regularizer_weight)
    if norm_weight > 0.0 and messages.numel():
        norms = messages.float().norm(dim=-1).to(dtype=messages.dtype)
        target = torch.as_tensor(float(message_config.message_norm_regularizer_target), dtype=messages.dtype, device=messages.device)
        loss = loss + norm_weight * (norms - target).pow(2).mean()
    return loss


def _format_explicit_evidence_clone_prompt(example: MultiViewTaskExample, role_index: int) -> str:
    evidence = ",".join(str(int(value)) for value in example_oracle_metadata(example).get("evidence_bits", [0, 0, 0, 0]))
    rows = [
        format_clone_prompt(example, role_index),
        "",
        f"Visible explicit evidence tuple for positive control: {evidence}",
        "Candidate attribute tuples:",
    ]
    for index, candidate in enumerate(example.candidates):
        rows.append(f"{index}: {','.join(candidate.attributes)}")
    return "\n".join(rows)


def _format_avenue_clone_prompt(
    example: MultiViewTaskExample,
    role_index: int,
    avenue_index: int,
    message_config: MessageChannelConfig,
) -> str:
    avenue_count = max(1, int(message_config.num_avenues))
    if avenue_count <= 1 or str(message_config.avenue_prompt_mode) == "single":
        return format_clone_prompt(example, role_index)

    view = example.views[role_index]
    mode = str(message_config.avenue_prompt_mode)
    if mode == "goal":
        goals = (
            "Identify the role-specific schema facts that constrain the missing import.",
            "Compare candidate patch intent against the observed program-analysis evidence.",
            "Check whether local naming and provider/category clues support each candidate.",
            "Check whether import-slot and call-site compatibility support each candidate.",
        )
        focus = goals[int(avenue_index) % len(goals)]
    elif mode in {"stage6_patch_selection", "swe_patch_selection"}:
        goals = (
            "Check local syntax, API compatibility, imports, names, and patch applicability.",
            "Check behavioral compatibility with the issue, failing test, traceback, and visible test evidence.",
            "Check consistency with surrounding source context, style, invariants, and edited files.",
            "Check regression, security, dependency, and related-file risk introduced by the patch.",
        )
        focus = goals[int(avenue_index) % len(goals)]
    elif mode == "schema_value_split":
        goals = (
            "Use schema field names and their attached values together.",
            "Use only concrete evidence values in the schema-aware view.",
            "Use compatibility between schema values and candidate text.",
            "Use contradictions between candidates and the role view.",
        )
        focus = goals[int(avenue_index) % len(goals)]
    else:
        raise ValueError(f"unknown avenue prompt mode: {message_config.avenue_prompt_mode}")

    return (
        f"Program-analysis artifact avenue {int(avenue_index) + 1} of {avenue_count}:\n"
        f"role_schema={view.role}\n"
        f"analysis_goal={focus}\n"
        f"{view.text}\n\n"
        f"{format_candidate_block(example)}"
    )


def _context_prompts(examples: Sequence[MultiViewTaskExample], prompt_mode: str) -> List[str]:
    if prompt_mode == "full":
        return [format_full_context_prompt(example) for example in examples]
    if prompt_mode == "partial":
        return [format_partial_context_prompt(example, role_index=0) for example in examples]
    raise ValueError(f"unknown prompt mode: {prompt_mode}")


def _labels_tensor(
    examples: Sequence[MultiViewTaskExample],
    label_map: Dict[str, int] | None,
    device: str | torch.device,
) -> torch.Tensor:
    labels = [example.label if label_map is None else label_map[example.id] for example in examples]
    return torch.as_tensor(labels, dtype=torch.long, device=device)


def _label_map(examples: Sequence[MultiViewTaskExample], labels: np.ndarray | None) -> Dict[str, int] | None:
    if labels is None:
        return None
    return {example.id: int(labels[index]) for index, example in enumerate(examples)}


def _batches(indices: Sequence[int], batch_size: int) -> Iterable[np.ndarray]:
    values = np.asarray(indices, dtype=np.int64)
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def _example_batches(examples: Sequence[MultiViewTaskExample], batch_size: int) -> Iterable[List[MultiViewTaskExample]]:
    for start in range(0, len(examples), batch_size):
        yield list(examples[start : start + batch_size])


def _system_checkpoint(system: SharedClonedAgentSystem) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in system.state_dict().items()}


def _load_system_checkpoint(
    system: SharedClonedAgentSystem,
    checkpoint: Dict[str, torch.Tensor],
    device: str,
) -> None:
    system.load_state_dict(checkpoint)
    system.to(device)


def _flat_named_parameters(items: Iterable[Tuple[str, nn.Parameter]]) -> torch.Tensor:
    values = [parameter.detach().cpu().reshape(-1).float() for _name, parameter in items]
    return torch.cat(values) if values else torch.zeros(0)


def _flat_module_parameters(module: nn.Module) -> torch.Tensor:
    return _flat_named_parameters(module.named_parameters())


def _parameter_delta(items: Iterable[Tuple[str, nn.Parameter]], initial: torch.Tensor) -> float:
    current = _flat_named_parameters(items)
    if current.numel() != initial.numel():
        raise ValueError("parameter set changed while computing delta")
    return float(torch.linalg.vector_norm(current - initial).item()) if current.numel() else 0.0


def _grad_norm(parameters: Iterable[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        total += float(parameter.grad.detach().float().pow(2).sum().cpu().item())
    return float(total ** 0.5)


def _tensor_accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return float((torch.argmax(model(x), dim=1) == y).float().mean().item())


def _fit_metadata(result: FitResult) -> Dict[str, object]:
    return {
        "agent_trainable": result.trainable_agent,
        "agent_mode": result.agent.mode,
        "agent_grad_norm_mean": result.audit.get("agent_grad_norm_mean"),
        "agent_parameter_delta": result.audit.get("agent_parameter_delta"),
        "coordinator_grad_norm_mean": result.audit.get("coordinator_grad_norm_mean"),
        "coordinator_parameter_delta": result.audit.get("coordinator_parameter_delta"),
        "active_message_readout_grad_norm_mean": result.audit.get("active_message_readout_grad_norm_mean"),
        "active_message_readout_parameter_delta": result.audit.get("active_message_readout_parameter_delta"),
        "message_head_grad_norm_mean": result.audit.get("message_head_grad_norm_mean"),
        "message_head_parameter_delta": result.audit.get("message_head_parameter_delta"),
        "private_cue_head_grad_norm_mean": result.audit.get("private_cue_head_grad_norm_mean"),
        "private_cue_head_parameter_delta": result.audit.get("private_cue_head_parameter_delta"),
        "uses_msg_token": result.audit.get("uses_msg_token"),
        "readout_source": result.audit.get("readout_source"),
        "uses_active_message_readout": result.audit.get("uses_active_message_readout"),
        "active_message_readout_type": result.audit.get("active_message_readout_type"),
        "active_message_topk": result.audit.get("active_message_topk"),
        "residual_message_readout": result.audit.get("residual_message_readout"),
        "uses_message_head": result.audit.get("uses_message_head"),
        "message_head_type": result.audit.get("message_head_type"),
        "uses_private_cue_aux": result.audit.get("uses_private_cue_aux"),
        "message_coordinator_family": result.audit.get("message_coordinator_family"),
        "shared_parameter_identity": result.audit.get("shared_parameter_identity"),
    }


def _message_variant_specs(
    base: MessageChannelConfig,
    agent_config: SharedTransformerAgentConfig,
) -> List[Dict[str, object]]:
    message_dim = int(base.message_dim) if int(base.message_dim) > 0 else int(agent_config.hidden_dim)

    def cfg(**updates) -> MessageChannelConfig:
        values = asdict(base)
        values.update({"message_dim": message_dim})
        values.update(updates)
        return MessageChannelConfig(**values)

    return [
        {
            "variant": "B",
            "method": PROPOSED_MESSAGE_METHOD,
            "trainable_agent": True,
            "message_config": cfg(
                use_msg_token=False,
                readout_source="active_message",
                use_message_head=True,
                use_private_cue_aux=True,
                coordinator_family="candidate_query_cross_attention",
            ),
            "architecture": "Learned shared message query plus role embedding cross-attends over selected token hidden states, then passes through the shared message head and candidate-query coordinator.",
            "why_tried": "Primary active-message readout test after passive [MSG] outputs collapsed.",
            "diagnostics_trigger": "Required active message readout architecture.",
        },
        {
            "variant": "C",
            "method": PROPOSED_FROZEN_MESSAGE_METHOD,
            "trainable_agent": False,
            "message_config": cfg(
                use_msg_token=False,
                readout_source="active_message",
                use_message_head=True,
                use_private_cue_aux=True,
                coordinator_family="candidate_query_cross_attention",
            ),
            "architecture": "Frozen shared agent with the same active message readout, shared message head, auxiliary cue loss, and candidate-query coordinator.",
            "why_tried": "Tests whether active readout extracts information already present in frozen token states.",
            "diagnostics_trigger": "Required frozen comparator with the exact same active readout.",
        },
        {
            "variant": "D",
            "method": "trainable_shared_agent_active_msg_head_candidate_query_no_aux",
            "trainable_agent": True,
            "message_config": cfg(
                use_msg_token=False,
                readout_source="active_message",
                use_message_head=True,
                use_private_cue_aux=False,
                coordinator_family="candidate_query_cross_attention",
            ),
            "architecture": "Active message readout with shared message head and candidate-query coordinator, without private-cue auxiliary loss.",
            "why_tried": "Ablates whether the auxiliary cue objective is necessary for active message formation.",
            "diagnostics_trigger": "Required first message-channel variant.",
        },
    ]


def _layer_token_sweep_variant_specs(
    base: MessageChannelConfig,
    agent_config: SharedTransformerAgentConfig,
) -> List[Dict[str, object]]:
    message_dim = int(base.message_dim) if int(base.message_dim) > 0 else int(agent_config.hidden_dim)

    def cfg(**updates) -> MessageChannelConfig:
        values = asdict(base)
        values.update({"message_dim": message_dim})
        values.update(updates)
        return MessageChannelConfig(**values)

    return [
        {
            "variant": "H",
            "method": EVIDENCE_TOKEN_UPPER_BOUND_METHOD,
            "trainable_agent": True,
            "message_config": cfg(
                use_msg_token=True,
                msg_position="prepend",
                readout_source="evidence_tokens",
                use_message_head=True,
                use_private_cue_aux=True,
                coordinator_family="candidate_query_cross_attention",
            ),
            "architecture": "Evidence-token final-layer readout plus shared message head, auxiliary cue loss, and candidate-query cross-attention.",
            "why_tried": "Diagnostic upper bound showing what direct cue-token extraction can solve.",
            "diagnostics_trigger": "Evidence-token diagnostic retained only as an upper bound.",
        },
        {
            "variant": "I",
            "method": FROZEN_EVIDENCE_TOKEN_UPPER_BOUND_METHOD,
            "trainable_agent": False,
            "message_config": cfg(
                use_msg_token=True,
                msg_position="prepend",
                readout_source="evidence_tokens",
                use_message_head=True,
                use_private_cue_aux=True,
                coordinator_family="candidate_query_cross_attention",
            ),
            "architecture": "Frozen shared agent with evidence-token readout and the same message head, auxiliary loss, and candidate-query coordinator.",
            "why_tried": "Frozen comparator for the evidence-token upper bound; solving here means frozen token extraction, not shared-agent learning.",
            "diagnostics_trigger": "Evidence-token diagnostic retained only as an upper bound.",
        },
    ]


def _message_variant_descriptor_rows(
    base: MessageChannelConfig,
    agent_config: SharedTransformerAgentConfig,
    seed: int,
    include_layer_token_sweep: bool = False,
) -> List[Dict[str, object]]:
    rows = [
        {
            "benchmark": BENCHMARK,
            "seed": seed,
            "split": "dev",
            "probe": "message_channel_variant",
            "variant": "A",
            "method": "trainable_shared_agent_latent_coordinator",
            "architecture": "Current raw pooled hidden activation coordinator.",
            "why_tried": "Required baseline for the failure diagnosis; it preserves the prior main path.",
            "diagnostics_trigger": "Required variant A.",
        }
    ]
    for spec in _message_variant_specs(base, agent_config):
        rows.append(
            {
                "benchmark": BENCHMARK,
                "seed": seed,
                "split": "dev",
                "probe": "message_channel_variant",
                "variant": spec["variant"],
                "method": spec["method"],
                "architecture": spec["architecture"],
                "why_tried": spec["why_tried"],
                "diagnostics_trigger": spec["diagnostics_trigger"],
            }
        )
    if include_layer_token_sweep:
        for spec in _layer_token_sweep_variant_specs(base, agent_config):
            rows.append(
                {
                    "benchmark": BENCHMARK,
                    "seed": seed,
                    "split": "dev",
                    "probe": "message_channel_variant",
                    "variant": spec["variant"],
                    "method": spec["method"],
                    "architecture": spec["architecture"],
                    "why_tried": spec["why_tried"],
                    "diagnostics_trigger": spec["diagnostics_trigger"],
                }
            )
    return rows


def _message_mechanism_diagnostic_rows(
    result: FitResult,
    splits: Dict[str, Sequence[MultiViewTaskExample]],
    seed: int,
    num_classes: int,
    device: str,
) -> List[Dict[str, object]]:
    reps = {
        split: _capture_clone_representations(result.system, examples)
        for split, examples in splits.items()
    }
    rows: List[Dict[str, object]] = []
    source_names = {
        "raw_pooled": "raw_pooled_activation",
        "msg": "msg_token",
        "evidence": "evidence_token_activation",
        "active_message": "active_message_readout",
        "message": "message_head_output" if result.system.message_head is not None else "message_channel_input",
    }
    available_sources = ["raw_pooled"]
    if bool(result.audit.get("uses_msg_token", False)):
        available_sources.append("msg")
        available_sources.append("evidence")
    if bool(result.audit.get("uses_active_message_readout", False)):
        available_sources.append("active_message")
    if result.system.message_head is not None:
        available_sources.append("message")

    for source in available_sources:
        source_label = source_names[source]
        for role_index in range(result.system.n_roles):
            scores = _fixed_linear_probe_scores(
                reps["train"][source][:, role_index, :],
                _evidence_bit_array(splits["train"], role_index),
                reps["dev"][source][:, role_index, :],
                _evidence_bit_array(splits["dev"], role_index),
                reps["test"][source][:, role_index, :],
                _evidence_bit_array(splits["test"], role_index),
                num_classes=2,
                seed=seed + 70_000 + role_index,
                device=device,
            )
            for split, accuracy in scores.items():
                rows.append(
                    {
                        "benchmark": BENCHMARK,
                        "seed": seed,
                        "split": split,
                        "probe": "private_cue_probe",
                        "method": result.method,
                        "condition": result.condition,
                        "feature_source": source_label,
                        "role_index": role_index,
                        "accuracy": accuracy,
                    }
                )
                if source == "message":
                    rows.append(
                        {
                            "benchmark": BENCHMARK,
                            "seed": seed,
                            "split": split,
                            "probe": "per_role_message_decodability",
                            "method": result.method,
                            "condition": result.condition,
                            "feature_source": source_label,
                            "role_index": role_index,
                            "accuracy": accuracy,
                        }
                    )

    combined_sources = ["raw_pooled"]
    if "message" in available_sources:
        combined_sources.append("message")
    elif "active_message" in available_sources:
        combined_sources.append("active_message")
    elif "msg" in available_sources:
        combined_sources.append("msg")
    for source in combined_sources:
        source_label = "combined_activation" if source == "raw_pooled" else "combined_message"
        scores = _fixed_linear_probe_scores(
            reps["train"][source].reshape(len(splits["train"]), -1),
            _label_array(splits["train"]),
            reps["dev"][source].reshape(len(splits["dev"]), -1),
            _label_array(splits["dev"]),
            reps["test"][source].reshape(len(splits["test"]), -1),
            _label_array(splits["test"]),
            num_classes=num_classes,
            seed=seed + 71_000,
            device=device,
        )
        for split, accuracy in scores.items():
            rows.append(
                {
                    "benchmark": BENCHMARK,
                    "seed": seed,
                    "split": split,
                    "probe": "combined_representation_probe",
                    "method": result.method,
                    "condition": result.condition,
                    "feature_source": source_label,
                    "accuracy": accuracy,
                }
            )

    for split, split_reps in reps.items():
        labels = _label_array(splits[split])
        for source in available_sources:
            stats = _message_collapse_stats(split_reps[source].reshape(len(splits[split]), -1), labels)
            rows.append(
                {
                    "benchmark": BENCHMARK,
                    "seed": seed,
                    "split": split,
                    "probe": "message_collapse_statistics",
                    "method": result.method,
                    "condition": result.condition,
                    "feature_source": source_names[source],
                    **stats,
                }
            )
    return rows


def _capture_clone_representations(system: SharedClonedAgentSystem, examples: Sequence[MultiViewTaskExample]) -> Dict[str, np.ndarray]:
    system.eval()
    rows: Dict[str, List[np.ndarray]] = {"raw_pooled": [], "msg": [], "evidence": [], "active_message": [], "message": []}
    with torch.no_grad():
        for batch in _example_batches(examples, 128):
            readouts = system.collect_clone_representations(batch)
            for key in rows:
                rows[key].append(readouts[key].detach().cpu().numpy().astype(np.float32))
    return {key: np.concatenate(values, axis=0) if values else np.zeros((0, system.n_roles, system.coordinator_input_dim), dtype=np.float32) for key, values in rows.items()}


def _fixed_linear_probe_scores(
    train_x: np.ndarray,
    train_y: np.ndarray,
    dev_x: np.ndarray,
    dev_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    num_classes: int,
    seed: int,
    device: str,
) -> Dict[str, float]:
    torch.manual_seed(seed + 8_191)
    input_dim = int(train_x.shape[-1])
    model = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, num_classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0001)
    tx = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    ty = torch.as_tensor(train_y, dtype=torch.long, device=device)
    dx = torch.as_tensor(dev_x, dtype=torch.float32, device=device)
    dy = torch.as_tensor(dev_y, dtype=torch.long, device=device)
    vx = torch.as_tensor(test_x, dtype=torch.float32, device=device)
    vy = torch.as_tensor(test_y, dtype=torch.long, device=device)
    rng = np.random.default_rng(seed + 9_191)
    best_state = None
    best_dev = -1.0
    for _epoch in range(16):
        model.train()
        for batch_ids in _batches(rng.permutation(len(train_y)), 128):
            idx = torch.as_tensor(batch_ids, dtype=torch.long, device=device)
            loss = F.cross_entropy(model(tx.index_select(0, idx)), ty.index_select(0, idx))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        dev_acc = _tensor_accuracy(model, dx, dy)
        if dev_acc > best_dev:
            best_dev = dev_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    return {
        "train": _tensor_accuracy(model, tx, ty),
        "dev": _tensor_accuracy(model, dx, dy),
        "test": _tensor_accuracy(model, vx, vy),
    }


def _message_collapse_stats(features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    if features.size == 0:
        return {
            "variance_mean": 0.0,
            "norm_mean": 0.0,
            "norm_std": 0.0,
            "mean_cosine_similarity": 0.0,
            "within_label_cosine": 0.0,
            "between_label_cosine": 0.0,
        }
    x = features.astype(np.float32, copy=False)
    norms = np.linalg.norm(x, axis=1)
    normalized = x / np.maximum(norms[:, None], 1e-8)
    if len(normalized) > 256:
        normalized = normalized[:256]
        labels = labels[:256]
    cosine = normalized @ normalized.T
    off_diag = ~np.eye(cosine.shape[0], dtype=bool)
    same = labels[:, None] == labels[None, :]
    within_mask = off_diag & same
    between_mask = off_diag & ~same
    return {
        "variance_mean": float(np.var(x, axis=0).mean()),
        "norm_mean": float(np.mean(norms)),
        "norm_std": float(np.std(norms)),
        "mean_cosine_similarity": float(np.mean(cosine[off_diag])) if np.any(off_diag) else 0.0,
        "within_label_cosine": float(np.mean(cosine[within_mask])) if np.any(within_mask) else 0.0,
        "between_label_cosine": float(np.mean(cosine[between_mask])) if np.any(between_mask) else 0.0,
    }


def _evidence_bit_array(examples: Sequence[MultiViewTaskExample], role_index: int) -> np.ndarray:
    return np.asarray([int(example_oracle_metadata(example).get("evidence_bits", [0, 0, 0, 0])[role_index]) for example in examples], dtype=np.int64)


def _label_array(examples: Sequence[MultiViewTaskExample]) -> np.ndarray:
    return np.asarray([int(example.label) for example in examples], dtype=np.int64)


def _learning_curve_rows(result: FitResult, seed: int) -> List[Dict[str, object]]:
    rows = []
    for item in result.history:
        for split, field in (("train", "train_acc"), ("dev", "dev_acc")):
            rows.append(
                {
                    "benchmark": BENCHMARK,
                    "seed": seed,
                    "split": split,
                    "probe": "real_shared_weight_learning_curve",
                    "method": result.method,
                    "condition": result.condition,
                    "epoch": int(item["epoch"]),
                    "accuracy": float(item[field]),
                }
            )
    return rows


def _validation_summary(
    metrics: List[Dict[str, object]],
    diagnostics: List[Dict[str, object]],
    audit: List[Dict[str, object]],
    num_classes: int,
) -> Dict[str, object]:
    test_none = {
        str(row["method"]): float(row["accuracy"])
        for row in metrics
        if row.get("benchmark") == BENCHMARK and row.get("split") == "test" and row.get("condition") == "none"
    }
    proposed_method = PROPOSED_MESSAGE_METHOD if PROPOSED_MESSAGE_METHOD in test_none else "trainable_shared_agent_latent_coordinator"
    frozen_method = PROPOSED_FROZEN_MESSAGE_METHOD if PROPOSED_FROZEN_MESSAGE_METHOD in test_none else "frozen_shared_agent_latent_coordinator"
    trainable_audit = next(
        (
            row
            for row in diagnostics
            if row.get("probe") == "real_shared_weight_training_audit"
            and row.get("method") == proposed_method
            and row.get("condition") == "none"
        ),
        {},
    )
    frozen_audit = next(
        (
            row
            for row in diagnostics
            if row.get("probe") == "real_shared_weight_training_audit"
            and row.get("method") == frozen_method
            and row.get("condition") == "none"
        ),
        {},
    )
    trainable_acc = test_none.get(proposed_method)
    frozen_acc = test_none.get(frozen_method)
    text_acc = test_none.get("text_output_only_multi_agent_coordinator")
    partial_acc = test_none.get("single_agent_partial_view")
    control_acc = {
        str(row["condition"]): float(row["accuracy"])
        for row in metrics
        if row.get("benchmark") == BENCHMARK
        and row.get("split") == "test"
        and row.get("method") == proposed_method
        and row.get("condition") != "none"
    }
    chance = 1.0 / max(1, int(num_classes))
    controls_collapse = bool(
        control_acc.get("randomized_labels", 1.0) <= chance + 0.10
        and control_acc.get("hidden_states_shuffled_across_examples", 1.0) <= chance + 0.10
        and control_acc.get("view_masked", 1.0) <= chance + 0.10
        and control_acc.get("view_shuffled", 1.0) <= chance + 0.10
        and abs(control_acc.get("physical_order_shuffled_roles_preserved", trainable_acc) - trainable_acc) <= 0.08
        and control_acc.get("role_labels_shuffled", 1.0) <= chance + 0.10
    )
    split_rows = [
        row
        for row in audit
        if isinstance(row, dict) and row.get("type") == "real_shared_weight_split_leakage"
    ]
    no_split_duplicates = all(
        int(row.get("train_test_id_overlap", 1)) == 0
        and int(row.get("train_test_candidate_hash_overlap", 1)) == 0
        and int(row.get("train_test_file_patch_overlap", 1)) == 0
        and int(row.get("train_test_problem_family_overlap", 1)) == 0
        for row in split_rows
    )
    trivial_methods = {
        "majority_class_baseline",
        "candidate_order_baseline",
        "bag_of_words_candidate_patch_only",
        "text_only_partial_view_baseline",
        "single_view_text_role_0",
        "single_view_text_role_1",
        "single_view_text_role_2",
        "single_view_text_role_3",
        "masked_evidence_text_baseline",
        "role_labels_only_baseline",
    }
    trivial_acc = {
        str(row["method"]): float(row["accuracy"])
        for row in metrics
        if row.get("benchmark") == BENCHMARK
        and row.get("split") == "test"
        and row.get("method") in trivial_methods
    }
    trivial_pass = bool(trivial_acc and max(trivial_acc.values()) <= chance + 0.12)
    positive_methods = {
        "explicit_evidence_neural_tuple_model",
        "raw_structured_evidence_coordinator",
        "larger_tiny_transformer_full_context",
        "trainable_shared_agent_latent_visible_explicit_evidence",
    }
    positive_acc = {
        str(row["method"]): float(row["accuracy"])
        for row in metrics
        if row.get("benchmark") == BENCHMARK
        and row.get("split") == "test"
        and row.get("method") in positive_methods
    }
    positive_control_pass = bool(positive_acc and max(positive_acc.values()) >= 0.8)
    active_collapse_rows = [
        row
        for row in diagnostics
        if row.get("probe") == "message_collapse_statistics"
        and row.get("method") == proposed_method
        and row.get("split") == "test"
        and row.get("feature_source") in {"active_message_readout", "message_head_output"}
    ]
    active_message_collapsed = bool(
        active_collapse_rows
        and all(
            float(row.get("variance_mean", 0.0)) <= 1e-6
            or float(row.get("mean_cosine_similarity", 0.0)) >= 0.98
            for row in active_collapse_rows
        )
    )
    solve_threshold = max(0.8, chance + 0.25)
    if trainable_acc is not None and frozen_acc is not None and trainable_acc >= solve_threshold and frozen_acc >= trainable_acc - 0.02:
        active_message_interpretation = "active_readout_solved_but_frozen_solves_too_report_frozen_token_extraction_not_shared_agent_learning"
    elif trainable_acc is not None and frozen_acc is not None and trainable_acc > frozen_acc and controls_collapse:
        active_message_interpretation = "trainable_active_readout_beats_frozen_and_controls_pass_evidence_for_learned_shared_agent_latent_message_formation"
    elif active_message_collapsed:
        active_message_interpretation = "active_readout_still_collapses_diagnose_message_readout_optimization_before_changing_coordinator"
    else:
        active_message_interpretation = "active_message_evidence_not_established"
    return {
        "real_shared_weight_latent_coordination": {
            "gate_a_valid_training_graph": bool(
                trainable_audit.get("shared_parameter_identity")
                and float(trainable_audit.get("agent_grad_norm_mean", 0.0)) > 0.0
                and float(trainable_audit.get("agent_parameter_delta", 0.0)) > 0.0
                and (
                    not trainable_audit.get("uses_active_message_readout")
                    or float(trainable_audit.get("active_message_readout_grad_norm_mean", 0.0)) > 0.0
                )
                and float(trainable_audit.get("coordinator_grad_norm_mean", 0.0)) > 0.0
                and trainable_audit.get("activation_requires_grad_before_coordinator")
                and trainable_audit.get("per_clone_gradient_contribution")
            ),
            "gate_b_frozen_control": bool(
                frozen_audit.get("shared_parameter_identity")
                and float(frozen_audit.get("agent_grad_norm_mean", 1.0)) == 0.0
                and float(frozen_audit.get("agent_parameter_delta", 1.0)) == 0.0
                and float(frozen_audit.get("coordinator_grad_norm_mean", 0.0)) > 0.0
            ),
            "gate_c_trainable_beats_frozen": bool(trainable_acc is not None and frozen_acc is not None and trainable_acc > frozen_acc),
            "gate_d_trainable_beats_text_only": bool(trainable_acc is not None and text_acc is not None and trainable_acc > text_acc),
            "gate_e_controls_collapse": controls_collapse,
            "gate_f_dataset_not_trivial": bool(trivial_pass and no_split_duplicates),
            "gate_positive_control_learnable": positive_control_pass,
            "proposed_method": proposed_method,
            "frozen_comparator_method": frozen_method,
            "active_message_interpretation": active_message_interpretation,
            "active_message_collapsed": active_message_collapsed,
            "test_accuracy": test_none,
            "control_accuracy": control_acc,
            "trivial_baseline_accuracy": trivial_acc,
            "positive_control_accuracy": positive_acc,
            "conservative_claim_allowed": bool(
                trainable_acc is not None
                and frozen_acc is not None
                and trainable_acc > frozen_acc
                and controls_collapse
                and not active_message_collapsed
            ),
            "dataset_note": "real local-code import-restoration benchmark from AST/import/use analysis; not evidence of open-ended code repair",
        }
    }


def _active_message_accuracy_comparison_rows(
    metrics: Sequence[Dict[str, object]],
    num_classes: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    by_key: Dict[Tuple[int, str, str], float] = {}
    for row in metrics:
        if row.get("benchmark") != BENCHMARK:
            continue
        method = str(row.get("method"))
        if method not in {PROPOSED_MESSAGE_METHOD, PROPOSED_FROZEN_MESSAGE_METHOD}:
            continue
        condition = str(row.get("condition"))
        split = str(row.get("split"))
        seed = int(row.get("seed", 0))
        by_key[(seed, split, f"{method}:{condition}")] = float(row.get("accuracy", 0.0))

    chance = 1.0 / max(1, int(num_classes))
    seeds_and_splits = sorted({(seed, split) for seed, split, _method_condition in by_key})
    for seed, split in seeds_and_splits:
        trainable = by_key.get((seed, split, f"{PROPOSED_MESSAGE_METHOD}:none"))
        frozen = by_key.get((seed, split, f"{PROPOSED_FROZEN_MESSAGE_METHOD}:none"))
        if trainable is None or frozen is None:
            continue
        if trainable >= 0.8 and frozen >= trainable - 0.02:
            interpretation = "active_readout_solved_by_frozen_token_extraction"
        elif trainable > frozen and trainable > chance + 0.10:
            interpretation = "trainable_active_readout_beats_frozen"
        else:
            interpretation = "active_readout_not_established"
        rows.append(
            {
                "benchmark": BENCHMARK,
                "seed": seed,
                "split": split,
                "probe": "trainable_vs_frozen_active_message_accuracy",
                "trainable_method": PROPOSED_MESSAGE_METHOD,
                "frozen_method": PROPOSED_FROZEN_MESSAGE_METHOD,
                "trainable_accuracy": trainable,
                "frozen_accuracy": frozen,
                "delta_trainable_minus_frozen": trainable - frozen,
                "chance": chance,
                "interpretation": interpretation,
            }
        )
    return rows


def _benchmark_validity_diagnostic_rows(
    splits: Dict[str, Sequence[MultiViewTaskExample]],
    seed: int,
    num_candidates: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for split, examples in splits.items():
        labels = [int(example.label) for example in examples]
        lengths_by_position: Dict[int, List[int]] = {index: [] for index in range(num_candidates)}
        value_counts_by_attribute: Dict[str, Dict[str, int]] = {name: {} for name in ATTRIBUTE_NAMES}
        single_view_ambiguity = []
        for example in examples:
            single_view_ambiguity.append(int(example_oracle_metadata(example).get("single_view_candidate_ambiguity_min", example.metadata.get("single_view_candidate_ambiguity_min", 0))))
            for index, candidate in enumerate(example.candidates):
                lengths_by_position[index].append(len(candidate.text))
                for attr_name, value in zip(ATTRIBUTE_NAMES, candidate.attributes):
                    value_counts_by_attribute[attr_name][str(value)] = value_counts_by_attribute[attr_name].get(str(value), 0) + 1
        rows.append(
            {
                "benchmark": BENCHMARK,
                "seed": seed,
                "split": split,
                "probe": "benchmark_validity_label_and_candidate_audit",
                "label_counts": {str(index): int(labels.count(index)) for index in range(num_candidates)},
                "chance": 1.0 / num_candidates,
                "dataset_source": examples[0].metadata.get("dataset_source") if examples else "none",
                "private_cue_style": examples[0].metadata.get("private_cue_style") if examples else "none",
                "program_analysis_artifacts": examples[0].metadata.get("program_analysis_artifacts") if examples else [],
                "real_correct_patch": all(bool(example.metadata.get("real_correct_patch")) for example in examples),
                "minimum_single_view_candidate_ambiguity": min(single_view_ambiguity) if single_view_ambiguity else 0,
                "single_view_sufficient_by_construction": bool(single_view_ambiguity and min(single_view_ambiguity) <= 1),
                "candidate_patch_length_mean_by_position": {
                    str(index): float(np.mean(lengths_by_position[index])) if lengths_by_position[index] else 0.0
                    for index in range(num_candidates)
                },
                "candidate_value_counts_by_attribute": {
                    key: dict(sorted(values.items())[:25]) for key, values in value_counts_by_attribute.items()
                },
                "problem_families": sorted({str(example_oracle_metadata(example).get("problem_family")) for example in examples}),
            }
        )
    rows.append(
        {
            "benchmark": BENCHMARK,
            "seed": seed,
            "split": "all",
            "probe": "benchmark_validity_fixes_applied",
            "failed_shortcuts_found": [
                "constructed semantic cue tokens were removed",
                "old candidates encoded abstract repair attributes rather than program-analysis patch facts",
            ],
            "dataset_fixes_applied": [
                "gold patch restores a real import statement used by real AST Name nodes",
                "private views are AST/export-kind, module-area, provider-name parity, and traceback/import-location artifacts",
                "candidate patch representations are per-example program-analysis attributes",
                "candidate patch order is randomized with balanced gold position",
                "target files are partitioned by split for leakage control",
                "masked control removes all program-analysis artifact values",
            ],
        }
    )
    return rows


def _control_correct_examples(
    examples: Sequence[MultiViewTaskExample],
    predictions: np.ndarray,
    limit: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for example, prediction in zip(examples, predictions):
        if int(prediction) != int(example.label):
            continue
        rows.append(
            {
                "id": example.id,
                "prediction": int(prediction),
                "label": int(example.label),
                "evidence_bits": example_oracle_metadata(example).get("evidence_bits"),
                "problem_family": example_oracle_metadata(example).get("problem_family"),
                "gold_tuple": example_oracle_metadata(example).get("gold_tuple"),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def _environment_summary(device: str) -> Dict[str, object]:
    info: Dict[str, object] = {
        "python": os.sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": device,
    }
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_total_memory"] = int(torch.cuda.get_device_properties(0).total_memory)
    return info


def _previous_run_summary(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    validation = data.get("validation", {}).get("real_shared_weight_latent_coordination", {})
    control_accuracy = validation.get("control_accuracy", {})
    test_accuracy = validation.get("test_accuracy", {})
    trivial_accuracy = validation.get("trivial_baseline_accuracy", {})
    return {
        "path": str(path),
        "gate_e_controls_collapse": validation.get("gate_e_controls_collapse"),
        "gate_f_dataset_not_trivial": validation.get("gate_f_dataset_not_trivial"),
        "control_accuracy": control_accuracy,
        "test_accuracy": test_accuracy,
        "trivial_baseline_accuracy": trivial_accuracy,
    }


def _cuda_max_memory(device: str) -> int:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated())
    return 0


def _autocast_context(device: str, mixed_precision: str):
    precision = str(mixed_precision or "none").lower()
    if not str(device).startswith("cuda") or not torch.cuda.is_available() or precision in {"", "none", "fp32", "float32"}:
        return nullcontext()
    if precision in {"bf16", "bfloat16"}:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision in {"fp16", "float16"}:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"unknown mixed precision mode: {mixed_precision}")


def _role_permutation(batch_size: int, n_roles: int, rng: np.random.Generator, device: torch.device) -> torch.Tensor:
    rows = np.zeros((batch_size, n_roles), dtype=np.int64)
    base = np.arange(n_roles)
    for row_id in range(batch_size):
        perm = rng.permutation(n_roles)
        if n_roles > 1 and np.array_equal(perm, base):
            perm = np.roll(perm, 1)
        rows[row_id] = perm
    return torch.as_tensor(rows, dtype=torch.long, device=device)


def _gather_role_axis(tensor: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    if tensor.shape[:2] != order.shape:
        raise ValueError(f"role gather shape mismatch: tensor={tuple(tensor.shape)} order={tuple(order.shape)}")
    index = order
    for _dim in tensor.shape[2:]:
        index = index.unsqueeze(-1)
    return tensor.gather(1, index.expand(-1, -1, *tensor.shape[2:]))


def _avenue_ids(batch_size: int, n_roles: int, n_avenues: int, device: torch.device) -> torch.Tensor:
    ids = torch.arange(n_avenues, dtype=torch.long, device=device).view(1, 1, n_avenues)
    return ids.expand(batch_size, n_roles, n_avenues)


def _avenue_permutation(
    batch_size: int,
    n_roles: int,
    n_avenues: int,
    rng: np.random.Generator,
    device: torch.device,
) -> torch.Tensor:
    rows = np.zeros((batch_size, n_roles, n_avenues), dtype=np.int64)
    base = np.arange(n_avenues)
    for batch_index in range(batch_size):
        for role_index in range(n_roles):
            perm = rng.permutation(n_avenues)
            if n_avenues > 1 and np.array_equal(perm, base):
                perm = np.roll(perm, 1)
            rows[batch_index, role_index] = perm
    return torch.as_tensor(rows, dtype=torch.long, device=device)


def _gather_avenue_axis(tensor: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    if tensor.shape[:3] != order.shape:
        raise ValueError(f"avenue gather shape mismatch: tensor={tuple(tensor.shape)} order={tuple(order.shape)}")
    index = order
    for _dim in tensor.shape[3:]:
        index = index.unsqueeze(-1)
    return tensor.gather(2, index.expand(-1, -1, -1, *tensor.shape[3:]))


def _apply_avenue_dropout(token_mask: torch.Tensor, drop_prob: float) -> torch.Tensor:
    if token_mask.dim() != 4:
        raise ValueError(f"avenue dropout expects [batch, roles, avenues, tokens], got {tuple(token_mask.shape)}")
    if drop_prob <= 0.0 or token_mask.shape[2] <= 1:
        return token_mask
    keep = torch.rand(token_mask.shape[:3], device=token_mask.device) >= float(drop_prob)
    fallback = torch.zeros_like(keep)
    fallback[:, :, 0] = True
    keep = torch.where(keep.any(dim=2, keepdim=True), keep, fallback)
    return token_mask.bool() & keep.unsqueeze(-1)


def _non_identity_permutation(n: int, rng: np.random.Generator) -> np.ndarray:
    perm = rng.permutation(n)
    if n > 1 and np.array_equal(perm, np.arange(n)):
        perm = np.roll(perm, 1)
    return perm.astype(np.int64, copy=False)


def _compatible_heads(hidden_dim: int, requested: int) -> int:
    requested = max(1, int(requested))
    if hidden_dim % requested == 0:
        return requested
    for heads in range(requested, 0, -1):
        if hidden_dim % heads == 0:
            return heads
    return 1


def _as_int_tuple(values: object) -> Tuple[int, ...]:
    if values is None:
        return (-1,)
    if isinstance(values, int):
        return (int(values),)
    if isinstance(values, (list, tuple)):
        out = tuple(int(value) for value in values)
        return out or (-1,)
    return (int(values),)


def _select_layer_tensors(
    layers: Sequence[torch.Tensor],
    selected_layer_ids: Sequence[int] | None,
) -> List[torch.Tensor]:
    if not layers:
        raise ValueError("at least one hidden-state layer is required for active message readout")
    selected = []
    for layer_id in _as_int_tuple(selected_layer_ids):
        index = int(layer_id)
        if index < 0:
            index = len(layers) + index
        index = min(max(index, 0), len(layers) - 1)
        selected.append(layers[index])
    return selected


def _make_dataclass(cls, data: object):
    values = dict(data or {})
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in values.items() if key in allowed})


def _replace_dataclass(instance, **updates):
    values = asdict(instance)
    values.update(updates)
    return type(instance)(**values)


def _make_mlp_training_config(data: object) -> MLPTrainingConfig:
    values = dict(data or {})
    hidden_dims = values.get("hidden_dims", (64,))
    return MLPTrainingConfig(
        epochs=int(values.get("epochs", 4)),
        batch_size=int(values.get("batch_size", 16)),
        lr=float(values.get("lr", 0.001)),
        weight_decay=float(values.get("weight_decay", 0.0001)),
        patience=int(values.get("patience", 3)),
        hidden_dims=tuple(int(value) for value in hidden_dims),
    )


def _resolve_device(requested: str) -> str:
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _split_offset(split: str) -> int:
    return {"train": 0, "dev": 10_000, "test": 20_000}.get(split, 30_000)


def _mean(values: Sequence[float]) -> float:
    return mean(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


if __name__ == "__main__":
    main()
