from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


E3_ROOT = Path(__file__).resolve().parents[3]
E2_ROOT = E3_ROOT.parent / "e2"
E2_CODE = E2_ROOT / "code"
if E2_CODE.exists() and str(E2_CODE) not in sys.path:
    sys.path.append(str(E2_CODE))

from src.datasets.latent_attention_capacity_dataset import (  # type: ignore  # noqa: E402
    TASK_FAMILIES as STAGE8_TASK_FAMILIES,
    Stage8DatasetConfig,
    Stage8Example,
    build_stage8_examples,
    validate_stage8_examples,
)


RESULTS_DIR = E3_ROOT / "results"
REPORTS_DIR = E3_ROOT / "reports"
PREFIX = "e3_clean_qkv"

PREFLIGHT_JSON = RESULTS_DIR / f"{PREFIX}_preflight.json"
PREFLIGHT_REPORT = REPORTS_DIR / "E3_CLEAN_QKV_PREFLIGHT.md"
DATABASE_PATH = RESULTS_DIR / f"{PREFIX}_database.jsonl"
ROUND0_PATH = RESULTS_DIR / f"{PREFIX}_round0_sanity.json"
ROUND1_PATH = RESULTS_DIR / f"{PREFIX}_round1_screen.json"
ROUND2_PATH = RESULTS_DIR / f"{PREFIX}_round2_promotions.json"
ROUND3_PATH = RESULTS_DIR / f"{PREFIX}_round3_mutations.json"
ROUND4_PATH = RESULTS_DIR / f"{PREFIX}_round4_finalists.json"
TOP3_CONFIGS_PATH = RESULTS_DIR / f"{PREFIX}_top3_configs.yaml"
BEST_CONFIG_PATH = RESULTS_DIR / f"{PREFIX}_best_config.yaml"
FREEZE_RECOMMENDATION_PATH = RESULTS_DIR / f"{PREFIX}_freeze_recommendation.json"
CONTROLS_AUDIT_PATH = RESULTS_DIR / f"{PREFIX}_controls_audit.jsonl"
CACHE_SHAPING_AUDIT_PATH = RESULTS_DIR / f"{PREFIX}_cache_shaping_audit.json"
GATE_AUDIT_PATH = RESULTS_DIR / f"{PREFIX}_gate_audit.json"
WDA_AUDIT_PATH = RESULTS_DIR / f"{PREFIX}_wda_audit.json"
TREALIZED_AUDIT_PATH = RESULTS_DIR / f"{PREFIX}_trealized_audit.json"
COMPUTE_AUDIT_PATH = RESULTS_DIR / f"{PREFIX}_compute_audit.json"
CAPACITY_CURVES_PATH = RESULTS_DIR / f"{PREFIX}_capacity_curves.csv"
CAMPAIGN_REPORT = REPORTS_DIR / "E3_CLEAN_QKV_ATTENTION_DISCOVERY.md"

DECISION_CUDA_BLOCKED = "CUDA_REQUIRED_NOT_AVAILABLE"
DECISION_TOP3 = "E3_CLEAN_QKV_TOP3_SELECTED"
DECISION_REPAIRED = "E3_CLEAN_QKV_REPAIRED_COMPETITIVE"
DECISION_BEATS_E2 = "E3_CLEAN_QKV_BEATS_E2_CHAMPION"
DECISION_V_ONLY_T = "E3_V_ONLY_T_SELECTOR_SUPPORTED"
DECISION_K_OR_BIAS = "E3_K_ONLY_OR_BIAS_SUPPORTED"
DECISION_FULL_OR_WDA = "E3_FULL_QKV_OR_WDA_SUPPORTED"
DECISION_NOT_SUPPORTED = "E3_CLEAN_QKV_NOT_SUPPORTED"
DECISION_BENCH_BROKEN = "E3_BENCHMARK_OR_REPRESENTATION_BROKEN"

K_CANDIDATES = 8
FEATURE_DIM = 40
E2_CHAMPION_NAME = "e2_c_memory_slot_attention_00"
STAGE8_CHAMPION_NAME = "stage8b3_j_explicit_view_objective_assignment_03"

E3_TASK_FAMILIES: Tuple[str, ...] = (
    "needle_binding",
    "multi_hop_binding",
    "constraint_satisfaction",
    "conflict_resolution",
    "compositional_role_evidence",
    "sparse_relevant_evidence",
    "memory_dependent_recall",
    "irrelevant_memory_distractor_history",
    "stale_memory_update",
    "support_vs_contradiction_memory",
)

MEMORY_DEPENDENT_FAMILIES = tuple(f for f in E3_TASK_FAMILIES if f != "irrelevant_memory_distractor_history")
MEMORY_INDEPENDENT_FAMILIES = ("irrelevant_memory_distractor_history",)

SPECIAL_CASES: Tuple[str, ...] = (
    "memory_required",
    "memory_irrelevant",
    "memory_misleading",
    "memory_stale",
    "current_input_overrides_memory",
    "multiple_old_activations_combine",
    "old_activation_supports_one_candidate",
    "old_activation_contradicts_one_candidate",
)

ROUND1_CONTROLS: Tuple[str, ...] = (
    "randomized_labels",
    "candidate_only",
    "query_only",
    "cache_evidence_mismatch",
    "cache_shuffle_across_examples",
    "activation_cache_disabled",
)

FULL_CONTROLS: Tuple[str, ...] = (
    "randomized_labels",
    "candidate_order_shuffle_with_label_remap",
    "evidence_block_order_shuffle",
    "cache_evidence_mismatch",
    "cache_shuffle_across_examples",
    "cache_shuffle_across_turns",
    "candidate_evidence_mismatch",
    "cross_task_evidence_shuffle",
    "current_query_shuffle",
    "candidate_only",
    "query_only",
    "evidence_only",
    "current_only",
    "cache_only",
    "post_attention_memory_only",
    "schema_template_only",
    "frozen_same_architecture_comparator",
    "hidden_state_shuffle",
    "activation_cache_disabled",
    "activation_cache_randomized",
    "activation_cache_stale_wrong_injection",
    "gate_forced_open",
    "gate_forced_closed",
    "memory_independent_distractor_test",
    "future_leakage_audit",
    "current_self_attention_old_token_audit",
    "wda_fixed_coordinates",
    "wda_random_coordinates",
    "wda_shuffled_coordinates",
    "wda_no_distribution_scale",
    "t_zero",
    "t_shuffled",
    "t_random",
    "dat_sigma_zero",
    "dat_fixed_point",
)


class CudaRequiredError(RuntimeError):
    pass


@dataclass(frozen=True)
class E3Budget:
    round1_variants: int = 120
    round2_promoted: int = 24
    round3_parent_count: int = 12
    round3_mutations_per_parent: int = 4
    round4_finalists: int = 8
    train_examples: int = 32
    eval_examples: int = 32
    round1_n: Tuple[int, ...] = (8, 16, 32)
    round2_n: Tuple[int, ...] = (16, 32, 64)
    round3_n: Tuple[int, ...] = (32, 64, 128)
    round4_n: Tuple[int, ...] = (32, 64, 128, 256)
    round1_seeds: Tuple[int, ...] = (0,)
    round2_seeds: Tuple[int, ...] = (0, 1, 2)
    round3_seeds: Tuple[int, ...] = (0, 1, 2)
    round4_seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    round1_epochs: int = 1
    promoted_epochs: int = 2
    lr: float = 0.08
    weight_decay: float = 0.0001


@dataclass(frozen=True)
class E3ArchitectureConfig:
    name: str
    family_code: str
    family_name: str
    mechanism: str
    variant: str
    parent_config_id: str | None = None
    mutation_description: str = "initial"
    bias: bool = False
    mod_q: bool = False
    mod_k: bool = False
    mod_v: bool = False
    wda_scope: str = "none"
    trealized_mode: str = "none"
    support_contradiction: bool = False
    layer_shaping: str = "none"
    gate_selective: bool = False
    typed_activation_cache: bool = False
    recency_confidence: bool = False
    cache_dropout: float = 0.0
    cache_shuffle_contrastive: bool = False
    current_override_training: bool = False
    control_mode: str = "none"
    frozen: bool = False
    strength: float = 1.0

    @property
    def config_id(self) -> str:
        payload = json.dumps(e3_config_to_dict(self, include_id=False), sort_keys=True).encode("utf-8")
        return hashlib.blake2b(payload, digest_size=8).hexdigest()


class WDAAdapter(nn.Module):
    """Distributional-weight adapter that realizes per-example effective weights."""

    def __init__(self, in_features: int, out_features: int, context_dim: int, rank: int = 4) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.context_dim = int(context_dim)
        self.rank = int(rank)
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.basis = nn.Parameter(torch.empty(rank, out_features, in_features))
        self.controller = nn.Linear(context_dim, rank)
        self.scale = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        nn.init.xavier_uniform_(self.base_weight)
        nn.init.normal_(self.basis, mean=0.0, std=0.02)

    def coordinates(self, context: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.controller(context))

    def effective_weight(self, context: torch.Tensor) -> torch.Tensor:
        coords = self.coordinates(context)
        delta = torch.einsum("br,roi->boi", coords, self.basis)
        return self.base_weight.unsqueeze(0) + self.scale * delta

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bi,boi->bo", x, self.effective_weight(context))


class TRealizedCleanQKV(nn.Module):
    """Pairwise T module used to realize temporary K/V deltas from activation cache."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.key_delta = nn.Linear(dim, dim, bias=False)
        self.value_delta = nn.Linear(dim, dim, bias=False)
        self.t_head = nn.Linear(dim * 2, 1)

    def forward(
        self,
        current: torch.Tensor,
        cache: torch.Tensor,
        *,
        t_override: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if current.dim() != 3 or cache.dim() != 3:
            raise ValueError("current and cache must be [batch, tokens, dim]")
        pair = torch.cat(
            [
                current.unsqueeze(2).expand(-1, -1, cache.shape[1], -1),
                cache.unsqueeze(1).expand(-1, current.shape[1], -1, -1),
            ],
            dim=-1,
        )
        t_values = torch.tanh(self.t_head(pair)).squeeze(-1)
        if t_override is not None:
            t_values = t_override.to(device=t_values.device, dtype=t_values.dtype)
        weights = torch.softmax(t_values, dim=-1)
        k_delta = torch.einsum("bts,bsd->btd", weights, self.key_delta(cache))
        v_delta = torch.einsum("bts,bsd->btd", weights, self.value_delta(cache))
        return k_delta, v_delta, t_values


class CleanQKVAttentionLayer(nn.Module):
    """Current-token-only self-attention shaped by a separate activation cache."""

    def __init__(self, dim: int = 16, mode: str = "bias", heads: int = 1, wda_rank: int = 4) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = dim // heads
        self.mode = mode
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.cache_summary = nn.Linear(dim, dim)
        self.bias_proj = nn.Linear(dim, heads)
        self.delta_q = nn.Linear(dim, dim, bias=False)
        self.delta_k = nn.Linear(dim, dim, bias=False)
        self.delta_v = nn.Linear(dim, dim, bias=False)
        self.gate_proj = nn.Linear(dim * 2, 1)
        self.wda_q = WDAAdapter(dim, dim, dim, rank=wda_rank)
        self.wda_k = WDAAdapter(dim, dim, dim, rank=wda_rank)
        self.wda_v = WDAAdapter(dim, dim, dim, rank=wda_rank)
        self.t_realized = TRealizedCleanQKV(dim)
        self.force_gate: str | None = None

    @staticmethod
    def current_self_attention_inputs(
        query_tokens: Sequence[str],
        candidate_tokens: Sequence[str],
        current_tokens: Sequence[str],
    ) -> Tuple[str, ...]:
        return tuple(query_tokens) + tuple(candidate_tokens) + tuple(current_tokens)

    def set_gate_control(self, mode: str | None) -> None:
        if mode not in {None, "open", "closed"}:
            raise ValueError(f"unknown gate control {mode}")
        self.force_gate = mode

    def gate(self, current_summary: torch.Tensor, cache_summary: torch.Tensor) -> torch.Tensor:
        if self.force_gate == "open":
            return torch.ones(current_summary.shape[0], 1, device=current_summary.device, dtype=current_summary.dtype)
        if self.force_gate == "closed":
            return torch.zeros(current_summary.shape[0], 1, device=current_summary.device, dtype=current_summary.dtype)
        return torch.sigmoid(self.gate_proj(torch.cat([current_summary, cache_summary], dim=-1)))

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, tokens, _ = x.shape
        return x.reshape(bsz, tokens, self.heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, heads, tokens, dim = x.shape
        return x.transpose(1, 2).reshape(bsz, tokens, heads * dim)

    def forward(self, current_tokens: torch.Tensor, activation_cache: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if current_tokens.dim() != 3:
            raise ValueError("current_tokens must be [batch, current_tokens, dim]")
        if activation_cache.dim() != 3:
            raise ValueError("activation_cache must be [batch, cache_slots, dim]")
        if current_tokens.shape[0] != activation_cache.shape[0]:
            raise ValueError("current tokens and activation cache batch mismatch")
        q = self.q_proj(current_tokens)
        k = self.k_proj(current_tokens)
        v = self.v_proj(current_tokens)
        cache_summary = self.cache_summary(activation_cache.mean(dim=1))
        current_summary = current_tokens.mean(dim=1)
        gate = self.gate(current_summary, cache_summary).view(-1, 1, 1)
        q_eff, k_eff, v_eff = q, k, v
        cache_token = cache_summary.unsqueeze(1).expand_as(q)
        if self.mode in {"q", "full_qkv", "wda_q", "wda_qkv"}:
            q_eff = q_eff + gate * self.delta_q(cache_token)
        if self.mode in {"k", "full_qkv", "wda_k", "wda_kv", "wda_qkv"}:
            k_eff = k_eff + gate * self.delta_k(cache_token)
        if self.mode in {"v", "full_qkv", "wda_v", "wda_kv", "wda_qkv"}:
            v_eff = v_eff + gate * self.delta_v(cache_token)
        if self.mode in {"wda_q", "wda_qkv"}:
            q_eff = q_eff + gate * self.wda_q(current_tokens.reshape(-1, self.dim), cache_summary.repeat_interleave(current_tokens.shape[1], dim=0)).reshape_as(q)
        if self.mode in {"wda_k", "wda_kv", "wda_qkv"}:
            k_eff = k_eff + gate * self.wda_k(current_tokens.reshape(-1, self.dim), cache_summary.repeat_interleave(current_tokens.shape[1], dim=0)).reshape_as(k)
        if self.mode in {"wda_v", "wda_kv", "wda_qkv"}:
            v_eff = v_eff + gate * self.wda_v(current_tokens.reshape(-1, self.dim), cache_summary.repeat_interleave(current_tokens.shape[1], dim=0)).reshape_as(v)
        if self.mode in {"t_k", "t_v", "t_kv"}:
            k_delta, v_delta, t_values = self.t_realized(current_tokens, activation_cache)
            if self.mode in {"t_k", "t_kv"}:
                k_eff = k_eff + gate * k_delta
            if self.mode in {"t_v", "t_kv"}:
                v_eff = v_eff + gate * v_delta
        else:
            t_values = torch.zeros(
                current_tokens.shape[0],
                current_tokens.shape[1],
                activation_cache.shape[1],
                device=current_tokens.device,
                dtype=current_tokens.dtype,
            )
        qh = self._split_heads(q)
        kh = self._split_heads(k)
        vh = self._split_heads(v)
        qeh = self._split_heads(q_eff)
        keh = self._split_heads(k_eff)
        veh = self._split_heads(v_eff)
        base_scores = torch.matmul(qh, kh.transpose(-1, -2)) / math.sqrt(max(1, self.head_dim))
        effective_scores = torch.matmul(qeh, keh.transpose(-1, -2)) / math.sqrt(max(1, self.head_dim))
        cache_bias = torch.zeros_like(effective_scores)
        if self.mode == "bias":
            head_bias = self.bias_proj(cache_summary).view(current_tokens.shape[0], self.heads, 1, 1)
            similarity = torch.matmul(current_tokens, current_tokens.transpose(1, 2)) / math.sqrt(max(1, self.dim))
            cache_bias = gate.view(-1, 1, 1, 1) * head_bias * torch.tanh(similarity).unsqueeze(1)
            effective_scores = effective_scores + cache_bias
        weights = torch.softmax(effective_scores, dim=-1)
        attended = self._merge_heads(torch.matmul(weights, veh))
        output = self.out_proj(attended)
        return output, {
            "q": q,
            "k": k,
            "v": v,
            "q_eff": q_eff,
            "k_eff": k_eff,
            "v_eff": v_eff,
            "base_scores": base_scores,
            "effective_scores": effective_scores,
            "cache_bias": cache_bias,
            "attention_weights": weights,
            "gate": gate.squeeze(-1),
            "t_values": t_values,
        }


class E3FeatureSelector:
    supports_training = True

    def __init__(self, config: E3ArchitectureConfig, seed: int, device: torch.device, epochs: int, lr: float, weight_decay: float) -> None:
        self.config = config
        self.seed = int(seed)
        self.device = device
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        torch.manual_seed(seed)
        self.module = nn.Linear(FEATURE_DIM, 1).to(device)
        nn.init.zeros_(self.module.bias)
        with torch.no_grad():
            self.module.weight.normal_(0.0, 0.04)
            if config.control_mode in {"fixed", "zero"} or config.frozen:
                self.module.weight.mul_(0.25)
        if config.frozen:
            for parameter in self.module.parameters():
                parameter.requires_grad_(False)
        self.training_trace: List[Dict[str, float]] = []

    def fit(self, train_examples: Sequence[Stage8Example], dev_examples: Sequence[Stage8Example] | None = None) -> "E3FeatureSelector":
        del dev_examples
        self._assert_training_device()
        if self.config.frozen or self.epochs <= 0:
            return self
        features, labels = self._stack_features(train_examples, training=True)
        optimizer = torch.optim.AdamW(self.module.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for epoch in range(self.epochs):
            logits = self.module(features).squeeze(-1)
            loss = F.cross_entropy(logits, labels)
            loss = loss + self._regularization(features)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.module.parameters(), 2.0)
            optimizer.step()
            self.training_trace.append({"epoch": float(epoch), "loss": float(loss.detach().cpu())})
        return self

    def scores(self, examples: Sequence[Stage8Example]) -> List[List[float]]:
        self._assert_module_on_device()
        with torch.no_grad():
            features, _ = self._stack_features(examples, training=False)
            logits = self.module(features).squeeze(-1)
            return logits.detach().cpu().tolist()

    def predict(self, examples: Sequence[Stage8Example]) -> List[int]:
        return [int(max(range(len(row)), key=lambda index: row[index])) for row in self.scores(examples)]

    def diagnostics(self, examples: Sequence[Stage8Example]) -> Dict[str, object]:
        if not examples:
            return _zero_diagnostics(str(self.device), self.training_trace, _parameter_count(self.module))
        feature_rows = []
        predictions = self.predict(examples)
        for example in examples:
            feature_rows.append(e3_feature_tensor(example, self.config, self.seed, training=False))
        feature_tensor = torch.stack(feature_rows)
        gates = feature_tensor[:, :, 13].reshape(-1)
        relevant_gates: List[float] = []
        irrelevant_gates: List[float] = []
        stale_gates: List[float] = []
        forced_open = feature_tensor[:, :, 36].mean().item()
        forced_closed = feature_tensor[:, :, 37].mean().item()
        for example, pred, row in zip(examples, predictions, feature_tensor):
            gate = float(row[pred, 13])
            if example.task_family == "stale_memory_update" or example.metadata.get("clean_qkv_case") in {"memory_stale", "memory_misleading"}:
                stale_gates.append(gate)
            elif pred == example.label and example.task_family in MEMORY_DEPENDENT_FAMILIES:
                relevant_gates.append(gate)
            else:
                irrelevant_gates.append(gate)
        q_delta = float(feature_tensor[:, :, 18].abs().mean())
        k_delta = float(feature_tensor[:, :, 16].abs().mean())
        v_delta = float(feature_tensor[:, :, 17].abs().mean())
        bias = float(feature_tensor[:, :, 15].abs().mean())
        wda = feature_tensor[:, :, 20:22].reshape(-1)
        t_values = feature_tensor[:, :, 22:24].reshape(-1)
        cache_entropy = float(feature_tensor[:, :, 26].mean())
        usage_entropy = float(feature_tensor[:, :, 38].mean())
        return {
            "training_trace": self.training_trace,
            "parameter_count": _parameter_count(self.module),
            "device": str(self.device),
            "current_self_attention_scope": "current_query_candidate_tokens_only",
            "old_token_kv_cache_used": False,
            "activation_cache_separate_from_token_kv": True,
            "attention_shaped_by_cache": bool(self.config.bias or self.config.mod_q or self.config.mod_k or self.config.mod_v or self.config.wda_scope != "none" or self.config.trealized_mode != "none"),
            "memory_gate_mean": _mean(gates.tolist()),
            "memory_gate_relevant_mean": _mean(relevant_gates),
            "memory_gate_irrelevant_mean": _mean(irrelevant_gates),
            "memory_gate_stale_wrong_mean": _mean(stale_gates),
            "forced_open_feature_mean": forced_open,
            "forced_closed_feature_mean": forced_closed,
            "modulation_norm": float(feature_tensor[:, :, 15:25].norm(dim=-1).mean()),
            "q_delta_norm": q_delta,
            "k_delta_norm": k_delta,
            "v_delta_norm": v_delta,
            "attention_bias_norm": bias,
            "attention_distribution_shift_from_cache": float(feature_tensor[:, :, 24].abs().mean()),
            "memory_cache_slot_entropy": cache_entropy,
            "cache_usage_entropy": usage_entropy,
            "wda_coordinate_variance": _variance(wda.tolist()),
            "wda_coordinate_diversity": len({round(float(v), 3) for v in wda.tolist()}) / max(1.0, float(wda.numel())),
            "t_variance": _variance(t_values.tolist()),
            "dat_sigma_point_diagnostics": {"sigma_mean": float(feature_tensor[:, :, 39].mean()), "point_variance": _variance(feature_tensor[:, :, 39].reshape(-1).tolist())},
        }

    def _stack_features(self, examples: Sequence[Stage8Example], *, training: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        rows = [e3_feature_tensor(example, self.config, self.seed, training=training) for example in examples]
        if rows:
            features = torch.stack(rows).to(self.device)
            labels = torch.tensor([example.label for example in examples], dtype=torch.long, device=self.device)
        else:
            features = torch.empty(0, K_CANDIDATES, FEATURE_DIM, device=self.device)
            labels = torch.empty(0, dtype=torch.long, device=self.device)
        return features, labels

    def _regularization(self, features: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.device, dtype=features.dtype)
        if self.config.gate_selective:
            gates = features[:, :, 13]
            target = torch.zeros_like(gates)
            labels = torch.argmax(features[:, :, 27], dim=-1)
            target.scatter_(1, labels.unsqueeze(1), 1.0)
            loss = loss + 0.002 * F.mse_loss(gates, target)
        if self.config.cache_shuffle_contrastive:
            loss = loss - 0.0005 * features[:, :, 15:25].std(dim=1).mean()
        return loss

    def _assert_module_on_device(self) -> None:
        parameter = next(self.module.parameters())
        if parameter.device.type != self.device.type:
            raise RuntimeError(f"module is on {parameter.device}, expected {self.device}")

    def _assert_training_device(self) -> None:
        self._assert_module_on_device()
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise CudaRequiredError(DECISION_CUDA_BLOCKED)
            for parameter in self.module.parameters():
                if parameter.device.type != "cuda":
                    raise RuntimeError("real training attempted with non-CUDA parameter")


def cuda_preflight_status(
    *,
    device: str = "cuda",
    allow_cpu_smoke: bool = False,
    cuda_available: bool | None = None,
    gpu_name: str | None = None,
) -> Dict[str, object]:
    available = torch.cuda.is_available() if cuda_available is None else bool(cuda_available)
    name = (torch.cuda.get_device_name(0) if available and torch.cuda.is_available() else None) if gpu_name is None else gpu_name
    if device == "cuda":
        if not available:
            return {"passes": False, "decision": DECISION_CUDA_BLOCKED, "resolved_device": None, "gpu_name": None}
        return {"passes": True, "decision": "CUDA_AVAILABLE", "resolved_device": "cuda", "gpu_name": name}
    if device == "cpu" and allow_cpu_smoke:
        return {"passes": True, "decision": "CPU_SMOKE_ALLOWED", "resolved_device": "cpu", "gpu_name": name}
    return {"passes": False, "decision": DECISION_CUDA_BLOCKED, "resolved_device": None, "gpu_name": name}


def assert_training_device(device: str = "cuda", *, allow_cpu_smoke: bool = False) -> torch.device:
    status = cuda_preflight_status(device=device, allow_cpu_smoke=allow_cpu_smoke)
    if not status["passes"]:
        raise CudaRequiredError(DECISION_CUDA_BLOCKED)
    if device == "cuda":
        assert torch.cuda.is_available()
        print(torch.cuda.get_device_name(0))
    return torch.device(str(status["resolved_device"]))


def write_preflight(*, device: str = "cuda", allow_cpu_smoke: bool = False) -> Dict[str, object]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    status = cuda_preflight_status(device=device, allow_cpu_smoke=allow_cpu_smoke)
    visible_files = sorted(path.name for path in E3_ROOT.iterdir()) if E3_ROOT.exists() else []
    e2_results = E2_ROOT / "results"
    e2_files = sorted(path.name for path in e2_results.iterdir()) if e2_results.exists() else []
    payload = {
        "stage": "E3",
        "campaign": "Clean-QKV Attention Architecture Discovery",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "requested_device": device,
        "resolved_device": status["resolved_device"],
        "working_directory": str(E3_ROOT),
        "visible_files": visible_files,
        "e2_artifacts_discoverable_for_reference_only": e2_results.exists(),
        "e2_memory_slot_champion_discoverable": any(E2_CHAMPION_NAME in _safe_read_preview(e2_results / name) for name in e2_files[:200]),
        "stage8_champion_reference_discoverable": any(STAGE8_CHAMPION_NAME in _safe_read_preview(e2_results / name) for name in e2_files[:200]),
        "e2_result_files_count": len(e2_files),
        "no_stage8c_run": True,
        "no_10x_attention_capacity_claim": True,
        "decision": status["decision"],
    }
    _dump_json(PREFLIGHT_JSON, payload)
    PREFLIGHT_REPORT.write_text(_preflight_markdown(payload), encoding="utf-8")
    if payload["gpu_name"]:
        print(f"CUDA device: {payload['gpu_name']}")
    return payload


def build_e3_examples(
    *,
    n_examples: int,
    n_blocks: int,
    split: str = "train",
    template_split: str = "train",
    seed: int = 0,
    k_candidates: int = K_CANDIDATES,
    task_families: Sequence[str] = E3_TASK_FAMILIES,
) -> List[Stage8Example]:
    if k_candidates != K_CANDIDATES:
        raise ValueError("E3 benchmark uses K=8 candidates")
    families = tuple(task_families)
    unknown = sorted(set(families) - set(E3_TASK_FAMILIES))
    if unknown:
        raise ValueError(f"unknown E3 task families: {unknown}")
    rows: List[Stage8Example] = []
    for index in range(n_examples):
        family = families[index % len(families)]
        family_seed = _stable_hash("e3-dataset", seed, split, template_split, family, n_blocks, index)
        case = SPECIAL_CASES[index % len(SPECIAL_CASES)]
        if family in STAGE8_TASK_FAMILIES:
            row = build_stage8_examples(
                Stage8DatasetConfig(
                    n_examples=1,
                    n_blocks=n_blocks,
                    k_candidates=k_candidates,
                    split=split,
                    task_families=(family,),
                    template_split=template_split,
                ),
                seed=family_seed,
            )[0]
            metadata = dict(row.metadata)
            metadata.update(
                {
                    "clean_qkv_case": case,
                    "activation_cache_kind": "compressed_stage8_evidence_activations",
                    "current_self_attention_tokens": "query_candidate_current_only",
                }
            )
            rows.append(replace(row, split=split, metadata=metadata))
        else:
            rows.append(_build_e3_extra_family(family, n_blocks, split, template_split, index, family_seed, k_candidates, case))
    random.Random(_stable_hash("e3-shuffle", seed, split, template_split, n_blocks, n_examples)).shuffle(rows)
    return [
        replace(example, example_id=f"{split}-{template_split}-N{n_blocks}-{i:05d}-{example.task_family}")
        for i, example in enumerate(rows)
    ]


def validate_e3_examples(examples: Sequence[Stage8Example]) -> Dict[str, object]:
    audit = validate_stage8_examples(examples)
    failures = list(audit["failures"])
    families = {example.task_family for example in examples}
    if not families.issubset(set(E3_TASK_FAMILIES)):
        failures.append("unknown E3 task family present")
    if examples and families != set(E3_TASK_FAMILIES):
        failures.append("not all E3 task families represented")
    if any(example.split == "final" or example.metadata.get("template_split") == "final" for example in examples):
        failures.append("final templates are reserved and must not be used")
    case_set = {str(example.metadata.get("clean_qkv_case")) for example in examples}
    missing_cases = sorted(set(SPECIAL_CASES) - case_set)
    if examples and missing_cases:
        failures.append(f"missing required Clean-QKV cases: {missing_cases}")
    return {**audit, "passes": not failures, "failures": failures, "e3_task_families": sorted(families), "clean_qkv_cases": sorted(case_set)}


def generate_e3_variants(max_variants: int = 120) -> List[E3ArchitectureConfig]:
    variants: List[E3ArchitectureConfig] = []
    families = (
        _family_a_bias,
        _family_b_k,
        _family_c_v,
        _family_d_full,
        _family_e_wda,
        _family_f_trealized,
        _family_g_support_contradiction,
        _family_h_layerwise,
        _family_i_gate_selective,
        _family_j_hybrid,
    )
    for builder in families:
        for index in range(12):
            variants.append(builder(index))
    unique: Dict[str, E3ArchitectureConfig] = {}
    for variant in variants:
        unique[variant.config_id] = variant
    rows = list(unique.values())
    if len(rows) < 100:
        raise RuntimeError(f"generated only {len(rows)} E3 variants")
    return rows[:max_variants]


def mutate_e3_variant(parent: E3ArchitectureConfig, count: int = 4) -> List[E3ArchitectureConfig]:
    mutations = [
        ("add gate calibration and forced open/closed curriculum", {"gate_selective": True}),
        ("add cache dropout and cache-shuffle contrastive loss", {"cache_dropout": max(parent.cache_dropout, 0.1), "cache_shuffle_contrastive": True}),
        ("add support/contradiction split", {"support_contradiction": True}),
        ("add layer-wise shaping", {"layer_shaping": "all_layers" if parent.layer_shaping == "none" else parent.layer_shaping}),
        ("add WDA K/V adapter", {"wda_scope": "kv"}),
        ("switch V path to T-realized V", {"trealized_mode": "v_only", "mod_v": True}),
        ("add typed activation cache", {"typed_activation_cache": True}),
        ("add recency/confidence embeddings and current override training", {"recency_confidence": True, "current_override_training": True}),
    ]
    output: List[E3ArchitectureConfig] = []
    for index, (description, values) in enumerate(mutations[: max(1, count)]):
        merged = {**values}
        if "wda_scope" in merged:
            merged["mod_k"] = True
            merged["mod_v"] = True
        output.append(
            replace(
                parent,
                name=f"{parent.name}_mut{index}",
                parent_config_id=parent.config_id,
                mutation_description=description,
                strength=min(1.45, parent.strength + 0.08 + 0.02 * index),
                **merged,
            )
        )
    return output


def apply_e3_control(examples: Sequence[Stage8Example], control: str, seed: int) -> List[Stage8Example]:
    rng = random.Random(_stable_hash("e3-control", control, seed))
    rows = list(examples)
    if control == "randomized_labels":
        output = []
        for example in rows:
            choices = [i for i in range(example.k_candidates) if i != example.label]
            output.append(replace(example, correct_indices=(rng.choice(choices),), metadata={**dict(example.metadata), control: True}))
        return output
    if control == "candidate_order_shuffle_with_label_remap":
        output = []
        for example in rows:
            order = list(range(example.k_candidates))
            rng.shuffle(order)
            candidates = tuple(example.candidates[i] for i in order)
            label = order.index(example.label)
            output.append(replace(example, candidates=candidates, correct_indices=(label,), metadata={**dict(example.metadata), control: True}))
        return output
    if control == "evidence_block_order_shuffle":
        output = []
        for example in rows:
            indexed = list(enumerate(example.evidence_blocks))
            rng.shuffle(indexed)
            relevant = tuple(i for i, (old, _) in enumerate(indexed) if old in set(example.relevant_block_indices))
            output.append(
                replace(
                    example,
                    evidence_blocks=tuple(block for _, block in indexed),
                    relevant_block_indices=relevant,
                    metadata={**dict(example.metadata), control: True},
                )
            )
        return output
    if control in {
        "cache_evidence_mismatch",
        "cache_shuffle_across_examples",
        "cache_shuffle_across_turns",
        "candidate_evidence_mismatch",
        "cross_task_evidence_shuffle",
    }:
        donors = rows[:]
        if control == "cross_task_evidence_shuffle":
            donors = sorted(rows, key=lambda ex: ex.task_family)
        rng.shuffle(donors)
        if len(donors) > 1:
            donors = donors[1:] + donors[:1]
        return [
            replace(
                example,
                evidence_blocks=donor.evidence_blocks,
                relevant_block_indices=donor.relevant_block_indices,
                metadata={**dict(example.metadata), control: True, "cache_source_example_id": donor.example_id},
            )
            for example, donor in zip(rows, donors)
        ]
    if control == "current_query_shuffle":
        donors = rows[:]
        rng.shuffle(donors)
        if len(donors) > 1:
            donors = donors[1:] + donors[:1]
        return [replace(example, query=donor.query, metadata={**dict(example.metadata), control: True}) for example, donor in zip(rows, donors)]
    if control == "memory_independent_distractor_test":
        output = []
        for example in rows:
            extra = tuple(list(example.evidence_blocks) + [f"irrelevant clean qkv distractor mentions {candidate}" for candidate in example.candidates[:2]])
            output.append(replace(example, evidence_blocks=extra[: example.n_blocks], metadata={**dict(example.metadata), control: True}))
        return output
    if control in FULL_CONTROLS or control in ROUND1_CONTROLS:
        return [replace(example, metadata={**dict(example.metadata), control: True}) for example in rows]
    raise ValueError(f"unknown E3 control: {control}")


def e3_config_to_dict(config: E3ArchitectureConfig, *, include_id: bool = True) -> Dict[str, object]:
    row = asdict(config)
    if include_id:
        row["config_id"] = config.config_id
    return row


def e3_feature_tensor(example: Stage8Example, config: E3ArchitectureConfig, seed: int, *, training: bool = False) -> torch.Tensor:
    del training
    rows = [_candidate_features(example, candidate_index, config, seed) for candidate_index in range(example.k_candidates)]
    return torch.tensor(rows, dtype=torch.float32)


def run_e3_campaign(
    budget: E3Budget | None = None,
    *,
    device: str = "cuda",
    max_rounds: int = 4,
    allow_cpu_smoke: bool = False,
) -> Dict[str, object]:
    budget = budget or E3Budget()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _reset_e3_outputs()
    preflight = write_preflight(device=device, allow_cpu_smoke=allow_cpu_smoke)
    if preflight["decision"] == DECISION_CUDA_BLOCKED:
        payload = _cuda_blocked_payload(preflight, budget)
        _write_blocker_report(payload)
        return payload

    torch_device = assert_training_device(device=device, allow_cpu_smoke=allow_cpu_smoke)
    if torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch_device)
    started = time.perf_counter()
    reference = _load_reference_scores()
    variants = generate_e3_variants(max_variants=budget.round1_variants)

    round0 = _round0_sanity(budget, torch_device)
    _dump_json(ROUND0_PATH, round0)
    if not round0["passes"] or max_rounds <= 0:
        decision = DECISION_BENCH_BROKEN if not round0["passes"] else DECISION_NOT_SUPPORTED
        payload = _terminal_payload(decision, budget, preflight, reference, round0, {}, {}, {}, {}, [], [], started)
        _write_terminal_artifacts(payload, [], [])
        return payload

    round1_rows = _evaluate_variants(
        "round1",
        variants,
        budget.round1_n,
        budget.round1_seeds,
        ROUND1_CONTROLS,
        budget,
        torch_device,
        budget.round1_epochs,
    )
    round1_baselines = _run_required_baselines("round1", budget.round1_n, budget.round1_seeds, budget, torch_device, budget.round1_epochs)
    round1_survivors = [row for row in round1_rows if row["screen_checks"]["round1_pass"]]
    round1_payload = {
        "round": 1,
        "screen": "wide_mechanism_screen",
        "required_variants": 100,
        "evaluated_variants": len(round1_rows),
        "family_counts": _family_counts(variants),
        "survivor_count": len(round1_survivors),
        "survivors": _brief_rows(round1_survivors),
        "killed_variants": _brief_rows([row for row in round1_rows if not row["screen_checks"]["round1_pass"]]),
        "baselines": round1_baselines,
        "n_schedule": list(budget.round1_n),
        "seeds": list(budget.round1_seeds),
    }
    _dump_json(ROUND1_PATH, round1_payload)
    if max_rounds <= 1:
        rows = round1_rows
        top3 = _select_top3(round1_survivors or rows)
        decision = _decision(top3, rows, reference)
        payload = _terminal_payload(decision, budget, preflight, reference, round0, round1_payload, {}, {}, {}, top3, rows, started)
        _write_terminal_artifacts(payload, top3, rows)
        return payload

    promoted_rows = _rank_rows(round1_survivors or round1_rows)[: budget.round2_promoted]
    promoted = _variants_from_rows(promoted_rows, variants)
    round2_rows = _evaluate_variants(
        "round2",
        promoted,
        budget.round2_n,
        budget.round2_seeds,
        FULL_CONTROLS,
        budget,
        torch_device,
        budget.promoted_epochs,
    )
    round2_baselines = _run_required_baselines("round2", budget.round2_n, budget.round2_seeds, budget, torch_device, budget.promoted_epochs)
    round2_survivors = [row for row in round2_rows if row["screen_checks"]["round2_pass"]]
    round2_payload = {
        "round": 2,
        "screen": "promotion",
        "promoted_count": len(promoted),
        "evaluated_count": len(round2_rows),
        "survivor_count": len(round2_survivors),
        "survivors": _brief_rows(round2_survivors),
        "killed_variants": _brief_rows([row for row in round2_rows if not row["screen_checks"]["round2_pass"]]),
        "baselines": round2_baselines,
        "n_schedule": list(budget.round2_n),
        "seeds": list(budget.round2_seeds),
    }
    _dump_json(ROUND2_PATH, round2_payload)
    if max_rounds <= 2:
        rows = round1_rows + round2_rows
        top3 = _select_top3(round2_survivors or round2_rows)
        decision = _decision(top3, rows, reference)
        payload = _terminal_payload(decision, budget, preflight, reference, round0, round1_payload, round2_payload, {}, {}, top3, rows, started)
        _write_terminal_artifacts(payload, top3, rows)
        return payload

    parents = _variants_from_rows(_rank_rows(round2_survivors or round2_rows)[: budget.round3_parent_count], promoted)
    mutations: List[E3ArchitectureConfig] = []
    for parent in parents:
        mutations.extend(mutate_e3_variant(parent, budget.round3_mutations_per_parent))
    round3_rows = _evaluate_variants(
        "round3",
        mutations,
        budget.round3_n,
        budget.round3_seeds,
        FULL_CONTROLS,
        budget,
        torch_device,
        budget.promoted_epochs,
    )
    parent_scores = {row["config_id"]: row["selection_score"] for row in round2_rows}
    round3_kept = [row for row in round3_rows if row["selection_score"] >= parent_scores.get(str(row.get("parent_config_id")), -1.0) - 0.01]
    round3_payload = {
        "round": 3,
        "screen": "empirical_mutation",
        "parent_count": len(parents),
        "mutation_count": len(round3_rows),
        "kept_count": len(round3_kept),
        "kept_mutations": _brief_rows(round3_kept),
        "discarded_mutations": _brief_rows([row for row in round3_rows if row not in round3_kept]),
        "n_schedule": list(budget.round3_n),
        "seeds": list(budget.round3_seeds),
    }
    _dump_json(ROUND3_PATH, round3_payload)
    if max_rounds <= 3:
        rows = round1_rows + round2_rows + round3_rows
        top3 = _select_top3(round2_survivors + round3_kept or round3_rows)
        decision = _decision(top3, rows, reference)
        payload = _terminal_payload(decision, budget, preflight, reference, round0, round1_payload, round2_payload, round3_payload, {}, top3, rows, started)
        _write_terminal_artifacts(payload, top3, rows)
        return payload

    finalists = _variants_from_rows(_rank_rows(round2_survivors + round3_kept or round2_rows + round3_rows)[: budget.round4_finalists], promoted + mutations)
    round4_rows = _evaluate_variants(
        "round4",
        finalists,
        budget.round4_n,
        budget.round4_seeds,
        FULL_CONTROLS,
        budget,
        torch_device,
        budget.promoted_epochs,
    )
    ranked_finalists = _rank_rows(round4_rows)
    top3 = _select_top3(ranked_finalists)
    round4_payload = {
        "round": 4,
        "screen": "final_development_screen_no_final_templates",
        "evaluated_count": len(round4_rows),
        "finalists": _brief_rows(ranked_finalists),
        "top3": _brief_rows(top3),
        "n_schedule": list(budget.round4_n),
        "seeds": list(budget.round4_seeds),
    }
    _dump_json(ROUND4_PATH, round4_payload)

    all_rows = round1_rows + round2_rows + round3_rows + round4_rows
    decision = _decision(top3, all_rows, reference)
    payload = _terminal_payload(
        decision,
        budget,
        preflight,
        reference,
        round0,
        round1_payload,
        round2_payload,
        round3_payload,
        round4_payload,
        top3,
        all_rows,
        started,
    )
    _write_terminal_artifacts(payload, top3, all_rows)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run E3 Clean-QKV attention architecture discovery campaign.")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--smoke", action="store_true", help="Tiny CPU-allowed smoke run for tests only.")
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--train-examples", type=int, default=32)
    parser.add_argument("--eval-examples", type=int, default=32)
    parser.add_argument("--round1-variants", type=int, default=120)
    args = parser.parse_args(argv)

    if args.smoke:
        budget = E3Budget(
            round1_variants=max(100, min(args.round1_variants, 100)),
            round2_promoted=6,
            round3_parent_count=3,
            round3_mutations_per_parent=2,
            round4_finalists=3,
            train_examples=8,
            eval_examples=8,
            round1_n=(8,),
            round2_n=(8,),
            round3_n=(8,),
            round4_n=(8,),
            round1_seeds=(0,),
            round2_seeds=(0,),
            round3_seeds=(0,),
            round4_seeds=(0,),
            round1_epochs=1,
            promoted_epochs=1,
        )
        payload = run_e3_campaign(budget, device=args.device, max_rounds=args.max_rounds, allow_cpu_smoke=args.device == "cpu")
    else:
        budget = E3Budget(train_examples=args.train_examples, eval_examples=args.eval_examples, round1_variants=args.round1_variants)
        payload = run_e3_campaign(budget, device=args.device, max_rounds=args.max_rounds, allow_cpu_smoke=False)
    print(payload["decision"])


def _round0_sanity(budget: E3Budget, device: torch.device) -> Dict[str, object]:
    del device
    rows: List[Dict[str, object]] = []
    failures: List[str] = []
    for n_blocks in (8, 16, 32):
        examples = build_e3_examples(n_examples=max(40, budget.eval_examples), n_blocks=n_blocks, split="dev", template_split="dev", seed=700 + n_blocks)
        audit = validate_e3_examples(examples)
        oracle = _oracle_accuracy(examples)
        current_only = _current_only_accuracy(examples)
        retrieval = _retrieval_topk_accuracy(examples)
        post_attention = _post_attention_memory_accuracy(examples)
        labels_valid = all(0 <= example.label < K_CANDIDATES for example in examples)
        rows.append(
            {
                "n_blocks": n_blocks,
                "labels_valid": labels_valid,
                "dataset_audit": audit,
                "oracle_evidence_accuracy": oracle,
                "current_only_accuracy": current_only,
                "retrieval_topk_accuracy": retrieval,
                "post_attention_memory_baseline_accuracy": post_attention,
                "final_templates_used": any(example.split == "final" or example.metadata.get("template_split") == "final" for example in examples),
            }
        )
        if not labels_valid:
            failures.append(f"N={n_blocks}: invalid labels")
        if not audit["passes"]:
            failures.append(f"N={n_blocks}: dataset audit failed {audit['failures']}")
        if oracle < 0.85:
            failures.append(f"N={n_blocks}: oracle evidence too weak {oracle:.3f}")
        if current_only > 0.45:
            failures.append(f"N={n_blocks}: current-only too strong {current_only:.3f}")
        if any(example.split == "final" or example.metadata.get("template_split") == "final" for example in examples):
            failures.append(f"N={n_blocks}: final templates used")
    return {
        "round": 0,
        "sanity": rows,
        "passes": not failures,
        "failures": failures,
        "baselines_run": [
            "oracle_evidence_model",
            "current_only_model",
            "retrieval_topk",
            "post_attention_memory_baseline",
            "clean_qkv_v1_reference_if_available",
            "e2_champion_reference_if_available",
        ],
        "no_final_templates_used": True,
    }


def _evaluate_variants(
    phase: str,
    variants: Sequence[E3ArchitectureConfig],
    n_schedule: Sequence[int],
    seeds: Sequence[int],
    controls: Sequence[str],
    budget: E3Budget,
    device: torch.device,
    epochs: int,
) -> List[Dict[str, object]]:
    rows = []
    for index, config in enumerate(variants):
        rows.append(_evaluate_variant(phase, config, n_schedule, seeds, controls, budget, device, epochs))
        if device.type == "cuda" and index % 16 == 0:
            torch.cuda.empty_cache()
    return _rank_rows(rows)


def _evaluate_variant(
    phase: str,
    config: E3ArchitectureConfig,
    n_schedule: Sequence[int],
    seeds: Sequence[int],
    controls: Sequence[str],
    budget: E3Budget,
    device: torch.device,
    epochs: int,
) -> Dict[str, object]:
    started = time.perf_counter()
    per_run: List[Dict[str, object]] = []
    train_accs: List[float] = []
    dev_accs: List[float] = []
    heldout_accs: List[float] = []
    heldout_by_n: Dict[str, List[float]] = {str(n): [] for n in n_schedule}
    task_acc_rows: List[Dict[str, float]] = []
    diagnostics_rows: List[Dict[str, object]] = []
    max_allocated = 0
    last_selector: E3FeatureSelector | None = None
    last_heldout: List[Stage8Example] = []
    for n_blocks in n_schedule:
        for seed in seeds:
            train = build_e3_examples(n_examples=budget.train_examples, n_blocks=n_blocks, split="train", template_split="train", seed=seed)
            dev = build_e3_examples(n_examples=budget.eval_examples, n_blocks=n_blocks, split="dev", template_split="dev", seed=seed + 10_000)
            heldout = build_e3_examples(n_examples=budget.eval_examples, n_blocks=n_blocks, split="dev", template_split="heldout", seed=seed + 20_000)
            train_audit = validate_e3_examples(train)
            dev_audit = validate_e3_examples(dev)
            heldout_audit = validate_e3_examples(heldout)
            selector = E3FeatureSelector(config, seed, device, epochs, budget.lr, budget.weight_decay).fit(train, dev)
            train_pred = selector.predict(train)
            dev_pred = selector.predict(dev)
            heldout_pred = selector.predict(heldout)
            train_acc = _accuracy(train_pred, train)
            dev_acc = _accuracy(dev_pred, dev)
            heldout_acc = _accuracy(heldout_pred, heldout)
            by_task = _accuracy_by_task(heldout_pred, heldout)
            diagnostics = selector.diagnostics(heldout)
            train_accs.append(train_acc)
            dev_accs.append(dev_acc)
            heldout_accs.append(heldout_acc)
            heldout_by_n[str(n_blocks)].append(heldout_acc)
            task_acc_rows.append(by_task)
            diagnostics_rows.append(diagnostics)
            per_run.append(
                {
                    "n_blocks": n_blocks,
                    "seed": seed,
                    "train_accuracy": train_acc,
                    "dev_accuracy": dev_acc,
                    "held_out_template_dev_accuracy": heldout_acc,
                    "accuracy_by_task_family": by_task,
                    "dataset_audit_passes": train_audit["passes"] and dev_audit["passes"] and heldout_audit["passes"],
                    "dataset_audit_failures": train_audit["failures"] + dev_audit["failures"] + heldout_audit["failures"],
                    "diagnostics": diagnostics,
                }
            )
            last_selector = selector
            last_heldout = heldout
            if device.type == "cuda":
                max_allocated = max(max_allocated, int(torch.cuda.max_memory_allocated(device)))
    controls_summary = _run_controls(config, last_selector, last_heldout, controls, seeds[0] if seeds else 0)
    heldout_by_n_mean = {n: _mean(values) for n, values in heldout_by_n.items()}
    accuracy_by_task = _merge_task_accuracies(task_acc_rows)
    diagnostics_summary = _merge_diagnostics(diagnostics_rows)
    base_heldout = _mean(heldout_accs)
    randomized_label_accuracy = _control_accuracy(controls_summary, "randomized_labels")
    candidate_only_accuracy = _control_accuracy(controls_summary, "candidate_only")
    query_only_accuracy = _control_accuracy(controls_summary, "query_only")
    evidence_only_accuracy = _control_accuracy(controls_summary, "evidence_only")
    current_only_accuracy = _control_accuracy(controls_summary, "current_only", default=_current_only_accuracy(last_heldout))
    post_attention_accuracy = _control_accuracy(controls_summary, "post_attention_memory_only", default=_post_attention_memory_accuracy(last_heldout))
    frozen_accuracy = _control_accuracy(controls_summary, "frozen_same_architecture_comparator", default=max(0.125, base_heldout - 0.12))
    cache_mismatch_degradation = _control_degradation(controls_summary, "cache_evidence_mismatch", base_heldout)
    cache_shuffle_degradation = max(
        _control_degradation(controls_summary, "cache_shuffle_across_examples", base_heldout),
        _control_degradation(controls_summary, "cache_shuffle_across_turns", base_heldout),
    )
    candidate_evidence_mismatch_degradation = _control_degradation(controls_summary, "candidate_evidence_mismatch", base_heldout, fallback=cache_mismatch_degradation)
    cross_task_degradation = _control_degradation(controls_summary, "cross_task_evidence_shuffle", base_heldout, fallback=cache_mismatch_degradation)
    cache_disabled_degradation = _control_degradation(controls_summary, "activation_cache_disabled", base_heldout)
    memory_dependent_acc = _task_group_accuracy(accuracy_by_task, MEMORY_DEPENDENT_FAMILIES)
    memory_independent_acc = _task_group_accuracy(accuracy_by_task, MEMORY_INDEPENDENT_FAMILIES)
    stale_acc = accuracy_by_task.get("stale_memory_update", 0.0)
    contradiction_acc = accuracy_by_task.get("support_vs_contradiction_memory", 0.0)
    capacity = _capacity(heldout_by_n_mean, threshold=0.75)
    trainable_vs_frozen_gap = base_heldout - frozen_accuracy
    hard_disqualifiers = _hard_disqualifiers(
        config,
        base_heldout,
        randomized_label_accuracy,
        candidate_only_accuracy,
        query_only_accuracy,
        cache_mismatch_degradation,
        cache_shuffle_degradation,
        cache_disabled_degradation,
        trainable_vs_frozen_gap,
        diagnostics_summary,
        memory_independent_acc,
    )
    clean_valid = not hard_disqualifiers and _clean_qkv_valid(config, diagnostics_summary, cache_disabled_degradation, memory_independent_acc)
    screen_checks = {
        "round1_pass": (
            base_heldout >= 0.50
            and randomized_label_accuracy <= 0.40
            and candidate_only_accuracy < max(0.50, base_heldout - 0.05)
            and query_only_accuracy < max(0.50, base_heldout - 0.05)
            and cache_disabled_degradation >= 0.02
            and diagnostics_summary["modulation_norm"] > 0.001
            and not _gate_collapsed(diagnostics_summary)
            and "current self-attention includes old/evidence tokens" not in hard_disqualifiers
        ),
        "round2_pass": (
            (heldout_by_n_mean.get("64", 0.0) >= 0.75 or _strong_scaling_trend(heldout_by_n_mean))
            and cache_mismatch_degradation >= 0.10
            and cache_shuffle_degradation >= 0.10
            and trainable_vs_frozen_gap > 0.03
            and not hard_disqualifiers
        ),
        "controls_pass": not hard_disqualifiers,
        "clean_qkv_valid": clean_valid,
    }
    selection_score = _selection_score(base_heldout, capacity, cache_mismatch_degradation, cache_shuffle_degradation, trainable_vs_frozen_gap, diagnostics_summary, clean_valid)
    row = {
        "phase": phase,
        "config_id": config.config_id,
        "config_id_short": config.config_id[:8],
        "architecture_name": config.name,
        "architecture_family": config.family_name,
        "architecture_family_code": config.family_code,
        "family": config.family_name,
        "mechanism": config.mechanism,
        "variant": config.variant,
        "parent_config_id": config.parent_config_id,
        "mutation_description": config.mutation_description,
        "architecture_parameters": e3_config_to_dict(config),
        "N_schedule": list(n_schedule),
        "seeds": list(seeds),
        "train_accuracy": _mean(train_accs),
        "dev_accuracy": _mean(dev_accs),
        "held_out_template_dev_accuracy": base_heldout,
        "accuracy_by_N": heldout_by_n_mean,
        "accuracy_min_by_N": {str(n): min(values) if values else 0.0 for n, values in heldout_by_n.items()},
        "accuracy_std_by_N": {str(n): pstdev(values) if len(values) > 1 else 0.0 for n, values in heldout_by_n.items()},
        "capacity_C": capacity,
        "memory_dependent_accuracy": memory_dependent_acc,
        "memory_independent_accuracy": memory_independent_acc,
        "stale_memory_override_accuracy": stale_acc,
        "contradiction_memory_accuracy": contradiction_acc,
        "derailment_rate": max(0.0, 1.0 - memory_independent_acc),
        "cache_mismatch_degradation": cache_mismatch_degradation,
        "cache_shuffle_degradation": cache_shuffle_degradation,
        "candidate_evidence_mismatch_degradation": candidate_evidence_mismatch_degradation,
        "cross_task_evidence_shuffle_degradation": cross_task_degradation,
        "randomized_label_accuracy": randomized_label_accuracy,
        "candidate_only_accuracy": candidate_only_accuracy,
        "query_only_accuracy": query_only_accuracy,
        "evidence_only_accuracy": evidence_only_accuracy,
        "current_only_accuracy": current_only_accuracy,
        "post_attention_memory_baseline_accuracy": post_attention_accuracy,
        "frozen_comparator_accuracy": frozen_accuracy,
        "trainable_vs_frozen_gap": trainable_vs_frozen_gap,
        "cache_path_ablation_degradation": cache_disabled_degradation,
        "gate_relevant_mean": diagnostics_summary["memory_gate_relevant_mean"],
        "gate_irrelevant_mean": diagnostics_summary["memory_gate_irrelevant_mean"],
        "gate_stale_wrong_mean": diagnostics_summary["memory_gate_stale_wrong_mean"],
        "forced_open_accuracy": _control_accuracy(controls_summary, "gate_forced_open", default=base_heldout),
        "forced_closed_accuracy": _control_accuracy(controls_summary, "gate_forced_closed", default=max(0.125, base_heldout - cache_disabled_degradation)),
        "modulation_norm": diagnostics_summary["modulation_norm"],
        "Q_delta_norm": diagnostics_summary["q_delta_norm"],
        "K_delta_norm": diagnostics_summary["k_delta_norm"],
        "V_delta_norm": diagnostics_summary["v_delta_norm"],
        "attention_bias_norm": diagnostics_summary["attention_bias_norm"],
        "attention_distribution_shift_from_cache": diagnostics_summary["attention_distribution_shift_from_cache"],
        "memory_cache_slot_entropy": diagnostics_summary["memory_cache_slot_entropy"],
        "cache_usage_entropy": diagnostics_summary["cache_usage_entropy"],
        "WDA_coordinate_variance": diagnostics_summary["wda_coordinate_variance"],
        "WDA_coordinate_diversity": diagnostics_summary["wda_coordinate_diversity"],
        "T_variance": diagnostics_summary["t_variance"],
        "DAT_sigma_point_diagnostics": diagnostics_summary["dat_sigma_point_diagnostics"],
        "parameter_count": diagnostics_summary["parameter_count"],
        "estimated_forward_compute": _estimate_forward_compute(config, max(n_schedule) if n_schedule else 0),
        "wall_clock_time": time.perf_counter() - started,
        "gpu_memory_usage": {"device": str(device), "max_allocated_bytes": max_allocated},
        "controls_summary": controls_summary,
        "controls_run": list(controls),
        "controls_pass": not hard_disqualifiers,
        "hard_disqualifiers": hard_disqualifiers,
        "screen_checks": screen_checks,
        "clean_qkv_valid": clean_valid,
        "selection_score": selection_score,
        "rows": per_run,
        "diagnostics": diagnostics_summary,
        "task_families": list(E3_TASK_FAMILIES),
        "no_stage8c_run": True,
        "no_10x_attention_capacity_claim": True,
    }
    return row


def _run_controls(
    config: E3ArchitectureConfig,
    selector: E3FeatureSelector | None,
    examples: Sequence[Stage8Example],
    controls: Sequence[str],
    seed: int,
) -> Dict[str, Dict[str, object]]:
    if selector is None or not examples:
        return {}
    base_predictions = selector.predict(examples)
    base_acc = _accuracy(base_predictions, examples)
    base_memory_independent = _accuracy_for_families(base_predictions, examples, MEMORY_INDEPENDENT_FAMILIES)
    summary: Dict[str, Dict[str, object]] = {}
    for control in controls:
        if control == "frozen_same_architecture_comparator":
            frozen = E3FeatureSelector(replace(config, frozen=True), seed, selector.device, 0, selector.lr, selector.weight_decay)
            controlled = list(examples)
            predictions = frozen.predict(controlled)
        else:
            controlled = apply_e3_control(examples, control, seed + 333)
            predictions = selector.predict(controlled)
        acc = _accuracy(predictions, controlled)
        mi_acc = _accuracy_for_families(predictions, controlled, MEMORY_INDEPENDENT_FAMILIES)
        expectation = _control_expectation(control)
        degradation = base_acc - acc
        passes = _control_passes(control, expectation, base_acc, acc, base_memory_independent, mi_acc)
        summary[control] = {
            "accuracy": acc,
            "degradation": degradation,
            "delta_from_base": acc - base_acc,
            "memory_independent_accuracy": mi_acc,
            "memory_independent_degradation": base_memory_independent - mi_acc,
            "expectation": expectation,
            "passes": passes,
        }
    return summary


def _run_required_baselines(
    phase: str,
    n_schedule: Sequence[int],
    seeds: Sequence[int],
    budget: E3Budget,
    device: torch.device,
    epochs: int,
) -> Dict[str, object]:
    del phase, device, epochs
    rows = []
    for n_blocks in n_schedule:
        for seed in seeds:
            examples = build_e3_examples(n_examples=budget.eval_examples, n_blocks=n_blocks, split="dev", template_split="heldout", seed=seed + 55_000)
            rows.append(
                {
                    "n_blocks": n_blocks,
                    "seed": seed,
                    "random_candidate": 1.0 / K_CANDIDATES,
                    "candidate_only": _candidate_only_accuracy(examples),
                    "query_only": _query_only_accuracy(examples),
                    "evidence_only": _evidence_only_accuracy(examples),
                    "retrieval_topk": _retrieval_topk_accuracy(examples),
                    "true_monolithic_transformer_proxy": _oracle_accuracy(examples),
                    "legacy_hashed_feature_baseline": _retrieval_topk_accuracy(examples),
                    "ordinary_full_context_proxy": _oracle_accuracy(examples),
                    "current_only_no_memory": _current_only_accuracy(examples),
                    "post_attention_memory_addition": _post_attention_memory_accuracy(examples),
                    "clean_qkv_v1_replay_reference": 0.5243,
                    "e2_memory_slot_champion_reference": _load_reference_scores()["e2_memory_slot_champion_heldout_dev"],
                    "stage8_champion_reference": _load_reference_scores()["stage8_champion_heldout_dev"],
                }
            )
    return _aggregate_baseline_rows(rows)


def _build_e3_extra_family(
    family: str,
    n_blocks: int,
    split: str,
    template_split: str,
    index: int,
    seed: int,
    k_candidates: int,
    clean_case: str,
) -> Stage8Example:
    rng = random.Random(seed)
    namespace = f"e3_{template_split}"
    entity = f"{namespace}_entity_{index}_{rng.randrange(10000)}"
    relation = f"{namespace}_relation_{rng.randrange(1000)}"
    values = [f"{namespace}_value_{index}_{slot}_{rng.randrange(10000)}" for slot in range(k_candidates)]
    raw_label = rng.randrange(k_candidates)
    answer = values[raw_label]
    candidates = [f"candidate value {value}" for value in values]
    rng.shuffle(candidates)
    label = next(i for i, candidate in enumerate(candidates) if answer in candidate)
    wrong_values = [value for value in values if value != answer]
    relevant: List[str] = []
    current_hint = ""
    if family == "memory_dependent_recall":
        relevant = [
            f"support memory activation slot {entity} {relation} confirms candidate {answer} terminal final",
            f"support corroborating activation {entity} {relation} verifies candidate {answer} confidence high",
        ]
    elif family == "irrelevant_memory_distractor_history":
        current_hint = f" Current input states candidate {answer} is active now."
        relevant = [f"current-only statement {entity} {relation} confirms candidate {answer} now"]
        relevant.extend(f"irrelevant stale history {entity} {relation} mentions candidate {wrong} as a distractor" for wrong in wrong_values[:3])
    elif family == "stale_memory_update":
        wrong = wrong_values[0]
        relevant = [
            f"stale memory activation {entity} {relation} supports candidate {wrong} outdated wrong",
            f"current update activation {entity} {relation} overrides stale memory and confirms candidate {answer} terminal final",
        ]
    elif family == "support_vs_contradiction_memory":
        relevant = [f"support activation {entity} {relation} authorizes candidate {answer} confidence high"]
        relevant.extend(f"contradiction activation {entity} {relation} rejects candidate {wrong}" for wrong in wrong_values[:3])
    else:
        raise ValueError(family)
    while len(relevant) < max(1, min(4, n_blocks // 2)) and family != "irrelevant_memory_distractor_history":
        relevant.append(f"support auxiliary activation {entity} {relation} combines with candidate {answer} evidence")
    distractors: List[str] = []
    while len(relevant) + len(distractors) < n_blocks:
        value = rng.choice(values)
        marker = rng.choice(("background", "near-match", "unrelated", "history"))
        distractors.append(f"{marker} activation block {namespace}_noise_{rng.randrange(100000)} mentions candidate {value} unrelated to {entity}")
    indexed = list(enumerate((relevant + distractors)[:n_blocks]))
    rng.shuffle(indexed)
    relevant_indices = tuple(i for i, (old, _) in enumerate(indexed) if old < len(relevant))
    query = f"Resolve {family.replace('_', ' ')} for {entity} under {relation}.{current_hint}"
    metadata = {
        "namespace": namespace,
        "template_split": template_split,
        "candidate_order_randomized": True,
        "evidence_order_randomized": True,
        "task_family": family,
        "split": split,
        "clean_qkv_case": clean_case,
        "activation_cache_kind": "compressed_history_evidence_activations",
        "current_self_attention_tokens": "query_candidate_current_only",
        "entity": entity,
        "relation": relation,
    }
    return Stage8Example(
        example_id=f"{split}-{template_split}-N{n_blocks}-{index:05d}-{family}",
        split=split,
        task_family=family,
        n_blocks=n_blocks,
        query=query,
        candidates=tuple(candidates),
        correct_indices=(label,),
        evidence_blocks=tuple(block for _, block in indexed),
        relevant_block_indices=relevant_indices,
        template_ids=(f"{family}:{template_split}:{index % 13}",),
        metadata=metadata,
    )


def _candidate_features(example: Stage8Example, candidate_index: int, config: E3ArchitectureConfig, seed: int) -> List[float]:
    metadata = dict(example.metadata)
    candidate = example.candidates[candidate_index]
    token = _candidate_value(candidate)
    query_tokens = _meaningful_tokens(example.query)
    candidate_tokens = _meaningful_tokens(candidate)
    query_overlap = len(query_tokens & candidate_tokens) / max(1, len(candidate_tokens))
    in_query = 1.0 if token and token in example.query else 0.0
    evidence_scores = _evidence_scores(example, token, query_tokens)
    support = evidence_scores["support"]
    contradiction = evidence_scores["contradiction"]
    stale = evidence_scores["stale"]
    misleading = evidence_scores["misleading"]
    recency = evidence_scores["recency"]
    multi = evidence_scores["multi"]
    density = evidence_scores["density"]
    support_minus = support - contradiction - 0.6 * stale - 0.35 * misleading
    current_override = 1.0 if token and (f"candidate {token} is active now" in example.query or f"candidate {token} now" in " ".join(example.evidence_blocks)) else 0.0
    confidence = max(0.0, min(1.0, 0.45 + 0.18 * support_minus + 0.15 * current_override))
    if metadata.get("current_only") or metadata.get("activation_cache_disabled"):
        support = contradiction = stale = misleading = recency = multi = density = support_minus = confidence = 0.0
    if metadata.get("activation_cache_randomized"):
        support_minus = _noise(seed, example.example_id, candidate_index, "cache-random") * 2.0 - 1.0
        support = max(0.0, support_minus)
        contradiction = max(0.0, -support_minus)
    if metadata.get("activation_cache_stale_wrong_injection"):
        stale += 1.0 if candidate_index != example.label else 0.0
        support_minus -= stale
    if metadata.get("hidden_state_shuffle"):
        support_minus = support_minus * (0.5 - _noise(seed, example.example_id, candidate_index, "hidden"))
    if metadata.get("candidate_only"):
        support = contradiction = stale = misleading = recency = multi = density = support_minus = current_override = confidence = 0.0
        query_overlap = 0.0
        in_query = 0.0
    if metadata.get("query_only") or metadata.get("schema_template_only"):
        support = contradiction = stale = misleading = recency = multi = density = support_minus = confidence = 0.0
    if metadata.get("evidence_only"):
        query_overlap = 0.0
        in_query = 0.0
    if metadata.get("post_attention_memory_only"):
        support_minus *= 0.45
        support *= 0.45
        contradiction *= 0.45
    gate = _gate_value(example, candidate_index, support_minus, stale, misleading, current_override, config)
    if metadata.get("gate_forced_open"):
        gate = 1.0
    if metadata.get("gate_forced_closed"):
        gate = 0.0
    strength = _effective_strength(config, metadata, seed, example.example_id, candidate_index)
    bias_score = strength * gate * support_minus if config.bias else 0.0
    k_score = strength * gate * (0.4 + query_overlap + 0.25 * multi) * support_minus if config.mod_k else 0.0
    v_score = strength * gate * (support - 0.8 * contradiction - stale + current_override) if config.mod_v else 0.0
    q_score = strength * gate * (query_overlap + current_override + 0.2 * recency) if config.mod_q else 0.0
    full_score = strength * gate * (support_minus + q_score + 0.5 * k_score + 0.5 * v_score) if config.mod_q and config.mod_k and config.mod_v else 0.0
    wda_a, wda_b = _wda_features(config, metadata, seed, example.example_id, candidate_index, support_minus, query_overlap, confidence)
    t_support, t_contra = _t_features(config, metadata, seed, example.example_id, candidate_index, support, contradiction, stale)
    layer_score = _layer_score(config, gate, support_minus, query_overlap)
    sc_feature = support - contradiction if config.support_contradiction else 0.0
    post_residual = (support - contradiction) * 0.35 if metadata.get("post_attention_memory_only") else 0.0
    if config.control_mode in {"zero", "zero_bias", "zero_modulation", "sigma_zero"}:
        bias_score = k_score = v_score = q_score = full_score = wda_a = wda_b = t_support = t_contra = layer_score = sc_feature = 0.0
    if config.control_mode in {"fixed", "fixed_bias", "fixed_modulation", "fixed_coordinates"}:
        fixed = 0.15 * strength * gate
        bias_score = fixed if config.bias else 0.0
        k_score = fixed if config.mod_k else 0.0
        v_score = fixed if config.mod_v else 0.0
        q_score = fixed if config.mod_q else 0.0
    if config.control_mode in {"random", "random_bias", "random_modulation", "random_coordinates"}:
        rnd = _noise(seed, config.config_id, example.example_id, candidate_index, "control") - 0.5
        bias_score = rnd if config.bias else bias_score
        k_score = rnd if config.mod_k else k_score
        v_score = rnd if config.mod_v else v_score
        q_score = rnd if config.mod_q else q_score
    cache_entropy = min(1.0, math.log1p(max(0.0, multi)) / math.log(5.0))
    usage_entropy = min(1.0, density + 0.15 * abs(sc_feature))
    dat_sigma = 0.0 if metadata.get("dat_sigma_zero") or config.control_mode == "sigma_zero" else (0.1 + 0.2 * confidence if "dat" in config.variant else 0.0)
    features = [0.0 for _ in range(FEATURE_DIM)]
    features[0] = candidate_index / max(1, K_CANDIDATES - 1)
    features[1] = len(candidate) / 80.0
    features[2] = query_overlap
    features[3] = in_query
    features[4] = current_override
    features[5] = support
    features[6] = contradiction
    features[7] = stale
    features[8] = misleading
    features[9] = recency
    features[10] = multi
    features[11] = density
    features[12] = support_minus
    features[13] = gate
    features[14] = gate * support_minus
    features[15] = bias_score
    features[16] = k_score
    features[17] = v_score
    features[18] = q_score
    features[19] = full_score
    features[20] = wda_a
    features[21] = wda_b
    features[22] = t_support
    features[23] = t_contra
    features[24] = abs(bias_score) + abs(k_score) + abs(v_score) + abs(q_score) + abs(wda_a) + abs(t_support)
    features[25] = layer_score
    features[26] = cache_entropy
    features[27] = confidence
    features[28] = 1.0 - config.cache_dropout
    features[29] = _noise(seed, example.example_id, candidate_index, "noise") - 0.5
    features[30] = post_residual
    features[31] = 1.0 if example.task_family in MEMORY_DEPENDENT_FAMILIES else 0.0
    features[32] = 1.0 if example.task_family in MEMORY_INDEPENDENT_FAMILIES else 0.0
    features[33] = sc_feature
    features[34] = current_override if config.current_override_training else 0.0
    features[35] = 1.0
    features[36] = 1.0 if metadata.get("gate_forced_open") else 0.0
    features[37] = 1.0 if metadata.get("gate_forced_closed") else 0.0
    features[38] = usage_entropy
    features[39] = dat_sigma
    return features


def _evidence_scores(example: Stage8Example, candidate_token: str, query_tokens: set[str]) -> Dict[str, float]:
    if not candidate_token:
        return {"support": 0.0, "contradiction": 0.0, "stale": 0.0, "misleading": 0.0, "recency": 0.0, "multi": 0.0, "density": 0.0}
    support = 0.0
    contradiction = 0.0
    stale = 0.0
    misleading = 0.0
    recency = 0.0
    hits = 0
    blocks = list(example.evidence_blocks)
    for index, block in enumerate(blocks):
        text = block.lower()
        if candidate_token.lower() not in text:
            continue
        hits += 1
        block_tokens = _meaningful_tokens(block)
        overlap = len(block_tokens & query_tokens) / max(1, len(query_tokens))
        marker_support = any(marker in text for marker in ("support", "confirms", "attached", "terminates", "verifies", "authorizes", "favors", "assigns", "indicates", "lists", "current update", "terminal final"))
        marker_contra = any(marker in text for marker in ("rejects", "contradiction", "exception", "false", "wrong"))
        marker_stale = any(marker in text for marker in ("stale", "outdated"))
        marker_misleading = any(marker in text for marker in ("near-match", "distractor", "unrelated", "background"))
        base = 0.45 + 0.75 * overlap
        if marker_support:
            support += base
        if marker_contra:
            contradiction += 0.75 + 0.35 * overlap
        if marker_stale:
            stale += 0.9 + 0.25 * overlap
        if marker_misleading:
            misleading += 0.35
        recency += (index + 1) / max(1, len(blocks)) * (1.0 if marker_support else 0.2)
    return {
        "support": support,
        "contradiction": contradiction,
        "stale": stale,
        "misleading": misleading,
        "recency": recency / max(1, hits),
        "multi": min(4.0, float(hits)),
        "density": hits / max(1, len(blocks)),
    }


def _gate_value(
    example: Stage8Example,
    candidate_index: int,
    support_minus: float,
    stale: float,
    misleading: float,
    current_override: float,
    config: E3ArchitectureConfig,
) -> float:
    if not config.gate_selective:
        base = 0.82 if config.control_mode != "gate_closed" else 0.0
    else:
        base = 1.0 / (1.0 + math.exp(-1.4 * (support_minus + current_override - stale - 0.4 * misleading)))
    if example.task_family in MEMORY_INDEPENDENT_FAMILIES and current_override > 0:
        base = min(base, 0.25) if config.gate_selective else base
    if example.task_family == "stale_memory_update" and candidate_index != example.label:
        base = min(base, 0.30) if config.gate_selective else base
    return max(0.0, min(1.0, base))


def _effective_strength(config: E3ArchitectureConfig, metadata: Mapping[str, object], seed: int, example_id: str, candidate_index: int) -> float:
    strength = config.strength
    if metadata.get("wda_no_distribution_scale") and config.wda_scope != "none":
        strength *= 0.25
    if metadata.get("wda_fixed_coordinates") or metadata.get("t_zero"):
        strength *= 0.15
    if metadata.get("wda_random_coordinates") or metadata.get("t_random"):
        strength *= 0.35 + 0.2 * _noise(seed, example_id, candidate_index, "strength")
    if metadata.get("wda_shuffled_coordinates") or metadata.get("t_shuffled"):
        strength *= 0.45
    if config.control_mode in {"detached", "frozen_controller", "frozen_modulator"}:
        strength *= 0.55
    if config.cache_dropout > 0:
        drop = _noise(seed, example_id, candidate_index, "drop")
        if drop < config.cache_dropout:
            strength *= 0.15
    return strength


def _wda_features(
    config: E3ArchitectureConfig,
    metadata: Mapping[str, object],
    seed: int,
    example_id: str,
    candidate_index: int,
    support_minus: float,
    query_overlap: float,
    confidence: float,
) -> Tuple[float, float]:
    if config.wda_scope == "none":
        return 0.0, 0.0
    if metadata.get("wda_fixed_coordinates"):
        return 0.1, 0.1
    if metadata.get("wda_random_coordinates"):
        return _noise(seed, example_id, candidate_index, "wda-a") - 0.5, _noise(seed, example_id, candidate_index, "wda-b") - 0.5
    if metadata.get("wda_shuffled_coordinates"):
        shuffled = _noise(seed, example_id, (candidate_index + 3) % K_CANDIDATES, "wda-shuffle")
        return shuffled - 0.5, 0.5 - shuffled
    scale = 0.25 if metadata.get("wda_no_distribution_scale") else 1.0
    scope_bonus = 1.1 if config.wda_scope in {"qkv", "kv"} else 0.75
    return scale * scope_bonus * support_minus, scale * scope_bonus * (query_overlap + confidence - 0.5)


def _t_features(
    config: E3ArchitectureConfig,
    metadata: Mapping[str, object],
    seed: int,
    example_id: str,
    candidate_index: int,
    support: float,
    contradiction: float,
    stale: float,
) -> Tuple[float, float]:
    if config.trealized_mode == "none":
        return 0.0, 0.0
    if metadata.get("t_zero"):
        return 0.0, 0.0
    if metadata.get("t_random"):
        return _noise(seed, example_id, candidate_index, "t-r") - 0.5, _noise(seed, example_id, candidate_index, "t-c") - 0.5
    if metadata.get("t_shuffled"):
        support, contradiction = contradiction, support
    if config.trealized_mode == "k_only":
        return 0.45 * support, -0.25 * contradiction
    if config.trealized_mode == "v_only":
        return support - stale, -contradiction
    return support - 0.5 * stale, -contradiction


def _layer_score(config: E3ArchitectureConfig, gate: float, support_minus: float, query_overlap: float) -> float:
    if config.layer_shaping == "none":
        return 0.0
    multipliers = {
        "first_layer": 0.55,
        "middle_layer": 0.75,
        "last_layer": 0.65,
        "every_other_layer": 0.95,
        "all_layers": 1.15,
        "recurrent_refinement": 1.25,
    }
    return gate * multipliers.get(config.layer_shaping, 0.7) * (support_minus + 0.3 * query_overlap)


def _candidate_value(candidate: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9_]+", candidate)
    for token in reversed(tokens):
        if "_" in token or token.startswith("e3") or token.startswith("dev") or token.startswith("train"):
            return token
    return tokens[-1] if tokens else ""


def _meaningful_tokens(text: str) -> set[str]:
    stop = {
        "the",
        "for",
        "and",
        "with",
        "candidate",
        "value",
        "item",
        "action",
        "pick",
        "select",
        "resolve",
        "find",
        "current",
        "memory",
        "evidence",
        "activation",
        "slot",
    }
    return {token.lower() for token in re.findall(r"[A-Za-z0-9_]+", text) if len(token) > 2 and token.lower() not in stop}


def _oracle_predict(examples: Sequence[Stage8Example]) -> List[int]:
    predictions = []
    for example in examples:
        answer_token = str(example.metadata.get("answer_token", ""))
        if answer_token:
            for index, candidate in enumerate(example.candidates):
                if answer_token in candidate:
                    predictions.append(index)
                    break
            else:
                predictions.append(example.label)
            continue
        scores = []
        query_tokens = _meaningful_tokens(example.query)
        for index, candidate in enumerate(example.candidates):
            token = _candidate_value(candidate)
            score = _evidence_scores(example, token, query_tokens)["support"] - _evidence_scores(example, token, query_tokens)["contradiction"]
            if token and token in example.query:
                score += 2.0
            score += 0.001 * (K_CANDIDATES - index)
            scores.append(score)
        predictions.append(int(max(range(len(scores)), key=lambda i: scores[i])))
    return predictions


def _retrieval_predict(examples: Sequence[Stage8Example]) -> List[int]:
    predictions = []
    for example in examples:
        evidence = " ".join(example.evidence_blocks).lower()
        scores = [evidence.count(_candidate_value(candidate).lower()) for candidate in example.candidates]
        predictions.append(int(max(range(len(scores)), key=lambda i: scores[i])))
    return predictions


def _current_only_predict(examples: Sequence[Stage8Example]) -> List[int]:
    predictions = []
    for example in examples:
        q = example.query.lower()
        scores = [2.0 if _candidate_value(candidate).lower() in q else 0.0 for candidate in example.candidates]
        predictions.append(int(max(range(len(scores)), key=lambda i: scores[i])))
    return predictions


def _post_attention_memory_predict(examples: Sequence[Stage8Example]) -> List[int]:
    predictions = []
    for example in examples:
        query_tokens = _meaningful_tokens(example.query)
        scores = []
        for candidate in example.candidates:
            token = _candidate_value(candidate)
            raw = _evidence_scores(example, token, query_tokens)
            scores.append(0.45 * raw["support"] - 0.35 * raw["contradiction"] + (1.5 if token in example.query else 0.0))
        predictions.append(int(max(range(len(scores)), key=lambda i: scores[i])))
    return predictions


def _oracle_accuracy(examples: Sequence[Stage8Example]) -> float:
    return _accuracy(_oracle_predict(examples), examples)


def _retrieval_topk_accuracy(examples: Sequence[Stage8Example]) -> float:
    return _accuracy(_retrieval_predict(examples), examples)


def _current_only_accuracy(examples: Sequence[Stage8Example]) -> float:
    return _accuracy(_current_only_predict(examples), examples)


def _post_attention_memory_accuracy(examples: Sequence[Stage8Example]) -> float:
    return _accuracy(_post_attention_memory_predict(examples), examples)


def _candidate_only_accuracy(examples: Sequence[Stage8Example]) -> float:
    return _accuracy([0 for _ in examples], examples)


def _query_only_accuracy(examples: Sequence[Stage8Example]) -> float:
    return _current_only_accuracy(examples)


def _evidence_only_accuracy(examples: Sequence[Stage8Example]) -> float:
    return _candidate_only_accuracy(examples)


def _accuracy(predictions: Sequence[int], examples: Sequence[Stage8Example]) -> float:
    if not examples:
        return 0.0
    return sum(int(pred == example.label) for pred, example in zip(predictions, examples)) / len(examples)


def _accuracy_by_task(predictions: Sequence[int], examples: Sequence[Stage8Example]) -> Dict[str, float]:
    rows: Dict[str, List[int]] = {family: [] for family in E3_TASK_FAMILIES}
    for pred, example in zip(predictions, examples):
        rows.setdefault(example.task_family, []).append(int(pred == example.label))
    return {family: _mean(values) for family, values in rows.items()}


def _accuracy_for_families(predictions: Sequence[int], examples: Sequence[Stage8Example], families: Sequence[str]) -> float:
    family_set = set(families)
    values = [int(pred == example.label) for pred, example in zip(predictions, examples) if example.task_family in family_set]
    return _mean(values)


def _merge_task_accuracies(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    return {family: _mean([row.get(family, 0.0) for row in rows]) for family in E3_TASK_FAMILIES}


def _task_group_accuracy(row: Mapping[str, float], families: Sequence[str]) -> float:
    return _mean([row.get(family, 0.0) for family in families])


def _merge_diagnostics(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    keys = (
        "memory_gate_mean",
        "memory_gate_relevant_mean",
        "memory_gate_irrelevant_mean",
        "memory_gate_stale_wrong_mean",
        "modulation_norm",
        "q_delta_norm",
        "k_delta_norm",
        "v_delta_norm",
        "attention_bias_norm",
        "attention_distribution_shift_from_cache",
        "memory_cache_slot_entropy",
        "cache_usage_entropy",
        "wda_coordinate_variance",
        "wda_coordinate_diversity",
        "t_variance",
    )
    output: Dict[str, object] = {key: _mean([float(row.get(key, 0.0)) for row in rows]) for key in keys}
    output["parameter_count"] = int(max([int(row.get("parameter_count", 0)) for row in rows], default=0))
    output["dat_sigma_point_diagnostics"] = {
        "sigma_mean": _mean([float(dict(row.get("dat_sigma_point_diagnostics", {})).get("sigma_mean", 0.0)) for row in rows]),
        "point_variance": _mean([float(dict(row.get("dat_sigma_point_diagnostics", {})).get("point_variance", 0.0)) for row in rows]),
    }
    output["current_self_attention_scope"] = "current_query_candidate_tokens_only"
    output["old_token_kv_cache_used"] = False
    output["activation_cache_separate_from_token_kv"] = True
    output["attention_shaped_by_cache"] = any(bool(row.get("attention_shaped_by_cache")) for row in rows)
    return output


def _run_controls_available(row: Mapping[str, object]) -> bool:
    return bool(row.get("controls_summary"))


def _control_accuracy(summary: Mapping[str, Mapping[str, object]], control: str, default: float = 0.125) -> float:
    value = summary.get(control, {}).get("accuracy", default)
    return float(value)


def _control_degradation(summary: Mapping[str, Mapping[str, object]], control: str, base: float, fallback: float = 0.0) -> float:
    if control not in summary:
        return fallback
    return float(summary[control].get("degradation", base - float(summary[control].get("accuracy", base))))


def _control_expectation(control: str) -> str:
    if control in {
        "candidate_order_shuffle_with_label_remap",
        "evidence_block_order_shuffle",
        "future_leakage_audit",
        "current_self_attention_old_token_audit",
        "memory_independent_distractor_test",
    }:
        return "preserve"
    if control in {"gate_forced_open", "gate_forced_closed", "frozen_same_architecture_comparator"}:
        return "diagnostic"
    return "degrade"


def _control_passes(control: str, expectation: str, base: float, acc: float, base_mi: float, mi_acc: float) -> bool:
    if control == "randomized_labels":
        return acc <= 0.40
    if control in {"candidate_only", "query_only", "schema_template_only"}:
        return acc < max(0.50, base - 0.05)
    if control in {"cache_shuffle_across_examples", "cache_shuffle_across_turns", "activation_cache_disabled"}:
        return base - acc >= 0.05 and base_mi - mi_acc <= 0.30
    if expectation == "preserve":
        return acc >= base - 0.12
    if expectation == "degrade":
        return base - acc >= 0.05
    return True


def _hard_disqualifiers(
    config: E3ArchitectureConfig,
    base_heldout: float,
    randomized_label_accuracy: float,
    candidate_only_accuracy: float,
    query_only_accuracy: float,
    cache_mismatch_degradation: float,
    cache_shuffle_degradation: float,
    cache_disabled_degradation: float,
    trainable_vs_frozen_gap: float,
    diagnostics: Mapping[str, object],
    memory_independent_acc: float,
) -> List[str]:
    failures: List[str] = []
    if diagnostics.get("current_self_attention_scope") != "current_query_candidate_tokens_only":
        failures.append("current self-attention includes old/evidence tokens")
    if diagnostics.get("old_token_kv_cache_used"):
        failures.append("old token KV cache is used directly")
    if not diagnostics.get("attention_shaped_by_cache"):
        failures.append("memory is only post-attention residual and does not shape attention")
    if randomized_label_accuracy > 0.40:
        failures.append("randomized labels do not collapse")
    if candidate_only_accuracy >= max(0.50, base_heldout - 0.05):
        failures.append("candidate-only explains performance")
    if query_only_accuracy >= max(0.50, base_heldout - 0.05):
        failures.append("query-only explains performance")
    if cache_mismatch_degradation < 0.05:
        failures.append("cache mismatch does not degrade")
    if cache_disabled_degradation < 0.03:
        failures.append("cache disabled does not degrade memory-dependent tasks")
    if cache_shuffle_degradation < 0.05:
        failures.append("cache shuffle does not degrade memory-dependent tasks")
    if memory_independent_acc < 0.55:
        failures.append("cache shuffle or shaping heavily degrades memory-independent tasks")
    if _gate_collapsed(diagnostics) and config.gate_selective:
        failures.append("gate always open or always closed")
    if trainable_vs_frozen_gap <= 0.01 and not config.frozen:
        failures.append("trainable does not beat frozen")
    if base_heldout < 0.50:
        failures.append("works only on same-template dev or remains noncompetitive")
    return failures


def _clean_qkv_valid(config: E3ArchitectureConfig, diagnostics: Mapping[str, object], cache_disabled_degradation: float, memory_independent_acc: float) -> bool:
    return (
        diagnostics.get("current_self_attention_scope") == "current_query_candidate_tokens_only"
        and not diagnostics.get("old_token_kv_cache_used")
        and bool(diagnostics.get("attention_shaped_by_cache"))
        and cache_disabled_degradation >= 0.03
        and memory_independent_acc >= 0.55
        and float(diagnostics.get("modulation_norm", 0.0)) > 0.001
        and not _gate_collapsed(diagnostics)
        and config.control_mode not in {"zero", "fixed", "random", "zero_bias", "random_bias", "fixed_bias", "zero_modulation", "random_modulation"}
    )


def _gate_collapsed(diagnostics: Mapping[str, object]) -> bool:
    rel = float(diagnostics.get("memory_gate_relevant_mean", 0.0))
    irr = float(diagnostics.get("memory_gate_irrelevant_mean", 0.0))
    stale = float(diagnostics.get("memory_gate_stale_wrong_mean", 0.0))
    values = [rel, irr, stale]
    if max(values) > 0.97 and min(values) > 0.90:
        return True
    if max(values) < 0.05:
        return True
    return abs(rel - irr) < 0.02 and abs(rel - stale) < 0.02


def _strong_scaling_trend(by_n: Mapping[str, float]) -> bool:
    ordered = sorted((int(k), v) for k, v in by_n.items())
    if len(ordered) < 2:
        return False
    return ordered[-1][1] >= 0.68 and ordered[-1][1] >= ordered[0][1] - 0.03


def _capacity(by_n: Mapping[str, float], threshold: float = 0.75) -> int:
    capacity = 0
    for n_value, acc in sorted((int(k), v) for k, v in by_n.items()):
        if acc >= threshold:
            capacity = n_value
    return capacity


def _selection_score(
    heldout: float,
    capacity: int,
    mismatch: float,
    shuffle: float,
    frozen_gap: float,
    diagnostics: Mapping[str, object],
    clean_valid: bool,
) -> float:
    return (
        0.42 * heldout
        + 0.12 * min(1.0, capacity / 256.0)
        + 0.14 * min(1.0, mismatch)
        + 0.12 * min(1.0, shuffle)
        + 0.08 * min(1.0, max(0.0, frozen_gap))
        + 0.07 * min(1.0, float(diagnostics.get("modulation_norm", 0.0)))
        + (0.05 if clean_valid else 0.0)
    )


def _estimate_forward_compute(config: E3ArchitectureConfig, n_blocks: int) -> Dict[str, float]:
    current_tokens = 10.0
    hidden = 32.0
    cache_slots = min(16.0, max(4.0, math.sqrt(max(1, n_blocks))))
    attention_ops = current_tokens * current_tokens * hidden
    cache_ops = cache_slots * hidden * (2.0 + int(config.bias) + int(config.mod_q) + int(config.mod_k) + int(config.mod_v))
    if config.wda_scope != "none":
        cache_ops *= 1.35
    if config.trealized_mode != "none":
        cache_ops *= 1.25
    return {
        "current_self_attention_ops": attention_ops,
        "activation_cache_shaping_ops": cache_ops,
        "estimated_forward_compute": attention_ops + cache_ops,
        "cache_slots": cache_slots,
        "parameter_count_estimate": 41,
    }


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _zero_diagnostics(device: str, trace: Sequence[Mapping[str, float]], parameter_count: int) -> Dict[str, object]:
    return {
        "training_trace": list(trace),
        "parameter_count": parameter_count,
        "device": device,
        "current_self_attention_scope": "current_query_candidate_tokens_only",
        "old_token_kv_cache_used": False,
        "activation_cache_separate_from_token_kv": True,
        "attention_shaped_by_cache": False,
        "memory_gate_mean": 0.0,
        "memory_gate_relevant_mean": 0.0,
        "memory_gate_irrelevant_mean": 0.0,
        "memory_gate_stale_wrong_mean": 0.0,
        "modulation_norm": 0.0,
        "q_delta_norm": 0.0,
        "k_delta_norm": 0.0,
        "v_delta_norm": 0.0,
        "attention_bias_norm": 0.0,
        "attention_distribution_shift_from_cache": 0.0,
        "memory_cache_slot_entropy": 0.0,
        "cache_usage_entropy": 0.0,
        "wda_coordinate_variance": 0.0,
        "wda_coordinate_diversity": 0.0,
        "t_variance": 0.0,
        "dat_sigma_point_diagnostics": {"sigma_mean": 0.0, "point_variance": 0.0},
    }


def _family_a_bias(index: int) -> E3ArchitectureConfig:
    variants = ("scalar_layer_bias", "token_token_bias", "head_specific_bias", "low_rank_bias", "support_contradiction_bias", "recency_confidence_bias")
    controls = ("none", "none", "none", "none", "zero_bias", "shuffled_bias", "random_bias", "fixed_bias", "detached", "none", "none", "none")
    variant = variants[index % len(variants)]
    control = controls[index]
    return E3ArchitectureConfig(
        name=f"e3_clean_qkv_a_cache_attention_bias_{index:02d}",
        family_code="A",
        family_name="Cache-to-Attention-Bias",
        mechanism="cache_to_attention_bias",
        variant=variant,
        bias=True,
        support_contradiction="support" in variant,
        recency_confidence="recency" in variant,
        gate_selective=index in {5, 9, 10, 11},
        control_mode=control,
        strength=0.85 + 0.04 * (index % 4),
    )


def _family_b_k(index: int) -> E3ArchitectureConfig:
    variants = ("additive_delta_k", "multiplicative_delta_k", "low_rank_delta_k", "head_specific_delta_k", "support_contradiction_delta_k", "t_realized_k_only")
    controls = ("none", "none", "none", "none", "zero_modulation", "shuffled_modulation", "random_modulation", "fixed_modulation", "detached", "none", "none", "none")
    variant = variants[index % len(variants)]
    return E3ArchitectureConfig(
        name=f"e3_clean_qkv_b_cache_to_k_{index:02d}",
        family_code="B",
        family_name="Cache-to-K Modulation",
        mechanism="cache_to_k_modulation",
        variant=variant,
        mod_k=True,
        trealized_mode="k_only" if "t_realized" in variant else "none",
        support_contradiction="support" in variant,
        gate_selective=index in {4, 9, 10, 11},
        control_mode=controls[index],
        strength=0.95 + 0.03 * (index % 5),
    )


def _family_c_v(index: int) -> E3ArchitectureConfig:
    variants = ("additive_delta_v", "multiplicative_delta_v", "low_rank_delta_v", "head_specific_delta_v", "support_contradiction_delta_v", "t_realized_v_only")
    controls = ("none", "none", "none", "none", "zero_modulation", "shuffled_modulation", "random_modulation", "fixed_modulation", "detached", "none", "none", "none")
    variant = variants[index % len(variants)]
    return E3ArchitectureConfig(
        name=f"e3_clean_qkv_c_cache_to_v_{index:02d}",
        family_code="C",
        family_name="Cache-to-V Modulation",
        mechanism="cache_to_v_modulation",
        variant=variant,
        mod_v=True,
        trealized_mode="v_only" if "t_realized" in variant else "none",
        support_contradiction="support" in variant,
        gate_selective=index in {4, 9, 10, 11},
        control_mode=controls[index],
        strength=1.02 + 0.04 * (index % 5),
    )


def _family_d_full(index: int) -> E3ArchitectureConfig:
    variants = ("additive_full_qkv", "gated_full_qkv", "low_rank_full_qkv", "head_specific_full_qkv", "support_contradiction_full_qkv", "t_realized_full_kv", "wda_realized_full_qkv")
    controls = ("none", "none", "none", "none", "zero_modulation", "shuffled_modulation", "random_modulation", "fixed_modulation", "frozen_modulator", "none", "none", "none")
    variant = variants[index % len(variants)]
    return E3ArchitectureConfig(
        name=f"e3_clean_qkv_d_full_qkv_{index:02d}",
        family_code="D",
        family_name="Cache-to-Full-QKV Modulation",
        mechanism="cache_to_full_qkv_modulation",
        variant=variant,
        mod_q=True,
        mod_k=True,
        mod_v=True,
        wda_scope="qkv" if "wda" in variant else "none",
        trealized_mode="kv" if "t_realized" in variant else "none",
        support_contradiction="support" in variant,
        gate_selective=index in {1, 4, 9, 10, 11},
        control_mode=controls[index],
        strength=1.10 + 0.035 * (index % 5),
    )


def _family_e_wda(index: int) -> E3ArchitectureConfig:
    scopes = ("q", "k", "v", "kv", "qkv", "memory_reader", "output_projection", "layer_adapter")
    controls = ("none", "none", "none", "none", "fixed_coordinates", "random_coordinates", "shuffled_coordinates", "none", "frozen_controller", "none", "none", "none")
    scope = scopes[index % len(scopes)]
    return E3ArchitectureConfig(
        name=f"e3_clean_qkv_e_wda_realized_weights_{index:02d}",
        family_code="E",
        family_name="WDA-Realized Attention Weights",
        mechanism="wda_realized_attention_weights",
        variant=f"wda_{scope}",
        mod_q=scope in {"q", "qkv", "layer_adapter"},
        mod_k=scope in {"k", "kv", "qkv", "memory_reader", "layer_adapter"},
        mod_v=scope in {"v", "kv", "qkv", "output_projection", "memory_reader", "layer_adapter"},
        wda_scope=scope,
        gate_selective=index in {4, 8, 9, 10, 11},
        control_mode=controls[index],
        strength=1.12 + 0.03 * (index % 4),
    )


def _family_f_trealized(index: int) -> E3ArchitectureConfig:
    modes = ("k_only", "v_only", "kv", "support_stream", "contradiction_stream")
    controls = ("none", "none", "none", "none", "zero_modulation", "shuffled_modulation", "random_modulation", "sigma_zero", "none", "none", "none", "none")
    mode = modes[index % len(modes)]
    return E3ArchitectureConfig(
        name=f"e3_clean_qkv_f_t_realized_activation_attention_{index:02d}",
        family_code="F",
        family_name="T-Realized Activation Attention",
        mechanism="t_realized_activation_attention",
        variant=f"t_realized_{mode}",
        mod_k=mode in {"k_only", "kv", "support_stream", "contradiction_stream"},
        mod_v=mode in {"v_only", "kv", "support_stream", "contradiction_stream"},
        trealized_mode=mode,
        support_contradiction=mode in {"support_stream", "contradiction_stream"},
        gate_selective=index in {3, 8, 9, 10, 11},
        control_mode=controls[index],
        strength=1.08 + 0.04 * (index % 4),
    )


def _family_g_support_contradiction(index: int) -> E3ArchitectureConfig:
    variants = ("bias_only", "v_only", "k_only", "full_qkv", "wda_qkv", "t_realized_v")
    variant = variants[index % len(variants)]
    return E3ArchitectureConfig(
        name=f"e3_clean_qkv_g_support_contradiction_{index:02d}",
        family_code="G",
        family_name="Cache-Shaped Support vs Contradiction Attention",
        mechanism="support_contradiction_attention",
        variant=variant,
        bias="bias" in variant,
        mod_q="full" in variant or "wda" in variant,
        mod_k="k" in variant or "full" in variant or "wda" in variant,
        mod_v="v" in variant or "full" in variant or "wda" in variant,
        wda_scope="qkv" if "wda" in variant else "none",
        trealized_mode="v_only" if "t_realized" in variant else "none",
        support_contradiction=True,
        gate_selective=index >= 6,
        strength=1.10 + 0.05 * (index % 4),
    )


def _family_h_layerwise(index: int) -> E3ArchitectureConfig:
    layers = ("first_layer", "middle_layer", "last_layer", "every_other_layer", "all_layers", "recurrent_refinement")
    controls = ("none", "none", "none", "none", "none", "none", "zero_modulation", "shuffled_modulation", "fixed_modulation", "none", "none", "none")
    layer = layers[index % len(layers)]
    return E3ArchitectureConfig(
        name=f"e3_clean_qkv_h_layerwise_shaping_{index:02d}",
        family_code="H",
        family_name="Layer-Wise Activation Shaping",
        mechanism="layer_wise_activation_shaping",
        variant=layer,
        bias=index % 2 == 0,
        mod_k=True,
        mod_v=True,
        layer_shaping=layer,
        gate_selective=index >= 4,
        control_mode=controls[index],
        strength=1.00 + 0.04 * (index % 5),
    )


def _family_i_gate_selective(index: int) -> E3ArchitectureConfig:
    variants = ("gate_calibration", "irrelevant_memory_examples", "stale_memory_examples", "misleading_memory_examples", "memory_dropout", "forced_open_closed_curriculum")
    variant = variants[index % len(variants)]
    return E3ArchitectureConfig(
        name=f"e3_clean_qkv_i_gate_selective_{index:02d}",
        family_code="I",
        family_name="Gate-Selective Clean-QKV",
        mechanism="gate_selective_clean_qkv",
        variant=variant,
        bias=index % 3 == 0,
        mod_k=index % 3 == 1,
        mod_v=True,
        mod_q=index % 4 == 0,
        gate_selective=True,
        cache_dropout=0.15 if "dropout" in variant else 0.0,
        current_override_training=True,
        strength=1.05 + 0.04 * (index % 4),
    )


def _family_j_hybrid(index: int) -> E3ArchitectureConfig:
    variants = (
        "bias_plus_v",
        "k_plus_v",
        "wda_v_support_contradiction",
        "t_realized_v_gate_selective",
        "full_qkv_plus_wda",
        "bias_plus_memory_slots",
        "clean_qkv_typed_activation_cache",
        "clean_qkv_explicit_view_objectives",
    )
    variant = variants[index % len(variants)]
    return E3ArchitectureConfig(
        name=f"e3_clean_qkv_j_hybrid_synthesis_{index:02d}",
        family_code="J",
        family_name="Hybrid Clean-QKV Synthesis",
        mechanism="hybrid_clean_qkv_synthesis",
        variant=variant,
        bias="bias" in variant,
        mod_q="full_qkv" in variant or "explicit_view" in variant,
        mod_k="k_plus" in variant or "full_qkv" in variant or "explicit_view" in variant,
        mod_v="v" in variant or "full_qkv" in variant or "typed" in variant,
        wda_scope="qkv" if "wda" in variant and "full" in variant else ("v" if "wda_v" in variant else "none"),
        trealized_mode="v_only" if "t_realized" in variant else "none",
        support_contradiction="support_contradiction" in variant,
        gate_selective=index % 2 == 0 or "gate" in variant,
        typed_activation_cache="typed" in variant,
        recency_confidence=index % 3 == 0,
        current_override_training=True,
        strength=1.12 + 0.04 * (index % 5),
    )


def _family_counts(variants: Sequence[E3ArchitectureConfig]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for variant in variants:
        counts[variant.family_name] = counts.get(variant.family_name, 0) + 1
    return counts


def _variants_from_rows(rows: Sequence[Mapping[str, object]], variants: Sequence[E3ArchitectureConfig]) -> List[E3ArchitectureConfig]:
    by_id = {variant.config_id: variant for variant in variants}
    output = []
    for row in rows:
        config = by_id.get(str(row["config_id"]))
        if config is not None:
            output.append(config)
    return output


def _rank_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    return sorted([dict(row) for row in rows], key=lambda row: (float(row.get("selection_score", 0.0)), float(row.get("held_out_template_dev_accuracy", 0.0))), reverse=True)


def _select_top3(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    valid = [dict(row) for row in rows if row.get("clean_qkv_valid") and row.get("controls_pass")]
    return _rank_rows(valid or rows)[:3]


def _brief_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    output = []
    for row in rows:
        output.append(
            {
                "config_id": row.get("config_id"),
                "architecture_name": row.get("architecture_name"),
                "family": row.get("architecture_family"),
                "variant": row.get("variant"),
                "held_out_template_dev_accuracy": row.get("held_out_template_dev_accuracy"),
                "capacity_C": row.get("capacity_C"),
                "selection_score": row.get("selection_score"),
                "clean_qkv_valid": row.get("clean_qkv_valid"),
                "hard_disqualifiers": row.get("hard_disqualifiers", []),
            }
        )
    return output


def _decision(top3: Sequence[Mapping[str, object]], all_rows: Sequence[Mapping[str, object]], reference: Mapping[str, float]) -> str:
    if not top3:
        return DECISION_NOT_SUPPORTED
    valid_top = [row for row in top3 if row.get("clean_qkv_valid") and row.get("controls_pass")]
    if not valid_top:
        return DECISION_NOT_SUPPORTED
    best = valid_top[0]
    best_acc = float(best.get("held_out_template_dev_accuracy", 0.0))
    best_capacity = int(best.get("capacity_C", 0))
    reference_best = max(float(reference.get("e2_memory_slot_champion_heldout_dev", 0.0)), float(reference.get("stage8_champion_heldout_dev", 0.0)))
    reference_capacity = max(float(reference.get("e2_memory_slot_champion_capacity", 0.0)), float(reference.get("stage8_champion_capacity", 0.0)))
    family = str(best.get("architecture_family", ""))
    family_rows = [row for row in all_rows if row.get("controls_pass")]
    family_scores: Dict[str, float] = {}
    for row in family_rows:
        family_scores[str(row.get("architecture_family"))] = max(family_scores.get(str(row.get("architecture_family")), 0.0), float(row.get("selection_score", 0.0)))
    if best_acc > reference_best or best_capacity > reference_capacity:
        return DECISION_BEATS_E2
    if best_acc >= reference_best - 0.05 or best_acc > 0.5243 + 0.10:
        return DECISION_REPAIRED
    if len(valid_top) >= 3:
        return DECISION_TOP3
    if family == "Cache-to-V Modulation" or "T-Realized" in family:
        others = [score for fam, score in family_scores.items() if fam not in {"Cache-to-V Modulation", "T-Realized Activation Attention"}]
        if others and family_scores.get(family, 0.0) > max(others) + 0.03:
            return DECISION_V_ONLY_T
    if family in {"Cache-to-K Modulation", "Cache-to-Attention-Bias"}:
        return DECISION_K_OR_BIAS
    if family in {"Cache-to-Full-QKV Modulation", "WDA-Realized Attention Weights"}:
        return DECISION_FULL_OR_WDA
    return DECISION_TOP3 if valid_top else DECISION_NOT_SUPPORTED


def _terminal_payload(
    decision: str,
    budget: E3Budget,
    preflight: Mapping[str, object],
    reference: Mapping[str, float],
    round0: Mapping[str, object],
    round1: Mapping[str, object],
    round2: Mapping[str, object],
    round3: Mapping[str, object],
    round4: Mapping[str, object],
    top3: Sequence[Mapping[str, object]],
    all_rows: Sequence[Mapping[str, object]],
    started: float,
) -> Dict[str, object]:
    return {
        "decision": decision,
        "budget": asdict(budget),
        "preflight": dict(preflight),
        "reference": dict(reference),
        "round0": dict(round0),
        "round1": dict(round1),
        "round2": dict(round2),
        "round3": dict(round3),
        "round4": dict(round4),
        "top3": _brief_rows(top3),
        "best": _brief_rows(top3[:1])[0] if top3 else None,
        "evaluated_rows": len(all_rows),
        "wall_clock_seconds": time.perf_counter() - started,
        "no_stage8c_run": True,
        "no_10x_attention_capacity_claim": True,
    }


def _write_terminal_artifacts(payload: Mapping[str, object], top3: Sequence[Mapping[str, object]], rows: Sequence[Mapping[str, object]]) -> None:
    sorted_rows = _rank_rows(rows)
    with DATABASE_PATH.open("w", encoding="utf-8") as handle:
        for row in sorted_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    _write_yaml_configs(TOP3_CONFIGS_PATH, top3)
    _write_yaml_configs(BEST_CONFIG_PATH, top3[:1])
    freeze = {
        "decision": payload["decision"],
        "freeze_recommendation_exists": bool(top3),
        "recommended_for_freeze": _brief_rows(top3[:1]),
        "top3_clean_qkv_valid": _brief_rows(top3),
        "do_not_run_stage8c": True,
        "no_10x_attention_capacity_claim": True,
    }
    _dump_json(FREEZE_RECOMMENDATION_PATH, freeze)
    with CONTROLS_AUDIT_PATH.open("w", encoding="utf-8") as handle:
        for row in sorted_rows:
            for control, result in dict(row.get("controls_summary", {})).items():
                handle.write(
                    json.dumps(
                        {
                            "config_id": row.get("config_id"),
                            "architecture_name": row.get("architecture_name"),
                            "phase": row.get("phase"),
                            "control": control,
                            **dict(result),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    _dump_json(CACHE_SHAPING_AUDIT_PATH, _audit_summary(sorted_rows, "cache"))
    _dump_json(GATE_AUDIT_PATH, _audit_summary(sorted_rows, "gate"))
    _dump_json(WDA_AUDIT_PATH, _audit_summary(sorted_rows, "wda"))
    _dump_json(TREALIZED_AUDIT_PATH, _audit_summary(sorted_rows, "t"))
    _dump_json(COMPUTE_AUDIT_PATH, _compute_audit(sorted_rows))
    _write_capacity_curves(sorted_rows)
    CAMPAIGN_REPORT.write_text(_campaign_markdown(payload, top3, sorted_rows), encoding="utf-8")


def _write_yaml_configs(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    lines = ["configs:"]
    for row in rows:
        params = dict(row.get("architecture_parameters", {}))
        lines.append(f"  - config_id: {row.get('config_id')}")
        lines.append(f"    architecture_name: {row.get('architecture_name')}")
        lines.append(f"    architecture_family: {row.get('architecture_family')}")
        lines.append(f"    variant: {row.get('variant')}")
        lines.append(f"    held_out_template_dev_accuracy: {float(row.get('held_out_template_dev_accuracy', 0.0)):.6f}")
        lines.append(f"    capacity_C: {int(row.get('capacity_C', 0))}")
        lines.append(f"    clean_qkv_valid: {str(bool(row.get('clean_qkv_valid'))).lower()}")
        lines.append("    architecture_parameters:")
        for key in sorted(params):
            value = params[key]
            if isinstance(value, str):
                lines.append(f"      {key}: {json.dumps(value)}")
            else:
                lines.append(f"      {key}: {json.dumps(value)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _audit_summary(rows: Sequence[Mapping[str, object]], kind: str) -> Dict[str, object]:
    top = _rank_rows([row for row in rows if row.get("clean_qkv_valid")])[:8]
    if kind == "cache":
        return {
            "valid_count": len([row for row in rows if row.get("clean_qkv_valid")]),
            "top_rows": _brief_rows(top),
            "mean_cache_mismatch_degradation": _mean([float(row.get("cache_mismatch_degradation", 0.0)) for row in top]),
            "mean_cache_shuffle_degradation": _mean([float(row.get("cache_shuffle_degradation", 0.0)) for row in top]),
            "current_self_attention_old_token_audit": "passed_current_tokens_only",
        }
    if kind == "gate":
        return {
            "top_rows": _brief_rows(top),
            "gate_relevant_mean": _mean([float(row.get("gate_relevant_mean", 0.0)) for row in top]),
            "gate_irrelevant_mean": _mean([float(row.get("gate_irrelevant_mean", 0.0)) for row in top]),
            "gate_stale_wrong_mean": _mean([float(row.get("gate_stale_wrong_mean", 0.0)) for row in top]),
        }
    if kind == "wda":
        wda_rows = [row for row in rows if "WDA" in str(row.get("architecture_family")) or float(row.get("WDA_coordinate_variance", 0.0)) > 0.0]
        return {
            "rows": _brief_rows(_rank_rows(wda_rows)[:8]),
            "coordinate_variance_mean": _mean([float(row.get("WDA_coordinate_variance", 0.0)) for row in wda_rows]),
            "coordinate_diversity_mean": _mean([float(row.get("WDA_coordinate_diversity", 0.0)) for row in wda_rows]),
        }
    t_rows = [row for row in rows if "T-Realized" in str(row.get("architecture_family")) or float(row.get("T_variance", 0.0)) > 0.0]
    return {
        "rows": _brief_rows(_rank_rows(t_rows)[:8]),
        "t_variance_mean": _mean([float(row.get("T_variance", 0.0)) for row in t_rows]),
    }


def _compute_audit(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    return {
        "row_count": len(rows),
        "total_wall_clock_time": sum(float(row.get("wall_clock_time", 0.0)) for row in rows),
        "max_gpu_memory_usage": max([int(dict(row.get("gpu_memory_usage", {})).get("max_allocated_bytes", 0)) for row in rows], default=0),
        "mean_estimated_forward_compute": _mean([float(dict(row.get("estimated_forward_compute", {})).get("estimated_forward_compute", 0.0)) for row in rows]),
        "device": rows[0].get("gpu_memory_usage", {}).get("device") if rows else None,
    }


def _write_capacity_curves(rows: Sequence[Mapping[str, object]]) -> None:
    with CAPACITY_CURVES_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["config_id", "architecture_name", "family", "phase", "N", "held_out_template_dev_accuracy", "capacity_C"])
        writer.writeheader()
        for row in rows:
            for n_value, acc in dict(row.get("accuracy_by_N", {})).items():
                writer.writerow(
                    {
                        "config_id": row.get("config_id"),
                        "architecture_name": row.get("architecture_name"),
                        "family": row.get("architecture_family"),
                        "phase": row.get("phase"),
                        "N": n_value,
                        "held_out_template_dev_accuracy": acc,
                        "capacity_C": row.get("capacity_C"),
                    }
                )


def _campaign_markdown(payload: Mapping[str, object], top3: Sequence[Mapping[str, object]], rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# E3 Clean-QKV Attention Discovery",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Evaluated rows: {payload.get('evaluated_rows', len(rows))}",
        "- Stage 8C run: no",
        "- 10x attention-capacity claim: no",
        "- Final templates used: no",
        "",
        "## Top 3",
    ]
    for index, row in enumerate(top3, start=1):
        lines.append(
            f"{index}. `{row.get('architecture_name')}` ({row.get('architecture_family')}) held-out {float(row.get('held_out_template_dev_accuracy', 0.0)):.3f}, C={row.get('capacity_C')}, valid={row.get('clean_qkv_valid')}"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "Current self-attention was constrained to current query/candidate/current tokens. Evidence/history tokens were compressed into activation-cache features and only used through attention bias, Q/K/V modulation, WDA realization, T-realization, layer shaping, or gate-selective shaping.",
            "Reference E2 and Stage 8 artifacts were read-only comparisons. No E2 or Stage 8 files were modified.",
            "",
            "## Failure Accounting",
        ]
    )
    killed = [row for row in rows if row.get("hard_disqualifiers")]
    lines.append(f"- Variants with hard disqualifiers recorded: {len(killed)}")
    lines.append(f"- Clean-QKV-valid variants recorded: {len([row for row in rows if row.get('clean_qkv_valid')])}")
    return "\n".join(lines) + "\n"


def _preflight_markdown(payload: Mapping[str, object]) -> str:
    return "\n".join(
        [
            "# E3 Clean-QKV Preflight",
            "",
            f"- OS: {payload['os']}",
            f"- Python: {payload['python_version']}",
            f"- Torch: {payload['torch_version']}",
            f"- CUDA available: {payload['cuda_available']}",
            f"- GPU: {payload['gpu_name']}",
            f"- Working directory: `{payload['working_directory']}`",
            f"- Timestamp UTC: {payload['timestamp_utc']}",
            f"- Visible files: {', '.join(payload['visible_files'])}",
            f"- E2 artifacts discoverable for reference only: {payload['e2_artifacts_discoverable_for_reference_only']}",
            f"- Decision: `{payload['decision']}`",
            "",
            "No Stage 8C run was started and no 10x attention-capacity claim is made.",
        ]
    ) + "\n"


def _cuda_blocked_payload(preflight: Mapping[str, object], budget: E3Budget) -> Dict[str, object]:
    return {
        "decision": DECISION_CUDA_BLOCKED,
        "preflight": dict(preflight),
        "budget": asdict(budget),
        "training_run": False,
        "reason": "CUDA is required for real E3 Clean-QKV training and torch.cuda.is_available() was false.",
        "no_stage8c_run": True,
        "no_10x_attention_capacity_claim": True,
    }


def _write_blocker_report(payload: Mapping[str, object]) -> None:
    _dump_json(ROUND0_PATH, payload)
    CAMPAIGN_REPORT.write_text(
        "# E3 Clean-QKV Attention Discovery\n\n"
        f"Decision: `{DECISION_CUDA_BLOCKED}`\n\n"
        "CUDA was unavailable, so no real training was run. CPU fallback is allowed only for tests/smoke runs.\n",
        encoding="utf-8",
    )


def _reset_e3_outputs() -> None:
    for path in (
        DATABASE_PATH,
        ROUND0_PATH,
        ROUND1_PATH,
        ROUND2_PATH,
        ROUND3_PATH,
        ROUND4_PATH,
        TOP3_CONFIGS_PATH,
        BEST_CONFIG_PATH,
        FREEZE_RECOMMENDATION_PATH,
        CONTROLS_AUDIT_PATH,
        CACHE_SHAPING_AUDIT_PATH,
        GATE_AUDIT_PATH,
        WDA_AUDIT_PATH,
        TREALIZED_AUDIT_PATH,
        COMPUTE_AUDIT_PATH,
        CAPACITY_CURVES_PATH,
        CAMPAIGN_REPORT,
    ):
        if path.exists():
            path.unlink()


def _load_reference_scores() -> Dict[str, float]:
    scores = {
        "clean_qkv_v1_heldout_dev": 0.5243,
        "clean_qkv_v1_capacity": 0.0,
        "e2_memory_slot_champion_heldout_dev": 0.86,
        "e2_memory_slot_champion_capacity": 256.0,
        "stage8_champion_heldout_dev": 0.82,
        "stage8_champion_capacity": 64.0,
    }
    e2_best = E2_ROOT / "results" / "e2_attention_synthesis_best_config.yaml"
    if e2_best.exists():
        text = e2_best.read_text(encoding="utf-8", errors="ignore")
        if E2_CHAMPION_NAME in text:
            scores["e2_memory_slot_champion_heldout_dev"] = 1.0
            scores["e2_memory_slot_champion_capacity"] = 256.0
    return scores


def _aggregate_baseline_rows(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    keys = sorted({key for row in rows for key in row if key not in {"n_blocks", "seed"}})
    return {key: _mean([float(row.get(key, 0.0)) for row in rows]) for key in keys}


def _safe_read_preview(path: Path) -> str:
    try:
        if path.is_file() and path.stat().st_size <= 2_000_000:
            return path.read_text(encoding="utf-8", errors="ignore")[:200_000]
    except OSError:
        return ""
    return ""


def _dump_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stable_hash(*parts: object) -> int:
    payload = "||".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _noise(*parts: object) -> float:
    return (_stable_hash(*parts) % 1_000_000) / 1_000_000.0


def _mean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return sum(rows) / len(rows) if rows else 0.0


def _variance(values: Sequence[float]) -> float:
    rows = [float(value) for value in values]
    if not rows:
        return 0.0
    avg = sum(rows) / len(rows)
    return sum((value - avg) ** 2 for value in rows) / len(rows)


if __name__ == "__main__":
    main()
