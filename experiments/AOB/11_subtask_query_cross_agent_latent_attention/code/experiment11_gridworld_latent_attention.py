from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROLE_NAMES = ("planner", "reader", "critic", "actor")
ACTION_NAMES = ("U", "D", "L", "R")
ACTION_DELTAS: Dict[int, Tuple[int, int]] = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}

PAD = 0
BOS = 1
SEP = 2
TASK_GRIDWORLD = 3
ROLE_PLACEHOLDER = 4
OBSTACLES = 5
HISTORY = 6
MASK = 7
ROLE_BASE = 10
STEP_BASE = 20
CUR_BASE = 60
GOAL_BASE = 124
OBS_BASE = 188
ACTION_BASE = 260
ACTION_QUERY_BASE = 280
ACTION_DECISION_TOKEN = 284
VOCAB_SIZE = 300

ANSWER_NAMES = ("YES", "NO", "IRRELEVANT", "UNKNOWN")
ANSWER_YES = 0
ANSWER_NO = 1
ANSWER_IRRELEVANT = 2
ANSWER_UNKNOWN = 3


@dataclass(frozen=True)
class GridWorld:
    id: str
    split: str
    grid_size: int
    start: Tuple[int, int]
    goal: Tuple[int, int]
    obstacles: Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class StepExample:
    world_id: str
    split: str
    current: Tuple[int, int]
    goal: Tuple[int, int]
    obstacles: Tuple[Tuple[int, int], ...]
    history: Tuple[int, ...]
    step_index: int
    label: int


@dataclass(frozen=True)
class RunConfig:
    grid_size: int = 7
    obstacle_density: float = 0.16
    min_path_len: int = 4
    max_steps: int = 14
    max_seq_len: int = 64
    train_worlds: int = 192
    dev_worlds: int = 64
    recovery_states_per_world: int = 4
    d_model: int = 64
    n_layers: int = 3
    n_heads: int = 4
    ff_mult: int = 4
    batch_size: int = 96
    epochs: int = 8
    lr: float = 0.0015
    weight_decay: float = 0.0001
    cross_layers: Tuple[int, ...] = (1, 2)
    label_mode: str = "shortest"


@dataclass(frozen=True)
class MethodSpec:
    name: str
    roles: Tuple[int, ...]
    cross_mode: str
    shared_query: bool = False
    query_mode: str = "subtask"
    summary_mode: str = "last_token"
    cross_layers: Tuple[int, ...] = ()
    slot_count: int = 0
    read_topk: int = 0
    role_dropout: float = 0.0
    lora_rank: int = 0
    lora_basis_count: int = 0
    lora_alpha: float = 1.0
    lora_gate_mode: str = "softmax"
    private_view: bool = False
    history_private_view: bool = False
    qa_aux_weight: float = 0.0
    qa_temperature: float = 0.75
    action_query_tokens: bool = False
    action_query_hybrid: bool = False
    action_query_decision: bool = False


METHODS: Dict[str, MethodSpec] = {
    "single_actor": MethodSpec(
        name="single_actor",
        roles=(3,),
        cross_mode="none",
        shared_query=False,
    ),
    "role_clones_no_path": MethodSpec(
        name="role_clones_no_path",
        roles=(0, 1, 2, 3),
        cross_mode="none",
        shared_query=False,
    ),
    "capacity_control_self_read": MethodSpec(
        name="capacity_control_self_read",
        roles=(0, 1, 2, 3),
        cross_mode="self_only",
        shared_query=False,
    ),
    "cross_agent_latent_attention": MethodSpec(
        name="cross_agent_latent_attention",
        roles=(0, 1, 2, 3),
        cross_mode="cross_peer",
        shared_query=False,
    ),
    "shared_query_control": MethodSpec(
        name="shared_query_control",
        roles=(0, 1, 2, 3),
        cross_mode="cross_peer",
        shared_query=True,
    ),
    "cross_late_only": MethodSpec(
        name="cross_late_only",
        roles=(0, 1, 2, 3),
        cross_mode="cross_peer",
        cross_layers=(2,),
    ),
    "cross_every_layer": MethodSpec(
        name="cross_every_layer",
        roles=(0, 1, 2, 3),
        cross_mode="cross_peer",
        cross_layers=(0, 1, 2),
    ),
    "cross_subtask_state_query": MethodSpec(
        name="cross_subtask_state_query",
        roles=(0, 1, 2, 3),
        cross_mode="cross_peer",
        query_mode="subtask_state",
    ),
    "cross_causal_mean_summary": MethodSpec(
        name="cross_causal_mean_summary",
        roles=(0, 1, 2, 3),
        cross_mode="cross_peer",
        summary_mode="causal_mean",
    ),
    "cross_two_role_actor_critic": MethodSpec(
        name="cross_two_role_actor_critic",
        roles=(2, 3),
        cross_mode="cross_peer",
    ),
    "slot_self_read_m2": MethodSpec(
        name="slot_self_read_m2",
        roles=(0, 1, 2, 3),
        cross_mode="self_only",
        summary_mode="causal_mean",
        slot_count=2,
    ),
    "slot_bottleneck_m2": MethodSpec(
        name="slot_bottleneck_m2",
        roles=(0, 1, 2, 3),
        cross_mode="cross_peer",
        summary_mode="causal_mean",
        slot_count=2,
    ),
    "slot_bottleneck_m2_topk2": MethodSpec(
        name="slot_bottleneck_m2_topk2",
        roles=(0, 1, 2, 3),
        cross_mode="cross_peer",
        summary_mode="causal_mean",
        slot_count=2,
        read_topk=2,
    ),
    "slot_bottleneck_m2_dropout": MethodSpec(
        name="slot_bottleneck_m2_dropout",
        roles=(0, 1, 2, 3),
        cross_mode="cross_peer",
        summary_mode="causal_mean",
        slot_count=2,
        role_dropout=0.25,
    ),
    "slot_bottleneck_m4_topk2": MethodSpec(
        name="slot_bottleneck_m4_topk2",
        roles=(0, 1, 2, 3),
        cross_mode="cross_peer",
        summary_mode="causal_mean",
        slot_count=4,
        read_topk=2,
    ),
    "role_lora_no_path": MethodSpec(
        name="role_lora_no_path",
        roles=(0, 1, 2, 3),
        cross_mode="none",
        lora_rank=8,
        lora_basis_count=4,
        lora_alpha=8.0,
    ),
    "role_lora_self_read": MethodSpec(
        name="role_lora_self_read",
        roles=(0, 1, 2, 3),
        cross_mode="self_only",
        summary_mode="causal_mean",
        lora_rank=8,
        lora_basis_count=4,
        lora_alpha=8.0,
    ),
    "role_lora_cross_causal_mean": MethodSpec(
        name="role_lora_cross_causal_mean",
        roles=(0, 1, 2, 3),
        cross_mode="cross_peer",
        summary_mode="causal_mean",
        lora_rank=8,
        lora_basis_count=4,
        lora_alpha=8.0,
    ),
    "role_lora_cross_slots_m2_topk2": MethodSpec(
        name="role_lora_cross_slots_m2_topk2",
        roles=(0, 1, 2, 3),
        cross_mode="cross_peer",
        summary_mode="causal_mean",
        slot_count=2,
        read_topk=2,
        lora_rank=8,
        lora_basis_count=4,
        lora_alpha=8.0,
    ),
    "role_lora_hard_no_path": MethodSpec(
        name="role_lora_hard_no_path",
        roles=(0, 1, 2, 3),
        cross_mode="none",
        lora_rank=8,
        lora_basis_count=4,
        lora_alpha=16.0,
        lora_gate_mode="fixed_role",
    ),
    "role_lora_hard_self_read": MethodSpec(
        name="role_lora_hard_self_read",
        roles=(0, 1, 2, 3),
        cross_mode="self_only",
        summary_mode="causal_mean",
        lora_rank=8,
        lora_basis_count=4,
        lora_alpha=16.0,
        lora_gate_mode="fixed_role",
    ),
    "role_lora_hard_cross_causal_mean": MethodSpec(
        name="role_lora_hard_cross_causal_mean",
        roles=(0, 1, 2, 3),
        cross_mode="cross_peer",
        summary_mode="causal_mean",
        lora_rank=8,
        lora_basis_count=4,
        lora_alpha=16.0,
        lora_gate_mode="fixed_role",
    ),
    "private_view_no_path": MethodSpec(
        name="private_view_no_path",
        roles=(0, 1, 2, 3),
        cross_mode="none",
        summary_mode="causal_mean",
        private_view=True,
    ),
    "private_view_cross_causal_mean": MethodSpec(
        name="private_view_cross_causal_mean",
        roles=(0, 1, 2, 3),
        cross_mode="cross_peer",
        summary_mode="causal_mean",
        private_view=True,
    ),
    "private_view_qa_soft": MethodSpec(
        name="private_view_qa_soft",
        roles=(0, 1, 2, 3),
        cross_mode="qa_soft",
        summary_mode="causal_mean",
        private_view=True,
    ),
    "private_view_qa_hard": MethodSpec(
        name="private_view_qa_hard",
        roles=(0, 1, 2, 3),
        cross_mode="qa_hard",
        summary_mode="causal_mean",
        private_view=True,
    ),
    "private_view_qa_hard_aux": MethodSpec(
        name="private_view_qa_hard_aux",
        roles=(0, 1, 2, 3),
        cross_mode="qa_hard",
        summary_mode="causal_mean",
        private_view=True,
        qa_aux_weight=0.35,
    ),
    "private_view_qa_hard_aux_late": MethodSpec(
        name="private_view_qa_hard_aux_late",
        roles=(0, 1, 2, 3),
        cross_mode="qa_hard",
        summary_mode="causal_mean",
        cross_layers=(2,),
        private_view=True,
        qa_aux_weight=0.35,
    ),
    "private_view_qa_hard_aux_every": MethodSpec(
        name="private_view_qa_hard_aux_every",
        roles=(0, 1, 2, 3),
        cross_mode="qa_hard",
        summary_mode="causal_mean",
        cross_layers=(0, 1, 2),
        private_view=True,
        qa_aux_weight=0.35,
    ),
    "private_view_qa_soft_bias": MethodSpec(
        name="private_view_qa_soft_bias",
        roles=(0, 1, 2, 3),
        cross_mode="qa_soft_bias",
        summary_mode="causal_mean",
        private_view=True,
    ),
    "private_view_qa_hard_aux_bias": MethodSpec(
        name="private_view_qa_hard_aux_bias",
        roles=(0, 1, 2, 3),
        cross_mode="qa_hard_bias",
        summary_mode="causal_mean",
        private_view=True,
        qa_aux_weight=0.35,
    ),
    "private_view_qa_hard_aux_bias_late": MethodSpec(
        name="private_view_qa_hard_aux_bias_late",
        roles=(0, 1, 2, 3),
        cross_mode="qa_hard_bias",
        summary_mode="causal_mean",
        cross_layers=(2,),
        private_view=True,
        qa_aux_weight=0.35,
    ),
    "private_view_qa_hard_aux_bias_every": MethodSpec(
        name="private_view_qa_hard_aux_bias_every",
        roles=(0, 1, 2, 3),
        cross_mode="qa_hard_bias",
        summary_mode="causal_mean",
        cross_layers=(0, 1, 2),
        private_view=True,
        qa_aux_weight=0.35,
    ),
    "full_view_qa_hard_aux": MethodSpec(
        name="full_view_qa_hard_aux",
        roles=(0, 1, 2, 3),
        cross_mode="qa_hard",
        summary_mode="causal_mean",
        qa_aux_weight=0.35,
    ),
    "private_view_action_tokens_no_path": MethodSpec(
        name="private_view_action_tokens_no_path",
        roles=(0, 1, 2, 3),
        cross_mode="none",
        summary_mode="causal_mean",
        private_view=True,
        action_query_tokens=True,
    ),
    "private_view_qa_token_soft": MethodSpec(
        name="private_view_qa_token_soft",
        roles=(0, 1, 2, 3),
        cross_mode="qa_soft_token",
        summary_mode="causal_mean",
        private_view=True,
        action_query_tokens=True,
    ),
    "private_view_qa_token_hard": MethodSpec(
        name="private_view_qa_token_hard",
        roles=(0, 1, 2, 3),
        cross_mode="qa_hard_token",
        summary_mode="causal_mean",
        private_view=True,
        action_query_tokens=True,
    ),
    "private_view_qa_token_hard_aux": MethodSpec(
        name="private_view_qa_token_hard_aux",
        roles=(0, 1, 2, 3),
        cross_mode="qa_hard_token",
        summary_mode="causal_mean",
        private_view=True,
        qa_aux_weight=0.35,
        action_query_tokens=True,
    ),
    "private_view_qa_token_hard_aux_late": MethodSpec(
        name="private_view_qa_token_hard_aux_late",
        roles=(0, 1, 2, 3),
        cross_mode="qa_hard_token",
        summary_mode="causal_mean",
        cross_layers=(2,),
        private_view=True,
        qa_aux_weight=0.35,
        action_query_tokens=True,
    ),
    "private_view_qa_token_hard_aux_every": MethodSpec(
        name="private_view_qa_token_hard_aux_every",
        roles=(0, 1, 2, 3),
        cross_mode="qa_hard_token",
        summary_mode="causal_mean",
        cross_layers=(0, 1, 2),
        private_view=True,
        qa_aux_weight=0.35,
        action_query_tokens=True,
    ),
    "full_view_qa_token_hard_aux": MethodSpec(
        name="full_view_qa_token_hard_aux",
        roles=(0, 1, 2, 3),
        cross_mode="qa_hard_token",
        summary_mode="causal_mean",
        qa_aux_weight=0.35,
        action_query_tokens=True,
    ),
    "private_view_action_tokens_hybrid_no_path": MethodSpec(
        name="private_view_action_tokens_hybrid_no_path",
        roles=(0, 1, 2, 3),
        cross_mode="none",
        summary_mode="causal_mean",
        private_view=True,
        action_query_tokens=True,
        action_query_hybrid=True,
    ),
    "private_view_qa_token_hard_aux_hybrid": MethodSpec(
        name="private_view_qa_token_hard_aux_hybrid",
        roles=(0, 1, 2, 3),
        cross_mode="qa_hard_token",
        summary_mode="causal_mean",
        private_view=True,
        qa_aux_weight=0.35,
        action_query_tokens=True,
        action_query_hybrid=True,
    ),
    "private_view_qa_token_hard_aux_hybrid_late": MethodSpec(
        name="private_view_qa_token_hard_aux_hybrid_late",
        roles=(0, 1, 2, 3),
        cross_mode="qa_hard_token",
        summary_mode="causal_mean",
        cross_layers=(2,),
        private_view=True,
        qa_aux_weight=0.35,
        action_query_tokens=True,
        action_query_hybrid=True,
    ),
    "private_view_qa_token_hard_aux_hybrid_every": MethodSpec(
        name="private_view_qa_token_hard_aux_hybrid_every",
        roles=(0, 1, 2, 3),
        cross_mode="qa_hard_token",
        summary_mode="causal_mean",
        cross_layers=(0, 1, 2),
        private_view=True,
        qa_aux_weight=0.35,
        action_query_tokens=True,
        action_query_hybrid=True,
    ),
    "full_view_qa_token_hard_aux_hybrid": MethodSpec(
        name="full_view_qa_token_hard_aux_hybrid",
        roles=(0, 1, 2, 3),
        cross_mode="qa_hard_token",
        summary_mode="causal_mean",
        qa_aux_weight=0.35,
        action_query_tokens=True,
        action_query_hybrid=True,
    ),
    "private_view_action_tokens_decision_no_path": MethodSpec(
        name="private_view_action_tokens_decision_no_path",
        roles=(0, 1, 2, 3),
        cross_mode="none",
        summary_mode="causal_mean",
        private_view=True,
        action_query_tokens=True,
        action_query_decision=True,
    ),
    "private_view_qa_token_hard_aux_decision": MethodSpec(
        name="private_view_qa_token_hard_aux_decision",
        roles=(0, 1, 2, 3),
        cross_mode="qa_hard_token",
        summary_mode="causal_mean",
        cross_layers=(0, 1),
        private_view=True,
        qa_aux_weight=0.35,
        action_query_tokens=True,
        action_query_decision=True,
    ),
    "private_view_qa_token_hard_aux_decision_every": MethodSpec(
        name="private_view_qa_token_hard_aux_decision_every",
        roles=(0, 1, 2, 3),
        cross_mode="qa_hard_token",
        summary_mode="causal_mean",
        cross_layers=(0, 1, 2),
        private_view=True,
        qa_aux_weight=0.35,
        action_query_tokens=True,
        action_query_decision=True,
    ),
    "full_view_qa_token_hard_aux_decision": MethodSpec(
        name="full_view_qa_token_hard_aux_decision",
        roles=(0, 1, 2, 3),
        cross_mode="qa_hard_token",
        summary_mode="causal_mean",
        cross_layers=(0, 1),
        qa_aux_weight=0.35,
        action_query_tokens=True,
        action_query_decision=True,
    ),
    "private_view_history_no_path": MethodSpec(
        name="private_view_history_no_path",
        roles=(0, 1, 2, 3),
        cross_mode="none",
        summary_mode="causal_mean",
        private_view=True,
        history_private_view=True,
    ),
    "private_view_history_cross_causal_mean": MethodSpec(
        name="private_view_history_cross_causal_mean",
        roles=(0, 1, 2, 3),
        cross_mode="cross_peer",
        summary_mode="causal_mean",
        private_view=True,
        history_private_view=True,
    ),
    "private_view_shared_bus": MethodSpec(
        name="private_view_shared_bus",
        roles=(0, 1, 2, 3),
        cross_mode="shared_bus",
        summary_mode="causal_mean",
        private_view=True,
        history_private_view=True,
    ),
    "private_view_shared_bus_late": MethodSpec(
        name="private_view_shared_bus_late",
        roles=(0, 1, 2, 3),
        cross_mode="shared_bus",
        summary_mode="causal_mean",
        cross_layers=(2,),
        private_view=True,
        history_private_view=True,
    ),
    "private_view_shared_bus_every": MethodSpec(
        name="private_view_shared_bus_every",
        roles=(0, 1, 2, 3),
        cross_mode="shared_bus",
        summary_mode="causal_mean",
        cross_layers=(0, 1, 2),
        private_view=True,
        history_private_view=True,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 11 minimal agentic gridworld screen.")
    parser.add_argument("--mode", choices=("smoke", "screen", "variant_screen"), default="smoke")
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/AOB/11_subtask_query_cross_agent_latent_attention/results/stage11a_gridworld"))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seeds", type=int, nargs="*", default=[101, 102, 103])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-worlds", type=int, default=None)
    parser.add_argument("--dev-worlds", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--label-mode", choices=("shortest", "history_tiebreak"), default=None)
    parser.add_argument("--methods", type=str, nargs="*", default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--report-title", type=str, default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    config = RunConfig()
    if args.mode == "smoke":
        config = replace_run_config(
            config,
            train_worlds=24,
            dev_worlds=10,
            epochs=2,
            batch_size=24,
            d_model=32,
            n_layers=2,
            n_heads=4,
            max_steps=10,
            cross_layers=(0, 1),
            label_mode=args.label_mode if args.label_mode is not None else config.label_mode,
        )
        seeds = [args.seeds[0] if args.seeds else 101]
        run_smoke(config, seeds[0], device, args.out_dir)
    elif args.mode == "screen":
        config = replace_run_config(
            config,
            epochs=args.epochs if args.epochs is not None else config.epochs,
            train_worlds=args.train_worlds if args.train_worlds is not None else config.train_worlds,
            dev_worlds=args.dev_worlds if args.dev_worlds is not None else config.dev_worlds,
            batch_size=args.batch_size if args.batch_size is not None else config.batch_size,
            label_mode=args.label_mode if args.label_mode is not None else config.label_mode,
        )
        run_screen(config, args.seeds, device, args.out_dir, method_names=tuple(args.methods) if args.methods else None)
    else:
        config = replace_run_config(
            config,
            epochs=args.epochs if args.epochs is not None else config.epochs,
            train_worlds=args.train_worlds if args.train_worlds is not None else config.train_worlds,
            dev_worlds=args.dev_worlds if args.dev_worlds is not None else config.dev_worlds,
            batch_size=args.batch_size if args.batch_size is not None else config.batch_size,
            label_mode=args.label_mode if args.label_mode is not None else config.label_mode,
        )
        run_variant_screen(
            config,
            args.seeds,
            device,
            args.out_dir,
            method_names=tuple(args.methods) if args.methods else None,
            report_path=args.report_path,
            report_title=args.report_title,
        )


def replace_run_config(config: RunConfig, **updates: object) -> RunConfig:
    values = asdict(config)
    values.update(updates)
    if isinstance(values.get("cross_layers"), list):
        values["cross_layers"] = tuple(int(v) for v in values["cross_layers"])
    return RunConfig(**values)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must divide n_heads={n_heads}")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = int(d_model // n_heads)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.out(y)


class CausalDecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(ff_mult) * d_model),
            nn.GELU(),
            nn.Linear(int(ff_mult) * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class RoleGatedLoRAAdapter(nn.Module):
    """Shared low-rank bases with role-conditioned gates to break role-axis symmetry."""

    def __init__(self, d_model: int, rank: int, basis_count: int, alpha: float, gate_mode: str) -> None:
        super().__init__()
        self.rank = int(rank)
        self.basis_count = int(basis_count)
        self.scale = float(alpha) / float(max(1, rank))
        self.gate_mode = str(gate_mode)
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Parameter(torch.empty(self.basis_count, d_model, self.rank))
        self.up = nn.Parameter(torch.empty(self.basis_count, self.rank, d_model))
        if self.gate_mode == "fixed_role":
            self.role_basis_logits = None
        else:
            self.role_basis_logits = nn.Embedding(len(ROLE_NAMES), self.basis_count)
        nn.init.normal_(self.down, mean=0.0, std=1.0 / math.sqrt(max(1, d_model)))
        nn.init.normal_(self.up, mean=0.0, std=1.0e-3)
        if self.role_basis_logits is not None:
            nn.init.normal_(self.role_basis_logits.weight, mean=0.0, std=0.02)
        self.last_stats: Dict[str, object] = {}

    def forward(self, x: torch.Tensor, role_ids: torch.Tensor) -> torch.Tensor:
        batch, roles, seq_len, dim = x.shape
        x_n = self.norm(x)
        low = torch.einsum("brtd,kdq->brtkq", x_n, self.down)
        basis_delta = torch.einsum("brtkq,kqd->brtkd", low, self.up)
        if self.gate_mode == "fixed_role":
            gates = F.one_hot(role_ids.remainder(self.basis_count), num_classes=self.basis_count).to(dtype=x.dtype)
        else:
            if self.role_basis_logits is None:
                raise RuntimeError("soft LoRA gates requested without role gate logits")
            gates = F.softmax(self.role_basis_logits(role_ids), dim=-1)
        delta = torch.einsum("rk,brtkd->brtd", gates, basis_delta) * self.scale
        out = x + delta
        with torch.no_grad():
            base_norm = x.norm(dim=-1).mean().clamp_min(1.0e-9)
            delta_norm = delta.norm(dim=-1).mean()
            gate_entropy = -(gates.clamp_min(1.0e-9) * gates.clamp_min(1.0e-9).log()).sum(dim=-1)
            self.last_stats = {
                "lora_delta_norm_ratio": float((delta_norm / base_norm).detach().cpu().item()),
                "lora_gate_entropy": float(gate_entropy.mean().detach().cpu().item()),
                "lora_role_basis_gates": gates.detach().cpu().tolist(),
            }
        return out


class CrossAgentLatentBlock(nn.Module):
    def __init__(self, d_model: int, role_count: int, slot_count: int = 0) -> None:
        super().__init__()
        self.role_query = nn.Embedding(len(ROLE_NAMES), d_model)
        self.summary_norm = nn.LayerNorm(d_model)
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.inject = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model * 2, d_model)
        self.action_query = nn.Embedding(len(ACTION_NAMES), d_model)
        self.answer_embedding = nn.Embedding(len(ANSWER_NAMES), d_model)
        self.answer_head = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, len(ANSWER_NAMES)),
        )
        self.answer_action_score = nn.Linear(d_model, 1)
        self.shared_query = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.shared_query, mean=0.0, std=1.0 / math.sqrt(max(1, d_model)))
        self.role_count = int(role_count)
        self.slot_count = max(0, int(slot_count))
        if self.slot_count:
            self.slot_query = nn.Parameter(torch.empty(self.slot_count, d_model))
            self.slot_token_norm = nn.LayerNorm(d_model)
            self.slot_token_key = nn.Linear(d_model, d_model, bias=False)
            self.slot_token_value = nn.Linear(d_model, d_model, bias=False)
            nn.init.normal_(self.slot_query, mean=0.0, std=1.0 / math.sqrt(max(1, d_model)))
        else:
            self.register_parameter("slot_query", None)
            self.slot_token_norm = None
            self.slot_token_key = None
            self.slot_token_value = None
        self.last_stats: Dict[str, object] = {}
        self.last_answer_logits_tensor: torch.Tensor | None = None
        self.last_answer_source_mask_tensor: torch.Tensor | None = None
        self.last_action_bias_tensor: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        role_ids: torch.Tensor,
        last_indices: torch.Tensor,
        mode: str,
        shared_query: bool,
        query_mode: str,
        summary_mode: str,
        read_topk: int = 0,
        role_dropout: float = 0.0,
        qa_temperature: float = 0.75,
        action_query_indices: torch.Tensor | None = None,
        control: str = "none",
    ) -> torch.Tensor:
        self.last_answer_logits_tensor = None
        self.last_answer_source_mask_tensor = None
        self.last_action_bias_tensor = None
        if mode == "none" or control == "remove_path":
            self.last_stats = {
                "attention_entropy": 0.0,
                "inject_norm_ratio": 0.0,
                "attention_mass": [],
                "qa_answer_entropy": 0.0,
                "qa_yes_rate": 0.0,
                "qa_no_rate": 0.0,
                "qa_irrelevant_rate": 0.0,
            }
            return x

        batch, roles, _seq_len, dim = x.shape
        summary = summarize_role_states(x, last_indices, summary_mode)
        summary_n = self.summary_norm(summary)
        if mode.startswith("qa_"):
            return self._forward_qa_relay(
                x=x,
                role_ids=role_ids,
                mode=mode,
                summary=summary,
                summary_n=summary_n,
                qa_temperature=qa_temperature,
                action_query_indices=action_query_indices,
                control=control,
            )
        if self.slot_count:
            return self._forward_slots(
                x=x,
                role_ids=role_ids,
                last_indices=last_indices,
                mode=mode,
                shared_query=shared_query,
                query_mode=query_mode,
                summary_n=summary_n,
                summary=summary,
                read_topk=read_topk,
                role_dropout=role_dropout,
                control=control,
            )
        if shared_query:
            query_source = self.shared_query.view(1, 1, dim).expand(batch, roles, dim)
        elif query_mode == "subtask_state":
            query_source = self.role_query(role_ids).view(1, roles, dim).expand(batch, roles, dim) + summary_n
        else:
            query_source = self.role_query(role_ids).view(1, roles, dim).expand(batch, roles, dim)
        q = self.wq(query_source)
        k = self.wk(summary_n)
        v = self.wv(summary_n)

        if control == "role_kv_shuffle" and roles > 1:
            k = torch.roll(k, shifts=1, dims=1)
            v = torch.roll(v, shifts=1, dims=1)
        if control == "cross_task_kv_shuffle" and batch > 1:
            k = torch.roll(k, shifts=1, dims=0)
            v = torch.roll(v, shifts=1, dims=0)

        logits = torch.einsum("bid,bjd->bij", q, k) / math.sqrt(max(1, dim))
        allowed = allowed_source_matrix(
            roles=roles,
            role_ids=role_ids,
            mode=mode,
            mask_actor_peer_reads=control == "mask_actor_peer_reads",
            device=x.device,
        )
        logits = logits.masked_fill(~allowed.view(1, roles, roles), -1.0e9)
        weights = F.softmax(logits, dim=-1)
        context = torch.einsum("bij,bjd->bid", weights, v)
        update = self.inject(context)
        gate = torch.sigmoid(self.gate(torch.cat([summary, context], dim=-1)))
        update = gate * update
        out = x + update.unsqueeze(2)

        with torch.no_grad():
            entropy = -(weights.clamp_min(1.0e-9) * weights.clamp_min(1.0e-9).log()).sum(dim=-1)
            summary_norm = summary.norm(dim=-1).mean().clamp_min(1.0e-9)
            inject_norm = update.norm(dim=-1).mean()
            self.last_stats = {
                "attention_entropy": float(entropy.mean().detach().cpu().item()),
                "inject_norm_ratio": float((inject_norm / summary_norm).detach().cpu().item()),
                "attention_mass": weights.mean(dim=0).detach().cpu().tolist(),
            }
        return out

    def _forward_qa_relay(
        self,
        x: torch.Tensor,
        role_ids: torch.Tensor,
        mode: str,
        summary: torch.Tensor,
        summary_n: torch.Tensor,
        qa_temperature: float,
        action_query_indices: torch.Tensor | None,
        control: str,
    ) -> torch.Tensor:
        batch, roles, seq_len, dim = x.shape
        actor_positions = torch.nonzero(role_ids.eq(3), as_tuple=False).flatten()
        if actor_positions.numel() == 0:
            self.last_stats = {
                "attention_entropy": 0.0,
                "inject_norm_ratio": 0.0,
                "attention_mass": [],
                "qa_answer_entropy": 0.0,
                "qa_yes_rate": 0.0,
                "qa_no_rate": 0.0,
                "qa_irrelevant_rate": 0.0,
            }
            return x
        actor_pos = int(actor_positions[0].detach().cpu().item())
        source_summary = summary_n
        if control == "role_kv_shuffle" and roles > 1:
            source_summary = torch.roll(source_summary, shifts=1, dims=1)
        if control == "cross_task_kv_shuffle" and batch > 1:
            source_summary = torch.roll(source_summary, shifts=1, dims=0)

        action_ids = torch.arange(len(ACTION_NAMES), dtype=torch.long, device=x.device)
        action_query = self.action_query(action_ids)
        token_mode = "token" in mode and action_query_indices is not None
        if token_mode:
            actor_action_states = gather_sequence_positions(x[:, actor_pos, :, :], action_query_indices)
            actor_query = self.wq(actor_action_states + action_query.view(1, len(ACTION_NAMES), dim))
        else:
            actor_action_states = None
            actor_query = self.wq(summary_n[:, actor_pos : actor_pos + 1, :] + action_query.view(1, len(ACTION_NAMES), dim))
        source_expanded = source_summary.unsqueeze(2).expand(batch, roles, len(ACTION_NAMES), dim)
        query_expanded = actor_query.unsqueeze(1).expand(batch, roles, len(ACTION_NAMES), dim)
        action_expanded = action_query.view(1, 1, len(ACTION_NAMES), dim).expand(batch, roles, -1, -1)
        answer_logits = self.answer_head(torch.cat([source_expanded, query_expanded, action_expanded], dim=-1))

        source_mask = role_ids.ne(3)
        if control == "mask_actor_peer_reads":
            source_mask = torch.zeros_like(source_mask)
        self.last_answer_logits_tensor = answer_logits
        self.last_answer_source_mask_tensor = source_mask

        answer_soft = F.softmax(answer_logits, dim=-1)
        if mode.startswith("qa_soft"):
            answer = answer_soft
        else:
            if self.training:
                answer = F.gumbel_softmax(answer_logits, tau=float(qa_temperature), hard=True, dim=-1)
            else:
                answer = F.one_hot(answer_logits.argmax(dim=-1), num_classes=len(ANSWER_NAMES)).to(dtype=x.dtype)
        if control == "action_answer_shuffle":
            answer = torch.roll(answer, shifts=1, dims=2)

        answer_vec = torch.einsum("brac,cd->brad", answer, self.answer_embedding.weight)
        role_vec = self.role_query(role_ids).view(1, roles, 1, dim)
        action_vec = action_query.view(1, 1, len(ACTION_NAMES), dim)
        message = answer_vec + role_vec + action_vec
        message = message * source_mask.view(1, roles, 1, 1).to(dtype=message.dtype)
        source_denom = source_mask.to(dtype=message.dtype).sum().clamp_min(1.0)
        context_by_action = message.sum(dim=1) / source_denom
        if "bias" in mode:
            self.last_action_bias_tensor = self.answer_action_score(context_by_action).squeeze(-1)
        if token_mode and actor_action_states is not None:
            update_by_action = self.inject(context_by_action)
            gate_by_action = torch.sigmoid(self.gate(torch.cat([actor_action_states, context_by_action], dim=-1)))
            update_by_action = gate_by_action * update_by_action
            out = x.clone()
            actor_stream = out[:, actor_pos, :, :]
            index = action_query_indices.view(batch, len(ACTION_NAMES), 1).expand(-1, -1, dim)
            actor_stream = actor_stream.scatter_add(1, index, update_by_action)
            out[:, actor_pos, :, :] = actor_stream
            update_for_stats = update_by_action
        else:
            update_for_stats = None
        denom = (source_denom * float(len(ACTION_NAMES))).clamp_min(1.0)
        context = message.sum(dim=(1, 2)) / denom
        actor_summary = summary[:, actor_pos, :]
        if not token_mode:
            update = self.inject(context)
            gate = torch.sigmoid(self.gate(torch.cat([actor_summary, context], dim=-1)))
            update = gate * update
            out = x.clone()
            out[:, actor_pos, :, :] = out[:, actor_pos, :, :] + update.view(batch, 1, dim).expand(-1, seq_len, -1)
            update_for_stats = update

        with torch.no_grad():
            mask = source_mask.view(1, roles, 1).expand(batch, roles, len(ACTION_NAMES))
            if bool(mask.any().detach().cpu().item()):
                masked_soft = answer_soft[mask]
                entropy = -(masked_soft.clamp_min(1.0e-9) * masked_soft.clamp_min(1.0e-9).log()).sum(dim=-1)
                answer_rates = masked_soft.mean(dim=0)
            else:
                entropy = torch.zeros((), dtype=x.dtype, device=x.device)
                answer_rates = torch.zeros(len(ANSWER_NAMES), dtype=x.dtype, device=x.device)
            if token_mode and actor_action_states is not None:
                summary_norm = actor_action_states.norm(dim=-1).mean().clamp_min(1.0e-9)
            else:
                summary_norm = actor_summary.norm(dim=-1).mean().clamp_min(1.0e-9)
            inject_norm = update_for_stats.norm(dim=-1).mean() if update_for_stats is not None else torch.zeros((), dtype=x.dtype, device=x.device)
            mass = torch.zeros((roles, roles), dtype=x.dtype, device=x.device)
            if bool(source_mask.any().detach().cpu().item()):
                mass[actor_pos, :] = source_mask.to(dtype=x.dtype) / source_mask.to(dtype=x.dtype).sum().clamp_min(1.0)
            self.last_stats = {
                "attention_entropy": float(entropy.mean().detach().cpu().item()),
                "inject_norm_ratio": float((inject_norm / summary_norm).detach().cpu().item()),
                "attention_mass": mass.detach().cpu().tolist(),
                "qa_answer_entropy": float(entropy.mean().detach().cpu().item()),
                "qa_yes_rate": float(answer_rates[ANSWER_YES].detach().cpu().item()),
                "qa_no_rate": float(answer_rates[ANSWER_NO].detach().cpu().item()),
                "qa_irrelevant_rate": float(answer_rates[ANSWER_IRRELEVANT].detach().cpu().item()),
            }
        return out

    def _forward_slots(
        self,
        x: torch.Tensor,
        role_ids: torch.Tensor,
        last_indices: torch.Tensor,
        mode: str,
        shared_query: bool,
        query_mode: str,
        summary_n: torch.Tensor,
        summary: torch.Tensor,
        read_topk: int,
        role_dropout: float,
        control: str,
    ) -> torch.Tensor:
        if self.slot_query is None or self.slot_token_norm is None or self.slot_token_key is None or self.slot_token_value is None:
            raise RuntimeError("slot bottleneck requested without slot modules")
        batch, roles, seq_len, dim = x.shape
        token_states = self.slot_token_norm(x)
        token_k = self.slot_token_key(token_states)
        token_v = self.slot_token_value(token_states)
        slot_q = self.slot_query.view(1, 1, self.slot_count, dim)
        slot_logits = torch.einsum("brmd,brtd->brmt", slot_q.expand(batch, roles, -1, -1), token_k) / math.sqrt(max(1, dim))
        positions = torch.arange(seq_len, dtype=torch.long, device=x.device).view(1, 1, 1, seq_len)
        token_mask = positions <= last_indices.view(batch, 1, 1, 1)
        slot_logits = slot_logits.masked_fill(~token_mask, -1.0e9)
        slot_weights = F.softmax(slot_logits, dim=-1)
        slots = torch.einsum("brmt,brtd->brmd", slot_weights, token_v)

        if shared_query:
            query_source = self.shared_query.view(1, 1, dim).expand(batch, roles, dim)
        elif query_mode == "subtask_state":
            query_source = self.role_query(role_ids).view(1, roles, dim).expand(batch, roles, dim) + summary_n
        else:
            query_source = self.role_query(role_ids).view(1, roles, dim).expand(batch, roles, dim)
        q = self.wq(query_source)
        source_slots = slots
        if control == "role_kv_shuffle" and roles > 1:
            source_slots = torch.roll(source_slots, shifts=1, dims=1)
        k = self.wk(source_slots).reshape(batch, roles * self.slot_count, dim)
        v = self.wv(source_slots).reshape(batch, roles * self.slot_count, dim)
        if control == "cross_task_kv_shuffle" and batch > 1:
            k = torch.roll(k, shifts=1, dims=0)
            v = torch.roll(v, shifts=1, dims=0)

        logits = torch.einsum("bid,bsd->bis", q, k) / math.sqrt(max(1, dim))
        allowed = allowed_slot_source_matrix(
            roles=roles,
            slot_count=self.slot_count,
            role_ids=role_ids,
            mode=mode,
            mask_actor_peer_reads=control == "mask_actor_peer_reads",
            device=x.device,
        ).unsqueeze(0).expand(batch, -1, -1)
        if self.training and float(role_dropout) > 0.0 and mode == "cross_peer":
            keep_by_source_role = torch.rand(batch, roles, device=x.device) >= float(role_dropout)
            keep_by_source_slot = keep_by_source_role.repeat_interleave(self.slot_count, dim=1)
            dropped = allowed & keep_by_source_slot.unsqueeze(1)
            empty = ~dropped.any(dim=-1, keepdim=True)
            allowed = torch.where(empty, allowed, dropped)
        logits = logits.masked_fill(~allowed, -1.0e9)
        if int(read_topk) > 0 and int(read_topk) < roles * self.slot_count:
            k_count = max(1, min(int(read_topk), roles * self.slot_count))
            topk_indices = logits.topk(k_count, dim=-1).indices
            topk_mask = torch.zeros_like(allowed)
            topk_mask.scatter_(-1, topk_indices, True)
            logits = logits.masked_fill(~(allowed & topk_mask), -1.0e9)
        weights = F.softmax(logits, dim=-1)
        context = torch.einsum("bis,bsd->bid", weights, v)
        update = self.inject(context)
        gate = torch.sigmoid(self.gate(torch.cat([summary, context], dim=-1)))
        update = gate * update
        out = x + update.unsqueeze(2)

        with torch.no_grad():
            entropy = -(weights.clamp_min(1.0e-9) * weights.clamp_min(1.0e-9).log()).sum(dim=-1)
            summary_norm = summary.norm(dim=-1).mean().clamp_min(1.0e-9)
            inject_norm = update.norm(dim=-1).mean()
            role_mass = weights.view(batch, roles, roles, self.slot_count).sum(dim=-1).mean(dim=0)
            self.last_stats = {
                "attention_entropy": float(entropy.mean().detach().cpu().item()),
                "inject_norm_ratio": float((inject_norm / summary_norm).detach().cpu().item()),
                "attention_mass": role_mass.detach().cpu().tolist(),
                "slot_attention_entropy": float(
                    (-(slot_weights.clamp_min(1.0e-9) * slot_weights.clamp_min(1.0e-9).log()).sum(dim=-1)).mean().detach().cpu().item()
                ),
            }
        return out


class SharedCrossRoleBus(nn.Module):
    """Partitioned residual bus: each role owns one slot, every role reads the whole bus."""

    def __init__(self, d_model: int, role_count: int, bus_dim: int | None = None) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.role_count = int(role_count)
        self.bus_dim = int(bus_dim if bus_dim is not None else d_model)
        if self.bus_dim % self.role_count != 0:
            raise ValueError(f"bus_dim={self.bus_dim} must divide role_count={self.role_count}")
        self.slot_dim = self.bus_dim // self.role_count
        self.write_norm = nn.LayerNorm(d_model)
        self.read_norm = nn.LayerNorm(self.bus_dim)
        self.writers = nn.ModuleList([nn.Linear(d_model, self.slot_dim) for _ in range(self.role_count)])
        self.readers = nn.ModuleList([nn.Linear(self.bus_dim, d_model) for _ in range(self.role_count)])
        self.last_stats: Dict[str, object] = {}

    def forward(
        self,
        x: torch.Tensor,
        bus: torch.Tensor,
        role_ids: torch.Tensor,
        last_indices: torch.Tensor,
        summary_mode: str,
        control: str = "none",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, roles, seq_len, dim = x.shape
        if roles != self.role_count:
            raise ValueError(f"expected {self.role_count} roles, got {roles}")
        if control == "remove_path":
            self.last_stats = {
                "attention_entropy": 0.0,
                "inject_norm_ratio": 0.0,
                "attention_mass": [],
                "bus_write_norm_ratio": 0.0,
                "bus_read_norm_ratio": 0.0,
            }
            return x, torch.zeros_like(bus)

        summary = summarize_role_states(x, last_indices, summary_mode)
        summary_n = self.write_norm(summary)
        write_chunks: List[torch.Tensor] = []
        write_norms: List[torch.Tensor] = []
        zero_controls = {
            "role_write_zero_planner": 0,
            "role_write_zero_reader": 1,
            "role_write_zero_critic": 2,
            "role_write_zero_actor": 3,
            "ablation_compound_planner_zero_actor_mask": 0,
        }
        zero_role_id = zero_controls.get(control)
        for role_index, writer in enumerate(self.writers):
            chunk = writer(summary_n[:, role_index, :])
            if zero_role_id is not None and int(role_ids[role_index].detach().cpu().item()) == zero_role_id:
                chunk = torch.zeros_like(chunk)
            write_chunks.append(chunk)
            write_norms.append(chunk.norm(dim=-1))
        write = torch.cat(write_chunks, dim=-1)
        new_bus = bus + write
        if control == "cross_task_bus_shuffle" and batch > 1:
            new_bus = torch.roll(new_bus, shifts=1, dims=0)

        bus_for_read = self.read_norm(new_bus)
        read_updates: List[torch.Tensor] = []
        for role_index, reader in enumerate(self.readers):
            update = reader(bus_for_read)
            if control in {"mask_actor_peer_reads", "ablation_compound_planner_zero_actor_mask"} and int(role_ids[role_index].detach().cpu().item()) == 3:
                update = torch.zeros_like(update)
            read_updates.append(update)
        read = torch.stack(read_updates, dim=1)
        out = x + read.unsqueeze(2).expand(batch, roles, seq_len, dim)

        with torch.no_grad():
            summary_norm = summary.norm(dim=-1).mean().clamp_min(1.0e-9)
            write_norm = write.norm(dim=-1).mean()
            read_norm = read.norm(dim=-1).mean()
            slot_norm = torch.stack(write_norms, dim=1).mean(dim=0)
            source_mass = slot_norm / slot_norm.sum().clamp_min(1.0e-9)
            mass = source_mass.view(1, roles).expand(roles, roles)
            self.last_stats = {
                "attention_entropy": 0.0,
                "inject_norm_ratio": float((read_norm / summary_norm).detach().cpu().item()),
                "attention_mass": mass.detach().cpu().tolist(),
                "bus_write_norm_ratio": float((write_norm / summary_norm).detach().cpu().item()),
                "bus_read_norm_ratio": float((read_norm / summary_norm).detach().cpu().item()),
            }
        return out, new_bus


class RoleStreamAgent(nn.Module):
    def __init__(self, config: RunConfig, method: MethodSpec) -> None:
        super().__init__()
        self.config = config
        self.method = method
        self.role_ids_tuple = tuple(int(v) for v in method.roles)
        self.actor_role_id = 3
        self.actor_index = self.role_ids_tuple.index(self.actor_role_id)
        self.token_embedding = nn.Embedding(VOCAB_SIZE, config.d_model, padding_idx=PAD)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList(
            [
                CausalDecoderBlock(config.d_model, config.n_heads, config.ff_mult)
                for _ in range(config.n_layers)
            ]
        )
        self.role_lora_adapters = nn.ModuleDict()
        if method.lora_rank > 0 and method.lora_basis_count > 0:
            self.role_lora_adapters.update(
                {
                    str(layer_id): RoleGatedLoRAAdapter(
                        config.d_model,
                        rank=method.lora_rank,
                        basis_count=method.lora_basis_count,
                        alpha=method.lora_alpha,
                        gate_mode=method.lora_gate_mode,
                    )
                    for layer_id in range(config.n_layers)
                }
            )
        requested_cross_layers = method.cross_layers if method.cross_layers else config.cross_layers
        self.cross_layers = tuple(int(v) for v in requested_cross_layers if 0 <= int(v) < config.n_layers)
        self.uses_shared_bus = method.cross_mode == "shared_bus"
        self.bus_dim = int(config.d_model)
        self.cross_blocks = nn.ModuleDict()
        self.shared_bus_blocks = nn.ModuleDict()
        if method.cross_mode != "none" and not self.uses_shared_bus:
            self.cross_blocks.update(
                {
                    str(layer_id): CrossAgentLatentBlock(config.d_model, len(self.role_ids_tuple), slot_count=method.slot_count)
                    for layer_id in self.cross_layers
                }
            )
        if self.uses_shared_bus:
            self.shared_bus_blocks.update(
                {
                    str(layer_id): SharedCrossRoleBus(config.d_model, len(self.role_ids_tuple), bus_dim=self.bus_dim)
                    for layer_id in self.cross_layers
                }
            )
            self.bus_final_norm = nn.LayerNorm(self.bus_dim)
            self.bus_action_head = nn.Linear(config.d_model + self.bus_dim, len(ACTION_NAMES))
        else:
            self.bus_final_norm = None
            self.bus_action_head = None
        self.final_norm = nn.LayerNorm(config.d_model)
        self.action_head = nn.Linear(config.d_model, len(ACTION_NAMES))
        self.action_token_head = nn.Linear(config.d_model, 1)
        self._last_debug: Dict[str, object] = {}
        self._last_qa_aux_tensors: List[Dict[str, torch.Tensor]] = []

    def forward(self, base_ids: torch.Tensor, last_indices: torch.Tensor, control: str = "none") -> torch.Tensor:
        batch, seq_len = base_ids.shape
        role_ids = torch.as_tensor(self.role_ids_tuple, dtype=torch.long, device=base_ids.device)
        role_tokens = role_token_ids(role_ids).view(1, -1, 1).expand(batch, -1, 1)
        ids = base_ids.unsqueeze(1).expand(-1, len(self.role_ids_tuple), -1).clone()
        ids[:, :, 1:2] = role_tokens
        if self.method.private_view:
            ids = apply_private_role_view_masks(ids, role_ids, history_private=self.method.history_private_view)
        action_query_indices: torch.Tensor | None = None
        action_decision_indices: torch.Tensor | None = None
        if self.method.action_query_tokens:
            ids, action_query_indices, action_decision_indices = insert_action_query_tokens(
                ids,
                last_indices,
                include_decision=self.method.action_query_decision,
            )
        positions = torch.arange(seq_len, dtype=torch.long, device=base_ids.device)
        hidden = self.token_embedding(ids) + self.position_embedding(positions).view(1, 1, seq_len, -1)
        bus = hidden.new_zeros(batch, self.bus_dim) if self.uses_shared_bus else None

        diagnostics: List[Dict[str, object]] = []
        lora_diagnostics: List[Dict[str, object]] = []
        qa_aux_tensors: List[Dict[str, torch.Tensor]] = []
        qa_action_biases: List[torch.Tensor] = []
        for layer_id, block in enumerate(self.blocks):
            flat = hidden.reshape(batch * len(self.role_ids_tuple), seq_len, self.config.d_model)
            flat = block(flat)
            hidden = flat.view(batch, len(self.role_ids_tuple), seq_len, self.config.d_model)
            lora_key = str(layer_id)
            adapter = self.role_lora_adapters[lora_key] if lora_key in self.role_lora_adapters else None
            if adapter is not None:
                hidden = adapter(hidden, role_ids=role_ids)
                lora_diagnostics.append({"layer": layer_id, **adapter.last_stats})
            cross_key = str(layer_id)
            cross = self.cross_blocks[cross_key] if cross_key in self.cross_blocks else None
            if cross is not None:
                hidden = cross(
                    hidden,
                    role_ids=role_ids,
                    last_indices=last_indices,
                    mode=self.method.cross_mode,
                    shared_query=self.method.shared_query,
                    query_mode=self.method.query_mode,
                    summary_mode=self.method.summary_mode,
                    read_topk=self.method.read_topk,
                    role_dropout=self.method.role_dropout,
                    qa_temperature=self.method.qa_temperature,
                    action_query_indices=action_query_indices,
                    control=control,
                )
                diagnostics.append({"layer": layer_id, **cross.last_stats})
                if cross.last_answer_logits_tensor is not None and cross.last_answer_source_mask_tensor is not None:
                    qa_aux_tensors.append(
                        {
                            "layer": torch.as_tensor(layer_id, dtype=torch.long, device=base_ids.device),
                            "answer_logits": cross.last_answer_logits_tensor,
                            "source_mask": cross.last_answer_source_mask_tensor,
                        }
                    )
                if cross.last_action_bias_tensor is not None:
                    qa_action_biases.append(cross.last_action_bias_tensor)
            bus_key = str(layer_id)
            bus_block = self.shared_bus_blocks[bus_key] if bus_key in self.shared_bus_blocks else None
            if bus_block is not None:
                if bus is None:
                    raise RuntimeError("shared bus block was configured without a bus tensor")
                hidden, bus = bus_block(
                    hidden,
                    bus=bus,
                    role_ids=role_ids,
                    last_indices=last_indices,
                    summary_mode=self.method.summary_mode,
                    control=control,
                )
                diagnostics.append({"layer": layer_id, **bus_block.last_stats})
        hidden = self.final_norm(hidden)
        if self.uses_shared_bus:
            if bus is None or self.bus_final_norm is None or self.bus_action_head is None:
                raise RuntimeError("shared bus action head requested without bus modules")
            actor_state = gather_last_token(hidden[:, self.actor_index : self.actor_index + 1], last_indices).squeeze(1)
            bus_for_head = self.bus_final_norm(bus)
            if control in {"remove_path", "mask_action_bus"}:
                bus_for_head = torch.zeros_like(bus_for_head)
            logits = self.bus_action_head(torch.cat([actor_state, bus_for_head], dim=-1))
        elif self.method.action_query_decision and action_decision_indices is not None:
            decision_state = gather_sequence_positions(hidden[:, self.actor_index, :, :], action_decision_indices.view(batch, 1)).squeeze(1)
            logits = self.action_head(decision_state)
        elif self.method.action_query_tokens and action_query_indices is not None:
            actor_action_states = gather_sequence_positions(hidden[:, self.actor_index, :, :], action_query_indices)
            logits = self.action_token_head(actor_action_states).squeeze(-1)
            if self.method.action_query_hybrid:
                actor_state = gather_last_token(hidden[:, self.actor_index : self.actor_index + 1], last_indices).squeeze(1)
                logits = logits + self.action_head(actor_state)
        else:
            actor_state = gather_last_token(hidden[:, self.actor_index : self.actor_index + 1], last_indices).squeeze(1)
            logits = self.action_head(actor_state)
        if qa_action_biases:
            logits = logits + torch.stack(qa_action_biases, dim=0).sum(dim=0)
        self._last_qa_aux_tensors = qa_aux_tensors
        self._last_debug = {
            "logits_shape": list(logits.shape),
            "role_count": len(self.role_ids_tuple),
            "cross_layers": list(self.cross_layers),
            "slot_count": int(self.method.slot_count),
            "read_topk": int(self.method.read_topk),
            "role_dropout": float(self.method.role_dropout),
            "private_view": bool(self.method.private_view),
            "history_private_view": bool(self.method.history_private_view),
            "qa_aux_weight": float(self.method.qa_aux_weight),
            "qa_temperature": float(self.method.qa_temperature),
            "action_query_tokens": bool(self.method.action_query_tokens),
            "action_query_hybrid": bool(self.method.action_query_hybrid),
            "action_query_decision": bool(self.method.action_query_decision),
            "lora_rank": int(self.method.lora_rank),
            "lora_basis_count": int(self.method.lora_basis_count),
            "lora_alpha": float(self.method.lora_alpha),
            "lora_gate_mode": str(self.method.lora_gate_mode),
            "diagnostics": diagnostics,
            "lora_diagnostics": lora_diagnostics,
        }
        return logits

    def qa_auxiliary_loss(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if not self._last_qa_aux_tensors or "qa_answer_labels" not in batch:
            return torch.zeros((), dtype=self.action_head.weight.dtype, device=self.action_head.weight.device)
        role_index = torch.as_tensor(self.role_ids_tuple, dtype=torch.long, device=batch["qa_answer_labels"].device)
        labels = batch["qa_answer_labels"].index_select(1, role_index)
        losses: List[torch.Tensor] = []
        for item in self._last_qa_aux_tensors:
            logits = item["answer_logits"]
            source_mask = item["source_mask"].to(device=logits.device)
            mask = source_mask.view(1, -1, 1).expand(labels.shape[0], -1, labels.shape[-1])
            if bool(mask.any().detach().cpu().item()):
                losses.append(F.cross_entropy(logits[mask], labels.to(device=logits.device)[mask]))
        if not losses:
            return torch.zeros((), dtype=self.action_head.weight.dtype, device=self.action_head.weight.device)
        return torch.stack(losses).mean()

    def diagnostics(self) -> Dict[str, object]:
        rows = list(self._last_debug.get("diagnostics", []))
        lora_rows = list(self._last_debug.get("lora_diagnostics", []))
        entropy = mean(float(row.get("attention_entropy", 0.0)) for row in rows) if rows else 0.0
        inject = mean(float(row.get("inject_norm_ratio", 0.0)) for row in rows) if rows else 0.0
        qa_entropy = mean(float(row.get("qa_answer_entropy", 0.0)) for row in rows) if rows else 0.0
        qa_yes = mean(float(row.get("qa_yes_rate", 0.0)) for row in rows) if rows else 0.0
        qa_no = mean(float(row.get("qa_no_rate", 0.0)) for row in rows) if rows else 0.0
        qa_irrelevant = mean(float(row.get("qa_irrelevant_rate", 0.0)) for row in rows) if rows else 0.0
        bus_write = mean(float(row.get("bus_write_norm_ratio", 0.0)) for row in rows) if rows else 0.0
        bus_read = mean(float(row.get("bus_read_norm_ratio", 0.0)) for row in rows) if rows else 0.0
        source_mass: Dict[str, float] = {}
        for row in rows:
            matrix = row.get("attention_mass", [])
            if not matrix:
                continue
            actor_row = matrix[self.actor_index]
            for source_index, mass in enumerate(actor_row):
                role_id = self.role_ids_tuple[source_index]
                source_mass.setdefault(ROLE_NAMES[role_id], 0.0)
                source_mass[ROLE_NAMES[role_id]] += float(mass) / max(1, len(rows))
        lora_delta = mean(float(row.get("lora_delta_norm_ratio", 0.0)) for row in lora_rows) if lora_rows else 0.0
        lora_gate_entropy = mean(float(row.get("lora_gate_entropy", 0.0)) for row in lora_rows) if lora_rows else 0.0
        return {
            "attention_entropy": float(entropy),
            "inject_norm_ratio": float(inject),
            "actor_attention_mass_by_source": source_mass,
            "qa_answer_entropy": float(qa_entropy),
            "qa_yes_rate": float(qa_yes),
            "qa_no_rate": float(qa_no),
            "qa_irrelevant_rate": float(qa_irrelevant),
            "bus_write_norm_ratio": float(bus_write),
            "bus_read_norm_ratio": float(bus_read),
            "lora_delta_norm_ratio": float(lora_delta),
            "lora_gate_entropy": float(lora_gate_entropy),
        }


def allowed_source_matrix(
    roles: int,
    role_ids: torch.Tensor,
    mode: str,
    mask_actor_peer_reads: bool,
    device: torch.device,
) -> torch.Tensor:
    eye = torch.eye(roles, dtype=torch.bool, device=device)
    if mode == "self_only":
        allowed = eye.clone()
    elif mode == "cross_peer":
        allowed = ~eye if roles > 1 else eye.clone()
    else:
        allowed = torch.zeros((roles, roles), dtype=torch.bool, device=device)
    if mask_actor_peer_reads:
        actor_positions = torch.nonzero(role_ids.eq(3), as_tuple=False).flatten()
        for actor_pos in actor_positions.tolist():
            allowed[actor_pos, :] = False
            allowed[actor_pos, actor_pos] = True
    return allowed


def allowed_slot_source_matrix(
    roles: int,
    slot_count: int,
    role_ids: torch.Tensor,
    mode: str,
    mask_actor_peer_reads: bool,
    device: torch.device,
) -> torch.Tensor:
    role_allowed = allowed_source_matrix(
        roles=roles,
        role_ids=role_ids,
        mode=mode,
        mask_actor_peer_reads=mask_actor_peer_reads,
        device=device,
    )
    return role_allowed.repeat_interleave(max(1, int(slot_count)), dim=1)


def apply_private_role_view_masks(ids: torch.Tensor, role_ids: torch.Tensor, history_private: bool = False) -> torch.Tensor:
    masked_ids = ids.clone()
    goal_cells = (masked_ids >= GOAL_BASE) & (masked_ids < OBS_BASE)
    obstacle_cells = (masked_ids >= OBS_BASE) & (masked_ids < ACTION_BASE)
    obstacle_tokens = obstacle_cells | masked_ids.eq(OBSTACLES)
    history_tokens = masked_ids.eq(HISTORY) | ((masked_ids >= ACTION_BASE) & (masked_ids < ACTION_QUERY_BASE))
    for role_index, role_id in enumerate(role_ids.detach().cpu().tolist()):
        hide = torch.zeros_like(masked_ids[:, role_index, :], dtype=torch.bool)
        if int(role_id) == 0:
            hide = obstacle_tokens[:, role_index, :]
        elif int(role_id) in (1, 2):
            hide = goal_cells[:, role_index, :]
        elif int(role_id) == 3:
            hide = goal_cells[:, role_index, :] | obstacle_tokens[:, role_index, :]
        if history_private and int(role_id) != 2:
            hide = hide | history_tokens[:, role_index, :]
        masked_ids[:, role_index, :] = torch.where(
            hide,
            torch.full_like(masked_ids[:, role_index, :], MASK),
            masked_ids[:, role_index, :],
        )
    return masked_ids


def insert_action_query_tokens(
    ids: torch.Tensor,
    last_indices: torch.Tensor,
    include_decision: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    out = ids.clone()
    batch, _roles, seq_len = out.shape
    action_offsets = torch.arange(len(ACTION_NAMES), dtype=torch.long, device=ids.device)
    inserted_count = len(ACTION_NAMES) + int(bool(include_decision))
    max_start = max(0, seq_len - inserted_count)
    start = (last_indices.to(device=ids.device) + 1).clamp(max=max_start)
    query_indices = start.view(batch, 1) + action_offsets.view(1, len(ACTION_NAMES))
    action_tokens = (ACTION_QUERY_BASE + action_offsets).view(1, 1, len(ACTION_NAMES)).expand(batch, out.shape[1], -1)
    scatter_index = query_indices.view(batch, 1, len(ACTION_NAMES)).expand(-1, out.shape[1], -1)
    out.scatter_(2, scatter_index, action_tokens)
    decision_indices = None
    if include_decision:
        decision_indices = start + len(ACTION_NAMES)
        decision_tokens = torch.full((batch, out.shape[1], 1), ACTION_DECISION_TOKEN, dtype=out.dtype, device=out.device)
        decision_scatter = decision_indices.view(batch, 1, 1).expand(-1, out.shape[1], -1)
        out.scatter_(2, decision_scatter, decision_tokens)
    return out, query_indices, decision_indices


def gather_sequence_positions(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch, _seq_len, dim = x.shape
    gather_index = indices.view(batch, -1, 1).expand(-1, -1, dim)
    return x.gather(1, gather_index)


def gather_last_token(x: torch.Tensor, last_indices: torch.Tensor) -> torch.Tensor:
    batch, roles, _seq_len, dim = x.shape
    index = last_indices.view(batch, 1, 1, 1).expand(batch, roles, 1, dim)
    return x.gather(2, index).squeeze(2)


def summarize_role_states(x: torch.Tensor, last_indices: torch.Tensor, summary_mode: str) -> torch.Tensor:
    if summary_mode == "causal_mean":
        seq_len = x.shape[2]
        positions = torch.arange(seq_len, dtype=torch.long, device=x.device).view(1, 1, seq_len, 1)
        mask = positions <= last_indices.view(-1, 1, 1, 1)
        masked = x * mask.to(dtype=x.dtype)
        denom = mask.to(dtype=x.dtype).sum(dim=2).clamp(min=1.0)
        return masked.sum(dim=2) / denom
    if summary_mode != "last_token":
        raise ValueError(f"unknown summary mode: {summary_mode}")
    return gather_last_token(x, last_indices)


def role_token_ids(role_ids: torch.Tensor) -> torch.Tensor:
    return ROLE_BASE + role_ids.to(dtype=torch.long)


def build_worlds(config: RunConfig, split: str, count: int, seed: int) -> List[GridWorld]:
    rng = np.random.default_rng(seed)
    worlds: List[GridWorld] = []
    attempts = 0
    while len(worlds) < count:
        attempts += 1
        if attempts > count * 800:
            raise RuntimeError(f"failed to build enough {split} worlds")
        grid_size = int(config.grid_size)
        cells = [(r, c) for r in range(grid_size) for c in range(grid_size)]
        start = tuple(cells[int(rng.integers(0, len(cells)))])
        goal = tuple(cells[int(rng.integers(0, len(cells)))])
        if start == goal or manhattan(start, goal) < int(config.min_path_len):
            continue
        obstacle_budget = int(round(float(config.obstacle_density) * grid_size * grid_size))
        candidates = [cell for cell in cells if cell not in {start, goal}]
        rng.shuffle(candidates)
        obstacles = tuple(sorted(candidates[:obstacle_budget]))
        path = shortest_path(grid_size, start, goal, obstacles)
        if path is None:
            obstacles = tuple(sorted(obstacles[: max(0, len(obstacles) // 2)]))
            path = shortest_path(grid_size, start, goal, obstacles)
        if path is None:
            continue
        if len(path) < config.min_path_len or len(path) > config.max_steps:
            continue
        worlds.append(
            GridWorld(
                id=f"{split}_{len(worlds):05d}",
                split=split,
                grid_size=grid_size,
                start=start,
                goal=goal,
                obstacles=obstacles,
            )
        )
    return worlds


def build_examples(config: RunConfig, worlds: Sequence[GridWorld], seed: int) -> List[StepExample]:
    rng = np.random.default_rng(seed)
    examples: List[StepExample] = []
    for world in worlds:
        path = shortest_path(world.grid_size, world.start, world.goal, world.obstacles)
        if path is None:
            continue
        positions = positions_from_actions(world.start, path)
        for step_index, action in enumerate(path[: config.max_steps]):
            history = tuple(path[:step_index])
            examples.append(
                StepExample(
                    world_id=world.id,
                    split=world.split,
                    current=positions[step_index],
                    goal=world.goal,
                    obstacles=world.obstacles,
                    history=history,
                    step_index=step_index,
                    label=select_action_label(
                        config=config,
                        current=positions[step_index],
                        goal=world.goal,
                        obstacles=world.obstacles,
                        history=history,
                        default_label=int(action),
                    ),
                )
            )
        free_cells = [
            (r, c)
            for r in range(world.grid_size)
            for c in range(world.grid_size)
            if (r, c) not in set(world.obstacles) and (r, c) != world.goal
        ]
        rng.shuffle(free_cells)
        added = 0
        for current in free_cells:
            if added >= int(config.recovery_states_per_world):
                break
            recovery_path = shortest_path(world.grid_size, current, world.goal, world.obstacles)
            if not recovery_path:
                continue
            history_len = int(rng.integers(0, max(1, config.max_steps // 2)))
            history = tuple(int(v) for v in rng.integers(0, len(ACTION_NAMES), size=history_len).tolist())
            examples.append(
                StepExample(
                    world_id=world.id,
                    split=world.split,
                    current=current,
                    goal=world.goal,
                    obstacles=world.obstacles,
                    history=history,
                    step_index=min(history_len, config.max_steps),
                    label=select_action_label(
                        config=config,
                        current=current,
                        goal=world.goal,
                        obstacles=world.obstacles,
                        history=history,
                        default_label=int(recovery_path[0]),
                    ),
                )
            )
            added += 1
    return examples


def select_action_label(
    config: RunConfig,
    current: Tuple[int, int],
    goal: Tuple[int, int],
    obstacles: Sequence[Tuple[int, int]],
    history: Sequence[int],
    default_label: int,
) -> int:
    if config.label_mode == "shortest":
        return int(default_label)
    if config.label_mode != "history_tiebreak":
        raise ValueError(f"unknown label mode: {config.label_mode}")
    candidates = shortest_action_candidates(config.grid_size, current, goal, obstacles)
    if len(candidates) <= 1 or not history:
        return int(default_label)
    return int(candidates[int(history[-1]) % len(candidates)])


def shortest_action_candidates(
    grid_size: int,
    current: Tuple[int, int],
    goal: Tuple[int, int],
    obstacles: Sequence[Tuple[int, int]],
) -> List[int]:
    base_path = shortest_path(grid_size, current, goal, obstacles)
    if base_path is None:
        return []
    base_len = len(base_path)
    candidates: List[int] = []
    for action in ACTION_DELTAS:
        next_pos, valid = apply_action(grid_size, current, obstacles, int(action))
        if not valid:
            continue
        rest = shortest_path(grid_size, next_pos, goal, obstacles)
        if rest is not None and 1 + len(rest) == base_len:
            candidates.append(int(action))
    return candidates


def tensorize_examples(config: RunConfig, examples: Sequence[StepExample]) -> Dict[str, torch.Tensor]:
    ids: List[List[int]] = []
    last_indices: List[int] = []
    labels: List[int] = []
    qa_labels: List[List[List[int]]] = []
    for example in examples:
        row, last = encode_state(
            grid_size=config.grid_size,
            current=example.current,
            goal=example.goal,
            obstacles=example.obstacles,
            history=example.history,
            step_index=example.step_index,
            max_seq_len=config.max_seq_len,
        )
        ids.append(row)
        last_indices.append(last)
        labels.append(int(example.label))
        qa_labels.append(build_qa_answer_labels(config, example))
    return {
        "input_ids": torch.as_tensor(ids, dtype=torch.long),
        "last_indices": torch.as_tensor(last_indices, dtype=torch.long),
        "labels": torch.as_tensor(labels, dtype=torch.long),
        "qa_answer_labels": torch.as_tensor(qa_labels, dtype=torch.long),
    }


def build_qa_answer_labels(config: RunConfig, example: StepExample) -> List[List[int]]:
    labels = [[ANSWER_IRRELEVANT for _action in ACTION_NAMES] for _role in ROLE_NAMES]
    current_distance = manhattan(example.current, example.goal)
    obstacle_set = set(tuple(v) for v in example.obstacles)
    for action, (dr, dc) in ACTION_DELTAS.items():
        candidate = (int(example.current[0]) + int(dr), int(example.current[1]) + int(dc))
        in_grid = in_bounds(candidate, int(config.grid_size))
        valid_map_move = in_grid and candidate not in obstacle_set
        closer_to_goal = in_grid and manhattan(candidate, example.goal) < current_distance
        labels[0][int(action)] = ANSWER_YES if closer_to_goal else ANSWER_NO
        labels[1][int(action)] = ANSWER_YES if valid_map_move else ANSWER_NO
        labels[2][int(action)] = ANSWER_YES if valid_map_move else ANSWER_NO
        labels[3][int(action)] = ANSWER_IRRELEVANT
    return labels


def encode_state(
    grid_size: int,
    current: Tuple[int, int],
    goal: Tuple[int, int],
    obstacles: Sequence[Tuple[int, int]],
    history: Sequence[int],
    step_index: int,
    max_seq_len: int,
) -> Tuple[List[int], int]:
    del grid_size
    tokens = [
        BOS,
        ROLE_PLACEHOLDER,
        TASK_GRIDWORLD,
        STEP_BASE + min(int(step_index), 31),
        CUR_BASE + cell_id(current),
        GOAL_BASE + cell_id(goal),
        OBSTACLES,
    ]
    for pos in sorted(tuple(p) for p in obstacles):
        tokens.append(OBS_BASE + cell_id(pos))
    tokens.append(HISTORY)
    for action in list(history)[-24:]:
        tokens.append(ACTION_BASE + int(action))
    tokens.append(SEP)
    tokens = tokens[:max_seq_len]
    last_index = len(tokens) - 1
    if len(tokens) < max_seq_len:
        tokens.extend([PAD] * (max_seq_len - len(tokens)))
    return tokens, last_index


def train_and_evaluate(
    config: RunConfig,
    method: MethodSpec,
    seed: int,
    device: torch.device,
) -> Dict[str, object]:
    set_seed(seed)
    train_worlds = build_worlds(config, "train", config.train_worlds, seed * 1000 + 11)
    dev_worlds = build_worlds(config, "dev", config.dev_worlds, seed * 1000 + 29)
    train_examples = build_examples(config, train_worlds, seed * 1000 + 43)
    dev_examples = build_examples(config, dev_worlds, seed * 1000 + 59)
    train_tensors = tensorize_examples(config, train_examples)
    dev_tensors = tensorize_examples(config, dev_examples)

    model = RoleStreamAgent(config, method).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history: List[Dict[str, float]] = []
    first_grad: Dict[str, float] = {}
    start = time.perf_counter()

    for epoch in range(config.epochs):
        model.train()
        perm = torch.randperm(train_tensors["labels"].shape[0])
        losses: List[float] = []
        aux_losses: List[float] = []
        for start_index in range(0, len(perm), config.batch_size):
            batch_index = perm[start_index : start_index + config.batch_size]
            batch = move_batch(slice_batch(train_tensors, batch_index), device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["last_indices"])
            loss = F.cross_entropy(logits, batch["labels"])
            if float(method.qa_aux_weight) > 0.0:
                aux_loss = model.qa_auxiliary_loss(batch)
                loss = loss + float(method.qa_aux_weight) * aux_loss
                aux_losses.append(float(aux_loss.detach().cpu().item()))
            loss.backward()
            if not first_grad:
                first_grad = gradient_summary(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        model.eval()
        dev_acc = supervised_action_accuracy(model, dev_tensors, device, config.batch_size)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": mean(losses),
                "qa_aux_loss": mean(aux_losses) if aux_losses else 0.0,
                "dev_action_accuracy": dev_acc,
            }
        )

    elapsed = time.perf_counter() - start
    model.eval()
    base_eval = evaluate_all(config, model, dev_tensors, dev_worlds, device, config.batch_size, control="none")
    ablations = {}
    if method.cross_mode == "shared_bus":
        controls = [
            "remove_path",
            "mask_action_bus",
            "mask_actor_peer_reads",
            "ablation_compound_planner_zero_actor_mask",
            "cross_task_bus_shuffle",
            "role_write_zero_planner",
            "role_write_zero_reader",
            "role_write_zero_critic",
            "role_write_zero_actor",
        ]
        for control in controls:
            ablations[control] = evaluate_all(config, model, dev_tensors, dev_worlds, device, config.batch_size, control=control)
    elif method.cross_mode == "cross_peer" or method.cross_mode.startswith("qa_"):
        controls = ["remove_path", "role_kv_shuffle", "cross_task_kv_shuffle", "mask_actor_peer_reads"]
        if method.cross_mode.startswith("qa_"):
            controls.append("action_answer_shuffle")
        for control in controls:
            ablations[control] = evaluate_all(config, model, dev_tensors, dev_worlds, device, config.batch_size, control=control)

    return {
        "method": method.name,
        "seed": int(seed),
        "metrics": base_eval,
        "ablations": ablations,
        "history": history,
        "first_gradient_summary": first_grad,
        "param_count": int(sum(p.numel() for p in model.parameters())),
        "trainable_param_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "dataset": {
            "train_worlds": len(train_worlds),
            "dev_worlds": len(dev_worlds),
            "train_step_examples": len(train_examples),
            "dev_step_examples": len(dev_examples),
        },
        "runtime_seconds": float(elapsed),
        "model_debug": model._last_debug,
        "config": method_config_dict(method),
    }


def evaluate_all(
    config: RunConfig,
    model: RoleStreamAgent,
    dev_tensors: Mapping[str, torch.Tensor],
    dev_worlds: Sequence[GridWorld],
    device: torch.device,
    batch_size: int,
    control: str,
) -> Dict[str, object]:
    action_acc = supervised_action_accuracy(model, dev_tensors, device, batch_size, control=control)
    rollout = rollout_evaluate(config, model, dev_worlds, device, control=control)
    diagnostics = model.diagnostics()
    return {
        "supervised_action_accuracy": float(action_acc),
        **rollout,
        **diagnostics,
    }


@torch.no_grad()
def supervised_action_accuracy(
    model: RoleStreamAgent,
    tensors: Mapping[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
    control: str = "none",
) -> float:
    model.eval()
    correct = 0
    total = 0
    for start in range(0, tensors["labels"].shape[0], batch_size):
        batch = move_batch(slice_range(tensors, start, min(start + batch_size, tensors["labels"].shape[0])), device)
        logits = model(batch["input_ids"], batch["last_indices"], control=control)
        pred = logits.argmax(dim=-1)
        correct += int(pred.eq(batch["labels"]).sum().detach().cpu().item())
        total += int(batch["labels"].numel())
    return float(correct / max(1, total))


@torch.no_grad()
def rollout_evaluate(
    config: RunConfig,
    model: RoleStreamAgent,
    worlds: Sequence[GridWorld],
    device: torch.device,
    control: str = "none",
) -> Dict[str, object]:
    positions = [tuple(world.start) for world in worlds]
    histories: List[List[int]] = [[] for _world in worlds]
    done = [False for _world in worlds]
    success_flags = [False for _world in worlds]
    invalid_seen = [False for _world in worlds]
    action_counts = [0 for _world in worlds]
    total_actions = 0
    valid_actions = 0

    for step in range(config.max_steps):
        active = [index for index, is_done in enumerate(done) if not is_done]
        if not active:
            break
        rows: List[List[int]] = []
        last_indices: List[int] = []
        for index in active:
            world = worlds[index]
            row, last = encode_state(
                grid_size=world.grid_size,
                current=positions[index],
                goal=world.goal,
                obstacles=world.obstacles,
                history=histories[index],
                step_index=step,
                max_seq_len=config.max_seq_len,
            )
            rows.append(row)
            last_indices.append(last)
        batch = {
            "input_ids": torch.as_tensor(rows, dtype=torch.long, device=device),
            "last_indices": torch.as_tensor(last_indices, dtype=torch.long, device=device),
        }
        logits = model(batch["input_ids"], batch["last_indices"], control=control)
        actions = logits.argmax(dim=-1).detach().cpu().tolist()
        for index, action in zip(active, actions):
            world = worlds[index]
            next_pos, valid = apply_action(world.grid_size, positions[index], world.obstacles, int(action))
            total_actions += 1
            valid_actions += int(valid)
            invalid_seen[index] = invalid_seen[index] or not valid
            histories[index].append(int(action))
            positions[index] = next_pos
            action_counts[index] = step + 1
            if next_pos == tuple(world.goal):
                done[index] = True
                success_flags[index] = True

    for index, is_done in enumerate(done):
        if not is_done:
            action_counts[index] = config.max_steps

    successes = int(sum(success_flags))
    total_actions = 0
    valid_actions = 0
    for history_index, history in enumerate(histories):
        pos = tuple(worlds[history_index].start)
        for action in history[: action_counts[history_index]]:
            pos, valid = apply_action(worlds[history_index].grid_size, pos, worlds[history_index].obstacles, int(action))
            total_actions += 1
            valid_actions += int(valid)
    failure_counts = {"collision_or_wall_budget": 0, "budget_exhausted": 0}
    traces: List[Dict[str, object]] = []
    for index, world in enumerate(worlds):
        if not success_flags[index]:
            if invalid_seen[index]:
                failure_counts["collision_or_wall_budget"] += 1
            else:
                failure_counts["budget_exhausted"] += 1
        if len(traces) < 5:
            traces.append(
                {
                    "world_id": world.id,
                    "success": bool(success_flags[index]),
                    "actions": "".join(ACTION_NAMES[a] for a in histories[index]),
                    "final": list(positions[index]),
                    "goal": list(world.goal),
                }
            )
    return {
        "rollout_success_rate": float(successes / max(1, len(worlds))),
        "valid_action_rate": float(valid_actions / max(1, total_actions)),
        "mean_actions": float(mean(action_counts) if action_counts else 0.0),
        "failure_counts": failure_counts,
        "trace_examples": traces,
    }


def run_smoke(config: RunConfig, seed: int, device: torch.device, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result = train_and_evaluate(config, METHODS["cross_agent_latent_attention"], seed, device)
    grad = result["first_gradient_summary"]
    metrics = result["metrics"]
    smoke = {
        "status": "pass"
        if (
            result["model_debug"].get("logits_shape")
            and float(grad.get("cross_qkv_grad_norm", 0.0)) > 0.0
            and float(grad.get("backbone_grad_norm", 0.0)) > 0.0
            and 0.0 <= float(metrics.get("rollout_success_rate", -1.0)) <= 1.0
        )
        else "fail",
        "hypothesis": "Implementation viability: causal role streams can train, backpropagate through subtask-query cross-agent latent attention, roll out in an executable gridworld, and log diagnostics.",
        "seed": int(seed),
        "device": environment_summary(device),
        "config": asdict(config),
        "result": result,
        "causality_audit": causality_audit(),
    }
    write_json(out_dir / "smoke.json", smoke)
    print(json.dumps({"smoke_status": smoke["status"], "path": str(out_dir / "smoke.json")}, indent=2))


def run_screen(
    config: RunConfig,
    seeds: Sequence[int],
    device: torch.device,
    out_dir: Path,
    method_names: Tuple[str, ...] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if method_names is None:
        method_names = (
            "single_actor",
            "role_clones_no_path",
            "capacity_control_self_read",
            "cross_agent_latent_attention",
            "shared_query_control",
        )
    rows: List[Dict[str, object]] = []
    metrics_jsonl = out_dir / "metrics.jsonl"
    if metrics_jsonl.exists():
        metrics_jsonl.unlink()
    for seed in seeds:
        for method_name in method_names:
            result = train_and_evaluate(config, METHODS[method_name], int(seed), device)
            rows.append(result)
            append_jsonl(metrics_jsonl, flatten_result_rows(result))
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "method": method_name,
                        "action_acc": result["metrics"]["supervised_action_accuracy"],
                        "rollout_success": result["metrics"]["rollout_success_rate"],
                    }
                )
            )
    aggregate = aggregate_results(rows)
    artifact = {
        "hypothesis": "A subtask-query cross-agent latent path should beat no-path role clones and a self-read capacity control on a cheap multi-step gridworld screen if the implementation creates useful collaboration.",
        "claim_boundary": "Development-only Stage 11A/11B bootstrap. The backbone is a small randomly initialized GPT-style causal decoder because no local pretrained decoder checkpoint is part of the Experiment 11 assets.",
        "environment": environment_summary(device),
        "config": asdict(config),
        "seeds": [int(s) for s in seeds],
        "results": rows,
        "aggregate": aggregate,
        "causality_audit": causality_audit(),
    }
    write_json(out_dir / "results.json", artifact)
    report_path = Path("experiments/AOB/11_subtask_query_cross_agent_latent_attention/reports/STAGE11A_AGENTIC_GRIDWORLD_SMOKE_SCREEN.md")
    write_report(report_path, artifact)
    print(json.dumps({"screen_results": str(out_dir / "results.json"), "report": str(report_path)}, indent=2))


def run_variant_screen(
    config: RunConfig,
    seeds: Sequence[int],
    device: torch.device,
    out_dir: Path,
    method_names: Tuple[str, ...] | None = None,
    report_path: Path | None = None,
    report_title: str | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if method_names is None:
        method_names = (
            "single_actor",
            "role_clones_no_path",
            "capacity_control_self_read",
            "cross_agent_latent_attention",
            "cross_late_only",
            "cross_every_layer",
            "cross_subtask_state_query",
            "cross_causal_mean_summary",
            "cross_two_role_actor_critic",
        )
    rows: List[Dict[str, object]] = []
    metrics_jsonl = out_dir / "metrics.jsonl"
    if metrics_jsonl.exists():
        metrics_jsonl.unlink()
    for seed in seeds:
        for method_name in method_names:
            result = train_and_evaluate(config, METHODS[method_name], int(seed), device)
            rows.append(result)
            append_jsonl(metrics_jsonl, flatten_result_rows(result))
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "method": method_name,
                        "action_acc": result["metrics"]["supervised_action_accuracy"],
                        "rollout_success": result["metrics"]["rollout_success_rate"],
                    }
                )
            )
    aggregate = aggregate_results(rows)
    default_method_set = set(method_names) == {
        "single_actor",
        "role_clones_no_path",
        "capacity_control_self_read",
        "cross_agent_latent_attention",
        "cross_late_only",
        "cross_every_layer",
        "cross_subtask_state_query",
        "cross_causal_mean_summary",
        "cross_two_role_actor_critic",
    }
    resolved_title = report_title or ("Stage 11B Architecture Variant Screen" if default_method_set else "Stage 11C Best Variant Retest")
    slot_screen = any(METHODS[name].slot_count > 0 or METHODS[name].role_dropout > 0.0 or METHODS[name].read_topk > 0 for name in method_names)
    lora_screen = any(METHODS[name].lora_rank > 0 or METHODS[name].lora_basis_count > 0 for name in method_names)
    if str(resolved_title).startswith("Stage 11H"):
        hypothesis = (
            "A partitioned shared residual bus should make cross-role state structurally load-bearing: "
            "Planner, Reader, and Critic write private goal, obstacle, and history evidence into fixed bus subspaces, "
            "all roles read the full bus, and the Actor action head receives the final bus as a required input."
        )
        claim_boundary = (
            "Development-only Stage 11H shared-bus branch on the same small randomly initialized causal decoder. "
            "This tests an explicit standard-transformer-style shared residual channel under private role views and history-dependent labels; it is not a final held-out claim."
        )
    elif str(resolved_title).startswith("Stage 11G"):
        hypothesis = (
            "Explicit action-query tokens may make the private-view Q/A relay more GPT-native than the Stage 11F direct action-logit bias: "
            "peer answers update the Actor's per-action token states, and the final action is scored from those tokens after causal decoding."
        )
        claim_boundary = (
            "Development-only Stage 11G refinement of the Stage 11F private-view branch on the same small randomly initialized GPT-style decoder. "
            "This tests a cleaner architectural route for constrained cooperation, not a final agentic claim."
        )
    elif str(resolved_title).startswith("Stage 11F"):
        hypothesis = (
            "Private role views plus a constrained latent question/answer relay may create real inter-agent dependence: "
            "the Actor is missing goal and map evidence, specialists hold complementary private state, and the Actor can only recover it through low-bandwidth action-conditioned answers."
        )
        claim_boundary = (
            "Development-only Stage 11F task-redesign branch on the same small randomly initialized GPT-style causal decoder. "
            "This tests whether architectural information partitioning can create cooperation pressure; it is not evidence for spontaneous specialization from identical role clones."
        )
    elif str(resolved_title).startswith("Stage 11E") or lora_screen:
        hypothesis = (
            "Role-gated shared-basis LoRA adapters may break transverse linearity across cloned role streams. "
            "The critical test is whether LoRA plus cross-agent latent reads beats the matching LoRA no-path control."
        )
        claim_boundary = (
            "Development-only Stage 11E diagnostic branch on the same randomly initialized causal decoder gridworld harness. "
            "Role-gated LoRA is an explicit specialization side study, not a replacement for the locked primary mechanism."
        )
    elif str(resolved_title).startswith("Stage 11D") or slot_screen:
        hypothesis = (
            "Slot-export bottlenecks, sparse top-k peer reads, and role dropout may force more complementary role specialization "
            "if the previous failures came from redundant all-role summaries rather than the cross-agent premise itself."
        )
        claim_boundary = (
            "Development-only Stage 11D complementary-specialization screen on the same randomly initialized causal decoder gridworld harness. "
            "No final claim and no pretrained-backbone claim."
        )
    else:
        hypothesis = (
            "Direct architecture-faithful variants of the subtask-query cross-agent block may rescue the negative Stage 11A result "
            "if failure came from layer placement, query source, summary source, or role count."
        )
        claim_boundary = (
            "Development-only Stage 11B architecture screen on the same randomly initialized causal decoder gridworld harness. No final claim and no pretrained-backbone claim."
            if default_method_set
            else "Development-only Stage 11C selected-variant retest on the same randomly initialized causal decoder gridworld harness. No final claim and no pretrained-backbone claim."
        )
    still_not_tried = tried_not_tried_inventory()["not_tried_after_stage11b"]
    if str(resolved_title).startswith("Stage 11H"):
        still_not_tried = [
            item
            for item in still_not_tried
            if not item.startswith("Partitioned shared residual cross-role bus")
        ]
    artifact = {
        "report_title": resolved_title,
        "hypothesis": hypothesis,
        "claim_boundary": claim_boundary,
        "environment": environment_summary(device),
        "config": asdict(config),
        "seeds": [int(s) for s in seeds],
        "tried_before_stage11b": tried_not_tried_inventory()["tried"],
        "new_stage11b_variants": [method_config_dict(METHODS[name]) for name in method_names],
        "still_not_tried_after_stage11b": still_not_tried,
        "results": rows,
        "aggregate": aggregate,
        "causality_audit": causality_audit(),
    }
    write_json(out_dir / "results.json", artifact)
    if report_path is None:
        report_path = Path("experiments/AOB/11_subtask_query_cross_agent_latent_attention/reports/STAGE11B_ARCHITECTURE_VARIANT_SCREEN.md")
    write_variant_report(report_path, artifact)
    print(json.dumps({"variant_results": str(out_dir / "results.json"), "report": str(report_path)}, indent=2))


def aggregate_results(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    by_method: Dict[str, List[Mapping[str, object]]] = {}
    ablation_rows: Dict[str, List[Mapping[str, object]]] = {}
    all_ablation_rows: Dict[str, List[Mapping[str, object]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)
        for name, metrics in dict(row.get("ablations", {})).items():
            all_ablation_rows.setdefault(f"{row['method']}::{name}", []).append({"metrics": metrics, "param_count": row["param_count"]})
            if row["method"] == "cross_agent_latent_attention":
                ablation_rows.setdefault(str(name), []).append({"metrics": metrics, "param_count": row["param_count"]})
    return {
        "methods": {name: summarize_metric_rows(items) for name, items in sorted(by_method.items())},
        "proposed_eval_ablations": {name: summarize_metric_rows(items) for name, items in sorted(ablation_rows.items())},
        "all_eval_ablations": {name: summarize_metric_rows(items) for name, items in sorted(all_ablation_rows.items())},
    }


def summarize_metric_rows(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    metrics = [dict(row["metrics"]) for row in rows]
    out: Dict[str, object] = {"seeds": len(rows)}
    for key in (
        "supervised_action_accuracy",
        "rollout_success_rate",
        "valid_action_rate",
        "mean_actions",
        "attention_entropy",
        "inject_norm_ratio",
        "qa_answer_entropy",
        "qa_yes_rate",
        "qa_no_rate",
        "qa_irrelevant_rate",
        "bus_write_norm_ratio",
        "bus_read_norm_ratio",
        "lora_delta_norm_ratio",
        "lora_gate_entropy",
    ):
        values = [float(m.get(key, 0.0)) for m in metrics]
        out[f"{key}_mean"] = float(mean(values) if values else 0.0)
        out[f"{key}_std"] = float(pstdev(values) if len(values) > 1 else 0.0)
        out[f"{key}_raw"] = values
    params = [int(row.get("param_count", 0)) for row in rows]
    out["param_count_mean"] = float(mean(params) if params else 0.0)
    source_keys = sorted(
        {
            key
            for metric in metrics
            for key in dict(metric.get("actor_attention_mass_by_source", {})).keys()
        }
    )
    out["actor_attention_mass_by_source_mean"] = {
        key: float(mean([float(dict(metric.get("actor_attention_mass_by_source", {})).get(key, 0.0)) for metric in metrics]))
        for key in source_keys
    }
    return out


def flatten_result_rows(result: Mapping[str, object]) -> List[Dict[str, object]]:
    rows = [
        {
            "seed": result["seed"],
            "method": result["method"],
            "condition": "none",
            **metric_scalars(dict(result["metrics"])),
            "param_count": result["param_count"],
            "runtime_seconds": result["runtime_seconds"],
        }
    ]
    for condition, metrics in dict(result.get("ablations", {})).items():
        rows.append(
            {
                "seed": result["seed"],
                "method": result["method"],
                "condition": condition,
                **metric_scalars(dict(metrics)),
                "param_count": result["param_count"],
                "runtime_seconds": result["runtime_seconds"],
            }
        )
    return rows


def metric_scalars(metrics: Mapping[str, object]) -> Dict[str, object]:
    out = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, str, bool)):
            out[key] = value
    return out


def write_report(path: Path, artifact: Mapping[str, object]) -> None:
    aggregate = dict(artifact["aggregate"])
    method_rows = dict(aggregate["methods"])
    ablation_rows = dict(aggregate["proposed_eval_ablations"])
    all_ablation_rows = dict(aggregate.get("all_eval_ablations", {}))
    lines: List[str] = []
    lines.append("# Stage 11A Agentic Gridworld Smoke and Cheap Screen")
    lines.append("")
    lines.append("## Hypothesis Tested")
    lines.append("")
    report_title = str(artifact.get("report_title", ""))
    hypothesis_text = str(artifact["hypothesis"])
    if report_title.startswith("Stage 11D"):
        hypothesis_text = (
            "Slot-export bottlenecks, sparse top-k peer reads, and role dropout may force more complementary role specialization "
            "if the previous failures came from redundant all-role summaries rather than the cross-agent premise itself."
        )
    elif report_title.startswith("Stage 11H"):
        hypothesis_text = (
            "A partitioned shared residual bus should make cross-role state structurally load-bearing: "
            "role-private goal, obstacle, and history evidence is written into fixed subspaces, every role reads the full bus, "
            "and the Actor action head receives the final bus as a required input."
        )
    elif report_title.startswith("Stage 11E"):
        hypothesis_text = (
            "Role-gated shared-basis LoRA adapters may break transverse linearity across cloned role streams. "
            "The critical test is whether LoRA plus cross-agent latent reads beats the matching LoRA no-path control."
        )
    lines.append(hypothesis_text)
    lines.append("")
    lines.append("## Architecture Variant")
    lines.append("")
    lines.append(
        "Tiny GPT-style causal decoder role streams with shared token, position, decoder-block, layer-norm, and action-head weights. "
        "The proposed variant inserts subtask-query cross-agent latent attention after the configured decoder layers. "
        "Role summaries are last-token hidden states; Q is derived from learned subtask/role embeddings; K/V are projected peer role summaries; "
        "the weighted value read is gated and injected residually into the receiving role stream before the actor action head."
    )
    lines.append("")
    lines.append("Important limitation: this batch used a randomly initialized small causal decoder, not a pretrained checkpoint. It is an implementation and development screen, not final evidence for the locked primary claim.")
    lines.append("")
    lines.append("## Task")
    lines.append("")
    lines.append(
        "Executable gridworld navigation. At each decision step, the agent observes the current position, goal, obstacle set, and action history, emits one of U/D/L/R, receives the updated position, and succeeds only by reaching the goal inside the action budget."
    )
    lines.append("")
    lines.append("## Budgets")
    lines.append("")
    cfg = dict(artifact["config"])
    lines.append(
        f"- Train worlds: `{cfg['train_worlds']}`; dev worlds: `{cfg['dev_worlds']}`; epochs: `{cfg['epochs']}`; batch size: `{cfg['batch_size']}`; max steps: `{cfg['max_steps']}`."
    )
    lines.append(f"- Seeds: `{artifact['seeds']}`.")
    env = dict(artifact["environment"])
    lines.append(f"- Hardware: `{env.get('device')}`; CUDA available: `{env.get('cuda_available')}`; GPU: `{env.get('cuda_device_name', 'none')}`.")
    lines.append("")
    lines.append("## Primary Results")
    lines.append("")
    lines.append("| method | seeds | params | action acc mean | rollout success mean | valid action mean | attention entropy | inject norm ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in sorted(method_rows):
        row = dict(method_rows[name])
        lines.append(
            f"| `{name}` | {row['seeds']} | {row['param_count_mean']:.0f} | "
            f"{row['supervised_action_accuracy_mean']:.4f} +/- {row['supervised_action_accuracy_std']:.4f} | "
            f"{row['rollout_success_rate_mean']:.4f} +/- {row['rollout_success_rate_std']:.4f} | "
            f"{row['valid_action_rate_mean']:.4f} | {row['attention_entropy_mean']:.4f} | {row['inject_norm_ratio_mean']:.4f} |"
        )
    lines.append("")
    proposed_success = float(method_rows.get("cross_agent_latent_attention", {}).get("rollout_success_rate_mean", 0.0))
    no_path_success = float(method_rows.get("role_clones_no_path", {}).get("rollout_success_rate_mean", 0.0))
    single_success = float(method_rows.get("single_actor", {}).get("rollout_success_rate_mean", 0.0))
    shared_success = float(method_rows.get("shared_query_control", {}).get("rollout_success_rate_mean", 0.0))
    if proposed_success > max(no_path_success, single_success):
        lines.append(
            f"Decision: the proposed route beat the single/no-path controls on this development screen "
            f"(`{proposed_success:.4f}` vs `{max(no_path_success, single_success):.4f}` mean rollout success), "
            "but this remains development-only until mechanism controls and a pretrained backbone are tested."
        )
    else:
        lines.append(
            f"Decision: this screen is negative for the proposed route. The cross-agent latent attention model did not beat "
            f"Single Actor or Role Clones, No Path on rollout success (`{proposed_success:.4f}` vs "
            f"`{max(no_path_success, single_success):.4f}` mean), and the trained shared-query control was close "
            f"(`{shared_success:.4f}`). The result does not pass the baseline gate even as a development screen."
        )
    lines.append("")
    lines.append("## Mechanism Ablations")
    lines.append("")
    lines.append("These are eval-time interventions on the trained proposed model, so they are mechanism smoke tests rather than full retrained controls.")
    lines.append("")
    lines.append("| ablation | seeds | action acc mean | rollout success mean | valid action mean | attention entropy | inject norm ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name in sorted(ablation_rows):
        row = dict(ablation_rows[name])
        lines.append(
            f"| `{name}` | {row['seeds']} | "
            f"{row['supervised_action_accuracy_mean']:.4f} +/- {row['supervised_action_accuracy_std']:.4f} | "
            f"{row['rollout_success_rate_mean']:.4f} +/- {row['rollout_success_rate_std']:.4f} | "
            f"{row['valid_action_rate_mean']:.4f} | {row['attention_entropy_mean']:.4f} | {row['inject_norm_ratio_mean']:.4f} |"
        )
    lines.append("")
    damaging = [
        name
        for name, row in ablation_rows.items()
        if float(row.get("rollout_success_rate_mean", proposed_success)) < proposed_success - 0.02
    ]
    if damaging:
        lines.append(f"Mechanism smoke result: `{damaging}` reduced rollout success by more than 0.02 absolute relative to the proposed model.")
    else:
        lines.append(
            "Mechanism smoke result: none of the eval-time K/V shuffles, Actor peer-read mask, or path removal controls materially reduced rollout success relative to the proposed model."
        )
    lines.append("")
    lines.append("## Routing Diagnostics")
    lines.append("")
    proposed = method_rows.get("cross_agent_latent_attention", {})
    source_mass = dict(proposed.get("actor_attention_mass_by_source_mean", {})) if proposed else {}
    lines.append(f"- Proposed actor read mass by source role: `{source_mass}`.")
    lines.append("- Attention diagnostics are used only as support; the behavioral ablations above are the relevant mechanism checks.")
    lines.append("")
    lines.append("## Failed or Invalid Runs")
    lines.append("")
    lines.append("- No runtime failures were recorded by the screen script.")
    lines.append("- The batch is claim-limited because it does not use a pretrained decoder and does not touch a final held-out split.")
    lines.append("")
    lines.append("## What This Supports")
    lines.append("")
    lines.append("- The Experiment 11 architecture path is implementable in this repo with causal shared-weight role streams, cross-agent Q/K/V reads, residual injection, gradient flow, rollout evaluation, and machine-readable logging.")
    if proposed_success > max(no_path_success, single_success):
        lines.append("- The cheap screen provides development evidence only; the result still needs mechanism controls and a pretrained backbone before any claim.")
    else:
        lines.append("- The task and harness are sufficient to expose a negative baseline result quickly: the proposed path did not improve the agentic rollout metric under this tiny randomly initialized setup.")
    lines.append("")
    lines.append("## What This Does Not Support")
    lines.append("")
    lines.append("- No publishable or final Experiment 11 claim is supported.")
    lines.append("- No claim about pretrained LLM agents is supported by this batch.")
    lines.append("- Eval-time ablations do not replace retrained shared-query and capacity-matched controls for a final claim.")
    lines.append("")
    lines.append("## Next Smallest Discriminating Experiment")
    lines.append("")
    lines.append(
        "Before scaling final evaluation, separate task/model weakness from mechanism weakness. The next smallest run should either use a locally available pretrained causal decoder, "
        "or explicitly document that no pretrained checkpoint is available and run a stronger randomized-decoder dev validation with more epochs, difficulty slices, and retrained mechanism controls. "
        "Continue only if the proposed path beats the no-path role clone under repeated dev seeds and at least one mechanism ablation damages performance."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def tried_not_tried_inventory() -> Dict[str, List[str]]:
    return {
        "tried": [
            "Single Actor baseline with one Actor stream and no cross-agent path.",
            "Role Clones, No Latent Cross-Agent Path with Planner/Reader/Critic/Actor streams.",
            "Capacity Control, Self Read with the cross block restricted to self-state reads.",
            "Base Proposed Cross-Agent Latent Attention with middle-plus-late insertion, last-token summaries, and subtask-only Q.",
            "Shared-Query Control replacing role/subtask-specific Q with a learned shared query.",
            "Eval-time ablations on the base proposed model: remove path, role K/V shuffle, cross-task K/V shuffle, Actor peer-read mask.",
            "Stage 11B layer, query, summary, and role-count variants: late-only, every-layer, subtask+state Q, causal-mean summaries, and Actor/Critic two-role.",
            "Stage 11C longer retest of the causal-mean summary variant against selected controls.",
            "Stage 11D complementary-specialization variants: slot self-read, slot bottlenecks, sparse top-k slot reads, and source-role dropout.",
            "Stage 11E transverse-linearity diagnostics: role-gated shared-basis LoRA, fixed-role hard LoRA, no-path/self-read controls, and LoRA plus cross-agent causal-mean/slot reads.",
            "Stage 11F private-view Q/A relay variants with complementary goal/map visibility and auxiliary answer supervision.",
            "Stage 11G explicit action-query token variants for the private-view Q/A relay.",
        ],
        "not_tried_after_stage11a": [
            "Layer insertion variants such as late-only and every-layer cross-agent blocks.",
            "Q derived from subtask plus receiving role state.",
            "Peer K/V summaries beyond last-token state, such as causal prefix mean summaries.",
            "Different number of role streams.",
            "Retrained mechanism controls for every promising variant.",
            "A pretrained causal decoder checkpoint.",
            "Full token-level cross-role attention instead of role summaries.",
            "Longer-medium validation, difficulty slices, and final held-out evaluation.",
        ],
        "not_tried_after_stage11b": [
            "A pretrained causal decoder checkpoint.",
            "Full token-level cross-role attention instead of role summaries.",
            "Learned or external subtask decomposition beyond fixed role/subtask tokens.",
            "Learned residual gate schedules or initialized near-zero gates.",
            "Retrained mechanism controls for every variant.",
            "Actor-only source routing or fixed sparse role-routing graphs.",
            "Stateful multi-turn latent Q/A with learned query selection rather than fixed action-conditioned questions.",
            "Partitioned shared residual cross-role bus with the final Actor action head forced to consume the bus.",
            "More role-count schedules beyond four roles and Actor/Critic two-role.",
            "Larger model/data/epoch scaling and difficulty-sliced validation.",
            "Held-out final evaluation.",
        ],
    }


def write_variant_report(path: Path, artifact: Mapping[str, object]) -> None:
    aggregate = dict(artifact["aggregate"])
    method_rows = dict(aggregate["methods"])
    ablation_rows = dict(aggregate["proposed_eval_ablations"])
    all_ablation_rows = dict(aggregate.get("all_eval_ablations", {}))
    if "private_view_shared_bus" in method_rows:
        reference_name = "private_view_shared_bus"
    elif "cross_agent_latent_attention" in method_rows:
        reference_name = "cross_agent_latent_attention"
    elif "cross_causal_mean_summary" in method_rows:
        reference_name = "cross_causal_mean_summary"
    else:
        reference_name = ""
    reference_success = float(method_rows.get(reference_name, {}).get("rollout_success_rate_mean", 0.0)) if reference_name else 0.0
    no_path = float(method_rows.get("role_clones_no_path", {}).get("rollout_success_rate_mean", 0.0))
    single = float(method_rows.get("single_actor", {}).get("rollout_success_rate_mean", 0.0))
    private_history_no_path = float(method_rows.get("private_view_history_no_path", {}).get("rollout_success_rate_mean", 0.0))
    control_best = max(single, no_path, private_history_no_path)
    variant_names = [
        name
        for name in method_rows
        if name not in {"single_actor", "role_clones_no_path", "capacity_control_self_read", "cross_agent_latent_attention"}
    ]
    best_name = max(method_rows, key=lambda name: float(method_rows[name].get("rollout_success_rate_mean", 0.0))) if method_rows else "none"
    best_success = float(method_rows.get(best_name, {}).get("rollout_success_rate_mean", 0.0))
    report_title = str(artifact.get("report_title", ""))

    lines: List[str] = []
    lines.append(f"# {artifact.get('report_title', 'Stage 11B Architecture Variant Screen')}")
    lines.append("")
    lines.append("## Hypothesis Tested")
    lines.append("")
    hypothesis_text = str(artifact["hypothesis"])
    if report_title.startswith("Stage 11D"):
        hypothesis_text = (
            "Slot-export bottlenecks, sparse top-k peer reads, and role dropout may force more complementary role specialization "
            "if the previous failures came from redundant all-role summaries rather than the cross-agent premise itself."
        )
    elif report_title.startswith("Stage 11H"):
        hypothesis_text = (
            "A partitioned shared residual bus should make cross-role state structurally load-bearing: "
            "role-private goal, obstacle, and history evidence is written into fixed subspaces, every role reads the full bus, "
            "and the Actor action head receives the final bus as a required input."
        )
    lines.append(hypothesis_text)
    lines.append("")
    lines.append("## Tried Before This Batch")
    lines.append("")
    for item in artifact["tried_before_stage11b"]:
        lines.append(f"- {item}")
    lines.append("")
    if report_title.startswith("Stage 11C"):
        lines.append("## Retested In Stage 11C")
    elif report_title.startswith("Stage 11D"):
        lines.append("## Newly Tried Or Retested In Stage 11D")
    elif report_title.startswith("Stage 11E"):
        lines.append("## Newly Tried Or Retested In Stage 11E")
    elif report_title.startswith("Stage 11F"):
        lines.append("## Newly Tried Or Retested In Stage 11F")
    elif report_title.startswith("Stage 11G"):
        lines.append("## Newly Tried Or Retested In Stage 11G")
    elif report_title.startswith("Stage 11H"):
        lines.append("## Newly Tried Or Retested In Stage 11H")
    else:
        lines.append("## Newly Tried In Stage 11B")
    lines.append("")
    for config in artifact["new_stage11b_variants"]:
        lines.append(
            f"- `{config['name']}`: roles={config['roles']}, cross_mode=`{config['cross_mode']}`, "
            f"query=`{config['query_mode']}`, summary=`{config['summary_mode']}`, layers={config['cross_layers'] or 'default'}, "
            f"slots={config['slot_count']}, topk={config['read_topk']}, role_dropout={config['role_dropout']}, "
            f"lora_rank={config.get('lora_rank', 0)}, lora_bases={config.get('lora_basis_count', 0)}, "
            f"lora_alpha={config.get('lora_alpha', 1.0)}, lora_gate={config.get('lora_gate_mode', 'softmax')}, "
            f"private_view={config.get('private_view', False)}, qa_aux={config.get('qa_aux_weight', 0.0)}, "
            f"history_private={config.get('history_private_view', False)}, "
            f"action_query_tokens={config.get('action_query_tokens', False)}, "
            f"action_query_hybrid={config.get('action_query_hybrid', False)}, "
            f"action_query_decision={config.get('action_query_decision', False)}."
        )
    lines.append("")
    lines.append("## Still Not Tried")
    lines.append("")
    for item in artifact["still_not_tried_after_stage11b"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Task And Budget")
    lines.append("")
    cfg = dict(artifact["config"])
    env = dict(artifact["environment"])
    lines.append(
        f"Same executable gridworld harness as Stage 11A. Train worlds `{cfg['train_worlds']}`, dev worlds `{cfg['dev_worlds']}`, epochs `{cfg['epochs']}`, batch size `{cfg['batch_size']}`, seeds `{artifact['seeds']}`."
    )
    lines.append(f"Hardware `{env.get('device')}`, CUDA `{env.get('cuda_available')}`, GPU `{env.get('cuda_device_name', 'none')}`.")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    has_lora = any(float(dict(row).get("lora_delta_norm_ratio_mean", 0.0)) > 0.0 for row in method_rows.values())
    has_qa = any(float(dict(row).get("qa_answer_entropy_mean", 0.0)) > 0.0 for row in method_rows.values())
    has_bus = any(float(dict(row).get("bus_write_norm_ratio_mean", 0.0)) > 0.0 for row in method_rows.values())
    if has_lora:
        lines.append("| method | seeds | params | action acc mean | rollout success mean | valid action mean | entropy | inject norm | lora norm | lora gate entropy |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    elif has_qa:
        lines.append("| method | seeds | params | action acc mean | rollout success mean | valid action mean | entropy | inject norm | qa entropy | yes | no | irrelevant |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    elif has_bus:
        lines.append("| method | seeds | params | action acc mean | rollout success mean | valid action mean | entropy | inject norm | bus write norm | bus read norm |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    else:
        lines.append("| method | seeds | params | action acc mean | rollout success mean | valid action mean | entropy | inject norm |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in sorted(method_rows):
        row = dict(method_rows[name])
        base_cells = (
            f"| `{name}` | {row['seeds']} | {row['param_count_mean']:.0f} | "
            f"{row['supervised_action_accuracy_mean']:.4f} +/- {row['supervised_action_accuracy_std']:.4f} | "
            f"{row['rollout_success_rate_mean']:.4f} +/- {row['rollout_success_rate_std']:.4f} | "
            f"{row['valid_action_rate_mean']:.4f} | {row['attention_entropy_mean']:.4f} | {row['inject_norm_ratio_mean']:.4f}"
        )
        if has_lora:
            lines.append(
                f"{base_cells} | {float(row.get('lora_delta_norm_ratio_mean', 0.0)):.4f} | "
                f"{float(row.get('lora_gate_entropy_mean', 0.0)):.4f} |"
            )
        elif has_qa:
            lines.append(
                f"{base_cells} | {float(row.get('qa_answer_entropy_mean', 0.0)):.4f} | "
                f"{float(row.get('qa_yes_rate_mean', 0.0)):.4f} | "
                f"{float(row.get('qa_no_rate_mean', 0.0)):.4f} | "
                f"{float(row.get('qa_irrelevant_rate_mean', 0.0)):.4f} |"
            )
        elif has_bus:
            lines.append(
                f"{base_cells} | {float(row.get('bus_write_norm_ratio_mean', 0.0)):.4f} | "
                f"{float(row.get('bus_read_norm_ratio_mean', 0.0)):.4f} |"
            )
        else:
            lines.append(f"{base_cells} |")
    lines.append("")
    lines.append("## Mechanism Ablations On Base Proposed")
    lines.append("")
    if ablation_rows:
        lines.append("| ablation | rollout success mean | action acc mean |")
        lines.append("|---|---:|---:|")
        for name in sorted(ablation_rows):
            row = dict(ablation_rows[name])
            lines.append(
                f"| `{name}` | {row['rollout_success_rate_mean']:.4f} +/- {row['rollout_success_rate_std']:.4f} | "
                f"{row['supervised_action_accuracy_mean']:.4f} +/- {row['supervised_action_accuracy_std']:.4f} |"
            )
    else:
        lines.append("No mechanism ablations were logged.")
    lines.append("")
    if all_ablation_rows:
        lines.append("## Eval-Time Ablations By Cross Method")
        lines.append("")
        lines.append("| method | ablation | rollout success mean | delta vs method |")
        lines.append("|---|---|---:|---:|")
        for key in sorted(all_ablation_rows):
            method_name, condition = str(key).split("::", 1)
            row = dict(all_ablation_rows[key])
            parent = float(method_rows.get(method_name, {}).get("rollout_success_rate_mean", 0.0))
            value = float(row["rollout_success_rate_mean"])
            lines.append(f"| `{method_name}` | `{condition}` | {value:.4f} | {value - parent:+.4f} |")
        lines.append("")
    lines.append("## Conclusions")
    lines.append("")
    lines.append(f"- Best method in this screen: `{best_name}` at `{best_success:.4f}` mean rollout success.")
    if reference_name:
        lines.append(f"- Reference cross method `{reference_name}`: `{reference_success:.4f}` mean rollout success.")
    lines.append(f"- Single/no-path control ceiling in this screen: `{control_best:.4f}` mean rollout success.")
    for name in sorted(variant_names):
        row = dict(method_rows[name])
        delta_reference = float(row["rollout_success_rate_mean"]) - reference_success if reference_name else 0.0
        delta_control = float(row["rollout_success_rate_mean"]) - control_best
        lines.append(
            f"- `{name}`: rollout `{float(row['rollout_success_rate_mean']):.4f}`, "
            f"delta vs reference `{delta_reference:+.4f}`, delta vs best single/no-path `{delta_control:+.4f}`."
        )
    if any(str(name).startswith("role_lora_") for name in method_rows):
        lora_pairs = (
            ("role_lora_cross_causal_mean", "role_lora_no_path"),
            ("role_lora_cross_slots_m2_topk2", "role_lora_no_path"),
            ("role_lora_hard_cross_causal_mean", "role_lora_hard_no_path"),
        )
        for lora_cross_name, lora_control_name in lora_pairs:
            if lora_cross_name in method_rows and lora_control_name in method_rows:
                lora_control = float(method_rows[lora_control_name].get("rollout_success_rate_mean", 0.0))
                lora_cross = float(method_rows[lora_cross_name].get("rollout_success_rate_mean", 0.0))
                lines.append(
                    f"- LoRA cross-path test `{lora_cross_name}` vs `{lora_control_name}`: "
                    f"`{lora_cross:.4f}` vs `{lora_control:.4f}` (`{lora_cross - lora_control:+.4f}`)."
                )
    if report_title.startswith("Stage 11H"):
        bus_no_path = float(method_rows.get("private_view_history_no_path", {}).get("rollout_success_rate_mean", 0.0))
        bus_candidates = {
            name: float(row.get("rollout_success_rate_mean", 0.0))
            for name, row in method_rows.items()
            if "shared_bus" in str(name)
        }
        best_bus_name = max(bus_candidates, key=bus_candidates.get) if bus_candidates else ""
        best_bus = bus_candidates.get(best_bus_name, 0.0)
        lines.append(
            f"- Stage 11H private history no-path baseline: `{bus_no_path:.4f}`. "
            f"Best shared-bus method `{best_bus_name or 'none'}`: `{best_bus:.4f}` (`{best_bus - bus_no_path:+.4f}` vs private history no-path)."
        )
        if best_bus > bus_no_path:
            lines.append(
                "- The shared bus carried useful private role information relative to the Actor-blind private no-path control."
            )
        else:
            lines.append(
                "- The shared bus did not beat its private no-path control in this screen."
            )
    elif report_title.startswith("Stage 11G"):
        token_no_path = float(method_rows.get("private_view_action_tokens_no_path", {}).get("rollout_success_rate_mean", 0.0))
        token_candidates = {
            name: float(row.get("rollout_success_rate_mean", 0.0))
            for name, row in method_rows.items()
            if "qa_token" in str(name)
        }
        best_token_name = max(token_candidates, key=token_candidates.get) if token_candidates else ""
        best_token = token_candidates.get(best_token_name, 0.0)
        stage11f_bias = float(method_rows.get("private_view_qa_hard_aux_bias", {}).get("rollout_success_rate_mean", 0.0))
        lines.append(
            f"- Stage 11G action-token no-path baseline: `{token_no_path:.4f}`. "
            f"Best action-token Q/A method `{best_token_name or 'none'}`: `{best_token:.4f}` (`{best_token - token_no_path:+.4f}` vs token no-path)."
        )
        if stage11f_bias:
            lines.append(f"- Stage 11F direct-bias reference in this screen: `{stage11f_bias:.4f}`.")
        if best_token > token_no_path:
            lines.append(
                "- Explicit action-query tokens carried useful peer-answer information relative to the private action-token no-path control."
            )
        else:
            lines.append(
                "- Explicit action-query tokens did not beat their private no-path control in this screen."
            )
    elif report_title.startswith("Stage 11F"):
        private_no_path = float(method_rows.get("private_view_no_path", {}).get("rollout_success_rate_mean", 0.0))
        private_cross = float(method_rows.get("private_view_cross_causal_mean", {}).get("rollout_success_rate_mean", 0.0))
        qa_candidates = {
            name: float(row.get("rollout_success_rate_mean", 0.0))
            for name, row in method_rows.items()
            if str(name).startswith("private_view_qa_")
        }
        best_qa_name = max(qa_candidates, key=qa_candidates.get) if qa_candidates else ""
        best_qa = qa_candidates.get(best_qa_name, 0.0)
        lines.append(
            f"- Stage 11F private-view no-path baseline: `{private_no_path:.4f}`. "
            f"Best private Q/A relay `{best_qa_name or 'none'}`: `{best_qa:.4f}` (`{best_qa - private_no_path:+.4f}` vs private no-path)."
        )
        if private_cross:
            lines.append(f"- Private continuous causal-mean peer read: `{private_cross:.4f}` (`{private_cross - private_no_path:+.4f}` vs private no-path).")
        if best_qa > private_no_path:
            lines.append(
                "- The private-view task redesign created a useful-dependence signal relative to the Actor-blind no-path baseline. "
                "This supports the information-partition premise, but the relay still needs stronger mechanism ablations and harder task slices before claim escalation."
            )
        else:
            lines.append(
                "- The private-view task redesign did not make the tested Q/A relay beat its Actor-blind no-path baseline in this screen."
            )
    elif report_title.startswith("Stage 11E"):
        if best_success > control_best and "_cross_" not in best_name:
            lines.append(
                "- The best LoRA result beat the single/no-path controls, but it was not a peer-read cross-agent method. "
                "This is capacity/specialization side evidence, not evidence for latent collaboration."
            )
        elif best_success > control_best:
            lines.append("- A LoRA cross-agent variant beat the single/no-path controls on this development screen. It still requires matching no-path controls and damaging mechanism ablations before claim escalation.")
        else:
            lines.append("- No LoRA cross-agent variant beat the single/no-path controls. The LoRA branch does not rescue the randomized-decoder gridworld setup.")
    elif best_success > control_best:
        lines.append("- A variant beat the single/no-path controls on this development screen. It still requires retrained mechanism controls and a stronger/pretrained backbone before claim escalation.")
    else:
        lines.append("- No cross-agent variant beat the single/no-path controls. The current evidence says these architectural tweaks do not rescue the randomly initialized gridworld setup.")
    best_ablation_drops = []
    for key, row in all_ablation_rows.items():
        method_name, condition = str(key).split("::", 1)
        if method_name != best_name:
            continue
        delta = float(row.get("rollout_success_rate_mean", best_success)) - best_success
        if delta < -0.02:
            best_ablation_drops.append((condition, delta))
    if best_ablation_drops:
        lines.append(f"- Best-variant eval-time mechanism drops: `{best_ablation_drops}`.")
    elif best_name.startswith("cross_"):
        lines.append("- Best-variant eval-time ablations did not produce a >0.02 rollout drop; the behavioral mechanism is still not established.")
    if ablation_rows and "cross_agent_latent_attention" in method_rows:
        base = float(method_rows["cross_agent_latent_attention"].get("rollout_success_rate_mean", 0.0))
        damaging = [
            name
            for name, row in ablation_rows.items()
            if float(row.get("rollout_success_rate_mean", base)) < base - 0.02
        ]
        if damaging:
            lines.append(f"- Mechanism ablations with >0.02 rollout drop: `{damaging}`.")
        else:
            lines.append("- Base proposed mechanism ablations still did not materially damage rollout success, so attention patterns remain non-causal evidence.")
    lines.append("")
    lines.append("## Next Decision")
    lines.append("")
    if report_title.startswith("Stage 11H"):
        bus_no_path = float(method_rows.get("private_view_history_no_path", {}).get("rollout_success_rate_mean", 0.0))
        bus_candidates = {
            name: float(row.get("rollout_success_rate_mean", 0.0))
            for name, row in method_rows.items()
            if "shared_bus" in str(name)
        }
        best_bus_name = max(bus_candidates, key=bus_candidates.get) if bus_candidates else ""
        best_bus = bus_candidates.get(best_bus_name, 0.0)
        damaging_bus_ablations = []
        if best_bus_name:
            for key, row in all_ablation_rows.items():
                method_name, condition = str(key).split("::", 1)
                if method_name != best_bus_name:
                    continue
                delta = float(row.get("rollout_success_rate_mean", best_bus)) - best_bus
                if delta < -0.02:
                    damaging_bus_ablations.append((condition, delta))
        if best_bus_name and best_bus > bus_no_path and damaging_bus_ablations:
            lines.append(
                f"Retain `{best_bus_name}` for a targeted Stage 11H retest with more seeds and difficulty slices; "
                f"its eval-time bus ablations were damaging: `{damaging_bus_ablations}`."
            )
        else:
            lines.append(
                "Do not escalate the shared-bus branch until it beats the private history no-path control and writer/action-bus ablations damage rollout under repeated seeds."
            )
    elif report_title.startswith("Stage 11G"):
        token_no_path = float(method_rows.get("private_view_action_tokens_no_path", {}).get("rollout_success_rate_mean", 0.0))
        token_candidates = {
            name: float(row.get("rollout_success_rate_mean", 0.0))
            for name, row in method_rows.items()
            if "qa_token" in str(name)
        }
        best_token_name = max(token_candidates, key=token_candidates.get) if token_candidates else ""
        best_token = token_candidates.get(best_token_name, 0.0)
        if best_token_name and best_token > token_no_path:
            lines.append(
                f"Retain `{best_token_name}` only if mechanism ablations are damaging and it remains competitive with the Stage 11F direct-bias relay under the same seed budget."
            )
        else:
            lines.append(
                "Do not escalate the action-token relay yet; inspect whether the query tokens are undertrained, too late in the causal stream, or need an explicit per-action auxiliary loss."
            )
    elif report_title.startswith("Stage 11F"):
        private_no_path = float(method_rows.get("private_view_no_path", {}).get("rollout_success_rate_mean", 0.0))
        qa_candidates = {
            name: float(row.get("rollout_success_rate_mean", 0.0))
            for name, row in method_rows.items()
            if str(name).startswith("private_view_qa_")
        }
        best_qa_name = max(qa_candidates, key=qa_candidates.get) if qa_candidates else ""
        best_qa = qa_candidates.get(best_qa_name, 0.0)
        if best_qa_name and best_qa > private_no_path:
            lines.append(
                f"Retain `{best_qa_name}` for a targeted Stage 11F retest with more seeds, difficulty slices, and retrained private-view no-path/continuous-read controls. "
                "Do not compare it as a replacement for the original same-view Experiment 11 claim; it is a task-redesign branch."
            )
        else:
            lines.append(
                "Do not escalate the fixed Q/A relay yet. Try either learned query selection or a cleaner compositional task where Planner and Map evidence are both necessary on most episodes."
            )
    elif report_title.startswith("Stage 11E"):
        lora_cross_wins = False
        for lora_cross_name, lora_control_name in (
            ("role_lora_cross_causal_mean", "role_lora_no_path"),
            ("role_lora_cross_slots_m2_topk2", "role_lora_no_path"),
            ("role_lora_hard_cross_causal_mean", "role_lora_hard_no_path"),
        ):
            if lora_cross_name in method_rows and lora_control_name in method_rows:
                lora_cross = float(method_rows[lora_cross_name].get("rollout_success_rate_mean", 0.0))
                lora_control = float(method_rows[lora_control_name].get("rollout_success_rate_mean", 0.0))
                lora_cross_wins = lora_cross_wins or (lora_cross > lora_control and lora_cross > control_best)
        if lora_cross_wins:
            lines.append(
                "Keep the winning LoRA cross variant only as a side-branch candidate and rerun with matched no-path, self-read, remove-path, role-shuffle, and cross-task-shuffle controls. Do not touch final evaluation yet."
            )
        else:
            lines.append(
                "Do not escalate the LoRA cross-agent branch on this randomized gridworld. Explicit symmetry breaking can change behavior, but it has not made peer latent reads beat their matching no-path controls and the original single/no-path baselines."
            )
    elif best_success > control_best:
        lines.append(
            "Retain only the best Stage 11B variant for a medium dev validation with retrained no-path, shared-query, K/V shuffle, and Actor-mask controls. Do not touch final evaluation yet."
        )
    else:
        lines.append(
            "Do not spend final-evaluation compute on this randomized-decoder gridworld line. The next discriminating step is a pretrained causal decoder or a task redesign that demonstrably requires complementary role-state information."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def shortest_path(
    grid_size: int,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    obstacles: Sequence[Tuple[int, int]],
) -> Tuple[int, ...] | None:
    obstacle_set = set(tuple(v) for v in obstacles)
    queue: deque[Tuple[int, int]] = deque([tuple(start)])
    parent: Dict[Tuple[int, int], Tuple[Tuple[int, int], int]] = {}
    seen = {tuple(start)}
    while queue:
        pos = queue.popleft()
        if pos == tuple(goal):
            actions: List[int] = []
            while pos != tuple(start):
                prev, action = parent[pos]
                actions.append(action)
                pos = prev
            actions.reverse()
            return tuple(actions)
        for action, (dr, dc) in ACTION_DELTAS.items():
            nxt = (pos[0] + dr, pos[1] + dc)
            if not in_bounds(nxt, grid_size) or nxt in obstacle_set or nxt in seen:
                continue
            seen.add(nxt)
            parent[nxt] = (pos, int(action))
            queue.append(nxt)
    return None


def positions_from_actions(start: Tuple[int, int], actions: Sequence[int]) -> List[Tuple[int, int]]:
    pos = tuple(start)
    positions = [pos]
    for action in actions:
        dr, dc = ACTION_DELTAS[int(action)]
        pos = (pos[0] + dr, pos[1] + dc)
        positions.append(pos)
    return positions


def apply_action(
    grid_size: int,
    pos: Tuple[int, int],
    obstacles: Sequence[Tuple[int, int]],
    action: int,
) -> Tuple[Tuple[int, int], bool]:
    dr, dc = ACTION_DELTAS[int(action)]
    nxt = (int(pos[0]) + dr, int(pos[1]) + dc)
    valid = in_bounds(nxt, grid_size) and nxt not in set(tuple(v) for v in obstacles)
    return (nxt if valid else tuple(pos), bool(valid))


def in_bounds(pos: Tuple[int, int], grid_size: int) -> bool:
    return 0 <= int(pos[0]) < int(grid_size) and 0 <= int(pos[1]) < int(grid_size)


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def cell_id(pos: Tuple[int, int]) -> int:
    return int(pos[0]) * 8 + int(pos[1])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def slice_batch(tensors: Mapping[str, torch.Tensor], index: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {key: value[index] for key, value in tensors.items()}


def slice_range(tensors: Mapping[str, torch.Tensor], start: int, end: int) -> Dict[str, torch.Tensor]:
    return {key: value[start:end] for key, value in tensors.items()}


def move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def gradient_summary(model: RoleStreamAgent) -> Dict[str, float]:
    groups = {
        "backbone_grad_norm": ["token_embedding", "position_embedding", "blocks"],
        "cross_qkv_grad_norm": [
            "wq",
            "wk",
            "wv",
            "inject",
            "gate",
            "role_query",
            "action_query",
            "answer_embedding",
            "answer_head",
            "answer_action_score",
            "shared_bus_blocks",
            "bus_final_norm",
            "bus_action_head",
        ],
        "lora_grad_norm": ["role_lora_adapters"],
        "action_head_grad_norm": ["action_head", "action_token_head", "bus_action_head"],
    }
    out: Dict[str, float] = {}
    for group_name, needles in groups.items():
        sq = 0.0
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            if any(needle in name for needle in needles):
                sq += float(parameter.grad.detach().float().pow(2).sum().cpu().item())
        out[group_name] = math.sqrt(sq)
    return out


def method_config_dict(method: MethodSpec) -> Dict[str, object]:
    return {
        "name": method.name,
        "roles": [ROLE_NAMES[i] for i in method.roles],
        "cross_mode": method.cross_mode,
        "shared_query": method.shared_query,
        "query_mode": method.query_mode,
        "summary_mode": method.summary_mode,
        "cross_layers": list(method.cross_layers),
        "slot_count": int(method.slot_count),
        "read_topk": int(method.read_topk),
        "role_dropout": float(method.role_dropout),
        "lora_rank": int(method.lora_rank),
        "lora_basis_count": int(method.lora_basis_count),
        "lora_alpha": float(method.lora_alpha),
        "lora_gate_mode": str(method.lora_gate_mode),
        "private_view": bool(method.private_view),
        "history_private_view": bool(method.history_private_view),
        "qa_aux_weight": float(method.qa_aux_weight),
        "qa_temperature": float(method.qa_temperature),
        "action_query_tokens": bool(method.action_query_tokens),
        "action_query_hybrid": bool(method.action_query_hybrid),
        "action_query_decision": bool(method.action_query_decision),
    }


def causality_audit() -> Dict[str, object]:
    return {
        "future_observations_in_input": False,
        "gold_future_actions_in_input": False,
        "success_labels_in_input": False,
        "role_specific_tools_or_privileged_observations": "False for Stage 11A-E; Stage 11F+ private-view methods intentionally mask different current-step fields by role. Stage 11H history-private methods additionally expose action history only to Critic.",
        "subtask_query_source": "learned role/subtask embedding plus role token in the causal stream",
        "summary_source": "last non-pad token or causal-prefix mean hidden state at the current decision step, depending on method config",
        "shared_bus_source": "Stage 11H shared_bus methods write causal role summaries into fixed non-overlapping bus subspaces and concatenate the final bus into the Actor action head.",
        "actor_only_emits_action": True,
    }


def environment_summary(device: torch.device) -> Dict[str, object]:
    data: Dict[str, object] = {
        "python": os.sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
    }
    if torch.cuda.is_available():
        data["cuda_device_name"] = torch.cuda.get_device_name(0)
        data["cuda_total_memory"] = int(torch.cuda.get_device_properties(0).total_memory)
    return data


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def write_json(path: Path, data: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
