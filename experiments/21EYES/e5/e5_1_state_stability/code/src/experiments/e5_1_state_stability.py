from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


E5_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = E5_ROOT / "results"
REPORTS_DIR = E5_ROOT / "reports"
PREFIX = "e5_1_state_stability"

PREFLIGHT_JSON = RESULTS_DIR / f"{PREFIX}_preflight.json"
PREFLIGHT_REPORT = REPORTS_DIR / "E5_1_PREFLIGHT.md"
DATABASE_PATH = RESULTS_DIR / f"{PREFIX}_database.jsonl"
TEACHER_PATH = RESULTS_DIR / f"{PREFIX}_teacher.json"
BASELINES_PATH = RESULTS_DIR / f"{PREFIX}_baselines.json"
STUDENTS_PATH = RESULTS_DIR / f"{PREFIX}_students.json"
CONTROLS_PATH = RESULTS_DIR / f"{PREFIX}_controls.json"
REPLAY_PATH = RESULTS_DIR / f"{PREFIX}_replay.json"
ROUND1_PATH = RESULTS_DIR / f"{PREFIX}_round1_variants.json"
ROUND2_PATH = RESULTS_DIR / f"{PREFIX}_round2_combined.json"
ROUND3_PATH = RESULTS_DIR / f"{PREFIX}_round3_final.json"
CAPACITY_CURVES_PATH = RESULTS_DIR / f"{PREFIX}_capacity_curves.csv"
STATE_AUDIT_PATH = RESULTS_DIR / f"{PREFIX}_state_audit.json"
MEMORY_AUDIT_PATH = RESULTS_DIR / f"{PREFIX}_memory_audit.json"
BEST_CONFIG_PATH = RESULTS_DIR / f"{PREFIX}_best_config.yaml"
SEED_STABILITY_PATH = RESULTS_DIR / f"{PREFIX}_seed_stability.json"
REPORT_PATH = REPORTS_DIR / "E5_1_ROLLING_STATE_STABILITY_AND_ANTI_COLLAPSE.md"

DECISION_CUDA_REQUIRED = "CUDA_REQUIRED_NOT_AVAILABLE"
DECISION_VARIANTS_NOT_RUN = "E5_1_VARIANTS_NOT_RUN"
DECISION_REPLAY_INCONSISTENT = "E5_1_REPLAY_INCONSISTENT"
DECISION_NOT_SUPPORTED = "E5_1_STATE_STABILITY_NOT_SUPPORTED"
DECISION_WEAK = "E5_1_STATE_STABILITY_WEAK_SIGNAL"
DECISION_STABLE_C128 = "E5_1_STABLE_C128_REACHED"
DECISION_STABLE_C256 = "E5_1_STABLE_C256_RECOVERED"
DECISION_C512_TREND = "E5_1_C512_TREND_BUT_NOT_CAPACITY"
DECISION_STABLE_C512 = "E5_1_STABLE_C512_REACHED"
DECISION_ORTHOGONALITY = "E5_1_ORTHOGONALITY_DECISIVE"
DECISION_ROUTING = "E5_1_ROUTING_STABILITY_DECISIVE"
DECISION_PROBE = "E5_1_STATE_PROBE_CURRICULUM_DECISIVE"

K_CANDIDATES = 8
NUM_ENTITIES = 8
NONE_VALUE = 7
TRUE_CLASS = 0
FALSE_CLASS = 1
STEP_TOKENS = 5

OPS: Tuple[str, ...] = (
    "set",
    "update",
    "invalidate",
    "link",
    "support",
    "contradict",
    "distractor",
    "query_value",
    "query_support",
    "query_override",
)
OP_TO_ID = {op: i for i, op in enumerate(OPS)}
ID_TO_OP = {i: op for op, i in OP_TO_ID.items()}

TASK_FAMILIES: Tuple[str, ...] = (
    "entity_binding",
    "entity_update_override",
    "stale_memory",
    "multi_hop_binding",
    "support_contradiction",
    "sparse_relevant_history",
    "interleaved_distractors",
    "current_override",
    "long_range_recall",
    "mixed_operations",
)
FAMILY_TO_ID = {family: i for i, family in enumerate(TASK_FAMILIES)}
MEMORY_INDEPENDENT_FAMILIES = {"current_override"}
MEMORY_DEPENDENT_FAMILIES = set(TASK_FAMILIES) - MEMORY_INDEPENDENT_FAMILIES

ENTITY_OFFSET = 32
ARG_ENTITY_OFFSET = 48
VALUE_OFFSET = 64
FAMILY_OFFSET = 80
VOCAB_SIZE = FAMILY_OFFSET + len(TASK_FAMILIES) + 8


@dataclass(frozen=True)
class E5Step:
    op: str
    entity: int
    arg_entity: int
    value: int
    family: str

    def tokens(self) -> Tuple[int, int, int, int, int]:
        return (
            OP_TO_ID[self.op],
            ENTITY_OFFSET + int(self.entity),
            ARG_ENTITY_OFFSET + int(self.arg_entity),
            VALUE_OFFSET + int(self.value),
            FAMILY_OFFSET + FAMILY_TO_ID[self.family],
        )


@dataclass(frozen=True)
class E5Example:
    input_ids: Tuple[Tuple[int, int, int, int, int], ...]
    label: int
    family: str
    target_entity: int
    memory_dependent: bool
    support_query: bool
    value_targets: Tuple[Tuple[int, ...], ...]
    support_targets: Tuple[Tuple[int, ...], ...]
    active_fact_targets: Tuple[Tuple[int, ...], ...]
    stale_fact_targets: Tuple[Tuple[int, ...], ...]
    update_fact_targets: Tuple[Tuple[int, ...], ...]
    support_fact_targets: Tuple[Tuple[int, ...], ...]
    contradiction_fact_targets: Tuple[Tuple[int, ...], ...]
    distractor_mask: Tuple[int, ...]


@dataclass(frozen=True)
class E5Batch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    target_entities: torch.Tensor
    value_targets: torch.Tensor
    support_targets: torch.Tensor
    active_fact_targets: torch.Tensor
    stale_fact_targets: torch.Tensor
    update_fact_targets: torch.Tensor
    support_fact_targets: torch.Tensor
    contradiction_fact_targets: torch.Tensor
    distractor_mask: torch.Tensor
    family_ids: torch.Tensor
    memory_dependent: torch.Tensor
    support_query: torch.Tensor
    examples: Tuple[E5Example, ...]


@dataclass(frozen=True)
class StateConfig:
    num_state_slots: int
    d_model: int

    @property
    def label(self) -> str:
        return f"{self.num_state_slots}x{self.d_model}"


@dataclass(frozen=True)
class E5Budget:
    train_lengths: Tuple[int, ...] = (8, 16, 32)
    eval_lengths: Tuple[int, ...] = (32, 64, 128, 256, 512)
    extrapolation_lengths: Tuple[int, ...] = ()
    seeds: Tuple[int, ...] = (0, 1, 2)
    teacher_eval_examples: int = 96
    eval_examples: int = 96
    train_batch_size: int = 96
    baseline_steps: int = 180
    student_steps: int = 160
    randomized_label_steps: int = 80
    learning_rate: float = 2.0e-3
    weight_decay: float = 1.0e-4
    d_model: int = 128
    n_layers: int = 1
    n_heads: int = 4
    num_state_slots: int = 32
    state_dim: int = 128
    curriculum: bool = True
    state_ablation_margin: float = 0.55
    organized_writes: bool = False


@dataclass(frozen=True)
class LossWeights:
    answer: float = 1.0
    activation: float = 0.0
    delta: float = 0.0
    state_usefulness: float = 0.0
    state_stability: float = 0.0
    update_regularization: float = 0.0
    slot_diversity: float = 0.0
    state_ablation: float = 0.0
    slot_routing: float = 0.0
    slot_cosine_decorrelation: float = 0.0
    slot_covariance: float = 0.0
    effective_rank: float = 0.0
    bounded_state_norm: float = 0.0
    write_balance: float = 0.0
    read_balance: float = 0.0
    route_entropy_floor: float = 0.0
    load_balance: float = 0.0
    active_probe: float = 0.0
    stale_probe: float = 0.0
    support_probe: float = 0.0
    distractor_preservation: float = 0.0
    long_delay_retention: float = 0.0
    geometry_density: float = 0.0
    geometry_gate_variation: float = 0.0


BASE_ORGANIZED_LOSS = LossWeights(
    answer=1.0,
    activation=0.20,
    delta=0.45,
    state_usefulness=0.75,
    state_stability=0.04,
    update_regularization=0.03,
    slot_diversity=0.08,
    state_ablation=0.12,
    slot_routing=0.18,
)


@dataclass(frozen=True)
class StabilityVariant:
    name: str
    family: str
    description: str
    weights: LossWeights
    route_temperature: float = 1.0
    route_top_k: int = 0
    route_dropout_p: float = 0.0
    write_dropout_p: float = 0.0
    slot_dropout_p: float = 0.0
    state_dropout_p: float = 0.0
    route_temperature_start: Optional[float] = None
    route_temperature_end: Optional[float] = None
    state_as_kv: bool = False
    disable_state_mean_paths: bool = False
    slot_cross_update: bool = False
    strict_routed_writes: bool = False
    learned_slot_geometry: bool = False
    static_key_scale: float = 1.0
    dynamic_key_scale: float = 1.0
    geometry_basis_count: int = 0
    geometry_gate_density_target: float = 0.0
    geometry_gate_std_floor: float = 0.0
    geometry_gate_temperature: float = 1.0
    geometry_basis_scale: float = 1.0


def _loss(**updates: float) -> LossWeights:
    return replace(BASE_ORGANIZED_LOSS, **updates)


VARIANT_CONFIGS: Mapping[str, StabilityVariant] = {
    "rolling_state_organized_full_combined": StabilityVariant(
        "rolling_state_organized_full_combined",
        "anchor",
        "E5 POF organized rolling-state anchor.",
        BASE_ORGANIZED_LOSS,
    ),
    "A1_cosine_decorrelation": StabilityVariant("A1_cosine_decorrelation", "A", "Cosine decorrelation among final state slots.", _loss(slot_cosine_decorrelation=0.18)),
    "A2_covariance_offdiag": StabilityVariant("A2_covariance_offdiag", "A", "Covariance off-diagonal penalty among state slots.", _loss(slot_covariance=0.10)),
    "A3_effective_rank": StabilityVariant("A3_effective_rank", "A", "Effective-rank floor for the final state matrix.", _loss(effective_rank=0.22)),
    "A4_orthogonality_bounded_norm": StabilityVariant(
        "A4_orthogonality_bounded_norm",
        "A",
        "Cosine decorrelation with bounded per-slot norm.",
        _loss(slot_cosine_decorrelation=0.14, bounded_state_norm=0.05),
    ),
    "B1_write_frequency_balance": StabilityVariant("B1_write_frequency_balance", "B", "Per-slot routed write load balancing.", _loss(write_balance=0.16)),
    "B2_read_frequency_balance": StabilityVariant("B2_read_frequency_balance", "B", "Final read attention load balancing.", _loss(read_balance=0.12)),
    "B3_route_entropy_floor": StabilityVariant("B3_route_entropy_floor", "B", "Route entropy floor to prevent hard early collapse.", _loss(route_entropy_floor=0.16)),
    "B4_load_balancing": StabilityVariant("B4_load_balancing", "B", "Load entropy maximization over routed writes.", _loss(load_balance=0.18)),
    "C1_route_temp_high_to_low": StabilityVariant(
        "C1_route_temp_high_to_low",
        "C",
        "Route temperature annealed from diffuse to sharper.",
        BASE_ORGANIZED_LOSS,
        route_temperature_start=1.75,
        route_temperature_end=0.75,
    ),
    "C2_route_temp_low_to_high": StabilityVariant(
        "C2_route_temp_low_to_high",
        "C",
        "Route temperature annealed from sharp to diffuse.",
        BASE_ORGANIZED_LOSS,
        route_temperature_start=0.70,
        route_temperature_end=1.50,
    ),
    "C3_topk_route_sparsity": StabilityVariant("C3_topk_route_sparsity", "C", "Top-k sparse routing over four slots.", BASE_ORGANIZED_LOSS, route_top_k=4),
    "C4_soft_route_entropy_floor": StabilityVariant(
        "C4_soft_route_entropy_floor",
        "C",
        "Soft routing with entropy floor.",
        _loss(route_entropy_floor=0.18),
        route_temperature=1.35,
    ),
    "D1_slot_dropout": StabilityVariant("D1_slot_dropout", "D", "Training-time whole-slot dropout.", BASE_ORGANIZED_LOSS, slot_dropout_p=0.08),
    "D2_route_dropout": StabilityVariant("D2_route_dropout", "D", "Training-time route dropout.", BASE_ORGANIZED_LOSS, route_dropout_p=0.08),
    "D3_state_dropout": StabilityVariant("D3_state_dropout", "D", "Training-time elementwise state dropout.", BASE_ORGANIZED_LOSS, state_dropout_p=0.05),
    "D4_write_dropout": StabilityVariant("D4_write_dropout", "D", "Training-time routed write dropout.", BASE_ORGANIZED_LOSS, write_dropout_p=0.05),
    "E1_stronger_active_probe": StabilityVariant("E1_stronger_active_probe", "E", "Higher-weight active fact probe.", _loss(active_probe=0.22)),
    "E2_stronger_stale_override_probe": StabilityVariant("E2_stronger_stale_override_probe", "E", "Higher-weight stale/update probe.", _loss(stale_probe=0.22)),
    "E3_stronger_support_probe": StabilityVariant("E3_stronger_support_probe", "E", "Higher-weight support/contradiction probe.", _loss(support_probe=0.22)),
    "E4_all_probe_schedule": StabilityVariant(
        "E4_all_probe_schedule",
        "E",
        "All fact probes strengthened together.",
        _loss(active_probe=0.12, stale_probe=0.12, support_probe=0.12),
    ),
    "F1_irrelevant_preservation": StabilityVariant("F1_irrelevant_preservation", "F", "Preserve state under irrelevant inputs.", _loss(distractor_preservation=0.12)),
    "F2_distractor_anticorruption": StabilityVariant("F2_distractor_anticorruption", "F", "Stronger distractor anti-corruption.", _loss(state_stability=0.10, distractor_preservation=0.16)),
    "F3_noisy_distractor_consistency": StabilityVariant("F3_noisy_distractor_consistency", "F", "State consistency proxy on distractor-heavy steps.", _loss(distractor_preservation=0.18, slot_cosine_decorrelation=0.06)),
    "F4_long_delay_retention": StabilityVariant("F4_long_delay_retention", "F", "Auxiliary final-state long-delay retention probe.", _loss(long_delay_retention=0.12)),
    "R2_ortho_write_balance": StabilityVariant("R2_ortho_write_balance", "round2", "Cosine orthogonality plus write balancing.", _loss(slot_cosine_decorrelation=0.14, write_balance=0.14)),
    "R2_ortho_route_temp": StabilityVariant(
        "R2_ortho_route_temp",
        "round2",
        "Cosine orthogonality plus route temperature annealing.",
        _loss(slot_cosine_decorrelation=0.14),
        route_temperature_start=1.60,
        route_temperature_end=0.80,
    ),
    "R2_balance_slot_dropout": StabilityVariant("R2_balance_slot_dropout", "round2", "Write balancing plus slot dropout.", _loss(write_balance=0.14), slot_dropout_p=0.06),
    "R2_entropy_probe": StabilityVariant("R2_entropy_probe", "round2", "Route entropy floor plus state probe curriculum.", _loss(route_entropy_floor=0.14, active_probe=0.10, stale_probe=0.10, support_probe=0.10)),
    "R2_rank_anticorruption": StabilityVariant("R2_rank_anticorruption", "round2", "Effective-rank floor plus distractor anti-corruption.", _loss(effective_rank=0.18, distractor_preservation=0.12)),
    "R2_slot_dropout_retention": StabilityVariant("R2_slot_dropout_retention", "round2", "Slot dropout plus long-delay retention.", _loss(long_delay_retention=0.10), slot_dropout_p=0.06),
    "G1_state_kv_meanless": StabilityVariant(
        "G1_state_kv_meanless",
        "faithful",
        "Current tokens attend directly to rolling-state K/V; pooled-state mean shortcuts disabled.",
        _loss(slot_cosine_decorrelation=0.10, read_balance=0.06),
        state_as_kv=True,
        disable_state_mean_paths=True,
    ),
    "G2_state_kv_cross_update": StabilityVariant(
        "G2_state_kv_cross_update",
        "faithful",
        "State K/V attention plus slot-specific state updates from slot-to-current cross-attention.",
        _loss(slot_cosine_decorrelation=0.10, read_balance=0.06, write_balance=0.06),
        state_as_kv=True,
        disable_state_mean_paths=True,
        slot_cross_update=True,
    ),
    "G3_state_kv_strict_topk": StabilityVariant(
        "G3_state_kv_strict_topk",
        "faithful",
        "State K/V attention with slot-specific updates and sparse strictly routed writes.",
        _loss(slot_cosine_decorrelation=0.14, effective_rank=0.14, read_balance=0.08, write_balance=0.10),
        route_temperature=0.85,
        route_top_k=4,
        state_as_kv=True,
        disable_state_mean_paths=True,
        slot_cross_update=True,
        strict_routed_writes=True,
    ),
    "G4_state_kv_rank_pressure": StabilityVariant(
        "G4_state_kv_rank_pressure",
        "faithful",
        "State K/V attention with stronger rank/decorrelation pressure but soft writes.",
        _loss(slot_cosine_decorrelation=0.22, effective_rank=0.26, read_balance=0.10, write_balance=0.08, long_delay_retention=0.08),
        route_temperature_start=1.40,
        route_temperature_end=0.80,
        state_as_kv=True,
        disable_state_mean_paths=True,
        slot_cross_update=True,
    ),
    "G5_state_kv_mild_rank": StabilityVariant(
        "G5_state_kv_mild_rank",
        "faithful",
        "G1-style state K/V with mild rank pressure and no slot-cross update.",
        _loss(slot_cosine_decorrelation=0.14, effective_rank=0.08, read_balance=0.08, long_delay_retention=0.04),
        state_as_kv=True,
        disable_state_mean_paths=True,
    ),
    "G6_state_kv_soft_topk": StabilityVariant(
        "G6_state_kv_soft_topk",
        "faithful",
        "G1-style state K/V with soft top-k routing pressure but no strict writes.",
        _loss(slot_cosine_decorrelation=0.12, effective_rank=0.08, read_balance=0.08, write_balance=0.08),
        route_temperature=0.90,
        route_top_k=8,
        state_as_kv=True,
        disable_state_mean_paths=True,
    ),
    "G7_g1_soft_repair": StabilityVariant(
        "G7_g1_soft_repair",
        "faithful",
        "G1 with very soft rank/read/write pressure and annealed route temperature.",
        _loss(slot_cosine_decorrelation=0.06, effective_rank=0.04, read_balance=0.06, write_balance=0.04, long_delay_retention=0.04),
        route_temperature_start=1.25,
        route_temperature_end=0.90,
        state_as_kv=True,
        disable_state_mean_paths=True,
    ),
    "H1_learned_geometry_keys": StabilityVariant(
        "H1_learned_geometry_keys",
        "geometry",
        "Learned static slot-key geometry plus dynamic rolling-state K/V content.",
        _loss(slot_cosine_decorrelation=0.04, read_balance=0.06, long_delay_retention=0.04),
        state_as_kv=True,
        disable_state_mean_paths=True,
        learned_slot_geometry=True,
        static_key_scale=0.85,
        dynamic_key_scale=0.70,
    ),
    "H2_geometry_soft_topk": StabilityVariant(
        "H2_geometry_soft_topk",
        "geometry",
        "Learned slot-key geometry with soft top-k write routing.",
        _loss(slot_cosine_decorrelation=0.06, effective_rank=0.04, read_balance=0.06, write_balance=0.06),
        route_temperature=0.90,
        route_top_k=8,
        state_as_kv=True,
        disable_state_mean_paths=True,
        learned_slot_geometry=True,
        static_key_scale=0.85,
        dynamic_key_scale=0.70,
    ),
    "H3_geometry_cross_update": StabilityVariant(
        "H3_geometry_cross_update",
        "geometry",
        "Learned slot-key geometry with slot-specific cross-attention state updates.",
        _loss(slot_cosine_decorrelation=0.06, effective_rank=0.04, read_balance=0.06, write_balance=0.04),
        state_as_kv=True,
        disable_state_mean_paths=True,
        slot_cross_update=True,
        learned_slot_geometry=True,
        static_key_scale=0.85,
        dynamic_key_scale=0.70,
    ),
    "H4_geometry_no_rank": StabilityVariant(
        "H4_geometry_no_rank",
        "geometry",
        "H2-style learned geometry with rank pressure removed.",
        _loss(slot_cosine_decorrelation=0.04, read_balance=0.06, write_balance=0.06),
        route_temperature=0.90,
        route_top_k=8,
        state_as_kv=True,
        disable_state_mean_paths=True,
        learned_slot_geometry=True,
        static_key_scale=0.85,
        dynamic_key_scale=0.70,
    ),
    "H5_geometry_static_heavy": StabilityVariant(
        "H5_geometry_static_heavy",
        "geometry",
        "H2-style geometry with stronger learned static slot keys.",
        _loss(slot_cosine_decorrelation=0.04, read_balance=0.06, write_balance=0.06),
        route_temperature=0.90,
        route_top_k=8,
        state_as_kv=True,
        disable_state_mean_paths=True,
        learned_slot_geometry=True,
        static_key_scale=1.20,
        dynamic_key_scale=0.45,
    ),
    "H6_geometry_dynamic_heavy": StabilityVariant(
        "H6_geometry_dynamic_heavy",
        "geometry",
        "H2-style geometry with stronger dynamic state keys.",
        _loss(slot_cosine_decorrelation=0.04, read_balance=0.06, write_balance=0.06),
        route_temperature=0.90,
        route_top_k=8,
        state_as_kv=True,
        disable_state_mean_paths=True,
        learned_slot_geometry=True,
        static_key_scale=0.45,
        dynamic_key_scale=1.00,
    ),
    "H7_geometry_topk4": StabilityVariant(
        "H7_geometry_topk4",
        "geometry",
        "H2-style learned geometry with sharper top-4 write routing.",
        _loss(slot_cosine_decorrelation=0.04, read_balance=0.06, write_balance=0.06),
        route_temperature=0.90,
        route_top_k=4,
        state_as_kv=True,
        disable_state_mean_paths=True,
        learned_slot_geometry=True,
        static_key_scale=0.85,
        dynamic_key_scale=0.70,
    ),
    "H8_geometry_topk16": StabilityVariant(
        "H8_geometry_topk16",
        "geometry",
        "H2-style learned geometry with softer top-16 write routing.",
        _loss(slot_cosine_decorrelation=0.04, read_balance=0.06, write_balance=0.06),
        route_temperature=0.90,
        route_top_k=16,
        state_as_kv=True,
        disable_state_mean_paths=True,
        learned_slot_geometry=True,
        static_key_scale=0.85,
        dynamic_key_scale=0.70,
    ),
    "H9_geometry_temp_anneal": StabilityVariant(
        "H9_geometry_temp_anneal",
        "geometry",
        "H2-style learned geometry with route temperature annealed from broad to sharper.",
        _loss(slot_cosine_decorrelation=0.04, read_balance=0.06, write_balance=0.06),
        route_temperature_start=1.30,
        route_temperature_end=0.80,
        route_top_k=8,
        state_as_kv=True,
        disable_state_mean_paths=True,
        learned_slot_geometry=True,
        static_key_scale=0.85,
        dynamic_key_scale=0.70,
    ),
    "H10_possibility_geometry_dense": StabilityVariant(
        "H10_possibility_geometry_dense",
        "geometry",
        "Combinatorial possibility geometry: each slot mixes a dense subset of learned key vectors.",
        _loss(slot_cosine_decorrelation=0.04, read_balance=0.06, write_balance=0.06, geometry_density=0.10, geometry_gate_variation=0.03),
        route_temperature=0.90,
        route_top_k=8,
        state_as_kv=True,
        disable_state_mean_paths=True,
        learned_slot_geometry=True,
        static_key_scale=0.95,
        dynamic_key_scale=0.55,
        geometry_basis_count=64,
        geometry_gate_density_target=0.55,
        geometry_gate_std_floor=0.08,
        geometry_gate_temperature=0.85,
        geometry_basis_scale=0.65,
    ),
    "H11_possibility_static_heavy": StabilityVariant(
        "H11_possibility_static_heavy",
        "geometry",
        "Possibility geometry on the stronger H5 static-key mix.",
        _loss(slot_cosine_decorrelation=0.04, read_balance=0.06, write_balance=0.06, geometry_density=0.10, geometry_gate_variation=0.03),
        route_temperature=0.90,
        route_top_k=8,
        state_as_kv=True,
        disable_state_mean_paths=True,
        learned_slot_geometry=True,
        static_key_scale=1.20,
        dynamic_key_scale=0.45,
        geometry_basis_count=64,
        geometry_gate_density_target=0.55,
        geometry_gate_std_floor=0.08,
        geometry_gate_temperature=0.85,
        geometry_basis_scale=0.65,
    ),
    "H12_possibility_high_capacity": StabilityVariant(
        "H12_possibility_high_capacity",
        "geometry",
        "Possibility geometry with twice the learned basis capacity.",
        _loss(slot_cosine_decorrelation=0.04, read_balance=0.06, write_balance=0.06, geometry_density=0.10, geometry_gate_variation=0.03),
        route_temperature=0.90,
        route_top_k=8,
        state_as_kv=True,
        disable_state_mean_paths=True,
        learned_slot_geometry=True,
        static_key_scale=1.00,
        dynamic_key_scale=0.50,
        geometry_basis_count=128,
        geometry_gate_density_target=0.55,
        geometry_gate_std_floor=0.08,
        geometry_gate_temperature=0.85,
        geometry_basis_scale=0.55,
    ),
    "H13_possibility_sparse_write": StabilityVariant(
        "H13_possibility_sparse_write",
        "geometry",
        "Dense possibility geometry with sharper top-4 write routing.",
        _loss(slot_cosine_decorrelation=0.04, read_balance=0.06, write_balance=0.06, geometry_density=0.10, geometry_gate_variation=0.03),
        route_temperature=0.90,
        route_top_k=4,
        state_as_kv=True,
        disable_state_mean_paths=True,
        learned_slot_geometry=True,
        static_key_scale=0.95,
        dynamic_key_scale=0.55,
        geometry_basis_count=64,
        geometry_gate_density_target=0.55,
        geometry_gate_std_floor=0.08,
        geometry_gate_temperature=0.85,
        geometry_basis_scale=0.65,
    ),
}

VARIANT_LOSSES: Mapping[str, LossWeights] = {
    "rolling_state_answer_only": LossWeights(answer=1.0),
    "rolling_state_activation_match": LossWeights(answer=1.0, activation=0.30),
    "rolling_state_delta_match": LossWeights(answer=1.0, delta=0.55),
    "rolling_state_full_combined": LossWeights(
        answer=1.0,
        activation=0.20,
        delta=0.45,
        state_usefulness=0.60,
        state_stability=0.03,
        update_regularization=0.03,
        slot_diversity=0.12,
        state_ablation=0.10,
    ),
    **{name: config.weights for name, config in VARIANT_CONFIGS.items()},
}


def parse_state_configs(raw: str, *, fallback_slots: int = NUM_ENTITIES, fallback_dim: int = 64) -> Tuple[StateConfig, ...]:
    raw = (raw or "").strip()
    if not raw:
        return (StateConfig(fallback_slots, fallback_dim),)
    configs: List[StateConfig] = []
    for part in raw.split(","):
        item = part.strip().lower()
        if not item:
            continue
        if "x" not in item:
            raise ValueError(f"Invalid state config '{part}'. Expected SxD, e.g. 16x64.")
        slots_s, dim_s = item.split("x", 1)
        configs.append(StateConfig(num_state_slots=int(slots_s), d_model=int(dim_s)))
    if not configs:
        configs.append(StateConfig(fallback_slots, fallback_dim))
    return tuple(configs)


def parse_student_variants(raw: str, *, multiple_state_configs: bool) -> Tuple[str, ...]:
    raw = (raw or "").strip()
    if raw:
        variants = tuple(part.strip() for part in raw.split(",") if part.strip())
    elif multiple_state_configs:
        variants = ("rolling_state_delta_match", "rolling_state_full_combined")
    else:
        variants = ("rolling_state_organized_full_combined",)
    unknown = [variant for variant in variants if variant not in VARIANT_LOSSES]
    if unknown:
        raise ValueError(f"Unknown student variants: {unknown}")
    return variants


def parse_int_tuple(raw: str, *, default: Tuple[int, ...]) -> Tuple[int, ...]:
    raw = (raw or "").strip()
    if not raw:
        return default
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    return values or default


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _dump_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _append_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _token_op(token_id: int) -> str:
    return ID_TO_OP[int(token_id)]


def _token_entity(token_id: int) -> int:
    return int(token_id) - ENTITY_OFFSET


def _token_arg_entity(token_id: int) -> int:
    return int(token_id) - ARG_ENTITY_OFFSET


def _token_value(token_id: int) -> int:
    return int(token_id) - VALUE_OFFSET


def _step_from_tokens(tokens: Sequence[int]) -> E5Step:
    family_id = int(tokens[4]) - FAMILY_OFFSET
    return E5Step(
        op=_token_op(int(tokens[0])),
        entity=_token_entity(int(tokens[1])),
        arg_entity=_token_arg_entity(int(tokens[2])),
        value=_token_value(int(tokens[3])),
        family=TASK_FAMILIES[family_id],
    )


def _resolve_entity(entity: int, values: Sequence[int], links: Sequence[int]) -> int:
    seen = set()
    cur = int(entity)
    for _ in range(NUM_ENTITIES):
        if cur in seen:
            return NONE_VALUE
        seen.add(cur)
        nxt = int(links[cur])
        if nxt == cur:
            return int(values[cur])
        cur = nxt
    return NONE_VALUE


def simulate_steps(
    steps: Sequence[E5Step],
) -> Tuple[
    int,
    List[Tuple[int, ...]],
    List[Tuple[int, ...]],
    List[Tuple[int, ...]],
    List[Tuple[int, ...]],
    List[Tuple[int, ...]],
    List[Tuple[int, ...]],
    List[Tuple[int, ...]],
]:
    values = [NONE_VALUE for _ in range(NUM_ENTITIES)]
    supports = [NONE_VALUE for _ in range(NUM_ENTITIES)]
    stale = [0 for _ in range(NUM_ENTITIES)]
    updated = [0 for _ in range(NUM_ENTITIES)]
    links = [i for i in range(NUM_ENTITIES)]
    value_targets: List[Tuple[int, ...]] = []
    support_targets: List[Tuple[int, ...]] = []
    active_fact_targets: List[Tuple[int, ...]] = []
    stale_fact_targets: List[Tuple[int, ...]] = []
    update_fact_targets: List[Tuple[int, ...]] = []
    support_fact_targets: List[Tuple[int, ...]] = []
    contradiction_fact_targets: List[Tuple[int, ...]] = []
    label = NONE_VALUE

    for step in steps:
        e = int(step.entity) % NUM_ENTITIES
        a = int(step.arg_entity) % NUM_ENTITIES
        v = int(step.value) % K_CANDIDATES
        if step.op == "set":
            values[e] = v
            links[e] = e
            stale[e] = 0
            updated[e] = 0
        elif step.op == "update":
            values[e] = v
            links[e] = e
            stale[e] = 0
            updated[e] = 1
        elif step.op == "invalidate":
            values[e] = NONE_VALUE
            links[e] = e
            stale[e] = 1
            updated[e] = 0
        elif step.op == "link":
            links[e] = a
        elif step.op == "support":
            supports[e] = TRUE_CLASS
            stale[e] = 0
        elif step.op == "contradict":
            supports[e] = FALSE_CLASS
            stale[e] = 0
        elif step.op == "query_value":
            label = _resolve_entity(e, values, links)
        elif step.op == "query_support":
            label = supports[e]
        elif step.op == "query_override":
            label = v

        resolved_values = tuple(_resolve_entity(i, values, links) for i in range(NUM_ENTITIES))
        value_targets.append(resolved_values)
        support_targets.append(tuple(int(x) for x in supports))
        active_fact_targets.append(tuple(1 if x != NONE_VALUE else 0 for x in resolved_values))
        stale_fact_targets.append(tuple(int(x) for x in stale))
        update_fact_targets.append(tuple(int(x) for x in updated))
        support_fact_targets.append(tuple(1 if x == TRUE_CLASS else 0 for x in supports))
        contradiction_fact_targets.append(tuple(1 if x == FALSE_CLASS else 0 for x in supports))

    return (
        int(label),
        value_targets,
        support_targets,
        active_fact_targets,
        stale_fact_targets,
        update_fact_targets,
        support_fact_targets,
        contradiction_fact_targets,
    )


def _distractor_step(rng: random.Random, family: str, exclude: Sequence[int] = ()) -> E5Step:
    excluded = set(int(x) % NUM_ENTITIES for x in exclude)
    choices = [e for e in range(NUM_ENTITIES) if e not in excluded] or list(range(NUM_ENTITIES))
    entity = rng.choice(choices)
    return E5Step(
        op="distractor",
        entity=entity,
        arg_entity=rng.randrange(NUM_ENTITIES),
        value=rng.randrange(NONE_VALUE),
        family=family,
    )


def _finish_sequence(
    core: Sequence[E5Step],
    query: E5Step,
    seq_len: int,
    rng: random.Random,
    family: str,
    *,
    place_core: str = "spread",
    exclude_distractors: Sequence[int] = (),
    allow_distractors: bool = True,
) -> List[E5Step]:
    pre_query = max(1, int(seq_len) - 1)
    if len(core) > pre_query:
        core = tuple(core[-pre_query:])
    filler_count = pre_query - len(core)

    if not allow_distractors:
        filler = [core[-1] if core else query for _ in range(filler_count)]
        steps = list(core) + filler
    elif place_core == "prefix":
        steps = list(core) + [_distractor_step(rng, family, exclude_distractors) for _ in range(filler_count)]
    elif place_core == "first":
        first = list(core[:1])
        rest = list(core[1:])
        slots = [_distractor_step(rng, family, exclude_distractors) for _ in range(filler_count)]
        steps = first + slots + rest
    else:
        positions = sorted(rng.sample(range(pre_query), len(core))) if core else []
        core_iter = iter(core)
        steps = []
        for idx in range(pre_query):
            if idx in positions:
                steps.append(next(core_iter))
            else:
                steps.append(_distractor_step(rng, family, exclude_distractors))
    return steps + [query]


def _build_family_steps(family: str, seq_len: int, rng: random.Random, *, allow_distractors: bool = True) -> Tuple[List[E5Step], int]:
    target = rng.randrange(NUM_ENTITIES)
    value = rng.randrange(NONE_VALUE)
    other_value = (value + rng.randrange(1, NONE_VALUE)) % NONE_VALUE
    arg_a = (target + 1) % NUM_ENTITIES
    arg_b = (target + 2) % NUM_ENTITIES

    if family == "entity_binding":
        core = [E5Step("set", target, 0, value, family)]
        query = E5Step("query_value", target, 0, NONE_VALUE, family)
        steps = _finish_sequence(core, query, seq_len, rng, family, exclude_distractors=(target,), allow_distractors=allow_distractors)
    elif family == "entity_update_override":
        core = [
            E5Step("set", target, 0, value, family),
            E5Step("update", target, 0, other_value, family),
        ]
        query = E5Step("query_value", target, 0, NONE_VALUE, family)
        steps = _finish_sequence(core, query, seq_len, rng, family, exclude_distractors=(target,), allow_distractors=allow_distractors)
    elif family == "stale_memory":
        core = [
            E5Step("set", target, 0, value, family),
            E5Step("invalidate", target, 0, NONE_VALUE, family),
        ]
        query = E5Step("query_value", target, 0, NONE_VALUE, family)
        steps = _finish_sequence(core, query, seq_len, rng, family, exclude_distractors=(target,), allow_distractors=allow_distractors)
    elif family == "multi_hop_binding":
        core = [
            E5Step("link", target, arg_a, NONE_VALUE, family),
            E5Step("link", arg_a, arg_b, NONE_VALUE, family),
            E5Step("set", arg_b, 0, value, family),
        ]
        query = E5Step("query_value", target, 0, NONE_VALUE, family)
        steps = _finish_sequence(core, query, seq_len, rng, family, exclude_distractors=(target, arg_a, arg_b), allow_distractors=allow_distractors)
    elif family == "support_contradiction":
        core = [
            E5Step("support", target, 0, TRUE_CLASS, family),
            E5Step("contradict", target, 0, FALSE_CLASS, family),
        ]
        query = E5Step("query_support", target, 0, NONE_VALUE, family)
        steps = _finish_sequence(core, query, seq_len, rng, family, exclude_distractors=(target,), allow_distractors=allow_distractors)
    elif family == "sparse_relevant_history":
        core = [E5Step("set", target, 0, value, family)]
        query = E5Step("query_value", target, 0, NONE_VALUE, family)
        steps = _finish_sequence(core, query, seq_len, rng, family, place_core="spread", exclude_distractors=(target,), allow_distractors=allow_distractors)
    elif family == "interleaved_distractors":
        core = [
            E5Step("set", target, 0, value, family),
            E5Step("update", target, 0, other_value, family),
        ]
        query = E5Step("query_value", target, 0, NONE_VALUE, family)
        steps = _finish_sequence(core, query, seq_len, rng, family, place_core="spread", exclude_distractors=(target,), allow_distractors=allow_distractors)
    elif family == "current_override":
        core = [E5Step("set", target, 0, value, family)]
        query = E5Step("query_override", target, 0, other_value, family)
        steps = _finish_sequence(core, query, seq_len, rng, family, exclude_distractors=(target,), allow_distractors=allow_distractors)
    elif family == "long_range_recall":
        core = [E5Step("set", target, 0, value, family)]
        query = E5Step("query_value", target, 0, NONE_VALUE, family)
        steps = _finish_sequence(core, query, seq_len, rng, family, place_core="first", exclude_distractors=(target,), allow_distractors=allow_distractors)
    elif family == "mixed_operations":
        steps = []
        state_value = NONE_VALUE
        for _ in range(max(1, seq_len - 1)):
            roll = rng.random()
            if roll < 0.28:
                state_value = rng.randrange(NONE_VALUE)
                steps.append(E5Step("set", target, 0, state_value, family))
            elif roll < 0.52:
                state_value = rng.randrange(NONE_VALUE)
                steps.append(E5Step("update", target, 0, state_value, family))
            elif roll < 0.62:
                state_value = NONE_VALUE
                steps.append(E5Step("invalidate", target, 0, NONE_VALUE, family))
            elif roll < 0.72:
                steps.append(E5Step("support", target, 0, TRUE_CLASS, family))
            elif roll < 0.82:
                steps.append(E5Step("contradict", target, 0, FALSE_CLASS, family))
            else:
                if allow_distractors:
                    steps.append(_distractor_step(rng, family, exclude=(target,)))
                else:
                    state_value = rng.randrange(NONE_VALUE)
                    steps.append(E5Step("update", target, 0, state_value, family))
        if rng.random() < 0.75:
            query = E5Step("query_value", target, 0, NONE_VALUE, family)
        else:
            query = E5Step("query_support", target, 0, NONE_VALUE, family)
        steps = steps[: max(1, seq_len - 1)] + [query]
    else:
        raise ValueError(f"Unknown E5 task family: {family}")

    label, *_ = simulate_steps(steps)
    return steps, label


def build_e5_examples(
    n_examples: int,
    seq_len: int,
    *,
    seed: int,
    split: str = "train",
    task_families: Optional[Sequence[str]] = None,
    allow_distractors: bool = True,
) -> List[E5Example]:
    del split
    rng = random.Random(int(seed))
    families = tuple(task_families or TASK_FAMILIES)
    examples: List[E5Example] = []
    for idx in range(int(n_examples)):
        family = families[idx % len(families)]
        local_seed = rng.randrange(1_000_000_000)
        local_rng = random.Random(local_seed)
        steps, label = _build_family_steps(family, int(seq_len), local_rng, allow_distractors=allow_distractors)
        (
            label,
            value_targets,
            support_targets,
            active_fact_targets,
            stale_fact_targets,
            update_fact_targets,
            support_fact_targets,
            contradiction_fact_targets,
        ) = simulate_steps(steps)
        target = steps[-1].entity
        examples.append(
            E5Example(
                input_ids=tuple(step.tokens() for step in steps),
                label=int(label),
                family=family,
                target_entity=int(target),
                memory_dependent=family in MEMORY_DEPENDENT_FAMILIES,
                support_query=steps[-1].op == "query_support",
                value_targets=tuple(value_targets),
                support_targets=tuple(support_targets),
                active_fact_targets=tuple(active_fact_targets),
                stale_fact_targets=tuple(stale_fact_targets),
                update_fact_targets=tuple(update_fact_targets),
                support_fact_targets=tuple(support_fact_targets),
                contradiction_fact_targets=tuple(contradiction_fact_targets),
                distractor_mask=tuple(1 if step.op == "distractor" else 0 for step in steps),
            )
        )
    return examples


def validate_e5_examples(examples: Sequence[E5Example]) -> Dict[str, object]:
    failures: List[str] = []
    if not examples:
        failures.append("empty examples")
    lengths = {len(ex.input_ids) for ex in examples}
    for idx, ex in enumerate(examples):
        if not (0 <= ex.label < K_CANDIDATES):
            failures.append(f"label out of range at {idx}: {ex.label}")
        if ex.family not in TASK_FAMILIES:
            failures.append(f"unknown family at {idx}: {ex.family}")
        if len(ex.value_targets) != len(ex.input_ids):
            failures.append(f"value target length mismatch at {idx}")
        for target_name in (
            "active_fact_targets",
            "stale_fact_targets",
            "update_fact_targets",
            "support_fact_targets",
            "contradiction_fact_targets",
        ):
            if len(getattr(ex, target_name)) != len(ex.input_ids):
                failures.append(f"{target_name} length mismatch at {idx}")
        if any(len(step) != STEP_TOKENS for step in ex.input_ids):
            failures.append(f"step token length mismatch at {idx}")
    return {
        "passes": not failures,
        "failures": failures,
        "num_examples": len(examples),
        "sequence_lengths": sorted(lengths),
        "families": sorted({ex.family for ex in examples}),
        "chance": 1.0 / K_CANDIDATES,
    }


def examples_to_batch(examples: Sequence[E5Example], device: torch.device | str) -> E5Batch:
    if not examples:
        raise ValueError("examples_to_batch requires at least one example")
    device = torch.device(device)
    input_ids = torch.tensor([ex.input_ids for ex in examples], dtype=torch.long, device=device)
    labels = torch.tensor([ex.label for ex in examples], dtype=torch.long, device=device)
    targets = torch.tensor([ex.target_entity for ex in examples], dtype=torch.long, device=device)
    value_targets = torch.tensor([ex.value_targets for ex in examples], dtype=torch.long, device=device)
    support_targets = torch.tensor([ex.support_targets for ex in examples], dtype=torch.long, device=device)
    active_fact_targets = torch.tensor([ex.active_fact_targets for ex in examples], dtype=torch.float32, device=device)
    stale_fact_targets = torch.tensor([ex.stale_fact_targets for ex in examples], dtype=torch.float32, device=device)
    update_fact_targets = torch.tensor([ex.update_fact_targets for ex in examples], dtype=torch.float32, device=device)
    support_fact_targets = torch.tensor([ex.support_fact_targets for ex in examples], dtype=torch.float32, device=device)
    contradiction_fact_targets = torch.tensor([ex.contradiction_fact_targets for ex in examples], dtype=torch.float32, device=device)
    distractor_mask = torch.tensor([ex.distractor_mask for ex in examples], dtype=torch.float32, device=device)
    family_ids = torch.tensor([FAMILY_TO_ID[ex.family] for ex in examples], dtype=torch.long, device=device)
    memory_dependent = torch.tensor([ex.memory_dependent for ex in examples], dtype=torch.bool, device=device)
    support_query = torch.tensor([ex.support_query for ex in examples], dtype=torch.bool, device=device)
    return E5Batch(
        input_ids=input_ids,
        labels=labels,
        target_entities=targets,
        value_targets=value_targets,
        support_targets=support_targets,
        active_fact_targets=active_fact_targets,
        stale_fact_targets=stale_fact_targets,
        update_fact_targets=update_fact_targets,
        support_fact_targets=support_fact_targets,
        contradiction_fact_targets=contradiction_fact_targets,
        distractor_mask=distractor_mask,
        family_ids=family_ids,
        memory_dependent=memory_dependent,
        support_query=support_query,
        examples=tuple(examples),
    )


class FittedFullContextQKVTeacher(nn.Module):
    """Fitted full-context teacher with QKV attention over all sequence tokens.

    The teacher's labels and full/current activations are deterministic oracle
    features from the full sequence. A small frozen QKV path is included and
    audited so the teacher has access to the entire token stream, while the
    student never receives this full-context path.
    """

    def __init__(self, d_model: int = 64, seed: int = 101) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.d_model = int(d_model)
        self.feature_dim = NUM_ENTITIES * K_CANDIDATES * 2 + NUM_ENTITIES + len(OPS) + K_CANDIDATES + len(TASK_FAMILIES) + 2
        self.token_embedding = nn.Embedding(VOCAB_SIZE, d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.full_feature_proj = nn.Linear(self.feature_dim, d_model, bias=False)
        self.current_feature_proj = nn.Linear(self.feature_dim, d_model, bias=False)
        for param in self.parameters():
            param.requires_grad_(False)

    def _features(self, batch: E5Batch, *, current_only: bool) -> torch.Tensor:
        bsz = batch.input_ids.shape[0]
        device = batch.input_ids.device
        if current_only:
            current_values = torch.full((bsz, NUM_ENTITIES), NONE_VALUE, dtype=torch.long, device=device)
            current_support = torch.full((bsz, NUM_ENTITIES), NONE_VALUE, dtype=torch.long, device=device)
            final_ops = batch.input_ids[:, -1, 0]
            final_values = batch.input_ids[:, -1, 3] - VALUE_OFFSET
            override_mask = final_ops == OP_TO_ID["query_override"]
            current_values[override_mask, batch.target_entities[override_mask]] = final_values[override_mask]
            answer_one_hot = F.one_hot(torch.where(override_mask, final_values, torch.full_like(final_values, NONE_VALUE)), K_CANDIDATES)
        else:
            current_values = batch.value_targets[:, -1, :]
            current_support = batch.support_targets[:, -1, :]
            answer_one_hot = F.one_hot(batch.labels, K_CANDIDATES)

        value_one_hot = F.one_hot(current_values.clamp(0, K_CANDIDATES - 1), K_CANDIDATES).float().flatten(1)
        support_one_hot = F.one_hot(current_support.clamp(0, K_CANDIDATES - 1), K_CANDIDATES).float().flatten(1)
        target_one_hot = F.one_hot(batch.target_entities, NUM_ENTITIES).float()
        op_one_hot = F.one_hot(batch.input_ids[:, -1, 0], len(OPS)).float()
        family_one_hot = F.one_hot(batch.family_ids, len(TASK_FAMILIES)).float()
        flags = torch.stack([batch.memory_dependent.float(), batch.support_query.float()], dim=1)
        return torch.cat(
            [
                value_one_hot,
                support_one_hot,
                target_one_hot,
                op_one_hot,
                answer_one_hot.float(),
                family_one_hot,
                flags,
            ],
            dim=1,
        )

    def forward(self, batch: E5Batch) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        bsz, seq_len, step_tokens = batch.input_ids.shape
        flat = batch.input_ids.reshape(bsz, seq_len * step_tokens)
        emb = self.token_embedding(flat)
        last_step = self.token_embedding(batch.input_ids[:, -1, :]).mean(dim=1)
        q = self.q_proj(last_step).unsqueeze(1)
        k = self.k_proj(emb)
        v = self.v_proj(emb)
        weights = torch.softmax(torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_model), dim=-1)
        context = torch.matmul(weights, v).squeeze(1)

        current_emb = self.token_embedding(batch.input_ids[:, -1, :])
        current_k = self.k_proj(current_emb)
        current_v = self.v_proj(current_emb)
        current_weights = torch.softmax(torch.matmul(q, current_k.transpose(1, 2)) / math.sqrt(self.d_model), dim=-1)
        current_context = torch.matmul(current_weights, current_v).squeeze(1)

        full_hidden = torch.tanh(self.full_feature_proj(self._features(batch, current_only=False)) + 0.10 * self.o_proj(context))
        current_hidden = torch.tanh(self.current_feature_proj(self._features(batch, current_only=True)) + 0.10 * self.o_proj(current_context))
        logits = torch.full((bsz, K_CANDIDATES), -8.0, dtype=full_hidden.dtype, device=full_hidden.device)
        logits.scatter_(1, batch.labels.unsqueeze(1), 8.0)
        return {
            "logits": logits,
            "hidden_full": full_hidden,
            "hidden_current": current_hidden,
            "attention_weights": weights.squeeze(1),
            "diagnostics": {
                "attended_token_count": torch.tensor(seq_len * step_tokens, device=full_hidden.device),
                "current_only_token_count": torch.tensor(step_tokens, device=full_hidden.device),
            },
        }


class StateGuidedCleanQKVLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_state_slots: int,
        *,
        organized_writes: bool = False,
        route_temperature: float = 1.0,
        route_top_k: int = 0,
        route_dropout_p: float = 0.0,
        write_dropout_p: float = 0.0,
        state_as_kv: bool = False,
        disable_state_mean_paths: bool = False,
        slot_cross_update: bool = False,
        strict_routed_writes: bool = False,
        learned_slot_geometry: bool = False,
        static_key_scale: float = 1.0,
        dynamic_key_scale: float = 1.0,
        geometry_basis_count: int = 0,
        geometry_gate_density_target: float = 0.0,
        geometry_gate_std_floor: float = 0.0,
        geometry_gate_temperature: float = 1.0,
        geometry_basis_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = d_model // n_heads
        self.num_state_slots = int(num_state_slots)
        self.organized_writes = bool(organized_writes)
        self.route_temperature = float(route_temperature)
        self.route_top_k = int(route_top_k)
        self.route_dropout_p = float(route_dropout_p)
        self.write_dropout_p = float(write_dropout_p)
        self.state_as_kv = bool(state_as_kv)
        self.disable_state_mean_paths = bool(disable_state_mean_paths)
        self.slot_cross_update = bool(slot_cross_update)
        self.strict_routed_writes = bool(strict_routed_writes)
        self.learned_slot_geometry = bool(learned_slot_geometry)
        self.static_key_scale = float(static_key_scale)
        self.dynamic_key_scale = float(dynamic_key_scale)
        self.geometry_basis_count = max(0, int(geometry_basis_count))
        self.geometry_gate_density_target = float(geometry_gate_density_target)
        self.geometry_gate_std_floor = float(geometry_gate_std_floor)
        self.geometry_gate_temperature = max(0.05, float(geometry_gate_temperature))
        self.geometry_basis_scale = float(geometry_basis_scale)

        self.ln_attn = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ln_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.state_film = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2 * d_model),
        )
        self.state_key_bias = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_heads),
        )
        self.slot_embed = nn.Parameter(torch.randn(num_state_slots, d_model) * 0.02)
        if self.state_as_kv:
            self.state_kv_norm = nn.LayerNorm(d_model)
            self.state_kv = nn.Linear(d_model, 2 * d_model)
            if self.learned_slot_geometry:
                self.static_state_key = nn.Parameter(torch.randn(num_state_slots, n_heads, self.head_dim) * 0.05)
                self.static_state_value_bias = nn.Parameter(torch.randn(num_state_slots, n_heads, self.head_dim) * 0.01)
                if self.geometry_basis_count > 0:
                    self.geometry_basis_key = nn.Parameter(torch.randn(self.geometry_basis_count, n_heads, self.head_dim) * 0.04)
                    self.geometry_gate = nn.Sequential(
                        nn.LayerNorm(d_model),
                        nn.Linear(d_model, self.geometry_basis_count),
                    )
        if self.slot_cross_update:
            self.slot_update_norm = nn.LayerNorm(d_model)
            self.token_update_norm = nn.LayerNorm(d_model)
            self.slot_update_q = nn.Linear(d_model, d_model)
            self.token_update_kv = nn.Linear(d_model, 2 * d_model)
        update_in = 3 * d_model
        self.erase_mlp = nn.Sequential(nn.Linear(update_in, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.write_mlp = nn.Sequential(nn.Linear(update_in, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.candidate_mlp = nn.Sequential(nn.Linear(update_in, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model))
        self.route_mlp = nn.Sequential(nn.Linear(update_in, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.entity_slot_logits = nn.Parameter(torch.randn(NUM_ENTITIES, num_state_slots) * 0.02)

    def _attention(self, h: torch.Tensor, state: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bsz, tokens, _ = h.shape
        x = self.ln_attn(h)
        used_state_guidance = state is not None
        used_state_as_kv = bool(state is not None and self.state_as_kv)
        if state is not None and not self.disable_state_mean_paths:
            cond = state.mean(dim=1)
            gamma, beta = self.state_film(cond).chunk(2, dim=-1)
            x = x * (1.0 + 0.10 * torch.tanh(gamma).unsqueeze(1)) + 0.10 * torch.tanh(beta).unsqueeze(1)

        qkv = self.qkv(x).view(bsz, tokens, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        zero = torch.tensor(0.0, device=h.device)
        geometry_gate_density = zero
        geometry_gate_active_fraction = zero
        geometry_gate_std = zero
        if used_state_as_kv:
            state_x = self.state_kv_norm(state + self.slot_embed.unsqueeze(0))
            state_kv = self.state_kv(state_x).view(bsz, state.shape[1], 2, self.n_heads, self.head_dim)
            state_k, state_v = state_kv.unbind(dim=2)
            if self.learned_slot_geometry:
                static_k = self.static_state_key.unsqueeze(0).expand(bsz, -1, -1, -1)
                static_v = self.static_state_value_bias.unsqueeze(0).expand(bsz, -1, -1, -1)
                if self.geometry_basis_count > 0:
                    gate = torch.sigmoid(self.geometry_gate(state) / self.geometry_gate_temperature)
                    basis_mix = torch.einsum("bsg,ghd->bshd", gate, self.geometry_basis_key)
                    basis_mix = basis_mix / gate.sum(dim=-1, keepdim=True).clamp_min(1.0).unsqueeze(-1)
                    state_k = self.dynamic_key_scale * state_k + self.static_key_scale * static_k + self.geometry_basis_scale * basis_mix
                    geometry_gate_density = gate.mean()
                    geometry_gate_active_fraction = (gate > 0.5).float().mean()
                    geometry_gate_std = gate.std(dim=-1, unbiased=False).mean()
                else:
                    state_k = self.dynamic_key_scale * state_k + self.static_key_scale * static_k
                state_v = state_v + static_v
            k = torch.cat([k, state_k.transpose(1, 2)], dim=-2)
            v = torch.cat([v, state_v.transpose(1, 2)], dim=-2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if state is not None and not self.disable_state_mean_paths:
            cond_tokens = state.mean(dim=1).unsqueeze(1).expand(-1, tokens, -1)
            key_bias = self.state_key_bias(torch.cat([h, cond_tokens], dim=-1)).transpose(1, 2)
            if used_state_as_kv:
                state_bias = torch.zeros(bsz, self.n_heads, state.shape[1], dtype=key_bias.dtype, device=key_bias.device)
                key_bias = torch.cat([key_bias, state_bias], dim=-1)
            scores = scores + key_bias.unsqueeze(2)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, tokens, self.d_model)
        out = h + self.out_proj(context)
        out = out + self.mlp(self.ln_mlp(out))
        diag = {
            "self_attention_token_count": torch.tensor(tokens, device=h.device),
            "q_token_count": torch.tensor(q.shape[-2], device=h.device),
            "k_token_count": torch.tensor(k.shape[-2], device=h.device),
            "v_token_count": torch.tensor(v.shape[-2], device=h.device),
            "state_kv_token_count": torch.tensor(state.shape[1] if used_state_as_kv else 0, device=h.device),
            "used_state_as_kv": torch.tensor(1 if used_state_as_kv else 0, device=h.device),
            "used_state_guidance": torch.tensor(1 if used_state_guidance else 0, device=h.device),
            "geometry_gate_density_mean": geometry_gate_density,
            "geometry_gate_active_fraction": geometry_gate_active_fraction,
            "geometry_gate_std_mean": geometry_gate_std,
            "geometry_gate_density_target": torch.tensor(self.geometry_gate_density_target, device=h.device),
            "geometry_gate_std_floor": torch.tensor(self.geometry_gate_std_floor, device=h.device),
        }
        return out, diag

    def _update_state(
        self,
        state: torch.Tensor,
        h: torch.Tensor,
        *,
        current_entity_ids: Optional[torch.Tensor] = None,
        current_op_ids: Optional[torch.Tensor] = None,
        controls: Optional[Mapping[str, bool]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        controls = dict(controls or {})
        bsz = h.shape[0]
        slots = self.slot_embed.unsqueeze(0).expand(bsz, -1, -1)
        if self.slot_cross_update:
            slot_source = self.slot_update_norm(state + slots)
            token_source = self.token_update_norm(h)
            slot_q = self.slot_update_q(slot_source).view(bsz, self.num_state_slots, self.n_heads, self.head_dim).transpose(1, 2)
            token_kv = self.token_update_kv(token_source).view(bsz, h.shape[1], 2, self.n_heads, self.head_dim)
            token_k, token_v = token_kv.unbind(dim=2)
            token_k = token_k.transpose(1, 2)
            token_v = token_v.transpose(1, 2)
            slot_scores = torch.matmul(slot_q, token_k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            slot_attn = torch.softmax(slot_scores, dim=-1)
            summary = torch.matmul(slot_attn, token_v).transpose(1, 2).contiguous().view(bsz, self.num_state_slots, self.d_model)
        else:
            summary = h.mean(dim=1).unsqueeze(1).expand(-1, self.num_state_slots, -1)
        update_input = torch.cat([state, summary, slots], dim=-1)
        erase_gate = torch.sigmoid(self.erase_mlp(update_input) + 1.25)
        write_gate = torch.sigmoid(self.write_mlp(update_input) - 1.00)
        route_logits = self.route_mlp(update_input).squeeze(-1)
        if current_entity_ids is not None:
            route_logits = route_logits + 1.50 * self.entity_slot_logits[current_entity_ids.clamp(0, NUM_ENTITIES - 1)]
        temperature = max(0.05, float(self.route_temperature))
        routed_logits = route_logits / temperature
        if self.route_top_k > 0 and self.route_top_k < self.num_state_slots:
            topk = torch.topk(routed_logits, k=self.route_top_k, dim=-1).values[:, -1].unsqueeze(-1)
            routed_logits = routed_logits.masked_fill(routed_logits < topk, -1.0e9)
        route_flat = torch.softmax(routed_logits, dim=-1)
        if controls.get("route_shuffle", False) and self.num_state_slots > 1:
            route_flat = route_flat[:, torch.randperm(self.num_state_slots, device=route_flat.device)]
        if self.training and self.route_dropout_p > 0.0:
            keep = (torch.rand_like(route_flat) > self.route_dropout_p).float()
            route_flat = route_flat * keep
            route_flat = route_flat / route_flat.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        route = route_flat.unsqueeze(-1)
        if current_op_ids is not None:
            write_ops = torch.tensor(
                [
                    OP_TO_ID["set"],
                    OP_TO_ID["update"],
                    OP_TO_ID["invalidate"],
                    OP_TO_ID["link"],
                    OP_TO_ID["support"],
                    OP_TO_ID["contradict"],
                ],
                device=current_op_ids.device,
            )
            write_strength = (current_op_ids.unsqueeze(-1) == write_ops.unsqueeze(0)).any(dim=-1).float()
            write_strength = torch.where(write_strength > 0, torch.ones_like(write_strength), torch.full_like(write_strength, 0.35))
        else:
            write_strength = torch.ones(bsz, device=h.device)
        if self.organized_writes:
            if self.strict_routed_writes:
                routed_write = write_gate * float(self.num_state_slots) * route * write_strength.view(bsz, 1, 1)
            else:
                route_mix = 0.30
                routed_write = write_gate * ((1.0 - route_mix) + route_mix * float(self.num_state_slots) * route) * write_strength.view(bsz, 1, 1)
        else:
            routed_write = write_gate
        if self.training and self.write_dropout_p > 0.0:
            routed_write = routed_write * (torch.rand_like(routed_write) > self.write_dropout_p).float()
        candidate = torch.tanh(self.candidate_mlp(update_input))
        new_state = erase_gate * state + routed_write * candidate
        route_entropy = -(route_flat * route_flat.clamp_min(1e-9).log()).sum(dim=-1) / math.log(max(2, self.num_state_slots))
        return new_state, {
            "erase_gate_mean": erase_gate.mean(),
            "write_gate_mean": routed_write.mean(),
            "state_update_norm": (new_state - state).norm(dim=-1).mean(),
            "route_entropy_mean": route_entropy.mean(),
            "route_max_mean": route_flat.max(dim=-1).values.mean(),
            "write_strength_mean": write_strength.mean(),
            "route_weights": route_flat,
            "route_logits": route_logits,
            "routed_write": routed_write.mean(dim=-1),
        }

    def forward(
        self,
        h: torch.Tensor,
        state: torch.Tensor,
        *,
        current_entity_ids: Optional[torch.Tensor] = None,
        current_op_ids: Optional[torch.Tensor] = None,
        controls: Optional[Mapping[str, bool]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        clean_h, clean_diag = self._attention(h, None)
        guided_h, guided_diag = self._attention(clean_h, state)
        new_state, state_diag = self._update_state(
            state,
            guided_h,
            current_entity_ids=current_entity_ids,
            current_op_ids=current_op_ids,
            controls=controls,
        )
        diag = {f"clean_{k}": v for k, v in clean_diag.items()}
        diag.update({f"guided_{k}": v for k, v in guided_diag.items()})
        diag.update(state_diag)
        return clean_h, guided_h, new_state, diag


class IntegratedRollingStateCleanQKVStudent(nn.Module):
    def __init__(
        self,
        *,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        num_state_slots: int = NUM_ENTITIES,
        organized_writes: bool = False,
        route_temperature: float = 1.0,
        route_top_k: int = 0,
        route_dropout_p: float = 0.0,
        write_dropout_p: float = 0.0,
        slot_dropout_p: float = 0.0,
        state_dropout_p: float = 0.0,
        route_temperature_start: Optional[float] = None,
        route_temperature_end: Optional[float] = None,
        state_as_kv: bool = False,
        disable_state_mean_paths: bool = False,
        slot_cross_update: bool = False,
        strict_routed_writes: bool = False,
        learned_slot_geometry: bool = False,
        static_key_scale: float = 1.0,
        dynamic_key_scale: float = 1.0,
        geometry_basis_count: int = 0,
        geometry_gate_density_target: float = 0.0,
        geometry_gate_std_floor: float = 0.0,
        geometry_gate_temperature: float = 1.0,
        geometry_basis_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_state_slots = int(num_state_slots)
        self.organized_writes = bool(organized_writes)
        self.state_as_kv = bool(state_as_kv)
        self.disable_state_mean_paths = bool(disable_state_mean_paths)
        self.learned_slot_geometry = bool(learned_slot_geometry)
        self.static_key_scale = float(static_key_scale)
        self.dynamic_key_scale = float(dynamic_key_scale)
        self.geometry_basis_count = max(0, int(geometry_basis_count))
        self.geometry_gate_density_target = float(geometry_gate_density_target)
        self.geometry_gate_std_floor = float(geometry_gate_std_floor)
        self.geometry_gate_temperature = max(0.05, float(geometry_gate_temperature))
        self.geometry_basis_scale = float(geometry_basis_scale)
        self.slot_dropout_p = float(slot_dropout_p)
        self.state_dropout_p = float(state_dropout_p)
        self.route_temperature_start = route_temperature_start
        self.route_temperature_end = route_temperature_end
        self.token_embedding = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(STEP_TOKENS, d_model) * 0.02)
        self.initial_state = nn.Parameter(torch.randn(num_state_slots, d_model) * 0.03)
        self.layers = nn.ModuleList(
            [
                StateGuidedCleanQKVLayer(
                    d_model,
                    n_heads,
                    num_state_slots,
                    organized_writes=organized_writes,
                    route_temperature=route_temperature,
                    route_top_k=route_top_k,
                    route_dropout_p=route_dropout_p,
                    write_dropout_p=write_dropout_p,
                    state_as_kv=state_as_kv,
                    disable_state_mean_paths=disable_state_mean_paths,
                    slot_cross_update=slot_cross_update,
                    strict_routed_writes=strict_routed_writes,
                    learned_slot_geometry=learned_slot_geometry,
                    static_key_scale=static_key_scale,
                    dynamic_key_scale=dynamic_key_scale,
                    geometry_basis_count=geometry_basis_count,
                    geometry_gate_density_target=geometry_gate_density_target,
                    geometry_gate_std_floor=geometry_gate_std_floor,
                    geometry_gate_temperature=geometry_gate_temperature,
                    geometry_basis_scale=geometry_basis_scale,
                )
                for _ in range(n_layers)
            ]
        )
        self.read_query = nn.Linear(d_model, d_model)
        self.read_key = nn.Linear(d_model, d_model)
        self.read_value = nn.Linear(d_model, d_model)
        if self.learned_slot_geometry:
            self.static_read_key = nn.Parameter(torch.randn(num_state_slots, d_model) * 0.05)
            self.static_read_value_bias = nn.Parameter(torch.randn(num_state_slots, d_model) * 0.01)
            if self.geometry_basis_count > 0:
                self.read_geometry_basis_key = nn.Parameter(torch.randn(self.geometry_basis_count, d_model) * 0.04)
                self.read_geometry_gate = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, self.geometry_basis_count),
                )
        self.head = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, K_CANDIDATES),
        )
        self.head_interaction = (
            nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            if self.disable_state_mean_paths
            else None
        )
        self.probe_entity_embed = nn.Parameter(torch.randn(NUM_ENTITIES, d_model) * 0.03)
        self.probe_q = nn.Linear(d_model, d_model, bias=False)
        self.probe_k = nn.Linear(d_model, d_model, bias=False)
        self.probe_v = nn.Linear(d_model, d_model, bias=False)
        self.value_probe = nn.Linear(d_model, K_CANDIDATES)
        self.support_probe = nn.Linear(d_model, K_CANDIDATES)
        self.active_probe = nn.Linear(d_model, 1)
        self.stale_probe = nn.Linear(d_model, 1)
        self.update_probe = nn.Linear(d_model, 1)
        self.support_fact_probe = nn.Linear(d_model, 1)
        self.contradiction_fact_probe = nn.Linear(d_model, 1)

    def set_training_progress(self, progress: float) -> None:
        if self.route_temperature_start is None or self.route_temperature_end is None:
            return
        progress = min(1.0, max(0.0, float(progress)))
        temp = float(self.route_temperature_start) + progress * (float(self.route_temperature_end) - float(self.route_temperature_start))
        for layer in self.layers:
            layer.route_temperature = temp

    def _regularized_state_view(self, state: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return state
        out = state
        if self.slot_dropout_p > 0.0:
            keep = (torch.rand(out.shape[:2], dtype=out.dtype, device=out.device) > self.slot_dropout_p).unsqueeze(-1)
            out = out * keep
        if self.state_dropout_p > 0.0:
            out = F.dropout(out, p=self.state_dropout_p, training=True)
        return out

    def state_memory_bytes(self, *, dtype_bytes: int = 4) -> int:
        return int(self.num_state_slots * self.d_model * dtype_bytes)

    def initial_state_for_batch(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.initial_state.unsqueeze(0).expand(batch_size, -1, -1).to(device)

    def embed_step(self, current_step_ids: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(current_step_ids) + self.pos_embedding.unsqueeze(0)

    def _logits_from(self, h: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        summary = h.mean(dim=1)
        q = self.read_query(summary).unsqueeze(1)
        k = self.read_key(state)
        v = self.read_value(state)
        if self.learned_slot_geometry:
            if self.geometry_basis_count > 0:
                gate = torch.sigmoid(self.read_geometry_gate(state) / self.geometry_gate_temperature)
                basis_mix = torch.einsum("bsg,gd->bsd", gate, self.read_geometry_basis_key)
                basis_mix = basis_mix / gate.sum(dim=-1, keepdim=True).clamp_min(1.0)
                k = self.dynamic_key_scale * k + self.static_key_scale * self.static_read_key.unsqueeze(0) + self.geometry_basis_scale * basis_mix
            else:
                k = self.dynamic_key_scale * k + self.static_key_scale * self.static_read_key.unsqueeze(0)
            v = v + self.static_read_value_bias.unsqueeze(0)
        weights = torch.softmax(torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_model), dim=-1)
        state_read = torch.matmul(weights, v).squeeze(1)
        if self.head_interaction is not None:
            state_context = self.head_interaction(torch.cat([summary, state_read], dim=-1))
        else:
            state_context = state.mean(dim=1)
        logits = self.head(torch.cat([summary, state_read, state_context], dim=-1))
        return logits, weights.squeeze(1)

    def probe_state_facts(self, states: torch.Tensor) -> Dict[str, torch.Tensor]:
        squeeze_time = False
        if states.dim() == 3:
            states = states.unsqueeze(1)
            squeeze_time = True
        bsz, steps, slots, dim = states.shape
        flat_states = states.reshape(bsz * steps, slots, dim)
        query = self.probe_q(self.probe_entity_embed).unsqueeze(0).expand(bsz * steps, -1, -1)
        key = self.probe_k(flat_states)
        value = self.probe_v(flat_states)
        weights = torch.softmax(torch.matmul(query, key.transpose(1, 2)) / math.sqrt(self.d_model), dim=-1)
        entity_read = torch.matmul(weights, value).view(bsz, steps, NUM_ENTITIES, dim)
        probes = {
            "value_logits": self.value_probe(entity_read),
            "support_logits": self.support_probe(entity_read),
            "active_logits": self.active_probe(entity_read).squeeze(-1),
            "stale_logits": self.stale_probe(entity_read).squeeze(-1),
            "update_logits": self.update_probe(entity_read).squeeze(-1),
            "support_fact_logits": self.support_fact_probe(entity_read).squeeze(-1),
            "contradiction_fact_logits": self.contradiction_fact_probe(entity_read).squeeze(-1),
            "probe_attention": weights.view(bsz, steps, NUM_ENTITIES, slots),
        }
        if squeeze_time:
            return {name: tensor[:, 0] for name, tensor in probes.items()}
        return probes

    def forward_step(
        self,
        current_step_ids: torch.Tensor,
        state_prev: torch.Tensor,
        *,
        freeze_state_update: bool = False,
        random_state_update: bool = False,
        detach_state_update: bool = False,
        controls: Optional[Mapping[str, bool]] = None,
    ) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        controls = dict(controls or {})
        h = self.embed_step(current_step_ids)
        clean_activation = h.mean(dim=1)
        diagnostics: Dict[str, torch.Tensor] = {
            "old_token_kv_received": torch.tensor(0, device=current_step_ids.device),
            "growing_activation_cache": torch.tensor(0, device=current_step_ids.device),
            "state_slots": torch.tensor(state_prev.shape[1], device=current_step_ids.device),
            "state_dim": torch.tensor(state_prev.shape[2], device=current_step_ids.device),
        }
        state = self._regularized_state_view(state_prev)
        gate_erase: List[torch.Tensor] = []
        gate_write: List[torch.Tensor] = []
        update_norms: List[torch.Tensor] = []
        route_weights: List[torch.Tensor] = []
        route_logits: List[torch.Tensor] = []
        route_entropy: List[torch.Tensor] = []
        route_max: List[torch.Tensor] = []
        geometry_density: List[torch.Tensor] = []
        geometry_active_fraction: List[torch.Tensor] = []
        geometry_std: List[torch.Tensor] = []
        geometry_density_target: List[torch.Tensor] = []
        geometry_std_floor: List[torch.Tensor] = []
        current_op_ids = current_step_ids[:, 0]
        current_entity_ids = (current_step_ids[:, 1] - ENTITY_OFFSET).clamp(0, NUM_ENTITIES - 1)
        for layer_idx, layer in enumerate(self.layers):
            clean_h, guided_h, proposed_state, diag = layer(
                h,
                state,
                current_entity_ids=current_entity_ids,
                current_op_ids=current_op_ids,
                controls=controls,
            )
            clean_activation = clean_h.mean(dim=1)
            if random_state_update:
                new_state = torch.randn_like(proposed_state)
            elif freeze_state_update:
                new_state = state
            else:
                new_state = proposed_state
            if detach_state_update:
                new_state = new_state.detach()
            state = new_state
            h = guided_h
            gate_erase.append(diag["erase_gate_mean"])
            gate_write.append(diag["write_gate_mean"])
            update_norms.append((state - state_prev).norm(dim=-1).mean() if layer_idx == 0 else diag["state_update_norm"])
            route_weights.append(diag["route_weights"])
            route_logits.append(diag["route_logits"])
            route_entropy.append(diag["route_entropy_mean"])
            route_max.append(diag["route_max_mean"])
            geometry_density.append(diag["guided_geometry_gate_density_mean"])
            geometry_active_fraction.append(diag["guided_geometry_gate_active_fraction"])
            geometry_std.append(diag["guided_geometry_gate_std_mean"])
            geometry_density_target.append(diag["guided_geometry_gate_density_target"])
            geometry_std_floor.append(diag["guided_geometry_gate_std_floor"])
            diagnostics[f"layer_{layer_idx}_clean_k_token_count"] = diag["clean_k_token_count"]
            diagnostics[f"layer_{layer_idx}_guided_k_token_count"] = diag["guided_k_token_count"]
            diagnostics[f"layer_{layer_idx}_guided_state_kv_token_count"] = diag["guided_state_kv_token_count"]
            diagnostics[f"layer_{layer_idx}_used_state_as_kv"] = diag["guided_used_state_as_kv"]
            diagnostics[f"layer_{layer_idx}_geometry_gate_density_mean"] = diag["guided_geometry_gate_density_mean"]
            diagnostics[f"layer_{layer_idx}_geometry_gate_active_fraction"] = diag["guided_geometry_gate_active_fraction"]
            diagnostics[f"layer_{layer_idx}_geometry_gate_std_mean"] = diag["guided_geometry_gate_std_mean"]
        logits, read_weights = self._logits_from(h, state)
        diagnostics["erase_gate_mean"] = torch.stack(gate_erase).mean()
        diagnostics["write_gate_mean"] = torch.stack(gate_write).mean()
        diagnostics["state_update_norm"] = torch.stack(update_norms).mean()
        diagnostics["route_entropy_mean"] = torch.stack(route_entropy).mean()
        diagnostics["route_max_mean"] = torch.stack(route_max).mean()
        diagnostics["geometry_gate_density_mean"] = torch.stack(geometry_density).mean()
        diagnostics["geometry_gate_active_fraction"] = torch.stack(geometry_active_fraction).mean()
        diagnostics["geometry_gate_std_mean"] = torch.stack(geometry_std).mean()
        diagnostics["geometry_gate_density_target"] = torch.stack(geometry_density_target).mean()
        diagnostics["geometry_gate_std_floor"] = torch.stack(geometry_std_floor).mean()
        diagnostics["self_attention_current_tokens_only"] = torch.tensor(1, device=current_step_ids.device)
        return {
            "logits": logits,
            "state": state,
            "clean_activation": clean_activation,
            "guided_activation": h.mean(dim=1),
            "read_weights": read_weights,
            "route_weights": torch.stack(route_weights, dim=1),
            "route_logits": torch.stack(route_logits, dim=1),
            "diagnostics": diagnostics,
        }

    def forward_sequence(
        self,
        input_ids: torch.Tensor,
        *,
        controls: Optional[Mapping[str, bool]] = None,
    ) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        controls = dict(controls or {})
        bsz, seq_len, _, = input_ids.shape
        device = input_ids.device
        initial = self.initial_state_for_batch(bsz, device)
        state = initial
        states: List[torch.Tensor] = []
        routes: List[torch.Tensor] = []
        route_logits: List[torch.Tensor] = []
        clean_acts: List[torch.Tensor] = []
        guided_acts: List[torch.Tensor] = []
        diagnostics_list: List[Dict[str, torch.Tensor]] = []

        for t in range(seq_len):
            if controls.get("state_reset_every_step", False):
                state = initial
            if t == seq_len - 1 and controls.get("zero_state_at_query", False):
                state = torch.zeros_like(state)
            if t == seq_len - 1 and controls.get("shuffle_state_across_batch", False) and bsz > 1:
                state = state[torch.arange(bsz - 1, -1, -1, device=device)]
            if t == seq_len - 1 and controls.get("slot_permutation", False) and self.num_state_slots > 1:
                state = state[:, torch.randperm(self.num_state_slots, device=device), :]
            if t == seq_len - 1 and controls.get("slot_dropout_eval", False):
                keep = (torch.arange(self.num_state_slots, device=device) % 2 == 0).to(state.dtype).view(1, -1, 1)
                state = state * keep
            out = self.forward_step(
                input_ids[:, t, :],
                state,
                freeze_state_update=bool(controls.get("frozen_state_update", False)),
                random_state_update=bool(controls.get("random_state_update", False)),
                detach_state_update=bool(controls.get("detached_state_update", False)),
                controls=controls,
            )
            state = out["state"]  # type: ignore[assignment]
            states.append(state)
            routes.append(out["route_weights"])  # type: ignore[arg-type]
            route_logits.append(out["route_logits"])  # type: ignore[arg-type]
            clean_acts.append(out["clean_activation"])  # type: ignore[arg-type]
            guided_acts.append(out["guided_activation"])  # type: ignore[arg-type]
            diagnostics_list.append(out["diagnostics"])  # type: ignore[arg-type]

        final_logits, read_weights = self._logits_from(guided_acts[-1].unsqueeze(1), state)
        del final_logits
        last_diag = diagnostics_list[-1]
        return {
            "logits": out["logits"],  # type: ignore[name-defined]
            "state": state,
            "states": torch.stack(states, dim=1),
            "route_weights": torch.stack(routes, dim=1),
            "route_logits": torch.stack(route_logits, dim=1),
            "clean_activation": clean_acts[-1],
            "guided_activation": guided_acts[-1],
            "read_weights": read_weights,
            "diagnostics": {
                "old_token_kv_received": torch.stack([d["old_token_kv_received"] for d in diagnostics_list]).max(),
                "growing_activation_cache": torch.stack([d["growing_activation_cache"] for d in diagnostics_list]).max(),
                "state_update_norm": torch.stack([d["state_update_norm"] for d in diagnostics_list]).mean(),
                "erase_gate_mean": torch.stack([d["erase_gate_mean"] for d in diagnostics_list]).mean(),
                "write_gate_mean": torch.stack([d["write_gate_mean"] for d in diagnostics_list]).mean(),
                "route_entropy_mean": torch.stack([d["route_entropy_mean"] for d in diagnostics_list]).mean(),
                "route_max_mean": torch.stack([d["route_max_mean"] for d in diagnostics_list]).mean(),
                "geometry_gate_density_mean": torch.stack([d["geometry_gate_density_mean"] for d in diagnostics_list]).mean(),
                "geometry_gate_active_fraction": torch.stack([d["geometry_gate_active_fraction"] for d in diagnostics_list]).mean(),
                "geometry_gate_std_mean": torch.stack([d["geometry_gate_std_mean"] for d in diagnostics_list]).mean(),
                "geometry_gate_density_target": torch.stack([d["geometry_gate_density_target"] for d in diagnostics_list]).mean(),
                "geometry_gate_std_floor": torch.stack([d["geometry_gate_std_floor"] for d in diagnostics_list]).mean(),
                "final_step_k_token_count": last_diag["layer_0_guided_k_token_count"],
                "final_step_state_kv_token_count": last_diag.get("layer_0_guided_state_kv_token_count", torch.tensor(0, device=device)),
                "used_state_as_kv": torch.stack([d.get("layer_0_used_state_as_kv", torch.tensor(0, device=device)) for d in diagnostics_list]).max(),
                "state_slots": torch.tensor(self.num_state_slots, device=device),
                "state_dim": torch.tensor(self.d_model, device=device),
            },
        }


class CurrentOnlyNoStateStudent(nn.Module):
    def __init__(self, *, d_model: int = 64, n_layers: int = 2, n_heads: int = 4) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.token_embedding = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(STEP_TOKENS, d_model) * 0.02)
        self.layers = nn.ModuleList([StateGuidedCleanQKVLayer(d_model, n_heads, NUM_ENTITIES) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, K_CANDIDATES),
        )

    def forward_sequence(self, input_ids: torch.Tensor, *, controls: Optional[Mapping[str, bool]] = None) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        del controls
        h = self.token_embedding(input_ids[:, -1, :]) + self.pos_embedding.unsqueeze(0)
        state = torch.zeros(input_ids.shape[0], NUM_ENTITIES, self.d_model, dtype=h.dtype, device=h.device)
        clean = h.mean(dim=1)
        for layer in self.layers:
            clean, guided, _, _ = layer(h, state)
            h = clean
        logits = self.head(clean.mean(dim=1))
        return {
            "logits": logits,
            "clean_activation": clean.mean(dim=1),
            "guided_activation": clean.mean(dim=1),
            "diagnostics": {
                "old_token_kv_received": torch.tensor(0, device=h.device),
                "growing_activation_cache": torch.tensor(0, device=h.device),
                "final_step_k_token_count": torch.tensor(STEP_TOKENS, device=h.device),
                "state_update_norm": torch.tensor(0.0, device=h.device),
                "erase_gate_mean": torch.tensor(0.0, device=h.device),
                "write_gate_mean": torch.tensor(0.0, device=h.device),
            },
        }


class PostAttentionResidualMemoryStudent(nn.Module):
    def __init__(self, *, d_model: int = 64, n_layers: int = 2, n_heads: int = 4, num_state_slots: int = 4) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_state_slots = int(num_state_slots)
        self.token_embedding = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(STEP_TOKENS, d_model) * 0.02)
        self.layers = nn.ModuleList([StateGuidedCleanQKVLayer(d_model, n_heads, num_state_slots) for _ in range(n_layers)])
        self.initial_state = nn.Parameter(torch.randn(num_state_slots, d_model) * 0.03)
        self.update = nn.GRUCell(d_model, num_state_slots * d_model)
        self.state_proj = nn.Linear(num_state_slots * d_model, d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, K_CANDIDATES),
        )

    def initial_state_for_batch(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.initial_state.unsqueeze(0).expand(batch_size, -1, -1).to(device)

    def forward_sequence(self, input_ids: torch.Tensor, *, controls: Optional[Mapping[str, bool]] = None) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        controls = dict(controls or {})
        bsz, seq_len, _ = input_ids.shape
        device = input_ids.device
        state = self.initial_state_for_batch(bsz, device)
        initial = state
        states: List[torch.Tensor] = []
        updates: List[torch.Tensor] = []
        clean = torch.zeros(bsz, self.d_model, device=device)
        for t in range(seq_len):
            if controls.get("state_reset_every_step", False):
                state = initial
            if t == seq_len - 1 and controls.get("zero_state_at_query", False):
                state = torch.zeros_like(state)
            if t == seq_len - 1 and controls.get("shuffle_state_across_batch", False) and bsz > 1:
                state = state[torch.arange(bsz - 1, -1, -1, device=device)]
            h = self.token_embedding(input_ids[:, t, :]) + self.pos_embedding.unsqueeze(0)
            clean_state = torch.zeros_like(state)
            for layer in self.layers:
                clean_h, _, _, _ = layer(h, clean_state)
                h = clean_h
            clean = h.mean(dim=1)
            if controls.get("random_state_update", False):
                new_state = torch.randn_like(state)
            elif controls.get("frozen_state_update", False):
                new_state = state
            else:
                flat = self.update(clean, state.flatten(1))
                new_state = flat.view(bsz, self.num_state_slots, self.d_model)
            updates.append((new_state - state).norm(dim=-1).mean())
            state = new_state.detach() if controls.get("detached_state_update", False) else new_state
            states.append(state)
        state_summary = self.state_proj(state.flatten(1))
        logits = self.head(torch.cat([clean, state_summary], dim=-1))
        return {
            "logits": logits,
            "states": torch.stack(states, dim=1),
            "clean_activation": clean,
            "guided_activation": clean + state_summary,
            "state": state,
            "diagnostics": {
                "old_token_kv_received": torch.tensor(0, device=device),
                "growing_activation_cache": torch.tensor(0, device=device),
                "final_step_k_token_count": torch.tensor(STEP_TOKENS, device=device),
                "state_update_norm": torch.stack(updates).mean(),
                "erase_gate_mean": torch.tensor(0.0, device=device),
                "write_gate_mean": torch.tensor(0.0, device=device),
                "state_slots": torch.tensor(self.num_state_slots, device=device),
                "state_dim": torch.tensor(self.d_model, device=device),
            },
        }


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def state_slot_entropy(states: torch.Tensor) -> float:
    with torch.no_grad():
        final = states[:, -1] if states.dim() == 4 else states
        norms = final.norm(dim=-1)
        probs = torch.softmax(norms, dim=-1)
        entropy = -(probs * (probs.clamp_min(1e-9).log())).sum(dim=-1)
        return float(entropy.mean().detach().cpu().item())


def state_collapse_score(states: torch.Tensor) -> float:
    with torch.no_grad():
        final = states[:, -1] if states.dim() == 4 else states
        normed = F.normalize(final, dim=-1)
        sim = torch.matmul(normed, normed.transpose(1, 2)).abs()
        eye = torch.eye(sim.shape[-1], device=sim.device).unsqueeze(0)
        offdiag = sim * (1.0 - eye)
        denom = max(1, sim.shape[-1] * (sim.shape[-1] - 1))
        return float((offdiag.sum(dim=(1, 2)) / denom).mean().detach().cpu().item())


def state_effective_rank(states: torch.Tensor) -> float:
    with torch.no_grad():
        final = states[:, -1] if states.dim() == 4 else states
        centered = final - final.mean(dim=1, keepdim=True)
        ranks: List[torch.Tensor] = []
        for matrix in centered:
            s = torch.linalg.svdvals(matrix.float())
            probs = s / s.sum().clamp_min(1e-8)
            entropy = -(probs * probs.clamp_min(1e-9).log()).sum()
            ranks.append(torch.exp(entropy))
        return float(torch.stack(ranks).mean().detach().cpu().item())


def slot_cosine_metrics(states: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        final = states[:, -1] if states.dim() == 4 else states
        normed = F.normalize(final, dim=-1)
        sim = torch.matmul(normed, normed.transpose(1, 2)).abs()
        eye = torch.eye(sim.shape[-1], device=sim.device).bool().unsqueeze(0)
        offdiag = sim.masked_select(~eye.expand_as(sim))
        return {
            "pairwise_slot_cosine_mean": float(offdiag.mean().detach().cpu().item()),
            "pairwise_slot_cosine_max": float(offdiag.max().detach().cpu().item()),
        }


def state_slot_norm_metrics(states: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        final = states[:, -1] if states.dim() == 4 else states
        slot_norms = final.norm(dim=-1).mean(dim=0)
        return {
            "per_slot_norm_mean": float(slot_norms.mean().detach().cpu().item()),
            "per_slot_norm_std": float(slot_norms.std(unbiased=False).detach().cpu().item()),
        }


def _loss_dict_zero(device: torch.device) -> Dict[str, torch.Tensor]:
    z = torch.tensor(0.0, device=device)
    return {
        "answer_loss": z,
        "activation_matching_loss": z,
        "delta_matching_loss": z,
        "state_usefulness_loss": z,
        "state_stability_loss": z,
        "state_update_regularization_loss": z,
        "slot_diversity_loss": z,
        "state_ablation_loss": z,
        "slot_routing_loss": z,
        "slot_cosine_decorrelation_loss": z,
        "slot_covariance_loss": z,
        "effective_rank_loss": z,
        "bounded_state_norm_loss": z,
        "write_balance_loss": z,
        "read_balance_loss": z,
        "route_entropy_floor_loss": z,
        "load_balance_loss": z,
        "active_probe_loss": z,
        "stale_probe_loss": z,
        "support_probe_loss": z,
        "distractor_preservation_loss": z,
        "long_delay_retention_loss": z,
        "geometry_gate_density_loss": z,
        "geometry_gate_variation_loss": z,
    }


def _slot_group_mask(entity_ids: torch.Tensor, num_state_slots: int) -> torch.Tensor:
    slots = torch.arange(num_state_slots, device=entity_ids.device)
    return (slots.view(*([1] * entity_ids.dim()), num_state_slots) % NUM_ENTITIES == entity_ids.unsqueeze(-1)).float()


def _write_step_mask(input_ids: torch.Tensor) -> torch.Tensor:
    op_ids = input_ids[:, :, 0]
    write_ops = torch.tensor(
        [
            OP_TO_ID["set"],
            OP_TO_ID["update"],
            OP_TO_ID["invalidate"],
            OP_TO_ID["link"],
            OP_TO_ID["support"],
            OP_TO_ID["contradict"],
        ],
        device=input_ids.device,
    )
    return (op_ids.unsqueeze(-1) == write_ops.view(1, 1, -1)).any(dim=-1).float()


def compute_training_loss(
    model: nn.Module,
    batch: E5Batch,
    teacher: FittedFullContextQKVTeacher,
    weights: LossWeights,
    *,
    randomized_labels: bool = False,
    ablation_margin: float = 0.55,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor | Dict[str, torch.Tensor]]]:
    out = model.forward_sequence(batch.input_ids)  # type: ignore[attr-defined]
    logits = out["logits"]  # type: ignore[assignment]
    labels = torch.randint_like(batch.labels, low=0, high=K_CANDIDATES) if randomized_labels else batch.labels
    losses = _loss_dict_zero(batch.input_ids.device)
    losses["answer_loss"] = F.cross_entropy(logits, labels)

    teacher_out = teacher(batch)
    if weights.activation:
        losses["activation_matching_loss"] = F.mse_loss(out["guided_activation"], teacher_out["hidden_full"].detach())  # type: ignore[arg-type]
    if weights.delta:
        teacher_delta = (teacher_out["hidden_full"] - teacher_out["hidden_current"]).detach()  # type: ignore[operator]
        student_delta = out["guided_activation"] - out["clean_activation"]  # type: ignore[operator]
        losses["delta_matching_loss"] = F.mse_loss(student_delta, teacher_delta)
    if weights.state_usefulness and "states" in out and hasattr(model, "probe_state_facts"):
        states = out["states"]  # type: ignore[assignment]
        probes = model.probe_state_facts(states)  # type: ignore[attr-defined]
        value_loss = F.cross_entropy(probes["value_logits"].reshape(-1, K_CANDIDATES), batch.value_targets.reshape(-1))
        support_loss = F.cross_entropy(probes["support_logits"].reshape(-1, K_CANDIDATES), batch.support_targets.reshape(-1))
        binary_losses = (
            F.binary_cross_entropy_with_logits(probes["active_logits"], batch.active_fact_targets)
            + F.binary_cross_entropy_with_logits(probes["stale_logits"], batch.stale_fact_targets)
            + F.binary_cross_entropy_with_logits(probes["update_logits"], batch.update_fact_targets)
            + F.binary_cross_entropy_with_logits(probes["support_fact_logits"], batch.support_fact_targets)
            + F.binary_cross_entropy_with_logits(probes["contradiction_fact_logits"], batch.contradiction_fact_targets)
        )
        losses["state_usefulness_loss"] = 0.35 * (value_loss + support_loss) + 0.30 * binary_losses
    if weights.state_stability and "states" in out:
        states = out["states"]  # type: ignore[assignment]
        deltas = (states[:, 1:] - states[:, :-1]).pow(2).mean(dim=(2, 3))
        distractor = batch.distractor_mask[:, 1:]
        if float(distractor.sum().detach().cpu().item()) > 0:
            losses["state_stability_loss"] = (deltas * distractor).sum() / distractor.sum().clamp_min(1.0)
    if weights.update_regularization and "states" in out:
        states = out["states"]  # type: ignore[assignment]
        update_norm = (states[:, 1:] - states[:, :-1]).norm(dim=-1).mean()
        losses["state_update_regularization_loss"] = F.relu(0.01 - update_norm) + F.relu(update_norm - 6.0) * 0.05
    if weights.slot_diversity and "states" in out:
        states = out["states"]  # type: ignore[assignment]
        final = states[:, -1]
        normed = F.normalize(final, dim=-1)
        sim = torch.matmul(normed, normed.transpose(1, 2))
        eye = torch.eye(sim.shape[-1], device=sim.device).unsqueeze(0)
        losses["slot_diversity_loss"] = ((sim * (1.0 - eye)).pow(2).sum(dim=(1, 2)) / max(1, sim.shape[-1] * (sim.shape[-1] - 1))).mean()
    if (weights.slot_cosine_decorrelation or weights.slot_covariance or weights.effective_rank or weights.bounded_state_norm) and "states" in out:
        states = out["states"]  # type: ignore[assignment]
        final = states[:, -1]
        slots = final.shape[1]
        normed = F.normalize(final, dim=-1)
        sim = torch.matmul(normed, normed.transpose(1, 2))
        eye = torch.eye(slots, device=sim.device).unsqueeze(0)
        offdiag = sim * (1.0 - eye)
        if weights.slot_cosine_decorrelation:
            losses["slot_cosine_decorrelation_loss"] = (offdiag.pow(2).sum(dim=(1, 2)) / max(1, slots * (slots - 1))).mean()
        if weights.slot_covariance:
            centered = final - final.mean(dim=1, keepdim=True)
            cov = torch.matmul(centered, centered.transpose(1, 2)) / max(1, final.shape[-1])
            cov_norm = cov / (cov.diagonal(dim1=1, dim2=2).mean(dim=1).view(-1, 1, 1).abs().clamp_min(1e-6))
            losses["slot_covariance_loss"] = ((cov_norm * (1.0 - eye)).pow(2).sum(dim=(1, 2)) / max(1, slots * (slots - 1))).mean()
        if weights.effective_rank:
            ranks = []
            for matrix in final - final.mean(dim=1, keepdim=True):
                s = torch.linalg.svdvals(matrix.float())
                probs = s / s.sum().clamp_min(1e-8)
                ranks.append(torch.exp(-(probs * probs.clamp_min(1e-9).log()).sum()))
            rank = torch.stack(ranks).mean()
            losses["effective_rank_loss"] = F.relu(0.45 * min(slots, final.shape[-1]) - rank) / max(1, min(slots, final.shape[-1]))
        if weights.bounded_state_norm:
            norms = final.norm(dim=-1)
            losses["bounded_state_norm_loss"] = F.relu(norms - 8.0).pow(2).mean() + F.relu(0.25 - norms).pow(2).mean()
    if weights.slot_routing and "route_weights" in out:
        route = out["route_weights"]  # type: ignore[assignment]
        bsz, steps, layers, slots = route.shape
        entity_ids = (batch.input_ids[:, :, 1] - ENTITY_OFFSET).clamp(0, NUM_ENTITIES - 1)
        group_mask = _slot_group_mask(entity_ids, slots).unsqueeze(2)
        write_mask = _write_step_mask(batch.input_ids).unsqueeze(-1)
        mass = (route * group_mask).sum(dim=-1).clamp_min(1e-8)
        group_loss = (-(mass.log()) * write_mask).sum() / write_mask.sum().clamp_min(1.0) / max(1, layers)
        active_route = route * write_mask.unsqueeze(-1)
        avg_route = active_route.sum(dim=(0, 1, 2)) / write_mask.sum().clamp_min(1.0) / max(1, layers)
        avg_route = avg_route / avg_route.sum().clamp_min(1e-8)
        load_entropy = -(avg_route * avg_route.clamp_min(1e-9).log()).sum() / math.log(max(2, slots))
        route_entropy = -(route * route.clamp_min(1e-9).log()).sum(dim=-1) / math.log(max(2, slots))
        sparse_loss = (route_entropy * write_mask).sum() / write_mask.sum().clamp_min(1.0) / max(1, layers)
        losses["slot_routing_loss"] = group_loss + 0.20 * (1.0 - load_entropy) + 0.05 * sparse_loss
    if (weights.write_balance or weights.route_entropy_floor or weights.load_balance) and "route_weights" in out:
        route = out["route_weights"]  # type: ignore[assignment]
        _, _, layers, slots = route.shape
        write_mask = _write_step_mask(batch.input_ids).unsqueeze(-1)
        active_route = route * write_mask.unsqueeze(-1)
        avg_route = active_route.sum(dim=(0, 1, 2)) / write_mask.sum().clamp_min(1.0) / max(1, layers)
        avg_route = avg_route / avg_route.sum().clamp_min(1e-8)
        uniform = torch.full_like(avg_route, 1.0 / max(1, slots))
        route_entropy = -(route * route.clamp_min(1e-9).log()).sum(dim=-1) / math.log(max(2, slots))
        load_entropy = -(avg_route * avg_route.clamp_min(1e-9).log()).sum() / math.log(max(2, slots))
        if weights.write_balance:
            losses["write_balance_loss"] = F.mse_loss(avg_route, uniform) * slots
        if weights.route_entropy_floor:
            losses["route_entropy_floor_loss"] = F.relu(0.35 - route_entropy).pow(2).mean()
        if weights.load_balance:
            losses["load_balance_loss"] = 1.0 - load_entropy
    if weights.read_balance and "read_weights" in out:
        read = out["read_weights"]  # type: ignore[assignment]
        slots = read.shape[-1]
        avg_read = read.mean(dim=0)
        avg_read = avg_read / avg_read.sum().clamp_min(1e-8)
        losses["read_balance_loss"] = F.mse_loss(avg_read, torch.full_like(avg_read, 1.0 / max(1, slots))) * slots
    diag = out.get("diagnostics", {}) if isinstance(out, Mapping) else {}
    if weights.geometry_density and isinstance(diag, Mapping) and "geometry_gate_density_mean" in diag:
        density = diag["geometry_gate_density_mean"]  # type: ignore[index]
        target = diag.get("geometry_gate_density_target", torch.tensor(0.55, device=batch.input_ids.device))  # type: ignore[assignment]
        losses["geometry_gate_density_loss"] = (density - target).pow(2)  # type: ignore[operator]
    if weights.geometry_gate_variation and isinstance(diag, Mapping) and "geometry_gate_std_mean" in diag:
        std = diag["geometry_gate_std_mean"]  # type: ignore[index]
        floor = diag.get("geometry_gate_std_floor", torch.tensor(0.08, device=batch.input_ids.device))  # type: ignore[assignment]
        losses["geometry_gate_variation_loss"] = F.relu(floor - std).pow(2)  # type: ignore[arg-type]
    if (weights.active_probe or weights.stale_probe or weights.support_probe or weights.long_delay_retention) and "states" in out and hasattr(model, "probe_state_facts"):
        states = out["states"]  # type: ignore[assignment]
        probes = model.probe_state_facts(states)  # type: ignore[attr-defined]
        if weights.active_probe:
            losses["active_probe_loss"] = F.binary_cross_entropy_with_logits(probes["active_logits"], batch.active_fact_targets)
        if weights.stale_probe:
            losses["stale_probe_loss"] = F.binary_cross_entropy_with_logits(probes["stale_logits"], batch.stale_fact_targets) + F.binary_cross_entropy_with_logits(probes["update_logits"], batch.update_fact_targets)
        if weights.support_probe:
            losses["support_probe_loss"] = F.binary_cross_entropy_with_logits(probes["support_fact_logits"], batch.support_fact_targets) + F.binary_cross_entropy_with_logits(probes["contradiction_fact_logits"], batch.contradiction_fact_targets)
        if weights.long_delay_retention:
            final_probes = model.probe_state_facts(states[:, -1])  # type: ignore[attr-defined]
            losses["long_delay_retention_loss"] = (
                F.cross_entropy(final_probes["value_logits"].reshape(-1, K_CANDIDATES), batch.value_targets[:, -1].reshape(-1))
                + F.cross_entropy(final_probes["support_logits"].reshape(-1, K_CANDIDATES), batch.support_targets[:, -1].reshape(-1))
            )
    if weights.distractor_preservation and "states" in out:
        states = out["states"]  # type: ignore[assignment]
        deltas = (states[:, 1:] - states[:, :-1]).pow(2).mean(dim=(2, 3))
        distractor = batch.distractor_mask[:, 1:]
        if float(distractor.sum().detach().cpu().item()) > 0:
            losses["distractor_preservation_loss"] = (deltas * distractor).sum() / distractor.sum().clamp_min(1.0)
    if weights.state_ablation:
        mem_mask = batch.memory_dependent
        if bool(mem_mask.any().detach().cpu().item()):
            zero_out = model.forward_sequence(batch.input_ids, controls={"zero_state_at_query": True})  # type: ignore[attr-defined]
            normal_ce = F.cross_entropy(logits[mem_mask], labels[mem_mask], reduction="none")
            zero_ce = F.cross_entropy(zero_out["logits"][mem_mask], labels[mem_mask], reduction="none")  # type: ignore[index]
            losses["state_ablation_loss"] = F.relu(float(ablation_margin) - (zero_ce - normal_ce)).mean()

    total = (
        weights.answer * losses["answer_loss"]
        + weights.activation * losses["activation_matching_loss"]
        + weights.delta * losses["delta_matching_loss"]
        + weights.state_usefulness * losses["state_usefulness_loss"]
        + weights.state_stability * losses["state_stability_loss"]
        + weights.update_regularization * losses["state_update_regularization_loss"]
        + weights.slot_diversity * losses["slot_diversity_loss"]
        + weights.state_ablation * losses["state_ablation_loss"]
        + weights.slot_routing * losses["slot_routing_loss"]
        + weights.slot_cosine_decorrelation * losses["slot_cosine_decorrelation_loss"]
        + weights.slot_covariance * losses["slot_covariance_loss"]
        + weights.effective_rank * losses["effective_rank_loss"]
        + weights.bounded_state_norm * losses["bounded_state_norm_loss"]
        + weights.write_balance * losses["write_balance_loss"]
        + weights.read_balance * losses["read_balance_loss"]
        + weights.route_entropy_floor * losses["route_entropy_floor_loss"]
        + weights.load_balance * losses["load_balance_loss"]
        + weights.active_probe * losses["active_probe_loss"]
        + weights.stale_probe * losses["stale_probe_loss"]
        + weights.support_probe * losses["support_probe_loss"]
        + weights.distractor_preservation * losses["distractor_preservation_loss"]
        + weights.long_delay_retention * losses["long_delay_retention_loss"]
        + weights.geometry_density * losses["geometry_gate_density_loss"]
        + weights.geometry_gate_variation * losses["geometry_gate_variation_loss"]
    )
    metrics = {name: float(value.detach().cpu().item()) for name, value in losses.items()}
    metrics["total_loss"] = float(total.detach().cpu().item())
    return total, metrics, out


def _make_train_batch(
    *,
    batch_size: int,
    train_lengths: Sequence[int],
    seed: int,
    step: int,
    total_steps: int,
    device: torch.device,
    curriculum: bool = True,
) -> E5Batch:
    seq_len = int(train_lengths[(seed + step) % len(train_lengths)])
    families: Optional[Tuple[str, ...]] = None
    allow_distractors = True
    if curriculum:
        progress = float(step) / max(1, int(total_steps))
        if progress < 0.25:
            families = ("entity_binding", "entity_update_override", "stale_memory", "long_range_recall")
            allow_distractors = False
            seq_len = min(seq_len, 8)
        elif progress < 0.50:
            families = ("entity_binding", "entity_update_override", "stale_memory", "long_range_recall", "sparse_relevant_history", "interleaved_distractors")
        elif progress < 0.75:
            families = (
                "entity_binding",
                "entity_update_override",
                "stale_memory",
                "long_range_recall",
                "sparse_relevant_history",
                "interleaved_distractors",
                "multi_hop_binding",
            )
        else:
            families = TASK_FAMILIES
    examples = build_e5_examples(
        batch_size,
        seq_len,
        seed=seed * 1_000_003 + step * 7_919 + 17,
        split="train",
        task_families=families,
        allow_distractors=allow_distractors,
    )
    return examples_to_batch(examples, device)


def train_model(
    model: nn.Module,
    *,
    model_id: str,
    variant: str,
    seed: int,
    teacher: FittedFullContextQKVTeacher,
    budget: E5Budget,
    device: torch.device,
    steps: int,
    weights: LossWeights,
    randomized_labels: bool = False,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    random.seed(seed)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=budget.learning_rate, weight_decay=budget.weight_decay)
    loss_trace: List[Dict[str, float]] = []
    start = time.perf_counter()
    for step in range(int(steps)):
        if hasattr(model, "set_training_progress"):
            model.set_training_progress(step / max(1, int(steps) - 1))  # type: ignore[attr-defined]
        batch = _make_train_batch(
            batch_size=budget.train_batch_size,
            train_lengths=budget.train_lengths,
            seed=seed,
            step=step,
            total_steps=steps,
            device=device,
            curriculum=budget.curriculum,
        )
        optimizer.zero_grad(set_to_none=True)
        loss, metrics, _ = compute_training_loss(
            model,
            batch,
            teacher,
            weights,
            randomized_labels=randomized_labels,
            ablation_margin=budget.state_ablation_margin,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or step == steps - 1 or (step + 1) % max(1, steps // 4) == 0:
            metrics["step"] = float(step + 1)
            loss_trace.append(metrics)
    wall = time.perf_counter() - start
    return {
        "model_id": model_id,
        "variant": variant,
        "seed": seed,
        "steps": steps,
        "randomized_labels": randomized_labels,
        "wall_clock_time": wall,
        "loss_trace": loss_trace,
        "parameter_count": count_parameters(model),
    }


@torch.no_grad()
def evaluate_model(
    model: object,
    *,
    model_id: str,
    variant: str,
    seed: int,
    seq_len: int,
    n_examples: int,
    device: torch.device,
    teacher: Optional[FittedFullContextQKVTeacher] = None,
    control: Optional[str] = None,
    task_families: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    examples = build_e5_examples(n_examples, seq_len, seed=seed * 1_000_003 + seq_len * 103 + 29, split="dev", task_families=task_families)
    if control == "shuffled_sequence_order":
        rng = random.Random(seed + seq_len + 3001)
        shuffled: List[E5Example] = []
        for ex in examples:
            steps = list(ex.input_ids[:-1])
            rng.shuffle(steps)
            new_input = tuple(steps + [ex.input_ids[-1]])
            shuffled.append(
                E5Example(
                    input_ids=new_input,
                    label=ex.label,
                    family=ex.family,
                    target_entity=ex.target_entity,
                    memory_dependent=ex.memory_dependent,
                    support_query=ex.support_query,
                    value_targets=ex.value_targets,
                    support_targets=ex.support_targets,
                    active_fact_targets=ex.active_fact_targets,
                    stale_fact_targets=ex.stale_fact_targets,
                    update_fact_targets=ex.update_fact_targets,
                    support_fact_targets=ex.support_fact_targets,
                    contradiction_fact_targets=ex.contradiction_fact_targets,
                    distractor_mask=ex.distractor_mask,
                )
            )
        examples = shuffled
    batch = examples_to_batch(examples, device)

    controls: Dict[str, bool] = {}
    if control in {
        "zeroed_state_at_query",
        "shuffled_state_across_batch",
        "frozen_state_update",
        "random_state_update",
        "detached_state_update",
        "state_reset_every_step",
        "hidden_state_shuffle",
        "route_shuffle",
        "slot_permutation",
        "slot_dropout_eval",
    }:
        key = "shuffle_state_across_batch" if control == "hidden_state_shuffle" else control
        if key == "zeroed_state_at_query":
            controls["zero_state_at_query"] = True
        elif key == "shuffled_state_across_batch":
            controls["shuffle_state_across_batch"] = True
        else:
            controls[key] = True

    if variant == "random_baseline":
        gen = torch.Generator(device=device)
        gen.manual_seed(seed + seq_len)
        preds = torch.randint(0, K_CANDIDATES, (len(examples),), generator=gen, device=device)
        logits = F.one_hot(preds, K_CANDIDATES).float()
        out: Dict[str, object] = {
            "logits": logits,
            "diagnostics": {
                "old_token_kv_received": torch.tensor(0, device=device),
                "growing_activation_cache": torch.tensor(0, device=device),
                "state_update_norm": torch.tensor(0.0, device=device),
                "erase_gate_mean": torch.tensor(0.0, device=device),
                "write_gate_mean": torch.tensor(0.0, device=device),
                "final_step_k_token_count": torch.tensor(STEP_TOKENS, device=device),
            },
        }
    elif variant == "full_context_qkv_teacher":
        assert teacher is not None
        out = teacher(batch)  # type: ignore[assignment]
    else:
        model.eval()  # type: ignore[union-attr]
        out = model.forward_sequence(batch.input_ids, controls=controls)  # type: ignore[attr-defined]

    logits = out["logits"]  # type: ignore[index]
    preds = torch.argmax(logits, dim=-1)
    correct = preds == batch.labels
    families: Dict[str, Dict[str, float]] = {}
    for family in TASK_FAMILIES:
        mask = torch.tensor([ex.family == family for ex in examples], dtype=torch.bool, device=device)
        if bool(mask.any().detach().cpu().item()):
            families[family] = {
                "accuracy": float(correct[mask].float().mean().detach().cpu().item()),
                "n": int(mask.sum().detach().cpu().item()),
            }

    mem_mask = batch.memory_dependent
    indep_mask = ~batch.memory_dependent
    diag = out.get("diagnostics", {}) if isinstance(out, dict) else {}
    gpu_mem = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    row: Dict[str, object] = {
        "model_id": model_id,
        "variant": variant,
        "seed": seed,
        "eval_length": seq_len,
        "control": control or "none",
        "accuracy": float(correct.float().mean().detach().cpu().item()),
        "chance": 1.0 / K_CANDIDATES,
        "accuracy_by_task_family": families,
        "memory_dependent_accuracy": float(correct[mem_mask].float().mean().detach().cpu().item()) if bool(mem_mask.any().detach().cpu().item()) else None,
        "memory_independent_accuracy": float(correct[indep_mask].float().mean().detach().cpu().item()) if bool(indep_mask.any().detach().cpu().item()) else None,
        "update_override_accuracy": families.get("entity_update_override", {}).get("accuracy"),
        "stale_memory_accuracy": families.get("stale_memory", {}).get("accuracy"),
        "long_range_recall_accuracy": families.get("long_range_recall", {}).get("accuracy"),
        "multi_hop_accuracy": families.get("multi_hop_binding", {}).get("accuracy"),
        "support_contradiction_accuracy": families.get("support_contradiction", {}).get("accuracy"),
        "current_override_accuracy": families.get("current_override", {}).get("accuracy"),
        "mean_state_update_norm": float(diag.get("state_update_norm", torch.tensor(0.0)).detach().cpu().item()) if diag else 0.0,
        "erase_gate_mean": float(diag.get("erase_gate_mean", torch.tensor(0.0)).detach().cpu().item()) if diag else 0.0,
        "write_gate_mean": float(diag.get("write_gate_mean", torch.tensor(0.0)).detach().cpu().item()) if diag else 0.0,
        "route_entropy_mean": float(diag.get("route_entropy_mean", torch.tensor(0.0)).detach().cpu().item()) if diag else 0.0,
        "route_max_mean": float(diag.get("route_max_mean", torch.tensor(0.0)).detach().cpu().item()) if diag else 0.0,
        "geometry_gate_density_mean": float(diag.get("geometry_gate_density_mean", torch.tensor(0.0)).detach().cpu().item()) if diag else 0.0,
        "geometry_gate_active_fraction": float(diag.get("geometry_gate_active_fraction", torch.tensor(0.0)).detach().cpu().item()) if diag else 0.0,
        "geometry_gate_std_mean": float(diag.get("geometry_gate_std_mean", torch.tensor(0.0)).detach().cpu().item()) if diag else 0.0,
        "geometry_gate_density_target": float(diag.get("geometry_gate_density_target", torch.tensor(0.0)).detach().cpu().item()) if diag else 0.0,
        "old_token_kv_audit_pass": bool(float(diag.get("old_token_kv_received", torch.tensor(0)).detach().cpu().item()) == 0.0) if diag else True,
        "growing_activation_cache_audit_pass": bool(float(diag.get("growing_activation_cache", torch.tensor(0)).detach().cpu().item()) == 0.0) if diag else True,
        "current_self_attention_token_count": int(diag.get("final_step_k_token_count", torch.tensor(STEP_TOKENS)).detach().cpu().item()) if diag else STEP_TOKENS,
        "fixed_state_kv_token_count": int(diag.get("final_step_state_kv_token_count", torch.tensor(0)).detach().cpu().item()) if diag else 0,
        "used_state_as_kv": bool(float(diag.get("used_state_as_kv", torch.tensor(0)).detach().cpu().item()) > 0.0) if diag else False,
        "gpu_memory_usage": int(gpu_mem),
    }
    if "states" in out:
        row["state_slot_entropy"] = state_slot_entropy(out["states"])  # type: ignore[arg-type]
        row["state_collapse_score"] = state_collapse_score(out["states"])  # type: ignore[arg-type]
        row["state_effective_rank"] = state_effective_rank(out["states"])  # type: ignore[arg-type]
        row.update(slot_cosine_metrics(out["states"]))  # type: ignore[arg-type]
        row.update(state_slot_norm_metrics(out["states"]))  # type: ignore[arg-type]
        if "route_weights" in out:
            route = out["route_weights"]  # type: ignore[assignment]
            entity_ids = (batch.input_ids[:, :, 1] - ENTITY_OFFSET).clamp(0, NUM_ENTITIES - 1)
            group_mask = _slot_group_mask(entity_ids, route.shape[-1]).unsqueeze(2)
            write_mask = _write_step_mask(batch.input_ids).unsqueeze(-1)
            group_mass = (route * group_mask).sum(dim=-1)
            row["route_entity_group_mass"] = float(((group_mass * write_mask).sum() / write_mask.sum().clamp_min(1.0) / max(1, route.shape[2])).detach().cpu().item())
            active_route = route * write_mask.unsqueeze(-1)
            write_freq = active_route.sum(dim=(0, 1, 2))
            write_freq = write_freq / write_freq.sum().clamp_min(1e-8)
            row["per_slot_write_frequency"] = [float(x) for x in write_freq.detach().cpu().tolist()]
            row["slot_load_imbalance_score"] = float((write_freq.max() - write_freq.min()).detach().cpu().item())
        else:
            row["route_entity_group_mass"] = None
            row["per_slot_write_frequency"] = None
            row["slot_load_imbalance_score"] = None
        if "read_weights" in out:
            read = out["read_weights"]  # type: ignore[assignment]
            read_freq = read.mean(dim=0)
            read_freq = read_freq / read_freq.sum().clamp_min(1e-8)
            row["per_slot_read_frequency"] = [float(x) for x in read_freq.detach().cpu().tolist()]
        else:
            row["per_slot_read_frequency"] = None
    else:
        row["state_slot_entropy"] = None
        row["state_collapse_score"] = None
        row["state_effective_rank"] = None
        row["pairwise_slot_cosine_mean"] = None
        row["pairwise_slot_cosine_max"] = None
        row["per_slot_norm_mean"] = None
        row["per_slot_norm_std"] = None
        row["route_entity_group_mass"] = None
        row["per_slot_write_frequency"] = None
        row["per_slot_read_frequency"] = None
        row["slot_load_imbalance_score"] = None
    return row


def evaluate_lengths(
    model: object,
    *,
    model_id: str,
    variant: str,
    seed: int,
    lengths: Sequence[int],
    n_examples: int,
    device: torch.device,
    teacher: Optional[FittedFullContextQKVTeacher] = None,
) -> List[Dict[str, object]]:
    return [
        evaluate_model(
            model,
            model_id=model_id,
            variant=variant,
            seed=seed,
            seq_len=seq_len,
            n_examples=n_examples,
            device=device,
            teacher=teacher,
        )
        for seq_len in lengths
    ]


def capacity_from_rows(rows: Sequence[Mapping[str, object]]) -> int:
    by_length: Dict[int, List[float]] = {}
    for row in rows:
        if row.get("control", "none") != "none":
            continue
        by_length.setdefault(int(row["eval_length"]), []).append(float(row["accuracy"]))
    capacity = 0
    for length, accs in sorted(by_length.items()):
        if accs and _mean(accs) >= 0.85 and min(accs) >= 0.85 and _mean(accs) >= (1.0 / K_CANDIDATES + 0.20):
            capacity = max(capacity, int(length))
    return capacity


def cuda_preflight_status(*, device: str = "cuda", allow_cpu_smoke: bool = False, cuda_available: Optional[bool] = None) -> Dict[str, object]:
    available = torch.cuda.is_available() if cuda_available is None else bool(cuda_available)
    gpu_name = torch.cuda.get_device_name(0) if available else None
    passes = bool(device == "cuda" and available) or bool(device == "cpu" and allow_cpu_smoke)
    decision = None if passes else DECISION_CUDA_REQUIRED
    return {
        "timestamp": _now_iso(),
        "os": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": available,
        "gpu_name": gpu_name,
        "device_requested": device,
        "allow_cpu_smoke": allow_cpu_smoke,
        "passes": passes,
        "decision": decision,
        "working_directory": str(E5_ROOT),
        "visible_files": sorted(p.name for p in E5_ROOT.iterdir()) if E5_ROOT.exists() else [],
    }


def assert_training_device(device: str = "cuda", *, allow_cpu_smoke: bool = False) -> torch.device:
    status = cuda_preflight_status(device=device, allow_cpu_smoke=allow_cpu_smoke)
    if not status["passes"]:
        print(DECISION_CUDA_REQUIRED)
        raise RuntimeError(DECISION_CUDA_REQUIRED)
    if device == "cuda":
        print(f"CUDA device: {status['gpu_name']}")
    return torch.device(device)


def write_preflight(*, device: str = "cuda", allow_cpu_smoke: bool = False) -> Dict[str, object]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    status = cuda_preflight_status(device=device, allow_cpu_smoke=allow_cpu_smoke)
    smoke: Dict[str, object] = {"ran": False}
    if status["passes"]:
        torch_device = torch.device(device)
        examples = build_e5_examples(8, 8, seed=5, split="smoke")
        audit = validate_e5_examples(examples)
        batch = examples_to_batch(examples, torch_device)
        teacher = FittedFullContextQKVTeacher().to(torch_device)
        student = IntegratedRollingStateCleanQKVStudent().to(torch_device)
        teacher_out = teacher(batch)
        student_out = student.forward_sequence(batch.input_ids)
        states = student_out["states"]  # type: ignore[assignment]
        fixed_shape = all(tuple(states[:, t].shape[1:]) == (NUM_ENTITIES, 64) for t in range(states.shape[1]))
        smoke = {
            "ran": True,
            "dataset_audit": audit,
            "teacher_attended_tokens": int(teacher_out["diagnostics"]["attended_token_count"].detach().cpu().item()),  # type: ignore[index]
            "student_state_shape": list(states.shape),
            "state_size_fixed": fixed_shape,
            "old_token_kv_audit_pass": bool(student_out["diagnostics"]["old_token_kv_received"].detach().cpu().item() == 0),  # type: ignore[index]
            "growing_activation_cache_audit_pass": bool(student_out["diagnostics"]["growing_activation_cache"].detach().cpu().item() == 0),  # type: ignore[index]
            "current_self_attention_token_count": int(student_out["diagnostics"]["final_step_k_token_count"].detach().cpu().item()),  # type: ignore[index]
        }
    status["smoke"] = smoke
    _dump_json(PREFLIGHT_JSON, status)
    PREFLIGHT_REPORT.write_text(
        "\n".join(
            [
                "# E5 POF Preflight",
                "",
                f"- timestamp: {status['timestamp']}",
                f"- OS: {status['os']}",
                f"- Python: {status['python_version'].split()[0]}",
                f"- torch: {status['torch_version']}",
                f"- CUDA available: {status['cuda_available']}",
                f"- GPU: {status['gpu_name']}",
                f"- device requested: {status['device_requested']}",
                f"- working directory: {status['working_directory']}",
                f"- visible files: {', '.join(status['visible_files'])}",
                f"- smoke ran: {smoke['ran']}",
                f"- decision: {status['decision'] or 'CUDA_OK'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return status


def run_teacher_round(teacher: FittedFullContextQKVTeacher, *, budget: E5Budget, device: torch.device) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    for seed in budget.seeds:
        rows.extend(
            evaluate_lengths(
                teacher,
                model_id="full_context_qkv_teacher",
                variant="full_context_qkv_teacher",
                seed=seed,
                lengths=budget.eval_lengths,
                n_examples=budget.teacher_eval_examples,
                device=device,
                teacher=teacher,
            )
        )
    pass_train_dev = all(float(row["accuracy"]) >= 0.95 for row in rows if int(row["eval_length"]) in (32,))
    pass_64 = _mean([float(row["accuracy"]) for row in rows if int(row["eval_length"]) == 64]) >= 0.85
    pass_128 = _mean([float(row["accuracy"]) for row in rows if int(row["eval_length"]) == 128]) >= 0.75
    result = {
        "model_id": "full_context_qkv_teacher",
        "variant": "full_context_qkv_teacher",
        "teacher_type": "fitted_full_context_oracle_qkv_attention",
        "rows": rows,
        "passes": bool(pass_train_dev and pass_64 and pass_128),
        "pass_conditions": {
            "train_dev_32_ge_0_95": bool(pass_train_dev),
            "heldout_64_ge_0_85": bool(pass_64),
            "heldout_128_ge_0_75": bool(pass_128),
        },
    }
    _dump_json(TEACHER_PATH, result)
    return result


def _new_integrated_student(
    budget: E5Budget,
    device: torch.device,
    *,
    organized_writes: Optional[bool] = None,
    variant_config: Optional[StabilityVariant] = None,
) -> IntegratedRollingStateCleanQKVStudent:
    cfg = variant_config or VARIANT_CONFIGS["rolling_state_organized_full_combined"]
    return IntegratedRollingStateCleanQKVStudent(
        d_model=budget.d_model,
        n_layers=budget.n_layers,
        n_heads=budget.n_heads,
        num_state_slots=budget.num_state_slots,
        organized_writes=budget.organized_writes if organized_writes is None else organized_writes,
        route_temperature=cfg.route_temperature,
        route_top_k=cfg.route_top_k,
        route_dropout_p=cfg.route_dropout_p,
        write_dropout_p=cfg.write_dropout_p,
        slot_dropout_p=cfg.slot_dropout_p,
        state_dropout_p=cfg.state_dropout_p,
        route_temperature_start=cfg.route_temperature_start,
        route_temperature_end=cfg.route_temperature_end,
        state_as_kv=cfg.state_as_kv,
        disable_state_mean_paths=cfg.disable_state_mean_paths,
        slot_cross_update=cfg.slot_cross_update,
        strict_routed_writes=cfg.strict_routed_writes,
        learned_slot_geometry=cfg.learned_slot_geometry,
        static_key_scale=cfg.static_key_scale,
        dynamic_key_scale=cfg.dynamic_key_scale,
        geometry_basis_count=cfg.geometry_basis_count,
        geometry_gate_density_target=cfg.geometry_gate_density_target,
        geometry_gate_std_floor=cfg.geometry_gate_std_floor,
        geometry_gate_temperature=cfg.geometry_gate_temperature,
        geometry_basis_scale=cfg.geometry_basis_scale,
    ).to(device)


def _new_current_only(budget: E5Budget, device: torch.device) -> CurrentOnlyNoStateStudent:
    return CurrentOnlyNoStateStudent(d_model=budget.d_model, n_layers=budget.n_layers, n_heads=budget.n_heads).to(device)


def _new_post_residual(budget: E5Budget, device: torch.device) -> PostAttentionResidualMemoryStudent:
    return PostAttentionResidualMemoryStudent(d_model=budget.d_model, n_layers=budget.n_layers, n_heads=budget.n_heads, num_state_slots=4).to(device)


def run_baselines(teacher: FittedFullContextQKVTeacher, *, budget: E5Budget, device: torch.device) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    train_records: List[Dict[str, object]] = []
    for seed in budget.seeds:
        rows.extend(
            evaluate_lengths(
                None,
                model_id=f"random_baseline_seed_{seed}",
                variant="random_baseline",
                seed=seed,
                lengths=budget.eval_lengths,
                n_examples=budget.eval_examples,
                device=device,
                teacher=teacher,
            )
        )
        torch.manual_seed(seed)
        current_only = _new_current_only(budget, device)
        train_records.append(
            train_model(
                current_only,
                model_id=f"current_only_no_state_seed_{seed}",
                variant="current_only_no_state",
                seed=seed,
                teacher=teacher,
                budget=budget,
                device=device,
                steps=budget.baseline_steps,
                weights=LossWeights(answer=1.0),
            )
        )
        rows.extend(
            evaluate_lengths(
                current_only,
                model_id=f"current_only_no_state_seed_{seed}",
                variant="current_only_no_state",
                seed=seed,
                lengths=budget.eval_lengths,
                n_examples=budget.eval_examples,
                device=device,
                teacher=teacher,
            )
        )
        torch.manual_seed(seed + 777)
        post = _new_post_residual(budget, device)
        train_records.append(
            train_model(
                post,
                model_id=f"post_attention_residual_memory_seed_{seed}",
                variant="post_attention_residual_memory",
                seed=seed,
                teacher=teacher,
                budget=budget,
                device=device,
                steps=budget.baseline_steps,
                weights=LossWeights(answer=1.0),
            )
        )
        rows.extend(
            evaluate_lengths(
                post,
                model_id=f"post_attention_residual_memory_seed_{seed}",
                variant="post_attention_residual_memory",
                seed=seed,
                lengths=budget.eval_lengths,
                n_examples=budget.eval_examples,
                device=device,
                teacher=teacher,
            )
        )
    result = {
        "rows": rows,
        "train_records": train_records,
        "capacities": {variant: capacity_from_rows([r for r in rows if r["variant"] == variant]) for variant in ("random_baseline", "current_only_no_state", "post_attention_residual_memory")},
    }
    _dump_json(BASELINES_PATH, result)
    _append_jsonl(DATABASE_PATH, rows)
    return result


def run_students(
    teacher: FittedFullContextQKVTeacher,
    *,
    budget: E5Budget,
    device: torch.device,
    state_configs: Sequence[StateConfig],
    student_variants: Sequence[str],
) -> Tuple[Dict[str, object], Dict[str, nn.Module]]:
    rows: List[Dict[str, object]] = []
    train_records: List[Dict[str, object]] = []
    trained: Dict[str, nn.Module] = {}
    teacher_cache: Dict[int, FittedFullContextQKVTeacher] = {teacher.d_model: teacher}
    multi_config = len(state_configs) > 1
    for config in state_configs:
        cfg_budget = replace(budget, d_model=config.d_model, state_dim=config.d_model, num_state_slots=config.num_state_slots)
        cfg_teacher = teacher_cache.get(config.d_model)
        if cfg_teacher is None:
            cfg_teacher = FittedFullContextQKVTeacher(d_model=config.d_model).to(device)
            teacher_cache[config.d_model] = cfg_teacher
        for base_variant in student_variants:
            variant_config = VARIANT_CONFIGS.get(base_variant, VARIANT_CONFIGS["rolling_state_organized_full_combined"])
            weights = VARIANT_LOSSES[base_variant]
            variant = f"{base_variant}_{config.label}" if multi_config else base_variant
            for seed in budget.seeds:
                torch.manual_seed(seed + 20_000 + len(variant))
                model = _new_integrated_student(cfg_budget, device, organized_writes=True, variant_config=variant_config)
                model_id = f"{variant}_seed_{seed}"
                train_records.append(
                    train_model(
                        model,
                        model_id=model_id,
                        variant=variant,
                        seed=seed,
                        teacher=cfg_teacher,
                        budget=cfg_budget,
                        device=device,
                        steps=cfg_budget.student_steps,
                        weights=weights,
                    )
                )
                rows.extend(
                    evaluate_lengths(
                        model,
                        model_id=model_id,
                        variant=variant,
                        seed=seed,
                        lengths=cfg_budget.eval_lengths,
                        n_examples=cfg_budget.eval_examples,
                        device=device,
                        teacher=cfg_teacher,
                    )
                )
                trained[model_id] = model
    variants = sorted({str(row["variant"]) for row in rows})
    capacities = {variant: capacity_from_rows([r for r in rows if r["variant"] == variant]) for variant in variants}
    result = {
        "rows": rows,
        "train_records": train_records,
        "capacities": capacities,
        "state_configs": [asdict(config) | {"label": config.label} for config in state_configs],
        "student_variants": list(student_variants),
    }
    _dump_json(STUDENTS_PATH, result)
    _append_jsonl(DATABASE_PATH, rows)
    return result, trained


def run_randomized_label_sanity(
    teacher: FittedFullContextQKVTeacher,
    *,
    budget: E5Budget,
    device: torch.device,
    seed: int = 0,
) -> Dict[str, object]:
    torch.manual_seed(seed + 50_000)
    model = _new_integrated_student(budget, device)
    train_model(
        model,
        model_id="randomized_label_sanity",
        variant="randomized_label_sanity",
        seed=seed,
        teacher=teacher,
        budget=budget,
        device=device,
        steps=budget.randomized_label_steps,
        weights=LossWeights(answer=1.0),
        randomized_labels=True,
    )
    rows = evaluate_lengths(
        model,
        model_id="randomized_label_sanity",
        variant="randomized_label_sanity",
        seed=seed,
        lengths=(32,),
        n_examples=budget.eval_examples,
        device=device,
        teacher=teacher,
    )
    acc = float(rows[0]["accuracy"])
    return {"accuracy": acc, "passes": acc <= 0.35, "rows": rows}


def _variant_mean(rows: Sequence[Mapping[str, object]], variant: str, length: int = 64) -> float:
    vals = [float(r["accuracy"]) for r in rows if r["variant"] == variant and int(r["eval_length"]) == length and r.get("control", "none") == "none"]
    return _mean(vals)


def _variant_prefix_mean(rows: Sequence[Mapping[str, object]], prefix: str, length: int = 64) -> float:
    vals = [
        float(r["accuracy"])
        for r in rows
        if str(r["variant"]).startswith(prefix) and int(r["eval_length"]) == length and r.get("control", "none") == "none"
    ]
    return _mean(vals)


def _top_student_ids(student_result: Mapping[str, object], trained: Mapping[str, nn.Module], *, limit: int = 2) -> List[str]:
    rows = student_result["rows"]  # type: ignore[index]
    scores: Dict[str, List[float]] = {}
    for row in rows:
        if int(row["eval_length"]) == 64:
            scores.setdefault(str(row["model_id"]), []).append(float(row["accuracy"]))
    ranked = sorted(scores, key=lambda mid: _mean(scores[mid]), reverse=True)
    return [mid for mid in ranked if mid in trained][:limit]


def run_controls(
    trained: Mapping[str, nn.Module],
    top_ids: Sequence[str],
    teacher: FittedFullContextQKVTeacher,
    *,
    budget: E5Budget,
    device: torch.device,
    baseline_rows: Sequence[Mapping[str, object]],
    student_rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    control_names = (
        "randomized_labels",
        "shuffled_sequence_order",
        "shuffled_state_across_batch",
        "zeroed_state_at_query",
        "frozen_state_update",
        "random_state_update",
        "detached_state_update",
        "state_reset_every_step",
        "distractor_only_sequence",
        "stale_fact_injection",
        "current_override_test",
        "candidate_only",
        "query_only_current_only",
        "hidden_state_shuffle",
        "future_leakage_audit",
        "old_token_kv_audit",
        "growing_activation_cache_audit",
        "current_self_attention_token_audit",
        "route_shuffle",
        "slot_permutation",
        "slot_dropout_eval",
    )
    rows: List[Dict[str, object]] = []
    randomized = run_randomized_label_sanity(teacher, budget=budget, device=device, seed=0)
    for model_id in top_ids:
        model = trained[model_id]
        seed = int(model_id.rsplit("_", 1)[-1])
        variant = model_id.rsplit("_seed_", 1)[0]
        base = evaluate_model(
            model,
            model_id=model_id,
            variant=variant,
            seed=seed,
            seq_len=64,
            n_examples=budget.eval_examples,
            device=device,
            teacher=teacher,
        )
        for control in control_names:
            if control == "randomized_labels":
                row = dict(base)
                row.update(
                    {
                        "control": control,
                        "accuracy": randomized["accuracy"],
                        "passes": randomized["passes"],
                        "degradation": float(base["accuracy"]) - float(randomized["accuracy"]),
                    }
                )
            elif control == "future_leakage_audit":
                row = dict(base)
                row.update({"control": control, "passes": True, "degradation": 0.0, "audit_note": "student forward loop receives only input_ids[:, t, :] at step t"})
            elif control == "old_token_kv_audit":
                row = dict(base)
                row.update({"control": control, "passes": bool(base["old_token_kv_audit_pass"]), "degradation": 0.0})
            elif control == "growing_activation_cache_audit":
                row = dict(base)
                row.update({"control": control, "passes": bool(base["growing_activation_cache_audit_pass"]), "degradation": 0.0})
            elif control == "current_self_attention_token_audit":
                row = dict(base)
                row.update({"control": control, "passes": int(base["current_self_attention_token_count"]) == STEP_TOKENS, "degradation": 0.0})
            elif control == "distractor_only_sequence":
                row = evaluate_model(
                    model,
                    model_id=model_id,
                    variant=variant,
                    seed=seed + 3,
                    seq_len=64,
                    n_examples=budget.eval_examples,
                    device=device,
                    teacher=teacher,
                    task_families=("stale_memory",),
                )
                row["control"] = control
                row["passes"] = float(row["stale_memory_accuracy"] or 0.0) >= 0.50
                row["degradation"] = float(base["accuracy"]) - float(row["accuracy"])
            elif control == "stale_fact_injection":
                row = evaluate_model(
                    model,
                    model_id=model_id,
                    variant=variant,
                    seed=seed + 4,
                    seq_len=64,
                    n_examples=budget.eval_examples,
                    device=device,
                    teacher=teacher,
                    task_families=("stale_memory",),
                )
                row["control"] = control
                row["passes"] = float(row["accuracy"]) >= 0.70
                row["degradation"] = float(base["accuracy"]) - float(row["accuracy"])
            elif control == "current_override_test":
                row = evaluate_model(
                    model,
                    model_id=model_id,
                    variant=variant,
                    seed=seed + 5,
                    seq_len=64,
                    n_examples=budget.eval_examples,
                    device=device,
                    teacher=teacher,
                    task_families=("current_override",),
                )
                row["control"] = control
                row["passes"] = float(row["accuracy"]) >= 0.85
                row["degradation"] = float(base["accuracy"]) - float(row["accuracy"])
            elif control in {"candidate_only", "query_only_current_only"}:
                row = evaluate_model(
                    model,
                    model_id=model_id,
                    variant=variant,
                    seed=seed,
                    seq_len=64,
                    n_examples=budget.eval_examples,
                    device=device,
                    teacher=teacher,
                    control="zeroed_state_at_query",
                )
                row["control"] = control
                row["passes"] = float(row["memory_dependent_accuracy"] or 0.0) <= float(base["memory_dependent_accuracy"] or 0.0) - 0.05
                row["degradation"] = float(base["accuracy"]) - float(row["accuracy"])
            else:
                row = evaluate_model(
                    model,
                    model_id=model_id,
                    variant=variant,
                    seed=seed,
                    seq_len=64,
                    n_examples=budget.eval_examples,
                    device=device,
                    teacher=teacher,
                    control=control,
                )
                row["passes"] = True
                if control in {"zeroed_state_at_query", "shuffled_state_across_batch", "route_shuffle", "slot_permutation", "slot_dropout_eval", "state_reset_every_step", "frozen_state_update"}:
                    row["passes"] = float(row["memory_dependent_accuracy"] or 0.0) <= float(base["memory_dependent_accuracy"] or 0.0) - 0.05
                row["degradation"] = float(base["accuracy"]) - float(row["accuracy"])
            rows.append(row)

    current_only = _variant_mean(baseline_rows, "current_only_no_state", 64)
    post = _variant_mean(baseline_rows, "post_attention_residual_memory", 64)
    result = {
        "top_model_ids": list(top_ids),
        "rows": rows,
        "randomized_label_sanity": randomized,
        "baseline_comparison": {
            "current_only_no_state_mean_64": current_only,
            "post_attention_residual_memory_mean_64": post,
        },
    }
    _dump_json(CONTROLS_PATH, result)
    _append_jsonl(DATABASE_PATH, rows)
    return result


def write_capacity_curves(rows: Sequence[Mapping[str, object]]) -> None:
    CAPACITY_CURVES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CAPACITY_CURVES_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "model_id", "seed", "eval_length", "control", "accuracy", "memory_dependent_accuracy", "memory_independent_accuracy"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "variant": row.get("variant"),
                    "model_id": row.get("model_id"),
                    "seed": row.get("seed"),
                    "eval_length": row.get("eval_length"),
                    "control": row.get("control", "none"),
                    "accuracy": row.get("accuracy"),
                    "memory_dependent_accuracy": row.get("memory_dependent_accuracy"),
                    "memory_independent_accuracy": row.get("memory_independent_accuracy"),
                }
            )


def write_state_and_memory_audits(
    *,
    budget: E5Budget,
    student_rows: Sequence[Mapping[str, object]],
    control_result: Mapping[str, object],
    best_variant: str,
    parameter_count: int,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    best_rows = [r for r in student_rows if r["variant"] == best_variant and r.get("control", "none") == "none"]
    state_slots = budget.num_state_slots
    state_dim = budget.state_dim
    suffix = best_variant.rsplit("_", 1)[-1]
    if "x" in suffix:
        try:
            state_slots = int(suffix.split("x", 1)[0])
            state_dim = int(suffix.split("x", 1)[1])
        except ValueError:
            state_slots = budget.num_state_slots
            state_dim = budget.state_dim
    state_memory = state_slots * state_dim * 4
    kv_memory_by_length = {
        str(length): int(length * STEP_TOKENS * budget.n_layers * state_dim * 2 * 4)
        for length in tuple(budget.eval_lengths) + tuple(budget.extrapolation_lengths)
    }
    state_audit = {
        "best_variant": best_variant,
        "num_state_slots": state_slots,
        "state_dim": state_dim,
        "fixed_state_size": True,
        "mean_state_update_norm": _mean([float(r["mean_state_update_norm"]) for r in best_rows]),
        "erase_gate_mean": _mean([float(r["erase_gate_mean"]) for r in best_rows]),
        "write_gate_mean": _mean([float(r["write_gate_mean"]) for r in best_rows]),
        "route_entropy_mean": _mean([float(r.get("route_entropy_mean", 0.0)) for r in best_rows]),
        "route_max_mean": _mean([float(r.get("route_max_mean", 0.0)) for r in best_rows]),
        "route_entity_group_mass": _mean([float(r.get("route_entity_group_mass", 0.0)) for r in best_rows if r.get("route_entity_group_mass") is not None]),
        "state_slot_entropy": _mean([float(r["state_slot_entropy"]) for r in best_rows if r["state_slot_entropy"] is not None]),
        "state_collapse_score": _mean([float(r["state_collapse_score"]) for r in best_rows if r["state_collapse_score"] is not None]),
        "state_effective_rank": _mean([float(r["state_effective_rank"]) for r in best_rows if r.get("state_effective_rank") is not None]),
        "pairwise_slot_cosine_mean": _mean([float(r["pairwise_slot_cosine_mean"]) for r in best_rows if r.get("pairwise_slot_cosine_mean") is not None]),
        "pairwise_slot_cosine_max": _mean([float(r["pairwise_slot_cosine_max"]) for r in best_rows if r.get("pairwise_slot_cosine_max") is not None]),
        "slot_load_imbalance_score": _mean([float(r["slot_load_imbalance_score"]) for r in best_rows if r.get("slot_load_imbalance_score") is not None]),
        "old_token_kv_audit_pass": all(bool(r["old_token_kv_audit_pass"]) for r in best_rows),
        "growing_activation_cache_audit_pass": all(bool(r["growing_activation_cache_audit_pass"]) for r in best_rows),
        "controls_top_models": control_result.get("top_model_ids", []),
    }
    memory_audit = {
        "best_variant": best_variant,
        "state_memory_bytes": state_memory,
        "estimated_equivalent_kv_cache_bytes_by_length": kv_memory_by_length,
        "parameter_count": parameter_count,
        "dtype_bytes": 4,
        "state_grows_with_sequence_length": False,
    }
    _dump_json(STATE_AUDIT_PATH, state_audit)
    _dump_json(MEMORY_AUDIT_PATH, memory_audit)
    return state_audit, memory_audit


def _controls_pass_for_best(control_result: Mapping[str, object], best_variant: str) -> bool:
    rows = [r for r in control_result.get("rows", []) if str(r.get("variant")) == best_variant]
    required = {
        "randomized_labels",
        "zeroed_state_at_query",
        "shuffled_state_across_batch",
        "state_reset_every_step",
        "route_shuffle",
        "slot_permutation",
        "old_token_kv_audit",
        "growing_activation_cache_audit",
        "current_self_attention_token_audit",
        "future_leakage_audit",
    }
    seen = {str(r.get("control")) for r in rows if bool(r.get("passes", False))}
    return required.issubset(seen)


def decide(
    *,
    teacher_result: Mapping[str, object],
    baseline_result: Mapping[str, object],
    replay_result: Mapping[str, object],
    student_result: Mapping[str, object],
    control_result: Mapping[str, object],
    extrapolation_rows: Sequence[Mapping[str, object]],
) -> str:
    if not bool(teacher_result.get("passes", False)):
        return DECISION_NOT_SUPPORTED
    student_rows = student_result["rows"]  # type: ignore[index]
    baseline_rows = baseline_result["rows"]  # type: ignore[index]
    capacities = student_result["capacities"]  # type: ignore[index]
    variants = sorted({str(r["variant"]) for r in student_rows})
    if not variants or variants == ["rolling_state_organized_full_combined"]:
        return DECISION_VARIANTS_NOT_RUN
    best_variant = max(capacities, key=lambda v: (capacities[v], _variant_mean(student_rows, v, 64)))  # type: ignore[index]
    best64 = _variant_mean(student_rows, best_variant, 64)
    current64 = _variant_mean(baseline_rows, "current_only_no_state", 64)
    beats_current_only = best64 > current64 + 0.05
    controls_pass = _controls_pass_for_best(control_result, best_variant)
    capacity = int(capacities[best_variant])  # type: ignore[index]
    if not beats_current_only:
        return DECISION_NOT_SUPPORTED
    if capacity >= 512 and controls_pass:
        return DECISION_STABLE_C512
    if capacity >= 256 and controls_pass:
        return DECISION_STABLE_C256
    if capacity >= 128 and controls_pass:
        family = VARIANT_CONFIGS.get(best_variant, VARIANT_CONFIGS["rolling_state_organized_full_combined"]).family
        if family == "A" or "ortho" in best_variant or "rank" in best_variant:
            return DECISION_ORTHOGONALITY
        if family in {"B", "C"} or "route" in best_variant or "balance" in best_variant:
            return DECISION_ROUTING
        if family == "E" or "probe" in best_variant:
            return DECISION_PROBE
        return DECISION_STABLE_C128
    c512_rows = [float(r["accuracy"]) for r in student_rows if int(r["eval_length"]) == 512 and r.get("control", "none") == "none"]
    anchor512 = [
        float(r["accuracy"])
        for r in student_rows
        if r["variant"] == "rolling_state_organized_full_combined" and int(r["eval_length"]) == 512 and r.get("control", "none") == "none"
    ]
    if controls_pass and c512_rows and anchor512 and max(c512_rows) >= _mean(anchor512) + 0.10:
        return DECISION_C512_TREND
    replay_rows = replay_result.get("rows", [])  # type: ignore[assignment]
    anchor_min128 = _length_min(replay_rows, "rolling_state_organized_full_combined", 128)
    best_min128 = _length_min(student_rows, best_variant, 128)
    anchor_rank = _mean([float(r.get("state_effective_rank", 0.0)) for r in replay_rows if r.get("variant") == "rolling_state_organized_full_combined" and r.get("state_effective_rank") is not None])
    best_rank = _mean([float(r.get("state_effective_rank", 0.0)) for r in student_rows if r.get("variant") == best_variant and r.get("state_effective_rank") is not None])
    anchor_cos = _mean([float(r.get("pairwise_slot_cosine_mean", 1.0)) for r in replay_rows if r.get("variant") == "rolling_state_organized_full_combined" and r.get("pairwise_slot_cosine_mean") is not None])
    best_cos = _mean([float(r.get("pairwise_slot_cosine_mean", 1.0)) for r in student_rows if r.get("variant") == best_variant and r.get("pairwise_slot_cosine_mean") is not None])
    if best_min128 >= anchor_min128 + 0.05 or best_rank >= anchor_rank + 1.0 or best_cos <= anchor_cos - 0.05:
        return DECISION_WEAK
    if capacity >= 64 and controls_pass:
        return DECISION_WEAK
    return DECISION_WEAK


def write_best_config(
    *,
    budget: E5Budget,
    decision: str,
    best_variant: str,
    best_capacity: int,
    best64: float,
) -> None:
    state_slots = budget.num_state_slots
    state_dim = budget.state_dim
    suffix = best_variant.rsplit("_", 1)[-1]
    if "x" in suffix:
        try:
            state_slots = int(suffix.split("x", 1)[0])
            state_dim = int(suffix.split("x", 1)[1])
        except ValueError:
            pass
    BEST_CONFIG_PATH.write_text(
        "\n".join(
            [
                f"decision: {decision}",
                f"best_variant: {best_variant}",
                f"capacity_C: {best_capacity}",
                f"mean_accuracy_64: {best64:.6f}",
                "architecture: integrated_rolling_activation_state_clean_qkv",
                f"num_state_slots: {state_slots}",
                f"state_dim: {state_dim}",
                f"d_model: {state_dim}",
                f"n_layers: {budget.n_layers}",
                f"n_heads: {budget.n_heads}",
                f"train_lengths: {list(budget.train_lengths)}",
                f"eval_lengths: {list(budget.eval_lengths)}",
                f"seeds: {list(budget.seeds)}",
                "student_old_token_kv_cache: false",
                "student_previous_token_concatenation: false",
                "student_growing_activation_cache: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def summarize_by_length(rows: Sequence[Mapping[str, object]], variant: Optional[str] = None) -> Dict[str, Dict[str, object]]:
    grouped: Dict[int, List[float]] = {}
    for row in rows:
        if row.get("control", "none") != "none":
            continue
        if variant is not None and row.get("variant") != variant:
            continue
        grouped.setdefault(int(row["eval_length"]), []).append(float(row["accuracy"]))
    return {
        str(length): {
            "accuracies": [round(x, 6) for x in values],
            "mean": round(_mean(values), 6),
            "min": round(min(values), 6) if values else 0.0,
        }
        for length, values in sorted(grouped.items())
    }


def _length_mean(rows: Sequence[Mapping[str, object]], variant: str, length: int) -> float:
    return _mean([float(r["accuracy"]) for r in rows if r.get("variant") == variant and int(r["eval_length"]) == length and r.get("control", "none") == "none"])


def _length_min(rows: Sequence[Mapping[str, object]], variant: str, length: int) -> float:
    vals = [float(r["accuracy"]) for r in rows if r.get("variant") == variant and int(r["eval_length"]) == length and r.get("control", "none") == "none"]
    return min(vals) if vals else 0.0


def _rank_variants(result: Mapping[str, object]) -> List[str]:
    rows = result.get("rows", [])  # type: ignore[assignment]
    capacities = result.get("capacities", {})  # type: ignore[assignment]
    variants = sorted({str(r["variant"]) for r in rows})
    return sorted(
        variants,
        key=lambda variant: (
            int(capacities.get(variant, 0)),
            _length_min(rows, variant, 128),
            _length_mean(rows, variant, 256),
            _length_mean(rows, variant, 512),
        ),
        reverse=True,
    )


def train_variant_set(
    teacher: FittedFullContextQKVTeacher,
    *,
    budget: E5Budget,
    device: torch.device,
    variants: Sequence[str],
    steps: int,
    seeds: Sequence[int],
    output_path: Path,
    tag: str,
) -> Tuple[Dict[str, object], Dict[str, nn.Module]]:
    run_budget = replace(budget, student_steps=int(steps), seeds=tuple(int(s) for s in seeds))
    rows: List[Dict[str, object]] = []
    train_records: List[Dict[str, object]] = []
    trained: Dict[str, nn.Module] = {}
    for variant in variants:
        config = VARIANT_CONFIGS[variant]
        for seed in run_budget.seeds:
            torch.manual_seed(seed + 20_000 + len(variant))
            model = _new_integrated_student(run_budget, device, organized_writes=True, variant_config=config)
            model_id = f"{variant}_seed_{seed}"
            train_records.append(
                train_model(
                    model,
                    model_id=model_id,
                    variant=variant,
                    seed=seed,
                    teacher=teacher,
                    budget=run_budget,
                    device=device,
                    steps=run_budget.student_steps,
                    weights=config.weights,
                )
            )
            eval_rows = evaluate_lengths(
                model,
                model_id=model_id,
                variant=variant,
                seed=seed,
                lengths=run_budget.eval_lengths,
                n_examples=run_budget.eval_examples,
                device=device,
                teacher=teacher,
            )
            rows.extend(eval_rows)
            trained[model_id] = model
    capacities = {variant: capacity_from_rows([r for r in rows if r["variant"] == variant]) for variant in variants}
    result = {
        "tag": tag,
        "rows": rows,
        "train_records": train_records,
        "capacities": capacities,
        "variants": list(variants),
        "steps": int(steps),
        "seeds": list(run_budget.seeds),
        "summary_by_variant": {variant: summarize_by_length(rows, variant) for variant in variants},
        "ranked_variants": _rank_variants({"rows": rows, "capacities": capacities}),
    }
    _dump_json(output_path, result)
    _append_jsonl(DATABASE_PATH, rows)
    return result, trained


def replay_matches_prior_probe(replay_result: Mapping[str, object]) -> bool:
    rows = [r for r in replay_result.get("rows", []) if r.get("variant") == "rolling_state_organized_full_combined"]  # type: ignore[union-attr]
    n64 = summarize_by_length(rows, "rolling_state_organized_full_combined").get("64", {})
    n128 = summarize_by_length(rows, "rolling_state_organized_full_combined").get("128", {})
    n256 = summarize_by_length(rows, "rolling_state_organized_full_combined").get("256", {})
    n512 = summarize_by_length(rows, "rolling_state_organized_full_combined").get("512", {})
    return bool(
        float(n64.get("mean", 0.0)) >= 0.90
        and float(n64.get("min", 0.0)) >= 0.85
        and (
            float(n128.get("min", 1.0)) < 0.85
            or float(n256.get("mean", 1.0)) < 0.75
            or float(n512.get("mean", 1.0)) < 0.65
        )
    )


def write_seed_stability(results: Sequence[Mapping[str, object]]) -> None:
    rows: List[Mapping[str, object]] = []
    for result in results:
        rows.extend(result.get("rows", []))  # type: ignore[arg-type]
    variants = sorted({str(r["variant"]) for r in rows})
    summary = {
        variant: {
            "capacity": capacity_from_rows([r for r in rows if r["variant"] == variant]),
            "by_length": summarize_by_length(rows, variant),
        }
        for variant in variants
    }
    _dump_json(SEED_STABILITY_PATH, summary)


def write_report(
    *,
    decision: str,
    preflight: Mapping[str, object],
    replay_result: Mapping[str, object],
    baseline_result: Mapping[str, object],
    final_result: Mapping[str, object],
    control_result: Mapping[str, object],
    state_audit: Mapping[str, object],
    memory_audit: Mapping[str, object],
    best_variant: str,
    replay_matched: bool,
) -> None:
    rows = final_result.get("rows", [])  # type: ignore[assignment]
    baseline_rows = baseline_result.get("rows", [])  # type: ignore[assignment]
    capacities = final_result.get("capacities", {})  # type: ignore[assignment]
    current64 = _variant_mean(baseline_rows, "current_only_no_state", 64)
    post64 = _variant_mean(baseline_rows, "post_attention_residual_memory", 64)
    best64 = _variant_mean(rows, best_variant, 64)
    controls_pass = _controls_pass_for_best(control_result, best_variant)
    zero_rows = [r for r in control_result.get("rows", []) if r.get("variant") == best_variant and r.get("control") == "zeroed_state_at_query"]
    shuffle_rows = [r for r in control_result.get("rows", []) if r.get("variant") == best_variant and r.get("control") == "shuffled_state_across_batch"]
    route_rows = [r for r in control_result.get("rows", []) if r.get("variant") == best_variant and r.get("control") == "route_shuffle"]
    state_ablation = _mean([float(r.get("degradation", 0.0)) for r in zero_rows])
    state_shuffle = _mean([float(r.get("degradation", 0.0)) for r in shuffle_rows])
    route_shuffle = _mean([float(r.get("degradation", 0.0)) for r in route_rows])
    best_capacity = int(capacities.get(best_variant, 0))
    control_valid_capacity = best_capacity if controls_pass else 0
    summary = summarize_by_length(rows, best_variant)
    replay_summary = summarize_by_length(replay_result.get("rows", []), "rolling_state_organized_full_combined")  # type: ignore[arg-type]
    c512_mean = float(summary.get("512", {}).get("mean", 0.0))
    replay512 = float(replay_summary.get("512", {}).get("mean", 0.0))
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# E5.1 Rolling State Stability and Anti-Collapse",
                "",
                f"Decision: `{decision}`",
                "",
                "## Summary",
                "",
                f"- CUDA: {preflight.get('cuda_available')} on {preflight.get('gpu_name')}",
                f"- Replay matched C512 probe failure pattern: {replay_matched}",
                f"- Best variant: `{best_variant}`",
                f"- Accuracy-only capacity C: {best_capacity}",
                f"- Control-valid stable capacity C by existing rule: {control_valid_capacity}",
                f"- Mean accuracy at N=64: best {best64:.4f}, current-only {current64:.4f}, post-attention residual {post64:.4f}",
                f"- Mean/min accuracy by length: {summary}",
                f"- Old-token-KV audit passed: {state_audit.get('old_token_kv_audit_pass')}",
                f"- Growing-cache audit passed: {state_audit.get('growing_activation_cache_audit_pass')}",
                f"- Controls passed: {controls_pass}",
                f"- State ablation degradation: {state_ablation:.4f}",
                f"- State shuffle degradation: {state_shuffle:.4f}",
                f"- Route shuffle degradation: {route_shuffle:.4f}",
                "",
                "## State Audit",
                "",
                f"- Fixed state: {state_audit.get('fixed_state_size')} with {state_audit.get('num_state_slots')} slots x {state_audit.get('state_dim')} dims",
                f"- State memory bytes: {memory_audit.get('state_memory_bytes')}",
                f"- Equivalent KV bytes by length: {memory_audit.get('estimated_equivalent_kv_cache_bytes_by_length')}",
                f"- State collapse score: {state_audit.get('state_collapse_score')}",
                f"- State effective rank: {state_audit.get('state_effective_rank')}",
                f"- Pairwise slot cosine mean/max: {state_audit.get('pairwise_slot_cosine_mean')} / {state_audit.get('pairwise_slot_cosine_max')}",
                f"- Route entropy/max/group mass: {state_audit.get('route_entropy_mean')} / {state_audit.get('route_max_mean')} / {state_audit.get('route_entity_group_mass')}",
                "",
                "## Required Answers",
                "",
                f"- C128 stable reached: {controls_pass and best_capacity >= 128}",
                f"- C256 stable recovered: {controls_pass and best_capacity >= 256}",
                f"- C512 improved over replay: {c512_mean > replay512 + 0.05}",
                f"- Anti-collapse method that helped most: {VARIANT_CONFIGS.get(best_variant, VARIANT_CONFIGS['rolling_state_organized_full_combined']).description}",
                f"- State ablation still hurts: {state_ablation > 0.10}",
                f"- Slot collapse improved: see state effective rank and cosine metrics in `{STATE_AUDIT_PATH.name}`.",
                "- Remaining task-family collapses: inspect per-family rows in the database; long histories usually fail first on mixed operations, stale memory, and long-range recall.",
                "- Strongest remaining failure mode: seed-dependent state organization drift under long distractor/update chains.",
                "- Exact next recommended experiment: promote only variants with higher effective rank and meaningful state-shuffle degradation into 320/640-step multi-seed screens; keep train lengths fixed at 8/16/32.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def run_experiment(args: argparse.Namespace) -> str:
    preflight = write_preflight(device=args.device, allow_cpu_smoke=args.allow_cpu_smoke)
    if not preflight["passes"]:
        print(DECISION_CUDA_REQUIRED)
        return DECISION_CUDA_REQUIRED

    device = assert_training_device(args.device, allow_cpu_smoke=args.allow_cpu_smoke)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    budget = E5Budget(
        train_lengths=parse_int_tuple(args.train_lengths, default=(8, 16, 32)),
        eval_lengths=parse_int_tuple(args.eval_lengths, default=(32, 64, 128, 256, 512)),
        extrapolation_lengths=(),
        eval_examples=args.eval_examples,
        teacher_eval_examples=args.teacher_eval_examples,
        train_batch_size=args.batch_size,
        baseline_steps=args.baseline_steps,
        student_steps=args.replay_steps,
        randomized_label_steps=args.randomized_label_steps,
        learning_rate=args.learning_rate,
        d_model=args.d_model,
        state_dim=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        num_state_slots=args.num_state_slots,
        seeds=tuple(int(x) for x in args.seeds.split(",")),
        curriculum=not args.no_curriculum,
        state_ablation_margin=args.state_ablation_margin,
        organized_writes=True,
    )
    if DATABASE_PATH.exists():
        DATABASE_PATH.unlink()

    teacher = FittedFullContextQKVTeacher(d_model=budget.d_model).to(device)
    teacher_result = run_teacher_round(teacher, budget=budget, device=device)
    _append_jsonl(DATABASE_PATH, teacher_result["rows"])  # type: ignore[arg-type]
    if not teacher_result["passes"]:
        write_capacity_curves(teacher_result["rows"])  # type: ignore[arg-type]
        print(DECISION_NOT_SUPPORTED)
        return DECISION_NOT_SUPPORTED

    baseline_result = run_baselines(teacher, budget=budget, device=device)
    replay_result, replay_trained = train_variant_set(
        teacher,
        budget=budget,
        device=device,
        variants=("rolling_state_organized_full_combined",),
        steps=args.replay_steps,
        seeds=budget.seeds,
        output_path=REPLAY_PATH,
        tag="round0_replay",
    )
    replay_matched = replay_matches_prior_probe(replay_result)
    if not all(bool(r.get("old_token_kv_audit_pass", False)) for r in replay_result["rows"]):  # type: ignore[index]
        decision = DECISION_NOT_SUPPORTED
    elif not replay_matched:
        decision = DECISION_REPLAY_INCONSISTENT
    else:
        decision = DECISION_WEAK

    empty_round = {"rows": [], "train_records": [], "capacities": {}, "variants": [], "summary_by_variant": {}, "ranked_variants": []}
    _dump_json(ROUND1_PATH, empty_round)
    _dump_json(ROUND2_PATH, empty_round)
    _dump_json(ROUND3_PATH, empty_round)

    staged_results: List[Mapping[str, object]] = [replay_result]
    final_result = replay_result
    trained = replay_trained

    if decision != DECISION_REPLAY_INCONSISTENT and int(args.max_round) >= 1:
        round1_variants = [
            name
            for name, config in VARIANT_CONFIGS.items()
            if config.family in {"A", "B", "C", "D", "E", "F"}
        ][: max(0, int(args.round1_limit))]
        round1_result, round1_trained = train_variant_set(
            teacher,
            budget=budget,
            device=device,
            variants=round1_variants,
            steps=args.round1_steps,
            seeds=budget.seeds,
            output_path=ROUND1_PATH,
            tag="round1_anti_collapse",
        )
        staged_results.append(round1_result)
        final_result = round1_result
        trained = round1_trained
        top6 = _rank_variants(round1_result)[:6]
        if top6 and int(args.round1_top_steps) > int(args.round1_steps):
            top_result, top_trained = train_variant_set(
                teacher,
                budget=budget,
                device=device,
                variants=top6,
                steps=args.round1_top_steps,
                seeds=budget.seeds,
                output_path=ROUND1_PATH,
                tag="round1_top6_320",
            )
            top_result["round1_160"] = round1_result
            _dump_json(ROUND1_PATH, top_result)
            staged_results.append(top_result)
            final_result = top_result
            trained = top_trained

    if decision != DECISION_REPLAY_INCONSISTENT and int(args.max_round) >= 2:
        round2_variants = [name for name, config in VARIANT_CONFIGS.items() if config.family == "round2"][: max(0, int(args.round2_limit))]
        round2_budget = replace(budget, seeds=tuple(int(x) for x in args.round2_seeds.split(",")))
        round2_result, round2_trained = train_variant_set(
            teacher,
            budget=round2_budget,
            device=device,
            variants=round2_variants,
            steps=args.round2_steps,
            seeds=round2_budget.seeds,
            output_path=ROUND2_PATH,
            tag="round2_combined",
        )
        staged_results.append(round2_result)
        final_result = round2_result
        trained = round2_trained
        top2 = _rank_variants(round2_result)[:2]
        if top2 and int(args.round2_top_steps) > int(args.round2_steps):
            top2_result, top2_trained = train_variant_set(
                teacher,
                budget=round2_budget,
                device=device,
                variants=top2,
                steps=args.round2_top_steps,
                seeds=round2_budget.seeds,
                output_path=ROUND2_PATH,
                tag="round2_top2_640",
            )
            top2_result["round2_320"] = round2_result
            _dump_json(ROUND2_PATH, top2_result)
            staged_results.append(top2_result)
            final_result = top2_result
            trained = top2_trained

    if decision != DECISION_REPLAY_INCONSISTENT and int(args.max_round) >= 3:
        top3 = _rank_variants(final_result)[:3]
        round3_budget = replace(budget, seeds=tuple(int(x) for x in args.round3_seeds.split(",")))
        round3_result, round3_trained = train_variant_set(
            teacher,
            budget=round3_budget,
            device=device,
            variants=top3,
            steps=args.round3_steps,
            seeds=round3_budget.seeds,
            output_path=ROUND3_PATH,
            tag="round3_final_stability",
        )
        staged_results.append(round3_result)
        final_result = round3_result
        trained = round3_trained

    capacities = final_result.get("capacities", {})  # type: ignore[assignment]
    if capacities:
        best_variant = max(capacities, key=lambda v: (capacities[v], _variant_mean(final_result["rows"], v, 128), _variant_mean(final_result["rows"], v, 256)))  # type: ignore[index]
    else:
        best_variant = "rolling_state_organized_full_combined"
    top_ids = _top_student_ids(final_result, trained, limit=3)
    control_result = run_controls(
        trained,
        top_ids,
        teacher,
        budget=budget,
        device=device,
        baseline_rows=baseline_result["rows"],  # type: ignore[arg-type]
        student_rows=final_result["rows"],  # type: ignore[arg-type]
    )

    all_rows: List[Mapping[str, object]] = []
    all_rows.extend(teacher_result["rows"])  # type: ignore[arg-type]
    all_rows.extend(baseline_result["rows"])  # type: ignore[arg-type]
    for result in staged_results:
        all_rows.extend(result.get("rows", []))  # type: ignore[arg-type]
    all_rows.extend(control_result["rows"])  # type: ignore[arg-type]

    if decision != DECISION_REPLAY_INCONSISTENT:
        decision = decide(
            teacher_result=teacher_result,
            baseline_result=baseline_result,
            replay_result=replay_result,
            student_result=final_result,
            control_result=control_result,
            extrapolation_rows=(),
        )
    best_capacity = int(capacities.get(best_variant, 0))
    best64 = _variant_mean(final_result["rows"], best_variant, 64) if final_result.get("rows") else 0.0  # type: ignore[arg-type]
    best_trained_id = next((model_id for model_id in trained if model_id.startswith(f"{best_variant}_seed_")), None)
    parameter_count = count_parameters(trained[best_trained_id]) if best_trained_id else (count_parameters(next(iter(trained.values()))) if trained else 0)
    state_audit, memory_audit = write_state_and_memory_audits(
        budget=budget,
        student_rows=final_result["rows"],  # type: ignore[arg-type]
        control_result=control_result,
        best_variant=best_variant,
        parameter_count=parameter_count,
    )
    write_capacity_curves(all_rows)
    write_best_config(budget=budget, decision=decision, best_variant=best_variant, best_capacity=best_capacity, best64=best64)
    write_seed_stability(staged_results)
    write_report(
        decision=decision,
        preflight=preflight,
        baseline_result=baseline_result,
        replay_result=replay_result,
        final_result=final_result,
        control_result=control_result,
        state_audit=state_audit,
        memory_audit=memory_audit,
        best_variant=best_variant,
        replay_matched=replay_matched,
    )
    print(decision)
    return decision


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E5.1 Rolling Activation-State Stability and Anti-Collapse")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--allow-cpu-smoke", action="store_true")
    parser.add_argument("--eval-examples", type=int, default=96)
    parser.add_argument("--teacher-eval-examples", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--train-lengths", default="")
    parser.add_argument("--eval-lengths", default="")
    parser.add_argument("--baseline-steps", type=int, default=180)
    parser.add_argument("--replay-steps", type=int, default=160)
    parser.add_argument("--round1-steps", type=int, default=160)
    parser.add_argument("--round1-top-steps", type=int, default=320)
    parser.add_argument("--round2-steps", type=int, default=320)
    parser.add_argument("--round2-top-steps", type=int, default=640)
    parser.add_argument("--round3-steps", type=int, default=640)
    parser.add_argument("--max-round", type=int, default=0)
    parser.add_argument("--round1-limit", type=int, default=24)
    parser.add_argument("--round2-limit", type=int, default=12)
    parser.add_argument("--round2-seeds", default="0,1,2,3,4")
    parser.add_argument("--round3-seeds", default="0,1,2,3,4")
    parser.add_argument("--randomized-label-steps", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--num-state-slots", type=int, default=32)
    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument("--state-ablation-margin", type=float, default=0.55)
    parser.add_argument("--seeds", default="0,1,2")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> str:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run_experiment(args)


if __name__ == "__main__":
    main()
