from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
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

from src.datasets.latent_attention_capacity_dataset import (
    TASK_FAMILIES as STAGE8_TASK_FAMILIES,
    Stage8DatasetConfig,
    Stage8Example,
    build_stage8_examples,
    validate_stage8_examples,
)
from src.experiments.stage8_architecture_registry import architecture_config_from_dict, architecture_config_to_dict
from src.experiments.stage8_controls import REQUIRED_STAGE8_CONTROLS, apply_stage8_control, control_expectation, shortcut_audit
from src.models.latent_attention_variants import (
    Stage8ArchitectureConfig,
    Stage8HashFeaturizer,
    build_stage8_selector,
    count_parameters,
    estimate_stage8_compute,
    _any_token_hit,
    _block_token_row,
    _candidate_identity_tokens,
    _clean_qkv_feature_tensor,
    _content_tokens,
    _distribution_entropy,
    _rank_for_claim,
    _source_rank_map,
    _stage8b3_feature_tensor,
    _stable_hash,
)


E2_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = E2_ROOT / "results"
REPORTS_DIR = E2_ROOT / "reports"

PREFIX = "e2_attention_synthesis"
PREFLIGHT_JSON = RESULTS_DIR / f"{PREFIX}_preflight.json"
PREFLIGHT_REPORT = REPORTS_DIR / "E2_ATTENTION_SYNTHESIS_PREFLIGHT.md"
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
VIEW_MEMORY_AUDIT_PATH = RESULTS_DIR / f"{PREFIX}_view_memory_audit.json"
WDA_AUDIT_PATH = RESULTS_DIR / f"{PREFIX}_wda_audit.json"
TREALIZED_AUDIT_PATH = RESULTS_DIR / f"{PREFIX}_trealized_audit.json"
COMPUTE_AUDIT_PATH = RESULTS_DIR / f"{PREFIX}_compute_audit.json"
CAPACITY_CURVES_PATH = RESULTS_DIR / f"{PREFIX}_capacity_curves.csv"
CAMPAIGN_REPORT = REPORTS_DIR / "E2_ATTENTION_SYNTHESIS_CAMPAIGN.md"

STAGE8B4_FREEZE_RECOMMENDATION = RESULTS_DIR / "stage8b4_freeze_recommendation.json"
STAGE8B3_FREEZE_MANIFEST = RESULTS_DIR / "stage8b3_top3_freeze_manifest.json"
STAGE8_BASELINE_CAPACITY = RESULTS_DIR / "stage8_baseline_capacity.json"

DECISION_CUDA_BLOCKED = "CUDA_REQUIRED_NOT_AVAILABLE"
DECISION_TOP3 = "E2_TOP3_SELECTED"
DECISION_NEW_BEATS_CHAMPION = "E2_NEW_ARCHITECTURE_BEATS_CHAMPION"
DECISION_CHAMPION_REMAINS_BEST = "E2_CHAMPION_REMAINS_BEST"
DECISION_CLEANQKV_REPAIRED = "E2_CLEANQKV_REPAIRED_COMPETITIVE"
DECISION_DISTRIBUTIONAL_PROMISING = "E2_DISTRIBUTIONAL_ATTENTION_PROMISING"
DECISION_NO_PROMISING = "E2_NO_PROMISING_NEW_ARCHITECTURE"

E2_TASK_FAMILIES: Tuple[str, ...] = (
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

ROUND1_CONTROLS: Tuple[str, ...] = (
    "randomized_labels",
    "candidate_evidence_mismatch",
    "cross_task_evidence_shuffle",
    "candidate_only",
    "query_only",
    "hidden_state_shuffle",
)

BASE_FULL_CONTROLS: Tuple[str, ...] = (
    "randomized_labels",
    "candidate_order_shuffle_with_label_remap",
    "evidence_block_order_shuffle",
    "candidate_evidence_mismatch",
    "cross_task_evidence_shuffle",
    "cross_task_query_shuffle",
    "candidate_only",
    "query_only",
    "evidence_only",
    "distractor_only",
    "schema_template_only",
    "frozen_same_architecture_comparator",
    "hidden_state_shuffle",
    "role_permutation",
    "avenue_permutation",
    "memory_slot_permutation",
    "memory_disabled",
    "memory_shuffle_across_examples",
)

MEMORY_GATE_CONTROLS: Tuple[str, ...] = (
    "memory_gate_forced_open",
    "memory_gate_forced_closed",
)

WDA_CONTROLS: Tuple[str, ...] = (
    "wda_fixed_coordinates",
    "wda_random_coordinates",
    "wda_shuffled_coordinates",
    "wda_no_distribution_scale",
)

DAT_CONTROLS: Tuple[str, ...] = (
    "dat_sigma_zero",
    "dat_fixed_point",
    "dat_shuffled_point",
    "dat_no_distribution",
)

TREALIZED_CONTROLS: Tuple[str, ...] = (
    "trealized_t_zero",
    "trealized_t_shuffle",
    "trealized_t_random",
    "trealized_fixed_normal_attention",
)

E2_DEGRADE_CONTROLS = set(
    (
        "wda_fixed_coordinates",
        "wda_random_coordinates",
        "wda_shuffled_coordinates",
        "wda_no_distribution_scale",
        "dat_sigma_zero",
        "dat_fixed_point",
        "dat_shuffled_point",
        "dat_no_distribution",
        "trealized_t_zero",
        "trealized_t_shuffle",
        "trealized_t_random",
    )
)

E2_DIAGNOSTIC_CONTROLS = {
    "frozen_same_architecture_comparator",
    "trealized_fixed_normal_attention",
}

E2_FEATURE_DIM = 16 + 18 + 18
STAGE8B4_CHAMPION_NAME = "stage8b3_j_explicit_view_objective_assignment_03"


class CudaRequiredError(RuntimeError):
    pass


@dataclass(frozen=True)
class E2Budget:
    round1_variants: int = 80
    round2_promoted: int = 24
    round3_parent_count: int = 12
    round3_mutations_per_parent: int = 3
    round4_finalists: int = 8
    train_examples: int = 48
    eval_examples: int = 48
    round1_n: Tuple[int, ...] = (8, 16, 32)
    round2_n: Tuple[int, ...] = (16, 32, 64)
    round3_n: Tuple[int, ...] = (32, 64, 128)
    round4_n: Tuple[int, ...] = (32, 64, 128, 256)
    round1_seeds: Tuple[int, ...] = (0,)
    round2_seeds: Tuple[int, ...] = (0, 1, 2)
    round3_seeds: Tuple[int, ...] = (0, 1, 2)
    round4_seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    round1_epochs: int = 2
    promoted_epochs: int = 4
    lr: float = 0.05


@dataclass(frozen=True)
class E2ArchitectureConfig:
    name: str
    family_code: str
    family_name: str
    stage8_config: Stage8ArchitectureConfig
    mechanisms: Tuple[str, ...] = ()
    parent_config_id: str | None = None
    mutation_description: str = "initial"
    wda_scope: str = "none"
    dat_mode: str = "none"
    trealized_mode: str = "none"
    activation_cache: bool = False
    support_contradiction: bool = False
    hierarchical: bool = False
    typed_memory_slots: bool = False
    cleanqkv_v2: bool = False
    champion_reference: bool = False
    cleanqkv_v1_replay: bool = False
    frozen: bool = False

    @property
    def config_id(self) -> str:
        payload = json.dumps(e2_config_to_dict(self, include_id=False), sort_keys=True).encode("utf-8")
        return hashlib.blake2b(payload, digest_size=8).hexdigest()


class WDAAdapter(nn.Module):
    """Distribution-coordinate adapter that realizes per-example effective weights."""

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
        weight = self.effective_weight(context)
        return torch.einsum("bi,boi->bo", x, weight)


class DATDistributionalMemorySlots(nn.Module):
    """DAT-style memory slots with mu/sigma and candidate-selected realization points."""

    def __init__(self, num_slots: int, slot_dim: int, context_dim: int) -> None:
        super().__init__()
        self.mu = nn.Parameter(torch.randn(num_slots, slot_dim) * 0.02)
        self.log_sigma = nn.Parameter(torch.full((num_slots, slot_dim), -2.0))
        self.attn = nn.Linear(context_dim, num_slots)
        self.point = nn.Linear(context_dim, slot_dim)

    def realize(
        self,
        context: torch.Tensor,
        *,
        sigma_zero: bool = False,
        fixed_point: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.attn(context), dim=-1)
        point = torch.zeros_like(self.point(context)) if fixed_point else torch.tanh(self.point(context))
        sigma = torch.zeros_like(self.log_sigma).exp() if sigma_zero else F.softplus(self.log_sigma)
        if sigma_zero:
            sigma = torch.zeros_like(sigma)
        slots = self.mu.unsqueeze(0) + point.unsqueeze(1) * sigma.unsqueeze(0)
        realized = torch.einsum("bs,bsd->bd", weights, slots)
        return realized, weights, point


class TRealizedCandidateMemoryAttention(nn.Module):
    """Candidate-memory attention where pairwise T realizes temporary K/V tensors."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value_delta = nn.Linear(dim, dim, bias=False)
        self.t_head = nn.Linear(dim * 2, 1)

    def forward(
        self,
        candidate: torch.Tensor,
        memory: torch.Tensor,
        *,
        t_override: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if candidate.dim() == 1:
            candidate = candidate.unsqueeze(0)
        query = self.query(candidate)
        keys = self.key(memory)
        pair = torch.cat([candidate.unsqueeze(1).expand(-1, memory.shape[0], -1), memory.unsqueeze(0).expand(candidate.shape[0], -1, -1)], dim=-1)
        t_values = torch.tanh(self.t_head(pair)).squeeze(-1)
        if t_override is not None:
            t_values = t_override.to(device=t_values.device, dtype=t_values.dtype)
        realized_keys = keys.unsqueeze(0) + t_values.unsqueeze(-1) * self.value_delta(memory).unsqueeze(0)
        logits = torch.einsum("bd,bsd->bs", query, realized_keys) / math.sqrt(max(1, memory.shape[-1]))
        weights = torch.softmax(logits, dim=-1)
        values = memory.unsqueeze(0) + t_values.unsqueeze(-1) * self.value_delta(memory).unsqueeze(0)
        return torch.einsum("bs,bsd->bd", weights, values), t_values


class CleanQKVActivationCacheV2(nn.Module):
    """Clean-QKV v2 cache: current self-attention stays query/candidate/current-only."""

    def __init__(self, vector_dim: int = 32, slots: int = 8) -> None:
        super().__init__()
        self.vector_dim = int(vector_dim)
        self.slots = int(slots)
        self.writer = nn.Linear(vector_dim, vector_dim)
        self.reader_gate = nn.Linear(vector_dim * 2, 1)
        self.force_gate: str | None = None

    def current_self_attention_inputs(self, query_tokens: Sequence[str], candidate_tokens: Sequence[str], current_tokens: Sequence[str]) -> Tuple[str, ...]:
        return tuple(query_tokens) + tuple(candidate_tokens) + tuple(current_tokens)

    def write_cache(self, evidence_activations: torch.Tensor) -> torch.Tensor:
        if evidence_activations.numel() == 0:
            return evidence_activations.reshape(0, self.vector_dim)
        projected = self.writer(evidence_activations)
        if projected.shape[0] <= self.slots:
            return projected
        chunks = torch.chunk(projected, self.slots, dim=0)
        return torch.stack([chunk.mean(dim=0) for chunk in chunks])

    def gate(self, current: torch.Tensor, memory_read: torch.Tensor) -> torch.Tensor:
        if self.force_gate == "closed":
            return torch.zeros(current.shape[0], 1, device=current.device, dtype=current.dtype)
        if self.force_gate == "open":
            return torch.ones(current.shape[0], 1, device=current.device, dtype=current.dtype)
        return torch.sigmoid(self.reader_gate(torch.cat([current, memory_read], dim=-1)))

    def set_gate_control(self, mode: str | None) -> None:
        if mode not in {None, "open", "closed"}:
            raise ValueError(f"unknown gate control {mode}")
        self.force_gate = mode


class E2FeatureSelector:
    supports_training = True

    def __init__(self, config: E2ArchitectureConfig, seed: int, device: torch.device, epochs: int, lr: float) -> None:
        self.config = config
        self.seed = int(seed)
        self.device = device
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.featurizer = Stage8HashFeaturizer(config.stage8_config.vector_dim)
        torch.manual_seed(seed)
        self.module = nn.Linear(E2_FEATURE_DIM, 1).to(device)
        nn.init.zeros_(self.module.bias)
        if config.champion_reference:
            with torch.no_grad():
                self.module.weight.zero_()
                self.module.weight[0, 0] = 2.0
                self.module.weight[0, 1] = 2.0
                self.module.weight[0, 2] = -3.0
                self.module.weight[0, 3] = 2.0
                self.module.weight[0, 5] = 0.6
                self.module.weight[0, 8] = 1.3
                self.module.weight[0, 10] = 0.3
        if config.frozen:
            for parameter in self.module.parameters():
                parameter.requires_grad_(False)
        self.training_trace: List[Dict[str, float]] = []

    def fit(self, train_examples: Sequence[Stage8Example], dev_examples: Sequence[Stage8Example] | None = None) -> "E2FeatureSelector":
        del dev_examples
        if self.config.frozen or self.epochs <= 0:
            self._assert_module_on_device()
            return self
        self._assert_cuda_if_required()
        rows = list(train_examples)[: self.config.stage8_config.max_train_examples]
        optimizer = torch.optim.AdamW(self.module.parameters(), lr=self.lr, weight_decay=self.config.stage8_config.weight_decay)
        for epoch in range(self.epochs):
            random.Random(_stable_hash(self.seed, self.config.config_id, "e2", epoch)).shuffle(rows)
            losses: List[float] = []
            for example in rows:
                if example.label < 0:
                    continue
                features = e2_feature_tensor(example, self.config, self.featurizer, self.seed, training=True).to(self.device)
                logits = self.module(features).squeeze(-1)
                label = torch.tensor([example.label], dtype=torch.long, device=self.device)
                loss = F.cross_entropy(logits.unsqueeze(0), label)
                loss = loss + self._regularization(example, features)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.module.parameters(), 2.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            self.training_trace.append({"epoch": float(epoch), "loss": float(sum(losses) / max(1, len(losses)))})
        return self

    def scores(self, examples: Sequence[Stage8Example]) -> List[List[float]]:
        self._assert_module_on_device()
        with torch.no_grad():
            rows = []
            for example in examples:
                features = e2_feature_tensor(example, self.config, self.featurizer, self.seed, training=False).to(self.device)
                rows.append(self.module(features).squeeze(-1).detach().cpu().tolist())
            return rows

    def predict(self, examples: Sequence[Stage8Example]) -> List[int]:
        return [int(max(range(len(row)), key=lambda index: row[index])) for row in self.scores(examples)]

    def diagnostics(self, examples: Sequence[Stage8Example]) -> Dict[str, object]:
        if not examples:
            return {"training_trace": self.training_trace, "parameter_count": count_parameters(self.module)}
        role_counts = [0 for _ in range(max(1, self.config.stage8_config.roles))]
        avenue_counts = [0 for _ in range(max(1, self.config.stage8_config.avenues))]
        routing_entropies: List[float] = []
        pairwise_sims: List[float] = []
        memory_entropies: List[float] = []
        gate_values: List[float] = []
        relevant_gate_values: List[float] = []
        irrelevant_gate_values: List[float] = []
        wda_coords: List[float] = []
        t_values: List[float] = []
        dat_sigmas: List[float] = []
        with torch.no_grad():
            for example in examples:
                features = e2_feature_tensor(example, self.config, self.featurizer, self.seed, training=False)
                scores = self.module(features.to(self.device)).squeeze(-1).detach().cpu()
                predicted = int(torch.argmax(scores).item())
                weights = _e2_view_weights(example, self.config, predicted, self.seed)
                routing_entropies.append(_normalized_entropy(weights.tolist()))
                top_view = int(torch.argmax(weights).item()) if weights.numel() else 0
                role_counts[top_view // max(1, self.config.stage8_config.avenues) % len(role_counts)] += 1
                avenue_counts[top_view % len(avenue_counts)] += 1
                pairwise_sims.append(_e2_pairwise_similarity(example, self.config, self.seed))
                metrics = _e2_candidate_metrics(example, predicted, self.config, self.seed)
                memory_entropies.append(float(metrics["memory_slot_entropy"]))
                gate = float(metrics["memory_gate"])
                gate_values.append(gate)
                if predicted == example.label:
                    relevant_gate_values.append(gate)
                else:
                    irrelevant_gate_values.append(gate)
                wda_coords.extend(float(value) for value in metrics["wda_coordinates"])
                t_values.extend(float(value) for value in metrics["t_values"])
                dat_sigmas.append(float(metrics["dat_sigma_mean"]))
        total = max(1, len(examples))
        role_dist = [count / total for count in role_counts]
        avenue_dist = [count / total for count in avenue_counts]
        gate_mean = _mean(gate_values)
        gate_relevant = _mean(relevant_gate_values)
        gate_irrelevant = _mean(irrelevant_gate_values)
        return {
            "training_trace": self.training_trace,
            "parameter_count": count_parameters(self.module),
            "device": str(self.device),
            "clean_qkv_current_self_attention": "candidate_query_current_only_no_evidence_history_tokens"
            if self.config.cleanqkv_v2 or self.config.activation_cache
            else "not_applicable",
            "activation_cache_guidance_only": bool(self.config.cleanqkv_v2 or self.config.activation_cache),
            "routing_entropy_mean": _mean(routing_entropies),
            "role_usage_distribution": role_dist,
            "avenue_usage_distribution": avenue_dist,
            "role_entropy": _normalized_entropy(role_dist),
            "avenue_entropy": _normalized_entropy(avenue_dist),
            "pairwise_hidden_state_similarity": _mean(pairwise_sims),
            "memory_slot_entropy_mean": _mean(memory_entropies),
            "memory_slot_usage_distribution": _e2_memory_slot_distribution(examples, self.config, self.seed),
            "memory_gate_mean": gate_mean,
            "memory_gate_relevant_mean": gate_relevant,
            "memory_gate_irrelevant_mean": gate_irrelevant,
            "memory_gate_selectivity": gate_relevant - gate_irrelevant,
            "wda_coordinate_variance": _variance(wda_coords),
            "wda_coordinate_diversity": len({round(value, 3) for value in wda_coords}) / max(1.0, float(len(wda_coords))),
            "t_realized_t_variance": _variance(t_values),
            "dat_sigma_mean": _mean(dat_sigmas),
        }

    def _regularization(self, example: Stage8Example, features: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, dtype=features.dtype, device=features.device)
        if self.config.cleanqkv_v2:
            gate = features[:, 34]
            target = torch.zeros_like(gate)
            if example.label >= 0:
                target[example.label] = 1.0
            loss = loss + 0.01 * F.mse_loss(gate, target)
        if "anti_collapse" in self.config.mechanisms:
            spread = features[:, :16].std(dim=0).mean()
            loss = loss - 0.001 * spread
        return loss

    def _assert_module_on_device(self) -> None:
        parameter = next(self.module.parameters())
        if parameter.device.type != self.device.type:
            raise RuntimeError(f"module is on {parameter.device}, expected {self.device}")

    def _assert_cuda_if_required(self) -> None:
        self._assert_module_on_device()
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise CudaRequiredError(DECISION_CUDA_BLOCKED)
        if self.device.type == "cuda":
            for parameter in self.module.parameters():
                if parameter.device.type != "cuda":
                    raise RuntimeError("real training attempted with non-CUDA parameter")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run E2 empirical attention architecture synthesis campaign.")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--smoke", action="store_true", help="Tiny CPU-allowed smoke run for tests only.")
    parser.add_argument("--train-examples", type=int, default=48)
    parser.add_argument("--eval-examples", type=int, default=48)
    parser.add_argument("--round1-variants", type=int, default=80)
    parser.add_argument("--max-rounds", type=int, default=4)
    args = parser.parse_args(argv)

    if args.smoke:
        budget = E2Budget(
            round1_variants=10,
            round2_promoted=4,
            round3_parent_count=2,
            round3_mutations_per_parent=1,
            round4_finalists=2,
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
        payload = run_e2_campaign(budget, device=args.device, max_rounds=args.max_rounds, allow_cpu_smoke=True)
    else:
        budget = E2Budget(train_examples=args.train_examples, eval_examples=args.eval_examples, round1_variants=max(80, args.round1_variants))
        payload = run_e2_campaign(budget, device=args.device, max_rounds=args.max_rounds, allow_cpu_smoke=False)
    print(payload["decision"])


def run_e2_campaign(
    budget: E2Budget | None = None,
    *,
    device: str = "cuda",
    max_rounds: int = 4,
    allow_cpu_smoke: bool = False,
) -> Dict[str, object]:
    budget = budget or E2Budget()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _reset_e2_outputs()
    preflight = write_preflight(device=device, allow_cpu_smoke=allow_cpu_smoke)
    if preflight["decision"] == DECISION_CUDA_BLOCKED:
        payload = _cuda_blocked_payload(preflight, budget)
        _write_blocker_report(payload)
        return payload

    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch_device)
    started = time.perf_counter()
    baseline_capacity = _load_stage8_baseline_capacity()
    champion_prior = _load_stage8b4_champion_prior()
    variants = generate_e2_variants(max_variants=budget.round1_variants)

    round0 = _round0_sanity(budget, torch_device)
    _dump_json(ROUND0_PATH, round0)
    if not round0["passes"] or max_rounds <= 0:
        payload = _terminal_payload(
            decision=DECISION_NO_PROMISING,
            budget=budget,
            preflight=preflight,
            champion_prior=champion_prior,
            baseline_capacity=baseline_capacity,
            round0=round0,
            round1={},
            round2={},
            round3={},
            round4={},
            top3=[],
            all_rows=[],
            started=started,
        )
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
        baseline_capacity,
    )
    round1_baselines = _run_required_baselines("round1", budget.round1_n, budget.round1_seeds, budget, torch_device, budget.round1_epochs)
    round1_survivors = [row for row in round1_rows if row["screen_checks"]["round1_pass"]]
    round1_payload = {
        "round": 1,
        "screen": "wide_screen",
        "required_variants": 80,
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
        top3 = _select_top3(round1_survivors)
        decision = _decision(top3, round1_survivors, champion_prior)
        payload = _terminal_payload(decision, budget, preflight, champion_prior, baseline_capacity, round0, round1_payload, {}, {}, {}, top3, round1_rows, started)
        _write_terminal_artifacts(payload, top3, round1_rows)
        return payload

    promoted = _variants_from_rows(_ensure_champion_present(round1_survivors[: budget.round2_promoted], round1_rows), variants)
    round2_rows = _evaluate_variants(
        "round2",
        promoted,
        budget.round2_n,
        budget.round2_seeds,
        (),
        budget,
        torch_device,
        budget.promoted_epochs,
        baseline_capacity,
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
        top3 = _select_top3(round2_survivors or round2_rows)
        decision = _decision(top3, round2_rows, champion_prior)
        payload = _terminal_payload(decision, budget, preflight, champion_prior, baseline_capacity, round0, round1_payload, round2_payload, {}, {}, top3, round1_rows + round2_rows, started)
        _write_terminal_artifacts(payload, top3, round1_rows + round2_rows)
        return payload

    parents = _variants_from_rows(_ensure_champion_present(round2_survivors[: budget.round3_parent_count], round2_rows), promoted)
    mutations: List[E2ArchitectureConfig] = []
    for parent in parents:
        mutations.extend(mutate_e2_variant(parent, budget.round3_mutations_per_parent))
    round3_rows = _evaluate_variants(
        "round3",
        mutations,
        budget.round3_n,
        budget.round3_seeds,
        (),
        budget,
        torch_device,
        budget.promoted_epochs,
        baseline_capacity,
    )
    parent_by_id = {row["config_id"]: row for row in round2_rows}
    round3_kept = [row for row in round3_rows if _mutation_improved(row, parent_by_id)]
    round3_payload = {
        "round": 3,
        "screen": "hybrid_mutation",
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
        top3 = _select_top3(round2_rows + round3_kept)
        decision = _decision(top3, round2_rows + round3_rows, champion_prior)
        all_rows = round1_rows + round2_rows + round3_rows
        payload = _terminal_payload(decision, budget, preflight, champion_prior, baseline_capacity, round0, round1_payload, round2_payload, round3_payload, {}, top3, all_rows, started)
        _write_terminal_artifacts(payload, top3, all_rows)
        return payload

    finalist_seed_rows = _ensure_champion_present(_rank_rows(round2_survivors + round3_kept)[: budget.round4_finalists], round2_rows + round3_rows)
    finalists = _variants_from_rows(finalist_seed_rows, promoted + mutations + variants)
    round4_rows = _evaluate_variants(
        "round4",
        finalists,
        budget.round4_n,
        budget.round4_seeds,
        (),
        budget,
        torch_device,
        budget.promoted_epochs,
        baseline_capacity,
    )
    ranked_finalists = _rank_rows(round4_rows)
    top3 = _select_top3(ranked_finalists)
    round4_payload = {
        "round": 4,
        "screen": "final_development_screen",
        "evaluated_count": len(round4_rows),
        "finalists": _brief_rows(ranked_finalists),
        "top3": _brief_rows(top3),
        "n_schedule": list(budget.round4_n),
        "seeds": list(budget.round4_seeds),
    }
    _dump_json(ROUND4_PATH, round4_payload)

    all_rows = round1_rows + round2_rows + round3_rows + round4_rows
    decision = _decision(top3, ranked_finalists, champion_prior)
    payload = _terminal_payload(
        decision,
        budget,
        preflight,
        champion_prior,
        baseline_capacity,
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


def write_preflight(*, device: str = "cuda", allow_cpu_smoke: bool = False) -> Dict[str, object]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    status = cuda_preflight_status(device=device, allow_cpu_smoke=allow_cpu_smoke)
    visible_files = sorted(path.name for path in E2_ROOT.iterdir())
    previous_results = sorted(path.name for path in RESULTS_DIR.iterdir()) if RESULTS_DIR.exists() else []
    existing_e2 = [name for name in previous_results if name.startswith(PREFIX)]
    payload = {
        "stage": "E2",
        "campaign": "Empirical Attention Architecture Synthesis",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "os": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "requested_device": device,
        "resolved_device": status["resolved_device"],
        "working_directory": str(E2_ROOT),
        "visible_files": visible_files,
        "previous_result_files_exist": bool(previous_results),
        "previous_result_files_count": len(previous_results),
        "previous_e2_attention_synthesis_files": existing_e2,
        "stage8_artifacts_present": any(name.startswith("stage8") for name in previous_results),
        "stage8b3_artifacts_present": any(name.startswith("stage8b3") for name in previous_results),
        "stage8b4_artifacts_present": any(name.startswith("stage8b4") for name in previous_results),
        "no_stage8c_run": True,
        "no_10x_attention_capacity_claim": True,
        "decision": status["decision"],
    }
    _dump_json(PREFLIGHT_JSON, payload)
    PREFLIGHT_REPORT.write_text(_preflight_markdown(payload), encoding="utf-8")
    if payload["gpu_name"]:
        print(f"CUDA device: {payload['gpu_name']}")
    return payload


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


def build_e2_examples(
    *,
    n_examples: int,
    n_blocks: int,
    split: str = "train",
    template_split: str = "train",
    seed: int = 0,
    k_candidates: int = 8,
    task_families: Sequence[str] = E2_TASK_FAMILIES,
) -> List[Stage8Example]:
    if k_candidates != 8:
        raise ValueError("E2 benchmark uses K=8 candidates")
    families = tuple(task_families)
    unknown = sorted(set(families) - set(E2_TASK_FAMILIES))
    if unknown:
        raise ValueError(f"unknown E2 task families: {unknown}")
    examples: List[Stage8Example] = []
    for index in range(n_examples):
        family = families[index % len(families)]
        family_seed = _stable_hash("e2-dataset", seed, split, template_split, family, n_blocks, index)
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
            examples.append(replace(row, example_id=f"{split}-{template_split}-N{n_blocks}-{index:05d}-{family}", split=split))
        else:
            examples.append(_build_e2_extra_family(family, n_blocks, split, template_split, index, family_seed, k_candidates))
    random.Random(_stable_hash("e2-shuffle", seed, split, template_split, n_blocks, n_examples)).shuffle(examples)
    return [
        replace(example, example_id=f"{split}-{template_split}-N{n_blocks}-{i:05d}-{example.task_family}")
        for i, example in enumerate(examples)
    ]


def validate_e2_examples(examples: Sequence[Stage8Example]) -> Dict[str, object]:
    audit = validate_stage8_examples(examples)
    families = {example.task_family for example in examples}
    failures = list(audit["failures"])
    if not families.issubset(set(E2_TASK_FAMILIES)):
        failures.append("unknown E2 task family present")
    if examples and not families.intersection(set(E2_TASK_FAMILIES) - set(STAGE8_TASK_FAMILIES)):
        failures.append("no E2-only task family present")
    return {**audit, "passes": not failures, "failures": failures, "e2_task_families": sorted(families)}


def generate_e2_variants(max_variants: int = 80) -> List[E2ArchitectureConfig]:
    champion = load_stage8b4_champion_config()
    variants: List[E2ArchitectureConfig] = [
        E2ArchitectureConfig(
            name=champion.name,
            family_code="A",
            family_name="Stage 8 Champion Replay",
            stage8_config=champion,
            mechanisms=("explicit_views", "memory_slots"),
            mutation_description="exact Stage 8B.4 champion config replay; no architecture modification",
            champion_reference=True,
        )
    ]
    for index in range(8):
        variants.append(_family_b_variant(index))
        variants.append(_family_c_variant(index))
        variants.append(_family_d_variant(index))
        variants.append(_family_e_variant(index))
        variants.append(_family_f_variant(index))
        variants.append(_family_g_variant(index))
        variants.append(_family_h_variant(index))
        variants.append(_family_i_variant(index))
    for index in range(max(15, max_variants - len(variants))):
        variants.append(_family_j_variant(index, champion))
    unique: Dict[str, E2ArchitectureConfig] = {}
    for variant in variants:
        unique[variant.config_id] = variant
    rows = list(unique.values())
    if len(rows) < max_variants:
        raise RuntimeError(f"generated only {len(rows)} E2 variants")
    return rows[:max_variants]


def mutate_e2_variant(parent: E2ArchitectureConfig, count: int = 3) -> List[E2ArchitectureConfig]:
    mutations = [
        ("add WDA to memory reader", {"mechanisms": tuple(sorted(set(parent.mechanisms + ("wda",)))), "wda_scope": "memory_reader"}),
        ("add T-realized memory K/V", {"mechanisms": tuple(sorted(set(parent.mechanisms + ("t_realized",)))), "trealized_mode": "k_v"}),
        ("add support/contradiction split", {"mechanisms": tuple(sorted(set(parent.mechanisms + ("support_contradiction",)))), "support_contradiction": True}),
        ("add activation-cache gate supervision", {"mechanisms": tuple(sorted(set(parent.mechanisms + ("activation_cache",)))), "activation_cache": True, "cleanqkv_v2": True}),
        ("add typed memory slots", {"mechanisms": tuple(sorted(set(parent.mechanisms + ("typed_memory_slots",)))), "typed_memory_slots": True}),
        ("add DAT memory realization", {"mechanisms": tuple(sorted(set(parent.mechanisms + ("dat",)))), "dat_mode": "mu_sigma_point"}),
        ("add hierarchical routing", {"mechanisms": tuple(sorted(set(parent.mechanisms + ("hierarchical",)))), "hierarchical": True}),
    ]
    output: List[E2ArchitectureConfig] = []
    for index, (description, values) in enumerate(mutations[: max(1, count)]):
        stage = replace(
            parent.stage8_config,
            name=f"{parent.stage8_config.name}_e2_mut{index}",
            coord_hops=max(parent.stage8_config.coord_hops, 2 if "hierarchical" in str(values) else parent.stage8_config.coord_hops),
        )
        output.append(
            replace(
                parent,
                name=f"{parent.name}_mut{index}",
                stage8_config=stage,
                parent_config_id=parent.config_id,
                mutation_description=description,
                **values,
            )
        )
    return output


def load_stage8b4_champion_config() -> Stage8ArchitectureConfig:
    if STAGE8B4_FREEZE_RECOMMENDATION.exists():
        payload = json.loads(STAGE8B4_FREEZE_RECOMMENDATION.read_text(encoding="utf-8"))
        selected = payload.get("selected_candidate", {})
        params = selected.get("architecture_parameters", {}) if isinstance(selected, Mapping) else {}
        if isinstance(params, Mapping) and params.get("name") == STAGE8B4_CHAMPION_NAME:
            return architecture_config_from_dict(dict(params))
    if STAGE8B3_FREEZE_MANIFEST.exists():
        payload = json.loads(STAGE8B3_FREEZE_MANIFEST.read_text(encoding="utf-8"))
        for row in payload.get("top3_full_configs", []):
            if isinstance(row, Mapping) and row.get("name") == STAGE8B4_CHAMPION_NAME:
                return architecture_config_from_dict(dict(row))
    raise FileNotFoundError("missing Stage 8B.4 champion config")


def e2_feature_tensor(
    example: Stage8Example,
    config: E2ArchitectureConfig,
    featurizer: Stage8HashFeaturizer,
    seed: int,
    *,
    training: bool = False,
) -> torch.Tensor:
    stage_features = _stage8b3_feature_tensor(example, config.stage8_config, seed, training=training)
    clean_features = _clean_qkv_feature_tensor(example, config.stage8_config, featurizer, seed, training=training)
    if not (config.cleanqkv_v2 or config.activation_cache or "activation_cache" in config.mechanisms):
        clean_features = torch.zeros_like(clean_features)
    extras = torch.stack([_e2_extra_feature_row(example, candidate_index, config, seed) for candidate_index in range(example.k_candidates)])
    return torch.cat([stage_features, clean_features, extras], dim=-1)


def apply_e2_control(examples: Sequence[Stage8Example], control: str, seed: int) -> List[Stage8Example]:
    if control == "frozen_same_architecture_comparator":
        return [_with_e2_metadata_flag(example, control, frozen_same_architecture_comparator=True) for example in examples]
    if control in REQUIRED_STAGE8_CONTROLS:
        return apply_stage8_control(examples, control, seed)
    if control in set(WDA_CONTROLS + DAT_CONTROLS + TREALIZED_CONTROLS):
        return [_with_e2_metadata_flag(example, control, **{control: True}) for example in examples]
    raise ValueError(f"unknown E2 control: {control}")


def e2_control_expectation(control: str) -> str:
    if control in E2_DEGRADE_CONTROLS:
        return "degrade"
    if control in E2_DIAGNOSTIC_CONTROLS:
        return "diagnostic"
    if control in REQUIRED_STAGE8_CONTROLS:
        return control_expectation(control)
    return "diagnostic"


def e2_config_to_dict(config: E2ArchitectureConfig, *, include_id: bool = True) -> Dict[str, object]:
    row = {
        "name": config.name,
        "family_code": config.family_code,
        "family_name": config.family_name,
        "stage8_config": architecture_config_to_dict(config.stage8_config),
        "mechanisms": list(config.mechanisms),
        "parent_config_id": config.parent_config_id,
        "mutation_description": config.mutation_description,
        "wda_scope": config.wda_scope,
        "dat_mode": config.dat_mode,
        "trealized_mode": config.trealized_mode,
        "activation_cache": config.activation_cache,
        "support_contradiction": config.support_contradiction,
        "hierarchical": config.hierarchical,
        "typed_memory_slots": config.typed_memory_slots,
        "cleanqkv_v2": config.cleanqkv_v2,
        "champion_reference": config.champion_reference,
        "cleanqkv_v1_replay": config.cleanqkv_v1_replay,
        "frozen": config.frozen,
    }
    if include_id:
        row["config_id"] = config.config_id
    return row


def _build_e2_extra_family(
    family: str,
    n_blocks: int,
    split: str,
    template_split: str,
    index: int,
    seed: int,
    k_candidates: int,
) -> Stage8Example:
    rng = random.Random(seed)
    namespace = f"e2_{template_split}"
    entity = f"{namespace}_entity_{index}_{rng.randrange(10000)}"
    relation = f"{namespace}_relation_{rng.randrange(1000)}"
    values = [f"{namespace}_value_{index}_{slot}_{rng.randrange(10000)}" for slot in range(k_candidates)]
    label = rng.randrange(k_candidates)
    answer = values[label]
    candidates = [f"choice {value}" for value in values]
    rng.shuffle(candidates)
    label = next(i for i, candidate in enumerate(candidates) if answer in candidate)
    relevant: List[str] = []
    metadata: Dict[str, object] = {
        "namespace": namespace,
        "template_split": template_split,
        "candidate_order_randomized": True,
        "answer_token": answer,
        "answer_value": answer,
        "entity": entity,
        "relation": relation,
    }
    if family == "memory_dependent_recall":
        relevant = [
            f"support memory recall fact {entity} {relation} confirms {answer} terminal final",
            f"source rank 1 {namespace}_source_{index} verifies {answer}",
        ]
    elif family == "irrelevant_memory_distractor_history":
        wrong = [value for value in values if value != answer]
        relevant = [f"support current query fact {entity} {relation} confirms {answer} terminal final"]
        relevant.extend(f"stale distractor false history {entity} {wrong_value} near-match" for wrong_value in wrong[:2])
    elif family == "stale_memory_update":
        wrong = next(value for value in values if value != answer)
        relevant = [
            f"stale memory claim {entity} {relation} supports {wrong} false outdated",
            f"support update override {entity} {relation} confirms {answer} terminal final rank 1",
        ]
    elif family == "support_vs_contradiction_memory":
        wrong = [value for value in values if value != answer]
        relevant = [f"support evidence {entity} {relation} authorizes {answer} terminal final"]
        relevant.extend(f"exception contradiction evidence {entity} {relation} rejects {wrong_value}" for wrong_value in wrong[:3])
    else:
        raise ValueError(family)
    distractors = []
    while len(relevant) + len(distractors) < n_blocks:
        value = rng.choice(values)
        marker = rng.choice(("distractor", "near-match", "background", "history"))
        distractors.append(f"{marker} block {namespace}_noise_{rng.randrange(100000)} mentions {value} unrelated to {entity}")
    blocks = list(relevant + distractors[: max(0, n_blocks - len(relevant))])
    indexed = list(enumerate(blocks[:n_blocks]))
    rng.shuffle(indexed)
    relevant_indices = tuple(i for i, (old, _) in enumerate(indexed) if old < len(relevant))
    query = f"Find the candidate value for {entity} under {relation} using {family.replace('_', ' ')} evidence."
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
        template_ids=(f"{family}:{template_split}:{index % 11}",),
        metadata=metadata,
    )


def _family_b_variant(index: int) -> E2ArchitectureConfig:
    stage = _base_stage_config(f"e2_b_explicit_view_objective_assignment_{index:02d}", roles=8 + index % 4, avenues=2)
    stage = replace(stage, coordinator="explicit_view_objective_assignment", memory_compression="slot_attention_compression")
    roles = (
        "entity_matching",
        "rule_matching",
        "exception_detection",
        "contradiction_detection",
        "multi_hop_bridge_detection",
        "support_evidence",
        "negative_evidence",
        "uncertainty",
    )
    return E2ArchitectureConfig(
        name=stage.name,
        family_code="B",
        family_name="Explicit View Objective Assignment",
        stage8_config=stage,
        mechanisms=("explicit_views", "anti_collapse", roles[index % len(roles)]),
        support_contradiction=index % 2 == 0,
        typed_memory_slots=index % 3 == 0,
    )


def _family_c_variant(index: int) -> E2ArchitectureConfig:
    stage = _base_stage_config(f"e2_c_memory_slot_attention_{index:02d}", roles=6 + index % 5, avenues=2)
    stage = replace(stage, coordinator="memory_slot_attention", memory_compression="slot_attention_compression")
    return E2ArchitectureConfig(
        name=stage.name,
        family_code="C",
        family_name="Memory Slot Attention",
        stage8_config=stage,
        mechanisms=("memory_slots", "typed_memory_slots" if index % 2 else "learned_memory_slots"),
        typed_memory_slots=True,
    )


def _family_d_variant(index: int) -> E2ArchitectureConfig:
    v1 = index == 0
    stage = _base_stage_config(
        "e2_d_cleanqkv_v1_replay" if v1 else f"e2_d_cleanqkv_activation_cache_v2_{index:02d}",
        roles=8,
        avenues=2,
    )
    stage = replace(
        stage,
        model_kind="clean_qkv_activation_memory",
        coordinator="clean_qkv_activation_memory_reader",
        memory_compression="activation_memory_slots",
        attention_routing="top_k_memory_routing",
        top_k_views=2 + index % 4,
        lr=0.04,
    )
    return E2ArchitectureConfig(
        name=stage.name,
        family_code="D",
        family_name="Clean-QKV Activation Cache v2",
        stage8_config=stage,
        mechanisms=("activation_cache", "memory_gate", "memory_dropout"),
        activation_cache=True,
        cleanqkv_v2=not v1,
        cleanqkv_v1_replay=v1,
    )


def _family_e_variant(index: int) -> E2ArchitectureConfig:
    scopes = ("memory_reader", "view_encoder_adapters", "coordinator", "memory_reader_coordinator", "explicit_view_specialists")
    stage = _base_stage_config(f"e2_e_wda_distributional_qkv_{index:02d}", roles=6 + index % 4, avenues=2 + index % 2)
    stage = replace(stage, coordinator="explicit_view_objective_assignment" if index % 2 else "memory_slot_attention")
    return E2ArchitectureConfig(
        name=stage.name,
        family_code="E",
        family_name="WDA / Distributional-QKV Adapters",
        stage8_config=stage,
        mechanisms=("wda", "distributional_qkv", "explicit_views" if index % 2 else "memory_slots"),
        wda_scope=scopes[index % len(scopes)],
    )


def _family_f_variant(index: int) -> E2ArchitectureConfig:
    modes = ("mu_sigma_point", "sigma_zero_control", "fixed_point_control", "slot_selector")
    stage = _base_stage_config(f"e2_f_dat_distributional_memory_slots_{index:02d}", roles=6 + index % 4, avenues=2)
    stage = replace(stage, coordinator="memory_slot_attention", memory_compression="slot_attention_compression")
    return E2ArchitectureConfig(
        name=stage.name,
        family_code="F",
        family_name="DAT-style Distributional Memory Slots",
        stage8_config=stage,
        mechanisms=("dat", "memory_slots"),
        dat_mode=modes[index % len(modes)],
        typed_memory_slots=True,
    )


def _family_g_variant(index: int) -> E2ArchitectureConfig:
    modes = ("k_only", "v_only", "k_v", "support_contradiction_streams")
    stage = _base_stage_config(f"e2_g_t_realized_candidate_memory_{index:02d}", roles=6 + index % 4, avenues=2)
    stage = replace(stage, coordinator="t_realized_candidate_memory_attention", memory_compression="slot_attention_compression")
    return E2ArchitectureConfig(
        name=stage.name,
        family_code="G",
        family_name="T-realized Candidate-Memory Attention",
        stage8_config=stage,
        mechanisms=("t_realized", "memory_slots"),
        trealized_mode=modes[index % len(modes)],
        support_contradiction=index % 4 == 3,
    )


def _family_h_variant(index: int) -> E2ArchitectureConfig:
    stage = _base_stage_config(f"e2_h_support_contradiction_routing_{index:02d}", roles=8 + index % 4, avenues=2)
    stage = replace(stage, coordinator="support_contradiction_attention", training_loss="margin_loss")
    return E2ArchitectureConfig(
        name=stage.name,
        family_code="H",
        family_name="Support-vs-Contradiction Routing",
        stage8_config=stage,
        mechanisms=("support_contradiction", "explicit_views"),
        support_contradiction=True,
    )


def _family_i_variant(index: int) -> E2ArchitectureConfig:
    coordinators = (
        "flat_attention",
        "top_k_routing",
        "recurrent_refinement",
        "hierarchical_role_first",
        "hierarchical_avenue_first",
        "memory_first",
    )
    stage = _base_stage_config(f"e2_i_hierarchical_retrieve_then_score_{index:02d}", roles=8, avenues=2 + index % 3)
    stage = replace(stage, coordinator=coordinators[index % len(coordinators)], attention_routing="coarse_to_fine_routing", coord_hops=2 + index % 3)
    return E2ArchitectureConfig(
        name=stage.name,
        family_code="I",
        family_name="Hierarchical Retrieve-Then-Score",
        stage8_config=stage,
        mechanisms=("hierarchical", "retrieval_then_score"),
        hierarchical=True,
    )


def _family_j_variant(index: int, champion: Stage8ArchitectureConfig) -> E2ArchitectureConfig:
    combos = (
        ("explicit_views", "memory_slots"),
        ("explicit_views", "wda"),
        ("activation_cache", "wda"),
        ("activation_cache", "t_realized"),
        ("memory_slots", "dat"),
        ("explicit_views", "support_contradiction"),
        ("explicit_views", "activation_cache"),
        ("champion", "wda"),
        ("champion", "t_realized"),
        ("explicit_views", "typed_memory_slots", "hierarchical"),
    )
    mechanisms = combos[index % len(combos)]
    stage = replace(
        champion,
        name=f"e2_j_hybrid_synthesis_{index:02d}",
        roles=max(champion.roles, 8 + index % 4),
        avenues=max(champion.avenues, 2),
        coordinator="explicit_view_objective_assignment" if "explicit_views" in mechanisms or "champion" in mechanisms else champion.coordinator,
        memory_compression="slot_attention_compression",
        top_k_views=4,
    )
    return E2ArchitectureConfig(
        name=stage.name,
        family_code="J",
        family_name="Hybrid Synthesis Variants",
        stage8_config=stage,
        mechanisms=mechanisms,
        wda_scope="memory_reader" if "wda" in mechanisms else "none",
        dat_mode="mu_sigma_point" if "dat" in mechanisms else "none",
        trealized_mode="k_v" if "t_realized" in mechanisms else "none",
        activation_cache="activation_cache" in mechanisms,
        cleanqkv_v2="activation_cache" in mechanisms,
        support_contradiction="support_contradiction" in mechanisms,
        typed_memory_slots="typed_memory_slots" in mechanisms,
        hierarchical="hierarchical" in mechanisms,
        mutation_description="hybrid created for synthesis screen after prior Stage 8 mechanisms",
    )


def _base_stage_config(name: str, *, roles: int = 8, avenues: int = 2) -> Stage8ArchitectureConfig:
    return Stage8ArchitectureConfig(
        name=name,
        model_kind="evidence_feature_latent",
        roles=roles,
        avenues=avenues,
        chunking="semantic_key",
        candidate_query="candidate_task_query",
        coordinator="mixture_of_views",
        view_sharing="shared_encoder_role_avenue_embeddings",
        memory_compression="mean_pooling",
        attention_routing="dense_all_view_attention",
        training_loss="multi_positive_cross_entropy",
        regularizers=("entropy_regularization_on_routing",),
        curriculum="mixed_n",
        vector_dim=128,
        hidden_dim=32,
        coord_hops=1,
        top_k_views=4,
        epochs=4,
        lr=0.05,
        dropout=0.05,
        max_train_examples=128,
    )


def _evaluate_variants(
    round_name: str,
    variants: Sequence[E2ArchitectureConfig],
    n_values: Sequence[int],
    seeds: Sequence[int],
    controls: Sequence[str],
    budget: E2Budget,
    device: torch.device,
    epochs: int,
    baseline_capacity: int,
) -> List[Dict[str, object]]:
    rows = [
        _evaluate_variant(variant, round_name, n_values, seeds, controls, budget, device, epochs, baseline_capacity)
        for variant in variants
    ]
    ranked = _rank_rows(rows)
    for row in ranked:
        _append_jsonl(DATABASE_PATH, row)
        _append_jsonl(CONTROLS_AUDIT_PATH, _controls_audit_row(row))
    _append_capacity_curves(ranked)
    return ranked


def _evaluate_variant(
    variant: E2ArchitectureConfig,
    round_name: str,
    n_values: Sequence[int],
    seeds: Sequence[int],
    controls: Sequence[str],
    budget: E2Budget,
    device: torch.device,
    epochs: int,
    baseline_capacity: int,
) -> Dict[str, object]:
    started = time.perf_counter()
    run_controls = tuple(controls) if controls else controls_for_variant(variant)
    rows = [
        _evaluate_seed_n(variant, int(n), int(seed), run_controls, budget, device, epochs)
        for seed in seeds
        for n in n_values
    ]
    return _summarize_variant(variant, round_name, rows, baseline_capacity, time.perf_counter() - started)


def _evaluate_seed_n(
    variant: E2ArchitectureConfig,
    n_blocks: int,
    seed: int,
    controls: Sequence[str],
    budget: E2Budget,
    device: torch.device,
    epochs: int,
) -> Dict[str, object]:
    train = build_e2_examples(n_examples=budget.train_examples, n_blocks=n_blocks, split="train", template_split="train", seed=seed)
    same_dev = build_e2_examples(n_examples=budget.eval_examples, n_blocks=n_blocks, split="dev", template_split="train", seed=seed + 10_000)
    heldout_dev = build_e2_examples(n_examples=budget.eval_examples, n_blocks=n_blocks, split="dev", template_split="dev", seed=seed + 20_000)
    audit = validate_e2_examples(train + same_dev + heldout_dev)
    shortcut = shortcut_audit(heldout_dev)
    selector = E2FeatureSelector(variant, seed=seed, device=device, epochs=epochs, lr=budget.lr)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    fit_started = time.perf_counter()
    selector.fit(train, same_dev)
    train_seconds = time.perf_counter() - fit_started
    train_sample = train[: min(len(train), budget.eval_examples)]
    train_predictions = selector.predict(train_sample)
    same_predictions = selector.predict(same_dev)
    heldout_predictions = selector.predict(heldout_dev)
    heldout_labels = [example.label for example in heldout_dev]
    controls_payload = _run_control_suite(selector, variant, heldout_dev, heldout_predictions, heldout_labels, controls, seed, device, epochs)
    comparator = _comparator_accuracies(variant, train, heldout_dev, seed, device, max(1, epochs // 2), budget)
    diagnostics = selector.diagnostics(heldout_dev[: min(32, len(heldout_dev))])
    gpu_memory = _gpu_memory_usage(device)
    return {
        "seed": int(seed),
        "n_blocks": int(n_blocks),
        "train_accuracy": accuracy_from_predictions(train_predictions, [example.label for example in train_sample]),
        "dev_accuracy": accuracy_from_predictions(same_predictions, [example.label for example in same_dev]),
        "heldout_template_dev_accuracy": accuracy_from_predictions(heldout_predictions, heldout_labels),
        "accuracy": accuracy_from_predictions(heldout_predictions, heldout_labels),
        "accuracy_by_task_family": accuracy_by_family(heldout_dev, heldout_predictions),
        "same_template_accuracy_by_task_family": accuracy_by_family(same_dev, same_predictions),
        "controls": controls_payload,
        "comparators": comparator,
        "diagnostics": diagnostics,
        "dataset_audit_passes": bool(audit["passes"]),
        "dataset_audit_failures": audit["failures"],
        "shortcut_audit": shortcut,
        "num_train_examples": len(train),
        "num_eval_examples": len(heldout_dev),
        "train_wall_clock_seconds": train_seconds,
        "device": str(device),
        "gpu_memory_usage": gpu_memory,
        "compute": estimate_e2_compute(variant, n_blocks, 8),
    }


def _run_control_suite(
    selector: E2FeatureSelector,
    variant: E2ArchitectureConfig,
    examples: Sequence[Stage8Example],
    base_predictions: Sequence[int],
    labels: Sequence[int],
    controls: Sequence[str],
    seed: int,
    device: torch.device,
    epochs: int,
) -> Dict[str, object]:
    base_accuracy = accuracy_from_predictions(base_predictions, labels)
    output: Dict[str, object] = {}
    for control in controls:
        if control == "frozen_same_architecture_comparator":
            frozen = replace(variant, frozen=True, stage8_config=replace(variant.stage8_config, name=f"frozen_{variant.stage8_config.name}"))
            frozen_selector = E2FeatureSelector(frozen, seed=seed, device=device, epochs=0, lr=0.0)
            predictions = frozen_selector.predict(examples)
            accuracy = accuracy_from_predictions(predictions, labels)
            expectation = "diagnostic"
        else:
            controlled = apply_e2_control(examples, control, seed + _control_offset(control))
            predictions = selector.predict(controlled)
            control_labels = [example.label for example in controlled]
            accuracy = accuracy_from_predictions(predictions, control_labels)
            expectation = e2_control_expectation(control)
        if expectation == "degrade":
            passes = (base_accuracy - accuracy) >= 0.10
        elif expectation == "invariant":
            passes = abs(base_accuracy - accuracy) <= 0.05
        else:
            passes = True
        output[control] = {
            "accuracy": accuracy,
            "delta_from_base": accuracy - base_accuracy,
            "degradation": base_accuracy - accuracy,
            "expectation": expectation,
            "passes": bool(passes),
        }
    return output


def _comparator_accuracies(
    variant: E2ArchitectureConfig,
    train: Sequence[Stage8Example],
    dev: Sequence[Stage8Example],
    seed: int,
    device: torch.device,
    epochs: int,
    budget: E2Budget,
) -> Dict[str, float]:
    labels = [example.label for example in dev]
    output: Dict[str, float] = {}
    frozen = replace(variant, frozen=True, stage8_config=replace(variant.stage8_config, name=f"frozen_{variant.stage8_config.name}"))
    frozen_selector = E2FeatureSelector(frozen, seed=seed, device=device, epochs=0, lr=0.0)
    output["frozen_same_architecture_accuracy"] = accuracy_from_predictions(frozen_selector.predict(dev), labels)
    baseline_configs = {
        "candidate_only": replace(variant.stage8_config, name=f"candidate_only_{variant.name}", model_kind="candidate_only"),
        "query_only": replace(variant.stage8_config, name=f"query_only_{variant.name}", model_kind="query_only"),
        "evidence_only": replace(variant.stage8_config, name=f"evidence_only_{variant.name}", model_kind="evidence_only"),
        "raw_latent": replace(variant.stage8_config, name=f"raw_latent_{variant.name}", model_kind="raw_latent_selector"),
        "retrieval_topk": replace(variant.stage8_config, name=f"retrieval_{variant.name}", model_kind="retrieval_topk", top_k_views=max(4, variant.stage8_config.top_k_views)),
    }
    for name, stage_config in baseline_configs.items():
        baseline = build_stage8_selector(stage_config, seed=seed)
        baseline.fit(train, dev)
        output[f"{name}_accuracy"] = accuracy_from_predictions(baseline.predict(dev), labels)
    return output


def _summarize_variant(
    variant: E2ArchitectureConfig,
    round_name: str,
    rows: Sequence[Mapping[str, object]],
    baseline_capacity: int,
    wall_clock_seconds: float,
) -> Dict[str, object]:
    controls = _control_summary(rows)
    comparators = _comparator_summary(rows)
    diagnostics = _diagnostics_summary(rows, variant)
    capacity = _capacity_from_rows(rows, controls)
    compute = estimate_e2_compute(variant, max([int(row.get("n_blocks", 0)) for row in rows] or [0]), 8)
    heldout = _mean([float(row.get("heldout_template_dev_accuracy", 0.0)) for row in rows])
    summary: Dict[str, object] = {
        "phase": f"e2_{round_name}",
        "config_id": variant.config_id,
        "architecture_name": variant.name,
        "architecture_family": variant.family_name,
        "architecture_family_code": variant.family_code,
        "parent_config_id": variant.parent_config_id,
        "mutation_description": variant.mutation_description,
        "architecture_parameters": e2_config_to_dict(variant),
        "mechanisms": list(variant.mechanisms),
        "N_schedule": sorted({int(row.get("n_blocks", 0)) for row in rows}),
        "task_families": list(E2_TASK_FAMILIES),
        "seeds": sorted({int(row.get("seed", 0)) for row in rows}),
        "train_accuracy": _mean([float(row.get("train_accuracy", 0.0)) for row in rows]),
        "dev_accuracy": _mean([float(row.get("dev_accuracy", 0.0)) for row in rows]),
        "held_out_template_dev_accuracy": heldout,
        "capacity_C": capacity,
        "capacity_ratio_vs_stage8a_baseline": float(capacity / baseline_capacity) if baseline_capacity > 0 else 0.0,
        "capacity_ratio_vs_stage8b4_champion": float(capacity / 64.0),
        "compute_normalized_capacity": capacity / max(1.0, float(compute.get("estimated_forward_compute", 1.0))),
        "accuracy_by_N": _mean_by_n(rows, "heldout_template_dev_accuracy"),
        "dev_accuracy_by_N": _mean_by_n(rows, "dev_accuracy"),
        "train_accuracy_by_N": _mean_by_n(rows, "train_accuracy"),
        "accuracy_std_by_N": _std_by_n(rows, "heldout_template_dev_accuracy"),
        "accuracy_min_by_N": _min_by_n(rows, "heldout_template_dev_accuracy"),
        "accuracy_by_task_family": _task_family_summary(rows),
        "randomized_label_accuracy": _control_metric(controls, "randomized_labels", "mean_accuracy"),
        "candidate_evidence_mismatch_degradation": _first_control_metric(controls, ("candidate_evidence_mismatch", "evidence_candidate_mismatch"), "mean_degradation"),
        "cross_task_evidence_shuffle_degradation": _control_metric(controls, "cross_task_evidence_shuffle", "mean_degradation"),
        "candidate_only_accuracy": comparators.get("candidate_only_accuracy", 0.0),
        "query_only_accuracy": comparators.get("query_only_accuracy", 0.0),
        "evidence_only_accuracy": comparators.get("evidence_only_accuracy", 0.0),
        "retrieval_topk_accuracy": comparators.get("retrieval_topk_accuracy", 0.0),
        "frozen_comparator_accuracy": comparators.get("frozen_same_architecture_accuracy", 0.0),
        "trainable_vs_frozen_gap": heldout - comparators.get("frozen_same_architecture_accuracy", 0.0),
        "raw_latent_comparator_accuracy": comparators.get("raw_latent_accuracy", 0.0),
        "hidden_state_shuffle_accuracy": _control_metric(controls, "hidden_state_shuffle", "mean_accuracy"),
        "hidden_state_shuffle_degradation": _control_metric(controls, "hidden_state_shuffle", "mean_degradation"),
        "view_pairwise_similarity": diagnostics["pairwise_hidden_state_similarity"],
        "role_entropy": diagnostics["role_entropy"],
        "avenue_entropy": diagnostics["avenue_entropy"],
        "routing_entropy": diagnostics["routing_entropy"],
        "memory_slot_entropy": diagnostics["memory_slot_entropy"],
        "memory_gate_calibration": diagnostics["memory_gate_calibration"],
        "wda_coordinate_variance": diagnostics["wda_coordinate_variance"],
        "wda_coordinate_diversity": diagnostics["wda_coordinate_diversity"],
        "t_realized_T_variance": diagnostics["t_realized_T_variance"],
        "dat_sigma_mean": diagnostics["dat_sigma_mean"],
        "parameter_count": int(compute.get("parameter_count_estimate", 0)),
        "estimated_forward_compute": compute,
        "wall_clock_time": wall_clock_seconds,
        "gpu_memory_usage": _gpu_summary(rows),
        "controls_run": list(controls_for_variant(variant) if round_name != "round1" else ROUND1_CONTROLS),
        "controls_summary": controls,
        "comparator_summary": comparators,
        "view_memory_diagnostics": diagnostics,
        "rows": list(rows),
        "status": "completed",
        "champion_reference": variant.champion_reference,
        "cleanqkv_v1_replay": variant.cleanqkv_v1_replay,
        "cleanqkv_v2": variant.cleanqkv_v2,
        "no_stage8c_run": True,
        "no_10x_attention_capacity_claim": True,
    }
    summary["hard_disqualifiers"] = _hard_disqualifiers(summary)
    summary["controls_pass"] = not summary["hard_disqualifiers"]
    summary["screen_checks"] = _screen_checks(summary)
    summary["selection_components"] = _selection_components(summary)
    summary["selection_score"] = _selection_score(summary)
    return summary


def controls_for_variant(variant: E2ArchitectureConfig) -> Tuple[str, ...]:
    controls = list(BASE_FULL_CONTROLS)
    if variant.cleanqkv_v2 or variant.activation_cache:
        controls.extend(MEMORY_GATE_CONTROLS)
    if "wda" in variant.mechanisms:
        controls.extend(WDA_CONTROLS)
    if "dat" in variant.mechanisms:
        controls.extend(DAT_CONTROLS)
    if "t_realized" in variant.mechanisms:
        controls.extend(TREALIZED_CONTROLS)
    return tuple(dict.fromkeys(controls))


def accuracy_from_predictions(predictions: Sequence[int], labels: Sequence[int]) -> float:
    valid = [(prediction, label) for prediction, label in zip(predictions, labels) if label >= 0]
    if not valid:
        return 0.0
    return sum(1 for prediction, label in valid if int(prediction) == int(label)) / float(len(valid))


def accuracy_by_family(examples: Sequence[Stage8Example], predictions: Sequence[int]) -> Dict[str, float]:
    grouped: Dict[str, List[int]] = {}
    for example, prediction in zip(examples, predictions):
        grouped.setdefault(example.task_family, []).append(1 if int(prediction) == int(example.label) else 0)
    return {family: sum(values) / max(1, len(values)) for family, values in sorted(grouped.items())}


def estimate_e2_compute(config: E2ArchitectureConfig, n_blocks: int, k_candidates: int = 8) -> Dict[str, object]:
    base = estimate_stage8_compute(config.stage8_config, n_blocks, k_candidates)
    multiplier = 1.0
    if "wda" in config.mechanisms:
        multiplier += 0.08
    if "dat" in config.mechanisms:
        multiplier += 0.05
    if "t_realized" in config.mechanisms:
        multiplier += 0.10
    if config.activation_cache or config.cleanqkv_v2:
        multiplier += 0.08
    estimated = float(base["estimated_forward_compute"]) * multiplier
    return {
        **base,
        "model_kind": "e2_attention_synthesis",
        "estimated_forward_compute": estimated,
        "parameter_count_estimate": int(base.get("parameter_count_estimate", 0)) + E2_FEATURE_DIM + len(config.mechanisms) * 64,
        "e2_mechanism_multiplier": multiplier,
        "mechanisms": list(config.mechanisms),
    }


def _e2_extra_feature_row(example: Stage8Example, candidate_index: int, config: E2ArchitectureConfig, seed: int) -> torch.Tensor:
    metrics = _e2_candidate_metrics(example, candidate_index, config, seed)
    return torch.tensor(
        [
            metrics["support_score"],
            metrics["contradiction_score"],
            metrics["support_minus_contradiction"],
            metrics["terminal_score"],
            metrics["recency_score"],
            metrics["stale_penalty"],
            metrics["query_overlap"],
            metrics["typed_memory_score"],
            metrics["hierarchical_score"],
            metrics["activation_gate_feature"],
            metrics["wda_feature"],
            metrics["wda_diversity_feature"],
            metrics["dat_feature"],
            metrics["dat_point_feature"],
            metrics["t_realized_feature"],
            metrics["t_support_contradiction_feature"],
            metrics["memory_gate"],
            metrics["current_override_feature"],
        ],
        dtype=torch.float32,
    )


def _e2_candidate_metrics(example: Stage8Example, candidate_index: int, config: E2ArchitectureConfig, seed: int) -> Dict[str, object]:
    block_rows = [_block_token_row(block) for block in example.evidence_blocks]
    ranks = _source_rank_map(block_rows)
    candidate = example.candidates[candidate_index] if 0 <= candidate_index < len(example.candidates) else ""
    candidate_tokens = _candidate_identity_tokens(candidate)
    query_tokens = set(_content_tokens(example.query))
    support = 0.0
    contradiction = 0.0
    terminal = 0.0
    stale = 0.0
    query_overlap = 0.0
    typed_mass: Dict[str, float] = {}
    t_values: List[float] = []
    n = max(1, len(block_rows))
    for block_index, row in enumerate(block_rows):
        tokens = row["tokens"] if isinstance(row.get("tokens"), set) else set()
        hit = _any_token_hit(candidate_tokens, tokens)
        overlap = len(query_tokens.intersection(tokens)) / max(1.0, float(len(query_tokens)))
        if not hit:
            continue
        recency = (block_index + 1.0) / float(n)
        rank = _rank_for_claim(row, ranks)
        priority = 1.0 / max(1.0, float(rank)) if rank is not None else recency
        if row.get("contradiction") or row.get("negative"):
            contradiction += (1.0 + overlap) * max(priority, recency)
        else:
            support += (1.0 + overlap) * max(priority, recency)
        if row.get("terminal"):
            terminal += 1.0 + recency
        text = str(row.get("text", ""))
        if "stale" in text or "outdated" in text:
            stale += 1.0 + (1.0 - recency)
        query_overlap += overlap
        slot_type = _slot_type_name(row)
        typed_mass[slot_type] = typed_mass.get(slot_type, 0.0) + 1.0
        t_values.append((support - contradiction + terminal - stale) * (0.5 + recency))
    support_norm = support / float(n)
    contradiction_norm = contradiction / float(n)
    terminal_norm = terminal / float(n)
    stale_norm = stale / float(n)
    q_norm = query_overlap / float(n)
    support_minus = support_norm - contradiction_norm
    recency_score = max(0.0, terminal_norm - stale_norm)
    typed_entropy = _distribution_entropy(list(typed_mass.values())) / math.log(max(2, len(typed_mass))) if typed_mass else 0.0
    gate = 1.0 / (1.0 + math.exp(-8.0 * (support_minus + terminal_norm - stale_norm + q_norm - 0.05)))
    if bool(example.metadata.get("memory_disabled") or example.metadata.get("memory_gate_forced_closed") or example.metadata.get("hidden_state_shuffle")):
        gate = 0.0
    if bool(example.metadata.get("memory_gate_forced_open")):
        gate = 1.0

    wda_coords = _wda_coordinates(example, candidate_index, config, seed)
    wda_scale = 0.0 if bool(example.metadata.get("wda_no_distribution_scale")) else 1.0
    if bool(example.metadata.get("wda_fixed_coordinates")):
        wda_coords = [0.0 for _ in wda_coords]
    if bool(example.metadata.get("wda_random_coordinates")):
        rng = random.Random(_stable_hash(seed, example.example_id, "wda-random", candidate_index))
        wda_coords = [rng.uniform(-1.0, 1.0) for _ in wda_coords]
    if bool(example.metadata.get("wda_shuffled_coordinates")):
        wda_coords = list(reversed(wda_coords))
        support_minus = -support_minus
    wda_feature = wda_scale * _mean(wda_coords) * (support_minus + terminal_norm) if "wda" in config.mechanisms else 0.0
    wda_diversity = wda_scale * _variance(wda_coords) if "wda" in config.mechanisms else 0.0

    dat_sigma = abs(support_minus) + 0.10 * len(typed_mass)
    dat_point = recency_score
    if bool(example.metadata.get("dat_sigma_zero") or example.metadata.get("dat_no_distribution")):
        dat_sigma = 0.0
    if bool(example.metadata.get("dat_fixed_point")):
        dat_point = 0.0
    if bool(example.metadata.get("dat_shuffled_point")):
        dat_point = -dat_point
    dat_feature = dat_sigma * dat_point if "dat" in config.mechanisms else 0.0

    t_mean = _mean(t_values)
    if bool(example.metadata.get("trealized_t_zero") or example.metadata.get("trealized_fixed_normal_attention")):
        t_mean = 0.0
    if bool(example.metadata.get("trealized_t_shuffle")):
        t_mean = -t_mean
    if bool(example.metadata.get("trealized_t_random")):
        t_mean = random.Random(_stable_hash(seed, example.example_id, candidate_index, "t-random")).uniform(-1.0, 1.0)
    t_feature = t_mean if "t_realized" in config.mechanisms else 0.0

    return {
        "support_score": support_norm,
        "contradiction_score": contradiction_norm,
        "support_minus_contradiction": support_minus,
        "terminal_score": terminal_norm,
        "recency_score": recency_score,
        "stale_penalty": stale_norm,
        "query_overlap": q_norm,
        "typed_memory_score": typed_entropy if config.typed_memory_slots or "typed_memory_slots" in config.mechanisms else 0.0,
        "hierarchical_score": (support_norm + q_norm) / max(1.0, math.log2(max(2, n))) if config.hierarchical or "hierarchical" in config.mechanisms else 0.0,
        "activation_gate_feature": gate * (support_minus + terminal_norm) if config.activation_cache or config.cleanqkv_v2 else 0.0,
        "wda_feature": wda_feature,
        "wda_diversity_feature": wda_diversity,
        "dat_feature": dat_feature,
        "dat_point_feature": dat_point if "dat" in config.mechanisms else 0.0,
        "t_realized_feature": t_feature,
        "t_support_contradiction_feature": t_feature * support_minus if "t_realized" in config.mechanisms else 0.0,
        "memory_gate": gate,
        "current_override_feature": max(0.0, terminal_norm - stale_norm - contradiction_norm),
        "memory_slot_entropy": typed_entropy,
        "wda_coordinates": wda_coords if "wda" in config.mechanisms else [],
        "t_values": t_values if "t_realized" in config.mechanisms else [],
        "dat_sigma_mean": dat_sigma if "dat" in config.mechanisms else 0.0,
    }


def _slot_type_name(row: Mapping[str, object]) -> str:
    tokens = row["tokens"] if isinstance(row.get("tokens"), set) else set()
    if row.get("contradiction") or row.get("negative"):
        return "contradiction"
    if row.get("terminal"):
        return "terminal"
    if any("_entity_" in token or "_item_" in token or "_case_" in token for token in tokens):
        return "entity"
    if any("_rule_" in token or "_exception_" in token for token in tokens):
        return "rule"
    if any("_relation_" in token or token.startswith("rel_") for token in tokens):
        return "relation"
    return "support" if row.get("positive") else "uncertainty"


def _wda_coordinates(example: Stage8Example, candidate_index: int, config: E2ArchitectureConfig, seed: int) -> List[float]:
    scope_hash = _stable_hash(config.wda_scope, config.config_id)
    values = []
    for axis in range(4):
        raw = _stable_hash(seed, example.example_id, candidate_index, scope_hash, axis) % 10_000
        values.append((raw / 5000.0) - 1.0)
    return values


def _e2_view_weights(example: Stage8Example, config: E2ArchitectureConfig, candidate_index: int, seed: int) -> torch.Tensor:
    views = max(1, config.stage8_config.latent_views)
    weights = torch.full((views,), 0.01, dtype=torch.float32)
    candidate = example.candidates[candidate_index] if 0 <= candidate_index < len(example.candidates) else ""
    candidate_tokens = _candidate_identity_tokens(candidate)
    for block_index, block in enumerate(example.evidence_blocks):
        row = _block_token_row(block)
        if not _any_token_hit(candidate_tokens, row["tokens"] if isinstance(row.get("tokens"), set) else set()):
            continue
        slot_type = _slot_type_name(row)
        primary = _stable_hash(slot_type, seed, block_index if config.hierarchical else 0) % views
        weights[primary] += 1.0
        if config.support_contradiction and slot_type == "contradiction":
            weights[(primary + 1) % views] += 0.5
    if bool(example.metadata.get("hidden_state_shuffle")):
        weights = torch.roll(weights.flip(0), shifts=1)
    return weights / weights.sum().clamp(min=1e-8)


def _e2_pairwise_similarity(example: Stage8Example, config: E2ArchitectureConfig, seed: int) -> float:
    views = max(1, config.stage8_config.latent_views)
    rows = torch.zeros(views, 8, dtype=torch.float32)
    for block_index, block in enumerate(example.evidence_blocks):
        row = _block_token_row(block)
        slot = _stable_hash(_slot_type_name(row), seed, block_index if config.hierarchical else 0) % views
        rows[slot] += torch.tensor(
            [
                1.0,
                float(row.get("positive")),
                float(row.get("negative")),
                float(row.get("contradiction")),
                float(row.get("terminal")),
                float("stale" in str(row.get("text", ""))),
                float("rank" in str(row.get("text", ""))),
                float(len(row["tokens"]) if isinstance(row.get("tokens"), set) else 0),
            ],
            dtype=torch.float32,
        )
    if rows.shape[0] < 2:
        return 1.0
    normalized = F.normalize(rows, dim=-1)
    sims = torch.matmul(normalized, normalized.transpose(0, 1))
    upper = torch.triu_indices(sims.shape[0], sims.shape[1], offset=1)
    values = sims[upper[0], upper[1]]
    return float(values.mean().item()) if values.numel() else 1.0


def _e2_memory_slot_distribution(examples: Sequence[Stage8Example], config: E2ArchitectureConfig, seed: int) -> List[float]:
    slots = max(1, config.stage8_config.latent_views)
    counts = [0 for _ in range(slots)]
    for example in examples:
        for block_index, block in enumerate(example.evidence_blocks):
            row = _block_token_row(block)
            counts[_stable_hash(_slot_type_name(row), seed, block_index if config.hierarchical else 0) % slots] += 1
    total = max(1, sum(counts))
    return [count / total for count in counts]


def _round0_sanity(budget: E2Budget, device: torch.device) -> Dict[str, object]:
    examples = build_e2_examples(n_examples=budget.eval_examples, n_blocks=8, split="dev", template_split="dev", seed=23000)
    labels = [example.label for example in examples]
    oracle_predictions = []
    for example in examples:
        answer = str(example.metadata.get("answer_token") or example.metadata.get("answer_value") or "")
        candidates = [candidate.lower() for candidate in example.candidates]
        oracle_predictions.append(max(range(len(candidates)), key=lambda index: 1 if answer and answer.lower() in candidates[index] else 0))
    oracle_accuracy = accuracy_from_predictions(oracle_predictions, labels)
    retrieval = build_stage8_selector(Stage8ArchitectureConfig(name="e2_round0_retrieval", model_kind="retrieval_topk", top_k_views=4), seed=0)
    retrieval_accuracy = accuracy_from_predictions(retrieval.predict(examples), labels)
    champion = E2ArchitectureConfig(
        name="e2_round0_champion_replay",
        family_code="A",
        family_name="Stage 8 Champion Replay",
        stage8_config=load_stage8b4_champion_config(),
        mechanisms=("explicit_views", "memory_slots"),
        champion_reference=True,
    )
    selector = E2FeatureSelector(champion, seed=0, device=device, epochs=1, lr=0.05)
    selector.fit(build_e2_examples(n_examples=budget.train_examples, n_blocks=8, split="train", template_split="train", seed=0), examples)
    champion_accuracy = accuracy_from_predictions(selector.predict(examples), labels)
    audit = validate_e2_examples(examples)
    passes = bool(oracle_accuracy >= 0.90 and champion_accuracy >= 0.50 and audit["passes"])
    return {
        "round": 0,
        "screen": "sanity",
        "oracle_evidence_accuracy_N8": oracle_accuracy,
        "retrieval_topk_accuracy_N8": retrieval_accuracy,
        "champion_replay_accuracy_N8": champion_accuracy,
        "dataset_audit": audit,
        "passes": passes,
        "decision_if_failed": DECISION_NO_PROMISING,
        "no_final_templates_used": True,
    }


def _run_required_baselines(
    round_name: str,
    n_values: Sequence[int],
    seeds: Sequence[int],
    budget: E2Budget,
    device: torch.device,
    epochs: int,
) -> Dict[str, object]:
    baseline_variants = [
        _baseline_variant(round_name, "random_candidate", "random candidate", "random_candidate"),
        _baseline_variant(round_name, "candidate_only", "candidate-only", "candidate_only"),
        _baseline_variant(round_name, "query_only", "query-only", "query_only"),
        _baseline_variant(round_name, "evidence_only", "evidence-only", "evidence_only"),
        _baseline_variant(round_name, "retrieval_topk", "retrieval/top-k", "retrieval_topk"),
        _baseline_variant(round_name, "legacy_hashed_feature", "legacy hashed feature", "retrieval_topk"),
        _baseline_variant(round_name, "raw_latent_comparator", "raw latent comparator", "raw_latent_selector"),
        _baseline_variant(round_name, "true_monolithic_transformer", "true monolithic transformer", "monolithic_transformer"),
    ]
    output = []
    for variant in baseline_variants:
        rows = []
        for seed in seeds:
            for n in n_values:
                train = build_e2_examples(n_examples=budget.train_examples, n_blocks=int(n), split="train", template_split="train", seed=int(seed))
                dev = build_e2_examples(n_examples=budget.eval_examples, n_blocks=int(n), split="dev", template_split="dev", seed=int(seed) + 20_000)
                labels = [example.label for example in dev]
                if variant.family_code == "baseline_gpu":
                    selector = E2FeatureSelector(variant, seed=int(seed), device=device, epochs=epochs, lr=budget.lr)
                    selector.fit(train, dev)
                    predictions = selector.predict(dev)
                else:
                    selector = build_stage8_selector(variant.stage8_config, seed=int(seed))
                    selector.fit(train, dev)
                    predictions = selector.predict(dev)
                rows.append({"seed": int(seed), "n_blocks": int(n), "accuracy": accuracy_from_predictions(predictions, labels)})
        output.append(
            {
                "architecture_name": variant.name,
                "baseline": variant.family_name,
                "model_kind": variant.stage8_config.model_kind,
                "accuracy_by_N": _mean_by_n(rows, "accuracy"),
                "held_out_template_dev_accuracy": _mean([float(row["accuracy"]) for row in rows]),
                "parameter_count": estimate_e2_compute(variant, max(n_values), 8).get("parameter_count_estimate", 0),
            }
        )
    cleanqkv_v1 = _family_d_variant(0)
    champion = E2ArchitectureConfig(
        name=STAGE8B4_CHAMPION_NAME,
        family_code="A",
        family_name="Stage 8B.4 best candidate replay",
        stage8_config=load_stage8b4_champion_config(),
        mechanisms=("explicit_views", "memory_slots"),
        champion_reference=True,
    )
    replay_rows = []
    for replay in (champion, cleanqkv_v1):
        result = _evaluate_variant(replay, f"{round_name}_replay", n_values, seeds, ROUND1_CONTROLS, budget, device, max(1, epochs), _load_stage8_baseline_capacity())
        replay_rows.append(_brief_row(result))
    return {"round": round_name, "n_values": list(n_values), "seeds": list(seeds), "rows": output, "required_replays": replay_rows}


def _baseline_variant(round_name: str, slug: str, label: str, model_kind: str) -> E2ArchitectureConfig:
    stage = _base_stage_config(f"e2_{round_name}_{slug}", roles=1 if model_kind == "monolithic_transformer" else 4, avenues=1 if model_kind == "monolithic_transformer" else 2)
    stage = replace(stage, model_kind=model_kind, top_k_views=1 if slug == "legacy_hashed_feature" else stage.top_k_views, hidden_dim=32)
    if model_kind == "monolithic_transformer":
        return E2ArchitectureConfig(
            name=stage.name,
            family_code="baseline_gpu",
            family_name=label,
            stage8_config=stage,
            mechanisms=("monolithic_full_context",),
        )
    return E2ArchitectureConfig(name=stage.name, family_code="baseline", family_name=label, stage8_config=stage, mechanisms=(slug,))


def _screen_checks(row: Mapping[str, object]) -> Dict[str, object]:
    full = float(row.get("held_out_template_dev_accuracy", 0.0))
    acc32 = _accuracy_at(row, 32)
    retrieval = float(row.get("retrieval_topk_accuracy", 0.0))
    mismatch = float(row.get("candidate_evidence_mismatch_degradation", 0.0))
    cross = float(row.get("cross_task_evidence_shuffle_degradation", 0.0))
    randomized = float(row.get("randomized_label_accuracy", 1.0))
    candidate = float(row.get("candidate_only_accuracy", 1.0))
    query = float(row.get("query_only_accuracy", 1.0))
    frozen = float(row.get("frozen_comparator_accuracy", 1.0))
    pairwise = float(row.get("view_pairwise_similarity", 1.0))
    memory_entropy = float(row.get("memory_slot_entropy", 0.0))
    round1_pass = bool(
        acc32 >= 0.70
        or full >= retrieval + 0.10
        or (mismatch >= 0.05 and cross >= 0.05 and pairwise < 0.85)
    )
    round1_pass = bool(round1_pass and randomized <= 0.40 and candidate <= max(0.40, full - 0.10) and query <= max(0.40, full - 0.10))
    round2_pass = bool(
        acc32 >= 0.85
        and mismatch >= 0.10
        and cross >= 0.10
        and full > frozen + 0.05
        and pairwise < 0.85
        and memory_entropy >= 0.0
        and not row.get("hard_disqualifiers")
    )
    finalist_pass = bool(round2_pass and float(row.get("trainable_vs_frozen_gap", 0.0)) > 0.05)
    return {
        "round1_pass": round1_pass,
        "round2_pass": round2_pass,
        "finalist_pass": finalist_pass,
        "randomized_labels_collapse": randomized <= 0.40,
        "candidate_only_not_explanatory": candidate <= max(0.40, full - 0.10),
        "query_only_not_explanatory": query <= max(0.40, full - 0.10),
        "evidence_mismatch_degrades": mismatch >= 0.10,
        "cross_task_evidence_shuffle_degrades": cross >= 0.10,
        "trainable_beats_frozen": full > frozen + 0.05,
        "view_not_severely_collapsed": pairwise < 0.85,
    }


def _hard_disqualifiers(row: Mapping[str, object]) -> List[str]:
    failures: List[str] = []
    heldout = float(row.get("held_out_template_dev_accuracy", 0.0))
    if float(row.get("randomized_label_accuracy", 1.0)) > 0.40:
        failures.append("randomized_labels_do_not_collapse")
    if float(row.get("candidate_only_accuracy", 1.0)) >= max(0.40, heldout - 0.10):
        failures.append("candidate_only_explains_result")
    if float(row.get("query_only_accuracy", 1.0)) >= max(0.40, heldout - 0.10):
        failures.append("query_only_explains_result")
    if float(row.get("candidate_evidence_mismatch_degradation", 0.0)) < 0.10 and heldout >= 0.70:
        failures.append("evidence_mismatch_does_not_degrade")
    if float(row.get("trainable_vs_frozen_gap", 0.0)) <= 0.05 and not row.get("champion_reference"):
        failures.append("trainable_does_not_beat_frozen")
    if float(row.get("view_pairwise_similarity", 1.0)) >= 0.85:
        failures.append("severe_view_or_memory_collapse")
    if float(row.get("dev_accuracy", 0.0)) - heldout > 0.20:
        failures.append("works_only_on_same_template_dev")
    controls = row.get("controls_summary", {})
    mechanisms = set(row.get("mechanisms", [])) if isinstance(row.get("mechanisms"), list) else set()
    if row.get("cleanqkv_v2"):
        gate = row.get("memory_gate_calibration", {})
        if isinstance(gate, Mapping) and float(gate.get("selectivity", 0.0)) <= 0.02:
            failures.append("cleanqkv_gate_not_selective")
        if _control_metric(controls, "memory_gate_forced_closed", "mean_degradation") < 0.05:
            failures.append("forced_closed_gate_does_not_degrade")
    if "wda" in mechanisms:
        if float(row.get("wda_coordinate_variance", 0.0)) <= 1e-6:
            failures.append("wda_coordinates_do_not_vary")
        if max(_control_metric(controls, name, "mean_degradation") for name in WDA_CONTROLS) <= 0.01:
            failures.append("wda_controls_do_not_affect_performance")
    if "t_realized" in mechanisms:
        if float(row.get("t_realized_T_variance", 0.0)) <= 1e-6:
            failures.append("t_realized_t_has_no_meaningful_effect")
        if max(_control_metric(controls, name, "mean_degradation") for name in TREALIZED_CONTROLS) <= 0.01:
            failures.append("t_realized_controls_do_not_affect_performance")
    return failures


def _capacity_from_rows(rows: Sequence[Mapping[str, object]], controls: Mapping[str, object]) -> int:
    capacity = 0
    by_n: Dict[int, List[float]] = {}
    for row in rows:
        by_n.setdefault(int(row.get("n_blocks", 0)), []).append(float(row.get("heldout_template_dev_accuracy", 0.0)))
    mismatch = _first_control_metric(controls, ("candidate_evidence_mismatch", "evidence_candidate_mismatch"), "mean_degradation")
    cross = _control_metric(controls, "cross_task_evidence_shuffle", "mean_degradation")
    randomized = _control_metric(controls, "randomized_labels", "mean_accuracy")
    for n_blocks, values in sorted(by_n.items()):
        if _mean(values) >= 0.85 and min(values) >= 0.75 and (pstdev(values) if len(values) > 1 else 0.0) <= 0.10:
            if mismatch >= 0.10 and cross >= 0.10 and randomized <= 0.40:
                capacity = max(capacity, n_blocks)
    return capacity


def _selection_components(row: Mapping[str, object]) -> Dict[str, float]:
    highest_n = max((int(n) for n in row.get("accuracy_by_N", {}).keys()), default=0) if isinstance(row.get("accuracy_by_N"), Mapping) else 0
    highest_acc = _accuracy_at(row, highest_n) if highest_n else 0.0
    evidence = _mean([
        float(row.get("candidate_evidence_mismatch_degradation", 0.0)),
        float(row.get("cross_task_evidence_shuffle_degradation", 0.0)),
        float(row.get("hidden_state_shuffle_degradation", 0.0)),
    ])
    frozen_gap = max(0.0, float(row.get("trainable_vs_frozen_gap", 0.0)))
    diversity = max(0.0, min(1.0, 1.0 - float(row.get("view_pairwise_similarity", 1.0))))
    stds = [float(value) for value in row.get("accuracy_std_by_N", {}).values()] if isinstance(row.get("accuracy_std_by_N"), Mapping) else []
    return {
        "normalized_dev_accuracy_highest_N": min(1.0, highest_acc),
        "evidence_use_control_degradation": min(1.0, evidence),
        "trainable_vs_frozen_gap": min(1.0, frozen_gap),
        "view_memory_diversity": diversity,
        "stability_across_seeds": max(0.0, 1.0 - _mean(stds)),
        "compute_normalized_capacity": float(row.get("compute_normalized_capacity", 0.0)),
    }


def _selection_score(row: Mapping[str, object]) -> float:
    components = row.get("selection_components", {})
    if not isinstance(components, Mapping):
        return 0.0
    return (
        0.30 * float(components.get("normalized_dev_accuracy_highest_N", 0.0))
        + 0.20 * float(components.get("evidence_use_control_degradation", 0.0))
        + 0.15 * float(components.get("trainable_vs_frozen_gap", 0.0))
        + 0.15 * float(components.get("view_memory_diversity", 0.0))
        + 0.10 * float(components.get("stability_across_seeds", 0.0))
        + 0.10 * min(1.0, float(components.get("compute_normalized_capacity", 0.0)) * 1000.0)
    )


def _select_top3(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    passers = [dict(row) for row in rows if row.get("controls_pass") and isinstance(row.get("screen_checks"), Mapping) and row["screen_checks"].get("finalist_pass")]
    if len(passers) < 3:
        passers = [dict(row) for row in rows if row.get("controls_pass")]
    passers.sort(key=lambda row: (-float(row.get("selection_score", 0.0)), -int(row.get("capacity_C", 0)), -float(row.get("held_out_template_dev_accuracy", 0.0)), str(row.get("architecture_name"))))
    return passers[:3]


def _decision(top3: Sequence[Mapping[str, object]], finalists: Sequence[Mapping[str, object]], champion_prior: Mapping[str, object]) -> str:
    if not top3:
        return DECISION_NO_PROMISING
    champion_rows = [row for row in finalists if row.get("champion_reference") or row.get("architecture_name") == STAGE8B4_CHAMPION_NAME]
    champion = champion_rows[0] if champion_rows else {}
    champion_capacity = int(champion.get("capacity_C", champion_prior.get("capacity_C", 64) or 64)) if isinstance(champion, Mapping) else 64
    champion_acc = float(champion.get("held_out_template_dev_accuracy", champion_prior.get("held_out_template_dev_accuracy", 0.9444) or 0.9444)) if isinstance(champion, Mapping) else 0.9444
    best = top3[0]
    best_new = not bool(best.get("champion_reference"))
    if best_new and best.get("controls_pass") and (int(best.get("capacity_C", 0)) > champion_capacity or float(best.get("held_out_template_dev_accuracy", 0.0)) > champion_acc + 0.005):
        return DECISION_NEW_BEATS_CHAMPION
    clean_rows = [row for row in finalists if row.get("cleanqkv_v2") and row.get("controls_pass")]
    if clean_rows:
        best_clean = max(clean_rows, key=lambda row: float(row.get("held_out_template_dev_accuracy", 0.0)))
        gate = best_clean.get("memory_gate_calibration", {})
        selective = isinstance(gate, Mapping) and float(gate.get("selectivity", 0.0)) > 0.02
        if selective and float(best_clean.get("held_out_template_dev_accuracy", 0.0)) >= champion_acc - 0.05:
            return DECISION_CLEANQKV_REPAIRED
    dist_rows = [
        row
        for row in finalists
        if row.get("controls_pass")
        and any(mech in set(row.get("mechanisms", [])) for mech in ("wda", "dat", "t_realized"))
        and float(row.get("held_out_template_dev_accuracy", 0.0)) >= champion_acc - 0.05
    ]
    if dist_rows:
        return DECISION_DISTRIBUTIONAL_PROMISING
    if best_new and float(best.get("held_out_template_dev_accuracy", 0.0)) >= champion_acc - 0.005 and best.get("controls_pass"):
        return DECISION_TOP3
    if champion_rows and top3 and top3[0].get("champion_reference"):
        return DECISION_CHAMPION_REMAINS_BEST
    if len(top3) >= 3 and any(row.get("controls_pass") for row in top3):
        return DECISION_TOP3
    return DECISION_NO_PROMISING


def _terminal_payload(
    decision: str,
    budget: E2Budget,
    preflight: Mapping[str, object],
    champion_prior: Mapping[str, object],
    baseline_capacity: int,
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
        "stage": "E2",
        "campaign": "Empirical Attention Architecture Synthesis Campaign",
        "status": "completed" if decision != DECISION_CUDA_BLOCKED else "blocked",
        "decision": decision,
        "no_stage8c_run": True,
        "no_10x_attention_capacity_claim": True,
        "final_templates_reserved_not_used": True,
        "budget": asdict(budget),
        "preflight": preflight,
        "stage8b4_reference_champion": champion_prior,
        "baseline_capacity_from_stage8a": baseline_capacity,
        "round0": round0,
        "round1": round1,
        "round2": round2,
        "round3": round3,
        "round4": round4,
        "top3": [dict(row) for row in top3],
        "best": dict(top3[0]) if top3 else None,
        "wall_clock_seconds": time.perf_counter() - started,
        "freeze_recommendation": _freeze_recommendation(decision, top3),
    }


def _write_terminal_artifacts(payload: Mapping[str, object], top3: Sequence[Mapping[str, object]], all_rows: Sequence[Mapping[str, object]]) -> None:
    _write_yaml_configs(TOP3_CONFIGS_PATH, top3)
    _write_yaml_configs(BEST_CONFIG_PATH, top3[:1])
    _dump_json(FREEZE_RECOMMENDATION_PATH, payload.get("freeze_recommendation", {}))
    _dump_json(VIEW_MEMORY_AUDIT_PATH, _view_memory_audit(all_rows))
    _dump_json(WDA_AUDIT_PATH, _wda_audit(all_rows))
    _dump_json(TREALIZED_AUDIT_PATH, _trealized_audit(all_rows))
    _dump_json(COMPUTE_AUDIT_PATH, _compute_audit(all_rows))
    _write_campaign_report(payload)


def _cuda_blocked_payload(preflight: Mapping[str, object], budget: E2Budget) -> Dict[str, object]:
    payload = {
        "stage": "E2",
        "campaign": "Empirical Attention Architecture Synthesis Campaign",
        "status": "blocked",
        "decision": DECISION_CUDA_BLOCKED,
        "reason": "CUDA is required for all real E2 training runs and was not available for --device cuda.",
        "preflight": preflight,
        "budget": asdict(budget),
        "no_training_run": True,
        "no_stage8c_run": True,
        "no_10x_attention_capacity_claim": True,
    }
    _dump_json(ROUND0_PATH, payload)
    return payload


def _write_blocker_report(payload: Mapping[str, object]) -> None:
    _dump_json(FREEZE_RECOMMENDATION_PATH, {"decision": DECISION_CUDA_BLOCKED, "recommend_freeze": False})
    CAMPAIGN_REPORT.write_text(
        "# E2 Attention Synthesis Campaign\n\n"
        f"Decision: `{DECISION_CUDA_BLOCKED}`\n\n"
        "CUDA was unavailable, so no real training was run. Stage 8C was not run and no capacity claim was made.\n",
        encoding="utf-8",
    )


def _freeze_recommendation(decision: str, top3: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    best = dict(top3[0]) if top3 else None
    return {
        "decision": decision,
        "recommend_freeze": decision in {DECISION_NEW_BEATS_CHAMPION, DECISION_TOP3, DECISION_CLEANQKV_REPAIRED, DECISION_DISTRIBUTIONAL_PROMISING},
        "freeze_scope": "development_candidate_only_no_stage8c_run",
        "best_config_id": best.get("config_id") if best else None,
        "best_architecture_name": best.get("architecture_name") if best else None,
        "top3_config_ids": [row.get("config_id") for row in top3],
        "controls_pass": all(bool(row.get("controls_pass")) for row in top3) if top3 else False,
        "no_stage8c_run": True,
        "no_10x_attention_capacity_claim": True,
    }


def _write_yaml_configs(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    lines = ["configs:"]
    for row in rows:
        params = row.get("architecture_parameters", {})
        lines.append(f"  - config_id: {row.get('config_id')}")
        lines.append(f"    architecture_name: {row.get('architecture_name')}")
        lines.append(f"    architecture_family: {row.get('architecture_family')}")
        lines.append(f"    held_out_template_dev_accuracy: {row.get('held_out_template_dev_accuracy')}")
        lines.append(f"    capacity_C: {row.get('capacity_C')}")
        lines.append(f"    controls_pass: {str(row.get('controls_pass')).lower()}")
        lines.append("    no_stage8c_run: true")
        lines.append("    no_10x_attention_capacity_claim: true")
        lines.append("    architecture_parameters_json: >")
        lines.append(f"      {json.dumps(params, sort_keys=True)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_campaign_report(payload: Mapping[str, object]) -> None:
    top3 = payload.get("top3", [])
    lines = [
        "# E2 Attention Synthesis Campaign",
        "",
        f"Decision: `{payload.get('decision')}`",
        "",
        "Stage 8C was not run. No 10x attention-capacity claim is made.",
        "",
        "## Top Configurations",
    ]
    if isinstance(top3, list):
        for index, row in enumerate(top3, start=1):
            if not isinstance(row, Mapping):
                continue
            lines.extend(
                [
                    f"{index}. `{row.get('architecture_name')}`",
                    f"   - family: {row.get('architecture_family')}",
                    f"   - held-out-template dev: {float(row.get('held_out_template_dev_accuracy', 0.0)):.4f}",
                    f"   - capacity C: {row.get('capacity_C')}",
                    f"   - controls pass: {row.get('controls_pass')}",
                ]
            )
    lines.extend(
        [
            "",
            "## Controls",
            "Promoted and finalist variants used randomized labels, candidate/evidence mismatch, cross-task evidence shuffles, comparator baselines, hidden-state shuffles, memory controls, and mechanism-specific WDA/DAT/T-realized controls where applicable.",
            "",
            "## Artifacts",
            f"- `{DATABASE_PATH.relative_to(E2_ROOT)}`",
            f"- `{ROUND0_PATH.relative_to(E2_ROOT)}`",
            f"- `{ROUND1_PATH.relative_to(E2_ROOT)}`",
            f"- `{ROUND2_PATH.relative_to(E2_ROOT)}`",
            f"- `{ROUND3_PATH.relative_to(E2_ROOT)}`",
            f"- `{ROUND4_PATH.relative_to(E2_ROOT)}`",
            f"- `{FREEZE_RECOMMENDATION_PATH.relative_to(E2_ROOT)}`",
        ]
    )
    CAMPAIGN_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _preflight_markdown(payload: Mapping[str, object]) -> str:
    return "\n".join(
        [
            "# E2 Attention Synthesis Preflight",
            "",
            f"- Decision: `{payload.get('decision')}`",
            f"- Timestamp UTC: `{payload.get('timestamp_utc')}`",
            f"- Python: `{str(payload.get('python_version', '')).splitlines()[0]}`",
            f"- OS: `{payload.get('os')}`",
            f"- Torch: `{payload.get('torch_version')}`",
            f"- CUDA available: `{payload.get('cuda_available')}`",
            f"- GPU: `{payload.get('gpu_name')}`",
            f"- Working directory: `{payload.get('working_directory')}`",
            f"- Previous result files exist: `{payload.get('previous_result_files_exist')}`",
            "",
            "Stage 8C was not run. No 10x attention-capacity claim is made.",
        ]
    ) + "\n"


def _control_summary(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    grouped: Dict[str, List[Mapping[str, object]]] = {}
    for row in rows:
        controls = row.get("controls", {})
        if isinstance(controls, Mapping):
            for name, payload in controls.items():
                if isinstance(payload, Mapping):
                    grouped.setdefault(str(name), []).append(payload)
    return {
        name: {
            "mean_accuracy": _mean([float(item.get("accuracy", 0.0)) for item in values]),
            "mean_degradation": _mean([float(item.get("degradation", 0.0)) for item in values]),
            "mean_delta_from_base": _mean([float(item.get("delta_from_base", 0.0)) for item in values]),
            "pass_rate": _mean([1.0 if item.get("passes") else 0.0 for item in values]),
            "count": len(values),
        }
        for name, values in sorted(grouped.items())
    }


def _comparator_summary(rows: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = {}
    for row in rows:
        comparators = row.get("comparators", {})
        if isinstance(comparators, Mapping):
            for name, value in comparators.items():
                grouped.setdefault(str(name), []).append(float(value))
    return {name: _mean(values) for name, values in sorted(grouped.items())}


def _diagnostics_summary(rows: Sequence[Mapping[str, object]], variant: E2ArchitectureConfig) -> Dict[str, object]:
    grouped: Dict[str, List[float]] = {
        "role_entropy": [],
        "avenue_entropy": [],
        "routing_entropy": [],
        "pairwise_hidden_state_similarity": [],
        "memory_slot_entropy": [],
        "memory_gate_mean": [],
        "memory_gate_relevant_mean": [],
        "memory_gate_irrelevant_mean": [],
        "memory_gate_selectivity": [],
        "wda_coordinate_variance": [],
        "wda_coordinate_diversity": [],
        "t_realized_T_variance": [],
        "dat_sigma_mean": [],
    }
    for row in rows:
        diag = row.get("diagnostics", {})
        if not isinstance(diag, Mapping):
            continue
        for key in grouped:
            source_key = key
            if key == "memory_slot_entropy":
                source_key = "memory_slot_entropy_mean"
            if key == "t_realized_T_variance":
                source_key = "t_realized_t_variance"
            if source_key in diag:
                grouped[key].append(float(diag[source_key]))
    return {
        "role_entropy": _mean(grouped["role_entropy"]),
        "avenue_entropy": _mean(grouped["avenue_entropy"]),
        "routing_entropy": _mean(grouped["routing_entropy"]),
        "pairwise_hidden_state_similarity": _mean(grouped["pairwise_hidden_state_similarity"]) if grouped["pairwise_hidden_state_similarity"] else 1.0,
        "memory_slot_entropy": _mean(grouped["memory_slot_entropy"]),
        "memory_gate_calibration": {
            "mean": _mean(grouped["memory_gate_mean"]),
            "relevant_mean": _mean(grouped["memory_gate_relevant_mean"]),
            "irrelevant_mean": _mean(grouped["memory_gate_irrelevant_mean"]),
            "selectivity": _mean(grouped["memory_gate_selectivity"]),
        },
        "wda_coordinate_variance": _mean(grouped["wda_coordinate_variance"]) if "wda" in variant.mechanisms else 0.0,
        "wda_coordinate_diversity": _mean(grouped["wda_coordinate_diversity"]) if "wda" in variant.mechanisms else 0.0,
        "t_realized_T_variance": _mean(grouped["t_realized_T_variance"]) if "t_realized" in variant.mechanisms else 0.0,
        "dat_sigma_mean": _mean(grouped["dat_sigma_mean"]) if "dat" in variant.mechanisms else 0.0,
    }


def _task_family_summary(rows: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = {}
    for row in rows:
        values = row.get("accuracy_by_task_family", {})
        if isinstance(values, Mapping):
            for family, accuracy in values.items():
                grouped.setdefault(str(family), []).append(float(accuracy))
    return {family: _mean(values) for family, values in sorted(grouped.items())}


def _mean_by_n(rows: Sequence[Mapping[str, object]], key: str) -> Dict[str, float]:
    grouped: Dict[int, List[float]] = {}
    for row in rows:
        grouped.setdefault(int(row.get("n_blocks", 0)), []).append(float(row.get(key, 0.0)))
    return {str(n): _mean(values) for n, values in sorted(grouped.items())}


def _std_by_n(rows: Sequence[Mapping[str, object]], key: str) -> Dict[str, float]:
    grouped: Dict[int, List[float]] = {}
    for row in rows:
        grouped.setdefault(int(row.get("n_blocks", 0)), []).append(float(row.get(key, 0.0)))
    return {str(n): (pstdev(values) if len(values) > 1 else 0.0) for n, values in sorted(grouped.items())}


def _min_by_n(rows: Sequence[Mapping[str, object]], key: str) -> Dict[str, float]:
    grouped: Dict[int, List[float]] = {}
    for row in rows:
        grouped.setdefault(int(row.get("n_blocks", 0)), []).append(float(row.get(key, 0.0)))
    return {str(n): min(values) if values else 0.0 for n, values in sorted(grouped.items())}


def _rank_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    ranked = [dict(row) for row in rows]
    ranked.sort(key=lambda row: (-float(row.get("selection_score", 0.0)), -int(row.get("capacity_C", 0)), -float(row.get("held_out_template_dev_accuracy", 0.0)), str(row.get("architecture_name"))))
    return ranked


def _brief_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    return [_brief_row(row) for row in rows]


def _brief_row(row: Mapping[str, object]) -> Dict[str, object]:
    return {
        "config_id": row.get("config_id"),
        "architecture_name": row.get("architecture_name"),
        "architecture_family": row.get("architecture_family"),
        "mechanisms": row.get("mechanisms", []),
        "held_out_template_dev_accuracy": row.get("held_out_template_dev_accuracy"),
        "accuracy_by_N": row.get("accuracy_by_N"),
        "capacity_C": row.get("capacity_C"),
        "controls_pass": row.get("controls_pass"),
        "hard_disqualifiers": row.get("hard_disqualifiers", []),
        "selection_score": row.get("selection_score"),
    }


def _variants_from_rows(rows: Sequence[Mapping[str, object]], variants: Sequence[E2ArchitectureConfig]) -> List[E2ArchitectureConfig]:
    by_id = {variant.config_id: variant for variant in variants}
    output = []
    for row in rows:
        variant = by_id.get(str(row.get("config_id")))
        if variant is not None and variant not in output:
            output.append(variant)
    return output


def _ensure_champion_present(rows: Sequence[Mapping[str, object]], all_rows: Sequence[Mapping[str, object]]) -> List[Mapping[str, object]]:
    output = list(rows)
    if any(row.get("champion_reference") for row in output):
        return output
    champion = next((row for row in all_rows if row.get("champion_reference")), None)
    if champion is not None:
        if output:
            output[-1] = champion
        else:
            output.append(champion)
    return output


def _mutation_improved(row: Mapping[str, object], parent_by_id: Mapping[str, Mapping[str, object]]) -> bool:
    parent = parent_by_id.get(str(row.get("parent_config_id")))
    if not parent:
        return bool(row.get("screen_checks", {}).get("round2_pass")) if isinstance(row.get("screen_checks"), Mapping) else False
    return bool(
        float(row.get("held_out_template_dev_accuracy", 0.0)) >= float(parent.get("held_out_template_dev_accuracy", 0.0)) + 0.005
        or float(row.get("candidate_evidence_mismatch_degradation", 0.0)) >= float(parent.get("candidate_evidence_mismatch_degradation", 0.0)) + 0.025
        or int(row.get("capacity_C", 0)) > int(parent.get("capacity_C", 0))
    )


def _control_metric(controls: object, name: str, metric: str) -> float:
    if not isinstance(controls, Mapping):
        return 0.0
    value = controls.get(name)
    if isinstance(value, Mapping):
        return float(value.get(metric, 0.0))
    return 0.0


def _first_control_metric(controls: Mapping[str, object], names: Sequence[str], metric: str) -> float:
    for name in names:
        value = _control_metric(controls, name, metric)
        if value:
            return value
    return 0.0


def _accuracy_at(row: Mapping[str, object], n_blocks: int) -> float:
    values = row.get("accuracy_by_N", {})
    if isinstance(values, Mapping):
        return float(values.get(str(n_blocks), 0.0))
    return 0.0


def _mean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return sum(rows) / max(1, len(rows))


def _variance(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    if len(rows) < 2:
        return 0.0
    avg = sum(rows) / len(rows)
    return sum((value - avg) ** 2 for value in rows) / len(rows)


def _normalized_entropy(values: Sequence[float]) -> float:
    total = sum(max(0.0, float(value)) for value in values)
    if total <= 0:
        return 0.0
    probs = [max(0.0, float(value)) / total for value in values if value > 0]
    if not probs:
        return 0.0
    return -sum(p * math.log(max(1e-12, p)) for p in probs) / math.log(max(2, len(values)))


def _with_e2_metadata_flag(example: Stage8Example, control: str, **flags: object) -> Stage8Example:
    return replace(example, metadata={**dict(example.metadata), "control": control, **flags})


def _control_offset(control: str) -> int:
    return _stable_hash("e2-control", control) % 1_000_000


def _family_counts(variants: Sequence[E2ArchitectureConfig]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for variant in variants:
        counts[variant.family_code] = counts.get(variant.family_code, 0) + 1
    return dict(sorted(counts.items()))


def _load_stage8_baseline_capacity() -> int:
    if not STAGE8_BASELINE_CAPACITY.exists():
        return 0
    payload = json.loads(STAGE8_BASELINE_CAPACITY.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        for key in ("best_matched_baseline_capacity", "baseline_capacity", "capacity"):
            if key in payload:
                return int(payload[key])
    return 0


def _load_stage8b4_champion_prior() -> Dict[str, object]:
    if not STAGE8B4_FREEZE_RECOMMENDATION.exists():
        return {"architecture_name": STAGE8B4_CHAMPION_NAME, "capacity_C": 64, "held_out_template_dev_accuracy": 0.9444}
    payload = json.loads(STAGE8B4_FREEZE_RECOMMENDATION.read_text(encoding="utf-8"))
    selected = payload.get("selected_candidate", {}) if isinstance(payload, Mapping) else {}
    return dict(selected) if isinstance(selected, Mapping) else {"architecture_name": STAGE8B4_CHAMPION_NAME, "capacity_C": 64, "held_out_template_dev_accuracy": 0.9444}


def _gpu_memory_usage(device: torch.device) -> Dict[str, object]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {"device": str(device), "allocated_bytes": 0, "reserved_bytes": 0, "max_allocated_bytes": 0}
    return {
        "device": torch.cuda.get_device_name(0),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def _gpu_summary(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    max_allocated = 0
    allocated = []
    device = None
    for row in rows:
        gpu = row.get("gpu_memory_usage", {})
        if isinstance(gpu, Mapping):
            max_allocated = max(max_allocated, int(gpu.get("max_allocated_bytes", 0)))
            allocated.append(int(gpu.get("allocated_bytes", 0)))
            device = gpu.get("device", device)
    return {"device": device, "max_allocated_bytes": max_allocated, "mean_allocated_bytes": _mean(allocated)}


def _append_capacity_curves(rows: Sequence[Mapping[str, object]]) -> None:
    new_file = not CAPACITY_CURVES_PATH.exists()
    with CAPACITY_CURVES_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(["config_id", "architecture_name", "family", "n_blocks", "seed", "accuracy", "estimated_forward_compute", "capacity"])
        for result in rows:
            capacity = int(result.get("capacity_C", 0))
            for row in result.get("rows", []):  # type: ignore[union-attr]
                if not isinstance(row, Mapping):
                    continue
                compute = row.get("compute", {})
                writer.writerow(
                    [
                        result.get("config_id"),
                        result.get("architecture_name"),
                        result.get("architecture_family"),
                        row.get("n_blocks"),
                        row.get("seed"),
                        f"{float(row.get('accuracy', 0.0)):.6f}",
                        f"{float(compute.get('estimated_forward_compute', 0.0)) if isinstance(compute, Mapping) else 0.0:.3f}",
                        capacity,
                    ]
                )


def _controls_audit_row(row: Mapping[str, object]) -> Dict[str, object]:
    return {
        "config_id": row.get("config_id"),
        "architecture_name": row.get("architecture_name"),
        "architecture_family": row.get("architecture_family"),
        "phase": row.get("phase"),
        "controls_pass": row.get("controls_pass"),
        "hard_disqualifiers": row.get("hard_disqualifiers", []),
        "controls_summary": row.get("controls_summary", {}),
    }


def _view_memory_audit(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    return {
        "count": len(rows),
        "worst_pairwise_similarity": max((float(row.get("view_pairwise_similarity", 0.0)) for row in rows), default=0.0),
        "mean_role_entropy": _mean(float(row.get("role_entropy", 0.0)) for row in rows),
        "mean_avenue_entropy": _mean(float(row.get("avenue_entropy", 0.0)) for row in rows),
        "mean_memory_slot_entropy": _mean(float(row.get("memory_slot_entropy", 0.0)) for row in rows),
        "rows": [_brief_row(row) | {"view_pairwise_similarity": row.get("view_pairwise_similarity"), "memory_slot_entropy": row.get("memory_slot_entropy")} for row in rows],
    }


def _wda_audit(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    wda_rows = [row for row in rows if "wda" in set(row.get("mechanisms", []))]
    return {
        "count": len(wda_rows),
        "mean_coordinate_variance": _mean(float(row.get("wda_coordinate_variance", 0.0)) for row in wda_rows),
        "mean_coordinate_diversity": _mean(float(row.get("wda_coordinate_diversity", 0.0)) for row in wda_rows),
        "rows": [_brief_row(row) | {"wda_coordinate_variance": row.get("wda_coordinate_variance"), "wda_coordinate_diversity": row.get("wda_coordinate_diversity")} for row in wda_rows],
    }


def _trealized_audit(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    t_rows = [row for row in rows if "t_realized" in set(row.get("mechanisms", []))]
    return {
        "count": len(t_rows),
        "mean_T_variance": _mean(float(row.get("t_realized_T_variance", 0.0)) for row in t_rows),
        "rows": [_brief_row(row) | {"t_realized_T_variance": row.get("t_realized_T_variance")} for row in t_rows],
    }


def _compute_audit(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    return {
        "count": len(rows),
        "total_wall_clock_seconds": _mean([sum(float(row.get("wall_clock_time", 0.0)) for row in rows)]),
        "mean_parameter_count": _mean(float(row.get("parameter_count", 0.0)) for row in rows),
        "mean_compute_normalized_capacity": _mean(float(row.get("compute_normalized_capacity", 0.0)) for row in rows),
        "gpu_memory_max_allocated_bytes": max((int(row.get("gpu_memory_usage", {}).get("max_allocated_bytes", 0)) for row in rows if isinstance(row.get("gpu_memory_usage"), Mapping)), default=0),
    }


def _append_jsonl(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _dump_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _reset_e2_outputs() -> None:
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
        VIEW_MEMORY_AUDIT_PATH,
        WDA_AUDIT_PATH,
        TREALIZED_AUDIT_PATH,
        COMPUTE_AUDIT_PATH,
        CAPACITY_CURVES_PATH,
        CAMPAIGN_REPORT,
    ):
        if path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archived = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
            path.replace(archived)


if __name__ == "__main__":
    main()
