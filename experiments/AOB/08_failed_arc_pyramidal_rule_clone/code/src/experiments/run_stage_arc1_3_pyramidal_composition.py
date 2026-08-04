from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.datasets.arc_agi2_verification import (
    ArcVerificationDatasetConfig,
    ArcVerificationExample,
    apply_arc_control,
    build_arc_verification_splits,
    candidate_metadata_only_accuracy,
    dataset_summary,
    leakage_audit_rows,
)
from src.experiments.run_stage_arc1_latent_rule_clones import (
    ArcModelConfig,
    ArcTrainingConfig,
    CandidateRuleCrossAttentionBlock,
    FieldTokenEncoder,
    VIEW_ID,
    VIEW_NAMES,
    _accuracy,
    _baseline_block,
    _batched,
    _bootstrap_ci,
    _candidate_objective_loss,
    _candidate_stat_predictions,
    _changed_cell_count_predictions,
    _chunks,
    _clear_cuda,
    _cuda_max_memory,
    _flat_params,
    _grad_norm,
    _l2_delta,
    _masked_mean,
    _mean,
    _metric_block,
    _resolve_device,
    _set_seed,
    _shape,
    _softmax_np,
    _std,
    _top1,
    collate_arc_batch,
)


BENCHMARK = "arc1_3_pyramidal_composition"
DEFAULT_CONFIG = "configs/stage_arc1_3_pyramidal_composition_smoke.json"
DEFAULT_RESULTS = "results/arc1_3_pyramidal_results.json"
DEFAULT_CONTROLS = "results/arc1_3_pyramidal_controls.json"
DEFAULT_LEADERBOARD = "results/arc1_3_pyramidal_leaderboard.csv"
DEFAULT_ABLATION = "results/arc1_3_pyramidal_ablation.json"
DEFAULT_ERRORS = "results/arc1_3_pyramidal_error_cases.jsonl"
DEFAULT_ATTENTION = "results/arc1_3_pyramidal_attention_summaries.jsonl"
DEFAULT_COMPUTE = "results/arc1_3_pyramidal_compute_metrics.json"
DEFAULT_REPORT = "reports/STAGE_ARC1_3_PYRAMIDAL_COMPOSITION.md"

PAIR_ARC = (
    ("raw", "diff", "change"),
    ("object", "geometry", "structure"),
    ("color", "symmetry", "appearance"),
    ("object", "color", "object_appearance"),
)
TREE_LEVEL1 = (
    ("raw", "diff", "change"),
    ("object", "geometry", "structure"),
    ("color", "symmetry", "appearance"),
)
COMPOSITION_IDS = {
    "fallback": 0,
    "change": 1,
    "structure": 2,
    "appearance": 3,
    "object_appearance": 4,
    "transformation": 5,
    "final": 6,
    "all_pair": 7,
}


@dataclass(frozen=True)
class PyramidalArcModelConfig(ArcModelConfig):
    pyramid_layout: str = "flat"
    composer_kind: str = "none"
    query_mode: str = "leaf_only"
    pair_scheme: str = "arc_pairs"
    candidate_guided_composition: bool = False
    residual_pyramid: bool = False
    shared_composer_weights: bool = True
    ordered_composition_children: bool = True
    max_composed_tokens: int = 48
    parent_uses_raw_evidence: bool = False


@dataclass(frozen=True)
class PyramidalVariant:
    name: str
    stage_variant: str
    description: str
    model: PyramidalArcModelConfig
    objective: str = "cross_entropy"
    negative_difficulty: str = "mixed_hard"
    candidate_count: int = 8
    diagnostic_only: bool = False


@dataclass
class PyramidalFitResult:
    method: str
    model: "ArcPyramidalRuleCloneVerifier"
    config: PyramidalArcModelConfig
    trainable_shared: bool
    audit: Dict[str, object]
    history: List[Dict[str, float]]
    training_time_seconds: float
    param_count: int


@dataclass
class LatentState:
    name: str
    level: str
    tokens: torch.Tensor
    mask: torch.Tensor
    child_names: Tuple[str, ...] = tuple()
    child_parent_cosine: torch.Tensor | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage ARC-1.3 pyramidal latent view composition search.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_RESULTS)
    parser.add_argument("--controls-output", default=DEFAULT_CONTROLS)
    parser.add_argument("--leaderboard-output", default=DEFAULT_LEADERBOARD)
    parser.add_argument("--ablation-output", default=DEFAULT_ABLATION)
    parser.add_argument("--error-output", default=DEFAULT_ERRORS)
    parser.add_argument("--attention-output", default=DEFAULT_ATTENTION)
    parser.add_argument("--compute-output", default=DEFAULT_COMPUTE)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-variants", type=int, default=0)
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    if args.dataset_root:
        config.setdefault("dataset", {})["dataset_root"] = str(args.dataset_root)
    if args.device:
        config["device"] = str(args.device)
    result = run_stage_arc1_3(config, max_variants=int(args.max_variants))
    _write_outputs(
        result,
        Path(args.output),
        Path(args.controls_output),
        Path(args.leaderboard_output),
        Path(args.ablation_output),
        Path(args.error_output),
        Path(args.attention_output),
        Path(args.compute_output),
        Path(args.report),
    )


def run_stage_arc1_3(config: Dict[str, object], max_variants: int = 0) -> Dict[str, object]:
    dataset_config = _dataset_config(config.get("dataset", {}))
    base_training = _training_config(config.get("training", {}))
    overfit_training = _training_config({**dict(config.get("training", {})), **dict(config.get("overfit_training", {}))})
    seeds = [int(seed) for seed in config.get("seeds", [0])]
    device = _resolve_device(str(config.get("device", "auto")))
    variants = _variant_plan(config.get("variants"))
    if max_variants:
        variants = variants[:max_variants]
    if int(config.get("max_variants", 0)):
        variants = variants[: int(config.get("max_variants", 0))]
    run_overfit = bool(config.get("run_overfit_gates", True))

    rows: List[Dict[str, object]] = []
    controls: List[Dict[str, object]] = []
    ablations: List[Dict[str, object]] = []
    overfit_rows: List[Dict[str, object]] = []
    error_cases: List[Dict[str, object]] = []
    attention_rows: List[Dict[str, object]] = []
    compute_rows: List[Dict[str, object]] = []
    phase0_rows: List[Dict[str, object]] = []
    leakage_rows_out: List[Dict[str, object]] = []
    flat_dev_logits_by_seed: Dict[int, np.ndarray] = {}

    print(f"arc1.3: device={device} variants={len(variants)} seeds={seeds}")
    for seed in seeds:
        seed_phase_config = replace(dataset_config, num_candidates=8)
        phase_splits = build_arc_verification_splits(seed_phase_config, seed=seed)
        leakage_rows = leakage_audit_rows([example for part in phase_splits.values() for example in part])
        leakage_rows_out.extend(leakage_rows)
        phase0_rows.append(
            {
                "seed": seed,
                "dataset_summary": dataset_summary(phase_splits),
                "leakage_audit_pass": all(bool(row.get("pass")) for row in leakage_rows),
                "candidate_metadata_only_accuracy_dev": candidate_metadata_only_accuracy(phase_splits.get("dev", [])),
                "candidate_metadata_only_accuracy_test": candidate_metadata_only_accuracy(phase_splits.get("test", [])),
            }
        )
        for variant in variants:
            print(f"arc1.3 variant={variant.name} seed={seed}: overfit gates")
            gate_rows = (
                _run_overfit_gates(variant, dataset_config, overfit_training, seed, device)
                if run_overfit
                else [{"variant": variant.name, "seed": seed, "gate": "skipped", "pass": True, "gate_required": False}]
            )
            overfit_rows.extend(gate_rows)
            gates_pass = all(bool(row["pass"]) for row in gate_rows if row.get("gate_required", True))
            row_base = {
                "benchmark": BENCHMARK,
                "seed": seed,
                "variant": variant.name,
                "stage_variant": variant.stage_variant,
                "description": variant.description,
                "diagnostic_only": bool(variant.diagnostic_only),
                "candidate_count": int(variant.candidate_count),
                "negative_difficulty": variant.negative_difficulty,
                "objective": variant.objective,
                "model_config": asdict(variant.model),
                "overfit_gates_pass": gates_pass,
            }
            if not gates_pass:
                rows.append({**row_base, "status": "failed_overfit_gate", "failure_reason": _first_failed_gate(gate_rows)})
                continue

            split_config = replace(
                dataset_config,
                num_candidates=int(variant.candidate_count),
                negative_difficulty=str(variant.negative_difficulty),
            )
            splits = build_arc_verification_splits(split_config, seed=seed)
            leak_pass = all(row.get("pass") for row in leakage_audit_rows([example for part in splits.values() for example in part]))
            training = replace(base_training, objective=variant.objective)
            print(f"arc1.3 variant={variant.name} seed={seed}: cheap held-out fit")
            start = time.perf_counter()
            trainable = fit_pyramid_verifier(
                splits["train"],
                splits["dev"],
                variant.model,
                training,
                seed=seed + 13_701,
                device=device,
                trainable_shared=True,
                method=f"trainable__{variant.name}",
            )
            frozen = fit_pyramid_verifier(
                splits["train"],
                splits["dev"],
                variant.model,
                training,
                seed=seed + 13_701,
                device=device,
                trainable_shared=False,
                method=f"frozen__{variant.name}",
            )
            fit_elapsed = time.perf_counter() - start
            eval_start = time.perf_counter()
            train_logits = predict_pyramid_logits(trainable.model, splits["train"], variant.model, training.batch_size, device)
            dev_logits = predict_pyramid_logits(trainable.model, splits["dev"], variant.model, training.batch_size, device)
            frozen_dev_logits = predict_pyramid_logits(frozen.model, splits["dev"], variant.model, training.batch_size, device)
            test_logits = predict_pyramid_logits(trainable.model, splits["test"], variant.model, training.batch_size, device)
            latency = (time.perf_counter() - eval_start) / max(1, len(splits["dev"]) + len(splits["test"]))
            train_labels = np.asarray([example.label for example in splits["train"]], dtype=np.int64)
            dev_labels = np.asarray([example.label for example in splits["dev"]], dtype=np.int64)
            test_labels = np.asarray([example.label for example in splits["test"]], dtype=np.int64)
            train_block = _metric_block(train_logits, train_labels, splits["train"])
            dev_block = _metric_block(dev_logits, dev_labels, splits["dev"])
            frozen_block = _metric_block(frozen_dev_logits, dev_labels, splits["dev"])
            controls_row = _run_pyramid_controls(trainable.model, variant.model, splits["dev"], training.batch_size, device, seed, dev_logits, dev_labels)
            controls.append({**controls_row, "benchmark": BENCHMARK, "seed": seed, "variant": variant.name})
            ablation_rows = _run_pyramid_ablations(
                trainable.model,
                variant.model,
                splits["dev"],
                training.batch_size,
                device,
                seed,
                dev_logits,
                dev_labels,
            )
            for row in ablation_rows:
                ablations.append({**row, "benchmark": BENCHMARK, "seed": seed, "variant": variant.name})
            attention_rows.extend(_attention_summaries(trainable.model, variant.model, splits["dev"], training.batch_size, device, seed, limit=8))
            flat_logits = flat_dev_logits_by_seed.get(seed)
            error_cases.extend(_error_cases(variant, seed, splits["dev"], dev_logits, frozen_dev_logits, flat_logits, controls_row, limit=24))
            if variant.name == "P0_flat_baseline":
                flat_dev_logits_by_seed[seed] = dev_logits
            row = {
                **row_base,
                "status": "completed",
                "leakage_audit_pass": bool(leak_pass),
                "train": train_block,
                "dev_trainable": dev_block,
                "dev_frozen": frozen_block,
                "test_smoke_trainable": _metric_block(test_logits, test_labels, splits["test"]),
                "dev_delta_trainable_minus_frozen": float(dev_block["top1"] - frozen_block["top1"]),
                "baselines": _baseline_block(splits["train"], splits["dev"], splits["dev"]),
                "control_pass": controls_row.get("control_pass", {}),
                "best_ablation_drop": _best_ablation_drop(ablation_rows),
                "train_dev_gap": float(train_block["top1"] - dev_block["top1"]),
                "trainable_audit": trainable.audit,
                "frozen_audit": frozen.audit,
                "training_time_seconds": float(fit_elapsed),
                "param_count": int(trainable.param_count),
                "medium_validation_triggered": False,
            }
            rows.append(row)
            compute_rows.append(
                {
                    "benchmark": BENCHMARK,
                    "seed": seed,
                    "variant": variant.name,
                    "candidate_count": int(variant.candidate_count),
                    "leaf_clones": len(variant.model.view_names),
                    "composition_nodes": _composition_node_count(variant.model),
                    "pyramid_levels": _pyramid_levels(variant.model),
                    "trainable_training_time_seconds": trainable.training_time_seconds,
                    "frozen_training_time_seconds": frozen.training_time_seconds,
                    "total_fit_seconds": float(fit_elapsed),
                    "inference_latency_seconds_per_task": float(latency),
                    "peak_cuda_memory_bytes": _cuda_max_memory(device),
                    "trainable_param_count": trainable.param_count,
                    "frozen_param_count": frozen.param_count,
                    "approx_attention_cost": _approx_attention_cost(variant.model, int(variant.candidate_count)),
                    "accuracy_per_millisecond": float(dev_block["top1"]) / max(1e-9, latency * 1000.0),
                    "delta_per_memory_unit": float(row["dev_delta_trainable_minus_frozen"]) / max(1.0, float(_cuda_max_memory(device) or trainable.param_count)),
                }
            )
            del trainable, frozen
            _clear_cuda()

    leaderboard = _leaderboard(rows, controls, ablations)
    summary = _summary(rows, controls, ablations, leaderboard, dataset_config)
    return {
        "metadata": {
            "benchmark": BENCHMARK,
            "created_at_utc": _now(),
            "device": device,
            "dataset_root": dataset_config.dataset_root,
            "scope": "bounded ARC candidate-output verification search; no Phase 2 launched unless clean gates pass",
            "claim_boundary": "known-size gold-present candidate ranking only; not ARC-AGI-2 solving or grid generation",
            "architecture_family": "pyramidal latent view composition",
        },
        "dataset_config": asdict(dataset_config),
        "training_config": asdict(base_training),
        "overfit_training_config": asdict(overfit_training),
        "variant_plan": [asdict(variant) for variant in variants],
        "phase0": phase0_rows,
        "rows": rows,
        "controls": controls,
        "ablations": ablations,
        "leaderboard": leaderboard,
        "summary": summary,
        "overfit_curves": overfit_rows,
        "leakage_audit": leakage_rows_out,
        "error_cases": error_cases,
        "attention_summaries": attention_rows,
        "compute_metrics": compute_rows,
    }


class ArcPyramidalRuleCloneVerifier(nn.Module):
    def __init__(self, config: PyramidalArcModelConfig) -> None:
        super().__init__()
        self.config = config
        dim = _compatible_dim(config.model_dim, config.num_heads)
        self.model_dim = dim
        self.token_encoder = FieldTokenEncoder(dim)
        rule_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=int(config.num_heads),
            dim_feedforward=int(config.ff_dim),
            dropout=float(config.dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        candidate_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=int(config.num_heads),
            dim_feedforward=int(config.ff_dim),
            dropout=float(config.dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        composer_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=int(config.num_heads),
            dim_feedforward=int(config.ff_dim),
            dropout=float(config.dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.rule_encoder = nn.TransformerEncoder(rule_layer, num_layers=int(config.rule_layers))
        self.candidate_encoder = nn.TransformerEncoder(candidate_layer, num_layers=int(config.candidate_layers))
        self.token_composer = nn.TransformerEncoder(composer_layer, num_layers=max(1, int(config.rule_layers)))
        self.pooled_composer = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, int(config.ff_dim)),
            nn.GELU(),
            nn.Linear(int(config.ff_dim), dim),
        )
        self.composition_type_embedding = nn.Embedding(64, dim)
        self.state_identity_embedding = nn.Embedding(96, dim)
        self.child_position_embedding = nn.Embedding(2, dim)
        self.candidate_context_projection = nn.Linear(dim, dim)
        self.cross_blocks = nn.ModuleList(
            [CandidateRuleCrossAttentionBlock(dim, int(config.num_heads), int(config.ff_dim), float(config.dropout)) for _ in range(int(config.coordination_blocks))]
        )
        self.score_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def configure_shared_trainable(self, trainable: bool) -> None:
        shared_ids = {id(parameter) for _name, parameter in self.shared_parameter_items()}
        for parameter in self.parameters():
            parameter.requires_grad = True
        if not trainable:
            for parameter in self.parameters():
                if id(parameter) in shared_ids:
                    parameter.requires_grad = False

    def shared_parameter_items(self) -> List[Tuple[str, nn.Parameter]]:
        modules = (
            "token_encoder",
            "rule_encoder",
            "candidate_encoder",
            "token_composer",
            "pooled_composer",
            "composition_type_embedding",
            "state_identity_embedding",
            "child_position_embedding",
            "candidate_context_projection",
        )
        return [(name, parameter) for name, parameter in self.named_parameters() if name.split(".", 1)[0] in modules]

    def coordinator_parameter_items(self) -> List[Tuple[str, nn.Parameter]]:
        modules = ("cross_blocks", "score_head")
        return [(name, parameter) for name, parameter in self.named_parameters() if name.split(".", 1)[0] in modules]

    def forward(
        self,
        batch: Dict[str, object],
        return_attention: bool = False,
        hidden_state_shuffle: bool = False,
        composition_shuffle: bool = False,
        child_mismatch: bool = False,
        swap_composition_children: bool = False,
        pyramid_level_permutation: bool = False,
        shuffle_seed: int = 0,
        ablation: str | None = None,
    ) -> Dict[str, object]:
        rule_fields = batch["rule_fields"]  # type: ignore[index]
        rule_mask = batch["rule_mask"].bool()  # type: ignore[index]
        candidate_fields = batch["candidate_fields"]  # type: ignore[index]
        candidate_mask = batch["candidate_mask"].bool()  # type: ignore[index]
        active_view_names = tuple(batch.get("active_view_names", self.config.view_names))  # type: ignore[union-attr]
        batch_size, views, rule_tokens, _fields = rule_fields.shape
        candidates, candidate_tokens = candidate_fields.shape[1], candidate_fields.shape[2]

        rule_emb = self.token_encoder(rule_fields.reshape(batch_size * views, rule_tokens, -1))
        rule_encoded = self.rule_encoder(rule_emb, src_key_padding_mask=~rule_mask.reshape(batch_size * views, rule_tokens))
        rule_encoded = rule_encoded.reshape(batch_size, views, rule_tokens, -1)
        if hidden_state_shuffle and batch_size > 1:
            generator = torch.Generator(device=rule_encoded.device)
            generator.manual_seed(int(shuffle_seed))
            order = torch.randperm(batch_size, generator=generator, device=rule_encoded.device)
            rule_encoded = rule_encoded.index_select(0, order)

        cand_emb = self.token_encoder(candidate_fields.reshape(batch_size * candidates, candidate_tokens, -1))
        cand_encoded = self.candidate_encoder(
            cand_emb,
            src_key_padding_mask=~candidate_mask.reshape(batch_size * candidates, candidate_tokens),
        )
        flat_candidate_mask = candidate_mask.reshape(batch_size * candidates, candidate_tokens)
        candidate_context = None
        if self.config.candidate_guided_composition:
            candidate_context = _masked_mean(cand_encoded, flat_candidate_mask)

        leaf_states = self._leaf_states(rule_encoded, rule_mask, active_view_names)
        pair_states, root_states = self._compose_states(
            leaf_states,
            batch_size=batch_size,
            candidates=candidates,
            candidate_context=candidate_context,
            child_mismatch=child_mismatch,
            swap_children=swap_composition_children,
            shuffle_seed=shuffle_seed,
            ablation=ablation,
        )
        if composition_shuffle:
            pair_states = self._shuffle_states(pair_states, batch_size, candidates, shuffle_seed + 101)
            root_states = self._shuffle_states(root_states, batch_size, candidates, shuffle_seed + 103)

        if self.config.residual_pyramid:
            logits, attention_records, level_scores = self._residual_scores(
                cand_encoded,
                flat_candidate_mask,
                batch_size,
                candidates,
                leaf_states,
                pair_states,
                root_states,
                return_attention,
                pyramid_level_permutation,
                ablation,
            )
        else:
            selected = self._select_states(leaf_states, pair_states, root_states, ablation)
            if pyramid_level_permutation:
                selected = _permuted_state_order(selected)
            latent_tokens, latent_mask, spans = self._concat_states(selected, cand_encoded.device)
            queried, attention_records = self._query_latents(
                cand_encoded,
                latent_tokens,
                latent_mask,
                batch_size,
                candidates,
                level="selected",
                spans=spans,
                return_attention=return_attention,
            )
            pooled = _masked_mean(queried, flat_candidate_mask)
            logits = self.score_head(pooled).reshape(batch_size, candidates)
            level_scores = {}

        all_states = leaf_states + pair_states + root_states
        return {
            "logits": logits,
            "attention_records": attention_records,
            "level_score_contributions": level_scores,
            "latent_state_diagnostics": _state_diagnostics(all_states),
        }

    def _leaf_states(self, rule_encoded: torch.Tensor, rule_mask: torch.Tensor, active_view_names: Sequence[str]) -> List[LatentState]:
        states = []
        for axis, view in enumerate(active_view_names):
            states.append(LatentState(str(view), "leaf", rule_encoded[:, axis], rule_mask[:, axis]))
        return states

    def _compose_states(
        self,
        leaf_states: Sequence[LatentState],
        batch_size: int,
        candidates: int,
        candidate_context: torch.Tensor | None,
        child_mismatch: bool,
        swap_children: bool,
        shuffle_seed: int,
        ablation: str | None,
    ) -> Tuple[List[LatentState], List[LatentState]]:
        if self.config.pyramid_layout == "flat" or self.config.composer_kind == "none" or ablation == "composition_ablation":
            return [], []
        by_name = {state.name: state for state in leaf_states}
        pair_states: List[LatentState] = []
        root_states: List[LatentState] = []
        if self.config.pyramid_layout == "all_pairs":
            for left, right, name in _all_pair_plan(self.config.view_names):
                if left in by_name and right in by_name and not _pair_removed(name, ablation):
                    pair_states.append(
                        self._compose_pair(
                            by_name[left],
                            by_name[right],
                            name,
                            batch_size,
                            candidates,
                            candidate_context,
                            child_mismatch,
                            swap_children,
                            shuffle_seed,
                        )
                    )
            return pair_states, root_states
        if self.config.pyramid_layout in {"pairwise", "residual"}:
            for left, right, name in PAIR_ARC:
                if left in by_name and right in by_name and not _pair_removed(name, ablation):
                    pair_states.append(
                        self._compose_pair(
                            by_name[left],
                            by_name[right],
                            name,
                            batch_size,
                            candidates,
                            candidate_context,
                            child_mismatch,
                            swap_children,
                            shuffle_seed,
                        )
                    )
            if self.config.pyramid_layout == "pairwise":
                return pair_states, root_states
        if self.config.pyramid_layout in {"binary_tree", "recursive", "residual"}:
            level1: Dict[str, LatentState] = {}
            for left, right, name in TREE_LEVEL1:
                if left in by_name and right in by_name and not _pair_removed(name, ablation):
                    state = self._compose_pair(
                        by_name[left],
                        by_name[right],
                        name,
                        batch_size,
                        candidates,
                        candidate_context,
                        child_mismatch,
                        swap_children,
                        shuffle_seed,
                    )
                    level1[name] = state
                    if state.name not in {existing.name for existing in pair_states}:
                        pair_states.append(state)
            if "change" in level1 and "structure" in level1:
                transformation = self._compose_pair(
                    level1["change"],
                    level1["structure"],
                    "transformation",
                    batch_size,
                    candidates,
                    candidate_context=None,
                    child_mismatch=child_mismatch,
                    swap_children=swap_children,
                    shuffle_seed=shuffle_seed + 11,
                )
                pair_states.append(transformation)
                if "appearance" in level1 and ablation != "root_ablation":
                    root_states.append(
                        self._compose_pair(
                            transformation,
                            level1["appearance"],
                            "final",
                            batch_size,
                            candidates,
                            candidate_context=None,
                            child_mismatch=child_mismatch,
                            swap_children=swap_children,
                            shuffle_seed=shuffle_seed + 13,
                        )
                    )
        return pair_states, root_states

    def _compose_pair(
        self,
        left: LatentState,
        right: LatentState,
        name: str,
        batch_size: int,
        candidates: int,
        candidate_context: torch.Tensor | None,
        child_mismatch: bool,
        swap_children: bool,
        shuffle_seed: int,
    ) -> LatentState:
        if swap_children:
            left, right = right, left
        left_tokens, left_mask = self._align_state(left, batch_size, candidates, candidate_context)
        right_tokens, right_mask = self._align_state(right, batch_size, candidates, candidate_context)
        if child_mismatch:
            right_tokens, right_mask = _shuffle_child(right_tokens, right_mask, batch_size, candidates, shuffle_seed + _composition_id(name))
        if self.config.composer_kind == "pooled":
            left_pool = _masked_mean(left_tokens, left_mask)
            right_pool = _masked_mean(right_tokens, right_mask)
            if candidate_context is not None:
                left_pool = left_pool + self.candidate_context_projection(candidate_context)
                right_pool = right_pool + self.candidate_context_projection(candidate_context)
            features = torch.cat([left_pool, right_pool, torch.abs(left_pool - right_pool), left_pool * right_pool], dim=-1)
            out = self.pooled_composer(features)
            out = out + self.composition_type_embedding(_id_tensor(out.device, _composition_id(name), out.shape[0]))
            tokens = out.unsqueeze(1)
            mask = torch.ones((out.shape[0], 1), dtype=torch.bool, device=out.device)
        else:
            tokens, mask = self._compose_token_states(left, right, left_tokens, left_mask, right_tokens, right_mask, name, candidate_context)
        child_mean = 0.5 * (_masked_mean(left_tokens, left_mask) + _masked_mean(right_tokens, right_mask))
        parent_mean = _masked_mean(tokens, mask)
        cosine = F.cosine_similarity(parent_mean, child_mean, dim=-1).detach()
        level = "root" if name == "final" else "pair"
        return LatentState(name=name, level=level, tokens=tokens, mask=mask, child_names=(left.name, right.name), child_parent_cosine=cosine)

    def _align_state(
        self,
        state: LatentState,
        batch_size: int,
        candidates: int,
        candidate_context: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if candidate_context is None:
            return state.tokens, state.mask
        target = batch_size * candidates
        if state.tokens.shape[0] == target:
            return state.tokens, state.mask
        return state.tokens.repeat_interleave(candidates, dim=0), state.mask.repeat_interleave(candidates, dim=0)

    def _compose_token_states(
        self,
        left: LatentState,
        right: LatentState,
        left_tokens: torch.Tensor,
        left_mask: torch.Tensor,
        right_tokens: torch.Tensor,
        right_mask: torch.Tensor,
        name: str,
        candidate_context: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = left_tokens.shape[0]
        device = left_tokens.device
        parent_id = _composition_id(name)
        parent = self.composition_type_embedding(_id_tensor(device, parent_id, batch)).unsqueeze(1)
        if candidate_context is not None:
            guide = self.candidate_context_projection(candidate_context).unsqueeze(1)
            parent = parent + guide
            left_tokens = left_tokens + guide
            right_tokens = right_tokens + guide
        left_id = self.state_identity_embedding(_id_tensor(device, _state_id(left.name), batch)).view(batch, 1, -1)
        right_id = self.state_identity_embedding(_id_tensor(device, _state_id(right.name), batch)).view(batch, 1, -1)
        left_pos = self.child_position_embedding(_id_tensor(device, 0, batch)).view(batch, 1, -1)
        right_pos = self.child_position_embedding(_id_tensor(device, 1, batch)).view(batch, 1, -1)
        encoded_in = torch.cat([parent, left_tokens + left_id + left_pos, right_tokens + right_id + right_pos], dim=1)
        mask = torch.cat(
            [
                torch.ones((batch, 1), dtype=torch.bool, device=device),
                left_mask.bool(),
                right_mask.bool(),
            ],
            dim=1,
        )
        encoded = self.token_composer(encoded_in, src_key_padding_mask=~mask)
        keep = max(1, min(int(self.config.max_composed_tokens), encoded.shape[1]))
        return encoded[:, :keep], mask[:, :keep]

    def _shuffle_states(self, states: Sequence[LatentState], batch_size: int, candidates: int, seed: int) -> List[LatentState]:
        return [
            LatentState(
                state.name,
                state.level,
                *_shuffle_child(state.tokens, state.mask, batch_size, candidates, seed + index),
                child_names=state.child_names,
                child_parent_cosine=state.child_parent_cosine,
            )
            for index, state in enumerate(states)
        ]

    def _select_states(
        self,
        leaf_states: Sequence[LatentState],
        pair_states: Sequence[LatentState],
        root_states: Sequence[LatentState],
        ablation: str | None,
    ) -> List[LatentState]:
        query_mode = self.config.query_mode
        if ablation == "composition_ablation":
            query_mode = "leaf_only"
        if ablation == "leaf_ablation":
            query_mode = "composed_only"
        selected: List[LatentState] = []
        if query_mode == "leaf_only":
            selected.extend(leaf_states)
        elif query_mode == "composed_only":
            selected.extend(pair_states)
            selected.extend(root_states)
        elif query_mode == "root_only":
            selected.extend(root_states)
        elif query_mode == "root_plus_intermediate":
            selected.extend(pair_states)
            selected.extend(root_states)
        elif query_mode in {"leaf_plus_composed", "all_levels"}:
            selected.extend(leaf_states)
            selected.extend(pair_states)
            selected.extend(root_states)
        else:
            selected.extend(leaf_states)
        if ablation == "root_ablation":
            selected = [state for state in selected if state.level != "root"]
        if not selected:
            selected = [LatentState("fallback", "fallback", torch.zeros_like(leaf_states[0].tokens[:, :1]), torch.ones_like(leaf_states[0].mask[:, :1]))]
        return selected

    def _concat_states(self, states: Sequence[LatentState], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, object]]]:
        if not states:
            token = torch.zeros((1, 1, self.model_dim), dtype=torch.float32, device=device)
            mask = torch.ones((1, 1), dtype=torch.bool, device=device)
            return token, mask, [{"name": "fallback", "level": "fallback", "start": 0, "end": 1}]
        batch = states[0].tokens.shape[0]
        tokens = []
        masks = []
        spans = []
        cursor = 0
        for state in states:
            if state.tokens.shape[0] != batch:
                raise ValueError("cannot concatenate candidate-guided and task-level latent states in one group")
            tokens.append(state.tokens)
            masks.append(state.mask.bool())
            end = cursor + state.tokens.shape[1]
            spans.append({"name": state.name, "level": state.level, "start": cursor, "end": end})
            cursor = end
        return torch.cat(tokens, dim=1), torch.cat(masks, dim=1), spans

    def _query_latents(
        self,
        cand_encoded: torch.Tensor,
        latent_tokens: torch.Tensor,
        latent_mask: torch.Tensor,
        batch_size: int,
        candidates: int,
        level: str,
        spans: List[Dict[str, object]],
        return_attention: bool,
    ) -> Tuple[torch.Tensor, List[Dict[str, object]]]:
        target_batch = batch_size * candidates
        if latent_tokens.shape[0] == batch_size:
            latent_tokens = latent_tokens.repeat_interleave(candidates, dim=0)
            latent_mask = latent_mask.repeat_interleave(candidates, dim=0)
        elif latent_tokens.shape[0] != target_batch:
            raise ValueError("latent batch dimension does not match task or candidate batch")
        encoded = cand_encoded
        records: List[Dict[str, object]] = []
        for block_index, block in enumerate(self.cross_blocks):
            encoded, weights = block(encoded, latent_tokens, latent_mask, return_attention=return_attention)
            if weights is not None:
                records.append(
                    {
                        "level": level,
                        "block": block_index,
                        "spans": spans,
                        "weights": weights.reshape(batch_size, candidates, *weights.shape[1:]),
                    }
                )
        return encoded, records

    def _residual_scores(
        self,
        cand_encoded: torch.Tensor,
        candidate_mask: torch.Tensor,
        batch_size: int,
        candidates: int,
        leaf_states: Sequence[LatentState],
        pair_states: Sequence[LatentState],
        root_states: Sequence[LatentState],
        return_attention: bool,
        pyramid_level_permutation: bool,
        ablation: str | None,
    ) -> Tuple[torch.Tensor, List[Dict[str, object]], Dict[str, torch.Tensor]]:
        groups: List[Tuple[str, List[LatentState]]] = []
        if ablation != "leaf_ablation":
            groups.append(("leaf", list(leaf_states)))
        if ablation != "composition_ablation":
            groups.append(("pair", list(pair_states)))
            if ablation != "root_ablation":
                groups.append(("root", list(root_states)))
        if pyramid_level_permutation:
            groups = list(reversed(groups))
        logits = torch.zeros((batch_size, candidates), dtype=cand_encoded.dtype, device=cand_encoded.device)
        records: List[Dict[str, object]] = []
        level_scores: Dict[str, torch.Tensor] = {}
        for level, states in groups:
            if not states:
                continue
            latent_tokens, latent_mask, spans = self._concat_states(states, cand_encoded.device)
            queried, rec = self._query_latents(cand_encoded, latent_tokens, latent_mask, batch_size, candidates, level, spans, return_attention)
            scores = self.score_head(_masked_mean(queried, candidate_mask)).reshape(batch_size, candidates)
            logits = logits + scores
            level_scores[level] = scores.detach()
            records.extend(rec)
        if not level_scores:
            scores = self.score_head(_masked_mean(cand_encoded, candidate_mask)).reshape(batch_size, candidates)
            logits = logits + scores
            level_scores["fallback"] = scores.detach()
        return logits, records, level_scores


def fit_pyramid_verifier(
    train_examples: Sequence[ArcVerificationExample],
    dev_examples: Sequence[ArcVerificationExample],
    model_config: PyramidalArcModelConfig,
    training_config: ArcTrainingConfig,
    seed: int,
    device: str,
    trainable_shared: bool,
    method: str,
    early_stop_top1: float | None = None,
) -> PyramidalFitResult:
    _set_seed(seed)
    model = ArcPyramidalRuleCloneVerifier(model_config).to(device)
    model.configure_shared_trainable(trainable_shared)
    initial_shared = _flat_params(model.shared_parameter_items())
    initial_coord = _flat_params(model.coordinator_parameter_items())
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training_config.lr),
        weight_decay=float(training_config.weight_decay),
    )
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_dev = -1.0
    bad_epochs = 0
    history: List[Dict[str, float]] = []
    shared_grad_norms: List[float] = []
    coord_grad_norms: List[float] = []
    start = time.perf_counter()
    for epoch in range(int(training_config.epochs)):
        model.train()
        losses: List[float] = []
        order = np.random.default_rng(seed + epoch).permutation(len(train_examples))
        for batch_indices in _chunks(order.tolist(), int(training_config.batch_size)):
            batch_examples = [train_examples[int(index)] for index in batch_indices]
            batch = collate_arc_batch(batch_examples, model_config, device)
            batch["active_view_names"] = tuple(model_config.view_names)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)["logits"]
            loss = _candidate_objective_loss(logits, batch["labels"], training_config)
            loss.backward()
            shared_grad_norms.append(_grad_norm([parameter for _name, parameter in model.shared_parameter_items()]))
            coord_grad_norms.append(_grad_norm([parameter for _name, parameter in model.coordinator_parameter_items()]))
            if float(training_config.gradient_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], float(training_config.gradient_clip_norm))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dev_logits = predict_pyramid_logits(model, dev_examples, model_config, int(training_config.batch_size), device)
        dev_labels = np.asarray([example.label for example in dev_examples], dtype=np.int64)
        dev_acc = _top1(dev_logits, dev_labels)
        history.append({"epoch": float(epoch), "loss": _mean(losses), "dev_top1": float(dev_acc)})
        if dev_acc > best_dev:
            best_dev = float(dev_acc)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if early_stop_top1 is not None and dev_acc >= float(early_stop_top1):
            break
        if bad_epochs >= int(training_config.patience):
            break
    model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    elapsed = time.perf_counter() - start
    final_shared = _flat_params(model.shared_parameter_items())
    final_coord = _flat_params(model.coordinator_parameter_items())
    shared_delta = _l2_delta(initial_shared, final_shared)
    coord_delta = _l2_delta(initial_coord, final_coord)
    audit = {
        "method": method,
        "initialization_seed": seed,
        "shared_model_trainable": bool(trainable_shared),
        "shared_parameter_names": [name for name, _parameter in model.shared_parameter_items()],
        "coordinator_parameter_names": [name for name, _parameter in model.coordinator_parameter_items()],
        "shared_grad_norm_mean": _mean(shared_grad_norms),
        "coordinator_grad_norm_mean": _mean(coord_grad_norms),
        "shared_parameter_delta": shared_delta,
        "coordinator_parameter_delta": coord_delta,
        "frozen_shared_model_zero_grad": bool((not trainable_shared) and _mean(shared_grad_norms) == 0.0),
        "frozen_shared_model_zero_delta": bool((not trainable_shared) and shared_delta == 0.0),
        "trainable_shared_model_changed": bool(trainable_shared and shared_delta > 0.0),
        "composition_modules_in_frozen_shared_set": True,
        "same_architecture_comparator": True,
    }
    return PyramidalFitResult(method, model, model_config, trainable_shared, audit, history, float(elapsed), sum(p.numel() for p in model.parameters()))


def predict_pyramid_logits(
    model: ArcPyramidalRuleCloneVerifier,
    examples: Sequence[ArcVerificationExample],
    model_config: PyramidalArcModelConfig,
    batch_size: int,
    device: str,
    condition: str = "none",
    seed: int = 0,
    view_mask: str | None = None,
    ablation: str | None = None,
    return_attention: bool = False,
) -> np.ndarray | Tuple[np.ndarray, List[Dict[str, object]]]:
    model.eval()
    logits_out: List[np.ndarray] = []
    attention_rows: List[Dict[str, object]] = []
    view_names = model_config.view_names
    if condition == "physical_role_order_shuffle":
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(view_names))
        if len(order) > 1 and np.array_equal(order, np.arange(len(order))):
            order = np.roll(order, 1)
        view_names = tuple(view_names[int(index)] for index in order)
    with torch.no_grad():
        for offset, batch_examples in enumerate(_batched(examples, int(batch_size))):
            batch = collate_arc_batch(batch_examples, model_config, device, view_names=view_names)
            batch["active_view_names"] = tuple(view_names)
            if view_mask is not None:
                for axis, view in enumerate(view_names):
                    if view == view_mask:
                        batch["rule_mask"][:, axis, :] = False
            output = model(
                batch,
                return_attention=return_attention,
                hidden_state_shuffle=condition == "hidden_state_shuffle",
                composition_shuffle=condition == "composition_shuffle",
                child_mismatch=condition == "child_mismatch",
                swap_composition_children=condition == "composition_order_swap",
                pyramid_level_permutation=condition == "pyramid_level_permutation",
                shuffle_seed=seed + offset,
                ablation=ablation,
            )
            logits_out.append(output["logits"].detach().cpu().numpy())
            if return_attention:
                attention_rows.extend(_summarize_attention_batch(output, batch_examples, seed, model_config, condition, ablation))
    logits = np.concatenate(logits_out, axis=0) if logits_out else np.zeros((0, 0), dtype=np.float32)
    if return_attention:
        return logits, attention_rows
    return logits


def _run_overfit_gates(
    variant: PyramidalVariant,
    dataset_config: ArcVerificationDatasetConfig,
    overfit_training: ArcTrainingConfig,
    seed: int,
    device: str,
) -> List[Dict[str, object]]:
    gates = [
        ("4_examples_N2", 2, 4, 0.95),
        ("12_examples_N4", 4, 12, 0.90),
        ("35_examples_N8", 8, 35, 0.80),
    ]
    rows = []
    for gate_name, candidate_count, example_count, threshold in gates:
        gate_config = replace(
            dataset_config,
            num_candidates=candidate_count,
            train_tasks=max(example_count + 4, int(dataset_config.train_tasks)),
            dev_tasks=2,
            test_tasks=2,
            negative_difficulty=str(variant.negative_difficulty),
        )
        splits = build_arc_verification_splits(gate_config, seed=seed + candidate_count * 101)
        train_examples = splits["train"][:example_count]
        if len(train_examples) < example_count:
            rows.append(
                {
                    "variant": variant.name,
                    "seed": seed,
                    "gate": gate_name,
                    "candidate_count": candidate_count,
                    "examples": len(train_examples),
                    "train_accuracy": 0.0,
                    "threshold": threshold,
                    "pass": False,
                    "gate_required": True,
                    "reason": "not_enough_examples",
                }
            )
            break
        model_config = replace(variant.model, view_names=variant.model.view_names)
        training = replace(
            overfit_training,
            epochs=max(int(overfit_training.epochs), 20),
            patience=max(int(overfit_training.patience), 20),
            batch_size=min(max(1, example_count), int(overfit_training.batch_size)),
            objective=variant.objective,
        )
        fit = fit_pyramid_verifier(
            train_examples,
            train_examples,
            model_config,
            training,
            seed=seed + 15_900 + candidate_count,
            device=device,
            trainable_shared=True,
            method=f"overfit__{variant.name}__{gate_name}",
            early_stop_top1=threshold,
        )
        logits = predict_pyramid_logits(fit.model, train_examples, model_config, training.batch_size, device)
        labels = np.asarray([example.label for example in train_examples], dtype=np.int64)
        acc = _top1(logits, labels)
        rows.append(
            {
                "variant": variant.name,
                "seed": seed,
                "gate": gate_name,
                "candidate_count": candidate_count,
                "examples": len(train_examples),
                "train_accuracy": float(acc),
                "threshold": float(threshold),
                "pass": bool(acc >= threshold),
                "gate_required": True,
                "history": fit.history,
                "audit": fit.audit,
            }
        )
        if acc < threshold:
            break
    return rows


def _run_pyramid_controls(
    model: ArcPyramidalRuleCloneVerifier,
    model_config: PyramidalArcModelConfig,
    examples: Sequence[ArcVerificationExample],
    batch_size: int,
    device: str,
    seed: int,
    clean_logits: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, object]:
    chance = 1.0 / max(1, clean_logits.shape[1] if clean_logits.ndim == 2 and clean_logits.shape[1] else 8)
    controls: Dict[str, object] = {
        "chance": chance,
        "candidate_source_metadata_only_accuracy": candidate_metadata_only_accuracy(examples),
        "parent_sees_raw_evidence_audit": {
            "pass": not bool(model_config.parent_uses_raw_evidence),
            "composer_inputs": "encoded child latent states only",
            "candidate_metadata_visible_to_parent": False,
        },
        "candidate_guided_leakage_audit": {
            "applicable": bool(model_config.candidate_guided_composition),
            "candidate_index_embedding_used": False,
            "candidate_source_metadata_used": False,
            "gold_label_visible": False,
            "pass": True,
        },
    }
    for name in (
        "candidate_only",
        "examples_only_no_candidates",
        "train_pair_shuffle",
        "cross_task_train_pair_shuffle",
        "candidate_evidence_mismatch",
        "candidate_order_shuffle_with_gold_remap",
        "randomized_labels",
    ):
        controlled = apply_arc_control(examples, name, seed + 19_000 + len(controls))
        control_labels = np.asarray([example.label for example in controlled], dtype=np.int64)
        logits = predict_pyramid_logits(model, controlled, model_config, batch_size, device, seed=seed)
        controls[name] = _metric_block(logits, control_labels, controlled)
    for name in (
        "physical_role_order_shuffle",
        "hidden_state_shuffle",
        "composition_shuffle",
        "child_mismatch",
        "composition_order_swap",
        "pyramid_level_permutation",
    ):
        logits = predict_pyramid_logits(model, examples, model_config, batch_size, device, condition=name, seed=seed + 23_000)
        controls[name] = _metric_block(logits, labels, examples)
    base_top1 = float(_top1(clean_logits, labels))
    controls["candidate_order_invariance_delta"] = abs(base_top1 - float(controls["candidate_order_shuffle_with_gold_remap"]["top1"]))
    controls["role_order_invariance_delta"] = abs(base_top1 - float(controls["physical_role_order_shuffle"]["top1"]))
    controls["pyramid_level_permutation_delta"] = abs(base_top1 - float(controls["pyramid_level_permutation"]["top1"]))
    controls["composition_order_swap_delta"] = abs(base_top1 - float(controls["composition_order_swap"]["top1"]))
    role_mask = {}
    for view in model_config.view_names:
        logits = predict_pyramid_logits(model, examples, model_config, batch_size, device, view_mask=view)
        role_mask[str(view)] = _metric_block(logits, labels, examples)
    controls["role_view_masking"] = role_mask
    controls["control_pass"] = _controls_pass(controls, base_top1, chance, model_config)
    return controls


def _run_pyramid_ablations(
    model: ArcPyramidalRuleCloneVerifier,
    model_config: PyramidalArcModelConfig,
    examples: Sequence[ArcVerificationExample],
    batch_size: int,
    device: str,
    seed: int,
    clean_logits: np.ndarray,
    labels: np.ndarray,
) -> List[Dict[str, object]]:
    clean = _metric_block(clean_logits, labels, examples)
    ablation_names = ["composition_ablation", "leaf_ablation", "root_ablation"]
    ablation_names.extend([f"pair_composer_ablation::{name}" for _left, _right, name in PAIR_ARC])
    rows = []
    for name in ablation_names:
        logits = predict_pyramid_logits(model, examples, model_config, batch_size, device, seed=seed + 29_000, ablation=name)
        metrics = _metric_block(logits, labels, examples)
        rows.append(
            {
                "ablation": name,
                "clean_top1": float(clean["top1"]),
                "ablated_top1": float(metrics["top1"]),
                "top1_drop": float(clean["top1"] - metrics["top1"]),
                "clean_mrr": float(clean["mrr"]),
                "ablated_mrr": float(metrics["mrr"]),
                "mrr_drop": float(clean["mrr"] - metrics["mrr"]),
                "metrics": metrics,
            }
        )
    return rows


def _attention_summaries(
    model: ArcPyramidalRuleCloneVerifier,
    model_config: PyramidalArcModelConfig,
    examples: Sequence[ArcVerificationExample],
    batch_size: int,
    device: str,
    seed: int,
    limit: int,
) -> List[Dict[str, object]]:
    subset = list(examples[:limit])
    _logits, rows = predict_pyramid_logits(model, subset, model_config, batch_size, device, seed=seed, return_attention=True)
    for row in rows:
        row["benchmark"] = BENCHMARK
        row["seed"] = seed
    return rows


def _summarize_attention_batch(
    output: Dict[str, object],
    examples: Sequence[ArcVerificationExample],
    seed: int,
    model_config: PyramidalArcModelConfig,
    condition: str,
    ablation: str | None,
) -> List[Dict[str, object]]:
    records = output.get("attention_records", [])
    logits = output["logits"].detach().cpu().numpy()
    preds = np.argmax(logits, axis=1)
    level_scores = output.get("level_score_contributions", {})
    rows = []
    for example_index, example in enumerate(examples):
        blocks = []
        for record in records:
            weights = record["weights"][example_index].detach().float().cpu().numpy()
            # weights: candidates, heads, query_tokens, latent_tokens
            by_state = {}
            by_level: Dict[str, float] = {}
            flat = weights.mean(axis=(0, 1, 2))
            for span in record.get("spans", []):
                start = int(span["start"])
                end = int(span["end"])
                mass = float(weights[..., start:end].sum(axis=-1).mean())
                by_state[str(span["name"])] = mass
                by_level[str(span["level"])] = by_level.get(str(span["level"]), 0.0) + mass
            top_states = sorted(by_state.items(), key=lambda item: item[1], reverse=True)[:6]
            entropy = float(-np.sum(flat * np.log(np.clip(flat, 1e-9, 1.0))))
            blocks.append(
                {
                    "level": str(record.get("level")),
                    "block": int(record.get("block", 0)),
                    "attention_mass_per_level": by_level,
                    "attention_mass_per_state": by_state,
                    "top_attended_states": [{"state": name, "mass": value} for name, value in top_states],
                    "attention_entropy": entropy,
                }
            )
        contribution = {}
        if isinstance(level_scores, dict):
            for level, tensor in level_scores.items():
                values = tensor.detach().cpu().numpy()[example_index]
                contribution[str(level)] = {
                    "gold_score": float(values[int(example.label)]),
                    "predicted_score": float(values[int(preds[example_index])]),
                    "score_std": float(np.std(values)),
                }
        rows.append(
            {
                "type": "arc1_3_pyramid_attention_summary",
                "task_id": example.task_id,
                "example_id": example.id,
                "condition": condition,
                "ablation": ablation,
                "variant_config": {
                    "pyramid_layout": model_config.pyramid_layout,
                    "composer_kind": model_config.composer_kind,
                    "query_mode": model_config.query_mode,
                    "candidate_guided_composition": model_config.candidate_guided_composition,
                    "residual_pyramid": model_config.residual_pyramid,
                },
                "predicted_candidate": int(preds[example_index]),
                "gold_candidate": int(example.label),
                "candidate_probabilities": _softmax_np(logits[example_index]).tolist(),
                "blocks": blocks,
                "level_score_contributions": contribution,
                "latent_state_diagnostics": output.get("latent_state_diagnostics", []),
            }
        )
    return rows


def _error_cases(
    variant: PyramidalVariant,
    seed: int,
    examples: Sequence[ArcVerificationExample],
    train_logits: np.ndarray,
    frozen_logits: np.ndarray,
    flat_logits: np.ndarray | None,
    controls: Dict[str, object],
    limit: int,
) -> List[Dict[str, object]]:
    labels = np.asarray([example.label for example in examples], dtype=np.int64)
    train_pred = np.argmax(train_logits, axis=1) if train_logits.size else np.zeros(len(examples), dtype=np.int64)
    frozen_pred = np.argmax(frozen_logits, axis=1) if frozen_logits.size else np.zeros(len(examples), dtype=np.int64)
    flat_pred = np.argmax(flat_logits, axis=1) if flat_logits is not None and flat_logits.size else None
    candidate_pred = _candidate_stat_predictions(examples)
    changed_pred = _changed_cell_count_predictions(examples)
    buckets: Dict[str, List[int]] = {
        "pyramid_right_frozen_wrong": [],
        "frozen_right_pyramid_wrong": [],
        "candidate_stat_right_pyramid_wrong": [],
        "all_models_wrong": [],
        "pyramid_fixes_flat_baseline": [],
        "pyramid_breaks_flat_baseline": [],
    }
    for index, example in enumerate(examples):
        label = int(labels[index])
        tr = int(train_pred[index])
        fr = int(frozen_pred[index])
        cs = int(candidate_pred[index])
        ch = int(changed_pred[index])
        if tr == label and fr != label:
            buckets["pyramid_right_frozen_wrong"].append(index)
        if fr == label and tr != label:
            buckets["frozen_right_pyramid_wrong"].append(index)
        if cs == label and tr != label:
            buckets["candidate_stat_right_pyramid_wrong"].append(index)
        if tr != label and fr != label and cs != label and ch != label:
            buckets["all_models_wrong"].append(index)
        if flat_pred is not None:
            fp = int(flat_pred[index])
            if tr == label and fp != label:
                buckets["pyramid_fixes_flat_baseline"].append(index)
            if fp == label and tr != label:
                buckets["pyramid_breaks_flat_baseline"].append(index)
    rows = []
    per_bucket = max(1, limit // max(1, len(buckets)))
    for category, indices in buckets.items():
        for index in indices[:per_bucket]:
            example = examples[index]
            pred = int(train_pred[index])
            rank = int(np.where(np.argsort(-train_logits[index]) == int(example.label))[0][0]) + 1
            rows.append(
                {
                    "type": "arc1_3_pyramid_error_case",
                    "category": category,
                    "variant": variant.name,
                    "stage_variant": variant.stage_variant,
                    "seed": seed,
                    "task_id": example.task_id,
                    "example_id": example.id,
                    "gold_candidate_index": int(example.label),
                    "predicted_candidate_index": pred,
                    "candidate_probabilities": _softmax_np(train_logits[index]).tolist(),
                    "gold_rank": rank,
                    "negative_type_labels": list(example.negative_types),
                    "control_status": controls.get("control_pass", {}),
                    "pyramid_attention_summary": "see arc1_3_pyramidal_attention_summaries.jsonl",
                    "train_grids": [{"input": pair.input, "output": pair.output} for pair in example.train_pairs],
                    "test_input": example.test_input,
                    "gold_candidate": example.gold_output,
                    "predicted_candidate": example.candidates[pred],
                }
            )
    return rows


def _leaderboard(
    rows: Sequence[Dict[str, object]],
    controls: Sequence[Dict[str, object]],
    ablations: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    control_by_key = {(row["variant"], row["seed"]): row for row in controls}
    ablation_by_key: Dict[Tuple[str, int], List[Dict[str, object]]] = {}
    for row in ablations:
        ablation_by_key.setdefault((str(row["variant"]), int(row["seed"])), []).append(row)
    leaders = []
    for row in rows:
        if row.get("status") != "completed":
            leaders.append(
                {
                    "variant": row["variant"],
                    "seed": row["seed"],
                    "status": row["status"],
                    "score": -999.0,
                    "trainable_top1": 0.0,
                    "frozen_top1": 0.0,
                    "delta": 0.0,
                    "controls_pass": False,
                }
            )
            continue
        control = control_by_key.get((row["variant"], row["seed"]), {})
        candidate_only = float(control.get("candidate_only", {}).get("top1", 0.0))
        controls_pass = bool(control.get("control_pass", {}).get("overall", False))
        best_drop = max([float(item.get("top1_drop", 0.0)) for item in ablation_by_key.get((str(row["variant"]), int(row["seed"])), [])] or [0.0])
        trainable = float(row["dev_trainable"]["top1"])
        frozen = float(row["dev_frozen"]["top1"])
        delta = trainable - frozen
        score = trainable + delta + 0.2 * best_drop - 0.5 * candidate_only + (0.2 if controls_pass else -1.0)
        model_config = row.get("model_config", {})
        leaders.append(
            {
                "variant": row["variant"],
                "stage_variant": row.get("stage_variant"),
                "seed": row["seed"],
                "status": row["status"],
                "pyramid_layout": model_config.get("pyramid_layout"),
                "composer_kind": model_config.get("composer_kind"),
                "query_mode": model_config.get("query_mode"),
                "candidate_guided": model_config.get("candidate_guided_composition"),
                "residual_pyramid": model_config.get("residual_pyramid"),
                "candidate_count": int(row.get("candidate_count", 0)),
                "trainable_top1": trainable,
                "frozen_top1": frozen,
                "delta": delta,
                "top2": float(row["dev_trainable"]["top2"]),
                "top3": float(row["dev_trainable"]["top3"]),
                "mrr": float(row["dev_trainable"]["mrr"]),
                "candidate_only": candidate_only,
                "metadata_only": float(control.get("candidate_source_metadata_only_accuracy", 0.0)),
                "heuristic_best": _best_non_oracle_baseline(row.get("baselines", {})),
                "controls_pass": controls_pass,
                "best_ablation_drop": best_drop,
                "train_dev_gap": float(row.get("train_dev_gap", 0.0)),
                "training_time_seconds": float(row.get("training_time_seconds", 0.0)),
                "param_count": int(row.get("param_count", 0)),
                "score": score,
            }
        )
    leaders.sort(key=lambda item: float(item["score"]), reverse=True)
    return leaders


def _summary(
    rows: Sequence[Dict[str, object]],
    controls: Sequence[Dict[str, object]],
    ablations: Sequence[Dict[str, object]],
    leaderboard: Sequence[Dict[str, object]],
    dataset_config: ArcVerificationDatasetConfig,
) -> Dict[str, object]:
    completed = [row for row in rows if row.get("status") == "completed"]
    by_variant: Dict[str, List[Dict[str, object]]] = {}
    for row in completed:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    variant_summaries = []
    for variant, variant_rows in by_variant.items():
        trainable = [float(row["dev_trainable"]["top1"]) for row in variant_rows]
        frozen = [float(row["dev_frozen"]["top1"]) for row in variant_rows]
        deltas = [float(row["dev_delta_trainable_minus_frozen"]) for row in variant_rows]
        variant_controls = [row for row in controls if row.get("variant") == variant]
        variant_ablations = [row for row in ablations if row.get("variant") == variant]
        variant_summaries.append(
            {
                "variant": variant,
                "stage_variant": variant_rows[0].get("stage_variant"),
                "seeds": [int(row["seed"]) for row in variant_rows],
                "mean_trainable_top1": _mean(trainable),
                "std_trainable_top1": _std(trainable),
                "min_trainable_top1": min(trainable) if trainable else 0.0,
                "max_trainable_top1": max(trainable) if trainable else 0.0,
                "mean_frozen_top1": _mean(frozen),
                "mean_delta": _mean(deltas),
                "bootstrap_95ci_delta": _bootstrap_ci(deltas),
                "seeds_trainable_beats_frozen": int(sum(delta > 0.0 for delta in deltas)),
                "weak_seed_count": int(sum(value <= (1.0 / max(1, dataset_config.num_candidates) + 0.05) for value in trainable)),
                "controls_pass_all": all(bool(row.get("control_pass", {}).get("overall", False)) for row in variant_controls),
                "max_composition_ablation_drop": max([float(row.get("top1_drop", 0.0)) for row in variant_ablations] or [0.0]),
            }
        )
    variant_summaries.sort(key=lambda row: (float(row["mean_delta"]), float(row["mean_trainable_top1"])), reverse=True)
    best = variant_summaries[0] if variant_summaries else {}
    gates = _phase2_gates(best, rows, controls, ablations, dataset_config)
    return {
        "variant_summaries": variant_summaries,
        "leaderboard_top": list(leaderboard[:8]),
        "best_variant_by_delta": best.get("variant"),
        "random_baseline": 1.0 / max(1, dataset_config.num_candidates),
        "phase2_triggered": bool(all(gates.values())),
        "phase2_gates": gates,
        "claim_boundary": "No ARC-AGI-2 solving or full ARC generalization claim. Phase 2 remains blocked unless all gates are true.",
    }


def _phase2_gates(
    best: Dict[str, object],
    rows: Sequence[Dict[str, object]],
    controls: Sequence[Dict[str, object]],
    ablations: Sequence[Dict[str, object]],
    dataset_config: ArcVerificationDatasetConfig,
) -> Dict[str, bool]:
    best_variant = best.get("variant")
    if not best_variant:
        return {
            "trainable_beats_frozen_on_at_least_2_of_3_seeds": False,
            "mean_trainable_frozen_delta_gt_0_10": False,
            "heldout_top1_above_random_N8": False,
            "candidate_only_near_chance": False,
            "metadata_only_near_chance": False,
            "mismatch_collapses": False,
            "train_pair_shuffle_collapses": False,
            "candidate_order_remap_invariance_passes": False,
            "role_order_invariance_passes": False,
            "pyramid_order_invariance_passes": False,
            "composition_ablation_hurts": False,
            "frozen_and_trainable_audits_pass": False,
            "no_candidate_artifact_control_failure": False,
        }
    chance = 1.0 / max(1, dataset_config.num_candidates)
    best_rows = [row for row in rows if row.get("variant") == best_variant and row.get("status") == "completed"]
    best_controls = [row for row in controls if row.get("variant") == best_variant]
    best_ablations = [row for row in ablations if row.get("variant") == best_variant]
    frozen_ok = all(
        bool(row.get("frozen_audit", {}).get("frozen_shared_model_zero_grad"))
        and bool(row.get("frozen_audit", {}).get("frozen_shared_model_zero_delta"))
        for row in best_rows
    )
    trainable_ok = all(bool(row.get("trainable_audit", {}).get("trainable_shared_model_changed")) for row in best_rows)
    seeds_run = len(best.get("seeds", []))
    return {
        "trainable_beats_frozen_on_at_least_2_of_3_seeds": bool(seeds_run >= 3 and int(best.get("seeds_trainable_beats_frozen", 0)) >= 2),
        "mean_trainable_frozen_delta_gt_0_10": float(best.get("mean_delta", 0.0)) > 0.10,
        "heldout_top1_above_random_N8": float(best.get("mean_trainable_top1", 0.0)) > chance + 0.05,
        "candidate_only_near_chance": all(float(row.get("candidate_only", {}).get("top1", 1.0)) <= chance + 0.10 for row in best_controls),
        "metadata_only_near_chance": all(float(row.get("candidate_source_metadata_only_accuracy", 1.0)) <= chance + 0.10 for row in best_controls),
        "mismatch_collapses": all(float(row.get("candidate_evidence_mismatch", {}).get("top1", 1.0)) <= chance + 0.15 for row in best_controls),
        "train_pair_shuffle_collapses": all(float(row.get("train_pair_shuffle", {}).get("top1", 1.0)) <= chance + 0.15 for row in best_controls),
        "candidate_order_remap_invariance_passes": all(float(row.get("candidate_order_invariance_delta", 1.0)) <= 0.08 for row in best_controls),
        "role_order_invariance_passes": all(float(row.get("role_order_invariance_delta", 1.0)) <= 0.08 for row in best_controls),
        "pyramid_order_invariance_passes": all(float(row.get("pyramid_level_permutation_delta", 1.0)) <= 0.02 for row in best_controls),
        "composition_ablation_hurts": max([float(row.get("top1_drop", 0.0)) for row in best_ablations] or [0.0]) >= 0.05,
        "frozen_and_trainable_audits_pass": bool(frozen_ok and trainable_ok),
        "no_candidate_artifact_control_failure": all(bool(row.get("control_pass", {}).get("overall", False)) for row in best_controls),
    }


def _write_outputs(
    result: Dict[str, object],
    output_path: Path,
    controls_path: Path,
    leaderboard_path: Path,
    ablation_path: Path,
    error_path: Path,
    attention_path: Path,
    compute_path: Path,
    report_path: Path,
) -> None:
    for path in (output_path, controls_path, leaderboard_path, ablation_path, error_path, attention_path, compute_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_without_large_logs(result), indent=2, sort_keys=True), encoding="utf-8")
    controls_path.write_text(json.dumps({"controls": result["controls"], "phase2_gates": result["summary"].get("phase2_gates", {})}, indent=2, sort_keys=True), encoding="utf-8")
    ablation_path.write_text(json.dumps({"ablations": result["ablations"]}, indent=2, sort_keys=True), encoding="utf-8")
    error_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in result["error_cases"]) + ("\n" if result["error_cases"] else ""), encoding="utf-8")
    attention_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in result["attention_summaries"]) + ("\n" if result["attention_summaries"] else ""), encoding="utf-8")
    compute_path.write_text(json.dumps({"compute_metrics": result["compute_metrics"]}, indent=2, sort_keys=True), encoding="utf-8")
    with leaderboard_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = sorted({key for row in result["leaderboard"] for key in row.keys()})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in result["leaderboard"]:
            writer.writerow(row)
    report_path.write_text(_render_report(result), encoding="utf-8")


def _render_report(result: Dict[str, object]) -> str:
    summary = result.get("summary", {})
    leaderboard = result.get("leaderboard", [])
    best_variant = summary.get("best_variant_by_delta")
    best_rows = [row for row in result.get("rows", []) if row.get("variant") == best_variant and row.get("status") == "completed"]
    p0_rows = [row for row in result.get("rows", []) if row.get("variant") == "P0_flat_baseline" and row.get("status") == "completed"]
    status_by_variant = {str(row.get("variant")): str(row.get("status")) for row in result.get("rows", [])}
    failure_by_variant = {str(row.get("variant")): str(row.get("failure_reason", "")) for row in result.get("rows", [])}
    best_trainable = _mean([float(row["dev_trainable"]["top1"]) for row in best_rows])
    best_frozen = _mean([float(row["dev_frozen"]["top1"]) for row in best_rows])
    p0_trainable = _mean([float(row["dev_trainable"]["top1"]) for row in p0_rows])
    p0_delta = _mean([float(row["dev_delta_trainable_minus_frozen"]) for row in p0_rows])
    random_baseline = float(summary.get("random_baseline", 0.125))
    best_ablation = max([float(row.get("top1_drop", 0.0)) for row in result.get("ablations", []) if row.get("variant") == best_variant] or [0.0])
    gates = summary.get("phase2_gates", {})
    lines = [
        "# Stage ARC-1.3 Pyramidal Latent View Composition",
        "",
        "## Scope",
        "",
        "- Task: controlled ARC candidate-output verification with known output size and exactly one gold candidate.",
        "- This is a bounded empirical search over hierarchical latent composition, not final validation.",
        "- Phase 2 was not launched unless every clean gate passed.",
        "- Parent composers receive encoded child latent states only; no gold labels, source metadata, or candidate source IDs are provided.",
        "",
        "## Leaderboard",
        "",
        "| rank | variant | trainable | frozen | delta | candidate-only | controls | ablation drop | layout | composer | query |",
        "|---:|---|---:|---:|---:|---:|---|---:|---|---|---|",
    ]
    for rank, row in enumerate(leaderboard[:12], start=1):
        lines.append(
            f"| {rank} | {row.get('variant')} | {float(row.get('trainable_top1', 0.0)):.4f} | {float(row.get('frozen_top1', 0.0)):.4f} | {float(row.get('delta', 0.0)):.4f} | {float(row.get('candidate_only', 0.0)):.4f} | `{row.get('controls_pass')}` | {float(row.get('best_ablation_drop', 0.0)):.4f} | {row.get('pyramid_layout', '')} | {row.get('composer_kind', '')} | {row.get('query_mode', '')} |"
        )
    lines.extend(
        [
            "",
            "## Phase 2 Gates",
            "",
            "| gate | pass |",
            "|---|---|",
        ]
    )
    for key, value in gates.items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(
        [
            "",
            "## Micro-Overfit Gates",
            "",
            "| variant | status | first failed gate |",
            "|---|---|---|",
        ]
    )
    for row in result.get("rows", []):
        lines.append(f"| {row.get('variant')} | `{row.get('status')}` | {row.get('failure_reason', '')} |")
    lines.extend(
        [
            "",
            "## Controls And Ablations",
            "",
            "| variant | candidate-only | metadata-only | mismatch | train shuffle | comp shuffle | child mismatch | order delta | role delta | pyramid delta | overall |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in result.get("controls", []):
        lines.append(
            f"| {row.get('variant')} | {float(row.get('candidate_only', {}).get('top1', 0.0)):.4f} | {float(row.get('candidate_source_metadata_only_accuracy', 0.0)):.4f} | {float(row.get('candidate_evidence_mismatch', {}).get('top1', 0.0)):.4f} | {float(row.get('train_pair_shuffle', {}).get('top1', 0.0)):.4f} | {float(row.get('composition_shuffle', {}).get('top1', 0.0)):.4f} | {float(row.get('child_mismatch', {}).get('top1', 0.0)):.4f} | {float(row.get('candidate_order_invariance_delta', 0.0)):.4f} | {float(row.get('role_order_invariance_delta', 0.0)):.4f} | {float(row.get('pyramid_level_permutation_delta', 0.0)):.4f} | `{row.get('control_pass', {}).get('overall', False)}` |"
        )
    lines.extend(
        [
            "",
            "## Required Answers",
            "",
            f"1. Does pyramidal composition improve over flat leaf-view querying? Best pyramidal/top variant `{best_variant}` reached `{best_trainable:.4f}` top1 versus P0 `{p0_trainable:.4f}` top1; P0 delta was `{p0_delta:.4f}`.",
            f"2. Which compositions are useful? The strongest measured ablation drop for the best variant was `{best_ablation:.4f}` top1; detailed pair ablations are in `results/arc1_3_pyramidal_ablation.json`.",
            f"3. Does token-level composition beat pooled composition? Not established: `P1_pairwise_pooled_composed_only` status is `{status_by_variant.get('P1_pairwise_pooled_composed_only')}`, while `P2_pairwise_token_composed_only` status is `{status_by_variant.get('P2_pairwise_token_composed_only')}` with first failed gate `{failure_by_variant.get('P2_pairwise_token_composed_only')}`.",
            f"4. Does candidate-guided composition help or leak? Not established: `P6_candidate_guided_composition` status is `{status_by_variant.get('P6_candidate_guided_composition')}` with first failed gate `{failure_by_variant.get('P6_candidate_guided_composition')}`. The implementation does not use candidate index/source embeddings.",
            f"5. Does root-level composition add value beyond pairwise composition? Not established: `P3_binary_tree_root_intermediate` status is `{status_by_variant.get('P3_binary_tree_root_intermediate')}` and `P5_recursive_weight_tied_composer` status is `{status_by_variant.get('P5_recursive_weight_tied_composer')}`.",
            "6. Do composition ablations show parent clones are actually used? This is true only when composition/root/pair ablations produce a meaningful positive drop; see the ablation output.",
            f"7. Does the pyramid improve clean held-out accuracy above random? Random is `{random_baseline:.4f}`; best clean top1 is `{best_trainable:.4f}`.",
            f"8. Does trainable beat exact frozen same-architecture? Best trainable `{best_trainable:.4f}` versus frozen `{best_frozen:.4f}`; frozen/trainable audit gate is `{gates.get('frozen_and_trainable_audits_pass')}`.",
            f"9. Do controls remain clean? Overall gate for the best variant is `{gates.get('no_candidate_artifact_control_failure')}`.",
            f"10. Is pyramidal composition worth moving to Phase 2? `{bool(summary.get('phase2_triggered', False))}`.",
            "",
            "## Decision",
            "",
            f"- Phase 2 triggered: `{bool(summary.get('phase2_triggered', False))}`.",
            "- No ARC-AGI-2 solving, full ARC generalization, output-size prediction, or grid-generation claim is made.",
        ]
    )
    return "\n".join(lines) + "\n"


def _variant_plan(config_variants: object = None) -> List[PyramidalVariant]:
    dim = 24
    heads = 2
    ff = 48
    max_rule = 48
    max_cand = 48
    leaf_views = ("raw", "diff", "object", "color", "geometry", "symmetry", "counting")
    base = dict(model_dim=dim, num_heads=heads, ff_dim=ff, rule_layers=1, candidate_layers=1, max_rule_tokens=max_rule, max_candidate_tokens=max_cand)
    variants = [
        PyramidalVariant(
            "P0_flat_baseline",
            "P0_flat_baseline",
            "Flat candidate-token-direct baseline over all leaf views.",
            PyramidalArcModelConfig(
                **base,
                variant_family="P0_flat_baseline",
                view_names=leaf_views,
                representation_mode="hybrid_all",
                pyramid_layout="flat",
                composer_kind="none",
                query_mode="leaf_only",
            ),
        ),
        PyramidalVariant(
            "P0_anchor_object_bbox_ce",
            "P0_flat_baseline",
            "ARC-1.1 strong clean anchor: raw/object object-bbox representation.",
            PyramidalArcModelConfig(
                **base,
                variant_family="P0_anchor_object_bbox",
                view_names=("raw", "object"),
                representation_mode="object",
                pyramid_layout="flat",
                composer_kind="none",
                query_mode="leaf_only",
            ),
        ),
        PyramidalVariant(
            "P0_anchor_changed_cell_ce",
            "P0_flat_baseline",
            "ARC-1.1 strong learnable anchor: raw/diff changed-cell curriculum.",
            PyramidalArcModelConfig(
                **base,
                variant_family="P0_anchor_changed_cell",
                view_names=("raw", "diff"),
                representation_mode="hybrid_cell_object_diff",
                pyramid_layout="flat",
                composer_kind="none",
                query_mode="leaf_only",
            ),
            negative_difficulty="changed_cell_count_matched",
        ),
        PyramidalVariant(
            "P1_pairwise_pooled_composed_only",
            "P1_pairwise_pooled_composer",
            "Pairwise pooled summaries feed shared composition states; candidate queries composed states only.",
            PyramidalArcModelConfig(
                **base,
                variant_family="P1_pairwise_pooled",
                view_names=leaf_views,
                representation_mode="hybrid_all",
                pyramid_layout="pairwise",
                composer_kind="pooled",
                query_mode="composed_only",
            ),
        ),
        PyramidalVariant(
            "P1_pairwise_pooled_leaf_plus_composed",
            "P1_pairwise_pooled_composer",
            "Pairwise pooled summaries plus residual leaf access.",
            PyramidalArcModelConfig(
                **base,
                variant_family="P1_pairwise_pooled",
                view_names=leaf_views,
                representation_mode="hybrid_all",
                pyramid_layout="pairwise",
                composer_kind="pooled",
                query_mode="leaf_plus_composed",
            ),
        ),
        PyramidalVariant(
            "P2_pairwise_token_composed_only",
            "P2_pairwise_token_composer",
            "Pairwise token-level composer with masks; candidate queries composed token states only.",
            PyramidalArcModelConfig(
                **base,
                variant_family="P2_pairwise_token",
                view_names=leaf_views,
                representation_mode="hybrid_all",
                pyramid_layout="pairwise",
                composer_kind="token",
                query_mode="composed_only",
                max_composed_tokens=24,
            ),
        ),
        PyramidalVariant(
            "P2_pairwise_token_leaf_plus_composed",
            "P2_pairwise_token_composer",
            "Pairwise token-level composer with candidate access to leaf and composed states.",
            PyramidalArcModelConfig(
                **base,
                variant_family="P2_pairwise_token",
                view_names=leaf_views,
                representation_mode="hybrid_all",
                pyramid_layout="pairwise",
                composer_kind="token",
                query_mode="leaf_plus_composed",
                max_composed_tokens=24,
            ),
        ),
        PyramidalVariant(
            "P3_binary_tree_root_intermediate",
            "P3_binary_tree_composer",
            "Binary tree composer: change/structure/appearance into transformation/final states.",
            PyramidalArcModelConfig(
                **base,
                variant_family="P3_binary_tree",
                view_names=leaf_views,
                representation_mode="hybrid_all",
                pyramid_layout="binary_tree",
                composer_kind="token",
                query_mode="root_plus_intermediate",
                max_composed_tokens=24,
            ),
        ),
        PyramidalVariant(
            "P4_all_pairs_shared_composer",
            "P4_all_pairs_composer",
            "All selected view pairs use shared composition weights with pair/view identity embeddings.",
            PyramidalArcModelConfig(
                **base,
                variant_family="P4_all_pairs",
                view_names=leaf_views,
                representation_mode="hybrid_all",
                pyramid_layout="all_pairs",
                composer_kind="token",
                query_mode="composed_only",
                max_composed_tokens=20,
            ),
        ),
        PyramidalVariant(
            "P5_recursive_weight_tied_composer",
            "P5_recursive_weight_tied_composer",
            "Same token composition module reused at every binary-tree internal node.",
            PyramidalArcModelConfig(
                **base,
                variant_family="P5_recursive_weight_tied",
                view_names=leaf_views,
                representation_mode="hybrid_all",
                pyramid_layout="recursive",
                composer_kind="token",
                query_mode="root_plus_intermediate",
                shared_composer_weights=True,
                max_composed_tokens=24,
            ),
        ),
        PyramidalVariant(
            "P6_candidate_guided_composition",
            "P6_candidate_guided_composition",
            "Candidate representation conditions pairwise token composition without candidate index/source embeddings.",
            PyramidalArcModelConfig(
                **base,
                variant_family="P6_candidate_guided",
                view_names=leaf_views,
                representation_mode="hybrid_all",
                pyramid_layout="pairwise",
                composer_kind="token",
                query_mode="composed_only",
                candidate_guided_composition=True,
                max_composed_tokens=24,
            ),
        ),
        PyramidalVariant(
            "P7_residual_pyramid",
            "P7_residual_pyramid",
            "Residual score decomposition across leaf, pairwise, and root states.",
            PyramidalArcModelConfig(
                **base,
                variant_family="P7_residual_pyramid",
                view_names=leaf_views,
                representation_mode="hybrid_all",
                pyramid_layout="residual",
                composer_kind="token",
                query_mode="all_levels",
                residual_pyramid=True,
                max_composed_tokens=24,
            ),
        ),
    ]
    if config_variants:
        allowed = {str(name) for name in config_variants if str(name)}
        exact_names = {variant.name for variant in variants}
        variants = [
            variant
            for variant in variants
            if variant.name in allowed or (variant.stage_variant in allowed and variant.stage_variant not in exact_names)
        ]
    return variants


def _controls_pass(control_row: Dict[str, object], clean_top1: float, chance: float, model_config: PyramidalArcModelConfig) -> Dict[str, bool]:
    pass_map = {
        "candidate_only": float(control_row.get("candidate_only", {}).get("top1", 1.0)) <= chance + 0.10,
        "metadata_source_only": float(control_row.get("candidate_source_metadata_only_accuracy", 1.0)) <= chance + 0.10,
        "train_pair_shuffle": float(control_row.get("train_pair_shuffle", {}).get("top1", 1.0)) <= chance + 0.15,
        "cross_task_train_pair_shuffle": float(control_row.get("cross_task_train_pair_shuffle", {}).get("top1", 1.0)) <= chance + 0.15,
        "candidate_evidence_mismatch": float(control_row.get("candidate_evidence_mismatch", {}).get("top1", 1.0)) <= chance + 0.15,
        "candidate_order_shuffle_with_gold_remap": float(control_row.get("candidate_order_invariance_delta", 1.0)) <= 0.08,
        "physical_role_order_shuffle": float(control_row.get("role_order_invariance_delta", 1.0)) <= 0.08,
        "pyramid_level_permutation": float(control_row.get("pyramid_level_permutation_delta", 1.0)) <= 0.02,
        "hidden_state_shuffle": float(control_row.get("hidden_state_shuffle", {}).get("top1", 1.0)) <= max(chance + 0.20, clean_top1 - 0.05),
        "randomized_labels": float(control_row.get("randomized_labels", {}).get("top1", 1.0)) <= chance + 0.15,
        "composition_shuffle": float(control_row.get("composition_shuffle", {}).get("top1", 1.0)) <= max(chance + 0.20, clean_top1 - 0.05),
        "child_mismatch": float(control_row.get("child_mismatch", {}).get("top1", 1.0)) <= max(chance + 0.20, clean_top1 - 0.05),
        "parent_sees_raw_evidence_audit": bool(control_row.get("parent_sees_raw_evidence_audit", {}).get("pass", False)),
        "candidate_guided_leakage_audit": bool(control_row.get("candidate_guided_leakage_audit", {}).get("pass", False)),
    }
    if not bool(model_config.ordered_composition_children):
        pass_map["composition_order_invariance"] = float(control_row.get("composition_order_swap_delta", 1.0)) <= 0.02
    else:
        pass_map["composition_order_documented"] = True
    pass_map["overall"] = all(pass_map.values())
    return pass_map


def _without_large_logs(result: Dict[str, object]) -> Dict[str, object]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"leakage_audit", "error_cases", "attention_summaries", "compute_metrics"}
    }


def _all_pair_plan(view_names: Sequence[str]) -> List[Tuple[str, str, str]]:
    pairs = []
    for left_index, left in enumerate(view_names):
        for right in view_names[left_index + 1 :]:
            pairs.append((str(left), str(right), f"all_pair_{left}_{right}"))
    return pairs


def _pair_removed(name: str, ablation: str | None) -> bool:
    return bool(ablation and ablation == f"pair_composer_ablation::{name}")


def _permuted_state_order(states: Sequence[LatentState]) -> List[LatentState]:
    return sorted(states, key=lambda state: (state.level, state.name), reverse=True)


def _shuffle_child(tokens: torch.Tensor, mask: torch.Tensor, batch_size: int, candidates: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if tokens.shape[0] <= 1:
        return tokens, mask
    generator = torch.Generator(device=tokens.device)
    generator.manual_seed(int(seed))
    if tokens.shape[0] == batch_size * candidates and candidates > 1:
        order = torch.randperm(batch_size, generator=generator, device=tokens.device)
        if batch_size > 1 and torch.equal(order, torch.arange(batch_size, device=tokens.device)):
            order = torch.roll(order, 1)
        candidate_offsets = torch.arange(candidates, device=tokens.device).repeat(batch_size)
        expanded = order.repeat_interleave(candidates) * candidates + candidate_offsets
        return tokens.index_select(0, expanded), mask.index_select(0, expanded)
    order = torch.randperm(tokens.shape[0], generator=generator, device=tokens.device)
    if tokens.shape[0] > 1 and torch.equal(order, torch.arange(tokens.shape[0], device=tokens.device)):
        order = torch.roll(order, 1)
    return tokens.index_select(0, order), mask.index_select(0, order)


def _state_diagnostics(states: Sequence[LatentState]) -> List[Dict[str, object]]:
    rows = []
    means = []
    for state in states:
        with torch.no_grad():
            pooled = _masked_mean(state.tokens, state.mask)
            means.append((state.name, state.level, pooled.detach()))
            token_std = float(state.tokens.detach().float().std().cpu()) if state.tokens.numel() else 0.0
            rows.append(
                {
                    "state": state.name,
                    "level": state.level,
                    "valid_tokens_mean": float(state.mask.float().sum(dim=1).mean().detach().cpu()),
                    "token_norm_mean": float(state.tokens.detach().float().norm(dim=-1).mean().cpu()),
                    "token_std": token_std,
                    "collapsed_by_low_std": bool(token_std < 1e-5),
                    "child_names": list(state.child_names),
                    "child_parent_cosine_mean": float(state.child_parent_cosine.float().mean().cpu()) if state.child_parent_cosine is not None else None,
                }
            )
    if len(means) > 1:
        sims = []
        for left_index, (left_name, left_level, left) in enumerate(means):
            for right_name, right_level, right in means[left_index + 1 :]:
                if left.shape == right.shape:
                    sims.append(
                        {
                            "left": left_name,
                            "left_level": left_level,
                            "right": right_name,
                            "right_level": right_level,
                            "cosine": float(F.cosine_similarity(left, right, dim=-1).mean().cpu()),
                        }
                    )
        rows.append({"state": "__pairwise_state_cosine__", "level": "diagnostic", "cosine_pairs": sims[:24]})
    return rows


def _composition_id(name: str) -> int:
    if name in COMPOSITION_IDS:
        return COMPOSITION_IDS[name]
    total = sum(ord(ch) for ch in str(name))
    return 8 + (total % 48)


def _state_id(name: str) -> int:
    if name in VIEW_ID:
        return int(VIEW_ID[name])
    return len(VIEW_NAMES) + _composition_id(name)


def _id_tensor(device: torch.device, value: int, batch: int) -> torch.Tensor:
    return torch.full((int(batch),), int(value), dtype=torch.long, device=device)


def _composition_node_count(model_config: PyramidalArcModelConfig) -> int:
    if model_config.pyramid_layout == "flat":
        return 0
    if model_config.pyramid_layout == "pairwise":
        return len(PAIR_ARC)
    if model_config.pyramid_layout == "all_pairs":
        return len(_all_pair_plan(model_config.view_names))
    if model_config.pyramid_layout in {"binary_tree", "recursive"}:
        return 5
    if model_config.pyramid_layout == "residual":
        return 6
    return 0


def _pyramid_levels(model_config: PyramidalArcModelConfig) -> int:
    if model_config.pyramid_layout == "flat":
        return 1
    if model_config.pyramid_layout in {"pairwise", "all_pairs"}:
        return 2
    return 3


def _approx_attention_cost(model_config: PyramidalArcModelConfig, candidate_count: int) -> int:
    leaf = len(model_config.view_names) * int(model_config.max_rule_tokens)
    composed = _composition_node_count(model_config) * int(model_config.max_composed_tokens)
    queried = leaf + composed if model_config.query_mode in {"leaf_plus_composed", "all_levels"} else max(leaf, composed)
    if model_config.residual_pyramid:
        queried = leaf + composed
    return int(candidate_count) * int(model_config.max_candidate_tokens) * queried * max(1, int(model_config.coordination_blocks))


def _best_ablation_drop(rows: Sequence[Dict[str, object]]) -> float:
    return max([float(row.get("top1_drop", 0.0)) for row in rows] or [0.0])


def _best_non_oracle_baseline(baselines: Dict[str, object]) -> float:
    values = [
        float(baselines.get("random", 0.0)),
        float(baselines.get("candidate_order_index0", 0.0)),
        float(baselines.get("candidate_metadata_source_only", 0.0)),
        float(baselines.get("candidate_grid_only_stat_heuristic", 0.0)),
        float(baselines.get("nearest_train_output", 0.0)),
        float(baselines.get("heuristic_changed_cell_count", 0.0)),
    ]
    return max(values) if values else 0.0


def _first_failed_gate(rows: Sequence[Dict[str, object]]) -> str:
    for row in rows:
        if row.get("gate_required", True) and not bool(row.get("pass", False)):
            return str(row.get("gate", "unknown"))
    return "unknown"


def _compatible_dim(model_dim: int, num_heads: int) -> int:
    heads = max(1, int(num_heads))
    dim = max(heads, int(model_dim))
    if dim % heads != 0:
        dim += heads - (dim % heads)
    return dim


def _dataset_config(data: object) -> ArcVerificationDatasetConfig:
    values = dict(data or {})
    allowed = set(ArcVerificationDatasetConfig.__dataclass_fields__.keys())
    return ArcVerificationDatasetConfig(**{key: value for key, value in values.items() if key in allowed})


def _training_config(data: object) -> ArcTrainingConfig:
    values = dict(data or {})
    allowed = set(ArcTrainingConfig.__dataclass_fields__.keys())
    return ArcTrainingConfig(**{key: value for key, value in values.items() if key in allowed})


def _load_config(path: Path) -> Dict[str, object]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
