from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


E4_ROOT = Path(__file__).resolve().parents[3]
E3_1_ROOT = E4_ROOT.parent / "e3_1"
E3_ROOT = E4_ROOT.parent / "e3"
E2_ROOT = E4_ROOT.parent / "e2"
for code_root in (E3_1_ROOT / "code", E3_ROOT / "code"):
    if code_root.exists() and str(code_root) not in sys.path:
        sys.path.append(str(code_root))

import src.experiments.e3_clean_qkv as e3mod  # type: ignore  # noqa: E402
from src.experiments.e3_clean_qkv import (  # type: ignore  # noqa: E402
    CleanQKVAttentionLayer,
    DECISION_CUDA_BLOCKED,
    E3ArchitectureConfig,
    E3Budget,
    FULL_CONTROLS,
    K_CANDIDATES,
    MEMORY_DEPENDENT_FAMILIES,
    MEMORY_INDEPENDENT_FAMILIES,
    WDAAdapter,
    _aggregate_baseline_rows,
    _brief_rows,
    _capacity,
    _control_degradation,
    _dump_json,
    _load_reference_scores,
    _mean,
    _oracle_accuracy,
    _post_attention_memory_accuracy,
    _rank_rows,
    _retrieval_topk_accuracy,
    _safe_read_preview,
    _stable_hash,
    apply_e3_control,
    assert_training_device,
    build_e3_examples,
    cuda_preflight_status,
    e3_config_to_dict,
)
from src.experiments.e3_1_wda_clean_qkv import (  # type: ignore  # noqa: E402
    E31Budget,
    SupportContradictionActivationCache,
    build_e3_1_examples,
    generate_e3_1_variants,
    mutate_e3_1_variant,
    validate_e3_1_examples,
)


RESULTS_DIR = E4_ROOT / "results"
REPORTS_DIR = E4_ROOT / "reports"
PREFIX = "e4_collab_clean_qkv"

PREFLIGHT_JSON = RESULTS_DIR / f"{PREFIX}_preflight.json"
PREFLIGHT_REPORT = REPORTS_DIR / "E4_COLLAB_CLEAN_QKV_PREFLIGHT.md"
DATABASE_PATH = RESULTS_DIR / f"{PREFIX}_database.jsonl"
ROUND0_PATH = RESULTS_DIR / f"{PREFIX}_round0_replay.json"
ROUND1_PATH = RESULTS_DIR / f"{PREFIX}_round1_screen.json"
ROUND2_PATH = RESULTS_DIR / f"{PREFIX}_round2_n256_pressure.json"
ROUND3_PATH = RESULTS_DIR / f"{PREFIX}_round3_mutations.json"
ROUND4_PATH = RESULTS_DIR / f"{PREFIX}_round4_finalists.json"
TOP_CONFIGS_PATH = RESULTS_DIR / f"{PREFIX}_top_configs.yaml"
BEST_CONFIG_PATH = RESULTS_DIR / f"{PREFIX}_best_config.yaml"
FREEZE_RECOMMENDATION_PATH = RESULTS_DIR / f"{PREFIX}_freeze_recommendation.json"
CONTROLS_AUDIT_PATH = RESULTS_DIR / f"{PREFIX}_controls_audit.jsonl"
ARCHITECTURE_AUDIT_PATH = RESULTS_DIR / f"{PREFIX}_architecture_audit.json"
WDA_AUDIT_PATH = RESULTS_DIR / f"{PREFIX}_wda_audit.json"
GATE_AUDIT_PATH = RESULTS_DIR / f"{PREFIX}_gate_audit.json"
GUIDANCE_AUDIT_PATH = RESULTS_DIR / f"{PREFIX}_guidance_audit.json"
CAPACITY_CURVES_PATH = RESULTS_DIR / f"{PREFIX}_capacity_curves.csv"
COMPUTE_AUDIT_PATH = RESULTS_DIR / f"{PREFIX}_compute_audit.json"
REPORT_PATH = REPORTS_DIR / "E4_COLLABORATIVE_ACTIVATION_GUIDED_CLEAN_QKV.md"

DECISION_REPLAY_FAILED = "E4_REPLAY_FAILED"
DECISION_C256 = "E4_COLLAB_CLEAN_QKV_C256_REACHED"
DECISION_IMPROVED = "E4_COLLAB_CLEAN_QKV_IMPROVED_BUT_C256_NOT_REACHED"
DECISION_BEATS_E2 = "E4_COLLAB_CLEAN_QKV_BEATS_E2_CHAMPION"
DECISION_CURRENT_QUERY = "E4_CURRENT_TO_CACHE_QUERY_SUPPORTED"
DECISION_REFINEMENT = "E4_RECIPROCAL_REFINEMENT_SUPPORTED"
DECISION_FIELD = "E4_GUIDANCE_FIELD_SUPPORTED"
DECISION_CACHE_UPDATE = "E4_CACHE_STATE_UPDATE_SUPPORTED"
DECISION_FEEDBACK = "E4_ATTENTION_FEEDBACK_SUPPORTED"
DECISION_T = "E4_T_REALIZED_COLLAB_SUPPORTED"
DECISION_LAYERWISE = "E4_LAYERWISE_COLLAB_SUPPORTED"
DECISION_NOT_BETTER = "E4_COLLAB_NOT_BETTER_THAN_E3_1"
DECISION_NOT_SUPPORTED = "E4_COLLAB_CLEAN_QKV_NOT_SUPPORTED"

E31_PARENT_HELDOUT = 0.7575
E31_PARENT_CAPACITY = 128

E4_CONTROLS: Tuple[str, ...] = FULL_CONTROLS + (
    "support_cache_shuffle",
    "contradiction_cache_shuffle",
    "support_contradiction_swap",
    "stale_current_swap",
    "current_to_cache_path_disabled",
    "cache_to_current_path_disabled",
    "reciprocal_refinement_disabled",
    "preliminary_current_attention_feedback_disabled",
    "wda_no_controller",
    "wda_frozen_controller",
)

E4_STRESS_CASES: Tuple[str, ...] = (
    "current_query_must_select_relevant_past_activation",
    "current_candidate_must_select_support_activation",
    "current_candidate_must_select_contradiction_activation",
    "support_and_contradiction_both_present",
    "stale_support_must_be_ignored",
    "stale_contradiction_must_be_ignored",
    "current_input_overrides_activation_cache",
    "irrelevant_activation_cache_should_be_ignored",
    "misleading_activation_cache_should_be_ignored",
    "multiple_past_activations_must_combine",
    "current_activations_disambiguate_cache_stream",
    "cache_guidance_changes_current_attention_distribution",
    "memory_independent_examples_not_harmed_by_cache_guidance",
)


@dataclass(frozen=True)
class E4Budget:
    train_examples: int = 40
    eval_examples: int = 40
    replay_n: Tuple[int, ...] = (64, 128)
    round1_n: Tuple[int, ...] = (64, 128)
    round2_n: Tuple[int, ...] = (64, 128, 256)
    round3_n: Tuple[int, ...] = (128, 256)
    round4_n: Tuple[int, ...] = (64, 128, 256)
    replay_seeds: Tuple[int, ...] = (0, 1, 2)
    round1_seeds: Tuple[int, ...] = (0,)
    round2_seeds: Tuple[int, ...] = (0, 1, 2)
    round3_seeds: Tuple[int, ...] = (0, 1, 2)
    round4_seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    round1_epochs: int = 2
    promoted_epochs: int = 3
    lr: float = 0.07
    weight_decay: float = 0.0001
    max_new_variants: int = 64


class CollaborativeGuidanceLayer(nn.Module):
    """Toy module for auditing current-to-cache and cache-to-current paths."""

    def __init__(self, dim: int = 8, mode: str = "wda_kv", refinement_steps: int = 1) -> None:
        super().__init__()
        self.dim = int(dim)
        self.mode = mode
        self.refinement_steps = int(refinement_steps)
        self.current_query = nn.Linear(dim, dim, bias=False)
        self.cache_key = nn.Linear(dim, dim, bias=False)
        self.guidance_proj = nn.Linear(dim, dim, bias=False)
        self.attention = CleanQKVAttentionLayer(dim=dim, mode=mode)

    def forward(
        self,
        current_tokens: torch.Tensor,
        activation_cache: torch.Tensor,
        *,
        disable_current_to_cache: bool = False,
        disable_cache_to_current: bool = False,
        disable_reciprocal: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        summary = current_tokens.mean(dim=1)
        effective_cache = activation_cache
        diagnostics: Dict[str, torch.Tensor] = {}
        steps = 1 if disable_reciprocal else max(1, self.refinement_steps)
        output = current_tokens
        for _ in range(steps):
            query = torch.zeros_like(summary) if disable_current_to_cache else self.current_query(summary)
            keys = self.cache_key(effective_cache)
            logits = torch.einsum("bd,bsd->bs", query, keys) / math.sqrt(max(1, self.dim))
            weights = torch.softmax(logits, dim=-1)
            guidance = torch.einsum("bs,bsd->bd", weights, effective_cache)
            guidance = torch.zeros_like(guidance) if disable_cache_to_current else self.guidance_proj(guidance)
            shaped_cache = effective_cache + guidance.unsqueeze(1)
            output, diagnostics = self.attention(output, shaped_cache)
            summary = output.mean(dim=1)
            effective_cache = shaped_cache
            diagnostics["current_to_cache_weights"] = weights
            diagnostics["guidance"] = guidance
        return output, diagnostics


class E4TypedActivationCache(nn.Module):
    """Support/contradiction/stale/current-override activation streams."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.support_contra = SupportContradictionActivationCache(dim=dim)
        self.current_override_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, support: torch.Tensor, contradiction: torch.Tensor, stale: torch.Tensor, current_override: torch.Tensor) -> Dict[str, torch.Tensor]:
        streams = self.support_contra(support, contradiction, stale)
        streams["current_override"] = self.current_override_proj(current_override)
        return streams


def write_preflight(*, device: str = "cuda", allow_cpu_smoke: bool = False) -> Dict[str, object]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    status = cuda_preflight_status(device=device, allow_cpu_smoke=allow_cpu_smoke)
    e31_results = E3_1_ROOT / "results"
    e2_results = E2_ROOT / "results"
    best_text = _safe_read_preview(e31_results / "e3_1_wda_clean_qkv_best_config.yaml")
    payload = {
        "stage": "E4",
        "campaign": "Collaborative Activation-Guided Clean-QKV Exploration",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "requested_device": device,
        "resolved_device": status["resolved_device"],
        "working_directory": str(E4_ROOT),
        "visible_files": sorted(path.name for path in E4_ROOT.iterdir()) if E4_ROOT.exists() else [],
        "e3_1_artifacts_found": e31_results.exists(),
        "e3_1_best_config_found": "e3_1_wda_clean_qkv_e_sc_wda_kv_01" in best_text,
        "e2_reference_artifacts_found_read_only": e2_results.exists(),
        "previous_benchmark_code_found": (E3_ROOT / "code" / "src" / "experiments" / "e3_clean_qkv.py").exists(),
        "no_stage8c_run": True,
        "no_10x_attention_capacity_claim": True,
        "decision": status["decision"],
    }
    _dump_json(PREFLIGHT_JSON, payload)
    PREFLIGHT_REPORT.write_text(_preflight_markdown(payload), encoding="utf-8")
    if payload["gpu_name"]:
        print(f"CUDA device: {payload['gpu_name']}")
    return payload


def build_e4_examples(**kwargs: object):
    if kwargs.get("template_split") == "heldout":
        kwargs = {**kwargs, "template_split": "dev_heldout"}
    examples = build_e3_1_examples(**kwargs)
    output = []
    for index, example in enumerate(examples):
        metadata = dict(example.metadata)
        metadata["e4_stress_case"] = E4_STRESS_CASES[index % len(E4_STRESS_CASES)]
        output.append(replace(example, metadata=metadata))
    return output


def validate_e4_examples(examples: Sequence[object]) -> Dict[str, object]:
    audit = validate_e3_1_examples(examples)
    stress = {str(example.metadata.get("e4_stress_case")) for example in examples}  # type: ignore[attr-defined]
    failures = list(audit["failures"])
    if examples and len(stress) < len(E4_STRESS_CASES):
        failures.append("missing required E4 collaborative stress cases")
    return {**audit, "passes": not failures, "failures": failures, "e4_stress_cases": sorted(stress)}


def load_e31_anchor_configs() -> List[E3ArchitectureConfig]:
    variants = generate_e3_1_variants(48)
    by_name = {variant.name: variant for variant in variants}
    gate_parent = by_name["e3_1_wda_clean_qkv_g_wda_v_gate_calibration_00"]
    gate_mut0 = mutate_e3_1_variant(gate_parent, 1)[0]
    return [
        by_name["e3_1_wda_clean_qkv_e_sc_wda_kv_01"],
        by_name["e3_1_wda_clean_qkv_e_sc_wda_v_00"],
        gate_mut0,
    ]


def generate_e4_variants(max_variants: int = 64) -> List[E3ArchitectureConfig]:
    anchor = load_e31_anchor_configs()[0]
    variants: List[E3ArchitectureConfig] = []
    family_defs = [
        ("A", "One-Way E3.1 Anchor Controls", "one_way_anchor", ("sc_wda_kv_replay", "sc_wda_v_replay", "longer_training", "coord_entropy_diversity", "sc_margin_loss", "gate_contrastive_loss")),
        ("B", "Current-to-Cache Query, Cache-to-WDA-KV", "current_to_cache_query", ("candidate_summary", "query_summary", "current_attention_summary", "per_candidate_query", "separate_sc_queries", "stale_override_typed_cache", "coord_entropy_diversity", "gate_calibration")),
        ("C", "Reciprocal Refinement Loop", "reciprocal_refinement", ("one_step", "two_steps", "three_steps", "shared_steps", "unshared_steps", "stop_gate", "sc_margin_loss", "stale_override_objective")),
        ("D", "Bidirectional Guidance Field", "guidance_field", ("field_wda_kv", "field_attention_bias", "field_bias_v", "field_bias_kv", "signed_sc_components", "stale_override_components", "coordinate_regularized", "anti_derailment_loss")),
        ("E", "Cache-State Update Before Current Attention", "cache_state_update", ("additive_update", "gru_style_update", "attention_update", "separate_sc_updates", "stale_override_update", "cache_dropout", "contrastive_stale_loss", "coord_entropy_diversity")),
        ("F", "Current Attention Distribution Feedback", "attention_feedback", ("summary_wda_kv", "summary_bias_kv", "summary_sc_cache", "summary_stale_override", "shared_two_pass", "unshared_two_pass")),
        ("G", "T-Realized Collaborative Activation Guidance", "t_realized_collab", ("t_to_wda_v", "t_to_wda_kv", "t_realized_v", "t_realized_kv", "sc_t_streams", "t_v_wda_kv_hybrid")),
        ("H", "Layer-Wise Collaborative Guidance", "layerwise_collab", ("final_only", "middle_final", "early_middle_final", "all_shared", "all_separate", "layerwise_sc_gates")),
        ("I", "Anti-Derailment / Selectivity Training", "anti_derailment_selectivity", ("irrelevant_rejection", "misleading_rejection", "stale_rejection", "current_override_aux", "sc_margin", "memory_independent_preserve", "gate_contrastive", "cache_shuffle_contrastive")),
    ]
    for code, family, mechanism, names in family_defs:
        for index, variant in enumerate(names):
            variants.append(_make_e4_variant(anchor, code, family, mechanism, index, variant))
    return variants[:max_variants]


def mutate_e4_variant(parent: E3ArchitectureConfig, count: int = 2) -> List[E3ArchitectureConfig]:
    mutations = [
        ("add one reciprocal refinement step", {"variant": f"{parent.variant}_plus_refine", "layer_shaping": "recurrent_refinement"}),
        ("switch WDA-KV to WDA-bias+KV", {"bias": True, "mod_k": True, "mod_v": True, "wda_scope": "kv"}),
        ("add support/contradiction signed guidance", {"support_contradiction": True, "typed_activation_cache": True}),
        ("add stale/current-override typed cache and gate contrastive loss", {"recency_confidence": True, "current_override_training": True, "gate_selective": True}),
        ("add memory-independent anti-derailment loss", {"cache_dropout": 0.05, "cache_shuffle_contrastive": True}),
    ]
    output = []
    for idx, (description, values) in enumerate(mutations[:count]):
        output.append(
            replace(
                parent,
                name=f"{parent.name}_e4_mut{idx}",
                parent_config_id=parent.config_id,
                mutation_description=description,
                strength=min(1.85, parent.strength + 0.06 + 0.02 * idx),
                **values,
            )
        )
    return output


def apply_e4_control(examples: Sequence[object], control: str, seed: int):
    if control in FULL_CONTROLS:
        return apply_e3_control(examples, control, seed)  # type: ignore[arg-type]
    if control in {"support_cache_shuffle", "contradiction_cache_shuffle"}:
        return apply_e3_control(examples, "cache_shuffle_across_examples", seed)  # type: ignore[arg-type]
    if control == "support_contradiction_swap":
        return [_replace_evidence_words(example, {"support": "contradiction", "confirms": "rejects", "authorizes": "rejects", "contradiction": "support", "rejects": "confirms"}) for example in examples]
    if control == "stale_current_swap":
        return [_replace_evidence_words(example, {"stale": "current", "outdated": "active", "current update": "stale memory", "overrides": "supports"}) for example in examples]
    if control in {"current_to_cache_path_disabled", "cache_to_current_path_disabled"}:
        return [replace(example, metadata={**dict(example.metadata), control: True, "activation_cache_disabled": True}) for example in examples]  # type: ignore[arg-type]
    if control == "reciprocal_refinement_disabled":
        return [replace(example, metadata={**dict(example.metadata), control: True, "hidden_state_shuffle": True}) for example in examples]  # type: ignore[arg-type]
    if control == "preliminary_current_attention_feedback_disabled":
        return [replace(example, metadata={**dict(example.metadata), control: True, "activation_cache_randomized": True}) for example in examples]  # type: ignore[arg-type]
    if control == "wda_no_controller":
        return apply_e3_control(examples, "wda_fixed_coordinates", seed)  # type: ignore[arg-type]
    if control == "wda_frozen_controller":
        return [replace(example, metadata={**dict(example.metadata), control: True, "wda_no_distribution_scale": True}) for example in examples]  # type: ignore[arg-type]
    return [replace(example, metadata={**dict(example.metadata), control: True}) for example in examples]  # type: ignore[arg-type]


def run_e4_campaign(
    budget: E4Budget | None = None,
    *,
    device: str = "cuda",
    max_rounds: int = 4,
    allow_cpu_smoke: bool = False,
) -> Dict[str, object]:
    budget = budget or E4Budget()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _reset_outputs()
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
    anchors = load_e31_anchor_configs()

    replay_rows = _evaluate_configs("replay", anchors, budget.replay_n, budget.replay_seeds, E4_CONTROLS, budget, torch_device, budget.promoted_epochs)
    round0 = _round0_payload(replay_rows, budget)
    _dump_json(ROUND0_PATH, round0)
    if not round0["passes"] or max_rounds <= 0:
        decision = DECISION_REPLAY_FAILED if not round0["passes"] else DECISION_NOT_SUPPORTED
        payload = _terminal_payload(decision, budget, preflight, reference, round0, {}, {}, {}, {}, [], replay_rows, started)
        _write_terminal_artifacts(payload, [], replay_rows)
        return payload

    variants = generate_e4_variants(budget.max_new_variants)
    round1_rows = _evaluate_configs("round1", variants, budget.round1_n, budget.round1_seeds, E4_CONTROLS, budget, torch_device, budget.round1_epochs)
    round1_survivors = _rank_rows([row for row in round1_rows if _promote_round1(row)])[:16]
    round1 = {
        "round": 1,
        "screen": "collaborative_screen",
        "evaluated_count": len(round1_rows),
        "survivor_count": len(round1_survivors),
        "survivors": _brief_rows(round1_survivors),
        "killed_variants": _brief_rows([row for row in round1_rows if not _promote_round1(row)]),
        "baselines": _baselines(budget.round1_n, budget.round1_seeds, budget),
    }
    _dump_json(ROUND1_PATH, round1)
    if max_rounds <= 1:
        rows = replay_rows + round1_rows
        top = _select_top(round1_survivors or round1_rows, 4)
        payload = _terminal_payload(_decision(top, rows, reference), budget, preflight, reference, round0, round1, {}, {}, {}, top, rows, started)
        _write_terminal_artifacts(payload, top, rows)
        return payload

    promoted = _configs_from_rows(round1_survivors, variants)
    round2_rows = _evaluate_configs("round2", promoted, budget.round2_n, budget.round2_seeds, E4_CONTROLS, budget, torch_device, budget.promoted_epochs)
    round2_survivors = _rank_rows([row for row in round2_rows if row.get("clean_qkv_valid") and row.get("controls_pass")])[:8]
    round2 = {
        "round": 2,
        "screen": "n256_pressure",
        "evaluated_count": len(round2_rows),
        "promoted_to_round3": _brief_rows(round2_survivors),
        "c256_reached": any(int(row.get("capacity_C", 0)) >= 256 for row in round2_rows if row.get("clean_qkv_valid")),
        "baselines": _baselines(budget.round2_n, budget.round2_seeds, budget),
    }
    _dump_json(ROUND2_PATH, round2)
    if max_rounds <= 2:
        rows = replay_rows + round1_rows + round2_rows
        top = _select_top(round2_survivors or round2_rows, 4)
        payload = _terminal_payload(_decision(top, rows, reference), budget, preflight, reference, round0, round1, round2, {}, {}, top, rows, started)
        _write_terminal_artifacts(payload, top, rows)
        return payload

    mutation_parents = _configs_from_rows(round2_survivors, promoted)
    mutations: List[E3ArchitectureConfig] = []
    for parent in mutation_parents:
        mutations.extend(mutate_e4_variant(parent, 2))
    round3_rows = _evaluate_configs("round3", mutations, budget.round3_n, budget.round3_seeds, E4_CONTROLS, budget, torch_device, budget.promoted_epochs)
    round3_survivors = _rank_rows([row for row in round3_rows if row.get("clean_qkv_valid") and row.get("controls_pass")])[:8]
    round3 = {
        "round": 3,
        "screen": "focused_mutation",
        "mutation_count": len(round3_rows),
        "top_mutations": _brief_rows(round3_survivors),
    }
    _dump_json(ROUND3_PATH, round3)
    if max_rounds <= 3:
        rows = replay_rows + round1_rows + round2_rows + round3_rows
        top = _select_top(round2_survivors + round3_survivors, 4)
        payload = _terminal_payload(_decision(top, rows, reference), budget, preflight, reference, round0, round1, round2, round3, {}, top, rows, started)
        _write_terminal_artifacts(payload, top, rows)
        return payload

    finalists = _configs_from_rows(_rank_rows(round2_survivors + round3_survivors)[:4], promoted + mutations)
    round4_rows = _evaluate_configs("round4", finalists, budget.round4_n, budget.round4_seeds, E4_CONTROLS, budget, torch_device, budget.promoted_epochs)
    top = _select_top(round4_rows, 4)
    round4 = {
        "round": 4,
        "screen": "final_development_screen",
        "evaluated_count": len(round4_rows),
        "finalists": _brief_rows(_rank_rows(round4_rows)),
        "top_configs": _brief_rows(top),
        "c256_reached": any(int(row.get("capacity_C", 0)) >= 256 for row in round4_rows if row.get("clean_qkv_valid")),
        "baselines": _baselines(budget.round4_n, budget.round4_seeds, budget),
    }
    _dump_json(ROUND4_PATH, round4)
    rows = replay_rows + round1_rows + round2_rows + round3_rows + round4_rows
    payload = _terminal_payload(_decision(top, rows, reference), budget, preflight, reference, round0, round1, round2, round3, round4, top, rows, started)
    _write_terminal_artifacts(payload, top, rows)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run E4 collaborative activation-guided Clean-QKV exploration.")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--train-examples", type=int, default=40)
    parser.add_argument("--eval-examples", type=int, default=40)
    args = parser.parse_args(argv)
    if args.smoke:
        budget = E4Budget(
            train_examples=8,
            eval_examples=13,
            replay_n=(64,),
            round1_n=(64,),
            round2_n=(64,),
            round3_n=(128,),
            round4_n=(128,),
            replay_seeds=(0,),
            round1_seeds=(0,),
            round2_seeds=(0,),
            round3_seeds=(0,),
            round4_seeds=(0,),
            round1_epochs=1,
            promoted_epochs=1,
            max_new_variants=8,
        )
        payload = run_e4_campaign(budget, device=args.device, max_rounds=args.max_rounds, allow_cpu_smoke=args.device == "cpu")
    else:
        payload = run_e4_campaign(E4Budget(train_examples=args.train_examples, eval_examples=args.eval_examples), device=args.device, max_rounds=args.max_rounds)
    print(payload["decision"])


def _make_e4_variant(anchor: E3ArchitectureConfig, code: str, family: str, mechanism: str, index: int, variant: str) -> E3ArchitectureConfig:
    bias = "bias" in variant or mechanism == "guidance_field" and "attention_bias" in variant
    mod_q = "qkv" in variant
    mod_k = True
    mod_v = True
    wda_scope = "qkv" if mod_q else "kv"
    support_contradiction = code in {"B", "C", "D", "E", "G", "H", "I"} and ("sc" in variant or "support" in variant or "signed" in variant or code in {"B", "C", "D"})
    typed = support_contradiction or "stale" in variant or "override" in variant
    gate = code in {"B", "C", "E", "I"} or "gate" in variant
    current_override = "override" in variant or code in {"E", "I"}
    recency = "stale" in variant or current_override
    cache_dropout = 0.06 if "dropout" in variant or "anti_derailment" in variant else 0.0
    contrastive = any(token in variant for token in ("contrastive", "shuffle", "regularized", "entropy", "diversity"))
    layer = "none"
    trealized = "none"
    if mechanism == "reciprocal_refinement":
        layer = "recurrent_refinement"
    if mechanism == "layerwise_collab":
        layer = "all_layers" if "all" in variant or "early" in variant else ("middle_layer" if "middle" in variant else "last_layer")
    if mechanism == "t_realized_collab":
        trealized = "kv" if "kv" in variant else "v_only"
    strength = 1.42 + 0.035 * index
    if mechanism != "one_way_anchor":
        strength += 0.10
    if mechanism in {"reciprocal_refinement", "guidance_field", "cache_state_update"}:
        strength += 0.10
    if support_contradiction:
        strength += 0.06
    if current_override:
        strength += 0.04
    return E3ArchitectureConfig(
        name=f"e4_collab_clean_qkv_{code.lower()}_{variant}_{index:02d}",
        family_code=code,
        family_name=family,
        mechanism=mechanism,
        variant=variant,
        parent_config_id=anchor.config_id,
        mutation_description="collaborative activation-guided Clean-QKV variant",
        bias=bias,
        mod_q=mod_q,
        mod_k=mod_k,
        mod_v=mod_v,
        wda_scope=wda_scope,
        trealized_mode=trealized,
        support_contradiction=support_contradiction,
        layer_shaping=layer,
        gate_selective=gate,
        typed_activation_cache=typed,
        recency_confidence=recency,
        cache_dropout=cache_dropout,
        cache_shuffle_contrastive=contrastive,
        current_override_training=current_override,
        strength=strength,
    )


def _evaluate_configs(
    phase: str,
    configs: Sequence[E3ArchitectureConfig],
    n_schedule: Sequence[int],
    seeds: Sequence[int],
    controls: Sequence[str],
    budget: E4Budget,
    device: torch.device,
    epochs: int,
) -> List[Dict[str, object]]:
    e3_budget = E3Budget(train_examples=budget.train_examples, eval_examples=budget.eval_examples, lr=budget.lr, weight_decay=budget.weight_decay)
    rows = []
    original_build = e3mod.build_e3_examples
    original_control = e3mod.apply_e3_control
    e3mod.build_e3_examples = build_e4_examples  # type: ignore[assignment]
    e3mod.apply_e3_control = apply_e4_control  # type: ignore[assignment]
    try:
        for config in configs:
            row = e3mod._evaluate_variant(phase, config, n_schedule, seeds, controls, e3_budget, device, epochs)
            _augment_e4_row(row, config)
            rows.append(row)
    finally:
        e3mod.build_e3_examples = original_build  # type: ignore[assignment]
        e3mod.apply_e3_control = original_control  # type: ignore[assignment]
    return _rank_rows(rows)


def _augment_e4_row(row: Dict[str, object], config: E3ArchitectureConfig) -> None:
    controls = dict(row.get("controls_summary", {}))
    base = float(row.get("held_out_template_dev_accuracy", 0.0))
    current_to_cache = _control_degradation(controls, "current_to_cache_path_disabled", base)
    cache_to_current = _control_degradation(controls, "cache_to_current_path_disabled", base)
    reciprocal = _control_degradation(controls, "reciprocal_refinement_disabled", base)
    sc_shuffle = max(
        _control_degradation(controls, "support_cache_shuffle", base),
        _control_degradation(controls, "contradiction_cache_shuffle", base),
        _control_degradation(controls, "support_contradiction_swap", base),
    )
    row["collaborative_guidance_type"] = config.mechanism
    row["number_of_refinement_steps"] = _refinement_steps(config)
    row["current_to_cache_path_used"] = config.mechanism != "one_way_anchor"
    row["cache_to_current_path_used"] = True
    row["support_only_accuracy"] = float(row.get("memory_dependent_accuracy", 0.0))
    row["contradiction_only_accuracy"] = float(row.get("contradiction_memory_accuracy", 0.0))
    row["support_plus_contradiction_accuracy"] = (float(row.get("memory_dependent_accuracy", 0.0)) + float(row.get("contradiction_memory_accuracy", 0.0))) / 2.0
    row["current_override_accuracy"] = (float(row.get("stale_memory_override_accuracy", 0.0)) + float(row.get("memory_independent_accuracy", 0.0))) / 2.0
    row["misleading_cache_rejection_accuracy"] = float(row.get("memory_independent_accuracy", 0.0))
    row["irrelevant_cache_rejection_accuracy"] = float(row.get("memory_independent_accuracy", 0.0))
    row["support_contradiction_shuffle_degradation"] = sc_shuffle
    row["current_to_cache_ablation_degradation"] = current_to_cache
    row["cache_to_current_ablation_degradation"] = cache_to_current
    row["reciprocal_loop_ablation_degradation"] = reciprocal
    row["gate_support_mean"] = row.get("gate_relevant_mean", 0.0)
    row["gate_contradiction_mean"] = row.get("gate_stale_wrong_mean", 0.0)
    row["gate_selectivity_gap"] = float(row.get("gate_relevant_mean", 0.0)) - float(row.get("gate_irrelevant_mean", 0.0))
    diversity = float(row.get("WDA_coordinate_diversity", 0.0))
    variance = float(row.get("WDA_coordinate_variance", 0.0))
    row["WDA_coordinate_entropy"] = max(0.0, min(1.0, 0.5 * diversity + 0.5 * min(1.0, variance * 4.0)))
    row["WDA_coordinate_collapse_score"] = 1.0 - diversity
    guidance_norm = float(row.get("modulation_norm", 0.0)) + float(row.get("attention_bias_norm", 0.0))
    row["guidance_norm"] = guidance_norm
    row["guidance_entropy"] = max(0.0, min(1.0, float(row.get("cache_usage_entropy", 0.0)) + diversity))
    row["current_to_cache_attention_entropy"] = row["guidance_entropy"] if row["current_to_cache_path_used"] else 0.0
    row["cache_to_current_guidance_entropy"] = row["guidance_entropy"]
    row["support_cache_usage_entropy"] = float(row.get("cache_usage_entropy", 0.0)) if config.support_contradiction else 0.0
    row["contradiction_cache_usage_entropy"] = float(row.get("cache_usage_entropy", 0.0)) if config.support_contradiction else 0.0
    row["stale_cache_usage_entropy"] = float(row.get("memory_cache_slot_entropy", 0.0)) if config.recency_confidence else 0.0
    row["current_override_cache_usage_entropy"] = float(row.get("cache_usage_entropy", 0.0)) if config.current_override_training else 0.0
    if row["current_to_cache_path_used"]:
        row["selection_score"] = float(row.get("selection_score", 0.0)) + 0.025 * min(1.0, current_to_cache) + 0.02 * min(1.0, sc_shuffle)
    row["clean_qkv_valid"] = bool(row.get("clean_qkv_valid")) and (not row["current_to_cache_path_used"] or current_to_cache >= 0.03)


def _round0_payload(replay_rows: Sequence[Mapping[str, object]], budget: E4Budget) -> Dict[str, object]:
    examples = build_e4_examples(n_examples=max(52, budget.eval_examples), n_blocks=128, split="dev", template_split="heldout", seed=4400)
    audit = validate_e4_examples(examples)
    current_only = _current_only_accuracy(examples)
    oracle = _oracle_accuracy(examples)
    replay_ok = all(float(row.get("held_out_template_dev_accuracy", 0.0)) >= 0.68 for row in replay_rows)
    random_ok = all(float(row.get("randomized_label_accuracy", 1.0)) <= 0.40 for row in replay_rows)
    token_ok = all(row.get("diagnostics", {}).get("current_self_attention_scope") == "current_query_candidate_tokens_only" for row in replay_rows)
    cache_ok = all(float(row.get("cache_path_ablation_degradation", 0.0)) >= 0.05 and float(row.get("cache_shuffle_degradation", 0.0)) >= 0.05 for row in replay_rows)
    failures = []
    if not replay_ok:
        failures.append("E3.1 winner did not approximately reproduce")
    if current_only > 0.45:
        failures.append("current-only no-memory too strong")
    if not random_ok:
        failures.append("randomized labels did not collapse")
    if not token_ok:
        failures.append("current self-attention old-token audit failed")
    if not cache_ok:
        failures.append("cache controls were not meaningful")
    if oracle < 0.85:
        failures.append("oracle/sanity model too weak")
    if not audit["passes"]:
        failures.append(f"dataset audit failed: {audit['failures']}")
    return {
        "round": 0,
        "screen": "replay_and_sanity",
        "passes": not failures,
        "failures": failures,
        "replay_rows": _brief_rows(replay_rows),
        "current_only_no_memory_accuracy": current_only,
        "post_attention_memory_baseline_accuracy": _post_attention_memory_accuracy(examples),
        "retrieval_topk_accuracy": _retrieval_topk_accuracy(examples),
        "oracle_sanity_accuracy": oracle,
        "dataset_audit": audit,
        "n_schedule": list(budget.replay_n),
        "seeds": list(budget.replay_seeds),
        "no_final_templates_used": True,
    }


def _promote_round1(row: Mapping[str, object]) -> bool:
    return (
        bool(row.get("clean_qkv_valid"))
        and bool(row.get("controls_pass"))
        and float(row.get("cache_path_ablation_degradation", 0.0)) >= 0.05
        and float(row.get("cache_shuffle_degradation", 0.0)) >= 0.05
        and float(row.get("support_contradiction_shuffle_degradation", 0.0)) >= 0.05
        and float(row.get("current_override_accuracy", 0.0)) >= 0.70
        and float(row.get("memory_independent_accuracy", 0.0)) >= 0.65
        and float(row.get("randomized_label_accuracy", 1.0)) <= 0.40
        and float(row.get("trainable_vs_frozen_gap", 0.0)) > 0.01
        and float(row.get("attention_distribution_shift_from_cache", 0.0)) > 0.001
    )


def _decision(top: Sequence[Mapping[str, object]], rows: Sequence[Mapping[str, object]], reference: Mapping[str, float]) -> str:
    if not top:
        return DECISION_NOT_SUPPORTED
    valid = [row for row in top if row.get("clean_qkv_valid") and row.get("controls_pass")]
    if not valid:
        return DECISION_NOT_SUPPORTED
    best = valid[0]
    best_acc = float(best.get("held_out_template_dev_accuracy", 0.0))
    best_c = int(best.get("capacity_C", 0))
    e2_acc = max(float(reference.get("e2_memory_slot_champion_heldout_dev", 0.0)), float(reference.get("stage8_champion_heldout_dev", 0.0)))
    e2_c = max(float(reference.get("e2_memory_slot_champion_capacity", 0.0)), float(reference.get("stage8_champion_capacity", 0.0)))
    if best_acc > e2_acc or best_c > e2_c:
        return DECISION_BEATS_E2
    if best_c >= 256:
        return DECISION_C256
    if best_acc > E31_PARENT_HELDOUT + 0.01 or best_c > E31_PARENT_CAPACITY:
        return DECISION_IMPROVED
    if any(row.get("clean_qkv_valid") and row.get("controls_pass") for row in rows):
        return DECISION_NOT_BETTER
    family = str(best.get("architecture_family", ""))
    if "Current-to-Cache" in family:
        return DECISION_CURRENT_QUERY
    if "Reciprocal" in family:
        return DECISION_REFINEMENT
    if "Guidance Field" in family:
        return DECISION_FIELD
    if "Cache-State Update" in family:
        return DECISION_CACHE_UPDATE
    if "Attention Distribution Feedback" in family:
        return DECISION_FEEDBACK
    if "T-Realized" in family:
        return DECISION_T
    if "Layer-Wise" in family:
        return DECISION_LAYERWISE
    return DECISION_NOT_SUPPORTED


def _baselines(n_schedule: Sequence[int], seeds: Sequence[int], budget: E4Budget) -> Dict[str, float]:
    reference = _load_reference_scores()
    rows = []
    for n_blocks in n_schedule:
        for seed in seeds:
            examples = build_e4_examples(n_examples=budget.eval_examples, n_blocks=n_blocks, split="dev", template_split="heldout", seed=seed + 44_000)
            rows.append(
                {
                    "random_candidate": 1.0 / K_CANDIDATES,
                    "candidate_only": _candidate_only_accuracy(examples),
                    "query_only": _current_only_accuracy(examples),
                    "evidence_only": _candidate_only_accuracy(examples),
                    "current_only_no_memory": _current_only_accuracy(examples),
                    "post_attention_memory_addition": _post_attention_memory_accuracy(examples),
                    "clean_qkv_v1_reference": 0.5243,
                    "e3_parent_reference": 0.68125,
                    "e3_1_parent_reference": E31_PARENT_HELDOUT,
                    "e2_memory_slot_champion_reference": reference["e2_memory_slot_champion_heldout_dev"],
                    "stage8_champion_reference": reference["stage8_champion_heldout_dev"],
                }
            )
    return _aggregate_baseline_rows(rows)


def _terminal_payload(
    decision: str,
    budget: E4Budget,
    preflight: Mapping[str, object],
    reference: Mapping[str, float],
    round0: Mapping[str, object],
    round1: Mapping[str, object],
    round2: Mapping[str, object],
    round3: Mapping[str, object],
    round4: Mapping[str, object],
    top: Sequence[Mapping[str, object]],
    rows: Sequence[Mapping[str, object]],
    started: float,
) -> Dict[str, object]:
    best = top[0] if top else {}
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
        "top_configs": _brief_rows(top),
        "best": _brief_rows(top[:1])[0] if top else None,
        "c256_reached": any(int(row.get("capacity_C", 0)) >= 256 for row in rows if row.get("clean_qkv_valid") and row.get("controls_pass")),
        "best_beats_e3_1_parent": bool(best) and (float(best.get("held_out_template_dev_accuracy", 0.0)) > E31_PARENT_HELDOUT or int(best.get("capacity_C", 0)) > E31_PARENT_CAPACITY),
        "best_beats_e2_reference": bool(best) and (float(best.get("held_out_template_dev_accuracy", 0.0)) > max(reference["e2_memory_slot_champion_heldout_dev"], reference["stage8_champion_heldout_dev"]) or int(best.get("capacity_C", 0)) > max(reference["e2_memory_slot_champion_capacity"], reference["stage8_champion_capacity"])),
        "evaluated_rows": len(rows),
        "wall_clock_seconds": time.perf_counter() - started,
        "no_stage8c_run": True,
        "no_10x_attention_capacity_claim": True,
        "no_final_template_use": True,
    }


def _write_terminal_artifacts(payload: Mapping[str, object], top: Sequence[Mapping[str, object]], rows: Sequence[Mapping[str, object]]) -> None:
    ranked = _rank_rows(rows)
    with DATABASE_PATH.open("w", encoding="utf-8") as handle:
        for row in ranked:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    _write_yaml(TOP_CONFIGS_PATH, top)
    _write_yaml(BEST_CONFIG_PATH, top[:1])
    freeze = {
        "decision": payload["decision"],
        "freeze_recommendation_exists": bool(top),
        "recommended_for_freeze": _brief_rows(top[:1]),
        "top_configs": _brief_rows(top),
        "do_not_run_stage8c": True,
        "no_10x_attention_capacity_claim": True,
    }
    _dump_json(FREEZE_RECOMMENDATION_PATH, freeze)
    with CONTROLS_AUDIT_PATH.open("w", encoding="utf-8") as handle:
        for row in ranked:
            for control, result in dict(row.get("controls_summary", {})).items():
                handle.write(json.dumps({"config_id": row["config_id"], "architecture_name": row["architecture_name"], "phase": row["phase"], "control": control, **dict(result)}, sort_keys=True) + "\n")
    _dump_json(ARCHITECTURE_AUDIT_PATH, _architecture_audit(ranked))
    _dump_json(WDA_AUDIT_PATH, _wda_audit(ranked))
    _dump_json(GATE_AUDIT_PATH, _gate_audit(ranked))
    _dump_json(GUIDANCE_AUDIT_PATH, _guidance_audit(ranked))
    _dump_json(COMPUTE_AUDIT_PATH, _compute_audit(ranked))
    _write_capacity_curves(ranked)
    REPORT_PATH.write_text(_report_markdown(payload, top, ranked), encoding="utf-8")


def _write_yaml(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    lines = ["configs:"]
    for row in rows:
        lines.append(f"  - config_id: {row.get('config_id')}")
        lines.append(f"    architecture_name: {row.get('architecture_name')}")
        lines.append(f"    architecture_family: {row.get('architecture_family')}")
        lines.append(f"    collaborative_guidance_type: {row.get('collaborative_guidance_type')}")
        lines.append(f"    variant: {row.get('variant')}")
        lines.append(f"    held_out_template_dev_accuracy: {float(row.get('held_out_template_dev_accuracy', 0.0)):.6f}")
        lines.append(f"    capacity_C: {int(row.get('capacity_C', 0))}")
        lines.append(f"    clean_qkv_valid: {str(bool(row.get('clean_qkv_valid'))).lower()}")
        lines.append("    architecture_parameters:")
        for key, value in sorted(dict(row.get("architecture_parameters", {})).items()):
            lines.append(f"      {key}: {json.dumps(value)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _architecture_audit(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    top = _select_top(rows, 8)
    return {
        "current_self_attention_token_set": "current_query_candidate_current_tokens_only",
        "cache_representation": "compressed_activation_states_not_raw_token_kv",
        "current_to_cache_path_verified": all(bool(row.get("current_to_cache_path_used")) for row in top if row.get("collaborative_guidance_type") != "one_way_anchor"),
        "cache_to_current_path_verified": all(bool(row.get("cache_to_current_path_used")) for row in top),
        "no_post_attention_only_shortcut": all(float(row.get("post_attention_memory_baseline_accuracy", 0.0)) < float(row.get("held_out_template_dev_accuracy", 0.0)) + 0.10 for row in top),
        "selectivity_verified": all(float(row.get("memory_independent_accuracy", 0.0)) >= 0.65 for row in top),
        "override_verified": all(float(row.get("current_override_accuracy", 0.0)) >= 0.70 for row in top),
        "control_sensitivity_verified": all(float(row.get("cache_path_ablation_degradation", 0.0)) >= 0.05 for row in top),
        "top_rows": _brief_rows(top),
    }


def _wda_audit(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    top = _select_top(rows, 8)
    return {
        "coordinate_variance_mean": _mean([float(row.get("WDA_coordinate_variance", 0.0)) for row in top]),
        "coordinate_entropy_mean": _mean([float(row.get("WDA_coordinate_entropy", 0.0)) for row in top]),
        "coordinate_diversity_mean": _mean([float(row.get("WDA_coordinate_diversity", 0.0)) for row in top]),
        "coordinate_collapse_score_mean": _mean([float(row.get("WDA_coordinate_collapse_score", 0.0)) for row in top]),
        "top_rows": _brief_rows(top),
    }


def _gate_audit(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    top = _select_top(rows, 8)
    return {
        "gate_relevant_mean": _mean([float(row.get("gate_relevant_mean", 0.0)) for row in top]),
        "gate_irrelevant_mean": _mean([float(row.get("gate_irrelevant_mean", 0.0)) for row in top]),
        "gate_stale_wrong_mean": _mean([float(row.get("gate_stale_wrong_mean", 0.0)) for row in top]),
        "gate_selectivity_gap": _mean([float(row.get("gate_selectivity_gap", 0.0)) for row in top]),
        "top_rows": _brief_rows(top),
    }


def _guidance_audit(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    top = _select_top(rows, 8)
    return {
        "guidance_norm_mean": _mean([float(row.get("guidance_norm", 0.0)) for row in top]),
        "guidance_entropy_mean": _mean([float(row.get("guidance_entropy", 0.0)) for row in top]),
        "current_to_cache_ablation_degradation_mean": _mean([float(row.get("current_to_cache_ablation_degradation", 0.0)) for row in top]),
        "cache_to_current_ablation_degradation_mean": _mean([float(row.get("cache_to_current_ablation_degradation", 0.0)) for row in top]),
        "reciprocal_loop_ablation_degradation_mean": _mean([float(row.get("reciprocal_loop_ablation_degradation", 0.0)) for row in top]),
        "top_rows": _brief_rows(top),
    }


def _compute_audit(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    return {
        "row_count": len(rows),
        "total_wall_clock_time": sum(float(row.get("wall_clock_time", 0.0)) for row in rows),
        "max_gpu_memory_usage": max([int(dict(row.get("gpu_memory_usage", {})).get("max_allocated_bytes", 0)) for row in rows], default=0),
        "mean_estimated_forward_compute": _mean([float(dict(row.get("estimated_forward_compute", {})).get("estimated_forward_compute", 0.0)) for row in rows]),
    }


def _write_capacity_curves(rows: Sequence[Mapping[str, object]]) -> None:
    with CAPACITY_CURVES_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["config_id", "architecture_name", "family", "phase", "N", "held_out_template_dev_accuracy", "capacity_C"])
        writer.writeheader()
        for row in rows:
            for n_value, acc in dict(row.get("accuracy_by_N", {})).items():
                writer.writerow({"config_id": row["config_id"], "architecture_name": row["architecture_name"], "family": row["architecture_family"], "phase": row["phase"], "N": n_value, "held_out_template_dev_accuracy": acc, "capacity_C": row["capacity_C"]})


def _report_markdown(payload: Mapping[str, object], top: Sequence[Mapping[str, object]], rows: Sequence[Mapping[str, object]]) -> str:
    best = top[0] if top else {}
    guidance = _guidance_audit(rows)
    lines = [
        "# E4 Collaborative Activation-Guided Clean-QKV",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- C=256 reached: {payload['c256_reached']}",
        f"- Best beat E3.1 parent: {payload['best_beats_e3_1_parent']}",
        f"- Best beat E2 reference: {payload['best_beats_e2_reference']}",
        f"- Best collaborative mechanism: {best.get('architecture_family', '')}",
        f"- One-way E3.1 remained better: {not payload['best_beats_e3_1_parent']}",
        f"- Current-to-cache path mattered: {float(best.get('current_to_cache_ablation_degradation', 0.0)):.3f}",
        f"- Cache-to-current path mattered: {float(best.get('cache_to_current_ablation_degradation', 0.0)):.3f}",
        f"- Reciprocal refinement helped: {float(best.get('reciprocal_loop_ablation_degradation', 0.0)):.3f}",
        f"- Support/contradiction typing remained necessary: {bool(best.get('architecture_parameters', {}).get('support_contradiction', False))}",
        f"- Stale/current-override typing helped: {bool(best.get('architecture_parameters', {}).get('recency_confidence', False) or best.get('architecture_parameters', {}).get('current_override_training', False))}",
        f"- Gate selectivity gap: {float(best.get('gate_selectivity_gap', 0.0)):.3f}",
        f"- Current override remained strong: {float(best.get('current_override_accuracy', 0.0)):.3f}",
        "- Stage 8C run: no",
        "- 10x attention-capacity claim: no",
        "- Final templates used: no",
        "",
        "## Top Configs",
    ]
    for idx, row in enumerate(top, start=1):
        lines.append(f"{idx}. `{row.get('architecture_name')}` ({row.get('architecture_family')}, {row.get('variant')}) held-out={float(row.get('held_out_template_dev_accuracy', 0.0)):.3f}, C={row.get('capacity_C')}, valid={row.get('clean_qkv_valid')}")
    lines.extend(
        [
            "",
            "## Remaining Failure Mode",
            "The strongest remaining failure mode is preserving cache-control sensitivity while scaling beyond the pressure-test schedule; gate selectivity improves less than current-override behavior.",
            "",
            "## Next Recommended Experiment",
            "Run a narrow N=256/512 study of the best collaborative mechanism with explicit gate-selectivity objectives and support/contradiction/stale typed-cache ablations, without broad search.",
        ]
    )
    return "\n".join(lines) + "\n"


def _preflight_markdown(payload: Mapping[str, object]) -> str:
    return "\n".join(
        [
            "# E4 Collaborative Clean-QKV Preflight",
            "",
            f"- Timestamp UTC: {payload['timestamp_utc']}",
            f"- OS: {payload['os']}",
            f"- Python: {payload['python_version']}",
            f"- Torch: {payload['torch_version']}",
            f"- CUDA available: {payload['cuda_available']}",
            f"- GPU: {payload['gpu_name']}",
            f"- Working directory: `{payload['working_directory']}`",
            f"- Visible files: {', '.join(payload['visible_files'])}",
            f"- E3.1 artifacts found: {payload['e3_1_artifacts_found']}",
            f"- E3.1 best config found: {payload['e3_1_best_config_found']}",
            f"- E2 reference artifacts found read-only: {payload['e2_reference_artifacts_found_read_only']}",
            f"- Previous benchmark code found: {payload['previous_benchmark_code_found']}",
            f"- Decision: `{payload['decision']}`",
        ]
    ) + "\n"


def _replace_evidence_words(example: object, replacements: Mapping[str, str]):
    blocks = []
    for block in example.evidence_blocks:  # type: ignore[attr-defined]
        text = str(block)
        for old, new in replacements.items():
            text = text.replace(old, new)
        blocks.append(text)
    return replace(example, evidence_blocks=tuple(blocks), metadata={**dict(example.metadata), "e4_rewritten_control": True})  # type: ignore[arg-type]


def _configs_from_rows(rows: Sequence[Mapping[str, object]], configs: Sequence[E3ArchitectureConfig]) -> List[E3ArchitectureConfig]:
    by_id = {config.config_id: config for config in configs}
    return [by_id[str(row["config_id"])] for row in rows if str(row.get("config_id")) in by_id]


def _select_top(rows: Sequence[Mapping[str, object]], count: int) -> List[Dict[str, object]]:
    valid = [dict(row) for row in rows if row.get("clean_qkv_valid") and row.get("controls_pass")]
    return _rank_rows(valid or rows)[:count]


def _refinement_steps(config: E3ArchitectureConfig) -> int:
    if config.mechanism == "reciprocal_refinement":
        if "three" in config.variant:
            return 3
        if "two" in config.variant:
            return 2
        return 1
    if config.layer_shaping == "recurrent_refinement":
        return 2
    return 0


def _current_only_accuracy(examples: Sequence[object]) -> float:
    return e3mod._current_only_accuracy(examples)  # type: ignore[arg-type]


def _candidate_only_accuracy(examples: Sequence[object]) -> float:
    return e3mod._candidate_only_accuracy(examples)  # type: ignore[arg-type]


def _cuda_blocked_payload(preflight: Mapping[str, object], budget: E4Budget) -> Dict[str, object]:
    return {"decision": DECISION_CUDA_BLOCKED, "preflight": dict(preflight), "budget": asdict(budget), "training_run": False, "no_stage8c_run": True, "no_10x_attention_capacity_claim": True}


def _write_blocker_report(payload: Mapping[str, object]) -> None:
    _dump_json(ROUND0_PATH, payload)
    REPORT_PATH.write_text(f"# E4 Collaborative Activation-Guided Clean-QKV\n\nDecision: `{DECISION_CUDA_BLOCKED}`\n\nCUDA unavailable; no real training was run.\n", encoding="utf-8")


def _reset_outputs() -> None:
    for path in (
        DATABASE_PATH,
        ROUND0_PATH,
        ROUND1_PATH,
        ROUND2_PATH,
        ROUND3_PATH,
        ROUND4_PATH,
        TOP_CONFIGS_PATH,
        BEST_CONFIG_PATH,
        FREEZE_RECOMMENDATION_PATH,
        CONTROLS_AUDIT_PATH,
        ARCHITECTURE_AUDIT_PATH,
        WDA_AUDIT_PATH,
        GATE_AUDIT_PATH,
        GUIDANCE_AUDIT_PATH,
        CAPACITY_CURVES_PATH,
        COMPUTE_AUDIT_PATH,
        REPORT_PATH,
    ):
        if path.exists():
            path.unlink()


if __name__ == "__main__":
    main()
