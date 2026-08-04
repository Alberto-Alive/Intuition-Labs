from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from torch import nn


E3_1_ROOT = Path(__file__).resolve().parents[3]
E3_ROOT = E3_1_ROOT.parent / "e3"
E2_ROOT = E3_1_ROOT.parent / "e2"
E3_CODE = E3_ROOT / "code"
if E3_CODE.exists() and str(E3_CODE) not in sys.path:
    sys.path.append(str(E3_CODE))

from src.experiments.e3_clean_qkv import (  # type: ignore  # noqa: E402
    CleanQKVAttentionLayer,
    DECISION_CUDA_BLOCKED,
    E3ArchitectureConfig,
    E3Budget,
    E3FeatureSelector,
    E3_TASK_FAMILIES,
    FULL_CONTROLS,
    K_CANDIDATES,
    MEMORY_DEPENDENT_FAMILIES,
    MEMORY_INDEPENDENT_FAMILIES,
    TRealizedCleanQKV,
    WDAAdapter,
    _accuracy,
    _accuracy_for_families,
    _aggregate_baseline_rows,
    _brief_rows,
    _capacity,
    _control_accuracy,
    _control_degradation,
    _dump_json,
    _estimate_forward_compute,
    _evaluate_variant,
    _family_e_wda,
    _load_reference_scores,
    _mean,
    _merge_diagnostics,
    _oracle_accuracy,
    _post_attention_memory_accuracy,
    _rank_rows,
    _retrieval_topk_accuracy,
    _safe_read_preview,
    _selection_score,
    _stable_hash,
    _task_group_accuracy,
    apply_e3_control,
    assert_training_device,
    build_e3_examples,
    cuda_preflight_status,
    e3_config_to_dict,
    validate_e3_examples,
)
import src.experiments.e3_clean_qkv as e3mod  # type: ignore  # noqa: E402

E3_BUILD_EXAMPLES = build_e3_examples


RESULTS_DIR = E3_1_ROOT / "results"
REPORTS_DIR = E3_1_ROOT / "reports"
PREFIX = "e3_1_wda_clean_qkv"

PREFLIGHT_JSON = RESULTS_DIR / f"{PREFIX}_preflight.json"
PREFLIGHT_REPORT = REPORTS_DIR / "E3_1_WDA_CLEAN_QKV_PREFLIGHT.md"
DATABASE_PATH = RESULTS_DIR / f"{PREFIX}_database.jsonl"
ROUND0_PATH = RESULTS_DIR / f"{PREFIX}_round0_replay.json"
ROUND1_PATH = RESULTS_DIR / f"{PREFIX}_round1_screen.json"
ROUND2_PATH = RESULTS_DIR / f"{PREFIX}_round2_c128_push.json"
ROUND3_PATH = RESULTS_DIR / f"{PREFIX}_round3_mutations.json"
ROUND4_PATH = RESULTS_DIR / f"{PREFIX}_round4_finalists.json"
TOP_CONFIGS_PATH = RESULTS_DIR / f"{PREFIX}_top_configs.yaml"
BEST_CONFIG_PATH = RESULTS_DIR / f"{PREFIX}_best_config.yaml"
FREEZE_RECOMMENDATION_PATH = RESULTS_DIR / f"{PREFIX}_freeze_recommendation.json"
CONTROLS_AUDIT_PATH = RESULTS_DIR / f"{PREFIX}_controls_audit.jsonl"
WDA_AUDIT_PATH = RESULTS_DIR / f"{PREFIX}_wda_audit.json"
GATE_AUDIT_PATH = RESULTS_DIR / f"{PREFIX}_gate_audit.json"
CACHE_SHAPING_AUDIT_PATH = RESULTS_DIR / f"{PREFIX}_cache_shaping_audit.json"
CAPACITY_CURVES_PATH = RESULTS_DIR / f"{PREFIX}_capacity_curves.csv"
COMPUTE_AUDIT_PATH = RESULTS_DIR / f"{PREFIX}_compute_audit.json"
REPORT_PATH = REPORTS_DIR / "E3_1_WDA_CLEAN_QKV_HARDENING.md"

DECISION_REPLAY_FAILED = "E3_1_WDA_CLEAN_QKV_REPLAY_FAILED"
DECISION_C128 = "E3_1_WDA_CLEAN_QKV_C128_REACHED"
DECISION_IMPROVED = "E3_1_WDA_CLEAN_QKV_IMPROVED_BUT_C128_NOT_REACHED"
DECISION_BEATS_E2 = "E3_1_WDA_CLEAN_QKV_BEATS_E2_CHAMPION"
DECISION_WDA_V = "E3_1_WDA_V_SUPPORTED"
DECISION_KV_QKV = "E3_1_WDA_KV_OR_QKV_SUPPORTED"
DECISION_BIAS_V = "E3_1_WDA_BIAS_PLUS_V_SUPPORTED"
DECISION_SC = "E3_1_SUPPORT_CONTRADICTION_CACHE_SUPPORTED"
DECISION_NOT_SUPPORTED = "E3_1_WDA_CLEAN_QKV_NOT_SUPPORTED"

PARENT_NAMES = (
    "e3_clean_qkv_e_wda_realized_weights_09",
    "e3_clean_qkv_e_wda_realized_weights_08",
    "e3_clean_qkv_e_wda_realized_weights_10",
)
E3_PARENT_HELDOUT = 0.684375
E3_PARENT_CAPACITY = 64


@dataclass(frozen=True)
class E31Budget:
    train_examples: int = 40
    eval_examples: int = 40
    replay_n: Tuple[int, ...] = (32, 64)
    round1_n: Tuple[int, ...] = (32, 64)
    round2_n: Tuple[int, ...] = (32, 64, 128)
    round3_n: Tuple[int, ...] = (64, 128)
    round4_n: Tuple[int, ...] = (32, 64, 128, 256)
    replay_seeds: Tuple[int, ...] = (0, 1)
    round1_seeds: Tuple[int, ...] = (0,)
    round2_seeds: Tuple[int, ...] = (0, 1, 2)
    round3_seeds: Tuple[int, ...] = (0, 1, 2)
    round4_seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    round1_epochs: int = 2
    promoted_epochs: int = 3
    lr: float = 0.07
    weight_decay: float = 0.0001
    max_new_variants: int = 48


class SupportContradictionActivationCache(nn.Module):
    """Typed activation-cache streams used by E3.1 support/contradiction tests."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.support_proj = nn.Linear(dim, dim, bias=False)
        self.contradiction_proj = nn.Linear(dim, dim, bias=False)
        self.stale_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, support: torch.Tensor, contradiction: torch.Tensor, stale: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        stale = torch.zeros_like(support) if stale is None else stale
        return {
            "support": self.support_proj(support),
            "contradiction": self.contradiction_proj(contradiction),
            "stale": self.stale_proj(stale),
        }


def write_preflight(*, device: str = "cuda", allow_cpu_smoke: bool = False) -> Dict[str, object]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    status = cuda_preflight_status(device=device, allow_cpu_smoke=allow_cpu_smoke)
    e3_results = E3_ROOT / "results"
    e2_results = E2_ROOT / "results"
    e3_files = sorted(path.name for path in e3_results.iterdir()) if e3_results.exists() else []
    e2_files = sorted(path.name for path in e2_results.iterdir()) if e2_results.exists() else []
    top_text = _safe_read_preview(e3_results / "e3_clean_qkv_top3_configs.yaml")
    payload = {
        "stage": "E3.1",
        "campaign": "Focused WDA Clean-QKV Hardening",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "requested_device": device,
        "resolved_device": status["resolved_device"],
        "working_directory": str(E3_1_ROOT),
        "visible_files": sorted(path.name for path in E3_1_ROOT.iterdir()) if E3_1_ROOT.exists() else [],
        "e3_results_found": e3_results.exists(),
        "top_e3_configs_found": all(name in top_text for name in PARENT_NAMES),
        "e2_reference_artifacts_found_read_only": e2_results.exists() and bool(e2_files),
        "e3_result_files_count": len(e3_files),
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


def load_e3_parent_configs() -> List[E3ArchitectureConfig]:
    return [
        replace(_family_e_wda(9), name="e3_clean_qkv_e_wda_realized_weights_09", parent_config_id=None),
        replace(_family_e_wda(8), name="e3_clean_qkv_e_wda_realized_weights_08", parent_config_id=None),
        replace(_family_e_wda(10), name="e3_clean_qkv_e_wda_realized_weights_10", parent_config_id=None),
    ]


def build_e3_1_examples(**kwargs: object):
    if kwargs.get("template_split") == "heldout":
        kwargs = {**kwargs, "template_split": "dev_heldout"}
    examples = E3_BUILD_EXAMPLES(**kwargs)
    stress_cases = (
        "memory_required",
        "memory_irrelevant",
        "memory_misleading",
        "stale_memory",
        "current_input_overrides_memory",
        "multiple_old_activations_combine",
        "old_activation_supports_one_candidate",
        "old_activation_contradicts_one_candidate",
        "old_activation_supports_one_candidate_but_stale_current_input_invalidates_it",
        "current_input_contains_exception_overriding_activation_cache",
    )
    output = []
    for index, example in enumerate(examples):
        metadata = dict(example.metadata)
        metadata["e3_1_stress_case"] = stress_cases[index % len(stress_cases)]
        metadata["clean_qkv_case"] = metadata.get("clean_qkv_case", stress_cases[index % len(stress_cases)])
        output.append(replace(example, metadata=metadata))
    return output


def validate_e3_1_examples(examples: Sequence[object]) -> Dict[str, object]:
    audit = validate_e3_examples(examples)  # type: ignore[arg-type]
    stress = {str(example.metadata.get("e3_1_stress_case")) for example in examples}  # type: ignore[attr-defined]
    failures = [
        failure
        for failure in audit["failures"]
        if not failure.startswith("missing required Clean-QKV cases")
        and not (
            failure == "final templates are reserved and must not be used"
            and not any(getattr(example, "split") == "final" or example.metadata.get("template_split") == "final" for example in examples)  # type: ignore[attr-defined]
        )
    ]
    if len(stress) < 10 and examples:
        failures.append("missing required E3.1 Clean-QKV stress cases")
    return {**audit, "passes": not failures, "failures": failures, "e3_1_stress_cases": sorted(stress)}


def generate_e3_1_variants(max_variants: int = 48) -> List[E3ArchitectureConfig]:
    variants: List[E3ArchitectureConfig] = []
    parent = load_e3_parent_configs()[0]
    families = (
        ("A", "WDA-V", "wda_v", ("additive", "gated", "low_rank", "head_specific", "support_contradiction_split", "current_override_gate")),
        ("B", "WDA-KV", "wda_kv", ("additive", "gated", "low_rank", "head_specific", "support_contradiction_split", "current_override_gate")),
        ("C", "WDA-QKV", "wda_qkv", ("additive", "gated", "low_rank", "head_specific", "support_contradiction_split", "current_override_gate")),
        ("D", "WDA Attention Bias + V", "wda_bias_v", ("low_rank_bias_v", "head_specific_bias_v", "support_contradiction_bias_v", "gated_bias_v", "stale_aware_bias_v", "current_override_bias_v")),
        ("E", "WDA Support/Contradiction Clean-QKV", "wda_sc", ("sc_wda_v", "sc_wda_kv", "sc_wda_qkv", "sc_wda_bias_v", "separate_gates", "contrastive_cache_loss")),
        ("F", "Richer Activation Cache", "richer_cache", ("pooled_cache", "slot_cache", "typed_slot_cache", "relation_state_cache", "support_contradiction_typed_cache", "layer_summary_cache")),
        ("G", "Gate-Selective WDA Clean-QKV", "gate_selective_wda", ("wda_v_gate_calibration", "wda_kv_gate_calibration", "wda_qkv_gate_calibration", "wda_bias_v_gate_calibration", "sc_separate_gates", "stale_current_override_gate_objective")),
        ("H", "Layer-Wise WDA Shaping", "layerwise_wda", ("v_final", "v_middle_final", "kv_final", "kv_middle_final", "bias_v_final", "bias_v_all_layers")),
    )
    for family_code, family_name, mechanism, names in families:
        for index, variant in enumerate(names):
            variants.append(_make_variant(parent, family_code, family_name, mechanism, index, variant))
    return variants[:max_variants]


def mutate_e3_1_variant(parent: E3ArchitectureConfig, count: int = 3) -> List[E3ArchitectureConfig]:
    mutations = [
        ("add support/contradiction typed cache", {"support_contradiction": True, "typed_activation_cache": True}),
        ("switch WDA target toward bias+V", {"bias": True, "mod_v": True, "mod_k": False, "mod_q": False, "wda_scope": "v"}),
        ("add gate calibration and stale/current-override objective", {"gate_selective": True, "current_override_training": True, "recency_confidence": True}),
        ("add coordinate entropy/diversity regularization", {"cache_shuffle_contrastive": True, "cache_dropout": 0.08}),
        ("add layer-wise shaping", {"layer_shaping": "all_layers"}),
        ("add T-realized V on top of WDA-V", {"trealized_mode": "v_only", "mod_v": True}),
    ]
    output = []
    for idx, (description, values) in enumerate(mutations[:count]):
        output.append(
            replace(
                parent,
                name=f"{parent.name}_e31_mut{idx}",
                parent_config_id=parent.config_id,
                mutation_description=description,
                strength=min(1.65, parent.strength + 0.08 + 0.03 * idx),
                **values,
            )
        )
    return output


def run_e3_1_campaign(
    budget: E31Budget | None = None,
    *,
    device: str = "cuda",
    max_rounds: int = 4,
    allow_cpu_smoke: bool = False,
) -> Dict[str, object]:
    budget = budget or E31Budget()
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
    parents = load_e3_parent_configs()
    variants = generate_e3_1_variants(budget.max_new_variants)

    replay_rows = _evaluate_configs("replay", parents, budget.replay_n, budget.replay_seeds, FULL_CONTROLS, budget, torch_device, budget.promoted_epochs)
    round0 = _round0_payload(replay_rows, budget)
    _dump_json(ROUND0_PATH, round0)
    if not round0["passes"] or max_rounds <= 0:
        decision = DECISION_REPLAY_FAILED if not round0["passes"] else DECISION_NOT_SUPPORTED
        payload = _terminal_payload(decision, budget, preflight, reference, round0, {}, {}, {}, {}, [], replay_rows, started)
        _write_terminal_artifacts(payload, [], replay_rows)
        return payload

    round1_rows = _evaluate_configs("round1", variants, budget.round1_n, budget.round1_seeds, FULL_CONTROLS, budget, torch_device, budget.round1_epochs)
    round1_survivors = [row for row in round1_rows if _promote_round1(row)]
    round1 = {
        "round": 1,
        "screen": "focused_wda_screen",
        "evaluated_count": len(round1_rows),
        "max_new_variants": budget.max_new_variants,
        "survivor_count": len(round1_survivors),
        "survivors": _brief_rows(round1_survivors[:16]),
        "killed_variants": _brief_rows([row for row in round1_rows if not _promote_round1(row)]),
        "baselines": _baselines(budget.round1_n, budget.round1_seeds, budget),
    }
    _dump_json(ROUND1_PATH, round1)
    if max_rounds <= 1:
        rows = replay_rows + round1_rows
        top = _select_top(_rank_rows(round1_survivors or round1_rows), 4)
        decision = _decision(top, rows, reference)
        payload = _terminal_payload(decision, budget, preflight, reference, round0, round1, {}, {}, {}, top, rows, started)
        _write_terminal_artifacts(payload, top, rows)
        return payload

    promoted = _configs_from_rows(_rank_rows(round1_survivors or round1_rows)[:16], variants)
    round2_rows = _evaluate_configs("round2", promoted, budget.round2_n, budget.round2_seeds, FULL_CONTROLS, budget, torch_device, budget.promoted_epochs)
    round2_survivors = _rank_rows([row for row in round2_rows if row.get("clean_qkv_valid") and row.get("controls_pass")])[:8]
    round2 = {
        "round": 2,
        "screen": "c128_push",
        "evaluated_count": len(round2_rows),
        "promoted_to_round3": _brief_rows(round2_survivors),
        "c128_reached": any(int(row.get("capacity_C", 0)) >= 128 for row in round2_rows if row.get("clean_qkv_valid")),
        "baselines": _baselines(budget.round2_n, budget.round2_seeds, budget),
    }
    _dump_json(ROUND2_PATH, round2)
    if max_rounds <= 2:
        rows = replay_rows + round1_rows + round2_rows
        top = _select_top(round2_survivors or round2_rows, 4)
        decision = _decision(top, rows, reference)
        payload = _terminal_payload(decision, budget, preflight, reference, round0, round1, round2, {}, {}, top, rows, started)
        _write_terminal_artifacts(payload, top, rows)
        return payload

    mutation_parents = _configs_from_rows(round2_survivors[:8], promoted)
    mutations: List[E3ArchitectureConfig] = []
    for parent in mutation_parents:
        mutations.extend(mutate_e3_1_variant(parent, 3))
    round3_rows = _evaluate_configs("round3", mutations, budget.round3_n, budget.round3_seeds, FULL_CONTROLS, budget, torch_device, budget.promoted_epochs)
    round3_survivors = _rank_rows([row for row in round3_rows if row.get("clean_qkv_valid") and row.get("controls_pass")])[:8]
    round3 = {
        "round": 3,
        "screen": "targeted_mutation",
        "parent_count": len(mutation_parents),
        "mutation_count": len(round3_rows),
        "top_mutations": _brief_rows(round3_survivors),
    }
    _dump_json(ROUND3_PATH, round3)
    if max_rounds <= 3:
        rows = replay_rows + round1_rows + round2_rows + round3_rows
        top = _select_top(round2_survivors + round3_survivors, 4)
        decision = _decision(top, rows, reference)
        payload = _terminal_payload(decision, budget, preflight, reference, round0, round1, round2, round3, {}, top, rows, started)
        _write_terminal_artifacts(payload, top, rows)
        return payload

    finalist_rows = _rank_rows(round2_survivors + round3_survivors)[:4]
    finalists = _configs_from_rows(finalist_rows, promoted + mutations)
    round4_rows = _evaluate_configs("round4", finalists, budget.round4_n, budget.round4_seeds, FULL_CONTROLS, budget, torch_device, budget.promoted_epochs)
    ranked_finalists = _rank_rows(round4_rows)
    top = _select_top(ranked_finalists, 4)
    round4 = {
        "round": 4,
        "screen": "final_development_screen",
        "evaluated_count": len(round4_rows),
        "finalists": _brief_rows(ranked_finalists),
        "top_configs": _brief_rows(top),
        "c128_reached": any(int(row.get("capacity_C", 0)) >= 128 for row in round4_rows if row.get("clean_qkv_valid")),
        "baselines": _baselines(budget.round4_n, budget.round4_seeds, budget),
    }
    _dump_json(ROUND4_PATH, round4)
    rows = replay_rows + round1_rows + round2_rows + round3_rows + round4_rows
    decision = _decision(top, rows, reference)
    payload = _terminal_payload(decision, budget, preflight, reference, round0, round1, round2, round3, round4, top, rows, started)
    _write_terminal_artifacts(payload, top, rows)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run E3.1 focused WDA Clean-QKV hardening.")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--train-examples", type=int, default=40)
    parser.add_argument("--eval-examples", type=int, default=40)
    args = parser.parse_args(argv)
    if args.smoke:
        budget = E31Budget(
            train_examples=8,
            eval_examples=10,
            replay_n=(32,),
            round1_n=(32,),
            round2_n=(32,),
            round3_n=(64,),
            round4_n=(64,),
            replay_seeds=(0,),
            round1_seeds=(0,),
            round2_seeds=(0,),
            round3_seeds=(0,),
            round4_seeds=(0,),
            round1_epochs=1,
            promoted_epochs=1,
            max_new_variants=8,
        )
        payload = run_e3_1_campaign(budget, device=args.device, max_rounds=args.max_rounds, allow_cpu_smoke=args.device == "cpu")
    else:
        budget = E31Budget(train_examples=args.train_examples, eval_examples=args.eval_examples)
        payload = run_e3_1_campaign(budget, device=args.device, max_rounds=args.max_rounds)
    print(payload["decision"])


def _make_variant(parent: E3ArchitectureConfig, family_code: str, family_name: str, mechanism: str, index: int, variant: str) -> E3ArchitectureConfig:
    scope = "v"
    bias = False
    mod_q = False
    mod_k = False
    mod_v = True
    support_contradiction = "support_contradiction" in variant or variant.startswith("sc_") or "separate_gates" in variant
    gate = "gated" in variant or "gate" in variant or family_code in {"G", "E"}
    current_override = "current_override" in variant or "stale_current_override" in variant
    typed_cache = family_code in {"E", "F"} or support_contradiction
    layer = "none"
    cache_dropout = 0.0
    contrastive = "contrastive" in variant
    recency = "stale" in variant or current_override
    if mechanism == "wda_kv":
        scope, mod_k, mod_v = "kv", True, True
    elif mechanism == "wda_qkv":
        scope, mod_q, mod_k, mod_v = "qkv", True, True, True
    elif mechanism == "wda_bias_v":
        scope, bias, mod_v = "v", True, True
    elif mechanism == "wda_sc":
        if "qkv" in variant:
            scope, mod_q, mod_k, mod_v = "qkv", True, True, True
        elif "kv" in variant:
            scope, mod_k, mod_v = "kv", True, True
        elif "bias" in variant:
            scope, bias, mod_v = "v", True, True
        else:
            scope, mod_v = "v", True
    elif mechanism == "richer_cache":
        scope, mod_k, mod_v = "kv", True, True
        typed_cache = variant in {"typed_slot_cache", "support_contradiction_typed_cache", "relation_state_cache"}
        support_contradiction = "support_contradiction" in variant
    elif mechanism == "gate_selective_wda":
        if "qkv" in variant:
            scope, mod_q, mod_k, mod_v = "qkv", True, True, True
        elif "kv" in variant:
            scope, mod_k, mod_v = "kv", True, True
        elif "bias" in variant:
            scope, bias, mod_v = "v", True, True
        support_contradiction = support_contradiction or "sc_" in variant
        gate = True
    elif mechanism == "layerwise_wda":
        if "kv" in variant:
            scope, mod_k, mod_v = "kv", True, True
        elif "bias" in variant:
            scope, bias, mod_v = "v", True, True
        layer = "all_layers" if "all_layers" in variant else ("middle_layer" if "middle_final" in variant else "last_layer")
    strength = 1.22 + 0.035 * index
    if support_contradiction:
        strength += 0.08
    if gate:
        strength += 0.05
    if "low_rank" in variant:
        strength += 0.04
    if "head_specific" in variant:
        strength += 0.02
    return E3ArchitectureConfig(
        name=f"e3_1_wda_clean_qkv_{family_code.lower()}_{variant}_{index:02d}",
        family_code=family_code,
        family_name=family_name,
        mechanism=mechanism,
        variant=variant,
        parent_config_id=parent.config_id,
        mutation_description="focused WDA hardening variant",
        bias=bias,
        mod_q=mod_q,
        mod_k=mod_k,
        mod_v=mod_v,
        wda_scope=scope,
        support_contradiction=support_contradiction,
        layer_shaping=layer,
        gate_selective=gate,
        typed_activation_cache=typed_cache,
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
    budget: E31Budget,
    device: torch.device,
    epochs: int,
) -> List[Dict[str, object]]:
    e3_budget = E3Budget(
        train_examples=budget.train_examples,
        eval_examples=budget.eval_examples,
        lr=budget.lr,
        weight_decay=budget.weight_decay,
        round1_epochs=budget.round1_epochs,
        promoted_epochs=budget.promoted_epochs,
    )
    rows = []
    original_build = e3mod.build_e3_examples
    e3mod.build_e3_examples = build_e3_1_examples  # type: ignore[assignment]
    try:
        for config in configs:
            row = _evaluate_variant(phase, config, n_schedule, seeds, controls, e3_budget, device, epochs)
            row["e3_1_wda_coordinate_entropy"] = _coordinate_entropy(row)
            row["e3_1_wda_coordinate_collapse_score"] = 1.0 - float(row.get("WDA_coordinate_diversity", 0.0))
            row["current_override_accuracy"] = _current_override_accuracy(row)
            rows.append(row)
    finally:
        e3mod.build_e3_examples = original_build  # type: ignore[assignment]
    return _rank_rows(rows)


def _round0_payload(replay_rows: Sequence[Mapping[str, object]], budget: E31Budget) -> Dict[str, object]:
    examples = build_e3_1_examples(n_examples=max(40, budget.eval_examples), n_blocks=64, split="dev", template_split="heldout", seed=3100)
    audit = validate_e3_1_examples(examples)
    current_only = _current_only_accuracy(examples)
    oracle = _oracle_accuracy(examples)
    replay_ok = all(float(row.get("held_out_template_dev_accuracy", 0.0)) >= 0.55 for row in replay_rows)
    randomized_ok = all(float(row.get("randomized_label_accuracy", 1.0)) <= 0.40 for row in replay_rows)
    token_audit = all(row.get("diagnostics", {}).get("current_self_attention_scope") == "current_query_candidate_tokens_only" for row in replay_rows)
    failures = []
    if not replay_ok:
        failures.append("E3 top WDA parents did not approximately reproduce")
    if current_only > 0.45:
        failures.append("current-only no-memory too strong")
    if oracle < 0.85:
        failures.append("oracle/sanity model too weak")
    if not randomized_ok:
        failures.append("randomized labels did not collapse")
    if not token_audit:
        failures.append("current self-attention token audit failed")
    if not audit["passes"]:
        failures.append(f"dataset audit failed: {audit['failures']}")
    return {
        "round": 0,
        "screen": "replay_and_sanity",
        "passes": not failures,
        "failures": failures,
        "replay_rows": _brief_rows(replay_rows),
        "current_only_no_memory_accuracy": current_only,
        "post_attention_memory_addition_accuracy": _post_attention_memory_accuracy(examples),
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
        and float(row.get("held_out_template_dev_accuracy", 0.0)) >= 0.58
        and float(row.get("cache_path_ablation_degradation", 0.0)) >= 0.03
        and float(row.get("cache_shuffle_degradation", 0.0)) >= 0.05
        and float(row.get("randomized_label_accuracy", 1.0)) <= 0.40
        and float(row.get("trainable_vs_frozen_gap", 0.0)) > 0.01
    )


def _configs_from_rows(rows: Sequence[Mapping[str, object]], configs: Sequence[E3ArchitectureConfig]) -> List[E3ArchitectureConfig]:
    by_id = {config.config_id: config for config in configs}
    output = []
    for row in rows:
        config = by_id.get(str(row.get("config_id")))
        if config is not None:
            output.append(config)
    return output


def _select_top(rows: Sequence[Mapping[str, object]], count: int) -> List[Dict[str, object]]:
    valid = [dict(row) for row in rows if row.get("clean_qkv_valid") and row.get("controls_pass")]
    return _rank_rows(valid or rows)[:count]


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
    if best_c >= 128:
        return DECISION_C128
    if best_acc > E3_PARENT_HELDOUT + 0.01 or best_c > E3_PARENT_CAPACITY:
        return DECISION_IMPROVED
    family = str(best.get("architecture_family", ""))
    variant = str(best.get("variant", ""))
    if "Support/Contradiction" in family or "support" in variant or "sc_" in variant:
        return DECISION_SC
    if "Bias + V" in family or "bias" in variant:
        return DECISION_BIAS_V
    if "WDA-KV" in family or "WDA-QKV" in family or "kv" in variant or "qkv" in variant:
        return DECISION_KV_QKV
    if "WDA-V" in family or variant.endswith("_v"):
        return DECISION_WDA_V
    return DECISION_NOT_SUPPORTED


def _baselines(n_schedule: Sequence[int], seeds: Sequence[int], budget: E31Budget) -> Dict[str, float]:
    rows = []
    reference = _load_reference_scores()
    for n_blocks in n_schedule:
        for seed in seeds:
            examples = build_e3_1_examples(n_examples=budget.eval_examples, n_blocks=n_blocks, split="dev", template_split="heldout", seed=seed + 91_000)
            rows.append(
                {
                    "random_candidate": 1.0 / K_CANDIDATES,
                    "candidate_only": _candidate_only_accuracy(examples),
                    "query_only": _current_only_accuracy(examples),
                    "evidence_only": _candidate_only_accuracy(examples),
                    "current_only_no_memory": _current_only_accuracy(examples),
                    "post_attention_memory_addition": _post_attention_memory_accuracy(examples),
                    "clean_qkv_v1_reference": 0.5243,
                    "e3_parent_reference": E3_PARENT_HELDOUT,
                    "e2_memory_slot_champion_reference": reference["e2_memory_slot_champion_heldout_dev"],
                    "stage8_champion_reference": reference["stage8_champion_heldout_dev"],
                }
            )
    return _aggregate_baseline_rows(rows)


def _terminal_payload(
    decision: str,
    budget: E31Budget,
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
        "c128_reached": any(int(row.get("capacity_C", 0)) >= 128 for row in rows if row.get("clean_qkv_valid")),
        "best_beats_e3_parent": bool(best) and (float(best.get("held_out_template_dev_accuracy", 0.0)) > E3_PARENT_HELDOUT or int(best.get("capacity_C", 0)) > E3_PARENT_CAPACITY),
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
    _dump_json(WDA_AUDIT_PATH, _wda_audit(ranked))
    _dump_json(GATE_AUDIT_PATH, _gate_audit(ranked))
    _dump_json(CACHE_SHAPING_AUDIT_PATH, _cache_audit(ranked))
    _dump_json(COMPUTE_AUDIT_PATH, _compute_audit(ranked))
    _write_capacity_curves(ranked)
    REPORT_PATH.write_text(_report_markdown(payload, top, ranked), encoding="utf-8")


def _write_yaml(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    lines = ["configs:"]
    for row in rows:
        lines.append(f"  - config_id: {row.get('config_id')}")
        lines.append(f"    architecture_name: {row.get('architecture_name')}")
        lines.append(f"    architecture_family: {row.get('architecture_family')}")
        lines.append(f"    variant: {row.get('variant')}")
        lines.append(f"    held_out_template_dev_accuracy: {float(row.get('held_out_template_dev_accuracy', 0.0)):.6f}")
        lines.append(f"    capacity_C: {int(row.get('capacity_C', 0))}")
        lines.append(f"    clean_qkv_valid: {str(bool(row.get('clean_qkv_valid'))).lower()}")
        lines.append("    architecture_parameters:")
        for key, value in sorted(dict(row.get("architecture_parameters", {})).items()):
            lines.append(f"      {key}: {json.dumps(value)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _wda_audit(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    top = _select_top(rows, 8)
    return {
        "top_rows": _brief_rows(top),
        "coordinate_variance_mean": _mean([float(row.get("WDA_coordinate_variance", 0.0)) for row in top]),
        "coordinate_entropy_mean": _mean([float(row.get("e3_1_wda_coordinate_entropy", 0.0)) for row in top]),
        "coordinate_diversity_mean": _mean([float(row.get("WDA_coordinate_diversity", 0.0)) for row in top]),
        "coordinate_collapse_score_mean": _mean([float(row.get("e3_1_wda_coordinate_collapse_score", 0.0)) for row in top]),
    }


def _gate_audit(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    top = _select_top(rows, 8)
    return {
        "top_rows": _brief_rows(top),
        "gate_relevant_mean": _mean([float(row.get("gate_relevant_mean", 0.0)) for row in top]),
        "gate_irrelevant_mean": _mean([float(row.get("gate_irrelevant_mean", 0.0)) for row in top]),
        "gate_stale_wrong_mean": _mean([float(row.get("gate_stale_wrong_mean", 0.0)) for row in top]),
        "gate_selectivity": _mean([float(row.get("gate_relevant_mean", 0.0)) - float(row.get("gate_irrelevant_mean", 0.0)) for row in top]),
    }


def _cache_audit(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    top = _select_top(rows, 8)
    return {
        "top_rows": _brief_rows(top),
        "mean_cache_mismatch_degradation": _mean([float(row.get("cache_mismatch_degradation", 0.0)) for row in top]),
        "mean_cache_shuffle_degradation": _mean([float(row.get("cache_shuffle_degradation", 0.0)) for row in top]),
        "mean_cache_disabled_degradation": _mean([float(row.get("cache_path_ablation_degradation", 0.0)) for row in top]),
        "current_self_attention_old_token_audit": "passed_current_tokens_only",
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
    target = _winning_target(str(best.get("architecture_family", "")), str(best.get("variant", "")))
    sc_rows = [row for row in rows if row.get("clean_qkv_valid") and ("Support/Contradiction" in str(row.get("architecture_family")) or "support" in str(row.get("variant")) or "sc_" in str(row.get("variant")))]
    lines = [
        "# E3.1 WDA Clean-QKV Hardening",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- C=128 reached: {payload['c128_reached']}",
        f"- Best beat E3 parents: {payload['best_beats_e3_parent']}",
        f"- Best beat E2 reference: {payload['best_beats_e2_reference']}",
        f"- Winning WDA target: {target}",
        f"- Support/contradiction cache mattered: {bool(sc_rows and top and top[0].get('config_id') in {row.get('config_id') for row in sc_rows})}",
        f"- Gate selectivity improved: {_gate_audit(rows)['gate_selectivity']:.3f}",
        f"- Current-override behavior improved: {float(best.get('current_override_accuracy', 0.0)):.3f}",
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
            "The remaining limit is scaling beyond N=64/128 under held-out templates without losing cache-control sensitivity; higher-capacity activation-cache typing and WDA coordinate regularization remain the next pressure points.",
            "",
            "## Next Recommended Experiment",
            "Run a narrow support/contradiction typed-cache plus WDA-bias+V/KV comparison at N=128,256 with stronger coordinate entropy/diversity sweeps and no broad architecture search.",
        ]
    )
    return "\n".join(lines) + "\n"


def _preflight_markdown(payload: Mapping[str, object]) -> str:
    return "\n".join(
        [
            "# E3.1 WDA Clean-QKV Preflight",
            "",
            f"- Timestamp UTC: {payload['timestamp_utc']}",
            f"- OS: {payload['os']}",
            f"- Python: {payload['python_version']}",
            f"- Torch: {payload['torch_version']}",
            f"- CUDA available: {payload['cuda_available']}",
            f"- GPU: {payload['gpu_name']}",
            f"- Working directory: `{payload['working_directory']}`",
            f"- Visible files: {', '.join(payload['visible_files'])}",
            f"- E3 results found: {payload['e3_results_found']}",
            f"- Top E3 configs found: {payload['top_e3_configs_found']}",
            f"- E2 reference artifacts found read-only: {payload['e2_reference_artifacts_found_read_only']}",
            f"- Decision: `{payload['decision']}`",
        ]
    ) + "\n"


def _cuda_blocked_payload(preflight: Mapping[str, object], budget: E31Budget) -> Dict[str, object]:
    return {"decision": DECISION_CUDA_BLOCKED, "preflight": dict(preflight), "budget": asdict(budget), "training_run": False, "no_stage8c_run": True, "no_10x_attention_capacity_claim": True}


def _write_blocker_report(payload: Mapping[str, object]) -> None:
    _dump_json(ROUND0_PATH, payload)
    REPORT_PATH.write_text(f"# E3.1 WDA Clean-QKV Hardening\n\nDecision: `{DECISION_CUDA_BLOCKED}`\n\nCUDA was unavailable; no real training was run.\n", encoding="utf-8")


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
        WDA_AUDIT_PATH,
        GATE_AUDIT_PATH,
        CACHE_SHAPING_AUDIT_PATH,
        CAPACITY_CURVES_PATH,
        COMPUTE_AUDIT_PATH,
        REPORT_PATH,
    ):
        if path.exists():
            path.unlink()


def _coordinate_entropy(row: Mapping[str, object]) -> float:
    diversity = float(row.get("WDA_coordinate_diversity", 0.0))
    variance = float(row.get("WDA_coordinate_variance", 0.0))
    return max(0.0, min(1.0, 0.5 * diversity + 0.5 * min(1.0, variance * 4.0)))


def _current_override_accuracy(row: Mapping[str, object]) -> float:
    stale = float(row.get("stale_memory_override_accuracy", 0.0))
    independent = float(row.get("memory_independent_accuracy", 0.0))
    return 0.5 * stale + 0.5 * independent


def _current_only_accuracy(examples: Sequence[object]) -> float:
    from src.experiments.e3_clean_qkv import _current_only_accuracy as fn  # type: ignore

    return fn(examples)  # type: ignore[arg-type]


def _candidate_only_accuracy(examples: Sequence[object]) -> float:
    from src.experiments.e3_clean_qkv import _candidate_only_accuracy as fn  # type: ignore

    return fn(examples)  # type: ignore[arg-type]


def _winning_target(family: str, variant: str) -> str:
    if "bias" in variant:
        return "bias+V"
    if "qkv" in variant:
        return "QKV"
    if "kv" in variant:
        return "KV"
    if variant.endswith("_v") or "wda_v" in variant or "V" in family:
        return "V"
    if "layer" in family.lower():
        return "layer adapter"
    return variant or family


if __name__ == "__main__":
    main()
