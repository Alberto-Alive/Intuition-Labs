from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from src.coordinators.mlp import MLPTrainingConfig
from src.datasets.multiview_code_patch_selection import (
    BALANCED_34B_CANDIDATE_REPRESENTATION,
    BALANCED_34B_DATASET_SOURCE,
    MultiViewTaskExample,
    apply_example_control,
    build_multiview_code_patch_splits,
    dataset_summary,
)
from src.experiments.architecture_search import (
    _agent_config,
    _candidate_config,
    _candidate_specs,
    _coordinator_config,
    _training_config,
)
from src.experiments.real_shared_weight_latent_coordination import (
    BENCHMARK,
    TextOutputOnlyCoordinator,
    fit_latent_system,
    predict_latent_system,
    _candidate_feature_tensor,
    _gather_role_axis,
)
from src.experiments.run_latent_vs_text_efficiency import _load_latent_checkpoint
from src.experiments.run_stage34_candidate_sanitization_diagnostics import _accuracy, _structured_oracle_predictions
from src.experiments.run_stage38_full_hard_validation import (
    _dataset_config_for_stage,
    _labels,
    _stage38_full_config,
    _stage_from_config,
)
from src.experiments.run_stage38_schema_aware_controls import (
    _candidate_evidence_mismatch,
    _candidate_only,
    _schema_only,
    _value_shuffle_within_schema,
    _zero_role_embeddings,
)
from src.experiments.run_stage3_gpu_hard_validation import _clear_cuda, _configure_cuda


DEFAULT_CONFIG = "configs/stage4_final_candidate_token_direct_validation.json"
DEFAULT_FINAL_RESULTS = "results/stage4_final_candidate_token_direct_validation_results.json"
DEFAULT_OUTPUT = "results/stage41_physical_order_invariance_audit.json"
DEFAULT_SMOKE_OUTPUT = "results/stage41_physical_order_invariance_smoke.json"
DEFAULT_REPORT = "reports/STAGE41_PHYSICAL_ORDER_INVARIANCE_AUDIT.md"
SELECTED_ARCHITECTURE = "candidate_token_direct_lr3e4_clip1"
REPAIRED_VARIANT = "candidate_token_direct_role_canonical"
PERMUTATIONS = tuple(itertools.permutations(range(4)))
ROLE_TENSOR_KEYS = ("raw_pooled", "msg", "evidence", "active_message", "message", "token_states", "token_mask")
NEAR_CHANCE = 0.18


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 4.1 physical-order invariance audit and cheap repair smoke.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--final-results", default=DEFAULT_FINAL_RESULTS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke-output", default=DEFAULT_SMOKE_OUTPUT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--reuse-checkpoint-audit", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    final_results = json.loads(Path(args.final_results).read_text(encoding="utf-8"))
    if bool(args.reuse_checkpoint_audit) and Path(args.output).exists():
        result = json.loads(Path(args.output).read_text(encoding="utf-8"))
        stage_config = _stage41_base_config(config)
        device, hardware = _configure_cuda(stage_config)
        result.setdefault("metadata", {})
        result["metadata"].update(
            {
                "created_at_utc": _now(),
                "device": device,
                "hardware": hardware,
                "checkpoint_audit_reused": True,
                "full_validation_run": False,
                "success_claim": "not claimed",
            }
        )
        if not bool(args.skip_smoke):
            print("stage4.1: reusing checkpoint audit and running cheap validation")
            result["cheap_validation"] = _cheap_validation(stage_config, device)
        result["shortcut_diagnostics"] = _shortcut_diagnostics(final_results)
        result["conclusion"] = _conclusion(result["checkpoint_audit"], result.get("cheap_validation"))
    else:
        result = run_stage41(
            config=config,
            config_path=config_path,
            final_results=final_results,
            run_smoke=not bool(args.skip_smoke),
        )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if result.get("cheap_validation") is not None:
        smoke_path = Path(args.smoke_output)
        smoke_path.parent.mkdir(parents=True, exist_ok=True)
        smoke_path.write_text(json.dumps(result["cheap_validation"], indent=2, sort_keys=True), encoding="utf-8")
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(result), encoding="utf-8")


def run_stage41(
    config: Dict[str, object],
    config_path: Path,
    final_results: Dict[str, object],
    run_smoke: bool,
) -> Dict[str, object]:
    config = _stage41_base_config(config)
    device, hardware = _configure_cuda(config)
    rows = [row for row in final_results.get("stage4_final_rows", []) if row.get("status") == "completed"]
    rows = [row for row in rows if row.get("candidate") == SELECTED_ARCHITECTURE]
    rows.sort(key=lambda row: int(row["seed"]))
    if not rows:
        raise RuntimeError("No completed Stage 4 final candidate-token-direct rows were found.")

    print("stage4.1: checkpoint-only all-permutation audit")
    checkpoint_audit = _checkpoint_audit(config, rows, device)
    smoke = None
    if run_smoke:
        print("stage4.1: cheap validation for canonical repaired variant")
        smoke = _cheap_validation(config, device)
    return {
        "metadata": {
            "stage": "stage41_physical_order_invariance_audit",
            "created_at_utc": _now(),
            "config_path": str(config_path),
            "fixed_benchmark": "Stage 3.8 schema-aware balanced_categories_v3",
            "dataset_or_control_changes": "none",
            "selected_architecture": SELECTED_ARCHITECTURE,
            "repair_variant": REPAIRED_VARIANT,
            "full_validation_run": False,
            "success_claim": "not claimed",
            "device": device,
            "hardware": hardware,
        },
        "source_trace": _implementation_trace(),
        "shortcut_diagnostics": _shortcut_diagnostics(final_results),
        "checkpoint_audit": checkpoint_audit,
        "cheap_validation": smoke,
        "conclusion": _conclusion(checkpoint_audit, smoke),
    }


def _checkpoint_audit(config: Dict[str, object], final_rows: Sequence[Dict[str, object]], device: str) -> Dict[str, object]:
    stage = _stage_from_config(config)
    dataset_config = _dataset_config_for_stage(config, stage)
    seed_rows = []
    for row in final_rows:
        seed = int(row["seed"])
        print(f"stage4.1 seed={seed}: loading final checkpoint")
        checkpoints = dict(row.get("checkpoint_paths", {}))
        trainable = _load_latent_checkpoint(Path(str(checkpoints["trainable"])), device=device)
        splits = build_multiview_code_patch_splits(dataset_config, seed=seed, repo_root=Path("."))
        examples = list(splits["test"])
        labels = _labels(examples)
        normal_predictions = _predict_with_role_order(trainable, examples, perm=(0, 1, 2, 3), seed=seed + 90_000)
        normal_accuracy = _accuracy(normal_predictions, labels)
        builtin_fixed = _accuracy(
            predict_latent_system(trainable, examples, "physical_order_shuffled_roles_preserved", seed + 91_000),
            labels,
        )
        permutation_rows = []
        for perm in PERMUTATIONS:
            perm_accuracy = _accuracy(_predict_with_role_order(trainable, examples, perm=perm, seed=seed + 92_000), labels)
            canonical_accuracy = _accuracy(
                _predict_with_role_order(trainable, examples, perm=perm, seed=seed + 93_000, canonicalize=True),
                labels,
            )
            legacy_accuracy = _accuracy(
                _predict_with_role_order(trainable, examples, perm=perm, seed=seed + 94_000, legacy_bug=True),
                labels,
            )
            integrity = _verify_permutation_integrity(trainable, examples[:4], perm)
            permutation_rows.append(
                {
                    "permutation": list(perm),
                    "accuracy": perm_accuracy,
                    "canonicalized_accuracy": canonical_accuracy,
                    "legacy_bug_accuracy": legacy_accuracy,
                    "integrity": integrity,
                }
            )
        accuracies = [float(item["accuracy"]) for item in permutation_rows]
        canonical = [float(item["canonicalized_accuracy"]) for item in permutation_rows]
        legacy = [float(item["legacy_bug_accuracy"]) for item in permutation_rows]
        seed_rows.append(
            {
                "seed": seed,
                "checkpoint_path": str(checkpoints["trainable"]),
                "final_recorded_normal_accuracy": float(row["test_accuracy"]["trainable"]),
                "final_recorded_failed_physical_order_accuracy": float(row["test_accuracy"]["physical_order_shuffled_roles_preserved"]),
                "normal_accuracy_recomputed": normal_accuracy,
                "builtin_physical_order_after_tensor_repair": builtin_fixed,
                "permutation_accuracy_summary": _summary_values(accuracies),
                "canonicalized_permutation_accuracy_summary": _summary_values(canonical),
                "legacy_bug_permutation_accuracy_summary": _summary_values(legacy),
                "worst_permutation": min(permutation_rows, key=lambda item: float(item["accuracy"]))["permutation"],
                "worst_accuracy": min(accuracies),
                "permutation_specific_failure": bool(max(accuracies) - min(accuracies) > 0.02),
                "universal_failure": bool(max(accuracies) < normal_accuracy - 0.08),
                "all_integrity_checks_pass": all(_integrity_passes(item["integrity"]) for item in permutation_rows),
                "permutations": permutation_rows,
            }
        )
        del trainable
        _clear_cuda()
    normal_values = [float(row["normal_accuracy_recomputed"]) for row in seed_rows]
    fixed_values = [float(row["permutation_accuracy_summary"]["mean"]) for row in seed_rows]
    canonical_values = [float(row["canonicalized_permutation_accuracy_summary"]["mean"]) for row in seed_rows]
    legacy_values = [float(row["legacy_bug_permutation_accuracy_summary"]["mean"]) for row in seed_rows]
    return {
        "checkpoint_only": True,
        "n_seeds": len(seed_rows),
        "n_permutations": len(PERMUTATIONS),
        "dataset_summary": dataset_summary(
            {"train": [], "dev": [], "test": build_multiview_code_patch_splits(dataset_config, seed=int(seed_rows[0]["seed"]), repo_root=Path("."))["test"][:1]}
        )
        if seed_rows
        else {},
        "seed_rows": seed_rows,
        "overall": {
            "normal_mean": _mean(normal_values),
            "correct_permutation_mean": _mean(fixed_values),
            "canonicalized_permutation_mean": _mean(canonical_values),
            "legacy_bug_permutation_mean": _mean(legacy_values),
            "mean_abs_delta_correct_permutation_from_normal": _mean([abs(a - b) for a, b in zip(fixed_values, normal_values)]),
            "mean_abs_delta_canonicalized_from_normal": _mean([abs(a - b) for a, b in zip(canonical_values, normal_values)]),
            "legacy_bug_reproduces_failure": _mean(legacy_values) < _mean(normal_values) - 0.30,
            "all_integrity_checks_pass": all(bool(row["all_integrity_checks_pass"]) for row in seed_rows),
            "canonical_sorting_fixes_checkpoint_behavior": _mean([abs(a - b) for a, b in zip(canonical_values, normal_values)]) <= 0.02,
            "fixed_tensor_permutation_fixes_checkpoint_behavior": _mean([abs(a - b) for a, b in zip(fixed_values, normal_values)]) <= 0.02,
        },
    }


def _predict_with_role_order(
    result,
    examples: Sequence[MultiViewTaskExample],
    perm: Sequence[int],
    seed: int,
    canonicalize: bool = False,
    legacy_bug: bool = False,
) -> np.ndarray:
    del seed
    result.system.eval()
    predictions = []
    with torch.no_grad():
        for batch in _example_batches(examples, 128):
            logits = _logits_with_role_order(result.system, batch, tuple(int(value) for value in perm), canonicalize, legacy_bug)
            predictions.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
    return np.concatenate(predictions).astype(np.int64)


def _logits_with_role_order(system, examples: Sequence[MultiViewTaskExample], perm: Tuple[int, ...], canonicalize: bool, legacy_bug: bool) -> torch.Tensor:
    device = next(system.parameters()).device
    batch_size = len(examples)
    n_roles = int(system.n_roles)
    base_role_ids = torch.arange(n_roles, dtype=torch.long, device=device).view(1, n_roles).expand(batch_size, n_roles)
    order = torch.as_tensor(perm, dtype=torch.long, device=device).view(1, n_roles).expand(batch_size, n_roles)
    readouts = system.collect_clone_representations(examples, condition="none", seed=0, role_ids=base_role_ids)
    role_ids = base_role_ids.gather(1, order)
    if legacy_bug:
        readouts = dict(readouts)
        readouts["message"] = _gather_role_axis(readouts["message"], order)
    else:
        readouts = _gather_readouts(readouts, order)
    if canonicalize:
        sort_order = torch.argsort(role_ids, dim=1)
        role_ids = role_ids.gather(1, sort_order)
        readouts = _gather_readouts(readouts, sort_order)
    return _coordinator_logits(system, examples, readouts, role_ids)


def _coordinator_logits(system, examples: Sequence[MultiViewTaskExample], readouts: Dict[str, torch.Tensor], role_ids: torch.Tensor) -> torch.Tensor:
    clone_activations = readouts["message"]
    if bool(getattr(system.coordinator, "requires_candidate_features", False)):
        candidate_features = _candidate_feature_tensor(examples, clone_activations.device)
        if bool(getattr(system.coordinator, "requires_token_states", False)):
            return system.coordinator(
                clone_activations,
                role_ids,
                candidate_features,
                readouts["token_states"],
                readouts["token_mask"],
            )
        return system.coordinator(clone_activations, role_ids, candidate_features)
    return system.coordinator(clone_activations, role_ids)


def _gather_readouts(readouts: Dict[str, torch.Tensor], order: torch.Tensor) -> Dict[str, torch.Tensor]:
    out = dict(readouts)
    for key in ROLE_TENSOR_KEYS:
        value = readouts.get(key)
        if isinstance(value, torch.Tensor):
            out[key] = _gather_role_axis(value, order)
    return out


def _verify_permutation_integrity(result, examples: Sequence[MultiViewTaskExample], perm: Sequence[int]) -> Dict[str, object]:
    if not examples:
        return {
            "view_text_moved_correctly": True,
            "role_id_moved_with_view": True,
            "role_embedding_moved_with_view": True,
            "attention_masks_moved_with_view": True,
            "token_states_moved_with_view": True,
            "candidate_token_direct_receives_same_role_identity": True,
            "labels_unchanged": True,
            "candidate_order_unchanged": True,
        }
    system = result.system
    device = next(system.parameters()).device
    n_roles = int(system.n_roles)
    order = torch.as_tensor(tuple(int(value) for value in perm), dtype=torch.long, device=device).view(1, n_roles).expand(len(examples), n_roles)
    base_role_ids = torch.arange(n_roles, dtype=torch.long, device=device).view(1, n_roles).expand(len(examples), n_roles)
    moved = _permute_example_views(examples, perm)
    moved_role_ids = base_role_ids.gather(1, order)
    with torch.no_grad():
        base = system.collect_clone_representations(examples, condition="none", seed=0, role_ids=base_role_ids)
        moved_reencoded = system.collect_clone_representations(moved, condition="none", seed=0, role_ids=moved_role_ids)
    gathered = _gather_readouts(base, order)
    token_delta = float((gathered["token_states"].detach().float() - moved_reencoded["token_states"].detach().float()).abs().max().cpu().item())
    message_delta = float((gathered["message"].detach().float() - moved_reencoded["message"].detach().float()).abs().max().cpu().item())
    mask_equal = bool(torch.equal(gathered["token_mask"].detach().cpu(), moved_reencoded["token_mask"].detach().cpu()))
    role_embedding_pass = True
    if hasattr(system.coordinator, "role_embedding"):
        base_roles = system.coordinator.role_embedding(base_role_ids.clamp(min=0, max=system.coordinator.role_embedding.num_embeddings - 1))
        moved_roles = system.coordinator.role_embedding(moved_role_ids.clamp(min=0, max=system.coordinator.role_embedding.num_embeddings - 1))
        role_embedding_pass = bool(torch.allclose(_gather_role_axis(base_roles, order), moved_roles, atol=0.0, rtol=0.0))
    return {
        "view_text_moved_correctly": all(
            moved[row_id].views[slot].text == examples[row_id].views[int(perm[slot])].text
            and moved[row_id].views[slot].role == examples[row_id].views[int(perm[slot])].role
            for row_id in range(len(examples))
            for slot in range(n_roles)
        ),
        "role_id_moved_with_view": bool(torch.equal(moved_role_ids.detach().cpu(), order.detach().cpu())),
        "role_embedding_moved_with_view": role_embedding_pass,
        "attention_masks_moved_with_view": mask_equal,
        "token_states_moved_with_view": token_delta <= 1e-6,
        "candidate_token_direct_receives_same_role_identity": bool(torch.equal(moved_role_ids.sort(dim=1).values.cpu(), base_role_ids.sort(dim=1).values.cpu())),
        "labels_unchanged": all(int(moved[row_id].label) == int(examples[row_id].label) for row_id in range(len(examples))),
        "candidate_order_unchanged": all(
            [candidate.candidate_id for candidate in moved[row_id].candidates]
            == [candidate.candidate_id for candidate in examples[row_id].candidates]
            for row_id in range(len(examples))
        ),
        "max_token_state_delta": token_delta,
        "max_message_delta": message_delta,
    }


def _cheap_validation(config: Dict[str, object], device: str) -> Dict[str, object]:
    smoke_config = _stage41_smoke_config(config)
    stage = _stage_from_config(smoke_config)
    dataset_config = _dataset_config_for_stage(smoke_config, stage)
    rows = []
    for seed in stage.seeds:
        seed = int(seed)
        print(f"stage4.1 smoke seed={seed}: building splits")
        splits = build_multiview_code_patch_splits(dataset_config, seed=seed, repo_root=Path("."))
        candidate = _repaired_candidate(stage)
        training = replace(
            _training_config(stage),
            epochs=50,
            patience=6,
            lr=0.0003,
            weight_decay=0.0001,
            gradient_clip_norm=1.0,
        )
        print(f"stage4.1 smoke seed={seed}: fitting trainable repaired variant")
        trainable = fit_latent_system(
            train_examples=splits["train"],
            dev_examples=splits["dev"],
            agent_config=_agent_config(stage),
            coordinator_config=candidate.coordinator_config,
            training_config=training,
            num_classes=8,
            seed=seed + 10_101,
            device=device,
            trainable_agent=True,
            method=f"stage41_trainable__{REPAIRED_VARIANT}",
            message_config=candidate.message_config,
        )
        print(f"stage4.1 smoke seed={seed}: fitting frozen repaired comparator")
        frozen = fit_latent_system(
            train_examples=splits["train"],
            dev_examples=splits["dev"],
            agent_config=_agent_config(stage),
            coordinator_config=candidate.coordinator_config,
            training_config=training,
            num_classes=8,
            seed=seed + 20_201,
            device=device,
            trainable_agent=False,
            method=f"stage41_frozen__{REPAIRED_VARIANT}",
            message_config=candidate.message_config,
        )
        print(f"stage4.1 smoke seed={seed}: fitting text/raw baselines")
        baselines = _fit_smoke_baselines(stage, splits, seed, device)
        row = {
            "seed": seed,
            "variant": REPAIRED_VARIANT,
            "architecture_config": _candidate_config(candidate),
            "dev_accuracy": _smoke_metrics(trainable, frozen, baselines, splits["dev"], seed + 1_000),
            "test_smoke_accuracy": _smoke_metrics(trainable, frozen, baselines, splits["test"], seed + 2_000),
            "trainable_audit": _compact_audit(trainable.audit),
            "frozen_audit": _compact_audit(frozen.audit),
            "dataset_summary": dataset_summary(splits),
        }
        row["dev_delta"] = float(row["dev_accuracy"]["trainable"] - row["dev_accuracy"]["frozen"])
        row["test_smoke_delta"] = float(row["test_smoke_accuracy"]["trainable"] - row["test_smoke_accuracy"]["frozen"])
        rows.append(row)
        del trainable, frozen, baselines
        _clear_cuda()
    summary = _cheap_summary(rows)
    return {
        "stage_config": asdict(stage),
        "dataset_config": asdict(dataset_config),
        "selection_metric": "dev only; test is smoke sanity",
        "rows": rows,
        "summary": summary,
    }


def _fit_smoke_baselines(stage, splits: Dict[str, Sequence[MultiViewTaskExample]], seed: int, device: str) -> Dict[str, object]:
    agent_config = _agent_config(stage)
    training = _training_config(stage)
    raw = fit_latent_system(
        train_examples=splits["train"],
        dev_examples=splits["dev"],
        agent_config=agent_config,
        coordinator_config=replace(_coordinator_config(stage, family="cross_attention", num_layers=1, dropout=0.0), input_dim=agent_config.hidden_dim),
        training_config=training,
        num_classes=8,
        seed=seed + 11_000,
        device=device,
        trainable_agent=True,
        method="stage41_raw_latent_baseline",
    )
    text = TextOutputOnlyCoordinator(
        num_classes=8,
        training=MLPTrainingConfig(
            epochs=max(3, stage.epochs),
            batch_size=stage.batch_size,
            lr=stage.lr,
            weight_decay=0.0001,
            patience=max(2, stage.patience),
            hidden_dims=(32,),
        ),
        seed=seed + 12_000,
        device=device,
        feature_dim=128,
    )
    text.fit(splits["train"], splits["dev"])
    return {"raw": raw, "text": text}


def _smoke_metrics(trainable, frozen, baselines: Dict[str, object], examples: Sequence[MultiViewTaskExample], seed: int) -> Dict[str, float]:
    y = _labels(examples)
    values = {
        "trainable": _accuracy(predict_latent_system(trainable, examples, "none", seed), y),
        "frozen": _accuracy(predict_latent_system(frozen, examples, "none", seed), y),
        "text_only": _accuracy(baselines["text"].predict(examples), y),
        "raw_latent": _accuracy(predict_latent_system(baselines["raw"], examples, "none", seed), y),
        "oracle": _accuracy(_structured_oracle_predictions(examples, [0, 1, 2, 3]), y),
        "physical_order_shuffled_roles_preserved": _accuracy(
            predict_latent_system(trainable, examples, "physical_order_shuffled_roles_preserved", seed), y
        ),
    }
    controls = {
        "candidate_only": _candidate_only(examples),
        "schema_only": _schema_only(examples),
        "view_masked_candidates_visible": apply_example_control(examples, "view_masked", seed=seed + 11_000),
        "value_shuffle_within_schema": _value_shuffle_within_schema(examples, seed + 12_000),
        "candidate_evidence_mismatch": _candidate_evidence_mismatch(examples, seed + 13_000),
    }
    for name, rows in controls.items():
        cy = _labels(rows)
        if name == "candidate_only":
            with _zero_role_embeddings(trainable.system):
                pred = predict_latent_system(trainable, rows, "none", seed)
        else:
            pred = predict_latent_system(trainable, rows, "none", seed)
        values[name] = _accuracy(pred, cy)
    shuffled = apply_example_control(examples, "candidate_order_shuffled", seed=seed + 14_000)
    values["candidate_order_shuffled_with_gold_remap"] = _accuracy(predict_latent_system(trainable, shuffled, "none", seed), _labels(shuffled))
    return values


def _repaired_candidate(stage):
    locked = next(candidate for candidate in _candidate_specs() if candidate.name == "topk_attention_no_head")
    msg = replace(
        locked.message_config,
        readout_source="pooled",
        use_message_head=False,
        use_private_cue_aux=False,
        aux_loss_weight=0.0,
        coordinator_family="candidate_token_cross_attention",
        canonicalize_role_order=True,
    )
    coord = replace(
        _coordinator_config(stage, family="candidate_token_cross_attention", num_layers=1, dropout=0.0),
        family="candidate_token_cross_attention",
    )
    return SimpleNamespace(
        name=REPAIRED_VARIANT,
        description="Candidate-token direct architecture with deterministic role-id canonicalization before readout.",
        message_config=msg,
        coordinator_config=coord,
    )


def _implementation_trace() -> List[Dict[str, object]]:
    return [
        {
            "component": "dataset physical_order_shuffled_roles_preserved",
            "uses_physical_order": False,
            "uses_role_id": False,
            "permutation_safe": True,
            "notes": "Dataset examples are unchanged; the control is a runtime tensor/role-slot perturbation.",
        },
        {
            "component": "SharedClonedAgentSystem.collect_clone_representations",
            "uses_physical_order": True,
            "uses_role_id": True,
            "permutation_safe": True,
            "notes": "Prompts are collected by slot; explicit role_ids can be supplied and are carried separately.",
        },
        {
            "component": "legacy physical-order control path",
            "uses_physical_order": True,
            "uses_role_id": True,
            "permutation_safe": False,
            "notes": "Before Stage 4.1 it permuted clone_activations and role_ids but not token_states/token_mask, which candidate-token direct consumes.",
        },
        {
            "component": "Stage 4.1 repaired physical-order control path",
            "uses_physical_order": True,
            "uses_role_id": True,
            "permutation_safe": True,
            "notes": "All role-axis tensors are gathered together: messages, pooled/msg/evidence, token_states, and token_mask.",
        },
        {
            "component": "CandidateTokenCrossAttentionCoordinator",
            "uses_physical_order": True,
            "uses_role_id": True,
            "permutation_safe": True,
            "notes": "It flattens role-token groups but has no learned slot embedding or role-slot positional encoding; attention is order invariant when role ids and token tensors stay aligned.",
        },
        {
            "component": "role embeddings",
            "uses_physical_order": False,
            "uses_role_id": True,
            "permutation_safe": True,
            "notes": "Embedding lookup depends on role_id, not physical slot.",
        },
        {
            "component": "canonicalize_role_order option",
            "uses_physical_order": False,
            "uses_role_id": True,
            "permutation_safe": True,
            "notes": "Sorts all role-axis tensors by role_id immediately before coordinator/readout.",
        },
    ]


def _conclusion(checkpoint_audit: Dict[str, object], smoke: Dict[str, object] | None) -> Dict[str, object]:
    overall = checkpoint_audit.get("overall", {})
    smoke_summary = (smoke or {}).get("summary", {})
    fixed = bool(overall.get("fixed_tensor_permutation_fixes_checkpoint_behavior", False))
    canonical = bool(overall.get("canonical_sorting_fixes_checkpoint_behavior", False))
    return {
        "failed_invariance_gate_cause": "control/tensor permutation bug" if fixed else "residual architecture order sensitivity",
        "role_tensors_ids_masks_permuted_correctly_after_repair": bool(overall.get("all_integrity_checks_pass", False)),
        "canonical_role_sorting_fixes_checkpoint_behavior": canonical,
        "cheap_validation_passed": bool(smoke_summary.get("cheap_validation_passed", False)) if smoke is not None else None,
        "full_final_validation_justified": bool(smoke_summary.get("cheap_validation_passed", False)) if smoke is not None else False,
        "success_claim": "not claimed",
    }


def _render_report(result: Dict[str, object]) -> str:
    audit = result["checkpoint_audit"]
    smoke = result.get("cheap_validation")
    conclusion = result["conclusion"]
    lines = [
        "# Stage 4.1 Physical-Order Invariance Audit",
        "",
        "## Scope",
        "",
        f"- Selected final architecture audited: `{SELECTED_ARCHITECTURE}`.",
        "- Dataset/control policy unchanged: Stage 3.8 schema-aware balanced_categories_v3 with role schemas preserved.",
        "- No full 10-seed validation was run.",
        "- No success claim is made.",
        "",
        "## Shortcut Diagnostics",
        "",
        f"- Status: `{bool(result.get('shortcut_diagnostics', {}).get('passes', False))}`.",
        "- Source: carried forward from the fixed Stage 4 final benchmark because Stage 4.1 did not change the dataset or controls.",
        "",
        "## No-Training Checkpoint Audit",
        "",
        "| seed | normal | prior failed physical | fixed builtin physical | all-perm mean | all-perm min | all-perm max | canonical mean | legacy-bug mean | integrity pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in audit.get("seed_rows", []):
        perm = row["permutation_accuracy_summary"]
        canon = row["canonicalized_permutation_accuracy_summary"]
        legacy = row["legacy_bug_permutation_accuracy_summary"]
        lines.append(
            "| {seed} | {normal:.4f} | {prior:.4f} | {builtin:.4f} | {pmean:.4f} | {pmin:.4f} | {pmax:.4f} | {cmean:.4f} | {lmean:.4f} | `{ok}` |".format(
                seed=row["seed"],
                normal=float(row["normal_accuracy_recomputed"]),
                prior=float(row["final_recorded_failed_physical_order_accuracy"]),
                builtin=float(row["builtin_physical_order_after_tensor_repair"]),
                pmean=float(perm["mean"]),
                pmin=float(perm["min"]),
                pmax=float(perm["max"]),
                cmean=float(canon["mean"]),
                lmean=float(legacy["mean"]),
                ok=bool(row["all_integrity_checks_pass"]),
            )
        )
    overall = audit.get("overall", {})
    lines.extend(
        [
            "",
            "## Checkpoint Summary",
            "",
            f"- Normal mean: `{float(overall.get('normal_mean', 0.0)):.4f}`",
            f"- Correct all-permutation mean: `{float(overall.get('correct_permutation_mean', 0.0)):.4f}`",
            f"- Canonicalized all-permutation mean: `{float(overall.get('canonicalized_permutation_mean', 0.0)):.4f}`",
            f"- Legacy buggy permutation mean: `{float(overall.get('legacy_bug_permutation_mean', 0.0)):.4f}`",
            f"- Fixed tensor permutation matches normal within 0.02: `{bool(overall.get('fixed_tensor_permutation_fixes_checkpoint_behavior', False))}`",
            f"- Canonical sorting matches normal within 0.02: `{bool(overall.get('canonical_sorting_fixes_checkpoint_behavior', False))}`",
            "",
            "## Implementation Trace",
            "",
            "| component | uses physical order? | uses role id? | permutation-safe? | notes |",
            "|---|---|---|---|---|",
        ]
    )
    for row in result.get("source_trace", []):
        lines.append(
            f"| {row['component']} | `{bool(row['uses_physical_order'])}` | `{bool(row['uses_role_id'])}` | `{bool(row['permutation_safe'])}` | {row['notes']} |"
        )
    if smoke is not None:
        summary = smoke.get("summary", {})
        lines.extend(
            [
                "",
                "## Cheap Validation",
                "",
                "| seed | dev trainable | dev frozen | dev delta | dev physical | dev cand-order | dev candidate-only | dev schema-only | dev value-shuffle | dev mismatch | test trainable | test frozen |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in smoke.get("rows", []):
            dev = row["dev_accuracy"]
            test = row["test_smoke_accuracy"]
            lines.append(
                "| {seed} | {train:.4f} | {frozen:.4f} | {delta:.4f} | {physical:.4f} | {corder:.4f} | {cand:.4f} | {schema:.4f} | {value:.4f} | {mismatch:.4f} | {tt:.4f} | {tf:.4f} |".format(
                    seed=row["seed"],
                    train=float(dev["trainable"]),
                    frozen=float(dev["frozen"]),
                    delta=float(row["dev_delta"]),
                    physical=float(dev["physical_order_shuffled_roles_preserved"]),
                    corder=float(dev["candidate_order_shuffled_with_gold_remap"]),
                    cand=float(dev["candidate_only"]),
                    schema=float(dev["schema_only"]),
                    value=float(dev["value_shuffle_within_schema"]),
                    mismatch=float(dev["candidate_evidence_mismatch"]),
                    tt=float(test["trainable"]),
                    tf=float(test["frozen"]),
                )
            )
        lines.extend(
            [
                "",
                "## Cheap Validation Summary",
                "",
                f"- Trainable beats frozen on dev seeds: `{summary.get('dev_trainable_beats_frozen_count')}/3`",
                f"- Mean dev trainable-frozen delta: `{float(summary.get('mean_dev_delta', 0.0)):.4f}`",
                f"- Mean dev physical-order delta from trainable: `{float(summary.get('mean_dev_physical_delta_from_trainable', 0.0)):.4f}`",
                f"- Mean dev candidate-order delta from trainable: `{float(summary.get('mean_dev_candidate_order_delta_from_trainable', 0.0)):.4f}`",
                f"- Controls near chance: `{bool(summary.get('controls_near_chance', False))}`",
                f"- Cheap validation passed: `{bool(summary.get('cheap_validation_passed', False))}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Conservative Interpretation",
            "",
            f"- Failed final invariance gate cause: `{conclusion.get('failed_invariance_gate_cause')}`.",
            f"- Role tensors/ids/masks are permuted correctly after repair: `{bool(conclusion.get('role_tensors_ids_masks_permuted_correctly_after_repair'))}`.",
            f"- Canonical role sorting fixes checkpoint behavior: `{bool(conclusion.get('canonical_role_sorting_fixes_checkpoint_behavior'))}`.",
            f"- Full final validation justified by this Stage 4.1 cheap gate: `{bool(conclusion.get('full_final_validation_justified'))}`.",
            "- Stage 4.1 does not claim benchmark success; it only audits and repairs the physical-order invariance failure.",
        ]
    )
    return "\n".join(lines) + "\n"


def _cheap_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    dev_train = [float(row["dev_accuracy"]["trainable"]) for row in rows]
    dev_frozen = [float(row["dev_accuracy"]["frozen"]) for row in rows]
    deltas = [float(row["dev_delta"]) for row in rows]
    physical = [float(row["dev_accuracy"]["physical_order_shuffled_roles_preserved"]) for row in rows]
    candidate_order = [float(row["dev_accuracy"]["candidate_order_shuffled_with_gold_remap"]) for row in rows]
    controls = ("candidate_only", "schema_only", "view_masked_candidates_visible", "value_shuffle_within_schema", "candidate_evidence_mismatch")
    control_means = {name: _mean([float(row["dev_accuracy"].get(name, 1.0)) for row in rows]) for name in controls}
    physical_delta = _mean(physical) - _mean(dev_train)
    candidate_order_delta = _mean(candidate_order) - _mean(dev_train)
    wins = sum(1 for row in rows if float(row["dev_accuracy"]["trainable"]) > float(row["dev_accuracy"]["frozen"]))
    controls_ok = all(value <= NEAR_CHANCE for value in control_means.values())
    return {
        "dev_trainable_beats_frozen_count": wins,
        "mean_dev_trainable": _mean(dev_train),
        "mean_dev_frozen": _mean(dev_frozen),
        "mean_dev_delta": _mean(deltas),
        "mean_dev_physical_delta_from_trainable": physical_delta,
        "mean_dev_candidate_order_delta_from_trainable": candidate_order_delta,
        "control_means": control_means,
        "controls_near_chance": controls_ok,
        "oracle_mean": _mean([float(row["dev_accuracy"]["oracle"]) for row in rows]),
        "cheap_validation_passed": bool(
            len(rows) >= 3
            and wins >= 2
            and _mean(deltas) >= 0.20
            and abs(physical_delta) <= 0.02
            and abs(candidate_order_delta) <= 0.08
            and controls_ok
        ),
    }


def _shortcut_diagnostics(final_results: Dict[str, object]) -> Dict[str, object]:
    pre_run = dict(final_results.get("pre_run", {}))
    summary = dict(pre_run.get("summary", {}))
    return {
        "source": "stage4_final_candidate_token_direct_validation",
        "carried_forward_because_dataset_and_controls_unchanged": True,
        "passes": bool(summary.get("passes", False)),
        "summary": summary,
    }


def _stage41_base_config(config: Dict[str, object]) -> Dict[str, object]:
    out = _stage38_full_config(config)
    out["architecture"] = SELECTED_ARCHITECTURE
    out["require_cuda"] = True
    out["dataset_config"] = {
        **dict(out.get("dataset_config", {})),
        "dataset_source": BALANCED_34B_DATASET_SOURCE,
        "generator_version": "real_import_restore_candidate_balanced_34b_stage41_physical_order_audit",
        "candidate_representation": BALANCED_34B_CANDIDATE_REPRESENTATION,
    }
    return out


def _stage41_smoke_config(config: Dict[str, object]) -> Dict[str, object]:
    out = json.loads(json.dumps(config))
    out["stage"] = {
        **dict(out["stage"]),
        "name": "stage41_physical_order_invariance_smoke",
        "n_train": 512,
        "n_dev": 256,
        "n_test": 256,
        "seeds": [0, 1, 2],
        "epochs": 50,
        "patience": 6,
        "batch_size": 32,
        "lr": 0.0003,
        "mixed_precision": "bf16",
    }
    return out


def _permute_example_views(examples: Sequence[MultiViewTaskExample], perm: Sequence[int]) -> List[MultiViewTaskExample]:
    from dataclasses import replace as dc_replace

    return [dc_replace(example, views=tuple(example.views[int(index)] for index in perm)) for example in examples]


def _example_batches(examples: Sequence[MultiViewTaskExample], batch_size: int) -> Sequence[Sequence[MultiViewTaskExample]]:
    return [examples[start : start + batch_size] for start in range(0, len(examples), batch_size)]


def _summary_values(values: Sequence[float]) -> Dict[str, float]:
    return {
        "mean": _mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": float(min(values)) if values else 0.0,
        "max": float(max(values)) if values else 0.0,
    }


def _integrity_passes(row: Dict[str, object]) -> bool:
    keys = (
        "view_text_moved_correctly",
        "role_id_moved_with_view",
        "role_embedding_moved_with_view",
        "attention_masks_moved_with_view",
        "token_states_moved_with_view",
        "candidate_token_direct_receives_same_role_identity",
        "labels_unchanged",
        "candidate_order_unchanged",
    )
    return all(bool(row.get(key, False)) for key in keys)


def _compact_audit(audit: Dict[str, object]) -> Dict[str, object]:
    keys = (
        "shared_parameter_identity",
        "agent_grad_norm_mean",
        "agent_parameter_delta",
        "coordinator_grad_norm_mean",
        "coordinator_parameter_delta",
        "message_coordinator_family",
        "readout_source",
        "canonicalize_role_order",
        "epochs_run",
        "best_dev_selection_acc",
    )
    return {key: audit.get(key) for key in keys if key in audit}


def _mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
