from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from src.datasets.multiview_code_patch_selection import (
    SANITIZED_DATASET_SOURCE,
    MultiViewTaskExample,
    apply_example_control,
    build_multiview_code_patch_splits,
    dataset_summary,
    output_leakage_audit,
    split_leakage_audit,
)
from src.experiments.architecture_search import (
    _agent_config,
    _candidate_config,
    _candidate_specs,
    _fit_baselines,
    _training_config,
)
from src.experiments.real_shared_weight_latent_coordination import BENCHMARK, fit_latent_system, predict_latent_system
from src.experiments.run_stage3_gpu_hard_validation import ARCHITECTURE, _clear_cuda, _configure_cuda, _stage_from_config
from src.experiments.run_stage34_candidate_sanitization_diagnostics import (
    DEFAULT_CONFIG,
    _accuracy,
    _structured_oracle_predictions,
    run_diagnostics,
    stage34_dataset_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 3.4 candidate-sanitization tiny CUDA smoke.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_smoke(config, config_path)


def run_smoke(config: Dict[str, object], config_path: Path) -> Dict[str, object]:
    diagnostics = _load_or_run_diagnostics(config, config_path)
    if not bool(diagnostics.get("summary", {}).get("pretraining_dataset_validity_passed", False)):
        raise RuntimeError("Stage 3.4 dataset diagnostics did not pass; refusing to run CUDA smoke")

    device, hardware = _configure_cuda(config)
    output_path = Path(str(config.get("output_path", "results/stage34_candidate_sanitization_smoke.json")))
    report_path = Path(str(config.get("report_path", "reports/STAGE34_CANDIDATE_SANITIZATION.md")))
    stage_base = _stage_from_config(config)
    candidate = next(candidate for candidate in _candidate_specs() if candidate.name == ARCHITECTURE)
    rows = []
    for seed in [int(value) for value in stage_base.seeds]:
        _clear_cuda()
        stage = stage_base
        dataset_config = stage34_dataset_config({**config, "stage": {**dict(config["stage"]), "batch_size": stage.batch_size}})
        splits = build_multiview_code_patch_splits(dataset_config, seed=seed, repo_root=Path("."))
        baselines = _fit_baselines(stage, splits, seed=seed, device=device)
        row = _run_seed(stage, seed, candidate, splits, baselines, device, hardware)
        row.update(
            {
                "dataset_config": asdict(dataset_config),
                "dataset_summary": dataset_summary(splits),
                "split_leakage_audit": split_leakage_audit(BENCHMARK, seed, splits),
                "output_leakage_audit": output_leakage_audit(BENCHMARK, seed, splits["train"] + splits["dev"] + splits["test"]),
            }
        )
        rows.append(row)
        _clear_cuda()
    result = {
        "metadata": {
            "stage": "stage3.4_candidate_sanitization_cuda_smoke",
            "config_path": str(config_path),
            "dataset": SANITIZED_DATASET_SOURCE,
            "architecture": ARCHITECTURE,
            "architecture_changes": "none",
            "device": device,
            "hardware": hardware,
            "created_at_utc": _now(),
            "scope": "tiny one-seed smoke only; not full validation",
        },
        "config": config,
        "dataset_diagnostics": diagnostics,
        "seed_rows": rows,
        "summary": _summary(rows, diagnostics),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(result), encoding="utf-8")
    return result


def _load_or_run_diagnostics(config: Dict[str, object], config_path: Path) -> Dict[str, object]:
    path = Path(str(config.get("dataset_diagnostics_output_path", "results/stage34_candidate_sanitization_diagnostics.json")))
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("dataset_config", {}).get("generator_version") == stage34_dataset_config(config).generator_version:
            return data
    return run_diagnostics(config, config_path=config_path)


def _run_seed(stage, seed: int, candidate, splits: Dict[str, Sequence[MultiViewTaskExample]], baselines: Dict[str, object], device: str, hardware: Dict[str, object]) -> Dict[str, object]:
    agent_config = _agent_config(stage)
    training = _training_config(stage)
    trainable = fit_latent_system(
        train_examples=splits["train"],
        dev_examples=splits["dev"],
        agent_config=agent_config,
        coordinator_config=candidate.coordinator_config,
        training_config=training,
        num_classes=8,
        seed=seed + 101,
        device=device,
        trainable_agent=True,
        method=f"stage34_trainable__{candidate.name}",
        message_config=candidate.message_config,
    )
    frozen = fit_latent_system(
        train_examples=splits["train"],
        dev_examples=splits["dev"],
        agent_config=agent_config,
        coordinator_config=candidate.coordinator_config,
        training_config=training,
        num_classes=8,
        seed=seed + 201,
        device=device,
        trainable_agent=False,
        method=f"stage34_frozen__{candidate.name}",
        message_config=candidate.message_config,
    )
    examples = list(splits["test"])
    labels = np.asarray([example.label for example in examples], dtype=np.int64)
    masked = apply_example_control(examples, "view_masked", seed=seed + 40_000)
    role_shuffled = apply_example_control(examples, "role_labels_shuffled", seed=seed + 50_000)
    with _zero_role_embeddings(trainable.system):
        candidate_only = predict_latent_system(trainable, masked, "none", seed=seed + 60_000)
    metrics = {
        "trainable": _accuracy(predict_latent_system(trainable, examples, "none", seed), labels),
        "frozen": _accuracy(predict_latent_system(frozen, examples, "none", seed), labels),
        "text_only": _accuracy(baselines["text"].predict(examples), labels),
        "raw_latent": _accuracy(predict_latent_system(baselines["raw"], examples, "none", seed), labels),
        "candidate_only": _accuracy(candidate_only, labels),
        "view_masked": _accuracy(predict_latent_system(trainable, masked, "view_masked", seed), labels),
        "role_labels_shuffled": _accuracy(predict_latent_system(trainable, role_shuffled, "role_labels_shuffled", seed), labels),
        "oracle": _accuracy(_structured_oracle_predictions(examples, [0, 1, 2, 3]), labels),
    }
    return {
        "seed": int(seed),
        "status": "completed",
        "completed_at_utc": _now(),
        "stage_config": asdict(stage),
        "architecture": candidate.name,
        "architecture_config": _candidate_config(candidate),
        "test_accuracy": metrics,
        "trainable_audit": trainable.audit,
        "frozen_audit": frozen.audit,
        "cuda_max_memory_allocated": int(torch.cuda.max_memory_allocated()) if str(device).startswith("cuda") and torch.cuda.is_available() else 0,
        "cuda_device_name": hardware.get("cuda_device_name"),
    }


@contextmanager
def _zero_role_embeddings(system):
    saved = []
    try:
        with torch.no_grad():
            for module in (system.active_message_readout, system.coordinator):
                embedding = getattr(module, "role_embedding", None) if module is not None else None
                if embedding is not None:
                    saved.append((embedding, embedding.weight.detach().clone()))
                    embedding.weight.zero_()
        yield
    finally:
        with torch.no_grad():
            for embedding, weight in saved:
                embedding.weight.copy_(weight)


def _summary(rows: Sequence[Dict[str, object]], diagnostics: Dict[str, object]) -> Dict[str, object]:
    controls = ["candidate_only", "view_masked", "role_labels_shuffled"]
    means = {
        name: float(np.mean([float(row["test_accuracy"][name]) for row in rows])) if rows else 0.0
        for name in ["trainable", "frozen", "text_only", "raw_latent", *controls, "oracle"]
    }
    smoke_controls_pass = bool(
        rows
        and all(means[name] <= 0.18 for name in controls)
        and means["oracle"] >= 0.90
        and bool(diagnostics.get("summary", {}).get("pretraining_dataset_validity_passed", False))
    )
    return {
        "n_completed": len(rows),
        "mean_test_accuracy": means,
        "candidate_metadata_fully_removed_from_model_inputs": bool(diagnostics.get("summary", {}).get("leakage_passes", False)),
        "candidate_only_collapsed": means.get("candidate_only", 1.0) <= 0.18,
        "view_masked_collapsed": means.get("view_masked", 1.0) <= 0.18,
        "role_pair_only_collapsed": float(diagnostics.get("summary", {}).get("mean_role_pair_only_accuracy", 1.0)) <= 0.18,
        "oracle_remains_high": means.get("oracle", 0.0) >= 0.90,
        "smoke_passed": smoke_controls_pass,
        "full_validation_justified": False,
    }


def _render_report(result: Dict[str, object]) -> str:
    summary = result.get("summary", {})
    diagnostics = result.get("dataset_diagnostics", {}).get("summary", {})
    rows = result.get("seed_rows", [])
    lines = [
        "# Stage 3.4 Candidate Sanitization",
        "",
        "## Dataset Repair",
        "",
        f"- Dataset mode: `{SANITIZED_DATASET_SOURCE}`.",
        "- Oracle-only compatibility fields are stored in `oracle_metadata`; model-facing `metadata`, candidate text, and candidate attributes are sanitized.",
        "- Locked architecture unchanged: `topk_attention_no_head`.",
        "",
        "## Dataset-Only Diagnostics",
        "",
        f"- Candidate-only baseline: `{float(diagnostics.get('mean_candidate_only_accuracy', 0.0)):.4f}`",
        f"- Candidate-metadata-only baseline: `{float(diagnostics.get('mean_candidate_metadata_only_accuracy', 0.0)):.4f}`",
        f"- Role-pair-only baseline: `{float(diagnostics.get('mean_role_pair_only_accuracy', 0.0)):.4f}`",
        f"- Family-only baseline: `{float(diagnostics.get('mean_family_only_accuracy', 0.0)):.4f}`",
        f"- View-masked + candidates-visible baseline: `{float(diagnostics.get('mean_view_masked_candidates_visible_accuracy', 0.0)):.4f}`",
        f"- Lexical-overlap baseline: `{float(diagnostics.get('mean_lexical_overlap_accuracy', 0.0)):.4f}`",
        f"- Static frequency baseline: `{float(diagnostics.get('mean_static_frequency_accuracy', 0.0)):.4f}`",
        f"- Max single-view baseline: `{float(diagnostics.get('max_single_view_accuracy', 0.0)):.4f}`",
        f"- Max pairwise structured oracle: `{float(diagnostics.get('max_pairwise_structured_oracle_accuracy', 0.0)):.4f}`",
        f"- All-role structured oracle: `{float(diagnostics.get('mean_all_role_structured_oracle_accuracy', 0.0)):.4f}`",
        f"- Dataset gates passed: `{bool(diagnostics.get('pretraining_dataset_validity_passed', False))}`",
        "",
        "## Tiny CUDA Smoke",
        "",
        "| seed | trainable | frozen | text | raw | candidate-only | view-masked | role-shuffled | oracle |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        test = row.get("test_accuracy", {})
        lines.append(
            "| {seed} | {trainable:.4f} | {frozen:.4f} | {text:.4f} | {raw:.4f} | {candidate:.4f} | {masked:.4f} | {role:.4f} | {oracle:.4f} |".format(
                seed=row.get("seed"),
                trainable=float(test.get("trainable", 0.0)),
                frozen=float(test.get("frozen", 0.0)),
                text=float(test.get("text_only", 0.0)),
                raw=float(test.get("raw_latent", 0.0)),
                candidate=float(test.get("candidate_only", 0.0)),
                masked=float(test.get("view_masked", 0.0)),
                role=float(test.get("role_labels_shuffled", 0.0)),
                oracle=float(test.get("oracle", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Candidate metadata fully removed from model inputs: `{bool(summary.get('candidate_metadata_fully_removed_from_model_inputs', False))}`",
            f"- Candidate-only collapsed: `{bool(summary.get('candidate_only_collapsed', False))}`",
            f"- View-masked collapsed: `{bool(summary.get('view_masked_collapsed', False))}`",
            f"- Role-pair-only collapsed: `{bool(summary.get('role_pair_only_collapsed', False))}`",
            f"- Oracle remains high: `{bool(summary.get('oracle_remains_high', False))}`",
            f"- Smoke passed: `{bool(summary.get('smoke_passed', False))}`",
            f"- Full validation justified now: `{bool(summary.get('full_validation_justified', False))}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    main()
