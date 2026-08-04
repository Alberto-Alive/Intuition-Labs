from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.datasets.arc_agi2_verification import (
    ArcPair,
    ArcVerificationDatasetConfig,
    ArcVerificationExample,
    apply_arc_control,
    build_arc_verification_splits,
    candidate_metadata_only_accuracy,
    dataset_summary,
    leakage_audit_rows,
)


BENCHMARK = "arc1_latent_rule_clones"
DEFAULT_CONFIG = "configs/stage_arc1_latent_rule_clones_smoke.json"
DEFAULT_RESULTS = "results/arc1_latent_rule_clones_results.json"
DEFAULT_CONTROLS = "results/arc1_latent_rule_clones_controls.json"
DEFAULT_LEAKAGE = "results/arc1_latent_rule_clones_leakage_audit.jsonl"
DEFAULT_ERRORS = "results/arc1_latent_rule_clones_error_cases.jsonl"
DEFAULT_ATTENTION = "results/arc1_latent_rule_clones_attention_summaries.jsonl"
DEFAULT_COMPUTE = "results/arc1_latent_rule_clones_compute_metrics.json"
DEFAULT_REPORT = "reports/STAGE_ARC1_LATENT_RULE_CLONES.md"

FIELD_VOCABS = (12, 12, 8, 32, 32, 32, 32, 16, 16, 8, 32, 32)
SUMMARY_ROWCOL = 31
ROLE_TRAIN_INPUT = 0
ROLE_TRAIN_OUTPUT = 1
ROLE_TRAIN_DIFF = 2
ROLE_TEST_INPUT = 3
ROLE_CANDIDATE_OUTPUT = 4
ROLE_CANDIDATE_DIFF = 5
ROLE_SUMMARY = 6
ROLE_EXECUTION_EVIDENCE = 7
VIEW_NAMES = (
    "raw",
    "diff",
    "object",
    "color",
    "geometry",
    "symmetry",
    "counting",
    "candidate_execution",
)
VIEW_ID = {name: index for index, name in enumerate(VIEW_NAMES)}


@dataclass(frozen=True)
class ArcModelConfig:
    variant_family: str = "view_biased_rule_clones"
    view_names: Tuple[str, ...] = ("raw", "diff", "object", "color", "geometry", "symmetry")
    learned_slots: bool = False
    representation_mode: str = "hybrid_cell_object_diff"
    interaction_style: str = "late_candidate_query"
    model_dim: int = 64
    num_heads: int = 4
    rule_layers: int = 1
    candidate_layers: int = 1
    coordination_blocks: int = 1
    ff_dim: int = 128
    dropout: float = 0.0
    max_rule_tokens: int = 192
    max_candidate_tokens: int = 128
    max_grid_size: int = 30


@dataclass(frozen=True)
class ArcTrainingConfig:
    epochs: int = 3
    batch_size: int = 12
    lr: float = 0.0003
    weight_decay: float = 0.0001
    patience: int = 2
    gradient_clip_norm: float = 1.0
    objective: str = "cross_entropy"
    margin: float = 0.25


@dataclass(frozen=True)
class ArcVariant:
    name: str
    description: str
    model: ArcModelConfig


@dataclass
class ArcFitResult:
    method: str
    model: "ArcRuleCloneVerifier"
    config: ArcModelConfig
    trainable_shared: bool
    audit: Dict[str, object]
    history: List[Dict[str, float]]
    training_time_seconds: float
    param_count: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage ARC-1 latent rule-clone ARC-AGI-2 verifier/ranker.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_RESULTS)
    parser.add_argument("--controls-output", default=DEFAULT_CONTROLS)
    parser.add_argument("--leakage-output", default=DEFAULT_LEAKAGE)
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
        config["dataset"]["dataset_root"] = str(args.dataset_root)
    if args.device:
        config["device"] = str(args.device)
    result_bundle = run_stage_arc1(config=config, max_variants=int(args.max_variants))
    _write_outputs(
        result_bundle,
        output_path=Path(args.output),
        controls_path=Path(args.controls_output),
        leakage_path=Path(args.leakage_output),
        error_path=Path(args.error_output),
        attention_path=Path(args.attention_output),
        compute_path=Path(args.compute_output),
        report_path=Path(args.report),
    )


def run_stage_arc1(config: Dict[str, object], max_variants: int = 0) -> Dict[str, object]:
    dataset_config = _dataset_config(config.get("dataset", {}))
    training = _training_config(config.get("training", {}))
    seeds = [int(value) for value in config.get("seeds", [0])]
    device = _resolve_device(str(config.get("device", "auto")))
    variants = _variant_plan()
    if max_variants > 0:
        variants = variants[:max_variants]
    if int(config.get("max_variants", 0)) > 0:
        variants = variants[: int(config.get("max_variants", 0))]
    checkpoint_dir = Path(str(config.get("checkpoint_dir", "results/arc1_latent_rule_clones_checkpoints")))

    all_rows: List[Dict[str, object]] = []
    all_controls: List[Dict[str, object]] = []
    all_attention: List[Dict[str, object]] = []
    all_errors: List[Dict[str, object]] = []
    leakage_rows_out: List[Dict[str, object]] = []
    compute_rows: List[Dict[str, object]] = []
    phase0_rows: List[Dict[str, object]] = []

    print(f"arc1: device={device} variants={len(variants)} seeds={seeds}")
    for seed in seeds:
        splits = build_arc_verification_splits(dataset_config, seed=seed)
        leakage_rows = leakage_audit_rows([example for rows in splits.values() for example in rows])
        leakage_rows_out.extend(leakage_rows)
        phase0 = _phase0_summary(splits, leakage_rows)
        phase0_rows.append({"seed": seed, **phase0})
        print(
            "arc1 seed={seed}: examples train/dev/test={train}/{dev}/{test} candidate-only-source={source:.4f}".format(
                seed=seed,
                train=len(splits["train"]),
                dev=len(splits["dev"]),
                test=len(splits["test"]),
                source=float(phase0["candidate_metadata_only_accuracy_dev"]),
            )
        )
        for variant in variants:
            print(f"arc1 seed={seed}: fitting {variant.name} trainable/frozen")
            trainable = fit_arc_verifier(
                train_examples=splits["train"],
                dev_examples=splits["dev"],
                model_config=variant.model,
                training_config=training,
                seed=seed + 10_001,
                device=device,
                trainable_shared=True,
                method=f"trainable__{variant.name}",
            )
            frozen = fit_arc_verifier(
                train_examples=splits["train"],
                dev_examples=splits["dev"],
                model_config=variant.model,
                training_config=training,
                seed=seed + 10_001,
                device=device,
                trainable_shared=False,
                method=f"frozen__{variant.name}",
            )
            eval_start = time.perf_counter()
            logits_train = predict_logits(trainable.model, splits["test"], variant.model, training.batch_size, device)
            logits_frozen = predict_logits(frozen.model, splits["test"], variant.model, training.batch_size, device)
            latency = (time.perf_counter() - eval_start) / max(1, len(splits["test"]))
            checkpoint_paths = _save_arc_checkpoints(checkpoint_dir, variant.name, seed, trainable, frozen, dataset_config, training)
            labels = np.asarray([example.label for example in splits["test"]], dtype=np.int64)
            row = {
                "benchmark": BENCHMARK,
                "seed": seed,
                "variant": variant.name,
                "variant_description": variant.description,
                "model_config": asdict(variant.model),
                "training_config": asdict(training),
                "trainable": _metric_block(logits_train, labels, splits["test"]),
                "frozen": _metric_block(logits_frozen, labels, splits["test"]),
                "delta_top1_trainable_minus_frozen": float(_top1(logits_train, labels) - _top1(logits_frozen, labels)),
                "baselines": _baseline_block(splits["train"], splits["dev"], splits["test"]),
                "trainable_audit": trainable.audit,
                "frozen_audit": frozen.audit,
                "checkpoint_paths": checkpoint_paths,
            }
            all_rows.append(row)
            controls = _run_controls(trainable.model, variant.model, splits["test"], training.batch_size, device, seed)
            controls.update(
                {
                    "benchmark": BENCHMARK,
                    "seed": seed,
                    "variant": variant.name,
                    "chance": 1.0 / max(1, dataset_config.num_candidates),
                    "control_pass": _controls_pass(controls, row["trainable"]["top1"], dataset_config.num_candidates),
                }
            )
            all_controls.append(controls)
            all_attention.extend(_attention_summaries(trainable.model, variant.model, splits["test"], training.batch_size, device, seed, limit=8))
            all_errors.extend(_error_cases(splits["test"], logits_train, logits_frozen, seed, variant.name, controls, limit=24))
            compute_rows.append(
                {
                    "benchmark": BENCHMARK,
                    "seed": seed,
                    "variant": variant.name,
                    "candidate_count": dataset_config.num_candidates,
                    "rule_clones": len(variant.model.view_names),
                    "coordination_blocks": variant.model.coordination_blocks,
                    "trainable_training_time_seconds": trainable.training_time_seconds,
                    "frozen_training_time_seconds": frozen.training_time_seconds,
                    "inference_latency_seconds_per_task": float(latency),
                    "peak_cuda_memory_bytes": _cuda_max_memory(device),
                    "trainable_param_count": trainable.param_count,
                    "frozen_param_count": frozen.param_count,
                    "approx_attention_cost": int(dataset_config.num_candidates)
                    * int(len(variant.model.view_names))
                    * int(variant.model.max_rule_tokens)
                    * int(variant.model.coordination_blocks),
                    "accuracy_per_millisecond": float(row["trainable"]["top1"]) / max(1e-9, latency * 1000.0),
                    "delta_over_frozen_per_attention_cost": float(row["delta_top1_trainable_minus_frozen"])
                    / max(
                        1,
                        int(dataset_config.num_candidates)
                        * int(len(variant.model.view_names))
                        * int(variant.model.max_rule_tokens)
                        * int(variant.model.coordination_blocks),
                    ),
                }
            )
            del trainable, frozen
            _clear_cuda()

    summary = _overall_summary(all_rows, all_controls, phase0_rows, dataset_config)
    return {
        "metadata": {
            "benchmark": BENCHMARK,
            "created_at_utc": _now(),
            "device": device,
            "dataset_root": dataset_config.dataset_root,
            "claim_boundary": "candidate-output verification/ranking only; known output size; gold-present candidate sets",
            "architecture_anchor": "candidate_token_direct_lr3e4_clip1 adapted to ARC grid tokens",
            "final_validation_run": False,
            "forbidden_claims": [
                "solved ARC-AGI-2",
                "full grid generation",
                "output-size prediction",
                "general ARC solver",
            ],
        },
        "dataset_config": asdict(dataset_config),
        "training_config": asdict(training),
        "variant_plan": [asdict(variant) for variant in variants],
        "phase0": phase0_rows,
        "rows": all_rows,
        "controls": all_controls,
        "leakage_audit": leakage_rows_out,
        "error_cases": all_errors,
        "attention_summaries": all_attention,
        "compute_metrics": compute_rows,
        "summary": summary,
    }


def fit_arc_verifier(
    train_examples: Sequence[ArcVerificationExample],
    dev_examples: Sequence[ArcVerificationExample],
    model_config: ArcModelConfig,
    training_config: ArcTrainingConfig,
    seed: int,
    device: str,
    trainable_shared: bool,
    method: str,
) -> ArcFitResult:
    _set_seed(seed)
    model = ArcRuleCloneVerifier(model_config).to(device)
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
        dev_logits = predict_logits(model, dev_examples, model_config, int(training_config.batch_size), device)
        dev_labels = np.asarray([example.label for example in dev_examples], dtype=np.int64)
        dev_acc = _top1(dev_logits, dev_labels)
        history.append({"epoch": float(epoch), "loss": _mean(losses), "dev_top1": float(dev_acc)})
        if dev_acc > best_dev:
            best_dev = float(dev_acc)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= int(training_config.patience):
            break
    model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    elapsed = time.perf_counter() - start
    final_shared = _flat_params(model.shared_parameter_items())
    final_coord = _flat_params(model.coordinator_parameter_items())
    audit = {
        "method": method,
        "initialization_seed": seed,
        "shared_model_trainable": bool(trainable_shared),
        "shared_parameter_names": [name for name, _parameter in model.shared_parameter_items()],
        "coordinator_parameter_names": [name for name, _parameter in model.coordinator_parameter_items()],
        "shared_grad_norm_mean": _mean(shared_grad_norms),
        "coordinator_grad_norm_mean": _mean(coord_grad_norms),
        "shared_parameter_delta": _l2_delta(initial_shared, final_shared),
        "coordinator_parameter_delta": _l2_delta(initial_coord, final_coord),
        "frozen_shared_model_zero_grad": bool((not trainable_shared) and _mean(shared_grad_norms) == 0.0),
        "frozen_shared_model_zero_delta": bool((not trainable_shared) and _l2_delta(initial_shared, final_shared) == 0.0),
        "trainable_shared_model_changed": bool(trainable_shared and _l2_delta(initial_shared, final_shared) > 0.0),
        "gradient_clip_norm": float(training_config.gradient_clip_norm),
        "same_architecture_comparator": True,
    }
    return ArcFitResult(
        method=method,
        model=model,
        config=model_config,
        trainable_shared=trainable_shared,
        audit=audit,
        history=history,
        training_time_seconds=float(elapsed),
        param_count=sum(parameter.numel() for parameter in model.parameters()),
    )


def predict_logits(
    model: "ArcRuleCloneVerifier",
    examples: Sequence[ArcVerificationExample],
    model_config: ArcModelConfig,
    batch_size: int,
    device: str,
    condition: str = "none",
    seed: int = 0,
    view_mask: str | None = None,
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
            if view_mask is not None:
                for axis, view in enumerate(view_names):
                    if view == view_mask:
                        batch["rule_mask"][:, axis, :] = False
            output = model(
                batch,
                return_attention=return_attention,
                hidden_state_shuffle=condition == "hidden_state_shuffle",
                shuffle_seed=seed + offset,
            )
            logits_out.append(output["logits"].detach().cpu().numpy())
            if return_attention:
                attention_rows.extend(_summarize_attention_batch(output, batch, batch_examples, view_names))
    logits = np.concatenate(logits_out, axis=0) if logits_out else np.zeros((0, 0), dtype=np.float32)
    if return_attention:
        return logits, attention_rows
    return logits


def _candidate_objective_loss(logits: torch.Tensor, labels: torch.Tensor, training_config: ArcTrainingConfig) -> torch.Tensor:
    objective = str(training_config.objective)
    if objective == "cross_entropy":
        return F.cross_entropy(logits, labels)
    gold = logits.gather(1, labels.view(-1, 1))
    wrong_mask = torch.ones_like(logits, dtype=torch.bool)
    wrong_mask.scatter_(1, labels.view(-1, 1), False)
    wrong = logits.masked_select(wrong_mask).view(logits.shape[0], -1)
    if wrong.numel() == 0:
        return F.cross_entropy(logits, labels)
    if objective == "pairwise_logistic":
        return F.softplus(wrong - gold).mean()
    if objective == "margin_ranking":
        return F.relu(float(training_config.margin) - gold + wrong).mean()
    if objective == "contrastive_infonce":
        centered = logits - logits.mean(dim=1, keepdim=True)
        return F.cross_entropy(centered, labels)
    if objective == "ce_plus_pairwise":
        return F.cross_entropy(logits, labels) + 0.5 * F.softplus(wrong - gold).mean()
    if objective == "ce_plus_margin":
        return F.cross_entropy(logits, labels) + 0.5 * F.relu(float(training_config.margin) - gold + wrong).mean()
    raise ValueError(f"unknown ARC candidate objective: {objective}")


class ArcRuleCloneVerifier(nn.Module):
    def __init__(self, config: ArcModelConfig) -> None:
        super().__init__()
        self.config = config
        dim = _compatible_dim(config.model_dim, config.num_heads)
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
        self.rule_encoder = nn.TransformerEncoder(rule_layer, num_layers=int(config.rule_layers))
        self.candidate_encoder = nn.TransformerEncoder(candidate_layer, num_layers=int(config.candidate_layers))
        self.cross_blocks = nn.ModuleList(
            [CandidateRuleCrossAttentionBlock(dim, int(config.num_heads), int(config.ff_dim), float(config.dropout)) for _ in range(int(config.coordination_blocks))]
        )
        self.rule_to_candidate_blocks = nn.ModuleList(
            [CandidateRuleCrossAttentionBlock(dim, int(config.num_heads), int(config.ff_dim), float(config.dropout)) for _ in range(int(config.coordination_blocks))]
        )
        self.candidate_to_rule_projection = nn.Linear(dim, dim)
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
        modules = ("token_encoder", "rule_encoder", "candidate_encoder")
        return [(name, parameter) for name, parameter in self.named_parameters() if name.split(".", 1)[0] in modules]

    def coordinator_parameter_items(self) -> List[Tuple[str, nn.Parameter]]:
        modules = ("cross_blocks", "rule_to_candidate_blocks", "candidate_to_rule_projection", "score_head")
        return [(name, parameter) for name, parameter in self.named_parameters() if name.split(".", 1)[0] in modules]

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        return_attention: bool = False,
        hidden_state_shuffle: bool = False,
        shuffle_seed: int = 0,
    ) -> Dict[str, object]:
        rule_fields = batch["rule_fields"]
        rule_mask = batch["rule_mask"].bool()
        candidate_fields = batch["candidate_fields"]
        candidate_mask = batch["candidate_mask"].bool()
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
        attention_weights: List[torch.Tensor] = []
        flat_rule = rule_encoded.reshape(batch_size, views * rule_tokens, -1)
        flat_rule_mask = rule_mask.reshape(batch_size, views * rule_tokens)
        interaction = str(self.config.interaction_style)
        for block_index, block in enumerate(self.cross_blocks):
            rule_for_candidate = flat_rule.repeat_interleave(candidates, dim=0)
            rule_mask_for_candidate = flat_rule_mask.repeat_interleave(candidates, dim=0)
            if interaction == "candidate_guided_rule_induction":
                pooled_candidate = _masked_mean(cand_encoded, candidate_mask.reshape(batch_size * candidates, candidate_tokens))
                rule_for_candidate = rule_for_candidate + self.candidate_to_rule_projection(pooled_candidate).unsqueeze(1)
            elif interaction == "bidirectional_candidate_rule":
                reverse_block = self.rule_to_candidate_blocks[block_index]
                rule_for_candidate, _reverse_weights = reverse_block(
                    rule_for_candidate,
                    cand_encoded,
                    candidate_mask.reshape(batch_size * candidates, candidate_tokens),
                    return_attention=False,
                )
            cand_encoded, weights = block(
                cand_encoded,
                rule_for_candidate,
                rule_mask_for_candidate,
                return_attention=return_attention,
            )
            if weights is not None:
                attention_weights.append(weights.reshape(batch_size, candidates, *weights.shape[1:]))
        pooled = _masked_mean(cand_encoded, candidate_mask.reshape(batch_size * candidates, candidate_tokens))
        scores = self.score_head(pooled).reshape(batch_size, candidates)
        return {"logits": scores, "attention_weights": attention_weights}


class FieldTokenEncoder(nn.Module):
    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(size, model_dim) for size in FIELD_VOCABS])
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        fields = fields.long()
        out = torch.zeros((*fields.shape[:2], self.embeddings[0].embedding_dim), dtype=torch.float32, device=fields.device)
        for index, embedding in enumerate(self.embeddings):
            values = fields[..., index].clamp(min=0, max=embedding.num_embeddings - 1)
            out = out + embedding(values)
        return self.norm(out)


class CandidateRuleCrossAttentionBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(model_dim)
        self.norm_ff = nn.LayerNorm(model_dim)
        self.ff = nn.Sequential(nn.Linear(model_dim, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, model_dim))

    def forward(
        self,
        query: torch.Tensor,
        rule_tokens: torch.Tensor,
        rule_mask: torch.Tensor,
        return_attention: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        attended, weights = self.attn(
            query=self.norm_q(query),
            key=rule_tokens,
            value=rule_tokens,
            key_padding_mask=~rule_mask.bool(),
            need_weights=return_attention,
            average_attn_weights=False,
        )
        query = query + attended
        query = query + self.ff(self.norm_ff(query))
        return query, weights if return_attention else None


def collate_arc_batch(
    examples: Sequence[ArcVerificationExample],
    model_config: ArcModelConfig,
    device: str,
    view_names: Sequence[str] | None = None,
) -> Dict[str, torch.Tensor]:
    views = tuple(view_names or model_config.view_names)
    rule_rows = []
    rule_masks = []
    cand_rows = []
    cand_masks = []
    labels = []
    for example in examples:
        rule_fields, rule_mask = _rule_token_fields(example, views, model_config)
        candidate_fields, candidate_mask = _candidate_token_fields(example, model_config)
        rule_rows.append(rule_fields)
        rule_masks.append(rule_mask)
        cand_rows.append(candidate_fields)
        cand_masks.append(candidate_mask)
        labels.append(int(example.label))
    return {
        "rule_fields": torch.as_tensor(np.stack(rule_rows), dtype=torch.long, device=device),
        "rule_mask": torch.as_tensor(np.stack(rule_masks), dtype=torch.bool, device=device),
        "candidate_fields": torch.as_tensor(np.stack(cand_rows), dtype=torch.long, device=device),
        "candidate_mask": torch.as_tensor(np.stack(cand_masks), dtype=torch.bool, device=device),
        "labels": torch.as_tensor(labels, dtype=torch.long, device=device),
    }


def _rule_token_fields(
    example: ArcVerificationExample,
    view_names: Sequence[str],
    model_config: ArcModelConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    per_view: List[List[List[int]]] = []
    for slot_index, view in enumerate(view_names):
        canonical_view = "raw" if model_config.learned_slots else str(view)
        view_id = slot_index if model_config.learned_slots else VIEW_ID.get(str(view), slot_index)
        tokens = _summary_tokens(example.train_pairs, view_id)
        for pair_id, pair in enumerate(example.train_pairs):
            if _include_cell_tokens(model_config.representation_mode):
                tokens.extend(_grid_cell_tokens(pair.input, ROLE_TRAIN_INPUT, pair_id, view_id, canonical_view, pair.output))
                tokens.extend(_grid_cell_tokens(pair.output, ROLE_TRAIN_OUTPUT, pair_id, view_id, canonical_view, pair.input))
            if _shape(pair.input) == _shape(pair.output):
                if _include_diff_tokens(model_config.representation_mode):
                    tokens.extend(_diff_tokens(pair.input, pair.output, pair_id, view_id, canonical_view))
            if _include_histogram_tokens(model_config.representation_mode):
                tokens.extend(_histogram_tokens(pair.output, ROLE_SUMMARY, pair_id, view_id))
            if _include_bbox_tokens(model_config.representation_mode):
                tokens.extend(_bbox_tokens(pair.output, ROLE_SUMMARY, pair_id, view_id))
        per_view.append(_cap_tokens(tokens, int(model_config.max_rule_tokens)))
    fields = np.zeros((len(view_names), int(model_config.max_rule_tokens), len(FIELD_VOCABS)), dtype=np.int64)
    mask = np.zeros((len(view_names), int(model_config.max_rule_tokens)), dtype=bool)
    for view_axis, tokens in enumerate(per_view):
        if not tokens:
            tokens = [_token(0, 0, ROLE_SUMMARY, SUMMARY_ROWCOL, SUMMARY_ROWCOL, 1, 1, 0, view_axis, 0, 0, 0)]
        count = min(len(tokens), int(model_config.max_rule_tokens))
        fields[view_axis, :count] = np.asarray(tokens[:count], dtype=np.int64)
        mask[view_axis, :count] = True
    return fields, mask


def _candidate_token_fields(
    example: ArcVerificationExample,
    model_config: ArcModelConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    fields = np.zeros((len(example.candidates), int(model_config.max_candidate_tokens), len(FIELD_VOCABS)), dtype=np.int64)
    mask = np.zeros((len(example.candidates), int(model_config.max_candidate_tokens)), dtype=bool)
    for candidate_index, candidate in enumerate(example.candidates):
        tokens = _candidate_summary_tokens(example.test_input, candidate)
        if _include_cell_tokens(model_config.representation_mode):
            tokens.extend(_grid_cell_tokens(example.test_input, ROLE_TEST_INPUT, 0, 0, "raw", candidate))
            tokens.extend(_grid_cell_tokens(candidate, ROLE_CANDIDATE_OUTPUT, 0, 0, "raw", example.test_input))
        if _shape(example.test_input) == _shape(candidate):
            if _include_diff_tokens(model_config.representation_mode):
                tokens.extend(_diff_tokens(example.test_input, candidate, 0, 0, "diff", role=ROLE_CANDIDATE_DIFF))
        if _include_histogram_tokens(model_config.representation_mode):
            tokens.extend(_histogram_tokens(candidate, ROLE_SUMMARY, 0, 0))
        if _include_bbox_tokens(model_config.representation_mode):
            tokens.extend(_bbox_tokens(candidate, ROLE_SUMMARY, 0, 0))
        if _include_execution_evidence_tokens(model_config.representation_mode):
            tokens.extend(_execution_evidence_tokens(example, candidate_index, candidate))
        tokens = _cap_tokens(tokens, int(model_config.max_candidate_tokens))
        count = min(len(tokens), int(model_config.max_candidate_tokens))
        fields[candidate_index, :count] = np.asarray(tokens[:count], dtype=np.int64)
        mask[candidate_index, :count] = True
    return fields, mask


def _summary_tokens(train_pairs: Sequence[ArcPair], view_id: int) -> List[List[int]]:
    tokens: List[List[int]] = []
    for pair_id, pair in enumerate(train_pairs):
        inp = np.asarray(pair.input, dtype=np.int64)
        out = np.asarray(pair.output, dtype=np.int64)
        dominant = _dominant(out)
        palette = len(np.unique(out)) if out.size else 0
        changed = int(np.sum(inp != out)) if inp.shape == out.shape else min(31, out.size)
        h, w = _shape(pair.output)
        tokens.append(_token(dominant, min(11, palette), ROLE_SUMMARY, SUMMARY_ROWCOL, SUMMARY_ROWCOL, h, w, pair_id, view_id, 0, min(31, out.size), min(31, changed)))
    if not tokens:
        tokens.append(_token(0, 0, ROLE_SUMMARY, SUMMARY_ROWCOL, SUMMARY_ROWCOL, 1, 1, 0, view_id, 0, 0, 0))
    return tokens


def _candidate_summary_tokens(test_input: List[List[int]], candidate: List[List[int]]) -> List[List[int]]:
    cand = np.asarray(candidate, dtype=np.int64)
    test = np.asarray(test_input, dtype=np.int64)
    changed = int(np.sum(cand != test)) if cand.shape == test.shape else min(31, cand.size)
    h, w = _shape(candidate)
    return [_token(_dominant(cand), min(11, len(np.unique(cand)) if cand.size else 0), ROLE_SUMMARY, SUMMARY_ROWCOL, SUMMARY_ROWCOL, h, w, 0, 0, 0, min(31, cand.size), min(31, changed))]


def _include_cell_tokens(mode: str) -> bool:
    return str(mode) in {
        "cell",
        "object",
        "bbox",
        "cell_object",
        "hybrid_cell_object_diff",
        "hybrid_cell_object_diff_hist",
        "hybrid_all",
        "hybrid_execution",
        "hybrid_all_execution",
    }


def _include_diff_tokens(mode: str) -> bool:
    return str(mode) in {
        "diff",
        "cell_diff",
        "raw_diff",
        "hybrid_cell_object_diff",
        "hybrid_cell_object_diff_hist",
        "hybrid_all",
        "hybrid_execution",
        "hybrid_all_execution",
    }


def _include_histogram_tokens(mode: str) -> bool:
    return str(mode) in {"histogram", "color_histogram", "hybrid_cell_object_diff_hist", "hybrid_all", "hybrid_all_execution"}


def _include_bbox_tokens(mode: str) -> bool:
    return str(mode) in {"object", "bbox", "cell_object", "hybrid_all", "hybrid_all_execution"}


def _include_execution_evidence_tokens(mode: str) -> bool:
    return str(mode) in {
        "execution_trace",
        "mismatch_map",
        "program_trace",
        "hybrid_execution",
        "hybrid_all_execution",
    }


def _execution_evidence_tokens(
    example: ArcVerificationExample,
    candidate_index: int,
    candidate: Sequence[Sequence[int]],
) -> List[List[int]]:
    metadata = example.metadata or {}
    if metadata.get("candidate_visible_evidence_stripped"):
        return []

    train_scores = _metadata_float_list(metadata, "candidate_train_execution_scores")
    mismatch_counts = _metadata_float_list(metadata, "candidate_mismatch_counts")
    mismatch_ratios = _metadata_float_list(metadata, "candidate_mismatch_ratios")
    program_lengths = _metadata_float_list(metadata, "candidate_program_lengths")
    trace_depths = _metadata_float_list(metadata, "candidate_trace_depths")
    if not any((train_scores, mismatch_counts, mismatch_ratios, program_lengths, trace_depths)):
        return []

    arr = np.asarray(candidate, dtype=np.int64)
    h, w = _shape(candidate)
    train_score = _list_value(train_scores, candidate_index, 0.0)
    mismatch_count = _list_value(mismatch_counts, candidate_index, 31.0)
    mismatch_ratio = _list_value(mismatch_ratios, candidate_index, 1.0)
    program_length = _list_value(program_lengths, candidate_index, 0.0)
    trace_depth = _list_value(trace_depths, candidate_index, 0.0)
    palette = len(np.unique(arr)) if arr.size else 0
    dominant = _dominant(arr)
    score_bucket = int(round(max(0.0, min(1.0, train_score)) * 9.0))
    mismatch_bucket = int(round(max(0.0, min(1.0, mismatch_ratio)) * 31.0))
    return [
        _token(
            score_bucket,
            min(11, int(palette)),
            ROLE_EXECUTION_EVIDENCE,
            SUMMARY_ROWCOL,
            SUMMARY_ROWCOL,
            h,
            w,
            0,
            VIEW_ID.get("candidate_execution", 0),
            int(train_score >= 0.999),
            min(31, int(round(program_length))),
            min(31, int(round(trace_depth))),
        ),
        _token(
            dominant,
            mismatch_bucket,
            ROLE_EXECUTION_EVIDENCE,
            SUMMARY_ROWCOL,
            SUMMARY_ROWCOL,
            h,
            w,
            0,
            VIEW_ID.get("candidate_execution", 0),
            int(mismatch_ratio <= 0.001),
            min(31, int(round(mismatch_count))),
            mismatch_bucket,
        ),
    ]


def _metadata_float_list(metadata: Dict[str, object], key: str) -> List[float]:
    values = metadata.get(key, [])
    if not isinstance(values, (list, tuple)):
        return []
    out = []
    for value in values:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def _list_value(values: Sequence[float], index: int, default: float) -> float:
    if 0 <= int(index) < len(values):
        return float(values[int(index)])
    return float(default)


def _histogram_tokens(grid: List[List[int]], role: int, pair_id: int, view_id: int) -> List[List[int]]:
    arr = np.asarray(grid, dtype=np.int64)
    if arr.size == 0:
        return []
    h, w = arr.shape
    tokens = []
    values, counts = np.unique(arr, return_counts=True)
    for color, count in zip(values, counts):
        tokens.append(_token(int(color), int(color), role, SUMMARY_ROWCOL, SUMMARY_ROWCOL, h, w, pair_id, view_id, 0, min(31, int(count)), min(31, int(count))))
    return tokens


def _bbox_tokens(grid: List[List[int]], role: int, pair_id: int, view_id: int) -> List[List[int]]:
    arr = np.asarray(grid, dtype=np.int64)
    if arr.size == 0:
        return []
    h, w = arr.shape
    bg = _dominant(arr)
    tokens = []
    for color in np.unique(arr):
        color = int(color)
        if color == bg:
            continue
        positions = np.argwhere(arr == color)
        if len(positions) == 0:
            continue
        r0, c0 = positions.min(axis=0)
        r1, c1 = positions.max(axis=0)
        area = int((int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1))
        tokens.append(_token(color, color, role, int(r0), int(c0), h, w, pair_id, view_id, 0, min(31, area), min(31, len(positions))))
    return tokens


def _grid_cell_tokens(
    grid: List[List[int]],
    role: int,
    pair_id: int,
    view_id: int,
    view: str,
    aux_grid: List[List[int]] | None = None,
) -> List[List[int]]:
    arr = np.asarray(grid, dtype=np.int64)
    if arr.size == 0:
        return []
    aux = np.asarray(aux_grid, dtype=np.int64) if aux_grid is not None and _shape(aux_grid) == _shape(grid) else None
    component_sizes = _component_size_map(arr)
    color_counts = {int(color): int(count) for color, count in zip(*np.unique(arr, return_counts=True))}
    h, w = arr.shape
    tokens: List[Tuple[int, List[int]]] = []
    bg = _dominant(arr)
    for row in range(h):
        for col in range(w):
            color = int(arr[row, col])
            aux_color = int(aux[row, col]) if aux is not None else color
            delta = 1 if aux is not None and color != aux_color else 0
            obj_bucket = min(31, int(component_sizes[row, col]))
            count_bucket = min(31, color_counts.get(color, 1))
            priority = _cell_priority(view, row, col, h, w, color, bg, delta, obj_bucket)
            tokens.append((priority, _token(color, aux_color, role, row, col, h, w, pair_id, view_id, delta, obj_bucket, count_bucket)))
    tokens.sort(key=lambda item: (-item[0], _token_sort_key(item[1])))
    return [token for _priority, token in tokens]


def _diff_tokens(
    input_grid: List[List[int]],
    output_grid: List[List[int]],
    pair_id: int,
    view_id: int,
    view: str,
    role: int = ROLE_TRAIN_DIFF,
) -> List[List[int]]:
    inp = np.asarray(input_grid, dtype=np.int64)
    out = np.asarray(output_grid, dtype=np.int64)
    if inp.shape != out.shape or out.size == 0:
        return []
    h, w = out.shape
    tokens = []
    for row, col in np.argwhere(inp != out):
        color = int(out[row, col])
        aux = int(inp[row, col])
        tokens.append(_token(color, aux, role, int(row), int(col), h, w, pair_id, view_id, 1, 1, min(31, len(tokens) + 1)))
    if not tokens and view == "diff":
        tokens.append(_token(_dominant(out), _dominant(inp), role, SUMMARY_ROWCOL, SUMMARY_ROWCOL, h, w, pair_id, view_id, 0, 0, 0))
    return tokens


def _token(
    color: int,
    aux_color: int,
    role: int,
    row: int,
    col: int,
    height: int,
    width: int,
    pair_id: int,
    view_id: int,
    delta: int,
    object_bucket: int,
    count_bucket: int,
) -> List[int]:
    values = [
        int(color),
        int(aux_color),
        int(role),
        int(row),
        int(col),
        int(height),
        int(width),
        int(pair_id),
        int(view_id),
        int(delta),
        int(object_bucket),
        int(count_bucket),
    ]
    return [max(0, min(FIELD_VOCABS[index] - 1, value)) for index, value in enumerate(values)]


def _cap_tokens(tokens: List[List[int]], limit: int) -> List[List[int]]:
    if len(tokens) <= limit:
        return tokens
    summary = [token for token in tokens if token[2] == ROLE_SUMMARY]
    rest = [token for token in tokens if token[2] != ROLE_SUMMARY]
    rest.sort(key=_token_sort_key)
    keep = max(0, int(limit) - len(summary))
    return (summary + rest[:keep])[:limit]


def _cell_priority(view: str, row: int, col: int, h: int, w: int, color: int, bg: int, delta: int, object_bucket: int) -> int:
    priority = 0
    if color != bg:
        priority += 5
    if delta:
        priority += 8
    if row in {0, h - 1} or col in {0, w - 1}:
        priority += 2
    if view == "diff":
        priority += 10 if delta else -4
    elif view == "object":
        priority += 6 if object_bucket > 0 else -3
    elif view == "color":
        priority += 4 if color != bg else 0
    elif view == "geometry":
        priority += 4 if row in {0, h - 1} or col in {0, w - 1} else 0
    elif view == "symmetry":
        priority += 3 if (row < h // 2 or col < w // 2) else 0
    elif view == "counting":
        priority += min(6, object_bucket)
    return priority


def _token_sort_key(token: Sequence[int]) -> Tuple[int, int, int, int]:
    return (int(token[7]), int(token[3]), int(token[4]), int(token[0]))


def _component_size_map(grid: np.ndarray) -> np.ndarray:
    if grid.size == 0:
        return np.zeros_like(grid)
    bg = _dominant(grid)
    sizes = np.zeros(grid.shape, dtype=np.int64)
    seen = np.zeros(grid.shape, dtype=bool)
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            if seen[row, col] or int(grid[row, col]) == bg:
                continue
            color = int(grid[row, col])
            stack = [(row, col)]
            seen[row, col] = True
            component: List[Tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                component.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            for rr, cc in component:
                sizes[rr, cc] = len(component)
    return sizes


def _run_controls(
    model: ArcRuleCloneVerifier,
    model_config: ArcModelConfig,
    examples: Sequence[ArcVerificationExample],
    batch_size: int,
    device: str,
    seed: int,
) -> Dict[str, object]:
    labels = np.asarray([example.label for example in examples], dtype=np.int64)
    controls: Dict[str, object] = {}
    for name in (
        "candidate_only",
        "examples_only_no_candidates",
        "train_pair_shuffle",
        "cross_task_train_pair_shuffle",
        "candidate_evidence_mismatch",
        "candidate_order_shuffle_with_gold_remap",
        "color_permutation",
        "coordinate_shift_padding",
        "randomized_labels",
    ):
        controlled = apply_arc_control(examples, name, seed + 31_000 + len(controls))
        controlled_labels = np.asarray([example.label for example in controlled], dtype=np.int64)
        logits = predict_logits(model, controlled, model_config, batch_size, device, condition="none", seed=seed)
        controls[name] = _metric_block(logits, controlled_labels, controlled)
    logits_role = predict_logits(model, examples, model_config, batch_size, device, condition="physical_role_order_shuffle", seed=seed + 44_000)
    controls["physical_role_order_shuffle"] = _metric_block(logits_role, labels, examples)
    logits_hidden = predict_logits(model, examples, model_config, batch_size, device, condition="hidden_state_shuffle", seed=seed + 55_000)
    controls["hidden_state_shuffle"] = _metric_block(logits_hidden, labels, examples)
    role_mask = {}
    for view in model_config.view_names:
        logits = predict_logits(model, examples, model_config, batch_size, device, view_mask=view)
        role_mask[str(view)] = _metric_block(logits, labels, examples)
    controls["role_view_masking"] = role_mask
    base_logits = predict_logits(model, examples, model_config, batch_size, device)
    shuffled = controls["candidate_order_shuffle_with_gold_remap"]["top1"]
    controls["candidate_order_invariance_delta"] = abs(float(_top1(base_logits, labels)) - float(shuffled))
    controls["role_order_invariance_delta"] = abs(float(_top1(base_logits, labels)) - float(controls["physical_role_order_shuffle"]["top1"]))
    controls["color_renaming_diagnostic_delta"] = abs(float(_top1(base_logits, labels)) - float(controls["color_permutation"]["top1"]))
    controls["grid_size_control_pass"] = all(len({tuple(_shape(candidate)) for candidate in example.candidates}) == 1 for example in examples)
    controls["candidate_source_metadata_only_accuracy"] = candidate_metadata_only_accuracy(examples)
    return controls


def _phase0_summary(splits: Dict[str, Sequence[ArcVerificationExample]], leakage_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "dataset_summary": dataset_summary(splits),
        "leakage_audit_pass": all(bool(row.get("pass")) for row in leakage_rows),
        "candidate_metadata_only_accuracy_dev": candidate_metadata_only_accuracy(splits.get("dev", [])),
        "candidate_metadata_only_accuracy_test": candidate_metadata_only_accuracy(splits.get("test", [])),
        "candidate_generator_recall_dev": dataset_summary({"dev": splits.get("dev", [])}).get("candidate_generator_recall", 0.0),
    }


def _baseline_block(
    train_examples: Sequence[ArcVerificationExample],
    dev_examples: Sequence[ArcVerificationExample],
    test_examples: Sequence[ArcVerificationExample],
) -> Dict[str, object]:
    del train_examples, dev_examples
    labels = np.asarray([example.label for example in test_examples], dtype=np.int64)
    return {
        "random": 1.0 / max(1, len(test_examples[0].candidates) if test_examples else 1),
        "candidate_order_index0": _accuracy(np.zeros(len(test_examples), dtype=np.int64), labels),
        "candidate_metadata_source_only": candidate_metadata_only_accuracy(test_examples),
        "candidate_grid_only_stat_heuristic": _accuracy(_candidate_stat_predictions(test_examples), labels),
        "nearest_train_output": _accuracy(_nearest_train_output_predictions(test_examples), labels),
        "heuristic_changed_cell_count": _accuracy(_changed_cell_count_predictions(test_examples), labels),
        "oracle_candidate_present": 1.0 if all(any(np.array_equal(np.asarray(c), np.asarray(e.gold_output)) for c in e.candidates) for e in test_examples) else 0.0,
    }


def _candidate_stat_predictions(examples: Sequence[ArcVerificationExample]) -> np.ndarray:
    preds = []
    for example in examples:
        scores = []
        train_palette = _mean([len(np.unique(np.asarray(pair.output))) for pair in example.train_pairs]) if example.train_pairs else 0.0
        for candidate in example.candidates:
            arr = np.asarray(candidate, dtype=np.int64)
            palette = len(np.unique(arr)) if arr.size else 0
            entropy = _entropy(arr)
            scores.append(abs(palette - train_palette) + 0.05 * entropy)
        preds.append(int(np.argmin(scores)) if scores else 0)
    return np.asarray(preds, dtype=np.int64)


def _nearest_train_output_predictions(examples: Sequence[ArcVerificationExample]) -> np.ndarray:
    preds = []
    for example in examples:
        train_outputs = [np.asarray(pair.output, dtype=np.int64) for pair in example.train_pairs]
        scores = []
        for candidate in example.candidates:
            cand = np.asarray(candidate, dtype=np.int64)
            distances = [_resized_hamming(cand, train) for train in train_outputs] or [1.0]
            scores.append(min(distances))
        preds.append(int(np.argmin(scores)) if scores else 0)
    return np.asarray(preds, dtype=np.int64)


def _changed_cell_count_predictions(examples: Sequence[ArcVerificationExample]) -> np.ndarray:
    preds = []
    for example in examples:
        train_ratios = []
        for pair in example.train_pairs:
            inp = np.asarray(pair.input, dtype=np.int64)
            out = np.asarray(pair.output, dtype=np.int64)
            if inp.shape == out.shape and out.size:
                train_ratios.append(float(np.mean(inp != out)))
        target = _mean(train_ratios) if train_ratios else 0.5
        test = np.asarray(example.test_input, dtype=np.int64)
        scores = []
        for candidate in example.candidates:
            cand = np.asarray(candidate, dtype=np.int64)
            ratio = float(np.mean(test != cand)) if test.shape == cand.shape and cand.size else 1.0
            scores.append(abs(ratio - target))
        preds.append(int(np.argmin(scores)) if scores else 0)
    return np.asarray(preds, dtype=np.int64)


def _metric_block(logits: np.ndarray, labels: np.ndarray, examples: Sequence[ArcVerificationExample]) -> Dict[str, object]:
    if logits.size == 0:
        return {"top1": 0.0, "top2": 0.0, "top3": 0.0, "mrr": 0.0, "mean_gold_rank": 0.0, "gold_ranks": []}
    order = np.argsort(-logits, axis=1)
    ranks = []
    for row, label in zip(order, labels):
        ranks.append(int(np.where(row == int(label))[0][0]) + 1)
    predictions = order[:, 0]
    margins = []
    for row_index, label in enumerate(labels):
        gold_logit = float(logits[row_index, int(label)])
        wrong = np.delete(logits[row_index], int(label))
        best_wrong = float(np.max(wrong)) if wrong.size else gold_logit
        margins.append(gold_logit - best_wrong)
    return {
        "top1": _accuracy(predictions, labels),
        "top2": float(np.mean([rank <= 2 for rank in ranks])) if ranks else 0.0,
        "top3": float(np.mean([rank <= 3 for rank in ranks])) if ranks else 0.0,
        "mrr": float(np.mean([1.0 / rank for rank in ranks])) if ranks else 0.0,
        "mean_gold_rank": float(np.mean(ranks)) if ranks else 0.0,
        "gold_ranks": ranks,
        "mean_gold_vs_best_wrong_logit_margin": float(np.mean(margins)) if margins else 0.0,
        "min_gold_vs_best_wrong_logit_margin": float(np.min(margins)) if margins else 0.0,
        "by_grid_size": _accuracy_by_key(predictions, labels, [f"{_shape(e.gold_output)[0]}x{_shape(e.gold_output)[1]}" for e in examples]),
        "by_train_examples": _accuracy_by_key(predictions, labels, [str(len(e.train_pairs)) for e in examples]),
        "by_palette_size": _accuracy_by_key(predictions, labels, [str(len(np.unique(np.asarray(e.gold_output)))) for e in examples]),
        "by_candidate_count": _accuracy_by_key(predictions, labels, [str(len(e.candidates)) for e in examples]),
        "by_transformation_type": _accuracy_by_key(predictions, labels, [str(e.metadata.get("transformation_type", "unknown")) for e in examples]),
        "negative_type_confusion": _negative_confusion(predictions, examples),
        "verifier_failure_rate": float(np.mean(predictions != labels)) if len(labels) else 0.0,
    }


def _attention_summaries(
    model: ArcRuleCloneVerifier,
    model_config: ArcModelConfig,
    examples: Sequence[ArcVerificationExample],
    batch_size: int,
    device: str,
    seed: int,
    limit: int,
) -> List[Dict[str, object]]:
    subset = list(examples[:limit])
    _logits, rows = predict_logits(model, subset, model_config, batch_size, device, seed=seed, return_attention=True)
    for row in rows:
        row["seed"] = seed
        row["benchmark"] = BENCHMARK
    return rows


def _summarize_attention_batch(
    output: Dict[str, object],
    batch: Dict[str, torch.Tensor],
    examples: Sequence[ArcVerificationExample],
    view_names: Sequence[str],
) -> List[Dict[str, object]]:
    weights_list = output.get("attention_weights", [])
    if not weights_list:
        return []
    logits = output["logits"].detach().cpu().numpy()
    preds = np.argmax(logits, axis=1)
    rows = []
    view_count = len(view_names)
    rule_tokens = batch["rule_fields"].shape[2]
    rule_fields = batch["rule_fields"].detach().cpu().numpy()
    for example_index, example in enumerate(examples):
        block_rows = []
        for block_index, weights in enumerate(weights_list):
            # weights: batch, candidates, heads, query_tokens, view_count * rule_tokens
            w = weights[example_index].detach().float().cpu().numpy()
            by_view = w.reshape(w.shape[0], w.shape[1], w.shape[2], view_count, rule_tokens).mean(axis=(0, 1, 2, 4))
            candidate_mass = w.sum(axis=(1, 2, 3)) / max(1, w.shape[1] * w.shape[2])
            flat = w.mean(axis=(0, 1, 2)).reshape(view_count, rule_tokens)
            top = []
            for flat_index in np.argsort(-flat.reshape(-1))[:5]:
                view_axis = int(flat_index // rule_tokens)
                token_axis = int(flat_index % rule_tokens)
                fields = rule_fields[example_index, view_axis, token_axis].tolist()
                top.append(
                    {
                        "view": str(view_names[view_axis]),
                        "role": int(fields[2]),
                        "row": int(fields[3]),
                        "col": int(fields[4]),
                        "color": int(fields[0]),
                        "mass": float(flat[view_axis, token_axis]),
                    }
                )
            entropy = float(-np.sum(flat.reshape(-1) * np.log(np.clip(flat.reshape(-1), 1e-9, 1.0))))
            block_rows.append(
                    {
                        "block": int(block_index),
                        "attention_mass_per_rule_clone": {str(view_names[i]): float(by_view[i]) for i in range(view_count)},
                        "attention_mass_per_candidate": {str(i): float(candidate_mass[i]) for i in range(w.shape[0])},
                        "attention_entropy": entropy,
                        "top_attended_tokens": top,
                    }
            )
        rows.append(
            {
                "type": "arc1_attention_summary",
                "task_id": example.task_id,
                "example_id": example.id,
                "predicted_candidate": int(preds[example_index]),
                "gold_candidate": int(example.label),
                "candidate_probabilities": _softmax_np(logits[example_index]).tolist(),
                "blocks": block_rows,
            }
        )
    return rows


def _error_cases(
    examples: Sequence[ArcVerificationExample],
    train_logits: np.ndarray,
    frozen_logits: np.ndarray,
    seed: int,
    variant: str,
    controls: Dict[str, object],
    limit: int,
) -> List[Dict[str, object]]:
    labels = np.asarray([example.label for example in examples], dtype=np.int64)
    train_pred = np.argmax(train_logits, axis=1) if train_logits.size else np.zeros(len(examples), dtype=np.int64)
    frozen_pred = np.argmax(frozen_logits, axis=1) if frozen_logits.size else np.zeros(len(examples), dtype=np.int64)
    cand_only_pred = _candidate_stat_predictions(examples)
    heuristic_pred = _changed_cell_count_predictions(examples)
    buckets = {
        "trainable_right_frozen_wrong": [],
        "frozen_right_trainable_wrong": [],
        "candidate_only_right_trainable_wrong": [],
        "all_models_wrong": [],
        "stacked_variant_fixes_baseline": [],
        "baseline_right_rule_clone_wrong": [],
    }
    for index, example in enumerate(examples):
        tr = int(train_pred[index])
        fr = int(frozen_pred[index])
        co = int(cand_only_pred[index])
        he = int(heuristic_pred[index])
        label = int(labels[index])
        if tr == label and fr != label:
            buckets["trainable_right_frozen_wrong"].append((index, tr, fr, co))
        if fr == label and tr != label:
            buckets["frozen_right_trainable_wrong"].append((index, tr, fr, co))
        if co == label and tr != label:
            buckets["candidate_only_right_trainable_wrong"].append((index, tr, fr, co))
        if tr != label and fr != label and co != label and he != label:
            buckets["all_models_wrong"].append((index, tr, fr, co))
        if he == label and tr != label:
            buckets["baseline_right_rule_clone_wrong"].append((index, tr, fr, co))
    rows = []
    for category, cases in buckets.items():
        for index, tr, _fr, _co in cases[: max(1, limit // max(1, len(buckets)))]:
            example = examples[index]
            probs = _softmax_np(train_logits[index]).tolist()
            rank = int(np.where(np.argsort(-train_logits[index]) == int(example.label))[0][0]) + 1
            rows.append(
                {
                    "type": "arc1_error_case",
                    "category": category,
                    "variant": variant,
                    "seed": seed,
                    "task_id": example.task_id,
                    "example_id": example.id,
                    "train_grids": [{"input": pair.input, "output": pair.output} for pair in example.train_pairs],
                    "test_input": example.test_input,
                    "gold_candidate": example.gold_output,
                    "predicted_candidate": example.candidates[tr],
                    "predicted_candidate_index": tr,
                    "gold_candidate_index": int(example.label),
                    "candidate_probabilities": probs,
                    "gold_rank": rank,
                    "negative_type_labels": list(example.negative_types),
                    "rule_clone_attention_summary": "see arc1_latent_rule_clones_attention_summaries.jsonl",
                    "ablation_sensitivity": controls.get("role_view_masking", {}),
                    "control_status": controls.get("control_pass", {}),
                }
            )
    return rows


def _overall_summary(
    rows: Sequence[Dict[str, object]],
    controls: Sequence[Dict[str, object]],
    phase0_rows: Sequence[Dict[str, object]],
    dataset_config: ArcVerificationDatasetConfig,
) -> Dict[str, object]:
    del phase0_rows
    by_variant: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    variant_summaries = []
    for variant, variant_rows in by_variant.items():
        train = [float(row["trainable"]["top1"]) for row in variant_rows]
        frozen = [float(row["frozen"]["top1"]) for row in variant_rows]
        deltas = [float(row["delta_top1_trainable_minus_frozen"]) for row in variant_rows]
        variant_controls = [row for row in controls if row.get("variant") == variant]
        variant_summaries.append(
            {
                "variant": variant,
                "seeds": [int(row["seed"]) for row in variant_rows],
                "mean_trainable_top1": _mean(train),
                "std_trainable_top1": _std(train),
                "min_trainable_top1": min(train) if train else 0.0,
                "max_trainable_top1": max(train) if train else 0.0,
                "mean_frozen_top1": _mean(frozen),
                "mean_delta": _mean(deltas),
                "bootstrap_95ci_delta": _bootstrap_ci(deltas),
                "seeds_trainable_beats_frozen": int(sum(delta > 0.0 for delta in deltas)),
                "weak_seed_count": int(sum(value <= (1.0 / max(1, dataset_config.num_candidates) + 0.05) for value in train)),
                "worst_seed_accuracy": min(train) if train else 0.0,
                "controls_pass_all": all(bool(row.get("control_pass", {}).get("overall", False)) for row in variant_controls),
            }
        )
    variant_summaries.sort(key=lambda row: float(row["mean_delta"]), reverse=True)
    best = variant_summaries[0] if variant_summaries else {}
    success_gates = _success_gates(best, controls, rows, dataset_config)
    return {
        "variant_summaries": variant_summaries,
        "best_variant_by_delta": best.get("variant"),
        "candidate_count": dataset_config.num_candidates,
        "random_baseline": 1.0 / max(1, dataset_config.num_candidates),
        "final_validation_run": False,
        "success_gates": success_gates,
        "allowed_claim": "No ARC-AGI-2 solving claim. This run only exercises controlled candidate-output verification with known output size.",
    }


def _success_gates(
    best: Dict[str, object],
    controls: Sequence[Dict[str, object]],
    rows: Sequence[Dict[str, object]],
    dataset_config: ArcVerificationDatasetConfig,
) -> Dict[str, object]:
    best_variant = best.get("variant")
    best_controls = [row for row in controls if row.get("variant") == best_variant]
    best_rows = [row for row in rows if row.get("variant") == best_variant]
    frozen_ok = all(
        bool(row.get("frozen_audit", {}).get("frozen_shared_model_zero_grad"))
        and bool(row.get("frozen_audit", {}).get("frozen_shared_model_zero_delta"))
        for row in best_rows
    )
    trainable_ok = all(bool(row.get("trainable_audit", {}).get("trainable_shared_model_changed")) for row in best_rows)
    return {
        "trainable_beats_frozen_on_at_least_8_of_10_final_seeds": False,
        "mean_trainable_frozen_delta_at_least_0_20": float(best.get("mean_delta", 0.0)) >= 0.20,
        "bootstrap_95ci_lower_gt_0_05": float(best.get("bootstrap_95ci_delta", [0.0, 0.0])[0]) > 0.05,
        "candidate_only_near_chance": all(
            float(row.get("candidate_only", {}).get("top1", 1.0)) <= 1.0 / max(1, dataset_config.num_candidates) + 0.10 for row in best_controls
        ),
        "candidate_metadata_source_only_near_chance": all(
            float(row.get("candidate_source_metadata_only_accuracy", 1.0)) <= 1.0 / max(1, dataset_config.num_candidates) + 0.10 for row in best_controls
        ),
        "train_pair_shuffle_collapses": all(
            float(row.get("train_pair_shuffle", {}).get("top1", 1.0)) <= 1.0 / max(1, dataset_config.num_candidates) + 0.15 for row in best_controls
        ),
        "candidate_evidence_mismatch_collapses": all(
            float(row.get("candidate_evidence_mismatch", {}).get("top1", 1.0)) <= 1.0 / max(1, dataset_config.num_candidates) + 0.15
            for row in best_controls
        ),
        "candidate_order_remap_invariance_passes": all(float(row.get("candidate_order_invariance_delta", 1.0)) <= 0.08 for row in best_controls),
        "role_order_invariance_passes": all(float(row.get("role_order_invariance_delta", 1.0)) <= 0.08 for row in best_controls),
        "frozen_comparator_remains_frozen": frozen_ok,
        "trainable_shared_model_receives_gradients_and_changes": trainable_ok,
        "final_validation_required_for_primary_success": True,
    }


def _controls_pass(control_row: Dict[str, object], clean_top1: float, candidate_count: int) -> Dict[str, bool]:
    chance = 1.0 / max(1, int(candidate_count))
    pass_map = {
        "candidate_only": float(control_row.get("candidate_only", {}).get("top1", 1.0)) <= chance + 0.10,
        "examples_only_no_candidates": float(control_row.get("examples_only_no_candidates", {}).get("top1", 1.0)) <= chance + 0.10,
        "train_pair_shuffle": float(control_row.get("train_pair_shuffle", {}).get("top1", 1.0)) <= chance + 0.15,
        "cross_task_train_pair_shuffle": float(control_row.get("cross_task_train_pair_shuffle", {}).get("top1", 1.0)) <= chance + 0.15,
        "candidate_evidence_mismatch": float(control_row.get("candidate_evidence_mismatch", {}).get("top1", 1.0)) <= chance + 0.15,
        "candidate_order_shuffle_with_gold_remap": float(control_row.get("candidate_order_invariance_delta", 1.0)) <= 0.08,
        "physical_role_order_shuffle": float(control_row.get("role_order_invariance_delta", 1.0)) <= 0.08,
        "grid_size_control": bool(control_row.get("grid_size_control_pass", False)),
        "candidate_source_audit": float(control_row.get("candidate_source_metadata_only_accuracy", 1.0)) <= chance + 0.10,
        "hidden_state_shuffle": float(control_row.get("hidden_state_shuffle", {}).get("top1", 1.0)) <= max(chance + 0.20, clean_top1 - 0.05),
        "randomized_labels": float(control_row.get("randomized_labels", {}).get("top1", 1.0)) <= chance + 0.15,
    }
    pass_map["overall"] = all(pass_map.values())
    return pass_map


def _write_outputs(
    result: Dict[str, object],
    output_path: Path,
    controls_path: Path,
    leakage_path: Path,
    error_path: Path,
    attention_path: Path,
    compute_path: Path,
    report_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    controls_path.parent.mkdir(parents=True, exist_ok=True)
    leakage_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)
    attention_path.parent.mkdir(parents=True, exist_ok=True)
    compute_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_without_large_logs(result), indent=2, sort_keys=True), encoding="utf-8")
    controls_path.write_text(json.dumps({"controls": result["controls"], "summary": result["summary"].get("success_gates", {})}, indent=2, sort_keys=True), encoding="utf-8")
    leakage_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in result["leakage_audit"]) + "\n", encoding="utf-8")
    error_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in result["error_cases"]) + ("\n" if result["error_cases"] else ""), encoding="utf-8")
    attention_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in result["attention_summaries"]) + ("\n" if result["attention_summaries"] else ""), encoding="utf-8")
    compute_path.write_text(json.dumps({"compute_metrics": result["compute_metrics"]}, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(_render_report(result), encoding="utf-8")


def _save_arc_checkpoints(
    checkpoint_dir: Path,
    variant: str,
    seed: int,
    trainable: ArcFitResult,
    frozen: ArcFitResult,
    dataset_config: ArcVerificationDatasetConfig,
    training_config: ArcTrainingConfig,
) -> Dict[str, str]:
    root = checkpoint_dir / str(variant) / f"seed_{int(seed)}"
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "trainable": root / "trainable.pt",
        "frozen": root / "frozen.pt",
    }
    for name, fit in (("trainable", trainable), ("frozen", frozen)):
        torch.save(
            {
                "benchmark": BENCHMARK,
                "variant": variant,
                "seed": int(seed),
                "method": fit.method,
                "model_config": asdict(fit.config),
                "dataset_config": asdict(dataset_config),
                "training_config": asdict(training_config),
                "trainable_shared": bool(fit.trainable_shared),
                "audit": fit.audit,
                "history": fit.history,
                "model_state": {key: value.detach().cpu() for key, value in fit.model.state_dict().items()},
            },
            paths[name],
        )
    return {key: str(path) for key, path in paths.items()}


def _without_large_logs(result: Dict[str, object]) -> Dict[str, object]:
    return {key: value for key, value in result.items() if key not in {"leakage_audit", "error_cases", "attention_summaries", "compute_metrics"}}


def _render_report(result: Dict[str, object]) -> str:
    best_variant = result.get("summary", {}).get("best_variant_by_delta")
    best_rows = [row for row in result.get("rows", []) if row.get("variant") == best_variant]
    best_trainable = _mean([float(row["trainable"]["top1"]) for row in best_rows])
    best_frozen = _mean([float(row["frozen"]["top1"]) for row in best_rows])
    best_candidate_stat = _mean([float(row["baselines"].get("candidate_grid_only_stat_heuristic", 0.0)) for row in best_rows])
    best_changed_count = _mean([float(row["baselines"].get("heuristic_changed_cell_count", 0.0)) for row in best_rows])
    best_random = _mean([float(row["baselines"].get("random", 0.0)) for row in best_rows])
    above_baselines = bool(best_trainable > max(best_candidate_stat, best_changed_count, best_random))
    lines = [
        "# Stage ARC-1 Latent Rule-Clone ARC-AGI-2 Verifier/Ranker",
        "",
        "## Scope",
        "",
        "- Task: select the correct output grid from correct-size candidates with exactly one gold candidate.",
        "- This is not a full ARC solver, does not predict output size, and does not generate grids.",
        "- Architecture anchor: shared-weight rule clones plus candidate-token direct cross-attention, with an exact frozen same-architecture comparator.",
        f"- Final validation run: `{bool(result['summary'].get('final_validation_run', False))}`.",
        "",
        "## Phase 0 Data Smoke",
        "",
    ]
    for row in result.get("phase0", []):
        ds = row.get("dataset_summary", {})
        lines.extend(
            [
                f"- Seed `{row['seed']}` examples: `{ds.get('num_examples', {})}`.",
                f"- Candidate generator recall: `{ds.get('candidate_generator_recall', 0.0)}`.",
                f"- Leakage audit pass: `{row.get('leakage_audit_pass')}`.",
                f"- Candidate metadata-only dev accuracy: `{float(row.get('candidate_metadata_only_accuracy_dev', 0.0)):.4f}`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| variant | seeds | trainable | frozen | delta | top2 | top3 | MRR | controls |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in result.get("summary", {}).get("variant_summaries", []):
        variant_rows = [r for r in result.get("rows", []) if r.get("variant") == row.get("variant")]
        top2 = _mean([float(r["trainable"]["top2"]) for r in variant_rows])
        top3 = _mean([float(r["trainable"]["top3"]) for r in variant_rows])
        mrr = _mean([float(r["trainable"]["mrr"]) for r in variant_rows])
        lines.append(
            f"| {row['variant']} | {len(row.get('seeds', []))} | {float(row['mean_trainable_top1']):.4f} | {float(row['mean_frozen_top1']):.4f} | {float(row['mean_delta']):.4f} | {top2:.4f} | {top3:.4f} | {mrr:.4f} | `{bool(row['controls_pass_all'])}` |"
        )
    gates = result.get("summary", {}).get("success_gates", {})
    lines.extend(
        [
            "",
            "## Controls",
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
            "## Required Questions",
            "",
            "1. Can latent rule clones infer enough from ARC examples to rank candidate outputs?",
            f"   - Current smoke answer: best variant `{result.get('summary', {}).get('best_variant_by_delta')}` has mean trainable-frozen delta `{_best_value(result, 'mean_delta'):.4f}`. Treat this as implementation smoke unless a full Phase 2/3 run is launched.",
            "2. Does trainable shared-weight coordination beat exact frozen same-architecture?",
            f"   - Smoke result: trainable `{best_trainable:.4f}`, frozen `{best_frozen:.4f}`; +0.20 mean-delta gate `{gates.get('mean_trainable_frozen_delta_at_least_0_20')}`; frozen audit gate `{gates.get('frozen_comparator_remains_frozen')}`.",
            "3. Are results above candidate-only and heuristic baselines?",
            f"   - Smoke answer: `{above_baselines}`. Best trainable `{best_trainable:.4f}` versus random `{best_random:.4f}`, candidate-stat `{best_candidate_stat:.4f}`, and changed-cell heuristic `{best_changed_count:.4f}`.",
            "4. Do controls show the model is using examples rather than candidate artifacts?",
            f"   - Control status is `{gates}`. Failed gates block any claim.",
            "5. Do different rule clones specialize?",
            "   - Attention mass and view ablation summaries are saved; specialization is diagnostic-only in this smoke run.",
            "6. Does stacking help ARC verification?",
            "   - The implementation supports `coordination_blocks`; the default Stage ARC-1 smoke only runs A/B/C with one block.",
            "7. Does multi-avenue rule cloning help?",
            "   - Multi-avenue cloning is represented by configurable repeated/expanded view lists, but no medium/final avenue search is claimed here.",
            "8. Are failures mostly due to hard negatives, weak rule inference, or clone collapse?",
            "   - Error cases and hard-negative confusion are logged. Full diagnosis requires a larger Phase 2/3 run.",
            "9. Is this worth extending into candidate generation/refinement later?",
            "   - Only if Phase 2/3 gates pass: trainable beats frozen, candidate-only remains near chance, mismatch/shuffle controls collapse, and clone ablations show multi-view contribution.",
            "",
            "## Claim Boundary",
            "",
            "No claim is made that this solves ARC-AGI-2. The allowed claim requires a controlled final run that passes the listed gates.",
        ]
    )
    return "\n".join(lines) + "\n"


def _variant_plan() -> List[ArcVariant]:
    return [
        ArcVariant(
            name="A_basic_latent_rule_clones",
            description="K=4 shared-weight clones over raw/diff/color/geometry views with candidate-token direct attention.",
            model=ArcModelConfig(variant_family="basic_latent_rule_clones", view_names=("raw", "diff", "color", "geometry"), learned_slots=False),
        ),
        ArcVariant(
            name="B_learned_rule_slot_clones",
            description="K=4 learned slot clones; each slot sees the same normalized raw train pairs with distinct slot embeddings.",
            model=ArcModelConfig(variant_family="learned_rule_slots", view_names=("slot_0", "slot_1", "slot_2", "slot_3"), learned_slots=True),
        ),
        ArcVariant(
            name="C_view_biased_rule_clones",
            description="K=6 engineered ARC views: raw, diff, object, color, geometry, symmetry.",
            model=ArcModelConfig(variant_family="view_biased_rule_clones", view_names=("raw", "diff", "object", "color", "geometry", "symmetry"), learned_slots=False),
        ),
    ]


def _load_config(path: Path) -> Dict[str, object]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _dataset_config(data: object) -> ArcVerificationDatasetConfig:
    values = dict(data or {})
    allowed = set(ArcVerificationDatasetConfig.__dataclass_fields__.keys())
    return ArcVerificationDatasetConfig(**{key: value for key, value in values.items() if key in allowed})


def _training_config(data: object) -> ArcTrainingConfig:
    values = dict(data or {})
    allowed = set(ArcTrainingConfig.__dataclass_fields__.keys())
    return ArcTrainingConfig(**{key: value for key, value in values.items() if key in allowed})


def _accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    return float(np.mean(predictions.astype(np.int64) == labels.astype(np.int64)))


def _top1(logits: np.ndarray, labels: np.ndarray) -> float:
    if logits.size == 0 or len(labels) == 0:
        return 0.0
    return _accuracy(np.argmax(logits, axis=1), labels)


def _accuracy_by_key(predictions: np.ndarray, labels: np.ndarray, keys: Sequence[str]) -> Dict[str, float]:
    out = {}
    for key in sorted(set(keys)):
        idx = [i for i, value in enumerate(keys) if value == key]
        if idx:
            out[str(key)] = _accuracy(predictions[idx], labels[idx])
    return out


def _negative_confusion(predictions: np.ndarray, examples: Sequence[ArcVerificationExample]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for prediction, example in zip(predictions, examples):
        label = "gold" if int(prediction) == int(example.label) else str(example.negative_types[int(prediction)])
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _shape(grid: Sequence[Sequence[int]]) -> Tuple[int, int]:
    return (len(grid), len(grid[0]) if grid else 0)


def _dominant(grid: np.ndarray) -> int:
    if grid.size == 0:
        return 0
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _entropy(grid: np.ndarray) -> float:
    if grid.size == 0:
        return 0.0
    _values, counts = np.unique(grid, return_counts=True)
    probs = counts.astype(np.float64) / float(grid.size)
    return float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))


def _resized_hamming(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 1.0
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    overlap = float(np.mean(a[:h, :w] != b[:h, :w]))
    size_penalty = abs(a.shape[0] - b.shape[0]) + abs(a.shape[1] - b.shape[1])
    return overlap + 0.05 * size_penalty


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp(min=1.0)
    return (values * weights).sum(dim=1) / denom


def _softmax_np(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    values = values - np.max(values)
    exp = np.exp(values)
    return exp / np.clip(exp.sum(), 1e-12, None)


def _bootstrap_ci(values: Sequence[float], iterations: int = 1000) -> List[float]:
    if not values:
        return [0.0, 0.0]
    rng = np.random.default_rng(12345)
    arr = np.asarray(values, dtype=np.float64)
    means = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(iterations)]
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _flat_params(items: Sequence[Tuple[str, nn.Parameter]]) -> torch.Tensor:
    tensors = [parameter.detach().float().cpu().reshape(-1) for _name, parameter in items]
    return torch.cat(tensors) if tensors else torch.zeros(0)


def _l2_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() != b.numel() or a.numel() == 0:
        return 0.0
    return float(torch.linalg.vector_norm(a - b).item())


def _grad_norm(parameters: Sequence[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        total += float(parameter.grad.detach().float().pow(2).sum().cpu())
    return float(math.sqrt(total)) if total > 0.0 else 0.0


def _chunks(values: Sequence[int], size: int) -> Iterable[List[int]]:
    for index in range(0, len(values), max(1, size)):
        yield list(values[index : index + max(1, size)])


def _batched(values: Sequence[ArcVerificationExample], size: int) -> Iterable[List[ArcVerificationExample]]:
    for index in range(0, len(values), max(1, size)):
        yield list(values[index : index + max(1, size)])


def _mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return float(pstdev(values)) if len(values) > 1 else 0.0


def _best_value(result: Dict[str, object], key: str) -> float:
    rows = result.get("summary", {}).get("variant_summaries", [])
    if not rows:
        return 0.0
    return float(rows[0].get(key, 0.0))


def _compatible_dim(model_dim: int, num_heads: int) -> int:
    model_dim = int(model_dim)
    num_heads = max(1, int(num_heads))
    remainder = model_dim % num_heads
    return model_dim if remainder == 0 else model_dim + num_heads - remainder


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(requested: str) -> str:
    if requested in {"auto", "", "none"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def _cuda_max_memory(device: str) -> int:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated())
    return 0


def _clear_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
