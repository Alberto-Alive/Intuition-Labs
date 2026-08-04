"""Small falsification harness utilities for DIGIT Extrapolation E10."""

from __future__ import annotations

import json
import math
import random
import copy
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

from eval_common import compute_success_safety_metrics, compute_threshold_sensitivity, apply_trace_perturbation

from .config import Config
from .models.digit import DIGITModel


PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

DOMAIN_WORDS = sorted(
    {
        "a",
        "an",
        "analysis",
        "and",
        "appears",
        "attention",
        "below",
        "but",
        "clear",
        "confidence",
        "consistent",
        "decreasing",
        "diffuse",
        "entropy",
        "evidence",
        "failure",
        "focused",
        "for",
        "from",
        "group",
        "high",
        "in",
        "increasing",
        "indicates",
        "is",
        "likely",
        "low",
        "medium",
        "mixed",
        "of",
        "outcome",
        "pattern",
        "query",
        "records",
        "remains",
        "shows",
        "signal",
        "stable",
        "suggests",
        "success",
        "supports",
        "the",
        "this",
        "trajectory",
        "uncertain",
        "volatile",
        "with",
        ".",
        ",",
    }
)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class SimpleVocabulary:
    """Token-to-id mapping for the text decoder."""

    def __init__(self):
        self.tokens = SPECIAL_TOKENS + DOMAIN_WORDS
        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self.pad_idx = self.token_to_id[PAD_TOKEN]
        self.bos_idx = self.token_to_id[BOS_TOKEN]
        self.eos_idx = self.token_to_id[EOS_TOKEN]
        self.unk_idx = self.token_to_id[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, text: str, max_len: int = 48) -> list[int]:
        words = text.lower().replace(".", " .").replace(",", " ,").split()
        ids = [self.bos_idx]
        for word in words:
            ids.append(self.token_to_id.get(word, self.unk_idx))
        ids.append(self.eos_idx)
        if len(ids) > max_len:
            ids = ids[: max_len - 1] + [self.eos_idx]
        while len(ids) < max_len:
            ids.append(self.pad_idx)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        tokens: list[str] = []
        for idx in ids:
            idx = int(idx)
            if idx == self.bos_idx:
                continue
            if idx == self.eos_idx:
                break
            if idx == self.pad_idx:
                continue
            tokens.append(self.id_to_token.get(idx, UNK_TOKEN))
        text = " ".join(tokens)
        return text.replace(" .", ".").replace(" ,", ",")


def load_trace_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_trace_split(trace_dir: str | Path, split: str) -> list[dict[str, Any]]:
    return load_trace_records(Path(trace_dir) / f"{split}.jsonl")


def load_trace_metadata(trace_dir: str | Path) -> dict[str, Any]:
    return load_json(Path(trace_dir) / "metadata.json")


def slice_records(records: Sequence[dict[str, Any]], limit: int | None = None, offset: int = 0) -> list[dict[str, Any]]:
    if limit is None:
        return list(records[offset:])
    return list(records[offset : offset + limit])


class TraceSliceDataset(Dataset):
    """Minimal trace dataset for evaluation and head-only finetuning."""

    def __init__(self, records: Sequence[dict[str, Any]], vocab: SimpleVocabulary, max_output_len: int):
        self.records = list(records)
        self.vocab = vocab
        self.max_output_len = int(max_output_len)
        self.target_ids = [self.vocab.encode(record["response_text"], self.max_output_len) for record in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        query = torch.tensor(record["query_fields"], dtype=torch.long)
        trace_inputs = torch.tensor(record["trace_inputs"], dtype=torch.float32)
        evidence_targets = torch.tensor(
            [record["evidence_targets"][name] for name in ("stability", "support", "success", "failure", "ambiguity")],
            dtype=torch.float32,
        )
        prim_targets = torch.tensor(
            [
                record["trajectory_shape"],
                record["attention_pattern"],
                record["confidence"],
                record["outcome"],
            ],
            dtype=torch.long,
        )
        target_ids = torch.tensor(self.target_ids[idx], dtype=torch.long)
        return query, trace_inputs, evidence_targets, prim_targets, target_ids


def trace_collate_fn(batch):
    queries, trace_inputs, evidence_targets, prim_targets, target_ids = zip(*batch)
    return (
        torch.stack(queries),
        torch.stack(trace_inputs),
        torch.stack(evidence_targets),
        torch.stack(prim_targets),
        torch.stack(target_ids),
    )


def build_loader(
    records: Sequence[dict[str, Any]],
    vocab: SimpleVocabulary,
    max_output_len: int,
    batch_size: int,
    *,
    shuffle: bool = False,
    seed: int = 0,
) -> tuple[TraceSliceDataset, DataLoader]:
    dataset = TraceSliceDataset(records, vocab=vocab, max_output_len=max_output_len)
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        collate_fn=trace_collate_fn,
        generator=generator if shuffle else None,
    )
    return dataset, loader


def _strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not state_dict:
        return state_dict
    if not all(isinstance(key, str) and key.startswith("module.") for key in state_dict):
        return state_dict
    return {key[len("module."):]: value for key, value in state_dict.items()}


def load_checkpoint_payload(checkpoint_path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(checkpoint_path), map_location="cpu")
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")


def resolve_checkpoint_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(path)
    candidates = [
        path / "checkpoint.pt",
        path / "model.pt",
        path / "state_dict.pt",
        path / "best.pt",
        path / "best_stage1.pt",
        path / "best_stage2.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No checkpoint file found in {path}. Tried: {', '.join(str(candidate) for candidate in candidates)}"
    )


def _apply_payload_config(config: Config, payload: dict[str, Any]) -> None:
    payload_config = payload.get("config")
    if not isinstance(payload_config, dict):
        return
    for key, value in payload_config.items():
        setattr(config, key, value)


def build_model(
    config: Config,
    vocab: SimpleVocabulary,
    *,
    checkpoint_path: str | Path | None = None,
    head_mode: str = "mlp",
    config_overrides: dict[str, Any] | None = None,
    trace_metadata: dict[str, Any] | None = None,
    device: torch.device | str | None = None,
    strict: bool | None = None,
) -> DIGITModel:
    config = copy.deepcopy(config)
    payload: dict[str, Any] | None = None
    if checkpoint_path is not None:
        resolved = resolve_checkpoint_path(checkpoint_path)
        payload = load_checkpoint_payload(resolved)
        _apply_payload_config(config, payload)
    if config_overrides:
        for key, value in config_overrides.items():
            setattr(config, key, value)
    config.head_mode = head_mode
    model = DIGITModel(config, vocab)
    if checkpoint_path is not None and payload is not None:
        state_dict = payload.get("model_state_dict")
        if state_dict is None:
            state_dict = payload.get("state_dict")
        if state_dict is None:
            state_dict = payload.get("model")
        if state_dict is None:
            state_dict = payload
        state_dict = _strip_module_prefix(dict(state_dict))
        load_strict = strict if strict is not None else (head_mode == "mlp")
        missing, unexpected = model.load_state_dict(state_dict, strict=load_strict)
        if not load_strict:
            print(f"Loaded checkpoint with strict={load_strict}; missing={len(missing)}, unexpected={len(unexpected)}")
    if trace_metadata is not None:
        model.set_trace_metadata(trace_metadata)
    if device is not None:
        model = model.to(device)
    return model


def freeze_all_parameters(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def trainable_head_parameters(model: DIGITModel, head_mode: str) -> list[torch.nn.Parameter]:
    freeze_all_parameters(model)
    head_mode = str(head_mode).strip().lower()
    if head_mode == "mlp":
        modules = [model.bottleneck.confidence_head, model.bottleneck.outcome_head]
    elif head_mode == "linear_evidence_only":
        modules = [model.bottleneck.confidence_linear_head, model.bottleneck.outcome_linear_head]
    elif head_mode == "ordered_threshold":
        modules = [model.bottleneck.confidence_threshold_head, model.bottleneck.outcome_threshold_head]
    else:
        raise ValueError(f"Unsupported head_mode {head_mode!r}")
    params: list[torch.nn.Parameter] = []
    for module in modules:
        for param in module.parameters():
            param.requires_grad = True
            params.append(param)
    return params


def active_head_modules(model: DIGITModel, head_mode: str) -> list[torch.nn.Module]:
    head_mode = str(head_mode).strip().lower()
    if head_mode == "mlp":
        return [model.bottleneck.confidence_head, model.bottleneck.outcome_head]
    if head_mode == "linear_evidence_only":
        return [model.bottleneck.confidence_linear_head, model.bottleneck.outcome_linear_head]
    if head_mode == "ordered_threshold":
        return [model.bottleneck.confidence_threshold_head, model.bottleneck.outcome_threshold_head]
    raise ValueError(f"Unsupported head_mode {head_mode!r}")


def compute_multiclass_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    head_names = ["trajectory", "pattern", "confidence", "outcome"]
    head_label_sizes = [4, 3, 3, 3]
    for idx, head_name in enumerate(head_names):
        pred = predictions[:, idx]
        target = targets[:, idx]
        metrics[f"{head_name}_acc"] = float((pred == target).float().mean().item())
        metrics[f"{head_name}_macro_f1"] = float(
            f1_score(target.cpu().numpy(), pred.cpu().numpy(), average="macro", zero_division=0)
        )
        metrics[f"{head_name}_confusion"] = confusion_matrix(
            target.cpu().numpy(),
            pred.cpu().numpy(),
            labels=list(range(head_label_sizes[idx])),
        ).tolist()

    metrics["joint_acc"] = float((predictions == targets).all(dim=1).float().mean().item())
    outcome_precision, outcome_recall, outcome_f1, _ = precision_recall_fscore_support(
        targets[:, 3].cpu().numpy(),
        predictions[:, 3].cpu().numpy(),
        labels=[0, 1, 2],
        zero_division=0,
    )
    for idx, label in enumerate(["SUCCESS_LIKELY", "UNCERTAIN", "FAILURE_LIKELY"]):
        label_key = label.lower()
        metrics[f"{label_key}_precision"] = float(outcome_precision[idx])
        metrics[f"{label_key}_recall"] = float(outcome_recall[idx])
        metrics[f"{label_key}_f1"] = float(outcome_f1[idx])
    metrics["failure_recall"] = float(outcome_recall[2])
    return metrics


def class_weights_from_records(
    records: Sequence[dict[str, Any]],
    key: str,
    *,
    num_classes: int,
    power: float = 0.5,
    clip: float = 4.0,
) -> torch.Tensor:
    counts = Counter(int(record[key]) for record in records)
    total = max(sum(counts.values()), 1)
    weights = torch.ones(num_classes, dtype=torch.float32)
    for cls, count in counts.items():
        if count > 0:
            raw = total / (num_classes * count)
            weights[cls] = float(max(min(raw ** power, clip), 0.5))
    return weights


def evaluate_model(
    model: DIGITModel,
    loader: DataLoader,
    device: torch.device,
    *,
    records: Sequence[dict[str, Any]] | None = None,
    collect_bundle: bool = False,
    evidence_override: dict[str, Any] | None = None,
    diffusion_override: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    model.eval()
    all_predictions: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_logits: dict[str, list[torch.Tensor]] = {"trajectory_shape": [], "attention_pattern": [], "confidence": [], "outcome": []}
    all_probs: dict[str, list[torch.Tensor]] = {"trajectory_shape": [], "attention_pattern": [], "confidence": [], "outcome": []}

    with torch.no_grad():
        for batch in loader:
            queries, trace_inputs, evidence_targets, prim_targets, target_ids = batch
            queries = queries.to(device)
            trace_inputs = trace_inputs.to(device)
            evidence_targets = evidence_targets.to(device)
            prim_targets = prim_targets.to(device)
            target_ids = target_ids.to(device)
            out = model(
                queries,
                trace_inputs,
                target_ids,
                bottleneck_mode="hard",
                tau=1.0,
                skip_decoder=True,
                evidence_override=evidence_override,
                diffusion_override=diffusion_override,
            )
            primitives = out["primitives"]
            predictions = torch.stack(
                [
                    primitives.trajectory_shape_logits.argmax(dim=-1),
                    primitives.attention_pattern_logits.argmax(dim=-1),
                    primitives.confidence_logits.argmax(dim=-1),
                    primitives.outcome_logits.argmax(dim=-1),
                ],
                dim=1,
            )
            all_predictions.append(predictions.cpu())
            all_targets.append(prim_targets.cpu())
            all_logits["trajectory_shape"].append(primitives.trajectory_shape_logits.cpu())
            all_logits["attention_pattern"].append(primitives.attention_pattern_logits.cpu())
            all_logits["confidence"].append(primitives.confidence_logits.cpu())
            all_logits["outcome"].append(primitives.outcome_logits.cpu())
            all_probs["trajectory_shape"].append(torch.softmax(primitives.trajectory_shape_logits, dim=-1).cpu())
            all_probs["attention_pattern"].append(torch.softmax(primitives.attention_pattern_logits, dim=-1).cpu())
            all_probs["confidence"].append(torch.softmax(primitives.confidence_logits, dim=-1).cpu())
            all_probs["outcome"].append(torch.softmax(primitives.outcome_logits, dim=-1).cpu())

    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = compute_multiclass_metrics(predictions, targets)
    if records is not None:
        outcome_probs = torch.cat(all_probs["outcome"], dim=0).numpy()
        is_correct = np.array([int(bool(record.get("is_correct", False))) for record in records], dtype=np.int64)
        metrics.update(
            compute_success_safety_metrics(
                outcome_probabilities=outcome_probs,
                outcome_predictions=predictions[:, 3].numpy(),
                outcome_targets=targets[:, 3].numpy(),
                is_correct=is_correct,
            )
        )
    bundle = {
        "predictions": predictions,
        "targets": targets,
        "logits": {key: torch.cat(values, dim=0) for key, values in all_logits.items()},
        "probs": {key: torch.cat(values, dim=0) for key, values in all_probs.items()},
    }
    if not collect_bundle:
        return metrics
    return metrics, bundle


def commitment_monotonicity_audit(
    model: DIGITModel,
    loader: DataLoader,
    device: torch.device,
    *,
    path_scales: dict[str, float] | None = None,
) -> dict[str, Any]:
    perturbation_kinds = [
        "lower_agreement",
        "lower_margin",
        "higher_entropy",
        "lower_attention_concentration",
        "higher_variation_ratio",
    ]
    total_checks = 0
    commitment_violations = 0
    success_prob_violations = 0
    per_kind: dict[str, dict[str, float]] = {
        kind: {
            "count": 0.0,
            "commitment": 0.0,
            "success_prob": 0.0,
            "commitment_mean_delta_sum": 0.0,
            "success_prob_mean_delta_sum": 0.0,
        }
        for kind in perturbation_kinds
    }
    model.eval()
    with torch.no_grad():
        for batch in loader:
            queries, trace_inputs, _, _, target_ids = batch
            queries = queries.to(device)
            trace_inputs = trace_inputs.to(device)
            target_ids = target_ids.to(device)
            base_out = model(
                queries,
                trace_inputs,
                target_ids,
                bottleneck_mode="hard",
                tau=1.0,
                skip_decoder=True,
                path_scales=path_scales,
            )
            base_commitment = base_out["primitives"].commitment_depth
            base_success_prob = torch.softmax(base_out["primitives"].outcome_logits, dim=-1)[:, 0]
            for kind in perturbation_kinds:
                perturbed_out = model(
                    queries,
                    apply_trace_perturbation(trace_inputs, kind),
                    target_ids,
                    bottleneck_mode="hard",
                    tau=1.0,
                    skip_decoder=True,
                    path_scales=path_scales,
                )
                perturbed_commitment = perturbed_out["primitives"].commitment_depth
                perturbed_success_prob = torch.softmax(perturbed_out["primitives"].outcome_logits, dim=-1)[:, 0]
                commitment_bad = perturbed_commitment > (base_commitment + 1e-6)
                success_prob_bad = perturbed_success_prob > (base_success_prob + 1e-6)
                count = float(commitment_bad.numel())
                total_checks += int(commitment_bad.numel())
                commitment_violations += int(commitment_bad.long().sum().item())
                success_prob_violations += int(success_prob_bad.long().sum().item())
                per_kind[kind]["count"] += count
                per_kind[kind]["commitment"] += float(commitment_bad.float().sum().item())
                per_kind[kind]["success_prob"] += float(success_prob_bad.float().sum().item())
                per_kind[kind]["commitment_mean_delta_sum"] += float(
                    (perturbed_commitment - base_commitment).sum().item()
                )
                per_kind[kind]["success_prob_mean_delta_sum"] += float(
                    (perturbed_success_prob - base_success_prob).sum().item()
                )
    payload = {
        "num_checks": int(total_checks),
        "commitment_monotonicity_violation_rate": float(commitment_violations / max(total_checks, 1)),
        "success_prob_monotonicity_violation_rate": float(success_prob_violations / max(total_checks, 1)),
        "by_perturbation": {},
    }
    for kind, stats in per_kind.items():
        denom = max(stats["count"], 1.0)
        payload["by_perturbation"][kind] = {
            "commitment_violation_rate": float(stats["commitment"] / denom),
            "success_prob_violation_rate": float(stats["success_prob"] / denom),
            "commitment_mean_delta": float(stats["commitment_mean_delta_sum"] / denom),
            "success_prob_mean_delta": float(stats["success_prob_mean_delta_sum"] / denom),
        }
    return payload


def threshold_sensitivity(records: Sequence[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    return compute_threshold_sensitivity(records, metadata)


def compute_model_threshold_sensitivity(
    outcome_probabilities: torch.Tensor | np.ndarray,
    *,
    thresholds: Sequence[float] | None = None,
    baseline_threshold: float = 0.5,
) -> dict[str, Any]:
    """Measure how outcome shares move under alternate decision thresholds.

    This is the model-facing counterpart to the corpus relabel threshold
    audit. It keeps the probability table fixed and sweeps the success/failure
    cutoffs used to convert probabilities into the 3-way outcome label.
    """

    probs = np.asarray(outcome_probabilities, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[-1] != 3:
        raise ValueError(f"Expected outcome probabilities with shape [N, 3], got {probs.shape!r}")

    if thresholds is None:
        thresholds = (0.35, 0.45, 0.50, 0.55, 0.65)
    thresholds = [float(value) for value in thresholds]
    baseline_threshold = float(baseline_threshold)

    def _predict(theta: float) -> np.ndarray:
        success_prob = probs[:, 0]
        failure_prob = probs[:, 2]
        prediction = np.full(probs.shape[0], 1, dtype=np.int64)
        success_mask = (success_prob >= theta) & (success_prob >= failure_prob)
        failure_mask = (failure_prob >= theta) & (failure_prob > success_prob)
        prediction[success_mask] = 0
        prediction[failure_mask] = 2
        return prediction

    def _share(prediction: np.ndarray) -> dict[str, float]:
        total = max(int(prediction.shape[0]), 1)
        return {
            "SUCCESS_LIKELY": float(np.sum(prediction == 0) / total),
            "UNCERTAIN": float(np.sum(prediction == 1) / total),
            "FAILURE_LIKELY": float(np.sum(prediction == 2) / total),
        }

    baseline_prediction = _predict(baseline_threshold)
    baseline_share = _share(baseline_prediction)

    scenarios: dict[str, dict[str, Any]] = {}
    max_outcome_share_swing = 0.0
    for theta in thresholds:
        prediction = _predict(theta)
        share = _share(prediction)
        share_swings = {
            label: abs(float(share[label]) - float(baseline_share[label]))
            for label in baseline_share
        }
        scenario_swing = max(share_swings.values(), default=0.0)
        max_outcome_share_swing = max(max_outcome_share_swing, scenario_swing)
        scenarios[f"theta_{theta:.2f}"] = {
            "threshold": float(theta),
            "outcome_share": share,
            "share_swings": share_swings,
            "max_outcome_share_swing": float(scenario_swing),
        }

    return {
        "baseline_threshold": baseline_threshold,
        "baseline_outcome_share": baseline_share,
        "max_outcome_share_swing": float(max_outcome_share_swing),
        "label_revision_trigger": bool(max_outcome_share_swing > 0.10),
        "scenarios": scenarios,
    }
