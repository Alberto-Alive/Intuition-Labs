from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.agents.types import AttemptBatch
from src.coordinators.activation import TextOnlyMLPCoordinator
from src.coordinators.mlp import MLPTrainingConfig
from src.coordinators.probes import HiddenStateOnlyLabelProbe
from src.datasets.strict_coordination import (
    StrictCoordinationConfig,
    StrictCoordinationExample,
    build_strict_coordination_splits,
    strict_label,
)
from src.evaluation.audit import audit_event, split_overlap_audit, summarize_test_access
from src.evaluation.metrics import evaluate_predictions
from src.evaluation.report import write_report


BENCHMARK = "shared_weight_cloned_agent"


@dataclass(frozen=True)
class SharedAgentConfig:
    n_agents: int = 4
    hidden_dim: int = 16
    num_layers: int = 1
    num_heads: int = 2
    ff_dim: int = 64
    evidence_embedding_scale: float = 0.001


@dataclass(frozen=True)
class SharedCoordinatorConfig:
    num_layers: int = 1
    num_heads: int = 2
    ff_dim: int = 64


@dataclass(frozen=True)
class SharedTrainingConfig:
    epochs: int = 60
    batch_size: int = 128
    lr: float = 0.003
    weight_decay: float = 0.0001
    patience: int = 60


@dataclass
class StrictCloneSplit:
    split: str
    example_ids: np.ndarray
    task_ids: np.ndarray
    evidence_bits: np.ndarray
    labels: np.ndarray

    @property
    def n_examples(self) -> int:
        return int(self.labels.shape[0])

    @property
    def n_agents(self) -> int:
        return int(self.evidence_bits.shape[1])


@dataclass
class FitResult:
    method: str
    agent: "SmallTransformerSharedAgent"
    coordinator: "RoleAwareQKVCoordinator"
    trainable_agent: bool
    param_count: int
    audit: Dict[str, object]
    history: List[Dict[str, float]]
    checkpoints: List[Dict[str, Dict[str, torch.Tensor]]]


class StrictCloneTokenizer:
    """Tiny fixed vocabulary for the strict private-evidence views."""

    def __init__(self, n_agents: int) -> None:
        self.n_agents = int(n_agents)
        self.pad_token_id = 0
        self.role_offset = 1
        self.slot_offset = self.role_offset + self.n_agents
        self.separator_token_id = self.slot_offset + self.n_agents
        self.bit_zero_token_id = self.separator_token_id + 1
        self.bit_one_token_id = self.separator_token_id + 2
        self.mask_token_id = self.separator_token_id + 3
        self.vocab_size = self.mask_token_id + 1

    def encode(self, evidence_bits: np.ndarray, mask_evidence: bool = False) -> np.ndarray:
        bits = np.asarray(evidence_bits, dtype=np.int64)
        if bits.ndim != 2 or bits.shape[1] != self.n_agents:
            raise ValueError(f"expected evidence bits with shape [batch,{self.n_agents}]")
        tokens = np.empty((bits.shape[0], self.n_agents, 4), dtype=np.int64)
        for role_id in range(self.n_agents):
            tokens[:, role_id, 0] = self.role_offset + role_id
            tokens[:, role_id, 1] = self.slot_offset + role_id
            tokens[:, role_id, 2] = self.separator_token_id
            if mask_evidence:
                tokens[:, role_id, 3] = self.mask_token_id
            else:
                tokens[:, role_id, 3] = np.where(bits[:, role_id] == 1, self.bit_one_token_id, self.bit_zero_token_id)
        return tokens


class SmallTransformerSharedAgent(nn.Module):
    """A small trainable transformer used as the single shared cloned agent."""

    def __init__(self, vocab_size: int, config: SharedAgentConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(4, config.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=_compatible_heads(config.hidden_dim, config.num_heads),
            dim_feedforward=config.ff_dim,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.output_norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(token_ids.shape[1], dtype=torch.long, device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions).unsqueeze(0)
        hidden = self.blocks(hidden)
        return self.output_norm(hidden[:, -1, :])


class RoleAwareQKVCoordinator(nn.Module):
    """Role-aware QKV coordinator over one activation per clone.

    The coordinator sorts clone tokens by role id before the final MLP, making
    physical clone order irrelevant while keeping role labels semantically
    meaningful.
    """

    def __init__(self, n_roles: int, hidden_dim: int, num_classes: int, config: SharedCoordinatorConfig) -> None:
        super().__init__()
        self.n_roles = int(n_roles)
        self.hidden_dim = int(hidden_dim)
        self.role_embedding = nn.Embedding(n_roles, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=_compatible_heads(hidden_dim, config.num_heads),
            dim_feedforward=config.ff_dim,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.qkv_layers = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(n_roles * hidden_dim),
            nn.Linear(n_roles * hidden_dim, config.ff_dim),
            nn.GELU(),
            nn.Linear(config.ff_dim, num_classes),
        )

    def forward(self, clone_activations: torch.Tensor, role_ids: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(role_ids, dim=1)
        hidden_order = order.unsqueeze(-1).expand(-1, -1, clone_activations.shape[-1])
        sorted_hidden = clone_activations.gather(1, hidden_order)
        sorted_roles = role_ids.gather(1, order)
        tokens = sorted_hidden + self.role_embedding(sorted_roles.clamp(min=0, max=self.n_roles - 1))
        contextual = self.qkv_layers(tokens)
        return self.head(contextual.reshape(contextual.shape[0], -1))


def initialize_stage6_agent(
    agent: SmallTransformerSharedAgent,
    tokenizer: StrictCloneTokenizer,
    evidence_embedding_scale: float,
) -> None:
    """Initialize evidence tokens as a tiny learnable signal.

    Frozen-M baselines cannot amplify this signal. The trainable shared agent
    can, but only through the group loss that reaches the shared embedding and
    transformer parameters.
    """

    with torch.no_grad():
        nn.init.normal_(agent.token_embedding.weight, mean=0.0, std=0.2)
        nn.init.normal_(agent.position_embedding.weight, mean=0.0, std=0.2)
        agent.token_embedding.weight[tokenizer.pad_token_id].zero_()
        base = torch.zeros_like(agent.token_embedding.weight[tokenizer.bit_zero_token_id])
        agent.token_embedding.weight[tokenizer.bit_zero_token_id].copy_(
            base + evidence_embedding_scale * torch.randn_like(base)
        )
        agent.token_embedding.weight[tokenizer.bit_one_token_id].copy_(
            base + evidence_embedding_scale * torch.randn_like(base)
        )


def run_experiment(config_path: str) -> Dict[str, object]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    deterministic = bool(config.get("deterministic", False))
    if deterministic:
        torch.use_deterministic_algorithms(True)
    device = _resolve_device(str(config.get("device", "cpu")))
    dataset_config = _make_dataclass(StrictCoordinationConfig, config["strict_dataset"])
    agent_config = _make_dataclass(SharedAgentConfig, config.get("agent", {}))
    coordinator_config = _make_dataclass(SharedCoordinatorConfig, config.get("coordinator", {}))
    training_config = _make_dataclass(SharedTrainingConfig, config.get("training", {}))
    probe_training = _make_mlp_training_config(config.get("probe_training", config.get("training", {})))
    probe_components = int(dict(config.get("probe_training", {})).get("pca_components", 8))
    if dataset_config.n_evidence_bits != agent_config.n_agents:
        raise ValueError("Stage 6A expects one clone per strict evidence bit")

    metrics: List[Dict[str, object]] = []
    diagnostics: List[Dict[str, object]] = []
    audit: List[Dict[str, object]] = []
    audit_log_rows: List[Dict[str, object]] = []

    for seed in [int(value) for value in config.get("seeds", [0])]:
        print(f"seed={seed}: building strict shared-weight clone splits")
        examples = build_strict_coordination_splits(dataset_config, seed=seed)
        splits = {name: _clone_split(name, rows, agent_config.n_agents) for name, rows in examples.items()}
        base_batches = {
            split: _attempt_batch_from_split(data, hidden_dim=agent_config.hidden_dim)
            for split, data in splits.items()
        }
        audit.append(split_overlap_audit(BENCHMARK, seed, base_batches))
        event_index = 0

        print(f"seed={seed}: fitting frozen shared-agent coordinator baseline")
        audit.append(audit_event(BENCHMARK, seed, event_index, "fit", "frozen_shared_agent_coordinator", "none", ("train", "dev")))
        frozen = fit_shared_weight_model(
            splits=splits,
            agent_config=agent_config,
            coordinator_config=coordinator_config,
            training_config=training_config,
            num_classes=dataset_config.num_classes,
            seed=seed,
            device=device,
            trainable_agent=False,
            method="frozen_shared_agent_coordinator",
        )
        diagnostics.append(frozen.audit)
        audit_log_rows.append(frozen.audit)
        event_index += 1

        print(f"seed={seed}: fitting trainable shared-agent coordinator")
        audit.append(audit_event(BENCHMARK, seed, event_index, "fit", "trainable_shared_agent_coordinator", "none", ("train", "dev")))
        trainable = fit_shared_weight_model(
            splits=splits,
            agent_config=agent_config,
            coordinator_config=coordinator_config,
            training_config=training_config,
            num_classes=dataset_config.num_classes,
            seed=seed,
            device=device,
            trainable_agent=True,
            method="trainable_shared_agent_coordinator",
        )
        diagnostics.append(trainable.audit)
        audit_log_rows.append(trainable.audit)
        event_index += 1

        print(f"seed={seed}: fitting randomized-label shared-agent sanity run")
        randomized_train = _balanced_randomized_labels(splits["train"], dataset_config.num_classes, seed + 87277)
        randomized_dev = _balanced_randomized_labels(splits["dev"], dataset_config.num_classes, seed + 97277)
        audit.append(audit_event(BENCHMARK, seed, event_index, "fit", "trainable_shared_agent_coordinator", "randomized_labels", ("train", "dev")))
        randomized = fit_shared_weight_model(
            splits=splits,
            agent_config=agent_config,
            coordinator_config=coordinator_config,
            training_config=training_config,
            num_classes=dataset_config.num_classes,
            seed=seed,
            device=device,
            trainable_agent=True,
            method="trainable_shared_agent_coordinator",
            condition="randomized_labels",
            train_targets=randomized_train,
            dev_targets=randomized_dev,
        )
        diagnostics.append(randomized.audit)
        audit_log_rows.append(randomized.audit)
        event_index += 1

        print(f"seed={seed}: fitting output-only/text-only and hidden-state diagnostic probes")
        text_only = TextOnlyMLPCoordinator(
            num_classes=dataset_config.num_classes,
            num_tasks=dataset_config.num_tasks,
            training=probe_training,
            seed=seed + 33_000,
            device=device,
        )
        audit.append(audit_event(BENCHMARK, seed, event_index, "fit", "text_only_coordinator", "none", ("train", "dev")))
        text_only.fit(base_batches["train"], base_batches["dev"])
        event_index += 1

        frozen_hidden_batches = capture_hidden_batches(
            frozen.agent,
            splits,
            agent_config=agent_config,
            device=device,
        )
        hidden_probe = HiddenStateOnlyLabelProbe(
            num_classes=dataset_config.num_classes,
            components=probe_components,
            training=probe_training,
            seed=seed + 44_000,
            device=device,
            pooling="mean",
        )
        audit.append(audit_event(BENCHMARK, seed, event_index, "fit", "hidden_state_only_probe", "none", ("train", "dev")))
        hidden_probe.fit(frozen_hidden_batches["train"], frozen_hidden_batches["dev"])
        event_index += 1

        audit.append(audit_event(BENCHMARK, seed, event_index, "test_gate_opened", None, None, ()))
        event_index += 1

        for split_name in ("train", "dev", "test"):
            split_inputs = (split_name,)
            if split_name == "test":
                audit.append(audit_event(BENCHMARK, seed, event_index, "predict", "shared_weight_methods", "none", split_inputs))
                event_index += 1
            batch = base_batches[split_name]
            for result in (frozen, trainable):
                predictions = predict_shared_weight_model(
                    result.agent,
                    result.coordinator,
                    splits[split_name],
                    agent_config=agent_config,
                    device=device,
                )
                metrics.append(
                    evaluate_predictions(
                        method=result.method,
                        condition="none",
                        predictions=predictions,
                        batch=batch,
                        seed=seed,
                        param_count=result.param_count,
                        benchmark=BENCHMARK,
                        metadata=_method_metadata(result),
                    )
                )
            metrics.append(
                evaluate_predictions(
                    method="evidence_sharing_rule_oracle",
                    condition="none",
                    predictions=_oracle_predictions(splits[split_name]),
                    batch=batch,
                    seed=seed,
                    param_count=0,
                    benchmark=BENCHMARK,
                    metadata={"uses_explicit_private_evidence": True},
                )
            )
            if split_name in {"dev", "test"}:
                metrics.append(
                    evaluate_predictions(
                        method="text_only_coordinator",
                        condition="none",
                        predictions=text_only.predict(batch),
                        batch=batch,
                        seed=seed,
                        param_count=int(text_only.param_count),
                        benchmark=BENCHMARK,
                        metadata=text_only.metadata(),
                    )
                )
                hidden_batch = frozen_hidden_batches[split_name]
                metrics.append(
                    evaluate_predictions(
                        method="hidden_state_only_probe",
                        condition="none",
                        predictions=hidden_probe.predict(hidden_batch),
                        batch=hidden_batch,
                        seed=seed,
                        param_count=int(hidden_probe.param_count),
                        benchmark=BENCHMARK,
                        metadata={**hidden_probe.metadata(), "diagnostic_only": True, "agent_frozen": True},
                    )
                )

        for split_name in ("dev", "test"):
            batch = base_batches[split_name]
            for condition in (
                "evidence_masked",
                "evidence_shuffled",
                "hidden_states_shuffled_across_examples",
                "slot_labels_shuffled",
                "physical_order_shuffled_roles_preserved",
            ):
                predictions = predict_shared_weight_model(
                    trainable.agent,
                    trainable.coordinator,
                    splits[split_name],
                    agent_config=agent_config,
                    device=device,
                    condition=condition,
                    seed=seed + _split_offset(split_name),
                )
                metrics.append(
                    evaluate_predictions(
                        method="trainable_shared_agent_coordinator",
                        condition=condition,
                        predictions=predictions,
                        batch=batch,
                        seed=seed,
                        param_count=trainable.param_count,
                        benchmark=BENCHMARK,
                        metadata=_method_metadata(trainable),
                    )
                )
            randomized_predictions = predict_shared_weight_model(
                randomized.agent,
                randomized.coordinator,
                splits[split_name],
                agent_config=agent_config,
                device=device,
            )
            metrics.append(
                evaluate_predictions(
                    method="trainable_shared_agent_coordinator",
                    condition="randomized_labels",
                    predictions=randomized_predictions,
                    batch=batch,
                    seed=seed,
                    param_count=randomized.param_count,
                    benchmark=BENCHMARK,
                    metadata={**_method_metadata(randomized), "train_labels_randomized": True},
                )
            )

        diagnostics.extend(_learning_curve_rows(frozen, splits, agent_config, device, seed))
        diagnostics.extend(_learning_curve_rows(trainable, splits, agent_config, device, seed))

    audit.append(summarize_test_access(audit))
    results = {
        "metadata": {
            "stage": "stage6a_shared_weight_cloned_agent_training",
            "config_path": str(Path(config_path)),
            "config": config,
            "strict_dataset_config": asdict(dataset_config),
            "shared_agent_config": asdict(agent_config),
            "shared_coordinator_config": asdict(coordinator_config),
            "shared_training_config": asdict(training_config),
            "device_used": device,
        },
        "metrics": metrics,
        "diagnostics": diagnostics,
        "audit": audit,
        "validation": {
            "shared_weight_cloned_agent": {
                "stage": "6A",
                "agent_model": "small_trainable_transformer_from_scratch",
                "clone_parameter_sharing": "same SmallTransformerSharedAgent module called for every clone",
                "loss": "single final group-answer cross entropy",
                "coordinator": "role-aware QKV self-attention over clone activations",
                "hidden_state_only_probe_role": "diagnostic_only",
            }
        },
    }

    output_path = Path(str(config["output_path"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    audit_path = Path(str(config.get("audit_log_path", "results/shared_weight_cloned_agent_audit.jsonl")))
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in audit_log_rows) + "\n",
        encoding="utf-8",
    )
    write_report(results, config["report_path"])
    print(f"wrote {output_path}")
    print(f"wrote {audit_path}")
    print(f"wrote {config['report_path']}")
    return results


def fit_shared_weight_model(
    splits: Dict[str, StrictCloneSplit],
    agent_config: SharedAgentConfig,
    coordinator_config: SharedCoordinatorConfig,
    training_config: SharedTrainingConfig,
    num_classes: int,
    seed: int,
    device: str,
    trainable_agent: bool,
    method: str,
    condition: str = "none",
    train_targets: np.ndarray | None = None,
    dev_targets: np.ndarray | None = None,
) -> FitResult:
    torch.manual_seed(seed + 16001)
    tokenizer = StrictCloneTokenizer(agent_config.n_agents)
    agent = SmallTransformerSharedAgent(tokenizer.vocab_size, agent_config).to(device)
    initialize_stage6_agent(agent, tokenizer, agent_config.evidence_embedding_scale)
    coordinator = RoleAwareQKVCoordinator(
        n_roles=agent_config.n_agents,
        hidden_dim=agent_config.hidden_dim,
        num_classes=num_classes,
        config=coordinator_config,
    ).to(device)
    if not trainable_agent:
        for parameter in agent.parameters():
            parameter.requires_grad_(False)

    shared_identity = shared_parameter_identity_check(agent, agent_config.n_agents)
    initial_agent = _flat_parameters(agent)
    initial_coordinator = _flat_parameters(coordinator)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in list(agent.parameters()) + list(coordinator.parameters()) if parameter.requires_grad],
        lr=training_config.lr,
        weight_decay=training_config.weight_decay,
    )
    train_y = np.asarray(train_targets if train_targets is not None else splits["train"].labels, dtype=np.int64)
    dev_y = np.asarray(dev_targets if dev_targets is not None else splits["dev"].labels, dtype=np.int64)
    rng = np.random.default_rng(seed + 91531)
    agent_grad_norms: List[float] = []
    coordinator_grad_norms: List[float] = []
    history: List[Dict[str, float]] = []
    checkpoints: List[Dict[str, Dict[str, torch.Tensor]]] = []
    best_state: Dict[str, Dict[str, torch.Tensor]] | None = None
    best_dev = -1.0
    stale_epochs = 0
    first_backward_audit: Dict[str, object] = {
        "clone_activation_requires_grad": False,
        "loss_backward_reaches_agent": False,
        "per_clone_activation_grad_norms": [0.0 for _ in range(agent_config.n_agents)],
        "per_clone_gradient_contribution": False,
    }

    train_bits_tensor = torch.as_tensor(splits["train"].evidence_bits, dtype=torch.long, device=device)
    train_targets_tensor = torch.as_tensor(train_y, dtype=torch.long, device=device)
    for epoch in range(training_config.epochs):
        agent.train(trainable_agent)
        coordinator.train()
        permutation = rng.permutation(splits["train"].n_examples)
        for start in range(0, splits["train"].n_examples, training_config.batch_size):
            batch_indices = permutation[start : start + training_config.batch_size]
            idx = torch.as_tensor(batch_indices, dtype=torch.long, device=device)
            bits = train_bits_tensor.index_select(0, idx)
            targets = train_targets_tensor.index_select(0, idx)
            clone_activations = run_shared_agent_clones(agent, _tokens_from_bits(bits, tokenizer))
            record_backward = trainable_agent and not bool(first_backward_audit["clone_activation_requires_grad"])
            if record_backward:
                clone_activations.retain_grad()
            role_ids = _role_ids(bits.shape[0], agent_config.n_agents, device)
            logits = coordinator(clone_activations, role_ids)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            agent_grad_norm = _grad_norm(agent.parameters())
            coordinator_grad_norm = _grad_norm(coordinator.parameters())
            agent_grad_norms.append(agent_grad_norm)
            coordinator_grad_norms.append(coordinator_grad_norm)
            if record_backward:
                per_clone = [
                    float(clone_activations.grad[:, clone_id, :].norm().detach().cpu().item())
                    for clone_id in range(agent_config.n_agents)
                ]
                first_backward_audit = {
                    "clone_activation_requires_grad": bool(clone_activations.requires_grad),
                    "loss_backward_reaches_agent": bool(agent_grad_norm > 0.0),
                    "per_clone_activation_grad_norms": per_clone,
                    "per_clone_gradient_contribution": all(value > 0.0 for value in per_clone),
                }
            optimizer.step()

        checkpoint = _checkpoint(agent, coordinator)
        checkpoints.append(checkpoint)
        train_acc = _accuracy_for_split(agent, coordinator, splits["train"], agent_config, device)
        dev_acc = _accuracy_for_split(agent, coordinator, splits["dev"], agent_config, device)
        dev_select_acc = _accuracy_for_split(agent, coordinator, splits["dev"], agent_config, device, labels=dev_y)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_acc": train_acc,
                "dev_acc": dev_acc,
                "dev_selection_acc": dev_select_acc,
            }
        )
        if dev_select_acc > best_dev:
            best_dev = dev_select_acc
            best_state = checkpoint
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= training_config.patience:
                break

    if best_state is not None:
        _load_checkpoint(agent, coordinator, best_state, device)
    agent_delta = _parameter_delta(agent, initial_agent)
    coordinator_delta = _parameter_delta(coordinator, initial_coordinator)
    param_count = int(
        sum(parameter.numel() for parameter in coordinator.parameters())
        + sum(parameter.numel() for parameter in agent.parameters() if parameter.requires_grad)
    )
    audit = {
        "benchmark": BENCHMARK,
        "seed": seed,
        "split": "dev",
        "probe": "shared_weight_training_audit",
        "method": method,
        "condition": condition,
        "shared_parameter_identity": shared_identity["pass"],
        "shared_parameter_count": shared_identity["parameter_count"],
        "agent_trainable": trainable_agent,
        "agent_grad_norm_mean": _mean(agent_grad_norms),
        "agent_grad_norm_std": _std(agent_grad_norms),
        "coordinator_grad_norm_mean": _mean(coordinator_grad_norms),
        "coordinator_grad_norm_std": _std(coordinator_grad_norms),
        "agent_parameter_delta": agent_delta,
        "coordinator_parameter_delta": coordinator_delta,
        "clone_activation_requires_grad": bool(first_backward_audit["clone_activation_requires_grad"]),
        "loss_backward_reaches_agent": bool(first_backward_audit["loss_backward_reaches_agent"]),
        "per_clone_gradient_contribution": bool(first_backward_audit["per_clone_gradient_contribution"]),
        "per_clone_activation_grad_norms": first_backward_audit["per_clone_activation_grad_norms"],
        "epochs_run": len(history),
        "best_dev_selection_acc": best_dev,
        "param_count": param_count,
    }
    return FitResult(
        method=method,
        agent=agent,
        coordinator=coordinator,
        trainable_agent=trainable_agent,
        param_count=param_count,
        audit=audit,
        history=history,
        checkpoints=checkpoints,
    )


def run_shared_agent_clones(agent: SmallTransformerSharedAgent, clone_tokens: torch.Tensor) -> torch.Tensor:
    """Run the same shared module once per clone and stack final-token states."""

    clone_outputs = []
    for clone_id in range(clone_tokens.shape[1]):
        clone_outputs.append(agent(clone_tokens[:, clone_id, :]))
    return torch.stack(clone_outputs, dim=1)


def shared_parameter_identity_check(agent: SmallTransformerSharedAgent, n_clones: int) -> Dict[str, object]:
    ids_by_clone = [[id(parameter) for parameter in agent.parameters()] for _ in range(n_clones)]
    first = ids_by_clone[0]
    return {
        "pass": all(ids == first for ids in ids_by_clone[1:]),
        "parameter_count": len(first),
        "clone_parameter_ids": ids_by_clone,
    }


def predict_shared_weight_model(
    agent: SmallTransformerSharedAgent,
    coordinator: RoleAwareQKVCoordinator,
    split: StrictCloneSplit,
    agent_config: SharedAgentConfig,
    device: str,
    condition: str = "none",
    seed: int = 0,
) -> np.ndarray:
    logits = _shared_weight_logits(agent, coordinator, split, agent_config, device, condition=condition, seed=seed)
    return torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64)


def capture_hidden_batches(
    agent: SmallTransformerSharedAgent,
    splits: Dict[str, StrictCloneSplit],
    agent_config: SharedAgentConfig,
    device: str,
) -> Dict[str, AttemptBatch]:
    return {
        split_name: _attempt_batch_from_split(
            split,
            hidden_dim=agent_config.hidden_dim,
            hidden_states=_capture_hidden(agent, split, agent_config, device),
        )
        for split_name, split in splits.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 6A shared-weight cloned-agent training.")
    parser.add_argument("--config", default="configs/shared_weight_cloned_agent_cpu.json")
    args = parser.parse_args()
    run_experiment(args.config)


def _shared_weight_logits(
    agent: SmallTransformerSharedAgent,
    coordinator: RoleAwareQKVCoordinator,
    split: StrictCloneSplit,
    agent_config: SharedAgentConfig,
    device: str,
    condition: str = "none",
    seed: int = 0,
) -> torch.Tensor:
    agent.eval()
    coordinator.eval()
    rng = np.random.default_rng(seed + 67121)
    evidence_bits = split.evidence_bits
    mask_evidence = condition == "evidence_masked"
    if condition == "evidence_shuffled":
        evidence_bits = _shuffle_evidence_bits(evidence_bits, seed + 991)
    tokenizer = StrictCloneTokenizer(agent_config.n_agents)
    rows: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, split.n_examples, 2048):
            bits = torch.as_tensor(evidence_bits[start : start + 2048], dtype=torch.long, device=device)
            clone_activations = run_shared_agent_clones(
                agent,
                _tokens_from_bits(bits, tokenizer, mask_evidence=mask_evidence),
            )
            role_ids = _role_ids(bits.shape[0], agent_config.n_agents, device)
            if condition == "hidden_states_shuffled_across_examples" and clone_activations.shape[0] > 1:
                perm = torch.as_tensor(rng.permutation(clone_activations.shape[0]), dtype=torch.long, device=device)
                if bool(torch.all(perm == torch.arange(clone_activations.shape[0], device=device)).item()):
                    perm = torch.roll(perm, shifts=1)
                clone_activations = clone_activations.index_select(0, perm)
            elif condition == "slot_labels_shuffled":
                role_ids = _shuffle_roles(role_ids, rng, preserve_pairing=False)
            elif condition == "physical_order_shuffled_roles_preserved":
                order = _role_permutation(role_ids.shape[0], role_ids.shape[1], rng, device)
                clone_activations = clone_activations.gather(
                    1,
                    order.unsqueeze(-1).expand(-1, -1, clone_activations.shape[-1]),
                )
                role_ids = role_ids.gather(1, order)
            rows.append(coordinator(clone_activations, role_ids).detach().cpu())
    return torch.cat(rows, dim=0)


def _capture_hidden(
    agent: SmallTransformerSharedAgent,
    split: StrictCloneSplit,
    agent_config: SharedAgentConfig,
    device: str,
) -> np.ndarray:
    agent.eval()
    tokenizer = StrictCloneTokenizer(agent_config.n_agents)
    hidden_rows: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, split.n_examples, 2048):
            bits = torch.as_tensor(split.evidence_bits[start : start + 2048], dtype=torch.long, device=device)
            hidden = run_shared_agent_clones(agent, _tokens_from_bits(bits, tokenizer))
            hidden_rows.append(hidden.unsqueeze(2).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(hidden_rows, axis=0)


def _accuracy_for_split(
    agent: SmallTransformerSharedAgent,
    coordinator: RoleAwareQKVCoordinator,
    split: StrictCloneSplit,
    agent_config: SharedAgentConfig,
    device: str,
    labels: np.ndarray | None = None,
) -> float:
    logits = _shared_weight_logits(agent, coordinator, split, agent_config, device)
    targets = split.labels if labels is None else np.asarray(labels, dtype=np.int64)
    predictions = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64)
    return float(np.mean(predictions == targets.astype(np.int64)))


def _tokens_from_bits(
    bits: torch.Tensor,
    tokenizer: StrictCloneTokenizer,
    mask_evidence: bool = False,
) -> torch.Tensor:
    tokens = torch.empty((bits.shape[0], tokenizer.n_agents, 4), dtype=torch.long, device=bits.device)
    for role_id in range(tokenizer.n_agents):
        tokens[:, role_id, 0] = tokenizer.role_offset + role_id
        tokens[:, role_id, 1] = tokenizer.slot_offset + role_id
        tokens[:, role_id, 2] = tokenizer.separator_token_id
        if mask_evidence:
            tokens[:, role_id, 3] = tokenizer.mask_token_id
        else:
            tokens[:, role_id, 3] = torch.where(
                bits[:, role_id] == 1,
                torch.as_tensor(tokenizer.bit_one_token_id, dtype=torch.long, device=bits.device),
                torch.as_tensor(tokenizer.bit_zero_token_id, dtype=torch.long, device=bits.device),
            )
    return tokens


def _role_ids(batch_size: int, n_agents: int, device: str | torch.device) -> torch.Tensor:
    return torch.arange(n_agents, dtype=torch.long, device=device).view(1, n_agents).expand(batch_size, n_agents)


def _shuffle_roles(role_ids: torch.Tensor, rng: np.random.Generator, preserve_pairing: bool) -> torch.Tensor:
    del preserve_pairing
    order = _role_permutation(role_ids.shape[0], role_ids.shape[1], rng, role_ids.device)
    shuffled = role_ids.gather(1, order)
    unchanged = torch.all(shuffled == role_ids, dim=1)
    if bool(unchanged.any().item()) and role_ids.shape[1] > 1:
        shuffled[unchanged] = torch.roll(shuffled[unchanged], shifts=1, dims=1)
    return shuffled


def _role_permutation(batch_size: int, n_agents: int, rng: np.random.Generator, device: str | torch.device) -> torch.Tensor:
    order = np.zeros((batch_size, n_agents), dtype=np.int64)
    base = np.arange(n_agents)
    for row_id in range(batch_size):
        perm = rng.permutation(n_agents)
        if n_agents > 1 and np.all(perm == base):
            perm = np.roll(perm, 1)
        order[row_id] = perm
    return torch.as_tensor(order, dtype=torch.long, device=device)


def _learning_curve_rows(
    result: FitResult,
    splits: Dict[str, StrictCloneSplit],
    agent_config: SharedAgentConfig,
    device: str,
    seed: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for item in result.history:
        for split_name, field in (("train", "train_acc"), ("dev", "dev_acc")):
            rows.append(
                {
                    "benchmark": BENCHMARK,
                    "seed": seed,
                    "split": split_name,
                    "probe": "shared_weight_learning_curve",
                    "method": result.method,
                    "condition": "none",
                    "epoch": int(item["epoch"]),
                    "accuracy": float(item[field]),
                }
            )
    for epoch_index, checkpoint in enumerate(result.checkpoints, start=1):
        _load_checkpoint(result.agent, result.coordinator, checkpoint, device)
        rows.append(
            {
                "benchmark": BENCHMARK,
                "seed": seed,
                "split": "test",
                "probe": "shared_weight_learning_curve",
                "method": result.method,
                "condition": "none",
                "epoch": epoch_index,
                "accuracy": _accuracy_for_split(result.agent, result.coordinator, splits["test"], agent_config, device),
            }
        )
    if result.checkpoints:
        best_epoch = max(result.history, key=lambda row: row["dev_selection_acc"])["epoch"]
        best_checkpoint = result.checkpoints[int(best_epoch) - 1]
        _load_checkpoint(result.agent, result.coordinator, best_checkpoint, device)
    return rows


def _attempt_batch_from_split(
    split: StrictCloneSplit,
    hidden_dim: int,
    hidden_states: np.ndarray | None = None,
) -> AttemptBatch:
    if hidden_states is None:
        hidden_states = np.zeros((split.n_examples, split.n_agents, 1, hidden_dim), dtype=np.float32)
    answers = np.zeros((split.n_examples, split.n_agents), dtype=np.int64)
    confidences = np.full((split.n_examples, split.n_agents), 0.25, dtype=np.float32)
    visible_texts = [
        [
            f"agent={agent_id} task=0 answer=0 confidence=0.250 status=withheld_private_evidence"
            for agent_id in range(split.n_agents)
        ]
        for _ in range(split.n_examples)
    ]
    private_agent_views = [
        [
            (
                f"Agent {agent_id}; private slot {agent_id}; "
                f"private evidence token BIT_{'ONE' if int(bits[agent_id]) else 'ZERO'}; "
                "visible reply STATUS_UNKNOWN"
            )
            for agent_id in range(split.n_agents)
        ]
        for bits in split.evidence_bits
    ]
    return AttemptBatch(
        split=split.split,
        example_ids=split.example_ids.copy(),
        task_ids=split.task_ids.copy(),
        labels=split.labels.copy(),
        answers=answers,
        confidences=confidences,
        hidden_states=hidden_states.astype(np.float32, copy=True),
        visible_texts=visible_texts,
        private_agent_views=private_agent_views,
    )


def _clone_split(
    split_name: str,
    examples: Sequence[StrictCoordinationExample],
    n_agents: int,
) -> StrictCloneSplit:
    bits = np.asarray([example.evidence_bits[:n_agents] for example in examples], dtype=np.int64)
    return StrictCloneSplit(
        split=split_name,
        example_ids=np.asarray([example.example_id for example in examples], dtype=object),
        task_ids=np.asarray([example.task_id for example in examples], dtype=np.int64),
        evidence_bits=bits,
        labels=np.asarray([example.label for example in examples], dtype=np.int64),
    )


def _oracle_predictions(split: StrictCloneSplit) -> np.ndarray:
    return np.asarray([strict_label(tuple(int(value) for value in row)) for row in split.evidence_bits], dtype=np.int64)


def _shuffle_evidence_bits(bits: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shuffled = bits.copy()
    for slot_id in range(bits.shape[1]):
        perm = rng.permutation(bits.shape[0])
        if bits.shape[0] > 1 and np.all(perm == np.arange(bits.shape[0])):
            perm = np.roll(perm, 1)
        shuffled[:, slot_id] = bits[perm, slot_id]
    return shuffled.astype(np.int64, copy=False)


def _balanced_randomized_labels(split: StrictCloneSplit, num_classes: int, seed: int) -> np.ndarray:
    """Random labels balanced within each identical private-evidence pattern."""

    rng = np.random.default_rng(seed)
    labels = np.empty_like(split.labels)
    patterns: Dict[tuple[int, ...], List[int]] = {}
    for row_id, bits in enumerate(split.evidence_bits):
        patterns.setdefault(tuple(int(value) for value in bits), []).append(row_id)
    for row_ids in patterns.values():
        random_cycle = np.tile(np.arange(num_classes, dtype=np.int64), int(np.ceil(len(row_ids) / num_classes)))
        random_cycle = random_cycle[: len(row_ids)]
        rng.shuffle(random_cycle)
        labels[np.asarray(row_ids, dtype=np.int64)] = random_cycle
    return labels.astype(np.int64, copy=False)


def _method_metadata(result: FitResult) -> Dict[str, object]:
    return {
        "agent_trainable": result.trainable_agent,
        "agent_parameter_delta": result.audit["agent_parameter_delta"],
        "coordinator_parameter_delta": result.audit["coordinator_parameter_delta"],
        "agent_grad_norm_mean": result.audit["agent_grad_norm_mean"],
        "coordinator_grad_norm_mean": result.audit["coordinator_grad_norm_mean"],
        "shared_parameter_identity": result.audit["shared_parameter_identity"],
    }


def _checkpoint(
    agent: SmallTransformerSharedAgent,
    coordinator: RoleAwareQKVCoordinator,
) -> Dict[str, Dict[str, torch.Tensor]]:
    return {
        "agent": {key: value.detach().cpu().clone() for key, value in agent.state_dict().items()},
        "coordinator": {key: value.detach().cpu().clone() for key, value in coordinator.state_dict().items()},
    }


def _load_checkpoint(
    agent: SmallTransformerSharedAgent,
    coordinator: RoleAwareQKVCoordinator,
    checkpoint: Dict[str, Dict[str, torch.Tensor]],
    device: str,
) -> None:
    agent.load_state_dict(checkpoint["agent"])
    coordinator.load_state_dict(checkpoint["coordinator"])
    agent.to(device)
    coordinator.to(device)


def _flat_parameters(module: nn.Module) -> torch.Tensor:
    values = [parameter.detach().cpu().reshape(-1).float() for parameter in module.parameters()]
    if not values:
        return torch.zeros(0)
    return torch.cat(values)


def _parameter_delta(module: nn.Module, initial: torch.Tensor) -> float:
    current = _flat_parameters(module)
    if current.numel() != initial.numel():
        raise ValueError("parameter shape changed while computing delta")
    return float(torch.linalg.vector_norm(current - initial).item())


def _grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        total += float(parameter.grad.detach().float().pow(2).sum().cpu().item())
    return float(total ** 0.5)


def _compatible_heads(hidden_dim: int, requested_heads: int) -> int:
    requested_heads = max(1, int(requested_heads))
    if hidden_dim % requested_heads == 0:
        return requested_heads
    for heads in range(requested_heads, 0, -1):
        if hidden_dim % heads == 0:
            return heads
    return 1


def _make_dataclass(cls, data: object):
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in dict(data).items() if key in allowed})


def _make_mlp_training_config(data: object) -> MLPTrainingConfig:
    values = dict(data)
    hidden_dims = values.get("hidden_dims", (32,))
    return MLPTrainingConfig(
        epochs=int(values.get("epochs", 20)),
        batch_size=int(values.get("batch_size", 128)),
        lr=float(values.get("lr", 0.003)),
        weight_decay=float(values.get("weight_decay", 0.0001)),
        patience=int(values.get("patience", 5)),
        hidden_dims=tuple(int(value) for value in hidden_dims),
    )


def _resolve_device(requested: str) -> str:
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _split_offset(split: str) -> int:
    return {"train": 0, "dev": 10_000, "test": 20_000}.get(split, 30_000)


def _mean(values: List[float]) -> float:
    return mean(values) if values else 0.0


def _std(values: List[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


if __name__ == "__main__":
    main()
