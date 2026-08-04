"""Deliverable 1 training loop and reporting helpers for DIGIT Extrapolation E23."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .config import E23Config
from .initialisation import E23InitialisedModels, initialise_e23_models, set_global_determinism
from .models import ActiveLearnedMemoryMLP, BaselineMLP, DualParameterAttentionMLP

WITHHELD_PART_A_VALUES = (0, 1, 2, 3)
WITHHELD_PART_B_VALUES = (4, 5, 6, 7)
SEEDS = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class CombinationRecord:
    """One combination in the frozen E23 compositional grid."""

    part_a: int
    part_b: int
    class_id: int
    is_withheld: bool


@dataclass(frozen=True)
class SplitTensors:
    """Tensorized split for one E23 run."""

    part_a: torch.Tensor
    part_b: torch.Tensor
    part_a_noise: torch.Tensor
    part_b_noise: torch.Tensor
    labels: torch.Tensor
    combo_part_a: torch.Tensor
    combo_part_b: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.labels.shape[0])


@dataclass(frozen=True)
class E23Dataset:
    """Train/test tensors for one seeded E23 run."""

    train: SplitTensors
    test: SplitTensors


@dataclass(frozen=True)
class EpochCheckpointMetric:
    """One seed-0 withheld accuracy checkpoint."""

    label: str
    epoch: int
    withheld_macro_accuracy: float


@dataclass(frozen=True)
class SeedTrainingResult:
    """Deliverable 1 outputs for one arm and one seed."""

    arm_name: str
    seed: int
    parameter_count: int
    final_training_loss: float
    final_training_accuracy: float
    stopping_epoch: int
    withheld_macro_accuracy: float


@dataclass
class SeedRunArtifacts:
    """Trained model, dataset, and requested Deliverable 1 metrics."""

    config: E23Config
    arm_name: str
    model: torch.nn.Module
    dataset: E23Dataset
    result: SeedTrainingResult
    seed0_checkpoint_metrics: tuple[EpochCheckpointMetric, ...] | None


@dataclass(frozen=True)
class WithheldEvaluationSummary:
    """Post-training withheld evaluation outputs for one model and one seed."""

    macro_accuracy: float
    mean_cross_entropy: float
    per_combination_accuracy: dict[tuple[int, int], float]


def fixed_class_id(part_a: int, part_b: int) -> int:
    """Return the frozen E23 class label."""
    return (part_a + part_b) % 4


def is_withheld_combination(part_a: int, part_b: int) -> bool:
    """Return whether the combination is in the frozen withheld quadrant."""
    return part_a in WITHHELD_PART_A_VALUES and part_b in WITHHELD_PART_B_VALUES


def build_combination_records() -> tuple[CombinationRecord, ...]:
    """Build the full frozen 8x8 grid for E23."""
    return tuple(
        CombinationRecord(
            part_a=part_a,
            part_b=part_b,
            class_id=fixed_class_id(part_a, part_b),
            is_withheld=is_withheld_combination(part_a, part_b),
        )
        for part_a in range(8)
        for part_b in range(8)
    )


def resolve_device(config: E23Config) -> torch.device:
    """Resolve the configured device and fail fast on unavailable CUDA requests."""
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "E23Config.device is set to CUDA, but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled PyTorch build or override device='cpu'."
        )
    return device


def build_dataset(config: E23Config) -> E23Dataset:
    """Materialize the frozen quadrant split with fixed train/test noise."""
    records = build_combination_records()
    train_noise_generator = torch.Generator()
    train_noise_generator.manual_seed(config.seed + config.train_noise_seed_offset)
    test_noise_generator = torch.Generator()
    test_noise_generator.manual_seed(config.seed + config.test_noise_seed_offset)

    train_part_a: list[int] = []
    train_part_b: list[int] = []
    train_part_a_noise: list[torch.Tensor] = []
    train_part_b_noise: list[torch.Tensor] = []
    train_labels: list[int] = []
    train_combo_part_a: list[int] = []
    train_combo_part_b: list[int] = []

    test_part_a: list[int] = []
    test_part_b: list[int] = []
    test_part_a_noise: list[torch.Tensor] = []
    test_part_b_noise: list[torch.Tensor] = []
    test_labels: list[int] = []
    test_combo_part_a: list[int] = []
    test_combo_part_b: list[int] = []

    for record in records:
        if record.is_withheld:
            for _ in range(config.test_repeats_per_combo):
                test_part_a.append(record.part_a)
                test_part_b.append(record.part_b)
                test_part_a_noise.append(
                    torch.randn(config.embedding_dim, generator=test_noise_generator) * config.input_jitter_std
                )
                test_part_b_noise.append(
                    torch.randn(config.embedding_dim, generator=test_noise_generator) * config.input_jitter_std
                )
                test_labels.append(record.class_id)
                test_combo_part_a.append(record.part_a)
                test_combo_part_b.append(record.part_b)
            continue

        for _ in range(config.train_repeats_per_combo):
            train_part_a.append(record.part_a)
            train_part_b.append(record.part_b)
            train_part_a_noise.append(
                torch.randn(config.embedding_dim, generator=train_noise_generator) * config.input_jitter_std
            )
            train_part_b_noise.append(
                torch.randn(config.embedding_dim, generator=train_noise_generator) * config.input_jitter_std
            )
            train_labels.append(record.class_id)
            train_combo_part_a.append(record.part_a)
            train_combo_part_b.append(record.part_b)

    return E23Dataset(
        train=SplitTensors(
            part_a=torch.tensor(train_part_a, dtype=torch.long),
            part_b=torch.tensor(train_part_b, dtype=torch.long),
            part_a_noise=torch.stack(train_part_a_noise).to(dtype=torch.float32),
            part_b_noise=torch.stack(train_part_b_noise).to(dtype=torch.float32),
            labels=torch.tensor(train_labels, dtype=torch.long),
            combo_part_a=torch.tensor(train_combo_part_a, dtype=torch.long),
            combo_part_b=torch.tensor(train_combo_part_b, dtype=torch.long),
        ),
        test=SplitTensors(
            part_a=torch.tensor(test_part_a, dtype=torch.long),
            part_b=torch.tensor(test_part_b, dtype=torch.long),
            part_a_noise=torch.stack(test_part_a_noise).to(dtype=torch.float32),
            part_b_noise=torch.stack(test_part_b_noise).to(dtype=torch.float32),
            labels=torch.tensor(test_labels, dtype=torch.long),
            combo_part_a=torch.tensor(test_combo_part_a, dtype=torch.long),
            combo_part_b=torch.tensor(test_combo_part_b, dtype=torch.long),
        ),
    )


def make_loader(split: SplitTensors, config: E23Config, *, shuffle: bool) -> DataLoader[tuple[torch.Tensor, ...]]:
    """Create a deterministic DataLoader for one split."""
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    dataset = TensorDataset(
        split.part_a,
        split.part_b,
        split.part_a_noise,
        split.part_b_noise,
        split.labels,
        split.combo_part_a,
        split.combo_part_b,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        pin_memory=config.device.startswith("cuda"),
    )


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Return the exact number of trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _move_batch_to_device(
    batch: tuple[torch.Tensor, ...],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    return tuple(item.to(device, non_blocking=device.type == "cuda") for item in batch)


def _forward_model(
    model: torch.nn.Module,
    part_a: torch.Tensor,
    part_b: torch.Tensor,
    part_a_noise: torch.Tensor,
    part_b_noise: torch.Tensor,
):
    return model(
        part_a,
        part_b,
        part_a_noise=part_a_noise,
        part_b_noise=part_b_noise,
    )


def evaluate_withheld_macro_accuracy(
    model: torch.nn.Module,
    split: SplitTensors,
    config: E23Config,
) -> float:
    """Evaluate withheld-quadrant macro accuracy over the 16 held-out combinations."""
    device = resolve_device(config)
    loader = make_loader(split, config, shuffle=False)
    model.eval()

    correct_by_combo = {
        (part_a, part_b): 0
        for part_a in WITHHELD_PART_A_VALUES
        for part_b in WITHHELD_PART_B_VALUES
    }
    total_by_combo = {
        (part_a, part_b): 0
        for part_a in WITHHELD_PART_A_VALUES
        for part_b in WITHHELD_PART_B_VALUES
    }

    with torch.no_grad():
        for batch in loader:
            part_a, part_b, part_a_noise, part_b_noise, labels, combo_part_a, combo_part_b = _move_batch_to_device(
                batch,
                device,
            )
            outputs = _forward_model(model, part_a, part_b, part_a_noise, part_b_noise)
            predictions = outputs.logits.argmax(dim=-1).cpu()
            labels_cpu = labels.cpu()
            combo_part_a_cpu = combo_part_a.cpu()
            combo_part_b_cpu = combo_part_b.cpu()

            for index in range(labels_cpu.size(0)):
                key = (int(combo_part_a_cpu[index].item()), int(combo_part_b_cpu[index].item()))
                total_by_combo[key] += 1
                correct_by_combo[key] += int(predictions[index].item() == labels_cpu[index].item())

    combo_accuracies = [
        correct_by_combo[key] / total_by_combo[key]
        for key in sorted(total_by_combo)
    ]
    return float(mean(combo_accuracies))


def evaluate_withheld_summary(
    model: torch.nn.Module,
    split: SplitTensors,
    config: E23Config,
) -> WithheldEvaluationSummary:
    """Evaluate withheld macro accuracy, mean cross-entropy, and per-combination accuracies."""
    device = resolve_device(config)
    loader = make_loader(split, config, shuffle=False)
    model.eval()

    correct_by_combo = {
        (part_a, part_b): 0
        for part_a in WITHHELD_PART_A_VALUES
        for part_b in WITHHELD_PART_B_VALUES
    }
    total_by_combo = {
        (part_a, part_b): 0
        for part_a in WITHHELD_PART_A_VALUES
        for part_b in WITHHELD_PART_B_VALUES
    }
    total_examples = 0
    total_cross_entropy = 0.0

    with torch.no_grad():
        for batch in loader:
            part_a, part_b, part_a_noise, part_b_noise, labels, combo_part_a, combo_part_b = _move_batch_to_device(
                batch,
                device,
            )
            outputs = _forward_model(model, part_a, part_b, part_a_noise, part_b_noise)
            loss = F.cross_entropy(outputs.logits, labels, reduction="sum")
            total_cross_entropy += float(loss.item())
            total_examples += labels.size(0)

            predictions = outputs.logits.argmax(dim=-1).cpu()
            labels_cpu = labels.cpu()
            combo_part_a_cpu = combo_part_a.cpu()
            combo_part_b_cpu = combo_part_b.cpu()

            for index in range(labels_cpu.size(0)):
                key = (int(combo_part_a_cpu[index].item()), int(combo_part_b_cpu[index].item()))
                total_by_combo[key] += 1
                correct_by_combo[key] += int(predictions[index].item() == labels_cpu[index].item())

    per_combination_accuracy = {
        key: correct_by_combo[key] / total_by_combo[key]
        for key in sorted(total_by_combo)
    }
    macro_accuracy = float(mean(list(per_combination_accuracy.values())))
    return WithheldEvaluationSummary(
        macro_accuracy=macro_accuracy,
        mean_cross_entropy=total_cross_entropy / total_examples,
        per_combination_accuracy=per_combination_accuracy,
    )


def _train_one_arm(
    *,
    config: E23Config,
    arm_name: str,
    model: torch.nn.Module,
    dataset: E23Dataset,
    update_mode: str,
) -> SeedRunArtifacts:
    """Train one arm for one seed and collect Deliverable 1 metrics."""
    if update_mode not in {"baseline", "e23", "ablation"}:
        raise ValueError("update_mode must be 'baseline', 'e23', or 'ablation'")

    device = resolve_device(config)
    model = model.to(device)
    if update_mode == "e23":
        assert isinstance(model, DualParameterAttentionMLP)
        model.layer1.record_update_inputs = False
        model.layer2.record_update_inputs = False
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_loader = make_loader(dataset.train, config, shuffle=True)

    best_loss = float("inf")
    epochs_without_meaningful_decrease = 0
    final_training_loss = float("inf")
    final_training_accuracy = 0.0
    stopping_epoch = 0
    seed0_checkpoints: list[EpochCheckpointMetric] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_correct = 0
        epoch_examples = 0

        for batch in train_loader:
            part_a, part_b, part_a_noise, part_b_noise, labels, _combo_part_a, _combo_part_b = _move_batch_to_device(
                batch,
                device,
            )

            optimizer.zero_grad(set_to_none=True)
            outputs = _forward_model(model, part_a, part_b, part_a_noise, part_b_noise)
            loss = F.cross_entropy(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            if update_mode == "e23":
                assert isinstance(model, DualParameterAttentionMLP)
                model.update_prototypes(outputs.layer1_semantic, outputs.layer2_semantic)

            batch_size = labels.size(0)
            epoch_loss_sum += float(loss.item()) * batch_size
            predictions = outputs.logits.argmax(dim=-1)
            epoch_correct += int((predictions == labels).sum().item())
            epoch_examples += batch_size

        final_training_loss = epoch_loss_sum / epoch_examples
        final_training_accuracy = epoch_correct / epoch_examples
        stopping_epoch = epoch

        if config.seed == 0 and epoch in config.seed0_report_epochs:
            seed0_checkpoints.append(
                EpochCheckpointMetric(
                    label=f"epoch_{epoch}",
                    epoch=epoch,
                    withheld_macro_accuracy=evaluate_withheld_macro_accuracy(model, dataset.test, config),
                )
            )

        if best_loss - final_training_loss > config.loss_improvement_tolerance:
            best_loss = final_training_loss
            epochs_without_meaningful_decrease = 0
        else:
            epochs_without_meaningful_decrease += 1

        if epochs_without_meaningful_decrease >= config.loss_patience_epochs:
            break

    final_withheld_macro_accuracy = evaluate_withheld_macro_accuracy(model, dataset.test, config)
    if config.seed == 0:
        seed0_checkpoints.append(
            EpochCheckpointMetric(
                label="final",
                epoch=stopping_epoch,
                withheld_macro_accuracy=final_withheld_macro_accuracy,
            )
        )

    result = SeedTrainingResult(
        arm_name=arm_name,
        seed=config.seed,
        parameter_count=count_trainable_parameters(model),
        final_training_loss=final_training_loss,
        final_training_accuracy=final_training_accuracy,
        stopping_epoch=stopping_epoch,
        withheld_macro_accuracy=final_withheld_macro_accuracy,
    )
    return SeedRunArtifacts(
        config=config,
        arm_name=arm_name,
        model=model,
        dataset=dataset,
        result=result,
        seed0_checkpoint_metrics=tuple(seed0_checkpoints) if config.seed == 0 else None,
    )


def train_three_arm_single_seed(config: E23Config) -> tuple[SeedRunArtifacts, SeedRunArtifacts, SeedRunArtifacts]:
    """Train baseline, E23, and learned-memory ablation for one seed."""
    set_global_determinism(config.seed)
    dataset = build_dataset(config)
    initialized: E23InitialisedModels = initialise_e23_models(config)

    baseline_run = _train_one_arm(
        config=config,
        arm_name="baseline",
        model=initialized.baseline,
        dataset=dataset,
        update_mode="baseline",
    )
    e23_run = _train_one_arm(
        config=config,
        arm_name="e23",
        model=initialized.e23,
        dataset=dataset,
        update_mode="e23",
    )
    ablation_run = _train_one_arm(
        config=config,
        arm_name="learned_memory_ablation",
        model=initialized.ablation,
        dataset=dataset,
        update_mode="ablation",
    )
    return baseline_run, e23_run, ablation_run


def run_deliverable1_suite(
    seeds: tuple[int, ...] | list[int] = SEEDS,
) -> tuple[list[SeedRunArtifacts], list[SeedRunArtifacts], list[SeedRunArtifacts]]:
    """Run Deliverable 1 training for all three arms across the requested seeds."""
    baseline_runs: list[SeedRunArtifacts] = []
    e23_runs: list[SeedRunArtifacts] = []
    ablation_runs: list[SeedRunArtifacts] = []

    for seed in seeds:
        baseline_run, e23_run, ablation_run = train_three_arm_single_seed(
            E23Config(
                seed=seed,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        )
        baseline_runs.append(baseline_run)
        e23_runs.append(e23_run)
        ablation_runs.append(ablation_run)

    return baseline_runs, e23_runs, ablation_runs


def summarise_metric(values: list[float]) -> dict[str, float]:
    """Return mean and population std for a list of floats."""
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)),
    }


def aggregate_per_combination_accuracy(
    summaries: list[WithheldEvaluationSummary],
) -> dict[tuple[int, int], dict[str, float]]:
    """Aggregate per-combination withheld accuracies across seeds."""
    aggregated: dict[tuple[int, int], dict[str, float]] = {}
    keys = [
        (part_a, part_b)
        for part_a in WITHHELD_PART_A_VALUES
        for part_b in WITHHELD_PART_B_VALUES
    ]
    for key in keys:
        values = [summary.per_combination_accuracy[key] for summary in summaries]
        aggregated[key] = summarise_metric(values)
    return aggregated
