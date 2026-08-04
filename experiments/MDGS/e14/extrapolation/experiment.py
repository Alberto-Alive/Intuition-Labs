"""Training and frozen-bank runners for DIGIT Extrapolation E14."""

from __future__ import annotations

import random
from csv import DictWriter
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .config import E14Config
from .data import build_combination_records
from .models import FrozenActivationBankMLP


@dataclass(frozen=True)
class SplitTensors:
    """Tensorized dataset split."""

    part_a: torch.Tensor
    part_b: torch.Tensor
    part_a_noise: torch.Tensor
    part_b_noise: torch.Tensor
    labels: torch.Tensor
    seen_unseen: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.labels.shape[0])


@dataclass(frozen=True)
class E14Dataset:
    """Train/test tensors for one seeded E14 run."""

    train: SplitTensors
    test: SplitTensors


@dataclass(frozen=True)
class EpochMetrics:
    """Training metrics for one epoch."""

    epoch: int
    loss: float
    accuracy: float


@dataclass(frozen=True)
class SeedTrainingResult:
    """Requested Deliverable 2 outputs for one seed."""

    seed: int
    stopping_epoch: int
    final_training_loss: float
    final_training_accuracy: float
    seen_test_accuracy: float
    unseen_test_accuracy: float
    seen_accuracy_substantially_above_chance: bool
    part_a_embedding_norm_mean: float
    part_b_embedding_norm_mean: float
    exact_seen_input_overlap_fraction: float


@dataclass
class TrainedSeedArtifacts:
    """Trained model plus dataset and stopping metrics for one seed."""

    config: E14Config
    model: FrozenActivationBankMLP
    dataset: E14Dataset
    result: SeedTrainingResult


@dataclass(frozen=True)
class FrozenBank:
    """Frozen activation bank built from one fully trained model."""

    layer1_bank: torch.Tensor
    layer2_bank: torch.Tensor


@dataclass(frozen=True)
class FrozenBankSummary:
    """Deliverable 3 reporting payload for one seed."""

    seed: int
    layer1_shape: tuple[int, int]
    layer2_shape: tuple[int, int]
    layer1_norm_mean: float
    layer1_norm_std: float
    layer2_norm_mean: float
    layer2_norm_std: float


@dataclass(frozen=True)
class TestExampleLogRecord:
    """Per-example logging payload required by the E14 specification."""

    seed: int
    part_a: int
    part_b: int
    target_label: int
    seen_unseen_label: int
    correctness: int
    predicted_class: int
    output_confidence: float
    input_nn_distance: float
    layer1_certainty: float
    layer2_certainty: float
    part_a_marginal_frequency: int
    part_b_marginal_frequency: int


@dataclass(frozen=True)
class MetricSeenUnseenSummary:
    """Mean/std summary of one metric for seen vs unseen pooled examples."""

    seen_mean: float
    seen_std: float
    unseen_mean: float
    unseen_std: float


@dataclass(frozen=True)
class SeedCertaintyVisualSummary:
    """Per-seed certainty means requested before regression."""

    seed: int
    layer1_seen_mean: float
    layer1_unseen_mean: float
    layer2_seen_mean: float
    layer2_unseen_mean: float


@dataclass(frozen=True)
class LoggingSummary:
    """Deliverable 4 aggregate summary plus per-seed certainty means."""

    layer1_certainty: MetricSeenUnseenSummary
    layer2_certainty: MetricSeenUnseenSummary
    output_confidence: MetricSeenUnseenSummary
    input_nn_distance: MetricSeenUnseenSummary
    per_seed_visual: list[SeedCertaintyVisualSummary]


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def set_global_determinism(seed: int) -> None:
    """Set the approved seed across Python and PyTorch."""
    random.seed(seed)
    torch.manual_seed(seed)


def build_dataset(config: E14Config) -> E14Dataset:
    """Materialize train/test tensors from the approved compositional split."""
    records = build_combination_records(seed=config.seed)
    train_noise_generator = torch.Generator()
    train_noise_generator.manual_seed(config.seed + config.train_noise_seed_offset)
    test_noise_generator = torch.Generator()
    test_noise_generator.manual_seed(config.seed + config.test_noise_seed_offset)

    train_part_a: list[int] = []
    train_part_b: list[int] = []
    train_part_a_noise: list[torch.Tensor] = []
    train_part_b_noise: list[torch.Tensor] = []
    train_labels: list[int] = []
    train_seen_unseen: list[int] = []

    test_part_a: list[int] = []
    test_part_b: list[int] = []
    test_part_a_noise: list[torch.Tensor] = []
    test_part_b_noise: list[torch.Tensor] = []
    test_labels: list[int] = []
    test_seen_unseen: list[int] = []

    for record in records:
        if not record.is_withheld:
            for _ in range(config.train_repeats_per_seen_combo):
                train_part_a.append(record.part_a)
                train_part_b.append(record.part_b)
                train_part_a_noise.append(
                    torch.randn(config.embedding_dim, generator=train_noise_generator) * config.input_jitter_std
                )
                train_part_b_noise.append(
                    torch.randn(config.embedding_dim, generator=train_noise_generator) * config.input_jitter_std
                )
                train_labels.append(record.class_id)
                train_seen_unseen.append(0)

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
            test_seen_unseen.append(int(record.is_withheld))

    return E14Dataset(
        train=SplitTensors(
            part_a=torch.tensor(train_part_a, dtype=torch.long),
            part_b=torch.tensor(train_part_b, dtype=torch.long),
            part_a_noise=torch.stack(train_part_a_noise).to(dtype=torch.float32),
            part_b_noise=torch.stack(train_part_b_noise).to(dtype=torch.float32),
            labels=torch.tensor(train_labels, dtype=torch.long),
            seen_unseen=torch.tensor(train_seen_unseen, dtype=torch.long),
        ),
        test=SplitTensors(
            part_a=torch.tensor(test_part_a, dtype=torch.long),
            part_b=torch.tensor(test_part_b, dtype=torch.long),
            part_a_noise=torch.stack(test_part_a_noise).to(dtype=torch.float32),
            part_b_noise=torch.stack(test_part_b_noise).to(dtype=torch.float32),
            labels=torch.tensor(test_labels, dtype=torch.long),
            seen_unseen=torch.tensor(test_seen_unseen, dtype=torch.long),
        ),
    )


def make_loader(split: SplitTensors, config: E14Config, *, shuffle: bool) -> DataLoader[tuple[torch.Tensor, ...]]:
    """Create a deterministic DataLoader for one split."""
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    dataset = TensorDataset(
        split.part_a,
        split.part_b,
        split.part_a_noise,
        split.part_b_noise,
        split.labels,
        split.seen_unseen,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def make_model(config: E14Config, *, bank_capacity: int | None = None) -> FrozenActivationBankMLP:
    """Instantiate the approved 2-layer MLP for one E14 run."""
    return FrozenActivationBankMLP(
        part_a_vocab_size=config.num_part_a_values,
        part_b_vocab_size=config.num_part_b_values,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
        bank_capacity=config.bank_capacity if bank_capacity is None else bank_capacity,
    )


def train_single_seed(config: E14Config) -> SeedTrainingResult:
    """Train the approved 2-layer MLP and return Deliverable 2 metrics."""
    return train_single_seed_artifacts(config).result


def train_single_seed_artifacts(config: E14Config) -> TrainedSeedArtifacts:
    """Train one seed and return the trained model plus materialized dataset."""
    set_global_determinism(config.seed)
    dataset = build_dataset(config)

    device = torch.device(config.device)
    model = make_model(config).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_loader = make_loader(dataset.train, config, shuffle=True)

    best_loss = float("inf")
    epochs_without_meaningful_decrease = 0
    final_epoch_metrics = EpochMetrics(epoch=0, loss=float("inf"), accuracy=0.0)

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_correct = 0
        epoch_examples = 0

        for part_a, part_b, part_a_noise, part_b_noise, labels, _ in train_loader:
            part_a = part_a.to(device)
            part_b = part_b.to(device)
            part_a_noise = part_a_noise.to(device)
            part_b_noise = part_b_noise.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                part_a,
                part_b,
                part_a_noise=part_a_noise,
                part_b_noise=part_b_noise,
                return_read_only_certainty=False,
            )
            loss = F.cross_entropy(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            epoch_loss_sum += float(loss.item()) * batch_size
            predictions = outputs.logits.argmax(dim=-1)
            epoch_correct += int((predictions == labels).sum().item())
            epoch_examples += batch_size

        epoch_loss = epoch_loss_sum / epoch_examples
        epoch_accuracy = epoch_correct / epoch_examples
        final_epoch_metrics = EpochMetrics(epoch=epoch, loss=epoch_loss, accuracy=epoch_accuracy)

        if best_loss - epoch_loss > config.loss_improvement_tolerance:
            best_loss = epoch_loss
            epochs_without_meaningful_decrease = 0
        else:
            epochs_without_meaningful_decrease += 1

        if epochs_without_meaningful_decrease >= config.loss_patience_epochs:
            break

    seen_test_accuracy, unseen_test_accuracy = evaluate_seen_unseen_accuracy(model, dataset.test, config)
    part_a_embedding_norm_mean = float(model.part_a_embedding.weight.norm(dim=-1).mean().item())
    part_b_embedding_norm_mean = float(model.part_b_embedding.weight.norm(dim=-1).mean().item())
    exact_seen_input_overlap_fraction = compute_exact_seen_input_overlap_fraction(dataset)

    result = SeedTrainingResult(
        seed=config.seed,
        stopping_epoch=final_epoch_metrics.epoch,
        final_training_loss=final_epoch_metrics.loss,
        final_training_accuracy=final_epoch_metrics.accuracy,
        seen_test_accuracy=seen_test_accuracy,
        unseen_test_accuracy=unseen_test_accuracy,
        seen_accuracy_substantially_above_chance=seen_test_accuracy > 0.25,
        part_a_embedding_norm_mean=part_a_embedding_norm_mean,
        part_b_embedding_norm_mean=part_b_embedding_norm_mean,
        exact_seen_input_overlap_fraction=exact_seen_input_overlap_fraction,
    )
    return TrainedSeedArtifacts(config=config, model=model, dataset=dataset, result=result)


def evaluate_seen_unseen_accuracy(
    model: FrozenActivationBankMLP,
    split: SplitTensors,
    config: E14Config,
) -> tuple[float, float]:
    """Evaluate accuracy separately on seen and unseen test combinations."""
    device = torch.device(config.device)
    loader = make_loader(split, config, shuffle=False)
    model.eval()

    seen_correct = 0
    seen_total = 0
    unseen_correct = 0
    unseen_total = 0

    with torch.no_grad():
        for part_a, part_b, part_a_noise, part_b_noise, labels, seen_unseen in loader:
            part_a = part_a.to(device)
            part_b = part_b.to(device)
            part_a_noise = part_a_noise.to(device)
            part_b_noise = part_b_noise.to(device)
            labels = labels.to(device)
            outputs = model(
                part_a,
                part_b,
                part_a_noise=part_a_noise,
                part_b_noise=part_b_noise,
                return_read_only_certainty=False,
            )
            predictions = outputs.logits.argmax(dim=-1).cpu()

            seen_mask = seen_unseen == 0
            unseen_mask = seen_unseen == 1

            if int(seen_mask.sum().item()) > 0:
                seen_correct += int((predictions[seen_mask] == labels.cpu()[seen_mask]).sum().item())
                seen_total += int(seen_mask.sum().item())
            if int(unseen_mask.sum().item()) > 0:
                unseen_correct += int((predictions[unseen_mask] == labels.cpu()[unseen_mask]).sum().item())
                unseen_total += int(unseen_mask.sum().item())

    seen_accuracy = seen_correct / seen_total if seen_total else 0.0
    unseen_accuracy = unseen_correct / unseen_total if unseen_total else 0.0
    return seen_accuracy, unseen_accuracy


def compute_exact_seen_input_overlap_fraction(dataset: E14Dataset) -> float:
    """Compute exact raw-input overlap between seen test examples and training examples."""
    train_examples = {
        (
            int(part_a),
            int(part_b),
            tuple(float(value) for value in part_a_noise.tolist()),
            tuple(float(value) for value in part_b_noise.tolist()),
        )
        for part_a, part_b, part_a_noise, part_b_noise in zip(
            dataset.train.part_a.tolist(),
            dataset.train.part_b.tolist(),
            dataset.train.part_a_noise,
            dataset.train.part_b_noise,
            strict=True,
        )
    }

    seen_test_total = 0
    seen_test_matches = 0
    for part_a, part_b, part_a_noise, part_b_noise, seen_unseen in zip(
        dataset.test.part_a.tolist(),
        dataset.test.part_b.tolist(),
        dataset.test.part_a_noise,
        dataset.test.part_b_noise,
        dataset.test.seen_unseen.tolist(),
        strict=True,
    ):
        if seen_unseen != 0:
            continue
        seen_test_total += 1
        signature = (
            int(part_a),
            int(part_b),
            tuple(float(value) for value in part_a_noise.tolist()),
            tuple(float(value) for value in part_b_noise.tolist()),
        )
        if signature in train_examples:
            seen_test_matches += 1

    return seen_test_matches / seen_test_total if seen_test_total else 0.0


def run_training_suite(seeds: list[int] | tuple[int, ...]) -> list[SeedTrainingResult]:
    """Run Deliverable 2 training over the requested seeds."""
    return [train_single_seed(E14Config(seed=seed)) for seed in seeds]


def build_frozen_banks(
    model: FrozenActivationBankMLP,
    train_split: SplitTensors,
    config: E14Config,
) -> FrozenBank:
    """Build one activation vector per training example per hidden layer."""
    device = torch.device(config.device)
    loader = make_loader(train_split, config, shuffle=False)
    layer1_chunks: list[torch.Tensor] = []
    layer2_chunks: list[torch.Tensor] = []
    was_training = model.training

    model.eval()
    with torch.no_grad():
        for part_a, part_b, part_a_noise, part_b_noise, _, _ in loader:
            outputs = model(
                part_a.to(device),
                part_b.to(device),
                part_a_noise=part_a_noise.to(device),
                part_b_noise=part_b_noise.to(device),
                return_read_only_certainty=False,
            )
            layer1_chunks.append(outputs.layer1_activation.detach().cpu())
            layer2_chunks.append(outputs.layer2_activation.detach().cpu())

    if was_training:
        model.train()

    layer1_bank = torch.cat(layer1_chunks, dim=0)
    layer2_bank = torch.cat(layer2_chunks, dim=0)

    if layer1_bank.shape != (train_split.size, config.hidden_dim):
        raise RuntimeError(
            f"Layer 1 bank shape mismatch: expected {(train_split.size, config.hidden_dim)}, got {tuple(layer1_bank.shape)}"
        )
    if layer2_bank.shape != (train_split.size, config.hidden_dim):
        raise RuntimeError(
            f"Layer 2 bank shape mismatch: expected {(train_split.size, config.hidden_dim)}, got {tuple(layer2_bank.shape)}"
        )

    return FrozenBank(layer1_bank=layer1_bank, layer2_bank=layer2_bank)


def summarize_frozen_bank(seed: int) -> FrozenBankSummary:
    """Train one seed, build its frozen bank, and summarize shapes and norms."""
    artifacts = train_single_seed_artifacts(E14Config(seed=seed))
    bank = build_frozen_banks(artifacts.model, artifacts.dataset.train, artifacts.config)
    layer1_norms = bank.layer1_bank.norm(dim=-1)
    layer2_norms = bank.layer2_bank.norm(dim=-1)
    return FrozenBankSummary(
        seed=seed,
        layer1_shape=tuple(bank.layer1_bank.shape),
        layer2_shape=tuple(bank.layer2_bank.shape),
        layer1_norm_mean=float(layer1_norms.mean().item()),
        layer1_norm_std=float(layer1_norms.std(unbiased=False).item()),
        layer2_norm_mean=float(layer2_norms.mean().item()),
        layer2_norm_std=float(layer2_norms.std(unbiased=False).item()),
    )


def run_frozen_bank_suite(seeds: list[int] | tuple[int, ...]) -> list[FrozenBankSummary]:
    """Run Deliverable 3 bank construction over the requested seeds."""
    return [summarize_frozen_bank(seed) for seed in seeds]


def clone_model_with_frozen_bank(
    artifacts: TrainedSeedArtifacts,
    bank: FrozenBank,
) -> FrozenActivationBankMLP:
    """Clone the trained weights into a bank-capacity-matched frozen eval model."""
    eval_model = make_model(artifacts.config, bank_capacity=artifacts.dataset.train.size)
    filtered_state = {
        name: tensor
        for name, tensor in artifacts.model.state_dict().items()
        if not name.startswith("layer1_bank")
        and not name.startswith("layer2_bank")
    }
    missing_keys, unexpected_keys = eval_model.load_state_dict(filtered_state, strict=False)
    if unexpected_keys:
        raise RuntimeError(f"Unexpected keys while loading frozen eval model: {unexpected_keys}")
    expected_missing = {"layer1_bank", "layer2_bank", "layer1_bank_mask", "layer2_bank_mask"}
    if set(missing_keys) != expected_missing:
        raise RuntimeError(f"Unexpected missing keys while loading frozen eval model: {missing_keys}")
    eval_model.seed_banks(bank.layer1_bank, bank.layer2_bank)
    eval_model.eval()
    return eval_model


def build_input_embedding_bank(
    model: FrozenActivationBankMLP,
    train_split: SplitTensors,
    config: E14Config,
) -> torch.Tensor:
    """Build the frozen input-embedding bank from the full training set."""
    device = torch.device(config.device)
    loader = make_loader(train_split, config, shuffle=False)
    chunks: list[torch.Tensor] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for part_a, part_b, part_a_noise, part_b_noise, _, _ in loader:
            outputs = model(
                part_a.to(device),
                part_b.to(device),
                part_a_noise=part_a_noise.to(device),
                part_b_noise=part_b_noise.to(device),
                return_read_only_certainty=False,
            )
            chunks.append(outputs.input_embedding.detach().cpu())
    if was_training:
        model.train()
    return torch.cat(chunks, dim=0)


def _euclidean_nearest_neighbor_distance(
    query_vectors: torch.Tensor,
    reference_vectors: torch.Tensor,
) -> torch.Tensor:
    """Compute per-query nearest-neighbour distance in Euclidean input space."""
    distances = torch.cdist(query_vectors, reference_vectors, p=2)
    return distances.min(dim=-1).values


def collect_test_example_logs_for_seed(seed: int) -> tuple[list[TestExampleLogRecord], SeedCertaintyVisualSummary]:
    """Train one seed, build the frozen bank, and log the full E14 test set."""
    config = E14Config(seed=seed)
    artifacts = train_single_seed_artifacts(config)
    bank = build_frozen_banks(artifacts.model, artifacts.dataset.train, config)
    eval_model = clone_model_with_frozen_bank(artifacts, bank)
    input_embedding_bank = build_input_embedding_bank(eval_model, artifacts.dataset.train, config)

    device = torch.device(config.device)
    eval_model = eval_model.to(device)
    input_embedding_bank = input_embedding_bank.to(device)

    train_part_a_frequencies = torch.bincount(
        artifacts.dataset.train.part_a,
        minlength=config.num_part_a_values,
    )
    train_part_b_frequencies = torch.bincount(
        artifacts.dataset.train.part_b,
        minlength=config.num_part_b_values,
    )

    loader = make_loader(artifacts.dataset.test, config, shuffle=False)
    records: list[TestExampleLogRecord] = []

    with torch.no_grad():
        for part_a, part_b, part_a_noise, part_b_noise, labels, seen_unseen in loader:
            part_a = part_a.to(device)
            part_b = part_b.to(device)
            part_a_noise = part_a_noise.to(device)
            part_b_noise = part_b_noise.to(device)
            labels = labels.to(device)

            outputs = eval_model(
                part_a,
                part_b,
                part_a_noise=part_a_noise,
                part_b_noise=part_b_noise,
                return_read_only_certainty=True,
            )
            if outputs.layer1_certainty is None or outputs.layer2_certainty is None:
                raise RuntimeError("Frozen eval model did not return certainty scores")

            probabilities = torch.softmax(outputs.logits, dim=-1)
            output_confidence = probabilities.max(dim=-1).values
            predicted_class = outputs.logits.argmax(dim=-1)
            correctness = (predicted_class == labels).to(torch.int64)
            input_nn_distance = _euclidean_nearest_neighbor_distance(
                outputs.input_embedding,
                input_embedding_bank,
            )

            batch_size = labels.size(0)
            for batch_index in range(batch_size):
                part_a_value = int(part_a[batch_index].item())
                part_b_value = int(part_b[batch_index].item())
                records.append(
                    TestExampleLogRecord(
                        seed=seed,
                        part_a=part_a_value,
                        part_b=part_b_value,
                        target_label=int(labels[batch_index].item()),
                        seen_unseen_label=int(seen_unseen[batch_index].item()),
                        correctness=int(correctness[batch_index].item()),
                        predicted_class=int(predicted_class[batch_index].item()),
                        output_confidence=float(output_confidence[batch_index].item()),
                        input_nn_distance=float(input_nn_distance[batch_index].item()),
                        layer1_certainty=float(outputs.layer1_certainty[batch_index].item()),
                        layer2_certainty=float(outputs.layer2_certainty[batch_index].item()),
                        part_a_marginal_frequency=int(train_part_a_frequencies[part_a_value].item()),
                        part_b_marginal_frequency=int(train_part_b_frequencies[part_b_value].item()),
                    )
                )

    return records, _seed_visual_summary(records, seed)


def summarize_logging_across_seeds(seeds: list[int] | tuple[int, ...]) -> LoggingSummary:
    """Aggregate Deliverable 4 certainty/logging summaries across seeds."""
    all_records: list[TestExampleLogRecord] = []
    visual_summaries: list[SeedCertaintyVisualSummary] = []
    for seed in seeds:
        seed_records, visual_summary = collect_test_example_logs_for_seed(seed)
        persist_per_example_logs(seed, seed_records)
        all_records.extend(seed_records)
        visual_summaries.append(visual_summary)

    return LoggingSummary(
        layer1_certainty=_metric_summary(all_records, metric_name="layer1_certainty"),
        layer2_certainty=_metric_summary(all_records, metric_name="layer2_certainty"),
        output_confidence=_metric_summary(all_records, metric_name="output_confidence"),
        input_nn_distance=_metric_summary(all_records, metric_name="input_nn_distance"),
        per_seed_visual=visual_summaries,
    )


def persist_per_example_logs(seed: int, records: list[TestExampleLogRecord]) -> Path:
    """Persist the per-example certainty logs for one seed to disk."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"per_example_logs_seed_{seed}.csv"
    fieldnames = [
        "seed",
        "part_a",
        "part_b",
        "target_label",
        "seen_unseen",
        "correctness",
        "predicted_class",
        "output_confidence",
        "input_nn_distance",
        "layer1_certainty",
        "layer2_certainty",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "seed": record.seed,
                    "part_a": record.part_a,
                    "part_b": record.part_b,
                    "target_label": record.target_label,
                    "seen_unseen": record.seen_unseen_label,
                    "correctness": record.correctness,
                    "predicted_class": record.predicted_class,
                    "output_confidence": record.output_confidence,
                    "input_nn_distance": record.input_nn_distance,
                    "layer1_certainty": record.layer1_certainty,
                    "layer2_certainty": record.layer2_certainty,
                }
            )
    return output_path


def _metric_summary(
    records: list[TestExampleLogRecord],
    *,
    metric_name: str,
) -> MetricSeenUnseenSummary:
    seen_values = torch.tensor(
        [getattr(record, metric_name) for record in records if record.seen_unseen_label == 0],
        dtype=torch.float64,
    )
    unseen_values = torch.tensor(
        [getattr(record, metric_name) for record in records if record.seen_unseen_label == 1],
        dtype=torch.float64,
    )
    return MetricSeenUnseenSummary(
        seen_mean=float(seen_values.mean().item()),
        seen_std=float(seen_values.std(unbiased=False).item()),
        unseen_mean=float(unseen_values.mean().item()),
        unseen_std=float(unseen_values.std(unbiased=False).item()),
    )


def _seed_visual_summary(records: list[TestExampleLogRecord], seed: int) -> SeedCertaintyVisualSummary:
    seen_layer1 = torch.tensor(
        [record.layer1_certainty for record in records if record.seen_unseen_label == 0],
        dtype=torch.float64,
    )
    unseen_layer1 = torch.tensor(
        [record.layer1_certainty for record in records if record.seen_unseen_label == 1],
        dtype=torch.float64,
    )
    seen_layer2 = torch.tensor(
        [record.layer2_certainty for record in records if record.seen_unseen_label == 0],
        dtype=torch.float64,
    )
    unseen_layer2 = torch.tensor(
        [record.layer2_certainty for record in records if record.seen_unseen_label == 1],
        dtype=torch.float64,
    )
    return SeedCertaintyVisualSummary(
        seed=seed,
        layer1_seen_mean=float(seen_layer1.mean().item()),
        layer1_unseen_mean=float(unseen_layer1.mean().item()),
        layer2_seen_mean=float(seen_layer2.mean().item()),
        layer2_unseen_mean=float(unseen_layer2.mean().item()),
    )
