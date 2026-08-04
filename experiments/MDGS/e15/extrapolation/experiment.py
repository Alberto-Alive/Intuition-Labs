"""Training loop and reporting helpers for DIGIT Extrapolation E15."""

from __future__ import annotations

from csv import DictReader, DictWriter
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.DIGIT.Extrapolation.e14.extrapolation.experiment import (
    E14Dataset,
    SplitTensors,
    build_dataset,
    make_loader,
)

from .config import E15Config
from .models import LiveActivationBankMLP


@dataclass(frozen=True)
class EpochBankDiagnostics:
    """Per-epoch live-bank occupancy and certainty summaries."""

    epoch: int
    layer1_occupancy: int
    layer2_occupancy: int
    layer1_mean_valid_certainty: float | None
    layer2_mean_valid_certainty: float | None


@dataclass(frozen=True)
class SeedTrainingResult:
    """Requested Deliverable 2 outputs for one E15 seed."""

    seed: int
    stopping_epoch: int
    final_training_loss: float
    final_training_accuracy: float
    seen_test_accuracy: float
    unseen_test_accuracy: float
    first_min_occupancy_epoch: int
    first_min_occupancy_step: int
    epoch_diagnostics: list[EpochBankDiagnostics]


@dataclass
class TrainedSeedArtifacts:
    """Trained model plus dataset and stopping metrics for one E15 seed."""

    config: E15Config
    model: LiveActivationBankMLP
    dataset: E14Dataset
    result: SeedTrainingResult


@dataclass(frozen=True)
class TestExampleLogRecord:
    """Per-example E15 logging payload with live and archived frozen certainty."""

    seed: int
    part_a: int
    part_b: int
    target_label: int
    seen_unseen_label: int
    correctness: int
    predicted_class: int
    output_confidence: float
    input_nn_distance: float
    layer1_live_certainty: float
    layer2_live_certainty: float
    layer1_frozen_e14_certainty: float
    layer2_frozen_e14_certainty: float


@dataclass(frozen=True)
class MetricSeenUnseenSummary:
    """Mean/std summary of one metric for seen vs unseen pooled examples."""

    seen_mean: float
    seen_std: float
    unseen_mean: float
    unseen_std: float


@dataclass(frozen=True)
class SeedCertaintyVisualSummary:
    """Per-seed live certainty means requested before analysis."""

    seed: int
    layer1_live_seen_mean: float
    layer1_live_unseen_mean: float
    layer2_live_seen_mean: float
    layer2_live_unseen_mean: float


@dataclass(frozen=True)
class LoggingSummary:
    """Deliverable 3 aggregate summary plus per-seed live certainty means."""

    layer1_live_certainty: MetricSeenUnseenSummary
    layer2_live_certainty: MetricSeenUnseenSummary
    layer1_frozen_e14_certainty: MetricSeenUnseenSummary
    per_seed_visual: list[SeedCertaintyVisualSummary]
    seed0_layer1_live_vs_frozen_correlation: float


E14_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "e14" / "outputs"
E15_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def set_global_determinism(seed: int) -> None:
    """Set the approved seed across Python and PyTorch."""
    random.seed(seed)
    torch.manual_seed(seed)


def make_model(config: E15Config) -> LiveActivationBankMLP:
    """Instantiate the approved 2-layer MLP with live circular-buffer banks."""
    return LiveActivationBankMLP(
        part_a_vocab_size=config.num_part_a_values,
        part_b_vocab_size=config.num_part_b_values,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
        bank_capacity=config.bank_capacity,
        min_bank_occupancy=config.min_bank_occupancy,
    )


def train_single_seed(config: E15Config) -> SeedTrainingResult:
    """Train one E15 seed and return the requested Deliverable 2 metrics."""
    return train_single_seed_artifacts(config).result


def train_single_seed_artifacts(config: E15Config) -> TrainedSeedArtifacts:
    """Train one E15 seed and return the trained model plus materialized dataset."""
    set_global_determinism(config.seed)
    dataset = build_dataset(config)

    device = torch.device(config.device)
    model = make_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_loader = make_loader(dataset.train, config, shuffle=True)

    best_loss = float("inf")
    epochs_without_meaningful_decrease = 0
    final_training_loss = float("inf")
    final_training_accuracy = 0.0
    first_min_occupancy_epoch: int | None = None
    first_min_occupancy_step: int | None = None
    epoch_diagnostics: list[EpochBankDiagnostics] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_correct = 0
        epoch_examples = 0
        layer1_valid_certainties: list[torch.Tensor] = []
        layer2_valid_certainties: list[torch.Tensor] = []

        for step, (part_a, part_b, part_a_noise, part_b_noise, labels, _) in enumerate(train_loader, start=1):
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
                return_live_certainty=True,
            )
            loss = F.cross_entropy(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            model.update_live_banks(outputs.layer1_activation, outputs.layer2_activation)

            if first_min_occupancy_epoch is None and model.layer1_bank.occupancy >= config.min_bank_occupancy:
                first_min_occupancy_epoch = epoch
                first_min_occupancy_step = step

            if outputs.layer1_certainty is not None:
                valid_mask = ~torch.isnan(outputs.layer1_certainty)
                if bool(valid_mask.any().item()):
                    layer1_valid_certainties.append(outputs.layer1_certainty[valid_mask].detach().cpu())

            if outputs.layer2_certainty is not None:
                valid_mask = ~torch.isnan(outputs.layer2_certainty)
                if bool(valid_mask.any().item()):
                    layer2_valid_certainties.append(outputs.layer2_certainty[valid_mask].detach().cpu())

            batch_size = labels.size(0)
            epoch_loss_sum += float(loss.item()) * batch_size
            predictions = outputs.logits.argmax(dim=-1)
            epoch_correct += int((predictions == labels).sum().item())
            epoch_examples += batch_size

        final_training_loss = epoch_loss_sum / epoch_examples
        final_training_accuracy = epoch_correct / epoch_examples

        epoch_diagnostics.append(
            EpochBankDiagnostics(
                epoch=epoch,
                layer1_occupancy=model.layer1_bank.occupancy,
                layer2_occupancy=model.layer2_bank.occupancy,
                layer1_mean_valid_certainty=_mean_from_chunks(layer1_valid_certainties),
                layer2_mean_valid_certainty=_mean_from_chunks(layer2_valid_certainties),
            )
        )

        if best_loss - final_training_loss > config.loss_improvement_tolerance:
            best_loss = final_training_loss
            epochs_without_meaningful_decrease = 0
        else:
            epochs_without_meaningful_decrease += 1

        if epochs_without_meaningful_decrease >= config.loss_patience_epochs:
            break

    if first_min_occupancy_epoch is None or first_min_occupancy_step is None:
        raise RuntimeError("Live bank never reached minimum occupancy during training")

    seen_test_accuracy, unseen_test_accuracy = evaluate_seen_unseen_accuracy(model, dataset.test, config)

    result = SeedTrainingResult(
        seed=config.seed,
        stopping_epoch=epoch_diagnostics[-1].epoch,
        final_training_loss=final_training_loss,
        final_training_accuracy=final_training_accuracy,
        seen_test_accuracy=seen_test_accuracy,
        unseen_test_accuracy=unseen_test_accuracy,
        first_min_occupancy_epoch=first_min_occupancy_epoch,
        first_min_occupancy_step=first_min_occupancy_step,
        epoch_diagnostics=epoch_diagnostics,
    )
    return TrainedSeedArtifacts(config=config, model=model, dataset=dataset, result=result)


def evaluate_seen_unseen_accuracy(
    model: LiveActivationBankMLP,
    split: SplitTensors,
    config: E15Config,
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
                return_live_certainty=False,
            )
            predictions = outputs.logits.argmax(dim=-1).cpu()
            labels_cpu = labels.cpu()

            seen_mask = seen_unseen == 0
            unseen_mask = seen_unseen == 1

            if int(seen_mask.sum().item()) > 0:
                seen_correct += int((predictions[seen_mask] == labels_cpu[seen_mask]).sum().item())
                seen_total += int(seen_mask.sum().item())
            if int(unseen_mask.sum().item()) > 0:
                unseen_correct += int((predictions[unseen_mask] == labels_cpu[unseen_mask]).sum().item())
                unseen_total += int(unseen_mask.sum().item())

    seen_accuracy = seen_correct / seen_total if seen_total else 0.0
    unseen_accuracy = unseen_correct / unseen_total if unseen_total else 0.0
    return seen_accuracy, unseen_accuracy


def run_training_suite(seeds: list[int] | tuple[int, ...]) -> list[SeedTrainingResult]:
    """Run Deliverable 2 training over the requested seeds."""
    return [train_single_seed(E15Config(seed=seed)) for seed in seeds]


def build_input_embedding_bank(
    model: LiveActivationBankMLP,
    train_split: SplitTensors,
    config: E15Config,
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
                return_live_certainty=False,
            )
            chunks.append(outputs.input_embedding.detach().cpu())
    if was_training:
        model.train()
    return torch.cat(chunks, dim=0)


def collect_test_example_logs_for_seed(seed: int) -> tuple[list[TestExampleLogRecord], SeedCertaintyVisualSummary]:
    """Train one seed, freeze the final live bank, and log the full E15 test set."""
    artifacts = train_single_seed_artifacts(E15Config(seed=seed))
    model = artifacts.model
    dataset = artifacts.dataset
    config = artifacts.config

    device = torch.device(config.device)
    model = model.to(device)
    model.eval()
    input_embedding_bank = build_input_embedding_bank(model, dataset.train, config).to(device)
    archived_e14_rows = _load_archived_e14_logs(seed)

    loader = make_loader(dataset.test, config, shuffle=False)
    records: list[TestExampleLogRecord] = []
    archived_index = 0

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
                return_live_certainty=True,
            )
            if outputs.layer1_certainty is None or outputs.layer2_certainty is None:
                raise RuntimeError("Live eval model did not return certainty scores")

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
                if archived_index >= len(archived_e14_rows):
                    raise RuntimeError(f"Archived E14 log for seed {seed} is shorter than the E15 test log")

                archived_row = archived_e14_rows[archived_index]
                archived_index += 1

                part_a_value = int(part_a[batch_index].item())
                part_b_value = int(part_b[batch_index].item())
                target_label = int(labels[batch_index].item())
                seen_unseen_value = int(seen_unseen[batch_index].item())
                _assert_archived_row_matches(
                    seed=seed,
                    archived_row=archived_row,
                    part_a=part_a_value,
                    part_b=part_b_value,
                    target_label=target_label,
                    seen_unseen=seen_unseen_value,
                )

                records.append(
                    TestExampleLogRecord(
                        seed=seed,
                        part_a=part_a_value,
                        part_b=part_b_value,
                        target_label=target_label,
                        seen_unseen_label=seen_unseen_value,
                        correctness=int(correctness[batch_index].item()),
                        predicted_class=int(predicted_class[batch_index].item()),
                        output_confidence=float(output_confidence[batch_index].item()),
                        input_nn_distance=float(input_nn_distance[batch_index].item()),
                        layer1_live_certainty=float(outputs.layer1_certainty[batch_index].item()),
                        layer2_live_certainty=float(outputs.layer2_certainty[batch_index].item()),
                        layer1_frozen_e14_certainty=float(archived_row["layer1_certainty"]),
                        layer2_frozen_e14_certainty=float(archived_row["layer2_certainty"]),
                    )
                )

    if archived_index != len(archived_e14_rows):
        raise RuntimeError(f"Archived E14 log for seed {seed} is longer than the E15 test log")

    return records, _seed_visual_summary(records, seed)


def summarize_logging_across_seeds(seeds: list[int] | tuple[int, ...]) -> LoggingSummary:
    """Aggregate Deliverable 3 live/frozen certainty summaries across seeds."""
    all_records: list[TestExampleLogRecord] = []
    visual_summaries: list[SeedCertaintyVisualSummary] = []
    seed0_records: list[TestExampleLogRecord] | None = None

    for seed in seeds:
        seed_records, visual_summary = collect_test_example_logs_for_seed(seed)
        persist_per_example_logs(seed, seed_records)
        all_records.extend(seed_records)
        visual_summaries.append(visual_summary)
        if seed == 0:
            seed0_records = seed_records

    if seed0_records is None:
        raise RuntimeError("Seed 0 records were not collected")

    return LoggingSummary(
        layer1_live_certainty=_metric_summary(all_records, metric_name="layer1_live_certainty"),
        layer2_live_certainty=_metric_summary(all_records, metric_name="layer2_live_certainty"),
        layer1_frozen_e14_certainty=_metric_summary(all_records, metric_name="layer1_frozen_e14_certainty"),
        per_seed_visual=visual_summaries,
        seed0_layer1_live_vs_frozen_correlation=_pearson_correlation(
            [record.layer1_live_certainty for record in seed0_records],
            [record.layer1_frozen_e14_certainty for record in seed0_records],
        ),
    )


def persist_per_example_logs(seed: int, records: list[TestExampleLogRecord]) -> Path:
    """Persist the E15 per-example logs for one seed to disk."""
    E15_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = E15_OUTPUT_DIR / f"per_example_logs_seed_{seed}.csv"
    fieldnames = [
        "seed",
        "part_a",
        "part_b",
        "target_label",
        "seen_unseen",
        "correctness",
        "predicted_class",
        "output_confidence",
        "layer1_live_certainty",
        "layer2_live_certainty",
        "gate_layer1",
        "gate_layer2",
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
                    "layer1_live_certainty": record.layer1_live_certainty,
                    "layer2_live_certainty": record.layer2_live_certainty,
                    "gate_layer1": _gate_from_certainty(record.layer1_live_certainty),
                    "gate_layer2": _gate_from_certainty(record.layer2_live_certainty),
                }
            )
    return output_path


def _mean_from_chunks(chunks: list[torch.Tensor]) -> float | None:
    if not chunks:
        return None
    return float(torch.cat(chunks).mean().item())


def _euclidean_nearest_neighbor_distance(
    query_vectors: torch.Tensor,
    reference_vectors: torch.Tensor,
) -> torch.Tensor:
    distances = torch.cdist(query_vectors, reference_vectors, p=2)
    return distances.min(dim=-1).values


def _load_archived_e14_logs(seed: int) -> list[dict[str, str]]:
    path = E14_OUTPUT_DIR / f"per_example_logs_seed_{seed}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Archived E14 per-example log not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(DictReader(handle))


def _assert_archived_row_matches(
    *,
    seed: int,
    archived_row: dict[str, str],
    part_a: int,
    part_b: int,
    target_label: int,
    seen_unseen: int,
) -> None:
    archived_seed = int(archived_row["seed"])
    archived_part_a = int(archived_row["part_a"])
    archived_part_b = int(archived_row["part_b"])
    archived_target_label = int(archived_row["target_label"])
    archived_seen_unseen = int(archived_row["seen_unseen"])
    if (
        archived_seed != seed
        or archived_part_a != part_a
        or archived_part_b != part_b
        or archived_target_label != target_label
        or archived_seen_unseen != seen_unseen
    ):
        raise RuntimeError(
            "Archived E14 row order does not match the E15 test example order"
        )


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
        [record.layer1_live_certainty for record in records if record.seen_unseen_label == 0],
        dtype=torch.float64,
    )
    unseen_layer1 = torch.tensor(
        [record.layer1_live_certainty for record in records if record.seen_unseen_label == 1],
        dtype=torch.float64,
    )
    seen_layer2 = torch.tensor(
        [record.layer2_live_certainty for record in records if record.seen_unseen_label == 0],
        dtype=torch.float64,
    )
    unseen_layer2 = torch.tensor(
        [record.layer2_live_certainty for record in records if record.seen_unseen_label == 1],
        dtype=torch.float64,
    )
    return SeedCertaintyVisualSummary(
        seed=seed,
        layer1_live_seen_mean=float(seen_layer1.mean().item()),
        layer1_live_unseen_mean=float(unseen_layer1.mean().item()),
        layer2_live_seen_mean=float(seen_layer2.mean().item()),
        layer2_live_unseen_mean=float(unseen_layer2.mean().item()),
    )


def _pearson_correlation(xs: list[float], ys: list[float]) -> float:
    x = torch.tensor(xs, dtype=torch.float64)
    y = torch.tensor(ys, dtype=torch.float64)
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = x_centered.norm() * y_centered.norm()
    if float(denominator.item()) == 0.0:
        raise RuntimeError("Cannot compute Pearson correlation for a constant input")
    return float(((x_centered * y_centered).sum() / denominator).item())


def _gate_from_certainty(certainty: float) -> float:
    if certainty != certainty:
        return 1.0
    return certainty * certainty
