"""Training and inference-time helpers for DIGIT Extrapolation E20."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import torch
from scipy.stats import norm, spearmanr

from experiments.DIGIT.Extrapolation.e19.extrapolation.config import E19Config
from experiments.DIGIT.Extrapolation.e19.extrapolation.data import (
    LEVEL_NAMES,
    LEVEL_NOVEL,
    build_frequency_level_records,
)
from experiments.DIGIT.Extrapolation.e19.extrapolation.experiment import (
    TrainedSeedArtifacts,
    run_training_suite,
)

from .config import E20Config
from .partitioned_bank import PartitionedActivationBank


ARCHIVED_E19_NOVEL_BASELINES = {
    0: {"layer1": 0.8510462045669556, "layer2": 0.9099333882331848},
    1: {"layer1": 0.8657646775245667, "layer2": 0.9223146438598633},
    2: {"layer1": 0.8332082033157349, "layer2": 0.905534029006958},
    3: {"layer1": 0.8688594698905945, "layer2": 0.9198400974273682},
    4: {"layer1": 0.8583950400352478, "layer2": 0.9237188100814819},
}


@dataclass(frozen=True)
class SeedTrainingPhaseResult:
    """Deliverable 1 training-phase outputs for one E20 seed."""

    seed: int
    final_training_loss: float
    final_training_accuracy: float
    stopping_epoch: int
    checkpoint0_layer1_novel_certainty: float
    checkpoint0_layer2_novel_certainty: float
    layer1_archived_difference: float
    layer2_archived_difference: float


@dataclass
class FrozenSeedArtifacts:
    """Frozen model and partitioned banks at the start of E20 inference."""

    seed: int
    trained_artifacts: TrainedSeedArtifacts
    layer1_partitioned_bank: PartitionedActivationBank
    layer2_partitioned_bank: PartitionedActivationBank
    result: SeedTrainingPhaseResult


@dataclass(frozen=True)
class CheckpointProbeSummary:
    """Checkpoint-level mean certainty and accuracy for one seed."""

    seed: int
    checkpoint: int
    layer1_mean_certainty: float
    layer2_mean_certainty: float
    novel_accuracy: float


@dataclass(frozen=True)
class LayerExposurePrimaryAnalysis:
    """Ordered-growth analysis for one seed and one certainty layer."""

    seed: int
    layer_name: str
    jonckheere_terpstra_statistic: float
    jonckheere_terpstra_p_value: float
    strict_mean_ordering: bool
    spearman_rho: float
    checkpoint_means: tuple[float, ...]


@dataclass(frozen=True)
class AggregatedExposurePrimaryAnalysis:
    """Cross-seed ordered-growth summary for one certainty layer."""

    layer_name: str
    significant_seed_count: int
    strict_mean_ordering_seed_count: int
    mean_spearman_rho: float
    std_spearman_rho: float


@dataclass(frozen=True)
class SeedExposureArtifacts:
    """Full exposure-protocol outputs for one seed."""

    seed: int
    checkpoint_summaries: tuple[CheckpointProbeSummary, ...]
    layer1_analysis: LayerExposurePrimaryAnalysis
    layer2_analysis: LayerExposurePrimaryAnalysis


def _compute_checkpoint0_novel_certainty(
    trained_artifacts: TrainedSeedArtifacts,
    *,
    inference_partition_capacity: int = 1024,
) -> tuple[float, float]:
    """Compute checkpoint-0 novel certainty from the frozen training partition alone."""
    model = trained_artifacts.model
    split = trained_artifacts.dataset.test
    device = torch.device(trained_artifacts.config.device)

    layer1_partitioned = PartitionedActivationBank.from_live_bank(
        model.layer1_bank,
        inference_capacity=inference_partition_capacity,
    ).to(device)
    layer2_partitioned = PartitionedActivationBank.from_live_bank(
        model.layer2_bank,
        inference_capacity=inference_partition_capacity,
    ).to(device)

    novel_mask = split.levels == LEVEL_NOVEL
    with torch.no_grad():
        input_embedding = model.encode_inputs_with_jitter(
            split.part_a[novel_mask].to(device),
            split.part_b[novel_mask].to(device),
            part_a_noise=split.part_a_noise[novel_mask].to(device),
            part_b_noise=split.part_b_noise[novel_mask].to(device),
        )
        layer1_pre_gate = torch.relu(model.hidden1(input_embedding))
        layer1_certainty = layer1_partitioned.query(layer1_pre_gate)
        layer1_gated = layer1_pre_gate * model._gate_from_certainty(layer1_certainty).unsqueeze(-1)

        layer2_pre_gate = torch.relu(model.hidden2(layer1_gated))
        layer2_certainty = layer2_partitioned.query(layer2_pre_gate)

    return float(layer1_certainty.mean().item()), float(layer2_certainty.mean().item())


def run_training_phase(
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    *,
    inference_partition_capacity: int = 1024,
) -> list[FrozenSeedArtifacts]:
    """Replay the frozen E19 training phase and snapshot checkpoint-0 certainty."""
    trained_artifacts = run_training_suite(seeds)
    frozen_artifacts: list[FrozenSeedArtifacts] = []

    for item in trained_artifacts:
        seed = item.result.seed
        layer1_checkpoint0, layer2_checkpoint0 = _compute_checkpoint0_novel_certainty(
            item,
            inference_partition_capacity=inference_partition_capacity,
        )
        layer1_partitioned = PartitionedActivationBank.from_live_bank(
            item.model.layer1_bank,
            inference_capacity=inference_partition_capacity,
        )
        layer2_partitioned = PartitionedActivationBank.from_live_bank(
            item.model.layer2_bank,
            inference_capacity=inference_partition_capacity,
        )
        archived = ARCHIVED_E19_NOVEL_BASELINES[seed]
        result = SeedTrainingPhaseResult(
            seed=seed,
            final_training_loss=item.result.final_training_loss,
            final_training_accuracy=item.result.final_training_accuracy,
            stopping_epoch=item.result.stopping_epoch,
            checkpoint0_layer1_novel_certainty=layer1_checkpoint0,
            checkpoint0_layer2_novel_certainty=layer2_checkpoint0,
            layer1_archived_difference=abs(layer1_checkpoint0 - archived["layer1"]),
            layer2_archived_difference=abs(layer2_checkpoint0 - archived["layer2"]),
        )
        frozen_artifacts.append(
            FrozenSeedArtifacts(
                seed=seed,
                trained_artifacts=item,
                layer1_partitioned_bank=layer1_partitioned,
                layer2_partitioned_bank=layer2_partitioned,
                result=result,
            )
        )

    return frozen_artifacts


def _novel_records():
    return tuple(record for record in build_frequency_level_records() if record.level == LEVEL_NOVEL)


def _forward_with_partitioned_banks(
    frozen_artifacts: FrozenSeedArtifacts,
    part_a: torch.Tensor,
    part_b: torch.Tensor,
    part_a_noise: torch.Tensor,
    part_b_noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one forward pass using partitioned banks for certainty queries."""
    model = frozen_artifacts.trained_artifacts.model
    device = torch.device(frozen_artifacts.trained_artifacts.config.device)

    input_embedding = model.encode_inputs_with_jitter(
        part_a.to(device),
        part_b.to(device),
        part_a_noise=part_a_noise.to(device),
        part_b_noise=part_b_noise.to(device),
    )
    layer1_pre_gate = torch.relu(model.hidden1(input_embedding))
    layer1_certainty = frozen_artifacts.layer1_partitioned_bank.query(layer1_pre_gate)
    gate_layer1 = model._gate_from_certainty(layer1_certainty)
    layer1_gated = layer1_pre_gate * gate_layer1.unsqueeze(-1)

    layer2_pre_gate = torch.relu(model.hidden2(layer1_gated))
    layer2_certainty = frozen_artifacts.layer2_partitioned_bank.query(layer2_pre_gate)
    gate_layer2 = model._gate_from_certainty(layer2_certainty)
    layer2_gated = layer2_pre_gate * gate_layer2.unsqueeze(-1)

    logits = model.classifier(layer2_gated)
    return logits, layer1_certainty, layer2_certainty


def _probe_checkpoint(
    frozen_artifacts: FrozenSeedArtifacts,
    checkpoint: int,
    config: E20Config,
) -> tuple[CheckpointProbeSummary, dict[str, np.ndarray]]:
    """Probe all novel combinations at one checkpoint with a fresh RNG stream."""
    novel_records = _novel_records()
    trained_config = frozen_artifacts.trained_artifacts.config
    device = torch.device(trained_config.device)
    probe_generator = torch.Generator()
    probe_generator.manual_seed(
        frozen_artifacts.seed + config.probe_noise_seed_offset + (checkpoint * 10_000)
    )

    layer1_values: list[torch.Tensor] = []
    layer2_values: list[torch.Tensor] = []
    accuracy_values: list[torch.Tensor] = []

    with torch.no_grad():
        for record in novel_records:
            part_a = torch.full((config.probe_repeats_per_combo,), record.part_a, dtype=torch.long)
            part_b = torch.full((config.probe_repeats_per_combo,), record.part_b, dtype=torch.long)
            part_a_noise = (
                torch.randn(
                    config.probe_repeats_per_combo,
                    trained_config.embedding_dim,
                    generator=probe_generator,
                )
                * trained_config.input_jitter_std
            )
            part_b_noise = (
                torch.randn(
                    config.probe_repeats_per_combo,
                    trained_config.embedding_dim,
                    generator=probe_generator,
                )
                * trained_config.input_jitter_std
            )
            logits, layer1_certainty, layer2_certainty = _forward_with_partitioned_banks(
                frozen_artifacts,
                part_a,
                part_b,
                part_a_noise,
                part_b_noise,
            )
            predictions = logits.argmax(dim=-1).cpu()
            labels = torch.full((config.probe_repeats_per_combo,), record.class_id, dtype=torch.long)

            layer1_values.append(layer1_certainty.detach().cpu())
            layer2_values.append(layer2_certainty.detach().cpu())
            accuracy_values.append((predictions == labels).to(dtype=torch.float32))

    layer1_concat = torch.cat(layer1_values)
    layer2_concat = torch.cat(layer2_values)
    accuracy_concat = torch.cat(accuracy_values)

    summary = CheckpointProbeSummary(
        seed=frozen_artifacts.seed,
        checkpoint=checkpoint,
        layer1_mean_certainty=float(layer1_concat.mean().item()),
        layer2_mean_certainty=float(layer2_concat.mean().item()),
        novel_accuracy=float(accuracy_concat.mean().item()),
    )
    return summary, {
        "layer1": layer1_concat.numpy(),
        "layer2": layer2_concat.numpy(),
        "accuracy": accuracy_concat.numpy(),
    }


def _run_exposure_updates(
    frozen_artifacts: FrozenSeedArtifacts,
    rounds: int,
    config: E20Config,
) -> None:
    """Append novel activations into the inference partitions over balanced rounds."""
    novel_records = _novel_records()
    trained_config = frozen_artifacts.trained_artifacts.config
    update_generator = torch.Generator()
    update_generator.manual_seed(frozen_artifacts.seed + config.update_noise_seed_offset)

    with torch.no_grad():
        for _round in range(rounds):
            for record in novel_records:
                part_a = torch.tensor([record.part_a], dtype=torch.long)
                part_b = torch.tensor([record.part_b], dtype=torch.long)
                part_a_noise = (
                    torch.randn(1, trained_config.embedding_dim, generator=update_generator)
                    * trained_config.input_jitter_std
                )
                part_b_noise = (
                    torch.randn(1, trained_config.embedding_dim, generator=update_generator)
                    * trained_config.input_jitter_std
                )
                model = frozen_artifacts.trained_artifacts.model
                input_embedding = model.encode_inputs_with_jitter(
                    part_a,
                    part_b,
                    part_a_noise=part_a_noise,
                    part_b_noise=part_b_noise,
                )
                layer1_pre_gate = torch.relu(model.hidden1(input_embedding))
                layer1_certainty = frozen_artifacts.layer1_partitioned_bank.query(layer1_pre_gate)
                gate_layer1 = model._gate_from_certainty(layer1_certainty)
                layer1_gated = layer1_pre_gate * gate_layer1.unsqueeze(-1)

                layer2_pre_gate = torch.relu(model.hidden2(layer1_gated))

                frozen_artifacts.layer1_partitioned_bank.update_inference_partition(layer1_pre_gate)
                frozen_artifacts.layer2_partitioned_bank.update_inference_partition(layer2_pre_gate)


def _jonckheere_terpstra_increasing(
    checkpoint_arrays: tuple[np.ndarray, ...],
) -> tuple[float, float]:
    """One-sided Jonckheere-Terpstra test for increasing ordered groups."""
    statistic = 0.0
    for left_index, left_values in enumerate(checkpoint_arrays[:-1]):
        left_column = left_values[:, None]
        for right_values in checkpoint_arrays[left_index + 1 :]:
            right_row = right_values[None, :]
            statistic += float(np.count_nonzero(left_column < right_row))
            statistic += 0.5 * float(np.count_nonzero(left_column == right_row))

    group_sizes = np.array([values.size for values in checkpoint_arrays], dtype=np.float64)
    total_size = float(group_sizes.sum())
    mean_under_null = (total_size**2 - float(np.square(group_sizes).sum())) / 4.0
    variance_under_null = (
        total_size**2 * (2.0 * total_size + 3.0)
        - float((np.square(group_sizes) * (2.0 * group_sizes + 3.0)).sum())
    ) / 72.0
    if variance_under_null <= 0.0:
        return statistic, 1.0

    z_score = (statistic - mean_under_null) / sqrt(variance_under_null)
    return statistic, float(norm.sf(z_score))


def _analyze_seed_layer(
    seed: int,
    layer_name: str,
    checkpoint_arrays: dict[int, np.ndarray],
    config: E20Config,
) -> LayerExposurePrimaryAnalysis:
    """Run the ordered-growth analysis for one seed and one certainty layer."""
    ordered_arrays = tuple(checkpoint_arrays[checkpoint] for checkpoint in config.checkpoint_rounds)
    checkpoint_means = tuple(float(values.mean()) for values in ordered_arrays)
    jt_statistic, jt_p_value = _jonckheere_terpstra_increasing(ordered_arrays)
    strict_mean_ordering = bool(
        all(
            checkpoint_means[index] < checkpoint_means[index + 1]
            for index in range(len(checkpoint_means) - 1)
        )
    )
    rank_vector = np.concatenate(
        [
            np.full(checkpoint_arrays[checkpoint].shape[0], rank, dtype=np.int64)
            for rank, checkpoint in enumerate(config.checkpoint_rounds, start=1)
        ]
    )
    certainty_vector = np.concatenate(
        [checkpoint_arrays[checkpoint] for checkpoint in config.checkpoint_rounds]
    )
    spearman_result = spearmanr(rank_vector, certainty_vector)
    return LayerExposurePrimaryAnalysis(
        seed=seed,
        layer_name=layer_name,
        jonckheere_terpstra_statistic=jt_statistic,
        jonckheere_terpstra_p_value=jt_p_value,
        strict_mean_ordering=strict_mean_ordering,
        spearman_rho=float(spearman_result.statistic),
        checkpoint_means=checkpoint_means,
    )


def aggregate_primary_analysis(
    layer_analyses: tuple[LayerExposurePrimaryAnalysis, ...],
) -> AggregatedExposurePrimaryAnalysis:
    """Aggregate the primary ordered-growth analysis across seeds for one layer."""
    spearman_rhos = np.array([item.spearman_rho for item in layer_analyses], dtype=np.float64)
    return AggregatedExposurePrimaryAnalysis(
        layer_name=layer_analyses[0].layer_name,
        significant_seed_count=sum(item.jonckheere_terpstra_p_value < 0.05 for item in layer_analyses),
        strict_mean_ordering_seed_count=sum(item.strict_mean_ordering for item in layer_analyses),
        mean_spearman_rho=float(spearman_rhos.mean()),
        std_spearman_rho=float(spearman_rhos.std(ddof=0)),
    )


def run_exposure_protocol(
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    *,
    training_phase_artifacts: list[FrozenSeedArtifacts] | None = None,
    config: E20Config | None = None,
) -> list[SeedExposureArtifacts]:
    """Run the full E20 balanced exposure protocol and ordered-growth analysis."""
    resolved_config = E20Config() if config is None else config
    frozen_artifacts = training_phase_artifacts
    if frozen_artifacts is None:
        frozen_artifacts = run_training_phase(
            seeds,
            inference_partition_capacity=resolved_config.inference_partition_capacity,
        )

    outputs: list[SeedExposureArtifacts] = []
    for frozen in frozen_artifacts:
        checkpoint_summaries: list[CheckpointProbeSummary] = []
        layer1_by_checkpoint: dict[int, np.ndarray] = {}
        layer2_by_checkpoint: dict[int, np.ndarray] = {}

        summary0, arrays0 = _probe_checkpoint(frozen, 0, resolved_config)
        checkpoint_summaries.append(summary0)
        layer1_by_checkpoint[0] = arrays0["layer1"]
        layer2_by_checkpoint[0] = arrays0["layer2"]

        completed_rounds = 0
        for target_round in resolved_config.checkpoint_rounds[1:]:
            rounds_to_run = target_round - completed_rounds
            _run_exposure_updates(frozen, rounds_to_run, resolved_config)
            completed_rounds = target_round

            summary, arrays = _probe_checkpoint(frozen, target_round, resolved_config)
            checkpoint_summaries.append(summary)
            layer1_by_checkpoint[target_round] = arrays["layer1"]
            layer2_by_checkpoint[target_round] = arrays["layer2"]

        layer1_analysis = _analyze_seed_layer(
            frozen.seed,
            "layer1",
            layer1_by_checkpoint,
            resolved_config,
        )
        layer2_analysis = _analyze_seed_layer(
            frozen.seed,
            "layer2",
            layer2_by_checkpoint,
            resolved_config,
        )

        outputs.append(
            SeedExposureArtifacts(
                seed=frozen.seed,
                checkpoint_summaries=tuple(checkpoint_summaries),
                layer1_analysis=layer1_analysis,
                layer2_analysis=layer2_analysis,
            )
        )

    return outputs
