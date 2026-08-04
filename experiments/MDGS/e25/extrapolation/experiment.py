"""DIGIT Extrapolation E25 — Training loop, familiarity exposure, and LTH procedure.

Deliverable 1 scope:
- Train the E22 dual-parameter architecture across 5 seeds.
- Save the step-0 initialisation checkpoint before any gradient update.
- Compute familiarity exposure scores E_jk for W_semantic_1 and W_semantic_2
  at epoch 5, 10, 20, and the final stopping epoch.
- Run one-shot 50% global magnitude pruning on the semantic weight matrices.
- Rewind surviving weights to step-0 values.
- Retrain the pruned network for the same epoch budget as the dense run.
- Report winning ticket results per seed.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from experiments.DIGIT.Extrapolation.e19.extrapolation.data import (
    LEVEL_NOVEL,
    LEVEL_CORE,
    LEVEL_FAMILIAR,
    LEVEL_RARE,
    LEVEL_NAMES,
    LEVEL_ORDER,
    build_frequency_level_records,
)
from experiments.DIGIT.Extrapolation.e22.extrapolation.models import DualParameterAttentionMLP

from .config import E25Config


# ---------------------------------------------------------------------------
# Determinism helpers
# ---------------------------------------------------------------------------

def set_global_determinism(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(config: E25Config) -> torch.device:
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "E25Config.device is set to CUDA but CUDA is not available. "
            "Override device='cpu' or install a CUDA-enabled PyTorch build."
        )
    return device


# ---------------------------------------------------------------------------
# Dataset — exact E19 four-level frequency construction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SplitTensors:
    part_a: torch.Tensor
    part_b: torch.Tensor
    part_a_noise: torch.Tensor
    part_b_noise: torch.Tensor
    labels: torch.Tensor
    levels: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.labels.shape[0])


@dataclass(frozen=True)
class E25Dataset:
    train: SplitTensors
    test: SplitTensors


def build_dataset(config: E25Config) -> E25Dataset:
    """Materialise the frozen E19 four-level frequency dataset."""
    records = build_frequency_level_records()
    train_noise_gen = torch.Generator()
    train_noise_gen.manual_seed(config.seed + config.train_noise_seed_offset)
    test_noise_gen = torch.Generator()
    test_noise_gen.manual_seed(config.seed + config.test_noise_seed_offset)

    repeats_by_level = {
        LEVEL_CORE: config.core_repeats_per_combo,
        LEVEL_FAMILIAR: config.familiar_repeats_per_combo,
        LEVEL_RARE: config.rare_repeats_per_combo,
        LEVEL_NOVEL: config.novel_repeats_per_combo,
    }

    train_a, train_b, train_a_noise, train_b_noise, train_labels, train_levels = (
        [], [], [], [], [], []
    )
    test_a, test_b, test_a_noise, test_b_noise, test_labels, test_levels = (
        [], [], [], [], [], []
    )

    for record in records:
        for _ in range(repeats_by_level[record.level]):
            train_a.append(record.part_a)
            train_b.append(record.part_b)
            train_a_noise.append(
                torch.randn(config.embedding_dim, generator=train_noise_gen) * config.input_jitter_std
            )
            train_b_noise.append(
                torch.randn(config.embedding_dim, generator=train_noise_gen) * config.input_jitter_std
            )
            train_labels.append(record.class_id)
            train_levels.append(record.level)

        for _ in range(config.test_repeats_per_combo):
            test_a.append(record.part_a)
            test_b.append(record.part_b)
            test_a_noise.append(
                torch.randn(config.embedding_dim, generator=test_noise_gen) * config.input_jitter_std
            )
            test_b_noise.append(
                torch.randn(config.embedding_dim, generator=test_noise_gen) * config.input_jitter_std
            )
            test_labels.append(record.class_id)
            test_levels.append(record.level)

    return E25Dataset(
        train=SplitTensors(
            part_a=torch.tensor(train_a, dtype=torch.long),
            part_b=torch.tensor(train_b, dtype=torch.long),
            part_a_noise=torch.stack(train_a_noise).float(),
            part_b_noise=torch.stack(train_b_noise).float(),
            labels=torch.tensor(train_labels, dtype=torch.long),
            levels=torch.tensor(train_levels, dtype=torch.long),
        ),
        test=SplitTensors(
            part_a=torch.tensor(test_a, dtype=torch.long),
            part_b=torch.tensor(test_b, dtype=torch.long),
            part_a_noise=torch.stack(test_a_noise).float(),
            part_b_noise=torch.stack(test_b_noise).float(),
            labels=torch.tensor(test_labels, dtype=torch.long),
            levels=torch.tensor(test_levels, dtype=torch.long),
        ),
    )


def make_loader(
    split: SplitTensors,
    config: E25Config,
    *,
    shuffle: bool,
) -> DataLoader[tuple[torch.Tensor, ...]]:
    gen = torch.Generator()
    gen.manual_seed(config.seed)
    ds = TensorDataset(
        split.part_a,
        split.part_b,
        split.part_a_noise,
        split.part_b_noise,
        split.labels,
        split.levels,
    )
    return DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=shuffle,
        generator=gen if shuffle else None,
        pin_memory=config.device.startswith("cuda"),
    )


def _move_batch(batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(t.to(device, non_blocking=device.type == "cuda") for t in batch)


# ---------------------------------------------------------------------------
# Model factory — identical parameterisation to E22
# ---------------------------------------------------------------------------

def make_model(config: E25Config) -> DualParameterAttentionMLP:
    return DualParameterAttentionMLP(
        part_a_vocab_size=config.num_part_a_values,
        part_b_vocab_size=config.num_part_b_values,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        key_dim=config.key_dim,
        num_classes=config.num_classes,
        num_prototypes=config.num_prototypes,
        prototype_lambda=config.prototype_lambda,
        eps=config.prototype_eps,
    )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_accuracy(
    model: DualParameterAttentionMLP,
    split: SplitTensors,
    config: E25Config,
) -> float:
    """Return overall test accuracy. Model must already be on device."""
    device = resolve_device(config)
    loader = make_loader(split, config, shuffle=False)
    model.eval()
    total_correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            part_a, part_b, pa_n, pb_n, labels, _ = _move_batch(batch, device)
            outputs = model(part_a, part_b, part_a_noise=pa_n, part_b_noise=pb_n)
            preds = outputs.logits.argmax(dim=-1)
            total_correct += int((preds == labels).sum().item())
            total += labels.size(0)
    return total_correct / total


def evaluate_accuracy_by_level(
    model: DualParameterAttentionMLP,
    split: SplitTensors,
    config: E25Config,
) -> dict[str, float]:
    """Return {level_name: accuracy} for all four levels."""
    device = resolve_device(config)
    loader = make_loader(split, config, shuffle=False)
    model.eval()
    correct: dict[int, int] = {lv: 0 for lv in LEVEL_ORDER}
    total: dict[int, int] = {lv: 0 for lv in LEVEL_ORDER}
    with torch.no_grad():
        for batch in loader:
            part_a, part_b, pa_n, pb_n, labels, levels = _move_batch(batch, device)
            outputs = model(part_a, part_b, part_a_noise=pa_n, part_b_noise=pb_n)
            preds = outputs.logits.argmax(dim=-1).cpu()
            labels_cpu = labels.cpu()
            levels_cpu = levels.cpu()
            for lv in LEVEL_ORDER:
                mask = levels_cpu == lv
                if mask.sum().item() == 0:
                    continue
                correct[lv] += int((preds[mask] == labels_cpu[mask]).sum().item())
                total[lv] += int(mask.sum().item())
    return {
        LEVEL_NAMES[lv]: (correct[lv] / total[lv] if total[lv] > 0 else float("nan"))
        for lv in LEVEL_ORDER
    }


# ---------------------------------------------------------------------------
# Familiarity exposure score computation
# ---------------------------------------------------------------------------

def compute_exposure_scores(
    model: DualParameterAttentionMLP,
    dataset: E25Dataset,
    config: E25Config,
) -> dict[str, np.ndarray]:
    """Compute E_jk for W_semantic_1 and W_semantic_2 over the full training set.

    Returns {"layer1": ndarray (32,32), "layer2": ndarray (32,32)}.

    This is a frozen evaluation pass — no weight updates, no prototype updates.

    For layer 1:
      - pre-weight activation a_k(x_i): the concatenated input embedding x_0 (B x 32)
      - post-ReLU M_j(x_i): (h_semantic_1 > 0) (B x 32)
      - F_l(x_i): layer-1 certainty from e22 max-cosine formula

    For layer 2:
      - pre-weight activation a_k(x_i): h_out_1, the output of layer 1 (B x 32)
      - post-ReLU M_j(x_i): (h_semantic_2 > 0) (B x 32)
      - F_l(x_i): layer-2 certainty from e22 max-cosine formula
    """
    device = resolve_device(config)
    loader = make_loader(dataset.train, config, shuffle=False)
    model.eval()

    eps = config.exposure_eps
    d = config.hidden_dim  # 32

    # Accumulate numerator and denominator for both layers
    # E_jk = sum_i [F_i * M_j(x_i) * |a_k(x_i)|] / (sum_i [M_j(x_i) * |a_k(x_i)|] + eps)
    # numerator shape: (d_out=32, d_in=32) — equivalent to M^T @ (F * a_mag)
    # denominator shape: (d_out=32, d_in=32) — equivalent to M^T @ a_mag

    num1 = torch.zeros(d, d, dtype=torch.float64)
    den1 = torch.zeros(d, d, dtype=torch.float64)
    num2 = torch.zeros(d, d, dtype=torch.float64)
    den2 = torch.zeros(d, d, dtype=torch.float64)

    with torch.no_grad():
        for batch in loader:
            part_a, part_b, pa_n, pb_n, _labels, _levels = _move_batch(batch, device)
            outputs = model(part_a, part_b, part_a_noise=pa_n, part_b_noise=pb_n)

            # --- Layer 1 ---
            # F_1: layer-1 certainty (B,) — already computed by the model
            F1 = outputs.layer1_certainty.detach().cpu().double()  # (B,)

            # a_k for layer 1 = |input_embedding| = |x_0|
            a1 = outputs.input_embedding.detach().cpu().double().abs()  # (B, 32)

            # M_j for layer 1 = (h_semantic_1 > 0)
            M1 = (outputs.layer1_semantic.detach().cpu() > 0).double()  # (B, 32)

            # weighted_a1 = F1[:,None] * a1  — (B, 32)
            weighted_a1 = F1.unsqueeze(1) * a1  # (B, 32)

            # numerator: M1^T @ weighted_a1  — (32, 32)
            num1 += M1.t() @ weighted_a1
            # denominator: M1^T @ a1  — (32, 32)
            den1 += M1.t() @ a1

            # --- Layer 2 ---
            # F_2: layer-2 certainty (B,)
            F2 = outputs.layer2_certainty.detach().cpu().double()  # (B,)

            # a_k for layer 2 = |h_out_1|
            a2 = outputs.layer1_output.detach().cpu().double().abs()  # (B, 32)

            # M_j for layer 2 = (h_semantic_2 > 0)
            M2 = (outputs.layer2_semantic.detach().cpu() > 0).double()  # (B, 32)

            weighted_a2 = F2.unsqueeze(1) * a2         # (B, 32)
            num2 += M2.t() @ weighted_a2               # (32, 32)
            den2 += M2.t() @ a2                        # (32, 32)

    E1 = (num1 / (den1 + eps)).float().numpy()
    E2 = (num2 / (den2 + eps)).float().numpy()

    return {"layer1": E1, "layer2": E2}


# ---------------------------------------------------------------------------
# LTH procedure
# ---------------------------------------------------------------------------

@dataclass
class PruningMask:
    """Per-layer binary pruning masks and the global threshold used."""
    mask1: np.ndarray   # (32, 32) — 1=survive, 0=pruned
    mask2: np.ndarray   # (32, 32)
    threshold: float
    layer1_sparsity: float
    layer2_sparsity: float
    global_sparsity: float


def compute_global_pruning_mask(
    w1: torch.Tensor,
    w2: torch.Tensor,
    sparsity: float = 0.5,
) -> PruningMask:
    """One-shot global magnitude pruning across W_semantic_1 and W_semantic_2 jointly."""
    flat = torch.cat([w1.detach().flatten().abs(), w2.detach().flatten().abs()])
    n_total = flat.numel()
    n_prune = int(round(sparsity * n_total))
    sorted_mags, _ = flat.sort()
    threshold = float(sorted_mags[n_prune - 1].item())

    mask1_t = (w1.detach().abs() > threshold).long()
    mask2_t = (w2.detach().abs() > threshold).long()

    n1_pruned = int((mask1_t == 0).sum().item())
    n2_pruned = int((mask2_t == 0).sum().item())
    n1_total = mask1_t.numel()
    n2_total = mask2_t.numel()

    return PruningMask(
        mask1=mask1_t.cpu().numpy(),
        mask2=mask2_t.cpu().numpy(),
        threshold=threshold,
        layer1_sparsity=n1_pruned / n1_total,
        layer2_sparsity=n2_pruned / n2_total,
        global_sparsity=(n1_pruned + n2_pruned) / (n1_total + n2_total),
    )


@dataclass
class LTHResult:
    """Per-seed LTH outcome."""
    seed: int
    dense_withheld_accuracy: float
    dense_stopping_epoch: int
    pruned_withheld_accuracy: float
    pruned_stopping_epoch: int
    accuracy_ratio: float          # pruned / dense
    winning_ticket: bool           # ratio >= 0.99
    global_sparsity: float
    layer1_sparsity: float
    layer2_sparsity: float
    pruning_threshold: float


def run_lth_for_seed(
    config: E25Config,
    dense_model: DualParameterAttentionMLP,
    dataset: E25Dataset,
    dense_withheld_accuracy: float,
    dense_stopping_epoch: int,
    init_checkpoint: dict[str, torch.Tensor],
    pruning_mask: PruningMask,
) -> LTHResult:
    """Run the LTH retraining procedure for one seed.

    Takes the trained dense model, prunes it globally, rewinds survivors to
    init_checkpoint values, retrains for dense_stopping_epoch epochs, and
    records whether the pruned network reaches the 99% threshold.
    """
    device = resolve_device(config)

    # Build a fresh model and copy the dense model's full state
    pruned_model = make_model(config).to(device)
    pruned_model.load_state_dict(copy.deepcopy(dense_model.state_dict()))

    # Convert masks to tensors on device
    mask1 = torch.tensor(pruning_mask.mask1, dtype=torch.float32, device=device)
    mask2 = torch.tensor(pruning_mask.mask2, dtype=torch.float32, device=device)

    # Rewind surviving weights to initialisation values.
    # Pruned positions are set to zero.
    with torch.no_grad():
        init_w1 = init_checkpoint["w_semantic_1"].to(device)
        init_w2 = init_checkpoint["w_semantic_2"].to(device)
        pruned_model.layer1.semantic.weight.copy_(init_w1 * mask1)
        pruned_model.layer2.semantic.weight.copy_(init_w2 * mask2)

    # Verify rewinding: surviving positions must match init_checkpoint exactly
    with torch.no_grad():
        w1_now = pruned_model.layer1.semantic.weight
        w2_now = pruned_model.layer2.semantic.weight
        surviving1 = mask1.bool()
        surviving2 = mask2.bool()
        max_dev1 = (w1_now[surviving1] - init_w1[surviving1]).abs().max().item()
        max_dev2 = (w2_now[surviving2] - init_w2[surviving2]).abs().max().item()
        assert max_dev1 < 1e-7, f"Seed {config.seed}: W1 rewind deviation {max_dev1:.2e}"
        assert max_dev2 < 1e-7, f"Seed {config.seed}: W2 rewind deviation {max_dev2:.2e}"

    # Retrain the pruned network
    optimizer = torch.optim.Adam(pruned_model.parameters(), lr=config.learning_rate)
    train_loader = make_loader(dataset.train, config, shuffle=True)

    best_loss = float("inf")
    epochs_without_improvement = 0
    stopping_epoch = 0

    for epoch in range(1, dense_stopping_epoch + 1):
        pruned_model.train()
        epoch_loss_sum = 0.0
        epoch_examples = 0

        for batch in train_loader:
            part_a, part_b, pa_n, pb_n, labels, _ = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = pruned_model(part_a, part_b, part_a_noise=pa_n, part_b_noise=pb_n)
            loss = F.cross_entropy(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            # Enforce pruning mask: zero out pruned positions after every step
            with torch.no_grad():
                pruned_model.layer1.semantic.weight.mul_(mask1)
                pruned_model.layer2.semantic.weight.mul_(mask2)


            epoch_loss_sum += float(loss.item()) * labels.size(0)
            epoch_examples += labels.size(0)

        epoch_loss = epoch_loss_sum / epoch_examples
        stopping_epoch = epoch

        if best_loss - epoch_loss > config.loss_improvement_tolerance:
            best_loss = epoch_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.loss_patience_epochs:
            break

    pruned_accuracy = evaluate_accuracy(pruned_model, dataset.test, config)
    ratio = pruned_accuracy / dense_withheld_accuracy if dense_withheld_accuracy > 0 else 0.0

    return LTHResult(
        seed=config.seed,
        dense_withheld_accuracy=dense_withheld_accuracy,
        dense_stopping_epoch=dense_stopping_epoch,
        pruned_withheld_accuracy=pruned_accuracy,
        pruned_stopping_epoch=stopping_epoch,
        accuracy_ratio=ratio,
        winning_ticket=ratio >= config.lth_performance_threshold,
        global_sparsity=pruning_mask.global_sparsity,
        layer1_sparsity=pruning_mask.layer1_sparsity,
        layer2_sparsity=pruning_mask.layer2_sparsity,
        pruning_threshold=pruning_mask.threshold,
    )


# ---------------------------------------------------------------------------
# Per-seed training artefacts
# ---------------------------------------------------------------------------

@dataclass
class E25SeedArtifacts:
    """Everything produced for one seed in Deliverable 1."""
    config: E25Config
    seed: int

    # Dense training
    dense_stopping_epoch: int
    dense_train_accuracy: float
    dense_withheld_accuracy: float
    dense_withheld_by_level: dict[str, float]
    dense_layer1_occupancy: int
    dense_layer2_occupancy: int

    # Familiarity exposure scores at each checkpoint
    # keys: "epoch_5", "epoch_10", "epoch_20", "final"
    exposure_layer1: dict[str, np.ndarray]   # checkpoint_label -> (32,32)
    exposure_layer2: dict[str, np.ndarray]

    # Weight matrices at each checkpoint (for baseline comparison in Deliverable 2)
    weights_layer1: dict[str, np.ndarray]    # checkpoint_label -> (32,32)
    weights_layer2: dict[str, np.ndarray]

    # Pruning mask
    pruning_mask: PruningMask

    # LTH result
    lth: LTHResult


# ---------------------------------------------------------------------------
# Main per-seed training function
# ---------------------------------------------------------------------------

def train_single_seed(config: E25Config) -> E25SeedArtifacts:
    """Train the dense E22 model, log checkpoints, prune, rewind, retrain."""
    set_global_determinism(config.seed)
    dataset = build_dataset(config)
    device = resolve_device(config)
    model = make_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_loader = make_loader(dataset.train, config, shuffle=True)

    # Save step-0 initialisation checkpoint BEFORE any gradient update
    with torch.no_grad():
        init_checkpoint = {
            "w_semantic_1": model.layer1.semantic.weight.detach().clone().cpu(),
            "w_semantic_2": model.layer2.semantic.weight.detach().clone().cpu(),
        }

    exposure_layer1: dict[str, np.ndarray] = {}
    exposure_layer2: dict[str, np.ndarray] = {}
    weights_layer1: dict[str, np.ndarray] = {}
    weights_layer2: dict[str, np.ndarray] = {}

    checkpoint_epochs = set(config.exposure_checkpoint_epochs)

    best_loss = float("inf")
    epochs_without_improvement = 0
    final_train_loss = float("inf")
    final_train_accuracy = 0.0
    stopping_epoch = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_correct = 0
        epoch_examples = 0

        for batch in train_loader:
            part_a, part_b, pa_n, pb_n, labels, _ = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(part_a, part_b, part_a_noise=pa_n, part_b_noise=pb_n)
            loss = F.cross_entropy(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            # Prototype update: run on CPU to avoid per-example GPU sync from
            # the sequential .item() calls inside the EMA loop.
            # Move prototype buffers to CPU, update, restore to GPU.
            h1_cpu = outputs.layer1_semantic.detach().cpu()
            h2_cpu = outputs.layer2_semantic.detach().cpu()
            model.layer1.prototypes = model.layer1.prototypes.cpu()
            model.layer1.valid_mask = model.layer1.valid_mask.cpu()
            model.layer2.prototypes = model.layer2.prototypes.cpu()
            model.layer2.valid_mask = model.layer2.valid_mask.cpu()
            model.update_prototypes(h1_cpu, h2_cpu)
            model.layer1.prototypes = model.layer1.prototypes.to(device)
            model.layer1.valid_mask = model.layer1.valid_mask.to(device)
            model.layer2.prototypes = model.layer2.prototypes.to(device)
            model.layer2.valid_mask = model.layer2.valid_mask.to(device)

            batch_n = labels.size(0)
            epoch_loss_sum += float(loss.item()) * batch_n
            preds = outputs.logits.argmax(dim=-1)
            epoch_correct += int((preds == labels).sum().item())
            epoch_examples += batch_n

        final_train_loss = epoch_loss_sum / epoch_examples
        final_train_accuracy = epoch_correct / epoch_examples
        stopping_epoch = epoch

        # Familiarity exposure checkpoint
        if epoch in checkpoint_epochs:
            label = f"epoch_{epoch}"
            scores = compute_exposure_scores(model, dataset, config)
            exposure_layer1[label] = scores["layer1"]
            exposure_layer2[label] = scores["layer2"]
            with torch.no_grad():
                weights_layer1[label] = model.layer1.semantic.weight.detach().cpu().numpy().copy()
                weights_layer2[label] = model.layer2.semantic.weight.detach().cpu().numpy().copy()

        # Early stopping check
        if best_loss - final_train_loss > config.loss_improvement_tolerance:
            best_loss = final_train_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.loss_patience_epochs:
            break

    # Final checkpoint (always, regardless of epoch number)
    final_scores = compute_exposure_scores(model, dataset, config)
    exposure_layer1["final"] = final_scores["layer1"]
    exposure_layer2["final"] = final_scores["layer2"]
    with torch.no_grad():
        weights_layer1["final"] = model.layer1.semantic.weight.detach().cpu().numpy().copy()
        weights_layer2["final"] = model.layer2.semantic.weight.detach().cpu().numpy().copy()

    # Evaluate dense model
    dense_withheld = evaluate_accuracy(model, dataset.test, config)
    dense_by_level = evaluate_accuracy_by_level(model, dataset.test, config)
    dense_layer1_occ = model.layer1.occupancy
    dense_layer2_occ = model.layer2.occupancy

    # Compute global 50% pruning mask from post-training semantic weights
    w1 = model.layer1.semantic.weight.detach().cpu()
    w2 = model.layer2.semantic.weight.detach().cpu()
    pruning_mask = compute_global_pruning_mask(w1, w2, sparsity=config.lth_sparsity)

    # Run LTH retraining
    lth = run_lth_for_seed(
        config=config,
        dense_model=model,
        dataset=dataset,
        dense_withheld_accuracy=dense_withheld,
        dense_stopping_epoch=stopping_epoch,
        init_checkpoint=init_checkpoint,
        pruning_mask=pruning_mask,
    )

    return E25SeedArtifacts(
        config=config,
        seed=config.seed,
        dense_stopping_epoch=stopping_epoch,
        dense_train_accuracy=final_train_accuracy,
        dense_withheld_accuracy=dense_withheld,
        dense_withheld_by_level=dense_by_level,
        dense_layer1_occupancy=dense_layer1_occ,
        dense_layer2_occupancy=dense_layer2_occ,
        exposure_layer1=exposure_layer1,
        exposure_layer2=exposure_layer2,
        weights_layer1=weights_layer1,
        weights_layer2=weights_layer2,
        pruning_mask=pruning_mask,
        lth=lth,
    )


def run_deliverable1(
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    device: str = "cpu",
) -> list[E25SeedArtifacts]:
    """Train all seeds sequentially and return per-seed artefacts."""
    all_artifacts: list[E25SeedArtifacts] = []
    for seed in seeds:
        print(f"  Seed {seed}: training dense model...", flush=True)
        config = E25Config(seed=seed, device=device)
        artifacts = train_single_seed(config)
        print(
            f"  Seed {seed}: done. "
            f"dense_acc={artifacts.dense_withheld_accuracy:.4f}, "
            f"stopping_epoch={artifacts.dense_stopping_epoch}, "
            f"pruned_acc={artifacts.lth.pruned_withheld_accuracy:.4f}, "
            f"winning_ticket={artifacts.lth.winning_ticket}",
            flush=True,
        )
        all_artifacts.append(artifacts)
    return all_artifacts
