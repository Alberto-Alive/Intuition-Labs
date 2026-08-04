import numpy as np

from src.agents.simulated import SimulatedAgentConfig, SimulatedAgentSwarm
from src.datasets.synthetic import SyntheticDatasetConfig, build_synthetic_splits
from src.telemetry.controls import apply_activation_control, randomized_train_labels


def _batches():
    dataset_config = SyntheticDatasetConfig(n_train=40, n_dev=20, n_test=20)
    splits = build_synthetic_splits(dataset_config, seed=0)
    swarm = SimulatedAgentSwarm(
        SimulatedAgentConfig(n_agents=3, hidden_dim=8, num_layers=2),
        num_tasks=dataset_config.num_tasks,
        num_classes=dataset_config.num_classes,
        seed=0,
    )
    return swarm.run(splits), dataset_config


def test_activation_controls_preserve_visible_outputs() -> None:
    batches, _ = _batches()
    for control in [
        "shuffled_activations",
        "random_activations",
        "wrong_task_activations",
        "wrong_agent_activations",
    ]:
        controlled = apply_activation_control(batches, control=control, seed=7)
        for split in ["train", "dev", "test"]:
            assert controlled[split].hidden_states.shape == batches[split].hidden_states.shape
            assert np.array_equal(controlled[split].answers, batches[split].answers)
            assert np.array_equal(controlled[split].confidences, batches[split].confidences)
            assert not np.allclose(controlled[split].hidden_states, batches[split].hidden_states)


def test_randomized_labels_change_train_labels_only_in_copy() -> None:
    batches, dataset_config = _batches()
    randomized = randomized_train_labels(batches["train"], seed=9, num_classes=dataset_config.num_classes)
    assert randomized.labels.shape == batches["train"].labels.shape
    assert not np.array_equal(randomized.labels, batches["train"].labels)
    assert np.array_equal(batches["train"].answers, randomized.answers)
    assert np.array_equal(batches["train"].labels, batches["train"].labels)

