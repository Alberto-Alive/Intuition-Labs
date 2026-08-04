import numpy as np

from src.agents.simulated import SimulatedAgentConfig, SimulatedAgentSwarm
from src.agents.strict_coordination import StrictCoordinationAgentConfig, StrictCoordinationSwarm
from src.datasets.strict_coordination import StrictCoordinationConfig, build_strict_coordination_splits
from src.datasets.synthetic import SyntheticDatasetConfig, build_synthetic_splits


def test_simulated_telemetry_channels_reconstruct_hidden_state() -> None:
    dataset_config = SyntheticDatasetConfig(n_train=12, n_dev=4, n_test=4)
    splits = build_synthetic_splits(dataset_config, seed=3)
    swarm = SimulatedAgentSwarm(
        SimulatedAgentConfig(n_agents=3, hidden_dim=8, num_layers=2),
        num_tasks=dataset_config.num_tasks,
        num_classes=dataset_config.num_classes,
        seed=3,
    )
    batch = swarm.run(splits)["train"]
    reconstructed = sum(batch.telemetry_channels.values())
    assert np.allclose(reconstructed, batch.hidden_states, atol=1e-5)


def test_strict_telemetry_channels_reconstruct_hidden_state() -> None:
    dataset_config = StrictCoordinationConfig(n_train=12, n_dev=4, n_test=4)
    splits = build_strict_coordination_splits(dataset_config, seed=4)
    swarm = StrictCoordinationSwarm(
        StrictCoordinationAgentConfig(n_agents=5, hidden_dim=8, num_layers=2),
        n_evidence_bits=dataset_config.n_evidence_bits,
        num_classes=dataset_config.num_classes,
        seed=4,
    )
    batch = swarm.run(splits)["train"]
    reconstructed = sum(batch.telemetry_channels.values())
    assert np.allclose(reconstructed, batch.hidden_states, atol=1e-5)

