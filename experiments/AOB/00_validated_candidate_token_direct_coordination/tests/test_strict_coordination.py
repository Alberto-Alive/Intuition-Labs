from collections import Counter

import numpy as np

from src.agents.strict_coordination import StrictCoordinationAgentConfig, StrictCoordinationSwarm
from src.datasets.strict_coordination import (
    StrictCoordinationConfig,
    build_strict_coordination_splits,
    strict_label,
)


def test_strict_coordination_labels_are_balanced_and_deterministic() -> None:
    config = StrictCoordinationConfig(n_train=200, n_dev=50, n_test=50)
    splits = build_strict_coordination_splits(config, seed=0)
    labels = [example.label for example in splits["train"]]
    counts = Counter(labels)
    assert set(counts) == {0, 1, 2, 3}
    assert max(counts.values()) - min(counts.values()) < 25
    for example in splits["train"][:20]:
        assert example.label == strict_label(example.evidence_bits)


def test_strict_coordination_attempt_batch_shapes() -> None:
    dataset_config = StrictCoordinationConfig(n_train=20, n_dev=10, n_test=10)
    splits = build_strict_coordination_splits(dataset_config, seed=1)
    agent_config = StrictCoordinationAgentConfig(n_agents=5, hidden_dim=8, num_layers=2)
    swarm = StrictCoordinationSwarm(
        agent_config,
        n_evidence_bits=dataset_config.n_evidence_bits,
        num_classes=dataset_config.num_classes,
        seed=1,
    )
    batches = swarm.run(splits)
    train = batches["train"]
    assert train.answers.shape == (20, 5)
    assert train.hidden_states.shape == (20, 5, 2, 8)
    assert np.all(train.answers >= 0)
    assert np.all(train.answers < dataset_config.num_classes)

