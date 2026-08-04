import numpy as np

from src.agents.simulated import SimulatedAgentConfig, SimulatedAgentSwarm
from src.coordinators.activation import (
    ActivationClusterRouter,
    ActivationPCAMLPCoordinator,
    ActivationPoolingMLPCoordinator,
    CapacityMatchedTextOnlyCoordinator,
    TextOnlyMLPCoordinator,
)
from src.coordinators.baselines import IndependentSwarmCoordinator, SingleAgentCoordinator
from src.coordinators.mlp import MLPTrainingConfig
from src.coordinators.role_attention import (
    CoordinatorTokenCrossAttentionCoordinator,
    RoleAttentionConfig,
    RoleAwareSelfAttentionCoordinator,
)
from src.datasets.synthetic import SyntheticDatasetConfig, build_synthetic_splits
from src.telemetry.features import activation_features, visible_features


def test_coordinators_fit_and_predict_on_tiny_problem() -> None:
    dataset_config = SyntheticDatasetConfig(n_train=80, n_dev=30, n_test=30)
    splits = build_synthetic_splits(dataset_config, seed=5)
    swarm = SimulatedAgentSwarm(
        SimulatedAgentConfig(n_agents=3, hidden_dim=8, num_layers=2),
        num_tasks=dataset_config.num_tasks,
        num_classes=dataset_config.num_classes,
        seed=5,
    )
    batches = swarm.run(splits)
    training = MLPTrainingConfig(epochs=3, batch_size=32, patience=2, hidden_dims=(16,))
    target_dim = (
        visible_features(batches["train"], dataset_config.num_classes, dataset_config.num_tasks).shape[1]
        + activation_features(batches["train"]).shape[1]
    )
    methods = [
        SingleAgentCoordinator(),
        IndependentSwarmCoordinator(),
        TextOnlyMLPCoordinator(dataset_config.num_classes, dataset_config.num_tasks, training, seed=1, device="cpu"),
        CapacityMatchedTextOnlyCoordinator(
            dataset_config.num_classes,
            dataset_config.num_tasks,
            target_dim=target_dim,
            training=training,
            seed=2,
            device="cpu",
        ),
        ActivationPoolingMLPCoordinator(dataset_config.num_classes, dataset_config.num_tasks, training, seed=3, device="cpu"),
        ActivationPCAMLPCoordinator(
            dataset_config.num_classes,
            dataset_config.num_tasks,
            components=4,
            training=training,
            seed=4,
            device="cpu",
        ),
        ActivationClusterRouter(clusters=4, seed=5),
    ]
    for method in methods:
        method.fit(batches["train"], batches["dev"])
        predictions = method.predict(batches["test"])
        assert predictions.shape == batches["test"].labels.shape
        assert np.all(predictions >= 0)
        assert np.all(predictions < dataset_config.num_classes)


def test_role_attention_coordinators_fit_and_preserve_order_api() -> None:
    dataset_config = SyntheticDatasetConfig(n_train=80, n_dev=30, n_test=30)
    splits = build_synthetic_splits(dataset_config, seed=8)
    swarm = SimulatedAgentSwarm(
        SimulatedAgentConfig(n_agents=4, hidden_dim=12, num_layers=3),
        num_tasks=dataset_config.num_tasks,
        num_classes=dataset_config.num_classes,
        seed=8,
    )
    batches = swarm.run(splits)
    for batch in batches.values():
        batch.private_agent_views = [
            [f"private evidence value BIT_{'ONE' if agent_id % 2 else 'ZERO'}" for agent_id in range(batch.n_agents)]
            for _row_id in range(batch.n_examples)
        ]
    training = MLPTrainingConfig(epochs=2, batch_size=32, patience=1, hidden_dims=(16,))
    configs = [
        RoleAttentionConfig(
            slot_indices=(2,),
            slot_labels=("final:layer_2",),
            family="self_attention",
            variant_name="smoke_self",
            num_attention_layers=1,
            num_heads=1,
            pooling="cls",
            use_aux_private_bit_loss=True,
        ),
        RoleAttentionConfig(
            slot_indices=(2,),
            slot_labels=("final:layer_2",),
            family="cross_attention",
            variant_name="smoke_cross",
            num_attention_layers=1,
            num_heads=1,
            pooling="coordinator",
        ),
    ]
    methods = [
        RoleAwareSelfAttentionCoordinator(dataset_config.num_classes, training, seed=1, device="cpu", config=configs[0]),
        CoordinatorTokenCrossAttentionCoordinator(dataset_config.num_classes, training, seed=2, device="cpu", config=configs[1]),
    ]
    for method in methods:
        method.fit(batches["train"], batches["dev"])
        predictions = method.predict(batches["test"])
        permuted = method.predict_with_permutation(batches["test"], seed=123)
        shuffled_roles = method.predict_with_slot_label_shuffle(batches["test"], seed=456)
        assert predictions.shape == batches["test"].labels.shape
        assert permuted.shape == predictions.shape
        assert shuffled_roles.shape == predictions.shape
        assert np.all(predictions >= 0)
        assert np.all(predictions < dataset_config.num_classes)
