import numpy as np
import torch

from src.datasets.strict_coordination import StrictCoordinationConfig, build_strict_coordination_splits
from src.experiments.shared_weight_cloned_agents import (
    RoleAwareQKVCoordinator,
    SharedAgentConfig,
    SharedCoordinatorConfig,
    SharedTrainingConfig,
    fit_shared_weight_model,
    shared_parameter_identity_check,
    _clone_split,
)


def _tiny_splits():
    dataset = StrictCoordinationConfig(n_train=32, n_dev=16, n_test=16, n_evidence_bits=4)
    examples = build_strict_coordination_splits(dataset, seed=123)
    return {name: _clone_split(name, rows, n_agents=4) for name, rows in examples.items()}, dataset


def test_trainable_shared_agent_receives_group_loss_gradients_and_updates() -> None:
    splits, dataset = _tiny_splits()
    agent_config = SharedAgentConfig(
        n_agents=4,
        hidden_dim=8,
        num_layers=1,
        num_heads=1,
        ff_dim=16,
        evidence_embedding_scale=0.01,
    )
    result = fit_shared_weight_model(
        splits=splits,
        agent_config=agent_config,
        coordinator_config=SharedCoordinatorConfig(num_layers=1, num_heads=1, ff_dim=16),
        training_config=SharedTrainingConfig(epochs=2, batch_size=16, lr=0.005, patience=2),
        num_classes=dataset.num_classes,
        seed=5,
        device="cpu",
        trainable_agent=True,
        method="trainable_shared_agent_coordinator",
    )
    audit = result.audit
    assert audit["shared_parameter_identity"] is True
    assert audit["agent_grad_norm_mean"] > 0.0
    assert audit["coordinator_grad_norm_mean"] > 0.0
    assert audit["agent_parameter_delta"] > 0.0
    assert audit["coordinator_parameter_delta"] > 0.0
    assert audit["clone_activation_requires_grad"] is True
    assert audit["loss_backward_reaches_agent"] is True
    assert audit["per_clone_gradient_contribution"] is True
    assert all(value > 0.0 for value in audit["per_clone_activation_grad_norms"])


def test_frozen_shared_agent_does_not_update_but_coordinator_does() -> None:
    splits, dataset = _tiny_splits()
    agent_config = SharedAgentConfig(
        n_agents=4,
        hidden_dim=8,
        num_layers=1,
        num_heads=1,
        ff_dim=16,
        evidence_embedding_scale=0.01,
    )
    result = fit_shared_weight_model(
        splits=splits,
        agent_config=agent_config,
        coordinator_config=SharedCoordinatorConfig(num_layers=1, num_heads=1, ff_dim=16),
        training_config=SharedTrainingConfig(epochs=1, batch_size=16, lr=0.005, patience=1),
        num_classes=dataset.num_classes,
        seed=7,
        device="cpu",
        trainable_agent=False,
        method="frozen_shared_agent_coordinator",
    )
    audit = result.audit
    assert audit["shared_parameter_identity"] is True
    assert audit["agent_grad_norm_mean"] == 0.0
    assert audit["agent_parameter_delta"] == 0.0
    assert audit["coordinator_grad_norm_mean"] > 0.0
    assert audit["coordinator_parameter_delta"] > 0.0


def test_role_aware_coordinator_preserves_physical_order_when_roles_move_with_hidden_states() -> None:
    torch.manual_seed(11)
    coordinator = RoleAwareQKVCoordinator(
        n_roles=4,
        hidden_dim=8,
        num_classes=4,
        config=SharedCoordinatorConfig(num_layers=1, num_heads=1, ff_dim=16),
    )
    coordinator.eval()
    hidden = torch.randn(6, 4, 8)
    roles = torch.arange(4).view(1, 4).expand(6, 4)
    order = torch.as_tensor(np.asarray([[2, 0, 3, 1]] * 6), dtype=torch.long)
    permuted_hidden = hidden.gather(1, order.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
    permuted_roles = roles.gather(1, order)
    with torch.no_grad():
        base_logits = coordinator(hidden, roles)
        permuted_logits = coordinator(permuted_hidden, permuted_roles)
    assert torch.allclose(base_logits, permuted_logits, atol=1e-6)


def test_shared_parameter_identity_check_uses_one_module_for_all_clones() -> None:
    splits, dataset = _tiny_splits()
    result = fit_shared_weight_model(
        splits=splits,
        agent_config=SharedAgentConfig(n_agents=4, hidden_dim=8, num_layers=1, num_heads=1, ff_dim=16),
        coordinator_config=SharedCoordinatorConfig(num_layers=1, num_heads=1, ff_dim=16),
        training_config=SharedTrainingConfig(epochs=1, batch_size=16, lr=0.005, patience=1),
        num_classes=dataset.num_classes,
        seed=9,
        device="cpu",
        trainable_agent=True,
        method="trainable_shared_agent_coordinator",
    )
    identity = shared_parameter_identity_check(result.agent, n_clones=4)
    assert identity["pass"] is True
    assert len({tuple(ids) for ids in identity["clone_parameter_ids"]}) == 1
