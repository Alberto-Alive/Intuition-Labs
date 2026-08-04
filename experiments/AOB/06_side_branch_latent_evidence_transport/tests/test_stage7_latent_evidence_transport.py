from pathlib import Path

import torch
from torch.nn import functional as F

from src.coordinators.latent_coordination import LatentCoordinatorConfig
from src.datasets.multiview_code_patch_selection import MultiViewCodePatchDatasetConfig, build_multiview_code_patch_splits
from src.experiments.real_shared_weight_latent_coordination import (
    MessageChannelConfig,
    RealSharedWeightTrainingConfig,
    SharedTransformerAgentConfig,
    fit_latent_system,
    predict_latent_system,
)
from src.models.latent_evidence_transport import LatentEvidenceTransportConfig, LatentEvidenceTransportCoordinator


def _inputs(batch: int = 3, roles: int = 4, tokens: int = 6, dim: int = 16, candidates: int = 8):
    torch.manual_seed(7)
    token_states = torch.randn(batch, roles, tokens, dim)
    token_mask = torch.ones(batch, roles, tokens, dtype=torch.bool)
    candidate_features = torch.randn(batch, candidates, 4)
    clone_activations = torch.randn(batch, roles, dim)
    role_ids = torch.arange(roles, dtype=torch.long).view(1, roles).expand(batch, roles)
    labels = torch.tensor([0, 3, 7], dtype=torch.long)[:batch]
    return clone_activations, role_ids, candidate_features, token_states, token_mask, labels


def _coordinator(config: LatentEvidenceTransportConfig | None = None) -> LatentEvidenceTransportCoordinator:
    return LatentEvidenceTransportCoordinator(
        n_roles=4,
        candidate_feature_dim=4,
        input_dim=16,
        model_dim=16,
        num_heads=2,
        ff_dim=32,
        dropout=0.0,
        transport_config=config or LatentEvidenceTransportConfig(refinement_steps=2),
    )


def test_stage7_transport_forward_auxiliary_loss_and_gradients() -> None:
    model = _coordinator(
        LatentEvidenceTransportConfig(
            family="D_energy_margin_refinement",
            refinement_steps=2,
            aux_loss="margin_improvement",
            aux_weight=0.10,
            gated_residual=True,
        )
    )
    clone_activations, role_ids, candidate_features, token_states, token_mask, labels = _inputs()
    logits = model(clone_activations, role_ids, candidate_features, token_states, token_mask)
    assert logits.shape == (3, 8)
    assert len(model._last_trajectory_logits) == 3

    loss = F.cross_entropy(logits, labels) + model.auxiliary_loss(labels)
    loss.backward()
    assert any(parameter.grad is not None and parameter.grad.detach().abs().sum() > 0 for parameter in model.parameters())


def test_stage7_transport_preserves_candidate_order_equivariance() -> None:
    model = _coordinator(
        LatentEvidenceTransportConfig(
            family="C_latent_workspace",
            refinement_steps=2,
            workspace_slots=4,
            workspace_init="learned",
            candidate_writes_workspace=True,
            update_evidence=True,
        )
    )
    model.eval()
    clone_activations, role_ids, candidate_features, token_states, token_mask, _labels = _inputs()
    with torch.no_grad():
        logits = model(clone_activations, role_ids, candidate_features, token_states, token_mask)
        order = torch.tensor([2, 0, 7, 1, 6, 3, 5, 4], dtype=torch.long)
        shuffled_logits = model(clone_activations, role_ids, candidate_features.index_select(1, order), token_states, token_mask)
    assert torch.allclose(shuffled_logits, logits.index_select(1, order), atol=1e-5)


def test_stage7_transport_preserves_physical_role_and_evidence_order_invariance() -> None:
    model = _coordinator(LatentEvidenceTransportConfig(refinement_steps=2, update_evidence=True, gated_residual=True))
    model.eval()
    clone_activations, role_ids, candidate_features, token_states, token_mask, _labels = _inputs()
    with torch.no_grad():
        logits = model(clone_activations, role_ids, candidate_features, token_states, token_mask)
        role_order = torch.tensor([2, 0, 3, 1], dtype=torch.long)
        token_order = torch.tensor([4, 2, 0, 5, 1, 3], dtype=torch.long)
        role_logits = model(
            clone_activations.index_select(1, role_order),
            role_ids.index_select(1, role_order),
            candidate_features,
            token_states.index_select(1, role_order),
            token_mask.index_select(1, role_order),
        )
        token_logits = model(
            clone_activations,
            role_ids,
            candidate_features,
            token_states.index_select(2, token_order),
            token_mask.index_select(2, token_order),
        )
    assert torch.allclose(role_logits, logits, atol=1e-5)
    assert torch.allclose(token_logits, logits, atol=1e-5)


def test_stage7_transport_integrates_with_shared_agent_training_and_frozen_comparator() -> None:
    config = MultiViewCodePatchDatasetConfig(
        n_train=8,
        n_dev=4,
        n_test=4,
        num_candidates=8,
        n_views=4,
        max_files=6,
        snippet_radius=1,
    )
    splits = build_multiview_code_patch_splits(config, seed=711, repo_root=Path("."))
    agent_config = SharedTransformerAgentConfig(
        agent_mode="tiny_transformer",
        max_length=80,
        hidden_dim=16,
        tiny_vocab_size=512,
        tiny_layers=1,
        tiny_heads=1,
        tiny_ff_dim=32,
        adapter_hidden_dim=16,
    )
    coordinator_config = LatentCoordinatorConfig(
        family="latent_evidence_transport",
        input_dim=16,
        model_dim=16,
        num_heads=1,
        num_layers=1,
        ff_dim=32,
        transport_config={
            "family": "unit_candidate_only",
            "refinement_steps": 1,
            "gated_residual": True,
            "aux_loss": "margin_improvement",
            "aux_weight": 0.05,
        },
    )
    training = RealSharedWeightTrainingConfig(epochs=1, batch_size=4, lr=0.003, patience=1)
    message_config = MessageChannelConfig(
        readout_source="pooled",
        use_message_head=False,
        coordinator_family="latent_evidence_transport",
        active_message_layers=(-1,),
        canonicalize_role_order=True,
    )
    trainable = fit_latent_system(
        train_examples=splits["train"],
        dev_examples=splits["dev"],
        agent_config=agent_config,
        coordinator_config=coordinator_config,
        training_config=training,
        num_classes=8,
        seed=81,
        device="cpu",
        trainable_agent=True,
        method="stage7_unit_trainable",
        message_config=message_config,
    )
    frozen = fit_latent_system(
        train_examples=splits["train"],
        dev_examples=splits["dev"],
        agent_config=agent_config,
        coordinator_config=coordinator_config,
        training_config=training,
        num_classes=8,
        seed=82,
        device="cpu",
        trainable_agent=False,
        method="stage7_unit_frozen",
        message_config=message_config,
    )
    assert trainable.audit["uses_latent_evidence_transport"] is True
    assert trainable.audit["agent_grad_norm_mean"] > 0.0
    assert trainable.audit["coordinator_grad_norm_mean"] > 0.0
    assert frozen.audit["agent_grad_norm_mean"] == 0.0
    assert frozen.audit["agent_parameter_delta"] == 0.0
    predictions = predict_latent_system(trainable, splits["dev"], "physical_order_shuffled_roles_avenues_preserved", seed=83)
    assert predictions.shape == (len(splits["dev"]),)

