import torch

from src.coordinators.latent_coordination import LatentCoordinatorConfig, make_candidate_query_coordinator
from src.experiments.run_stage5_stacked_coordination import _depth_metrics, _transition_counts, _variant_plan


def test_stacked_candidate_token_direct_outputs_and_diagnostics() -> None:
    config = LatentCoordinatorConfig(
        family="stacked_candidate_token_cross_attention",
        input_dim=12,
        model_dim=16,
        num_heads=2,
        num_layers=3,
        ff_dim=32,
    )
    coordinator = make_candidate_query_coordinator(n_roles=4, candidate_feature_dim=4, config=config)
    coordinator.capture_diagnostics = True
    logits = coordinator(
        torch.randn(5, 4, 12),
        torch.arange(4).view(1, 4).expand(5, 4),
        torch.randn(5, 8, 4),
        torch.randn(5, 4, 7, 12),
        torch.ones(5, 4, 7, dtype=torch.bool),
    )
    assert logits.shape == (5, 8)
    diagnostics = coordinator.last_diagnostics
    assert diagnostics["num_blocks"] == 3
    assert diagnostics["early_logits"].shape == (3, 5, 8)
    assert diagnostics["attention_weights"].shape == (3, 5, 8, 28)
    assert len(diagnostics["representation_change_l2"]) == 3


def test_stage5_depth_transition_metrics() -> None:
    before = torch.tensor([True, False, False, True]).numpy()
    after = torch.tensor([True, True, False, False]).numpy()
    assert _transition_counts(before, after) == {
        "wrong_to_right": 1,
        "right_to_wrong": 1,
        "wrong_to_wrong": 1,
        "right_to_right": 1,
    }

    early_logits = torch.tensor(
        [
            [[4.0, 1.0], [4.0, 1.0], [1.0, 4.0]],
            [[4.0, 1.0], [1.0, 4.0], [4.0, 1.0]],
        ]
    ).numpy()
    labels = torch.tensor([0, 1, 1]).numpy()
    metrics = _depth_metrics(early_logits, labels)
    assert metrics["early_logits_available"] is True
    assert metrics["accuracy_after_intermediate_block"]["block_1"] == 2 / 3
    assert metrics["accuracy_after_intermediate_block"]["block_2"] == 2 / 3
    assert metrics["later_blocks_transition_counts"]["block_1_to_2"]["wrong_to_right"] == 1
    assert metrics["later_blocks_transition_counts"]["block_1_to_2"]["right_to_wrong"] == 1


def test_stage5_default_variants_are_one_to_four_blocks() -> None:
    variants = _variant_plan({"include_optional_bounded_variants": False})
    assert [variant.block_count for variant in variants] == [1, 2, 3, 4]
    assert variants[0].name == "candidate_token_direct_lr3e4_clip1"
