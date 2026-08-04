import numpy as np
import pytest

from src.datasets.swe_patch_selection_dataset import (
    STAGE6_NUM_CANDIDATES,
    apply_stage6_control,
    label_matrix,
    patch_selection_to_multiview_many,
    stage6_output_leakage_audit,
    stage6_split_leakage_audit,
    validate_patch_selection_examples,
)
from src.experiments.stage6_candidate_pool_builder import build_synthetic_stage6_candidate_pool


BENCHMARK = "stage6_latent_patch_selector"


def _tiny_pool():
    examples = build_synthetic_stage6_candidate_pool(n_tasks=24, seed=3)
    return {
        "train": [example for example in examples if example.split == "train"][:10],
        "dev": [example for example in examples if example.split == "dev"][:4],
        "test": [example for example in examples if example.split == "test"][:4],
    }


def _torch_stage6_imports():
    torch = pytest.importorskip("torch")
    from src.coordinators.latent_coordination import LatentCoordinatorConfig
    from src.experiments.real_shared_weight_latent_coordination import (
        MessageChannelConfig,
        RealSharedWeightTrainingConfig,
        SharedTransformerAgentConfig,
    )
    from src.experiments.run_stage6_latent_patch_selector import (
        evaluate_scores,
        fit_stage6_latent_selector,
        multi_positive_selector_loss,
        predict_stage6_latent_logits,
    )

    return {
        "torch": torch,
        "LatentCoordinatorConfig": LatentCoordinatorConfig,
        "MessageChannelConfig": MessageChannelConfig,
        "RealSharedWeightTrainingConfig": RealSharedWeightTrainingConfig,
        "SharedTransformerAgentConfig": SharedTransformerAgentConfig,
        "evaluate_scores": evaluate_scores,
        "fit_stage6_latent_selector": fit_stage6_latent_selector,
        "multi_positive_selector_loss": multi_positive_selector_loss,
        "predict_stage6_latent_logits": predict_stage6_latent_logits,
    }


def _agent_config(imports):
    SharedTransformerAgentConfig = imports["SharedTransformerAgentConfig"]
    return SharedTransformerAgentConfig(
        agent_mode="tiny_transformer",
        max_length=72,
        hidden_dim=8,
        tiny_vocab_size=256,
        tiny_layers=1,
        tiny_heads=1,
        tiny_ff_dim=16,
        adapter_hidden_dim=8,
    )


def _coordinator_config(imports):
    LatentCoordinatorConfig = imports["LatentCoordinatorConfig"]
    return LatentCoordinatorConfig(
        family="candidate_token_cross_attention",
        input_dim=8,
        model_dim=8,
        num_heads=1,
        num_layers=1,
        ff_dim=16,
        num_avenues=4,
    )


def _message_config(imports):
    MessageChannelConfig = imports["MessageChannelConfig"]
    return MessageChannelConfig(
        readout_source="pooled",
        use_message_head=False,
        coordinator_family="candidate_token_cross_attention",
        num_avenues=4,
        avenue_prompt_mode="stage6_patch_selection",
    )


def _training_config(imports):
    RealSharedWeightTrainingConfig = imports["RealSharedWeightTrainingConfig"]
    return RealSharedWeightTrainingConfig(epochs=1, batch_size=4, lr=0.001, patience=1, gradient_clip_norm=1.0)


def test_stage6_synthetic_pool_has_strict_k8_multiview_shape_and_no_output_leakage() -> None:
    splits = _tiny_pool()
    examples = splits["train"] + splits["dev"] + splits["test"]
    validation = validate_patch_selection_examples(examples)
    assert validation["passes"] is True
    assert all(len(example.candidates) == STAGE6_NUM_CANDIDATES for example in examples)
    assert all(len(example.labels_pass_fail) == STAGE6_NUM_CANDIDATES for example in examples)
    assert {candidate.candidate_source_agent for example in examples for candidate in example.candidates} == {
        "generator_0",
        "generator_1",
        "generator_2",
        "generator_3",
    }
    mv = patch_selection_to_multiview_many(examples[:2])
    assert len(mv[0].views) == 4
    assert len(mv[0].candidates) == 8
    assert stage6_output_leakage_audit(BENCHMARK, 0, examples)["passes"] is True
    assert stage6_split_leakage_audit(BENCHMARK, 0, splits)["passes"] is True


def test_stage6_candidate_order_control_remaps_pass_labels_by_candidate_identity() -> None:
    example = _tiny_pool()["test"][0]
    controlled = apply_stage6_control([example], "candidate_order_shuffled_with_label_remap", seed=11)[0]
    original_positive_ids = {
        candidate.candidate_id for candidate, label in zip(example.candidates, example.labels_pass_fail) if label
    }
    controlled_positive_ids = {
        candidate.candidate_id for candidate, label in zip(controlled.candidates, controlled.labels_pass_fail) if label
    }
    assert controlled_positive_ids == original_positive_ids
    assert [candidate.candidate_id for candidate in controlled.candidates] != [candidate.candidate_id for candidate in example.candidates]


def test_multi_positive_loss_is_negative_log_sum_probability_on_positive_candidates() -> None:
    imports = _torch_stage6_imports()
    torch = imports["torch"]
    multi_positive_selector_loss = imports["multi_positive_selector_loss"]
    logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0]])
    labels = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    loss = multi_positive_selector_loss(logits, labels)
    probs = torch.softmax(logits, dim=1)
    expected = -torch.log(torch.tensor([probs[0, [0, 2]].sum(), probs[1, [1, 2]].sum()])).mean()
    assert torch.allclose(loss, expected)


def test_stage6_trainable_and_frozen_latent_selector_audits_and_metrics() -> None:
    imports = _torch_stage6_imports()
    fit_stage6_latent_selector = imports["fit_stage6_latent_selector"]
    predict_stage6_latent_logits = imports["predict_stage6_latent_logits"]
    evaluate_scores = imports["evaluate_scores"]
    splits = _tiny_pool()
    trainable = fit_stage6_latent_selector(
        train_examples=splits["train"],
        dev_examples=splits["dev"],
        agent_config=_agent_config(imports),
        coordinator_config=_coordinator_config(imports),
        training_config=_training_config(imports),
        seed=5,
        device="cpu",
        trainable_agent=True,
        method="unit_trainable_stage6",
        message_config=_message_config(imports),
    )
    frozen = fit_stage6_latent_selector(
        train_examples=splits["train"],
        dev_examples=splits["dev"],
        agent_config=_agent_config(imports),
        coordinator_config=_coordinator_config(imports),
        training_config=_training_config(imports),
        seed=6,
        device="cpu",
        trainable_agent=False,
        method="unit_frozen_stage6",
        message_config=_message_config(imports),
    )
    assert trainable.audit["shared_parameter_identity"] is True
    assert trainable.audit["agent_grad_norm_mean"] > 0.0
    assert trainable.audit["agent_parameter_delta"] > 0.0
    assert trainable.audit["coordinator_grad_norm_mean"] > 0.0
    assert trainable.audit["coordinator_parameter_delta"] > 0.0
    assert trainable.audit["activation_requires_grad_before_coordinator"] is True
    assert trainable.audit["no_detach_between_clone_activations_and_loss"] is True
    assert frozen.audit["agent_grad_norm_mean"] == 0.0
    assert frozen.audit["agent_parameter_delta"] == 0.0
    assert frozen.audit["coordinator_grad_norm_mean"] > 0.0
    scores = predict_stage6_latent_logits(trainable, splits["test"], condition="none", seed=7)
    assert scores.shape == (len(splits["test"]), STAGE6_NUM_CANDIDATES)
    metrics = evaluate_scores("unit_trainable_stage6", scores, splits["test"], split="test")
    assert 0.0 <= metrics["pass_at_1"] <= 1.0
    assert metrics["oracle_pass_at_8"] == float(np.mean(label_matrix(splits["test"]).sum(axis=1) > 0))
