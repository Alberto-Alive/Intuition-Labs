from pathlib import Path
import re

import numpy as np

from src.coordinators.latent_coordination import LatentCoordinatorConfig
from src.datasets.multiview_code_patch_selection import (
    CONTROL_NAMES,
    MultiViewCodePatchDatasetConfig,
    apply_example_control,
    attempt_batch_from_examples,
    build_multiview_code_patch_splits,
    split_leakage_audit,
)
from src.evaluation.metrics import evaluate_predictions
from src.evaluation.report import render_report
from src.experiments.real_shared_weight_latent_coordination import (
    BENCHMARK,
    MessageChannelConfig,
    RealSharedWeightTrainingConfig,
    SharedTransformerAgentConfig,
    fit_latent_system,
    predict_latent_system,
    shared_parameter_identity_check,
)


def _tiny_splits():
    config = MultiViewCodePatchDatasetConfig(
        n_train=16,
        n_dev=8,
        n_test=8,
        num_candidates=8,
        n_views=4,
        max_files=8,
        snippet_radius=1,
    )
    return build_multiview_code_patch_splits(config, seed=123, repo_root=Path(".")), config


def _agent_config() -> SharedTransformerAgentConfig:
    return SharedTransformerAgentConfig(
        agent_mode="tiny_transformer",
        max_length=96,
        hidden_dim=16,
        tiny_vocab_size=512,
        tiny_layers=1,
        tiny_heads=1,
        tiny_ff_dim=32,
        adapter_hidden_dim=16,
    )


def _coordinator_config() -> LatentCoordinatorConfig:
    return LatentCoordinatorConfig(
        family="cross_attention",
        input_dim=16,
        model_dim=16,
        num_heads=1,
        num_layers=1,
        ff_dim=32,
    )


def _training_config() -> RealSharedWeightTrainingConfig:
    return RealSharedWeightTrainingConfig(epochs=1, batch_size=8, lr=0.003, patience=1)


def _message_config(train_aux: bool = True) -> MessageChannelConfig:
    return MessageChannelConfig(
        use_msg_token=True,
        msg_position="prepend",
        use_message_head=True,
        message_dim=16,
        coordinator_family="candidate_query_cross_attention",
        use_private_cue_aux=train_aux,
        aux_loss_weight=0.05,
        aux_warmup_epochs=1,
        aux_decay_epochs=1,
    )


def _active_message_config(train_aux: bool = True) -> MessageChannelConfig:
    return MessageChannelConfig(
        use_msg_token=False,
        readout_source="active_message",
        use_message_head=True,
        message_dim=16,
        coordinator_family="candidate_query_cross_attention",
        use_private_cue_aux=train_aux,
        aux_loss_weight=0.05,
        aux_warmup_epochs=1,
        aux_decay_epochs=1,
        active_message_layers=(-1,),
        active_message_heads=1,
        active_message_ff_dim=32,
    )


def test_multiview_code_patch_splits_have_no_train_test_overlap_and_controls_are_callable() -> None:
    splits, _config = _tiny_splits()
    audit = split_leakage_audit(BENCHMARK, seed=123, splits=splits)
    assert audit["train_dev_id_overlap"] == 0
    assert audit["train_test_id_overlap"] == 0
    assert audit["dev_test_id_overlap"] == 0
    assert audit["train_test_candidate_hash_overlap"] == 0
    assert audit["train_test_target_file_overlap"] == 0
    assert audit["candidate_patch_hash_leakage_audit_passes"] is True
    for condition in CONTROL_NAMES:
        controlled = apply_example_control(splits["dev"], condition=condition, seed=7)
        assert len(controlled) == len(splits["dev"])
        assert attempt_batch_from_examples(controlled, "dev").labels.shape == (len(splits["dev"]),)


def test_real_program_analysis_variant_has_no_constructed_cue_tokens() -> None:
    config = MultiViewCodePatchDatasetConfig(
        n_train=8,
        n_dev=4,
        n_test=4,
        num_candidates=8,
        n_views=4,
        max_files=4,
        snippet_radius=1,
        include_private_signal_tokens=False,
    )
    splits = build_multiview_code_patch_splits(config, seed=321, repo_root=Path("."))
    for example in splits["train"] + splits["dev"] + splits["test"]:
        assert example.metadata["dataset_source"] == "real_program_analysis_import_restoration"
        assert example.metadata["private_cue_style"] == "real_program_analysis_artifacts_no_constructed_cue_tokens"
        assert example.metadata["real_correct_patch"] is True
        assert example.metadata["single_view_candidate_ambiguity_min"] >= 4
        assert set(example.metadata["program_analysis_artifacts"]) == {
            "ast_import_deletion_name_usage",
            "module_export_resolution",
            "provider_module_name_parity_analysis",
            "traceback_import_block_location",
        }
        for view in example.views:
            text = view.text.lower()
            assert "repair_cue" not in text
            assert "repair cue" not in text
            assert re.search(r"\b(?:high|low)\b", text) is None


def test_hardened_import_restore_variant_redacts_private_answer_strings() -> None:
    config = MultiViewCodePatchDatasetConfig(
        n_train=8,
        n_dev=4,
        n_test=4,
        num_candidates=8,
        n_views=4,
        max_files=8,
        snippet_radius=1,
        dataset_source="real_import_restore_hardened",
        generator_version="import_restore_redacted_v1_unit",
        candidate_representation="hardened_redacted_v1",
    )
    splits = build_multiview_code_patch_splits(config, seed=91, repo_root=Path("."))
    for example in splits["train"] + splits["dev"] + splits["test"]:
        assert example.metadata["dataset_source"] == "real_import_restore_hardened"
        assert example.metadata["candidate_representation"] == "hardened_redacted_v1"
        private_text = "\n".join(view.text for view in example.views).lower()
        gold_values = example.metadata["candidate_patch_values"][example.label]
        assert str(gold_values[0]).lower() not in private_text
        assert str(gold_values[1]).lower() not in private_text
        assert str(gold_values[3]).lower() not in private_text
        assert str(example.metadata["gold_import_statement"]).lower() not in private_text
        for candidate, values in zip(example.candidates, example.metadata["candidate_patch_values"]):
            assert candidate.text.lower() not in private_text
            assert str(values[0]).lower() not in private_text
            assert str(values[1]).lower() not in private_text
            assert str(values[3]).lower() not in private_text
            assert "IMPORTED_SYMBOL_REDACTED".lower() in candidate.text.lower()


def test_trainable_shared_agent_receives_gradients_and_parameter_delta() -> None:
    splits, config = _tiny_splits()
    result = fit_latent_system(
        train_examples=splits["train"],
        dev_examples=splits["dev"],
        agent_config=_agent_config(),
        coordinator_config=_coordinator_config(),
        training_config=_training_config(),
        num_classes=config.num_candidates,
        seed=5,
        device="cpu",
        trainable_agent=True,
        method="trainable_shared_agent_latent_coordinator",
    )
    audit = result.audit
    assert audit["shared_parameter_identity"] is True
    assert audit["agent_grad_norm_mean"] > 0.0
    assert audit["coordinator_grad_norm_mean"] > 0.0
    assert audit["agent_parameter_delta"] > 0.0
    assert audit["coordinator_parameter_delta"] > 0.0
    assert audit["activation_requires_grad_before_coordinator"] is True
    assert audit["no_detach_between_clone_activations_and_loss"] is True
    assert audit["per_clone_gradient_contribution"] is True
    assert all(value > 0.0 for value in audit["per_clone_activation_grad_norms"])
    identity = shared_parameter_identity_check(result.agent, n_clones=4)
    assert identity["pass"] is True


def test_active_message_readout_candidate_query_receives_gradients_and_delta() -> None:
    splits, config = _tiny_splits()
    result = fit_latent_system(
        train_examples=splits["train"],
        dev_examples=splits["dev"],
        agent_config=_agent_config(),
        coordinator_config=_coordinator_config(),
        training_config=_training_config(),
        num_classes=config.num_candidates,
        seed=14,
        device="cpu",
        trainable_agent=True,
        method="trainable_shared_agent_active_msg_head_aux_candidate_query",
        message_config=_active_message_config(train_aux=True),
    )
    audit = result.audit
    assert audit["uses_msg_token"] is False
    assert audit["uses_active_message_readout"] is True
    assert audit["readout_source"] == "active_message"
    assert audit["uses_message_head"] is True
    assert audit["uses_private_cue_aux"] is True
    assert audit["message_coordinator_family"] == "candidate_query_cross_attention"
    assert audit["agent_grad_norm_mean"] > 0.0
    assert audit["agent_parameter_delta"] > 0.0
    assert audit["active_message_readout_grad_norm_mean"] > 0.0
    assert audit["active_message_readout_parameter_delta"] > 0.0
    assert audit["message_head_grad_norm_mean"] > 0.0
    assert audit["message_head_parameter_delta"] > 0.0
    assert audit["private_cue_head_grad_norm_mean"] > 0.0
    assert audit["coordinator_grad_norm_mean"] > 0.0
    predictions = predict_latent_system(result, splits["dev"], condition="hidden_states_shuffled_across_examples", seed=15)
    assert predictions.shape == (len(splits["dev"]),)


def test_msg_token_message_head_candidate_query_receives_gradients_and_delta() -> None:
    splits, config = _tiny_splits()
    result = fit_latent_system(
        train_examples=splits["train"],
        dev_examples=splits["dev"],
        agent_config=_agent_config(),
        coordinator_config=_coordinator_config(),
        training_config=_training_config(),
        num_classes=config.num_candidates,
        seed=15,
        device="cpu",
        trainable_agent=True,
        method="trainable_shared_agent_msg_head_aux_candidate_query",
        message_config=_message_config(train_aux=True),
    )
    audit = result.audit
    assert audit["uses_msg_token"] is True
    assert audit["uses_message_head"] is True
    assert audit["uses_private_cue_aux"] is True
    assert audit["message_coordinator_family"] == "candidate_query_cross_attention"
    assert audit["agent_grad_norm_mean"] > 0.0
    assert audit["agent_parameter_delta"] > 0.0
    assert audit["message_head_grad_norm_mean"] > 0.0
    assert audit["message_head_parameter_delta"] > 0.0
    assert audit["private_cue_head_grad_norm_mean"] > 0.0
    assert audit["coordinator_grad_norm_mean"] > 0.0
    predictions = predict_latent_system(result, splits["dev"], condition="candidate_order_shuffled", seed=15)
    assert predictions.shape == (len(splits["dev"]),)


def test_frozen_shared_agent_does_not_receive_gradients_or_delta() -> None:
    splits, config = _tiny_splits()
    result = fit_latent_system(
        train_examples=splits["train"],
        dev_examples=splits["dev"],
        agent_config=_agent_config(),
        coordinator_config=_coordinator_config(),
        training_config=_training_config(),
        num_classes=config.num_candidates,
        seed=6,
        device="cpu",
        trainable_agent=False,
        method="frozen_shared_agent_latent_coordinator",
    )
    audit = result.audit
    assert audit["shared_parameter_identity"] is True
    assert audit["agent_grad_norm_mean"] == 0.0
    assert audit["agent_parameter_delta"] == 0.0
    assert audit["coordinator_grad_norm_mean"] > 0.0
    assert audit["coordinator_parameter_delta"] > 0.0


def test_frozen_msg_aux_ablation_keeps_shared_agent_frozen() -> None:
    splits, config = _tiny_splits()
    result = fit_latent_system(
        train_examples=splits["train"],
        dev_examples=splits["dev"],
        agent_config=_agent_config(),
        coordinator_config=_coordinator_config(),
        training_config=_training_config(),
        num_classes=config.num_candidates,
        seed=16,
        device="cpu",
        trainable_agent=False,
        method="frozen_shared_agent_msg_head_aux_candidate_query",
        message_config=_message_config(train_aux=True),
    )
    audit = result.audit
    assert audit["agent_grad_norm_mean"] == 0.0
    assert audit["agent_parameter_delta"] == 0.0
    assert audit["message_head_grad_norm_mean"] > 0.0
    assert audit["message_head_parameter_delta"] > 0.0
    assert audit["coordinator_grad_norm_mean"] > 0.0
    assert audit["coordinator_parameter_delta"] > 0.0


def test_frozen_active_message_comparator_keeps_shared_agent_frozen() -> None:
    splits, config = _tiny_splits()
    result = fit_latent_system(
        train_examples=splits["train"],
        dev_examples=splits["dev"],
        agent_config=_agent_config(),
        coordinator_config=_coordinator_config(),
        training_config=_training_config(),
        num_classes=config.num_candidates,
        seed=17,
        device="cpu",
        trainable_agent=False,
        method="frozen_shared_agent_active_msg_head_aux_candidate_query",
        message_config=_active_message_config(train_aux=True),
    )
    audit = result.audit
    assert audit["uses_active_message_readout"] is True
    assert audit["agent_grad_norm_mean"] == 0.0
    assert audit["agent_parameter_delta"] == 0.0
    assert audit["active_message_readout_grad_norm_mean"] > 0.0
    assert audit["active_message_readout_parameter_delta"] > 0.0
    assert audit["message_head_grad_norm_mean"] > 0.0
    assert audit["message_head_parameter_delta"] > 0.0
    assert audit["coordinator_grad_norm_mean"] > 0.0
    assert audit["coordinator_parameter_delta"] > 0.0


def test_controls_produce_prediction_records_and_report_renders() -> None:
    splits, config = _tiny_splits()
    result = fit_latent_system(
        train_examples=splits["train"],
        dev_examples=splits["dev"],
        agent_config=_agent_config(),
        coordinator_config=_coordinator_config(),
        training_config=_training_config(),
        num_classes=config.num_candidates,
        seed=8,
        device="cpu",
        trainable_agent=True,
        method="trainable_shared_agent_latent_coordinator",
    )
    metrics = []
    for condition in ["none", "view_masked", "hidden_states_shuffled_across_examples", "candidate_order_shuffled"]:
        examples = apply_example_control(splits["test"], condition=condition, seed=8)
        predictions = predict_latent_system(result, examples, condition=condition, seed=8)
        assert predictions.shape == (len(examples),)
        batch = attempt_batch_from_examples(examples, "test", hidden_dim=16)
        metrics.append(
            evaluate_predictions(
                method="trainable_shared_agent_latent_coordinator",
                condition=condition,
                predictions=predictions,
                batch=batch,
                seed=8,
                param_count=result.param_count,
                benchmark=BENCHMARK,
            )
        )
    report = render_report(
        {
            "metadata": {
                "agent_config": _agent_config().__dict__,
                "training_config": _training_config().__dict__,
                "dataset_summary_by_seed": {
                    "8": {
                        "dataset_source": "unit_test",
                        "split_sizes": {"train": 16, "dev": 8, "test": 8},
                        "num_candidates": 8,
                        "source_files_used": 1,
                        "publishable_proof_dataset": False,
                    }
                },
            },
            "metrics": metrics,
            "diagnostics": [result.audit],
            "audit": [],
            "validation": {"real_shared_weight_latent_coordination": {}},
        }
    )
    assert "Real Shared-Weight Latent Coordination" in report
    assert np.isfinite(metrics[0]["accuracy"])
