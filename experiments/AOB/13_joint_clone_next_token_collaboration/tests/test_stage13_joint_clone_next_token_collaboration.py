from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "src"
    / "experiments"
    / "run_stage13_joint_clone_next_token_collaboration.py"
)
SPEC = importlib.util.spec_from_file_location("stage13_joint_clone_next_token_collaboration", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_other_clone_concat_keeps_other_clones_distinct() -> None:
    hidden = torch.tensor(
        [
            [
                [[1.0, 2.0]],
                [[3.0, 4.0]],
                [[5.0, 6.0]],
            ]
        ],
        dtype=torch.float32,
    )
    other = MODULE._other_clone_concat(hidden)
    expected = torch.tensor(
        [
            [
                [[3.0, 4.0, 5.0, 6.0]],
                [[1.0, 2.0, 5.0, 6.0]],
                [[1.0, 2.0, 3.0, 4.0]],
            ]
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(other, expected)


def test_detached_other_concat_blocks_gradient_flow_from_other_clones() -> None:
    hidden = torch.arange(1, 9, dtype=torch.float32).reshape(1, 2, 1, 4).requires_grad_(True)
    conditioned = MODULE._concat_with_detached_other_clones(hidden)
    detached_only = conditioned[..., hidden.shape[-1] :].sum()
    detached_only.backward()
    assert hidden.grad is not None
    assert torch.allclose(hidden.grad, torch.zeros_like(hidden.grad))


def test_clone_attention_bias_changes_attention_geometry_even_with_identical_inputs() -> None:
    attention = MODULE.CollaborativeCloneSelfAttention(
        hidden_size=4,
        num_heads=1,
        num_clones=2,
        max_length=2,
        dropout=0.0,
        bias_init_std=0.02,
    )
    with torch.no_grad():
        attention.q_proj.weight.zero_()
        attention.q_proj.bias.zero_()
        attention.k_proj.weight.zero_()
        attention.k_proj.bias.zero_()
        attention.v_proj.weight.zero_()
        attention.v_proj.bias.zero_()
        attention.out_proj.weight.zero_()
        attention.out_proj.bias.zero_()
        attention.clone_attention_bias.zero_()
        attention.clone_attention_bias[0, 0, 1, 0] = 1.0
        attention.clone_attention_bias[1, 0, 1, 0] = -1.0
    hidden = torch.ones(1, 2, 2, 4, dtype=torch.float32)
    mask = torch.ones(1, 2, dtype=torch.bool)
    _out, probs = attention(hidden, mask)
    assert probs.shape == (1, 2, 1, 2, 2)
    assert probs[0, 0, 0, 1, 0] > probs[0, 1, 0, 1, 0]


def test_weighted_clone_sum_uses_supplied_weights() -> None:
    clone_states = torch.tensor(
        [
            [
                [[1.0, 1.0]],
                [[3.0, 3.0]],
                [[5.0, 5.0]],
                [[7.0, 7.0]],
            ]
        ],
        dtype=torch.float32,
    )
    weights = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
    joint = MODULE._weighted_clone_sum(clone_states, weights)
    assert torch.allclose(joint, torch.tensor([[[5.0, 5.0]]], dtype=torch.float32))


def test_attention_kl_and_bias_shift_report_positive_difference() -> None:
    probs_a = torch.tensor(
        [
            [
                [[[0.7, 0.3], [0.4, 0.6]]],
                [[[0.3, 0.7], [0.6, 0.4]]],
            ]
        ],
        dtype=torch.float32,
    )
    probs_b = torch.tensor(
        [
            [
                [[[0.5, 0.5], [0.5, 0.5]]],
                [[[0.5, 0.5], [0.5, 0.5]]],
            ]
        ],
        dtype=torch.float32,
    )
    mask = torch.ones(1, 2, dtype=torch.bool)
    pairwise = MODULE._symmetrized_attention_kl([probs_a], mask)
    shift = MODULE._mean_attention_distribution_shift([probs_a], [probs_b], mask)
    assert pairwise["overall_mean"] > 0.0
    assert shift["overall_mean"] > 0.0


def test_train_on_text_splits_smoke_runs_and_reports_requested_metrics() -> None:
    text_splits = {
        "train": [
            "import foo from bar if x == y",
            "class widget uses coordinator tokens",
            "patch local import boundary explicit policy",
            "function call returns value from module",
        ],
        "dev": [
            "widget import policy explicit",
            "module returns class value",
        ],
        "test": [
            "coordinator import value",
            "policy boundary function call",
        ],
    }
    config = MODULE.Stage13Config(
        corpus=MODULE.CorpusConfig(max_length=16, vocab_size=64),
        training=MODULE.TrainingConfig(epochs=1, batch_size=2, seed=7, device="cpu"),
    )
    results = MODULE.train_on_text_splits(text_splits, config)
    assert len(results["history"]) == 1
    assert len(results["final_test"]["clone_perplexity"]) == 4
    assert len(results["final_test"]["attention_entropy_per_clone"]) == 4
    assert len(results["final_test"]["joint_weights"]) == 4
    assert len(results["attention_bias_norms"]) == 3
    assert "pairwise_attention_kl" in results["final_test"]
    assert "bias_used_diagnostic" in results["final_test"]
    assert results["dataset_sizes"]["train_sequences"] == 4
    assert math.isfinite(float(results["final_test"]["joint_perplexity"]))
