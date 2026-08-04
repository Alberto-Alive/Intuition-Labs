from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from src.experiments.e3_clean_qkv import (  # noqa: E402
    CleanQKVAttentionLayer,
    E3_TASK_FAMILIES,
    TRealizedCleanQKV,
    WDAAdapter,
    apply_e3_control,
    build_e3_examples,
    cuda_preflight_status,
    validate_e3_examples,
)


def test_e3_dataset_builds_all_families_and_valid_labels() -> None:
    examples = build_e3_examples(n_examples=40, n_blocks=8, split="dev", template_split="dev", seed=21)
    audit = validate_e3_examples(examples)

    assert audit["passes"], audit["failures"]
    assert {example.task_family for example in examples} == set(E3_TASK_FAMILIES)
    assert all(0 <= example.label < 8 for example in examples)
    assert all(len(example.candidates) == 8 for example in examples)


def test_cuda_check_function_reports_blocker_without_cpu_fallback() -> None:
    blocked = cuda_preflight_status(device="cuda", cuda_available=False)
    smoke = cuda_preflight_status(device="cpu", allow_cpu_smoke=True, cuda_available=False)

    assert not blocked["passes"]
    assert blocked["decision"] == "CUDA_REQUIRED_NOT_AVAILABLE"
    assert smoke["passes"]
    assert smoke["decision"] == "CPU_SMOKE_ALLOWED"


def test_current_self_attention_receives_current_tokens_only() -> None:
    tokens = CleanQKVAttentionLayer.current_self_attention_inputs(
        query_tokens=("query_entity",),
        candidate_tokens=("candidate_value",),
        current_tokens=("current_fact",),
    )

    assert tokens == ("query_entity", "candidate_value", "current_fact")
    assert "old_evidence" not in tokens
    assert "history_token" not in tokens


def test_activation_cache_is_separate_from_token_kv() -> None:
    torch.manual_seed(0)
    layer = CleanQKVAttentionLayer(dim=8, mode="bias")
    current = torch.randn(2, 3, 8)
    cache = torch.randn(2, 5, 8)
    output, diagnostics = layer(current, cache)

    assert output.shape == current.shape
    assert diagnostics["k"].shape[1] == current.shape[1]
    assert diagnostics["v"].shape[1] == current.shape[1]
    assert diagnostics["k"].shape[1] != cache.shape[1]


def test_bias_only_changes_attention_scores() -> None:
    torch.manual_seed(1)
    layer = CleanQKVAttentionLayer(dim=8, mode="bias")
    current = torch.randn(1, 4, 8)
    cache = torch.randn(1, 3, 8)
    _, diagnostics = layer(current, cache)

    assert not torch.allclose(diagnostics["base_scores"], diagnostics["effective_scores"])
    assert torch.allclose(diagnostics["k"], diagnostics["k_eff"])
    assert torch.allclose(diagnostics["v"], diagnostics["v_eff"])


def test_k_only_changes_k_but_not_v() -> None:
    torch.manual_seed(2)
    layer = CleanQKVAttentionLayer(dim=8, mode="k")
    current = torch.randn(1, 4, 8)
    cache = torch.randn(1, 3, 8)
    _, diagnostics = layer(current, cache)

    assert not torch.allclose(diagnostics["k"], diagnostics["k_eff"])
    assert torch.allclose(diagnostics["v"], diagnostics["v_eff"])


def test_v_only_changes_v_but_not_k() -> None:
    torch.manual_seed(3)
    layer = CleanQKVAttentionLayer(dim=8, mode="v")
    current = torch.randn(1, 4, 8)
    cache = torch.randn(1, 3, 8)
    _, diagnostics = layer(current, cache)

    assert not torch.allclose(diagnostics["v"], diagnostics["v_eff"])
    assert torch.allclose(diagnostics["k"], diagnostics["k_eff"])


def test_full_qkv_changes_q_k_v() -> None:
    torch.manual_seed(4)
    layer = CleanQKVAttentionLayer(dim=8, mode="full_qkv")
    current = torch.randn(1, 4, 8)
    cache = torch.randn(1, 3, 8)
    _, diagnostics = layer(current, cache)

    assert not torch.allclose(diagnostics["q"], diagnostics["q_eff"])
    assert not torch.allclose(diagnostics["k"], diagnostics["k_eff"])
    assert not torch.allclose(diagnostics["v"], diagnostics["v_eff"])


def test_wda_coordinates_produce_per_example_effective_weights() -> None:
    torch.manual_seed(5)
    adapter = WDAAdapter(in_features=3, out_features=2, context_dim=4, rank=3)
    context = torch.randn(2, 4)
    weights = adapter.effective_weight(context)

    assert weights.shape == (2, 2, 3)
    assert not torch.allclose(weights[0], weights[1])


def test_t_realized_variants_change_outputs_when_t_nonzero() -> None:
    torch.manual_seed(6)
    module = TRealizedCleanQKV(dim=4)
    current = torch.randn(1, 2, 4)
    cache = torch.randn(1, 3, 4)
    k_delta, v_delta, t_values = module(current, cache)
    zero_k, zero_v, _ = module(current, cache, t_override=torch.zeros_like(t_values))

    assert not torch.allclose(k_delta, zero_k)
    assert not torch.allclose(v_delta, zero_v)


def test_gate_forced_open_closed_works() -> None:
    torch.manual_seed(7)
    layer = CleanQKVAttentionLayer(dim=8, mode="v")
    current = torch.ones(2, 8)
    cache = torch.zeros(2, 8)

    layer.set_gate_control("closed")
    assert torch.equal(layer.gate(current, cache), torch.zeros(2, 1))
    layer.set_gate_control("open")
    assert torch.equal(layer.gate(current, cache), torch.ones(2, 1))
    layer.set_gate_control(None)
    assert layer.gate(current, cache).shape == (2, 1)


def test_cache_shuffle_controls_run() -> None:
    examples = build_e3_examples(n_examples=16, n_blocks=8, split="dev", template_split="dev", seed=30)
    shuffled = apply_e3_control(examples, "cache_shuffle_across_examples", seed=31)
    randomized = apply_e3_control(examples, "randomized_labels", seed=32)

    assert len(shuffled) == len(examples)
    assert all(0 <= example.label < 8 for example in randomized)
    assert any(example.evidence_blocks != original.evidence_blocks for example, original in zip(shuffled, examples))


def test_cpu_smoke_is_explicitly_marked_without_running_campaign_outputs() -> None:
    status = cuda_preflight_status(device="cpu", allow_cpu_smoke=True, cuda_available=False)

    assert status["passes"]
    assert status["decision"] == "CPU_SMOKE_ALLOWED"
