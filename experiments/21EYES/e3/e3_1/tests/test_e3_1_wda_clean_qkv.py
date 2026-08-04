from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from src.experiments.e3_1_wda_clean_qkv import (  # noqa: E402
    SupportContradictionActivationCache,
    apply_e3_control,
    build_e3_1_examples,
    cuda_preflight_status,
    e3_config_to_dict,
    generate_e3_1_variants,
    validate_e3_1_examples,
)
from src.experiments.e3_clean_qkv import CleanQKVAttentionLayer, WDAAdapter  # noqa: E402


def test_dataset_builds_and_labels_valid() -> None:
    examples = build_e3_1_examples(n_examples=40, n_blocks=32, split="dev", template_split="dev", seed=11)
    audit = validate_e3_1_examples(examples)

    assert audit["passes"], audit["failures"]
    assert all(0 <= example.label < 8 for example in examples)
    assert len(set(audit["e3_1_stress_cases"])) == 10


def test_cuda_check_stops_correctly_if_unavailable() -> None:
    blocked = cuda_preflight_status(device="cuda", cuda_available=False)
    smoke = cuda_preflight_status(device="cpu", allow_cpu_smoke=True, cuda_available=False)

    assert not blocked["passes"]
    assert blocked["decision"] == "CUDA_REQUIRED_NOT_AVAILABLE"
    assert smoke["passes"]


def test_current_self_attention_receives_current_tokens_only() -> None:
    tokens = CleanQKVAttentionLayer.current_self_attention_inputs(
        query_tokens=("query",),
        candidate_tokens=("candidate",),
        current_tokens=("current_exception",),
    )

    assert tokens == ("query", "candidate", "current_exception")
    assert "old_evidence" not in tokens
    assert "history" not in tokens


def test_old_evidence_history_tokens_not_used_as_current_kv() -> None:
    torch.manual_seed(0)
    layer = CleanQKVAttentionLayer(dim=8, mode="wda_kv")
    current = torch.randn(2, 3, 8)
    cache = torch.randn(2, 7, 8)
    _, diagnostics = layer(current, cache)

    assert diagnostics["k_eff"].shape[1] == current.shape[1]
    assert diagnostics["v_eff"].shape[1] == current.shape[1]
    assert diagnostics["k_eff"].shape[1] != cache.shape[1]


def test_activation_cache_is_separate_from_token_kv() -> None:
    torch.manual_seed(1)
    layer = CleanQKVAttentionLayer(dim=8, mode="wda_v")
    current = torch.randn(1, 4, 8)
    cache = torch.randn(1, 6, 8)
    _, diagnostics = layer(current, cache)

    assert diagnostics["k"].shape[1] == 4
    assert diagnostics["v"].shape[1] == 4
    assert diagnostics["t_values"].shape[-1] == 6


def test_wda_coordinates_produce_per_example_effective_weights() -> None:
    torch.manual_seed(2)
    adapter = WDAAdapter(in_features=3, out_features=2, context_dim=4)
    weights = adapter.effective_weight(torch.randn(2, 4))

    assert weights.shape == (2, 2, 3)
    assert not torch.allclose(weights[0], weights[1])


def test_wda_v_changes_v_but_not_qk() -> None:
    torch.manual_seed(3)
    layer = CleanQKVAttentionLayer(dim=8, mode="wda_v")
    current = torch.randn(1, 4, 8)
    cache = torch.randn(1, 3, 8)
    _, diagnostics = layer(current, cache)

    assert torch.allclose(diagnostics["q"], diagnostics["q_eff"])
    assert torch.allclose(diagnostics["k"], diagnostics["k_eff"])
    assert not torch.allclose(diagnostics["v"], diagnostics["v_eff"])


def test_wda_kv_changes_kv_but_not_q() -> None:
    torch.manual_seed(4)
    layer = CleanQKVAttentionLayer(dim=8, mode="wda_kv")
    current = torch.randn(1, 4, 8)
    cache = torch.randn(1, 3, 8)
    _, diagnostics = layer(current, cache)

    assert torch.allclose(diagnostics["q"], diagnostics["q_eff"])
    assert not torch.allclose(diagnostics["k"], diagnostics["k_eff"])
    assert not torch.allclose(diagnostics["v"], diagnostics["v_eff"])


def test_wda_qkv_changes_qkv() -> None:
    torch.manual_seed(5)
    layer = CleanQKVAttentionLayer(dim=8, mode="wda_qkv")
    current = torch.randn(1, 4, 8)
    cache = torch.randn(1, 3, 8)
    _, diagnostics = layer(current, cache)

    assert not torch.allclose(diagnostics["q"], diagnostics["q_eff"])
    assert not torch.allclose(diagnostics["k"], diagnostics["k_eff"])
    assert not torch.allclose(diagnostics["v"], diagnostics["v_eff"])


def test_wda_bias_changes_attention_scores() -> None:
    torch.manual_seed(6)
    layer = CleanQKVAttentionLayer(dim=8, mode="bias")
    current = torch.randn(1, 4, 8)
    cache = torch.randn(1, 3, 8)
    _, diagnostics = layer(current, cache)

    assert not torch.allclose(diagnostics["base_scores"], diagnostics["effective_scores"])


def test_support_contradiction_cache_streams_are_separable() -> None:
    torch.manual_seed(7)
    cache = SupportContradictionActivationCache(dim=4)
    support = torch.randn(2, 4)
    contradiction = torch.randn(2, 4)
    streams = cache(support, contradiction)

    assert set(streams) == {"support", "contradiction", "stale"}
    assert not torch.allclose(streams["support"], streams["contradiction"])


def test_gate_forced_open_closed_works() -> None:
    layer = CleanQKVAttentionLayer(dim=8, mode="wda_v")
    current = torch.ones(2, 8)
    cache = torch.zeros(2, 8)

    layer.set_gate_control("open")
    assert torch.equal(layer.gate(current, cache), torch.ones(2, 1))
    layer.set_gate_control("closed")
    assert torch.equal(layer.gate(current, cache), torch.zeros(2, 1))


def test_cache_shuffle_controls_run() -> None:
    examples = build_e3_1_examples(n_examples=20, n_blocks=32, split="dev", template_split="dev", seed=21)
    shuffled = apply_e3_control(examples, "cache_shuffle_across_examples", seed=22)

    assert len(shuffled) == len(examples)
    assert any(a.evidence_blocks != b.evidence_blocks for a, b in zip(examples, shuffled))


def test_frozen_comparator_uses_exact_same_architecture_except_frozen_flag() -> None:
    config = generate_e3_1_variants(1)[0]
    frozen = replace(config, frozen=True)
    base = e3_config_to_dict(config)
    comp = e3_config_to_dict(frozen)
    base.pop("config_id")
    comp.pop("config_id")
    base["frozen"] = True

    assert base == comp
