from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
E3_CODE = ROOT.parent / "e3" / "code"
E31_CODE = ROOT.parent / "e3_1" / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
for extra in (E31_CODE, E3_CODE):
    if str(extra) not in sys.path:
        sys.path.append(str(extra))

import src.experiments.e3_clean_qkv as e3mod  # noqa: E402
from src.experiments.e3_clean_qkv import CleanQKVAttentionLayer, WDAAdapter, e3_config_to_dict  # noqa: E402
from src.experiments.e4_collab_clean_qkv import (  # noqa: E402
    CollaborativeGuidanceLayer,
    E4TypedActivationCache,
    apply_e4_control,
    build_e4_examples,
    cuda_preflight_status,
    generate_e4_variants,
    validate_e4_examples,
)


def test_dataset_builds_and_labels_valid() -> None:
    examples = build_e4_examples(n_examples=52, n_blocks=64, split="dev", template_split="dev", seed=4)
    audit = validate_e4_examples(examples)

    assert audit["passes"], audit["failures"]
    assert all(0 <= example.label < 8 for example in examples)
    assert len(set(audit["e4_stress_cases"])) == 13


def test_cuda_check_stops_correctly_if_unavailable() -> None:
    blocked = cuda_preflight_status(device="cuda", cuda_available=False)

    assert not blocked["passes"]
    assert blocked["decision"] == "CUDA_REQUIRED_NOT_AVAILABLE"


def test_current_self_attention_receives_current_tokens_only() -> None:
    tokens = CleanQKVAttentionLayer.current_self_attention_inputs(("query",), ("candidate",), ("current",))

    assert tokens == ("query", "candidate", "current")
    assert "old_evidence" not in tokens


def test_old_history_tokens_not_used_as_current_kv_and_cache_separate() -> None:
    torch.manual_seed(0)
    layer = CleanQKVAttentionLayer(dim=8, mode="wda_kv")
    current = torch.randn(2, 3, 8)
    cache = torch.randn(2, 7, 8)
    _, diagnostics = layer(current, cache)

    assert diagnostics["k_eff"].shape[1] == current.shape[1]
    assert diagnostics["v_eff"].shape[1] == current.shape[1]
    assert diagnostics["k_eff"].shape[1] != cache.shape[1]
    assert diagnostics["t_values"].shape[-1] == cache.shape[1]


def test_current_to_cache_query_path_exists_and_changes_guidance() -> None:
    torch.manual_seed(1)
    layer = CollaborativeGuidanceLayer(dim=8, mode="wda_kv", refinement_steps=1)
    current = torch.randn(1, 4, 8)
    cache = torch.randn(1, 5, 8)
    _, normal = layer(current, cache)
    _, disabled = layer(current, cache, disable_current_to_cache=True)

    assert "guidance" in normal
    assert not torch.allclose(normal["guidance"], disabled["guidance"])


def test_cache_to_current_guidance_changes_current_attention() -> None:
    torch.manual_seed(2)
    layer = CollaborativeGuidanceLayer(dim=8, mode="wda_kv", refinement_steps=1)
    current = torch.randn(1, 4, 8)
    cache = torch.randn(1, 5, 8)
    _, normal = layer(current, cache)
    _, disabled = layer(current, cache, disable_cache_to_current=True)

    assert not torch.allclose(normal["effective_scores"], disabled["effective_scores"])


def test_disabling_current_to_cache_and_cache_to_current_changes_outputs() -> None:
    torch.manual_seed(3)
    layer = CollaborativeGuidanceLayer(dim=8, mode="wda_kv", refinement_steps=1)
    current = torch.randn(1, 4, 8)
    cache = torch.randn(1, 5, 8)
    normal, _ = layer(current, cache)
    no_query, _ = layer(current, cache, disable_current_to_cache=True)
    no_guidance, _ = layer(current, cache, disable_cache_to_current=True)

    assert not torch.allclose(normal, no_query)
    assert not torch.allclose(normal, no_guidance)


def test_reciprocal_refinement_changes_outputs() -> None:
    torch.manual_seed(4)
    layer = CollaborativeGuidanceLayer(dim=8, mode="wda_kv", refinement_steps=2)
    current = torch.randn(1, 4, 8)
    cache = torch.randn(1, 5, 8)
    refined, _ = layer(current, cache)
    single, _ = layer(current, cache, disable_reciprocal=True)

    assert not torch.allclose(refined, single)


def test_support_contradiction_stale_current_streams_are_separable() -> None:
    torch.manual_seed(5)
    cache = E4TypedActivationCache(dim=4)
    streams = cache(torch.randn(2, 4), torch.randn(2, 4), torch.randn(2, 4), torch.randn(2, 4))

    assert set(streams) == {"support", "contradiction", "stale", "current_override"}
    assert not torch.allclose(streams["support"], streams["contradiction"])
    assert not torch.allclose(streams["stale"], streams["current_override"])


def test_wda_coordinates_produce_per_example_effective_weights() -> None:
    torch.manual_seed(6)
    adapter = WDAAdapter(in_features=3, out_features=2, context_dim=4)
    weights = adapter.effective_weight(torch.randn(2, 4))

    assert weights.shape == (2, 2, 3)
    assert not torch.allclose(weights[0], weights[1])


def test_wda_kv_changes_kv_but_not_q() -> None:
    torch.manual_seed(7)
    layer = CleanQKVAttentionLayer(dim=8, mode="wda_kv")
    current = torch.randn(1, 4, 8)
    cache = torch.randn(1, 3, 8)
    _, diagnostics = layer(current, cache)

    assert torch.allclose(diagnostics["q"], diagnostics["q_eff"])
    assert not torch.allclose(diagnostics["k"], diagnostics["k_eff"])
    assert not torch.allclose(diagnostics["v"], diagnostics["v_eff"])


def test_wda_bias_kv_changes_scores_and_kv() -> None:
    torch.manual_seed(8)
    bias_layer = CleanQKVAttentionLayer(dim=8, mode="bias")
    kv_layer = CleanQKVAttentionLayer(dim=8, mode="wda_kv")
    current = torch.randn(1, 4, 8)
    cache = torch.randn(1, 3, 8)
    _, bias_diag = bias_layer(current, cache)
    _, kv_diag = kv_layer(current, cache)

    assert not torch.allclose(bias_diag["base_scores"], bias_diag["effective_scores"])
    assert not torch.allclose(kv_diag["k"], kv_diag["k_eff"])
    assert not torch.allclose(kv_diag["v"], kv_diag["v_eff"])


def test_support_contradiction_swap_control_changes_predictions_when_relevant() -> None:
    examples = build_e4_examples(
        n_examples=20,
        n_blocks=64,
        split="dev",
        template_split="dev",
        seed=9,
        task_families=("support_vs_contradiction_memory",),
    )
    swapped = apply_e4_control(examples, "support_contradiction_swap", seed=10)
    config = [variant for variant in generate_e4_variants(64) if variant.support_contradiction][0]
    before = [int(torch.argmax(e3mod.e3_feature_tensor(example, config, 0)[:, 33]).item()) for example in examples]
    after = [int(torch.argmax(e3mod.e3_feature_tensor(example, config, 0)[:, 33]).item()) for example in swapped]

    assert before != after


def test_gate_forced_open_closed_works() -> None:
    layer = CleanQKVAttentionLayer(dim=8, mode="wda_kv")
    current = torch.ones(2, 8)
    cache = torch.zeros(2, 8)

    layer.set_gate_control("open")
    assert torch.equal(layer.gate(current, cache), torch.ones(2, 1))
    layer.set_gate_control("closed")
    assert torch.equal(layer.gate(current, cache), torch.zeros(2, 1))


def test_cache_shuffle_controls_run() -> None:
    examples = build_e4_examples(n_examples=20, n_blocks=64, split="dev", template_split="dev", seed=12)
    shuffled = apply_e4_control(examples, "cache_shuffle_across_examples", seed=13)

    assert len(shuffled) == len(examples)
    assert any(a.evidence_blocks != b.evidence_blocks for a, b in zip(examples, shuffled))


def test_frozen_comparator_uses_exact_same_architecture() -> None:
    config = generate_e4_variants(1)[0]
    frozen = replace(config, frozen=True)
    base = e3_config_to_dict(config)
    comp = e3_config_to_dict(frozen)
    base.pop("config_id")
    comp.pop("config_id")
    base["frozen"] = True

    assert base == comp
