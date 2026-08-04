from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from src.experiments.e2_attention_synthesis import (  # noqa: E402
    CleanQKVActivationCacheV2,
    E2Budget,
    E2_TASK_FAMILIES,
    TRealizedCandidateMemoryAttention,
    WDAAdapter,
    apply_e2_control,
    build_e2_examples,
    cuda_preflight_status,
    load_stage8b4_champion_config,
    run_e2_campaign,
    validate_e2_examples,
)


def test_e2_dataset_builds_all_families_and_valid_labels() -> None:
    examples = build_e2_examples(n_examples=40, n_blocks=8, split="dev", template_split="dev", seed=21)
    audit = validate_e2_examples(examples)

    assert audit["passes"], audit["failures"]
    assert {example.task_family for example in examples} == set(E2_TASK_FAMILIES)
    assert all(0 <= example.label < 8 for example in examples)
    assert all(len(example.candidates) == 8 for example in examples)


def test_gpu_check_function_reports_cuda_blocker_without_fallback() -> None:
    blocked = cuda_preflight_status(device="cuda", cuda_available=False)
    smoke = cuda_preflight_status(device="cpu", allow_cpu_smoke=True, cuda_available=False)

    assert not blocked["passes"]
    assert blocked["decision"] == "CUDA_REQUIRED_NOT_AVAILABLE"
    assert smoke["passes"]
    assert smoke["decision"] == "CPU_SMOKE_ALLOWED"


def test_champion_config_can_load() -> None:
    config = load_stage8b4_champion_config()

    assert config.name == "stage8b3_j_explicit_view_objective_assignment_03"
    assert config.model_kind == "evidence_feature_latent"
    assert config.roles == 9


def test_wda_module_produces_per_example_effective_weights() -> None:
    torch.manual_seed(0)
    adapter = WDAAdapter(in_features=3, out_features=2, context_dim=4, rank=3)
    context = torch.randn(2, 4)
    weights = adapter.effective_weight(context)

    assert weights.shape == (2, 2, 3)
    assert not torch.allclose(weights[0], weights[1])


def test_cleanqkv_current_self_attention_excludes_evidence_tokens() -> None:
    cache = CleanQKVActivationCacheV2(vector_dim=4, slots=2)
    current_tokens = cache.current_self_attention_inputs(
        query_tokens=("query_entity",),
        candidate_tokens=("candidate_value",),
        current_tokens=("current_fact",),
    )

    assert current_tokens == ("query_entity", "candidate_value", "current_fact")
    assert "old_evidence" not in current_tokens
    assert "history_token" not in current_tokens


def test_memory_gate_controls_can_be_toggled() -> None:
    cache = CleanQKVActivationCacheV2(vector_dim=4, slots=2)
    current = torch.ones(2, 4)
    memory = torch.zeros(2, 4)

    cache.set_gate_control("closed")
    assert torch.equal(cache.gate(current, memory), torch.zeros(2, 1))
    cache.set_gate_control("open")
    assert torch.equal(cache.gate(current, memory), torch.ones(2, 1))
    cache.set_gate_control(None)
    assert cache.gate(current, memory).shape == (2, 1)


def test_t_realized_module_changes_output_under_nonzero_t() -> None:
    torch.manual_seed(2)
    module = TRealizedCandidateMemoryAttention(dim=4)
    candidate = torch.randn(1, 4)
    memory = torch.randn(3, 4)

    normal, t_values = module(candidate, memory)
    zero, _ = module(candidate, memory, t_override=torch.zeros_like(t_values))

    assert normal.shape == zero.shape == (1, 4)
    assert not torch.allclose(normal, zero)


def test_controls_run_on_tiny_cpu_smoke_data() -> None:
    examples = build_e2_examples(n_examples=8, n_blocks=8, split="dev", template_split="dev", seed=30)
    randomized = apply_e2_control(examples, "randomized_labels", seed=31)
    wda_fixed = apply_e2_control(examples, "wda_fixed_coordinates", seed=32)

    assert all(0 <= example.label < 8 for example in randomized)
    assert all(example.metadata["wda_fixed_coordinates"] for example in wda_fixed)


def test_tiny_smoke_campaign_runs_without_cuda_training_claim() -> None:
    budget = E2Budget(
        round1_variants=10,
        round2_promoted=3,
        round3_parent_count=1,
        round3_mutations_per_parent=1,
        round4_finalists=1,
        train_examples=8,
        eval_examples=8,
        round1_n=(8,),
        round2_n=(8,),
        round3_n=(8,),
        round4_n=(8,),
        round1_seeds=(0,),
        round2_seeds=(0,),
        round3_seeds=(0,),
        round4_seeds=(0,),
        round1_epochs=1,
        promoted_epochs=1,
    )
    payload = run_e2_campaign(budget, device="cpu", max_rounds=1, allow_cpu_smoke=True)

    assert payload["no_stage8c_run"]
    assert payload["no_10x_attention_capacity_claim"]
    assert payload["preflight"]["decision"] == "CPU_SMOKE_ALLOWED"
