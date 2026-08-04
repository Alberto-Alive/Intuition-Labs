from __future__ import annotations

import random
from dataclasses import asdict, replace
from typing import Dict, Iterable, List, Sequence

from src.models.latent_attention_variants import Stage8ArchitectureConfig


ROLE_OPTIONS = (2, 4, 8, 16)
AVENUE_OPTIONS = (1, 2, 4, 8)
CHUNKING_OPTIONS = (
    "contiguous",
    "random",
    "semantic_key",
    "hashed",
    "learned_router",
    "overlapping",
    "hierarchical",
)
CANDIDATE_QUERY_OPTIONS = (
    "candidate_token_direct_query",
    "candidate_summary_query",
    "candidate_task_query",
    "candidate_role_query",
    "multi_query_per_candidate",
    "learned_query_bank",
)
COORDINATOR_OPTIONS = (
    "single_cross_attention_layer",
    "multi_hop_cross_attention",
    "recurrent_refinement_2",
    "recurrent_refinement_4",
    "recurrent_refinement_8",
    "top_k_latent_view_selection",
    "gated_latent_view_selection",
    "mixture_of_views",
    "hierarchical_coordinator",
    "role_first_avenue_aggregation",
    "avenue_first_role_aggregation",
)
VIEW_SHARING_OPTIONS = (
    "fully_shared_encoder",
    "shared_encoder_role_embeddings",
    "shared_encoder_avenue_embeddings",
    "shared_encoder_role_avenue_embeddings",
    "shared_encoder_low_rank_adapters",
    "partially_shared_role_adapters",
    "independent_encoders_ablation",
)
MEMORY_OPTIONS = (
    "no_compression",
    "mean_pooling",
    "learned_summary_token",
    "top_k_token_retention",
    "candidate_conditioned_token_retention",
    "slot_attention_compression",
    "recurrent_memory_slots",
)
ROUTING_OPTIONS = (
    "dense_all_view_attention",
    "top_k_view_attention",
    "sparsemax_routing",
    "entmax_routing",
    "hard_routing_straight_through",
    "two_stage_retrieve_then_score",
    "coarse_to_fine_routing",
)
LOSS_OPTIONS = (
    "multi_positive_cross_entropy",
    "pairwise_ranking_loss",
    "margin_loss",
    "focal_loss_hard_negatives",
    "auxiliary_evidence_location",
    "contrastive_candidate_evidence_alignment",
)
REGULARIZER_OPTIONS = (
    (),
    ("avenue_diversity_penalty",),
    ("role_diversity_penalty",),
    ("entropy_regularization_on_routing",),
    ("load_balancing_across_avenues",),
    ("dropout_on_roles",),
    ("dropout_on_avenues",),
    ("evidence_masking_curriculum",),
    ("entropy_regularization_on_routing", "load_balancing_across_avenues"),
)
CURRICULUM_OPTIONS = (
    "direct_target_n",
    "curriculum_8_to_target",
    "mixed_n",
    "hard_negative_curriculum",
    "distractor_similarity_curriculum",
)


def required_stage8_baselines(base: Stage8ArchitectureConfig | None = None) -> List[Stage8ArchitectureConfig]:
    base = base or default_latent_config()
    return [
        replace(base, name="standard_monolithic_transformer_full_context", model_kind="monolithic_transformer", roles=1, avenues=1),
        replace(base, name="same_parameter_count_monolithic_transformer", model_kind="monolithic_transformer", roles=1, avenues=1, hidden_dim=base.hidden_dim),
        replace(base, name="same_compute_budget_monolithic_transformer", model_kind="monolithic_transformer", roles=1, avenues=1, hidden_dim=max(16, base.hidden_dim // 2)),
        replace(base, name="frozen_same_architecture_latent", model_kind="frozen_latent", frozen=True),
        replace(base, name="raw_latent_non_trainable_selector", model_kind="raw_latent_selector"),
        replace(base, name="single_role_latent_model", model_kind="trainable_latent", roles=1, avenues=max(1, base.avenues)),
        replace(base, name="single_avenue_latent_model", model_kind="trainable_latent", roles=max(1, base.roles), avenues=1),
        replace(base, name="text_only_multi_agent_style_baseline", model_kind="text_only_multi_agent"),
        replace(base, name="retrieval_topk_baseline", model_kind="retrieval_topk", top_k_views=4),
        replace(base, name="random_candidate_baseline", model_kind="random_candidate"),
        replace(base, name="candidate_only_baseline", model_kind="candidate_only"),
        replace(base, name="evidence_only_baseline", model_kind="evidence_only"),
        replace(base, name="query_only_baseline", model_kind="query_only"),
        replace(base, name="oracle_evidence_location_baseline", model_kind="oracle_evidence_location", top_k_views=4),
    ]


def default_latent_config() -> Stage8ArchitectureConfig:
    return Stage8ArchitectureConfig(
        name="latent_r4_a2_candidate_task_mixture",
        model_kind="trainable_latent",
        roles=4,
        avenues=2,
        chunking="semantic_key",
        candidate_query="candidate_task_query",
        coordinator="mixture_of_views",
        view_sharing="shared_encoder_role_avenue_embeddings",
        memory_compression="mean_pooling",
        attention_routing="dense_all_view_attention",
        training_loss="multi_positive_cross_entropy",
        regularizers=("entropy_regularization_on_routing",),
        curriculum="mixed_n",
        top_k_views=0,
    )


def generate_stage8_search_space(
    max_configs: int = 200,
    seed: int = 0,
    min_configs: int = 50,
) -> List[Stage8ArchitectureConfig]:
    """Generate a deterministic ASHA candidate pool without enumerating the full Cartesian product."""
    if max_configs < min_configs:
        raise ValueError("max_configs must be >= min_configs")
    rng = random.Random(seed)
    configs: Dict[str, Stage8ArchitectureConfig] = {}

    seeded = [
        default_latent_config(),
        replace(default_latent_config(), name="latent_r8_a4_topk_semantic", roles=8, avenues=4, attention_routing="top_k_view_attention", top_k_views=8),
        replace(default_latent_config(), name="latent_r16_a2_hierarchical", roles=16, avenues=2, chunking="hierarchical", coordinator="hierarchical_coordinator"),
        replace(default_latent_config(), name="latent_r4_a8_overlap_sparsemax", roles=4, avenues=8, chunking="overlapping", attention_routing="sparsemax_routing"),
        replace(default_latent_config(), name="latent_r8_a8_two_stage", roles=8, avenues=8, attention_routing="two_stage_retrieve_then_score", top_k_views=8),
        replace(default_latent_config(), name="latent_single_role_ablation", roles=1, avenues=8),
        replace(default_latent_config(), name="latent_single_avenue_ablation", roles=8, avenues=1),
    ]
    for config in seeded:
        configs[config.config_id] = config

    attempts = 0
    while len(configs) < max_configs and attempts < max_configs * 100:
        attempts += 1
        roles = rng.choice(ROLE_OPTIONS)
        avenues = rng.choice(AVENUE_OPTIONS)
        routing = rng.choice(ROUTING_OPTIONS)
        coordinator = rng.choice(COORDINATOR_OPTIONS)
        coord_hops = _coord_hops(coordinator)
        top_k = 0
        if routing in {"top_k_view_attention", "two_stage_retrieve_then_score", "coarse_to_fine_routing"} or "top_k" in coordinator:
            top_k = rng.choice((2, 4, 8, 16))
        config = Stage8ArchitectureConfig(
            name=f"latent_r{roles}_a{avenues}_{rng.randrange(1_000_000):06d}",
            model_kind="trainable_latent",
            roles=roles,
            avenues=avenues,
            chunking=rng.choice(CHUNKING_OPTIONS),
            candidate_query=rng.choice(CANDIDATE_QUERY_OPTIONS),
            coordinator=coordinator,
            view_sharing=rng.choice(VIEW_SHARING_OPTIONS),
            memory_compression=rng.choice(MEMORY_OPTIONS),
            attention_routing=routing,
            training_loss=rng.choice(LOSS_OPTIONS),
            regularizers=tuple(rng.choice(REGULARIZER_OPTIONS)),
            curriculum=rng.choice(CURRICULUM_OPTIONS),
            coord_hops=coord_hops,
            top_k_views=top_k,
            dropout=0.05 if rng.random() < 0.35 else 0.0,
            hidden_dim=rng.choice((48, 64, 96)),
            epochs=rng.choice((4, 6, 8)),
            lr=rng.choice((1e-3, 3e-3, 5e-3)),
        )
        configs[config.config_id] = config
    rows = list(configs.values())
    if len(rows) < min_configs:
        raise RuntimeError(f"generated only {len(rows)} Stage 8 configs")
    return rows[:max_configs]


def architecture_config_to_dict(config: Stage8ArchitectureConfig) -> Dict[str, object]:
    row = asdict(config)
    row["config_id"] = config.config_id
    row["latent_views"] = config.latent_views
    return row


def architecture_config_from_dict(row: Dict[str, object]) -> Stage8ArchitectureConfig:
    values = dict(row)
    values.pop("config_id", None)
    values.pop("latent_views", None)
    if "regularizers" in values:
        values["regularizers"] = tuple(values["regularizers"])  # type: ignore[index]
    return Stage8ArchitectureConfig(**values)  # type: ignore[arg-type]


def _coord_hops(coordinator: str) -> int:
    if coordinator.endswith("_8"):
        return 8
    if coordinator.endswith("_4"):
        return 4
    if coordinator.endswith("_2") or coordinator == "multi_hop_cross_attention":
        return 2
    return 1
