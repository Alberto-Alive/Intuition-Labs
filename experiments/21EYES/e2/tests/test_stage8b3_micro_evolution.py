from collections import Counter

from src.datasets.latent_attention_capacity_dataset import (
    TASK_FAMILIES,
    Stage8DatasetConfig,
    build_stage8_examples,
    validate_stage8_examples,
)
from src.experiments.run_stage8b3_micro_evolution_tournament import generate_stage8b3_variants


def test_stage8_small_n_examples_validate_for_tournament_schedule() -> None:
    for n_blocks in (2, 4, 8):
        examples = build_stage8_examples(Stage8DatasetConfig(n_examples=60, n_blocks=n_blocks), seed=8303)
        audit = validate_stage8_examples(examples)

        assert audit["passes"], audit["failures"]
        assert {example.task_family for example in examples} == set(TASK_FAMILIES)
        assert all(len(example.evidence_blocks) == n_blocks for example in examples)


def test_stage8b3_generates_required_family_coverage() -> None:
    variants = generate_stage8b3_variants(max_variants=60)
    counts = Counter(variant.family_code for variant in variants)

    assert len(variants) == 60
    assert len({variant.config.config_id for variant in variants}) == 60
    assert set(counts) == set("ABCDEFGHIJ")
    assert all(count >= 5 for count in counts.values())
