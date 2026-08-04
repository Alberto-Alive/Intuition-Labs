from src.datasets.latent_attention_capacity_dataset import (
    TASK_FAMILIES,
    Stage8DatasetConfig,
    build_stage8_examples,
    validate_stage8_examples,
)
from src.experiments.stage8_architecture_registry import generate_stage8_search_space
from src.experiments.stage8_capacity_evaluator import (
    Stage8EvaluationConfig,
    compute_effective_capacity,
    evaluate_stage8_architecture,
)
from src.experiments.stage8_controls import apply_stage8_control, shortcut_audit
from src.models.latent_attention_variants import Stage8ArchitectureConfig, build_stage8_selector, estimate_stage8_compute


def test_stage8_dataset_generates_all_families_without_basic_leakage() -> None:
    examples = build_stage8_examples(
        Stage8DatasetConfig(n_examples=60, n_blocks=8, split="final", template_split="final"),
        seed=123,
    )
    audit = validate_stage8_examples(examples)
    shortcut = shortcut_audit(examples)

    assert audit["passes"], audit["failures"]
    assert {example.task_family for example in examples} == set(TASK_FAMILIES)
    assert all(len(example.candidates) == 8 for example in examples)
    assert all(len(example.evidence_blocks) == 8 for example in examples)
    assert all(example.metadata["namespace"] == "fin" for example in examples)
    assert shortcut["passes"], shortcut["failures"]


def test_stage8_controls_preserve_or_break_labels_as_expected() -> None:
    examples = build_stage8_examples(Stage8DatasetConfig(n_examples=12, n_blocks=16), seed=5)
    ordered = apply_stage8_control(examples, "randomized_candidate_order_with_label_remap", seed=7)
    randomized = apply_stage8_control(examples, "randomized_labels", seed=8)
    candidate_only = apply_stage8_control(examples, "candidate_only", seed=9)
    evidence_only = apply_stage8_control(examples, "evidence_only", seed=10)

    for before, after in zip(examples, ordered):
        assert sorted(before.candidates) == sorted(after.candidates)
        assert before.candidates[before.label] == after.candidates[after.label]
    assert all(0 <= example.label < 8 for example in randomized)
    assert all(len(example.evidence_blocks) == 0 for example in candidate_only)
    assert all(len(example.evidence_blocks) == len(before.evidence_blocks) for before, example in zip(examples, evidence_only))


def test_stage8_registry_generates_minimum_search_space() -> None:
    configs = generate_stage8_search_space(max_configs=50, min_configs=50, seed=11)
    assert len(configs) == 50
    assert len({config.config_id for config in configs}) == 50
    assert any(config.roles == 16 for config in configs)
    assert any(config.avenues == 8 for config in configs)


def test_stage8_capacity_requires_stable_seeds_and_controls() -> None:
    eval_config = Stage8EvaluationConfig(min_stable_seeds=5)
    controls = {
        "randomized_labels": {
            "accuracy": 0.14,
            "degradation": 0.74,
            "delta_from_base": -0.74,
            "expectation": "degrade",
            "passes": True,
        },
        "evidence_candidate_mismatch": {
            "accuracy": 0.65,
            "degradation": 0.23,
            "delta_from_base": -0.23,
            "expectation": "degrade",
            "passes": True,
        },
        "cross_task_evidence_shuffle": {
            "accuracy": 0.66,
            "degradation": 0.22,
            "delta_from_base": -0.22,
            "expectation": "degrade",
            "passes": True,
        },
        "cross_task_query_shuffle": {
            "accuracy": 0.66,
            "degradation": 0.22,
            "delta_from_base": -0.22,
            "expectation": "degrade",
            "passes": True,
        },
        "distractor_only": {
            "accuracy": 0.66,
            "degradation": 0.22,
            "delta_from_base": -0.22,
            "expectation": "degrade",
            "passes": True,
        },
        "schema_template_only": {
            "accuracy": 0.66,
            "degradation": 0.22,
            "delta_from_base": -0.22,
            "expectation": "degrade",
            "passes": True,
        },
        "randomized_candidate_order_with_label_remap": {
            "accuracy": 0.88,
            "degradation": 0.0,
            "delta_from_base": 0.0,
            "expectation": "invariant",
            "passes": True,
        },
        "randomized_evidence_block_order": {
            "accuracy": 0.87,
            "degradation": 0.01,
            "delta_from_base": -0.01,
            "expectation": "invariant",
            "passes": True,
        },
    }
    rows = [
        {"n_blocks": 64, "seed": seed, "accuracy": 0.88, "chance_accuracy": 0.125, "controls": controls}
        for seed in range(5)
    ]
    rows += [
        {"n_blocks": 128, "seed": seed, "accuracy": 0.70, "chance_accuracy": 0.125, "controls": controls}
        for seed in range(5)
    ]

    capacity = compute_effective_capacity(rows, eval_config=eval_config, require_controls=True)
    underseeded = compute_effective_capacity(rows[:4], eval_config=eval_config, require_controls=True)

    assert capacity["capacity"] == 64
    assert underseeded["capacity"] == 0


def test_stage8_latent_model_and_evaluator_smoke() -> None:
    train = build_stage8_examples(Stage8DatasetConfig(n_examples=10, n_blocks=8), seed=21)
    dev = build_stage8_examples(Stage8DatasetConfig(n_examples=6, n_blocks=8, split="dev", template_split="dev"), seed=22)
    config = Stage8ArchitectureConfig(
        name="unit_latent",
        roles=2,
        avenues=2,
        epochs=1,
        hidden_dim=24,
        vector_dim=64,
        max_train_examples=10,
    )
    selector = build_stage8_selector(config, seed=3)
    selector.fit(train, dev)
    predictions = selector.predict(dev)
    compute = estimate_stage8_compute(config, n_blocks=8)

    assert len(predictions) == len(dev)
    assert all(0 <= prediction < 8 for prediction in predictions)
    assert compute["latent_views"] == 4

    result = evaluate_stage8_architecture(
        config,
        n_values=(8,),
        seeds=(0,),
        eval_config=Stage8EvaluationConfig(train_examples=10, eval_examples=6),
        run_controls=False,
    )
    assert result["status"] == "completed"
    assert result["rows"][0]["num_eval_examples"] == 6
