from src.datasets.synthetic import (
    SyntheticDatasetConfig,
    build_synthetic_splits,
    deterministic_label,
)


def test_synthetic_splits_are_clean_and_deterministic() -> None:
    config = SyntheticDatasetConfig(n_train=50, n_dev=20, n_test=20)
    splits = build_synthetic_splits(config, seed=123)
    seen = set()
    for split_name, examples in splits.items():
        assert examples
        for example in examples:
            key = (example.task_id, example.x)
            assert key not in seen
            seen.add(key)
            assert example.split == split_name
            assert example.label == deterministic_label(
                example.task_id,
                example.x,
                config.num_classes,
            )

