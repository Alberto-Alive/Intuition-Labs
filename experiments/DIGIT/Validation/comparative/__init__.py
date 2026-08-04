"""Comparative DIGIT-vs-DP evaluation helpers."""

from .specs import (
    AttackMode,
    AttackRunConfig,
    DatasetAttackSpec,
    SystemSpec,
)
from .registry import (
    ComparativeDatasetBundle,
    get_dataset_attack_spec,
    get_supported_dataset_names,
    load_dataset_bundle,
)
from .systems import (
    DIGIT_VARIANTS,
    DEFAULT_DP_EPSILONS,
    system_dir_name,
    variant_name_for_mode,
    build_system,
    ensure_digit_checkpoints,
    make_digit_config,
)

__all__ = [
    "AttackMode",
    "AttackRunConfig",
    "DatasetAttackSpec",
    "SystemSpec",
    "ComparativeDatasetBundle",
    "get_dataset_attack_spec",
    "get_supported_dataset_names",
    "load_dataset_bundle",
    "DIGIT_VARIANTS",
    "DEFAULT_DP_EPSILONS",
    "system_dir_name",
    "variant_name_for_mode",
    "build_system",
    "ensure_digit_checkpoints",
    "make_digit_config",
]
