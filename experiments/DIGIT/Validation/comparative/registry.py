"""Dataset registry for comparative attacks and utility runs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ..data.base import PrivateDataset
from ..data.mimic_iv_demo import load_mimic_iv_demo
from ..data.nist_genomics import load_nist_genomics
from ..data.prism import load_prism
from ..data.tcga import load_tcga
from .specs import DatasetAttackSpec


@dataclass(frozen=True)
class ComparativeDatasetBundle:
    train: PrivateDataset
    val: PrivateDataset
    test: PrivateDataset
    attack_spec: DatasetAttackSpec


_DATASET_SPECS: Dict[str, DatasetAttackSpec] = {
    "nist_genomics": DatasetAttackSpec(
        dataset_name="nist_genomics",
        hidden_attribute_name="eur_ancestry",
        reconstruction_target_name="eur_ancestry",
        canary_label=1,
        linkage_overlap_fraction=0.50,
        utility_task="binary_classification",
        utility_metrics=["auroc", "auprc", "f1", "accuracy"],
        notes={"aux_knowledge": "public reference panel prevalence features"},
    ),
    "tcga": DatasetAttackSpec(
        dataset_name="tcga",
        hidden_attribute_name="mortality",
        reconstruction_target_name="mortality",
        canary_label=1,
        linkage_overlap_fraction=0.50,
        utility_task="binary_classification",
        utility_metrics=["auroc", "auprc", "f1", "accuracy"],
    ),
    "mimic_iv_demo": DatasetAttackSpec(
        dataset_name="mimic_iv_demo",
        hidden_attribute_name="hospital_expire",
        reconstruction_target_name="hospital_expire",
        canary_label=1,
        linkage_overlap_fraction=0.50,
        utility_task="binary_classification",
        utility_metrics=["auroc", "balanced_accuracy", "f1", "accuracy"],
    ),
    "prism": DatasetAttackSpec(
        dataset_name="prism",
        hidden_attribute_name="response_sensitive",
        reconstruction_target_name="response_sensitive",
        canary_label=1,
        linkage_overlap_fraction=0.50,
        utility_task="regression_ranking",
        utility_metrics=["rmse", "pearson", "spearman", "ndcg"],
    ),
}


def get_supported_dataset_names() -> List[str]:
    return sorted(_DATASET_SPECS)


def get_dataset_attack_spec(name: str) -> DatasetAttackSpec:
    if name not in _DATASET_SPECS:
        raise ValueError(f"Unknown dataset: {name}")
    return _DATASET_SPECS[name]


def load_dataset_bundle(
    name: str,
    data_cache: str = "data_cache",
    split_seed: int = 0,
) -> ComparativeDatasetBundle:
    cache_root = Path(data_cache)
    if name == "nist_genomics":
        train, val, test = load_nist_genomics(
            cache_dir=str(cache_root / "nist_genomics"),
            split_seed=split_seed,
        )
    elif name == "tcga":
        train, val, test = load_tcga(
            cache_dir=str(cache_root / "tcga"),
            split_seed=split_seed,
        )
    elif name == "mimic_iv_demo":
        train, val, test = load_mimic_iv_demo(
            data_dir=str(cache_root / "mimic_iv_demo"),
            split_seed=split_seed,
        )
    elif name == "prism":
        train, val, test = load_prism(
            data_dir=str(cache_root / "prism"),
            split_seed=split_seed,
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return ComparativeDatasetBundle(train=train, val=val, test=test, attack_spec=get_dataset_attack_spec(name))
