"""Locked fair-utility protocol for automated Validation runs.

Unlike the original proxy utility runner, this protocol is decision-complete:
it defines the downstream target, the query-schema exclusions needed to avoid
target leakage, the split rule, the metric family, and whether a dataset is
safe to use as a headline utility claim.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


class UtilityProtocolError(RuntimeError):
    """Raised when an automated utility run would violate the locked protocol."""


@dataclass(frozen=True)
class UtilityProtocol:
    dataset_name: str
    automated_runner_allowed: bool
    task_type: str
    subproblem_strategy: str
    target_name: str
    target_source: str
    split_rule: str
    excluded_query_fields: Tuple[str, ...] = ()
    derived_query_fields: Tuple[str, ...] = ()
    primary_metrics: Tuple[str, ...] = ()
    secondary_metrics: Tuple[str, ...] = ()
    headline_safe: bool = True
    published_role: str = "headline"
    rationale: str = ""
    notes: str = ""

    @property
    def primary_metric(self) -> str:
        if not self.primary_metrics:
            raise UtilityProtocolError(
                f"Utility protocol for `{self.dataset_name}` does not define a primary metric."
            )
        return self.primary_metrics[0]


_PROTOCOLS: Dict[str, UtilityProtocol] = {
    "nist_genomics": UtilityProtocol(
        dataset_name="nist_genomics",
        automated_runner_allowed=True,
        task_type="binary_classification",
        subproblem_strategy="direct_binary",
        target_name="eur_ancestry",
        target_source="dataset label emitted by load_nist_genomics()",
        split_rule="Use the frozen train/val/test split emitted by load_nist_genomics().",
        primary_metrics=("auroc", "auprc", "f1", "accuracy", "brier"),
        secondary_metrics=("primitive_accuracy_mean",),
        headline_safe=True,
        published_role="headline",
        rationale=(
            "Fair automated benchmark: the utility target is the frozen dataset label, and it is not present "
            "in the queryable feature schema."
        ),
        notes=(
            "This remains the first automated utility benchmark for NIST; competitor-pack-specific tasks are "
            "still out of scope for the current implementation."
        ),
    ),
    "mimic_iv_demo": UtilityProtocol(
        dataset_name="mimic_iv_demo",
        automated_runner_allowed=True,
        task_type="binary_classification",
        subproblem_strategy="direct_binary",
        target_name="hospital_expire",
        target_source="dataset label emitted by load_mimic_iv_demo()",
        split_rule="Use the frozen patient-level train/val/test split emitted by load_mimic_iv_demo().",
        primary_metrics=("auroc", "balanced_accuracy", "f1", "accuracy", "brier"),
        secondary_metrics=("primitive_accuracy_mean",),
        headline_safe=False,
        published_role="sanity_check",
        rationale=(
            "Fair automated benchmark: the mortality label is not directly encoded in the current query schema."
        ),
        notes="Sanity-check benchmark only; do not use as the primary external utility claim.",
    ),
    "tcga": UtilityProtocol(
        dataset_name="tcga",
        automated_runner_allowed=True,
        task_type="multiclass_classification",
        subproblem_strategy="one_vs_rest",
        target_name="morphology_group",
        target_source="feature column `morphology_group`, repurposed as the downstream target for utility only",
        split_rule="Use the frozen patient-level train/val/test split emitted by load_tcga().",
        excluded_query_fields=("morphology_group",),
        primary_metrics=("macro_f1", "accuracy", "log_loss"),
        secondary_metrics=("primitive_accuracy_mean",),
        headline_safe=True,
        published_role="headline",
        rationale=(
            "Fair automated benchmark: the utility target is removed from the utility query schema, so systems "
            "cannot trivially recover it from directly exposed query fields."
        ),
        notes=(
            "The automated runner evaluates multiclass morphology prediction via one-vs-rest binary subproblems "
            "because the current DIGIT/raw/DP query stack is binary-rate based."
        ),
    ),
    "prism": UtilityProtocol(
        dataset_name="prism",
        automated_runner_allowed=True,
        task_type="regression_ranking",
        subproblem_strategy="threshold_ladder",
        target_name="response",
        target_source="continuous response retained in PRISMPrivateDataset.responses",
        split_rule="Use the frozen held-out pair split emitted by load_prism().",
        excluded_query_fields=("response_bin", "abs_response_bin", "sensitivity_flag"),
        derived_query_fields=("response_bin", "abs_response_bin", "sensitivity_flag"),
        primary_metrics=("rmse", "pearson", "spearman", "ndcg"),
        secondary_metrics=("primitive_accuracy_mean",),
        headline_safe=True,
        published_role="headline",
        rationale=(
            "Fair automated benchmark: response-derived fields are removed from the utility query schema before "
            "utility evaluation."
        ),
        notes=(
            "The automated runner approximates continuous-response utility with a threshold ladder of binary "
            "subproblems while reporting regression/ranking metrics on the held-out pair targets."
        ),
    ),
}


def get_utility_protocol(dataset_name: str) -> UtilityProtocol:
    if dataset_name not in _PROTOCOLS:
        raise UtilityProtocolError(f"No locked utility protocol defined for dataset: {dataset_name}")
    return _PROTOCOLS[dataset_name]


def require_fair_automated_utility(dataset_name: str) -> UtilityProtocol:
    protocol = get_utility_protocol(dataset_name)
    if not protocol.automated_runner_allowed:
        raise UtilityProtocolError(
            f"Automated utility frontier is blocked for `{dataset_name}`. {protocol.rationale} {protocol.notes}"
        )
    return protocol


def allowed_automated_utility_datasets() -> List[str]:
    return sorted(
        name for name, protocol in _PROTOCOLS.items() if protocol.automated_runner_allowed
    )
