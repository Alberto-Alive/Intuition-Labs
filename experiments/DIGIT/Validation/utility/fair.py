"""Fair utility dataset views and binary subproblem adapters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch

from ..data.base import DatasetInfo, PrivateDataset, QueryField, TabularPrivateDataset
from ..data.mimic_iv_demo import load_mimic_iv_demo
from ..data.nist_genomics import load_nist_genomics
from ..data.prism import PRISMPrivateDataset, load_prism
from ..data.tcga import load_tcga
from .protocol import UtilityProtocol, UtilityProtocolError, require_fair_automated_utility


@dataclass(frozen=True)
class UtilitySubproblemBundle:
    key: str
    description: str
    checkpoint_dataset_name: str
    train: PrivateDataset
    val: PrivateDataset
    test: PrivateDataset


@dataclass(frozen=True)
class FairUtilityBundle:
    protocol: UtilityProtocol
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    subproblems: Tuple[UtilitySubproblemBundle, ...]


def _clone_dataset(
    dataset: PrivateDataset,
    *,
    features: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    query_fields: Sequence[QueryField] | None = None,
    dataset_name: str | None = None,
    label_name: str | None = None,
    responses: torch.Tensor | None = None,
    pair_ids: Sequence[str] | None = None,
) -> PrivateDataset:
    new_features = (features if features is not None else dataset.features).long().clone()
    new_labels = (labels if labels is not None else dataset.labels).long().clone()
    new_query_fields = list(query_fields if query_fields is not None else dataset.query_fields)
    field_vocab_sizes = [field.vocab_size for field in new_query_fields] + [len(new_query_fields) + 1]
    info = DatasetInfo(
        dataset_name=dataset_name or dataset.info.dataset_name,
        split_name=dataset.info.split_name,
        n_records=int(new_features.shape[0]),
        n_features=int(new_features.shape[1]),
        label_name=label_name or dataset.info.label_name,
        positive_rate=float(new_labels.float().mean()) if len(new_labels) else 0.0,
        query_fields=list(new_query_fields),
        field_vocab_sizes=field_vocab_sizes,
    )
    if isinstance(dataset, PRISMPrivateDataset) or responses is not None or hasattr(dataset, "responses"):
        new_responses = responses if responses is not None else dataset.responses
        new_pair_ids = list(pair_ids if pair_ids is not None else getattr(dataset, "pair_ids", []))
        return PRISMPrivateDataset(
            new_features,
            new_labels,
            new_responses.clone(),
            list(new_query_fields),
            info,
            pair_ids=new_pair_ids,
        )
    return TabularPrivateDataset(new_features, new_labels, list(new_query_fields), info)


def _drop_query_fields(dataset: PrivateDataset, drop_names: Iterable[str], dataset_name: str) -> PrivateDataset:
    drop_set = set(drop_names)
    kept_fields = [field for field in dataset.query_fields if field.name not in drop_set]
    if len(kept_fields) == len(dataset.query_fields):
        return _clone_dataset(dataset, dataset_name=dataset_name)
    kept_cols = [field.col_idx for field in kept_fields]
    remapped = [
        QueryField(name=field.name, col_idx=i, vocab_size=field.vocab_size, is_threshold=field.is_threshold)
        for i, field in enumerate(kept_fields)
    ]
    new_features = dataset.features[:, kept_cols]
    return _clone_dataset(
        dataset,
        features=new_features,
        query_fields=remapped,
        dataset_name=dataset_name,
    )


def _relabel_binary(dataset: PrivateDataset, labels_np: np.ndarray, dataset_name: str, label_name: str) -> PrivateDataset:
    labels = torch.tensor(labels_np.astype(np.int64), dtype=torch.long)
    return _clone_dataset(
        dataset,
        labels=labels,
        dataset_name=dataset_name,
        label_name=label_name,
    )


def _nist_bundle(data_cache: str, split_seed: int, protocol: UtilityProtocol) -> FairUtilityBundle:
    train, val, test = load_nist_genomics(cache_dir=f"{data_cache}/nist_genomics", split_seed=split_seed)
    y_train = train.labels.cpu().numpy().astype(int)
    y_val = val.labels.cpu().numpy().astype(int)
    y_test = test.labels.cpu().numpy().astype(int)
    subproblem = UtilitySubproblemBundle(
        key="eur_ancestry",
        description="Direct binary EUR ancestry classification.",
        checkpoint_dataset_name="nist_genomics_utility_eur_ancestry",
        train=_clone_dataset(train, dataset_name="nist_genomics_utility_eur_ancestry"),
        val=_clone_dataset(val, dataset_name="nist_genomics_utility_eur_ancestry"),
        test=_clone_dataset(test, dataset_name="nist_genomics_utility_eur_ancestry"),
    )
    return FairUtilityBundle(protocol=protocol, y_train=y_train, y_val=y_val, y_test=y_test, subproblems=(subproblem,))


def _mimic_bundle(data_cache: str, split_seed: int, protocol: UtilityProtocol) -> FairUtilityBundle:
    train, val, test = load_mimic_iv_demo(data_dir=f"{data_cache}/mimic_iv_demo", split_seed=split_seed)
    y_train = train.labels.cpu().numpy().astype(int)
    y_val = val.labels.cpu().numpy().astype(int)
    y_test = test.labels.cpu().numpy().astype(int)
    subproblem = UtilitySubproblemBundle(
        key="hospital_expire",
        description="Direct binary in-hospital mortality classification.",
        checkpoint_dataset_name="mimic_iv_demo_utility_hospital_expire",
        train=_clone_dataset(train, dataset_name="mimic_iv_demo_utility_hospital_expire"),
        val=_clone_dataset(val, dataset_name="mimic_iv_demo_utility_hospital_expire"),
        test=_clone_dataset(test, dataset_name="mimic_iv_demo_utility_hospital_expire"),
    )
    return FairUtilityBundle(protocol=protocol, y_train=y_train, y_val=y_val, y_test=y_test, subproblems=(subproblem,))


def _tcga_bundle(data_cache: str, split_seed: int, protocol: UtilityProtocol) -> FairUtilityBundle:
    train_raw, val_raw, test_raw = load_tcga(cache_dir=f"{data_cache}/tcga", split_seed=split_seed)
    y_train = train_raw.features[:, 5].cpu().numpy().astype(int) - 1
    y_val = val_raw.features[:, 5].cpu().numpy().astype(int) - 1
    y_test = test_raw.features[:, 5].cpu().numpy().astype(int) - 1

    train_view = _drop_query_fields(train_raw, protocol.excluded_query_fields, "tcga_utility_morphology")
    val_view = _drop_query_fields(val_raw, protocol.excluded_query_fields, "tcga_utility_morphology")
    test_view = _drop_query_fields(test_raw, protocol.excluded_query_fields, "tcga_utility_morphology")

    class_names = ["idc", "ilc", "mixed_other", "unknown"]
    subproblems: List[UtilitySubproblemBundle] = []
    for class_idx, class_name in enumerate(class_names):
        ckpt_name = f"tcga_utility_morphology_{class_name}"
        label_name = f"morphology_is_{class_name}"
        subproblems.append(
            UtilitySubproblemBundle(
                key=class_name,
                description=f"One-vs-rest morphology target for class `{class_name}`.",
                checkpoint_dataset_name=ckpt_name,
                train=_relabel_binary(train_view, y_train == class_idx, ckpt_name, label_name),
                val=_relabel_binary(val_view, y_val == class_idx, ckpt_name, label_name),
                test=_relabel_binary(test_view, y_test == class_idx, ckpt_name, label_name),
            )
        )
    return FairUtilityBundle(
        protocol=protocol,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        subproblems=tuple(subproblems),
    )


def _prism_bundle(data_cache: str, split_seed: int, protocol: UtilityProtocol) -> FairUtilityBundle:
    train_raw, val_raw, test_raw = load_prism(data_dir=f"{data_cache}/prism", split_seed=split_seed)
    y_train = train_raw.responses.cpu().numpy().astype(float)
    y_val = val_raw.responses.cpu().numpy().astype(float)
    y_test = test_raw.responses.cpu().numpy().astype(float)

    train_view = _drop_query_fields(train_raw, protocol.excluded_query_fields, "prism_utility_response")
    val_view = _drop_query_fields(val_raw, protocol.excluded_query_fields, "prism_utility_response")
    test_view = _drop_query_fields(test_raw, protocol.excluded_query_fields, "prism_utility_response")

    raw_thresholds = np.quantile(y_train, [0.2, 0.4, 0.6, 0.8])
    thresholds: List[float] = []
    for value in raw_thresholds.tolist():
        if not thresholds or abs(value - thresholds[-1]) > 1e-8:
            thresholds.append(float(value))
    if not thresholds:
        thresholds = [float(np.median(y_train))]

    subproblems: List[UtilitySubproblemBundle] = []
    for idx, threshold in enumerate(thresholds):
        pct = int(round(100.0 * (idx + 1) / (len(thresholds) + 1)))
        key = f"le_q{pct:02d}"
        ckpt_name = f"prism_utility_response_{key}"
        label_name = f"response_le_{threshold:.4f}"
        subproblems.append(
            UtilitySubproblemBundle(
                key=key,
                description=f"Thresholded response target `response <= {threshold:.4f}`.",
                checkpoint_dataset_name=ckpt_name,
                train=_relabel_binary(train_view, y_train <= threshold, ckpt_name, label_name),
                val=_relabel_binary(val_view, y_val <= threshold, ckpt_name, label_name),
                test=_relabel_binary(test_view, y_test <= threshold, ckpt_name, label_name),
            )
        )
    return FairUtilityBundle(
        protocol=protocol,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        subproblems=tuple(subproblems),
    )


def validate_fair_utility_bundle(bundle: FairUtilityBundle) -> None:
    protocol = bundle.protocol
    if not bundle.subproblems:
        raise UtilityProtocolError(f"Fair utility bundle for `{protocol.dataset_name}` has no subproblems.")

    expected_train_shape = None
    expected_val_shape = None
    expected_test_shape = None
    for subproblem in bundle.subproblems:
        for split_name, dataset in (("train", subproblem.train), ("val", subproblem.val), ("test", subproblem.test)):
            field_names = [field.name for field in dataset.query_fields]
            for forbidden in protocol.excluded_query_fields + protocol.derived_query_fields:
                if forbidden in field_names:
                    raise UtilityProtocolError(
                        f"Utility bundle `{protocol.dataset_name}` still exposes forbidden query field "
                        f"`{forbidden}` in {subproblem.key}/{split_name}."
                    )
            if protocol.target_name in field_names:
                raise UtilityProtocolError(
                    f"Utility bundle `{protocol.dataset_name}` still exposes target field "
                    f"`{protocol.target_name}` in {subproblem.key}/{split_name}."
                )
            labels = dataset.labels.cpu().numpy().astype(int)
            label_min = int(labels.min()) if labels.size else 0
            label_max = int(labels.max()) if labels.size else 0
            if label_min < 0 or label_max > 1:
                raise UtilityProtocolError(
                    f"Utility subproblem `{protocol.dataset_name}:{subproblem.key}` is not binary-labeled."
                )
        train_shape = tuple(subproblem.train.features.shape)
        val_shape = tuple(subproblem.val.features.shape)
        test_shape = tuple(subproblem.test.features.shape)
        if expected_train_shape is None:
            expected_train_shape = train_shape
            expected_val_shape = val_shape
            expected_test_shape = test_shape
        if train_shape != expected_train_shape or val_shape != expected_val_shape or test_shape != expected_test_shape:
            raise UtilityProtocolError(
                f"Utility subproblem `{protocol.dataset_name}:{subproblem.key}` uses inconsistent feature shapes "
                "across the task adapter."
            )


def build_fair_utility_bundle(
    dataset_name: str,
    *,
    data_cache: str = "data_cache",
    split_seed: int = 0,
) -> FairUtilityBundle:
    protocol = require_fair_automated_utility(dataset_name)
    if dataset_name == "nist_genomics":
        bundle = _nist_bundle(data_cache, split_seed, protocol)
    elif dataset_name == "mimic_iv_demo":
        bundle = _mimic_bundle(data_cache, split_seed, protocol)
    elif dataset_name == "tcga":
        bundle = _tcga_bundle(data_cache, split_seed, protocol)
    elif dataset_name == "prism":
        bundle = _prism_bundle(data_cache, split_seed, protocol)
    else:
        raise UtilityProtocolError(f"No fair utility bundle builder defined for dataset: {dataset_name}")
    validate_fair_utility_bundle(bundle)
    return bundle
