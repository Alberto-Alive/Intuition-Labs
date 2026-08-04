"""Smoke checks for the comparative Validation framework.

Run with:
    python3 experiments/DIGIT/Validation/scripts/test_comparative_smoke.py

This script assumes the Validation requirements are installed.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

_digit_root = Path(__file__).resolve().parents[2]
if str(_digit_root) not in sys.path:
    sys.path.insert(0, str(_digit_root))

from Validation.attacks.membership_inference import ShadowMIAConfig, run_mia_full
from Validation.attacks.repeated_query_attacks import (
    run_composition_stress_attack,
    run_query_averaging_attack,
)
from Validation.comparative.registry import get_supported_dataset_names
from Validation.comparative.querying import choose_canary_feature_values, dataset_with_appended_record
from Validation.comparative.specs import AttackMode, SystemSpec
from Validation.comparative.stability_gate import StabilityGateConfig, StabilityGatedDigitWrapper
from Validation.comparative.stats import bootstrap_ci, summarize_per_seed_metrics
from Validation.comparative.systems import comparative_system_key, disambiguate_digit_variants, make_digit_config
from Validation.comparative.utility import (
    _fit_binary_utility_model,
    _fit_multiclass_utility_model,
    _fit_regression_utility_model,
    metric_direction,
    utility_comparison_fields,
)
from Validation.data.base import DatasetInfo, QueryField, TabularPrivateDataset
from Validation.data.dataset import ValidationDataset
from Validation.harness.output_writer import write_seed_run
from Validation.models.digit import ValidationDIGITModel
from Validation.models.dp_baseline import GenericDPLaplaceBaseline
from Validation.models.raw_baseline import RawBaseline
from Validation.utility.fair import FairUtilityBundle, UtilitySubproblemBundle, _drop_query_fields, validate_fair_utility_bundle
from Validation.utility.protocol import (
    UtilityProtocolError,
    allowed_automated_utility_datasets,
    get_utility_protocol,
    require_fair_automated_utility,
)


def _toy_dataset(n: int = 120) -> TabularPrivateDataset:
    gen = torch.Generator().manual_seed(0)
    features = torch.randint(1, 5, (n, 12), generator=gen)
    labels = ((features[:, 0] + features[:, 1] + features[:, 2]) % 2).long()
    qfields = [QueryField(f"f{i}", i, 4) for i in range(12)]
    info = DatasetInfo(
        dataset_name="toy",
        split_name="train",
        n_records=n,
        n_features=12,
        label_name="label",
        positive_rate=float(labels.float().mean()),
        query_fields=qfields,
        field_vocab_sizes=[4] * 12 + [13],
    )
    return TabularPrivateDataset(features, labels, qfields, info)


def main() -> None:
    dataset = _toy_dataset()
    shadow_cfg = ShadowMIAConfig(n_shadow=4, n_queries_per_record=4, batch_size=16)

    assert "nist_genomics" in get_supported_dataset_names()
    assert "prism" in get_supported_dataset_names()
    assert allowed_automated_utility_datasets() == ["mimic_iv_demo", "nist_genomics", "prism", "tcga"]
    assert require_fair_automated_utility("nist_genomics").automated_runner_allowed
    assert require_fair_automated_utility("mimic_iv_demo").automated_runner_allowed
    assert require_fair_automated_utility("tcga").automated_runner_allowed
    assert require_fair_automated_utility("prism").automated_runner_allowed
    assert get_utility_protocol("tcga").task_type == "multiclass_classification"
    assert get_utility_protocol("tcga").excluded_query_fields == ("morphology_group",)
    assert get_utility_protocol("prism").task_type == "regression_ranking"
    assert get_utility_protocol("prism").derived_query_fields == (
        "response_bin",
        "abs_response_bin",
        "sensitivity_flag",
    )

    raw_mech = RawBaseline(dataset, min_group_size=AttackMode.MECHANISM.min_group_size)
    raw_sys = RawBaseline(dataset, min_group_size=AttackMode.SYSTEM.min_group_size)
    dp_mech = GenericDPLaplaceBaseline(dataset, epsilon=1.0, min_group_size=AttackMode.MECHANISM.min_group_size, seed=0)

    assert raw_mech.cfg.min_group_size == 1
    assert raw_sys.cfg.min_group_size == 10
    assert dp_mech.cfg.min_group_size == 1
    dp_out = dp_mech.answer_query(torch.zeros(13, dtype=torch.long))
    assert "true_rate" not in dp_out
    assert "noisy_rate" in dp_out
    assert "noisy_n" in dp_out

    digit_cfg = make_digit_config(dataset, mode=AttackMode.MECHANISM, digit_variant="a2_s2_c2_r2")
    digit = ValidationDIGITModel(digit_cfg)
    q = dataset.features[0].clone()
    query = torch.zeros(13, dtype=torch.long)
    query[:12] = q
    query[12] = 12
    out = digit.generate(query.unsqueeze(0), dataset)
    assert out["primitives"].answer_discrete.shape[-1] == 2

    coarse_ds = ValidationDataset(
        dataset,
        n_queries=64,
        seed=0,
        class_counts={
            "answer": 2,
            "support": 2,
            "confidence": 2,
            "risk": 2,
        },
    )
    coarse_targets = torch.stack(coarse_ds.targets)
    assert int(coarse_targets[:, 0].max().item()) < 2
    assert int(coarse_targets[:, 1].max().item()) < 2
    assert int(coarse_targets[:, 2].max().item()) < 2
    assert int(coarse_targets[:, 3].max().item()) < 2
    coarse_weights = coarse_ds.get_class_weights()
    assert tuple(coarse_weights["answer"].shape) == (2,)
    assert tuple(coarse_weights["support"].shape) == (2,)
    assert tuple(coarse_weights["confidence"].shape) == (2,)
    assert tuple(coarse_weights["risk"].shape) == (2,)

    gated_digit = StabilityGatedDigitWrapper(digit, dataset)
    gated_out = gated_digit.answer_query(query)
    assert gated_out["primitive_vector"].shape == (4,)
    assert gated_out["answer"] in {0, 1, 2}
    specificity_gated_digit = StabilityGatedDigitWrapper(
        digit,
        dataset,
        gate_cfg=StabilityGateConfig(max_drop_neighbors=8, max_public_specificity=6),
    )
    specificity_out = specificity_gated_digit.answer_query(query)
    assert bool(specificity_out["specificity_guarded"])

    single_digit_spec = SystemSpec(family="digit", digit_variant="a6_s4_c3_r3")
    single_gated_spec = SystemSpec(family="digit_stability_gate", digit_variant="a6_s4_c3_r3")
    single_specificity_spec = SystemSpec(family="digit_specificity_gate", digit_variant="a6_s4_c3_r3")
    multi_digit_specs = [
        SystemSpec(family="digit", digit_variant="a2_s2_c2_r2"),
        SystemSpec(family="digit", digit_variant="a6_s4_c3_r3"),
    ]
    assert not disambiguate_digit_variants([single_digit_spec])
    assert disambiguate_digit_variants(multi_digit_specs)
    assert comparative_system_key(single_digit_spec, disambiguate_digit=False) == "digit"
    assert comparative_system_key(single_digit_spec, disambiguate_digit=True) == "digit_a6_s4_c3_r3"
    assert comparative_system_key(single_gated_spec, disambiguate_digit=False) == "digit_stability_gate"
    assert comparative_system_key(single_gated_spec, disambiguate_digit=True) == "digit_stability_gate_a6_s4_c3_r3"
    assert comparative_system_key(single_specificity_spec, disambiguate_digit=False) == "digit_specificity_gate"
    assert comparative_system_key(single_specificity_spec, disambiguate_digit=True) == "digit_specificity_gate_a6_s4_c3_r3"
    assert [comparative_system_key(spec, disambiguate_digit=True) for spec in multi_digit_specs] == [
        "digit_a2_s2_c2_r2",
        "digit_a6_s4_c3_r3",
    ]
    variant_summary = {
        comparative_system_key(spec, disambiguate_digit=True): {}
        for spec in multi_digit_specs
    }
    assert sorted(variant_summary) == ["digit_a2_s2_c2_r2", "digit_a6_s4_c3_r3"]

    mia = run_mia_full(
        raw_mech,
        dataset,
        dataset,
        seed=0,
        num_samples=40,
        shadow_cfg=shadow_cfg,
        system_factory=lambda ds: RawBaseline(ds, min_group_size=1),
    )
    assert "query_count" in mia
    assert "direct" in mia and "shadow" in mia

    avg = run_query_averaging_attack(raw_mech, dataset, seed=0, query_budget=40)
    comp = run_composition_stress_attack(raw_mech, dataset, dataset, seed=0, query_budget=100, num_targets=20)
    assert avg["query_count"] > 0
    assert comp["query_count"] > 0

    ci = bootstrap_ci([0.5, 0.55, 0.6, 0.52, 0.58], seed=0)
    assert ci[0] <= ci[1]
    threshold_summary = summarize_per_seed_metrics(
        [
            {"query_count_to_80pct_accuracy": None},
            {"query_count_to_80pct_accuracy": 50},
        ],
        ["query_count_to_80pct_accuracy"],
        seed=0,
    )
    assert threshold_summary["query_count_to_80pct_accuracy_mean"] == 50.0

    tcga_like_qfields = [
        QueryField("age_at_diagnosis_bin", 0, 6),
        QueryField("gender", 1, 2),
        QueryField("morphology_group", 2, 4),
        QueryField("vital_status_feat", 3, 2),
    ]
    tcga_like_info = DatasetInfo(
        dataset_name="tcga_like",
        split_name="train",
        n_records=dataset.num_records,
        n_features=4,
        label_name="mortality",
        positive_rate=float(dataset.labels.float().mean()),
        query_fields=tcga_like_qfields,
        field_vocab_sizes=[6, 2, 4, 2, 5],
    )
    tcga_like = TabularPrivateDataset(dataset.features[:, :4].clone(), dataset.labels.clone(), tcga_like_qfields, tcga_like_info)
    tcga_view = _drop_query_fields(tcga_like, {"morphology_group"}, "tcga_utility_morphology")
    assert [field.name for field in tcga_view.query_fields] == [
        "age_at_diagnosis_bin",
        "gender",
        "vital_status_feat",
    ]
    assert tcga_view.features.shape[1] == 3
    tcga_y = np.array([0, 1, 0, 1] * 30, dtype=int)
    tcga_binary = TabularPrivateDataset(
        tcga_view.features.clone(),
        torch.tensor((tcga_y[: tcga_view.num_records] == 1).astype(np.int64)),
        tcga_view.query_fields,
        DatasetInfo(
            dataset_name="tcga_utility_morphology_class1",
            split_name="train",
            n_records=tcga_view.num_records,
            n_features=3,
            label_name="morphology_is_ilc",
            positive_rate=float((tcga_y[: tcga_view.num_records] == 1).mean()),
            query_fields=tcga_view.query_fields,
            field_vocab_sizes=[field.vocab_size for field in tcga_view.query_fields] + [len(tcga_view.query_fields) + 1],
        ),
    )
    validate_fair_utility_bundle(
        FairUtilityBundle(
            protocol=get_utility_protocol("tcga"),
            y_train=tcga_y[: tcga_view.num_records],
            y_val=tcga_y[: tcga_view.num_records],
            y_test=tcga_y[: tcga_view.num_records],
            subproblems=(
                UtilitySubproblemBundle(
                    key="ilc",
                    description="toy one-vs-rest",
                    checkpoint_dataset_name="tcga_utility_morphology_ilc",
                    train=tcga_binary,
                    val=tcga_binary,
                    test=tcga_binary,
                ),
            ),
        )
    )

    X_train = np.random.RandomState(0).randn(32, 6)
    X_val = np.random.RandomState(1).randn(16, 6)
    X_test = np.random.RandomState(2).randn(16, 6)
    y_train_bin = (X_train[:, 0] > 0).astype(int)
    y_val_bin = (X_val[:, 0] > 0).astype(int)
    y_test_bin = (X_test[:, 0] > 0).astype(int)
    binary_metrics = _fit_binary_utility_model(X_train, y_train_bin, X_val, y_val_bin, X_test, y_test_bin, seed=0)
    assert {"auroc", "auprc", "f1", "accuracy", "balanced_accuracy", "brier"}.issubset(binary_metrics)

    y_train_mc = np.argmax(np.stack([X_train[:, 0], X_train[:, 1], -X_train[:, 0]], axis=1), axis=1)
    y_val_mc = np.argmax(np.stack([X_val[:, 0], X_val[:, 1], -X_val[:, 0]], axis=1), axis=1)
    y_test_mc = np.argmax(np.stack([X_test[:, 0], X_test[:, 1], -X_test[:, 0]], axis=1), axis=1)
    multiclass_metrics = _fit_multiclass_utility_model(X_train, y_train_mc, X_val, y_val_mc, X_test, y_test_mc, seed=0)
    assert {"macro_f1", "accuracy", "log_loss"}.issubset(multiclass_metrics)

    y_train_reg = X_train[:, 0] * 0.7 + X_train[:, 1] * 0.3
    y_val_reg = X_val[:, 0] * 0.7 + X_val[:, 1] * 0.3
    y_test_reg = X_test[:, 0] * 0.7 + X_test[:, 1] * 0.3
    regression_metrics = _fit_regression_utility_model(X_train, y_train_reg, X_val, y_val_reg, X_test, y_test_reg, seed=0)
    assert {"rmse", "pearson", "spearman", "ndcg"}.issubset(regression_metrics)

    assert metric_direction("auroc") == "higher"
    assert metric_direction("rmse") == "lower"
    comparisons = utility_comparison_fields(
        get_utility_protocol("nist_genomics"),
        {
            "raw": {"auroc_mean": 0.90},
            "digit": {"auroc_mean": 0.81},
            "dp_laplace_e0p1": {"auroc_mean": 0.78},
        },
    )
    assert comparisons["best_dp_system"] == "dp_laplace_e0p1"
    assert abs(comparisons["per_system"]["digit"]["retained_utility_vs_raw"] - 0.9) < 1e-9

    canary_vals = choose_canary_feature_values(dataset)
    augmented = dataset_with_appended_record(dataset, canary_vals, 1)
    assert augmented.num_records == dataset.num_records + 1
    assert dataset.num_records == 120

    with tempfile.TemporaryDirectory() as tmpdir:
        write_seed_run(
            phase="phase1_privacy",
            dataset="toy",
            task="smoke_task",
            system_key="raw",
            variant="mechanism__default",
            seed=0,
            seed_mode="reduced",
            metrics={"auroc": 0.5, "query_count": 10},
            predictions=[{"x": 1}],
            events=[{"query_count": 10, "auroc": 0.5}],
            config_extra={"mode": "mechanism"},
            results_root=Path(tmpdir),
        )
        out_dir = Path(tmpdir) / "phase1_privacy" / "toy" / "smoke_task" / "raw" / "mechanism__default" / "seed_0"
        for filename in ["config.json", "metrics.json", "events.jsonl", "summary.md"]:
            assert (out_dir / filename).exists(), filename

        for system_key in ["digit_a2_s2_c2_r2", "digit_a6_s4_c3_r3"]:
            write_seed_run(
                phase="phase1_privacy",
                dataset="toy",
                task="variant_smoke",
                system_key=system_key,
                variant="mechanism__default",
                seed=0,
                seed_mode="reduced",
                metrics={"auroc": 0.5, "query_count_to_80pct_accuracy": None},
                predictions=[],
                events=[],
                config_extra={"mode": "mechanism"},
                results_root=Path(tmpdir),
            )
        variant_root = Path(tmpdir) / "phase1_privacy" / "toy" / "variant_smoke"
        assert (variant_root / "digit_a2_s2_c2_r2" / "mechanism__default" / "seed_0" / "config.json").exists()
        assert (variant_root / "digit_a6_s4_c3_r3" / "mechanism__default" / "seed_0" / "config.json").exists()

    legacy = Path("experiments/DIGIT/Validation/results/phase1_privacy/mia_summary.json")
    if legacy.exists():
        payload = json.loads(legacy.read_text())
        assert "nist_genomics" in payload
        assert "tcga" in payload

    print("comparative smoke checks: OK")


if __name__ == "__main__":
    main()
