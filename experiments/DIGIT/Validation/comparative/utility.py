"""Utility evaluation helpers for comparative runs."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_squared_error,
    ndcg_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor

from ..data.base import PrivateDataset
from ..data.ground_truth import GTConfig, GroundTruthComputer, coerce_gt_config
from ..data.query_gen import QueryGenerator
from ..utility.fair import FairUtilityBundle
from ..utility.protocol import UtilityProtocol
from .querying import build_partial_queries, primitive_vector_from_system
from .specs import AttackMode, SystemSpec
from .systems import build_system, ensure_digit_checkpoints


def evaluate_internal_primitive_utility(
    system: Any,
    private_data: PrivateDataset,
    seed: int,
    num_queries: int = 1000,
    gt_cfg: GTConfig | None = None,
) -> Dict[str, float]:
    gt = GroundTruthComputer(private_data, coerce_gt_config(gt_cfg or getattr(system, "cfg", None)))
    gen = QueryGenerator(private_data, include_specificity_field=True)
    queries = gen.generate(num_queries, seed=seed)
    gold = []
    pred = []
    for q in queries:
        g = gt.compute(q)
        r = primitive_vector_from_system(system, q, private_data)
        gold.append([g["answer"], g["support"], g["confidence"], g["risk"]])
        pred.append([r["answer"], r["support"], r["confidence"], r["risk"]])
    gold_arr = np.asarray(gold, dtype=int)
    pred_arr = np.asarray(pred, dtype=int)
    names = ["answer", "support", "confidence", "risk"]
    metrics: Dict[str, float] = {}
    accs = []
    for i, name in enumerate(names):
        acc = float(accuracy_score(gold_arr[:, i], pred_arr[:, i]))
        metrics[f"{name}_accuracy"] = acc
        accs.append(acc)
    metrics["primitive_accuracy_mean"] = float(np.mean(accs))
    return metrics


def _downstream_features(
    system: Any,
    private_data: PrivateDataset,
    target_data: PrivateDataset,
    seed: int,
    queries_per_record: int,
) -> np.ndarray:
    feats: List[np.ndarray] = []
    for idx in range(target_data.num_records):
        rec = target_data.features[idx]
        queries = build_partial_queries(rec, queries_per_record, seed * 1000 + idx)
        responses = [primitive_vector_from_system(system, q, private_data) for q in queries]
        prims = np.stack([r["primitive_vector"].astype(float) for r in responses], axis=0)
        feats.append(np.concatenate([prims.mean(axis=0), prims.std(axis=0)]))
    return np.asarray(feats, dtype=float)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    corr = np.corrcoef(x, y)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0


def _rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = np.argsort(np.argsort(x))
    y_rank = np.argsort(np.argsort(y))
    return _safe_corr(x_rank.astype(float), y_rank.astype(float))


def _safe_ndcg(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    shifted = y_true - float(np.min(y_true))
    if float(np.max(shifted)) == 0.0:
        return 0.0
    try:
        return float(ndcg_score(shifted.reshape(1, -1), y_pred.reshape(1, -1)))
    except Exception:
        return 0.0


def _safe_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(brier_score_loss(y_true, y_prob))
    except Exception:
        return 0.25


def _safe_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return 0.5


def _safe_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, y_prob))
    except Exception:
        return 0.5


def _safe_multiclass_log_loss(y_true: np.ndarray, prob: np.ndarray) -> float:
    try:
        labels = np.arange(prob.shape[1], dtype=int)
        return float(log_loss(y_true, prob, labels=labels))
    except Exception:
        return float("inf")


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    pred = (y_prob >= 0.5).astype(int)
    return {
        "auroc": _safe_auroc(y_true, y_prob),
        "auprc": _safe_auprc(y_true, y_prob),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "brier": _safe_brier(y_true, y_prob),
    }


def _multiclass_metrics(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    pred = prob.argmax(axis=1)
    return {
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "log_loss": _safe_multiclass_log_loss(y_true, prob),
    }


def _regression_metrics(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(mean_squared_error(y_true, pred, squared=False)),
        "pearson": _safe_corr(y_true, pred),
        "spearman": _rank_corr(y_true, pred),
        "ndcg": _safe_ndcg(y_true, pred),
    }


def _fit_binary_utility_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> Dict[str, float]:
    best = None
    for name, clf in [
        ("logreg", LogisticRegression(max_iter=1000, random_state=seed, C=1.0)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed)),
    ]:
        try:
            clf.fit(X_train, y_train)
            val_prob = clf.predict_proba(X_val)[:, 1]
            val_score = _safe_auroc(y_val, val_prob)
            test_prob = clf.predict_proba(X_test)[:, 1]
            metrics = _binary_metrics(y_test, test_prob)
            metrics["selected_model"] = name
            if best is None or val_score > best["selection_score"]:
                best = {"selection_score": val_score, **metrics}
        except Exception:
            continue
    if best is None:
        return {
            "auroc": 0.5,
            "auprc": 0.5,
            "f1": 0.0,
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "brier": 0.25,
            "selected_model": "none",
        }
    best.pop("selection_score", None)
    return best


def _fit_multiclass_utility_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> Dict[str, float]:
    best = None
    n_classes = int(max(np.max(y_train), np.max(y_val), np.max(y_test)) + 1)
    for name, clf in [
        ("logreg", LogisticRegression(max_iter=1000, random_state=seed)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed)),
    ]:
        try:
            clf.fit(X_train, y_train)
            val_prob = clf.predict_proba(X_val)
            if val_prob.shape[1] != n_classes:
                val_prob_full = np.full((len(y_val), n_classes), 1e-8, dtype=float)
                val_prob_full[:, clf.classes_.astype(int)] = val_prob
                val_prob = val_prob_full
            val_pred = val_prob.argmax(axis=1)
            val_score = float(f1_score(y_val, val_pred, average="macro", zero_division=0))
            test_prob = clf.predict_proba(X_test)
            if test_prob.shape[1] != n_classes:
                test_prob_full = np.full((len(y_test), n_classes), 1e-8, dtype=float)
                test_prob_full[:, clf.classes_.astype(int)] = test_prob
                test_prob = test_prob_full
            metrics = _multiclass_metrics(y_test, test_prob)
            metrics["selected_model"] = name
            if best is None or val_score > best["selection_score"]:
                best = {"selection_score": val_score, **metrics}
        except Exception:
            continue
    if best is None:
        return {
            "macro_f1": 0.0,
            "accuracy": 0.0,
            "log_loss": float("inf"),
            "selected_model": "none",
        }
    best.pop("selection_score", None)
    return best


def _fit_regression_utility_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> Dict[str, float]:
    best = None
    for name, reg in [
        ("linear", LinearRegression()),
        ("mlp", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed)),
    ]:
        try:
            reg.fit(X_train, y_train)
            val_pred = reg.predict(X_val)
            val_rmse = float(mean_squared_error(y_val, val_pred, squared=False))
            test_pred = reg.predict(X_test)
            metrics = _regression_metrics(y_test, test_pred)
            metrics["selected_model"] = name
            if best is None or val_rmse < best["selection_score"]:
                best = {"selection_score": val_rmse, **metrics}
        except Exception:
            continue
    if best is None:
        return {
            "rmse": float("inf"),
            "pearson": 0.0,
            "spearman": 0.0,
            "ndcg": 0.0,
            "selected_model": "none",
        }
    best.pop("selection_score", None)
    return best


def _average_metric_dicts(metric_dicts: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not metric_dicts:
        return {}
    out: Dict[str, float] = {}
    numeric_keys = {
        key
        for metrics in metric_dicts
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and np.isfinite(float(value))
    }
    for key in sorted(numeric_keys):
        vals = [float(metrics[key]) for metrics in metric_dicts if key in metrics and np.isfinite(float(metrics[key]))]
        if vals:
            out[key] = float(np.mean(vals))
    return out


def evaluate_proxy_binary_end_task_utility(
    system: Any,
    train_data: PrivateDataset,
    test_data: PrivateDataset,
    seed: int,
    queries_per_record: int = 8,
) -> Dict[str, float]:
    X_train = _downstream_features(system, train_data, train_data, seed=seed, queries_per_record=queries_per_record)
    X_test = _downstream_features(system, train_data, test_data, seed=seed + 1, queries_per_record=queries_per_record)
    y_train = train_data.labels.cpu().numpy().astype(int)
    y_test = test_data.labels.cpu().numpy().astype(int)
    return _fit_binary_utility_model(X_train, y_train, X_train, y_train, X_test, y_test, seed)


def evaluate_proxy_regression_end_task_utility(
    system: Any,
    train_data: PrivateDataset,
    test_data: PrivateDataset,
    seed: int,
    queries_per_record: int = 8,
) -> Dict[str, float]:
    if not hasattr(train_data, "responses") or not hasattr(test_data, "responses"):
        raise TypeError("Regression utility requires dataset.responses.")
    X_train = _downstream_features(system, train_data, train_data, seed=seed, queries_per_record=queries_per_record)
    X_test = _downstream_features(system, train_data, test_data, seed=seed + 1, queries_per_record=queries_per_record)
    y_train = train_data.responses.cpu().numpy().astype(float)
    y_test = test_data.responses.cpu().numpy().astype(float)
    return _fit_regression_utility_model(X_train, y_train, X_train, y_train, X_test, y_test, seed)


def evaluate_end_task_utility(
    task_type: str,
    system: Any,
    train_data: PrivateDataset,
    test_data: PrivateDataset,
    seed: int,
    queries_per_record: int = 8,
) -> Dict[str, float]:
    """Deprecated proxy helper kept for internal/debug use only."""
    if task_type == "regression_ranking":
        return evaluate_proxy_regression_end_task_utility(
            system,
            train_data,
            test_data,
            seed=seed,
            queries_per_record=queries_per_record,
        )
    return evaluate_proxy_binary_end_task_utility(
        system,
        train_data,
        test_data,
        seed=seed,
        queries_per_record=queries_per_record,
    )


def _build_subproblem_system(
    *,
    system_spec: SystemSpec,
    subproblem,
    mode: AttackMode,
    seed: int,
    ckpt_dir,
    device,
    train_missing: bool,
):
    if system_spec.family in {"digit", "digit_stability_gate"}:
        ensure_digit_checkpoints(
            dataset_name=subproblem.checkpoint_dataset_name,
            train_data=subproblem.train,
            val_data=subproblem.val,
            mode=mode,
            digit_variant=system_spec.digit_variant,
            seeds=[seed],
            ckpt_dir=ckpt_dir,
            device=device,
            force_train=train_missing,
        )
    system, _, config_extra = build_system(
        system_spec=system_spec,
        private_data=subproblem.train,
        mode=mode,
        seed=seed,
        ckpt_dir=ckpt_dir,
        dataset_name=subproblem.checkpoint_dataset_name,
        device=device,
    )
    return system, config_extra


def evaluate_fair_utility(
    *,
    bundle: FairUtilityBundle,
    system_spec: SystemSpec,
    mode: AttackMode,
    seed: int,
    ckpt_dir,
    device,
    train_missing: bool,
    internal_queries: int = 1000,
    queries_per_record: int = 8,
) -> Dict[str, float]:
    train_blocks: List[np.ndarray] = []
    val_blocks: List[np.ndarray] = []
    test_blocks: List[np.ndarray] = []
    internal_metrics: List[Dict[str, float]] = []
    config_extra = None
    subproblem_keys: List[str] = []

    for offset, subproblem in enumerate(bundle.subproblems):
        system, subproblem_config = _build_subproblem_system(
            system_spec=system_spec,
            subproblem=subproblem,
            mode=mode,
            seed=seed,
            ckpt_dir=ckpt_dir,
            device=device,
            train_missing=train_missing,
        )
        if config_extra is None:
            config_extra = dict(subproblem_config)
        subproblem_keys.append(subproblem.key)
        train_blocks.append(
            _downstream_features(system, subproblem.train, subproblem.train, seed=seed, queries_per_record=queries_per_record)
        )
        val_blocks.append(
            _downstream_features(
                system,
                subproblem.train,
                subproblem.val,
                seed=seed + 17,
                queries_per_record=queries_per_record,
            )
        )
        test_blocks.append(
            _downstream_features(
                system,
                subproblem.train,
                subproblem.test,
                seed=seed + 29,
                queries_per_record=queries_per_record,
            )
        )
        if internal_queries > 0:
            internal_metrics.append(
                evaluate_internal_primitive_utility(
                    system,
                    subproblem.train,
                    seed=seed + offset,
                    num_queries=internal_queries,
                )
            )

    X_train = np.concatenate(train_blocks, axis=1)
    X_val = np.concatenate(val_blocks, axis=1)
    X_test = np.concatenate(test_blocks, axis=1)
    y_train = np.asarray(bundle.y_train)
    y_val = np.asarray(bundle.y_val)
    y_test = np.asarray(bundle.y_test)

    if bundle.protocol.task_type == "binary_classification":
        task_metrics = _fit_binary_utility_model(X_train, y_train, X_val, y_val, X_test, y_test, seed)
    elif bundle.protocol.task_type == "multiclass_classification":
        task_metrics = _fit_multiclass_utility_model(X_train, y_train, X_val, y_val, X_test, y_test, seed)
    elif bundle.protocol.task_type == "regression_ranking":
        task_metrics = _fit_regression_utility_model(X_train, y_train, X_val, y_val, X_test, y_test, seed)
    else:
        raise ValueError(f"Unsupported fair utility task type: {bundle.protocol.task_type}")

    result: Dict[str, float] = {}
    result.update(_average_metric_dicts(internal_metrics))
    result.update(task_metrics)
    result["utility_feature_dim"] = float(X_train.shape[1])
    result["utility_subproblem_count"] = float(len(bundle.subproblems))
    result["queries_per_record"] = float(queries_per_record)
    result["utility_headline_safe"] = float(1.0 if bundle.protocol.headline_safe else 0.0)
    result["utility_target"] = bundle.protocol.target_name
    result["utility_task_type"] = bundle.protocol.task_type
    result["utility_primary_metric"] = bundle.protocol.primary_metric
    result["utility_role"] = bundle.protocol.published_role
    result["utility_subproblems"] = ",".join(subproblem_keys)
    result["_config_extra"] = {
        **(config_extra or {}),
        "utility_target": bundle.protocol.target_name,
        "utility_task_type": bundle.protocol.task_type,
        "utility_role": bundle.protocol.published_role,
        "utility_subproblem_strategy": bundle.protocol.subproblem_strategy,
        "utility_subproblems": subproblem_keys,
        "queries_per_record": queries_per_record,
    }
    return result


def metric_direction(metric_name: str) -> str:
    lower_is_better = {"rmse", "log_loss", "brier"}
    return "lower" if metric_name in lower_is_better else "higher"


def utility_comparison_fields(
    protocol: UtilityProtocol,
    systems_summary: Dict[str, Dict[str, float]],
) -> Dict[str, object]:
    primary = protocol.primary_metric
    direction = metric_direction(primary)
    raw_summary = systems_summary.get("raw", {})
    raw_value = raw_summary.get(f"{primary}_mean")
    dp_candidates = {
        key: value.get(f"{primary}_mean")
        for key, value in systems_summary.items()
        if key.startswith("dp_laplace_")
    }
    finite_dp = {key: val for key, val in dp_candidates.items() if isinstance(val, (int, float)) and np.isfinite(float(val))}
    if direction == "higher":
        best_dp_system = max(finite_dp, key=finite_dp.get) if finite_dp else None
    else:
        best_dp_system = min(finite_dp, key=finite_dp.get) if finite_dp else None
    best_dp_value = finite_dp.get(best_dp_system) if best_dp_system is not None else None

    per_system: Dict[str, Dict[str, float | str | None]] = {}
    for system_key, summary in systems_summary.items():
        value = summary.get(f"{primary}_mean")
        retained = None
        if isinstance(raw_value, (int, float)) and np.isfinite(float(raw_value)) and isinstance(value, (int, float)) and np.isfinite(float(value)):
            if direction == "higher":
                retained = float(value) / float(raw_value) if float(raw_value) != 0.0 else None
            else:
                retained = float(raw_value) / float(value) if float(value) != 0.0 else None
        delta_vs_best_dp = None
        if isinstance(best_dp_value, (int, float)) and np.isfinite(float(best_dp_value)) and isinstance(value, (int, float)) and np.isfinite(float(value)):
            delta_vs_best_dp = float(value) - float(best_dp_value) if direction == "higher" else float(best_dp_value) - float(value)

        delta_vs_dp: Dict[str, float] = {}
        for dp_key, dp_value in finite_dp.items():
            if isinstance(value, (int, float)) and np.isfinite(float(value)):
                delta_vs_dp[dp_key] = float(value) - float(dp_value) if direction == "higher" else float(dp_value) - float(value)
        per_system[system_key] = {
            "primary_metric": primary,
            "primary_metric_mean": value,
            "retained_utility_vs_raw": retained,
            "delta_vs_best_dp": delta_vs_best_dp,
            "best_dp_system": best_dp_system,
            "delta_vs_dp": delta_vs_dp,
        }
    return {
        "primary_metric": primary,
        "metric_direction": direction,
        "best_dp_system": best_dp_system,
        "best_dp_value": best_dp_value,
        "per_system": per_system,
    }


def render_utility_summary_markdown(summary: Dict[str, object]) -> str:
    lines = [
        "# Fair Utility Frontier Summary",
        "",
        "| Dataset | Mode | Role | Primary metric | System | Mean | Retained vs raw | Delta vs best DP |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for dataset_name, dataset_payload in summary.items():
        if not isinstance(dataset_payload, dict):
            continue
        for mode_name, mode_payload in dataset_payload.items():
            if not isinstance(mode_payload, dict):
                continue
            protocol = mode_payload.get("protocol", {})
            systems = mode_payload.get("systems", {})
            comparisons = mode_payload.get("comparisons", {})
            role = protocol.get("published_role", "headline")
            primary_metric = comparisons.get("primary_metric", protocol.get("primary_metrics", ["n/a"])[0] if protocol.get("primary_metrics") else "n/a")
            per_system = comparisons.get("per_system", {})
            for system_key in sorted(systems):
                comparison = per_system.get(system_key, {})
                mean_value = systems[system_key].get(f"{primary_metric}_mean")
                retained = comparison.get("retained_utility_vs_raw")
                delta = comparison.get("delta_vs_best_dp")
                lines.append(
                    f"| {dataset_name} | {mode_name} | {role} | {primary_metric} | {system_key} | "
                    f"{_fmt(mean_value)} | {_fmt(retained)} | {_fmt(delta)} |"
                )
    lines.append("")
    lines.append("Notes:")
    lines.append("- `Retained vs raw` is direction-aware: values closer to `1.0` mean the system preserves raw utility.")
    lines.append("- `Delta vs best DP` is direction-aware: positive means the system is better than the best DP baseline for that dataset/mode.")
    return "\n".join(lines)


def _fmt(value) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, float)):
        if not np.isfinite(float(value)):
            return "NA"
        return f"{float(value):.3f}"
    return str(value)
