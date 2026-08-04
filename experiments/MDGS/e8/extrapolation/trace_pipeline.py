"""Trace collection and baseline evaluation for Extrapolation."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from .data.adult_loader import load_adult_data
from .data.ground_truth import (
    ATTENTION_PATTERN_LABELS,
    OUTCOME_LABELS,
    TEMPLATE_RESPONSES,
    TRAJECTORY_SHAPE_LABELS,
    CONFIDENCE_LABELS,
)
from .data.private_store import PrivateAdultDataset
from .data.query_generator import QueryGenerator
from .data.trace_dataset import (
    EVIDENCE_TARGET_NAMES,
    load_trace_records,
    trace_inputs_from_record,
    write_trace_records,
)
from .models.trace_probe import QueryTraceProbe

PROBE_ANSWER_LABELS = ["ABOVE_BASELINE", "BELOW_BASELINE", "NEAR_BASELINE"]
TRACE_LABEL_VERSION = "v5"


def set_global_determinism(seed: int) -> None:
    """Seed Python, NumPy, and Torch for reproducible trace collection/training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


class _ProbeDataset(Dataset):
    """Simple query-answer dataset for the trace probe."""

    def __init__(self, examples: List[Dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        example = self.examples[idx]
        return (
            torch.tensor(example["query_fields"], dtype=torch.long),
            torch.tensor(example["gold_answer"], dtype=torch.long),
        )


def _probe_collate_fn(batch):
    queries, labels = zip(*batch)
    return torch.stack(queries), torch.stack(labels)


def compute_probe_answer(
    income_rate: float,
    overall_rate: float,
    income_margin: float,
) -> int:
    """Three-class answer label derived from subgroup income rate."""
    if income_rate > overall_rate + income_margin:
        return 0
    if income_rate < overall_rate - income_margin:
        return 1
    return 2


def _trajectory_summary(values: Iterable[float]) -> Dict[str, float]:
    values = [float(v) for v in values]
    return {
        "mean_entropy": float(np.mean(values)),
        "slope": float(values[-1] - values[0]),
        "std_entropy": float(np.std(values)),
        "entropy_range": float(max(values) - min(values)),
    }


def _safe_std(value: float) -> float:
    return value if abs(value) > 1e-6 else 1e-6


def _z_score(value: float, mean: float, std: float) -> float:
    return (float(value) - float(mean)) / _safe_std(float(std))


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _support_ratio_to_unit_interval(value: float) -> float:
    value = max(float(value), 0.0)
    return _clip01(np.log1p(value * 100.0) / np.log1p(100.0))


def _label_attention_pattern(
    mean_entropy: float,
    max_attention: float,
    metadata: Dict,
    threshold: float | None = None,
) -> int:
    threshold = float(metadata["pattern_z_threshold"] if threshold is None else threshold)
    mean_z = _z_score(
        mean_entropy,
        metadata["mean_entropy_mean"],
        metadata["mean_entropy_std"],
    )
    attention_z = _z_score(
        max_attention,
        metadata["max_attention_mean"],
        metadata["max_attention_std"],
    )

    if mean_z <= -threshold and attention_z >= threshold:
        return ATTENTION_PATTERN_LABELS.index("FOCUSED")
    if mean_z >= threshold and attention_z <= -threshold:
        return ATTENTION_PATTERN_LABELS.index("DIFFUSE")
    return ATTENTION_PATTERN_LABELS.index("MIXED")


def _pattern_distribution(
    records: List[Dict],
    metadata: Dict,
    threshold: float,
) -> Dict[str, int]:
    counts = Counter()
    for record in records:
        pattern = _label_attention_pattern(
            record["trace_summary"]["mean_entropy"],
            record["max_attention_mass"],
            metadata,
            threshold=threshold,
        )
        counts[ATTENTION_PATTERN_LABELS[pattern]] += 1

    for label in ATTENTION_PATTERN_LABELS:
        counts.setdefault(label, 0)
    return dict(counts)


def _select_pattern_threshold(
    train_records: List[Dict],
    metadata: Dict,
    config,
) -> tuple[float, Dict[str, int], Dict[str, Dict[str, int]], str]:
    if config.trace_pattern_threshold_override is not None:
        override = float(config.trace_pattern_threshold_override)
        distribution = _pattern_distribution(train_records, metadata, override)
        return (
            override,
            distribution,
            {f"{override:.2f}": distribution},
            "override",
        )

    selected_threshold = float(config.trace_pattern_z_candidates[-1])
    selected_distribution = _pattern_distribution(train_records, metadata, selected_threshold)
    search_history: Dict[str, Dict[str, int]] = {}
    total = max(len(train_records), 1)

    for candidate in config.trace_pattern_z_candidates:
        candidate = float(candidate)
        distribution = _pattern_distribution(train_records, metadata, candidate)
        search_history[f"{candidate:.2f}"] = distribution
        focused_share = distribution["FOCUSED"] / total
        diffuse_share = distribution["DIFFUSE"] / total
        selected_threshold = candidate
        selected_distribution = distribution
        if (
            focused_share >= config.trace_pattern_min_share
            and diffuse_share >= config.trace_pattern_min_share
        ):
            break

    return selected_threshold, selected_distribution, search_history, "auto"


def _select_agreement_thresholds(agreements: np.ndarray) -> tuple[float, float, List[float]]:
    values = sorted({float(value) for value in agreements.tolist()})
    if not values:
        return 1.0, 1.0, [1.0]
    if len(values) == 1:
        return values[0], values[0], values
    if len(values) == 2:
        return values[0], values[1], values
    return values[-2], values[-1], values


def _label_confidence(agreement: float, prob_margin: float, metadata: Dict) -> int:
    if (
        float(agreement) >= float(metadata["agreement_high"])
        and float(prob_margin) >= float(metadata["margin_high"])
    ):
        return CONFIDENCE_LABELS.index("HIGH")
    if (
        float(agreement) >= float(metadata["agreement_medium"])
        and float(prob_margin) >= float(metadata["margin_low"])
    ):
        return CONFIDENCE_LABELS.index("MEDIUM")
    return CONFIDENCE_LABELS.index("LOW")


def _query_key(query_fields: List[int] | Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(int(value) for value in query_fields)


def _probe_example_from_query(query, private_data: PrivateAdultDataset, config) -> Dict | None:
    stats = private_data.get_subgroup_stats(query)
    if stats["n"] < config.min_group_size:
        return None

    income_rate = float(stats["income_rate"])
    overall_rate = float(private_data.overall_income_rate)
    return {
        "query_fields": query.tolist(),
        "gold_answer": compute_probe_answer(
            income_rate=income_rate,
            overall_rate=overall_rate,
            income_margin=config.income_margin,
        ),
        "stats": {
            "n": int(stats["n"]),
            "income_rate": income_rate,
            "support_ratio": float(stats["support_ratio"]),
            "variance": float(stats["variance"]),
            "abs_margin": abs(income_rate - overall_rate),
        },
    }


def _candidate_probe_examples(
    private_data: PrivateAdultDataset,
    config,
    num_examples: int,
    seed: int,
    exclude_queries: set[Tuple[int, ...]] | None = None,
) -> List[Dict]:
    generator = QueryGenerator(private_data=private_data, min_group_size=config.min_group_size)
    seen = set(exclude_queries or set())
    examples: List[Dict] = []
    attempt = 0

    while len(examples) < num_examples and attempt < 12:
        remaining = num_examples - len(examples)
        num_candidates = max(remaining * 2, config.probe_pool_multiplier * remaining)
        queries = generator.generate_queries(num_candidates, seed=seed + attempt)
        for query in queries:
            key = _query_key(query.tolist())
            if key in seen:
                continue
            example = _probe_example_from_query(query, private_data=private_data, config=config)
            if example is None:
                continue
            seen.add(key)
            examples.append(example)
            if len(examples) >= num_examples:
                break
        attempt += 1

    return examples


def _rank_scores(values: List[float], higher_is_harder: bool) -> np.ndarray:
    if not values:
        return np.zeros(0, dtype=np.float32)

    values_arr = np.asarray(values, dtype=np.float32)
    order = np.argsort(values_arr, kind="mergesort")
    if higher_is_harder:
        order = order[::-1]

    scores = np.zeros(len(values), dtype=np.float32)
    if len(values) == 1:
        scores[order[0]] = 1.0
        return scores

    scores[order] = np.linspace(1.0, 0.0, len(values), endpoint=True, dtype=np.float32)
    return scores


def _score_hard_train_candidates(records: List[Dict]) -> List[Dict]:
    if not records:
        return []

    agreement_scores = _rank_scores([record["agreement"] for record in records], higher_is_harder=False)
    margin_scores = _rank_scores([record["prob_margin"] for record in records], higher_is_harder=False)
    variance_scores = _rank_scores(
        [record["stats"]["variance"] for record in records],
        higher_is_harder=True,
    )
    boundary_scores = _rank_scores(
        [record["stats"]["abs_margin"] for record in records],
        higher_is_harder=False,
    )

    scored: List[Dict] = []
    for idx, record in enumerate(records):
        incorrect_bonus = 4.0 if not record["is_correct"] else 0.0
        hardness_score = (
            incorrect_bonus
            + float(agreement_scores[idx])
            + float(margin_scores[idx])
            + float(variance_scores[idx])
            + float(boundary_scores[idx])
        )
        scored.append({**record, "hardness_score": float(hardness_score)})

    scored.sort(
        key=lambda record: (
            record["hardness_score"],
            int(not record["is_correct"]),
            record["stats"]["variance"],
            -record["agreement"],
        ),
        reverse=True,
    )
    return scored


def _merge_hard_train_records(
    base_records: List[Dict],
    hard_candidates: List[Dict],
    config,
) -> List[Dict]:
    max_hard_cases = int(
        round(
            len(base_records)
            * config.trace_train_hard_fraction
            / max(1.0 - config.trace_train_hard_fraction, 1e-6)
        )
    )
    failures = sum(int(record["is_correct"] is False) for record in base_records)
    selected: List[Dict] = []

    for record in hard_candidates:
        current_total = len(base_records) + len(selected)
        current_failures = failures + sum(int(item["is_correct"] is False) for item in selected)
        current_failure_share = current_failures / max(current_total, 1)
        if len(selected) >= max_hard_cases or current_failure_share >= config.trace_failure_target_share:
            break
        selected.append({**record, "split": "train", "is_hard_mined": True})

    return [{**record, "is_hard_mined": False} for record in base_records] + selected


def _run_trace_labeler_checks(records: List[Dict], metadata: Dict, config) -> None:
    failure_idx = OUTCOME_LABELS.index("FAILURE_LIKELY")
    uncertain_idx = OUTCOME_LABELS.index("UNCERTAIN")
    success_idx = OUTCOME_LABELS.index("SUCCESS_LIKELY")
    failure_threshold = float(metadata["failure_max_correctness_prob"])
    success_threshold = float(metadata["success_min_correctness_prob"])

    for record in records:
        probability = float(record["correctness_probability"])
        if probability < failure_threshold:
            assert record["outcome"] == failure_idx
        elif probability >= success_threshold and _supports_success_outcome(record, metadata):
            assert record["outcome"] == success_idx
        else:
            assert record["outcome"] == uncertain_idx

    if metadata.get("pattern_threshold_mode") == "auto":
        selected = float(metadata["pattern_z_threshold"])
        candidate_keys = [f"{float(candidate):.2f}" for candidate in config.trace_pattern_z_candidates]
        selected_idx = min(
            range(len(candidate_keys)),
            key=lambda idx: abs(float(candidate_keys[idx]) - selected),
        )
        for key in candidate_keys[:selected_idx]:
            distribution = metadata["pattern_threshold_search"][key]
            total = max(sum(distribution.values()), 1)
            focused_share = distribution["FOCUSED"] / total
            diffuse_share = distribution["DIFFUSE"] / total
            assert (
                focused_share < config.trace_pattern_min_share
                or diffuse_share < config.trace_pattern_min_share
            )


def _render_response_text(outcome: int, pattern: int) -> str:
    outcome_label = OUTCOME_LABELS[outcome]
    pattern_label = ATTENTION_PATTERN_LABELS[pattern]
    key = f"{outcome_label}_{pattern_label}"
    if key in TEMPLATE_RESPONSES:
        return TEMPLATE_RESPONSES[key][0]

    fallback_key = f"{outcome_label}_MIXED"
    if fallback_key in TEMPLATE_RESPONSES:
        return TEMPLATE_RESPONSES[fallback_key][0]

    return "attention trajectory remains mixed and the outcome is uncertain."


def _balanced_probe_examples(
    private_data: PrivateAdultDataset,
    config,
    num_examples: int,
    seed: int,
) -> List[Dict]:
    """Generate a balanced query-answer dataset for the probe."""
    generator = QueryGenerator(private_data=private_data, min_group_size=config.min_group_size)
    rng = random.Random(seed)
    target_per_class = max(1, num_examples // len(PROBE_ANSWER_LABELS))
    buckets = {i: [] for i in range(len(PROBE_ANSWER_LABELS))}

    attempt = 0
    while min(len(bucket) for bucket in buckets.values()) < target_per_class and attempt < 8:
        num_candidates = num_examples * config.probe_pool_multiplier
        queries = generator.generate_queries(num_candidates, seed=seed + attempt)
        for query in queries:
            stats = private_data.get_subgroup_stats(query)
            if stats["n"] < config.min_group_size:
                continue

            answer = compute_probe_answer(
                income_rate=stats["income_rate"],
                overall_rate=private_data.overall_income_rate,
                income_margin=config.income_margin,
            )
            buckets[answer].append(
                {
                    "query_fields": query.tolist(),
                    "gold_answer": answer,
                    "stats": {
                        "n": int(stats["n"]),
                        "income_rate": float(stats["income_rate"]),
                        "support_ratio": float(stats["support_ratio"]),
                        "variance": float(stats["variance"]),
                        "abs_margin": abs(float(stats["income_rate"]) - float(private_data.overall_income_rate)),
                    },
                }
            )
        attempt += 1

    per_class = min(target_per_class, *(len(bucket) for bucket in buckets.values()))
    examples = []
    for answer, bucket in buckets.items():
        rng.shuffle(bucket)
        examples.extend(bucket[:per_class])

    rng.shuffle(examples)
    return examples


def _train_probe_model(
    train_examples: List[Dict],
    val_examples: List[Dict],
    config,
    device: torch.device,
    probe_seed: int,
) -> QueryTraceProbe:
    """Train the attention probe used for trace collection."""
    set_global_determinism(probe_seed)
    train_loader = DataLoader(
        _ProbeDataset(train_examples),
        batch_size=config.probe_batch_size,
        shuffle=True,
        collate_fn=_probe_collate_fn,
        generator=_make_generator(probe_seed),
    )
    val_loader = DataLoader(
        _ProbeDataset(val_examples),
        batch_size=config.probe_batch_size,
        shuffle=False,
        collate_fn=_probe_collate_fn,
    )

    model = QueryTraceProbe(
        field_vocab_sizes=config.field_vocab_sizes,
        num_answer_classes=config.probe_num_answer_classes,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.probe_num_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.probe_learning_rate,
        weight_decay=config.probe_weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.probe_epochs):
        model.train()
        for queries, labels in train_loader:
            queries = queries.to(device)
            labels = labels.to(device)
            out = model(queries)
            loss = criterion(out.logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

        model.eval()
        total_loss = 0.0
        total_items = 0
        with torch.no_grad():
            for queries, labels in val_loader:
                queries = queries.to(device)
                labels = labels.to(device)
                out = model(queries)
                loss = criterion(out.logits, labels)
                total_loss += loss.item() * queries.size(0)
                total_items += queries.size(0)

        val_loss = total_loss / max(total_items, 1)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.probe_patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def _mc_dropout_stats(
    model: QueryTraceProbe,
    query_batch: torch.Tensor,
    config,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run MC dropout to estimate stability and margin."""
    probs_runs = []
    trajectory_runs = []
    max_attention_runs = []
    top2_attention_runs = []
    effective_support_runs = []
    attention_gini_runs = []
    concentration_drift_runs = []

    with torch.no_grad():
        for _ in range(config.probe_mc_passes):
            model.train()
            out = model(query_batch, collect_trace=True)
            probs_runs.append(torch.softmax(out.logits, dim=-1))
            trajectory_runs.append(out.attention_entropy_trajectory)
            max_attention_runs.append(out.max_attention_mass)
            top2_attention_runs.append(out.attention_top2_mass)
            effective_support_runs.append(out.attention_effective_support)
            attention_gini_runs.append(out.attention_gini)
            concentration_drift_runs.append(out.attention_concentration_drift)
        model.eval()

    probs_runs = torch.stack(probs_runs, dim=0)
    avg_probs = probs_runs.mean(dim=0)
    votes = probs_runs.argmax(dim=-1)
    agreement = []
    for i in range(votes.size(1)):
        counts = torch.bincount(votes[:, i], minlength=avg_probs.size(-1))
        agreement.append(counts.max().float() / votes.size(0))
    agreement = torch.stack(agreement, dim=0)

    top2 = torch.topk(avg_probs, k=2, dim=-1).values
    prob_margin = top2[:, 0] - top2[:, 1]
    avg_trajectory = torch.stack(trajectory_runs, dim=0).mean(dim=0)
    avg_max_attention = torch.stack(max_attention_runs, dim=0).mean(dim=0)
    avg_top2_attention = torch.stack(top2_attention_runs, dim=0).mean(dim=0)
    avg_effective_support = torch.stack(effective_support_runs, dim=0).mean(dim=0)
    avg_attention_gini = torch.stack(attention_gini_runs, dim=0).mean(dim=0)
    avg_concentration_drift = torch.stack(concentration_drift_runs, dim=0).mean(dim=0)

    return (
        agreement,
        prob_margin,
        avg_trajectory,
        avg_max_attention,
        avg_top2_attention,
        avg_effective_support,
        avg_attention_gini,
        avg_concentration_drift,
        avg_probs,
    )


def _fit_trace_labeler(train_records: List[Dict], config):
    """Fit cluster- and threshold-based labeling metadata from train traces."""
    trajectories = np.array(
        [record["attention_entropy_trajectory"] for record in train_records],
        dtype=np.float32,
    )
    kmeans = KMeans(
        n_clusters=config.num_trajectory_classes,
        random_state=config.train_loop_seed,
        n_init=20,
    )
    kmeans.fit(trajectories)

    centroids = kmeans.cluster_centers_
    slopes = centroids[:, -1] - centroids[:, 0]
    ranges = centroids.max(axis=1) - centroids.min(axis=1)

    decreasing_idx = int(np.argmin(slopes))
    increasing_idx = int(np.argmax(slopes))
    remaining = [
        idx
        for idx in range(len(centroids))
        if idx not in {decreasing_idx, increasing_idx}
    ]
    volatile_idx = remaining[int(np.argmax(ranges[remaining]))]
    stable_idx = [idx for idx in remaining if idx != volatile_idx][0]

    cluster_to_shape = {
        decreasing_idx: "DECREASING",
        increasing_idx: "INCREASING",
        volatile_idx: "VOLATILE",
        stable_idx: "STABLE",
    }

    mean_entropies = np.array([record["trace_summary"]["mean_entropy"] for record in train_records])
    max_attention = np.array([record["max_attention_mass"] for record in train_records])
    agreements = np.array([record["agreement"] for record in train_records])
    margins = np.array([record["prob_margin"] for record in train_records])
    agreement_medium, agreement_high, agreement_values = _select_agreement_thresholds(agreements)

    metadata = {
        "label_version": TRACE_LABEL_VERSION,
        "cluster_to_shape": {str(idx): label for idx, label in cluster_to_shape.items()},
        "mean_entropy_mean": float(np.mean(mean_entropies)),
        "mean_entropy_std": float(np.std(mean_entropies)),
        "max_attention_mean": float(np.mean(max_attention)),
        "max_attention_std": float(np.std(max_attention)),
        "agreement_values": [float(value) for value in agreement_values],
        "agreement_medium": float(agreement_medium),
        "agreement_high": float(agreement_high),
        "margin_low": float(np.quantile(margins, 0.33)),
        "margin_high": float(np.quantile(margins, 0.67)),
    }
    pattern_z_threshold, attention_pattern_distribution, search_history, threshold_mode = _select_pattern_threshold(
        train_records,
        metadata,
        config,
    )
    metadata["pattern_z_threshold"] = float(pattern_z_threshold)
    metadata["pattern_threshold_mode"] = threshold_mode
    metadata["attention_pattern_distribution"] = attention_pattern_distribution
    metadata["pattern_threshold_search"] = search_history
    return kmeans, metadata


def _fit_correctness_labeler(train_records: List[Dict], config) -> tuple[Any, np.ndarray, Dict]:
    """Fit a small correctness model and return OOF train probabilities."""
    _, x_train_full = _feature_matrices(train_records)
    y_train = np.array([int(bool(record["is_correct"])) for record in train_records], dtype=np.int64)

    unique, counts = np.unique(y_train, return_counts=True)
    min_class_count = int(counts.min()) if counts.size else 0
    if unique.size > 1:
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=int(config.train_loop_seed)),
        )
    else:
        estimator = DummyClassifier(strategy="constant", constant=int(unique[0]) if unique.size else 0)

    cv_folds = int(max(2, min(int(config.trace_correctness_cv_folds), min_class_count))) if unique.size > 1 else 2

    if unique.size > 1 and min_class_count >= 2:
        splitter = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=int(config.train_loop_seed),
        )
        train_probabilities = cross_val_predict(
            estimator,
            x_train_full,
            y_train,
            cv=splitter,
            method="predict_proba",
        )[:, 1]
    else:
        majority_prob = float(y_train.mean()) if y_train.size else 0.0
        train_probabilities = np.full(y_train.shape[0], fill_value=majority_prob, dtype=np.float64)

    estimator.fit(x_train_full, y_train)
    train_auroc = float(roc_auc_score(y_train, train_probabilities)) if unique.size > 1 else float("nan")
    train_brier = float(np.mean((train_probabilities - y_train) ** 2)) if y_train.size else float("nan")
    metadata = {
        "correctness_labeler": "logistic_regression_full_v1",
        "correctness_cv_folds": int(cv_folds),
        "success_min_correctness_prob": float(config.trace_success_correctness_threshold),
        "failure_max_correctness_prob": float(config.trace_failure_correctness_threshold),
        "success_min_support_ratio": float(config.trace_success_min_support_ratio),
        "success_min_group_size": int(config.trace_success_min_group_size),
        "correctness_train_auroc": train_auroc,
        "correctness_train_brier": train_brier,
    }
    return estimator, train_probabilities, metadata


def _predict_correctness_probabilities(estimator, records: List[Dict]) -> np.ndarray:
    """Score records with the fitted correctness model."""
    if not records:
        return np.zeros(0, dtype=np.float64)
    _, features = _feature_matrices(records)
    probabilities = estimator.predict_proba(features)
    if probabilities.shape[1] == 1:
        classes = getattr(estimator, "classes_", None)
        if classes is None and hasattr(estimator, "named_steps"):
            final_step = next(reversed(estimator.named_steps.values()))
            classes = getattr(final_step, "classes_", None)
        positive_prob = 1.0 if classes is not None and int(classes[0]) == 1 else 0.0
        return np.full(features.shape[0], fill_value=positive_prob, dtype=np.float64)
    return probabilities[:, 1]


def _attach_correctness_probabilities(records: List[Dict], probabilities: np.ndarray) -> List[Dict]:
    """Attach calibrated-ish correctness probabilities to records."""
    if len(records) != len(probabilities):
        raise ValueError(f"Expected {len(records)} probabilities, got {len(probabilities)}")
    output = []
    for record, probability in zip(records, probabilities):
        output.append({**record, "correctness_probability": float(probability)})
    return output


def _supports_success_outcome(record: Dict, metadata: Dict) -> bool:
    """Hard safety overrides that must hold before allowing SUCCESS."""
    stats = record.get("stats", {})
    return (
        float(record.get("agreement", 0.0)) >= float(metadata["agreement_high"])
        and float(record.get("prob_margin", 0.0)) >= float(metadata["margin_high"])
        and float(stats.get("support_ratio", 0.0)) >= float(metadata["success_min_support_ratio"])
        and int(stats.get("n", 0)) >= int(metadata["success_min_group_size"])
    )


def _label_outcome_from_correctness(record: Dict, metadata: Dict) -> int:
    """Map correctness probability into SUCCESS / UNCERTAIN / FAILURE buckets."""
    probability = float(record["correctness_probability"])
    if probability < float(metadata["failure_max_correctness_prob"]):
        return OUTCOME_LABELS.index("FAILURE_LIKELY")
    if probability >= float(metadata["success_min_correctness_prob"]) and _supports_success_outcome(record, metadata):
        return OUTCOME_LABELS.index("SUCCESS_LIKELY")
    return OUTCOME_LABELS.index("UNCERTAIN")


def _apply_trace_labels(record: Dict, kmeans: KMeans, metadata: Dict) -> Dict:
    """Attach Extrapolation labels to a collected trace record."""
    trajectory = np.array(record["attention_entropy_trajectory"], dtype=np.float32)
    cluster = int(kmeans.predict(trajectory.reshape(1, -1))[0])
    shape_label = metadata["cluster_to_shape"][str(cluster)]
    trajectory_shape = TRAJECTORY_SHAPE_LABELS.index(shape_label)

    mean_entropy = float(record["trace_summary"]["mean_entropy"])
    max_attention = float(record["max_attention_mass"])
    agreement = float(record["agreement"])
    prob_margin = float(record["prob_margin"])

    attention_pattern = _label_attention_pattern(mean_entropy, max_attention, metadata)
    confidence = _label_confidence(agreement, prob_margin, metadata)
    if "correctness_probability" not in record:
        raise KeyError("correctness_probability is required before applying v4 trace labels")
    outcome = _label_outcome_from_correctness(record, metadata)

    return {
        **record,
        "trajectory_shape": trajectory_shape,
        "attention_pattern": attention_pattern,
        "confidence": confidence,
        "outcome": outcome,
        "trace_inputs": trace_inputs_from_record(record),
        "response_text": _render_response_text(outcome, attention_pattern),
    }


def _evidence_targets_from_record(record: Dict, metadata: Dict) -> Dict[str, float]:
    """Construct cooperative evidence targets from fixed trace statistics."""
    stats = record.get("stats", {})
    mean_entropy = float(record["trace_summary"]["mean_entropy"])
    agreement = float(record.get("agreement", 0.0))
    prob_margin = float(record.get("prob_margin", 0.0))
    max_attention = float(record.get("max_attention_mass", 0.0))
    correctness_probability = float(record.get("correctness_probability", float(record.get("is_correct", False))))
    support_ratio = float(stats.get("support_ratio", 0.0))
    n_norm = min(float(stats.get("n", 0.0)) / 200.0, 1.0)
    variation_ratio = 1.0 - agreement

    margin_high = max(float(metadata.get("margin_high", 1.0)), 1e-6)
    margin_low = float(metadata.get("margin_low", 0.0))
    mean_entropy_mean = float(metadata.get("mean_entropy_mean", 0.0))
    mean_entropy_std = float(metadata.get("mean_entropy_std", 1.0))

    margin_norm = _clip01(prob_margin / margin_high)
    entropy_z = _z_score(mean_entropy, mean_entropy_mean, mean_entropy_std)
    entropy_good = _clip01(1.0 - max(0.0, entropy_z) / 3.0)
    low_margin_flag = 1.0 if prob_margin < margin_low else 0.0
    unstable_flag = 1.0 if agreement < float(metadata.get("agreement_medium", 0.0)) else 0.0

    trajectory_anchor = {
        TRAJECTORY_SHAPE_LABELS.index("DECREASING"): 0.85,
        TRAJECTORY_SHAPE_LABELS.index("STABLE"): 1.00,
        TRAJECTORY_SHAPE_LABELS.index("INCREASING"): 0.55,
        TRAJECTORY_SHAPE_LABELS.index("VOLATILE"): 0.15,
    }.get(int(record["trajectory_shape"]), 0.50)
    pattern_anchor = {
        ATTENTION_PATTERN_LABELS.index("FOCUSED"): 1.00,
        ATTENTION_PATTERN_LABELS.index("MIXED"): 0.55,
        ATTENTION_PATTERN_LABELS.index("DIFFUSE"): 0.10,
    }.get(int(record["attention_pattern"]), 0.50)
    confidence_anchor = {
        CONFIDENCE_LABELS.index("LOW"): 0.15,
        CONFIDENCE_LABELS.index("MEDIUM"): 0.55,
        CONFIDENCE_LABELS.index("HIGH"): 0.90,
    }.get(int(record["confidence"]), 0.50)
    outcome_success_anchor = {
        OUTCOME_LABELS.index("SUCCESS_LIKELY"): 1.00,
        OUTCOME_LABELS.index("UNCERTAIN"): 0.50,
        OUTCOME_LABELS.index("FAILURE_LIKELY"): 0.00,
    }.get(int(record["outcome"]), 0.50)
    outcome_failure_anchor = {
        OUTCOME_LABELS.index("SUCCESS_LIKELY"): 0.00,
        OUTCOME_LABELS.index("UNCERTAIN"): 0.50,
        OUTCOME_LABELS.index("FAILURE_LIKELY"): 1.00,
    }.get(int(record["outcome"]), 0.50)
    outcome_uncertain_anchor = 1.0 if int(record["outcome"]) == OUTCOME_LABELS.index("UNCERTAIN") else 0.0

    stability_target = _clip01(
        0.25 * agreement
        + 0.18 * margin_norm
        + 0.12 * max_attention
        + 0.12 * entropy_good
        + 0.10 * (1.0 - variation_ratio)
        + 0.10 * trajectory_anchor
        + 0.08 * pattern_anchor
        + 0.05 * (1.0 - low_margin_flag)
        + 0.10 * confidence_anchor
    )

    support_target = _clip01(
        0.65 * _support_ratio_to_unit_interval(support_ratio)
        + 0.35 * n_norm
    )

    success_target = _clip01(
        0.50 * correctness_probability
        + 0.20 * stability_target
        + 0.15 * support_target
        + 0.15 * outcome_success_anchor
    )

    failure_target = _clip01(
        0.55 * (1.0 - correctness_probability)
        + 0.20 * (1.0 - stability_target)
        + 0.10 * (1.0 - support_target)
        + 0.15 * outcome_failure_anchor
    )

    ambiguity_target = _clip01(
        0.40 * (1.0 - abs(success_target - failure_target))
        + 0.25 * (1.0 - max(success_target, failure_target))
        + 0.10 * (1.0 - stability_target)
        + 0.10 * (1.0 - support_target)
        + 0.15 * outcome_uncertain_anchor
    )

    targets = {
        "stability": stability_target,
        "support": support_target,
        "success": success_target,
        "failure": failure_target,
        "ambiguity": ambiguity_target,
    }
    return {name: _clip01(targets[name]) for name in EVIDENCE_TARGET_NAMES}


def _collect_trace_records(
    examples: List[Dict],
    split_name: str,
    model: QueryTraceProbe,
    config,
    device: torch.device,
    is_hard_mined: bool = False,
) -> List[Dict]:
    """Collect real attention traces and prediction metadata."""
    loader = DataLoader(
        _ProbeDataset(examples),
        batch_size=config.probe_batch_size,
        shuffle=False,
        collate_fn=_probe_collate_fn,
    )
    records = []
    offset = 0

    model.eval()
    with torch.no_grad():
        for query_batch, gold_labels in loader:
            query_batch = query_batch.to(device)
            gold_labels = gold_labels.to(device)

            deterministic = model(query_batch, collect_trace=True)
            (
                agreement,
                prob_margin,
                avg_trajectory,
                avg_max_attention,
                avg_top2_attention,
                avg_effective_support,
                avg_attention_gini,
                avg_concentration_drift,
                avg_probs,
            ) = _mc_dropout_stats(
                model,
                query_batch,
                config,
            )
            pred_answer = avg_probs.argmax(dim=-1)

            for i in range(query_batch.size(0)):
                trajectory = avg_trajectory[i].cpu().tolist()
                records.append(
                    {
                        "example_id": f"{split_name}_{offset + i:05d}",
                        "split": split_name,
                        "query_fields": query_batch[i].cpu().tolist(),
                        "gold_answer": int(gold_labels[i].item()),
                        "pred_answer": int(pred_answer[i].item()),
                        "prediction_probs": [float(v) for v in avg_probs[i].cpu().tolist()],
                        "agreement": float(agreement[i].item()),
                        "prob_margin": float(prob_margin[i].item()),
                        "max_attention_mass": float(avg_max_attention[i].item()),
                        "attention_top2_mass": float(avg_top2_attention[i].item()),
                        "attention_effective_support": float(avg_effective_support[i].item()),
                        "attention_gini": float(avg_attention_gini[i].item()),
                        "attention_concentration_drift": float(avg_concentration_drift[i].item()),
                        "attention_entropy_trajectory": [float(v) for v in trajectory],
                        "trace_summary": _trajectory_summary(trajectory),
                        "is_correct": bool(pred_answer[i].item() == gold_labels[i].item()),
                        "stats": dict(examples[offset + i]["stats"]),
                        "is_hard_mined": bool(is_hard_mined),
                    }
                )
            offset += query_batch.size(0)

    return records


def _feature_matrices(records: List[Dict]) -> tuple[np.ndarray, np.ndarray]:
    """Prepare summary/full feature matrices from raw trace records."""
    summary = np.array(
        [
            [
                record["trace_summary"]["mean_entropy"],
                record["trace_summary"]["slope"],
                record["trace_summary"]["std_entropy"],
                record["trace_summary"]["entropy_range"],
                record["max_attention_mass"],
                record["agreement"],
                record["prob_margin"],
            ]
            for record in records
        ],
        dtype=np.float32,
    )
    full = np.array(
        [
            trace_inputs_from_record(record)
            + [
                record["trace_summary"]["mean_entropy"],
                record["trace_summary"]["slope"],
                record["trace_summary"]["std_entropy"],
                record["trace_summary"]["entropy_range"],
            ]
            for record in records
        ],
        dtype=np.float32,
    )
    return summary, full


def _feature_arrays(records: List[Dict]):
    """Prepare arrays for simple baselines after primitive labels have been attached."""
    summary, full = _feature_matrices(records)
    targets = {
        "trajectory_shape": np.array([record["trajectory_shape"] for record in records]),
        "attention_pattern": np.array([record["attention_pattern"] for record in records]),
        "confidence": np.array([record["confidence"] for record in records]),
        "outcome": np.array([record["outcome"] for record in records]),
    }
    return summary, full, targets


def _corpus_signature(config, probe_seed: int, hard_mining_seed: int) -> Dict:
    return {
        "label_version": TRACE_LABEL_VERSION,
        "data_split_seed": int(config.data_split_seed),
        "train_loop_seed": int(config.train_loop_seed),
        "probe_seed": int(probe_seed),
        "hard_mining_seed": int(hard_mining_seed),
        "probe_train_examples": int(config.probe_train_examples),
        "probe_val_examples": int(config.probe_val_examples),
        "probe_test_examples": int(config.probe_test_examples),
        "probe_pool_multiplier": int(config.probe_pool_multiplier),
        "probe_mc_passes": int(config.probe_mc_passes),
        "trace_hard_case_pool_multiplier": int(config.trace_hard_case_pool_multiplier),
        "trace_train_hard_fraction": float(config.trace_train_hard_fraction),
        "trace_failure_target_share": float(config.trace_failure_target_share),
        "trace_success_correctness_threshold": float(config.trace_success_correctness_threshold),
        "trace_failure_correctness_threshold": float(config.trace_failure_correctness_threshold),
        "trace_success_min_support_ratio": float(config.trace_success_min_support_ratio),
        "trace_success_min_group_size": int(config.trace_success_min_group_size),
        "trace_evidence_target_version": str(config.trace_evidence_target_version),
        "trace_pattern_threshold_override": (
            None
            if config.trace_pattern_threshold_override is None
            else float(config.trace_pattern_threshold_override)
        ),
    }


def _compute_corpus_fingerprint(root_dir: Path, signature: Dict) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(signature, sort_keys=True).encode("utf-8"))
    for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
        digest.update((root_dir / name).read_bytes())
    return digest.hexdigest()[:16]


def _corpus_rebuild_reason(metadata: Dict, config) -> str | None:
    requested_mode = "auto" if config.trace_pattern_threshold_override is None else "override"
    if metadata.get("label_version") != TRACE_LABEL_VERSION:
        return "label_version"
    if str(metadata.get("evidence_target_version", "")) != str(config.trace_evidence_target_version):
        return "evidence_target_version"
    if metadata.get("correctness_labeler") != "logistic_regression_full_v1":
        return "correctness_labeler"
    if metadata.get("pattern_threshold_mode") != requested_mode:
        return "pattern_threshold_mode"
    if config.trace_pattern_threshold_override is not None and abs(
        float(metadata.get("pattern_z_threshold", -1.0)) - float(config.trace_pattern_threshold_override)
    ) > 1e-8:
        return "pattern_z_threshold"
    if abs(
        float(metadata.get("success_min_correctness_prob", -1.0)) - float(config.trace_success_correctness_threshold)
    ) > 1e-8:
        return "success_min_correctness_prob"
    if abs(
        float(metadata.get("failure_max_correctness_prob", -1.0)) - float(config.trace_failure_correctness_threshold)
    ) > 1e-8:
        return "failure_max_correctness_prob"
    if abs(
        float(metadata.get("success_min_support_ratio", -1.0)) - float(config.trace_success_min_support_ratio)
    ) > 1e-8:
        return "success_min_support_ratio"
    if int(metadata.get("success_min_group_size", -1)) != int(config.trace_success_min_group_size):
        return "success_min_group_size"
    if not metadata.get("corpus_fingerprint"):
        return "corpus_fingerprint"
    return None


def evaluate_trace_baselines(train_records: List[Dict], test_records: List[Dict], config=None) -> Dict[str, Dict]:
    """Evaluate simple baselines against the collected trace labels."""
    if config is None:
        raise ValueError("config is required for baseline evaluation")

    x_train_summary, x_train_full, y_train = _feature_arrays(train_records)
    x_test_summary, x_test_full, y_test = _feature_arrays(test_records)

    results: Dict[str, Dict] = {}

    train_entropy = x_train_summary[:, 0]
    class_centroids = {
        cls: train_entropy[y_train["outcome"] == cls].mean()
        for cls in sorted(np.unique(y_train["outcome"]))
    }
    threshold_preds = [
        min(class_centroids, key=lambda cls: abs(value - class_centroids[cls]))
        for value in x_test_summary[:, 0]
    ]

    results["entropy_threshold"] = {
        "outcome_accuracy": float(accuracy_score(y_test["outcome"], threshold_preds)),
        "outcome_macro_f1": float(f1_score(y_test["outcome"], threshold_preds, average="macro", zero_division=0)),
    }

    logistic = {}
    mlp = {}
    for target_name in ["trajectory_shape", "attention_pattern", "confidence", "outcome"]:
        lr_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=42),
        )
        lr_model.fit(x_train_summary, y_train[target_name])
        lr_pred = lr_model.predict(x_test_summary)
        logistic[target_name] = {
            "accuracy": float(accuracy_score(y_test[target_name], lr_pred)),
            "macro_f1": float(f1_score(y_test[target_name], lr_pred, average="macro", zero_division=0)),
        }

        mlp_model = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=config.baseline_mlp_max_iter,
                early_stopping=config.baseline_mlp_early_stopping,
                random_state=42,
            ),
        )
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            mlp_model.fit(x_train_full, y_train[target_name])
        mlp_pred = mlp_model.predict(x_test_full)
        mlp_inner = mlp_model.named_steps["mlpclassifier"]
        converged = not any(
            issubclass(warning.category, ConvergenceWarning)
            for warning in caught_warnings
        )
        mlp[target_name] = {
            "accuracy": float(accuracy_score(y_test[target_name], mlp_pred)),
            "macro_f1": float(f1_score(y_test[target_name], mlp_pred, average="macro", zero_division=0)),
            "converged": bool(converged),
            "n_iter": int(mlp_inner.n_iter_),
        }

    results["logistic_regression"] = logistic
    results["mlp"] = mlp
    results["mlp_converged"] = bool(mlp["outcome"]["converged"])
    return results


def build_trace_corpus(config, root_dir: str | Path, device: torch.device) -> Dict:
    """Collect traces, label them, and persist the train/val/test splits."""
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    set_global_determinism(config.train_loop_seed)
    probe_seed = config.train_loop_seed + 17
    hard_mining_seed = config.train_loop_seed + 100

    splits = load_adult_data(
        cache_dir=str(root_dir.parent / "data_cache"),
        split_seed=config.data_split_seed,
    )
    private_train = PrivateAdultDataset(splits.train, splits.categorical_maps)
    private_val = PrivateAdultDataset(splits.val, splits.categorical_maps)
    private_test = PrivateAdultDataset(splits.test, splits.categorical_maps)

    train_examples = _balanced_probe_examples(
        private_data=private_train,
        config=config,
        num_examples=config.probe_train_examples,
        seed=config.train_loop_seed,
    )
    val_examples = _balanced_probe_examples(
        private_data=private_val,
        config=config,
        num_examples=config.probe_val_examples,
        seed=config.train_loop_seed + 1,
    )
    test_examples = _balanced_probe_examples(
        private_data=private_test,
        config=config,
        num_examples=config.probe_test_examples,
        seed=config.train_loop_seed + 2,
    )

    probe = _train_probe_model(train_examples, val_examples, config, device, probe_seed=probe_seed)

    train_records_base = _collect_trace_records(
        train_examples,
        "train",
        probe,
        config,
        device,
        is_hard_mined=False,
    )
    train_query_keys = {_query_key(example["query_fields"]) for example in train_examples}
    hard_pool_examples = _candidate_probe_examples(
        private_data=private_train,
        config=config,
        num_examples=config.probe_train_examples * config.trace_hard_case_pool_multiplier,
        seed=hard_mining_seed,
        exclude_queries=train_query_keys,
    )
    hard_pool_records = _collect_trace_records(
        hard_pool_examples,
        "train_hard",
        probe,
        config,
        device,
        is_hard_mined=True,
    )
    hard_pool_records = _score_hard_train_candidates(hard_pool_records)
    train_records = _merge_hard_train_records(train_records_base, hard_pool_records, config)
    val_records = _collect_trace_records(
        val_examples,
        "val",
        probe,
        config,
        device,
        is_hard_mined=False,
    )
    test_records = _collect_trace_records(
        test_examples,
        "test",
        probe,
        config,
        device,
        is_hard_mined=False,
    )

    kmeans, metadata = _fit_trace_labeler(train_records, config)
    correctness_model, train_correctness_prob, correctness_metadata = _fit_correctness_labeler(train_records, config)
    metadata.update(correctness_metadata)
    train_records = _attach_correctness_probabilities(train_records, train_correctness_prob)
    val_records = _attach_correctness_probabilities(
        val_records,
        _predict_correctness_probabilities(correctness_model, val_records),
    )
    test_records = _attach_correctness_probabilities(
        test_records,
        _predict_correctness_probabilities(correctness_model, test_records),
    )
    train_records = [
        {**labeled, "evidence_targets": _evidence_targets_from_record(labeled, metadata)}
        for labeled in (_apply_trace_labels(record, kmeans, metadata) for record in train_records)
    ]
    val_records = [
        {**labeled, "evidence_targets": _evidence_targets_from_record(labeled, metadata)}
        for labeled in (_apply_trace_labels(record, kmeans, metadata) for record in val_records)
    ]
    test_records = [
        {**labeled, "evidence_targets": _evidence_targets_from_record(labeled, metadata)}
        for labeled in (_apply_trace_labels(record, kmeans, metadata) for record in test_records)
    ]
    _run_trace_labeler_checks(train_records + val_records + test_records, metadata, config)

    assert all(not record["is_hard_mined"] for record in val_records)
    assert all(not record["is_hard_mined"] for record in test_records)

    write_trace_records(root_dir / "train.jsonl", train_records)
    write_trace_records(root_dir / "val.jsonl", val_records)
    write_trace_records(root_dir / "test.jsonl", test_records)

    train_outcome_distribution = dict(Counter(record["outcome"] for record in train_records))
    failure_idx = OUTCOME_LABELS.index("FAILURE_LIKELY")
    train_hard_case_count = sum(int(record["is_hard_mined"]) for record in train_records)
    train_failure_share = sum(int(record["outcome"] == failure_idx) for record in train_records) / max(len(train_records), 1)
    val_failure_share = sum(int(record["outcome"] == failure_idx) for record in val_records) / max(len(val_records), 1)
    test_failure_share = sum(int(record["outcome"] == failure_idx) for record in test_records) / max(len(test_records), 1)
    corpus_fingerprint = _compute_corpus_fingerprint(
        root_dir,
        _corpus_signature(config, probe_seed=probe_seed, hard_mining_seed=hard_mining_seed),
    )
    with open(root_dir / "metadata.json", "w") as f:
        json.dump(
            {
                "probe_answer_labels": PROBE_ANSWER_LABELS,
                "trajectory_labels": TRAJECTORY_SHAPE_LABELS,
                "attention_pattern_labels": ATTENTION_PATTERN_LABELS,
                "confidence_labels": CONFIDENCE_LABELS,
                "outcome_labels": OUTCOME_LABELS,
                "train_distribution": train_outcome_distribution,
                "train_hard_case_count": train_hard_case_count,
                "train_failure_share": train_failure_share,
                "val_failure_share": val_failure_share,
                "test_failure_share": test_failure_share,
                "probe_seed": probe_seed,
                "hard_mining_seed": hard_mining_seed,
                "corpus_fingerprint": corpus_fingerprint,
                "mlp_converged": None,
                "evidence_target_names": list(EVIDENCE_TARGET_NAMES),
                "evidence_target_version": str(config.trace_evidence_target_version),
                **metadata,
            },
            f,
            indent=2,
        )

    return {
        "trace_dir": str(root_dir),
        "num_train_records": len(train_records),
        "num_val_records": len(val_records),
        "num_test_records": len(test_records),
        "train_distribution": train_outcome_distribution,
        "label_version": metadata["label_version"],
        "pattern_z_threshold": metadata["pattern_z_threshold"],
        "pattern_threshold_mode": metadata["pattern_threshold_mode"],
        "evidence_target_version": config.trace_evidence_target_version,
        "train_hard_case_count": train_hard_case_count,
        "train_failure_share": train_failure_share,
        "val_failure_share": val_failure_share,
        "test_failure_share": test_failure_share,
        "corpus_fingerprint": corpus_fingerprint,
        "mlp_converged": None,
    }


def ensure_trace_corpus(
    config,
    root_dir: str | Path,
    device: torch.device,
    rebuild: bool = False,
    strict_existing: bool = False,
) -> Dict:
    """Ensure trace files exist; build them if they do not."""
    root_dir = Path(root_dir)
    required = [root_dir / "train.jsonl", root_dir / "val.jsonl", root_dir / "test.jsonl", root_dir / "metadata.json"]
    if rebuild:
        return build_trace_corpus(config, root_dir=root_dir, device=device)
    if not all(path.exists() for path in required):
        if strict_existing:
            raise FileNotFoundError(
                f"Trace corpus is missing in {root_dir}. Run scripts/collect_traces.py --force or pass --rebuild-traces."
            )
        return build_trace_corpus(config, root_dir=root_dir, device=device)

    with open(root_dir / "metadata.json") as f:
        metadata = json.load(f)
    rebuild_reason = _corpus_rebuild_reason(metadata, config)
    if rebuild_reason is not None:
        if strict_existing:
            raise RuntimeError(
                f"Existing trace corpus in {root_dir} is not reusable ({rebuild_reason}). "
                "Run scripts/collect_traces.py --force or pass --rebuild-traces."
            )
        return build_trace_corpus(config, root_dir=root_dir, device=device)
    return {
        "trace_dir": str(root_dir),
        "num_train_records": len(load_trace_records(root_dir / "train.jsonl")),
        "num_val_records": len(load_trace_records(root_dir / "val.jsonl")),
        "num_test_records": len(load_trace_records(root_dir / "test.jsonl")),
        "train_distribution": metadata.get("train_distribution", {}),
        "label_version": metadata.get("label_version", "unknown"),
        "pattern_z_threshold": metadata.get("pattern_z_threshold"),
        "pattern_threshold_mode": metadata.get("pattern_threshold_mode"),
        "evidence_target_version": metadata.get("evidence_target_version"),
        "train_hard_case_count": metadata.get("train_hard_case_count", 0),
        "train_failure_share": metadata.get("train_failure_share", 0.0),
        "val_failure_share": metadata.get("val_failure_share", 0.0),
        "test_failure_share": metadata.get("test_failure_share", 0.0),
        "corpus_fingerprint": metadata.get("corpus_fingerprint"),
        "mlp_converged": metadata.get("mlp_converged"),
    }
