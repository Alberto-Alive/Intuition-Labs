"""Trace collection and baseline evaluation for Extrapolation."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
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
from .data.trace_dataset import load_trace_records, write_trace_records
from .models.trace_probe import QueryTraceProbe

PROBE_ANSWER_LABELS = ["ABOVE_BASELINE", "BELOW_BASELINE", "NEAR_BASELINE"]


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
    low_conf_idx = CONFIDENCE_LABELS.index("LOW")

    assert all(
        record["outcome"] == failure_idx
        for record in records
        if not record["is_correct"]
    )
    assert all(
        record["outcome"] == uncertain_idx
        for record in records
        if record["is_correct"] and record["confidence"] == low_conf_idx
    )
    assert all(
        record["outcome"] == success_idx
        for record in records
        if record["is_correct"] and record["confidence"] != low_conf_idx
    )

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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run MC dropout to estimate stability and margin."""
    probs_runs = []
    trajectory_runs = []
    max_attention_runs = []

    with torch.no_grad():
        for _ in range(config.probe_mc_passes):
            model.train()
            out = model(query_batch, collect_trace=True)
            probs_runs.append(torch.softmax(out.logits, dim=-1))
            trajectory_runs.append(out.attention_entropy_trajectory)
            max_attention_runs.append(out.max_attention_mass)
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

    return agreement, prob_margin, avg_trajectory, avg_max_attention, avg_probs


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
        "label_version": "v3",
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

    if not record["is_correct"]:
        outcome = OUTCOME_LABELS.index("FAILURE_LIKELY")
    elif confidence == CONFIDENCE_LABELS.index("LOW"):
        outcome = OUTCOME_LABELS.index("UNCERTAIN")
    else:
        outcome = OUTCOME_LABELS.index("SUCCESS_LIKELY")

    trace_inputs = list(record["attention_entropy_trajectory"])
    trace_inputs.extend([
        float(record["max_attention_mass"]),
        float(record["agreement"]),
        float(record["prob_margin"]),
    ])

    return {
        **record,
        "trajectory_shape": trajectory_shape,
        "attention_pattern": attention_pattern,
        "confidence": confidence,
        "outcome": outcome,
        "trace_inputs": trace_inputs,
        "response_text": _render_response_text(outcome, attention_pattern),
    }


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
            agreement, prob_margin, avg_trajectory, avg_max_attention, avg_probs = _mc_dropout_stats(
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
                        "attention_entropy_trajectory": [float(v) for v in trajectory],
                        "trace_summary": _trajectory_summary(trajectory),
                        "is_correct": bool(pred_answer[i].item() == gold_labels[i].item()),
                        "stats": dict(examples[offset + i]["stats"]),
                        "is_hard_mined": bool(is_hard_mined),
                    }
                )
            offset += query_batch.size(0)

    return records


def _feature_arrays(records: List[Dict]):
    """Prepare arrays for simple baselines."""
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
            list(record["trace_inputs"])
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
    targets = {
        "trajectory_shape": np.array([record["trajectory_shape"] for record in records]),
        "attention_pattern": np.array([record["attention_pattern"] for record in records]),
        "confidence": np.array([record["confidence"] for record in records]),
        "outcome": np.array([record["outcome"] for record in records]),
    }
    return summary, full, targets


def _corpus_signature(config, probe_seed: int, hard_mining_seed: int) -> Dict:
    return {
        "label_version": "v3",
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
    if metadata.get("label_version") != "v3":
        return "label_version"
    if metadata.get("pattern_threshold_mode") != requested_mode:
        return "pattern_threshold_mode"
    if config.trace_pattern_threshold_override is not None and abs(
        float(metadata.get("pattern_z_threshold", -1.0)) - float(config.trace_pattern_threshold_override)
    ) > 1e-8:
        return "pattern_z_threshold"
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
    train_records = [_apply_trace_labels(record, kmeans, metadata) for record in train_records]
    val_records = [_apply_trace_labels(record, kmeans, metadata) for record in val_records]
    test_records = [_apply_trace_labels(record, kmeans, metadata) for record in test_records]
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
        "train_hard_case_count": metadata.get("train_hard_case_count", 0),
        "train_failure_share": metadata.get("train_failure_share", 0.0),
        "val_failure_share": metadata.get("val_failure_share", 0.0),
        "test_failure_share": metadata.get("test_failure_share", 0.0),
        "corpus_fingerprint": metadata.get("corpus_fingerprint"),
        "mlp_converged": metadata.get("mlp_converged"),
    }
