"""Shared query helpers for comparative attacks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..data.base import PrivateDataset, QueryField
from ..data.ground_truth import (
    ANSWER_LABELS,
    GTConfig,
)


@dataclass
class QueryAccounting:
    """Track query usage for black-box attacks."""

    query_count: int = 0
    cumulative_epsilon: float = 0.0

    def record(self, n_queries: int, epsilon: Optional[float] = None) -> None:
        self.query_count += int(n_queries)
        if epsilon is not None:
            self.cumulative_epsilon += float(epsilon) * float(n_queries)


def active_specificity(query_vec: torch.Tensor, n_fields: int) -> int:
    return int(sum(1 for i in range(n_fields) if int(query_vec[i].item()) != 0))


def build_exact_match_query(record_features: torch.Tensor) -> torch.Tensor:
    n_fields = int(record_features.numel())
    q = torch.zeros(n_fields + 1, dtype=torch.long)
    q[:n_fields] = record_features.long()
    q[n_fields] = n_fields
    return q


def build_partial_queries(
    record_features: torch.Tensor,
    n_queries: int,
    seed: int,
    min_specificity: int = 1,
    max_specificity: Optional[int] = None,
) -> List[torch.Tensor]:
    rng = np.random.RandomState(seed)
    n_fields = int(record_features.numel())
    max_k = max_specificity or min(n_fields, 8)
    queries: List[torch.Tensor] = []
    for _ in range(n_queries):
        k = int(rng.randint(min_specificity, max_k + 1))
        active = rng.choice(n_fields, size=min(k, n_fields), replace=False)
        q = torch.zeros(n_fields + 1, dtype=torch.long)
        for idx in active:
            q[idx] = int(record_features[idx].item())
        q[n_fields] = int(k)
        queries.append(q)
    return queries


def build_refinement_pairs(
    record_features: torch.Tensor,
    n_pairs: int,
    seed: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Broad-vs-narrow query pairs.

    The current query language is monotone equality filtering, so it cannot
    express "exclude exactly one person". We approximate differencing by
    broad-query / one-more-filter refinement pairs.
    """
    rng = np.random.RandomState(seed)
    n_fields = int(record_features.numel())
    pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(n_pairs):
        broad_k = int(rng.randint(1, min(4, n_fields) + 1))
        extra_k = min(n_fields, broad_k + 1)
        broad_active = list(rng.choice(n_fields, size=broad_k, replace=False))
        remaining = [idx for idx in range(n_fields) if idx not in broad_active]
        if not remaining:
            remaining = broad_active
        extra_field = int(rng.choice(remaining))
        narrow_active = broad_active + [extra_field]

        broad = torch.zeros(n_fields + 1, dtype=torch.long)
        narrow = torch.zeros(n_fields + 1, dtype=torch.long)
        for idx in broad_active:
            broad[idx] = int(record_features[idx].item())
            narrow[idx] = int(record_features[idx].item())
        narrow[extra_field] = int(record_features[extra_field].item())
        broad[n_fields] = len(broad_active)
        narrow[n_fields] = len(narrow_active)
        pairs.append((broad, narrow))
    return pairs


def primitive_vector_from_system(
    system: Any,
    query: torch.Tensor,
    private_data: PrivateDataset,
) -> Dict[str, Any]:
    """Return a normalized response dict for any supported system."""
    if hasattr(system, "answer_query"):
        out = dict(system.answer_query(query))
        out.setdefault(
            "primitive_vector",
            np.array(
                [out["answer"], out["support"], out["confidence"], out["risk"]],
                dtype=np.int64,
            ),
        )
        return out

    if hasattr(system, "generate"):
        q_b = query.unsqueeze(0).to(next(system.parameters()).device)
        with torch.no_grad():
            out = system.generate(q_b, private_data)
        p = out["primitives"]
        answer = int(p.answer_discrete.argmax(-1)[0])
        support = int(p.support_discrete.argmax(-1)[0])
        confidence = int(p.confidence_discrete.argmax(-1)[0])
        risk = int(p.risk_discrete.argmax(-1)[0])
        return {
            "answer": answer,
            "support": support,
            "confidence": confidence,
            "risk": risk,
            "primitive_vector": np.array([answer, support, confidence, risk], dtype=np.int64),
        }

    raise TypeError(f"Unsupported system type: {type(system)}")


def primitive_rate_interval(
    response: Dict[str, Any],
    overall_rate: float,
    cfg: GTConfig,
) -> Tuple[float, float]:
    if "exact_rate" in response:
        rate = float(response["exact_rate"])
        return rate, rate
    if "noisy_rate" in response:
        rate = float(response["noisy_rate"])
        return rate, rate
    if "true_rate" in response:
        rate = float(response["true_rate"])
        return rate, rate

    answer_idx = int(response["answer"])
    answer = ANSWER_LABELS[answer_idx] if 0 <= answer_idx < len(ANSWER_LABELS) else "INCONSISTENT_SIGNAL"
    low_margin = max(0.0, overall_rate - cfg.rate_margin)
    high_margin = min(1.0, overall_rate + cfg.rate_margin)

    if answer == "YES":
        return high_margin, 1.0
    if answer == "NO":
        return 0.0, low_margin
    if answer == "MAYBE":
        return low_margin, high_margin
    if answer in {"INSUFFICIENT_EVIDENCE", "POLICY_BLOCKED", "INCONSISTENT_SIGNAL"}:
        return 0.0, 1.0
    return 0.0, 1.0


def primitive_rate_estimate(
    response: Dict[str, Any],
    overall_rate: float,
    cfg: GTConfig,
) -> float:
    lo, hi = primitive_rate_interval(response, overall_rate, cfg)
    return 0.5 * (lo + hi)


def batch_rate_estimates(
    responses: Sequence[Dict[str, Any]],
    overall_rate: float,
    cfg: GTConfig,
) -> np.ndarray:
    return np.array([primitive_rate_estimate(r, overall_rate, cfg) for r in responses], dtype=float)


def dataset_with_appended_record(
    private_data: PrivateDataset,
    feature_values: Sequence[int],
    label: int,
) -> PrivateDataset:
    from ..data.base import DatasetInfo, TabularPrivateDataset

    feat = torch.tensor(feature_values, dtype=torch.long).unsqueeze(0)
    lbl = torch.tensor([label], dtype=torch.long)
    features = torch.cat([private_data.features, feat], dim=0)
    labels = torch.cat([private_data.labels, lbl], dim=0)
    info = DatasetInfo(
        dataset_name=private_data.info.dataset_name,
        split_name=f"{private_data.info.split_name}_plus_canary",
        n_records=int(features.shape[0]),
        n_features=private_data.info.n_features,
        label_name=private_data.info.label_name,
        positive_rate=float(labels.float().mean()),
        query_fields=private_data.info.query_fields,
        field_vocab_sizes=private_data.info.field_vocab_sizes,
    )
    return TabularPrivateDataset(features, labels, private_data.query_fields, info)


def choose_canary_feature_values(
    private_data: PrivateDataset,
    override: Optional[Sequence[int]] = None,
) -> List[int]:
    if override is not None:
        return [int(v) for v in override]
    vals = []
    for field in private_data.query_fields:
        vals.append(max(1, int(field.vocab_size)))
    return vals


def find_answerable_queries(
    private_data: PrivateDataset,
    queries: Iterable[torch.Tensor],
    min_group_size: int,
) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    for q in queries:
        stats = private_data.get_subgroup_stats(q)
        if int(stats["n"]) >= int(min_group_size):
            out.append(q)
    return out
