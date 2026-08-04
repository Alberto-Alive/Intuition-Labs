"""Functional Quotient Coordination (FQC) core state.

Adapts the RFQ principle (see ../rfq_reference/: quotient a population of
computations by functional equivalence, keep one representative per class,
select a complementary basis) to multi-agent work coordination:

    shared goal
    -> clones propose contributions in parallel
    -> encode each contribution relative to the goal
    -> quotient functionally equivalent contributions
    -> estimate the unresolved residual
    -> clones condition their next proposal on the residual map

The state is deliberately tiny: an orthonormal basis of accepted contribution
embeddings (the resolved subspace) plus per-class bookkeeping. Everything an
agent conditions on is derivable from this shared public state; there is no
central scheduler, no per-agent reservation, and no ground-truth decomposition.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

TOL = 1e-6


def append_to_basis(q: np.ndarray, v: np.ndarray, tol: float = TOL) -> Tuple[np.ndarray, float]:
    residual = v - q @ (q.T @ v) if q.shape[1] else v.astype(float).copy()
    norm = float(np.linalg.norm(residual))
    if norm <= tol:
        return q, 0.0
    return np.column_stack([q, residual / norm]), norm


def residual_norms(vectors: np.ndarray, q: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    if q.shape[1] == 0:
        return np.linalg.norm(vectors, axis=1)
    projection = vectors @ q
    residual = vectors - projection @ q.T
    return np.linalg.norm(residual, axis=1)


@dataclass
class QuotientClass:
    representative: np.ndarray
    utility: float
    count: int = 1


@dataclass
class FunctionalQuotientState:
    """Shared set Q_t of quotient classes of accepted contributions."""

    dim: int
    equivalence_cos: float = 0.995
    novelty_tol: float = 1e-4
    classes: List[QuotientClass] = field(default_factory=list)
    basis: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.basis is None:
            self.basis = np.zeros((self.dim, 0), dtype=float)

    # -- residual map ------------------------------------------------------
    def novelty(self, vectors: np.ndarray) -> np.ndarray:
        """Unresolved-residual magnitude of each candidate contribution."""
        return residual_norms(np.atleast_2d(vectors), self.basis)

    def coverage(self, vectors: np.ndarray) -> np.ndarray:
        """Max cosine of each candidate to any accepted class representative."""
        vectors = np.atleast_2d(np.asarray(vectors, dtype=float))
        if not self.classes:
            return np.zeros(len(vectors))
        reps = np.stack([c.representative for c in self.classes])
        norms = np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=TOL)
        rep_norms = np.linalg.norm(reps, axis=1, keepdims=True).clip(min=TOL)
        cos = (vectors / norms) @ (reps / rep_norms).T
        return cos.max(axis=1)

    # -- quotient step -----------------------------------------------------
    def submit(self, vector: np.ndarray, utility: float = 0.0) -> str:
        """Process one contribution. Returns the quotient outcome:

        "duplicate"  equivalent to an accepted class, no extra utility
        "update"     equivalent, but better evidence -> replaces representative
        "new"        non-equivalent and reduces the residual -> new class
        "reject"     non-equivalent but adds no residual reduction
        """
        vector = np.asarray(vector, dtype=float)
        vec_norm = float(np.linalg.norm(vector))
        if vec_norm <= TOL:
            return "reject"

        best_cos, best_idx = -1.0, -1
        for idx, cls in enumerate(self.classes):
            rep = cls.representative
            cos = float(vector @ rep / (vec_norm * max(float(np.linalg.norm(rep)), TOL)))
            if cos > best_cos:
                best_cos, best_idx = cos, idx

        if best_idx >= 0 and best_cos >= self.equivalence_cos:
            cls = self.classes[best_idx]
            cls.count += 1
            if utility > cls.utility:
                cls.representative = vector.copy()
                cls.utility = utility
                return "update"
            return "duplicate"

        new_basis, gain = append_to_basis(self.basis, vector, tol=self.novelty_tol)
        if gain <= self.novelty_tol:
            return "reject"
        self.basis = new_basis
        self.classes.append(QuotientClass(representative=vector.copy(), utility=utility))
        return "new"

    # -- summaries ---------------------------------------------------------
    @property
    def class_count(self) -> int:
        return len(self.classes)

    @property
    def collapsed_count(self) -> int:
        """Contributions that were quotiented into existing classes."""
        return sum(c.count - 1 for c in self.classes)


def propose_stochastic(
    rng: np.random.Generator,
    novelty: np.ndarray,
    scores: np.ndarray,
    beta: float = 12.0,
    novelty_tol: float = 1e-4,
    viable: np.ndarray | None = None,
) -> int | None:
    """One agent's independent proposal from its (possibly private-noisy) view.

    Samples from softmax(beta * novelty) over candidates that still reduce the
    residual. Returns None when the agent sees nothing left to contribute
    (it idles / verifies instead of duplicating). Private randomness is the only
    symmetry breaker: no identity slots, no reservations.
    """
    if viable is None:
        viable = novelty > novelty_tol
    if not viable.any():
        return None
    logits = beta * novelty + 1e-4 * scores
    logits = np.where(viable, logits, -np.inf)
    logits = logits - logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    return int(rng.choice(len(novelty), p=probs))
