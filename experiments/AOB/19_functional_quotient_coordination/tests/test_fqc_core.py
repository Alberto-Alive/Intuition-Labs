import sys
from pathlib import Path

import numpy as np
import pytest

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from fqc_core import FunctionalQuotientState, propose_stochastic  # noqa: E402


def test_new_contribution_creates_class():
    state = FunctionalQuotientState(dim=4)
    assert state.submit(np.array([1.0, 0.0, 0.0, 0.0]), utility=1.0) == "new"
    assert state.class_count == 1
    assert state.basis.shape == (4, 1)


def test_equivalent_contribution_collapses_as_duplicate():
    state = FunctionalQuotientState(dim=4, equivalence_cos=0.99)
    state.submit(np.array([1.0, 0.0, 0.0, 0.0]), utility=1.0)
    outcome = state.submit(np.array([1.0, 1e-4, 0.0, 0.0]), utility=0.5)
    assert outcome == "duplicate"
    assert state.class_count == 1
    assert state.collapsed_count == 1


def test_equivalent_with_better_utility_updates_representative():
    state = FunctionalQuotientState(dim=4, equivalence_cos=0.99)
    state.submit(np.array([1.0, 0.0, 0.0, 0.0]), utility=1.0)
    outcome = state.submit(np.array([1.0, 1e-4, 0.0, 0.0]), utility=2.0)
    assert outcome == "update"
    assert state.class_count == 1
    assert state.classes[0].utility == 2.0


def test_orthogonal_contribution_adds_class():
    state = FunctionalQuotientState(dim=4)
    state.submit(np.array([1.0, 0.0, 0.0, 0.0]))
    assert state.submit(np.array([0.0, 1.0, 0.0, 0.0])) == "new"
    assert state.class_count == 2


def test_spanned_but_nonequivalent_contribution_rejected():
    state = FunctionalQuotientState(dim=4, equivalence_cos=0.999)
    state.submit(np.array([1.0, 0.0, 0.0, 0.0]))
    state.submit(np.array([0.0, 1.0, 0.0, 0.0]))
    # 45-degree mix: not equivalent to either class, adds no residual.
    assert state.submit(np.array([1.0, 1.0, 0.0, 0.0])) == "reject"
    assert state.class_count == 2


def test_zero_vector_rejected():
    state = FunctionalQuotientState(dim=4)
    assert state.submit(np.zeros(4)) == "reject"


def test_novelty_reflects_resolved_subspace():
    state = FunctionalQuotientState(dim=3)
    state.submit(np.array([1.0, 0.0, 0.0]))
    novelty = state.novelty(np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]))
    assert novelty[0] == pytest.approx(0.0, abs=1e-9)
    assert novelty[1] == pytest.approx(2.0)


def test_coverage_uses_class_representatives():
    state = FunctionalQuotientState(dim=3)
    state.submit(np.array([1.0, 0.0, 0.0]))
    cov = state.coverage(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    assert cov[0] == pytest.approx(1.0)
    assert cov[1] == pytest.approx(0.0, abs=1e-9)


def test_propose_idles_when_nothing_is_novel():
    rng = np.random.default_rng(0)
    assert propose_stochastic(rng, np.zeros(5), np.ones(5)) is None


def test_propose_prefers_high_novelty():
    rng = np.random.default_rng(0)
    novelty = np.array([0.01, 0.9, 0.02])
    picks = [propose_stochastic(rng, novelty, np.zeros(3), beta=12.0) for _ in range(50)]
    assert picks.count(1) > 40


def test_identical_agents_collide_without_slots():
    # Same shared state, same view, private rngs only: collisions must be
    # possible (no hidden forced-distinct mechanism).
    novelty = np.array([0.5, 0.5, 0.5, 0.5])
    rngs = [np.random.default_rng(i) for i in range(8)]
    picks = [propose_stochastic(rng, novelty, np.zeros(4), beta=12.0) for rng in rngs]
    assert len(set(picks)) < len(picks)
