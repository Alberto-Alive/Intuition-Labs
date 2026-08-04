import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

import run_experiment_a as A  # noqa: E402


def _tiny_task(n_agents: int, seed: int = 0):
    return A.S18.make_task(seed, 0, "ood", n_agents, rank=8, ambient=24, input_noise=0.0)


def test_fqc_oracle_embeddings_solve_without_duplicates():
    task = _tiny_task(4)
    predicted = A.S18.predict_embeddings(task, "oracle_previous", {})
    row = A.run_fqc_coordination(task, predicted, seed=0, n_agents=4, variant={"embed": "oracle_previous"})
    assert row["solved"]
    assert row["duplicate_fraction"] == 0.0
    assert row["quotient_class_count"] == task.rank


def test_fqc_is_deterministic_given_seed():
    task = _tiny_task(4)
    predicted = A.S18.predict_embeddings(task, "oracle_previous", {})
    rows = [
        A.run_fqc_coordination(task, predicted, seed=7, n_agents=4, variant={})
        for _ in range(2)
    ]
    assert rows[0] == rows[1]


def test_no_quotient_control_duplicates_heavily():
    task = _tiny_task(8)
    predicted = A.S18.predict_embeddings(task, "oracle_previous", {})
    full = A.run_fqc_coordination(task, predicted, seed=0, n_agents=8, variant={})
    control = A.run_fqc_coordination(task, predicted, seed=0, n_agents=8, variant={"no_state": True})
    assert control["duplicate_fraction"] > full["duplicate_fraction"]
    assert control["rounds"] >= full["rounds"]


def test_more_agents_do_not_slow_completion():
    predicted = None
    rounds = []
    for n_agents in (1, 4, 16):
        task = _tiny_task(n_agents)
        predicted = A.S18.predict_embeddings(task, "oracle_previous", {})
        row = A.run_fqc_coordination(task, predicted, seed=0, n_agents=n_agents, variant={})
        assert row["solved"]
        rounds.append(row["rounds"])
    assert rounds[0] >= rounds[1] >= rounds[2]


def test_excess_agents_stand_down_instead_of_duplicating():
    task = _tiny_task(32)  # 32 agents, only 8 latent directions
    predicted = A.S18.predict_embeddings(task, "oracle_previous", {})
    row = A.run_fqc_coordination(task, predicted, seed=0, n_agents=32, variant={})
    assert row["solved"]
    assert row["duplicate_fraction"] < 0.2
    assert row["idle_agent_rounds"] > 0
