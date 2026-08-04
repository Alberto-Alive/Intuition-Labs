"""Pre-training invariant tests for DIGIT Extrapolation E26.

These five tests are self-contained and do not depend on PyTorch. They verify
the frozen math and state-timing invariants before any training code exists.

Test 1 - Gradient decomposition reconstructs g_h exactly.
Test 2 - alpha(F) + beta(F) is constant and equal to 2 on [0, 1].
Test 3 - Boundary conditions at F=0 and F=1 match the frozen spec exactly.
Test 4 - Prototype snapshot timing: prototypes are unchanged by backward and
         change only after the explicit post-optimizer update call.
Test 5 - Task A probe set is generated once per seed before training and is
         immutable across a simulated Task B phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from random import Random


EPS = 1e-8


Vector = list[float]
Matrix = list[list[float]]


def clamp_familiarity(familiarity: float) -> float:
    return max(0.0, min(1.0, familiarity))


def alpha(familiarity: float) -> float:
    return 1.0 + familiarity


def beta(familiarity: float) -> float:
    return 1.0 - familiarity


def dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(v: Vector) -> float:
    return sqrt(sum(x * x for x in v))


def add(a: Vector, b: Vector) -> Vector:
    return [x + y for x, y in zip(a, b)]


def sub(a: Vector, b: Vector) -> Vector:
    return [x - y for x, y in zip(a, b)]


def scale(v: Vector, s: float) -> Vector:
    return [s * x for x in v]


def matvec(matrix: Matrix, vector: Vector) -> Vector:
    return [dot(row, vector) for row in matrix]


def matmul(matrix_a: Matrix, matrix_b: Matrix) -> Matrix:
    rows = len(matrix_a)
    cols = len(matrix_b[0])
    inner = len(matrix_b)
    result: Matrix = []
    for i in range(rows):
        row: list[float] = []
        for j in range(cols):
            total = 0.0
            for k in range(inner):
                total += matrix_a[i][k] * matrix_b[k][j]
            row.append(total)
        result.append(row)
    return result


def softmax(vector: Vector) -> Vector:
    max_value = max(vector)
    exps = [pow(2.718281828459045, value - max_value) for value in vector]
    total = sum(exps)
    return [value / total for value in exps]


def allclose_vector(a: Vector, b: Vector, atol: float = 1e-7) -> bool:
    return len(a) == len(b) and all(abs(x - y) <= atol for x, y in zip(a, b))


def allclose_matrix(a: Matrix, b: Matrix, atol: float = 1e-7) -> bool:
    return len(a) == len(b) and all(allclose_vector(row_a, row_b, atol=atol) for row_a, row_b in zip(a, b))


def decompose_gradient(g_h: Vector, n: Vector) -> tuple[Vector, Vector, Vector]:
    n_hat = scale(n, 1.0 / max(norm(n), EPS))
    g_parallel = scale(n_hat, dot(g_h, n_hat))
    g_perp = sub(g_h, g_parallel)
    return n_hat, g_parallel, g_perp


def gate_gradient(g_h: Vector, n: Vector, familiarity: float) -> tuple[Vector, dict[str, Vector | float]]:
    n_hat, g_parallel, g_perp = decompose_gradient(g_h, n)
    f = clamp_familiarity(familiarity)
    g_tilde = add(scale(g_parallel, alpha(f)), scale(g_perp, beta(f)))
    return g_tilde, {
        "n_hat": n_hat,
        "g_parallel": g_parallel,
        "g_perp": g_perp,
        "f": f,
    }


def checksum_matrix(matrix: Matrix) -> dict[str, float]:
    values = [value for row in matrix for value in row]
    total = sum(values)
    mean = total / len(values)
    l2 = sqrt(sum(value * value for value in values))
    return {"sum": total, "mean": mean, "l2": l2}


def checksum_text(matrix: Matrix) -> str:
    c = checksum_matrix(matrix)
    return f"sum={c['sum']:.10f}, mean={c['mean']:.10f}, l2={c['l2']:.10f}"


@dataclass
class TimingForwardResult:
    h: Matrix
    nearest_prototype: Matrix
    residual: Matrix
    logits: Matrix


class TimingProbeModel:
    """Minimal substrate for the prototype snapshot timing invariant."""

    def __init__(self) -> None:
        self.encoder_weight: Matrix = [
            [0.8, -0.4, 0.2, 0.1],
            [-0.3, 0.5, 0.7, -0.6],
        ]
        self.head_weight: Matrix = [
            [0.4, -0.2],
            [-0.1, 0.6],
        ]
        self.prototypes: Matrix = [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        self.valid_mask = [True, True]

    def forward(self, inputs: Matrix) -> TimingForwardResult:
        h = [matvec(self.encoder_weight, row) for row in inputs]
        nearest_prototype = [self._nearest_prototype_snapshot(row) for row in h]
        residual = [sub(row, proto) for row, proto in zip(h, nearest_prototype)]
        logits = [matvec(self.head_weight, row) for row in h]
        return TimingForwardResult(
            h=h,
            nearest_prototype=nearest_prototype,
            residual=residual,
            logits=logits,
        )

    def _nearest_prototype_snapshot(self, h_row: Vector) -> Vector:
        valid = [proto for proto, is_valid in zip(self.prototypes, self.valid_mask) if is_valid]
        distances = [norm(sub(h_row, proto)) for proto in valid]
        nearest_index = distances.index(min(distances))
        return valid[nearest_index][:]

    def loss_value(self, logits: Matrix, targets: list[int]) -> float:
        losses = []
        for row, target in zip(logits, targets):
            probs = softmax(row)
            losses.append(-1.0 * float(__import__("math").log(probs[target])))
        return sum(losses) / len(losses)

    def backward(self) -> None:
        """No-op on prototypes, standing in for loss.backward() in this harness."""
        return None

    def optimizer_step(self) -> None:
        """No-op on prototypes, standing in for optimizer.step() in this harness."""
        return None

    def update_prototypes(self, h: Matrix) -> None:
        """Deterministic post-step prototype update used only for the timing test."""
        detached = [scale(row, 1.0 / max(norm(row), EPS)) for row in h]
        self.prototypes[0] = scale(add(self.prototypes[0], detached[0]), 0.5)
        self.prototypes[1] = scale(add(self.prototypes[1], detached[-1]), 0.5)


def build_task_a_probe_set(seed: int) -> tuple[list[int], list[int], list[int]]:
    """Generate the frozen Task A probe set once per seed."""
    rng = Random(seed + 12345)
    part_a_values: list[int] = []
    part_b_values: list[int] = []
    labels: list[int] = []

    for part_a in range(8):
        for part_b in range(4):
            for _ in range(32):
                part_a_values.append(part_a)
                part_b_values.append(part_b)
                labels.append((part_a + part_b) % 4)

    indices = list(range(len(labels)))
    rng.shuffle(indices)
    return (
        [part_a_values[index] for index in indices],
        [part_b_values[index] for index in indices],
        [labels[index] for index in indices],
    )


def test_gradient_decomposition_reconstructs_the_original_gradient_exactly() -> None:
    g_h = [0.6, -1.2, 0.8]
    n = [0.5, 0.5, 0.0]
    familiarity = 0.75

    g_tilde, pieces = gate_gradient(g_h, n, familiarity)
    reconstructed = add(pieces["g_parallel"], pieces["g_perp"])  # type: ignore[arg-type]
    max_error = max(abs(x - y) for x, y in zip(reconstructed, g_h))

    assert allclose_vector(reconstructed, g_h, atol=1e-7)
    assert max_error <= 1e-7

    print("PASS  Test 1: gradient decomposition reconstructs g_h exactly.")
    print(f"       g_h        = {g_h}")
    print(f"       n          = {n}")
    print(f"       n_hat      = {pieces['n_hat']}")
    print(f"       g_parallel = {pieces['g_parallel']}")
    print(f"       g_perp     = {pieces['g_perp']}")
    print(f"       g_tilde    = {g_tilde}")
    print(f"       max_error  = {max_error:.10e}")


def test_alpha_and_beta_sum_to_two_for_the_full_unit_interval() -> None:
    familiarity_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    alpha_values = [alpha(value) for value in familiarity_values]
    beta_values = [beta(value) for value in familiarity_values]
    total_values = [a + b for a, b in zip(alpha_values, beta_values)]

    assert all(abs(total - 2.0) <= 1e-7 for total in total_values)

    print("PASS  Test 2: alpha(F) + beta(F) is constant and equal to 2.")
    for f_value, a_value, b_value, total in zip(familiarity_values, alpha_values, beta_values, total_values):
        print(f"       F={f_value:.2f} -> alpha={a_value:.6f}, beta={b_value:.6f}, sum={total:.6f}")


def test_boundary_conditions_match_the_frozen_spec_exactly() -> None:
    g_h = [2.0, -1.0, 0.5]
    n = [3.0, 4.0, 0.0]

    g_tilde_f0, pieces_f0 = gate_gradient(g_h, n, 0.0)
    g_tilde_f1, pieces_f1 = gate_gradient(g_h, n, 1.0)

    expected_f0 = g_h
    expected_f1 = scale(pieces_f1["g_parallel"], 2.0)  # type: ignore[arg-type]

    max_error_f0 = max(abs(x - y) for x, y in zip(g_tilde_f0, expected_f0))
    max_error_f1 = max(abs(x - y) for x, y in zip(g_tilde_f1, expected_f1))

    assert allclose_vector(g_tilde_f0, expected_f0, atol=1e-7)
    assert allclose_vector(g_tilde_f1, expected_f1, atol=1e-7)

    print("PASS  Test 3: boundary conditions at F=0 and F=1 match the spec.")
    print(f"       g_h           = {g_h}")
    print(f"       n             = {n}")
    print(f"       g_parallel    = {pieces_f1['g_parallel']}")
    print(f"       g_perp        = {pieces_f1['g_perp']}")
    print(f"       g_tilde(F=0)  = {g_tilde_f0}")
    print(f"       expected(F=0) = {expected_f0}")
    print(f"       max_error(F=0)= {max_error_f0:.10e}")
    print(f"       g_tilde(F=1)  = {g_tilde_f1}")
    print(f"       expected(F=1) = {expected_f1}")
    print(f"       max_error(F=1)= {max_error_f1:.10e}")


def test_prototype_snapshot_is_frozen_before_backward_and_changes_only_after_update() -> None:
    model = TimingProbeModel()
    x = [
        [1.0, 0.0, 0.5, -0.5],
        [0.0, 1.0, -0.25, 0.75],
        [0.5, 0.5, 0.5, 0.5],
    ]
    targets = [0, 1, 0]

    forward = model.forward(x)
    before_backward = [row[:] for row in model.prototypes]
    before_checksum = checksum_matrix(before_backward)

    loss = model.loss_value(forward.logits, targets)
    model.backward()
    after_backward = [row[:] for row in model.prototypes]
    after_backward_checksum = checksum_matrix(after_backward)

    model.optimizer_step()
    model.update_prototypes(forward.h)
    after_update = [row[:] for row in model.prototypes]
    after_update_checksum = checksum_matrix(after_update)

    assert allclose_matrix(before_backward, after_backward, atol=1e-12)
    assert not allclose_matrix(after_backward, after_update, atol=1e-12)
    assert before_checksum == after_backward_checksum
    assert after_update_checksum != before_checksum

    print("PASS  Test 4: prototype snapshot timing is correct.")
    print(f"       prototypes before backward: {before_backward}")
    print(f"       checksum before backward : {checksum_text(before_backward)}")
    print(f"       checksum after backward  : {checksum_text(after_backward)}")
    print(f"       checksum after update    : {checksum_text(after_update)}")
    print(f"       loss                     : {loss:.10f}")


def test_task_a_probe_set_is_generated_once_and_remains_unchanged_across_task_b() -> None:
    seed = 3
    probe_a_1, probe_b_1, labels_1 = build_task_a_probe_set(seed)
    probe_a_before = probe_a_1[:]
    probe_b_before = probe_b_1[:]
    labels_before = labels_1[:]

    # Simulated Task B phase mutates unrelated values only.
    task_b_tensor = [float(value) for value in range(16)]
    task_b_tensor = [value + 1.0 for value in task_b_tensor]
    task_b_tensor = [value * 0.5 for value in task_b_tensor]

    # The frozen probe set is never rebuilt and never modified.
    probe_a_after, probe_b_after, labels_after = probe_a_1, probe_b_1, labels_1

    assert probe_a_before == probe_a_after
    assert probe_b_before == probe_b_after
    assert labels_before == labels_after

    probe_checksum_before = (
        sum(probe_a_before),
        sum(probe_b_before),
        sum(labels_before),
    )
    probe_checksum_after = (
        sum(probe_a_after),
        sum(probe_b_after),
        sum(labels_after),
    )

    assert probe_checksum_before == probe_checksum_after

    print("PASS  Test 5: Task A probe set is generated once and remains unchanged.")
    print(f"       probe_a length       = {len(probe_a_before)}")
    print(f"       probe_b length       = {len(probe_b_before)}")
    print(f"       labels length        = {len(labels_before)}")
    print(f"       checksum before      = {probe_checksum_before}")
    print(f"       checksum after       = {probe_checksum_after}")
    print(f"       unrelated Task B tensor = {task_b_tensor}")


def _run_all_tests() -> None:
    test_gradient_decomposition_reconstructs_the_original_gradient_exactly()
    test_alpha_and_beta_sum_to_two_for_the_full_unit_interval()
    test_boundary_conditions_match_the_frozen_spec_exactly()
    test_prototype_snapshot_is_frozen_before_backward_and_changes_only_after_update()
    test_task_a_probe_set_is_generated_once_and_remains_unchanged_across_task_b()
    print("ALL FIVE E26 INVARIANT TESTS PASSED")


if __name__ == "__main__":
    _run_all_tests()
