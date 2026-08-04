import unittest
import numpy as np
import torch

from run_taskmap_experiment import (
    RelationalTaskMap,
    append_to_basis,
    make_task,
    orthogonal_matrix,
    residual_norms,
)


class TaskMapTests(unittest.TestCase):
    def test_strong_scaling_same_content(self):
        a = make_task(2, 4, "ood", 1, input_noise=0.1)
        b = make_task(2, 4, "ood", 16, input_noise=0.1)
        np.testing.assert_allclose(a.items, b.items)
        np.testing.assert_allclose(a.targets, b.targets)
        np.testing.assert_allclose(a.anchors, b.anchors)

    def test_relational_map_rotation_invariance_before_training(self):
        torch.manual_seed(0)
        model = RelationalTaskMap().eval()
        task = make_task(1, 1, "ood", 4, input_noise=0.0)
        q = orthogonal_matrix(task.items.shape[1], 888).astype(np.float32)
        with torch.no_grad():
            a = model(torch.tensor(task.items[None]), torch.tensor(task.anchors[None]))
            b = model(torch.tensor((task.items @ q)[None]), torch.tensor((task.anchors @ q)[None]))
        np.testing.assert_allclose(a.numpy(), b.numpy(), atol=1e-5)

    def test_goal_token_permutation_equivariance(self):
        torch.manual_seed(0)
        model = RelationalTaskMap().eval()
        task = make_task(3, 2, "ood", 4, input_noise=0.1)
        perm = np.random.default_rng(7).permutation(task.rank)
        with torch.no_grad():
            base = model(torch.tensor(task.items[None]), torch.tensor(task.anchors[None]))[0].numpy()
            moved = model(torch.tensor(task.items[None]), torch.tensor(task.anchors[perm][None]))[0].numpy()
        np.testing.assert_allclose(moved, base[:, perm], atol=1e-5)

    def test_basis_novelty(self):
        q = np.zeros((4, 0))
        q, n1 = append_to_basis(q, np.array([1.0, 0, 0, 0]))
        q, n2 = append_to_basis(q, np.array([1.0, 0, 0, 0]))
        self.assertGreater(n1, 0)
        self.assertEqual(n2, 0)
        self.assertAlmostEqual(float(residual_norms(np.array([[1.0, 0, 0, 0]]), q)[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
