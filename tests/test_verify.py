"""verify.py on a quadratic PES with three exact translation zero modes (a periodic cell, no fixed atoms)."""
import numpy as np

from saddleflow.classical_saddle_search.verify import central_curvature, local_hessian, lowest_modes_block


class Dofs:
    def __init__(self, n):
        self.free = np.ones(n, bool); self.n_free = n; self.dimension = 3 * n


class QuadraticPES:
    """E = 1/2 x^T H x with H built to annihilate uniform translations; one negative mode."""
    def __init__(self, n=12, seed=0):
        rng = np.random.default_rng(seed)
        dim = 3 * n
        T = np.zeros((n, 3, 3))
        for k in range(3):
            T[:, k, k] = 1.0
        T = T.reshape(dim, 3) / np.sqrt(n)
        Q, _ = np.linalg.qr(np.hstack([T, rng.normal(size=(dim, dim - 3))]))
        values = np.r_[0, 0, 0, -0.6, 0.3, 0.35, np.linspace(1, 40, dim - 6)]
        self.H = (Q * values) @ Q.T
        self.dofs, self.force_calls = Dofs(n), 0

    def forces(self, x):
        self.force_calls += 1
        return -self.H @ x

    def set_coordinates(self, x):
        pass


def test_lowest_modes_skip_translations():
    pes = QuadraticPES()
    x = np.zeros(pes.dofs.dimension)
    values, vectors, calls = lowest_modes_block(pes, x, k=3, tol=1e-6, maxiter=200)
    assert np.allclose(values, [-0.6, 0.3, 0.35], atol=1e-3)      # the zero translation modes are excluded
    assert calls == pes.force_calls


def test_central_and_local_block():
    pes = QuadraticPES()
    x = np.zeros(pes.dofs.dimension)
    v = np.random.default_rng(1).normal(size=pes.dofs.dimension)
    assert np.isclose(central_curvature(pes, x, v), v @ pes.H @ v / (v @ v), atol=1e-8)
    H, coords, calls = local_hessian(pes, x, atoms=[0, 3])
    assert np.allclose(H, pes.H[np.ix_(coords, coords)], atol=1e-8) and calls == 7
