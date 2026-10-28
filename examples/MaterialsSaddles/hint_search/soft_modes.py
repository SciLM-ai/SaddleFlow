"""Random soft-mode displacements of a minimum: a thermal-like snapshot with a random direction.

The full Hessian at the minimum (forward differences, 3N + 1 force calls, the three uniform translations
projected out) gives the vibration modes. A random combination with each mode weighted by
1/sqrt(eigenvalue) puts nearly all of the displacement into the softest, collective motions, as thermal
fluctuations do, where isotropic noise puts most of it into stiff bond stretches. The eigenvalues are
floored so that a nearly flat mode cannot take all of the weight.
"""
import numpy as np

FLOOR = 0.05      # eV/A^2


def soft_modes(pes, x):
    """(eigenvalues, modes (3N, 3N - 3), force calls) of the Hessian at x, translations removed.

    The structure must have no fixed atoms (the translations are then exact zero modes)."""
    from scipy.linalg import null_space
    from saddleflow.classical_saddle_search import local_hessian
    n = pes.dofs.n_free
    H, coords, calls = local_hessian(pes, x, np.flatnonzero(pes.dofs.free))
    assert np.array_equal(coords, np.arange(3 * n))
    T = np.zeros((3 * n, 3))
    for k in range(3):
        T[k::3, k] = 1.0
    Q = null_space(T.T)
    values, V = np.linalg.eigh(Q.T @ H @ Q)
    return values, Q @ V, calls


def soft_displacement(values, modes, rng, size):
    """(N, 3) random soft-mode displacement whose most-moving atom moves `size` A."""
    c = rng.normal(size=len(values)) / np.sqrt(np.maximum(values, FLOOR))
    d = (modes @ c).reshape(-1, 3)
    return d * size / np.linalg.norm(d, axis=1).max()
