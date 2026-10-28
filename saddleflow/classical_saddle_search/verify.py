"""Cheap curvature tools on a counted PES: lowest modes without a full Hessian, and local Hessian blocks.

local_hessian uses forward differences of the forces (Sella, Eq. 12): with F0 at x known, one more force call
per coordinate, H e_i ~ -(F(x + eps e_i) - F0) / eps. lowest_modes_block defaults to central differences
(2 calls per product): in stiff solids the forward-difference error (step x third derivative) reaches tenths
of an eV/A^2 along strongly curved modes at a 0.01 A step, and the slightly nonlinear operator also keeps
the block solver from converging.

lowest_modes_block
    The lowest k eigenpairs over the free degrees of freedom by LOBPCG, started from a block of vectors
    (e.g. the next eigenvectors of Sella's approximate Hessian) and with the three uniform translations
    projected out when no atom is fixed (they are exact zero modes of a periodic cell and otherwise stall
    the solver). A Ritz value with residual r guarantees an eigenvalue within |r| of it, not that no lower
    one exists, so the block start is what makes a hidden extra negative curvature unlikely to be missed.

local_hessian
    The dense Hessian block of a cluster of atoms with the rest of the structure held fixed: one forward
    difference per cluster coordinate. Holding the surroundings fixed makes every cluster motion cost
    energy, so the block has no zero modes; its modes are stiffer than the fully relaxed ones.
"""

import numpy as np


def translation_basis(pes):
    """Orthonormal uniform translations over the free degrees of freedom, or none if any atom is fixed."""
    if not np.all(pes.dofs.free):
        return np.zeros((pes.dofs.dimension, 0))
    n = pes.dofs.n_free
    T = np.zeros((n, 3, 3))
    for k in range(3):
        T[:, k, k] = 1.0
    return T.reshape(3 * n, 3) / np.sqrt(n)


def _hvp(pes, x, f0, v, eps, central):
    norm = np.linalg.norm(v)
    if norm < 1e-14:
        return np.zeros_like(v)
    u = v / norm
    if central:
        return -(pes.forces(x + eps * u) - pes.forces(x - eps * u)) * norm / (2 * eps)
    return -(pes.forces(x + eps * u) - f0) * norm / eps


def lowest_modes_block(pes, x, *, start=None, k=3, eps=0.002, tol=5e-3, maxiter=30, central=True, seed=0):
    """Lowest k (eigenvalues, vectors, force calls) at x, translations removed. `start`: (dim, >=k) or None.

    tol is the absolute residual norm in eV/A^2; the eigenvalue error is about residual^2 / gap."""
    from scipy.sparse.linalg import LinearOperator, lobpcg
    before = pes.force_calls
    x = np.asarray(x, float)
    f0 = None if central else pes.forces(x)
    T = translation_basis(pes)
    dim = len(x)
    rng = np.random.default_rng(seed)
    X = np.asarray(start, float)[:, :k] if start is not None else rng.normal(size=(dim, k))
    if X.shape[1] < k:
        X = np.hstack([X, rng.normal(size=(dim, k - X.shape[1]))])
    X = X - T @ (T.T @ X)

    def matmat(V):
        V = np.asarray(V)
        V = V[:, None] if V.ndim == 1 else V
        out = np.column_stack([_hvp(pes, x, f0, V[:, j], eps, central) for j in range(V.shape[1])])
        return out - T @ (T.T @ out) if T.shape[1] else out

    A = LinearOperator((dim, dim), matvec=lambda v: matmat(v.reshape(-1, 1))[:, 0], matmat=matmat, dtype=float)
    values, vectors = lobpcg(A, X, Y=T if T.shape[1] else None, largest=False, tol=tol, maxiter=maxiter)
    pes.set_coordinates(x)
    order = np.argsort(values)
    return values[order], vectors[:, order], pes.force_calls - before


def central_curvature(pes, x, v, eps=0.01):
    """v^T H v / v^T v by central differences (2 force calls); a check on the forward-difference values."""
    v = np.asarray(v, float) / np.linalg.norm(v)
    c = -np.dot(pes.forces(x + eps * v) - pes.forces(x - eps * v), v) / (2 * eps)
    pes.set_coordinates(x)
    return float(c)


def local_hessian(pes, x, atoms, eps=0.01):
    """(H_block, coordinate indices, force calls): the forward-difference Hessian over `atoms` (free indices)."""
    before = pes.force_calls
    x = np.asarray(x, float)
    f0 = pes.forces(x)
    free_index = {int(a): i for i, a in enumerate(np.flatnonzero(pes.dofs.free))}
    coords = np.array([3 * free_index[int(a)] + c for a in atoms for c in range(3)])
    H = np.empty((len(coords), len(coords)))
    for col, i in enumerate(coords):
        step = np.zeros_like(x); step[i] = eps
        H[:, col] = -(pes.forces(x + step) - f0)[coords] / eps
    pes.set_coordinates(x)
    return 0.5 * (H + H.T), coords, pes.force_calls - before
