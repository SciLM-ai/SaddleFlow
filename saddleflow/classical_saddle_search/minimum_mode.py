"""Lowest free-space modes without requiring a full-system Hessian."""

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh


class MinimumModeEstimator:
    """Small spaces: symmetrized FD Hessian. Large spaces: Lanczos + FD HVPs.

    'Oracle' means the numerical lowest mode of the chosen potential, not an
    analytic Hessian or a stored reference-saddle eigenvector.
    """

    def __init__(self, backend="auto", eps=0.002, dense_max_dimension=12,
                 lanczos_tol=1e-3, lanczos_maxiter=100, seed=0, ncv=None):
        if backend not in ("auto", "dense", "hvp"):
            raise ValueError("Mode backend must be auto, dense, or hvp.")
        if eps <= 0:
            raise ValueError("Finite-difference displacement must be positive.")
        self.backend = backend
        self.eps = eps
        self.dense_max_dimension = dense_max_dimension
        self.lanczos_tol = lanczos_tol
        self.lanczos_maxiter = lanczos_maxiter
        self.rng = np.random.default_rng(seed)
        self.ncv = ncv              # Lanczos subspace size; ARPACK's default (2k + 1) restarts often

    def use_dense(self, dimension, n_modes=2):
        return (self.backend == "dense" or dimension <= n_modes or
                (self.backend == "auto" and dimension <= self.dense_max_dimension))

    def lowest_modes(self, pes, x, n_modes=2):
        dimension = pes.dofs.dimension
        if self.use_dense(dimension, n_modes):
            hessian, asymmetry = pes.free_hessian(x, self.eps)
            values, vectors = np.linalg.eigh(hessian)
            return values[:n_modes], vectors[:, :n_modes], asymmetry
        operator = LinearOperator(
            (dimension, dimension), matvec=lambda v: pes.hvp(x, v, self.eps), dtype=float
        )
        values, vectors = eigsh(
            operator, k=min(n_modes, dimension - 1), which="SA",
            v0=self.rng.normal(size=dimension), tol=self.lanczos_tol,
            maxiter=self.lanczos_maxiter, ncv=None if self.ncv is None else min(self.ncv, dimension - 1),
        )
        order = np.argsort(values)
        return values[order], vectors[:, order], float("nan")
