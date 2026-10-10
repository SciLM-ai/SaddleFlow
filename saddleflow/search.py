"""Turn SaddleFlow proposals into verified first-order saddles with Sella.

    from saddleflow import SaddleFlowSearch
    atoms.calc = calculator                          # e.g. fairchem's FAIRChemCalculator for UMA-S-1.2
    SaddleFlowSearch(atoms).run(fmax=0.01)           # moves atoms to a saddle, like an ASE optimizer

or, for several noise draws with duplicates merged:

    from saddleflow.search import find_saddles
    saddles = find_saddles(atoms, calculator, n=8, seed=0)

Needs Sella (`pip install 'saddleflow[search]'`). The defaults follow the paper's protocol: first order,
fmax 0.01 eV/A, at most 300 steps, delta0 0.1, and a saddle counts as first order when exactly one
Hessian eigenvalue (over the free atoms) is below -0.01 eV/A^2.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.geometry import find_mic

_DEFAULT_MODEL = None


def _default_model():
    global _DEFAULT_MODEL
    if _DEFAULT_MODEL is None:
        from .pretrained import SaddleFlow
        _DEFAULT_MODEL = SaddleFlow.from_pretrained()
    return _DEFAULT_MODEL


def max_displacement(a: Atoms, b: Atoms) -> float:
    """Largest per-atom displacement between two structures of the same cell (minimum image)."""
    d, _ = find_mic(b.get_positions() - a.get_positions(), a.get_cell(), a.pbc)
    return float(np.linalg.norm(d, axis=1).max())


def _free_dofs(atoms: Atoms) -> np.ndarray:
    free = np.ones(len(atoms), dtype=bool)
    for c in atoms.constraints:
        if isinstance(c, FixAtoms):
            free[c.get_indices()] = False
    return (3 * np.where(free)[0][:, None] + np.arange(3)).ravel()


def hessian_eigenvalues(atoms: Atoms, k: int = 4, eps: float = 1e-3, tol: float = 1e-3,
                        maxiter: int = 300) -> np.ndarray:
    """Lowest `k` eigenvalues (eV/A^2) of the Hessian over the free atoms, from central differences of
    `atoms.calc` forces (Lanczos for large systems, the full matrix for small ones). Translations are
    not projected out; they show up as near-zero eigenvalues."""
    dofs = _free_dofs(atoms)
    probe = atoms.copy()
    probe.calc = atoms.calc
    x0 = atoms.get_positions().ravel()

    def hv(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm == 0.0:
            return np.zeros_like(v)
        out = []
        for sign in (1.0, -1.0):
            x = x0.copy()
            x[dofs] += sign * eps * v / norm
            probe.set_positions(x.reshape(-1, 3), apply_constraint=False)
            out.append(probe.get_forces(apply_constraint=False).ravel()[dofs])
        return -(out[0] - out[1]) / (2 * eps) * norm

    n = len(dofs)
    if n <= 12:
        h = np.column_stack([hv(e) for e in np.eye(n)])
        return np.linalg.eigvalsh(0.5 * (h + h.T))[:k]
    from scipy.sparse.linalg import LinearOperator, eigsh
    vals = eigsh(LinearOperator((n, n), matvec=hv, dtype=float), k=min(k, n - 1), which="SA",
                 tol=tol, maxiter=maxiter, return_eigenvectors=False)
    return np.sort(vals)


def count_negative_modes(atoms: Atoms, threshold: float = -0.01, **kw) -> int:
    """Number of Hessian eigenvalues below `threshold` (eV/A^2); 1 for a first-order saddle."""
    return int((hessian_eigenvalues(atoms, **kw) < threshold).sum())


def _sella(atoms: Atoms, fmax: float, steps: int, delta0: float, logfile) -> tuple[bool, int]:
    from sella import Sella
    opt = Sella(atoms, order=1, delta0=delta0, logfile=logfile)
    converged = bool(opt.run(fmax=fmax, steps=steps))
    return converged, int(opt.nsteps)


class SaddleFlowSearch:
    """Saddle search from a minimum, used like an ASE optimizer:

        atoms.calc = calculator
        search = SaddleFlowSearch(atoms, seed=0)
        search.run(fmax=0.01)        # True when atoms ends on a verified first-order saddle

    SaddleFlow proposes a saddle from `atoms` plus noise, `atoms` is moved there, and Sella converges it
    with `atoms.calc`. `atoms` is modified in place.
    """

    def __init__(self, atoms: Atoms, model=None, *, seed: int | None = None, sigma: float | None = None,
                 task_name: str | None = None, delta0: float = 0.1, check_order: bool = True, logfile=None):
        self.atoms, self.model = atoms, model
        self.seed, self.sigma, self.task_name = seed, sigma, task_name
        self.delta0, self.check_order, self.logfile = delta0, check_order, logfile
        self.converged, self.nsteps, self.n_negative = False, 0, None

    def run(self, fmax: float = 0.01, steps: int = 300) -> bool:
        model = self.model or _default_model()
        prop = model.propose(self.atoms, n=1, seed=self.seed, sigma=self.sigma, task_name=self.task_name)[0]
        self.atoms.set_positions(prop.get_positions(), apply_constraint=False)
        self.converged, self.nsteps = _sella(self.atoms, fmax, steps, self.delta0, self.logfile)
        if not (self.converged and self.check_order):
            return self.converged
        self.n_negative = count_negative_modes(self.atoms)
        return self.n_negative == 1


def find_saddles(atoms: Atoms, calculator, n: int = 8, *, model=None, seed: int | None = None,
                 fmax: float = 0.01, steps: int = 300, delta0: float = 0.1, check_order: bool = True,
                 dedup: float = 0.3, logfile=None, **propose_kwargs) -> list[Atoms]:
    """Propose `n` saddles from `atoms`, converge each with Sella and `calculator`, and return the
    distinct first-order saddles (two saddles closer than `dedup` A in maximum per-atom displacement
    count once). Each result records `sella_steps`, `start_to_saddle` (A), `n_negative_modes` and
    `energy` in `info`."""
    model = model or _default_model()
    found: list[Atoms] = []
    for prop in model.propose(atoms, n=n, seed=seed, **propose_kwargs):
        sad = prop.copy()
        sad.calc = calculator
        converged, nsteps = _sella(sad, fmax, steps, delta0, logfile)
        if not converged:
            continue
        sad.info.update(sella_steps=nsteps, start_to_saddle=max_displacement(prop, sad))
        if check_order:
            sad.info["n_negative_modes"] = count_negative_modes(sad)
            if sad.info["n_negative_modes"] != 1:
                continue
        if any(max_displacement(sad, other) < dedup for other in found):
            continue
        sad.info["energy"] = float(sad.get_potential_energy())
        found.append(sad)
    return found
