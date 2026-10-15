"""Select the Cartesian coordinates that ASE actually allows to move."""

import numpy as np
from ase.constraints import FixAtoms
from ase.geometry import find_mic


class FreeDOFs:
    """A fixed reference structure plus a flat vector of its free coordinates.

    Supports unconstrained structures and FixAtoms, including several FixAtoms
    constraints. Other constraints need a different projector and are rejected.
    Copying ASE Atoms preserves numbers, cell, PBC, info and constraints.
    """

    def __init__(self, atoms):
        self.template = atoms.copy()
        self.free = np.ones(len(atoms), dtype=bool)
        for constraint in atoms.constraints:
            if not isinstance(constraint, FixAtoms):
                raise NotImplementedError(
                    f"Only FixAtoms is supported, not {type(constraint).__name__}."
                )
            self.free[constraint.get_indices()] = False
        self.n_free = int(self.free.sum())
        self.dimension = 3 * self.n_free
        if self.n_free == 0:
            raise ValueError("There are no free coordinates to search.")

    def pack(self, positions):
        return np.asarray(positions)[self.free].reshape(-1).copy()

    def expand(self, x):
        positions = self.template.positions.copy()
        positions[self.free] = np.asarray(x).reshape(self.n_free, 3)
        return positions

    def atoms_at(self, x):
        atoms = self.template.copy()
        atoms.set_positions(self.expand(x), apply_constraint=False)
        return atoms

    def displacement(self, x, y):
        """MIC(x-y) in the free space; respects nonperiodic cell directions."""
        delta = (np.asarray(x) - np.asarray(y)).reshape(self.n_free, 3)
        if self.template.pbc.any():
            delta, _ = find_mic(delta, self.template.cell, self.template.pbc)
        return delta.reshape(-1)

    def rms(self, vector):
        """Per-free-atom vector RMS, NOT Cartesian-component RMS."""
        return float(np.linalg.norm(vector) / np.sqrt(self.n_free))

    def force_metrics(self, force):
        magnitudes = np.linalg.norm(np.asarray(force).reshape(self.n_free, 3), axis=1)
        return float(np.sqrt(np.mean(magnitudes ** 2))), float(magnitudes.max())
