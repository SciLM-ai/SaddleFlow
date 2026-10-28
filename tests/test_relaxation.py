"""saddleflow.utils.relaxation on ASE's EMT potential (CPU, no UMA download)."""
import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms

from saddleflow.utils.relaxation import relax


def rattled_copper(seed=0):
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat(2)
    atoms.rattle(0.05, seed=seed)
    return atoms


def test_relax_reaches_threshold():
    result = relax(rattled_copper(), EMT(), fmax=0.01)
    assert result.converged and result.fmax <= 0.01
    start = rattled_copper(); start.calc = EMT()
    assert result.energy < start.get_potential_energy()


def test_fixed_mask_freezes_atoms():
    atoms = rattled_copper()
    fixed = np.zeros(len(atoms), bool); fixed[:8] = True
    result = relax(atoms, EMT(), fixed=fixed, fmax=0.01)
    assert result.converged
    assert np.allclose(result.atoms.positions[fixed], atoms.positions[fixed])
    assert not np.allclose(result.atoms.positions[~fixed], atoms.positions[~fixed])


def test_existing_constraints_are_kept():
    atoms = rattled_copper()
    atoms.set_constraint(FixAtoms(indices=range(4)))
    result = relax(atoms, EMT(), fmax=0.01)
    assert np.allclose(result.atoms.positions[:4], atoms.positions[:4])


def test_input_is_not_modified():
    atoms = rattled_copper()
    before = atoms.positions.copy()
    relax(atoms, EMT(), fmax=0.01)
    assert np.array_equal(atoms.positions, before) and atoms.calc is None


def test_relax_cell_changes_lattice():
    atoms = bulk("Cu", "fcc", a=3.4, cubic=True)        # compressed; EMT's equilibrium is about 3.59 A
    result = relax(atoms, EMT(), fmax=0.005, relax_cell=True)
    assert result.converged and result.atoms.cell.lengths()[0] > 3.5
