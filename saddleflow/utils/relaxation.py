"""Relax a structure to a nearby local minimum with an ASE optimizer (UMA by default).

Simplified port of `evaluation/uma_relaxation.py` from ConstrDiff_Material: one function, `relax`, and a
helper that builds the UMA calculator. Fixed cell by default; positions only.

    from saddleflow.utils.relaxation import relax, uma_calculator
    calc = uma_calculator("uma-s-1p2", task_name="omat")
    result = relax(atoms, calc)               # general case: the structure's own constraints apply
    result = relax(atoms, calc, fixed=mask)   # explicit frozen atoms, e.g. the LiC carbons (True = fixed)
    result.atoms, result.energy, result.fmax, result.converged

Frozen atoms. With `fixed=None` the copy keeps whatever constraints `atoms` carries; the LiC frames already
store their carbons as FixAtoms, so they relax correctly with no mask. A `fixed` mask replaces the
constraints with FixAtoms on exactly those atoms (use it for structures read without constraints, or with
a record's `fixed` array). Either way the optimizer and the reported `fmax` see only the mobile atoms.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS, FIRE, LBFGS

OPTIMIZERS = {"FIRE": FIRE, "LBFGS": LBFGS, "BFGS": BFGS}


@dataclass
class RelaxResult:
    atoms: Atoms        # relaxed copy, with its constraints and the final energy/forces attached
    energy: float       # eV
    fmax: float         # largest per-atom force norm over mobile atoms, eV/A
    steps: int          # optimizer steps taken
    converged: bool     # the optimizer's criterion was met (forces, plus stress with relax_cell)


def uma_calculator(model: str = "uma-s-1p2", task_name: str = "omat", device: str | None = None):
    """An ASE calculator for a pretrained UMA model. The task must match the data (omat, oc20, ...)."""
    import torch
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return FAIRChemCalculator(pretrained_mlip.get_predict_unit(model, device=device), task_name=task_name)


def relax(atoms: Atoms, calculator, *, fixed=None, fmax: float = 0.01, steps: int = 500,
          optimizer: str = "FIRE", relax_cell: bool = False, trajectory: str | None = None,
          logfile: str | None = None) -> RelaxResult:
    """Relax a copy of `atoms` until every mobile atom's force is below `fmax` (eV/A) or `steps` is reached.

    `fixed` (bool array of length N) replaces the structure's constraints with FixAtoms on those atoms;
    None keeps the constraints it already has. `relax_cell` also relaxes the cell (needs stress, which
    the UMA omat task provides). `trajectory` and `logfile` are passed to the ASE optimizer.
    """
    work = atoms.copy()
    if fixed is not None:
        fixed = np.asarray(fixed, dtype=bool)
        if fixed.shape != (len(work),):
            raise ValueError(f"fixed has shape {fixed.shape}, expected ({len(work)},)")
        work.set_constraint(FixAtoms(mask=fixed))
    work.calc = calculator

    target = FrechetCellFilter(work) if relax_cell else work
    kwargs = {"maxstep": 0.2} if optimizer in ("LBFGS", "BFGS") else {}
    opt = OPTIMIZERS[optimizer](target, trajectory=trajectory, logfile=logfile, **kwargs)
    opt.run(fmax=fmax, steps=steps)

    energy = float(work.get_potential_energy())
    forces = work.get_forces()                     # constrained: frozen atoms report zero force
    largest = float(np.linalg.norm(forces, axis=1).max()) if len(work) else 0.0
    # With a cell filter the optimizer's "forces" also hold the stress terms, so test convergence on those.
    criterion = float(np.linalg.norm(target.get_forces(), axis=1).max()) if len(work) else 0.0
    work.calc = SinglePointCalculator(work, energy=energy, forces=forces)
    return RelaxResult(atoms=work, energy=energy, fmax=largest, steps=int(opt.nsteps),
                       converged=criterion <= fmax)
