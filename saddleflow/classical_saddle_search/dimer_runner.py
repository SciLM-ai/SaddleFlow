"""ASE's dimer method (Henkelman and Jonsson, JCP 111, 7010 (1999)) on the counted PES, always explicitly seeded."""

from time import perf_counter

import numpy as np
from ase.mep.dimer import DimerControl, MinModeAtoms, MinModeTranslate

from .pes_backend import QueryBudgetExceeded
from .result import finish_result, trace_row
from .sella_runner import SearchStopped

# DimerControl keeps its parameters in a CLASS dict that set_parameter() mutates, so one run's settings leak
# into every later DimerControl in the process. Each run gets its own copy of ASE's pristine defaults instead.
_ASE_PARAMETERS = dict(DimerControl.parameters)
# ASE's defaults except the image separation. ASE uses 1e-4 A, where UMA's float32 force noise (~1e-6 eV/A on
# every call) becomes ~1e-2 eV/A^2 in the curvature; 0.01 A is the VTST default (DimerDR).
DIMER_DEFAULTS = dict(dimer_separation=0.01)


def run_dimer(pes, x0, *, seed, max_steps=100, max_force_calls=800, fmax_tol=0.005,
              callback=None, v0=None, stop=None, **dimer_params):
    """ASE's MinModeAtoms + MinModeTranslate from x0 on the counted PES.

    seed: required; seeds the dimer's random initial mode. ASE seeds from the clock when it is not given, so an
        unseeded run would be a different search every time.
    v0: initial mode over the free degrees of freedom (pes.dofs.pack layout); None draws ASE's random 'gauss'
        mode from the seeded generator.
    stop: optional rule called at every dimer centre as stop(x, energy) -> reason or None (as in run_sella).
    dimer_params: DimerControl parameters on top of DIMER_DEFAULTS (e.g. max_num_rot, maximum_translation).

    Converged means the true (not projected) fmax <= fmax_tol at negative dimer curvature. ASE's own optimizer
    loop tests the projected forces, which can differ per atom from the true ones, so the loop is run here.
    """
    if seed is None:
        raise ValueError("run_dimer needs an explicit seed.")
    unknown = set(dimer_params) - set(_ASE_PARAMETERS)
    if unknown:
        raise ValueError(f"Unknown DimerControl parameters: {sorted(unknown)}")
    pes.max_force_calls = max_force_calls
    pes.set_coordinates(x0)
    trace = [trace_row(pes, x0)]
    start = perf_counter()
    status, error, steps = "max_steps", "", 0
    control = DimerControl(logfile=None)
    control.parameters = {**_ASE_PARAMETERS, **DIMER_DEFAULTS, **dimer_params, "mask": pes.dofs.free.tolist()}
    eigenmodes = None
    if v0 is not None:
        mode = pes.dofs.expand(np.asarray(v0, float)) - pes.dofs.expand(np.zeros_like(v0))
        eigenmodes = [mode / np.linalg.norm(mode)]
    curvature, q = np.nan, None
    try:
        dimer = MinModeAtoms(pes.atoms, control, eigenmodes=eigenmodes, random_seed=int(seed))
        translate = MinModeTranslate(dimer, logfile=None)
        for step in range(max_steps + 1):
            projected = dimer.get_forces()      # forces at the centre, then the dimer's rotation there
            x = pes.dofs.pack(dimer.get_positions())
            force = pes.dofs.pack(dimer.forces0)
            q = pes.dofs.pack(dimer.get_eigenmode())
            q = q / np.linalg.norm(q)
            curvature = float(dimer.get_curvature())
            shift = pes.dofs.rms(pes.dofs.displacement(x, trace[-1]["x"]))
            row = trace_row(pes, x, force, q=q, eigs=[curvature], shift_rms=shift)
            row["rotations"] = control.get_counter("rotcount")     # rotations made at this centre
            row["energy"] = float(dimer.energy0)
            trace.append(row)
            if callback is not None:
                callback(step, row)
            if stop is not None:
                reason = stop(x, row["energy"])
                if reason:
                    raise SearchStopped(reason)
            if row["force_max"] <= fmax_tol and curvature < 0:
                status = "converged"
                break
            if step == max_steps:
                break
            translate.step(projected)
            steps = step + 1
    except QueryBudgetExceeded:
        status = "query_budget"
    except SearchStopped as exc:
        status = str(exc)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        status, error = "error", str(exc)
    # A budget exception can occur at a dimer image or trial step. NEVER mistake that geometry for a path.
    x = trace[-1]["x"].copy()
    pes.set_coordinates(x)
    result = finish_result(pes, x, trace, status, steps, perf_counter() - start, error)
    result.update(curvature=curvature, eigenmode=q, seed=int(seed))
    return result
