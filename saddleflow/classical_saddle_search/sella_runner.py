"""Constraint-preserving Sella adapter using the same PES query counter."""

from time import perf_counter

import numpy as np

from .pes_backend import QueryBudgetExceeded
from .result import finish_result, trace_row


class SearchStopped(Exception):
    """Raised by a `stop` rule to end a Sella run early (status = the rule's reason)."""


def run_sella(pes, x0, *, max_steps=100, max_force_calls=800,
              fmax_tol=0.005, delta0=0.1, callback=None, v0=None, stop=None, return_hessian=False):
    """Sella from x0 on the counted PES.

    v0: initial guess for the lowest mode, over the free degrees of freedom (pes.dofs.pack layout); only
        Sella's first iterative diagonalisation uses it.
    stop: optional rule called after every accepted step as stop(x, energy) -> reason or None; a reason
        ends the run with that status (e.g. drifting too far from the start, or too high in energy).
    return_hessian: also return the eigenpairs of Sella's final approximate Hessian, mapped to the free
        degrees of freedom ("hessian_evals", "hessian_evecs"): exact on the directions its eigensolver
        explored, approximate elsewhere.
    """
    # Optional dependency: the oracle/two-flow modules do not require Sella/JAX.
    from sella import Sella
    from sella.internal import Constraints

    pes.max_force_calls = max_force_calls
    pes.set_coordinates(x0)
    trace = [trace_row(pes, x0)]
    start = perf_counter()
    status, error = "max_steps", ""
    # Sella's Constraints imports the incoming ASE FixAtoms constraints. Passing
    # this explicitly also makes the adapter's physical search space visible.
    optimizer = None

    def record_accepted_state():
        x = pes.dofs.pack(pes.atoms.positions)
        force = pes.forces(x)
        shift = pes.dofs.rms(pes.dofs.displacement(x, trace[-1]["x"]))
        row = trace_row(pes, x, force, shift_rms=shift)
        if not np.array_equal(trace[-1]["x"], x) or not np.isfinite(trace[-1]["force_max"]):
            trace.append(row)
        if callback is not None:
            callback(optimizer.get_number_of_steps(), row)
        if stop is not None:
            reason = stop(x, float(pes.atoms.get_potential_energy()))
            if reason:
                raise SearchStopped(reason)

    try:
        constraints = Constraints(pes.atoms)
        optimizer = Sella(pes.atoms, order=1, constraints=constraints,
                          internal=False, delta0=delta0, logfile=None)
        if v0 is not None:
            # Sella keeps v0 in its own free coordinates, known only once the optimizer exists.
            full = pes.dofs.expand(np.asarray(v0, float)) - pes.dofs.expand(np.zeros_like(v0))
            optimizer.pes.v0 = full.ravel() @ optimizer.pes.get_Ufree()
        optimizer.attach(record_accepted_state, interval=1)
        optimizer.run(fmax=fmax_tol, steps=max_steps)
        record_accepted_state()
        status = "converged" if trace[-1]["force_max"] <= fmax_tol else "max_steps"
    except QueryBudgetExceeded:
        status = "query_budget"
    except SearchStopped as exc:
        status = str(exc)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        status, error = "error", str(exc)
    # A budget exception can occur at a displaced Hessian probe or rejected
    # trust-region candidate. NEVER mistake that temporary geometry for a path.
    x = trace[-1]["x"].copy()
    pes.set_coordinates(x)
    steps = optimizer.get_number_of_steps() if optimizer is not None else 0
    result = finish_result(pes, x, trace, status, steps, perf_counter() - start, error)
    if return_hessian and optimizer is not None:
        H = optimizer.pes.get_H()
        if H is not None and H.B is not None:
            evecs = H.evecs.reshape(-1, 3, H.evecs.shape[1])[pes.dofs.free].reshape(-1, H.evecs.shape[1])
            result.update(hessian_evals=np.asarray(H.evals), hessian_evecs=evecs)
    return result
