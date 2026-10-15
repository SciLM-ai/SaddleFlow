"""Constraint-preserving Sella adapter using the same PES query counter."""

from time import perf_counter

import numpy as np

from .pes_backend import QueryBudgetExceeded
from .result import finish_result, trace_row


def run_sella(pes, x0, *, max_steps=100, max_force_calls=800,
              fmax_tol=0.005, delta0=0.1, callback=None):
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

    try:
        constraints = Constraints(pes.atoms)
        optimizer = Sella(pes.atoms, order=1, constraints=constraints,
                          internal=False, delta0=delta0, logfile=None)
        optimizer.attach(record_accepted_state, interval=1)
        optimizer.run(fmax=fmax_tol, steps=max_steps)
        record_accepted_state()
        status = "converged" if trace[-1]["force_max"] <= fmax_tol else "max_steps"
    except QueryBudgetExceeded:
        status = "query_budget"
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        status, error = "error", str(exc)
    # A budget exception can occur at a displaced Hessian probe or rejected
    # trust-region candidate. NEVER mistake that temporary geometry for a path.
    x = trace[-1]["x"].copy()
    pes.set_coordinates(x)
    steps = optimizer.get_number_of_steps() if optimizer is not None else 0
    return finish_result(pes, x, trace, status, steps,
                         perf_counter() - start, error)
