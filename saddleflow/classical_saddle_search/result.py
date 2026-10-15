"""Small dictionary helpers shared by the solvers and independent evaluator."""

import numpy as np


def trace_row(pes, x, force=None, *, q=None, eigs=None, shift_rms=0.0, asymmetry=np.nan):
    rms, fmax = pes.dofs.force_metrics(force) if force is not None else (np.nan, np.nan)
    parallel = float(abs(np.dot(q, force))) if q is not None and force is not None else np.nan
    perpendicular = (float(np.linalg.norm(force - np.dot(q, force) * q))
                     if q is not None and force is not None else np.nan)
    return dict(x=np.asarray(x).copy(), force_rms=rms, force_max=fmax,
                query_count=pes.force_calls, q=None if q is None else q.copy(),
                eigs=None if eigs is None else np.asarray(eigs).copy(),
                force_parallel=parallel, force_perpendicular=perpendicular,
                shift_rms=float(shift_rms), hessian_asymmetry=float(asymmetry))


def finish_result(pes, x, trace, status, steps, seconds, error=""):
    return dict(final=np.asarray(x).copy(), positions=pes.dofs.expand(x), trace=trace,
                status=status, steps=steps, force_calls=pes.force_calls,
                wall_seconds=seconds, error=error)


def evaluate_endpoint(pes, x, estimator, fmax_tol=0.005, eigenvalue_tol=0.01):
    """Independent force/index check, explicitly OUTSIDE the search query budget.

    Strict index-one certificate: lambda1 < -tol and lambda2 > tol. This is more
    stringent than counting negative eigenvalues when soft/translation modes
    exist; such results are not certified and their eigenvalues remain visible.
    """
    before = pes.force_calls
    old_budget = pes.max_force_calls
    pes.max_force_calls = None
    try:
        force = pes.forces(x)
        rms, fmax = pes.dofs.force_metrics(force)
        eigs, _, asymmetry = estimator.lowest_modes(pes, x, n_modes=2)
        index_one = bool(len(eigs) >= 2 and eigs[0] < -eigenvalue_tol and eigs[1] > eigenvalue_tol)
        return dict(force_rms=rms, force_max=fmax, eigs=eigs,
                    index_one=index_one, stationary=bool(fmax <= fmax_tol),
                    physical_success=bool(fmax <= fmax_tol and index_one),
                    hessian_asymmetry=asymmetry, diagnostic_force_calls=pes.force_calls - before)
    finally:
        pes.max_force_calls = old_budget
        pes.set_coordinates(x)
