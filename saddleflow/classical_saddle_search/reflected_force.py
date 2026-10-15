"""Reflected-force translation with oracle or dynamically rotated orientation."""

from time import perf_counter

import numpy as np

from .minimum_mode import MinimumModeEstimator
from .pes_backend import QueryBudgetExceeded
from .result import finish_result, trace_row


def reflected_force(force, q):
    return force - 2.0 * np.dot(q, force) * q


def rotate_orientation(q, hq, step_size):
    """Euler step of qdot=-(I-qq.T)Hq, followed by normalization."""
    updated = q - step_size * (hq - np.dot(q, hq) * q)
    return updated / np.linalg.norm(updated)


def run_mode_flow(pes, x0, *, orientation="oracle", estimator=None,
                  orientation_backend="dense", orientation_step_size=0.1,
                  rotations_per_step=3, seed=0, step_size=0.2, max_step=0.1,
                  max_steps=100, max_force_calls=800, fmax_tol=0.005,
                  eigenvalue_tol=0.01, callback=None):
    """Integrate a physical search, not a normalized FM-time trajectory.

    step_size and orientation_step_size have units A^2/eV. max_step caps the
    per-free-atom RMS displacement (A). There is deliberately no energy-descent
    line search: a saddle search must be allowed to climb in one direction.

    Two-flow q is initialized independently of labels. At an EXACT stationary
    minimum translation is zero; report stationary_not_index1. A merely small
    force at a minimum does not stop the search: the reflected flow can escape.
    """
    if step_size <= 0 or max_step <= 0 or max_steps < 0 or max_force_calls < 1:
        raise ValueError("Positive step sizes/query budget and nonnegative steps are required.")
    if orientation not in ("oracle", "two_flow") or orientation_backend not in ("dense", "hvp"):
        raise ValueError("Unknown orientation algorithm/backend.")
    if orientation_step_size <= 0 or rotations_per_step < 1:
        raise ValueError("Orientation step size and number of rotations must be positive.")
    estimator = estimator or MinimumModeEstimator(seed=seed)
    pes.max_force_calls = max_force_calls
    x = np.asarray(x0, dtype=float).copy()
    rng = np.random.default_rng(seed)
    q = rng.normal(size=pes.dofs.dimension)
    q /= np.linalg.norm(q)
    trace = [trace_row(pes, x)]
    status, error, steps = "max_steps", "", 0
    last_shift = 0.0
    start = perf_counter()
    try:
        for step in range(max_steps + 1):
            force = pes.forces(x)
            _, fmax = pes.dofs.force_metrics(force)
            eigs, asymmetry = None, np.nan
            # The oracle recomputes q each step; two-flow only uses eigenvalues
            # to verify stationarity, not to supply/initialize its orientation.
            if orientation == "oracle":
                eigs, vectors, asymmetry = estimator.lowest_modes(pes, x)
                q = vectors[:, 0]
            elif orientation_backend == "dense":
                hessian, asymmetry = pes.free_hessian(x, estimator.eps)
                eigs = np.linalg.eigvalsh(hessian)[:2]  # diagnostic only
                for _ in range(rotations_per_step):
                    q = rotate_orientation(q, hessian @ q, orientation_step_size)
            else:
                if fmax <= fmax_tol:
                    eigs, _, asymmetry = estimator.lowest_modes(pes, x)
                for _ in range(rotations_per_step):
                    hq = pes.hvp(x, q, estimator.eps)
                    q = rotate_orientation(q, hq, orientation_step_size)

            row = trace_row(pes, x, force, q=q, eigs=eigs,
                            shift_rms=last_shift, asymmetry=asymmetry)
            trace.append(row)
            if callback is not None:
                callback(step, row)
            if fmax <= fmax_tol:
                index_one = len(eigs) >= 2 and eigs[0] < -eigenvalue_tol and eigs[1] > eigenvalue_tol
                if index_one:
                    status = "converged"
                    break
                if np.linalg.norm(force) < 1e-12:
                    status = "stationary_not_index1"
                    break
            if step == max_steps:
                break
            delta = step_size * reflected_force(force, q)
            rms = pes.dofs.rms(delta)
            if rms > max_step:
                delta *= max_step / rms
            last_shift = pes.dofs.rms(delta)
            x = x + delta  # unwrapped coordinates; ASE handles PBC for evaluation
            steps = step + 1
            # This state is available immediately, without paying for the NEXT
            # state's force/mode. Needed for exact anytime query-budget tables.
            trace.append(trace_row(pes, x, shift_rms=last_shift))
    except QueryBudgetExceeded:
        status = "query_budget"
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        status, error = "error", str(exc)
    # A translation can use the last permitted query. Keep the translated state
    # as a usable checkpoint even if its force has not yet been evaluated.
    if not np.array_equal(trace[-1]["x"], x):
        trace.append(trace_row(pes, x, shift_rms=last_shift))
    pes.set_coordinates(x)
    return finish_result(pes, x, trace, status, steps, perf_counter() - start, error)


def run_oracle_mmf(pes, x0, **options):
    return run_mode_flow(pes, x0, orientation="oracle", **options)
