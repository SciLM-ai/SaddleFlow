"""Climb out of a minimum's bowl along a chosen direction, then hand over to Sella.

Started near a minimum, Sella follows the lowest-curvature mode, which there is the minimum's softest
vibration whatever direction the start was nudged in. `climb_along` instead pushes a fixed distance along
the chosen direction d each step and relaxes every other direction downhill (the activation phase of ARTn).
It is Hessian-free: one force call per step, plus one more only when a test of the curvature is due.

Downhill step. The step perpendicular to d uses the Barzilai-Borwein step size from the last two steps, so
stiff bonds (tens of eV/A^2 in oxides) relax instead of oscillating, capped at max_perp A per atom.

Leaving the bowl. The curvature along d itself, c_d = -(F(x + eps d) - F(x)) . d / eps, bounds the lowest
Hessian eigenvalue from above, so c_d < 0 means the bowl is behind us. Along a line the restoring force
-F.d grows until the inflection point, so c_d (one extra force call) is only computed once that force has
stopped growing.

    from saddleflow.classical_saddle_search import PESBackend, climb_along, run_sella
    pes = PESBackend(atoms, calculator)
    climb = climb_along(pes, pes.dofs.pack(atoms.positions), direction)
    if climb["status"] == "left_bowl":
        result = run_sella(pes, climb["x"], fmax_tol=0.01, max_steps=300, max_force_calls=10**6, delta0=0.1)

`direction` is a vector over the free degrees of freedom (pes.dofs.pack of an (N, 3) array). For an
epicenter hint it is zero except on the epicenter atom.
"""

import numpy as np


def climb_along(pes, x0, direction, *, push=0.05, max_perp=0.1, max_steps=40, max_rise=2.0,
                max_distance=2.0, eps=0.01, alpha0=0.01):
    """Push along `direction` until the curvature along it turns negative.

    push: A moved along d per step. Perpendicular step: Barzilai-Borwein size (A^2/eV, starting at alpha0,
    kept within [1e-3, 0.1]), capped at max_perp A per atom. Stops with status "left_bowl" (hand over),
    "wall" (energy rose more than max_rise eV), "too_far" (moved more than max_distance A along d) or
    "max_steps". Returns the final point, status, steps, force calls, energy rise and the RMS force
    perpendicular to d at the last point (how relaxed the rest of the structure is at handover).
    """
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    x = np.asarray(x0, float).copy()
    calls_before = pes.force_calls
    force = pes.forces(x)
    energy0 = pes.energy(x)              # same coordinates: the calculator returns its cached result
    perp = force - np.dot(force, d) * d
    status, steps, rise, alpha, restoring = "max_steps", 0, 0.0, alpha0, -np.dot(force, d)
    for steps in range(1, max_steps + 1):
        down = alpha * perp.reshape(-1, 3)
        size = np.linalg.norm(down, axis=1, keepdims=True)
        down *= np.minimum(1.0, max_perp / np.maximum(size, 1e-12))
        x_new = x + push * d + down.ravel()
        force_new = pes.forces(x_new)
        perp_new = force_new - np.dot(force_new, d) * d
        s, y = down.ravel(), perp - perp_new          # y: change of the perpendicular gradient
        if np.dot(s, y) > 1e-12:
            alpha = float(np.clip(np.dot(s, s) / np.dot(s, y), 1e-3, 0.1))
        x, force, perp = x_new, force_new, perp_new
        rise = pes.energy(x) - energy0
        restoring_new = -np.dot(force, d)
        if restoring_new < restoring:                 # past the steepest point along d: test the curvature
            curvature = -np.dot(pes.forces(x + eps * d) - force, d) / eps
            pes.set_coordinates(x)
            if curvature < 0:
                status = "left_bowl"; break
        restoring = restoring_new
        if rise > max_rise:
            status = "wall"; break
        if np.dot(x - x0, d) > max_distance:
            status = "too_far"; break
    return dict(x=x, status=status, steps=steps, force_calls=pes.force_calls - calls_before, energy_rise=rise,
                perp_force_rms=float(np.sqrt(np.mean(np.sum(perp.reshape(-1, 3) ** 2, axis=1)))))
