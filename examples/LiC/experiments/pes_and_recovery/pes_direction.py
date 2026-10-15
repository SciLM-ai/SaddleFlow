"""Detached local PES teachers, not endpoint penalties or inference guidance.

All vectors live in the full free-coordinate space. No bond axes, species
assumptions, or saddle labels enter the teacher. The ASE backend currently
supports unconstrained structures and FixAtoms, and rejects other constraints.
"""

import numpy as np
import torch

from saddleflow.classical_saddle_search import MinimumModeEstimator


def query_teacher(pes, positions, kind="mmf", fd_eps=0.002, seed=0):
    """Observe frozen force and optionally the numerical minimum mode at x_t.

    `pes` is the existing generic PESBackend for this structure. Coordinate
    autograd may be needed INSIDE UMA to obtain forces; its NumPy observations
    do not carry a backward graph to the student or potential parameters.
    Small free spaces use dense FD; larger ones use the existing Lanczos/HVP.
    """
    if kind not in ("force", "mmf"):
        raise ValueError("Teacher must be force or mmf")
    x = pes.dofs.pack(positions.detach().cpu().numpy())
    before = pes.force_calls
    with torch.enable_grad():
        force = pes.forces(x)
        eigs, q, asymmetry = None, None, None
        if kind == "mmf":
            estimator = MinimumModeEstimator(eps=fd_eps, seed=seed)
            eigs, vectors, asymmetry = estimator.lowest_modes(pes, x, n_modes=2)
            q = vectors[:, 0]
            teacher = force - 2 * np.dot(q, force) * q
        else:
            teacher = force.copy()
    full = np.zeros((len(pes.dofs.free), 3))
    full[pes.dofs.free] = teacher.reshape(-1, 3)
    rms, fmax = pes.dofs.force_metrics(force)
    return dict(vector=torch.as_tensor(full, device=positions.device, dtype=positions.dtype),
                force=force, q=q, eigs=eigs, force_rms=rms, force_max=fmax,
                force_calls=pes.force_calls-before, hessian_asymmetry=asymmetry)


def directional_loss(velocity, teacher, free_mask, *, force_rms,
                     teacher_deadband, velocity_floor=1e-4, force_floor=1e-6):
    """One system's smooth cosine, masked by the TEACHER force only.

    Floors are in per-free-atom vector RMS units (A per FM time, eV/A).
    Dividing both flattened vectors by sqrt(N_free) makes them independent
    of atom count and is equivalent to scaling the norm floors by sqrt(N_free).
    Fixed atom forces never influence the direction, mask, or loss.
    """
    free = torch.as_tensor(free_mask, device=velocity.device, dtype=torch.bool)
    count = int(free.sum())
    if count == 0:
        return velocity.sum() * 0.0
    if float(force_rms) <= teacher_deadband:
        return velocity.sum() * 0.0
    v = velocity[free].reshape(-1) / count**0.5
    b = teacher.detach().to(velocity)[free].reshape(-1) / count**0.5
    denominator = torch.sqrt(v.square().sum() + velocity_floor**2)
    denominator = denominator * torch.sqrt(b.square().sum() + force_floor**2)
    return 1.0 - torch.dot(v, b) / denominator


def alignment(target, teacher):
    """Diagnostic cosine; a zero-vector direction is undefined, not zero."""
    a, b = np.asarray(target).reshape(-1), np.asarray(teacher).reshape(-1)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b)/denominator) if denominator > 1e-14 else np.nan
