"""Small, explicit UMA force guidance for inference-time terminal estimates.

The functions here do not know anything about a particular SaddleFlow model.
An inference script supplies a batch of algebraic terminal estimates,

    S_hat = x_t + (1 - t) v_theta(x_t, t),

and this module makes a *small accepted-only correction* to those estimates.
It never changes model parameters and it never participates in training.

For a free-coordinate force vector F, the force-residual objective is

    J_F = 1/2 ||F||^2.

Because F = -grad(E), its descent direction is H F.  We avoid a Hessian by a
central finite difference of ordinary UMA force evaluations.  A correction is
accepted only when a new UMA force calculation confirms that the mobile-atom
force RMS decreased.
"""

from __future__ import annotations

import torch
from fairchem.core.datasets.collaters.simple_collater import data_list_collater

from saddleflow.data.transforms import mic_displacement, wrap_positions
from saddleflow.flow.matching import build_atomic_data
from saddleflow.utils.forces import compute_uma_forces


def uma_forces(
    positions: torch.Tensor,
    *,
    Z: torch.Tensor,
    cell: torch.Tensor,
    task_name: str,
    charge: int,
    spin: int,
    fixed: torch.Tensor,
    force_backbone,
    force_head,
    force_tasks: dict,
    device: torch.device | str,
) -> torch.Tensor:
    """Return frozen-pretrained UMA forces for a batch of structures.

    ``positions`` has shape ``(batch, atoms, 3)`` and can live on CPU; the
    returned force tensor is moved back to the same device.  UMA obtains force
    by differentiating its energy, so this function necessarily uses
    ``torch.enable_grad()`` for positions.  UMA parameters remain frozen and
    no optimizer or higher-order graph is created.
    """
    original_device = positions.device
    batch_size = len(positions)
    data_list = [
        build_atomic_data(positions[index], Z, cell, task_name, charge, spin, fixed)
        for index in range(batch_size)
    ]
    atomic_batch = data_list_collater(data_list, otf_graph=True).to(device)

    # Position gradients are required for F = -dE/dx.  `create_graph=False`
    # makes this an inference calculation, not a second-order training graph.
    with torch.enable_grad():
        atomic_batch["pos"].requires_grad_(True)
        features = force_backbone(atomic_batch)
        forces = compute_uma_forces(
            atomic_batch,
            features,
            force_head,
            force_tasks,
            create_graph=False,
            retain_graph=False,
            task_name=task_name,
        )
    return forces.detach().view(batch_size, positions.shape[1], 3).to(original_device)


def force_rms_and_max(forces: torch.Tensor, mobile_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mobile-atom force RMS and f_max, both in eV/A, per structure."""
    mobile_forces = forces[:, mobile_mask, :]
    squared_norm_per_atom = (mobile_forces ** 2).sum(dim=-1)
    force_rms = torch.sqrt(squared_norm_per_atom.mean(dim=1))
    force_max = torch.sqrt(squared_norm_per_atom).max(dim=1).values
    return force_rms, force_max


def mobile_vector_rms(vectors: torch.Tensor, mobile_mask: torch.Tensor) -> torch.Tensor:
    """RMS length of a vector field over free atoms, one scalar per structure."""
    mobile_vectors = vectors[:, mobile_mask, :]
    return torch.sqrt((mobile_vectors ** 2).sum(dim=-1).mean(dim=1))


def estimate_hf_direction(
    terminal: torch.Tensor,
    forces: torch.Tensor,
    mobile_mask: torch.Tensor,
    cell: torch.Tensor,
    force_function,
    fd_eps: float,
) -> torch.Tensor:
    """Estimate the force-residual descent direction H F by finite differences.

    The finite-difference displacement is only on mobile atoms.  Let
    ``u = F / ||F||`` on the flattened mobile coordinate vector.  Since
    ``dF/dx = -H``, the returned direction is

        H F ~= -||F|| [F(x + eps*u) - F(x - eps*u)] / (2*eps).

    Fixed atoms are exactly zero in the returned tensor.
    """
    if fd_eps <= 0:
        raise ValueError("fd_eps must be positive.")

    # This is the flattened Euclidean norm in the formula above, not force RMS.
    free_forces = forces[:, mobile_mask, :]
    force_l2 = torch.linalg.vector_norm(free_forces.reshape(len(forces), -1), dim=1)
    unit_force = torch.zeros_like(forces)
    # Indexing a boolean-selected batch would make a temporary tensor, so fill
    # the mobile slice directly.  Zero-force rows stay exactly zero.
    unit_force[:, mobile_mask, :] = free_forces / force_l2.clamp(min=1e-12)[:, None, None]

    plus = wrap_positions(terminal + fd_eps * unit_force, cell)
    minus = wrap_positions(terminal - fd_eps * unit_force, cell)
    force_plus = force_function(plus)
    force_minus = force_function(minus)
    hf = -force_l2[:, None, None] * (force_plus - force_minus) / (2.0 * fd_eps)
    hf[:, ~mobile_mask, :] = 0.0
    return hf


def correct_terminal(
    terminal: torch.Tensor,
    hf_direction: torch.Tensor,
    force_before: torch.Tensor,
    mobile_mask: torch.Tensor,
    cell: torch.Tensor,
    force_function,
    endpoint_step: float,
    max_endpoint_step: float,
    max_backtracks: int,
    backtrack_factor: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Try a normalized H F endpoint shift, accepting only force-reducing moves.

    ``endpoint_step`` is an interpretable RMS displacement in Angstrom.  The
    finite-difference H F magnitude is used only to choose direction.  Every
    rejected attempt halves its displacement; no rejected endpoint is used.
    """
    if endpoint_step <= 0 or max_endpoint_step <= 0:
        raise ValueError("endpoint_step and max_endpoint_step must be positive.")
    if max_backtracks < 0 or not 0 < backtrack_factor < 1:
        raise ValueError("Use max_backtracks >= 0 and 0 < backtrack_factor < 1.")

    batch_size = len(terminal)
    direction_rms = mobile_vector_rms(hf_direction, mobile_mask)
    usable_direction = direction_rms > 1e-12
    direction = torch.zeros_like(hf_direction)
    direction[usable_direction] = (
        hf_direction[usable_direction] / direction_rms[usable_direction, None, None]
    )
    direction[:, ~mobile_mask, :] = 0.0

    corrected = terminal.clone()
    force_after = force_before.clone()
    fmax_after = torch.full_like(force_before, float("nan"))
    accepted = torch.zeros(batch_size, dtype=torch.bool, device=terminal.device)
    attempts = torch.zeros(batch_size, dtype=torch.int64, device=terminal.device)
    step_size = torch.full(
        (batch_size,), min(endpoint_step, max_endpoint_step), dtype=terminal.dtype, device=terminal.device
    )

    # All pending structures share one UMA batch per backtracking attempt.
    # The comparison is always with the original endpoint force, which enforces
    # the simple safety property: accepted guidance reduces force residual.
    pending = usable_direction.clone()
    for attempt in range(max_backtracks + 1):
        if not pending.any():
            break
        pending_indices = torch.where(pending)[0]
        candidate = wrap_positions(
            terminal[pending_indices] + step_size[pending_indices, None, None] * direction[pending_indices],
            cell,
        )
        candidate_forces = force_function(candidate)
        candidate_rms, candidate_fmax = force_rms_and_max(candidate_forces, mobile_mask)
        improves = candidate_rms < force_before[pending_indices]

        accepted_indices = pending_indices[improves]
        if len(accepted_indices):
            corrected[accepted_indices] = candidate[improves]
            force_after[accepted_indices] = candidate_rms[improves]
            fmax_after[accepted_indices] = candidate_fmax[improves]
            accepted[accepted_indices] = True
            attempts[accepted_indices] = attempt

        rejected_indices = pending_indices[~improves]
        if len(rejected_indices):
            step_size[rejected_indices] *= backtrack_factor
        pending = torch.zeros_like(pending)
        pending[rejected_indices] = True

    endpoint_shift = mic_displacement(corrected, terminal, cell)
    endpoint_shift[:, ~mobile_mask, :] = 0.0
    shift_rms = mobile_vector_rms(endpoint_shift, mobile_mask)
    shift_max = torch.sqrt((endpoint_shift[:, mobile_mask, :] ** 2).sum(dim=-1)).max(dim=1).values
    return corrected, {
        "force_after": force_after,
        "fmax_after": fmax_after,
        "accepted": accepted,
        "backtracks": attempts,
        "shift_rms": shift_rms,
        "shift_max": shift_max,
    }


def guide_terminal_estimates(
    terminal: torch.Tensor,
    *,
    mobile_mask: torch.Tensor,
    cell: torch.Tensor,
    force_function,
    force_tol: float,
    fd_eps: float,
    endpoint_step: float,
    max_endpoint_step: float,
    max_backtracks: int,
    backtrack_factor: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply one optional, thresholded UMA force-guidance event to a batch.

    The returned dictionary is intentionally plain tensors so a caller can save
    a transparent per-trajectory trace without depending on a custom class.
    """
    if force_tol < 0:
        raise ValueError("force_tol must be non-negative.")

    all_forces = force_function(terminal)
    force_before, fmax_before = force_rms_and_max(all_forces, mobile_mask)
    should_correct = force_before > force_tol
    corrected = terminal.clone()
    force_after = force_before.clone()
    accepted = torch.zeros(len(terminal), dtype=torch.bool, device=terminal.device)
    backtracks = torch.zeros(len(terminal), dtype=torch.int64, device=terminal.device)
    shift_rms = torch.zeros(len(terminal), dtype=terminal.dtype, device=terminal.device)
    shift_max = torch.zeros(len(terminal), dtype=terminal.dtype, device=terminal.device)

    # Good endpoints bypass every finite-difference and line-search force call.
    if should_correct.any():
        active_indices = torch.where(should_correct)[0]
        active_terminal = terminal[active_indices]
        active_forces = all_forces[active_indices]

        def active_force_function(active_positions: torch.Tensor) -> torch.Tensor:
            return force_function(active_positions)

        hf = estimate_hf_direction(
            active_terminal, active_forces, mobile_mask, cell, active_force_function, fd_eps
        )
        active_corrected, details = correct_terminal(
            active_terminal,
            hf,
            force_before[active_indices],
            mobile_mask,
            cell,
            active_force_function,
            endpoint_step,
            max_endpoint_step,
            max_backtracks,
            backtrack_factor,
        )
        corrected[active_indices] = active_corrected
        force_after[active_indices] = details["force_after"]
        accepted[active_indices] = details["accepted"]
        backtracks[active_indices] = details["backtracks"]
        shift_rms[active_indices] = details["shift_rms"]
        shift_max[active_indices] = details["shift_max"]
        accepted_active = details["accepted"]
        fmax_after = fmax_before.clone()
        fmax_after[active_indices[accepted_active]] = details["fmax_after"][accepted_active]
    else:
        fmax_after = fmax_before.clone()

    return corrected, {
        "force_before": force_before,
        "fmax_before": fmax_before,
        "force_after": force_after,
        "fmax_after": fmax_after,
        "checked": torch.ones(len(terminal), dtype=torch.bool, device=terminal.device),
        "above_tolerance": should_correct,
        "accepted": accepted,
        "backtracks": backtracks,
        "shift_rms": shift_rms,
        "shift_max": shift_max,
    }
