"""UMA force evaluation for optional flow-matching training constraints.

This module implements one deliberately small training-time idea:

    S_hat = x_t + (1 - t) v_theta(x_t, t)

is the terminal structure implied by the current velocity prediction for the
straight/rectified flow path.  A frozen UMA model evaluates the force at that
structure, and a dead-band loss penalizes only predictions whose *free*
coordinates have clearly non-stationary forces.

The direct force-penalty mode keeps the terminal positions connected.
Consequently its graph is

    model parameters -> velocity -> S_hat -> UMA force -> force loss.

Backpropagation therefore uses an implicit UMA Hessian-vector product; it does
not build or diagonalize a full Hessian.

The optional ``stopgrad`` mode instead evaluates UMA on detached terminal
positions and returns a detached force-violation gate.  The caller can use
that number to reweight the ordinary supervised FM loss.  UMA then decides
how strongly an example is trained, never which direction the flow weights
are updated.  The functions are intentionally plain and accept ordinary
tensors plus the existing SaddleFlow sample dicts.  They are usable for any
number of atoms, not just the Li/C example.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import torch
from ase import Atoms
from ase.constraints import FixAtoms
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.collaters.simple_collater import data_list_collater

from saddleflow.data.transforms import mic_displacement
from saddleflow.utils.forces import compute_uma_forces


def free_coordinate_mask(sample: dict, device: torch.device) -> torch.Tensor:
    """Return the degrees of freedom that this sample is allowed to generate.

    Current SaddleFlow datasets provide ``fixed`` as an atom-level ``(N,)``
    `FixAtoms` mask.  That becomes three free/fixed Cartesian coordinates per
    atom.  The optional ``free_coordinate_mask`` key supports a future or
    external dataset with partial Cartesian constraints directly as ``(N, 3)``.
    A sample with neither key means every Cartesian coordinate is free.
    """
    if "free_coordinate_mask" in sample:
        mask = sample["free_coordinate_mask"].to(device=device, dtype=torch.bool)
        if mask.ndim != 2 or mask.shape[-1] != 3:
            raise ValueError("free_coordinate_mask must have shape (N, 3)")
        return mask
    if "fixed" in sample:
        fixed = sample["fixed"].to(device=device, dtype=torch.bool)
        if fixed.ndim != 1:
            raise ValueError("existing SaddleFlow fixed mask must have shape (N,)")
        return (~fixed).unsqueeze(-1).expand(-1, 3)
    n_atoms = int(sample["Z"].shape[0])
    return torch.ones(n_atoms, 3, dtype=torch.bool, device=device)


def _graph_data_for_terminal(terminal: torch.Tensor, sample: dict) -> AtomicData:
    """Build UMA's static neighbor graph from a detached terminal snapshot.

    FairChem's ``AtomicData.from_ase`` necessarily passes through NumPy.  That
    is appropriate for discrete graph construction, but it would detach
    coordinates if used as UMA's actual ``pos`` tensor.  The caller replaces
    the collated ``pos`` with the original graph-connected terminal tensor
    before the UMA forward.  Thus neighbor edges are locally fixed while the
    force and its coordinate derivatives remain differentiable.
    """
    positions = terminal.detach().float().cpu().numpy()
    numbers = sample["Z"].detach().cpu().numpy()
    cell = sample["cell"].detach().float().cpu().numpy()
    atoms = Atoms(positions=positions, numbers=numbers, cell=cell, pbc=True)

    # UMA's graph builder only needs the atom-level FixAtoms metadata.  A
    # partial coordinate mask is represented for the loss itself, not as an
    # ASE constraint here, because it does not change UMA's energy forward.
    if "fixed" in sample:
        fixed_indices = torch.where(sample["fixed"])[0].cpu().tolist()
        if fixed_indices:
            atoms.set_constraint(FixAtoms(indices=fixed_indices))

    data = AtomicData.from_ase(atoms, task_name=sample["task_name"])
    data.charge = torch.tensor([int(sample["charge"])], dtype=torch.long)
    data.spin = torch.tensor([int(sample["spin"])], dtype=torch.long)
    return data


def _force_and_metrics_for_task(
    terminals: Sequence[torch.Tensor],
    labels: Sequence[torch.Tensor],
    samples: Sequence[dict],
    *,
    force_backbone,
    force_head,
    force_tasks: dict,
    device: torch.device,
    stopgrad: bool,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Evaluate one same-task UMA mini-batch and return per-structure metrics.

    The three returned lists contain free-coordinate atom-wise force RMS,
    free-coordinate atom-wise force maximum, and PBC endpoint RMSD to the
    supervised saddle label.  In direct-loss mode these values retain their
    graph to ``terminals``.  In stop-gradient mode they are ordinary detached
    measurements.
    """
    graph_data = [_graph_data_for_terminal(terminal, sample)
                  for terminal, sample in zip(terminals, samples)]
    uma_batch = data_list_collater(graph_data, otf_graph=True).to(device)

    # This assignment is the key differentiability step in direct-loss mode.
    # The AtomicData graph was built using detached coordinates above, but UMA
    # receives the live predicted terminal positions as its ``pos`` field.
    #
    # In stop-gradient mode, UMA still needs a position tensor requiring grad:
    # forces are -dE/dx.  It is a fresh leaf detached from SaddleFlow, however,
    # and create_graph=False below.  This computes ordinary forces without
    # creating the force-to-flow-model Hessian path.
    if stopgrad:
        live_positions = torch.cat([
            terminal.detach().to(device=device, dtype=torch.float32)
            for terminal in terminals
        ], dim=0).requires_grad_(True)
    else:
        live_positions = torch.cat([
            terminal.to(device=device, dtype=torch.float32)
            for terminal in terminals
        ], dim=0)
    uma_batch["pos"] = live_positions
    if not uma_batch["pos"].requires_grad:
        raise RuntimeError(
            "Predicted terminal positions do not require gradients; the force "
            "constraint must be called from a live model forward pass."
        )

    features = force_backbone(uma_batch)
    forces = compute_uma_forces(
        uma_batch,
        features,
        force_head,
        force_tasks,
        create_graph=not stopgrad,
        retain_graph=not stopgrad,
        task_name=samples[0]["task_name"],
    )

    force_rms: list[torch.Tensor] = []
    force_max: list[torch.Tensor] = []
    endpoint_rmsd: list[torch.Tensor] = []
    offset = 0
    for terminal, label, sample in zip(terminals, labels, samples):
        n_atoms = int(sample["Z"].shape[0])
        force = forces[offset:offset + n_atoms]
        offset += n_atoms

        free = free_coordinate_mask(sample, device)
        n_free_atoms = int(free.any(dim=-1).sum().item())
        if n_free_atoms == 0:
            # A completely constrained structure has no stationarity condition
            # for a generator to satisfy.  Keep a graph-connected zero so the
            # caller can uniformly average the result.
            zero = force.sum() * 0.0
            force_rms.append(zero)
            force_max.append(zero)
            endpoint_rmsd.append(zero)
            continue

        projected_force = force * free.to(dtype=force.dtype)
        atom_force_norm = torch.linalg.vector_norm(projected_force, dim=-1)
        mobile_atoms = free.any(dim=-1)
        force_rms.append(torch.sqrt((atom_force_norm[mobile_atoms] ** 2).mean()))
        force_max.append(atom_force_norm[mobile_atoms].max())

        # This is diagnostic-only.  It uses the same PBC minimum-image
        # displacement convention as the rest of SaddleFlow.
        metric_terminal = terminal.detach() if stopgrad else terminal
        terminal_to_label = mic_displacement(
            metric_terminal.to(device=device, dtype=torch.float32),
            label.to(device=device, dtype=torch.float32),
            sample["cell"].to(device),
        )
        projected_error = terminal_to_label * free.to(dtype=terminal_to_label.dtype)
        atom_error_norm = torch.linalg.vector_norm(projected_error, dim=-1)
        endpoint_rmsd.append(torch.sqrt((atom_error_norm[mobile_atoms] ** 2).mean()))

    return force_rms, force_max, endpoint_rmsd


def algebraic_endpoint_force_loss(
    terminals: Sequence[torch.Tensor],
    labels: Sequence[torch.Tensor],
    samples: Sequence[dict],
    *,
    force_backbone,
    force_head,
    force_tasks: dict,
    epsilon: float,
    device: torch.device,
    stopgrad: bool = False,
    stopgrad_max_gate: float | None = 10.0,
) -> dict[str, torch.Tensor]:
    """Evaluate UMA at algebraic terminal predictions.

    Args:
        terminals: PBC-wrapped ``S_hat`` tensors, one ``(N_i, 3)`` tensor per
            selected training structure.  These remain connected to the
            velocity prediction for the direct loss; stop-gradient mode
            deliberately detaches them before UMA evaluation.
        labels: Corresponding PBC-wrapped supervised saddle labels, used only
            for the endpoint-RMSD diagnostic.
        samples: Existing SaddleFlow sample dicts for atom types, cells,
            task metadata, and constraints.
        epsilon: The force dead-band in eV/A.  It must be calibrated externally
            for the model/dataset; it is deliberately never guessed here.
        stopgrad: Return a detached continuous force gate instead of using the
            UMA force derivative in the returned loss.
        stopgrad_max_gate: Optional cap for the detached gate.  ``None`` means
            no cap.

    Returns plain tensors suitable for the parent loss module's logging.  In
    direct-loss mode, the central value is

        mean(relu(f_rms / epsilon - 1)^2).

    In stop-gradient mode, ``gate`` is instead

        detach(clamp(relu(f_rms / epsilon - 1), max=stopgrad_max_gate)),

    and ``loss`` is a graph-connected zero.  The caller applies ``gate`` to
    its normal supervised per-example FM loss.
    """
    if len(terminals) != len(labels) or len(terminals) != len(samples):
        raise ValueError("terminals, labels, and samples must have equal length")
    if not terminals:
        raise ValueError("force constraint requires at least one selected sample")
    if epsilon <= 0:
        raise ValueError("force-constraint epsilon must be positive (eV/A)")
    if stopgrad_max_gate is not None and stopgrad_max_gate < 0:
        raise ValueError("stopgrad_max_gate must be non-negative or None")

    # UMA's output wrapper selects a task-specific normalizer.  Grouping avoids
    # silently using the first sample's task for a heterogeneous general batch.
    by_task: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        by_task[str(sample["task_name"])].append(index)

    rms_by_index: list[torch.Tensor | None] = [None] * len(samples)
    max_by_index: list[torch.Tensor | None] = [None] * len(samples)
    rmsd_by_index: list[torch.Tensor | None] = [None] * len(samples)
    for indices in by_task.values():
        group_terminals = [terminals[index] for index in indices]
        group_labels = [labels[index] for index in indices]
        group_samples = [samples[index] for index in indices]
        rms, fmax, endpoint_rmsd = _force_and_metrics_for_task(
            group_terminals,
            group_labels,
            group_samples,
            force_backbone=force_backbone,
            force_head=force_head,
            force_tasks=force_tasks,
            device=device,
            stopgrad=stopgrad,
        )
        for index, value_rms, value_max, value_rmsd in zip(
            indices, rms, fmax, endpoint_rmsd,
        ):
            rms_by_index[index] = value_rms
            max_by_index[index] = value_max
            rmsd_by_index[index] = value_rmsd

    force_rms = torch.stack([value for value in rms_by_index if value is not None])
    force_max = torch.stack([value for value in max_by_index if value is not None])
    endpoint_rmsd = torch.stack([value for value in rmsd_by_index if value is not None])
    violation = torch.relu(force_rms / float(epsilon) - 1.0)
    if stopgrad:
        gate = violation.detach()
        if stopgrad_max_gate is not None:
            gate = gate.clamp(max=float(stopgrad_max_gate))
        # The parent loss module combines this detached gate with the normal
        # supervised FM loss.  Keep a graph-connected zero here so callers can
        # use one stable return schema for both modes.
        loss = sum(terminal.sum() * 0.0 for terminal in terminals)
    else:
        gate = violation.detach()
        loss = (violation ** 2).mean()
    return {
        "loss": loss,
        "force_rms": force_rms,
        "force_max": force_max,
        "endpoint_rmsd": endpoint_rmsd,
        "violation": violation,
        "gate": gate,
    }
