"""Michael: detached force/current free-Hessian inputs for the LiC student.

No saddle labels, eigenmode targets, auxiliary loss, or learned physics model.
The same observation routine runs at sampled training x_t and every ODE step.
"""

import torch
from pathlib import Path

from saddleflow.constr_gen import uma_forces


def head_options(extras):
    return dict(force_field_channels=int(extras.get("force_field_channels", 0)),
                curvature_conditioning=bool(extras.get("curvature_conditioning", False)),
                curvature_mlp_hidden=int(extras.get("curvature_mlp_hidden", 64)),
                identity_force_init=bool(extras.get("inject_force", False)),
                source_channels=int(extras.get("source_channels", 0)),
                source_conditioning=bool(extras.get("source_conditioning", False)))


def catalog_paths(extras, train_traj=None, test_traj=None):
    """Study-aware train/catalog complement; explicit CLI paths still win."""
    here = Path(__file__).resolve().parents[1]  # examples/LiC
    return (train_traj or extras.get("evaluation_train_traj") or str(here/"train_set.traj"),
            test_traj or extras.get("evaluation_test_traj") or str(here/"test_set.traj"))


def make_conditioner(force_backbone, force_head, force_tasks, *, inject_force=False,
                     curvature=False, fd_eps=.002, probe_batch_size=32):
    """Batch central forces and the six Li-only FD probes (FP32 oracle).

    Curvature inputs are [lambda1,lambda2,lambda2-lambda1], eV/A^2, from the
    symmetrized free Hessian. Cell/carbon positions are NEVER perturbed.
    Captured frozen models are not registered in the student checkpoint.
    """
    if not inject_force and not curvature: raise ValueError("Michael: enable force or curvature inputs")
    if fd_eps <= 0 or probe_batch_size < 1: raise ValueError("Michael: FD epsilon/chunk size must be positive")
    for module in [force_backbone, force_head]:
        module.eval()
        for parameter in module.parameters(): parameter.requires_grad_(False)

    def observe(positions, samples, device):
        sample = samples[0]
        # This is intentionally a LiC helper, not a dense all-system Hessian.
        free = torch.nonzero(~sample["fixed"], as_tuple=False).flatten()
        if len(free) != 1 or int(sample["Z"][free[0]]) != 3:
            raise ValueError("Michael: scalar-curvature example expects exactly one free Li")
        li = int(free[0]); x = torch.stack(list(positions)).detach().float()
        force, scalars = None, None
        before = observe.queries

        def forces(points):
            chunks = []
            for first in range(0, len(points), probe_batch_size):
                chunk = points[first:first+probe_batch_size]
                chunks.append(uma_forces(chunk, Z=sample["Z"], cell=sample["cell"],
                    task_name=sample["task_name"], charge=int(sample["charge"]), spin=int(sample["spin"]),
                    fixed=sample["fixed"], force_backbone=force_backbone, force_head=force_head,
                    force_tasks=force_tasks, device=device))
                observe.queries += len(chunk)  # Geometries, not batches.
            return torch.cat(chunks)

        # Autocast must not turn a 0.002-A numerical derivative into bf16 noise.
        with torch.autocast(device_type=torch.device(device).type, enabled=False):
            if inject_force:
                force = forces(x).detach().to(device)
                force[:, sample["fixed"].to(device)] = 0.
                force = force.reshape(-1, 3)
            if curvature:
                directions = torch.eye(3, device=x.device).repeat_interleave(2, dim=0)
                directions *= x.new_tensor([1., -1., 1., -1., 1., -1.])[:, None]
                probes = x[:, None].expand(-1, 6, -1, -1).clone()
                probes[:, :, li] += fd_eps * directions
                probe_forces = forces(probes.reshape(-1, x.shape[1], 3)).reshape(len(x), 6, x.shape[1], 3)
                raw = -(probe_forces[:, 0::2, li]-probe_forces[:, 1::2, li])/(2*fd_eps)
                raw = raw.transpose(-1, -2)
                H = .5*(raw+raw.transpose(-1, -2))
                eigs = torch.linalg.eigvalsh(H.double().cpu()).to(device=device, dtype=torch.float32)
                scalars = torch.stack([eigs[:, 0], eigs[:, 1], eigs[:, 1]-eigs[:, 0]], dim=-1).detach()
                if not torch.isfinite(scalars).all(): raise ValueError("Michael: nonfinite oracle curvature")
        return dict(force_field=force, curvature_scalars=scalars, queries=observe.queries-before)

    observe.queries = 0
    return observe


def conditioning_queries(head):
    return getattr(getattr(head, "_physics_conditioner", None), "queries", 0)
