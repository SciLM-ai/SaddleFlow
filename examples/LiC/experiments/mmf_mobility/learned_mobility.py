"""Learn only MMF translation scales; the potential and minimum mode stay fixed.

The oracle uses the existing ASE backend on detached coordinates. Its numerical
Hessian supplies a local force derivative in the tiny PyTorch rollout graph:
forward F is the actual observed force, backward dF/dx is -H. We intentionally
do NOT differentiate the minimum mode or Hessian eigenvalues.

This is an approximate rollout gradient, not full differentiation of the oracle.
There is no flow-matching velocity target or auxiliary force loss here.
"""

import math
from time import perf_counter

import numpy as np
import torch
from torch import nn

from saddleflow.classical_saddle_search.result import finish_result, trace_row


class MobilityPolicy(nn.Module):
    """Raw (1,1), two global parameters, or a small invariant eight-input MLP.

    The policy lives on CPU in float64. Only UMA needs the GPU; retaining the
    rollout graph never retains UMA activations. The outputs are dimensionless
    multipliers on the solver's shared base step size.
    """

    def __init__(self, kind="state", lower=0.05, upper=5.0, hidden=32):
        super().__init__()
        if kind not in ("raw", "global", "state"):
            raise ValueError("Policy must be raw, global, or state.")
        if not 0 < lower < 1 < upper:
            raise ValueError("Mobility bounds must satisfy 0 < lower < 1 < upper.")
        self.kind, self.lower, self.upper = kind, lower, upper
        # sigmoid(bias) = (1-lower)/(upper-lower), so a=b=1 initially.
        bias = math.log((1.0 - lower) / (upper - 1.0))
        if kind == "global":
            self.raw_scales = nn.Parameter(torch.full((2,), bias, dtype=torch.float64))
        elif kind == "state":
            self.network = nn.Sequential(
                nn.Linear(8, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2),
            ).double()
            nn.init.zeros_(self.network[-1].weight)
            nn.init.constant_(self.network[-1].bias, bias)

    def forward(self, features):
        if self.kind == "raw":
            return features.new_ones(2)
        raw = self.raw_scales if self.kind == "global" else self.network(features)
        return self.lower + (self.upper - self.lower) * torch.sigmoid(raw)


def force_with_local_derivative(x, observed_force, hessian):
    """Actual oracle force in forward; the already measured -H in backward.

    x - x.detach() is zero numerically but has identity derivative. H and the
    force observation are constants for this update. No custom autograd class
    and no extra potential queries are necessary.
    """
    force = torch.as_tensor(observed_force, dtype=x.dtype, device=x.device).detach()
    hessian = torch.as_tensor(hessian, dtype=x.dtype, device=x.device).detach()
    return force - hessian @ (x - x.detach())


def invariant_features(force, parallel, perpendicular, eigenvalues, progress):
    """Force norms, three lowest eigenvalues, eigengap, and k/K.

    log1p/asinh keep magnitudes moderate without fitting normalization to test
    states. Reference units are 1 eV/A and 1 eV/A^2. The features are rotationally
    invariant and unchanged by q -> -q. Force features retain their local graph;
    eigenvalues and the iteration coordinate are detached observations.
    """
    eigs = torch.as_tensor(eigenvalues, dtype=force.dtype, device=force.device).detach()
    if len(eigs) < 3:
        eigs = torch.cat([eigs, eigs[-1].expand(3 - len(eigs))])
    norms = torch.stack([torch.linalg.vector_norm(v) for v in (force, parallel, perpendicular)])
    return torch.cat([
        torch.log1p(norms), torch.asinh(eigs[:3]),
        torch.asinh(eigs[1:2] - eigs[:1]), force.new_tensor([progress]),
    ])


def trust_clip(delta, n_free, max_step):
    """Cap per-free-atom RMS displacement, retaining the clipping derivative."""
    rms = torch.linalg.vector_norm(delta) / math.sqrt(n_free)
    scale = (max_step / rms.clamp_min(1e-15)).clamp(max=1.0)
    return delta * scale, bool(rms.detach() > max_step)


def endpoint_loss(terminal, target, dofs, policy):
    """Free-atom MIC mean squared displacement; image shifts are constants.

    A trajectory can already be physically converged at its initial state. In
    that case its loss has a zero connection to the trainable parameters, so
    a uniform training loop can still call backward safely.
    """
    target = torch.as_tensor(target, dtype=terminal.dtype, device=terminal.device)
    error = terminal - target
    mic = dofs.displacement(terminal.detach().cpu().numpy(), target.detach().cpu().numpy())
    image_shift = torch.as_tensor(mic, dtype=error.dtype, device=error.device) - error.detach()
    loss = (error + image_shift).square().sum() / dofs.n_free
    for parameter in policy.parameters():
        loss = loss + 0.0 * parameter.sum()
    return loss


def run_mobility_flow(pes, x0, policy, *, max_steps=100, step_size=0.2,
                      max_step=0.1, fd_eps=0.002, fmax_tol=0.005,
                      eigenvalue_tol=0.01, callback=None):
    """Same oracle MMF loop for A/B/C, with a graph-connected terminal tensor.

    Free coordinates may be any small dimension, not a hard-coded Li index.
    This first learned prototype uses a dense FD Hessian. It is NOT the scalable
    full-system Lanczos implementation. Coordinates remain unwrapped internally,
    exactly as in run_oracle_mmf; ASE handles periodic geometry and the terminal
    loss uses MIC. Fixed coordinates are always restored by FreeDOFs.expand.

    There are at most K translations and K+1 evaluated states (including the
    terminal check). In LiC each state costs about seven geometry queries. The
    local force graph is used when gradients are enabled; inference can use
    torch.no_grad(). Early-stop decisions and mode selection are not differentiated.
    """
    if max_steps < 1 or min(step_size, max_step, fd_eps, fmax_tol) <= 0:
        raise ValueError("Positive horizon, step sizes, FD epsilon, and force tolerance required.")
    x = torch.as_tensor(np.asarray(x0).copy(), dtype=torch.float64)
    trace, last_shift, last_clipped = [], 0.0, False
    status, steps = "max_steps", 0
    start = perf_counter()
    for step in range(max_steps + 1):
        point = x.detach().cpu().numpy()
        # UMA's conservative forces need coordinate autograd internally even
        # when the policy rollout is evaluated under torch.no_grad(). Its
        # detached NumPy observations never enter the policy's graph directly.
        with torch.enable_grad():
            observed_force = pes.forces(point)
            hessian, asymmetry = pes.free_hessian(point, fd_eps)
        eigs, vectors = np.linalg.eigh(hessian)
        q = torch.as_tensor(vectors[:, 0], dtype=x.dtype).detach()
        force = force_with_local_derivative(x, observed_force, hessian)
        parallel = torch.dot(q, force) * q
        perpendicular = force - parallel
        features = invariant_features(force, parallel, perpendicular, eigs, step / max_steps)
        a, b = policy(features).unbind()
        row = trace_row(pes, point, observed_force, q=vectors[:, 0], eigs=eigs,
                        shift_rms=last_shift, asymmetry=asymmetry)
        row.update(step=step, mobility_a=float(a.detach()), mobility_b=float(b.detach()),
                   clipped=float(last_clipped), eigenvalue3=float(eigs[min(2, len(eigs)-1)]))
        trace.append(row)
        if callback is not None:
            callback(step, row)
        index_one = len(eigs) >= 2 and eigs[0] < -eigenvalue_tol and eigs[1] > eigenvalue_tol
        if row["force_max"] <= fmax_tol and index_one:
            status = "converged"
            break
        if np.linalg.norm(observed_force) < 1e-12 and not index_one:
            status = "stationary_not_index1"
            break
        if step == max_steps:
            break
        delta, last_clipped = trust_clip(step_size * (a * perpendicular - b * parallel),
                                        pes.dofs.n_free, max_step)
        last_shift = pes.dofs.rms(delta.detach().numpy())
        x = x + delta  # Never detach the position recurrence.
        steps = step + 1
        # Keep the last translated state usable at a strict PES-query budget.
        pending = trace_row(pes, x.detach().numpy(), shift_rms=last_shift)
        pending.update(step=steps, mobility_a=np.nan, mobility_b=np.nan,
                       clipped=float(last_clipped), eigenvalue3=np.nan)
        trace.append(pending)
    final = x.detach().numpy()
    pes.set_coordinates(final)
    result = finish_result(pes, final, trace, status, steps, perf_counter() - start)
    result["terminal_tensor"] = x
    result["model_forward_calls"] = len([r for r in trace if r["q"] is not None]) if policy.kind == "state" else 0
    return result
