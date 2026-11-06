"""Mirror equivariance by construction for a UMA (eSCN-MD) backbone: W2 = 0 in every m > 0 SO(2) layer.

UMA is exactly rotation-equivariant but not inversion-equivariant, and every mirror is an inversion times a
rotation. Each m > 0 SO(2) convolution maps the (real, imaginary) edge-frame coefficients of order +-m with
[[W1, -W2], [W2, W1]] (fairchem `SO2_m_Conv`: `fc` holds W1 in its first half of output rows, W2 in the second).
W2, which mixes the two parts, is the only parity-odd path. With W2 = 0 every feature of degree l picks up
(-1)^l under r -> -r, so the velocity (an l = 1 output) satisfies v(-x) = -v(x).

Only weight values change; fairchem's code is untouched. The zeroing is applied to the backbone object after
loading, so the separate frozen UMA that computes energies and forces for Sella is never affected.

    enforce_parity_even(m)  zero W2 and add a gradient hook so it stays zero in training (AdamW moments stay
                            zero, weight decay leaves zero at zero, EMA of zeros is zero)
    zero_w2_(m)             zero W2 only (evaluation)
    assert_parity_even(m)   fail if any W2 entry is nonzero, or if fairchem's fast block variant
                            (SO2_m_Conv_Block) is present: it caches [[W1, -W2], [W2, W1]] and would keep the old W2
    parity_from_env(m)      SF_PARITY_EVEN=1: zero W2 in any loaded checkpoint (no retraining)
    w2_report(m)            ||W1||, ||W2|| per layer
    parity_sign(lmax)       (-1)^l per spherical-harmonic coefficient
"""

import os

import torch
import torch.nn as nn

ENV = "SF_PARITY_EVEN"


def _so2_classes():
    from fairchem.core.models.uma.nn.so2_layers import SO2_m_Conv, SO2_m_Conv_Block
    return SO2_m_Conv, SO2_m_Conv_Block


def _w2(conv):
    """(weight tensor, index of its W2 half). Plain Linear: (out, in); mixture of experts: (experts, out, in)."""
    fc, half = conv.fc, conv.out_channels_half
    w = fc.weight if isinstance(getattr(fc, "weight", None), nn.Parameter) else fc.weights
    if w.shape[-2] != 2 * half:
        raise RuntimeError(f"{type(fc).__name__}: {w.shape[-2]} output rows, expected 2 x {half}")
    return w, (..., slice(half, None), slice(None))


def so2_layers(module):
    """[(name, SO2_m_Conv)] for every m > 0 SO(2) layer. Refuses the cached block variant."""
    plain, block = _so2_classes()
    cached = [n for n, c in module.named_modules() if isinstance(c, block)]
    if cached:
        raise RuntimeError(f"{len(cached)} SO2_m_Conv_Block layers (fairchem's fast backend) cache "
                           f"[[W1, -W2], [W2, W1]]; zeroing fc would not reach the cache: {cached[:3]}")
    layers = [(n, c) for n, c in module.named_modules() if isinstance(c, plain)]
    if not layers:
        raise RuntimeError("no m > 0 SO(2) layers found: is this a UMA backbone?")
    return layers


@torch.no_grad()
def zero_w2_(module):
    """Zero W2 in every m > 0 SO(2) layer. Returns the number of layers."""
    layers = so2_layers(module)
    for _, conv in layers:
        w, i = _w2(conv)
        w[i] = 0
    return len(layers)


def enforce_parity_even(module):
    """Hook to zero W2 and keep it zero under training"""
    zero_w2_(module)
    for _, conv in so2_layers(module):
        w, i = _w2(conv)
        if not getattr(w, "_parity_hook", False):        # mixture-of-experts layers can be shared

            def hook(g, i=i):
                g = g.clone()
                g[i] = 0
                return g

            w.register_hook(hook)
            w._parity_hook = True
    return module


def assert_parity_even(module):
    for name, conv in so2_layers(module):
        w, i = _w2(conv)
        if w[i].abs().max().item() != 0:
            raise RuntimeError(f"{name}: W2 is not zero (max |W2| = {w[i].abs().max().item():.3g})")


def parity_from_env(module):
    """If SF_PARITY_EVEN=1, zero W2 in `module` (a loaded model or backbone) and check it. Returns whether it did."""
    if os.environ.get(ENV) != "1":
        return False
    n = zero_w2_(module)
    assert_parity_even(module)
    print(f"[parity] {ENV}=1: W2 zeroed in {n} SO(2) layers", flush=True)
    return True


@torch.no_grad()
def w2_report(module):
    """[dict(name, m, w1, w2, ratio)] with Frobenius norms over all experts."""
    rows = []
    for name, conv in so2_layers(module):
        w, i = _w2(conv)
        w1, w2 = w[..., :conv.out_channels_half, :].norm().item(), w[i].norm().item()
        rows.append(dict(name=name, m=conv.m, w1=w1, w2=w2, ratio=w2 / w1 if w1 else float("nan")))
    return rows


def parity_sign(lmax):
    """(-1)^l for the (lmax + 1)^2 coefficients, in UMA's l-major order."""
    return torch.tensor([(-1.0) ** l for l in range(lmax + 1) for _ in range(2 * l + 1)])
