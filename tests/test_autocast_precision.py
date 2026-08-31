"""Geometry helpers must be immune to autocast.

`wrap_positions` and `mic_displacement` both do a Cartesian->fractional->Cartesian
round trip, i.e. two matmuls. torch.autocast silently demotes matmuls to bf16/fp16,
and bf16 keeps only 7 mantissa bits, so its representable-value spacing at a
coordinate of ~12 A is ~0.0625 A. That quantises the structure fed to the backbone
while the flow-matching target still encodes the exact displacement.

This was not hypothetical: it cost months on the LiC_simpler case. With the round
trip unpinned, bf16 training reached a LOWER loss (0.013 vs 0.183 at epoch 10k)
while the learned field lost its six-fold angular structure -- hexatic order 0.124
vs 0.732, on-orbit 21% vs 88%. See CLAUDE.md, latent-bug log.
"""
import numpy as np
import pytest
import torch

from saddleflow.data.transforms import mic_displacement, wrap_positions

# A cell of the size that actually bites: ~17 A, where bf16's grid is ~0.125 A.
CELL = torch.tensor([[17.1069, 0.0, 0.0],
                     [0.0, 17.2842, 0.0],
                     [0.0, 0.0, 20.0013]], dtype=torch.float32)


def _positions(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand((n, 3), generator=g) @ CELL


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_wrap_positions_is_autocast_immune(dtype):
    pos = _positions()
    ref = wrap_positions(pos, CELL)
    with torch.autocast(device_type="cpu", dtype=dtype):
        got = wrap_positions(pos, CELL).float()
    assert torch.equal(got, ref), (
        f"wrap_positions changed under {dtype} autocast by up to "
        f"{(got - ref).norm(dim=-1).max():.4f} A -- the fp32 pin was removed."
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mic_displacement_is_autocast_immune(dtype):
    a, b = _positions(seed=0), _positions(seed=1)
    ref = mic_displacement(a, b, CELL)
    with torch.autocast(device_type="cpu", dtype=dtype):
        got = mic_displacement(a, b, CELL).float()
    assert torch.equal(got, ref), (
        f"mic_displacement changed under {dtype} autocast by up to "
        f"{(got - ref).norm(dim=-1).max():.4f} A -- the fp32 pin was removed."
    )


def test_unpinned_roundtrip_really_is_lossy():
    """Guards the premise: without the pin the error is large enough to matter.

    If this ever stops failing, torch changed its autocast rules and the pins
    may no longer be load-bearing -- re-derive before deleting them.
    """
    pos = _positions()

    def unpinned(p, c):
        frac = p @ torch.linalg.inv(c)
        return (frac - torch.floor(frac)) @ c

    ref = unpinned(pos, CELL)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        got = unpinned(pos, CELL).float()
    assert (got - ref).norm(dim=-1).max() > 0.01
