"""W2 gating: keep the W2 path of every m > 0 SO(2) layer and stay mirror-equivariant by construction.

Each m > 0 SO(2) convolution in UMA maps the (+m, -m) pair of edge-frame coefficients of a bond with
[[W1, -W2], [W2, W1]] (saddleflow.utils.parity). A mirror conjugates the pair: W1 commutes with that and W2 changes
sign, so W2 is the only path that breaks mirror equivariance. Zeroing it (train.py --parity-even) removes the path.
Here it is kept and multiplied by a per-bond pseudoscalar (a number that also changes sign under a mirror), which
makes the product mirror-correct:

    u, w = A_m x_m, B_m x_m                    `width` real linear maps of the (+m, -m) halves of the block's
                                               edge-frame message, applied to both halves alike, m = 1 .. mmax
    p    = sum_m (u_+ * w_- - u_- * w_+)       `width` pseudoscalars per bond: the component of u x w along the
                                               bond for m = 1. A rotation about the bond turns both pairs by the
                                               same angle (p unchanged); a mirror conjugates both (p changes sign)
    h    = H (c p),   g = h / sqrt(1 + h^2)    one gate per output channel of each of the block's two SO(2)
                                               convolutions; odd, smooth and bounded, so the sign survives
    W2  -> g * W2                              in both convolutions of the block

p is a pseudoscalar for any A, B, so the layer is mirror-equivariant whatever the gate learns. Where a bond's
surroundings are mirror-symmetric, p = 0 and the block acts as with W2 = 0.

c is a fixed constant (DEFAULT_SCALE, recorded in the model config as w2_gate_scale). On FM2 at initialisation
|p| is about 1e-4 per bond, too small for H to open the gate in a short fine-tune; c = 1e4 brings c p to about 1.
A positive constant times a pseudoscalar is still a pseudoscalar, so c does not affect equivariance.

At initialisation H = 0, so g = 0 and every block equals the same block with W2 = 0: a model built from a
checkpoint starts exactly at that checkpoint with W2 zeroed. A and B start random, so H gets gradient at once,
provided W2 is nonzero (it is in FM2; with W2 = 0 and g = 0 neither would move).

install_w2_gates(backbone, width) changes only the blocks of the backbone object it is given: fairchem's code, and
any other UMA in the process (the frozen one that computes energies and forces for Sella), are untouched. The two
forward functions below are fairchem 2.21.0's `SO2_Convolution.forward` and `Edgewise.forward_chunk` with the gate
added; `Edgewise.forward` is not replaced, so both its plain and its activation-checkpointed paths use the gate.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from fairchem.core.models.uma.escn_md_block import Edgewise, set_mole_ac_start_index
from fairchem.core.models.uma.nn.so2_layers import SO2_Convolution, SO2_m_Conv
from torch.profiler import record_function

DEFAULT_SCALE = 1e4     # c: brings the pseudoscalars of FM2 at initialisation (|p| ~ 1e-4) to about 1


class W2Gate(nn.Module):
    """Per-bond gates for the W2 paths of one block's two SO(2) convolutions."""

    def __init__(self, m_in_features: list[int], m_split_sizes: list[int], width: int,
                 out_channels: tuple[int, int], scale: float = DEFAULT_SCALE, seed: int = 0):
        """m_in_features[m - 1]: the size of one half of the order-m block of the message (its (lmax - m + 1) x
        channels coefficients); m_split_sizes: how the message splits by order (SO2_Convolution.m_split_sizes);
        out_channels: the output channels of the block's two convolutions; scale: the fixed constant c."""
        super().__init__()
        if scale <= 0:
            raise ValueError(f"W2 gate scale must be positive, got {scale}")
        self.m_split_sizes = list(m_split_sizes)
        self.mmax = len(m_in_features)
        self.scale = float(scale)
        with torch.random.fork_rng(devices=[]):        # a fixed initialisation that leaves the global RNG alone
            torch.manual_seed(seed)
            self.proj_u = nn.ModuleList(nn.Linear(n, width, bias=False) for n in m_in_features)
            self.proj_w = nn.ModuleList(nn.Linear(n, width, bias=False) for n in m_in_features)
        self.heads = nn.ModuleList(nn.Linear(width, c, bias=False) for c in out_channels)
        for head in self.heads:
            nn.init.zeros_(head.weight)

    def pseudoscalars(self, x: torch.Tensor) -> torch.Tensor:
        """(E, width) from the edge-frame message x (E, coefficients, channels)."""
        x_by_m = x.split(self.m_split_sizes, dim=1)
        p = 0
        for m in range(1, self.mmax + 1):
            x_m = x_by_m[m].reshape(x.shape[0], 2, -1)        # the two halves, as SO2_Convolution views them
            u, w = self.proj_u[m - 1](x_m), self.proj_w[m - 1](x_m)
            p = p + u[:, 0] * w[:, 1] - u[:, 1] * w[:, 0]
        return p

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """[(E, C_out) gate in (-1, 1)] for the block's first and second SO(2) convolution."""
        p = self.scale * self.pseudoscalars(x)
        gates = []
        for head in self.heads:
            h = head(p)
            gates.append(h * torch.rsqrt(1 + h * h))
        return gates


def gated_m_conv(conv: SO2_m_Conv, x_m: torch.Tensor, gate: torch.Tensor):
    """fairchem's SO2_m_Conv.forward with its W2 terms multiplied by `gate` (E, C_out).

    fc's output splits into four pieces; in fairchem's names x_r_0 = W1 x_+, x_i_0 = W2 x_+, x_r_1 = W1 x_-,
    x_i_1 = W2 x_- (fc holds W1 in its first half of output rows, W2 in the second). fairchem returns
    (x_r_0 - x_i_1, x_r_1 + x_i_0); here both W2 terms carry the gate.
    """
    num_edges = x_m.shape[0]
    x_m = conv.fc(x_m)
    x_r_0, x_i_0, x_r_1, x_i_1 = x_m.reshape(num_edges, -1, conv.out_channels_half).split(1, dim=1)
    shape = (num_edges, -1, conv.m_output_channels)
    g = gate[:, None, :]
    x_m_r = x_r_0.view(shape) - g * x_i_1.view(shape)
    x_m_i = x_r_1.view(shape) + g * x_i_0.view(shape)
    return x_m_r, x_m_i


def gated_so2_convolution(conv: SO2_Convolution, x: torch.Tensor, x_edge: torch.Tensor | None, gate: torch.Tensor):
    """fairchem 2.21.0's SO2_Convolution.forward with every m > 0 layer replaced by gated_m_conv."""
    if conv.rad_func is not None:
        x_edge = conv.rad_func(x_edge)

    x_by_m = x.split(conv.m_split_sizes, dim=1)
    if x_edge is not None:
        x_edge_by_m = x_edge.split(conv.edge_split_sizes, dim=1)

    num_edges = x.shape[0]
    x_0 = x_by_m[0].view(num_edges, -1)
    if x_edge is not None:
        x_0 = x_0 * x_edge_by_m[0]
    x_0 = conv.fc_m0(x_0)

    if conv.extra_m0_output_channels is not None:
        x_0_extra, x_0 = x_0.split(
            (conv.extra_m0_output_channels, conv.fc_m0.out_features - conv.extra_m0_output_channels), -1)

    out = [x_0.view(num_edges, -1, conv.m_output_channels)]
    for m in range(1, conv.mmax + 1):
        x_m = x_by_m[m].view(num_edges, 2, -1)
        if x_edge is not None:
            x_m = x_m * x_edge_by_m[m].unsqueeze(1)
        out.extend(gated_m_conv(conv.so2_m_conv[m - 1], x_m, gate))
    out = torch.cat(out, dim=1)

    if conv.extra_m0_output_channels is not None:
        return out, x_0_extra
    return out


class W2GatedEdgewise(Edgewise):
    """fairchem's Edgewise with the W2 gate (attribute `w2_gate`) applied to both SO(2) convolutions."""

    def forward_chunk(self, x_full, x_original_shape, x_edge, edge_index, wigner, wigner_inv_envelope,
                      node_offset: int = 0, ac_mole_start_idx: int = 0):
        set_mole_ac_start_index(self, ac_mole_start_idx)

        with record_function("SO2Conv"):
            x_message = self.backend.node_to_edge_wigner_permute(x_full, edge_index, wigner)
            gate_1, gate_2 = self.w2_gate(x_message)
            x_message, x_0_gating = gated_so2_convolution(self.so2_conv_1, x_message, x_edge, gate_1)
            x_message = self.act(x_0_gating, x_message)
            x_message = gated_so2_convolution(self.so2_conv_2, x_message, None, gate_2)
            new_embedding = self.backend.permute_wigner_inv_edge_to_node(
                x_message, wigner_inv_envelope, edge_index, x_original_shape, node_offset)

        set_mole_ac_start_index(self, 0)
        return new_embedding


def install_w2_gates(backbone: nn.Module, width: int, scale: float = DEFAULT_SCALE, seed: int = 0) -> nn.Module:
    """Give every block of this UMA backbone a W2 gate of `width` pseudoscalars per bond, multiplied by the fixed
    constant `scale`. Returns the backbone."""
    if width <= 0:
        raise ValueError(f"W2 gate width must be positive, got {width}")
    for i, block in enumerate(backbone.blocks):
        edge_wise = block.edge_wise
        if type(edge_wise) is not Edgewise:
            raise RuntimeError(f"blocks[{i}].edge_wise is {type(edge_wise).__name__}, expected fairchem's Edgewise")
        convs = (edge_wise.so2_conv_1, edge_wise.so2_conv_2)
        for conv in convs:
            # The fast backend's converted layers (SO2_m_Conv_Block) cache [[W1, -W2], [W2, W1]] and would bypass
            # the gate; the forward functions above are written for these exact classes.
            if type(conv) is not SO2_Convolution or any(type(c) is not SO2_m_Conv for c in conv.so2_m_conv):
                raise RuntimeError(f"blocks[{i}]: expected fairchem's SO2_Convolution with SO2_m_Conv layers")
        conv_1 = edge_wise.so2_conv_1
        edge_wise.w2_gate = W2Gate(
            m_in_features=[c.fc.in_features for c in conv_1.so2_m_conv],
            m_split_sizes=conv_1.m_split_sizes,
            width=width,
            out_channels=(conv_1.m_output_channels, edge_wise.so2_conv_2.m_output_channels),
            scale=scale,
            seed=seed + i,
        )
        edge_wise.__class__ = W2GatedEdgewise
    return backbone
