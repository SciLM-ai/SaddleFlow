"""Conv-layer separation: keep every UMA weight and make each edge message exactly mirror-equivariant.

In the edge frame of a bond, each order m > 0 of the message is a set of (cos, sin) pairs (a, b), or z = a + i b.
Each of the two SO(2) convolutions of a block's Edgewise maps z with (W1 + i W2) (fairchem `SO2_m_Conv`: `fc` holds
W1 in its first half of output rows, W2 in the second). A mirror through a plane holding the bond conjugates z:
W1 commutes with that and W2 anticommutes. With the gate g of the activation between the two convolutions (a function
of the m = 0 inputs only, which the mirror leaves alone) and conv2's (V1 + i V2), fairchem's message is

    (V1 + i V2) g (W1 + i W2) z  =  [V1 g W1 - V2 g W2] z  +  i [V1 g W2 + V2 g W1] z
                                     commutes with the mirror     anticommutes: breaks mirror equivariance

Here conv1's W1 and W2 results are not added. They are carried as two streams and recombined in conv2:

    conv1     ordinary stream o = W1 z, with conv1's m = 0 output as its m = 0 part
              second stream   p = i W2 z, with a zero m = 0 part (W2 has no m = 0 counterpart)
    gate      the same gates on both streams; the SiLU acts on slot 0 (l = 0, m = 0), which belongs to o
    conv2     out = V1 o + i V2 p = [V1 g W1 - V2 g W2] z

so every message keeps exactly the mirror-commuting bracket. The result is exactly mirror- and rotation-equivariant
and has no new parameters: all of W1, W2, V1, V2 are used and trained, and the state dict is the stock one. On a
checkpoint's weights each message equals the mirror-symmetric part of that checkpoint's message, 1/2 [m(z) + R m(R z)]
with R the sign flip of the sin half of every m > 0 block. With W2 = 0 it equals the stock model with W2 = 0.

Because the state dict is unchanged, a checkpoint trained this way loads into a stock model without any error and
then silently runs as a stock model. Every loader therefore reads `conv_layer_separation` from the model config,
applies the separation, and checks it with `check_mode`.

apply_conv_layer_separation(backbone) changes only the blocks of the backbone object it is given, by swapping each
block's Edgewise class (which survives copy.deepcopy and pickling): fairchem's code, and any other UMA in the process
(the frozen one that computes energies and forces for Sella), are untouched. `Edgewise.forward` is not replaced, so
both its plain and its activation-checkpointed paths use the separation. The functions below follow fairchem
2.21.0's `SO2_Convolution.forward`, `SO2_m_Conv.forward`, `GateActivation.forward` and `Edgewise.forward_chunk`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from fairchem.core.models.uma.escn_md_block import Edgewise, set_mole_ac_start_index
from fairchem.core.models.uma.nn.activation import GateActivation
from fairchem.core.models.uma.nn.so2_layers import SO2_Convolution, SO2_m_Conv
from torch.profiler import record_function

CONFIG_KEY = "conv_layer_separation"
STOCK, SEPARATED = "stock", "conv_layer_separation"


def _m_terms(conv: SO2_m_Conv, x_m: torch.Tensor) -> list[torch.Tensor]:
    """fairchem's SO2_m_Conv.forward without its final sums: [W1 a, W2 a, W1 b, W2 b], each (E, L, C_out).

    x_m is (E, 2, L * C_in), the (a, b) halves of one order. fairchem names the four pieces x_r_0, x_i_0, x_r_1,
    x_i_1: the W2 terms are x_i_0 and x_i_1, and x_r_1 is a W1 term. `fc` may be a mixture-of-experts layer.
    """
    num_edges = x_m.shape[0]
    y = conv.fc(x_m).reshape(num_edges, -1, conv.out_channels_half)
    return [t.reshape(num_edges, -1, conv.m_output_channels) for t in y.split(1, dim=1)]


def conv1_split(conv: SO2_Convolution, x: torch.Tensor, x_edge: torch.Tensor):
    """Edgewise.so2_conv_1 with its W1 and W2 results kept apart: (o, p, gating scalars).

    o = W1 z with conv1's m = 0 output as its m = 0 part; p = i W2 z with a zero m = 0 part. Both have the layout
    of conv1's usual output, (E, coefficients, C_out).
    """
    num_edges = x.shape[0]
    x_edge = conv.rad_func(x_edge)
    x_by_m = x.split(conv.m_split_sizes, dim=1)
    x_edge_by_m = x_edge.split(conv.edge_split_sizes, dim=1)

    x_0 = conv.fc_m0(x_by_m[0].reshape(num_edges, -1) * x_edge_by_m[0])
    gating, x_0 = x_0.split(
        (conv.extra_m0_output_channels, conv.fc_m0.out_features - conv.extra_m0_output_channels), -1)
    x_0 = x_0.reshape(num_edges, -1, conv.m_output_channels)

    o, p = [x_0], [torch.zeros_like(x_0)]
    for m in range(1, conv.mmax + 1):
        x_m = x_by_m[m].reshape(num_edges, 2, -1) * x_edge_by_m[m].unsqueeze(1)
        w1a, w2a, w1b, w2b = _m_terms(conv.so2_m_conv[m - 1], x_m)
        o += [w1a, w1b]          # W1 z
        p += [-w2b, w2a]         # i W2 z
    return torch.cat(o, dim=1), torch.cat(p, dim=1), gating


def gate_pair(act: GateActivation, gating: torch.Tensor, o: torch.Tensor, p: torch.Tensor):
    """Edgewise.act on both streams: o exactly as fairchem's GateActivation, p with the same gates and no SiLU (its
    slot 0 is zero)."""
    g = act.gate_act(gating).view(gating.shape[0], act.lmax, act.num_channels)
    g = torch.index_select(g, dim=1, index=act.expand_index)
    o = torch.cat((act.scalar_act(o[:, :1]), o[:, 1:] * g), dim=1)
    p = torch.cat((p[:, :1], p[:, 1:] * g), dim=1)
    return o, p


def conv2_merge(conv: SO2_Convolution, o: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Edgewise.so2_conv_2 recombining the streams: V1 o + i V2 p, in the layout of conv2's usual output."""
    num_edges = o.shape[0]
    o_by_m = o.split(conv.m_split_sizes, dim=1)
    p_by_m = p.split(conv.m_split_sizes, dim=1)
    out = [conv.fc_m0(o_by_m[0].reshape(num_edges, -1)).reshape(num_edges, -1, conv.m_output_channels)]
    for m in range(1, conv.mmax + 1):
        layer = conv.so2_m_conv[m - 1]
        v1_oa, _, v1_ob, _ = _m_terms(layer, o_by_m[m].reshape(num_edges, 2, -1))
        _, v2_pa, _, v2_pb = _m_terms(layer, p_by_m[m].reshape(num_edges, 2, -1))
        out += [v1_oa - v2_pb, v1_ob + v2_pa]     # cos: V1 o_a - V2 p_b;  sin: V1 o_b + V2 p_a
    return torch.cat(out, dim=1)


def separated_message(edge_wise: Edgewise, x_message: torch.Tensor, x_edge: torch.Tensor) -> torch.Tensor:
    """The edge-frame message of one Edgewise with its conv layers separated (in place of conv1, act, conv2)."""
    o, p, gating = conv1_split(edge_wise.so2_conv_1, x_message, x_edge)
    o, p = gate_pair(edge_wise.act, gating, o, p)
    return conv2_merge(edge_wise.so2_conv_2, o, p)


class ConvLayerSeparationEdgewise(Edgewise):
    """fairchem's Edgewise with its two SO(2) convolutions combined by conv-layer separation."""

    def forward_chunk(self, x_full, x_original_shape, x_edge, edge_index, wigner, wigner_inv_envelope,
                      node_offset: int = 0, ac_mole_start_idx: int = 0):
        set_mole_ac_start_index(self, ac_mole_start_idx)

        with record_function("SO2Conv"):
            x_message = self.backend.node_to_edge_wigner_permute(x_full, edge_index, wigner)
            x_message = separated_message(self, x_message, x_edge)
            new_embedding = self.backend.permute_wigner_inv_edge_to_node(
                x_message, wigner_inv_envelope, edge_index, x_original_shape, node_offset)

        set_mole_ac_start_index(self, 0)
        return new_embedding


def _check_block(i: int, edge_wise: nn.Module) -> None:
    if type(edge_wise) is not Edgewise:
        raise RuntimeError(f"blocks[{i}].edge_wise is {type(edge_wise).__name__}, expected fairchem's Edgewise")
    if edge_wise.act_type != "gate" or type(edge_wise.act) is not GateActivation or not edge_wise.act.m_prime:
        raise RuntimeError(f"blocks[{i}]: act_type {edge_wise.act_type!r}; conv-layer separation needs the gate "
                           "activation (GateActivation, m_prime=True)")
    for name in ("so2_conv_1", "so2_conv_2"):
        conv = getattr(edge_wise, name)
        # The fast backend's converted layers (SO2_m_Conv_Block) cache [[W1, -W2], [W2, W1]] and would bypass the
        # separation; the functions above are written for these exact classes.
        if type(conv) is not SO2_Convolution or any(type(c) is not SO2_m_Conv for c in conv.so2_m_conv):
            raise RuntimeError(f"blocks[{i}].{name}: expected fairchem's SO2_Convolution with SO2_m_Conv layers")
    c1, c2 = edge_wise.so2_conv_1, edge_wise.so2_conv_2
    if c1.rad_func is None or c1.extra_m0_output_channels is None:
        raise RuntimeError(f"blocks[{i}].so2_conv_1: expected radial weights and extra m = 0 gating outputs")
    if c2.rad_func is not None or c2.extra_m0_output_channels is not None:
        raise RuntimeError(f"blocks[{i}].so2_conv_2: expected no radial weights and no extra m = 0 outputs")


def apply_conv_layer_separation(backbone: nn.Module) -> nn.Module:
    """Separate the conv layers of every block of this UMA backbone. Returns the backbone."""
    blocks = getattr(backbone, "blocks", None)
    if not blocks:
        raise RuntimeError(f"{type(backbone).__name__} has no blocks: is this a UMA backbone?")
    for i, block in enumerate(blocks):
        _check_block(i, block.edge_wise)
    for block in blocks:
        block.edge_wise.__class__ = ConvLayerSeparationEdgewise
    return backbone


def is_separated(module: nn.Module) -> bool:
    """Whether every Edgewise in `module` (a model, a TimeFiLMBackbone or a UMA backbone) has conv-layer separation.
    Refuses a module where only some do."""
    edge_wises = [m for m in module.modules() if isinstance(m, Edgewise)]
    if not edge_wises:
        raise RuntimeError(f"no Edgewise in {type(module).__name__}: is this a UMA model?")
    separated = [type(m) is ConvLayerSeparationEdgewise for m in edge_wises]
    if any(separated) and not all(separated):
        raise RuntimeError(f"conv-layer separation in {sum(separated)} of {len(separated)} Edgewise modules")
    return all(separated)


def check_mode(module: nn.Module, expected: bool) -> str:
    """The model's mode, STOCK or SEPARATED, after checking that it is the one its config asks for (`expected`)."""
    separated = is_separated(module)
    if separated != bool(expected):
        raise RuntimeError(f"the config says {CONFIG_KEY}={bool(expected)}, but the model is "
                           f"{SEPARATED if separated else STOCK}")
    return SEPARATED if separated else STOCK
