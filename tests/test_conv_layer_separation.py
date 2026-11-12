"""Conv-layer separation (saddleflow.models.conv_layer_separation) on the tiny random SaddleFlow model: each message is
the mirror-symmetric part of the stock message, the model is mirror- and rotation-equivariant, every weight is kept
and trained, and every loader runs the mode its config asks for.

The tests on the real FM2 weights run only when SF_FM2_DIR names a release directory (config.json, model.safetensors),
e.g. the Hugging Face snapshot of SciLM/SaddleFlow revision fm2; they use a GPU when there is one.
"""

import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("fairchem.core")

from fairchem.core.datasets.collaters.simple_collater import data_list_collater  # noqa: E402
from test_parity_even import ROT, errors, features, rel  # noqa: E402
from test_pretrained import TINY, _cell  # noqa: E402

from saddleflow.data.transforms import transform_structure  # noqa: E402
from saddleflow.flow.matching import build_atomic_data  # noqa: E402
from saddleflow.models.conv_layer_separation import (  # noqa: E402
    SEPARATED,
    STOCK,
    ConvLayerSeparationEdgewise,
    check_mode,
    apply_conv_layer_separation,
    is_separated,
    separated_message,
)
from saddleflow.pretrained import SaddleFlow  # noqa: E402
from saddleflow.utils.parity import _w2, parity_from_env, so2_layers, w2_report, zero_w2_  # noqa: E402

SEPARATED_CONFIG = {**TINY, "conv_layer_separation": True}


def _randomise_zero_params(m):
    """The FiLM output layers start at zero; give them values so that time matters."""
    with torch.no_grad():
        for p in m.parameters():
            if p.abs().sum() == 0:
                p.normal_(0, 0.01)


def edge_wises(m):
    return [b.edge_wise for b in m.backbone.backbone.blocks]


def no_activation_checkpointing(m):
    for ew in edge_wises(m):
        ew.activation_checkpoint_chunk_size = None
    return m


@pytest.fixture()
def stock():
    torch.manual_seed(0)
    m = SaddleFlow(TINY).to("cpu").eval()
    _randomise_zero_params(m)
    return m


@pytest.fixture()
def separated(stock):
    """The stock model's weights with conv-layer separation; the state dict loads strictly."""
    m = SaddleFlow(SEPARATED_CONFIG).to("cpu").eval()
    m.load_state_dict(stock.state_dict(), strict=True)
    return m


def with_config(m, config):
    c = SaddleFlow(config).to("cpu").eval()
    c.load_state_dict(m.state_dict())
    return c


# ---------------------------------------------------------------- applying the separation
def test_applied_to_every_block(separated, stock):
    assert all(type(ew) is ConvLayerSeparationEdgewise for ew in edge_wises(separated))
    assert separated.mode == SEPARATED and is_separated(separated)
    assert stock.mode == STOCK and not is_separated(stock)
    assert all(r["w2"] > 0 for r in w2_report(separated.backbone))            # W2 is kept, not zeroed


def test_refuses_other_activations(stock):
    raw = stock.backbone.backbone
    raw.blocks[1].edge_wise.act_type = "s2"
    with pytest.raises(RuntimeError, match="gate activation"):
        apply_conv_layer_separation(raw)
    assert not is_separated(stock)                                             # no block changed


def test_refuses_a_second_application(separated):
    with pytest.raises(RuntimeError, match="expected fairchem's Edgewise"):
        apply_conv_layer_separation(separated.backbone.backbone)


def test_survives_deepcopy(separated):
    c = copy.deepcopy(separated)
    assert c.mode == SEPARATED
    at = _cell(fix_first=False)
    with torch.no_grad():                    # not bitwise: two calls of the same model differ by ~3e-7 on CPU
        assert rel(features(c, at)[1], features(separated, at)[1]) < 1e-5


# ---------------------------------------------------------------- 1. each message is the mirror-symmetric part
def _edge_inputs(model):
    """[(x_message, x_edge)] that every block's so2_conv_1 sees in one forward pass of `model`."""
    seen, handles = [], []
    for ew in edge_wises(model):
        handles.append(ew.so2_conv_1.register_forward_hook(lambda mod, args, out: seen.append(args)))
    with torch.no_grad():
        features(model, _cell(fix_first=False))
    for h in handles:
        h.remove()
    return seen


def _sin_flip(conv):
    """(coefficients,) sign: -1 on the sin half of every m > 0 block of the edge-frame layout."""
    sign = [torch.ones(conv.m_split_sizes[0])]
    for size in conv.m_split_sizes[1:]:
        sign += [torch.ones(size // 2), -torch.ones(size // 2)]
    return torch.cat(sign)[None, :, None]


def _stock_message(ew, x, x_edge):
    out, gating = ew.so2_conv_1(x, x_edge)
    return ew.so2_conv_2(ew.act(gating, out))


def test_message_is_mirror_symmetric_part_of_stock(stock):
    no_activation_checkpointing(stock)
    inputs = _edge_inputs(stock)
    assert len(inputs) == len(edge_wises(stock))
    with torch.no_grad():
        for ew, (x, x_edge) in zip(edge_wises(stock), inputs):
            R = _sin_flip(ew.so2_conv_1)
            m = _stock_message(ew, x, x_edge)
            symmetric = 0.5 * (m + R * _stock_message(ew, R * x, x_edge))
            got = separated_message(ew, x, x_edge)
            assert rel(got, symmetric) < 1e-5
            assert rel(m, symmetric) > 1e-3                                    # the stock message is not symmetric


# ---------------------------------------------------------------- 2. the whole model is mirror- and rotation-equivariant
def test_stock_breaks_inversion(stock):
    rot, inv, feat = errors(stock)
    assert rot < 1e-4 and inv > 1e-2 and feat > 1e-2


@pytest.mark.parametrize("checkpointing", [True, False])
def test_mirror_and_rotation_equivariant(separated, checkpointing):
    if not checkpointing:
        no_activation_checkpointing(separated)
    rot, inv, feat = errors(separated)
    assert rot < 1e-4 and inv < 1e-4 and feat < 1e-4, (rot, inv, feat)


# ---------------------------------------------------------------- 3. with W2 = 0 it is the W2 = 0 model
def test_reduces_to_w2_zero(stock, separated):
    w2_zero, separated_w2_zero = with_config(stock, TINY), with_config(stock, SEPARATED_CONFIG)
    zero_w2_(w2_zero.backbone)
    zero_w2_(separated_w2_zero.backbone)
    at = _cell(fix_first=False)
    with torch.no_grad():
        h_a, v_a = features(w2_zero, at)
        h_b, v_b = features(separated_w2_zero, at)
        _, v_sep = features(separated, at)
    assert rel(v_b, v_a) < 1e-6 and rel(h_b, h_a) < 1e-6
    assert rel(v_sep, v_a) > 1e-2                     # with W2 kept it is not the W2 = 0 model


# ---------------------------------------------------------------- 4. no new parameters
def test_state_dict_is_stock(stock, separated):
    assert list(separated.state_dict()) == list(stock.state_dict())
    assert all(a.shape == b.shape for a, b in zip(separated.state_dict().values(), stock.state_dict().values()))


# ---------------------------------------------------------------- 5. W1 and W2 both train
def test_w1_and_w2_get_gradient(separated):
    _, v = features(separated, _cell(fix_first=False))
    v.pow(2).sum().backward()
    layers = so2_layers(separated.backbone)
    assert len(layers) == 2 * len(edge_wises(separated)) * TINY["backbone"]["mmax"]
    for name, conv in layers:
        w, i = _w2(conv)
        assert w.grad is not None, name
        w1_grad = w.grad[..., :conv.out_channels_half, :]
        assert w1_grad.abs().sum() > 0 and w.grad[i].abs().sum() > 0, name


# ---------------------------------------------------------------- 6. loaders
def _release(model, config, path):
    from safetensors.torch import save_file
    state = {k: v.contiguous() for k, v in model.state_dict().items() if not k.startswith("backbone.film.")}
    save_file(state, str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps(config))


def test_from_pretrained_follows_the_config(separated, tmp_path):
    _release(separated, SEPARATED_CONFIG, tmp_path)
    loaded = SaddleFlow.from_pretrained(str(tmp_path), device="cpu")
    assert loaded.mode == SEPARATED
    at = _cell(fix_first=False)
    with torch.no_grad():
        v = features(separated, at)[1]
        assert rel(features(loaded, at)[1], v) < 1e-6

        # The hazard the config key guards against: the same weights without it load cleanly and run as stock.
        (tmp_path / "config.json").write_text(json.dumps(TINY))
        as_stock = SaddleFlow.from_pretrained(str(tmp_path), device="cpu")
        assert as_stock.mode == STOCK and rel(features(as_stock, at)[1], v) > 1e-2


def test_mode_checks_the_config(stock, separated):
    stock.config = SEPARATED_CONFIG
    with pytest.raises(RuntimeError, match="the config says conv_layer_separation=True"):
        stock.mode
    with pytest.raises(RuntimeError, match="conv_layer_separation=False"):
        check_mode(separated.backbone, False)


def test_parity_env_refused(separated, stock, monkeypatch):
    monkeypatch.setenv("SF_PARITY_EVEN", "1")
    with pytest.raises(RuntimeError, match="conv-layer separation"):
        parity_from_env(separated.backbone)
    assert all(r["w2"] > 0 for r in w2_report(separated.backbone))            # refused before touching W2
    assert parity_from_env(stock.backbone)


# ---------------------------------------------------------------- the real FM2 weights (SF_FM2_DIR)
FM2_DIR = os.environ.get("SF_FM2_DIR")
needs_fm2 = pytest.mark.skipif(not FM2_DIR, reason="SF_FM2_DIR is not set")


def _features_on(model, atoms, t=0.3):
    rec = model._record(atoms, "oc20")
    x = torch.as_tensor(atoms.positions, dtype=torch.float32)
    batch = data_list_collater([build_atomic_data(x, rec["Z"], rec["cell"], rec["task_name"], rec["charge"],
                                                  rec["spin"], rec["fixed"])], otf_graph=True).to(model.device)
    tt = torch.full((1,), t, device=model.device)
    h = model.backbone(batch, tt, batch.batch)["node_embedding"]
    return h, model.velocity_head(h, tt, batch.batch)


def _fm2_errors(model):
    at = _cell(fix_first=False)
    with torch.no_grad():
        _, v = _features_on(model, at)
        _, v_rot = _features_on(model, transform_structure(at, ROT))
        _, v_inv = _features_on(model, transform_structure(at, -np.eye(3)))
    R = torch.as_tensor(ROT, dtype=torch.float32, device=v.device)
    return rel(v_rot, v @ R.T), rel(v_inv, -v), v


@needs_fm2
def test_fm2_loads_with_separation(tmp_path):
    src = Path(FM2_DIR)
    config = json.loads((src / "config.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps({**config, "conv_layer_separation": True}))
    (tmp_path / "model.safetensors").symlink_to((src / "model.safetensors").resolve())    # no copy of the weights
    device = "cuda" if torch.cuda.is_available() else "cpu"

    separated = SaddleFlow.from_pretrained(str(tmp_path), device=device)      # raises on missing or unexpected keys
    assert separated.mode == SEPARATED
    rot_sep, inv_sep, v_sep = _fm2_errors(separated)
    del separated

    stock = SaddleFlow.from_pretrained(str(src), device=device)
    assert stock.mode == STOCK
    rot_stock, inv_stock, v_stock = _fm2_errors(stock)
    zero_w2_(stock.backbone)
    _, _, v_zero = _fm2_errors(stock)
    print(f"FM2 stock:      rotation {rot_stock:.2e}  inversion {inv_stock:.2e}")
    print(f"FM2 separated:  rotation {rot_sep:.2e}  inversion {inv_sep:.2e}  "
          f"vs stock {rel(v_sep, v_stock):.2e}  vs W2 = 0 {rel(v_sep, v_zero):.2e}")
    assert inv_sep <= max(rot_sep, 1e-4)               # inversion error at the noise level of the rotation error
    assert inv_stock > 10 * inv_sep                    # so the test above can fail
    assert rel(v_sep, v_zero) > 1e-3                   # not the W2 = 0 model at step 0
