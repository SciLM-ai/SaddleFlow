"""W2 gating (saddleflow.models.w2_gate) on the tiny random SaddleFlow model: W2 kept, mirror equivariance by
construction, and the gated model at init equal to the same model with W2 = 0."""

import json

import torch
import pytest

pytest.importorskip("fairchem.core")

from test_parity_even import errors, features, rel  # noqa: E402
from test_pretrained import TINY, _cell  # noqa: E402

from saddleflow.models.w2_gate import W2Gate, W2GatedEdgewise  # noqa: E402
from saddleflow.pretrained import SaddleFlow  # noqa: E402
from saddleflow.utils.parity import so2_layers, w2_report, zero_w2_  # noqa: E402

WIDTH = 4
GATED = {**TINY, "w2_gate_width": WIDTH}


def _randomise_zero_params(m):
    """The FiLM output layers start at zero; give them values so that time matters."""
    with torch.no_grad():
        for p in m.parameters():
            if p.abs().sum() == 0:
                p.normal_(0, 0.01)


def gate_heads(m):
    return [p for n, p in m.named_parameters() if ".w2_gate.heads." in n]


def clone(m, config):
    """A fresh model with m's weights. Not copy.deepcopy: the copy's time-FiLM hooks still point at the original
    TimeFiLMBackbone, so a deep copy computes something else."""
    c = SaddleFlow(config).to("cpu").eval()
    c.load_state_dict(m.state_dict())
    return c


def without_w2(m, config):
    c = clone(m, config)
    zero_w2_(c.backbone)
    return c


@pytest.fixture()
def stock():
    torch.manual_seed(0)
    m = SaddleFlow(TINY).to("cpu").eval()
    _randomise_zero_params(m)
    return m


@pytest.fixture()
def gated_at_init(stock):
    """The gated model with the stock model's weights and the gate as built (heads zero)."""
    m = SaddleFlow(GATED).to("cpu").eval()
    missing, unexpected = m.load_state_dict(stock.state_dict(), strict=False)
    assert not unexpected and missing and all(".w2_gate." in k for k in missing)
    return m


@pytest.fixture()
def gated(gated_at_init):
    """The gated model with gates far from zero: mean |g| about 0.5 on the test cell (the random model's
    pseudoscalars are about 1e-3, times the fixed scale 1e4)."""
    with torch.no_grad():
        for p in gate_heads(gated_at_init):
            p.normal_(0, 0.03)
    return gated_at_init


def test_gate_installed_in_every_block(gated):
    blocks = gated.backbone.backbone.blocks
    assert all(type(b.edge_wise) is W2GatedEdgewise for b in blocks)
    assert all(isinstance(b.edge_wise.w2_gate, W2Gate) for b in blocks)
    assert all(r["w2"] > 0 for r in w2_report(gated.backbone))                  # W2 is kept, not zeroed
    assert len(so2_layers(gated.backbone)) == 2 * len(blocks) * TINY["backbone"]["mmax"]


def test_init_equals_w2_zero(stock, gated_at_init):
    w2_zero = without_w2(stock, TINY)
    at = _cell(fix_first=False)
    with torch.no_grad():
        h_gate, v_gate = features(gated_at_init, at)
        h_zero, v_zero = features(w2_zero, at)
        _, v_stock = features(stock, at)
    assert rel(v_gate, v_zero) < 1e-6 and rel(h_gate, h_zero) < 1e-6
    assert rel(v_gate, v_stock) > 1e-2                                          # so the test above can fail


def test_gated_is_mirror_equivariant(gated, stock):
    assert errors(stock)[1] > 1e-2                                              # stock breaks inversion
    at = _cell(fix_first=False)
    w2_zero = without_w2(gated, GATED)
    with torch.no_grad():
        _, v = features(gated, at)
        _, v_zero = features(w2_zero, at)
    assert rel(v, v_zero) > 1e-2                                                # the gated W2 path contributes
    rot, inv, feat = errors(gated)
    assert rot < 1e-4 and inv < 1e-4 and feat < 1e-4


def test_gated_is_mirror_equivariant_without_activation_checkpointing(gated):
    for block in gated.backbone.backbone.blocks:
        block.edge_wise.activation_checkpoint_chunk_size = None
    rot, inv, feat = errors(gated)
    assert rot < 1e-4 and inv < 1e-4 and feat < 1e-4


def test_ordinary_scalar_gate_breaks_mirror(gated, monkeypatch):
    """Negative control: gate W2 with |p|, which does not change sign under a mirror."""
    original = W2Gate.pseudoscalars
    monkeypatch.setattr(W2Gate, "pseudoscalars", lambda self, x: original(self, x).abs())
    rot, inv, feat = errors(gated)
    assert rot < 1e-4 and inv > 1e-2


def test_gate_heads_get_gradient_at_init(gated_at_init):
    at = _cell(fix_first=False)
    _, v = features(gated_at_init, at)
    v.pow(2).sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in gate_heads(gated_at_init))


def test_from_pretrained_round_trip(gated, tmp_path):
    from safetensors.torch import save_file
    state = {k: v.contiguous() for k, v in gated.state_dict().items() if not k.startswith("backbone.film.")}
    save_file(state, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(GATED))
    loaded = SaddleFlow.from_pretrained(str(tmp_path), device="cpu")
    at = _cell(fix_first=False)
    with torch.no_grad():
        assert rel(features(loaded, at)[1], features(gated, at)[1]) < 1e-6

    (tmp_path / "config.json").write_text(json.dumps(TINY))                    # config without the gate
    with pytest.raises(RuntimeError, match="unexpected keys"):
        SaddleFlow.from_pretrained(str(tmp_path), device="cpu")
