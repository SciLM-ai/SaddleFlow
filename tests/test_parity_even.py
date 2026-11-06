"""W2 = 0 (saddleflow.utils.parity) on the tiny random SaddleFlow model: mirror equivariance by construction."""

import copy

import numpy as np
import pytest
import torch

pytest.importorskip("fairchem.core")

from fairchem.core.datasets.collaters.simple_collater import data_list_collater  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
from test_pretrained import TINY, _cell  # noqa: E402

from saddleflow.data.transforms import transform_structure  # noqa: E402
from saddleflow.flow.matching import build_atomic_data  # noqa: E402
from saddleflow.pretrained import SaddleFlow  # noqa: E402
from saddleflow.utils.parity import (  # noqa: E402
    assert_parity_even,
    enforce_parity_even,
    parity_sign,
    so2_layers,
    w2_report,
)


@pytest.fixture()
def model():
    torch.manual_seed(0)
    m = SaddleFlow(TINY).to("cpu").eval()
    with torch.no_grad():                       # the FiLM output layers start at zero; make time matter
        for p in m.parameters():
            if p.abs().sum() == 0:
                p.normal_(0, 0.01)
    return m


def features(model, atoms, t=0.3):
    """(node features (N, num_sph, C), velocity (N, 3)) from one network call."""
    rec = model._record(atoms, None)
    x = torch.as_tensor(atoms.positions, dtype=torch.float32)
    batch = data_list_collater([build_atomic_data(x, rec["Z"], rec["cell"], rec["task_name"], rec["charge"],
                                                  rec["spin"], rec["fixed"])], otf_graph=True)
    tt = torch.full((1,), t)
    h = model.backbone(batch, tt, batch.batch)["node_embedding"]
    return h, model.velocity_head(h, tt, batch.batch)


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


ROT = Rotation.random(random_state=1).as_matrix()


def errors(model):
    """Relative velocity errors under a random rotation and under inversion, and the feature parity error."""
    at = _cell(fix_first=False)
    with torch.no_grad():
        h, v = features(model, at)
        _, v_rot = features(model, transform_structure(at, ROT))
        h_inv, v_inv = features(model, transform_structure(at, -np.eye(3)))
    sign = parity_sign(model.backbone.backbone.lmax)[None, :, None]
    R = torch.as_tensor(ROT, dtype=torch.float32)
    return rel(v_rot, v @ R.T), rel(v_inv, -v), rel(h_inv, sign * h)


def test_stock_model_breaks_inversion_only(model):
    rot, inv, feat = errors(model)
    assert rot < 1e-4
    assert inv > 1e-2 and feat > 1e-2               # so the parity tests below can fail


def test_w2_zero_is_mirror_equivariant(model):
    stock = copy.deepcopy(model)
    enforce_parity_even(model.backbone)
    assert_parity_even(model.backbone)
    rot, inv, feat = errors(model)
    assert rot < 1e-4 and inv < 1e-4 and feat < 1e-4
    assert any(r["w2"] > 0 for r in w2_report(stock.backbone))
    assert all(r["w2"] == 0 for r in w2_report(model.backbone))


def test_w2_stays_zero_in_training(model):
    enforce_parity_even(model.backbone)
    w1 = [p.detach().clone() for p in model.parameters()]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1)
    at = _cell(fix_first=False)
    for _ in range(3):
        opt.zero_grad()
        _, v = features(model, at)
        v.pow(2).sum().backward()
        opt.step()
    assert_parity_even(model.backbone)
    assert any(not torch.equal(a, b) for a, b in zip(w1, model.parameters()))      # it did train
    _, conv = so2_layers(model.backbone)[0]
    with torch.no_grad():
        conv.fc.weights[0, -1, 0] = 1e-3                                           # a leak is caught
    with pytest.raises(RuntimeError, match="W2 is not zero"):
        assert_parity_even(model.backbone)


def test_frozen_w2_needs_no_hook(model):
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    enforce_parity_even(model.backbone)
    assert_parity_even(model.backbone)


def test_cached_block_matrix_refused(model):
    from fairchem.core.models.uma.nn.so2_layers import SO2_m_Conv_Block
    model.backbone.extra = SO2_m_Conv_Block(1, 4, 4, 2, 2)
    with pytest.raises(RuntimeError, match="SO2_m_Conv_Block"):
        assert_parity_even(model.backbone)
