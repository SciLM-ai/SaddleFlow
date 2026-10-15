"""Tests for the user-facing API (saddleflow.pretrained, saddleflow.search) on a tiny random model."""

import json

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms

pytest.importorskip("fairchem.core")

from saddleflow.pretrained import SaddleFlow  # noqa: E402
from saddleflow.search import count_negative_modes, hessian_eigenvalues, max_displacement  # noqa: E402

TINY = {
    "model_type": "saddleflow-unconditional",
    "backbone": {
        "max_num_elements": 100, "sphere_channels": 16, "lmax": 2, "mmax": 2, "num_layers": 2,
        "hidden_channels": 16, "edge_channels": 16, "num_experts": 2, "cutoff": 6.0, "max_neighbors": 300,
        "distance_function": "gaussian", "num_distance_basis": 16, "norm_type": "rms_norm_sh",
        "act_type": "gate", "ff_type": "spectral", "dataset_list": ["oc20", "omat"],
        "regress_forces": False, "direct_forces": False, "chg_spin_emb_type": "rand_emb", "cs_emb_grad": True,
        "use_composition_embedding": True, "moe_layer_type": "pytorch",
    },
    "film_blocks": [0, 1],
    "head_depth": 3,
    "tasks": {"default": "omat", "map": {"omat": "omat", "oc20": "oc20", "oc22": "oc20"}, "trained": ["oc20", "omat"]},
    "inference": {"sigma": 0.3, "passes": 2, "steps": 2},
}


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = SaddleFlow(TINY).to("cpu").eval()
    with torch.no_grad():                       # the FiLM output layers start at zero; make time matter
        for p in m.parameters():
            if p.abs().sum() == 0:
                p.normal_(0, 0.01)
    return m


def _cell(fix_first=True):
    at = Atoms("Cu3Ag", positions=[[1, 1, 1], [3.5, 1.2, 1.1], [1.1, 3.4, 1.3], [2.4, 2.3, 3.0]],
               cell=[7.0, 7.0, 7.0], pbc=True)
    if fix_first:
        at.set_constraint(FixAtoms(indices=[0]))
    return at


def test_propose_keeps_structure_and_frozen_atoms(model):
    at = _cell()
    props = model.propose(at, n=3, seed=0)
    assert len(props) == 3
    for p in props:
        assert (p.numbers == at.numbers).all() and np.allclose(p.cell, at.cell)
        assert isinstance(p.constraints[0], FixAtoms)
        assert np.allclose(p.positions[0], at.positions[0], atol=1e-5)      # frozen atom never moves
        assert p.info["task_name"] == "omat" and p.info["saddleflow_seed"] == 0
    assert max_displacement(props[0], props[1]) > 0                        # different draws differ


def test_seed_reproducible_and_batching_invariant(model):
    at = [_cell(), _cell(fix_first=False)]
    a = model.propose(at, n=2, seed=3, batch_size=1)
    b = model.propose(at, n=2, seed=3, batch_size=8)
    for pa, pb in zip(sum(a, []), sum(b, [])):
        assert np.allclose(pa.positions, pb.positions, atol=1e-5)


def test_guards(model):
    with pytest.raises(ValueError):
        model.propose(_cell(), sigma=0.5)
    with pytest.raises(ValueError):
        model.propose(_cell(), task_name="omol")
    p = model.propose(_cell(), task_name="oc22", seed=0)[0]
    assert p.info["task_name"] == "oc22"                                    # routed through oc20 internally


def test_from_pretrained_local_directory(model, tmp_path):
    from safetensors.torch import save_file
    state = {k: v.contiguous() for k, v in model.state_dict().items() if not k.startswith("backbone.film.")}
    save_file(state, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(TINY))
    loaded = SaddleFlow.from_pretrained(str(tmp_path), device="cpu")
    x = model.propose(_cell(), n=2, seed=1)
    y = loaded.propose(_cell(), n=2, seed=1)
    for px, py in zip(x, y):
        assert np.allclose(px.positions, py.positions, atol=1e-6)


class _Quadratic(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, h, x0):
        super().__init__()
        self.h, self.x0 = h, x0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        dx = atoms.get_positions().ravel() - self.x0
        self.results = {"energy": 0.5 * dx @ self.h @ dx, "forces": -(self.h @ dx).reshape(-1, 3)}


@pytest.mark.parametrize("n_free", [2, 5])          # dense path (6 dofs) and Lanczos path (15 dofs)
def test_hessian_index_on_quadratic(n_free):
    rng = np.random.default_rng(0)
    at = Atoms("H" * (n_free + 1), positions=rng.uniform(0, 5, (n_free + 1, 3)), cell=[10] * 3, pbc=True)
    at.set_constraint(FixAtoms(indices=[0]))
    q, _ = np.linalg.qr(rng.normal(size=(3 * n_free, 3 * n_free)))
    spectrum = np.concatenate([[-0.5], np.linspace(0.5, 3.0, 3 * n_free - 1)])
    h = np.eye(3 * (n_free + 1)) * 10.0
    h[3:, 3:] = q @ np.diag(spectrum) @ q.T
    at.calc = _Quadratic(h, at.get_positions().ravel())
    vals = hessian_eigenvalues(at, k=2)
    assert np.isclose(vals[0], -0.5, atol=1e-3)
    assert count_negative_modes(at) == 1
