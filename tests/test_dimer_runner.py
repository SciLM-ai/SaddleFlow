"""run_dimer on an analytic double well, E = (x^2 - 1)^2 + 2 y^2 + 2 z^2 on atom 0 (saddle at the origin)."""
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms
from ase.mep.dimer import DimerControl

from saddleflow.classical_saddle_search import PESBackend, run_dimer


class DoubleWell(Calculator):
    """Atom 1 is fixed and feels nothing."""
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        x, y, z = self.atoms.positions[0]
        forces = np.zeros((len(self.atoms), 3))
        forces[0] = -np.array([4 * x * (x ** 2 - 1), 4 * y, 4 * z])
        self.results = dict(energy=(x ** 2 - 1) ** 2 + 2 * y ** 2 + 2 * z ** 2, forces=forces)


def backend():
    atoms = Atoms("H2", positions=[[-0.4, 0.1, 0.05], [5.0, 5.0, 5.0]])
    atoms.set_constraint(FixAtoms([1]))
    pes = PESBackend(atoms, DoubleWell())
    return pes, pes.dofs.pack(atoms.positions)


def test_converges_to_the_saddle_with_the_fixed_atom_untouched():
    pes, x0 = backend()
    out = run_dimer(pes, x0, seed=3, fmax_tol=1e-4, max_steps=200)
    assert out["status"] == "converged"
    assert np.abs(out["final"]).max() < 1e-3
    assert out["curvature"] == pytest.approx(-4.0, abs=0.1)
    assert abs(out["eigenmode"][0]) > 0.99
    assert np.array_equal(out["positions"][1], [5.0, 5.0, 5.0])
    assert out["force_calls"] == pes.force_calls


def test_same_seed_repeats_bitwise():
    runs = [run_dimer(*backend(), seed=7, fmax_tol=1e-4, max_steps=200) for _ in range(2)]
    assert [r["x"].tobytes() for r in runs[0]["trace"]] == [r["x"].tobytes() for r in runs[1]["trace"]]


def test_the_seed_is_required():
    with pytest.raises(ValueError):
        run_dimer(*backend(), seed=None)
    with pytest.raises(TypeError):
        run_dimer(*backend())


def test_query_budget_keeps_a_visited_centre():
    pes, x0 = backend()
    out = run_dimer(pes, x0, seed=3, fmax_tol=1e-4, max_force_calls=7)
    assert out["status"] == "query_budget" and out["force_calls"] <= 7
    centres = [r["x"] for r in out["trace"]]
    assert any(np.array_equal(out["final"], c) for c in centres)


def test_stop_rule_ends_the_run():
    out = run_dimer(*backend(), seed=3, stop=lambda x, e: "drift" if abs(x[0] + 0.4) > 0.05 else None)
    assert out["status"] == "drift"


def test_settings_do_not_leak_into_ase():
    before = dict(DimerControl.parameters)
    run_dimer(*backend(), seed=3, max_num_rot=4, maximum_translation=0.05)
    assert DimerControl.parameters == before
    with pytest.raises(ValueError):
        run_dimer(*backend(), seed=3, not_a_parameter=1)
