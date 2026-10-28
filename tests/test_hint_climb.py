"""climb_along on an analytic double well, E = (x^2 - 1)^2 + 2 y^2 + 2 z^2 (minima x = +-1, saddle at 0)."""
import numpy as np

from saddleflow.classical_saddle_search import climb_along


class DoubleWell:
    """The PESBackend interface climb_along uses: forces, energy and a force-call counter."""
    force_calls = 0

    def set_coordinates(self, x):
        pass

    def energy(self, x):
        return (x[0] ** 2 - 1) ** 2 + 2 * x[1] ** 2 + 2 * x[2] ** 2

    def forces(self, x):
        self.force_calls += 1
        return -np.array([4 * x[0] * (x[0] ** 2 - 1), 4 * x[1], 4 * x[2]])


def test_leaves_the_bowl_along_the_hint():
    out = climb_along(DoubleWell(), np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.3, 0.0]), push=0.05)
    # Negative curvature along x starts at |x| < 1/sqrt(3) = 0.577. The push keeps the hint's own y part;
    # only motion perpendicular to the hint is relaxed, and Sella removes the rest after handover.
    assert out["status"] == "left_bowl"
    assert -0.6 < out["x"][0] < 0.0 and abs(out["x"][2]) < 1e-12
    assert out["steps"] + 1 <= out["force_calls"] <= 2 * out["steps"] + 1


def test_a_wall_is_a_failure():
    out = climb_along(DoubleWell(), np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), push=0.1, max_rise=1.0)
    assert out["status"] == "wall"
