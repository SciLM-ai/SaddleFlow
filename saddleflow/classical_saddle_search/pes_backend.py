"""ASE potential access, finite differences, and honest PES-query counting."""

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from .free_dofs import FreeDOFs


class QueryBudgetExceeded(RuntimeError):
    pass


class CountingCalculator(Calculator):
    """Count actual calculator evaluations, not optimizer steps or cache reads.

    Energy and forces at one geometry are one PES query. In a batched evaluator,
    a batch of six geometries is still SIX queries, not one.
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, calculator, backend):
        super().__init__()
        self.calculator = calculator
        self.backend = backend

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        self.backend.reserve_queries(1)
        super().calculate(atoms, properties, system_changes)
        self.calculator.calculate(atoms, ["energy", "forces"], system_changes)
        self.results = {key: self.calculator.results[key] for key in ("energy", "forces")}
        self.results["free_energy"] = self.calculator.results.get("free_energy", self.results["energy"])


class PESBackend:
    """Any ASE energy/force calculator, restricted to the allowed coordinates.

    ``batch_evaluate``, if supplied, accepts a list of ASE structures and returns
    (energies, full_forces). Dense finite differences can then run in one GPU
    batch. Scalar and batched calls use the same potential and query accounting.
    No parameter gradients or second-order autograd are needed.
    """

    def __init__(self, atoms, calculator, batch_evaluate=None):
        self.dofs = FreeDOFs(atoms)
        self.atoms = atoms.copy()
        self.force_calls = 0
        self.max_force_calls = None
        self.batch_evaluate = batch_evaluate
        self.atoms.calc = CountingCalculator(calculator, self)

    def reserve_queries(self, count):
        if self.max_force_calls is not None and self.force_calls + count > self.max_force_calls:
            raise QueryBudgetExceeded("PES query budget exhausted")
        self.force_calls += count

    def set_coordinates(self, x):
        # Do not wrap fixed atoms, or change their positions even by a cell image.
        self.atoms.set_positions(self.dofs.expand(x), apply_constraint=False)

    def forces(self, x):
        self.set_coordinates(x)
        return self.dofs.pack(self.atoms.get_forces(apply_constraint=False))

    def energy(self, x):
        self.set_coordinates(x)
        return float(self.atoms.get_potential_energy())

    def forces_many(self, coordinates):
        if self.batch_evaluate is None:
            return np.asarray([self.forces(x) for x in coordinates])
        self.reserve_queries(len(coordinates))
        structures = [self.dofs.atoms_at(x) for x in coordinates]
        _, forces = self.batch_evaluate(structures)
        return np.asarray([self.dofs.pack(f) for f in forces])

    def hvp(self, x, vector, eps=0.002):
        """H v = -dF/dx v, central FD with a fixed displacement norm in A."""
        norm = np.linalg.norm(vector)
        if norm < 1e-14:
            return np.zeros_like(vector)
        direction = np.asarray(vector) / norm
        try:
            plus, minus = self.forces_many([x + eps * direction, x - eps * direction])
            return -(plus - minus) * norm / (2 * eps)
        finally:
            self.set_coordinates(x)

    def free_hessian(self, x, eps=0.002):
        """Numerical free-space Hessian. Only suitable for small free spaces."""
        axes = np.eye(self.dofs.dimension)
        coordinates = [point for axis in axes for point in (x + eps * axis, x - eps * axis)]
        try:
            force = self.forces_many(coordinates)
            raw = (-(force[0::2] - force[1::2]) / (2 * eps)).T
            asymmetry = float(np.linalg.norm(raw - raw.T) / max(np.linalg.norm(raw), 1e-14))
            return 0.5 * (raw + raw.T), asymmetry
        finally:
            self.set_coordinates(x)


def make_uma_calculator(model="uma-s-1p2", device="cuda", task_name="omat"):
    """Load one frozen pretrained UMA predictor; return scalar + batched access.

    Use the same FAIRChem converter and predictor for both access paths. Charge
    and spin are read from ASE metadata, just as in normal UMA calculations.
    The predictor handles coordinate autograd for forces internally: wrapping
    predict() in torch.inference_mode() would break conservative force models.
    """
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    from fairchem.core.datasets.collaters.simple_collater import data_list_collater

    predictor = pretrained_mlip.get_predict_unit(model, device=device)
    # Different FairChem releases expose the nn.Module directly or as a DDP
    # module. Prediction remains coordinate-differentiable for energy forces,
    # but this baseline never trains the potential's parameters.
    if hasattr(predictor, "model"):
        predictor.model.eval()
        for parameter in predictor.model.parameters():
            parameter.requires_grad_(False)
    calculator = FAIRChemCalculator(predictor, task_name=task_name)

    def batch_evaluate(structures):
        data = []
        for atoms in structures:
            predictor.validate_atoms_data(atoms, task_name)
            data.append(calculator.a2g(atoms))
        batch = data_list_collater(data, otf_graph=True)
        prediction = predictor.predict(batch)
        energies = prediction["energy"].detach().cpu().numpy().reshape(-1)
        forces = prediction["forces"].detach().cpu().numpy()
        offsets = np.cumsum([0] + [len(atoms) for atoms in structures])
        return energies, [forces[offsets[i]:offsets[i + 1]] for i in range(len(structures))]

    return calculator, batch_evaluate
