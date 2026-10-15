"""LiC-only starts and catalog metrics; the solvers know nothing about LiC."""

from itertools import combinations
import hashlib
from pathlib import Path

import numpy as np
from ase.geometry import find_mic, wrap_positions
from ase.io import read

from saddleflow.classical_saddle_search import FreeDOFs


def read_npz(path):
    with np.load(path,allow_pickle=False) as archive:
        return {key:archive[key] for key in archive.files}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_geometry(train_traj, test_traj):
    train = read(train_traj, index=":")
    test = read(test_traj, index=":")
    reference = train[0].copy()
    dofs = FreeDOFs(reference)
    free_indices = np.flatnonzero(dofs.free)
    if len(free_indices) != 1 or reference.numbers[free_indices[0]] != 3:
        raise ValueError("This LiC visualization expects one free Li atom; the solvers do not.")
    li = int(free_indices[0])
    carbon_indices = np.flatnonzero(reference.numbers == 6)
    for frame in train + test:
        if not np.array_equal(frame.numbers, reference.numbers):
            raise ValueError("LiC frames must use consistent atom ordering.")
        if not np.allclose(frame.cell, reference.cell) or not np.array_equal(frame.pbc, reference.pbc):
            raise ValueError("LiC frames must share the cell and periodic directions.")
        if not np.array_equal(FreeDOFs(frame).free, dofs.free):
            raise ValueError("LiC frames must share the same FixAtoms constraints.")
        if not np.allclose(frame.positions[~dofs.free], reference.positions[~dofs.free]):
            raise ValueError("LiC frames must share the fixed scaffold.")
    train_saddles = np.asarray([a.positions[li] for a in train[1::3]])
    test_saddles = np.asarray([a.positions[li] for a in test[1::3]])
    true_saddles = np.vstack([train_saddles, test_saddles])
    midpoints, bonds = [], []
    for i, j in combinations(carbon_indices, 2):
        bond, length = find_mic(reference.positions[j] - reference.positions[i],
                                reference.cell, reference.pbc)
        if 0.1 < length < 1.8:
            midpoints.append(reference.positions[i] + 0.5 * bond)
            bonds.append((i, j))
    midpoints = wrap_positions(np.asarray(midpoints), reference.cell, reference.pbc)
    # A false bridge is a C-C midpoint lacking a stored saddle in the XY plane.
    # This catalog is a diagnostic, not a definition of physical stationarity.
    xy_delta = midpoints[:, None] - true_saddles[None, :]
    xy_delta[:, :, 2] = 0
    xy_mic, _ = find_mic(xy_delta.reshape(-1, 3), reference.cell, [True, True, False])
    distances = np.linalg.norm(xy_mic.reshape(len(midpoints), -1, 3), axis=2).min(axis=1)
    false_bridges = midpoints[distances > 0.5].copy()
    false_bridges[:, 2] = true_saddles[:, 2].mean()
    if len(false_bridges) != 10:
        raise ValueError(f"Expected the existing 10 false bridges, found {len(false_bridges)}.")
    return reference, dict(
        li_index=li, carbon_indices=carbon_indices, bonds=np.asarray(bonds),
        train_saddles=train_saddles, test_saddles=test_saddles,
        false_bridges=false_bridges, cell=np.asarray(reference.cell),
        pbc=reference.pbc.copy(), positions=reference.positions.copy(),
        numbers=reference.numbers.copy(), free_mask=dofs.free,
    )


def nearest_distances(points, references, cell, pbc):
    delta = np.asarray(points)[:, None] - np.asarray(references)[None, :]
    mic, _ = find_mic(delta.reshape(-1, 3), cell, pbc)
    return np.linalg.norm(mic.reshape(len(points), len(references), 3), axis=2).min(axis=1)


def classify(points, geometry, true_radius=0.1, false_radius=0.3):
    true = np.vstack([geometry["train_saddles"], geometry["test_saddles"]])
    d_true = nearest_distances(points, true, geometry["cell"], geometry["pbc"])
    d_false = nearest_distances(points, geometry["false_bridges"], geometry["cell"], geometry["pbc"])
    outcome = np.full(len(points), 2, dtype=int)  # 0 true; 1 false region; 2 other
    outcome[d_false < false_radius] = 1
    outcome[d_true < true_radius] = 0  # a catalogued true hit has precedence
    return outcome, d_true, d_false


def generate_starts(args, geometry):
    """Generate starts once, WITHOUT consulting solver output or eigenmodes."""
    rng = np.random.default_rng(args.seed)
    starts, population, radius, parent = [], [], [], []

    def append(point, group, r=np.nan, owner=-1):
        starts.append(point)
        population.append(group)
        radius.append(r)
        parent.append(owner)

    groups = args.starts.split(",")
    if groups == ["mixed"]:
        groups = ["false", "true", "shells"]
    for group in groups:
        if group == "false":
            for point in geometry["false_bridges"]:
                append(point.copy(), "false_exact")
                for _ in range(args.samples_per_site):
                    append(point + rng.normal(0, args.false_noise_sigma, 3), "false_noise")
        elif group == "true":
            for i, point in enumerate(geometry["train_saddles"]):
                for _ in range(args.samples_per_site):
                    append(point + rng.normal(0, args.true_noise_sigma, 3), "true_noise", owner=i)
        elif group == "shells":
            radii = [float(r) for r in args.radii.split(",")]
            for i, point in enumerate(geometry["train_saddles"]):
                for r in radii:
                    for _ in range(args.directions_per_radius):
                        direction = rng.normal(size=3 if args.shell_dimensions == "xyz" else 2)
                        direction /= np.linalg.norm(direction)
                        delta = np.zeros(3)
                        delta[:len(direction)] = r * direction
                        append(point + delta, "shells", r, i)
        elif group == "grid":
            for i in range(args.grid):
                for j in range(args.grid):
                    point = ((i + 0.5) / args.grid * geometry["cell"][0]
                             + (j + 0.5) / args.grid * geometry["cell"][1])
                    point[2] = geometry["train_saddles"][:, 2].mean()
                    append(point, "grid")
        else:
            raise ValueError(f"Unknown start population: {group}")
    if args.starts_npz:
        # Reuse sources, never previous finals. Only t=0 sources are meaningful
        # for a side-by-side static saddle-search experiment.
        with np.load(args.starts_npz, allow_pickle=False) as archive:
            points = archive["starts"]
            use = (archive["start_step"] == 0 if "start_step" in archive
                   else np.ones(len(points), dtype=bool))
            starts = list(points[use])
            if np.asarray(starts).ndim != 2 or np.asarray(starts).shape[1] != 3:
                raise ValueError("The source archive must contain Li-only starts of shape (M,3).")
            population = ["archive"] * len(starts)
            radius = list(archive["radius"][use]) if "radius" in archive else [np.nan] * len(starts)
            parent = [-1] * len(starts)
    if args.limit_starts is not None:
        # Seeded subset instead of truncating to only the first population.
        choose = np.sort(rng.choice(len(starts), min(args.limit_starts, len(starts)), replace=False))
    else:
        choose = np.arange(len(starts))
    return dict(starts=np.asarray(starts)[choose],
                population=np.asarray(population)[choose],
                radius=np.asarray(radius)[choose], parent=np.asarray(parent)[choose])
