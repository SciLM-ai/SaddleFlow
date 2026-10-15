"""Compare UMA forces at verified LiC saddles and saddle-less C--C bridges.

This is a static physical probe, not a SaddleFlow inference script.  Each
evaluated structure has the same frozen 126-carbon sheet; only the Li position
changes.  It writes exactly one human-readable text report.

The two populations come from the distributed LiC trajectories:

* true saddles are the stored S frame of each [R, S, P] triplet;
* false bridges are periodic C--C bond midpoints farther than 0.5 A from every
  stored saddle.  The supplied data have 179 true saddles and 10 false bridges.

False bridges have no stored three-dimensional Li coordinate.  They are tested
at (1) a common Li height, the median true-saddle z, and (2) a constrained
height relaxation that holds x,y fixed at the bridge and chooses the lowest
UMA-energy z from a documented one-dimensional scan.

Example, from the repository root:

    CUDA_VISIBLE_DEVICES=0 python examples/LiC/experiments/diagnostics/probe_uma_false_saddles.py \
        --device cuda \
        --output /sfs/weka/scratch/gqs7zr/saddleflow/lic_uma_probe.txt

The script evaluates frozen pretrained UMA-S-1.2.  It intentionally does not
use a fine-tuned SaddleFlow backbone as an energy calculator.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase.io import read


N_CARBON = 126
LI = 126


@dataclass(frozen=True)
class Bridge:
    """One periodic C--C bridge plus its local force-coordinate frame."""

    carbon_i: int
    carbon_j: int
    midpoint: np.ndarray
    bond_hat: np.ndarray
    hop_hat: np.ndarray
    normal_hat: np.ndarray
    length: float


@dataclass(frozen=True)
class Measurement:
    """UMA energy and Li-force components at one nominated Li position."""

    energy: float
    force_norm: float
    abs_force_bond: float
    abs_force_hop: float
    abs_force_normal: float
    li_position: np.ndarray
    bridge_index: int


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parents[2]  # examples/LiC
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--train-traj", default=str(here / "train_set.traj"))
    parser.add_argument("--test-traj", default=str(here / "test_set.traj"))
    parser.add_argument(
        "--output", default=str(here / "uma_true_vs_false_report.txt"),
        help="The only file written by this script.",
    )
    parser.add_argument("--device", default="cuda", help="UMA device, e.g. cuda or cpu.")
    parser.add_argument("--backbone", default="uma-s-1p2")
    parser.add_argument("--task-name", default="omat")

    # These are deliberately the same geometric definitions used by the LiC
    # plotting code, so the report and red-cross sites cannot drift apart.
    parser.add_argument("--bond-cutoff", type=float, default=1.8, metavar="A")
    parser.add_argument("--false-distance-cutoff", type=float, default=0.5, metavar="A")
    parser.add_argument("--expected-false-count", type=int, default=10)
    parser.add_argument("--true-match-cutoff", type=float, default=0.30, metavar="A")

    # Control 2: coarse scan locates the vertical basin; fine scan reduces
    # discretisation error in the selected height and its residual F_normal.
    parser.add_argument("--z-search-half-width", type=float, default=0.75, metavar="A")
    parser.add_argument("--z-coarse-step", type=float, default=0.05, metavar="A")
    parser.add_argument("--z-fine-step", type=float, default=0.005, metavar="A")
    return parser.parse_args()


def unit(vector: np.ndarray, name: str) -> np.ndarray:
    """Return a unit vector, with a useful error for malformed geometry."""
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError(f"Cannot normalize zero-length {name} vector.")
    return vector / norm


def minimum_image_vector(delta: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Minimum-image Cartesian vector for ASE's row-vector cell convention."""
    fractional = np.asarray(delta) @ np.linalg.inv(cell)
    fractional -= np.round(fractional)
    return fractional @ cell


def wrap_into_cell(position: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Wrap one Cartesian position into the periodic cell."""
    fractional = np.asarray(position) @ np.linalg.inv(cell)
    fractional -= np.floor(fractional)
    return fractional @ cell


def read_and_validate(path: str) -> list:
    """Read LiC [R,S,P,...] frames and validate its known atom ordering."""
    frames = list(read(path, index=":"))
    if not frames or len(frames) % 3:
        raise ValueError(f"{path} must contain a non-empty multiple of three frames.")
    for frame_index, atoms in enumerate(frames):
        atomic_numbers = atoms.get_atomic_numbers()
        correct_layout = (
            len(atomic_numbers) == LI + 1
            and np.all(atomic_numbers[:N_CARBON] == 6)
            and atomic_numbers[LI] == 3
        )
        if not correct_layout:
            raise ValueError(
                f"{path}, frame {frame_index}: expected 126 C atoms followed by one Li."
            )
    return frames


def verify_common_sheet(frames: list) -> tuple[object, float]:
    """Ensure all data frames share one fixed carbon sheet and cell."""
    template = frames[0]
    reference_carbon = template.get_positions()[:N_CARBON]
    reference_cell = np.asarray(template.get_cell())
    maximum_difference = 0.0
    for frame_index, atoms in enumerate(frames):
        if not np.allclose(atoms.get_cell(), reference_cell, atol=1e-8, rtol=0.0):
            raise ValueError(f"Frame {frame_index} has a different cell from the reference.")
        carbon_difference = np.linalg.norm(
            atoms.get_positions()[:N_CARBON] - reference_carbon, axis=1
        ).max()
        maximum_difference = max(maximum_difference, float(carbon_difference))
    if maximum_difference > 1e-5:
        raise ValueError(
            "The LiC frames should have an identical frozen carbon sheet, but the "
            f"largest carbon displacement is {maximum_difference:.3e} A."
        )
    return template, maximum_difference


def extract_saddles(frames: list) -> np.ndarray:
    """Return Li positions from the S frame in each [R,S,P] triplet."""
    return np.asarray([atoms.get_positions()[LI] for atoms in frames[1::3]], dtype=float)


def build_cc_bridges(template, bond_cutoff: float) -> list[Bridge]:
    """Enumerate periodic C--C bonds and their bond/hop/surface-normal axes."""
    carbon = template.get_positions()[:N_CARBON]
    cell = np.asarray(template.get_cell())
    normal_hat = unit(np.cross(cell[0], cell[1]), "surface-normal")
    bridges: list[Bridge] = []

    for carbon_i, carbon_j in itertools.combinations(range(N_CARBON), 2):
        bond = minimum_image_vector(carbon[carbon_j] - carbon[carbon_i], cell)
        length = float(np.linalg.norm(bond))
        if not 0.1 < length < bond_cutoff:
            continue
        in_plane_bond = bond - np.dot(bond, normal_hat) * normal_hat
        bond_hat = unit(in_plane_bond, "in-plane C--C bond")
        hop_hat = unit(np.cross(normal_hat, bond_hat), "in-plane hop")
        midpoint = wrap_into_cell(carbon[carbon_i] + 0.5 * bond, cell)
        bridges.append(
            Bridge(carbon_i, carbon_j, midpoint, bond_hat, hop_hat, normal_hat, length)
        )
    return bridges


def nearest_in_plane(
    points: np.ndarray, references: np.ndarray, cell: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest reference index and x-y distance over a 3x3 set of PBC images."""
    nearest_index = np.full(len(points), -1, dtype=int)
    nearest_distance = np.full(len(points), np.inf, dtype=float)
    for shift_i, shift_j in itertools.product((-1, 0, 1), repeat=2):
        offset = shift_i * cell[0] + shift_j * cell[1]
        delta = points[:, None, :2] - (references[None, :, :2] + offset[:2])
        distances = np.linalg.norm(delta, axis=2)
        candidate_index = distances.argmin(axis=1)
        candidate_distance = distances.min(axis=1)
        use_candidate = candidate_distance < nearest_distance
        nearest_index[use_candidate] = candidate_index[use_candidate]
        nearest_distance[use_candidate] = candidate_distance[use_candidate]
    return nearest_index, nearest_distance


def identify_false_bridges(
    bridges: list[Bridge], all_saddles: np.ndarray, cell: np.ndarray, false_cutoff: float
) -> tuple[np.ndarray, np.ndarray]:
    """Identify C--C midpoints that carry no stored true saddle."""
    midpoints = np.asarray([bridge.midpoint for bridge in bridges])
    _, distance_to_saddle = nearest_in_plane(midpoints, all_saddles, cell)
    return np.flatnonzero(distance_to_saddle > false_cutoff), distance_to_saddle


def assign_saddles_to_bridges(
    saddles: np.ndarray, bridges: list[Bridge], cell: np.ndarray, cutoff: float
) -> tuple[np.ndarray, np.ndarray]:
    """Associate each stored saddle with the local axes of its nearest C--C bond."""
    midpoints = np.asarray([bridge.midpoint for bridge in bridges])
    bridge_indices, distances = nearest_in_plane(saddles, midpoints, cell)
    if float(distances.max()) > cutoff:
        raise ValueError(
            f"A stored saddle is {distances.max():.3f} A from the nearest C--C bridge; "
            f"expected <= {cutoff:.3f} A."
        )
    return bridge_indices, distances


def make_uma_calculator(backbone: str, device: str, task_name: str):
    """Load one frozen pretrained UMA calculator for all measurements."""
    # Delayed import means --help and geometry-only failures do not load UMA.
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    print(f"[UMA] loading frozen pretrained {backbone!r} on {device}", flush=True)
    predictor = pretrained_mlip.get_predict_unit(backbone, device=device)
    return FAIRChemCalculator(predictor, task_name=task_name)


def evaluate_li_position(calculator, template, position: np.ndarray, bridge: Bridge,
                         bridge_index: int) -> Measurement:
    """Evaluate one copy of the sheet with Li at ``position``."""
    atoms = template.copy()
    positions = atoms.get_positions()
    positions[LI] = position
    atoms.set_positions(positions)
    atoms.calc = calculator

    # We read only Li.  Asking for unconstrained forces makes the intent clear:
    # fixed-carbon constraints must not affect the reported physical Li force.
    li_force = np.asarray(atoms.get_forces(apply_constraint=False)[LI], dtype=float)
    energy = float(atoms.get_potential_energy())
    return Measurement(
        energy=energy,
        force_norm=float(np.linalg.norm(li_force)),
        abs_force_bond=float(abs(np.dot(li_force, bridge.bond_hat))),
        abs_force_hop=float(abs(np.dot(li_force, bridge.hop_hat))),
        abs_force_normal=float(abs(np.dot(li_force, bridge.normal_hat))),
        li_position=np.asarray(position, dtype=float),
        bridge_index=int(bridge_index),
    )


def evaluate_population(calculator, template, positions: np.ndarray, bridge_indices: np.ndarray,
                        bridges: list[Bridge], label: str) -> list[Measurement]:
    """Evaluate one population, with concise progress for long UMA runs."""
    results: list[Measurement] = []
    for number, (position, bridge_index) in enumerate(zip(positions, bridge_indices), start=1):
        results.append(
            evaluate_li_position(calculator, template, position, bridges[int(bridge_index)], int(bridge_index))
        )
        if number == len(positions) or number % 20 == 0:
            print(f"[UMA] {label}: {number}/{len(positions)}", flush=True)
    return results


def inclusive_grid(start: float, stop: float, step: float) -> np.ndarray:
    """Create an inclusive grid with stable endpoints."""
    count = int(round((stop - start) / step))
    return np.linspace(start, stop, count + 1)


def height_relax_false_bridges(
    calculator,
    template,
    bridges: list[Bridge],
    false_indices: np.ndarray,
    common_z: float,
    half_width: float,
    coarse_step: float,
    fine_step: float,
) -> tuple[list[Measurement], int]:
    """Perform the explicitly constrained, one-dimensional UMA height control."""
    if half_width <= 0 or coarse_step <= 0 or fine_step <= 0:
        raise ValueError("All z-search widths and steps must be positive.")
    if fine_step > coarse_step:
        raise ValueError("--z-fine-step cannot be larger than --z-coarse-step.")

    lower, upper = common_z - half_width, common_z + half_width
    coarse_grid = inclusive_grid(lower, upper, coarse_step)
    selected: list[Measurement] = []
    boundary_minima = 0

    for number, bridge_index in enumerate(false_indices, start=1):
        bridge = bridges[int(bridge_index)]
        coarse = [
            evaluate_li_position(
                calculator, template,
                np.array([bridge.midpoint[0], bridge.midpoint[1], z]),
                bridge, int(bridge_index),
            )
            for z in coarse_grid
        ]
        coarse_best = int(np.argmin([result.energy for result in coarse]))
        if coarse_best in (0, len(coarse) - 1):
            # Do not hide an insufficient window by silently changing it.
            boundary_minima += 1
            selected.append(coarse[coarse_best])
        else:
            center = coarse_grid[coarse_best]
            fine_grid = inclusive_grid(
                max(lower, center - coarse_step), min(upper, center + coarse_step), fine_step
            )
            fine = [
                evaluate_li_position(
                    calculator, template,
                    np.array([bridge.midpoint[0], bridge.midpoint[1], z]),
                    bridge, int(bridge_index),
                )
                for z in fine_grid
            ]
            selected.append(min(fine, key=lambda result: result.energy))
        print(f"[UMA] false bridges, height-relaxed: {number}/{len(false_indices)}", flush=True)
    return selected, boundary_minima


def summarize(results: list[Measurement], reference_energy: float) -> dict[str, float]:
    """Summarize one population for the human-readable report."""
    if not results:
        raise ValueError("Cannot summarize an empty population.")
    energy = np.asarray([result.energy for result in results])
    force = np.asarray([result.force_norm for result in results])
    bond = np.asarray([result.abs_force_bond for result in results])
    hop = np.asarray([result.abs_force_hop for result in results])
    normal = np.asarray([result.abs_force_normal for result in results])
    return {
        "N": float(len(results)),
        "mean_E": float(energy.mean()),
        "mean_dE": float((energy - reference_energy).mean()),
        "mean_F": float(force.mean()),
        "mean_bond": float(bond.mean()),
        "mean_hop": float(hop.mean()),
        "mean_normal": float(normal.mean()),
        "median_F": float(np.median(force)),
        "median_bond": float(np.median(bond)),
        "median_hop": float(np.median(hop)),
        "median_normal": float(np.median(normal)),
    }


def format_table(rows: list[tuple[str, dict[str, float]]], kind: str) -> str:
    """Format fixed-width mean or median metrics without external dependencies."""
    if kind == "mean":
        headers = ["population", "N", "E_UMA (eV)", "dE vs test (eV)", "|F_Li|", "|F_bond|", "|F_hop|", "|F_normal|"]
        keys = ["N", "mean_E", "mean_dE", "mean_F", "mean_bond", "mean_hop", "mean_normal"]
    elif kind == "median":
        headers = ["population", "N", "median |F_Li|", "median |F_bond|", "median |F_hop|", "median |F_normal|"]
        keys = ["N", "median_F", "median_bond", "median_hop", "median_normal"]
    else:
        raise ValueError(f"Unknown table kind {kind!r}")

    body = []
    for label, row in rows:
        values = [row[key] for key in keys]
        body.append([label, f"{int(values[0])}"] + [f"{value:.6f}" for value in values[1:]])
    widths = [max(len(headers[column]), *(len(line[column]) for line in body))
              for column in range(len(headers))]
    render = lambda line: "  ".join(value.ljust(width) for value, width in zip(line, widths))
    divider = "  ".join("-" * width for width in widths)
    return "\n".join([render(headers), divider] + [render(line) for line in body])


def make_report(
    args: argparse.Namespace,
    n_train: int,
    n_test: int,
    n_bridges: int,
    n_false: int,
    common_z: float,
    max_carbon_difference: float,
    max_true_midpoint_distance: float,
    boundary_minima: int,
    rows: list[tuple[str, dict[str, float]]],
) -> str:
    """Create the script's sole output file."""
    lines = [
        "UMA true-saddle versus false-bridge probe",
        "=======================================",
        "",
        "Purpose",
        "-------",
        f"Frozen pretrained {args.backbone} is evaluated on the LiC fixed-carbon sheet.",
        "No SaddleFlow checkpoint or flow trajectory is used.",
        "",
        "Input provenance",
        "----------------",
        f"train trajectory: {Path(args.train_traj).resolve()}",
        f"test trajectory : {Path(args.test_traj).resolve()}",
        f"stored saddles  : {n_train} train + {n_test} held-out test = {n_train + n_test}",
        "frozen sheet   : 126 C atoms shared by all frames; "
        f"largest coordinate difference = {max_carbon_difference:.3e} A",
        f"UMA task       : {args.task_name}",
        "",
        "False-bridge reconstruction",
        "---------------------------",
        f"Periodic C--C bonds shorter than {args.bond_cutoff:.3f} A: {n_bridges}",
        "False rule: nearest stored saddle farther than "
        f"{args.false_distance_cutoff:.3f} A in-plane.",
        f"False bridges found: {n_false}",
        "Stored-saddle check: maximum distance to nearest C--C midpoint = "
        f"{max_true_midpoint_distance:.6f} A",
        "",
        "False-bridge height controls",
        "----------------------------",
        "Control 1: Li z is the median stored-saddle z = "
        f"{common_z:.6f} A.",
        "Control 2: Li x,y are fixed at the same bridge midpoint; UMA chooses the",
        "lowest-energy z from a coarse-to-fine scan over "
        f"[{common_z - args.z_search_half_width:.3f}, {common_z + args.z_search_half_width:.3f}] A.",
        f"Coarse/fine z steps: {args.z_coarse_step:.3f}/{args.z_fine_step:.3f} A.",
        f"Control-2 minima at a scan boundary: {boundary_minima}.",
        "",
        "Interpretation",
        "--------------",
        "All force columns are Li-force magnitudes in eV/A.  F_bond is parallel",
        "to the local C--C bond; F_hop is perpendicular to it in the sheet;",
        "F_normal is perpendicular to the sheet.  Absolute components are used",
        "because bond-direction signs depend on arbitrary carbon-index ordering.",
        "E_UMA is the total UMA energy.  dE is relative to the mean held-out-test",
        "saddle energy.  The held-out-test-saddle row is the primary reference.",
        "",
        "Mean metrics",
        "------------",
        format_table(rows, "mean"),
        "",
        "Median force metrics",
        "--------------------",
        format_table(rows, "median"),
        "",
        "Decision rule",
        "-------------",
        "The desired UMA signature is an elevation of |F_Li|, especially |F_bond|,",
        "for both false-bridge controls versus held-out true saddles.  If those",
        "populations overlap strongly, pretrained UMA is not a reliable force-based",
        "post-filter for this false-saddle mechanism.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    train_frames = read_and_validate(args.train_traj)
    test_frames = read_and_validate(args.test_traj)
    template, max_carbon_difference = verify_common_sheet(train_frames + test_frames)
    cell = np.asarray(template.get_cell())

    train_saddles = extract_saddles(train_frames)
    test_saddles = extract_saddles(test_frames)
    all_saddles = np.vstack([train_saddles, test_saddles])
    bridges = build_cc_bridges(template, args.bond_cutoff)
    false_indices, _ = identify_false_bridges(
        bridges, all_saddles, cell, args.false_distance_cutoff
    )
    if len(false_indices) != args.expected_false_count:
        raise ValueError(
            f"Found {len(false_indices)} false bridges; expected {args.expected_false_count}. "
            "Do not silently compare a changed dataset."
        )

    all_bridge_indices, true_midpoint_distances = assign_saddles_to_bridges(
        all_saddles, bridges, cell, args.true_match_cutoff
    )
    train_bridge_indices = all_bridge_indices[:len(train_saddles)]
    test_bridge_indices = all_bridge_indices[len(train_saddles):]
    common_z = float(np.median(all_saddles[:, 2]))
    false_common_positions = np.asarray([
        [bridges[index].midpoint[0], bridges[index].midpoint[1], common_z]
        for index in false_indices
    ])

    print(
        f"[setup] {len(bridges)} periodic C--C bridges; {len(false_indices)} false; "
        f"{len(train_saddles)} train + {len(test_saddles)} test true saddles",
        flush=True,
    )
    calculator = make_uma_calculator(args.backbone, args.device, args.task_name)
    test_results = evaluate_population(
        calculator, template, test_saddles, test_bridge_indices, bridges, "held-out true saddles"
    )
    train_results = evaluate_population(
        calculator, template, train_saddles, train_bridge_indices, bridges, "training true saddles"
    )
    false_common_results = evaluate_population(
        calculator, template, false_common_positions, false_indices, bridges,
        "false bridges, control 1",
    )
    false_relaxed_results, boundary_minima = height_relax_false_bridges(
        calculator, template, bridges, false_indices, common_z,
        args.z_search_half_width, args.z_coarse_step, args.z_fine_step,
    )

    test_reference_energy = float(np.mean([result.energy for result in test_results]))
    rows = [
        ("held-out true saddles (primary)", summarize(test_results, test_reference_energy)),
        ("training true saddles (audit)", summarize(train_results, test_reference_energy)),
        ("false bridges: control 1 common z", summarize(false_common_results, test_reference_energy)),
        ("false bridges: control 2 relaxed z", summarize(false_relaxed_results, test_reference_energy)),
    ]
    report = make_report(
        args, len(train_saddles), len(test_saddles), len(bridges), len(false_indices),
        common_z, max_carbon_difference, float(true_midpoint_distances.max()),
        boundary_minima, rows,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report)
    # Keep the on-disk report as the durable artifact, but also emit its full
    # contents so an sbatch .out file contains the exact same scientific record.
    print("\n" + report, flush=True)
    print(f"[done] wrote human-readable UMA comparison to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
