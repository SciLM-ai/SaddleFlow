"""Render an existing LiC robustness-shell NPZ archive as a PDF.

This script is deliberately separate from ``robustness_shells.py``.  It never
loads SaddleFlow or UMA, never samples trajectories, and needs no GPU.  That
means visual choices can be changed and re-rendered from a saved archive
without repeating the expensive robustness experiment.

The PDF is ordered by increasing cumulative shell radius.  For a fixed flow
time, its first page shows only the r=0 control; each following page retains
all previous shells and adds the next radius.  Trajectory lines are colored by
their shell radius.  Starts are intentionally *not* plotted as dots; endpoint
crosses are colored by their landing classification.

Example:

    python examples/LiC/experiments/diagnostics/plot_robustness_shells.py \
        /scratch/gqs7zr/saddleflow/model_ckpts/lic_tsdenoise_sigma0.25_b200/robustness_shells/robustness_shells.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

# This script imports LiC modules from examples/LiC.
_LIC = Path(__file__).resolve().parents[2]  # examples/LiC
sys.path[:0] = [str(_LIC)]

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

try:  # Direct execution from examples/LiC.
    from plot_throws import MARGIN, sheet_geometry  # noqa: E402
except ModuleNotFoundError:  # Import/module execution from repository root.
    from examples.LiC.plot_throws import MARGIN, sheet_geometry  # noqa: E402


TRUE_HIT_RADIUS = 0.10
OUTCOME_NAMES = np.asarray(
    ["parent_saddle", "other_true_saddle", "known_false_bridge", "other_failure"]
)
OUTCOME_COLORS = ["tab:green", "darkorange", "tab:red", "tab:purple"]
OUTCOME_LABELS = ["parent saddle", "other true saddle", "known false bridge", "other failure"]


def parse_args() -> argparse.Namespace:
    """Choose an existing archive and purely visual rendering options."""
    here = Path(__file__).resolve().parents[2]  # examples/LiC
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archive", help="robustness_shells.npz written by robustness_shells.py")
    parser.add_argument("--out", default=None, help="PDF path; defaults to robustness_shells.pdf beside archive")
    parser.add_argument("--train-traj", default=str(here / "train_set.traj"))
    parser.add_argument("--test-traj", default=str(here / "test_set.traj"))
    parser.add_argument(
        "--path-limit", type=int, default=500,
        help="Maximum trajectory lines drawn on each cumulative page; endpoint crosses are never subsampled.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Deterministic path subsampling seed")
    args = parser.parse_args()
    if args.path_limit <= 0:
        raise SystemExit("--path-limit must be positive")
    return args


def scalar_text(value) -> str:
    """Read a NumPy scalar that may be stored as bytes or Unicode."""
    value = value.item() if isinstance(value, np.ndarray) and value.ndim == 0 else value
    return value.decode() if isinstance(value, bytes) else str(value)


def load_archive(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Load and validate the numerical fields needed for rendering only."""
    if not path.is_file():
        raise FileNotFoundError(f"Robustness archive does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = [
            "starts", "finals", "paths", "parent_index", "radius", "start_step",
            "outcome_code", "force_rms", "train_saddles", "test_saddles", "cell",
            "configuration_json",
        ]
        missing = [key for key in required if key not in archive]
        if missing:
            raise ValueError(f"{path} is not a compatible robustness archive; missing {missing}")
        data = {key: archive[key] for key in archive.files}
    config = json.loads(scalar_text(data["configuration_json"]))
    return data, config


def median_or_nan(values: np.ndarray) -> float:
    """Return a median while preserving the archive's no-UMA NaN convention."""
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else float("nan")


def make_summary_rows(data: dict[str, np.ndarray], K: int) -> list[dict]:
    """Recreate the compact table statistics directly from a saved archive."""
    rows: list[dict] = []
    for step in sorted(np.unique(data["start_step"])):
        rows_at_time = data["start_step"] == step
        for radius in sorted(np.unique(data["radius"])):
            use = rows_at_time & np.isclose(data["radius"], radius)
            outcomes = data["outcome_code"][use]
            rows.append({
                "step": int(step),
                "radius": float(radius),
                "n": int(use.sum()),
                "parent_pct": float(100 * np.mean(outcomes == 0)),
                "other_true_pct": float(100 * np.mean(outcomes == 1)),
                "false_pct": float(100 * np.mean(outcomes == 2)),
                "other_pct": float(100 * np.mean(outcomes == 3)),
                "median_d_parent": float(np.median(data["distance_to_parent"][use])),
                "median_d_any_true": float(np.median(data["distance_to_any_true"][use])),
                "median_force_rms": median_or_nan(data["force_rms"][use]),
            })
    return rows


def draw_sheet(axis, geometry: dict, train_saddles: np.ndarray, test_saddles: np.ndarray) -> None:
    """Draw the fixed carbon network and all known saddle/false-site labels."""
    carbon = geometry["C"]
    for first, second in geometry["bonds"]:
        axis.plot(
            [carbon[first, 0], carbon[second, 0]], [carbon[first, 1], carbon[second, 1]],
            color="0.58", lw=0.75, zorder=0,
        )
    axis.scatter(carbon[:, 0], carbon[:, 1], s=5, color="0.35", zorder=1)
    for point in train_saddles:
        axis.add_patch(plt.Circle(tuple(point[:2]), TRUE_HIT_RADIUS, fill=False,
                                  edgecolor="limegreen", lw=0.45, zorder=2))
    for point in test_saddles:
        axis.add_patch(plt.Circle(tuple(point[:2]), TRUE_HIT_RADIUS, fill=False,
                                  edgecolor="darkorange", lw=0.35, alpha=0.65, zorder=2))
    false_sites = geometry["ns"]
    axis.scatter(false_sites[:, 0], false_sites[:, 1], marker="x", c="red", s=16,
                 lw=0.9, zorder=6)
    cell = geometry["cell"]
    axis.set_xlim(-MARGIN, cell[0, 0] + MARGIN)
    axis.set_ylim(-MARGIN, cell[1, 1] + MARGIN)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])


def split_path_at_cell_crossings(path: np.ndarray, cell_width: float) -> list[np.ndarray]:
    """Split a wrapped trajectory rather than drawing a false line across the cell."""
    path = path[np.isfinite(path).all(axis=1)]
    if len(path) < 2:
        return []
    cuts = np.flatnonzero(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1) > cell_width / 2)
    if not len(cuts):
        return [path]
    pieces: list[np.ndarray] = []
    begin = 0
    for cut in cuts:
        if cut + 1 - begin >= 2:
            pieces.append(path[begin:cut + 1])
        begin = cut + 1
    if len(path) - begin >= 2:
        pieces.append(path[begin:])
    return pieces


def draw_cumulative_page(
    pdf: PdfPages,
    *,
    data: dict[str, np.ndarray],
    geometry: dict,
    K: int,
    step: int,
    radius: float,
    radii: np.ndarray,
    radius_norm,
    radius_map,
    path_limit: int,
    rng: np.random.Generator,
) -> None:
    """Draw one whole-sheet page retaining all shells through `radius`.

    No starting-point markers are drawn.  Radius appears only through each
    trajectory's line color; endpoint crosses show the categorical outcome.
    """
    rows_at_time = np.flatnonzero(data["start_step"] == step)
    rows = rows_at_time[data["radius"][rows_at_time] <= radius + 1e-8]
    newly_added = rows_at_time[np.isclose(data["radius"][rows_at_time], radius)]
    figure, axes = plt.subplots(
        1, 2, figsize=(13.0, 6.3), gridspec_kw={"width_ratios": [1.35, 1]}
    )
    axis = axes[0]
    draw_sheet(axis, geometry, data["train_saddles"], data["test_saddles"])
    axis.scatter(data["train_saddles"][:, 0], data["train_saddles"][:, 1], marker="*", s=65,
                 c="black", edgecolor="white", lw=0.45, zorder=9)

    drawn = rows if len(rows) <= path_limit else rng.choice(rows, size=path_limit, replace=False)
    segments: list[np.ndarray] = []
    segment_radius: list[float] = []
    for index in drawn:
        for segment in split_path_at_cell_crossings(data["paths"][index], geometry["cell"][0, 0]):
            segments.append(segment[:, :2])
            segment_radius.append(float(data["radius"][index]))
    if segments:
        collection = LineCollection(
            segments, cmap=radius_map, norm=radius_norm, linewidths=0.62,
            alpha=0.34, zorder=4,
        )
        collection.set_array(np.asarray(segment_radius))
        axis.add_collection(collection)

    for outcome, color in enumerate(OUTCOME_COLORS):
        group = rows[data["outcome_code"][rows] == outcome]
        if len(group):
            axis.scatter(data["finals"][group, 0], data["finals"][group, 1], s=17,
                         marker="x", c=color, lw=1.0, alpha=0.9, zorder=8)
    scalar = plt.cm.ScalarMappable(norm=radius_norm, cmap=radius_map)
    scalar.set_array(radii)
    colorbar = figure.colorbar(scalar, ax=axis, fraction=0.046, pad=0.02)
    colorbar.set_label("source-shell radius (Å): trajectory line color")
    axis.set_title(
        f"t={step / K:.2f}: cumulative source shells r <= {radius:.3f} Å\n"
        f"{len(rows)} rollouts retained; {len(drawn)} trajectory lines drawn; starts omitted"
    )
    legend = [
        Line2D([], [], color="black", marker="*", ls="", ms=8, label="training saddle"),
        Line2D([], [], color="0.45", lw=1, label="C--C bond"),
    ] + [
        Line2D([], [], color=color, marker="x", ls="", ms=6, label=label)
        for color, label in zip(OUTCOME_COLORS, OUTCOME_LABELS)
    ]
    axis.legend(handles=legend, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.02),
                ncol=2, frameon=False)

    side = axes[1]
    for bar_index, (label, group) in enumerate([
        (f"new r={radius:.3f}", newly_added),
        (f"cumulative r<={radius:.3f}", rows),
    ]):
        left = 0.0
        for outcome, color in enumerate(OUTCOME_COLORS):
            percent = 100 * np.mean(data["outcome_code"][group] == outcome)
            side.barh(bar_index, percent, left=left, color=color, height=0.62)
            left += percent
    side.set_yticks([0, 1], [f"new shell\nr={radius:.3f}", f"cumulative\nr<={radius:.3f}"])
    side.set_xlim(0, 100)
    side.set_xlabel("endpoint outcome probability (%)")
    side.grid(axis="x", alpha=0.25)
    side.set_title("New shell versus cumulative population")
    if np.isfinite(data["force_rms"][rows]).any():
        side.text(
            0.02, -0.17,
            f"cumulative median UMA |F_Li|: {np.nanmedian(data['force_rms'][rows]):.4f} eV/Å",
            transform=side.transAxes, fontsize=8,
        )
    figure.tight_layout()
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def draw_summary_page(pdf: PdfPages, summary_rows: list[dict], K: int, times: list[int]) -> None:
    """Append the probability and endpoint-statistic overview after map pages."""
    figure, axes = plt.subplots(len(times), 2, figsize=(11.5, 3.7 * len(times)), squeeze=False)
    for row_index, step in enumerate(times):
        selected = sorted((row for row in summary_rows if row["step"] == step), key=lambda row: row["radius"])
        radii = [row["radius"] for row in selected]
        left = axes[row_index, 0]
        for key, label, color in [
            ("parent_pct", "parent saddle", OUTCOME_COLORS[0]),
            ("other_true_pct", "other true", OUTCOME_COLORS[1]),
            ("false_pct", "known false bridge", OUTCOME_COLORS[2]),
            ("other_pct", "other failure", OUTCOME_COLORS[3]),
        ]:
            left.plot(radii, [row[key] for row in selected], "o-", color=color, label=label)
        left.set_ylim(-3, 103)
        left.grid(alpha=0.25)
        left.set_xlabel("fixed Li shell radius (Å)")
        left.set_ylabel("endpoint probability (%)")
        left.set_title(f"t={step / K:.2f} (step {step})")
        left.legend(fontsize=7, ncol=2)

        right = axes[row_index, 1]
        right.plot(radii, [row["median_d_parent"] for row in selected], "o-", color="tab:blue",
                   label="median distance to parent")
        right.plot(radii, [row["median_d_any_true"] for row in selected], "o--", color="0.2",
                   label="median distance to any true")
        if np.isfinite([row["median_force_rms"] for row in selected]).any():
            force_axis = right.twinx()
            force_axis.plot(radii, [row["median_force_rms"] for row in selected], "s-",
                            color="tab:red", label="median UMA |F_Li|")
            force_axis.set_ylabel("UMA Li force RMS (eV/Å)", color="tab:red")
            force_axis.tick_params(axis="y", colors="tab:red")
        right.grid(alpha=0.25)
        right.set_xlabel("fixed Li shell radius (Å)")
        right.set_ylabel("endpoint distance (Å)")
        right.set_title("Endpoint diagnostics")
        right.legend(fontsize=7, loc="upper left")
    figure.suptitle("LiC: robustness-shell numerical summary", y=1.01)
    figure.tight_layout()
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    archive_path = Path(args.archive)
    output_path = Path(args.out) if args.out else archive_path.with_name("robustness_shells.pdf")
    data, config = load_archive(archive_path)
    K = int(config.get("K", data["paths"].shape[1] - 1))
    if data["paths"].shape[1] != K + 1:
        raise ValueError("Archive path length and stored K disagree")

    geometry = sheet_geometry(args.train_traj, args.test_traj)
    radii = np.asarray(sorted(np.unique(data["radius"])), dtype=float)
    times = sorted(np.unique(data["start_step"]))
    radius_norm = plt.Normalize(vmin=float(radii.min()), vmax=float(max(radii.max(), 1e-8)))
    radius_map = plt.get_cmap("viridis")
    summary_rows = make_summary_rows(data, K)
    rng = np.random.default_rng(args.seed)

    print(
        f"[plot] archive: {archive_path}; {len(data['finals'])} rollouts; "
        f"{len(times)} time(s); {len(radii)} radius shells; K={K}",
        flush=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        for step in times:
            for radius in radii:
                print(f"[plot] t={step / K:.2f}, cumulative r<={radius:.3f} Å", flush=True)
                draw_cumulative_page(
                    pdf, data=data, geometry=geometry, K=K, step=int(step), radius=float(radius),
                    radii=radii, radius_norm=radius_norm, radius_map=radius_map,
                    path_limit=args.path_limit, rng=rng,
                )
        draw_summary_page(pdf, summary_rows, K, times)
    print(f"[done] wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
