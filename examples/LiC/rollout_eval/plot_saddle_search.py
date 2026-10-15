"""Render saved solver trajectories. No UMA, checkpoint, Sella or GPU required."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.geometry import find_mic, wrap_positions
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


OUTCOME_COLORS = np.asarray(["tab:green", "tab:red", "tab:purple"])


def draw_sheet(axis, data):
    positions, cell = data["positions"], data["cell"]
    for i, j in data["bonds"]:
        bond, _ = find_mic(positions[j] - positions[i], cell, data["pbc"])
        # Periodic edge bonds are drawn from either end, clipped by the cell.
        for start, end in ((positions[i], positions[i] + bond),
                           (positions[j] - bond, positions[j])):
            axis.plot([start[0], end[0]], [start[1], end[1]], color="0.7", lw=0.6, zorder=0)
    carbon = positions[data["carbon_indices"]]
    axis.scatter(carbon[:, 0], carbon[:, 1], s=4, c="0.5", zorder=1)
    for name, color in (("train_saddles", "limegreen"), ("test_saddles", "darkorange")):
        saddle = wrap_positions(data[name], cell, data["pbc"])
        axis.scatter(saddle[:, 0], saddle[:, 1], c="black", marker="*", s=13, zorder=4)
        axis.scatter(saddle[:, 0], saddle[:, 1], facecolors="none", edgecolors=color,
                     s=31, linewidths=0.6, zorder=3)
    false = wrap_positions(data["false_bridges"], cell, data["pbc"])
    axis.scatter(false[:, 0], false[:, 1], marker="x", c="red", s=25, linewidths=1, zorder=5)
    corners = np.asarray([np.zeros(3), cell[0], cell[1], cell[0] + cell[1]])
    axis.set_xlim(corners[:, 0].min() - 0.2, corners[:, 0].max() + 0.2)
    axis.set_ylim(corners[:, 1].min() - 0.2, corners[:, 1].max() + 0.2)
    axis.set_aspect("equal")
    axis.set_xlabel("x (A)")
    axis.set_ylabel("y (A)")


def path_segments(path, cell, pbc):
    path = path[np.isfinite(path).all(axis=1)]
    if not len(path):
        return []
    path = wrap_positions(path, cell, pbc)
    cuts = np.flatnonzero(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
                         > 0.5 * min(np.linalg.norm(cell[0]), np.linalg.norm(cell[1]))) + 1
    return np.split(path, cuts)


def draw_endpoint_paths(axis, paths, outcomes, data, indices=None, path_limit=100):
    """Shared geometry-only trajectory panel; never draw a layer of start dots."""
    draw_sheet(axis,data)
    indices=np.arange(len(paths)) if indices is None else np.asarray(indices)
    chosen=indices if len(indices)<=path_limit else np.random.default_rng(0).choice(indices,path_limit,replace=False)
    for i in chosen:
        for segment in path_segments(paths[i],data["cell"],data["pbc"]):
            axis.plot(segment[:,0],segment[:,1],color="lightskyblue",lw=.8,alpha=.6,zorder=2)
    final=wrap_positions(paths[indices,-1],data["cell"],data["pbc"])
    axis.scatter(final[:,0],final[:,1],c=OUTCOME_COLORS[np.asarray(outcomes)[indices]],s=17,zorder=6)


def draw_paths(axis, data, trial_indices, show_starts=False, path_limit=300):
    # Limit only the drawn curves, NEVER the statistics or endpoint markers.
    if len(trial_indices) > path_limit:
        trial_indices_for_lines = np.random.default_rng(0).choice(trial_indices, path_limit, replace=False)
    else:
        trial_indices_for_lines = trial_indices
    for i in trial_indices_for_lines:
        for segment in path_segments(data["path"][i], data["cell"], data["pbc"]):
            axis.plot(segment[:, 0], segment[:, 1], color="lightskyblue", lw=0.8, alpha=0.65, zorder=2)
    final = wrap_positions(data["final"][trial_indices], data["cell"], data["pbc"])
    outcome = data["outcome"][trial_indices].astype(int)
    axis.scatter(final[:, 0], final[:, 1], c=OUTCOME_COLORS[outcome], s=17, alpha=0.9, zorder=6)
    physical = data["physical_success"][trial_indices].astype(bool)
    axis.scatter(final[physical, 0], final[physical, 1], facecolors="none", edgecolors="blue",
                 s=37, linewidths=0.8, zorder=7)
    if show_starts:
        source = data["starts"][data["source_index"][trial_indices]]
        source = wrap_positions(source, data["cell"], data["pbc"])
        axis.scatter(source[:, 0], source[:, 1], c="red", s=7, zorder=2)


def trajectory_page(pdf, data, methods, selection, title, show_starts, path_limit):
    columns = min(2, len(methods))
    rows = int(np.ceil(len(methods) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(7 * columns, 7 * rows), squeeze=False)
    for axis, method in zip(axes.flat, methods):
        use = np.flatnonzero((data["method"] == method) & selection)
        draw_sheet(axis, data)
        if len(use):
            draw_paths(axis, data, use, show_starts, path_limit)
            out = data["outcome"][use]
            text = (f"N={len(use)}; true {100*np.mean(out==0):.1f}%; false {100*np.mean(out==1):.1f}%\n"
                    f"physical {100*np.mean(data['physical_success'][use]):.1f}%; "
                    f"median fmax {np.median(data['force_max'][use]):.4f} eV/A")
        else:
            text = "No completed trials in this population"
        axis.set_title(method + "\n" + text, fontsize=11)
    for axis in list(axes.flat)[len(methods):]:
        axis.set_visible(False)
    legend = [
        Line2D([], [], color="lightskyblue", label="trajectory"),
        Line2D([], [], color="black", marker="*", linestyle="none", label="stored true saddle"),
        Line2D([], [], color="red", marker="x", linestyle="none", label="known false bridge"),
    ]
    # Keep explicit construction: endpoint color and physical certification
    # carry different information, so neither is hidden behind a color scale.
    legend += [Line2D([], [], color=c, marker="o", linestyle="none", label=l)
                          for c, l in zip(OUTCOME_COLORS, ["true endpoint", "false-region endpoint", "other endpoint"])]
    legend.append(Line2D([], [], marker="o", color="blue", markerfacecolor="none",
                         linestyle="none", label="force + index1 certified"))
    figure.suptitle(title + "\nGreen/orange reference rings: train/held-out labels", fontsize=13)
    figure.legend(handles=legend, loc="lower center", ncol=4, fontsize=9)
    figure.tight_layout(rect=(0, 0.07, 1, 0.94))
    pdf.savefig(figure)
    plt.close(figure)


def summary_page(pdf, data, methods):
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    step_axis = "step_budgets" in data
    budgets = data["step_budgets"] if step_axis else data["force_budgets"]
    available = data.get("budget_available", np.ones((len(data["method"]), len(budgets)), dtype=bool))
    for method in methods:
        use = data["method"] == method
        if not available[use].any():
            continue  # Historical Sella/FM finals have no MMF-iteration snapshots.
        true, physical = [], []
        for j in range(len(budgets)):
            valid = use & available[:, j]
            true.append(100*np.mean(data["budget_outcome"][valid, j] == 0) if valid.any() else np.nan)
            physical.append(100*np.mean(data["budget_physical_success"][valid, j]) if valid.any() else np.nan)
        axes[0, 0].plot(budgets, true, "o-", label=method)
        axes[0, 1].plot(budgets, physical, "o-", label=method)
    axes[0, 0].set_title("Catalog true hit versus " + ("translation steps" if step_axis else "search-PES budget"))
    axes[0, 1].set_title("Force + index-one certificate versus budget")
    for axis in axes[0]:
        axis.set_ylim(-2, 102)
        axis.set_xlabel("Translation steps (same Kmax clock)" if step_axis else "PES evaluations (verification excluded)")
        axis.set_ylabel("Independent trials (%)")
        axis.legend()
        axis.grid(alpha=0.2)
    for method in methods:
        use = data["method"] == method
        force = data["force_max"][use]
        force = force[np.isfinite(force)]
        force.sort()
        if len(force):
            axes[1, 0].plot(np.maximum(force, 1e-8), np.arange(1, len(force)+1)/len(force), label=method)
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_title("Final free-atom fmax distribution")
    axes[1, 0].set_xlabel("eV/A")
    axes[1, 0].set_ylabel("Cumulative fraction")
    axes[1, 0].legend()
    axes[1, 1].axis("off")
    config = json.loads(str(data["config_json"]))
    text = ["Median cost per independent trial", "", "method          PES queries   seconds"]
    for method in methods:
        use = data["method"] == method
        text.append(f"{method:14s} {np.median(data['force_calls'][use]):8.1f}   {np.median(data['wall_seconds'][use]):8.2f}")
    text += ["", f"MMF maximum horizon: {config['max_steps']}" if step_axis else f"Two-flow restarts/source: {config['restarts']}",
             f"SaddleFlow: {config['K']} learned steps; batched amortized time.",
             "Not a comparison at equal total wall time.",
             "FD and rotation probes count as PES queries.",
             "Final/budget force + curvature checks are extra.",
             "All restarts retained, no label-based selection.",
             "Low force alone does not certify a saddle."]
    axes[1, 1].text(0, 0.98, "\n".join(text), va="top", family="monospace", fontsize=9)
    figure.suptitle("LiC physics saddle-search outcomes and cost")
    figure.tight_layout()
    pdf.savefig(figure)
    plt.close(figure)


def mobility_page(pdf, data, methods):
    """Plot observed a/b histories; stopped trajectories are not invented anew."""
    if "trace_mobility_a" not in data:
        return
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for method in methods:
        use = data["method"] == method
        if not np.isfinite(data["trace_mobility_a"][use]).any():
            continue
        steps = data["trace_step"][use]
        for axis, key in zip(axes, ("trace_mobility_a", "trace_mobility_b")):
            values = data[key][use]
            indices = np.unique(steps[np.isfinite(values) & np.isfinite(steps)])
            medians = [np.median(values[(steps == k) & np.isfinite(values)]) for k in indices]
            axis.plot(indices, medians, label=method)
    for axis, name in zip(axes, ("a: transverse relaxation", "b: minimum-mode ascent")):
        axis.set_title(name)
        axis.set_xlabel("Translation step")
        axis.set_ylabel("Median mobility multiplier (active trajectories)")
        axis.grid(alpha=0.2)
        axis.legend()
    figure.suptitle("Learned translation policy; late-step medians use only trajectories still active")
    figure.tight_layout()
    pdf.savefig(figure)
    plt.close(figure)


def render_pdf(archive_path, output_path=None, *, show_starts=False, path_limit=300):
    archive_path = Path(archive_path)
    output_path = Path(output_path) if output_path else archive_path.with_suffix(".pdf")
    with np.load(archive_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    methods = list(dict.fromkeys(data["method"].tolist()))
    if not methods:
        raise ValueError("Archive has no completed trials to plot.")
    print(f"[plot] writing {output_path}", flush=True)
    with PdfPages(output_path) as pdf:
        summary_page(pdf, data, methods)
        mobility_page(pdf, data, methods)
        trajectory_page(pdf, data, methods, np.ones(len(data["method"]), dtype=bool),
                        "Identical sources: all start populations", show_starts, path_limit)
        population = data["source_population"][data["source_index"]]
        radii = data["source_radius"][data["source_index"]]
        for group in np.unique(population):
            if group == "shells":
                for radius in np.unique(radii[population == group]):
                    use = (population == group) & (radii <= radius + 1e-8)
                    trajectory_page(pdf, data, methods, use, f"Cumulative training-parent shells: r <= {radius:g} A",
                                    show_starts, path_limit)
            else:
                trajectory_page(pdf, data, methods, population == group,
                                f"Start population: {group}", show_starts, path_limit)
    print(f"[plot] saved {output_path}", flush=True)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz")
    parser.add_argument("--out")
    parser.add_argument("--show-starts", action="store_true", help="Off by default to keep the carbon network visible.")
    parser.add_argument("--path-limit", type=int, default=300)
    arguments = parser.parse_args()
    render_pdf(arguments.npz, arguments.out, show_starts=arguments.show_starts, path_limit=arguments.path_limit)
