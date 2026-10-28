"""Where the never-reached saddles are, and where the flow goes instead.

    R=/scratch/gqs7zr/saddleflow/lic_source_study_seed42/coverage_curve
    python examples/LiC/experiments/connectivity/plot_missed_saddles.py \
        $R/sigma0.15/source_study.npz $R/sigma0.30/source_study.npz --out $R/missed_saddles.pdf

Page 1 is the plot_throws.py reference view (same sheet, rings, crosses, size) with:
    purple disc        a saddle never reached from one of its stored minima, at every noise level
    light purple disc  a saddle missed at only some noise levels (--no-partial to leave these out)
    purple diamond     the minimum it is never reached from, joined to the saddle by a purple line
Page 2 zooms on each of those minima and draws the flow trajectories that started there (the archive
given by --trajectories, default the last), coloured by the saddle they ended on; grey ended elsewhere.
CPU only.
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

LIC = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(LIC), str(LIC / "experiments" / "source_study")]
from plot_throws import HIT, _legend, _sheet, sheet_geometry  # noqa: E402
from coverage_curve import coverage, load  # noqa: E402

PURPLE, LIGHT = "darkviolet", "plum"
HALF = 3.0   # A: half-width of a zoom window


def image_near(points, centre, cell):
    """Each in-plane point moved to its periodic image nearest `centre`."""
    A = cell[:2, :2]
    f = (points[..., :2] - centre[:2]) @ np.linalg.inv(A)
    return points[..., :2] - np.round(f) @ A


def tiled_sheet(a, geo, tr, te):
    """The plot_throws sheet with its 3x3 periodic images, so a zoom near the cell edge is complete."""
    A = geo["cell"][:2, :2]
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            off = np.r_[i * A[0] + j * A[1], 0.0]
            _sheet(a, dict(geo, C=geo["C"] + off, ns=geo["ns"] + off), tr + off, te + off)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("archives", nargs="+", help="source_study.npz files from the coverage runs")
    p.add_argument("--out", required=True)
    p.add_argument("--trajectories", type=int, default=-1, help="Archive (index) whose trajectories page 2 draws")
    p.add_argument("--no-partial", action="store_true", help="Leave out saddles missed at only some noise levels")
    p.add_argument("--train-traj", default=str(LIC / "train_set.traj"))
    p.add_argument("--test-traj", default=str(LIC / "test_set.traj"))
    args = p.parse_args()
    warnings.filterwarnings("ignore", message=".*tight_layout.*")

    runs = [load(path, "n12_uncond") for path in args.archives]
    missed = [set(coverage(run, run["max_draws"])["missed"]) for run in runs]
    always, partial = sorted(set.intersection(*missed)), sorted(set.union(*missed) - set.intersection(*missed))
    if args.no_partial:
        partial = []
    data = np.load(args.archives[args.trajectories], allow_pickle=True)
    tr, te, minima, cell = data["train_saddles"], data["test_saddles"], data["minima"], data["cell"]
    saddles = np.vstack([tr, te])
    geo = sheet_geometry(args.train_traj, args.test_traj)
    sigmas = ", ".join(f"{run['sigma']:g}" for run in runs)

    with PdfPages(args.out) as pdf:
        # Page 1: the reference view plus the missed pairs.
        fig, ax = plt.subplots(1, 2, figsize=(12.4, 6.6), gridspec_kw={"width_ratios": [1.35, 1]})
        a = ax[0]
        _sheet(a, geo, tr, te)
        for pairs, colour in ((partial, LIGHT), (always, PURPLE)):
            for m, s in pairs:
                x = image_near(saddles[s], minima[m], cell)
                a.plot([minima[m, 0], x[0]], [minima[m, 1], x[1]], "-", c=colour, lw=1.2, zorder=7)
                a.scatter(*minima[m, :2], s=28, marker="D", c=colour, edgecolors="k", lw=.4, zorder=8)
            for s in sorted({s for _, s in pairs}):
                a.scatter(*saddles[s, :2], s=60, c=colour, edgecolors="k", lw=.5, zorder=9)
        a.set_title(f"never-reached connected saddles (n12_uncond, {runs[0]['max_draws']} draws per minimum, "
                    f"sigma = {sigmas} A)", fontsize=9)
        _legend(a, geo, tr, te, with_traj=False)
        extra = [Line2D([], [], color=PURPLE, marker="o", ls="", ms=7, mec="k",
                        label=f"never reached at any sigma ({len({s for _, s in always})})"),
                 Line2D([], [], color=PURPLE, marker="D", ls="-", ms=5, mec="k", label="minimum it is missed from")]
        if partial:
            extra.insert(1, Line2D([], [], color=LIGHT, marker="o", ls="", ms=7, mec="k",
                                   label=f"missed at some sigma only ({len({s for _, s in partial})})"))
        a.add_artist(a.get_legend())
        a.legend(handles=extra, loc="upper center", bbox_to_anchor=(0.5, -0.075), ncol=3, fontsize=8, frameon=False)
        b = ax[1]
        b.axis("off")
        rows = [f"{'(min, saddle)':14s} {'missed at':12s} {'min -> saddle':>13s}"]
        for (m, s), kind in [(x, "every sigma") for x in always] + [(x, "some sigma") for x in partial]:
            d = np.linalg.norm(image_near(saddles[s], minima[m], cell) - minima[m, :2])
            rows.append(f"{str((m, s)):14s} {kind:12s} {d:11.3f} A")
        b.text(0, 1, "\n".join(rows), family="monospace", fontsize=8, va="top", transform=b.transAxes)
        b.set_title("missed (minimum, saddle) pairs; in-plane distance", fontsize=8)
        fig.tight_layout(pad=1.9)
        pdf.savefig(fig); plt.close(fig)

        # Page 2: flow trajectories from each minimum with a never-reached saddle.
        use = (data["method"] == "n12_uncond") & (data["source_population"][data["source_index"]] == "connected")
        start_min = data["source_minimum_index"][data["source_index"][use]]
        paths, found, outcome = data["path"][use], data["nearest_true_index"][use], data["outcome"][use]
        edges = data["catalog_edges"]
        targets = sorted({m for m, _ in always})
        ncol = 3
        fig, ax = plt.subplots(-(-len(targets) // ncol), ncol, figsize=(12.4, 4.4 * -(-len(targets) // ncol)),
                               squeeze=False)
        for a, m in zip(ax.flat, targets):
            tiled_sheet(a, geo, tr, te)
            connected = np.flatnonzero(np.any(edges == m, axis=1))
            missed_here = {s for mm, s in always if mm == m}
            mine = start_min == m
            colours = {s: f"C{k}" for k, s in enumerate(s for s in connected if s not in missed_here)}
            for s, c in colours.items():
                x = image_near(saddles[s], minima[m], cell)
                a.add_patch(plt.Circle(x, HIT, fill=False, lw=1.4, edgecolor=c, zorder=6))
            for s in missed_here:
                x = image_near(saddles[s], minima[m], cell)
                a.scatter(*x, s=90, c=PURPLE, edgecolors="k", lw=.5, zorder=9)
            lines, line_colours, counts = [], [], {}
            for path, s, o in zip(paths[mine], found[mine], outcome[mine]):
                hit = o == 0 and s in colours
                counts[s if hit else -1] = counts.get(s if hit else -1, 0) + 1
                lines.append(image_near(path, minima[m], cell))
                line_colours.append(colours[s] if hit else "0.5")
            a.add_collection(LineCollection(lines, colors=line_colours, linewidths=.5, alpha=.5, zorder=3))
            ends = np.array([l[-1] for l in lines])
            a.scatter(ends[:, 0], ends[:, 1], s=4, c=line_colours, zorder=5)
            a.scatter(*minima[m, :2], s=40, marker="D", c=PURPLE, edgecolors="k", lw=.4, zorder=8)
            a.set_xlim(minima[m, 0] - HALF, minima[m, 0] + HALF)
            a.set_ylim(minima[m, 1] - HALF, minima[m, 1] + HALF)
            summary = ", ".join(f"{s}: {counts.get(s, 0)}" for s in colours) + f", other: {counts.get(-1, 0)}"
            a.set_title(f"minimum {m}: {mine.sum()} draws; never reached {sorted(missed_here)} (purple)\n"
                        f"ended on saddle  {summary}", fontsize=8)
        for a in ax.flat[len(targets):]:
            a.axis("off")
        fig.suptitle(f"flow trajectories from the minima with a never-reached saddle (sigma = {runs[args.trajectories]['sigma']:g} A);"
                     f" coloured ring = a connected saddle the draws reached, grey = ended elsewhere", fontsize=9)
        fig.tight_layout(pad=1.5)
        pdf.savefig(fig); plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
