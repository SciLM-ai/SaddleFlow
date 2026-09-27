"""Draw the trajectories written by random_throw.py over the LiC sheet.

    python examples/LiC/plot_throws.py \\
        examples/LiC/runs/tsdenoise_sigma0.25/random_throw/throw.npz

Writes one multi-page PDF, by default `throws.pdf` next to the first input: a
reference page with the sheet and the saddles only, then one page per throw file
with every trajectory, a histogram of start and end distances, and the metrics.
Pass several throw files to compare runs on the same reference frame.

Left panel
    grey          carbon atoms and C-C bonds; the two vacancies are the gaps
    green ring    training saddle, drawn at the 0.10 A hit radius
    orange ring   test saddle, never seen in training
    red cross     a C-C bond midpoint that carries NO saddle (see README.md)
    blue line     one throw from its start to its endpoint (black dot)

Metrics, all in-plane distances with periodic images
    p50 p90 p99 max   distance from each endpoint to the nearest saddle
    reached_pct       endpoints within 0.10 A of a saddle
    switch_pct        endpoints nearest a different saddle than their start was
    saddleless_pct    endpoints within 0.30 A of a saddle-less midpoint
    other_fail_pct    endpoints that reached neither

Needs only numpy, matplotlib and ASE; no GPU.
"""
import argparse
import itertools
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from ase.io import read

LI = 126         # the Li atom; 0-125 are the frozen carbons
HIT = 0.10       # A: an endpoint within this of a saddle has reached it
NS_R = 0.30      # A: an endpoint within this of a saddle-less midpoint is stuck there.
                 # Regional on purpose: the attractor near such a site misses the
                 # geometric midpoint by up to ~0.3 A.
NS_CUT = 0.5     # A: a bond midpoint with no saddle this close is saddle-less
MARGIN = 0.4     # A: plot margin around the cell


def nearest(pts, ref, A):
    """Index of and in-plane distance to the nearest `ref` point, over the 3x3
    periodic images spanned by the in-plane cell vectors A (2x2)."""
    bi = np.full(len(pts), -1)
    bd = np.full(len(pts), 1e9)
    for i, j in itertools.product((-1, 0, 1), repeat=2):
        off = i * A[0] + j * A[1]
        d = np.linalg.norm(pts[:, None, :2] - (ref[None, :, :2] + off), axis=2)
        k = d.argmin(1)
        v = d.min(1)
        m = v < bd
        bi[m] = k[m]
        bd[m] = v[m]
    return bi, bd


def sheet_geometry(train_traj, test_traj):
    """Carbons, drawable bonds, and the saddle-less C-C bond midpoints."""
    train, test = read(train_traj, index=":"), read(test_traj, index=":")
    ref = train[0]
    pos, Z = ref.get_positions(), ref.get_atomic_numbers()
    cell = np.array(ref.get_cell())
    C = pos[Z == 6]
    dv = C[:, None, :] - C[None, :, :]
    f = dv @ np.linalg.inv(cell)
    f -= np.round(f)
    vec = f @ cell                                   # minimum-image C_i - C_j
    dd = np.linalg.norm(vec, axis=-1)
    pairs = [(i, j) for i in range(len(C)) for j in range(i + 1, len(C))
             if 0.1 < dd[i, j] < 1.8]
    # Draw only the bonds that do not cross the cell edge.
    bonds = [(i, j) for i, j in pairs if np.linalg.norm(C[j] - C[i]) < 1.8]
    mids = np.array([C[i] - vec[i, j] / 2 for i, j in pairs])
    saddles = np.array([fr.get_positions()[LI] for fr in train[1::3] + test[1::3]])
    _, d = nearest(mids, saddles, cell[:2, :2])
    return dict(C=C, cell=cell, bonds=bonds, mids=mids, ns=mids[d > NS_CUT])


def metrics(z, geo):
    P, tr, te = z["paths"], z["train_saddles"], z["test_saddles"]
    A = z["cell"][:2, :2]
    allS = np.vstack([tr, te])
    si, _ = nearest(P[:, 0, :], allS, A)
    ei, ed = nearest(P[:, -1, :], allS, A)
    _, dn = nearest(P[:, -1, :], geo["ns"], A)
    reached = ed < HIT
    return dict(p50=round(float(np.percentile(ed, 50)), 4),
                p90=round(float(np.percentile(ed, 90)), 4),
                p99=round(float(np.percentile(ed, 99)), 4),
                max=round(float(ed.max()), 4),
                reached_pct=round(100 * float(reached.mean()), 1),
                switch_pct=round(100 * float(np.mean(si != ei)), 1),
                saddleless_pct=round(100 * float((dn < NS_R).mean()), 2),
                other_fail_pct=round(100 * float((~reached & (dn >= NS_R)).mean()), 2))


def _sheet(a, geo, tr, te):
    C = geo["C"]
    for i, j in geo["bonds"]:
        a.plot([C[i, 0], C[j, 0]], [C[i, 1], C[j, 1]], "-", c="0.55", lw=.85, zorder=0)
    a.scatter(C[:, 0], C[:, 1], s=6, c="0.35", zorder=1)
    # Rings are drawn at the hit radius itself, so an endpoint inside a ring has
    # reached that saddle by definition.
    for x, y in tr[:, :2]:
        a.add_patch(plt.Circle((x, y), HIT, fill=False, lw=.5, edgecolor="limegreen", zorder=4))
    for x, y in te[:, :2]:
        a.add_patch(plt.Circle((x, y), HIT, fill=False, lw=.5, edgecolor="darkorange", zorder=4))
    ns = geo["ns"]
    a.scatter(ns[:, 0], ns[:, 1], s=14, marker="x", c="red", lw=0.9, zorder=6)
    cell = geo["cell"]
    a.set_xlim(-MARGIN, cell[0, 0] + MARGIN)
    a.set_ylim(-MARGIN, cell[1, 1] + MARGIN)
    a.set_aspect("equal")
    a.set_xticks([])
    a.set_yticks([])


def _legend(a, geo, tr, te, with_traj=True, with_starts=False):
    h = []
    if with_traj:
        if with_starts:
            h += [Line2D([], [], color="C0", marker="o", ls="", ms=3, label="throw start")]
        h += [Line2D([], [], color="C0", lw=1.2, label="trajectory"),
              Line2D([], [], color="k", marker="o", ls="", ms=3, label="endpoint")]
    h += [Line2D([], [], color="limegreen", marker="o", ls="", mfc="none", ms=7,
                 label=f"train saddle ({len(tr)})  r = {HIT} Å"),
          Line2D([], [], color="darkorange", marker="o", ls="", mfc="none", ms=7,
                 label=f"test saddle ({len(te)})  r = {HIT} Å"),
          Line2D([], [], color="red", marker="x", ls="", ms=5,
                 label=f"saddle-less midpoint ({len(geo['ns'])})")]
    a.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.015),
             ncol=3, fontsize=8, frameon=False)


def _xlabel(n):
    return f"in-plane distance to the nearest of the {n} saddles (Å)"


def page(z, label, row, geo, show_starts=False):
    """Trajectories (left) and start/end distance histograms (right)."""
    P, tr, te = z["paths"], z["train_saddles"], z["test_saddles"]
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 6.6),
                           gridspec_kw={"width_ratios": [1.35, 1]})
    a = ax[0]
    _sheet(a, geo, tr, te)
    segs = P[:, :, :2]
    # A path that wraps through the cell edge would draw a line across the
    # whole sheet; leave those out of the drawing (they stay in the metrics).
    keep = np.linalg.norm(np.diff(segs, axis=1), axis=2).max(1) <= geo["cell"][0, 0] / 2
    a.add_collection(LineCollection(list(segs[keep]), linewidths=.25, colors="C0",
                                    alpha=.30, zorder=2))
    if show_starts:
        a.scatter(P[:, 0, 0], P[:, 0, 1], s=1.2, c="C0", alpha=.55, zorder=3)
    a.scatter(P[:, -1, 0], P[:, -1, 1], s=2.0, c="k", alpha=.75, zorder=5)
    a.set_title(f"{label}   {len(P)} throws ({int(keep.sum())} drawn, "
                f"{int((~keep).sum())} crossing the cell edge not drawn), K={P.shape[1] - 1}",
                fontsize=9)
    _legend(a, geo, tr, te, with_traj=True, with_starts=show_starts)

    b = ax[1]
    A = z["cell"][:2, :2]
    allS = np.vstack([tr, te])
    _, ed = nearest(P[:, -1, :], allS, A)
    _, sd = nearest(P[:, 0, :], allS, A)
    bins = np.logspace(-3, 1, 70)
    b.hist(sd, bins=bins, histtype="step", lw=1.2, color="0.5", label="start")
    b.hist(ed, bins=bins, histtype="step", lw=1.6, color="C0", label="endpoint")
    b.axvline(HIT, ls=":", c="r", lw=1.2, label=f"{HIT:.2f} Å hit radius")
    b.set_xscale("log")
    b.set_xlim(1e-3, 10)
    b.set_xlabel(_xlabel(len(allS)))
    b.set_ylabel("count")
    b.legend(fontsize=8)
    b.grid(alpha=.3)
    l1 = "  ".join(f"{k}={row[k]}" for k in ("p50", "p90", "p99", "max"))
    l2 = "  ".join(f"{k}={row[k]}" for k in
                   ("reached_pct", "switch_pct", "saddleless_pct", "other_fail_pct"))
    b.set_title(l1 + "\n" + l2, fontsize=8)
    fig.tight_layout(pad=1.9)
    return fig


def reference_page(z, geo, positions=None):
    """The same view with no trajectories, so the saddles can be read on their own."""
    tr, te = z["train_saddles"], z["test_saddles"]
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 6.6),
                           gridspec_kw={"width_ratios": [1.35, 1]})
    a = ax[0]
    _sheet(a, geo, tr, te)
    a.set_title("LiC reference frame: the sheet and its saddles, no trajectories\n"
                "(the following pages overlay the throws on this exact view)", fontsize=9)
    _legend(a, geo, tr, te, with_traj=False)
    b = ax[1]
    b.set_xscale("log")
    b.set_xlim(1e-3, 10)
    b.set_xlabel(_xlabel(len(tr) + len(te)))
    b.set_ylabel("count")
    b.grid(alpha=.3)
    b.set_title("reference page\nmetrics appear on the following pages", fontsize=8)
    fig.tight_layout(pad=1.9)
    if positions is not None:
        # This page's title and legend differ in height from a trajectory page's,
        # so tight_layout lands elsewhere; pin the trajectory pages' geometry so
        # the sheet does not jump when flipping pages.
        for ax_, bounds in zip(fig.axes, positions):
            ax_.set_position(bounds)
    return fig


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("throws", nargs="+", help="throw.npz files written by random_throw.py")
    p.add_argument("--labels", nargs="*", default=None,
                   help="one page title per throw file (default: the run directory name)")
    p.add_argument("--out", default=None,
                   help="output PDF (default: throws.pdf next to the first throw file)")
    p.add_argument("--show-starts", action="store_true",
                   help="also mark every start (clutters a 64x64 grid, useful for random starts)")
    p.add_argument("--train-traj", default=str(here / "train_set.traj"))
    p.add_argument("--test-traj", default=str(here / "test_set.traj"))
    args = p.parse_args()

    paths = [Path(t) for t in args.throws]
    labels = args.labels or [t.resolve().parent.parent.name for t in paths]
    if len(labels) != len(paths):
        raise SystemExit("--labels needs one label per throw file")
    out = Path(args.out) if args.out else paths[0].with_name("throws.pdf")
    geo = sheet_geometry(args.train_traj, args.test_traj)
    warnings.filterwarnings("ignore", message=".*tight_layout.*")

    with PdfPages(out) as pdf:
        z0 = np.load(paths[0])
        f0 = page(z0, labels[0], metrics(z0, geo), geo, args.show_starts)
        f0.canvas.draw()
        pos = [a.get_position().bounds for a in f0.axes]
        plt.close(f0)
        fig = reference_page(z0, geo, positions=pos)
        pdf.savefig(fig)
        plt.close(fig)
        for t, label in zip(paths, labels):
            z = np.load(t)
            row = metrics(z, geo)
            fig = page(z, label, row, geo, args.show_starts)
            pdf.savefig(fig)
            plt.close(fig)
            print(f"  {label}: " + "  ".join(f"{k} {v}" for k, v in row.items()))
    print(f"wrote {out} ({len(paths) + 1} pages)")


if __name__ == "__main__":
    main()
