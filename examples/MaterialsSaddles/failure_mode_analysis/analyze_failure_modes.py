"""How often do landing points miss a saddle, and does the starting force flag them?

Reads a run directory written by run_failure_mode_analysis.sh (make_proposals.py + sella_eval.py
--check-index). Each landing point is labelled by what Sella does from it, never by its force:

    real           Sella converged, exactly one negative mode, and moved it <= the maxd cutoff
    moved_far      converged to an index-1 saddle, but farther than the cutoff
    minimum        converged to a stationary point with no negative mode
    higher_order   converged to a stationary point with two or more negative modes
    not_converged  Sella did not reach fmax
    error          Sella or the calculator raised

Everything but `real` counts as a false landing point. Then `fmax_start` (UMA force at the landing
point, before Sella) is tested as a one-call filter. The maxd cutoff that defines "near" is not fixed in
advance; every table is repeated for each --maxd-cutoffs value. CPU only, no model or UMA calls.

    python examples/MaterialsSaddles/failure_mode_analysis/analyze_failure_modes.py <run_dir>
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
from ase.geometry import find_mic
from ase.io import read

CATEGORIES = ("real", "moved_far", "minimum", "higher_order", "not_converged", "error")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--maxd-cutoffs", default="0.1,0.2,0.5", help="A; max per-atom Sella displacement for 'real'.")
    p.add_argument("--primary-cutoff", type=float, default=0.2, help="Cutoff used for the plots.")
    p.add_argument("--force-thresholds", default="0.01,0.02,0.03,0.05,0.1,0.2,0.5,1.0", help="eV/A filter thresholds.")
    p.add_argument("--match-tol", type=float, default=0.1, help="A; Sella saddle counts as the stored one within this.")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def load(run_dir):
    run = Path(run_dir)
    records = []
    for path in sorted(glob.glob(str(run / "sella*.json"))):
        records += json.loads(Path(path).read_text())
    by_tid = {r["tid"]: r for r in records}
    proposals = read(run / "proposals.traj", index=":")
    minima = read(run / "minima.traj", index=":")
    saddles = read(run / "stored_saddles.traj", index=":")
    selection = json.loads((run / "selection.json").read_text())
    missing = [a.info["tid"] for a in proposals if a.info["tid"] not in by_tid]
    return selection, proposals, minima, saddles, [by_tid.get(a.info["tid"]) for a in proposals], missing


def max_mic(a_pos, b_pos, cell, pbc):
    return float(find_mic(np.asarray(a_pos) - np.asarray(b_pos), cell, pbc)[1].max())


def category(record, cutoff):
    if record.get("err") is not None:
        return "error"
    if not record["conv"]:
        return "not_converged"
    if record.get("nneg") is None:
        raise ValueError("Sella was run without --check-index; the index labels are missing")
    if record["nneg"] == 0:
        return "minimum"
    if record["nneg"] >= 2:
        return "higher_order"
    return "real" if record["maxd"] <= cutoff else "moved_far"


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def auc(false_scores, real_scores):
    """P(false landing point has the larger starting force); ties count half."""
    if not len(false_scores) or not len(real_scores):
        return np.nan
    f, r = np.asarray(false_scores)[:, None], np.asarray(real_scores)[None, :]
    return float(np.mean((f > r) + 0.5 * (f == r)))


def quantiles(values):
    if not len(values):
        return "        n/a"
    q = np.quantile(values, [0.1, 0.5, 0.9])
    return f"{q[0]:8.4f} {q[1]:8.4f} {q[2]:8.4f}"


def main():
    args = parse_args()
    cutoffs = sorted(map(float, args.maxd_cutoffs.split(",")))
    thresholds = list(map(float, args.force_thresholds.split(",")))
    selection, proposals, minima, saddles, records, missing = load(args.run_dir)
    keep = [i for i, r in enumerate(records) if r is not None]
    records = [records[i] for i in keep]
    proposals, minima, saddles = [proposals[i] for i in keep], [minima[i] for i in keep], [saddles[i] for i in keep]
    s = selection["settings"]
    fstart = np.array([r.get("fmax_start", np.nan) for r in records])
    landed = np.array([max_mic(p.positions, m.positions, p.cell, p.pbc) for p, m in zip(proposals, minima)])
    stored = np.array([max_mic(r["pos"], sd.positions, sd.cell, sd.pbc) if r.get("pos") is not None else np.nan
                       for r, sd in zip(records, saddles)])
    lines = ["failure-mode analysis of the unconditional model (Sella-labelled landing points)", "=" * 100,
             f"data={s['data']}", f"model={s['model']} inference={s.get('inference')} seed={s['seed']}",
             f"{s['n_endpoints']} endpoints -> {s['n_unique_minima']} unique minima -> {s['n_chosen']} starts; "
             f"{len(records)} with Sella results" + (f" ({len(missing)} MISSING, e.g. tids {missing[:5]})" if missing else ""),
             "Labels come from Sella (converged, index 1, displacement), never from fmax_start.",
             f"Model displacement from the minimum (max per-atom, A): p10/p50/p90 = {quantiles(landed)}",
             f"fmax_start NaN (failed before Sella) for {int(np.isnan(fstart).sum())} landing points", ""]
    summary = {}
    for cut in cutoffs:
        cats = np.array([category(r, cut) for r in records])
        false = cats != "real"
        lo, hi = wilson(int(false.sum()), len(cats))
        valid = ~np.isnan(fstart)
        a = auc(fstart[false & valid], fstart[~false & valid])
        summary[cut] = dict(cats=cats, false=false, auc=a)
        lines += [f"--- 'real' = converged, index 1, Sella maxd <= {cut:g} A " + "-" * 40,
                  f"false landing points: {false.sum()}/{len(cats)} = {100 * false.mean():.1f}%  "
                  f"(95% CI {100 * lo:.1f}-{100 * hi:.1f}%)",
                  "category          N      %   fmax_start p10/p50/p90 (eV/A)   Sella maxd p50   Sella nfc p50"]
        for c in CATEGORIES:
            m = cats == c
            if not m.any():
                continue
            maxd = np.array([r.get("maxd", np.nan) for r in records])[m]
            nfc = np.array([r.get("nfc", -1) for r in records])[m]
            lines.append(f"{c:14s} {m.sum():5d} {100 * m.mean():6.1f}   {quantiles(fstart[m & valid]):>32s}"
                         f"   {np.nanmedian(maxd) if np.isfinite(maxd).any() else np.nan:12.3f}"
                         f"   {np.median(nfc[nfc >= 0]) if (nfc >= 0).any() else np.nan:12.0f}")
        lines += [f"fmax_start as a filter: AUC = {a:.3f}  (P[false point has the larger force]; 0.5 = no signal)",
                  "  threshold   false caught   real discarded   kept that are real   kept"]
        for thr in thresholds:
            flagged = fstart > thr
            caught = flagged[false & valid].mean() if (false & valid).any() else np.nan
            lost = flagged[~false & valid].mean() if (~false & valid).any() else np.nan
            kept = valid & ~flagged
            purity = (~false[kept]).mean() if kept.any() else np.nan
            lines.append(f"  {thr:8.3f}   {100 * caught:11.1f}%   {100 * lost:13.1f}%   {100 * purity:17.1f}%   {kept.sum():5d}")
        real = cats == "real"
        if real.any():
            hit = stored[real] <= args.match_tol
            lines.append(f"real saddles that are the stored NEB saddle (within {args.match_tol:g} A): "
                         f"{hit.sum()}/{real.sum()} = {100 * hit.mean():.1f}%; the rest are other real saddles")
        lines.append("")
    text = "\n".join(lines) + "\n"
    out = Path(args.run_dir)
    (out / "report.txt").write_text(text)
    print(text, flush=True)
    if not args.no_plot:
        plot(out / "failure_modes.pdf", records, fstart, summary, args.primary_cutoff if args.primary_cutoff in summary
             else cutoffs[len(cutoffs) // 2])


def plot(path, records, fstart, summary, cut):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    colors = dict(real="tab:green", moved_far="tab:orange", minimum="tab:blue", higher_order="tab:purple",
                  not_converged="tab:red", error="black")
    maxd = np.array([r.get("maxd", np.nan) for r in records])
    cats = summary[cut]["cats"]
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        for c in CATEGORIES:
            m = (cats == c) & np.isfinite(fstart) & np.isfinite(maxd)
            if m.any():
                ax.scatter(fstart[m], np.maximum(maxd[m], 1e-4), s=10, alpha=.7, color=colors[c], label=f"{c} ({m.sum()})")
        ax.axhline(cut, color=".4", ls="--", lw=.8)
        ax.set(xscale="log", yscale="log", xlabel="UMA fmax at the landing point (eV/A)",
               ylabel="Sella displacement, max per atom (A)", title=f"Landing points ('real' needs maxd <= {cut:g} A)")
        ax.legend(fontsize=7); ax.grid(alpha=.3)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        fig, ax = plt.subplots(figsize=(5, 5))
        for c, entry in summary.items():
            false, valid = entry["false"], np.isfinite(fstart)
            if not (false & valid).any() or not (~false & valid).any():
                continue
            grid = np.sort(np.unique(fstart[valid]))[::-1]
            tpr = [(fstart[false & valid] >= g).mean() for g in grid]
            fpr = [(fstart[~false & valid] >= g).mean() for g in grid]
            ax.plot([0] + fpr, [0] + tpr, label=f"maxd <= {c:g} A (AUC {entry['auc']:.2f})")
        ax.plot([0, 1], [0, 1], color=".6", ls=":")
        ax.set(xlabel="real saddles discarded", ylabel="false landing points caught", title="fmax_start as a filter")
        ax.legend(fontsize=7); ax.grid(alpha=.3)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print(f"[plot] {path}")


if __name__ == "__main__":
    main()
