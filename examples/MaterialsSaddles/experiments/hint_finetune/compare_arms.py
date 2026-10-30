"""Compare hint fine-tunes on the 24 test minima, every arm against one reference pooled from all evaluations.

Each arm is an eval_hints.py output directory (the trimmed evaluation: --arms ft_hint_1x8,follow_ft is enough).
Each evaluation's own report.txt grows its reference with that evaluation's finds; here the reference is the
30-minimum experiment's verified saddles plus every listed evaluation's, merged as usual.

    saddles      distinct reference regions per minimum that ft_hint_1x8's Sella runs reach (mean over minima;
                 Wilcoxon p paired over minima against the first arm), and within 5k / 20k force-call equivalents
                 (eval_hints' costing and start order)
    conv, calls  Sella convergence and median force calls per run
    precision    on the regions every arm reaches: the median over (minimum, region) of each arm's per-region median
                 distance Sella moved the landing; and the share of all landings within 0.2 A of a reference region
    following    held-out Tier B regions (follow_ft): landing within 0.2 A of the target, Sella ending within 0.15 A
                 of it, split by whether the arm trained on that minimum (a training region in its labels.json; none
                 for --hint-rows dataset). Secondary: the test minima are the main comparison.
The count curve (tierb_subsample.py) is appended when it exists.

    python examples/MaterialsSaddles/experiments/hint_finetune/compare_arms.py --out <report.txt> \\
        pilot=<eval dir> armA=<eval dir> wide=<eval dir> ...
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_hints as ev  # noqa: E402
from eval_hints import hl, read_rows  # noqa: E402

ARM, BUDGETS = "ft_hint_1x8", (5000, 20000)


def trained_minima(eval_dir):
    """Tier B minima the arm trained on, from its fine-tune's config.json (run directory = eval name's prefix)."""
    run = Path(ev.DATA) / "hint_finetune" / "runs" / Path(eval_dir).name.rsplit("_checkpoint", 1)[0]
    cfg = json.loads((run / "config.json").read_text())["extras"]["finetune"]
    if cfg.get("hint_rows") == "dataset":
        return set()
    labels = Path(cfg["labels"].replace("/scratch/gqs7zr/saddleflow", ev.DATA))      # runs made on Rivanna
    return {e["tid"] for e in json.loads(labels.read_text())["minima"] if any(not r["held_out"] for r in e["regions"])}


def score(d, minima, ref):
    """Per minimum: regions reached with each one's Sella moves, regions within each budget; and landing precision."""
    runs = [r for r in read_rows(d / "runs" / "runs_*.json") if r["arm"] == ARM and r["key"] in minima]
    per_start = ev.ROLLOUT_CALLS_PER_EVAL * ev.EVALS[ARM]
    per = {}
    for t, m in minima.items():
        mine = sorted((r for r in runs if r["key"] == t), key=lambda r: r["id"])
        reg = [ev.endpoint_region(m, ref[t], r["pos"]) if r.get("conv") else -1 for r in mine]
        moved = {}
        for r, k in zip(mine, reg):
            if k >= 0:
                moved.setdefault(k, []).append(r["moved"])
        starts = [i for i, r in enumerate(mine) for _ in r["members"]]
        cost, seen, reached = 0.0, set(), []
        for j in np.random.default_rng([t, 51]).permutation(len(starts)):       # eval_hints' start order
            i = starts[j]
            cost += per_start
            if i not in seen:
                seen.add(i); cost += mine[i].get("sella_calls", 0)
                if reg[i] >= 0:
                    reached.append((cost, reg[i]))
        per[t] = dict(moved=moved, budget=[len({k for c, k in reached if c <= b}) for b in BUDGETS])
    lands = [l for l in read_rows(d / "landings" / f"{ARM}_*.json") if l["arm"] == ARM and l["key"] in minima]
    near = [ev.nearest_region(minima[l["key"]], ref[l["key"]], l["landing"], 0.2)[0] >= 0 for l in lands]
    return dict(per=per, conv=np.mean([bool(r.get("conv")) for r in runs]),
                calls=np.median([r.get("sella_calls", 0) for r in runs]), near=np.mean(near))


def following(d, args):
    """{trained: [(landing hit, Sella hit or None)]} over the held-out regions' follow_ft starts."""
    regions = {k: (m, r, e) for k, m, r, _, _, e in ev.follow_regions(args)}
    runs = {r["id"]: r for r in read_rows(d / "runs" / "runs_*.json") if r["arm"] == "follow_ft"}
    trained, out = trained_minima(d), {True: [], False: []}
    for l in read_rows(d / "landings" / "follow_ft_*.json"):
        m, r, e = regions[l["key"]]
        s = runs.get(l["id"])
        hit = (s is not None and s.get("conv")
               and hl.saddle_distance(m, np.array([s["pos"]]), np.array(r["saddle"]))[0] <= ev.LAND_MERGE)
        out[e["tid"] in trained].append((hl.saddle_distance(m, np.array([l["landing"]]), np.array(r["saddle"]))[0] <= 0.2,
                                         bool(hit) if s is not None else None))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("arms", nargs="+", help="name=<eval_hints.py output directory>; the first is the paired baseline")
    p.add_argument("--out", required=True)
    p.add_argument("--curve", default=f"{ev.DATA}/hint_finetune/subsample/count_curve.txt")
    a = p.parse_args()
    dirs = {name: Path(d) for name, d in (x.split("=", 1) for x in a.arms)}
    args = SimpleNamespace(old_dir=ev.OLD_DIR, limit=0, max_hints=0, labels=ev.LABELS, min_barrier=0.05)
    minima = {t: m for t, (m, _) in ev.test_minima(args).items()}
    ref = ev.reference(args, list(dirs.values()), minima)
    res = {name: score(d, minima, ref) for name, d in dirs.items()}
    first = next(iter(res))
    ts = list(minima)
    shared = {t: set.intersection(*(set(res[n]["per"][t]["moved"]) for n in res)) for t in ts}

    L = ["Hint fine-tunes compared on the 24 test minima (ft_hint_1x8, then Sella)", "=" * 118, "",
         f"reference: {sum(len(ref[t]['regions']) for t in ts)} regions pooled from {len(dirs)} evaluations and the "
         f"30-minimum experiment; shared regions (reached by every arm): {sum(map(len, shared.values()))}", "",
         f"  arm          saddles/min  p vs {first:9s} | {'  '.join(f'{b // 1000:>3d}k' for b in BUDGETS)} | conv%  calls p50 | "
         "shared regions: moved p50 | landings <= 0.2 A"]
    for name, r in res.items():
        found = [len(r["per"][t]["moved"]) for t in ts]
        base = [len(res[first]["per"][t]["moved"]) for t in ts]
        moved = [np.median(r["per"][t]["moved"][k]) for t in ts for k in shared[t]]
        L.append(f"  {name:12s} {np.mean(found):11.1f}  {ev.paired(found, base) if name != first else '-':>14s} | "
                 + "  ".join(f"{np.mean([r['per'][t]['budget'][i] for t in ts]):4.1f}" for i in range(len(BUDGETS)))
                 + f" | {100 * r['conv']:5.1f}  {r['calls']:9.0f} | {np.median(moved) if moved else np.nan:24.3f} | "
                 f"{100 * r['near']:16.1f}%")

    L += ["", "HINT FOLLOWING (secondary): held-out Tier B regions, follow_ft; by whether the arm trained on the minimum",
          "  arm          trained  n | landing within 0.2 A | Sella ends at the target"]
    for name, d in dirs.items():
        for trained, rows in following(d, args).items():
            if rows:
                sella = [s for _, s in rows if s is not None]
                L.append(f"  {name:12s} {'yes' if trained else 'no':>7s} {len(rows):4d} | {100 * np.mean([l for l, _ in rows]):19.1f}% | "
                         + (f"{100 * np.mean(sella):23.1f}%" if sella else f"{'-':>24s}"))
    if Path(a.curve).exists():
        L += ["", "", Path(a.curve).read_text()]
    text = "\n".join(L) + "\n"
    Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
