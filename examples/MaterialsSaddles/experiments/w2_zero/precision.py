"""Precision of the flow landings on the hint starts, against the reference saddles: CPU, from mirror-test output.

For each mirror-test directory (mirror_test.py landings.json; the same 1,536 hint starts for every model):
    d          distance of the unchanged landing to the nearest reference saddle (the 30-minimum experiment's
               verified regions; largest per-atom distance, rigid shift removed)
    hit        d <= 0.2 A (the mirror test's "near"); the table gives the share of starts within 0.05, 0.1, 0.2 A
               and, among hits, the median and p90 of d
    inverted   for the hits, the distance of the inverted start's landing (mapped back) to the SAME saddle, and
               the gap (inverted - unchanged): the precision lost to the missing mirror symmetry alone
Paired against the first directory (the same starts): Wilcoxon signed-rank on d over all starts, and the share of
starts where this model lands closer.

    python examples/MaterialsSaddles/experiments/w2_zero/precision.py <mirror dir> [<mirror dir> ...] --out <file>
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mirror_test"))
import mirror_test as mt  # noqa: E402
from mirror_test import ev, hl  # noqa: E402

NEAR = mt.NEAR


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+")
    p.add_argument("--out", default=None, help="also write the report here")
    p.add_argument("--fm-run", default=mt.FM_RUN)
    p.add_argument("--per-minimum", type=int, default=64)
    args = p.parse_args()
    sets, ea = mt.start_sets(SimpleNamespace(fm_run=args.fm_run, per_minimum=args.per_minimum))
    jobs = sets["hint"]
    minima = {t: m for t, (m, _) in ev.test_minima(ea).items()}
    ref = ev.reference(ea, [], minima)

    res = {}
    for d in args.dirs:
        f = json.loads((Path(d) / "landings.json").read_text())
        L = {k: [np.array(x) for x in f["landings"][f"hint/{k}"]] for k in ("unchanged", "inversion")}
        if len(L["unchanged"]) != len(jobs):
            raise SystemExit(f"{d}: {len(L['unchanged'])} hint landings, expected {len(jobs)}")
        near = [ev.nearest_region(minima[j[1]], ref[j[1]], x, NEAR) for j, x in zip(jobs, L["unchanged"])]
        dist = np.array([x for _, x in near])
        k = np.array([r for r, _ in near])
        inv = np.array([float(hl.saddle_distance(minima[j[1]], ref[j[1]]["regions"][r]["pos"][None], x)[0])
                        if r >= 0 else np.nan for j, x, r in zip(jobs, L["inversion"], k)])
        name = f["ckpt"] if f["ckpt"] == "released" else Path(f["ckpt"]).parent.name
        res[d] = dict(label=name + (" W2=0" if f.get("w2_zero") else ""), d=dist, hit=k >= 0, inv=inv)

    q = lambda v: f"{np.median(v):7.4f} {np.quantile(v, 0.9):7.4f}" if len(v) else "    n/a     n/a"
    first = args.dirs[0]
    L = [f"Precision on the {len(jobs)} hint starts against the reference saddles (A)", "=" * 100, "",
         f"  {'model':28s} | <=0.05  <=0.1  <=0.2 | hits: d p50     p90 | inverted p50   p90 | gap p50    p90 "
         f"| vs first: closer  p"]
    for d, r in res.items():
        h = r["hit"]
        gap = r["inv"][h] - r["d"][h]
        line = (f"  {r['label']:28s} | " + " ".join(f"{100 * np.mean(r['d'] <= c):5.1f}%" for c in (0.05, 0.1, NEAR))
                + f" | {q(r['d'][h])} | {q(r['inv'][h])} | {q(gap)}")
        if d != first:
            line += f" | {100 * np.mean(r['d'] < res[first]['d']):5.1f}%  {ev.paired(r['d'], res[first]['d'])}"
        L.append(line)
    text = "\n".join(L)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
