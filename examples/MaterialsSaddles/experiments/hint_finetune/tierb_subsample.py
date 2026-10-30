"""Tier B with fewer Sella runs: the count curve, and the "wide" and "deep" label sets (no new runs).

A subsample keeps k of a minimum's 128 runs (half atom pushes, half soft pushes, seeded). Its saddles are the
verified points (index 1, connected, barrier 0.05-1 eV) that at least one kept run reached, merged into regions as
Tier B's were (hint_labels.merge, priority = kept runs reaching the point). Cost: the kept runs' Sella force calls.

    curve   per k (8, 15, 32, 64, 128) and 3 seeds: regions per minimum, hint rows, Sella force calls
    wide    all minima of labels.json, k = 15 (seed 0)
    deep1/2 minima in a random order (seed 1 or 2), all 128 runs each, until the Sella force calls match wide's
Every label set keeps each minimum's held-out Tier B regions unchanged (validation and the hint-following test
stay the same) and drops training regions within 0.15 A of one of them.

    python examples/MaterialsSaddles/experiments/hint_finetune/tierb_subsample.py --out-dir <dir>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_hints as ev  # noqa: E402
import tierb  # noqa: E402
from eval_hints import hl, read_rows  # noqa: E402

TIERB = Path(ev.LABELS).parent
KS, SEEDS, WIDE_K, LOW, CUTOFF = (8, 15, 32, 64, 128), (0, 1, 2), 15, 0.05, 1.0


def subsample(runs, k, seed, tid):
    """Ids of k runs of one minimum, half atom pushes and half soft pushes."""
    rng = np.random.default_rng([tid, k, seed])
    keep = []
    for arm, n in (("atom", k - k // 2), ("soft", k // 2)):
        ids = sorted(r["id"] for r in runs if r["arm"] == arm)
        keep += [ids[i] for i in rng.choice(len(ids), min(n, len(ids)), replace=False)]
    return set(keep)


def regions(m, saddles, kept):
    """Representative verified points of the regions the kept runs reach."""
    S = [s for s in saddles if any(i in kept for i in s["members"])]
    if not S:
        return []
    _, reps = hl.merge(m, [s["pos"] for s in S], [s["barrier"] for s in S], [sum(i in kept for i in s["members"]) for s in S])
    return [S[r] for r in reps]


def label_set(entries, minima, saddles, kept):
    """labels.json minima: held-out regions as in Tier B, training regions from the kept runs ({tid: run ids})."""
    out = []
    for e in entries:
        m, held = minima[e["tid"]], [r for r in e["regions"] if r["held_out"]]
        train = []
        for s in regions(m, saddles[e["tid"]], kept.get(e["tid"], set())):
            if held and hl.saddle_distance(m, np.array([h["saddle"] for h in held]), np.array(s["pos"])).min() < hl.MERGE_RADIUS:
                continue
            train.append(dict(region=len(e["regions"]) + len(train), held_out=False, barrier=s["barrier"],
                              saddle=np.round(m.positions + hl.displacement(m, s["pos"]), 5).tolist(),
                              hints=[dict(atom=a, direction=np.round(u, 6).tolist(), move=round(mv, 4))
                                     for a, u, mv in hl.make_hints(m, s["pos"])]))
        out.append(dict(e, regions=held + train))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    out = Path(p.parse_args().out_dir); out.mkdir(parents=True, exist_ok=True)

    labels = json.loads(Path(ev.LABELS).read_text())
    entries = labels["minima"]
    minima = {e["tid"]: Atoms(numbers=e["numbers"], positions=e["positions"], cell=e["cell"], pbc=True) for e in entries}
    runs = {}
    for r in read_rows(TIERB / "runs" / "runs_*.json"):
        if r["tid"] in minima:
            runs.setdefault(r["tid"], []).append(r)
    calls = {r["id"]: r.get("sella_calls", 0) for R in runs.values() for r in R}
    saddles = {t: [s for s in P if s.get("nneg") == 1 and s.get("connected") and LOW <= s["barrier"] <= CUTOFF]
               for t, P in tierb.verified_points(TIERB, minima).items()}
    saddles = {t: saddles.get(t, []) for t in minima}

    L = ["Tier B subsampled: regions per minimum against Sella runs per minimum", "=" * 100, "",
         f"{len(minima)} minima; saddles: index 1, connected, {LOW}-{CUTOFF} eV; mean over {len(SEEDS)} seeds (min-max)", "",
         "     k | regions per minimum: mean  median | hint rows          | Sella force calls (all minima)"]
    for k in KS:
        reg, rows, cost = [], [], []
        for seed in SEEDS:
            kept = {t: subsample(runs[t], k, seed, t) for t in minima}
            R = {t: regions(minima[t], saddles[t], kept[t]) for t in minima}
            reg.append([len(R[t]) for t in minima])
            rows.append(sum(len(hl.make_hints(minima[t], s["pos"])) for t in minima for s in R[t]))
            cost.append(sum(calls[i] for t in minima for i in kept[t]))
        L.append(f"  {k:4d} | {np.mean([np.mean(r) for r in reg]):24.1f} {np.mean([np.median(r) for r in reg]):7.1f} | "
                 f"{np.mean(rows):7.0f} ({min(rows)}-{max(rows)}) | {np.mean(cost):10.0f} ({min(cost)}-{max(cost)})")

    wide = {t: subsample(runs[t], WIDE_K, 0, t) for t in minima}
    budget = sum(calls[i] for t in minima for i in wide[t])
    sets = {"wide": wide}
    for d in (1, 2):
        deep, spent = {}, 0
        for t in np.random.default_rng(d).permutation(sorted(minima)):
            full = {r["id"] for r in runs[int(t)]}
            if spent + sum(calls[i] for i in full) / 2 > budget:
                break
            deep[int(t)] = full; spent += sum(calls[i] for i in full)
        sets[f"deep{d}"] = deep
    L += ["", "label sets (training regions and rows exclude the held-out regions; barrier >= 0.05 eV)",
          "  set      minima searched  Sella force calls | training regions  hint rows"]
    for name, kept in sets.items():
        E = label_set(entries, minima, saddles, kept)
        (out / f"{name}.json").write_text(json.dumps(dict(labels, minima=E)))
        train = [r for e in E for r in e["regions"] if not r["held_out"]]
        L.append(f"  {name:8s} {len(kept):14d} {sum(calls[i] for t in kept for i in kept[t]):18d} | "
                 f"{len(train):16d} {sum(len(r['hints']) for r in train):10d}")
    full = [r for e in entries for r in e["regions"] if not r["held_out"] and r["barrier"] >= LOW]
    L.append(f"  {'tierb':8s} {len(minima):14d} {sum(calls.values()):18d} | {len(full):16d} {sum(len(r['hints']) for r in full):10d}")
    text = "\n".join(L) + "\n"
    (out / "count_curve.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
