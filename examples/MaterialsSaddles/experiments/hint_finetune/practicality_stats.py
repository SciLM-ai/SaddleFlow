"""Numbers for scaling the hint method to all of MaterialsSaddles: dataset sizes, atoms per structure, saddles the
dataset already holds per minimum, and how Tier B's cost and saddle count grew with the number of atoms.

    parquet  (needs pyarrow; no ase) metadata/triplets.parquet: transition states, distinct source structures
             (source_id) and chemical systems per subset, main files and tier2/ separately
    shards   (the saddleflow environment) one test shard per subset (mp20bat: train) read whole: atoms and fixed
             atoms per structure; per source structure the distinct minima (reactants and products merged within
             0.1 A), its "home" minimum (the one most triplets touch) and the distinct saddles <= 1 eV above the home
             minimum (merged within 0.1 A). Then Tier B: Sella seconds per run and connected regions per minimum
             against the number of atoms (power-law fits).

    python practicality_stats.py parquet --parquet <triplets.parquet> [--files <file in the repository> ...]
    python practicality_stats.py shards --shards <subset>=<file> ... --tierb <tierb dir>
"""
import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def parquet(args):
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    t = pq.read_table(args.parquet, columns=["subset", "file", "source_id", "chemical_system"])
    tier2 = pc.starts_with(t["file"], "tier2/")
    print("subset   files    transition states   sources   TS per source: mean  p50  p90   max   chemical systems")
    for sub in sorted(set(t["subset"].to_pylist())):
        for name, mask in (("main", pc.invert(tier2)), ("tier2", tier2), ("all", None)):
            k = pc.equal(t["subset"], sub) if mask is None else pc.and_(pc.equal(t["subset"], sub), mask)
            s = t.filter(k)
            per = np.array(list(Counter(s["source_id"].to_pylist()).values()))
            print(f"{sub:8s} {name:6s} {len(s):14d} {len(per):12d} {per.mean():18.1f} {np.median(per):4.0f} "
                  f"{np.percentile(per, 90):4.0f} {per.max():6d} {len(set(s['chemical_system'].to_pylist())):12d}")
    # Are a source's transition states in one file? (the shards stage counts minima and saddles within one file)
    total = Counter(t["source_id"].to_pylist())
    for f in args.files:
        s = t.filter(pc.equal(t["file"], f))
        here = Counter(s["source_id"].to_pylist())
        share = np.array([here[k] / total[k] for k in here])
        print(f"{f}: {len(s)} TS, {len(here)} sources; share of each source's TS in this file: mean "
              f"{share.mean():.2f}, p10 {np.percentile(share, 10):.2f}")


def source_of(subset, info):
    o = info.get("orig_info", {})
    if subset == "mp20bat":
        return f"{o.get('charge_id')}|{o.get('discharge_id')}|{o.get('working_ion')}"
    o = o.get("orig_info", {}) if isinstance(o, dict) else {}
    return str(o.get({"lemat": "immutable_id", "oc20": "source_file", "oc22": "sid"}[subset],
                     o.get("possible_immutable_ids", "?")))


def dist(cell, P, q):
    """Largest per-atom minimum-image distance from each structure in P to q, overall shift removed."""
    d = np.asarray(P) - np.asarray(q)[None]
    f = d @ np.linalg.inv(cell); d = (f - np.round(f)) @ cell
    d -= d.mean(-2, keepdims=True)
    return np.linalg.norm(d, axis=-1).max(-1)


def dedup(cell, structures, radius=0.1):
    """Index of the distinct structure for each input (first come)."""
    reps, out = [], []
    for x in structures:
        if reps:
            d = dist(cell, np.array([structures[r] for r in reps]), x)
            k = int(d.argmin())
            if d[k] < radius:
                out.append(k); continue
        out.append(len(reps)); reps.append(len(out) - 1)
    return out


def q(x, ps=(50, 90)):
    x = np.asarray(x, float)
    return f"{x.mean():7.1f} " + " ".join(f"{np.percentile(x, p):5.0f}" for p in ps)


def shards(args):
    from ase.db import connect
    print("subset   triplets sources | atoms mean p50 p90 | free atoms mean p50 p90 | minima per source mean p50 "
          "| triplets touching home | saddles <= 1 eV at home: mean p50 p90")
    for spec in args.shards:
        sub, path = spec.split("=", 1)
        rows = defaultdict(list)                        # source -> [(R, S, P, barrier_rs, barrier_ps)]
        natoms, nfree, cells = [], [], {}
        with connect(path) as db:
            trip = []
            for row in db.select():
                trip.append(row)
                if len(trip) < 3:
                    continue
                R, S, P = trip; trip = []
                info = R.data.get("info", {}); sinfo = S.data.get("info", {})
                a = R.toatoms()
                fixed = sum(len(c.get_indices()) for c in a.constraints)
                natoms.append(len(a)); nfree.append(len(a) - fixed)
                src = (source_of(sub, info), len(a))      # mp20bat: one source holds structures with and without a vacancy
                cells[src] = a.cell.array
                rows[src].append((a.positions, S.toatoms().positions, P.toatoms().positions,
                                  sinfo.get("barrier_rs", info.get("barrier_rs", np.nan)),
                                  sinfo.get("barrier_ps", info.get("barrier_ps", np.nan))))
        n_min, home_share, home_saddles = [], [], []
        for src, T in rows.items():
            cell = cells[src]
            ends = [t[0] for t in T] + [t[2] for t in T]
            gid = dedup(cell, ends)
            n_min.append(len(set(gid)))
            home = Counter(gid).most_common(1)[0][0]
            gR, gP = gid[:len(T)], gid[len(T):]
            touch = [i for i in range(len(T)) if gR[i] == home or gP[i] == home]
            home_share.append(len(touch) / len(T))
            S = [T[i][1] for i in touch if (T[i][3] if gR[i] == home else T[i][4]) <= 1.0]
            home_saddles.append(len(set(dedup(cell, S))) if S else 0)
        print(f"{sub:8s} {len(natoms):8d} {len(rows):7d} | {q(natoms)} | {q(nfree)} | {q(n_min, (50,))} "
              f"| {100 * np.mean(home_share):21.0f}% | {q(home_saddles)}")

    if args.tierb:
        tb = Path(args.tierb)
        minima = {m["tid"]: m for m in json.loads((tb / "labels.json").read_text())["minima"]}
        runs = [r for f in sorted(glob.glob(str(tb / "runs" / "*.json"))) for r in json.loads(Path(f).read_text())]
        sec = defaultdict(list); calls = defaultdict(list)
        for r in runs:
            if r["tid"] in minima and r.get("seconds"):
                sec[r["tid"]].append(r["seconds"]); calls[r["tid"]].append(r.get("sella_calls", 0))
        ts = sorted(sec)
        n = np.array([minima[t]["natoms"] for t in ts], float)
        s = np.array([np.mean(sec[t]) for t in ts]); c = np.array([np.mean(calls[t]) for t in ts])
        regs = np.array([len(minima[t]["regions"]) for t in ts], float)
        fit = lambda y: np.polyfit(np.log(n), np.log(np.maximum(y, 1e-9)), 1)
        print(f"\nTier B: {len(ts)} minima, {len(runs)} Sella runs; atoms mean {n.mean():.1f} (p10 {np.percentile(n, 10):.0f}, "
              f"p90 {np.percentile(n, 90):.0f})")
        for name, y in (("Sella seconds per run (8 workers per GPU)", s), ("Sella force calls per run", c),
                        ("connected regions <= 1 eV per minimum", regs)):
            b, a = fit(y)
            print(f"  {name:44s} mean {y.mean():7.1f}  ~ {np.exp(a):.3g} x atoms^{b:.2f}  "
                  f"(at 20 / 40 / 80 / 150 atoms: " + " / ".join(f"{np.exp(a) * x ** b:.1f}" for x in (20, 40, 80, 150)) + ")")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["parquet", "shards"])
    p.add_argument("--parquet")
    p.add_argument("--files", nargs="*", default=[], help="parquet: files whose sources to check for being whole")
    p.add_argument("--shards", nargs="*", default=[])
    p.add_argument("--tierb")
    a = p.parse_args()
    {"parquet": parquet, "shards": shards}[a.mode](a)


if __name__ == "__main__":
    main()
