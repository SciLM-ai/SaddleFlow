"""Collect a SaddleMill (method = Sella) run into (a) per-case JSON records in the `sella_eval.py`
format {tid, src, conv, nneg, pos, nfc, ...} so `build_pairs.py` works unchanged, and (b) a
statistics table of the max-atom displacement between each flow endpoint and its Sella saddle.

  python sm_collect.py --throws '<dir>/thr_*.traj' --sm-run <saddlemill run dir> --out-json <file> [--label NAME]
"""
import argparse, glob, json
import numpy as np
from ase.io import Trajectory


def mic_maxd(x, y, cell):
    d = x - y; f = np.linalg.solve(cell.T, d.T).T; f -= np.round(f); return float(np.sqrt(((f @ cell) ** 2).sum(1)).max())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--throws", required=True); p.add_argument("--sm-run", required=True); p.add_argument("--out-json", required=True)
    p.add_argument("--label", default="model")
    a = p.parse_args()
    ends = {}
    for f in sorted(glob.glob(a.throws)):
        for at in Trajectory(f):
            ends[(int(at.info["tid"]), str(at.info["src"]))] = (at.get_positions(), np.array(at.get_cell()))
    recs = []
    # Read only SaddleMill's per-rank result files.  The run dir also holds the input copy under data/ and, for
    # any job that was still running when the pass was ended, a live per-step `sella_*.traj` with thousands of
    # unconverged frames -- sweeping those in dilutes the converged / index-1 percentages.
    files = sorted(glob.glob(f"{a.sm_run}/Sella_trajes/collected_ts_rank_*.traj"))
    if not files: files = [f for f in sorted(glob.glob(f"{a.sm_run}/**/*.traj", recursive=True)) if "/data/" not in f]
    for f in files:
        try: frames = list(Trajectory(f))
        except Exception as e: print(f"  skipping {f}: {type(e).__name__}"); continue
        for at in frames:
            i = at.info; o = i.get("orig_info", {}) or {}
            tid = i.get("tid", o.get("tid")); src = i.get("src", o.get("src", "?"))
            if tid is None: continue
            recs.append({"tid": int(tid), "src": str(src), "conv": bool(i.get("converged", 0)), "nneg": (int(i["nneg"]) if i.get("nneg") is not None else None),
                         "pos": at.get_positions().tolist(), "nfc": int(i.get("n_force_calls", -1)), "nsteps": int(i.get("n_steps", -1)), "status": str(i.get("status", ""))})
    json.dump(recs, open(a.out_json, "w"))
    # Strict protocol: a case counts as converged only if Sella actually ran to fmax (status 'converged' or
    # 'converged_after_extension'; SaddleMill's 'converged_to_desorption' fires at step 0 on some bulk cells and
    # stamps converged=1 with a zero displacement), and as index-1 only if the Hessian check ran and found nneg == 1.
    # Throws with no record (Sella raised inside its own step -- systematically the largest cells) are failures,
    # so every percentage is over the number of THROWS, not over the records that happened to survive.
    d, dok, nfc = [], [], []
    n_conv = n_idx1 = n_desorb = n_unverified = 0
    for r in recs:
        key = (r["tid"], r["src"]); e = ends.get(key)
        if e is None: continue
        conv = r["status"] in ("converged", "converged_after_extension")
        if r["status"] == "converged_to_desorption": n_desorb += 1
        if conv: n_conv += 1; nfc.append(r["nfc"])
        if conv and r["nneg"] is None: n_unverified += 1
        ok = conv and r["nneg"] == 1
        if ok: n_idx1 += 1; dok.append(mic_maxd(e[0], np.array(r["pos"]), e[1]))
    n = len(recs); nt = max(1, len(ends)); dok = np.array(dok)
    # A SaddleMill pass that was killed part-way leaves the jobs that finished FIRST, i.e. the easy ones, so its
    # median looks good and is not comparable with a complete row.  Say so loudly rather than emit a normal row.
    if n < 0.9 * nt:
        print(f"{a.label}: *** TRUNCATED PASS: only {n} records for {nt} throws ({100*n/nt:.0f}%). "
              f"The survivors are the cases that finished first, so these statistics are biased optimistic "
              f"and must NOT be compared with complete rows -- re-run the Sella pass. ***")
    print(f"{a.label}: throws {len(ends)}, Sella records {n} (no record {len(ends)-n}, desorption-flagged {n_desorb}, index unverified {n_unverified}), "
          f"converged {100*n_conv/nt:.1f}% of throws, index-1 {100*n_idx1/nt:.1f}% of throws, force calls med {np.median(nfc) if nfc else float('nan'):.0f}")
    if len(dok):
        q = np.percentile(dok, [50, 75, 90, 95])
        print(f"{a.label}: maxd endpoint->Sella saddle (converged index-1, n={len(dok)}): mean {dok.mean():.3f}  median {q[0]:.3f}  p75 {q[1]:.3f}  p90 {q[2]:.3f}  p95 {q[3]:.3f}  >0.5A {100*np.mean(dok>0.5):.0f}%  >1A {100*np.mean(dok>1):.0f}%")


if __name__ == "__main__":
    main()
