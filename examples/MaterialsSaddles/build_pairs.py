"""Turn a multi-start dump + its Sella results into a `--pair-override` npz.

Pair = (endpoint the model reached, saddle Sella converged to from it), kept only
when Sella converged AND the endpoint is index-1 (`nneg == 1`). Pairs whose endpoint
had to move more than --moved-thr (max-atom, MIC) are 'corrections' and get
--moved-weight (repeated entries), as in the LiC round-2 recipe.

  python build_pairs.py --traj '<dir>/r2_*.traj' --sella '<dir>/sella_*.json' --out pairs.npz
"""
import argparse, glob, json
import numpy as np
from ase.io import Trajectory


def mic_maxd(x, y, cell):
    d = x - y; f = np.linalg.solve(cell.T, d.T).T; f -= np.round(f); return float(np.sqrt(((f @ cell) ** 2).sum(1)).max())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traj", required=True); p.add_argument("--sella", required=True); p.add_argument("--out", required=True)
    p.add_argument("--moved-thr", type=float, default=0.5); p.add_argument("--moved-weight", type=int, default=5)
    p.add_argument("--keep-sigma", default=None, help="comma list: keep only endpoints whose start sigma is in this set (e.g. '0')")
    a = p.parse_args()
    keep = None if a.keep_sigma is None else {float(x) for x in a.keep_sigma.split(",")}
    ends = {}
    for f in sorted(glob.glob(a.traj)):
        for at in Trajectory(f):
            if keep is not None and float(at.info.get("sigma", -1)) not in keep: continue
            ends[(int(at.info["tid"]), str(at.info["src"]))] = (at.get_positions(), np.array(at.get_cell()), float(at.info.get("sigma", -1)))
    sel = {}
    for f in sorted(glob.glob(a.sella)):
        try: rs = json.load(open(f))
        except Exception: continue
        for r in rs: sel[(int(r["tid"]), str(r["src"]))] = r
    T, X, S, W = [], [], [], []; n_conv = n_idx1 = n_moved = 0; per_sigma = {}
    for key, (x0, cell, sig) in ends.items():
        r = sel.get(key)
        if r is None: continue
        if not r.get("conv"): continue
        n_conv += 1
        if r.get("nneg") != 1 or r.get("pos") is None: continue
        n_idx1 += 1
        tgt = np.array(r["pos"]); moved = mic_maxd(x0, tgt, cell)
        w = a.moved_weight if moved > a.moved_thr else 1; n_moved += moved > a.moved_thr
        st = per_sigma.setdefault(sig, []); st.append(moved)
        T.append(key[0]); X.append(x0.astype(np.float32)); S.append(tgt.astype(np.float32)); W.append(w)
    off = np.concatenate([[0], np.cumsum([len(x) for x in X])])
    np.savez(a.out, tids=np.array(T), offsets=off, x0=np.concatenate(X, 0), target=np.concatenate(S, 0), weight=np.array(W))
    print(f"endpoints {len(ends)}, with Sella result {sum(1 for k in ends if k in sel)}, converged {n_conv}, index-1 kept {n_idx1} "
          f"({100*n_idx1/max(1,len(ends)):.1f}%), corrections (moved > {a.moved_thr} A) {n_moved} ({100*n_moved/max(1,n_idx1):.1f}%) x{a.moved_weight}; "
          f"triplets {len(set(T))}; wrote {a.out}")
    for sig in sorted(per_sigma):
        m = np.array(per_sigma[sig]); print(f"  sigma {sig:.2f}: n={len(m)} moved med {np.median(m):.3f} mean {m.mean():.3f} >0.5A {100*np.mean(m>0.5):.1f}%")


if __name__ == "__main__":
    main()
