"""Is the released model mirror-equivariant on mp20bat? The LiC mirror test, on a failure-mode run.

Every start of the run (its exact noise, regenerated) is rolled out three ways with the released schedule:
unchanged, rotated 180 degrees about z (the control: the model is rotation-equivariant by construction), and
inverted (r -> -r). Inversion is the improper operation to test: every mirror is an inversion times a
rotation, and it maps every lattice onto itself. Each landing point is mapped back and compared with the
unchanged one (largest per-atom distance, rigid shift removed). The differences are then set against the
run's own scale: the 0.2 A 'real' cutoff and how far Sella moved each landing point.

    python examples/MaterialsSaddles/failure_mode_analysis/mirror_test.py <run_dir>

Writes <run_dir>/mirror/report.txt and mirror.json. One GPU, about 20 minutes for 500 starts.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_failure_modes import category, load, max_mic_untranslated  # noqa: E402
from higher_order_test import original_starts  # noqa: E402

from saddleflow.data.transforms import transform_structure  # noqa: E402

MAPS = {"unchanged": np.eye(3), "rotation 180 about z": np.diag([-1.0, -1.0, 1.0]), "inversion": -np.eye(3)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--cutoff", type=float, default=0.2)
    p.add_argument("--task-name", default="omat")
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()
    out = Path(args.run_dir) / "mirror"; out.mkdir(exist_ok=True)

    import torch
    from saddleflow import SaddleFlow
    torch.backends.cuda.matmul.allow_tf32 = False
    selection, proposals, minima, _, records, _, _, _ = load(args.run_dir)
    s = selection["settings"]
    starts = original_starts(args.run_dir, s)
    tids = [int(a.info["tid"]) for a in proposals]
    sf = SaddleFlow.from_pretrained(s["model"], device="cuda")
    landing = {}
    for name, R in MAPS.items():
        moved = [transform_structure(starts[t], R) for t in tids]
        props = sf.propose(moved, n=1, sigma=0.0, task_name=args.task_name, batch_size=args.batch_size)
        landing[name] = [prop.positions @ R for (prop,) in props]            # back to the original frame
        print(f"  {name} done", flush=True)

    def diff(name):
        return np.array([max_mic_untranslated(_at(proposals[k], landing["unchanged"][k]),
                                              _at(proposals[k], landing[name][k])) for k in range(len(tids))])
    repro = np.array([max_mic_untranslated(proposals[k], _at(proposals[k], landing["unchanged"][k]))
                      for k in range(len(tids))])
    rot, inv = diff("rotation 180 about z"), diff("inversion")
    cats = np.array([category(r, args.cutoff) if r else "missing" for r in records])
    sella_move = np.array([r["maxd"] if r and np.isfinite(r.get("maxd", np.nan)) else np.nan for r in records])
    natoms = np.array([len(a) for a in proposals])

    q = lambda v: "  ".join(f"{x:7.4f}" for x in np.quantile(v, [0.5, 0.9, 0.99, 1.0]))
    lines = ["Mirror test on mp20bat: is the released model inversion-equivariant?", "=" * 100,
             f"{len(tids)} starts (exact noise of the run), released schedule {s['inference']}. Largest per-atom "
             "distance (A, rigid shift removed) between the landing point of the mapped start, mapped back, and the "
             "unchanged landing point.", "",
             "map                       p50      p90      p99      max",
             f"rerun vs stored          {q(repro)}",
             f"rotation 180 (control)   {q(rot)}",
             f"inversion                {q(inv)}", "",
             f"inversion differences larger than the {args.cutoff} A cutoff: {np.mean(inv > args.cutoff) * 100:.1f}% "
             f"of landing points; larger than 1 A: {np.mean(inv > 1) * 100:.1f}%",
             f"median ratio of the inversion difference to how far Sella moved the landing point: "
             f"{np.nanmedian(inv / np.maximum(sella_move, 1e-3)):.2f}", "",
             "by experiment-2 category (cutoff %g A):" % args.cutoff,
             "category          n    inversion p50   p90     Sella move p50"]
    for c in ("real", "moved_far", "higher_order", "minimum", "not_converged"):
        m = cats == c
        if m.any():
            lines.append(f"{c:14s} {m.sum():4d}   {np.median(inv[m]):9.4f}  {np.quantile(inv[m], 0.9):7.4f}   "
                         f"{np.nanmedian(sella_move[m]):9.3f}")
    lines += ["", "by cell size:", "atoms      n    inversion p50   p90"]
    for lo, hi in [(0, 8), (8, 20), (20, 40), (40, 80), (80, 10**4)]:
        m = (natoms > lo) & (natoms <= hi)
        if m.any():
            lines.append(f"{lo + 1:3d}-{hi if hi < 10**4 else '':<4} {m.sum():4d}   {np.median(inv[m]):9.4f}  "
                         f"{np.quantile(inv[m], 0.9):7.4f}")
    text = "\n".join(lines) + "\n"
    (out / "report.txt").write_text(text)
    (out / "mirror.json").write_text(json.dumps(dict(tids=tids, rerun=repro.tolist(), rotation=rot.tolist(),
                                                     inversion=inv.tolist(), category=cats.tolist())))
    print(text)


def _at(atoms, positions):
    b = atoms.copy(); b.positions = np.asarray(positions)
    return b


if __name__ == "__main__":
    main()
