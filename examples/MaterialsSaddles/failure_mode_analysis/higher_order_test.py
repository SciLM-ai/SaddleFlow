"""Is a higher-order landing point the model averaging two saddles, and is coarse integration the cause?

Reads a failure-mode run directory (make_proposals.py + sella_eval.py). The cases are the landing points
the model moved (>= 0.05 A from the minimum, rigid shift removed) that Sella labels higher_order: Sella
barely moved them onto a stationary point H with two or more negative modes.

    neighbours  For each H and each of its two lowest modes, push H --push A each way along the mode (the
                largest per-atom move) and run Sella with the paper protocol plus the index check. If H is the
                ridge between two passes, the two pushes along one mode reach two different index-1 saddles.
                GPU; --shard/--nshards split the cases.
    rollouts    (1) 16 new draws from each case's minimum, released schedule (5 passes x 4 steps).
                (2) The ORIGINAL noise draw of each case, rerun with finer and longer integration (--schedules).
                The noise is regenerated exactly as make_proposals.py drew it, and 5x4 must reproduce the
                stored landing point. GPU.
    analyze     Label every rollout as near H, near a found saddle, or elsewhere, and report. CPU, seconds.

    python examples/MaterialsSaddles/failure_mode_analysis/higher_order_test.py neighbours <run_dir>
    python examples/MaterialsSaddles/failure_mode_analysis/higher_order_test.py rollouts <run_dir>
    python examples/MaterialsSaddles/failure_mode_analysis/higher_order_test.py analyze <run_dir>

Everything is written to <run_dir>/higher_order/.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from ase.geometry import find_mic
from ase.io import read

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_failure_modes import category, found_saddle, load, max_mic_untranslated  # noqa: E402

SCHEDULES = "5x4,5x10,5x25,5x50,1x20,1x100,10x4,20x4"   # passes x Euler steps per pass


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["neighbours", "rollouts", "analyze"])
    p.add_argument("run_dir")
    p.add_argument("--cutoff", type=float, default=0.2, help="A; the run's 'real' cutoff and the label radius")
    p.add_argument("--push", type=float, default=0.1, help="A, largest per-atom move along a mode")
    p.add_argument("--draws", type=int, default=16)
    p.add_argument("--draw-seed", type=int, default=1, help="Seed of the new draws (the run used --seed 0)")
    p.add_argument("--schedules", default=SCHEDULES)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--task-name", default="omat")
    return p.parse_args()


def cases(run_dir, cutoff):
    """Higher-order landing points the model moved: (tid, proposal, minimum, H as Atoms)."""
    selection, proposals, minima, _, records, _, _, _ = load(run_dir)
    out = []
    for p, m, r in zip(proposals, minima, records):
        if r and category(r, cutoff) == "higher_order" and max_mic_untranslated(p, m) >= 0.05:
            out.append((int(p.info["tid"]), p, m, found_saddle(p, r)))
    return selection, out


def distance(a, b_positions):
    """Max per-atom MIC distance, rigid shift removed, from Atoms a to positions b."""
    b = a.copy(); b.positions = np.asarray(b_positions)
    return max_mic_untranslated(a, b)


def neighbours(args, out):
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    from saddleflow.classical_saddle_search import MinimumModeEstimator, PESBackend, run_sella

    unit = pretrained_mlip.get_predict_unit("uma-s-1p2", device="cuda")
    _, todo = cases(args.run_dir, args.cutoff)
    rows = []
    for tid, _, _, H in todo[args.shard::args.nshards]:
        pes = PESBackend(H.copy(), FAIRChemCalculator(unit, task_name=args.task_name))
        x = pes.dofs.pack(H.positions)
        values, vectors, _ = MinimumModeEstimator(lanczos_maxiter=300).lowest_modes(pes, x, n_modes=2)
        row = dict(tid=tid, eigs=values.tolist(), energy=pes.energy(x), modes=[])
        for k in range(2):
            v = vectors[:, k].reshape(-1, 3)
            v = args.push * v / np.linalg.norm(v, axis=1).max()
            ends = []
            for sign in (+1, -1):
                try:
                    result = run_sella(pes, x + sign * v.ravel(), fmax_tol=0.01, max_steps=300,
                                       max_force_calls=1_000_000, delta0=0.1)
                    pes.max_force_calls = None
                    _, fmax = pes.dofs.force_metrics(pes.forces(result["final"]))
                    eigs = MinimumModeEstimator(lanczos_maxiter=300).lowest_modes(pes, result["final"], n_modes=4)[0]
                    ends.append(dict(conv=bool(fmax < 0.01), fmax=fmax, nneg=int(np.sum(eigs < -0.01)),
                                     eigs=eigs.tolist(), energy=pes.energy(result["final"]),
                                     pos=result["positions"].tolist()))
                except Exception as exc:    # one bad case must not lose the shard
                    ends.append(dict(conv=False, nneg=None, err=str(exc)[:200]))
            row["modes"].append(ends)
        rows.append(row)
        (out / f"neighbours_{args.shard}.json").write_text(json.dumps(rows))
        print(f"  tid {tid}: eigs {np.round(values, 3)}; mode ends nneg "
              f"{[[e.get('nneg') for e in ends] for ends in row['modes']]}", flush=True)


def original_starts(run_dir, settings):
    """Every start of a make_proposals.py run, {tid: Atoms}, with its exact noise: one generator seeded
    with the run's seed, one draw per minimum in selection order, scaled by the inference sigma."""
    import torch

    gen = torch.Generator().manual_seed(settings["seed"])
    sigma = float(settings["inference"]["sigma"])
    starts = {}
    for m in read(Path(run_dir) / "minima.traj", index=":"):
        noise = torch.randn(m.positions.shape, generator=gen)
        fixed = np.zeros(len(m), bool)
        for c in m.constraints:
            fixed[c.get_indices()] = True
        pos = torch.as_tensor(m.positions, dtype=torch.float32)
        start = m.copy()
        start.set_positions((pos + sigma * noise * torch.as_tensor(~fixed).float().unsqueeze(1)).numpy().astype(float),
                            apply_constraint=False)
        starts[int(m.info["tid"])] = start
    return starts


def rollouts(args, out):
    from saddleflow import SaddleFlow

    selection, todo = cases(args.run_dir, args.cutoff)
    s = selection["settings"]
    sf = SaddleFlow.from_pretrained(s["model"], device="cuda")
    starts = original_starts(args.run_dir, s)
    picked = [starts[tid] for tid, *_ in todo]
    result = dict(tids=[tid for tid, *_ in todo], schedules={}, draws=[])
    for schedule in args.schedules.split(","):
        passes, steps = map(int, schedule.split("x"))
        props = sf.propose(picked, n=1, sigma=0.0, passes=passes, steps=steps, task_name=args.task_name,
                           return_passes=True)
        result["schedules"][schedule] = [prop.info["passes"].tolist() for (prop,) in props]
        print(f"  schedule {schedule} done", flush=True)
    reproduced = [distance(p, np.array(result["schedules"]["5x4"][k][-1])) if "5x4" in result["schedules"] else np.nan
                  for k, (_, p, _, _) in enumerate(todo)]
    result["reproduction_max_error"] = float(np.nanmax(reproduced))
    print(f"  5x4 reproduces the stored landing points to {result['reproduction_max_error']:.2e} A", flush=True)
    props = sf.propose([m for _, _, m, _ in todo], n=args.draws, seed=args.draw_seed, task_name=args.task_name)
    result["draws"] = [[d.positions.tolist() for d in per] for per in props]
    (out / "rollouts.json").write_text(json.dumps(result))


def analyze(args, out):
    _, todo = cases(args.run_dir, args.cutoff)
    by_tid = {r["tid"]: r for path in sorted(glob.glob(str(out / "neighbours_*.json")))
              for r in json.loads(Path(path).read_text())}
    roll = json.loads((out / "rollouts.json").read_text()) if (out / "rollouts.json").exists() else None
    cut = args.cutoff
    lines = [f"Higher-order landing points: averaging and integration test ({len(todo)} cases, "
             f"neighbour results for {len(by_tid)})", "=" * 100,
             f"H = Sella's stationary point next to the landing point. Pushes of {args.push} A along each of H's two "
             f"lowest modes; Sella fmax 0.01, index-1 = one eigenvalue < -0.01. Labels within {cut} A.", ""]

    # Neighbours: which pushes reach two different index-1 saddles, and where H sits between them.
    refs, n_split = {}, 0
    lines.append(" tid   eig1    eig2  | mode  ends' nneg  splits  |S+ - S-|  position of H  off-line     dE")
    for tid, _, _, H in todo:
        r = by_tid.get(tid)
        if r is None:
            continue
        refs[tid], split = [], False
        for k, ends in enumerate(r["modes"]):
            ok = [e for e in ends if e.get("conv") and e.get("nneg") == 1]
            refs[tid] += [e["pos"] for e in ok]
            head = f"{tid:4d}  {r['eigs'][0]:6.3f}  {r['eigs'][1]:6.3f}" if k == 0 else " " * 20
            text = f"{head}  | {k + 1:4d}  {str([e.get('nneg') for e in ends]):10s}"
            if len(ok) == 2 and max_mic_untranslated(_with(H, ok[0]["pos"]), _with(H, ok[1]["pos"])) > cut:
                split = True
                plus, minus = (_vector(H, e["pos"]) for e in ok)
                u = plus - minus
                w = float(np.dot(-minus.ravel(), u.ravel()) / np.dot(u.ravel(), u.ravel()))
                off = float(np.linalg.norm(-minus - w * u, axis=1).max())
                text += (f"  yes     {np.linalg.norm(u, axis=1).max():8.2f}  {w:13.2f}  {off:8.2f}  "
                         f"{abs(ok[0]['energy'] - ok[1]['energy']):6.3f}")
            else:
                text += "  no"
            lines.append(text)
        n_split += split
    lines += ["", f"cases where at least one mode splits into two different index-1 saddles: {n_split}/{len(by_tid)}",
              "position of H: 0 = at S-, 1 = at S+, 0.5 = midpoint. off-line: largest per-atom distance from the "
              "S- -> S+ line (A). dE: |E(S+) - E(S-)| (eV).", ""]

    if roll is None:
        lines.append("rollouts.json missing; run the rollouts mode")
    else:
        def label(tid, H, positions):
            if distance(H, positions) < cut:
                return "H"
            return "saddle" if any(max_mic_untranslated(_with(H, positions), _with(H, s)) < cut
                                   for s in refs.get(tid, [])) else "elsewhere"
        cases_by_tid = {tid: H for tid, _, _, H in todo}
        lines.append(f"5x4 rerun reproduces the stored landing points to {roll['reproduction_max_error']:.1e} A")
        lines += ["", f"(1) {len(roll['draws'][0])} new draws per minimum (released 5x4): share landing on",
                  "     H       a found saddle   elsewhere"]
        counts = np.zeros(3)
        for tid, draws in zip(roll["tids"], roll["draws"]):
            labs = [label(tid, cases_by_tid[tid], d) for d in draws]
            counts += [labs.count("H"), labs.count("saddle"), labs.count("elsewhere")]
        counts /= counts.sum()
        lines.append(f"   {100 * counts[0]:5.1f}%      {100 * counts[1]:5.1f}%       {100 * counts[2]:5.1f}%")
        lines += ["", "(2) the original noise draw, other integration schedules (passes x steps per pass):",
                  "  schedule  network calls   on H   on a found saddle   elsewhere   median distance to H (A)"]
        for schedule, traces in roll["schedules"].items():
            passes, steps = map(int, schedule.split("x"))
            labs = [label(tid, cases_by_tid[tid], tr[-1]) for tid, tr in zip(roll["tids"], traces)]
            dH = [distance(cases_by_tid[tid], tr[-1]) for tid, tr in zip(roll["tids"], traces)]
            n = len(labs)
            lines.append(f"  {schedule:8s}  {passes * steps:13d}  {labs.count('H'):5d}  {labs.count('saddle'):17d}  "
                         f"{labs.count('elsewhere'):10d}   {np.median(dH):8.3f}")
        lines += ["", "(3) share on H after each pass (original noise; long-pass schedules):"]
        for schedule, traces in roll["schedules"].items():
            if int(schedule.split("x")[0]) >= 5:
                share = [np.mean([distance(cases_by_tid[tid], tr[k]) < cut for tid, tr in zip(roll["tids"], traces)])
                         for k in range(len(traces[0]))]
                lines.append(f"  {schedule:8s} " + " ".join(f"{100 * x:4.0f}" for x in share) + "   (%, pass 0 = start)")
    text = "\n".join(lines) + "\n"
    (out / "report.txt").write_text(text)
    print(text)


def _with(atoms, positions):
    b = atoms.copy(); b.positions = np.asarray(positions)
    return b


def _vector(H, positions):
    """MIC displacement from H to positions, rigid shift removed (cells without fixed atoms)."""
    d = find_mic(np.asarray(positions) - H.positions, H.cell, H.pbc)[0]
    return d - d.mean(0) if not H.constraints else d


def main():
    args = parse_args()
    out = Path(args.run_dir) / "higher_order"
    out.mkdir(exist_ok=True)
    {"neighbours": neighbours, "rollouts": rollouts, "analyze": analyze}[args.mode](args, out)


if __name__ == "__main__":
    main()
