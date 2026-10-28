"""Score LiC hint models on all 63 minima with the same hint generator as the classical search (A3).

For every minimum and every hint direction d (12 evenly spaced in-plane directions plus one toward each
nearby C-C bond midpoint, as in hint_pilot.py a3), each model is rolled out once from
x_0 = minimum + step x d (no jitter), with the hint field held fixed for a model that injects one. Arms are
given as name=checkpoint; the usual set is
    A1  hint injected into the backbone           (train.py --hint-pairs-dir ... --hint-input)
    A2  hint only in the start                     (train.py --hint-pairs-dir ... --no-hint-input)
    B0  n12_uncond, untouched, from the same starts (the zero-training control)
Every endpoint is then refined by Sella (UMA-S-1.2 omat, fmax 0.005, <= 300 steps, delta0 0.1, 3x3 index).

    rollout   model rollouts for every arm (GPU, one process)
    sella     Sella from every endpoint (GPU; --shard/--nshards)
    analyze   coverage on training and held-out minima, raw (0.10 A) and after Sella (moved <= 0.2 A);
              agreement with the A3 label of the same hint; hints where Sella from the hint failed but the
              model reached a connected saddle; the long hop (10, 143); cost per hint (CPU)

    python examples/LiC/experiments/hint_model/evaluate_hint_model.py rollout --out-dir <dir> \
        --arm A1=<ckpt> --arm A2=<ckpt> --arm B0=<n12_uncond ckpt>
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LIC = HERE.parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent / "hint_pilot"), str(LIC / "rollout_eval")]
import hint_pilot as hp  # noqa: E402
from hint_pairs import DEFAULT_A3, lic_context  # noqa: E402

RAW, CAP = 0.10, 0.2          # A: raw hit radius; largest Sella move that still credits the model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["rollout", "sella", "analyze"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--arm", action="append", default=[], help="name=checkpoint directory (rollout mode)")
    p.add_argument("--step", type=float, default=0.3, help="A along the hint for x_0 (the training value)")
    p.add_argument("--K", type=int, default=20, help="Euler steps (one pass, the LiC sampler)")
    p.add_argument("--a3-dir", default=DEFAULT_A3)
    p.add_argument("--minima", default="", help="comma-separated minima to use (default: all 63); for quick tests")
    p.add_argument("--device", default="cuda")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    return p.parse_args()


def rollout(args, out, ctx):
    import torch
    from force_guided_sampling import load_flow_model
    from robustness_shells import rollout_from_step
    from saddleflow.data import atoms_to_sample_dict
    torch.backends.cuda.matmul.allow_tf32 = False
    ref, li = ctx["reference"], ctx["li"]
    sample = atoms_to_sample_dict(ref)
    rows = []
    for spec in args.arm:
        name, ckpt = spec.split("=", 1)
        _, _, backbone, attention, head = load_flow_model(Path(ckpt), args.device)
        injects = bool(getattr(backbone, "inject_hint", False))
        for m in ([int(x) for x in args.minima.split(",")] if args.minima else range(len(ctx["minima"]))):
            dirs = hp.directions(ctx, m)
            starts, hints = [], []
            for _, d in dirs:
                x = ref.positions.copy(); x[li] = ctx["minima"][m] + args.step * d
                h = np.zeros_like(x); h[li] = d
                starts.append(x); hints.append(h)
            hint = torch.as_tensor(np.array(hints), dtype=torch.float32) if injects else None
            final, _ = rollout_from_step(torch.as_tensor(np.array(starts), dtype=torch.float32), 0, sample=sample,
                                         backbone=backbone, attention=attention, velocity_head=head,
                                         device=args.device, K=args.K, hint=hint)
            for j, ((kind, d), end) in enumerate(zip(dirs, final.numpy()[:, li])):
                rows.append(dict(arm=name, minimum=m, direction=j, kind=kind, vector=d.tolist(), end=end.tolist()))
        print(f"  {name}: {sum(r['arm'] == name for r in rows)} rollouts (hint injected: {injects})", flush=True)
        del backbone, attention, head
        torch.cuda.empty_cache()
    (out / "rollouts.json").write_text(json.dumps(rows))


def sella(args, out, ctx):
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    from saddleflow.classical_saddle_search import MinimumModeEstimator, PESBackend, run_sella
    rows = json.loads((out / "rollouts.json").read_text())
    path = out / f"sella_{args.shard}.json"
    done = json.loads(path.read_text()) if path.exists() else []
    finished = {(r["arm"], r["minimum"], r["direction"]) for r in done}
    calc = FAIRChemCalculator(pretrained_mlip.get_predict_unit("uma-s-1p2", device=args.device), task_name="omat")
    li = ctx["li"]
    for r in rows[args.shard::args.nshards]:
        if (r["arm"], r["minimum"], r["direction"]) in finished:
            continue
        atoms = ctx["reference"].copy(); atoms.positions[li] = r["end"]
        pes = PESBackend(atoms, calc)
        x0 = pes.dofs.pack(atoms.positions)
        res = run_sella(pes, x0, fmax_tol=0.005, max_steps=300, max_force_calls=10 ** 6, delta0=0.1)
        pes.max_force_calls = None
        calls = pes.force_calls
        _, fmax = pes.dofs.force_metrics(pes.forces(res["final"]))
        values, _, _ = MinimumModeEstimator().lowest_modes(pes, res["final"], n_modes=2)
        final = np.asarray(res["positions"])[li]
        done.append(dict(arm=r["arm"], minimum=r["minimum"], direction=r["direction"], sella_calls=calls,
                         conv=bool(fmax <= 0.005), nneg=int(np.sum(values < -0.01)), li=final.tolist(),
                         moved=float(hp.mic_norm(final - np.array(r["end"]), ctx["cell"], ctx["pbc"])[0])))
        tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(done)); tmp.replace(path)
    print(f"shard {args.shard}: {len(done)} endpoints")


def analyze(args, out, ctx):
    rolls = {(r["arm"], r["minimum"], r["direction"]): r for r in json.loads((out / "rollouts.json").read_text())}
    sel = {(r["arm"], r["minimum"], r["direction"]): r
           for p in sorted(glob.glob(str(out / "sella_*.json"))) for r in json.loads(Path(p).read_text())}
    edges = ctx["edges"]
    train_minima = set(json.loads((Path(args.a3_dir) / "training_minima.json").read_text()))
    connected = lambda m, s: s >= 0 and m in edges[s]
    # A3 labels of the same hints (nudge0.8, first runs).
    a3 = {}
    for p in glob.glob(str(Path(args.a3_dir) / "runs_*.json")):
        for r in json.loads(Path(p).read_text()):
            if r["arm"] == "nudge0.8" and r["repeat"] == 0:
                res, s = hp.a3_outcome(r, ctx)
                a3[(r["minimum"], r["direction"])] = (s if res == "connected" else -1,
                                                      r.get("sella_calls", 0) + r.get("index_calls", 0))
    all_pairs = {(int(m), int(s)) for s in range(len(edges)) for m in edges[s]}
    groups = {"training": {p for p in all_pairs if p[0] in train_minima},
              "held-out": {p for p in all_pairs if p[0] not in train_minima}}
    # A saddle between a training and a held-out minimum was a training target from the other side; held-out
    # pairs with such a saddle are counted separately from the truly unseen ones.
    seen_saddles = {s for m, s in groups["training"]}
    unseen = {p for p in groups["held-out"] if p[1] not in seen_saddles}
    arms = sorted({k[0] for k in rolls})
    lines = ["LiC hint models: coverage, agreement with the classical search, cost", "=" * 104,
             f"Hints: 12 even + bond-midpoint directions per minimum (as in A3), x_0 = minimum + {args.step} A x hint, "
             f"one pass of {args.K} Euler steps. Raw hit: endpoint within {RAW} A of a connected catalog saddle. "
             f"After Sella: converged, index 1, catalog match {hp.MATCH} A, connected, Sella moved it <= {CAP} A.",
             f"Training minima: {len(train_minima)} ({len(groups['training'])} connected pairs); held-out: "
             f"{63 - len(train_minima)} ({len(groups['held-out'])} pairs).", "",
             "coverage of connected pairs                  raw (0.10 A)                 after Sella (<= 0.2 A)",
             "arm        hints   training   held-out    training   held-out    (10,143)"]

    def found(arm, mode):
        out_ = {}
        for (a, m, j), r in rolls.items():
            if a != arm:
                continue
            if mode == "raw":
                s, dist = hp.match(np.array(r["end"]), ctx)
                s = int(s[0]) if dist[0] < RAW and connected(m, int(s[0])) else -1
            else:
                q = sel.get((a, m, j))
                s = -1
                if q and q["conv"] and q["nneg"] == 1 and q["moved"] <= CAP:
                    k, dist = hp.match(np.array(q["li"]), ctx)
                    s = int(k[0]) if dist[0] < hp.MATCH and connected(m, int(k[0])) else -1
            out_[(m, j)] = s
        return out_

    results = {}
    for arm in arms:
        raw, refined = found(arm, "raw"), found(arm, "sella")
        results[arm] = refined
        pr = {(m, s) for (m, _), s in raw.items() if s >= 0}
        ps = {(m, s) for (m, _), s in refined.items() if s >= 0}
        lines.append(f"{arm:9s} {len(raw):6d}   {len(pr & groups['training']):4d}/{len(groups['training']):<4d} "
                     f"{len(pr & groups['held-out']):4d}/{len(groups['held-out']):<4d}   "
                     f"{len(ps & groups['training']):4d}/{len(groups['training']):<4d} "
                     f"{len(ps & groups['held-out']):4d}/{len(groups['held-out']):<4d}   "
                     f"{'yes' if (10, 143) in ps else 'no'}")
    lines += ["", f"held-out pairs whose saddle no training minimum touches ({len(unseen)} of {len(groups['held-out'])}), "
              "after Sella:"]
    for arm in arms:
        ps = {(m, s) for (m, _), s in results[arm].items() if s >= 0}
        lines.append(f"  {arm:9s} {len(ps & unseen):4d}/{len(unseen)}")
    a3_pairs = {(m, s) for (m, _), (s, _) in a3.items() if s >= 0}
    lines.append(f"{'A3 (0.8 A nudge, Sella)':24s}              "
                 f"{len(a3_pairs & groups['training']):4d}/{len(groups['training']):<4d} "
                 f"{len(a3_pairs & groups['held-out']):4d}/{len(groups['held-out']):<4d}   "
                 f"{'yes' if (10, 143) in a3_pairs else 'no'}")

    lines += ["", "per hint, after Sella (training / held-out minima):",
              "arm        connected%            same saddle as A3%    connected where A3 failed   Sella calls/hint"]
    for arm in arms:
        cells = []
        for g in (True, False):
            keys = [k for k in results[arm] if (k[0] in train_minima) == g]
            hit = [results[arm][k] >= 0 for k in keys]
            same = [results[arm][k] == a3[k][0] for k in keys if k in a3 and a3[k][0] >= 0]
            rescue = sum(results[arm][k] >= 0 for k in keys if k in a3 and a3[k][0] < 0)
            failed = sum(1 for k in keys if k in a3 and a3[k][0] < 0)
            cells.append((100 * np.mean(hit), 100 * np.mean(same), rescue, failed))
        calls = np.mean([q["sella_calls"] for (a, *_), q in sel.items() if a == arm]) if sel else float("nan")
        lines.append(f"{arm:9s}  {cells[0][0]:5.1f} / {cells[1][0]:5.1f}        {cells[0][1]:5.1f} / {cells[1][1]:5.1f}"
                     f"           {cells[0][2]}/{cells[0][3]} , {cells[1][2]}/{cells[1][3]}              "
                     f"{calls:6.1f} + 20 network calls")
    lines.append(f"A3 (0.8 A nudge): {np.mean([c for _, c in a3.values()]):.1f} force calls per hint (Sella + index)")
    text = "\n".join(lines) + "\n"
    (out / "report.txt").write_text(text)
    print(text)


def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    ctx = lic_context()
    {"rollout": rollout, "sella": sella, "analyze": analyze}[args.mode](args, out, ctx)


if __name__ == "__main__":
    main()
