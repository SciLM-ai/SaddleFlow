"""Is a model mirror-equivariant? Inversion test for the released model or any hint fine-tune checkpoint.

Every start is rolled out three ways: unchanged, rotated 180 degrees about z (the control: the backbone is
rotation-equivariant by construction), and inverted (r -> -r; every mirror is an inversion times a rotation, and
inversion maps every lattice onto itself). The start positions, their noise and the hint are all mapped, the
landing is mapped back and compared with the unchanged one: largest per-atom distance, rigid shift removed
(hint_labels.saddle_distance). An equivariant model gives zero up to float error, as the rotation should.

Start sets:
    noise   the 500 starts of the failure-mode run the released model was first tested on (failure_mode_analysis/
            mirror_test.py: its exact noise), hint zero, 5 passes x 4 Euler steps (the released schedule)
    hint    --per-minimum starts of the evaluation's hint grid on each of the 24 test minima (eval_hints.starts,
            the same seeded subset for every model), hint given, 1 pass x 8 steps (ft_hint_1x8); the released
            model has no hint input and runs its own 5 x 4 from the same starts. Also: of the unchanged landings
            within 0.2 A of a reference saddle (the 30-minimum experiment's verified regions), the share whose
            mapped landing is within 0.2 A of the same one (the mirrored start finds the mirrored saddle).

    python examples/MaterialsSaddles/experiments/mirror_test/mirror_test.py --ckpt <checkpoint dir> --out-dir <dir>
    python examples/MaterialsSaddles/experiments/mirror_test/mirror_test.py --ckpt released --out-dir <dir>

Writes <out-dir>/landings.json (reused if present) and report.txt.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ase.io import read

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "hint_finetune"))
sys.path.insert(0, str(HERE.parents[1] / "failure_mode_analysis"))
import eval_hints as ev  # noqa: E402
from eval_hints import hl  # noqa: E402
from higher_order_test import original_starts  # noqa: E402

from saddleflow.data.transforms import transform_structure  # noqa: E402

MAPS = {"unchanged": np.eye(3), "rotation 180 about z": np.diag([-1.0, -1.0, 1.0]), "inversion": -np.eye(3)}
FM_RUN = f"{ev.DATA}/failure_mode_analysis_mp20bat_21421000"
NEAR = 0.2


def start_sets(args):
    """{set: [(id, key, minimum, start, hint or None, extra)]} in eval_hints' job layout."""
    run = Path(args.fm_run)
    settings = json.loads((run / "selection.json").read_text())["settings"]
    starts = original_starts(run, settings)
    noise = [(str(t), t, m, starts[t].positions, None, {}) for t, m in
             ((int(m.info["tid"]), m) for m in read(run / "minima.traj", index=":"))]
    ea = SimpleNamespace(old_dir=ev.OLD_DIR, labels=ev.LABELS, limit=0, max_hints=args.per_minimum, min_barrier=0.05)
    hint = [j for g in range(ev.N_GROUPS) for j in ev.starts(ea, g, "grid")]
    return dict(noise=noise, hint=hint), ea


def mapped(jobs, R):
    return [(i, k, transform_structure(m, R), pos @ R.T, None if h is None else h @ R.T, x) for i, k, m, pos, h, x in jobs]


def load(args):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.ckpt == "released":
        from saddleflow import SaddleFlow
        return SaddleFlow.from_pretrained("SciLM/SaddleFlow", device=args.device)
    return ev.load_finetuned(args.ckpt, args.device)


def roll(args, model, jobs, kind):
    if args.ckpt == "released":
        return ev.roll_released(model, jobs, 5, 4, args.batch_size)
    return ev.roll_finetuned(model, jobs, kind == "hint", *((1, 8) if kind == "hint" else (5, 4)), args.batch_size)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="fine-tune checkpoint directory (ema.pt; config.json one level up), or 'released'")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fm-run", default=FM_RUN, help="failure-mode run (make_proposals.py) whose starts form the noise set")
    p.add_argument("--per-minimum", type=int, default=64, help="hint starts per test minimum")
    p.add_argument("--limit", type=int, default=0, help="smoke tests: the first N starts of each set")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    sets, ea = start_sets(args)
    sets = {k: v[:args.limit or None] for k, v in sets.items()}
    path = out / "landings.json"
    if not path.exists():
        lands, model = {}, load(args)
        for kind, jobs in sets.items():
            for name, R in MAPS.items():
                X = roll(args, model, mapped(jobs, R), kind)
                lands[f"{kind}/{name}"] = [np.round(x @ R, 5).tolist() for x in X]        # back to the original frame
                print(f"  {kind} {name}: {len(X)} landings", flush=True)
        path.write_text(json.dumps(dict(ckpt=args.ckpt, landings=lands)))
    lands = {k: [np.array(x) for x in v] for k, v in json.loads(path.read_text())["landings"].items()}

    minima = {t: m for t, (m, _) in ev.test_minima(ea).items()}
    ref = ev.reference(ea, [], minima)
    q = lambda v: "  ".join(f"{x:7.4f}" for x in np.quantile(v, [0.5, 0.9, 0.99, 1.0]))
    L = [f"Mirror test: {args.ckpt}", "=" * 100, "",
         "Largest per-atom distance (A, rigid shift removed) between the mapped start's landing, mapped back, and the "
         "unchanged landing.", ""]
    for kind, jobs in sets.items():
        base = lands[f"{kind}/unchanged"]
        L += [f"{kind} starts ({len(jobs)}): " + ("failure-mode run, hint zero, 5 x 4" if kind == "noise" else
              f"{args.per_minimum} hint-grid starts per test minimum, " + ("5 x 4, no hint input" if args.ckpt == "released" else "hint given, 1 x 8")),
              f"  map                        p50      p90      p99      max |  > {NEAR} A    > 1 A" +
              ("  | same saddle" if kind == "hint" else "")]
        near = [ev.nearest_region(minima[j[1]], ref[j[1]], x, NEAR)[0] for j, x in zip(jobs, base)] if kind == "hint" else []
        for name in list(MAPS)[1:]:
            X = lands[f"{kind}/{name}"]
            d = np.array([hl.saddle_distance(j[2], x[None], y)[0] for j, x, y in zip(jobs, X, base)])
            line = f"  {name:22s} {q(d)} | {100 * np.mean(d > NEAR):6.1f}%  {100 * np.mean(d > 1):6.1f}%"
            if kind == "hint":
                kept = [ev.nearest_region(minima[j[1]], ref[j[1]], x, NEAR)[0] == k for j, x, k in zip(jobs, X, near) if k >= 0]
                line += f"  | {100 * np.mean(kept):5.1f}% of {len(kept)}"
            L.append(line)
        L.append("")
    text = "\n".join(L)
    (out / "report.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
