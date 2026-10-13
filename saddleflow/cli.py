"""Command line: `saddleflow propose` and `saddleflow search`.

    saddleflow propose reactant.traj -o proposals.traj --n 8 --seed 0
    saddleflow search  reactant.traj -o saddles.traj   --n 8 --seed 0 --task omat

`propose` writes the flow endpoints; `search` also converges them with Sella on UMA-S-1.2 (fairchem's
calculator, gated access to facebook/UMA) and keeps the distinct first-order saddles.
"""

from __future__ import annotations

import argparse

from ase.io import read, write


def main(argv=None):
    p = argparse.ArgumentParser(prog="saddleflow", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("propose", "search"):
        s = sub.add_parser(name)
        s.add_argument("input", help="ASE-readable structure file (every frame is used)")
        s.add_argument("-o", "--output", required=True)
        s.add_argument("--n", type=int, default=8, help="noise draws per structure")
        s.add_argument("--seed", type=int, default=None)
        s.add_argument("--sigma", type=float, default=None, help="start noise (A), at most the training 0.3")
        s.add_argument("--task", default=None, help="omat (bulk), oc20 or oc22 (surfaces); default omat")
        s.add_argument("--model", default="SciLM/SaddleFlow", help="Hub repository or local directory")
        s.add_argument("--device", default=None)
        if name == "search":
            s.add_argument("--uma", default="uma-s-1p2", help="UMA model for the Sella refinement")
            s.add_argument("--fmax", type=float, default=0.01)
            s.add_argument("--steps", type=int, default=300)
    a = p.parse_args(argv)

    from .pretrained import SaddleFlow
    model = SaddleFlow.from_pretrained(a.model, device=a.device)
    frames = read(a.input, index=":")
    out = []
    if a.cmd == "propose":
        for props in model.propose(frames, n=a.n, seed=a.seed, sigma=a.sigma, task_name=a.task):
            out.extend(props)
    else:
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.calculate.pretrained_mlip import get_predict_unit

        from .search import find_saddles
        unit = get_predict_unit(a.uma, device=model.device.type)          # fairchem takes "cuda" or "cpu"
        for frame in frames:
            task = a.task or frame.info.get("task_name") or "omat"
            calc = FAIRChemCalculator(unit, task_name=task)
            out.extend(find_saddles(frame, calc, n=a.n, model=model, seed=a.seed, fmax=a.fmax, steps=a.steps,
                                    sigma=a.sigma, task_name=task))
    write(a.output, out)
    print(f"wrote {len(out)} structures to {a.output}")


if __name__ == "__main__":
    main()
