"""Does UMA carry handedness (pseudotensor) features, and does its mirror-correct output rely on them?

    python examples/MaterialsSaddles/experiments/pseudotensor_probe/pseudotensor_probe.py --model uma --dataset <oc20/test>
    python examples/MaterialsSaddles/experiments/pseudotensor_probe/pseudotensor_probe.py --model fm2 --dataset <oc22/test>

--model uma is stock UMA-M-1.1 (energy and forces, oc20 task); --model fm2 is SaddleFlow FM2 (velocity). Forward passes
only, no training. Every input x runs in one batch with its inverted copy -x (positions negated and wrapped into the same
cell; atom order, cell, constraints and task unchanged). In a mirror-correct network the node features after every block
obey h(-x) = P h(x), with P = (-1)^l per coefficient, the velocity v(-x) = -v(x), the energy E(-x) = E(x) and the forces
F(-x) = -F(x).

1. Odd fractions. h_odd = (h(x) - P h(-x)) / 2 is the handedness stored on the atoms; reported as ||h_odd|| / ||h|| per
   block, per degree l and per channel (channels in odd_fractions.npz). The odd part of each edge message is
   m_odd = (m_W2(z) - m_-W2(z)) / 2 on the same input z: negating W2 is the mirror through a plane holding the edge, so
   this is the part of the message that is not mirror-correct. The run checks this identity on the whole model:
   output_-W2(x) must equal the inverted output of -x, up to a small fraction of the mirror error. Noise floor: the same measurement with W2 zeroed (mirror-correct
   by construction; saddleflow.utils.parity), restored afterwards.
2. Outputs. v_even = (v(x) - v(-x)) / 2 and v_odd = (v(x) + v(-x)) / 2; for FM2 the error against the training target T
   on x (||v(x) - T||) and on -x (||v(-x) + T||). For UMA the energy and force mirror errors.
3. Ablation. After block k (each block, then every block at once) the features of both copies are replaced by their even
   part: h(x) -> h_even(x), h(-x) -> P h_even(x). Reported: the change in the mirror-correct output (v_even; E_even for
   UMA) and, for FM2, in the error on x. Control: subtract instead a random even perturbation with the same norm per
   atom and per degree l as the odd part, leaving the odd part in place.

Inputs. FM2 (oc22): x0 = E + u (S - E) + eps (u = 0 half the time, else U(0, 1); eps ~ N(0, 0.3^2) on every mobile
atom, all of them flagged), at each --times t: x_t = (1 - t) x0 + t S with the convergent target
T = (S - x_t) / (1 - t) (S - x0 above t = 0.9); plus each saddle at t = 0 (where a restart pass begins), target 0.
No x_t perturbation. UMA (oc20): each record's minimum, its saddle, and the minimum displaced by N(0, 0.1^2) A on its
mobile atoms. Velocity norms are RMS over mobile atoms; force norms RMS over all atoms.

Writes <out-dir>/report.txt, cases.jsonl (one row per input) and odd_fractions.npz.
"""
import argparse
import contextlib
import json
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
from fairchem.core.datasets.collaters.simple_collater import data_list_collater

from saddleflow.data.materials_saddles_dataset import MaterialsSaddlesDataset
from saddleflow.data.transforms import wrap_positions
from saddleflow.flow.matching import apply_output_projections, build_atomic_data
from saddleflow.utils.parity import _w2, parity_sign, so2_layers

SIGMA_FM2 = 0.3       # A: FM2's training noise
SIGMA_UMA = 0.1       # A: displacement of the "displaced" UMA inputs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=["uma", "fm2"])
    p.add_argument("--dataset", required=True, help="directory of MaterialsSaddles .aselmdb shards")
    p.add_argument("--n-records", type=int, default=64, help="records drawn at random with the seed")
    p.add_argument("--times", default="0,0.5,0.9", help="FM2 flow times")
    p.add_argument("--batch-size", type=int, default=4, help="inputs per forward pass (each with its inverted copy)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


# ---------------------------------------------------------------- W2
@contextlib.contextmanager
def w2_scaled(module, factor):
    """W2 of every m > 0 SO(2) layer multiplied by `factor` (0: mirror-correct; -1: mirrored), restored exactly after."""
    saved = []
    with torch.no_grad():
        for _, conv in so2_layers(module):
            w, i = _w2(conv)
            saved.append((w, i, w[i].clone()))
            w[i] = factor * w[i]
    try:
        yield
    finally:
        with torch.no_grad():
            for w, i, old in saved:
                w[i] = old


# ---------------------------------------------------------------- hooks
class Probe:
    """Forward hooks on every backbone block (odd-fraction statistics, ablation) and on every message (odd part)."""

    def __init__(self, backbone, device):
        self.blocks = list(backbone.blocks)
        self.P = parity_sign(backbone.lmax).to(device)[None, :, None]
        self.lidx = torch.tensor([l for l in range(backbone.lmax + 1) for _ in range(2 * l + 1)], device=device)
        self.L = backbone.lmax + 1
        self.n = None              # atoms in the original half of the batch (the inverted copies follow, same order)
        self.sys = None            # (n,) system of each original atom
        self.stats = None          # dict to accumulate into, or None
        self.messages = False      # measure the messages' odd parts too
        self.edit = None           # ("odd" | "control", set of blocks)
        self.gen = None
        self._z, self._busy = {}, False
        for k, block in enumerate(self.blocks):
            block.register_forward_hook(partial(self._block, k))
            block.edge_wise.so2_conv_1.register_forward_pre_hook(partial(self._message_input, k))
            block.edge_wise.so2_conv_2.register_forward_hook(partial(self._message, k))

    def per_l(self, sq):
        """(atoms, coefficients, channels) squares -> (atoms, L) sums."""
        return torch.zeros(sq.shape[0], self.L, device=sq.device, dtype=sq.dtype).index_add_(1, self.lidx, sq.sum(-1))

    def _block(self, k, module, inputs, out):
        if self.stats is None and (self.edit is None or k not in self.edit[1]):
            return None
        n = self.n
        hx, hi = out[:n], out[n:]
        odd = (hx - self.P * hi) / 2
        if self.stats is not None:          # UMA's backbone turns gradients on internally (for forces)
            s, od, hd = self.stats, odd.detach(), hx.detach()
            s["atom_odd"][k] += (od ** 2).sum(0).double().cpu().numpy()
            s["atom_tot"][k] += (hd ** 2).sum(0).double().cpu().numpy()
            B = int(self.sys.max()) + 1
            s["case_odd"].append(torch.zeros(B, device=out.device).index_add_(0, self.sys, (od ** 2).sum((1, 2))))
            s["case_tot"].append(torch.zeros(B, device=out.device).index_add_(0, self.sys, (hd ** 2).sum((1, 2))))
        if self.edit is None or k not in self.edit[1]:
            return None
        if self.edit[0] == "odd":
            even = (hx + self.P * hi) / 2
            return torch.cat([even, self.P * even])
        g = torch.randn(hx.shape, generator=self.gen, device=hx.device, dtype=hx.dtype)
        scale = (self.per_l(odd ** 2) / self.per_l(g ** 2).clamp_min(1e-30)).sqrt()
        delta = g * scale[:, self.lidx, None]
        return torch.cat([hx - delta, hi - self.P * delta])

    def _message_input(self, k, module, args):
        if not self._busy:
            self._z[k] = args

    def _message(self, k, module, inputs, out):
        if not self.messages or self._busy:
            return None
        edge_wise = self.blocks[k].edge_wise
        self._busy = True
        try:
            with torch.no_grad(), w2_scaled(edge_wise, -1.0):
                msg, gating = edge_wise.so2_conv_1(*self._z[k])
                mirrored = edge_wise.so2_conv_2(edge_wise.act(gating, msg))
        finally:
            self._busy = False
        out = out.detach()
        self.stats["msg_odd"][k] += float((((out - mirrored) / 2) ** 2).sum())
        self.stats["msg_tot"][k] += float((out ** 2).sum())
        return None


# ---------------------------------------------------------------- models
class Model:
    """FM2 (velocity) or stock UMA-M (energy, forces), with the batch layout [x_1 .. x_B, -x_1 .. -x_B]."""

    def __init__(self, name, device):
        self.name, self.device = name, device
        if name == "fm2":
            from saddleflow import SaddleFlow
            self.net = SaddleFlow.from_pretrained(revision="fm2", device=device)
            self.backbone = self.net.backbone.backbone
            self.task_map = self.net.config["tasks"]["map"]
        else:
            from fairchem.core.calculate.pretrained_mlip import get_predict_unit
            self.net = get_predict_unit("uma-m-1p1", device=device).model.module.to(device).eval()
            self.backbone = self.net.backbone
            self.task_map = {"oc20": "oc20", "oc22": "oc20", "omat": "omat"}
        self.probe = Probe(self.backbone, device)

    def batch(self, cases):
        pos = [wrap_positions(c["x"], c["cell"]) for c in cases] + [wrap_positions(-c["x"], c["cell"]) for c in cases]
        both = cases + cases
        data = [build_atomic_data(p, c["Z"], c["cell"], self.task_map[c["task_name"]], c["charge"], c["spin"], c["fixed"])
                for p, c in zip(pos, both)]
        b = data_list_collater(data, otf_graph=True).to(self.device)
        self.probe.n = sum(len(c["Z"]) for c in cases)
        self.probe.sys = b.batch[:self.probe.n]
        return b, both

    def features(self, cases):
        """Backbone only, no gradients (UMA's statistics pass)."""
        b, _ = self.batch(cases)
        with torch.no_grad():
            self.backbone(b)

    def outputs(self, cases):
        """FM2: {"v": (2n, 3)}; UMA: {"E": (2B,), "F": (2n, 3)}."""
        b, both = self.batch(cases)
        if self.name == "fm2":
            t = torch.tensor([c["t"] for c in both], dtype=torch.float32, device=self.device)
            flag = torch.cat([~c["fixed"] for c in both]).float().to(self.device)
            fixed = torch.cat([c["fixed"] for c in both]).to(self.device)
            with torch.no_grad():
                h = self.net.backbone(b, t, b.batch, flag=flag)["node_embedding"]
                v = self.net.velocity_head(h, t, b.batch)
                v = apply_output_projections(v, fixed, b.batch, len(both))
            return {"v": v.double().cpu()}
        with torch.enable_grad():
            out = self.net(b)
        return {"E": find(out, "energy").detach().double().cpu().reshape(-1),
                "F": find(out, "forces").detach().double().cpu()}


def find(out, key):
    if isinstance(out.get(key), torch.Tensor):
        return out[key]
    for v in out.values():
        if isinstance(v, dict):
            with contextlib.suppress(KeyError):
                return find(v, key)
    raise KeyError(key)


# ---------------------------------------------------------------- inputs
def make_cases(args, rng):
    ds = MaterialsSaddlesDataset(sorted(str(p) for p in Path(args.dataset).glob("*.aselmdb")))
    picks = rng.permutation(len(ds))[:args.n_records]
    times = [float(t) for t in args.times.split(",")]
    cases = defaultdict(list)
    for i in picks:
        r = ds[int(i)]
        base = dict(record=int(i), Z=r["Z"], cell=r["cell"].float(), fixed=r["fixed"].bool(),
                    task_name=r["task_name"], charge=int(r["charge"]), spin=int(r["spin"]))
        E, S = r["start_pos"].double(), r["saddle_un_pos"].double()
        mobile = (~base["fixed"]).double()[:, None]
        if args.model == "fm2":
            u = 0.0 if rng.random() < 0.5 else rng.random()
            x0 = E + u * (S - E) + SIGMA_FM2 * torch.as_tensor(rng.normal(size=E.shape)) * mobile
            for t in times:
                xt = (1 - t) * x0 + t * S
                T = (S - xt) / (1 - t) if t <= 0.9 else S - x0
                cases[f"t={t:g}"].append(dict(base, kind=f"t={t:g}", t=t, x=xt.float(), T=T))
            cases["saddle"].append(dict(base, kind="saddle", t=0.0, x=S.float(), T=torch.zeros_like(S)))
        else:
            displaced = E + SIGMA_UMA * torch.as_tensor(rng.normal(size=E.shape)) * mobile
            for kind, x in (("minimum", E), ("saddle", S), ("displaced", displaced)):
                cases[kind].append(dict(base, kind=kind, x=x.float()))
    return cases


# ---------------------------------------------------------------- per-case numbers
def rms(a, mask=None):
    a = a if mask is None else a[mask]
    return float(a.pow(2).sum(-1).mean().sqrt()) if len(a) else float("nan")


def split(o, cases):
    """Per input: (original-copy output, inverted-copy output)."""
    B = len(cases)
    if "v" in o:
        parts = torch.split(o["v"], [len(c["Z"]) for c in cases] * 2)
        return [(parts[i], parts[B + i]) for i in range(B)]
    F = torch.split(o["F"], [len(c["Z"]) for c in cases] * 2)
    return [((o["E"][i], F[i]), (o["E"][B + i], F[B + i])) for i in range(B)]


def output_numbers(model_name, c, x, inv):
    """Mirror-correct part, mirror error and (FM2) errors against the target, for one input."""
    if model_name == "fm2":
        m = ~c["fixed"]
        even, odd = (x - inv) / 2, (x + inv) / 2
        return dict(v=rms(x, m), v_even=rms(even, m), v_odd=rms(odd, m), T=rms(c["T"], m),
                    err_x=rms(x - c["T"], m), err_inv=rms(inv + c["T"], m), err_even=rms(even - c["T"], m)), even
    (Ex, Fx), (Ei, Fi) = x, inv
    N = len(c["Z"])
    return dict(E_odd=abs(float(Ex - Ei)) / 2 / N, F=rms(Fx), F_odd=rms((Fx + Fi) / 2), F_even=rms((Fx - Fi) / 2)), \
        float(Ex + Ei) / 2


def ablation_numbers(model_name, c, row, even0, x, inv):
    if model_name == "fm2":
        m = ~c["fixed"]
        even = (x - inv) / 2
        return dict(d_even=rms(even - even0, m), err_x=rms(x - c["T"], m), v_odd=rms((x + inv) / 2, m))
    (Ex, _), (Ei, _) = x, inv
    N = len(c["Z"])
    return dict(d_even=abs(float(Ex + Ei) / 2 - even0) / N, E_odd=abs(float(Ex - Ei)) / 2 / N)


def identity_error(model_name, plain, mirrored_x):
    """How far output_-W2(x) is from the inverted output of -x, as a fraction of the mirror error itself (the distance
    of output(x) from it). Near zero if negating W2 is the mirror conjugation."""
    x, inv = plain
    if model_name == "fm2":
        return rms(mirrored_x + inv) / max(rms(x + inv), 1e-30)
    (Ex, Fx), (Ei, Fi), (Em, Fm) = x, inv, mirrored_x
    return max(abs(float(Em - Ei)) / max(abs(float(Ex - Ei)), 1e-30), rms(Fm + Fi) / max(rms(Fx + Fi), 1e-30))


# ---------------------------------------------------------------- run
def run(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    cases = make_cases(args, rng)
    model = Model(args.model, device)
    probe = model.probe
    probe.gen = torch.Generator(device=device).manual_seed(args.seed)
    nb, C = len(probe.blocks), model.backbone.sphere_channels
    K = (model.backbone.lmax + 1) ** 2
    edits = [(str(k), {k}) for k in range(nb)] + [("all", set(range(nb)))]

    def fresh():
        return dict(atom_odd=np.zeros((nb, K, C)), atom_tot=np.zeros((nb, K, C)), msg_odd=np.zeros(nb),
                    msg_tot=np.zeros(nb), case_odd=[], case_tot=[])

    stats, rows, t0 = {}, [], time.time()
    for kind, group in cases.items():
        stats[kind], stats[kind + "/floor"] = fresh(), fresh()
        for s in range(0, len(group), args.batch_size):
            chunk = group[s:s + args.batch_size]

            def measured(st, messages):
                st_len = len(st["case_odd"])
                probe.stats, probe.messages = st, messages
                try:
                    o = model.outputs(chunk) if args.model == "fm2" else model.features(chunk)
                finally:
                    probe.stats, probe.messages = None, False
                frac = (torch.stack(st["case_odd"][st_len:]) / torch.stack(st["case_tot"][st_len:])).sqrt()
                del st["case_odd"][st_len:], st["case_tot"][st_len:]
                return o, frac.cpu().numpy()          # (blocks, B)

            o, frac = measured(stats[kind], True)
            if o is None:
                o = model.outputs(chunk)
            with w2_scaled(model.backbone, 0.0):
                of, frac_floor = measured(stats[kind + "/floor"], False)
                if of is None:
                    of = model.outputs(chunk)
            with w2_scaled(model.backbone, -1.0):
                om = model.outputs(chunk)

            plain, floor, mirrored = split(o, chunk), split(of, chunk), split(om, chunk)
            chunk_rows = []
            for i, c in enumerate(chunk):
                row, even0 = output_numbers(args.model, c, *plain[i])
                row_floor, _ = output_numbers(args.model, c, *floor[i])
                chunk_rows.append(dict(kind=kind, record=c["record"], n_atoms=len(c["Z"]),
                                       n_mobile=int((~c["fixed"]).sum()), **row,
                                       floor={k: v for k, v in row_floor.items() if "odd" in k},
                                       identity=identity_error(args.model, plain[i], mirrored[i][0]),
                                       atom_odd=frac[:, i].tolist(), atom_odd_floor=frac_floor[:, i].tolist(),
                                       ablation={}, control={}, _even0=even0))
            for how in ("odd", "control"):
                for name, blocks in edits:
                    probe.edit = (how, blocks)
                    try:
                        oe = split(model.outputs(chunk), chunk)
                    finally:
                        probe.edit = None
                    for i, c in enumerate(chunk):
                        r = chunk_rows[i]
                        r["ablation" if how == "odd" else "control"][name] = \
                            ablation_numbers(args.model, c, r, r["_even0"], *oe[i])
            for r in chunk_rows:
                del r["_even0"]
            rows += chunk_rows
            print(f"{kind}: {s + len(chunk)}/{len(group)}  ({time.time() - t0:.0f}s)", flush=True)
    return model, stats, rows


# ---------------------------------------------------------------- report
def med(xs):
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.median(xs)) if xs else float("nan")


def report(args, model, stats, rows):
    out = Path(args.out_dir)
    nb, L = len(model.probe.blocks), model.probe.L
    lidx = model.probe.lidx.cpu().numpy()
    fm2 = args.model == "fm2"
    arrays = {}
    lines = [f"Pseudotensor probe: {'SaddleFlow FM2' if fm2 else 'stock UMA-M-1.1'} on {args.dataset}",
             f"{args.n_records} records, seed {args.seed}; {len(rows)} inputs", "=" * 100, ""]
    worst = max(r["identity"] for r in rows)
    lines += [f"Check: output with W2 negated on x vs the inverted output of -x, as a fraction of the mirror error: "
              f"median {med([r['identity'] for r in rows]):.1e}, worst {worst:.1e} (near zero: negating W2 is the "
              "mirror conjugation, so the message odd parts measure the right thing).", ""]

    for kind in [k for k in stats if not k.endswith("/floor")]:
        s, f = stats[kind], stats[kind + "/floor"]
        kr = [r for r in rows if r["kind"] == kind]

        def by_l(a):
            out_ = np.zeros((nb, L, a.shape[-1]))
            np.add.at(out_, (slice(None), lidx), a)
            return out_
        odd_l, tot_l = by_l(s["atom_odd"]), by_l(s["atom_tot"])                # (blocks, L, channels)
        arrays[f"{kind}/odd_fraction_l_channel"] = np.sqrt(odd_l / np.maximum(tot_l, 1e-300))
        arrays[f"{kind}/floor_fraction_l_channel"] = np.sqrt(by_l(f["atom_odd"]) / np.maximum(by_l(f["atom_tot"]), 1e-300))
        frac = np.sqrt(s["atom_odd"].sum((1, 2)) / s["atom_tot"].sum((1, 2)))
        floor = np.sqrt(f["atom_odd"].sum((1, 2)) / f["atom_tot"].sum((1, 2)))
        msg = np.sqrt(s["msg_odd"] / s["msg_tot"])
        per_l = np.sqrt(odd_l.sum(-1) / tot_l.sum(-1))                          # (blocks, L)
        l0_max = arrays[f"{kind}/odd_fraction_l_channel"][:, 0].max(-1)

        lines += [f"[{kind}]  {len(kr)} inputs", "",
                  "1. Odd fractions ||odd|| / ||total|| (all atoms and edges pooled; median over inputs in brackets)",
                  "  block   atoms              floor (W2 = 0)   messages | by degree l: " +
                  " ".join(f"{'l=' + str(l):>7s}" for l in range(L)) + " | l=0 max channel"]
        for k in range(nb):
            lines.append(f"  {k:5d}   {frac[k]:.3e} [{med([r['atom_odd'][k] for r in kr]):.2e}]  {floor[k]:.2e}"
                         f"         {msg[k]:.3e} |             " + " ".join(f"{x:7.4f}" for x in per_l[k]) +
                         f" | {l0_max[k]:.4f}")
        lines.append("")
        if fm2:
            lines += ["2. Outputs (RMS over mobile atoms, A per unit time; median over inputs)",
                      f"  ||v|| {med([r['v'] for r in kr]):.4f}   ||v_even|| {med([r['v_even'] for r in kr]):.4f}   "
                      f"||v_odd|| {med([r['v_odd'] for r in kr]):.4f} (floor {med([r['floor']['v_odd'] for r in kr]):.1e})"
                      f"   ||T|| {med([r['T'] for r in kr]):.4f}",
                      f"  error on x ||v(x) - T|| {med([r['err_x'] for r in kr]):.4f}   on -x ||v(-x) + T|| "
                      f"{med([r['err_inv'] for r in kr]):.4f}   of v_even ||v_even - T|| {med([r['err_even'] for r in kr]):.4f}",
                      "", "3. Ablation: odd part removed after block k vs the matched-size even control (medians)",
                      "  block   ||dv_even||/||v_even||     ||dv_even||/||v(x)-T||    error on x, change   ||v_odd|| after",
                      "          removed    control         removed    control        removed    control    removed"]
            for name, _ in [(str(k), 0) for k in range(nb)] + [("all", 0)]:
                a = [r["ablation"][name] for r in kr]
                ctl = [r["control"][name] for r in kr]
                lines.append(
                    f"  {name:5s}   {med([x['d_even'] / r['v_even'] for x, r in zip(a, kr)]):8.4f}   "
                    f"{med([x['d_even'] / r['v_even'] for x, r in zip(ctl, kr)]):8.4f}        "
                    f"{med([x['d_even'] / r['err_x'] for x, r in zip(a, kr)]):8.4f}   "
                    f"{med([x['d_even'] / r['err_x'] for x, r in zip(ctl, kr)]):8.4f}      "
                    f"{med([x['err_x'] / r['err_x'] - 1 for x, r in zip(a, kr)]):+8.4f}   "
                    f"{med([x['err_x'] / r['err_x'] - 1 for x, r in zip(ctl, kr)]):+8.4f}   "
                    f"{med([x['v_odd'] for x in a]):8.4f}")
        else:
            lines += ["2. Outputs (median over inputs)",
                      f"  |E_odd| {1e3 * med([r['E_odd'] for r in kr]):.4f} meV/atom (floor "
                      f"{1e3 * med([r['floor']['E_odd'] for r in kr]):.1e})   ||F|| {med([r['F'] for r in kr]):.4f} eV/A   "
                      f"||F_odd|| {med([r['F_odd'] for r in kr]):.5f} (floor {med([r['floor']['F_odd'] for r in kr]):.1e})"
                      f"   ||F_odd||/||F|| {med([r['F_odd'] / r['F'] for r in kr]):.2e}",
                      "", "3. Ablation: odd part removed after block k vs the matched-size even control (medians)",
                      "  block   |dE_even| meV/atom        |E_odd| after", "          removed    control        removed"]
            for name in [str(k) for k in range(nb)] + ["all"]:
                a = [r["ablation"][name] for r in kr]
                ctl = [r["control"][name] for r in kr]
                lines.append(f"  {name:5s}   {1e3 * med([x['d_even'] for x in a]):8.4f}   "
                             f"{1e3 * med([x['d_even'] for x in ctl]):8.4f}       {1e3 * med([x['E_odd'] for x in a]):8.4f}")
        lines.append("")

    text = "\n".join(lines) + "\n"
    (out / "report.txt").write_text(text)
    np.savez(out / "odd_fractions.npz", **arrays)
    with open(out / "cases.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(text)


def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    model, stats, rows = run(args)
    report(args, model, stats, rows)


if __name__ == "__main__":
    main()
