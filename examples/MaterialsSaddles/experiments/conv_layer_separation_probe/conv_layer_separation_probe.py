"""Where does conv_layer_separation lose or keep accuracy, and how much does it use its turn-turn path?

    python examples/MaterialsSaddles/experiments/conv_layer_separation_probe/conv_layer_separation_probe.py \
        --dataset <oc22/test> --plain <release> --separated <release> --w2 <release> --fm2 <release> --out-dir <dir>

Forward passes only, no training. Every model sees exactly the same inputs and targets.

Inputs. Records drawn at random from --dataset (held-out oc22), --draws noised starts per record, shared by every model.
Each start is the fine-tunes' training start with every mobile atom noised and flagged (as the sampler does):
x0 = E + u (S - E) + eps, u = 0 half the time and U(0, 1) otherwise, eps ~ N(0, 0.3^2) on the mobile atoms (its mean
removed when no atom is fixed). At each --times t: x_t = (1 - t) x0 + t S. For t <= 0.9, x_t is perturbed by N(0, 0.05^2)
and the target is T = (S - x_t) / (1 - t); above 0.9 the target is S - x0 with no perturbation (the training schedule,
--xt-target-correction with t_floor 0.1). The target's mobile mean is removed when no atom is fixed, as the model's.

The sampler (SaddleFlow.propose, 5 passes x 4 Euler steps) calls the network only at t = 0, 0.25, 0.5 and 0.75, and
each pass restarts at t = 0 from the previous endpoint without new noise: those are the main rows. t = 0.9 and 0.95 are
extra rows, and 0.95 (other target) compares models with each other only.

1. Velocity error by flow time. Per input, the mean over mobile atoms of |v - T| (A per unit time), and the relative
   error ||v - T|| / ||T|| over the mobile atoms. Models: plain, conv_layer_separation, the W2 = 0 run, FM2, and FM2's
   weights run with conv-layer separation (step 0 of the fine-tune). Paired differences are averaged over each record's
   draws; 95% intervals resample the start minima (records sharing a minimum are correlated); also the fraction of
   records where the first model's error is lower.

2. The turn-turn path. In conv_layer_separation, conv2's output is V1 o + i V2 p with o = g W1 z and p = g (i W2 z),
   z conv1's input: the stretch path S = V1 o and the turn-turn path T = i V2 p = -V2 g W2 z (no m = 0 part). Per block,
   on the m > 0 coefficients, with each edge's vectors scaled by its cutoff envelope before pooling:
       share   sum |T| / (sum |S| + sum |T|)
       cosine  <S, T> / (|S| |T|), pooled: positive where the paths reinforce, negative where they cancel
       |T|/|M| sum |T| / sum |S + T|, and the envelope-weighted median of the per-edge ratio
   At the atoms (each atom's incoming messages rotated back and summed, envelope applied; M_atom is the block's whole
   Edgewise update, m = 0 included): |T_atom| / |M_atom|, the cosine between T_atom and the rest M_atom - T_atom, and
   how much each path cancels between neighbours: sum |T_atom| / sum over edges |T_edge| (and the same for S, m > 0).
   For the step-0 model and the trained one, at every t.

3. Reliance. In the trained conv_layer_separation, T removed from every message of every block (each message becomes
   V1 o, the W2 = 0 run's form of it). Control: remove instead a piece of S of the same size, |T| on each edge along S
   (at most all of S's m > 0 part). Reported: |dv| / |v - T| and the change of the error, at every t.

Writes <out-dir>/report.txt, cases.jsonl (one row per input and model) and turn_path.json (part 2's sums).
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from fairchem.core.datasets.collaters.simple_collater import data_list_collater
from fairchem.core.models.uma.escn_md_block import set_mole_ac_start_index

from saddleflow import SaddleFlow
from saddleflow.data.materials_saddles_dataset import MaterialsSaddlesDataset
from saddleflow.data.transforms import wrap_positions
from saddleflow.flow.matching import apply_output_projections, build_atomic_data
from saddleflow.models.conv_layer_separation import (ConvLayerSeparationEdgewise, _m_terms, apply_conv_layer_separation,
                                                      conv1_split, gate_pair, is_separated, separated_message)
from saddleflow.utils.parity import w2_report

SIGMA = 0.3               # A: start noise (training and inference)
XT_SIGMA = 0.05           # A: x_t perturbation for t <= 0.9
T_FLOOR = 0.1             # target switches above t = 1 - T_FLOOR
U0_PROB = 0.5             # start at the minimum (u = 0) this often
SAMPLER_TIMES = (0.0, 0.25, 0.5, 0.75)
TASK_MAP = {"oc22": "oc20"}
MODELS = ("plain", "conv_layer_separation", "w2", "fm2", "conv_layer_separation_step0")
LABELS = {"plain": "plain", "conv_layer_separation": "conv_layer_separation", "w2": "W2 = 0 run",
          "fm2": "FM2", "conv_layer_separation_step0": "FM2 + separation (step 0)"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="directory of MaterialsSaddles .aselmdb shards (held out)")
    p.add_argument("--plain", required=True, help="release directory of the plain fine-tune")
    p.add_argument("--separated", required=True, help="release directory of the conv_layer_separation fine-tune")
    p.add_argument("--w2", required=True, help="release directory of the W2 = 0 fine-tune")
    p.add_argument("--fm2", required=True, help="release directory of FM2 (the fine-tunes' starting weights)")
    p.add_argument("--n-records", type=int, default=128)
    p.add_argument("--draws", type=int, default=4, help="noised starts per record, identical for every model")
    p.add_argument("--times", default="0,0.25,0.5,0.75,0.9,0.95")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


# ---------------------------------------------------------------- the two paths of conv2
def conv2_paths(conv, o, p):
    """conv2_merge split into its stretch path (V1 o, with the m = 0 output) and turn-turn path (i V2 p, zero at
    m = 0), both in conv2's output layout; their sum is conv2_merge(conv, o, p)."""
    num_edges = o.shape[0]
    o_by_m, p_by_m = o.split(conv.m_split_sizes, dim=1), p.split(conv.m_split_sizes, dim=1)
    m0 = conv.fc_m0(o_by_m[0].reshape(num_edges, -1)).reshape(num_edges, -1, conv.m_output_channels)
    stretch, turn = [m0], [torch.zeros_like(m0)]
    for m in range(1, conv.mmax + 1):
        layer = conv.so2_m_conv[m - 1]
        v1_oa, _, v1_ob, _ = _m_terms(layer, o_by_m[m].reshape(num_edges, 2, -1))
        _, v2_pa, _, v2_pb = _m_terms(layer, p_by_m[m].reshape(num_edges, 2, -1))
        stretch += [v1_oa, v1_ob]
        turn += [-v2_pb, v2_pa]
    return torch.cat(stretch, dim=1), torch.cat(turn, dim=1), m0.shape[1]


class Recorder:
    """Shared state of the measuring Edgewise modules: what to measure (`stats`, keyed by block) and what to remove
    (`edit`: None, "turn" or "control")."""

    def __init__(self, n_blocks):
        self.n_blocks, self.stats, self.edit, self.check = n_blocks, None, None, None
        self.gen = torch.Generator().manual_seed(0)


def new_stats():
    keys = ("eS", "eT", "eM", "eS2", "eT2", "eST", "aT2", "aM2", "aR2", "aTR", "aT", "aSm", "eTsum", "eSsum")
    return {k: 0.0 for k in keys} | {"ratio": [], "weight": []}


class MeasuringEdgewise(ConvLayerSeparationEdgewise):
    """ConvLayerSeparationEdgewise that records the stretch and turn-turn paths and can remove one of them."""

    def forward_chunk(self, x_full, x_original_shape, x_edge, edge_index, wigner, wigner_inv_envelope,
                      node_offset: int = 0, ac_mole_start_idx: int = 0):
        rec, k = self._recorder, self._block
        set_mole_ac_start_index(self, ac_mole_start_idx)
        x_message = self.backend.node_to_edge_wigner_permute(x_full, edge_index, wigner)
        o, p, gating = conv1_split(self.so2_conv_1, x_message, x_edge)
        o, p = gate_pair(self.act, gating, o, p)
        stretch, turn, n0 = conv2_paths(self.so2_conv_2, o, p)
        S, T = stretch[:, n0:], turn[:, n0:]
        nS, nT = S.flatten(1).norm(dim=1), T.flatten(1).norm(dim=1)
        if rec.check is not None:
            with torch.no_grad():
                ref = separated_message(self, x_message, x_edge)
            rec.check["paths"] = max(rec.check.get("paths", 0.0),
                                     float((stretch + turn - ref).norm() / ref.norm().clamp_min(1e-30)))
            off = wigner_inv_envelope[:, 0, 1:].abs().max() / wigner_inv_envelope[:, 0, 0].abs().max().clamp_min(1e-30)
            rec.check["envelope"] = max(rec.check.get("envelope", 0.0), float(off))

        if rec.edit == "turn":
            message = stretch
        elif rec.edit == "control":
            f = (nT / nS.clamp_min(1e-30)).clamp(max=1.0)[:, None, None]
            message = torch.cat((stretch[:, :n0], S - f * S), dim=1) + turn
        else:
            message = stretch + turn
        new_embedding = self.backend.permute_wigner_inv_edge_to_node(
            message, wigner_inv_envelope, edge_index, x_original_shape, node_offset)

        if rec.stats is not None and rec.edit is None:          # UMA turns gradients on internally (forces)
            with torch.no_grad():
                self._record(rec.stats[self._block], stretch.detach(), turn.detach(), n0, nS.detach(), nT.detach(),
                             new_embedding.detach(), wigner_inv_envelope.detach(), edge_index, x_original_shape,
                             node_offset, rec.gen)

        set_mole_ac_start_index(self, 0)
        return new_embedding

    def _record(self, st, stretch, turn, n0, nS, nT, new_embedding, wigner_inv_envelope, edge_index, num_nodes,
                node_offset, gen):
        S, T = stretch[:, n0:], turn[:, n0:]
        env = wigner_inv_envelope[:, 0, 0]                   # the (l = 0, m = 0) entry of the rotation is 1
        M = S + T
        nM = M.flatten(1).norm(dim=1)
        dot = (S * T).flatten(1).sum(1)
        st["eS"] += float((env * nS).sum()); st["eT"] += float((env * nT).sum()); st["eM"] += float((env * nM).sum())
        st["eS2"] += float((env ** 2 * nS ** 2).sum()); st["eT2"] += float((env ** 2 * nT ** 2).sum())
        st["eST"] += float((env ** 2 * dot).sum())
        keep = torch.randperm(len(env), generator=gen)[:4096].to(env.device)
        st["ratio"].append((nT / nM.clamp_min(1e-30))[keep].double().cpu().numpy())
        st["weight"].append(env[keep].double().cpu().numpy())
        # at the atoms
        t_atom = self.backend.permute_wigner_inv_edge_to_node(turn, wigner_inv_envelope, edge_index, num_nodes,
                                                              node_offset)
        s_m = torch.cat((torch.zeros_like(stretch[:, :n0]), S), dim=1)
        s_atom = self.backend.permute_wigner_inv_edge_to_node(s_m, wigner_inv_envelope, edge_index, num_nodes,
                                                              node_offset)
        r_atom = new_embedding - t_atom
        st["aT2"] += float((t_atom ** 2).sum()); st["aM2"] += float((new_embedding ** 2).sum())
        st["aR2"] += float((r_atom ** 2).sum()); st["aTR"] += float((t_atom * r_atom).sum())
        st["aT"] += float(t_atom.flatten(1).norm(dim=1).sum()); st["aSm"] += float(s_atom.flatten(1).norm(dim=1).sum())
        st["eTsum"] += float((env * nT).sum()); st["eSsum"] += float((env * nS).sum())


def make_measuring(net, recorder):
    blocks = net.backbone.backbone.blocks
    if not is_separated(net.backbone.backbone):
        apply_conv_layer_separation(net.backbone.backbone)
    for k, block in enumerate(blocks):
        ew = block.edge_wise
        ew.activation_checkpoint_chunk_size = None           # an atom's incoming messages in one call
        ew.__class__ = MeasuringEdgewise
        ew._recorder, ew._block = recorder, k


# ---------------------------------------------------------------- inputs
def minimum_key(r):
    pos = wrap_positions(r["start_pos"].double(), r["cell"].double())
    return (tuple(r["Z"].tolist()), tuple(np.round(r["cell"].numpy().ravel(), 2)),
            tuple(np.round(pos.numpy().ravel(), 2)))


def make_inputs(args, rng):
    ds = MaterialsSaddlesDataset(sorted(str(p) for p in Path(args.dataset).glob("*.aselmdb")),
                                 task_name_map=TASK_MAP, align_com=True)
    picks = rng.permutation(len(ds))[:args.n_records]
    times = [float(t) for t in args.times.split(",")]
    groups, inputs = {}, defaultdict(list)
    for i in picks:
        r = ds[int(i)]
        key = minimum_key(r)
        group = groups.setdefault(key, len(groups))
        fixed = r["fixed"].bool()
        mobile = (~fixed).double()[:, None]
        base = dict(record=int(i), group=group, Z=r["Z"], cell=r["cell"].float(), fixed=fixed,
                    task_name=r["task_name"], charge=int(r["charge"]), spin=int(r["spin"]))
        E, S = r["start_pos"].double(), r["saddle_un_pos"].double()
        for d in range(args.draws):
            u = 0.0 if rng.random() < U0_PROB else float(rng.random())
            eps = SIGMA * torch.as_tensor(rng.normal(size=E.shape)) * mobile
            if not bool(fixed.any()):
                eps = eps - eps.mean(0)
            x0 = E + u * (S - E) + eps
            for t in times:
                xt = (1 - t) * x0 + t * S
                if (1 - t) >= T_FLOOR - 1e-9:
                    xt = xt + XT_SIGMA * torch.as_tensor(rng.normal(size=E.shape)) * mobile
                    target = (S - xt) / (1 - t)
                else:
                    target = S - x0
                if not bool(fixed.any()):
                    target = target - target.mean(0)
                inputs[t].append(dict(base, draw=d, u=u, t=t, x=xt.float(), target=target))
    return inputs, len(groups)


# ---------------------------------------------------------------- forward
def velocities(net, chunk, device):
    data = [build_atomic_data(wrap_positions(c["x"], c["cell"]), c["Z"], c["cell"], c["task_name"], c["charge"],
                              c["spin"], c["fixed"]) for c in chunk]
    b = data_list_collater(data, otf_graph=True).to(device)
    t = torch.tensor([c["t"] for c in chunk], dtype=torch.float32, device=device)
    fixed = torch.cat([c["fixed"] for c in chunk]).to(device)
    flag = (~fixed).float() if net.backbone.inject_flag else None
    with torch.no_grad():
        h = net.backbone(b, t, b.batch, flag=flag)["node_embedding"]
        v = net.velocity_head(h, t, b.batch)
        v = apply_output_projections(v, fixed, b.batch, len(chunk))
    return torch.split(v.double().cpu(), [len(c["Z"]) for c in chunk])


def error(v, c):
    m = ~c["fixed"]
    d = (v - c["target"])[m]
    return float(d.norm(dim=1).mean()), float(d.norm() / c["target"][m].norm().clamp_min(1e-30))


def load(name, path, device):
    net = SaddleFlow.from_pretrained(path, device=device).eval()
    if name == "conv_layer_separation_step0":
        apply_conv_layer_separation(net.backbone.backbone)
    mode = "conv_layer_separation" if is_separated(net.backbone.backbone) else "stock"
    w2 = max(r["ratio"] for r in w2_report(net.backbone.backbone))
    print(f"[load] {name}: {path}  mode {mode}  max ||W2||/||W1|| {w2:.3e}", flush=True)
    return net, dict(path=str(path), mode=mode, w2_ratio_max=w2)


def first_batch_checks(net, recorder, chunk, device):
    """On one batch: stretch + turn equals the separated message, the envelope is the rotation's (0, 0) entry, and
    the measuring modules give the same velocity as conv_layer_separation's own (relative differences)."""
    recorder.check = {}
    measured = velocities(net, chunk, device)
    check, recorder.check = recorder.check, None
    blocks = net.backbone.backbone.blocks
    for b in blocks:
        b.edge_wise.__class__ = ConvLayerSeparationEdgewise
    try:
        reference = velocities(net, chunk, device)
    finally:
        for b in blocks:
            b.edge_wise.__class__ = MeasuringEdgewise
    num = sum(float((a - r).norm() ** 2) for a, r in zip(measured, reference)) ** 0.5
    den = sum(float(r.norm() ** 2) for r in reference) ** 0.5
    check["velocity"] = num / max(den, 1e-30)
    return check


def run(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    inputs, n_groups = make_inputs(args, rng)
    paths = {"plain": args.plain, "conv_layer_separation": args.separated, "w2": args.w2, "fm2": args.fm2,
             "conv_layer_separation_step0": args.fm2}
    rows, turn_path, loaded, checks, t0 = [], {}, {}, {}, time.time()
    for name in MODELS:
        net, info = load(name, paths[name], device)
        expect = "stock" if name in ("plain", "w2", "fm2") else "conv_layer_separation"
        if info["mode"] != expect:
            raise RuntimeError(f"{name} runs as {info['mode']}, expected {expect}")
        loaded[name] = info
        measure = name in ("conv_layer_separation", "conv_layer_separation_step0")
        recorder = None
        if measure:
            recorder = Recorder(len(net.backbone.backbone.blocks))
            make_measuring(net, recorder)
            first = next(iter(inputs.values()))[:args.batch_size]
            checks[name] = first_batch_checks(net, recorder, first, device)
            print(f"[check] {name}: {checks[name]}", flush=True)
        for t, group in inputs.items():
            if measure:
                recorder.stats = {k: new_stats() for k in range(recorder.n_blocks)}
            for s in range(0, len(group), args.batch_size):
                chunk = group[s:s + args.batch_size]
                vs = velocities(net, chunk, device)
                edited = {}
                if name == "conv_layer_separation":
                    saved, recorder.stats = recorder.stats, None
                    try:
                        for edit in ("turn", "control"):
                            recorder.edit = edit
                            edited[edit] = velocities(net, chunk, device)
                    finally:
                        recorder.edit, recorder.stats = None, saved
                for i, c in enumerate(chunk):
                    err, rel = error(vs[i], c)
                    row = dict(model=name, t=t, record=c["record"], group=c["group"], draw=c["draw"], u=c["u"],
                               n_atoms=len(c["Z"]), n_mobile=int((~c["fixed"]).sum()), err=err, rel=rel,
                               target=float(c["target"][~c["fixed"]].norm(dim=1).mean()))
                    for edit, ve in edited.items():
                        e_err, _ = error(ve[i], c)
                        dv = float((ve[i] - vs[i])[~c["fixed"]].norm(dim=1).mean())
                        row[edit] = dict(err=e_err, dv=dv)
                    rows.append(row)
            if measure:
                turn_path[f"{name}/t={t:g}"] = finish(recorder.stats)
                recorder.stats = None
            print(f"{name}  t={t:g}: {len(group)} inputs  ({time.time() - t0:.0f}s)", flush=True)
        del net
        if device == "cuda":
            torch.cuda.empty_cache()
    return rows, turn_path, loaded, n_groups, checks


def finish(stats):
    out = []
    for k in sorted(stats):
        s = stats[k]
        ratio, weight = np.concatenate(s["ratio"]), np.concatenate(s["weight"])
        order = np.argsort(ratio)
        cw = np.cumsum(weight[order])
        wmed = float(ratio[order][np.searchsorted(cw, cw[-1] / 2)]) if cw[-1] > 0 else float("nan")
        out.append(dict(
            block=k,
            share=s["eT"] / max(s["eS"] + s["eT"], 1e-300),
            cosine=s["eST"] / max(np.sqrt(s["eS2"] * s["eT2"]), 1e-300),
            t_over_m=s["eT"] / max(s["eM"], 1e-300),
            t_over_m_median=wmed,
            atom_t_over_m=np.sqrt(s["aT2"] / max(s["aM2"], 1e-300)),
            atom_cosine=s["aTR"] / max(np.sqrt(s["aT2"] * s["aR2"]), 1e-300),
            atom_share=np.sqrt(s["aT2"]) / max(np.sqrt(s["aT2"]) + np.sqrt(s["aR2"]), 1e-300),
            keep_t=s["aT"] / max(s["eTsum"], 1e-300),
            keep_s=s["aSm"] / max(s["eSsum"], 1e-300),
            sums={key: v for key, v in s.items() if key not in ("ratio", "weight")}))
    return out


# ---------------------------------------------------------------- statistics
def per_record(rows, model, t, value):
    """{record: (group, mean over draws of value(row))}."""
    acc = defaultdict(list)
    group = {}
    for r in rows:
        if r["model"] == model and r["t"] == t:
            acc[r["record"]].append(value(r))
            group[r["record"]] = r["group"]
    return {k: (group[k], float(np.mean(v))) for k, v in acc.items()}


def paired(a, b, n_boot, seed):
    """Mean of a - b over records and the fraction of records with a < b, each with a 95% interval from resampling
    groups (start minima)."""
    recs = sorted(set(a) & set(b))
    d = np.array([a[k][1] - b[k][1] for k in recs])
    groups = np.array([a[k][0] for k in recs])
    members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    rng = np.random.default_rng(seed)
    means, fracs = [], []
    for _ in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))])
        means.append(d[idx].mean())
        fracs.append((d[idx] < 0).mean())
    lo_hi = lambda x: np.percentile(x, [2.5, 97.5])
    return dict(n=len(d), groups=len(members), mean=float(d.mean()), mean_ci=lo_hi(means).tolist(),
                lower=float((d < 0).mean()), lower_ci=lo_hi(fracs).tolist())


def med(xs):
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.median(xs)) if xs else float("nan")


# ---------------------------------------------------------------- report
def report(args, rows, turn_path, loaded, n_groups, checks):
    times = sorted({r["t"] for r in rows})
    n_records = len({r["record"] for r in rows})
    tag = lambda t: "sampler" if any(abs(t - s) < 1e-9 for s in SAMPLER_TIMES) else "extra"
    L = ["conv_layer_separation probe: velocity error by flow time, and the turn-turn path",
         f"dataset {args.dataset}",
         f"{n_records} records from {n_groups} start minima, {args.draws} draws each, seed {args.seed}; "
         f"intervals: {args.bootstrap} resamples of the minima",
         "The sampler (5 passes x 4 Euler steps) calls the network at t = 0, 0.25, 0.5, 0.75 only; other t are extra rows.",
         "At t > 0.9 the target is S - x0 (no x_t perturbation): compare models with each other there, not across t.",
         "=" * 110, "", "Models"]
    for name in MODELS:
        i = loaded[name]
        L.append(f"  {LABELS[name]:28s} mode {i['mode']:22s} max ||W2||/||W1|| {i['w2_ratio_max']:.2e}   {i['path']}")
    L += ["", "Checks on one batch (relative differences; all should be at float32 rounding):"]
    for name, c in checks.items():
        L.append(f"  {LABELS[name]:28s} stretch + turn vs the separated message {c['paths']:.1e}   "
                 f"envelope: off-(0,0) entries of row 0 {c['envelope']:.1e}   velocity vs unmeasured model "
                 f"{c['velocity']:.1e}")
    L.append("")

    # ---- part 1
    L += ["1. VELOCITY ERROR BY FLOW TIME",
          "   mean over mobile atoms of |v - T| (A per unit time), averaged over draws then records; rel = ||v - T|| / ||T||",
          ""]
    pairs = [("conv_layer_separation", "plain"), ("conv_layer_separation", "w2"), ("conv_layer_separation", "fm2"),
             ("plain", "fm2"), ("conv_layer_separation_step0", "conv_layer_separation")]
    part1 = {}
    for t in times:
        L.append(f"  t = {t:g}  ({tag(t)})")
        L.append(f"    {'model':28s} {'|v-T| mean':>10s} {'median':>8s} {'rel mean':>9s} {'|T| mean':>9s}")
        recs = {}
        for name in MODELS:
            e = per_record(rows, name, t, lambda r: r["err"])
            rl = per_record(rows, name, t, lambda r: r["rel"])
            tg = per_record(rows, name, t, lambda r: r["target"])
            recs[name] = e
            vals = [v for _, v in e.values()]
            L.append(f"    {LABELS[name]:28s} {np.mean(vals):10.4f} {np.median(vals):8.4f} "
                     f"{np.mean([v for _, v in rl.values()]):9.4f} {np.mean([v for _, v in tg.values()]):9.4f}")
        L.append(f"    {'paired, a - b':44s} {'mean':>8s} {'95% interval':>20s}   {'a lower':>7s} {'95% interval':>16s}")
        for a, b in pairs:
            p = paired(recs[a], recs[b], args.bootstrap, args.seed)
            part1[f"t={t:g}/{a}-{b}"] = p
            L.append(f"    {LABELS[a] + ' - ' + LABELS[b]:44s} {p['mean']:+8.4f}   [{p['mean_ci'][0]:+.4f}, "
                     f"{p['mean_ci'][1]:+.4f}]   {p['lower']:7.1%}   [{p['lower_ci'][0]:.1%}, {p['lower_ci'][1]:.1%}]")
        L.append("")

    # ---- part 2
    L += ["2. THE TURN-TURN PATH  T = i V2 p = -V2 g W2 z  vs the stretch path  S = V1 o  (m > 0 coefficients)",
          "   edges, envelope-weighted: share = sum|T| / (sum|S| + sum|T|); cos = <S, T> pooled (< 0: the paths cancel);",
          "   T/M = sum|T| / sum|S + T|, and the envelope-weighted median of the per-edge ratio.",
          "   atoms: T/M = |T_atom| / |whole Edgewise update|; cos between T_atom and the rest; keep = sum|path_atom| /",
          "   sum over edges |path_edge| (1: neighbours add up; small: they cancel), for T and for S (m > 0).", ""]
    for t in times:
        s0, tr = turn_path[f"conv_layer_separation_step0/t={t:g}"], turn_path[f"conv_layer_separation/t={t:g}"]
        L.append(f"  t = {t:g}  ({tag(t)})")
        L.append("          | edges: step 0                        | edges: trained                       |"
                 " atoms: step 0                    | atoms: trained")
        L.append("    block | share    cos    T/M  T/M med | share    cos    T/M  T/M med |"
                 "  T/M    cos keep T keep S |  T/M    cos keep T keep S")
        for a, b in zip(s0, tr):
            L.append(f"    {a['block']:5d} | {a['share']:.3f} {a['cosine']:+.3f}  {a['t_over_m']:.3f}  {a['t_over_m_median']:.3f}"
                     f"   | {b['share']:.3f} {b['cosine']:+.3f}  {b['t_over_m']:.3f}  {b['t_over_m_median']:.3f}   |"
                     f" {a['atom_t_over_m']:.3f} {a['atom_cosine']:+.3f}  {a['keep_t']:.3f}  {a['keep_s']:.3f} |"
                     f" {b['atom_t_over_m']:.3f} {b['atom_cosine']:+.3f}  {b['keep_t']:.3f}  {b['keep_s']:.3f}")
        L.append("")

    # ---- part 3
    L += ["3. RELIANCE: trained conv_layer_separation with T removed from every message (each message V1 o) vs the",
          "   control (a piece of S of the same size removed). Medians over inputs; mean errors over records.", "",
          f"    {'t':>5s}  {'|dv|/|v-T|':>21s}  {'error change':>21s}  {'mean |v-T|':>40s}",
          f"    {'':5s}  {'T removed':>10s} {'control':>10s}  {'T removed':>10s} {'control':>10s}  "
          f"{'separated':>9s} {'T removed':>10s} {'control':>9s} {'W2 = 0':>9s}"]
    for t in times:
        rs = [r for r in rows if r["model"] == "conv_layer_separation" and r["t"] == t]
        w2 = per_record(rows, "w2", t, lambda r: r["err"])
        mean_of = lambda f: np.mean([v for _, v in per_record(rows, "conv_layer_separation", t, f).values()])
        L.append(f"    {t:5g}  {med([r['turn']['dv'] / r['err'] for r in rs]):10.4f} "
                 f"{med([r['control']['dv'] / r['err'] for r in rs]):10.4f}  "
                 f"{med([r['turn']['err'] / r['err'] - 1 for r in rs]):+10.4f} "
                 f"{med([r['control']['err'] / r['err'] - 1 for r in rs]):+10.4f}  "
                 f"{mean_of(lambda r: r['err']):9.4f} {mean_of(lambda r: r['turn']['err']):10.4f} "
                 f"{mean_of(lambda r: r['control']['err']):9.4f} {np.mean([v for _, v in w2.values()]):9.4f}"
                 f"   ({tag(t)})")
    L.append("")
    return "\n".join(L) + "\n", part1


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, turn_path, loaded, n_groups, checks = run(args)
    text, part1 = report(args, rows, turn_path, loaded, n_groups, checks)
    (out / "report.txt").write_text(text)
    with open(out / "cases.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    (out / "turn_path.json").write_text(json.dumps(dict(turn_path=turn_path, paired=part1, models=loaded,
                                                        checks=checks), indent=1))
    print(text)


if __name__ == "__main__":
    main()
