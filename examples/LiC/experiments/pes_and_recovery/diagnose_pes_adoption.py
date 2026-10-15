"""Michael: fixed-probe E2-vs-E0 teacher adoption, with NO training or rollout.

The primary comparison is EMA (the inference weights); raw is diagnostic.
The original 192 audit probes are reproducibly reconstructed, verified, then
cached with their teachers ONCE. A second invocation does not query UMA again.
"""

import sys
from pathlib import Path

# Michael: this script imports LiC modules from examples/LiC/rollout_eval.
_LIC = Path(__file__).resolve().parents[2]  # examples/LiC
sys.path[:0] = [str(_LIC / 'rollout_eval')]

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from ase.io import read
from fairchem.core.datasets.collaters.simple_collater import data_list_collater
from safetensors.torch import load_model

from pes_direction import alignment, query_teacher
from saddleflow.data import atoms_to_sample_dict
from saddleflow.data.transforms import gaussian_perturbation, wrap_positions
from saddleflow.flow import FlowMatchingConfig, FlowMatchingLoss
from saddleflow.flow.matching import apply_output_projections, build_atomic_data
from saddleflow.models.time_filmed_backbone import TimeFiLMBackbone
from saddleflow.classical_saddle_search import FreeDOFs, PESBackend, make_uma_calculator
from saddleflow.utils import load_ema_weights

try:
    from force_guided_sampling import load_flow_model
    from saddle_search_data import file_hash
except ModuleNotFoundError:
    from examples.LiC.rollout_eval.force_guided_sampling import load_flow_model
    from examples.LiC.rollout_eval.saddle_search_data import file_hash


DEFAULT_RUN = "/scratch/gqs7zr/saddleflow/lic_pes_directions_21242185"
PAYLOAD_KEYS = ["positions", "t", "targets", "parent", "free_mask", "force", "q", "teacher",
                "force_rms", "eigenvalues", "active", "teacher_disagreement"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", default=DEFAULT_RUN)
    p.add_argument("--out-dir", help="Default: <run-dir>/fixed_probe_adoption")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--cache-only", action="store_true", help="Reconstruct/verify/cache teachers, without loading students")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def array_hash(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        h.update(str(array.dtype).encode()); h.update(str(array.shape).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def reconstruct_probes(run_dir):
    """Replay the original audit generator, without a model or PES evaluation."""
    root = Path(run_dir)
    configs = [json.loads((root/e/"config.json").read_text()) for e in ["E0", "E2"]]
    for key in ["init_weights", "noise_sigma", "seed", "fd_eps", "audit_samples_per_saddle", "updates",
                "batch_size", "learning_rate", "uma_lr", "warmup_steps", "ema_decay", "calibration_dir"]:
        if configs[0].get(key) != configs[1].get(key):
            raise ValueError(f"Michael: E0/E2 differ in {key}")
    for key in ["backbone", "attn_layers", "head_depth", "delta_endpoint_channels", "unfreeze_uma_all",
                "early_time_film", "early_time_film_blocks", "lambda_pes", "teacher_deadband"]:
        if configs[0]["extras"].get(key) != configs[1]["extras"].get(key):
            raise ValueError(f"Michael: E0/E2 architecture/calibration differ in {key}")
    config = configs[0]
    directory = Path(config["calibration_dir"])
    calibration = json.loads((directory/"calibration.json").read_text())
    data_hash = file_hash(config["train_traj"])
    if data_hash != calibration["settings"]["training_data_sha256"]:
        raise ValueError("Michael: training trajectory changed since the audit")
    with np.load(directory/"alignment_audit.npz", allow_pickle=False) as archive:
        audit = {key: archive[key] for key in archive.files}
    saddles = read(config["train_traj"], index=":")[1::3]
    free = FreeDOFs(saddles[0]).free
    for atoms in saddles:
        if not np.array_equal(FreeDOFs(atoms).free, free) or not np.array_equal(atoms.numbers, saddles[0].numbers):
            raise ValueError("Michael: this LiC diagnostic expects a common atom layout/free mask")
    rng = torch.Generator().manual_seed(config["seed"]+4000)
    positions, targets, parents, times = [], [], [], []
    for parent, atoms in enumerate(saddles):
        for _ in range(config["audit_samples_per_saddle"]):
            eps = gaussian_perturbation(torch.as_tensor(free), config["noise_sigma"], rng)
            t = float(torch.rand((), generator=rng))
            x = torch.as_tensor(atoms.positions, dtype=torch.float32)+(1-t)*eps
            x = wrap_positions(x, torch.as_tensor(np.asarray(atoms.cell), dtype=torch.float32))
            positions.append(x.numpy()); targets.append(-eps.numpy())
            parents.append(parent); times.append(t)
    probes = dict(positions=np.asarray(positions), targets=np.asarray(targets),
                  parent=np.asarray(parents), t=np.asarray(times), free_mask=free)
    for key in ["parent", "t"]:
        if not np.array_equal(probes[key], audit[key]):
            raise ValueError(f"Michael: reconstructed {key} does not exactly match the original audit")
    hashes = np.asarray([array_hash(x,t,v) for x,t,v in zip(probes["positions"], probes["t"], probes["targets"])])
    if "probe_sha256" in audit and not np.array_equal(hashes, audit["probe_sha256"]):
        raise ValueError("Michael: original probe hashes do not match")
    probes["probe_sha256"] = hashes
    context = dict(run_dir=str(root.resolve()), training_data_sha256=data_hash,
                   audit_sha256=file_hash(directory/"alignment_audit.npz"),
                   seed=config["seed"], noise_sigma=config["noise_sigma"], fd_eps=config["fd_eps"],
                   model=config["extras"]["backbone"], teacher_deadband=calibration["force_deadband"],
                   original_hashes_available="probe_sha256" in audit,
                   probe_set="original alignment audit; fresh bridge draws, NOT exact training-update samples")
    return probes, saddles, audit, context


def cached_teachers(path, probes, saddles, audit, context, device):
    """Create a write-once cache, or verify and reuse it without any UMA calls."""
    path = Path(path)
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        if json.loads(str(data["context_json"])) != context:
            raise ValueError("Michael: existing immutable cache belongs to different data/settings")
        for key in probes:
            if not np.array_equal(data[key], probes[key]):
                raise ValueError(f"Michael: cached {key} differs from reconstructed probes")
        fields = [data[k] for k in PAYLOAD_KEYS]
        if str(data["payload_sha256"]) != array_hash(*fields):
            raise ValueError("Michael: immutable cache checksum failed")
        print(f"[cache] Michael: reusing {path}; ZERO new UMA queries", flush=True)
        return data
    print(f"[verify] Michael: {len(probes['t'])} probes; parent/time replay exact; "
          f"original coordinate hashes available={context['original_hashes_available']}", flush=True)
    calculator, batch_evaluate = make_uma_calculator(context["model"], device, saddles[0].info.get("task_name", "omat"))
    backends = [PESBackend(atoms, calculator, batch_evaluate) for atoms in saddles]
    forces, modes, teachers, scores, eigenvalues, calls = [], [], [], [], [], []
    for i, (position, parent) in enumerate(zip(probes["positions"], probes["parent"])):
        pes = backends[parent]
        result = query_teacher(pes, torch.as_tensor(position), "mmf", context["fd_eps"], context["seed"])
        force = np.zeros_like(position, dtype=float); q = np.zeros_like(force)
        force[pes.dofs.free] = result["force"].reshape(-1,3)
        q[pes.dofs.free] = result["q"].reshape(-1,3)
        forces.append(force); modes.append(q); teachers.append(result["vector"].cpu().numpy())
        scores.append(result["force_rms"]); eigenvalues.append(result["eigs"]); calls.append(result["force_calls"])
        if (i+1)%16 == 0 or i+1 == len(probes["t"]):
            print(f"[teacher] Michael: {i+1}/{len(probes['t'])}; cumulative PES geometries={sum(calls)}", flush=True)
    data = dict(probes, force=np.asarray(forces), q=np.asarray(modes), teacher=np.asarray(teachers),
                force_rms=np.asarray(scores), eigenvalues=np.asarray(eigenvalues), teacher_calls=np.asarray(calls))
    free = data["free_mask"]
    reconstructed = dict(force_rms=data["force_rms"], lambda1=data["eigenvalues"][:,0],
        eigengap=np.diff(data["eigenvalues"], axis=1)[:,0],
        cosine_force=np.asarray([alignment(a[free], b[free]) for a,b in zip(data["targets"], data["force"])]),
        cosine_mmf=np.asarray([alignment(a[free], b[free]) for a,b in zip(data["targets"], data["teacher"])]),
        active=data["force_rms"] > context["teacher_deadband"])
    # Exact random-state verification above; numerical physics verification
    # allows small FD/GPU differences, but never silently accepts a failed check.
    errors = {}
    for key, value in reconstructed.items():
        if key == "active":
            ok = np.array_equal(value, audit[key])
        else:
            ok = np.allclose(value, audit[key], atol=5e-4, rtol=5e-4, equal_nan=True)
            errors[key] = float(np.nanmax(np.abs(value-audit[key])))
        if not ok:
            raise ValueError(f"Michael: original audit check failed for {key}; cache NOT written")
    data["active"] = reconstructed["active"]
    data["teacher_disagreement"] = np.clip(1-reconstructed["cosine_mmf"],0,2)
    data["context_json"] = json.dumps(context, sort_keys=True)
    data["verification_json"] = json.dumps(dict(max_abs_errors=errors, atol=5e-4, rtol=5e-4), sort_keys=True)
    data["payload_sha256"] = array_hash(*[data[k] for k in PAYLOAD_KEYS])
    path.parent.mkdir(parents=True, exist_ok=True)
    # 'xb' refuses overwrites even if another process created the cache meanwhile.
    with path.open("xb") as handle:
        np.savez_compressed(handle, **data)
    print(f"[cache] Michael: verified teacher cache saved once: {path}", flush=True)
    return data


def evaluate_velocities(model, probes, saddles, device, batch_size):
    """Forward-only evaluation at cached x_t and per-example t; no FM resampling."""
    model.eval()
    values = []
    samples = [atoms_to_sample_dict(atoms) for atoms in saddles]
    with torch.no_grad():
        for first in range(0, len(probes["t"]), batch_size):
            end = min(first+batch_size, len(probes["t"]))
            chunk = [samples[parent] for parent in probes["parent"][first:end]]
            dl = [build_atomic_data(torch.as_tensor(probes["positions"][i]), s["Z"], s["cell"],
                  s["task_name"], s["charge"], s["spin"], s["fixed"])
                  for i,s in zip(range(first,end),chunk)]
            batch = data_list_collater(dl, otf_graph=True).to(device)
            t = torch.as_tensor(probes["t"][first:end], dtype=torch.float32, device=device)
            feature = (model.backbone(batch,t,batch.batch) if isinstance(model.backbone,TimeFiLMBackbone)
                       else model.backbone(batch))
            hidden = model.global_attn(feature["node_embedding"], batch.batch)
            v = model.velocity_head(hidden,t,batch.batch)
            fixed = torch.cat([s["fixed"] for s in chunk]).to(device)
            v = apply_output_projections(v,fixed,batch.batch,len(chunk))
            values.extend(v.reshape(len(chunk),-1,3).cpu().numpy())
            print(f"[student] Michael: {end}/{len(probes['t'])} fixed probes", flush=True)
    return np.asarray(values)


def compare_metrics(velocities, probes):
    """Paired atom-vector RMS/MSE, not Cartesian-component averages."""
    free = probes["free_mask"]
    target = probes["targets"][:,free].astype(float)
    def rms(v): return np.sqrt(np.square(v).sum(-1).mean(-1))
    def cos(v): return np.asarray([alignment(a,b) for a,b in zip(v,probes["teacher"][:,free])])
    initial = velocities["initializer"][:,free].astype(float)
    result = dict(weights=np.asarray(["ema","raw"]), parent=probes["parent"], t=probes["t"],
                  active=probes["active"], teacher_disagreement=probes["teacher_disagreement"],
                  initializer_cosine=cos(initial), initializer_fm_error=rms(initial-target)**2,
                  initializer_velocity_rms=rms(initial))
    for key in ["cosine_e0","cosine_e2","delta_cosine","delta_velocity_rms","fm_error_e0","fm_error_e2",
                "delta_fm_error","velocity_rms_e0","velocity_rms_e2","e0_shift_from_init","e2_shift_from_init"]:
        result[key] = []
    for weights in result["weights"]:
        e0 = velocities["E0_"+weights][:,free].astype(float)
        e2 = velocities["E2_"+weights][:,free].astype(float)
        c0,c2 = cos(e0),cos(e2)
        error0,error2 = rms(e0-target)**2,rms(e2-target)**2
        values = [c0,c2,c2-c0,rms(e2-e0),error0,error2,error2-error0,rms(e0),rms(e2),rms(e0-initial),rms(e2-initial)]
        for key,value in zip(["cosine_e0","cosine_e2","delta_cosine","delta_velocity_rms","fm_error_e0","fm_error_e2",
                "delta_fm_error","velocity_rms_e0","velocity_rms_e2","e0_shift_from_init","e2_shift_from_init"],values):
            result[key].append(value)
    for key,value in list(result.items()):
        if isinstance(value,list): result[key] = np.asarray(value)
    for name,velocity in velocities.items(): result["velocity_"+name] = velocity
    return result


def make_report(data):
    from scipy.stats import spearmanr
    initial_valid = data["active"] & np.isfinite(data["initializer_cosine"])
    initial_cosine = float(data["initializer_cosine"][initial_valid].mean()) if initial_valid.any() else np.nan
    lines = ["Michael: fixed-probe E2-vs-E0 MMF teacher adoption", "="*120,
        "EMA is PRIMARY (inference weights); raw is diagnostic. No training or rollout occurs.",
        "Probes are the original training-distribution audit, not exact samples used in optimizer updates.",
        "Rows use the SAME cached teachers; N_active uses the original force deadband.",
        "Velocity units: A per unit FM time. FM error: free-atom vector MSE in A^2 per FM-time^2.",
        "Positive delta cosine = E2 aligns better; positive delta FM error = E2 hurts FM accuracy.",
        "Undefined zero-vector cosines are counted explicitly, not treated as successful alignment.",
        f"Initializer: mean active cosine={initial_cosine:.6f}; mean FM error={np.mean(data['initializer_fm_error']):.7g}; "
        f"median velocity RMS={np.median(data['initializer_velocity_rms']):.7g} A/FM-time.", "",
        "weights group            N_active N_cos   mean c0   mean c2   mean dc   median dc  improve%  med dv_RMS   mean e0   mean e2   mean de", "-"*135]
    selections = [("all",np.ones(len(data["t"]),bool))]
    selections += [(f"t={lo:.2f}-{lo+.25:.2f}",(data["t"]>=lo)&(data["t"]<lo+.25)) for lo in np.arange(0,1,.25)]
    edges = [0,.1,.25,.5,1,2.000001]
    selections += [(f"d={lo:g}-{min(hi,2):g}",(data["teacher_disagreement"]>=lo)&(data["teacher_disagreement"]<hi)) for lo,hi in zip(edges[:-1],edges[1:])]
    selections += [(f"parent={parent}",data["parent"]==parent) for parent in np.unique(data["parent"])]
    for i,weights in enumerate(data["weights"]):
        for group,selected in selections:
            active = selected & data["active"]
            valid = active & np.isfinite(data["delta_cosine"][i])
            if not active.any(): continue
            c0,c2,dc = [data[k][i,valid] for k in ["cosine_e0","cosine_e2","delta_cosine"]]
            means = [np.mean(c0),np.mean(c2),np.mean(dc),np.median(dc),100*np.mean(dc>0)] if valid.any() else [np.nan]*5
            lines.append(f"{weights:7s} {group:16s} {active.sum():8d} {valid.sum():5d} "
                + " ".join(f"{v:10.6f}" for v in means) + " "
                + " ".join(f"{v:11.6g}" for v in [np.median(data['delta_velocity_rms'][i,active]),
                    np.mean(data['fm_error_e0'][i,active]),np.mean(data['fm_error_e2'][i,active]),
                    np.mean(data['delta_fm_error'][i,active])]))
        valid = data["active"] & np.isfinite(data["delta_cosine"][i]) & np.isfinite(data["teacher_disagreement"])
        d,dc = data["teacher_disagreement"][valid],data["delta_cosine"][i,valid]
        rho = float(spearmanr(d,dc).statistic) if len(dc)>1 and np.std(dc)>1e-12 and np.std(d)>1e-12 else np.nan
        lines += [f"{weights}: disagreement vs adoption Spearman rho={rho:.6f} (descriptive; probes cluster by parent)",
                  f"{weights}: median shifts from initializer: E0={np.median(data['e0_shift_from_init'][i]):.7g}, "
                  f"E2={np.median(data['e2_shift_from_init'][i]):.7g} A/FM-time"]
    lines += ["", "Interpretation (no automatic pass/fail or test-based tuning):",
        "No change in dc or dv: local teacher adoption is not demonstrated.",
        "Positive dc with similar FM error: local adoption; failed basins may indicate inadequate support/transfer.",
        "Positive dc with substantially worse FM error: competing objectives.",
        "Raw adoption without EMA adoption: averaging suppresses the intervention.",
        "Large dv with little dc: inspect velocity magnitudes; change need not be directional.",
        "These probes alone do not prove capacity limits or uniquely establish a support-mismatch cause."]
    return "\n".join(lines)+"\n"


def main():
    args = parse_args()
    root = Path(args.run_dir)
    out = Path(args.out_dir) if args.out_dir else root/"fixed_probe_adoption"
    probes,saddles,audit,context = reconstruct_probes(root)
    cache = cached_teachers(out/"probes_teacher.npz",probes,saddles,audit,context,args.device)
    if args.cache_only: return
    _,_,backbone,attention,head = load_flow_model(root/"E0/checkpoint_step_00000000",args.device)
    model = FlowMatchingLoss(FlowMatchingConfig(),backbone,attention,head)
    velocities = {}
    cases = [("initializer",root/"E0/checkpoint_step_00000000","ema")]
    cases += [(f"{e}_{w}",root/e/"checkpoint_final",w) for w in ["ema","raw"] for e in ["E0","E2"]]
    for name,checkpoint,weights in cases:
        print(f"[model] Michael: {name}; {checkpoint}; {weights}",flush=True)
        if weights == "ema":
            load_ema_weights(str(checkpoint),[backbone,attention,head],args.device,use_ema=True)
        else:
            # Restore the ENTIRE saved student, including unfrozen backbone and
            # time-FiLM; the old raw inference helper only restores the head.
            load_model(model,str(checkpoint/"model.safetensors"),strict=True,device=args.device)
        velocities[name] = evaluate_velocities(model,cache,saddles,args.device,args.batch_size)
    results = compare_metrics(velocities,cache)
    results["teacher_cache_sha256"] = cache["payload_sha256"]
    results["context_json"] = cache["context_json"]
    results["models_json"] = json.dumps({name:dict(checkpoint=str(p.resolve()),weights=w) for name,p,w in cases})
    np.savez_compressed(out/"adoption.npz",**results)
    report = make_report(results)
    (out/"report.txt").write_text(report)
    print(report,flush=True)
    print(f"[save] Michael: numerical results and report saved in {out}",flush=True)
    if not args.no_plot:
        try:
            from plot_pes_adoption import render_pdf
        except ModuleNotFoundError:
            from examples.LiC.experiments.pes_and_recovery.plot_pes_adoption import render_pdf
        render_pdf(out/"adoption.npz")


if __name__ == "__main__":
    main()
