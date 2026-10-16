"""Post-hoc isopotential-curvature audit. NO steering or training.

Replay stored Li coordinates with frozen UMA and the existing free-space FD
Hessian. Physics is cached in restartable NPZ chunks. Analysis thresholds can
change without new UMA calls; PDF rendering is a separate, model-free script.
"""

import sys
from pathlib import Path

# This script imports LiC modules from examples/LiC/rollout_eval.
_LIC = Path(__file__).resolve().parents[2]  # examples/LiC
sys.path[:0] = [str(_LIC / 'rollout_eval')]

import argparse
import hashlib
import json
import signal
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy.linalg import null_space

from saddleflow.classical_saddle_search import PESBackend, make_uma_calculator

try:
    from saddle_search_data import load_geometry, nearest_distances, read_npz, file_hash
except ModuleNotFoundError:
    from examples.LiC.rollout_eval.saddle_search_data import load_geometry, nearest_distances, read_npz, file_hash


DEFAULT_SHELLS = "/scratch/gqs7zr/saddleflow/model_ckpts/lic_tsdenoise_sigma0.25_b200/robustness_shells/robustness_shells.npz"
DEFAULT_SEARCH = "/scratch/gqs7zr/saddleflow/lic_saddle_search_threeway_21107961/saddle_search.npz"
OUTCOME_NAMES = np.asarray(["true_certified","true_uncertified","false_bridge","other_failure"])
PHYSICS_KEYS = ["positions","energy","force","hessian","asymmetry","sensitivity_hessian"]


def save_npz(path, data):
    # Atomic replacement: an interrupted write leaves the previous snapshot intact.
    path = Path(path); temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary,**data); temporary.replace(path)


def array_hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode()); h.update(str(a.dtype).encode()); h.update(a.tobytes())
    return h.hexdigest()


def kappa_values(force, hessian, force_floor):
    """Paper Eq. 6: tangent-plane minimum curvature, not the full min mode.

    F and H must already be restricted to the free Cartesian coordinates.
    Both nu1 and kappa are undefined when the force normal is unreliable.
    Restrict to an orthonormal basis: min eig(P H P) would include an
    artificial zero normal mode and is NOT the correct calculation.
    """
    force = np.asarray(force,dtype=float).reshape(-1)
    hessian = .5*(np.asarray(hessian)+np.asarray(hessian).T)
    norm = float(np.linalg.norm(force))
    eigs = np.linalg.eigvalsh(hessian)
    valid = norm > force_floor
    basis = np.full((len(force),len(force)-1),np.nan)
    iso_eigs = np.full(len(force)-1,np.nan)
    if valid:
        basis = null_space((force/norm)[None,:])
        iso_eigs = np.linalg.eigvalsh(basis.T@hessian@basis)
    nu1 = float(iso_eigs[0])
    return dict(force_norm=norm,eigenvalues=eigs,tangent_basis=basis,iso_eigenvalues=iso_eigs,
                nu1=nu1,kappa=-nu1/norm if valid else np.nan,valid=valid)


def load_paths(args, geometry):
    """Adapt existing shell and solver archives; never regenerate a rollout."""
    records,provenance = [],[]
    rng = np.random.default_rng(args.seed)
    for archive_index,path in enumerate(args.archives):
        a = read_npz(path)
        for key in ["cell","train_saddles","test_saddles","false_bridges"]:
            if not np.allclose(a[key],geometry[key],atol=1e-4,rtol=0):
                raise ValueError(f"{path} differs from the reference geometry in {key}")
        if "numbers" in a and not np.array_equal(a["numbers"],geometry["numbers"]):
            raise ValueError("atom ordering differs")
        if "pbc" in a and not np.array_equal(a["pbc"],geometry["pbc"]):
            raise ValueError("periodic directions differ")
        if "positions" in a and not np.allclose(a["positions"][~geometry["free_mask"]],geometry["positions"][~geometry["free_mask"]]):
            raise ValueError("fixed scaffold differs")
        shell = "paths" in a
        paths = a["paths"] if shell else a["path"]
        methods = np.repeat("saddleflow",len(paths)) if shell else a["method"]
        config = json.loads(str(a["configuration_json" if shell else "config_json"]))
        K = config.get("K",20)
        for method in dict.fromkeys(methods):
            if args.methods != "all" and method not in args.methods.split(","): continue
            rows = np.flatnonzero(methods == method)
            if args.limit_per_method is not None:
                rows = np.sort(rng.choice(rows,min(args.limit_per_method,len(rows)),replace=False))
            for row in rows:
                p = paths[row,:int(a["trace_length"][row])] if "trace_length" in a else paths[row]
                p = np.asarray(p,dtype=float)
                p = p[np.isfinite(p).all(axis=1)]
                if not len(p): continue
                is_flow = shell or method == "saddleflow" or ("status" in a and a["status"][row] == "fixed_K")
                if not is_flow:
                    # The MMF archive records the same x before and after its
                    # force query. These are not two translation steps.
                    p = p[np.r_[True,np.any(np.diff(p,axis=0) != 0,axis=1)]]
                start = int(a["start_step"][row]) if shell else 0
                steps = np.arange(len(p))+start
                progress = np.arange(len(p))/max(len(p)-1,1)
                records.append(dict(path=p,step=steps,progress=progress,
                    flow_time=steps/K if is_flow else np.full(len(p),np.nan),is_flow=is_flow,
                    collection=f"{archive_index}:"+("shells" if shell else "search"),method=str(method),
                    archive_index=archive_index,archive_row=int(row)))
        provenance.append(dict(path=str(Path(path).resolve()),sha256=file_hash(path)))
    if not records: raise ValueError("no selected paths; use --methods all for differently named models")
    return records,provenance


def pack_points(records, geometry, args):
    """Exact-coordinate cache shared across trajectories and the fixed-z map."""
    points,lookup = [],{}
    def add(x):
        x = np.ascontiguousarray(x,dtype=float)
        key = x.tobytes()
        if key not in lookup: lookup[key] = len(points); points.append(x.copy())
        return lookup[key]
    length = max(len(r["path"]) for r in records)
    path = np.full((len(records),length,3),np.nan)
    index = np.full((len(records),length),-1,dtype=int)
    step,progress,time = [np.full((len(records),length),np.nan) for _ in range(3)]
    for i,r in enumerate(records):
        n = len(r["path"]); path[i,:n] = r["path"]
        index[i,:n] = [add(x) for x in r["path"]]
        step[i,:n],progress[i,:n],time[i,:n] = r["step"],r["progress"],r["flow_time"]
    grid = np.empty((0,3)); grid_index = np.empty((0,0),dtype=int)
    center = geometry["false_bridges"][args.map_basin].copy()
    if args.map_z is not None: center[2] = args.map_z
    if not args.no_map:
        offset = np.linspace(-args.map_half_width,args.map_half_width,args.grid_size)
        xx,yy = np.meshgrid(center[0]+offset,center[1]+offset)
        grid = np.column_stack([xx.ravel(),yy.ravel(),np.full(xx.size,center[2])])
        grid_index = np.asarray([add(x) for x in grid]).reshape(xx.shape)
    metadata = {key:np.asarray([r[key] for r in records]) for key in
        ["collection","method","archive_index","archive_row","is_flow"]}
    return np.asarray(points),dict(path=path,point_index=index,step=step,progress=progress,flow_time=time,
        trace_length=np.asarray([len(r["path"]) for r in records]),grid_positions=grid,grid_point_index=grid_index,
        map_center=center,**metadata)


def checked_chunk(filename, expected, context):
    chunk = read_npz(filename); n = len(chunk["positions"])
    if str(chunk["context_json"]) != context or not 0 < n <= len(expected) or not np.array_equal(chunk["positions"],expected[:n]):
        raise ValueError("cached physics belongs to different inputs/settings; use a new output directory")
    if str(chunk["checksum"]) != array_hash(*[chunk[k] for k in PHYSICS_KEYS]):
        raise ValueError("physics-cache checksum failed")
    return chunk


def evaluate_cached(points, cache_dir, settings, make_backend, sensitivity_indices, sensitivity_eps,
                    chunk_size=128, log_every=16, on_chunk=None, stop_requested=None):
    """Load UMA lazily. Completed chunks are reused without any PES calls."""
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True,exist_ok=True)
    context = json.dumps(settings,sort_keys=True)
    pieces,pes,new_calls = [],None,0
    began = perf_counter()
    for first in range(0,len(points),chunk_size):
        if stop_requested is not None and stop_requested(): break
        end = min(first+chunk_size,len(points)); filename = cache_dir/f"points_{first:07d}_{end:07d}.npz"
        chunk = dict(positions=points[first:end].copy(),energy=np.zeros(end-first),force=np.zeros((end-first,3)),
            hessian=np.zeros((end-first,3,3)),asymmetry=np.zeros(end-first),
            sensitivity_hessian=np.full((end-first,len(sensitivity_eps),3,3),np.nan),
            context_json=context,force_calls=0)
        done = 0
        if filename.exists():
            saved = checked_chunk(filename,points[first:end],context); done = len(saved["positions"])
            for key in PHYSICS_KEYS: chunk[key][:done] = saved[key]
            chunk["force_calls"] = int(saved["force_calls"])
            print(f"[cache] {first+done}/{len(points)} points reused; zero new UMA queries for this prefix",flush=True)
        if done < end-first:
            if pes is None: pes = make_backend()
            before = pes.force_calls; completed_calls = 0
            try:
                for local in range(done,end-first):
                    x = points[first+local]
                    force = pes.forces(x)
                    energy = pes.energy(x)  # cached central energy BEFORE FD moves atoms
                    hessian,asymmetry = pes.free_hessian(x,settings["fd_eps"])
                    chunk["force"][local],chunk["energy"][local] = force,energy
                    chunk["hessian"][local],chunk["asymmetry"][local] = hessian,asymmetry
                    if first+local in sensitivity_indices:
                        for j,eps in enumerate(sensitivity_eps):
                            chunk["sensitivity_hessian"][local,j] = pes.free_hessian(x,eps)[0]
                    done = local+1; completed_calls = pes.force_calls-before
                    if done%log_every == 0 or first+done == len(points):
                        print(f"[physics] {first+done}/{len(points)}; new PES geometries="
                              f"{new_calls+completed_calls}; elapsed={(perf_counter()-began)/60:.1f}min",flush=True)
                    if stop_requested is not None and stop_requested(): break
            finally:
                # On a catchable error or signal, persist only finished F/H points.
                # SIGKILL/power loss can still lose the current unsaved chunk.
                chunk = {k:(v[:done] if k in PHYSICS_KEYS else v) for k,v in chunk.items()}
                chunk["force_calls"] += completed_calls; new_calls += completed_calls
                if done:
                    chunk["checksum"] = array_hash(*[chunk[k] for k in PHYSICS_KEYS])
                    save_npz(filename,chunk)
        pieces.append(chunk)
        if on_chunk is not None: on_chunk(first//chunk_size+1,new_calls)
        if stop_requested is not None and stop_requested(): break
    if not pieces: return dict(new_force_calls=np.asarray(0))
    result = {key:np.concatenate([c[key] for c in pieces]) for key in
        PHYSICS_KEYS}
    result.update(new_force_calls=np.asarray(new_calls),cached_force_calls=np.asarray(sum(int(c["force_calls"]) for c in pieces)))
    return result


def point_distances(points, references, geometry):
    # Avoid allocating all points x 179 saddles x periodic images at once.
    if not len(points): return np.empty(0)
    return np.concatenate([nearest_distances(points[i:i+256],references,geometry["cell"],geometry["pbc"])
                           for i in range(0,len(points),256)])


def analyze(inputs, physics, geometry, args):
    rows = [kappa_values(f,h,args.force_floor) for f,h in zip(physics["force"],physics["hessian"])]
    data = dict(inputs,**geometry,outcome_names=OUTCOME_NAMES,config_json=json.dumps(vars(args)),
        point_positions=physics["positions"],point_force=physics["force"],point_energy=physics["energy"],
        point_hessian=physics["hessian"],point_asymmetry=physics["asymmetry"],
        new_force_calls=physics["new_force_calls"],cached_force_calls=physics["cached_force_calls"])
    for key in rows[0]: data["point_"+key] = np.asarray([r[key] for r in rows])
    index = data["point_index"]; present = index >= 0; safe = np.maximum(index,0)
    for key in ["nu1","kappa","force_norm"]:
        data["history_"+key] = np.where(present,data["point_"+key][safe],np.nan)
    data["history_valid"] = present & data["point_valid"][safe]
    data["history_negative"] = data["history_valid"] & (data["history_nu1"] < 0)
    data["history_warning"] = data["history_valid"] & (data["history_nu1"] < -args.curvature_tol)
    d_false = point_distances(physics["positions"],geometry["false_bridges"],geometry)
    data["history_distance_false"] = np.where(present,d_false[safe],np.nan)
    final_index = np.asarray([index[i,n-1] for i,n in enumerate(data["trace_length"])],dtype=int)
    final = physics["positions"][final_index]
    true = np.vstack([geometry["train_saddles"],geometry["test_saddles"]])
    d_true = point_distances(final,true,geometry)
    eigs = data["point_eigenvalues"][final_index]
    stationary = data["point_force_norm"][final_index] <= args.fmax
    physical = stationary & (eigs[:,0] < -args.eigenvalue_tol) & (eigs[:,1] > args.eigenvalue_tol)
    outcome = np.full(len(final),3,dtype=int)
    outcome[d_false[final_index] < args.false_radius] = 2
    outcome[d_true < args.true_radius] = 1
    outcome[(d_true < args.true_radius)&physical] = 0
    data.update(final=final,outcome=outcome,physical_success=physical,final_force_max=data["point_force_norm"][final_index],
                final_eigenvalues=eigs,distance_true=d_true,final_distance_false=d_false[final_index])
    summary = {k:[] for k in ["valid_fraction","negative_fraction","warning_fraction","any_warning",
        "first_warning_step","first_false_step","first_warning_time","first_false_time",
        "first_warning_progress","first_false_progress","first_negative_step","min_nu1","max_kappa","starts_inside_false",
        "lead_eligible","lead_observable","lead_steps","lead_progress"]}
    for i,n in enumerate(data["trace_length"]):
        valid = data["history_valid"][i,:n]; warning = data["history_warning"][i,:n]
        entered = data["history_distance_false"][i,:n] < args.false_radius
        kw = np.flatnonzero(warning); kf = np.flatnonzero(entered)
        negative = np.flatnonzero(data["history_negative"][i,:n])
        w = int(kw[0]) if len(kw) else None; f = int(kf[0]) if len(kf) else None
        eligible = outcome[i] == 2 and not entered[0] and f is not None
        observable = eligible and valid[:f].any()
        values = [valid.mean(),data["history_negative"][i,:n][valid].mean() if valid.any() else np.nan,
            warning[valid].mean() if valid.any() else np.nan,bool(warning.any()),
            data["step"][i,w] if w is not None else np.nan,data["step"][i,f] if f is not None else np.nan,
            data["flow_time"][i,w] if w is not None else np.nan,data["flow_time"][i,f] if f is not None else np.nan,
            data["progress"][i,w] if w is not None else np.nan,data["progress"][i,f] if f is not None else np.nan,
            data["step"][i,negative[0]] if len(negative) else np.nan,
            np.min(data["history_nu1"][i,:n][valid]) if valid.any() else np.nan,
            np.max(data["history_kappa"][i,:n][valid]) if valid.any() else np.nan,bool(entered[0]),bool(eligible),bool(observable),
            data["step"][i,f]-data["step"][i,w] if eligible and w is not None else np.nan,
            data["progress"][i,f]-data["progress"][i,w] if eligible and w is not None else np.nan]
        for key,value in zip(summary,values): summary[key].append(value)
    boolean_keys = {"any_warning","starts_inside_false","lead_eligible","lead_observable"}
    data.update({key:np.asarray(value,dtype=bool if key in boolean_keys else float) for key,value in summary.items()})
    sensitivity = np.flatnonzero(np.isfinite(physics["sensitivity_hessian"]).all(axis=(1,2,3))) if len(args.sensitivity_eps) else np.array([],int)
    data["sensitivity_index"] = sensitivity
    data["sensitivity_eps"] = np.asarray(args.sensitivity_eps)
    data["sensitivity_nu1"] = np.asarray([[kappa_values(physics["force"][i],h,args.force_floor)["nu1"]
        for h in physics["sensitivity_hessian"][i]] for i in sensitivity]).reshape(len(sensitivity),len(args.sensitivity_eps))
    return data


def save_plan(out, points, inputs, geometry, args, settings):
    """Everything needed for later CPU-only export, even without source archives."""
    filename = out/"audit_inputs.npz"; context = json.dumps(settings,sort_keys=True)
    if filename.exists() and str(read_npz(filename)["provenance_json"]) != context:
        raise ValueError("saved audit plan uses different physics inputs; use a new output directory")
    save_npz(filename,dict(**inputs,**geometry,planned_points=points,
        input_keys=np.asarray(list(inputs)),geometry_keys=np.asarray(list(geometry)),
        config_json=json.dumps(vars(args)),provenance_json=context))


def export_cached(out, new_calls=0, reason="CPU cache export", print_report=False):
    """Export completed trajectories and available map points. NEVER query UMA.

    Missing trajectories are excluded, not treated as failures or truncated
    endpoints. The selected subset is processing-order biased, not a new test set.
    """
    out = Path(out); plan = read_npz(out/"audit_inputs.npz")
    args = SimpleNamespace(**json.loads(str(plan["config_json"])))
    settings = json.loads(str(plan["provenance_json"])); context = str(plan["provenance_json"])
    points = plan["planned_points"]; chunk_size = settings["chunk_size"]
    pieces,point_ids = [],[]
    for first in range(0,len(points),chunk_size):
        end = min(first+chunk_size,len(points)); filename = out/"physics_cache"/f"points_{first:07d}_{end:07d}.npz"
        if not filename.exists(): continue  # .tmp files are never read
        chunk = checked_chunk(filename,points[first:end],context)
        pieces.append(chunk); point_ids.extend(range(first,first+len(chunk["positions"])))
    if not pieces:
        print("[partial] no finished physics points yet; nothing to export",flush=True)
        return None
    physics = {key:np.concatenate([c[key] for c in pieces]) for key in PHYSICS_KEYS}
    physics.update(new_force_calls=np.asarray(new_calls),
        cached_force_calls=np.asarray(sum(int(c["force_calls"]) for c in pieces)))
    # Compact indices prevent an unevaluated point from being read as real data.
    remap = np.full(len(points),-1,dtype=int); remap[point_ids] = np.arange(len(point_ids))
    inputs = {key:plan[key] for key in plan["input_keys"]}
    original_index = inputs["point_index"]; present = original_index >= 0
    completed = np.all(~present | (remap[np.maximum(original_index,0)] >= 0),axis=1)
    row_keys = ["path","point_index","step","progress","flow_time","trace_length",
                "collection","method","archive_index","archive_row","is_flow"]
    for key in row_keys: inputs[key] = inputs[key][completed]
    index = inputs["point_index"]
    inputs["point_index"] = np.where(index >= 0,remap[np.maximum(index,0)],-1)
    inputs["grid_point_index"] = remap[inputs["grid_point_index"]]
    geometry = {key:plan[key] for key in plan["geometry_keys"]}
    data = analyze(inputs,physics,geometry,args)
    data.update(provenance_json=context,audit_complete=np.asarray(len(point_ids)==len(points)),
        num_planned_points=np.asarray(len(points)),num_evaluated_points=np.asarray(len(point_ids)),
        num_planned_trajectories=np.asarray(len(completed)),num_complete_trajectories=np.asarray(completed.sum()),
        complete_trajectory_rows=np.flatnonzero(completed),evaluated_point_ids=np.asarray(point_ids),
        snapshot_reason=np.asarray(reason))
    save_npz(out/"kappa_audit.npz",data)
    report = make_report(data)
    temporary = out/"report.tmp.txt"; temporary.write_text(report); temporary.replace(out/"report.txt")
    print(f"[snapshot] {len(point_ids)}/{len(points)} physics points; "
          f"{completed.sum()}/{len(completed)} fully audited trajectories; saved {out/'kappa_audit.npz'}",flush=True)
    if print_report: print(report,flush=True)
    return data


def finite_median(values):
    values = np.asarray(values); values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else np.nan


def make_report(data):
    config = json.loads(str(data["config_json"]))
    coverage = []
    if "audit_complete" in data:
        coverage = [f"Audit status: {'COMPLETE' if data['audit_complete'] else 'PARTIAL'} ({data['snapshot_reason']}).",
            f"Physics points: {int(data['num_evaluated_points'])}/{int(data['num_planned_points'])}; "
            f"fully audited trajectories: {int(data['num_complete_trajectories'])}/{int(data['num_planned_trajectories'])}.",
            "Only fully audited trajectories contribute to outcome/lead statistics. Unfinished paths are excluded, NOT failures.",
            "Partial coverage follows processing order and is not representative of all selected methods/starts."]
    lines = ["post-hoc LiC isopotential-curvature audit", "="*130,
        *coverage,
        "kappa = -min eig(C.T H_free C)/||F_free||; C spans the plane perpendicular to the FULL 3D Li force.",
        f"Force floor={config['force_floor']} eV/A: BOTH nu1 and kappa are undefined at/below it. Warning: nu1 < -{config['curvature_tol']} eV/A^2.",
        "Positive kappa is a CANDIDATE warning, not a failure or Hessian-index certificate.",
        f"True hit <{config['true_radius']} A; false region <{config['false_radius']} A. False region is a geometric ball, not a dynamical basin definition.",
        f"Certified: true hit AND free fmax<={config['fmax']} AND lambda1<-{config['eigenvalue_tol']} AND lambda2>{config['eigenvalue_tol']}.",
        "Physics solver duplicate trace rows were removed; FM integration steps, including stationary steps, were retained.",
        "Lead steps are method-local updates. Normalized progress is descriptive, not a common physical clock across solvers.",
        "Statistics use all selected trajectories; plotting limits only the drawn curves. Related starts are not independent replicates.",
        f"Total cached PES geometries={int(data['cached_force_calls'])}; newly evaluated this invocation={int(data['new_force_calls'])}.","",
        "collection  method       outcome              N  med valid%  med negative%  any warn%  med first warn  med min nu1  med max kappa", "-"*130]
    groups = list(dict.fromkeys(zip(data["collection"],data["method"])))
    for collection,method in groups:
        selected = (data["collection"] == collection)&(data["method"] == method)
        for outcome,name in enumerate(OUTCOME_NAMES):
            use = selected & (data["outcome"] == outcome)
            observable = use & (data["valid_fraction"] > 0)
            if not use.any(): continue
            lines.append(f"{collection:11s} {method:12s} {name:19s} {use.sum():5d} "
                f"{100*np.median(data['valid_fraction'][use]):10.2f} {100*finite_median(data['negative_fraction'][use]):14.2f} "
                f"{100*data['any_warning'][observable].mean() if observable.any() else np.nan:10.2f} "
                f"{finite_median(data['first_warning_step'][use]):15.2f} {finite_median(data['min_nu1'][use]):12.5g} "
                f"{finite_median(data['max_kappa'][use]):14.5g}")
    lines += ["", "Advance warning for eventual FALSE endpoints starting OUTSIDE all false regions",
        "collection  method        false N  start-inside  eligible  observable  warned-before(%)  med lead steps  true-cert FPR(%)  true-cert observed N", "-"*145]
    for collection,method in groups:
        selected = (data["collection"] == collection)&(data["method"] == method)
        false = selected & (data["outcome"] == 2)
        eligible = selected & data["lead_eligible"]; observed = selected & data["lead_observable"]
        true = selected & (data["outcome"] == 0)&(data["valid_fraction"] > 0)
        lead = data["lead_steps"][observed]
        lines.append(f"{collection:11s} {method:12s} {false.sum():7d} {np.sum(false & data['starts_inside_false']):13d} "
            f"{eligible.sum():9d} {observed.sum():11d} {100*np.mean(lead>0) if observed.any() else np.nan:18.2f} "
            f"{finite_median(lead):15.2f} {100*data['any_warning'][true].mean() if true.any() else np.nan:17.2f} {true.sum():21d}")
    lines += ["Missing warnings count as misses in warned-before%; masked-only pre-entry states are reported as unobservable.",
              "Median lead excludes trajectories with no defined warning; use it WITH the warned-before denominator.","",
              "Finite-difference sensitivity (fixed seeded geometry subset; no threshold fitting)",
              "eps(A)    valid N  raw sign flips(%)  warning flips(%)  median |delta nu1|", "-"*85]
    idx = data["sensitivity_index"]; base = data["point_nu1"][idx]
    for j,eps in enumerate(data["sensitivity_eps"]):
        other = data["sensitivity_nu1"][:,j]; valid = np.isfinite(base)&np.isfinite(other)
        if valid.any():
            lines.append(f"{eps:.5f} {valid.sum():10d} {100*np.mean((base[valid]<0)!=(other[valid]<0)):18.2f} "
                f"{100*np.mean((base[valid]<-config['curvature_tol'])!=(other[valid]<-config['curvature_tol'])):17.2f} "
                f"{np.median(np.abs(base[valid]-other[valid])):20.6g}")
    lines += [f"Relative Hessian FD asymmetry: median={np.median(data['point_asymmetry']):.6g}, "
              f"p95={np.percentile(data['point_asymmetry'],95):.6g}, max={np.max(data['point_asymmetry']):.6g}."]
    lines += ["", "No steering, optimization, training, or new learned rollouts occurred.",
              "A discriminative signature requires useful lead AND low false-positive rate, not merely positive endpoint kappa."]
    return "\n".join(lines)+"\n"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archives",nargs="+",default=[DEFAULT_SHELLS,DEFAULT_SEARCH])
    p.add_argument("--out-dir",required=True)
    p.add_argument("--train-traj",default=str((Path(__file__).resolve().parents[2] / "train_set.traj")))
    p.add_argument("--test-traj",default=str((Path(__file__).resolve().parents[2] / "test_set.traj")))
    p.add_argument("--methods",default="saddleflow,oracle,sella",help="all includes every stored method")
    p.add_argument("--model",default="uma-s-1p2"); p.add_argument("--device",default="cuda")
    p.add_argument("--fd-eps",type=float,default=.002)
    p.add_argument("--force-floor",type=float,default=.001)
    p.add_argument("--curvature-tol",type=float,default=0.)
    p.add_argument("--fmax",type=float,default=.005); p.add_argument("--eigenvalue-tol",type=float,default=.01)
    p.add_argument("--true-radius",type=float,default=.1); p.add_argument("--false-radius",type=float,default=.3)
    p.add_argument("--grid-size",type=int,default=31); p.add_argument("--map-basin",type=int,default=0)
    p.add_argument("--map-half-width",type=float,default=1.5); p.add_argument("--map-z",type=float)
    p.add_argument("--no-map",action="store_true")
    p.add_argument("--sensitivity-points",type=int,default=64)
    p.add_argument("--sensitivity-eps",default="0.001,0.004")
    p.add_argument("--cache-chunk-size",type=int,default=128); p.add_argument("--log-every",type=int,default=16)
    p.add_argument("--save-every-chunks",type=int,default=0,
                   help="Write a CPU-plottable partial NPZ/report every N cache chunks (0=end only)")
    p.add_argument("--export-cached",action="store_true",
                   help="CPU-only: export available physics cache and exit; no missing UMA evaluations")
    p.add_argument("--limit-per-method",type=int,help="Seeded execution subset; default ALL")
    p.add_argument("--seed",type=int,default=1); p.add_argument("--dry-run",action="store_true")
    p.add_argument("--plot",action="store_true",help="Optionally plot AFTER saving numerical results")
    args = p.parse_args()
    args.sensitivity_eps = [float(x) for x in args.sensitivity_eps.split(",") if x]
    if min(args.force_floor,args.fd_eps,args.fmax,args.true_radius,args.false_radius,args.map_half_width) <= 0:
        raise ValueError("force floor and length/force scales must be positive")
    if min(args.cache_chunk_size,args.log_every,args.grid_size) <= 0 or args.curvature_tol < 0 or args.sensitivity_points < 0:
        raise ValueError("invalid counts or curvature tolerance")
    if args.save_every_chunks < 0: raise ValueError("snapshot interval cannot be negative")
    if any(x <= 0 for x in args.sensitivity_eps): raise ValueError("FD sensitivity steps must be positive")
    if not args.no_map and args.grid_size < 2: raise ValueError("the XY map needs at least a 2x2 grid")
    return args


def main():
    args = parse_args()
    out = Path(args.out_dir)
    if args.export_cached and (out/"audit_inputs.npz").exists():
        data = export_cached(out,print_report=True)
        if data is not None and args.plot: plot_saved(out)
        return
    reference,geometry = load_geometry(args.train_traj,args.test_traj)
    if not 0 <= args.map_basin < len(geometry["false_bridges"]): raise ValueError("invalid map-basin catalog ID")
    records,provenance = load_paths(args,geometry)
    points,inputs = pack_points(records,geometry,args)
    rng = np.random.default_rng(args.seed+5000)
    sensitivity = np.sort(rng.choice(len(points),min(args.sensitivity_points,len(points)),replace=False))
    print(f"[setup] {len(records)} saved trajectories; {len(points)} unique 3D Li geometries including map",flush=True)
    print(f"[setup] approximately {7*len(points)+6*len(args.sensitivity_eps)*len(sensitivity)} PES geometries before cache reuse",flush=True)
    for i,p in enumerate(provenance): print(f"[archive] {i}: {p['path']}",flush=True)
    if args.dry_run: return
    out.mkdir(parents=True,exist_ok=True)
    settings = dict(input_archives=provenance,points_sha256=array_hash(points),geometry_sha256=array_hash(
        geometry["positions"],geometry["numbers"],geometry["cell"],geometry["pbc"],geometry["free_mask"]),
        metadata=reference.info,model=args.model,fd_eps=args.fd_eps,sensitivity_eps=args.sensitivity_eps,
        sensitivity_index=sensitivity.tolist(),chunk_size=args.cache_chunk_size)
    save_plan(out,points,inputs,geometry,args,settings)
    if args.export_cached:
        # Legacy runs did not save audit_inputs.npz. Reconstruct it using the
        # original arguments, then export existing chunks without loading UMA.
        data = export_cached(out,print_report=True)
        if data is not None and args.plot: plot_saved(out)
        return
    def backend():
        print(f"[UMA] frozen {args.model}; no inference-mode wrapper around force evaluation",flush=True)
        calculator,batch_evaluate = make_uma_calculator(args.model,args.device,reference.info.get("task_name","omat"))
        return PESBackend(reference,calculator,batch_evaluate)
    stopped = []
    def request_stop(signum, frame):
        if not stopped: print(f"[stop] {signal.Signals(signum).name}; finishing this physics point and saving",flush=True)
        stopped.append(signum)
    signals = [signal.SIGINT,signal.SIGTERM,signal.SIGUSR1]
    previous = {s:signal.signal(s,request_stop) for s in signals}
    def snapshot(chunk_number, new_calls):
        if args.save_every_chunks and chunk_number%args.save_every_chunks == 0:
            export_cached(out,new_calls,"periodic snapshot")
    try:
        physics = evaluate_cached(points,out/"physics_cache",settings,backend,set(sensitivity),args.sensitivity_eps,
                                  args.cache_chunk_size,args.log_every,snapshot,lambda:bool(stopped))
    except BaseException:
        export_cached(out,reason="interrupted/error snapshot")
        raise
    finally:
        for s,handler in previous.items():
            # None denotes a native handler not restorable by Python's API.
            signal.signal(s,handler if handler is not None else signal.SIG_DFL)
    data = export_cached(out,int(physics["new_force_calls"]),
                         "graceful interruption" if stopped else "finished",print_report=True)
    if data is not None and args.plot: plot_saved(out)


def plot_saved(out):
    try: from plot_kappa_audit import render_pdf
    except ModuleNotFoundError: from examples.LiC.experiments.kappa_audit.plot_kappa_audit import render_pdf
    render_pdf(out/"kappa_audit.npz")


if __name__ == "__main__": main()
