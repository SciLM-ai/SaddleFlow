"""Michael: saved, endpoint-basin-disjoint recovery sources for the LiC ablation.

Each shell archive row is one complete rollout episode. We use its START,
never harvest intermediate path states into separate train/test records.
High-force near-misses outside known false regions have their own episode split.
No UMA/model calls occur here. This is not a spatial exclusion of entire bridges.
"""

import sys
from pathlib import Path

# Michael: this script imports LiC modules from examples/LiC/rollout_eval.
_LIC = Path(__file__).resolve().parents[2]  # examples/LiC
sys.path[:0] = [str(_LIC / 'rollout_eval')]

import json
from pathlib import Path

import numpy as np
import torch

from saddleflow.data.core import mic_unwrap
from saddleflow.data.transforms import wrap_positions

try:
    from train_pes_directions import batch_indices
    from saddle_search_data import read_npz, file_hash
except ModuleNotFoundError:
    from examples.LiC.experiments.pes_and_recovery.train_pes_directions import batch_indices
    from examples.LiC.rollout_eval.saddle_search_data import read_npz, file_hash


DEFAULT_ARCHIVE = "/scratch/gqs7zr/saddleflow/model_ckpts/lic_tsdenoise_sigma0.25_b200/robustness_shells/robustness_shells.npz"


def load_learnability_run(checkpoint):
    """Read a small recovery run and verify its exact source-episode provenance."""
    root=Path(checkpoint).parent
    config=json.loads((root/"config.json").read_text())
    selection=read_npz(root/"selection.npz")
    if file_hash(root/"selection.npz") != config["selection_sha256"]:
        raise ValueError("Michael: learnability training selection changed")
    split_path=Path(config["source_run"])/"split.npz"
    if file_hash(split_path) != str(selection["source_split_sha256"]):
        raise ValueError("Michael: source fold changed")
    return config,selection,read_npz(split_path)


def split_failures(archive, train_basins, holdout_fraction, seed, force_epsilon):
    """Force-only mining; basin grouping does not change failure eligibility."""
    high = archive["force_max"] > force_epsilon
    # True hits have precedence in the original geometric classification.
    known = high & (archive["outcome_code"] == 2)
    observed = np.unique(archive["nearest_false_index"][known])
    train_basins = np.asarray(train_basins, dtype=int)
    if not len(train_basins) or not np.isin(train_basins, observed).all():
        raise ValueError("Michael: training basins must be present among mined known-false endpoints")
    held_basins = np.setdiff1d(observed, train_basins)
    if not len(held_basins):
        raise ValueError("Michael: reserve at least one observed false basin")
    # -1: not mined; 0: training; 1: unseen episodes in seen basins;
    # 2: withheld false basins; 3: generic high-force episode holdout.
    partition = np.full(len(high), -1, dtype=int)
    partition[known & np.isin(archive["nearest_false_index"], held_basins)] = 2
    rng = np.random.default_rng(seed)

    def divide(rows, held_code):
        rows = rng.permutation(rows)
        n_held = max(1, int(round(len(rows)*holdout_fraction))) if len(rows) > 1 else 0
        partition[rows[n_held:]] = 0
        partition[rows[:n_held]] = held_code

    for basin in train_basins:
        divide(np.flatnonzero(known & (archive["nearest_false_index"] == basin)), 1)
    # Stratify the remaining force-only failures by parent and radius.
    # Independent stream keeps this generic split identical in both folds.
    rng = np.random.default_rng(seed+100)
    generic = high & ~known
    for parent in np.unique(archive["parent_index"][generic]):
        for radius in np.unique(archive["radius"][generic]):
            divide(np.flatnonzero(generic & (archive["parent_index"] == parent)
                                 & (archive["radius"] == radius)), 3)
    return partition, observed, held_basins


def prepare(args, dataset):
    root = Path(args.run_dir)
    root.mkdir(parents=True, exist_ok=True)
    settings = dict(split_version=1,shell_archive=str(Path(args.shell_archive).resolve()),
        shell_sha256=file_hash(args.shell_archive), train_traj=str(Path(args.train_traj).resolve()),
        train_sha256=file_hash(args.train_traj), train_basins=list(args.train_basins),
        init_weights=str(Path(args.init_weights).resolve()), init_ema_sha256=file_hash(Path(args.init_weights)/"ema.pt"),
        force_epsilon=args.force_epsilon, holdout_fraction=args.holdout_fraction,
        seed=args.seed, audit_sources=args.audit_sources)
    manifest_path = root/"split.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())["settings"] != settings:
            raise ValueError("Michael: this run already has a different immutable split")
        load_split(root, dataset)
        print(f"[split] Michael: reusing {manifest_path}", flush=True)
        return
    a = read_npz(args.shell_archive)
    if "configuration_json" in a:
        baseline = json.loads(str(a["configuration_json"]))["ckpt"]
        if Path(baseline).resolve() != Path(args.init_weights).resolve():
            raise ValueError("Michael: initialize from the same checkpoint that mined the archive")
    if np.any(a["start_step"] != 0):
        raise ValueError("Michael: use a t=0 archive; this experiment learns complete source-to-parent pairs")
    if not np.isfinite(a["force_max"]).all():
        raise ValueError("Michael: mining needs finite saved UMA forces")
    templates = [dataset[2*i] for i in range(len(a["train_saddles"]))]
    free_indices = torch.nonzero(~templates[0]["fixed"]).flatten().numpy()
    if len(free_indices) != 1:
        raise ValueError("Michael: this contained LiC experiment expects one mobile atom")
    li = int(free_indices[0])
    labels = np.asarray([s["saddle_un_pos"][li].numpy() for s in templates])
    cell = templates[0]["cell"].numpy()
    # Compare periodic images rather than raw wrapped coordinates.
    for expected, stored in zip(labels, a["train_saddles"]):
        if np.linalg.norm(mic_unwrap(expected, stored, cell)-expected) > 1e-4:
            raise ValueError("Michael: archive parent saddles do not match the training labels")
    if not np.allclose(cell, a["cell"]):
        raise ValueError("Michael: archive cell differs from the training cell")
    partition, observed, held = split_failures(a, args.train_basins, args.holdout_fraction,
                                             args.seed, args.force_epsilon)
    train_rows = np.flatnonzero(partition == 0)
    if not len(train_rows):
        raise ValueError("Michael: no recovery training rows")
    data = dict(starts=a["starts"], parent=a["parent_index"], radius=a["radius"],
        partition=partition, baseline_force_max=a["force_max"], baseline_outcome=a["outcome_code"],
        baseline_basin=a["nearest_false_index"], baseline_finals=a["finals"],
        train_basins=np.asarray(args.train_basins), holdout_basins=held, li_index=np.asarray(li),
        cell=a["cell"], train_saddles=a["train_saddles"], test_saddles=a["test_saddles"],
        false_bridges=a["false_bridges"])
    with (root/"split.npz").open("xb") as handle:
        np.savez_compressed(handle, **data)
    # New fixed probes on TRAINING recovery bridges only, evenly covering
    # known-false sources and other high-force sources when both are available.
    rng = np.random.default_rng(args.seed+4000)
    false = train_rows[a["outcome_code"][train_rows] == 2]
    n_false = min(len(false), args.audit_sources//2)
    chosen = list(rng.choice(false, n_false, replace=False)) if n_false else []
    remaining = np.setdiff1d(train_rows, chosen)
    chosen += rng.choice(remaining, min(len(remaining), args.audit_sources-len(chosen)), replace=False).tolist()
    probe_x, probe_t, probe_v, probe_parent, probe_rows = [], [], [], [], []
    for row in chosen:
        record = recovery_record(data, int(row), dataset)
        x0, target = record["flow_source_pos"], record["saddle_un_pos"]
        for t in [0., .25, .5, .75]:
            probe_x.append(wrap_positions(x0+t*(target-x0), record["cell"]).numpy())
            probe_v.append((target-x0).numpy()); probe_t.append(t)
            probe_parent.append(int(data["parent"][row])); probe_rows.append(row)
    with (root/"recovery_probes.npz").open("xb") as handle:
        np.savez_compressed(handle, positions=probe_x, targets=probe_v, t=probe_t,
            parent=probe_parent, archive_row=probe_rows, free_mask=(~templates[0]["fixed"]).numpy())
    summary = dict(settings=settings, split_sha256=file_hash(root/"split.npz"),
        probes_sha256=file_hash(root/"recovery_probes.npz"), observed_basins=observed.tolist(),
        heldout_basins=held.tolist(), total_high_force=int(np.sum(partition >= 0)),
        counts={name:int(np.sum(partition == code)) for code,name in enumerate(
            ["recovery_train", "seen_basin_holdout", "withheld_basin", "generic_holdout"])},
        known_false_train=int(np.sum((partition == 0)&(a["outcome_code"] == 2))),
        split_unit="one whole shell rollout per archive row; no intermediate-state harvesting",
        scope="endpoint-basin-disjoint recovery sources, NOT exclusion of all bridge coordinates",
        mining_rule="saved free-atom force_max > epsilon, regardless of geometric true hit")
    manifest_path.write_text(json.dumps(summary, indent=2))
    print(f"[split] Michael: {json.dumps(summary, indent=2)}", flush=True)


def load_split(root, dataset):
    root = Path(root)
    manifest = json.loads((root/"split.json").read_text())
    for name,key in [("split.npz","split_sha256"), ("recovery_probes.npz","probes_sha256")]:
        if file_hash(root/name) != manifest[key]:
            raise ValueError(f"Michael: immutable {name} checksum failed")
    if file_hash(manifest["settings"]["train_traj"]) != manifest["settings"]["train_sha256"]:
        raise ValueError("Michael: training labels changed since the split")
    data = read_npz(root/"split.npz")
    rows = np.flatnonzero(data["partition"] == 0)
    forbidden = (data["baseline_outcome"][rows] == 2) & np.isin(data["baseline_basin"][rows], data["holdout_basins"])
    if forbidden.any():
        raise ValueError("Michael: withheld-basin source leaked into training")
    return data, [recovery_record(data,int(row),dataset) for row in rows]


def recovery_record(data, row, dataset):
    """One full structure, with exact fixed scaffold and a MIC-unwrapped label."""
    parent = int(data["parent"][row]); template = dataset[2*parent]
    li = int(data["li_index"])
    source = template["saddle_un_pos"].clone().float()
    source[li] = torch.as_tensor(data["starts"][row], dtype=torch.float32)
    target = source.clone()
    target[li] = torch.as_tensor(mic_unwrap(data["starts"][row], data["train_saddles"][parent],
                                          data["cell"]), dtype=torch.float32)
    return dict(template, start_pos=source, flow_source_pos=source, saddle_un_pos=target,
                partner_un_pos=source, role="recovery", archive_row=row,
                triplet_id=parent, delta_norm=torch.linalg.vector_norm(target-source))


def mixed_batch(dataset, recovery, update, seed, experiment):
    """Exactly 3 original + 1 recovery; R0 replaces only the fourth member.

    Use an independent per-update bridge RNG in the trainer so R0's extra
    Gaussian draw cannot change the next update's shared original examples.
    """
    ids = batch_indices(len(dataset), update, 4, seed)
    if len(ids) != 4:
        raise ValueError("Michael: use a training dataset divisible by this experiment's batch size 4")
    batch = [dataset[i] for i in ids]
    if experiment != "R0":
        j = batch_indices(len(recovery), update, 1, seed+7000)[0]
        batch[3] = recovery[j]
    return batch
