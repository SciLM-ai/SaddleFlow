"""Build and mix LiC recovery pairs mined from a robustness-shell archive.

This module deliberately contains no model or UMA calls.  Those expensive
operations already happened when ``robustness_shells.py`` created its archive.
Here we use the saved final UMA force merely as a selection signal:

    force_rms(final baseline rollout) > epsilon

selects one exact shell source.  Its target is the training saddle that
generated that shell.  The resulting pair is ordinary straight flow-matching
data:

    x_0 = selected shell source,     x_1 = parent training saddle.

The LiC trainer uses this module only when ``--recovery-shell-archive`` is
explicitly supplied.  Normal LiC TS-denoising training remains unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from saddleflow.data.core import mic_unwrap


LI = 126


def _require_array(archive: np.lib.npyio.NpzFile, name: str) -> np.ndarray:
    """Read one required archive field with an immediately useful error."""
    if name not in archive.files:
        raise ValueError(
            f"shell archive is missing {name!r}. Expected an archive written by "
            "examples/LiC/rollout_eval/robustness_shells.py."
        )
    return archive[name]


def build_recovery_records(
    shell_archive: str | Path,
    original_dataset: Dataset,
    *,
    force_epsilon: float,
) -> tuple[list[dict], dict, dict[str, np.ndarray]]:
    """Select high-force shell starts and turn them into exact recovery pairs.

    ``original_dataset`` supplies atom numbers, fixed-atom constraints, cell,
    and UMA metadata for every one of the twelve training parents.  The shell
    archive stores Li coordinates only, because the graphene carbon scaffold
    is fixed.  We reconstruct a full structure by copying the parent saddle
    geometry and replacing only its mobile Li coordinate.

    Returns
    -------
    records
        Sample dictionaries understood by ``FlowMatchingLoss``.  Each has an
        explicit ``flow_source_pos`` field, so TS-denoising does not redraw
        Gaussian noise for the selected source.
    summary
        Small JSON-serializable provenance report for the run directory.
    buffer_arrays
        Compact numerical representation written alongside the run for audit
        and later reuse.
    """
    archive_path = Path(shell_archive)
    if force_epsilon <= 0:
        raise ValueError("recovery force_epsilon must be positive")
    if not archive_path.is_file():
        raise FileNotFoundError(f"recovery shell archive does not exist: {archive_path}")

    with np.load(archive_path, allow_pickle=False) as archive:
        starts_li = _require_array(archive, "starts")
        parent_index = _require_array(archive, "parent_index").astype(np.int64)
        force_rms = _require_array(archive, "force_rms").astype(np.float64)
        train_saddles_li = _require_array(archive, "train_saddles")
        radius = _require_array(archive, "radius").astype(np.float32)
        start_time = _require_array(archive, "start_time").astype(np.float32)

    n_rows = len(starts_li)
    if starts_li.shape != (n_rows, 3):
        raise ValueError(
            "LiC shell archive field 'starts' must have shape (N, 3) for Li coordinates; "
            f"got {starts_li.shape}"
        )
    if force_rms.shape != (n_rows,) or parent_index.shape != (n_rows,):
        raise ValueError("shell archive starts, parent_index, and force_rms must have matching lengths")
    if radius.shape != (n_rows,) or start_time.shape != (n_rows,):
        raise ValueError("shell archive radius and start_time must match starts")
    if train_saddles_li.ndim != 2 or train_saddles_li.shape[1] != 3:
        raise ValueError("shell archive field 'train_saddles' must have shape (N_train, 3)")
    if not np.isfinite(force_rms).all():
        raise ValueError(
            "shell archive has non-finite force_rms values. Re-run robustness_shells.py "
            "without --no-uma before mining recovery data."
        )
    if parent_index.min(initial=0) < 0 or parent_index.max(initial=-1) >= len(train_saddles_li):
        raise ValueError("shell archive parent_index refers outside its train_saddles array")
    if len(original_dataset) < 2 * len(train_saddles_li):
        raise ValueError(
            "the supplied training dataset has fewer triplets than the shell archive's "
            "training-saddle parents"
        )

    keep = np.flatnonzero(force_rms > force_epsilon)
    if not len(keep):
        raise ValueError(
            f"no shell sources exceed the recovery force threshold {force_epsilon:.6f} eV/A"
        )

    # Fetch one R-side record per parent.  LiC's TS-denoising objective ignores
    # R/P conditioning, and either side has the same fixed carbon scaffold and
    # saddle target.  Using a single side avoids silently duplicating a source.
    parent_templates = [original_dataset[2 * parent] for parent in range(len(train_saddles_li))]
    template = parent_templates[0]
    if template["Z"].numel() <= LI:
        raise ValueError(f"expected Li atom index {LI}, but the training structure is too small")
    if bool(template["fixed"][LI]):
        raise ValueError("Li is marked fixed in the training data; recovery sources require it mobile")

    records: list[dict] = []
    source_full: list[np.ndarray] = []
    target_full: list[np.ndarray] = []
    selected_parent = parent_index[keep]

    for archive_row in keep:
        parent = int(parent_index[archive_row])
        parent_sample = parent_templates[parent]
        cell = parent_sample["cell"].cpu().numpy()

        # Start from the known parent saddle to retain its fixed carbon atoms.
        # The archive stores a wrapped Li source.  Its target must be placed in
        # the source's nearest periodic image before straight interpolation.
        source = parent_sample["saddle_un_pos"].clone().float()
        source[LI] = torch.as_tensor(starts_li[archive_row], dtype=source.dtype)
        target = parent_sample["saddle_un_pos"].clone().float()
        target_li_unwrapped = mic_unwrap(
            starts_li[archive_row], train_saddles_li[parent], cell
        ).astype(np.float32)
        target[LI] = torch.from_numpy(target_li_unwrapped)

        # Fixed atoms must be identical in x0 and x1.  That is both the LiC
        # physical constraint and the general contract of this recovery pair.
        fixed = parent_sample["fixed"]
        if not torch.allclose(source[fixed], target[fixed], atol=1e-6, rtol=0.0):
            raise ValueError(f"recovery row {archive_row} moved a fixed atom")

        record = {
            "start_pos": source.clone(),
            "flow_source_pos": source.clone(),
            "saddle_un_pos": target.clone(),
            # The unconditioned LiC head never consumes this field, but keeping
            # it structurally valid lets the regular FlowMatchingLoss operate
            # without a LiC-only branch.
            "partner_un_pos": source.clone(),
            "Z": parent_sample["Z"].clone(),
            "cell": parent_sample["cell"].clone(),
            "fixed": fixed.clone(),
            "task_name": parent_sample["task_name"],
            "charge": parent_sample["charge"],
            "spin": parent_sample["spin"],
            "delta_norm": torch.linalg.vector_norm(target - source),
            "role": "recovery_shell_to_parent_saddle",
            "triplet_id": parent,
            "metadata": {
                "recovery_archive_row": int(archive_row),
                "recovery_parent_index": parent,
                "recovery_radius_angstrom": float(radius[archive_row]),
                "recovery_start_time": float(start_time[archive_row]),
                "baseline_endpoint_force_rms_ev_per_angstrom": float(force_rms[archive_row]),
            },
        }
        records.append(record)
        source_full.append(source.numpy())
        target_full.append(target.numpy())

    selected_radii, selected_radius_counts = np.unique(radius[keep], return_counts=True)
    summary = {
        "shell_archive": str(archive_path.resolve()),
        "selection_rule": f"baseline endpoint force_rms > {force_epsilon:.6f} eV/A",
        "force_epsilon_ev_per_angstrom": float(force_epsilon),
        "shell_rows_total": int(n_rows),
        "recovery_pairs_selected": int(len(records)),
        "baseline_force_rms_selected_min": float(force_rms[keep].min()),
        "baseline_force_rms_selected_median": float(np.median(force_rms[keep])),
        "baseline_force_rms_selected_max": float(force_rms[keep].max()),
        "selected_count_by_radius_angstrom": {
            f"{value:.3f}": int(count)
            for value, count in zip(selected_radii, selected_radius_counts)
        },
    }
    buffer_arrays = {
        "archive_row": keep.astype(np.int64),
        "parent_index": selected_parent.astype(np.int64),
        "source_pos": np.asarray(source_full, dtype=np.float32),
        "target_saddle_un_pos": np.asarray(target_full, dtype=np.float32),
        "radius": radius[keep],
        "start_time": start_time[keep],
        "baseline_endpoint_force_rms": force_rms[keep].astype(np.float32),
        "force_epsilon": np.asarray([force_epsilon], dtype=np.float32),
    }
    return records, summary, buffer_arrays


class RecoveryMixedDataset(Dataset):
    """Fixed-ratio mixture of the original LiC records and recovery pairs.

    One epoch contains every selected recovery pair exactly once.  Original
    records are repeated as needed to achieve the requested aggregate fraction.
    The original TS-denoising records remain stochastic because their Gaussian
    source noise is still drawn inside ``sample_endpoints`` on every visit.
    """

    def __init__(
        self,
        original_dataset: Dataset,
        recovery_records: list[dict],
        *,
        original_fraction: float,
    ) -> None:
        if not 0.0 < original_fraction < 1.0:
            raise ValueError("recovery original_fraction must lie strictly between 0 and 1")
        if not recovery_records:
            raise ValueError("RecoveryMixedDataset requires at least one recovery record")
        self.original_dataset = original_dataset
        self.recovery_records = recovery_records
        self.original_fraction = float(original_fraction)
        self.n_original_per_epoch = int(round(
            len(recovery_records) * original_fraction / (1.0 - original_fraction)
        ))
        self.n_original_per_epoch = max(1, self.n_original_per_epoch)

    def __len__(self) -> int:
        return self.n_original_per_epoch + len(self.recovery_records)

    def __getitem__(self, index: int) -> dict:
        if index < self.n_original_per_epoch:
            # Cycle evenly through the tiny original dataset.  DataLoader's
            # epoch shuffle randomizes its placement relative to recovery rows.
            return self.original_dataset[index % len(self.original_dataset)]
        return self.recovery_records[index - self.n_original_per_epoch]


def write_recovery_artifacts(
    output_dir: str | Path,
    summary: dict,
    buffer_arrays: dict[str, np.ndarray],
) -> tuple[Path, Path]:
    """Write selection provenance before model training begins."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "recovery_mining_summary.json"
    buffer_path = output / "recovery_buffer.npz"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(buffer_path, **buffer_arrays)
    return summary_path, buffer_path
