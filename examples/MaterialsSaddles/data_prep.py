"""
Data preparation helper — idempotent download + official-split loading.

Convention:
  * Single canonical location: ``$SCRATCH/MaterialsSaddles/`` on every machine.
  * Two artefacts per subset:
      ``$SCRATCH/MaterialsSaddles/<subset>/*.aselmdb``         (the data)
      ``$SCRATCH/MaterialsSaddles/splits/<subset>/{train,val,test}.parquet``
  * If both already exist with the expected file counts they are reused; otherwise
    the missing pieces are pulled from HuggingFace ``AnonymouScientist/MaterialsSaddles``.

Each parquet column is just ``ms_id`` (uint32). One row per ASE-LMDB row, so each
triplet contributes 3 ``ms_id`` rows that are guaranteed to be in the same split
(empirically: per-triplet ms_ids are 3 consecutive integers).

The helper is rank-aware via ``accelerate.PartialState`` — only the global main
process touches the network/filesystem; the others wait at a barrier and then
read the freshly-laid-down files.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Iterable

# How many .aselmdb shards each MaterialsSaddles subset is supposed to contain.
# Sourced from the dataset README; used as a sanity check before deciding the
# local copy is "complete".
EXPECTED_SHARDS = {
    "lemat":   256,
    "oc20":    96,
    "oc22":    32,
    "mp20bat": 32,
}

REPO_ID = "AnonymouScientist/MaterialsSaddles"


def materials_saddles_root() -> Path:
    """Resolve ``$SCRATCH/MaterialsSaddles`` (creating the directory if needed)."""
    scratch = os.environ.get("SCRATCH")
    if not scratch:
        raise SystemExit(
            "$SCRATCH is not set. SaddleFlow pins the dataset under "
            "$SCRATCH/MaterialsSaddles so it works across machines — please "
            "export SCRATCH (most Slurm sites set this automatically; on other clusters "
            "point it at a fast scratch path)."
        )
    root = Path(scratch) / "MaterialsSaddles"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _shard_count(d: Path) -> int:
    return sum(1 for p in d.glob("*.aselmdb")) if d.is_dir() else 0


def _splits_complete(d: Path) -> bool:
    if not d.is_dir():
        return False
    needed = {"train.parquet", "val.parquet", "test.parquet"}
    have = {p.name for p in d.iterdir()}
    return needed.issubset(have)


ALL_SUBSETS: tuple[str, ...] = tuple(EXPECTED_SHARDS.keys())


def ensure_subsets(
    subsets: Iterable[str] = ALL_SUBSETS,
    *,
    accelerator_state=None,
    max_workers: int = 32,
) -> dict[str, Path]:
    """Make sure ``$SCRATCH/MaterialsSaddles/<subset>`` and its splits exist on
    disk for every requested subset; if anything is missing, pull it from
    HuggingFace in a single ``snapshot_download`` call (idempotent — re-runs
    only fetch new/changed files, the HF equivalent of ``git pull``).

    Returns ``{subset: shards_dir}``.

    On a multi-rank launch only the global main process performs the download;
    other ranks wait on a barrier. Pass ``accelerator_state=PartialState()`` (or
    any object with ``is_main_process`` + ``wait_for_everyone()``); falls back
    to single-process behaviour if not given.

    ``max_workers`` controls per-file download concurrency. 32 is a sane default
    for NERSC-class WANs (≈ 32 × 50 MB/s ≈ 1.5 GB/s aggregate, well below HF's
    concurrent-connection throttle and far below pscratch's write speed).
    """
    subsets = list(subsets)
    for s in subsets:
        if s not in EXPECTED_SHARDS:
            raise ValueError(
                f"Unknown subset {s!r}. Known: {sorted(EXPECTED_SHARDS)}"
            )

    root = materials_saddles_root()
    is_main = (accelerator_state is None) or accelerator_state.is_main_process

    need_shards: list[str] = []
    need_splits: list[str] = []
    for s in subsets:
        if _shard_count(root / s) != EXPECTED_SHARDS[s]:
            need_shards.append(s)
        if not _splits_complete(root / "splits" / s):
            need_splits.append(s)

    if (need_shards or need_splits) and is_main:
        from huggingface_hub import snapshot_download
        patterns: list[str] = []
        for s in need_shards:
            patterns.append(f"{s}/*.aselmdb")
        for s in need_splits:
            patterns.append(f"splits/{s}/*.parquet")
        # Repo-root docs are tiny; refresh them on every git-pull-style call so
        # each scratch root stays self-documenting.
        patterns += ["README.md", "DATASHEET.md", "example_load.py"]
        print(f"[data_prep] downloading {patterns} from {REPO_ID} → {root} "
              f"(max_workers={max_workers})")
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=str(root),
            allow_patterns=patterns,
            token=os.environ.get("HF_TOKEN"),
            max_workers=max_workers,
        )

    if accelerator_state is not None:
        accelerator_state.wait_for_everyone()

    out: dict[str, Path] = {}
    for s in subsets:
        shards_dir = root / s
        splits_dir = root / "splits" / s
        n = _shard_count(shards_dir)
        expected = EXPECTED_SHARDS[s]
        if n != expected:
            raise SystemExit(
                f"[data_prep] {shards_dir} has {n} *.aselmdb shards, expected "
                f"{expected}. Re-run on a node with HF_TOKEN set, or remove the "
                f"directory to force a re-download."
            )
        if not _splits_complete(splits_dir):
            raise SystemExit(
                f"[data_prep] {splits_dir} is missing one of train/val/test.parquet. "
                f"Re-run on a node with HF_TOKEN set, or remove the directory."
            )
        print(f"[data_prep] {s}: {shards_dir} ({n} shards) + splits OK")
        out[s] = shards_dir
    return out


def ensure_subset(subset: str = "mp20bat", *, accelerator_state=None) -> Path:
    """Single-subset wrapper around :func:`ensure_subsets` (backward-compat —
    older training scripts call this directly)."""
    return ensure_subsets([subset], accelerator_state=accelerator_state)[subset]


def _build_or_load_msid_to_triplet(shards_dir: Path, *, cache_path: Path,
                                    is_main: bool, accelerator_state=None) -> dict[int, int]:
    """Build (and JSON-cache under ``cache_path``) the saddle-row ``ms_id ->
    triplet_id`` mapping for a MaterialsSaddles subset, where ``triplet_id`` is
    the dataset-wide index (concatenated across shards in lexicographic order).

    Each triplet's 3 rows have consecutive ms_ids; the saddle is the middle
    one. We cache only the saddle ms_id since that's the unambiguous anchor.
    """
    if cache_path.is_file():
        with cache_path.open() as f:
            data = json.load(f)
        return {int(k): int(v) for k, v in data["saddle_ms_to_triplet"].items()}

    if is_main:
        from ase.db import connect
        shard_paths = sorted(shards_dir.glob("*.aselmdb"))
        saddle_to_tid: dict[int, int] = {}
        triplet_id = 0
        # Fast path: ms_ids are perfectly consecutive within and across triplets
        # in every shard (verified empirically May 2026 on mp20bat + lemat;
        # README documents per-triplet consecutiveness, cross-triplet is implied
        # by the row-order convention). So we only need the FIRST row's ms_id
        # per shard + the row count; the saddle ms_id of triplet `i` in that
        # shard is `first_ms_id + 3*i + 1`. This is ~5 orders of magnitude
        # faster than the previous full-row walk (which JSON-decoded every
        # atoms.info dict) on multi-GB lemat shards.
        for shard_path in shard_paths:
            db = connect(str(shard_path), type="aselmdb",
                         readonly=True, use_lock_file=False)
            row_count = db.count()
            if row_count % 3 != 0:
                raise SystemExit(
                    f"[data_prep] {shard_path}: row count {row_count} is not a "
                    f"multiple of 3 — file is corrupt or not a triplet shard."
                )
            n_shard_triplets = row_count // 3
            first_row = next(db.select(limit=1))
            first_ms_id = int(first_row.data["info"]["ms_id"])
            for i in range(n_shard_triplets):
                saddle_to_tid[first_ms_id + 3 * i + 1] = triplet_id + i
            triplet_id += n_shard_triplets
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({
            "shards_dir": str(shards_dir),
            "num_triplets": triplet_id,
            "saddle_ms_to_triplet": {str(k): v for k, v in saddle_to_tid.items()},
        }))
        print(f"[data_prep] built {len(saddle_to_tid):,}-entry ms_id cache → {cache_path}")

    if accelerator_state is not None:
        accelerator_state.wait_for_everyone()

    with cache_path.open() as f:
        data = json.load(f)
    return {int(k): int(v) for k, v in data["saddle_ms_to_triplet"].items()}


def load_official_splits(subset: str = "mp20bat", *, accelerator_state=None
                         ) -> tuple[list[int], list[int], list[int]]:
    """Return ``(train_tids, val_tids, test_tids)`` for the requested subset,
    using the official ``splits/<subset>/{train,val,test}.parquet`` files
    shipped with the HuggingFace dataset.

    Triplet IDs are the index used by ``MaterialsSaddlesDataset`` (records
    ``2*tid`` and ``2*tid+1`` are the R→S and P→S samples).
    """
    import pyarrow.parquet as pq
    root = materials_saddles_root()
    shards_dir = root / subset
    splits_dir = root / "splits" / subset
    cache_path = root / f".msid_cache_{subset}.json"

    is_main = (accelerator_state is None) or accelerator_state.is_main_process
    saddle_to_tid = _build_or_load_msid_to_triplet(
        shards_dir, cache_path=cache_path, is_main=is_main,
        accelerator_state=accelerator_state,
    )

    out: dict[str, list[int]] = {}
    for split in ("train", "val", "test"):
        ms_ids = pq.read_table(str(splits_dir / f"{split}.parquet")).column("ms_id").to_pylist()
        tids: set[int] = set()
        unmatched = 0
        for ms in ms_ids:
            ms = int(ms)
            # Each triplet's 3 ms_ids are consecutive (R = saddle-1, P = saddle+1).
            # Use explicit `is not None` because triplet_id 0 is a valid value
            # that would be swallowed by `or` short-circuiting.
            tid = saddle_to_tid.get(ms)
            if tid is None:
                tid = saddle_to_tid.get(ms + 1)
            if tid is None:
                tid = saddle_to_tid.get(ms - 1)
            if tid is not None:
                tids.add(tid)
            else:
                unmatched += 1
        if unmatched:
            raise SystemExit(
                f"[data_prep] {unmatched} ms_ids in {split}.parquet did not "
                f"resolve to a triplet — the parquet and the local shards are "
                f"out of sync. Wipe {root}/{subset} + {root}/splits/{subset} "
                f"and re-run to refresh."
            )
        out[split] = sorted(tids)

    print(f"[data_prep] official splits: train={len(out['train']):,}  "
          f"val={len(out['val']):,}  test={len(out['test']):,}  "
          f"(total {sum(len(v) for v in out.values()):,} triplets)")
    return out["train"], out["val"], out["test"]


# ----- CLI: run this file standalone to pre-stage data on a new machine -----

def _cli():
    import argparse
    p = argparse.ArgumentParser(
        description="Idempotently stage MaterialsSaddles subsets under "
                    "$SCRATCH/MaterialsSaddles. Pass --all to stage every "
                    "subset (~640 GiB total, dominated by lemat 596 GiB), "
                    "or --subset NAME for a single one. Re-running this is "
                    "cheap — only files that are missing / changed upstream "
                    "are re-downloaded.",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--subset", choices=sorted(EXPECTED_SHARDS),
                   help="Stage one subset.")
    g.add_argument("--all", action="store_true",
                   help="Stage every subset listed in EXPECTED_SHARDS.")
    p.add_argument("--max-workers", type=int, default=32,
                   help="Per-file HF download concurrency. Default 32 is the "
                        "sweet spot on NERSC; bump higher only if you have "
                        "verified that the network rather than HF's "
                        "concurrent-connection throttle is the bottleneck.")
    args = p.parse_args()

    subsets = list(ALL_SUBSETS) if args.all else [args.subset]
    staged = ensure_subsets(subsets, max_workers=args.max_workers)
    print(f"[data_prep] staged: {list(staged)}")
    for s, shards_dir in staged.items():
        train, val, test = load_official_splits(s)
        print(f"[data_prep] {s}: train={len(train):,}  val={len(val):,}  test={len(test):,}")


if __name__ == "__main__":
    _cli()
