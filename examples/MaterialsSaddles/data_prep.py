"""
Data preparation helper for MaterialsSaddles v2 — idempotent download + split loading.

MaterialsSaddles v2 (Hugging Face ``SciLM/MaterialsSaddles``) stores every subset
already split and deduplicated:

    <root>/<subset>/{train,val,test}/<subset>_<split>_<NNNN>.aselmdb

Each file holds at most 50,000 transition states as consecutive (R, S, P) rows in
ascending ``ms_id`` order; the split is the directory a file lives in (chemical
systems are disjoint between splits). ``metadata/triplets.parquet`` maps every
saddle ``ms_id`` to its file and row.

Triplet ids ("tids"). SaddleFlow addresses a subset as ONE ``MaterialsSaddlesDataset``
over all of that subset's files in sorted path order (``subset_shards``), i.e.
``<subset>/test/*``, then ``<subset>/train/*``, then ``<subset>/val/*``. A tid is the
position of a triplet in that concatenation (records ``2*tid`` / ``2*tid+1`` are the
R->S / P->S samples). ``load_official_splits`` returns the tids of each split by
reading which directory each file is in — no ms_id lookup is involved. Use
``tid_to_saddle_msid`` to translate tids into the dataset's stable ``ms_id``s.

Location: ``$MATERIALSSADDLES_ROOT`` if set, else ``$SCRATCH/MaterialsSaddles_v2``.
Source: ``$MATERIALSSADDLES_REPO`` (default ``SciLM/MaterialsSaddles``) at revision
``$MATERIALSSADDLES_REVISION`` (default ``v2-dedup``, the branch holding v2 until it
is merged to ``main``). The download runs only for files that are missing, only on
the global main process; other ranks wait at a barrier.
"""

from __future__ import annotations

import json
import os
import zlib
from pathlib import Path
from typing import Iterable

REPO_ID = os.environ.get("MATERIALSSADDLES_REPO", "SciLM/MaterialsSaddles")
REVISION = os.environ.get("MATERIALSSADDLES_REVISION", "v2-dedup")
SPLITS: tuple[str, ...] = ("train", "val", "test")
TRIPLETS_PER_FILE = 50_000

# Triplet counts of the v2 release (verified against every file when it was written).
# They pin the file lists and double as a completeness / consistency check.
EXPECTED_TRIPLETS = {
    "lemat":   {"train": 28_223_516, "val": 1_533_392, "test": 1_566_009},
    "oc20":    {"train": 2_133_114,  "val": 123_751,   "test": 110_853},
    "oc22":    {"train": 139_175,    "val": 6_557,     "test": 6_861},
    "mp20bat": {"train": 31_046,     "val": 1_454,     "test": 1_531},
}
ALL_SUBSETS: tuple[str, ...] = tuple(EXPECTED_TRIPLETS.keys())


def materials_saddles_root() -> Path:
    """Resolve the v2 data root (creating the directory if needed)."""
    root = os.environ.get("MATERIALSSADDLES_ROOT")
    if not root:
        scratch = os.environ.get("SCRATCH")
        if not scratch:
            raise SystemExit(
                "Neither $MATERIALSSADDLES_ROOT nor $SCRATCH is set; point "
                "MATERIALSSADDLES_ROOT at a fast filesystem to hold the dataset.")
        root = str(Path(scratch) / "MaterialsSaddles_v2")
    p = Path(root)
    p.mkdir(parents=True, exist_ok=True)
    return p


def expected_files(subset: str, split: str) -> list[str]:
    """Repository paths of the files of one subset/split, in order."""
    n = EXPECTED_TRIPLETS[subset][split]
    k = (n + TRIPLETS_PER_FILE - 1) // TRIPLETS_PER_FILE
    return [f"{subset}/{split}/{subset}_{split}_{i:04d}.aselmdb" for i in range(k)]


def subset_shards(subset: str, root: Path | None = None) -> list[str]:
    """Every file of a subset (all splits), sorted by path — the order in which
    ``MaterialsSaddlesDataset`` concatenates them, which defines the tids."""
    root = root or materials_saddles_root()
    return sorted(str(root / f) for sp in SPLITS for f in expected_files(subset, sp))


def subset_glob(subset: str, root: Path | None = None) -> str:
    """Glob matching exactly ``subset_shards(subset)``."""
    root = root or materials_saddles_root()
    return str(root / subset / "*" / f"{subset}_*.aselmdb")


def _check_subset(s: str) -> None:
    if s not in EXPECTED_TRIPLETS:
        raise ValueError(f"Unknown subset {s!r}. Known: {sorted(EXPECTED_TRIPLETS)}")


def ensure_subsets(
    subsets: Iterable[str] = ALL_SUBSETS,
    *,
    accelerator_state=None,
    max_workers: int = 32,
    metadata: bool = True,
) -> dict[str, str]:
    """Make sure every file of the requested subsets (all three splits) is on disk,
    downloading only what is missing. Returns ``{subset: glob}``; pass the glob to
    ``MaterialsSaddlesDataset``."""
    subsets = list(subsets)
    for s in subsets:
        _check_subset(s)
    root = materials_saddles_root()
    is_main = (accelerator_state is None) or accelerator_state.is_main_process

    patterns: list[str] = []
    for s in subsets:
        for sp in SPLITS:
            if not all((root / f).is_file() for f in expected_files(s, sp)):
                patterns.append(f"{s}/{sp}/*.aselmdb")
    if metadata and not (root / "metadata" / "triplets.parquet").is_file():
        patterns.append("metadata/*.parquet")
    if patterns and is_main:
        from huggingface_hub import snapshot_download
        patterns += ["README.md", "DATASHEET.md"]
        print(f"[data_prep] downloading {patterns} from {REPO_ID}@{REVISION} -> {root} "
              f"(max_workers={max_workers})")
        snapshot_download(
            repo_id=REPO_ID, repo_type="dataset", revision=REVISION,
            local_dir=str(root), allow_patterns=patterns,
            token=os.environ.get("HF_TOKEN"), max_workers=max_workers,
        )
    if accelerator_state is not None:
        accelerator_state.wait_for_everyone()

    out: dict[str, str] = {}
    for s in subsets:
        missing = [f for sp in SPLITS for f in expected_files(s, sp) if not (root / f).is_file()]
        if missing:
            raise SystemExit(f"[data_prep] {len(missing)} files of {s} missing under {root}, "
                             f"e.g. {missing[:3]}. Re-run on a node with network access.")
        stray = sorted(set(map(str, root.glob(f"{s}/*/*.aselmdb"))) - set(subset_shards(s, root)))
        if stray:
            raise SystemExit(f"[data_prep] unexpected files under {root / s}: {stray[:3]} "
                             f"(they would shift every tid). Remove them.")
        print(f"[data_prep] {s}: {len(subset_shards(s, root))} files under {root / s} OK")
        out[s] = subset_glob(s, root)
    return out


def ensure_subset(subset: str = "mp20bat", *, accelerator_state=None) -> str:
    """Single-subset wrapper around :func:`ensure_subsets`."""
    return ensure_subsets([subset], accelerator_state=accelerator_state)[subset]


def _triplets_in_file(path: str) -> int:
    """Number of triplets in one ``.aselmdb`` (read from its LMDB ``nextid``; the
    release has no deleted rows)."""
    import lmdb
    env = lmdb.open(path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
    try:
        with env.begin() as t:
            nextid = int(json.loads(zlib.decompress(t.get(b"nextid")).decode()))
            if t.get(b"deleted_ids") is not None:
                raise SystemExit(f"[data_prep] {path}: has deleted rows; not a release file")
    finally:
        env.close()
    rows = nextid - 1
    if rows % 3:
        raise SystemExit(f"[data_prep] {path}: {rows} rows, not a multiple of 3")
    return rows // 3


def split_tids(subset: str, root: Path | None = None) -> dict[str, list[int]]:
    """``{split: tids}`` for one subset, from the directory each file is in."""
    _check_subset(subset)
    files = subset_shards(subset, root)
    out: dict[str, list[int]] = {sp: [] for sp in SPLITS}
    t0 = 0
    for f in files:
        n = _triplets_in_file(f)
        out[Path(f).parent.name].extend(range(t0, t0 + n))
        t0 += n
    for sp in SPLITS:
        if len(out[sp]) != EXPECTED_TRIPLETS[subset][sp]:
            raise SystemExit(f"[data_prep] {subset}/{sp}: {len(out[sp])} triplets on disk, "
                             f"expected {EXPECTED_TRIPLETS[subset][sp]}")
    return out


def load_official_splits(subset: str = "mp20bat", *, accelerator_state=None
                         ) -> tuple[list[int], list[int], list[int]]:
    """``(train_tids, val_tids, test_tids)`` of one subset (see module docstring)."""
    d = split_tids(subset)
    print(f"[data_prep] {subset} splits: train={len(d['train']):,}  val={len(d['val']):,}  "
          f"test={len(d['test']):,}")
    return d["train"], d["val"], d["test"]


def tid_to_saddle_msid(subset: str, root: Path | None = None):
    """numpy array ``msid[tid]`` = the saddle ``ms_id`` of every tid of a subset,
    from ``metadata/triplets.parquet``."""
    import numpy as np
    import pyarrow.parquet as pq
    root = root or materials_saddles_root()
    files = subset_shards(subset, root)
    fidx = {os.path.relpath(f, root): i for i, f in enumerate(files)}
    counts = np.array([_triplets_in_file(f) for f in files], np.int64)
    cum = np.concatenate([[0], np.cumsum(counts)])
    t = pq.read_table(str(root / "metadata" / "triplets.parquet"),
                      columns=["saddle_ms_id", "file", "reactant_row_id"],
                      filters=[("subset", "==", subset)]).to_pydict()
    msid = np.full(int(cum[-1]), -1, np.int64)
    fi = np.array([fidx[f] for f in t["file"]], np.int64)
    row = np.asarray(t["reactant_row_id"], np.int64)
    if ((row - 1) % 3).any():
        raise SystemExit("[data_prep] metadata reactant_row_id not of the form 3k+1")
    tid = cum[fi] + (row - 1) // 3
    msid[tid] = np.asarray(t["saddle_ms_id"], np.int64)
    if (msid < 0).any() or len(tid) != len(msid):
        raise SystemExit(f"[data_prep] metadata/triplets.parquet does not cover every tid of {subset}")
    return msid


def load_local_triplet_splits(
    shards_dir, manifest_csv, *, accelerator_state=None,
) -> tuple[list[int], list[int], list[int]]:
    """Split loader for a *local* triplet dataset (not on Hugging Face).

    Reads the split assignment from a CSV manifest with columns ``ms_id_R`` and
    ``split`` (the ``dataset1_split_manifest.csv`` of the lemat-bulk family) and
    the shards directly from ``shards_dir``. Returns ``(train_tids, val_tids,
    test_tids)`` as positional triplet indices over the sorted shards. Assumes each
    shard's R-row ms_ids are ``first_ms_id + 3*j`` (true for those local shards —
    NOT for the Hugging Face release, which uses :func:`load_official_splits`).
    """
    import csv
    from ase.db import connect

    shard_paths = sorted(Path(shards_dir).glob("*.aselmdb"))
    if not shard_paths:
        raise SystemExit(f"[data_prep] no *.aselmdb shards under {shards_dir}")
    msidR_to_tid: dict[int, int] = {}
    tid = 0
    for sp in shard_paths:
        db = connect(str(sp), type="aselmdb", readonly=True, use_lock_file=False)
        try:
            row_count = db.count()
            if row_count % 3 != 0:
                raise SystemExit(f"[data_prep] {sp}: row count {row_count} not a multiple of 3.")
            n = row_count // 3
            first_ms = int(next(db.select(limit=1)).data["info"]["ms_id"])
        finally:
            db.close()
        for j in range(n):
            msidR_to_tid[first_ms + 3 * j] = tid + j
        tid += n

    out: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    unmatched = 0
    with open(manifest_csv) as f:
        reader = csv.DictReader(f)
        if "ms_id_R" not in reader.fieldnames or "split" not in reader.fieldnames:
            raise SystemExit(f"[data_prep] {manifest_csv}: expected columns 'ms_id_R' and "
                             f"'split', got {reader.fieldnames}.")
        for row in reader:
            t = msidR_to_tid.get(int(row["ms_id_R"]))
            if t is None:
                unmatched += 1
                continue
            split = row["split"].strip()
            if split in out:
                out[split].append(t)
    if unmatched:
        print(f"[data_prep] WARNING: {unmatched} manifest rows had an ms_id_R "
              f"not present in the local shards (manifest/shards mismatch?).")
    for k in out:
        out[k] = sorted(out[k])
    print(f"[data_prep] local manifest splits ({manifest_csv}): "
          f"train={len(out['train']):,}  val={len(out['val']):,}  "
          f"test={len(out['test']):,}  (of {tid:,} triplets in {len(shard_paths)} shards)")
    return out["train"], out["val"], out["test"]


# ----- CLI: run this file standalone to pre-stage / check data on a new machine -----

def _cli():
    import argparse
    p = argparse.ArgumentParser(description="Stage MaterialsSaddles v2 subsets (idempotent) and "
                                            "report their split sizes.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--subset", choices=sorted(EXPECTED_TRIPLETS))
    g.add_argument("--all", action="store_true")
    p.add_argument("--max-workers", type=int, default=32)
    args = p.parse_args()
    subsets = list(ALL_SUBSETS) if args.all else [args.subset]
    ensure_subsets(subsets, max_workers=args.max_workers)
    for s in subsets:
        load_official_splits(s)


if __name__ == "__main__":
    _cli()
