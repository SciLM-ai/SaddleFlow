# scaling_bench — cross-machine training throughput & scaling

A portable kit to benchmark **SaddleFlow MP20Bat training** (the real production
step: UMA-S-1.2 full unfreeze, 4-block time-FiLM, bf16) on different GPU clusters
and compare per-GPU throughput, parallel efficiency, memory headroom, and
projected full-run wall-clock. Built to make the hardware case for an allocation:
**A100 (Perlmutter) → GH200 (Vista) → GB200 (Horizon)**.

It times the *real* `examples/MP20Bat/train.py` for a fixed window of optimizer
steps after warmup — same flags as `examples/MP20Bat/run.sh`, just capped via
`--max-steps`/`--bench-warmup`/`--bench-output` (added to the shared loop; no-ops
in normal training). Output is one small JSON per config; `analyze.py` combines
them.

## Files

| file | role |
|---|---|
| `prestage.sh`    | one-time: stage mp20bat (~0.6 GB) + pre-warm the UMA checkpoint |
| `bench_launch.sh`| run ONE config (`NPROC` GPUs) inside an allocation → one JSON |
| `sweep.sh`       | loop GPU counts (and optionally batch sizes) → the full curve |
| `analyze.py`     | combine JSONs → `summary.md` + `scaling.png` + `per_gpu_throughput.png` |

## Run on Perlmutter (A100, 4 GPU/node)

```bash
cd /global/cfs/cdirs/m1883/ilgar/codes/SaddleFlow
bash examples/scaling_bench/prestage.sh            # once (data + UMA cache already warm here)

# IMPORTANT: request 4 tasks/node + 4 gpus/node so srun can subdivide the nodes
# (an salloc defaulting to 1 task/node cannot srun 4 tasks/node — see CLAUDE.md).
salloc -N 4 -C gpu -q interactive -t 01:00:00 -A m1883_g \
       --ntasks-per-node=4 --gpus-per-node=4 \
       bash examples/scaling_bench/sweep.sh

python examples/scaling_bench/analyze.py \
       --results-dir $SCRATCH/saddleflow_bench \
       --out-dir     $SCRATCH/saddleflow_bench/report
```

Default sweep: GPU counts `1 2 4 8 16` at per-GPU batch 16, 200 timed steps each
(+30 warmup). ~15–20 min total. Add a memory-headroom probe with
`BATCH_LIST="16 32 48 64"`.

## Run on Vista (GH200, 1 GPU/node) — the handoff

Vista is **1 GPU per node**, so set `BENCH_GPUS_PER_NODE=1` and request one node
per GPU. Fill in your own python/modules/account/partition.

```bash
# 0. point at your checkout + env
export SADDLEFLOW_REPO=/path/to/SaddleFlow
export SADDLEFLOW_PYTHON=/path/to/your/saddleflow/env/bin/python
export FAIRCHEM_CACHE_DIR=$SCRATCH/fairchem_cache
export BENCH_MACHINE=vista
export BENCH_GPUS_PER_NODE=1
export BENCH_MODULES="cuda"          # whatever your site needs; "" to skip
export HF_HOME=$SCRATCH/hf HF_HUB_CACHE=$SCRATCH/hf/hub   # big pulls → scratch

# 1. one-time stage (needs internet on the node)
bash $SADDLEFLOW_REPO/examples/scaling_bench/prestage.sh

# 2. scaling sweep — 1 GPU/node, so 8 nodes covers 1,2,4,8 GPUs
export GPU_LIST="1 2 4 8"
sbatch -N 8 --ntasks-per-node=1 --gpus-per-node=1 -t 01:00:00 \
       -A <your_account> -p <gpu_partition> \
       --wrap "bash $SADDLEFLOW_REPO/examples/scaling_bench/sweep.sh"
# (or run sweep.sh inside an salloc the same way as Perlmutter)

# 3. send the JSONs in $SCRATCH/saddleflow_bench/ back; combine with Perlmutter's:
python $SADDLEFLOW_REPO/examples/scaling_bench/analyze.py \
       --results-dir $SCRATCH/saddleflow_bench \
       --results-dir /path/to/perlmutter_jsons \
       --out-dir ./report
```

GH200 has 96 GB HBM3e (vs A100 40 GB), so it should fit a much larger batch —
run `BATCH_LIST="16 32 64 96 128"` on 1 GPU to capture that headroom; it's a real
part of the efficiency story (fewer grad-accum steps, higher utilization).

## Fairness notes (so the numbers survive review)

- **Identical workload:** same model/flags/seed, same `LIMIT_TRIPLETS` slice of
  the same mp20bat data on both machines; warmup excluded; window = 200 steps.
- **Don't conflate per-GPU and per-node:** Vista is 1 GPU/node, Perlmutter 4/node.
  The table reports per-GPU and per-node separately. Per-GPU throughput is the
  clean device comparison; parallel efficiency is the interconnect comparison.
- **bf16 only:** these kernels run bf16 on both machines today. The Blackwell
  FP4 path is reported as *upside*, not baseline.
- **Same `summary.json`-style baseline check:** `analyze.py` keys on the GPU name
  read from each JSON — confirm Perlmutter shows A100 and Vista shows GH200.
```
