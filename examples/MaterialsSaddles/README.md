# MaterialsSaddles — full-dataset production training & evaluation

Mode-1 product-conditional flow-matching training of **SaddleFlow** on the
**entire MaterialsSaddles dataset** (NEB-CI / Dimer saddles across battery
materials, catalysis, and broad Materials Project / Alexandria / ICSD chemistry),
plus the full post-training evaluation pipeline used for the paper.

Dataset breakdown after staging (`python data_prep.py --all`):

| Subset | Triplets | Train | Val | Test | On-disk |
|---|---:|---:|---:|---:|---:|
| `lemat` | 31,346,419 | 28,211,777 | 1,567,321 | 1,567,321 | 596 GiB |
| `oc20` | 2,587,101 | 2,328,391 | 129,355 | 129,355 | 41 GiB |
| `oc22` | 167,335 | 150,602 | 8,367 | 8,366 | 3.1 GiB |
| `mp20bat` | 34,742 | 31,268 | 1,737 | 1,737 | 596 MiB |
| **Total** | **34,135,597** | **30,722,038** | **1,706,780** | **1,706,779** | **640 GiB** |

With R↔P doubling the training stream is **61.4M records / epoch**.

This bundle reproduces:
- the headline parity figures (PBC-RMSD and max-atom-displacement, log-log)
- the K=10 full-test-set histogram
- the K=10 vs K=50 stability comparison
- the fmax-coloured parity scatter (stationary-point sanity check)

## Architecture

UMA-S-1.2 with **all 4 backbone blocks unfrozen** at low LR (1e-4), 4-block
equivariant time-FiLM, depth-3 `VelocityHead`, no force injection / no eigenmode
auxiliary / no Dimer residual / no endpoint features / no multi-layer feature
stacks. Training uses a hybrid PBC-correct convergent v_target schedule with a
σ=0.05 Å Gaussian perturbation on x_t and a CoM-symmetric MSE loss. See
`train.py --help` and the `## Mode 1 architecture sweep` section of the
top-level `CLAUDE.md` for ablation history.

## Scaling to the full dataset

This example was originally tuned for the `mp20bat` subset (34,742 triplets,
~21 k optimizer steps at global batch 192, ~30 h on 4 nodes × 3 A100s). The
full dataset is **~1000× larger** in training records, so the run config has
been retuned. `run.sh` autoscales LR with the chosen node count; the table
below shows the math.

**Reference point (mp20bat baseline):** global batch 192, head LR 1e-3, UMA-blocks LR 1e-4, 60 epochs ≈ 21,700 steps, ~30 h on 12 GPUs.

### Steps per epoch and wall-clock estimates

Per-record cost on mp20bat measured at **~0.31 GPU-s/record** (12 GPUs × 30 h / 4.17 M record-passes). Average bytes-per-triplet across the full dataset is similar to mp20bat (~17–20 KiB/triplet for every subset), so we assume the per-record cost generalizes; total compute per epoch ≈ 61.4 M × 0.31 = **~5,300 GPU-h per epoch**.

| Allocation | Per-GPU bs | Global batch | Steps / epoch | 3-epoch wall-clock |
|---|---:|---:|---:|---:|
| 16 nodes × 4 GPUs = 64 GPUs | 8 | 512 | 120 k | ~250 h (≈ 10.4 d, needs 5–6 sbatches) |
| 32 nodes × 4 GPUs = 128 GPUs | 8 | 1024 | 60 k | ~124 h (≈ 5.2 d, needs 3 sbatches) |
| 64 nodes × 4 GPUs = 256 GPUs | 8 | 2048 | 30 k | ~62 h (needs 2 sbatches) |
| **128 nodes × 4 GPUs = 512 GPUs** (run.sh default, bs=4) | 4 | 2048 | 30 k | **~31 h — single sbatch** ✓ |
| 256 nodes × 4 GPUs = 1024 GPUs (bs=2) | 2 | 2048 | 30 k | ~16 h |

The 128-node default is the sweet spot:
- **Fits one sbatch** (well under the 48 h `regular` cap).
- **Clears NERSC's ≥128-node scale-discount bar** → ~15 % less allocation charge vs. the same total GPU-h spent on a smaller-node chain.
- **Training-efficient**: global batch 2048 is 10.7× the mp20bat baseline, sqrt-scaled LR lands at 3.27e-3 (well below the regime where LARS/LAMB would be needed in place of AdamW).
- Step-based checkpointing (`--save-every-steps 2000`) writes ~45 saves over the run, ~40 min apart — an interrupt loses well under an hour.

### LR and batch-size scaling rule

`run.sh` derives head LR from the global batch via **sqrt scaling**:

```
HEAD_LR = 1e-3 × √(global_batch / 192)
```

Sqrt (not linear) because UMA is pretrained: linear scaling would push LR to 2.7e-3 at global 512 / 5.3e-3 at 1024, high enough to risk destroying the pretrained features. Sqrt gives 1.6e-3 / 2.3e-3 — empirically stable in this regime.

**UMA-blocks LR stays fixed at 1e-4 regardless of batch size.** It's the discriminative-LR anchor; scaling it scales the destruction-of-pretrained-features risk one-for-one.

### Other hyperparameters that change at scale

| Hyperparameter | mp20bat | full-dataset | Why |
|---|---|---|---|
| `--batch-size` (per-process) | 16 | **4** | At -N 128 this gives global 2048. Lemat unit cells reach ~100 atoms; bs=4 leaves comfortable A100-40GB headroom. Bump to 8 (halves step count, doubles global batch) only after smoke confirms memory holds. |
| `--warmup-steps` | 1000 | **4500** | 5 % of the 90 k-step total. Long warmup matters at large global batches. |
| `--ema-decay` | 0.9995 | **0.99992** | Half-life ≈ ln 2/(1−decay) = 8.7 k steps ≈ 9.6 % of the 90 k total → middle of the Karras 5–20 % band. If you change step count substantially, recompute: target half-life ≈ 0.10 × total_steps. |
| `--num-epochs` | 60 | **3** | Big-data training: 1–3 epochs is usually enough; 60 would be GPU-years. |
| `--save-every-steps` | 0 | **2000** | New flag. 45 saves over the 31 h run, ~40 min apart — an interrupt loses < 1 h of work. |
| `--save-every-epochs` | 1 | **1** | Unchanged; step-based saves are the workhorse, epoch saves are convenient anchors. |

### Eval

Each subset is a distinct reaction domain (mp20bat = Li-ion batteries, oc20/oc22 = catalysis on slabs, lemat = broad materials chemistry), so the eval scripts stay **per-subset** — run them once per subset rather than reporting one combined number. The paper-relevant script is `eval_full_testset_K10.py` (RMSD parity on the full test split at K=10). `run.sh` does **not** auto-run eval; submit it manually after training:

```bash
for s in mp20bat oc22 oc20 lemat; do
  accelerate launch --num_processes $NUM_PROCS --multi_gpu --mixed_precision bf16 \
    eval_full_testset_K10.py --ckpt-dir $RUN/checkpoint_final --subset $s --K 10
done
```

`eval_full_testset_K10.py` shards the test set across processes (`my_indices = list(range(n_total))[rank::num_processes]`) and merges per-rank pickles at the end, so it scales linearly with `--num_processes`. Use `--num-cases N` to subsample — lemat alone has 1.57 M test triplets, so the full sweep on lemat is multi-day even at 64 GPUs; for a paper-style parity figure 5–20 k cases is usually plenty.

## Files

| File | What it does |
|---|---|
| `train.py` | Trains the model. Multi-node-aware via `accelerate`. |
| `data_prep.py` | Idempotent dataset stage + official-split loader. |
| `eval_full_testset_K10.py` | Runs Mode-1 deterministic sampling at K=10 across the full mp20bat test split, writes `results.npz` and histograms. |
| `compare_K10_K50.py` | Same model, paired K=10 vs K=50 predictions on a random subset; quantifies integration-error sensitivity. |
| `replot_parity.py` | Reads `results.npz` from `eval_full_testset_K10.py`'s output dir and writes the paper's `parity_maxdisp_log.{png,pdf}` and `parity_all_atoms_log.{png,pdf}`. No re-sampling. |
| `analysis_fmax_parity.py` | Reads `cases.pkl` from `sample_and_distance_eval.py`'s output, runs UMA on every prediction, writes `parity_fmax.png` (parity scatter coloured by fmax at the predicted structure). Helps spot cases where the prediction is far from the labelled saddle but is itself a stationary point. |
| `run.sh` | SLURM driver: trains only. Eval is run separately after (see Scaling section above). |

## Layout on disk

```
$SCRATCH/SaddleFlow_MaterialsSaddles/                         ← run root (override with $SADDLEFLOW_RUN_ROOT)
└── runs/MaterialsSaddles_<TIMESTAMP>/
    ├── config.json
    ├── dataset_stats.json
    ├── history.json
    ├── checkpoint_epoch_NNNNN/
    └── checkpoint_final/
        ├── model.safetensors  ema.pt
        ├── sample_distance_eval/      ← Phase 2 of run.sh
        │   ├── results.npz  cases.pkl  summary.json
        │   ├── parity_all_atoms{,_log}.png
        │   └── trajs/case*.traj
        └── full_testset_K10/          ← from eval_full_testset_K10.py
            ├── results.npz  cases.pkl  summary.json
            ├── hist_*.png
            └── parity_*.{png,pdf}     ← from replot_parity.py

$SCRATCH/MaterialsSaddles/                                   ← dataset, auto-staged
├── mp20bat/*.aselmdb                                         ← 32 LMDB shards
├── splits/mp20bat/{train,val,test}.parquet                   ← official ms_id splits
└── .msid_cache_mp20bat.json                                  ← built once
```

## Quick start

### 0. Environment

Required:
- `$SCRATCH` — fast scratch path. Most Slurm sites set this automatically.
- `$WORK` — long-term project path.
- `python` with `torch`, `accelerate`, `fairchem-core`, `ase`, `pyarrow`,
  `huggingface_hub` (and a `HF_TOKEN` env var if HF rate-limits you on the
  first download).

Optional overrides used by `run.sh`:

| Var | Default | Purpose |
|---|---|---|
| `SADDLEFLOW_PYTHON` | `python` | which python on `$PATH` |
| `SADDLEFLOW_REPO` | `<script>/../..` | root of the SaddleFlow repo |
| `SADDLEFLOW_RUN_ROOT` | `$SCRATCH/SaddleFlow_MaterialsSaddles` | where `runs/` is created |

### 1. Smoke test (single-node allocation)

```bash
SMOKE=1 bash run.sh
```

Runs 1 epoch × 4 triplets × 2 eval cases in ~2 minutes — exercises the full
training and eval code path.

### 2. Full training

Edit `#SBATCH -A` to your cluster allocation if it isn't already
`m1883_g`. The script's `#SBATCH -o/-e` already write to
`$SCRATCH/SaddleFlow_MaterialsSaddles/logs/`, so you can submit from anywhere —
but **`SLURM_SUBMIT_DIR` must be `examples/MaterialsSaddles/`** for `run.sh` to
find `train.py`, so cd there first:

```bash
cd /path/to/SaddleFlow/examples/MaterialsSaddles
sbatch run.sh
```

To resume from a checkpoint (typical, since runs span multiple sbatches at
128-node scale — node failures and SLURM walltimes are routine at this scale):

```bash
cd /path/to/SaddleFlow/examples/MaterialsSaddles
RESUME_FROM=$SCRATCH/SaddleFlow_MaterialsSaddles/runs/MaterialsSaddles_<TS>/checkpoint_step_NNNNNNNN \
    sbatch run.sh
```

Everything heavy (checkpoints, EMA, optimizer state, dataset shards, logs)
lands under `$SCRATCH/SaddleFlow_MaterialsSaddles/`. The repo only holds source.

**Launch mechanism (do not "modernize"):** `run.sh` uses srun-native SPMD —
one srun task per GPU with `RANK=$SLURM_PROCID`, `init_process_group` from
`env://`. We deliberately do NOT use `accelerate launch --num_machines=N`
because torch-elastic's dynamic c10d rendezvous strands stragglers and times
out at ≥128 nodes (verified May 2026). See run.sh's header + CLAUDE.md
"Training infrastructure" for the full rationale.

Expected runtime on 128 nodes × 4 A100 = 512 GPUs (run.sh default): **~21 h
for 2 epochs from a fresh resume of `checkpoint_step_00008000`**, ~5.6 s/step
measured. Fits in a single 48 h `regular`-QOS sbatch with margin for one
NODE_FAIL/resume cycle.

### 3. Post-training analyses (the paper figures)

```bash
RUN=$SCRATCH/SaddleFlow_MaterialsSaddles/runs/MaterialsSaddles_<TIMESTAMP>

# Full test set at K=10 (deterministic Euler, ~30 min on 3 GPUs)
accelerate launch --num_processes 3 --multi_gpu --mixed_precision bf16 \
    eval_full_testset_K10.py --ckpt-dir $RUN/checkpoint_final --K 10

# Paper-style parity PDFs (no re-sampling, just replots from results.npz)
python replot_parity.py $RUN/checkpoint_final/full_testset_K10

# K=10 vs K=50 stability (uses same seed → same triplet selection)
accelerate launch --num_processes 3 --multi_gpu --mixed_precision bf16 \
    compare_K10_K50.py --ckpt-dir $RUN/checkpoint_final --num-cases 100

# fmax-coloured parity (single GPU, ~5 s per case)
python analysis_fmax_parity.py --ckpt-dir $RUN/checkpoint_final
```

## Resuming an interrupted training

Checkpoints land at both `<output-dir>/checkpoint_epoch_NNNNN` (per epoch) and
`<output-dir>/checkpoint_step_NNNNNNNN` (every `SAVE_EVERY_STEPS=2000` steps,
controlled in run.sh). Intra-epoch checkpoints are the resume target after a
NODE_FAIL or walltime — losing only ≤2000 steps of work. After the resume,
`training.py`'s stop-at-`num_epochs × len(dataloader)` cap (added 2026-05-30)
ensures the loop exits at the planned step count rather than re-iterating
`num_epochs` full passes from scratch.

```bash
cd /path/to/SaddleFlow/examples/MaterialsSaddles
RESUME_FROM=$SCRATCH/SaddleFlow_MaterialsSaddles/runs/MaterialsSaddles_<TS>/checkpoint_step_00008000 \
    sbatch run.sh
```

Restores model + optimizer + EMA + scheduler step counter + RNG. LR picks up
mid-cosine at exactly where it left off (no re-warmup spike); the cap fix
means resume(8000) → 30000 is the same model+step count as a continuous
0 → 30000 run.

## Resolved data paths

- Dataset is auto-staged to `$SCRATCH/MaterialsSaddles/mp20bat/` on first run.
- Official train/val/test splits are downloaded as
  `$SCRATCH/MaterialsSaddles/splits/mp20bat/{train,val,test}.parquet` and used
  verbatim — no random splitting on the client side.
- The `ms_id → triplet_id` map is built once on first invocation
  (`$SCRATCH/MaterialsSaddles/.msid_cache_mp20bat.json`).

## Notes on reproducibility

- `seed` controls case selection / direction coin only; the model is fully
  deterministic at inference (`sigma_inf=0`, `n_perturbations=1`).
- The K=10 vs K=50 comparison uses the same seed across both K values, so the
  per-case pair difference is purely the integration error.
