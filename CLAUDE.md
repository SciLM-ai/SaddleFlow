# CLAUDE.md

Guide for anyone (person or coding agent) working on SaddleFlow. Machine- or person-specific notes (cluster paths,
job logs, unpublished experiments) belong in an untracked `CLAUDE.local.md` next to this file; it is git-ignored and
must never be committed.

## What SaddleFlow is

SaddleFlow generates transition-state (first-order saddle) structures for periodic materials by **flow matching**.
The velocity field is Meta FAIR's pretrained **UMA** backbone (`uma-s-1p2` or `uma-m-1p1`) with an equivariant
**time-FiLM** at selected backbone blocks (all of them in the paper model), an optional global-attention layer
(`GlobalAttn`, off in every released recipe), and a small SO(3)-equivariant `VelocityHead` that outputs a per-atom
velocity. Package import name `saddleflow`; written
**SaddleFlow** in prose.

Two ways to use it:

- **Unconditional, single-ended (Dimer-like).** Only the reactant is known. Start from the reactant plus Gaussian noise
  and flow to a nearby first-order saddle; the head never sees the product (`--delta-endpoint-channels 0`). This is
  the model in the MaterialsSaddles paper (NeurIPS 2026, Evaluations & Datasets), trained in `examples/MaterialsSaddles`.
- **Conditional, double-ended (NEB-like).** Both endpoints (R, P) are known. Start at their minimum-image midpoint and
  give the head the per-atom displacement to each endpoint at every step (`--delta-endpoint-channels 32`). Recipe in
  `examples/MP20Bat`.

## Repository layout

- `saddleflow/data/` — datasets (`TrajTripletDataset` for `.traj`, `AseDbSaddleDataset`, `MaterialsSaddlesDataset`
  for the released ASE-LMDB shards), coordinate transforms (`wrap_positions`, `mic_displacement`).
- `saddleflow/flow/` — `FlowMatchingConfig`, `FlowMatchingLoss`, start-distribution sampling, sampler.
- `saddleflow/models/` — `TimeFiLMBackbone`, `VelocityHead`, `GlobalAttn`.
- `saddleflow/utils/` — backbone loading, the `accelerate` training loop (`training.py`), EMA, evaluation helpers.
- `examples/` — `MaterialsSaddles` (paper model, data download, evaluation), `MP20Bat` (conditional recipe),
  `LiC` and `LiC_simpler` (small unconditioned examples with their own READMEs), `LematBulk`, `scaling_bench`.
- `tests/` — `pytest tests`.

## The paper model (unconditional, all four MaterialsSaddles subsets)

**Data.** `python examples/MaterialsSaddles/data_prep.py` downloads MaterialsSaddles from Hugging Face
(`SciLM/MaterialsSaddles`) into `$MATERIALSSADDLES_ROOT` (default `$SCRATCH/MaterialsSaddles_v2`). The release is
deduplicated and split 90/5/5 by element set into `<subset>/{train,val,test}/` directories; no chemical system
appears in two splits.

**Training command** (one epoch over all subsets with oc20 ×2 and oc22, mp20bat ×8 = 67.7 M records; 64 GH200 GPUs,
8 records per GPU, global batch 512, ~132 k steps). Launch one process per GPU with the SPMD pattern of
`examples/MaterialsSaddles/run.sh`:

```
python examples/MaterialsSaddles/train.py --all-subsets --output-dir <out> \
  --subset-repeat oc22=8,mp20bat=8,oc20=2 --task-name-map oc22=oc20 \
  --backbone uma-m-1p1 --mixed-precision no --grad-clip-norm 1.0 --batch-size 8 --num-epochs 1 \
  --learning-rate 1e-3 --uma-lr 3e-3 --warmup-steps $((250 * WORLD_SIZE)) --ema-decay 0.9996 \
  --loss-type huber --huber-delta 0.05 \
  --delta-endpoint-channels 0 --path-start-prob 1.0 --path-noise-sigma 0.3 --path-u-power 2.0 \
  --save-every-steps 2000 --val-every-steps 2000 --max-val-records 4000 \
  --unfreeze-uma-all --early-time-film-blocks 0,1,2,3,4,5,6,7,8,9 --head-depth 3 --attn-layers 0 \
  --com-symmetric-loss --xt-target-correction --xt-perturb-sigma 0.05 \
  --no-inject-force --no-dimer-residual --no-frozen-force-backbone --no-endpoint-features --eigenmode-aux-weight 0
```

**Why each choice.**
- **Path starts** (`--path-start-prob 1`). Each sample starts on the line from an endpoint E (reactant or product) to its
  saddle S plus noise: `x0 = E + u (S − E) + ε`, `ε ~ N(0, σ²)` on mobile atoms. Training only from `S + ε`
  (transition-state denoising) fails for this task: in 3N dimensions an isotropic draw almost never points along the
  reaction, so the model learns to undo noise and stays at the minimum.
- **`u ~ U(0,1)²`** (`--path-u-power 2`) biases starts toward the endpoint, where inference begins.
- **σ = 0.3 Å matches the inference noise.** Never deploy wider than the training σ (the model degrades sharply);
  narrower is safe. Ship the training σ with the weights.
- **x_t perturbation (0.05 Å)** is not redundant with the start noise: ε enters `x_t` scaled by `(1 − t)` and fades near
  the saddle, where the model's own endpoints land off the line.
- **Convergent target** (`--xt-target-correction`, `t_floor = 0.1`): `v = MIC(S − x_t)/(1 − t)` for `t ≤ 0.9`,
  `S − x0` above (perturbation off there).
- **Huber δ = 0.05** approximates the median of a multi-modal target where MSE averages the modes.
- **Backbone LR 3e-3 on UMA-M** (1e-2, the UMA-S optimum, diverges on UMA-M); head/FiLM LR 1e-3.
- **`--task-name-map oc22=oc20`**: `uma-m-1p1` has no `oc22` mixture-of-experts entry and raises `KeyError('oc22')`.

**Inference: ×5 K4.** Spend 20 network evaluations as five passes of four Euler steps, each pass restarting at `t = 0`
from the previous endpoint, rather than one pass of 20 (×1 K20). A single pass stops short of the saddle; because the
training starts span the whole endpoint–saddle line, the same model also knows how to refine its own endpoint.

```
SF_NAPPLY=5 python examples/MaterialsSaddles/dump_throws_batched.py \
  --data-glob "$MATERIALSSADDLES_ROOT/<subset>/*/<subset>_*.aselmdb" --subset <subset> --split test \
  --ckpt <ckpt> --use-ema --task-name-map oc22=oc20 --sides 0 --sigma 0.3 --K 4 --seed 0 --tag ev --outdir <dir>
```

Then converge every endpoint with Sella (see Evaluation) and collect with `sm_collect.py`.

## The conditional recipe (`examples/MP20Bat`)

UMA-S-1.2 with all four blocks unfrozen (LR 1e-4) and time-FiLM at every block; `VelocityHead(depth=3,
delta_endpoint_channels=32)` fed `(Δ_R, Δ_P) = MIC(R − x_t), MIC(P − x_t)` stacked as `(N, 2, 3)`; `x0 = (R + P)/2`
(minimum-image midpoint); x_t perturbation 0.05 Å with the convergent target; CoM-symmetric loss. `run.sh` holds the
exact hyperparameters. Later work found Huber loss and a much higher backbone LR (3e-3 to 1e-2) clearly better for
this recipe. Score conditional models against the dataset label (they are asked for that specific saddle).

## Architecture

**Backbone (UMA).** Loaded by `saddleflow.utils.load_uma_backbone`; only `.backbone` is used, the energy/force heads are
discarded. UMA-S-1.2: 4 blocks, lmax 2, 128 channels, embedding `(N, 9, 128)`. UMA-M-1.1: 10 blocks, lmax 4,
128 channels, embedding `(N, 25, 128)`, 1.4 B parameters. With a 6 Å cutoff, even the 4 UMA-S blocks reach ~24 Å,
beyond most released cells; this is why `GlobalAttn` stays off. Mixture-of-experts routing comes from each record's
`task_name` (`omat`, `oc20`, `oc22`, ...) via `csd_embedding(charge, spin, task)`; `charge`/`spin` default to 0.

**Time-FiLM.** A zero-initialised equivariant FiLM is hooked before each selected block (`TimeFiLMBackbone`) and in the
head: an additive bias on `l = 0` channels and a `(1 + γ)` gate on `l ≥ 1` channels, both functions of `t` only, so
equivariance is exact. The bias alone would never reach `l ≥ 1` outputs (`SO3_Linear` keeps the `l` paths separate);
the gate is what lets time act on the vector channels. At init the wrapped model equals UMA bit for bit.

**VelocityHead.** Built from UMA's `SO3_Linear` (not e3nn's `o3.Linear`; the two tensor layouts are not
interchangeable) with `depth − 1` blocks of `SO3_Linear → UMAGate` before the `l = 1` readout. Optional zero-initialised
inputs (endpoint displacements, forces) leave it identical to a plain force head when off.

**Output projections, in order.** (1) `v[fixed] = 0`. (2) Subtract the mobile-atom mean velocity **only in systems with
no frozen atoms** (`_com_projection_batched`). Do not make this unconditional: in a system with one mobile atom it
zeroes that atom's velocity and silently kills training.

## Flow formulation

- **State:** Cartesian positions in Å; UMA always sees physical positions, wrapped into the cell just before each
  forward pass. Interpolation happens in unwrapped space.
- **Minimum-image unwrap** of every target relative to the start (once, when a record is built):
  `Δs = inv(cell) @ (r_target − r_start); Δs −= round(Δs); r_target_un = r_start + cell @ Δs`.
- **R↔P doubling:** every triplet gives two records, one per endpoint as start; the unwrap is recomputed per anchor.
- **Start distributions** (`FlowMatchingConfig`): `(R+P)/2` midpoint (conditional), transition-state denoising
  (`ts_denoise_sigma`), path starts (`path_start_prob`, `path_noise_sigma`, `path_u_power`, optional σ schedule
  `path_noise_sigma_endpoint`/`_saddle`). All new options default off.
- **Path:** `x_t = (1 − t) x0 + t S`, `t ~ U(0, 1)`; target as above.
- **Inference:** forward Euler, `K` steps per pass, optional restarts (`SF_NAPPLY` in the dumpers). Higher-order
  integrators and larger `K` do not help; the error is in the field, not the integration.

## Data formats

All sources reduce to the same per-record dict: `start_pos`, `partner_un_pos`, `saddle_un_pos` (both unwrapped to the
start), `Z`, `cell`, `fixed`, `task_name`, `charge`, `spin`, `delta_norm`, `role` (`R2S`/`P2S`), `triplet_id`,
`metadata`. Triplets are stored as consecutive `[R, S, P]` frames. Small problems read `.traj` directly
(`TrajTripletDataset`); MaterialsSaddles reads ASE-LMDB shards (`MaterialsSaddlesDataset`, which also applies
`task_name_map`). Neighbour lists are rebuilt every forward pass (`AtomicData.from_ase`), since bonds change along the
flow.

## Training infrastructure

Plain PyTorch + Hugging Face `accelerate` (`saddleflow/utils/training.py`); no Hydra or Lightning.

- **Multi-node: SPMD, one process per GPU** (`RANK=$SLURM_PROCID`, `WORLD_SIZE=$SLURM_NTASKS`, static
  `init_process_group`), as in `examples/MaterialsSaddles/run.sh`. Do not use `accelerate launch --num_machines`
  across many nodes: its elastic rendezvous strands stragglers. `examples/MP20Bat/run.sh` still uses the old pattern.
- **Warmup and `--max-steps` count scheduler ticks, and accelerate ticks once per rank per optimizer step.** For a real
  warmup of W optimizer steps pass `--warmup-steps W × world_size`. `--max-steps` is converted internally.
- **Fixed global batch, free node count.** Changing nodes at fixed global batch changes nothing but speed; never
  resume a run on a different node count (the schedule is per-rank ticks).
- **Resume** (`--resume-from <checkpoint>`) continues mid-epoch with the same shuffle and skips trained batches.
  `--init-weights` loads weights only (fresh optimizer and schedule) and re-snapshots the EMA.
- **EMA:** choose `d = 1 − 1/window` with `window ≈ total_steps/50`. Too high a decay on a short run leaves the shadow
  at the initial weights (evaluation then looks broken; compare with live weights).
- **Large validation sets:** cap them with `--max-val-records`.
- **Precision:** fp32 is the safe default (the paper model is fp32). bf16 autocast is acceptable for UMA-S training
  because the coordinate helpers are pinned to fp32 (see bug log).

## Evaluation

- **Score with Sella, never with ASE Dimer.** Dimer wanders to a different saddle in a sizeable fraction of cases and
  fabricates a tail. Run Sella (via SaddleMill, `method = Sella`; example `examples/Sella_OC22` in
  `SciLM-ai/SaddleMill`) from each prediction: UMA-S-1.2 with the subset's own task (`omat` for lemat/mp20bat,
  `oc20`, `oc22`), first order, `fmax = 0.01` eV/Å, ≤ 300 steps, `delta0 = 0.1`, index check (exactly one eigenvalue
  below −0.01 eV/Å²). For SaddleFlow outputs use `dataset_type = bulk`: they carry their own `FixAtoms` and all-zero
  tags, and `oc` would re-freeze every tag-0 atom.
- **Check the task name against the subset before scoring.** Scoring through the wrong expert inflates errors.
- **Metric:** maximum per-atom displacement (minimum image) from the prediction to the saddle Sella reaches, over cases
  that converge and verify as first order; report mean and median and the Sella force calls. Unconditional models are
  scored against the Sella saddle, conditional models against the dataset label.
- **Collect strictly** (`sm_collect.py`): count only real Sella convergence and verified index-1; a desorption check
  can mark an unoptimised input as converged at step 0; Sella crashes are failures, not dropped cases; a truncated
  pass keeps the easy cases first and is biased.
- **Compare arms on identical starts** with paired tests. Different noise draws move medians by several hundredths
  of an Å, more than most training changes.

## Latent-bug log — re-check these if behaviour looks wrong

- **Autocast silently quantises coordinates.** `wrap_positions` and `mic_displacement` do a Cartesian→fractional→
  Cartesian round trip; under `torch.autocast` those matmuls run in bf16 (spacing 0.06 Å at 12 Å), so the structure
  fed to UMA is corrupted while the target is exact. Training still reaches a *lower* loss. Both helpers now force
  autocast off (`tests/test_autocast_precision.py`). Rule: never do arithmetic on absolute coordinates under autocast.
  fp16 hides the bug; do not test for it with fp16.
- **UMA dropouts leak into training.** `.train()` enables UMA's composition and mixture dropouts; `FlowMatchingLoss`
  re-calls `backbone.eval()`.
- **Time embedding base.** The transformer base 10000 is ~constant over `t ∈ [0, 1]`; the head uses geometric
  frequencies from 1 to `half` cycles per unit time.
- **RMSD over all atoms dilutes error by √N.** All evaluation helpers take `mobile_mask`.
- **CoM projection** must stay conditional on having no frozen atoms (see Architecture).
- **AdaLN-zero:** the first time-MLP layer has zero gradient at step 0 only; if it persists, something is broken.
- **Force injection uses the first sample's task name for the whole batch.** Fine for homogeneous batches or with
  `--no-inject-force` (all released recipes); wrong for mixed-subset batches with force injection.
- **Multi-node NCCL hang** on some clusters near an epoch boundary (a race between gradient all-reduce buckets):
  `TORCH_DISTRIBUTED_DEBUG=DETAIL` serialises the collectives (~40 % slower) and fixes it; a per-step barrier does not.
- **LiC:** the Li is atom 126 in `LiC` and 112 in `LiC_simpler`; the negative cell `z` is intentional vacuum.

## Contributing

- Work on a branch, open a pull request, merge it; do not push to `main` directly.
- Commit messages: one sentence; add a short body only when the reason is not visible in the diff.
- Figures as PDF.
- Never commit data, checkpoints, or `CLAUDE.local.md`.

## License

MIT — see [`LICENSE`](LICENSE).
