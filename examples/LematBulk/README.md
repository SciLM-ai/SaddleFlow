# lemat-bulk — Mode-1 reference models (1 & 2)

Public Mode-1 training on the **lemat-bulk DATASET 1** (both-converged R/S/P
triplets), `x0 = (R+P)/2` midpoint — the released midpoint→saddle scheme. Same
production recipe as `examples/MaterialsSaddles/run.sh` (full UMA-S-1.2 unfreeze,
4-block time-FiLM, convergent `v_target` `t_floor=0.1`, CoM-symmetric loss),
reusing that directory's `train.py` / `data_prep.py` via two additive flags:

- `--shards-dir <DIR>` — read local `.aselmdb` shards directly (skip HF staging).
- `--split-manifest <CSV>` — train/val/test from `dataset1_split_manifest.csv`
  (joined by `ms_id_R`), instead of the HF parquet splits.

Data (local scratch, not on HuggingFace):
```
shards : /pscratch/sd/i/ilgar/genTS/lematbulk_Elnara_ls6redo/doubleopt/aselmdb_no-EF   (32 shards)
split  : /pscratch/sd/i/ilgar/genTS/lematbulk_Elnara_ls6redo/datasets/dataset1_split_manifest.csv
counts : train 1,677,551 / val 98,924 / test 100,802 triplets  (verified)
```

## The two models

| model | what differs | submit |
|---|---|---|
| **1** (reference) | `xt_perturb_sigma = 0.05` | `sbatch run.sh` |
| **2** (wider off-line) | `xt_perturb_sigma = 0.15` | `sbatch --export=ALL,SIGMA=0.15,TAG=model2 run.sh` |

Everything else is identical, so model 1 vs 2 isolates the effect of a wider
synthetic off-line perturbation tube.

## Sizing (both)

Global batch 4096 (128 nodes × 4 × bs 8), `-t 24:00:00`. `NUM_EPOCHS=18` →
~14.7k optimizer steps (819 steps/epoch × 18). EMA decay **0.99969** (15%-of-run
half-life at ~15k steps; the production 0.99985 is calibrated for ~30k and would
lag here). Head LR sqrt-scaled from the batch (unchanged from production), UMA LR
1e-4. Intra-epoch checkpoints every 1000 steps; resume an interrupted run with
`RESUME_FROM=<…/checkpoint_step_NNNNNNNN> sbatch run.sh`.

> Step count is set via `--num-epochs` (the proven mechanism the paper used —
> there is no `--max-steps`; the cosine horizon is `num_epochs × steps/epoch`).
> If a 24 h wall is hit before epoch 18, resume to finish.

## Smoke test (inside an allocation)
```
SMOKE=1 bash run.sh        # 1 epoch × 8 triplets, exercises the full code path
```

## Eval
Use the MaterialsSaddles eval pipeline against `…/checkpoint_final` (the model
is architecturally identical). See `examples/MaterialsSaddles/README.md`.
