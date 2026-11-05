# FM1: how the paper model was trained

FM1 is the unconditional SaddleFlow foundation model of the MaterialsSaddles paper, released as
[SciLM/SaddleFlow](https://huggingface.co/SciLM/SaddleFlow) (revision `main`; EMA weights of the run's final checkpoint).
[FM2](../FM2) continues from it.

This directory records the run exactly (`provenance/`) and gives a script that repeats it with this repository's code
(`train_fm1.sh`).

## Recipe

UMA-M-1.1 (10 blocks, 1.4 B parameters) with every block trained, an equivariant time-FiLM before each of the 10 blocks and
a depth-3 velocity head; unconditional (the head never sees the product). One epoch over the train split of all four
MaterialsSaddles v2 subsets with oc20 repeated twice and oc22 and mp20bat eight times (67.7 M records; oc22 routed through
the oc20 task, which UMA-M-1.1 lacks). Training starts on the line from an endpoint E to its saddle S,
x₀ = E + u (S − E) + ε with u ~ U(0, 1)² and ε ~ N(0, 0.3² Å²) on every mobile atom. x_t perturbation 0.05 Å with the
convergent target (t_floor 0.1), CoM-symmetric Huber loss (δ 0.05 Å). AdamW, backbone LR 3 × 10⁻³, head and FiLM LR
10⁻³, 250-step warmup and cosine decay, EMA 0.9996, fp32, global batch 512. Inference: five passes of four Euler steps
(×5 K4) from the reactant with σ = 0.3 Å noise on the mobile atoms.

## The run

64 GH200 nodes, one GPU and 8 records each, 132,233 steps. Started on 2026-09-29 in job 1034084 (an attempt minutes
earlier failed on a port already in use), stopped at the allocation's end after step 90,000, resumed from
`checkpoint_step_00090000` in job 1036605 on 2026-10-01 (mid-epoch, same shuffle, trained batches skipped) and finished on
2026-10-02 (1,361 min for the resumed part). Test loss 0.134 (EMA) and 0.148 (live weights) on 4,000 records
(`provenance/test_results.json`).

## Reproducing it

```bash
# inside a Slurm allocation, on its first node; data from examples/MaterialsSaddles/data_prep.py
MATERIALSSADDLES_ROOT=<v2 data> OUT=<output dir> bash examples/FM1/train_fm1.sh
# after an interruption, in a new allocation with the same number of GPUs:
MATERIALSSADDLES_ROOT=<v2 data> OUT=<output dir> RESUME_FROM=<output dir>/checkpoint_step_NNNNN bash examples/FM1/train_fm1.sh
```

The script runs one rank per GPU (`GPUS_PER_NODE`, default 1) at 512 / (number of GPUs) records each, so the global batch
and the schedule stay those of FM1. Never resume on a different GPU count: warmup and the cosine horizon are counted in
per-rank scheduler ticks. Each checkpoint is about 22 GB.

## `provenance/`

| file | what it is |
|---|---|
| `scripts/launch_foundation.sh` | the launcher that ran both parts (`launch_foundation.sh <job> <nodes> runs/FM1`, plus `--resume-from` for the second) |
| `scripts/watch_and_launch.sh` | waited for the 64-node job and started the launcher on its first node |
| `config.json` | the configuration the run recorded (every argument) |
| `history.json`, `test_results.json`, `dataset_stats.json` | validation history, final test loss, dataset statistics |
| `run_log_excerpts.txt` | the launcher's line for each launch and the log lines of the start, the resume and the end |

The scripts are verbatim, with the cluster paths and job IDs of the run. The run used commit `7dc8a1a` of a local
working copy (not on GitHub); its code is identical to `main`'s `4b93656`. Later commits add options FM1 does not use
(FM2's mover starts and flag input among them), all off by default, and every argument of `train_fm1.sh` parses with the
current `train.py` to the values in `config.json` (`force_field_channels` is recorded as the effective 0, since force
injection is off). The order of random draws was not checked against the run, so a rerun gives an equivalent model, not
the same weights.
