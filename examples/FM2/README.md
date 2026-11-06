# FM2: how the released model was trained

FM2 is the SaddleFlow foundation model released as [SciLM/SaddleFlow](https://huggingface.co/SciLM/SaddleFlow), revision
`fm2` (EMA weights of the run's final checkpoint). It is FM1, the model of the MaterialsSaddles paper (revision `main`,
commit `f003c26`), trained for one more epoch on the same data with two changes: noise on a chosen set of atoms instead of
every atom, and a per-atom flag that tells the network which atoms were noised. At inference the flag lets a user say which
atoms should react, or flag every atom.

This directory records the run exactly (`provenance/`) and gives a script that repeats it with this repository's code
(`train_fm2.sh`).

## What changed from FM1

| | FM1 | FM2 |
|---|---|---|
| initial weights | UMA-M-1.1 | FM1 (370 tensors loaded; the 24 new ones are the zero-initialised flag layers and the FiLM alias) |
| training starts | x₀ = E + u (S − E) + ε, u ~ U(0, 1)², ε on every mobile atom | x₀ = R + u (S − R) + ε, u = 0 half the time, else U(0, 1); ε only on the movers |
| noised atoms (movers) | all mobile atoms | per sample: 20 % all mobile atoms; 20 % those within U(3, 5) Å of the most-moving atom; 60 % those moving more than a threshold drawn from U(0, 1) Å; the most-moving atom always |
| flag input | none | 0/1 per atom, a zero-initialised linear map into the ℓ = 0 channels at every time-FiLM block (`--inject-flag`) |
| rigid drift | kept | bulk systems (no fixed atoms): saddle and partner shifted so their mean displacement from the start is zero (`--align-com`) |
| inference | ×5 K4 | ×4 K5 (the two tied in a paired test on oc22) |

Everything else is FM1's recipe: UMA-M-1.1 with every block trained, time-FiLM at blocks 0–9, head depth 3, unconditional
(no product input), Huber δ 0.05 Å, x_t perturbation 0.05 Å with the convergent target, CoM-symmetric loss, backbone LR
3 × 10⁻³ and head LR 10⁻³ with a 250-step warmup and cosine decay, fp32, all four v2 subsets with oc20 × 2, oc22 × 8 and
mp20bat × 8 (67.7 M records, one epoch), oc22 routed through the oc20 task, global batch 512.

## The run

64 GH200 nodes, one GPU and 8 records each, 132,233 steps. Started on 2026-10-25 from FM1's weights, stopped at the
allocation's end, resumed from `checkpoint_step_00048000` in a new 64-node job (the checkpoint carries the optimizer, the
schedule and the EMA), finished on 2026-10-28 (2,418 min for the resumed part).

## Reproducing it

```bash
# inside a Slurm allocation, on its first node; data from examples/MaterialsSaddles/data_prep.py
MATERIALSSADDLES_ROOT=<v2 data> OUT=<output dir> bash examples/FM2/train_fm2.sh
# after an interruption, in a new allocation with the same number of GPUs:
MATERIALSSADDLES_ROOT=<v2 data> OUT=<output dir> RESUME_FROM=<output dir>/gen/checkpoint_step_NNNNN bash examples/FM2/train_fm2.sh
```

The script downloads FM1 from Hugging Face (accept the model's terms and log in first) unless `FM1` points at a directory
with its `config.json` and `model.safetensors`. It runs one rank per GPU (`GPUS_PER_NODE`, default 1) at 512 / (number of
GPUs) records each, so the global batch and the schedule stay those of FM2. Never resume on a different GPU count: warmup and
the cosine horizon are counted in per-rank scheduler ticks. Each checkpoint is about 22 GB.

## `provenance/`

| file | what it is |
|---|---|
| `scripts/start_ft.sh` | started the run (`start_ft.sh 19`: recipe 19 adds the ball sets); sets FM2's arguments |
| `scripts/launch_full_slot_d.sh` | the launcher it calls (`gen 0 64 full_64b/ft_run19m`): the base command, which those arguments override (later arguments win) |
| `scripts/resume_ft.sh`, `scripts/start_resume_ft.sh` | resumed the run in the next 64-node job |
| `scripts/night_lib.sh`, `scripts/msenv.sh` | sourced by them: paths (FM1's weights in `REL`) and environment |
| `config.json` | the configuration the run recorded (every argument) |
| `run_log_excerpts.txt` | the launchers' resolved settings and the log lines of the start from FM1, the resume and the end |
| `code_snapshot.diff` | from this branch's code to the code snapshot that ran |

The scripts are verbatim, with the cluster paths and job IDs of the run; files they source outside this directory are
not included. The run used a code snapshot outside git; this
branch carries the same training code without the experimental options FM2 left off (line, seed and patch starts, mover
editing, background noise and others; see `code_snapshot.diff`). It has no `--mover-base` option: starts are always
built from the reactant, as FM2's `--mover-base start`. Every
argument of `train_fm2.sh` parses with this branch's `train.py` to the values in `config.json`. The order of random draws
was not checked to be identical, so a rerun gives an equivalent model, not the same weights.
