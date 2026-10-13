# SaddleFlow

Generative AI for **transition-state structures** in periodic materials.

SaddleFlow uses flow matching on a fine-tuned [UMA](https://huggingface.co/facebook/UMA) backbone to propose
saddle points. The released model is unconditional, like a Dimer search: from a single minimum plus noise it
flows to a nearby first-order saddle, which Sella then converges in 25–61 force calls instead of 196–352
(MaterialsSaddles paper). A conditional recipe that also takes the product, like NEB, is in
[`examples/MP20Bat/`](https://github.com/SciLM-ai/SaddleFlow/tree/main/examples/MP20Bat/).

## Use the pretrained model

### Install

```bash
pip install "saddleflow[search] @ git+https://github.com/SciLM-ai/SaddleFlow"
python -c "import torch; print(torch.cuda.is_available())"   # should print True
```

**Linux on ARM (aarch64), e.g. NVIDIA GH200:** PyPI has only CPU builds of torch for aarch64, so the line above
silently installs one. Install the CUDA build first, then SaddleFlow:

```bash
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu129
pip install "saddleflow[search] @ git+https://github.com/SciLM-ai/SaddleFlow"
```

`torch==2.13.0` matches the newest supported fairchem-core (2.23); with fairchem-core 2.19–2.21 use `torch==2.8.0`
from the same index.

**Access.** The weights are gated: accept the license at [`SciLM/SaddleFlow`](https://huggingface.co/SciLM/SaddleFlow)
and, to converge proposals with Sella, at [`facebook/UMA`](https://huggingface.co/facebook/UMA); then run
`hf auth login`. The first `from_pretrained()` downloads 5.6 GB.

**Hardware.** One GPU. On one GH200 a proposal (20 network evaluations) takes about 1.8 s on its own, and 16–64
proposals batched together run at 2–3 per second for 40–80-atom cells. GPU memory is about 6 GB for one proposal,
12–15 GB for 16, and 26–34 GB for 64 (`batch_size` sets how many run together). CPU works but is slow. fairchem
compiles kernels on first use and caches them under `~/.triton` and `~/.cache`; on clusters with a small home
quota, set `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, and `XDG_CACHE_HOME` to a scratch directory.

### Run

[`examples/quickstart/`](https://github.com/SciLM-ai/SaddleFlow/tree/main/examples/quickstart/) runs all of this on a real 61-atom battery cell.

```python
from ase.io import read
from saddleflow import SaddleFlow, SaddleFlowSearch, find_saddles

sf = SaddleFlow.from_pretrained()                              # SciLM/SaddleFlow, GPU if available
atoms = read("reactant.traj")
guesses = sf.propose(atoms, n=8, seed=0, task_name="omat")     # 8 ase.Atoms, one per noise draw

# Converge them with Sella on UMA-S-1.2 and keep the distinct first-order saddles.
from fairchem.core import FAIRChemCalculator
from fairchem.core.calculate.pretrained_mlip import get_predict_unit
calc = FAIRChemCalculator(get_predict_unit("uma-s-1p2", device="cuda"), task_name="omat")
saddles = find_saddles(atoms, calc, n=8, model=sf, seed=0)

# Or use it like an ASE optimizer: atoms ends on a verified first-order saddle.
atoms.calc = calc
SaddleFlowSearch(atoms, sf, seed=0).run(fmax=0.01)
```

From the command line: `saddleflow propose reactant.traj -o proposals.traj --n 8` and
`saddleflow search reactant.traj -o saddles.traj --n 8 --task omat`.

Use `task_name="omat"` for bulk materials and `"oc20"` or `"oc22"` for adsorbates on catalyst surfaces. The start
noise defaults to the training value (0.3 Å) and cannot be raised; each proposal is five flow passes of four
Euler steps. `examples/MaterialsSaddles/export_pretrained.py` turns a training checkpoint into this format, and
`check_pretrained.py` checks that it reproduces recorded evaluation endpoints.

## Repository

- `examples/MaterialsSaddles/`: training, evaluation and export of the released model on all four
  MaterialsSaddles subsets.
- `examples/MP20Bat/`: the conditional recipe on the `mp20bat` subset.
- `examples/LiC/`, `examples/LiC_simpler/`: small Li-on-graphene examples for quick experiments.

See [`CLAUDE.md`](https://github.com/SciLM-ai/SaddleFlow/blob/main/CLAUDE.md) for the method and the training recipe.

## Install for training and development

```bash
conda create -n saddleflow python=3.12 -y
conda activate saddleflow
pip install -e '.[train,search]'
```

`train` adds what the training and evaluation scripts in `examples/` need (accelerate, wandb, matplotlib, pyarrow,
lmdb); `search` adds Sella for the saddle-search evaluation below, which is not needed to train or to sample. The
paper model was trained with fairchem-core 2.19 (`pip install 'fairchem-core==2.19.*'` to match it). On aarch64,
install a CUDA build of torch first, as above.

Use **Python 3.12** — fairchem-core builds cleanly on 3.12 but has had wheel issues on 3.11/3.13 on some platforms.

First UMA load downloads the `uma-s-1p2` checkpoint from HuggingFace and requires a valid `HF_TOKEN` (or a
`hf auth login`). If pip builds Sella from source with NVIDIA's `nvc` compiler, set `CC=gcc CXX=g++`.

## Sample from your own conditional checkpoint

```python
import torch
from ase.io import read

from saddleflow.flow.sampler import sample_saddles
# ... load_model is provided per-example; see examples/MP20Bat/sample_and_distance_eval.py

# Load a trained checkpoint (architecture is rebuilt from the run's config.json).
ckpt_dir = "$SCRATCH/SaddleFlow_mp20bat/runs/<RUN>/checkpoint_final"
loss_module, _config = load_model(ckpt_dir, device="cuda", use_ema=True)

# Read reactant and product geometries (ASE-readable formats).
R = read("my_reactant.traj")
P = read("my_product.traj")

# Run deterministic Mode-1 sampling (sigma_inf=0, n_perturbations=1, K Euler steps).
candidates = sample_saddles(
    sample_dict_from_atoms(R, P),    # builds {start_pos, partner_un_pos, Z, cell, fixed, ...}
    loss_module.backbone,
    loss_module.global_attn,
    loss_module.velocity_head,
    sigma_inf=0.0,
    n_perturbations=1,
    K=50,
    device="cuda",
    partner_pos=...,                  # MIC-unwrapped partner positions
    # Plus any optional heads / force backbones the run uses; see eval scripts.
)
```

For end-to-end working examples, see `examples/MP20Bat/sample_and_distance_eval.py` (a 50-case eval that loads a checkpoint, samples saddles for held-out triplets, and writes parity plots and 4-frame `.traj` files for visualization).

## Training your own model

The production example is `examples/MP20Bat/`:

```bash
cd $SCRATCH/SaddleFlow_mp20bat            # so SLURM logs land on scratch
sbatch $WORK/codes/SaddleFlow/examples/MP20Bat/run.sh
```

This trains on the full `mp20bat` subset (~34,742 triplets) for 60 epochs with 4 nodes × 3 A100s, then auto-runs a 50-case post-training eval. Heavy outputs (checkpoints, eval `.npz`, `.traj` files) all land under `$SCRATCH/SaddleFlow_mp20bat/runs/<TIMESTAMP>/`. See [`examples/MP20Bat/README.md`](https://github.com/SciLM-ai/SaddleFlow/blob/main/examples/MP20Bat/README.md) for the full pipeline (training → full-test-set eval at K=10 → K=10 vs K=50 stability → paper-style parity figures).

**For full-dataset / large-scale training (all four MaterialsSaddles subsets — lemat + oc20 + oc22 + mp20bat = 34 M triplets — at 128+ nodes), use [`examples/MaterialsSaddles/`](https://github.com/SciLM-ai/SaddleFlow/tree/main/examples/MaterialsSaddles/) instead.** Its `run.sh` uses srun-native SPMD (each SLURM task = one rank, env-var DDP), which scales reliably past 128 nodes. The MP20Bat launcher above is the proven 4-node paper config; its `accelerate launch --num_machines=N` pattern is fragile at large scale — see [`CLAUDE.md`](https://github.com/SciLM-ai/SaddleFlow/blob/main/CLAUDE.md) "Training infrastructure" for why.

For a quick sanity-check on small data:

- [`examples/LiC_simpler/`](https://github.com/SciLM-ai/SaddleFlow/tree/main/examples/LiC_simpler/) — **start here.** One Li hop on pristine graphene (1 triplet), ~20 min on one GPU. The site is 6-fold symmetric and the data contains only *one* of the six equivalent saddles, so whether the model recovers all six is a visual pass/fail on the physics. **It passes:** with the backbone unfrozen and per-block time-FiLM, all 48 sampled trajectories end in one of the six saddle petals — a six-fold orbit from one training saddle, with hexatic order **0.98–1.00** over two independent runs. That recipe is the script's default; a frozen backbone reaches only 0.91 (see its [README](https://github.com/SciLM-ai/SaddleFlow/blob/main/examples/LiC_simpler/README.md)).
- [`examples/LiC/`](https://github.com/SciLM-ai/SaddleFlow/tree/main/examples/LiC/) — defective graphene: 12 training saddles, 167 held-out ones (each Sella-reconverged at fmax 0.005 and verified index-1), and 10 bond midpoints that look like saddles but are not. An unconditioned model carries a Li thrown anywhere on the sheet to the nearest saddle; ~4 h on one GPU. Its [README](https://github.com/SciLM-ai/SaddleFlow/blob/main/examples/LiC/README.md) reproduces the trajectory figure.

Both use an *unconditioned* model rather than the `(R, P)`-conditioned scheme below: on graphene the midpoint of R and P already sits on the saddle, so conditioning would leave the flow nothing to learn.

## Method (in brief)

- **Flow matching** over atomic Cartesian coordinates with straight-line Optimal Transport from `x_0` to `x_1 = saddle`.
- **Product-conditional anchoring.** `x_0 = (R + P)/2` is the PBC-correct geodesic midpoint of the reactant–product pair, and the velocity head is conditioned on a per-atom Δ_partner = MIC(partner − x_t) at every flow step. This re-parameterises the L2-Bayes-optimal predictor from `E[saddle] − R ≈ midpoint − R` (large) to `E[saddle] − midpoint ≈ 0` (small) — the head only has to predict the residual deviation of the saddle from the midpoint, not rediscover the midpoint itself.
- **Velocity field** = pretrained **UMA-S-1.2** (fairchem, 6.6M active params) as the equivariant backbone, equivariant time-FiLM injected at each backbone block, and a small `VelocityHead` on top. Selectively-unfrozen UMA blocks at low LR (1e-5) let the backbone adapt its layer-N l ≥ 1 outputs to the velocity-prediction task.
- **R↔P doubling.** Each triplet contributes both `(start=R, partner=P)` and `(start=P, partner=R)` training pairs, so the head sees the saddle from both directions and the parity isn't a learned bias.
- **Inference perturbation.** A small Gaussian `ε_inf` on the start position at `t=0` lets multiple integrations from the same `(R, P)` pair land in different angular wedges of the local environment, producing a small ensemble of saddle candidates that can be clustered downstream.

## Evaluation — score against the saddle the prediction actually reaches

RMSD to a stored label answers "is this near *that* saddle", which is not the question. Run a saddle optimiser from the prediction and measure how far it had to move:

```bash
# 1. Integrate the flow and dump candidates (--ckpt2 runs the two-stage cascade).
python examples/MaterialsSaddles/dump_predictions.py \
    --data-glob '<MaterialsSaddles>/mp20bat/*.aselmdb' \
    --ckpt <stage1>/checkpoint_final --ckpt2 <stage2>/checkpoint_final \
    --tag cascade --outdir preds --shard $i --nshards $N

# 2. Optimise each candidate to a saddle and verify it is genuinely first-order.
python examples/MaterialsSaddles/sella_eval.py \
    --src preds/cascade_00.traj --out sella_00.json --check-index
```

Then keep records with `nneg == 1`. Reported per model: distance moved (rmsd/maxd), convergence rate, and force calls — a prediction that is close to a saddle *and* cheap to converge is the useful one.

**Use Sella, not the ASE Dimer.** On identical structures the Dimer walked to a different saddle in ~8% of cases, which silently inflates the error tail (in the worst decile, predictions sat 0.17 Å from the label but 2.21 Å from where the Dimer stopped). Sella converged 100% vs 77%, in 44 vs 384 force calls, with 93% of results verified index-1.

## Built on

- [fairchem](https://github.com/facebookresearch/fairchem) (Meta FAIR) — UMA backbone, `AtomicData`, PBC-aware graph building, ASE Calculator integration.
- [ASE](https://wiki.fysik.dtu.dk/ase/) — atomic simulation environment.
- [PyTorch](https://pytorch.org) + [HuggingFace `accelerate`](https://huggingface.co/docs/accelerate) — training loop, multi-GPU / multi-node scaling.

## Authors

Ilgar Baghishov, Sung Hoon Jung and Graeme Henkelman (The University of Texas at Austin).

## License

The code is MIT — see [`LICENSE`](https://github.com/SciLM-ai/SaddleFlow/blob/main/LICENSE). The released weights are a fine-tuned version of Meta's UMA-M-1.1
and are distributed under the FAIR Chemistry License v1 on the Hugging Face model page.
