# The NEB behind `one_saddle.traj`

`../one_saddle.traj` is a `[R, S, P]` triplet whose saddle `S` came out of a
SaddleMill climbing-image NEB. This directory regenerates that saddle from the
triplet's own reactant and product with SaddleMill, as a single serial job on
one GPU, and keeps the outputs of the recorded run so the result can be checked
without rerunning anything.

## Layout

| path | what |
|---|---|
| `make_input.py` | writes `data/LiC_simpler_RP.traj`: frames 0 and 2 of `../one_saddle.traj` (R and P, FixAtoms on the 112 carbons kept). The saddle frame is deliberately **not** included, SaddleMill interpolates the band itself. The training saddle's Li position and barrier are stashed as scalars in `.info` (`ref_saddle_li_pos`, `ref_barrier_eV`) so the output is self-contained. |
| `config.ini` | the SaddleMill run: `executorlib = False` (serial, no flux), `method = NEB`, UMA-S-1.2 (`omat` task) on CUDA. |
| `run.sh` | launcher, see below. |
| `data/` | the input pair. |
| `NEB_trajes/collected_ts_rank_None.traj` | **the recorded result**: the converged 7-image band. `rank_None` because serial mode has no executorlib worker id. |
| `NEB_status_csvs/`, `NEB_debug_zips/`, `traj_files_ordered.json`, `saddlemill.log` | the rest of the recorded run: status (`converged`), the per-step NEB trajectory, optimizer logs and the band plot (zipped), SaddleMill's resume bookkeeping, and the log. |

## Settings, and why

Everything mirrors what the training saddle's own metadata records
(`nimages = 7`, `interpolation_method = ase_linear`, a climbing image, and
image forces that sit just under 0.01 eV/Å):

- 7 images (R + 5 intermediates + P), linear interpolation with minimum-image
  convention, `improvedtangent` NEB with `k = 5` and `climb = True`.
- Band convergence `fmax = 0.01 eV/Å` with MDMin (`dt = 0.05`, `maxstep = 0.1`).
- Endpoints relaxed first with LBFGS to 0.01 eV/Å (they already are, so this is
  a zero-step check).
- OCPNEB batches all interior images through one UMA forward per step and
  zeroes the forces on the FixAtoms indices, so only the Li moves.

## Running it

Needs a SaddleMill checkout and a Python environment with `fairchem-core`,
`ase` and a CUDA build of torch (SaddleMill's "application env"); neither is a
dependency of `saddleflow` itself. UMA-S-1.2 is a gated Hugging Face model, so
the environment must be able to fetch it (or already have it cached).

```bash
bash run.sh          # no-op while the recorded outputs are present (SaddleMill resumes)
bash run.sh fresh    # delete the recorded outputs and rerun from scratch
```

`run.sh` defaults to the Vista paths the run was recorded with; override with
`SADDLEMILL_DIR`, `SADDLEMILL_ENV_BIN` and `FAIRCHEM_CACHE_DIR`. It runs
`python -u -m saddlemill` directly in this directory, on whatever GPU the
current node has. `git checkout -- .` brings the recorded outputs back after a
`fresh` run.

## Recorded result

One GH200, about 20 s wall time including model load, 27 MDMin steps.

| quantity | this NEB | training label |
|---|---|---|
| barrier | 0.3084 eV | 0.3085 eV |
| climbing-image Li vs training saddle Li | 0.0000 Å | |
| climbing-image effective fmax | 0.0047 eV/Å | 0.0047 eV/Å |
| band | `converged`, all 7 images below threshold | |

The band is symmetric about the C–C bond midpoint (images 1/5 and 2/4 pair up
in energy). The output frames carry the usual SaddleMill metadata
(`image_type`, `effective_fmax`, `image_converged`, `band_converged`, ...); the
frame with `image_type == "climbing"` is the saddle and additionally has
`eigenmode`, `barrier` and `dE`. The input `.info` (with the reference values
above) rides along under `orig_info`. Quick check:

```python
from ase.io import read
band = read("NEB_trajes/collected_ts_rank_None.traj", index=":")
ci = next(a for a in band if a.info["image_type"] == "climbing")
print(ci.info["barrier"], ci.positions[112], ci.info["orig_info"]["ref_saddle_li_pos"])
```
