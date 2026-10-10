"""Load the released SaddleFlow model and propose saddle points for ASE structures.

    from saddleflow import SaddleFlow
    sf = SaddleFlow.from_pretrained()               # SciLM/SaddleFlow on the Hugging Face Hub
    guesses = sf.propose(atoms, n=8, seed=0)        # list of ase.Atoms, one per noise draw

Each proposal starts from the input structure plus Gaussian noise on its mobile atoms and integrates the
flow; with the released defaults this is five passes of four Euler steps (20 network evaluations), each
pass restarting at t = 0 from the previous endpoint. The proposals are starting points for a saddle
optimizer: `saddleflow.search.find_saddles` and `SaddleFlowSearch` run Sella on them.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from ase import Atoms
from ase.constraints import FixAtoms
from fairchem.core.datasets.collaters.simple_collater import data_list_collater

from .data.transforms import wrap_positions
from .flow.matching import apply_output_projections, build_atomic_data
from .models.time_filmed_backbone import TimeFiLMBackbone
from .models.velocity_head import VelocityHead

DEFAULT_REPO = "SciLM/SaddleFlow"
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"

# Backbone overrides that fairchem's default inference settings apply when the model is loaded with
# get_predict_unit (models/uma/escn_md.py::build_inference_settings); training used that path, so the
# rebuilt backbone must use them too.
_INFERENCE_OVERRIDES = {"always_use_pbc": False, "otf_graph": True, "radius_pbc_version": 2,
                        "activation_checkpointing": True, "use_quaternion_wigner": True}


@contextlib.contextmanager
def _no_tf32():
    """Run in strict fp32: TF32 matmuls move flow endpoints enough to change the saddle they lead to."""
    if not torch.cuda.is_available():
        yield
        return
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


class SaddleFlow(nn.Module):
    """Unconditional SaddleFlow: a fine-tuned UMA backbone with time-FiLM and an equivariant velocity head.

    `config` holds the UMA backbone configuration (`backbone`), the blocks that carry a time-FiLM
    (`film_blocks`), the head depth (`head_depth`), the tasks the model was trained on (`tasks`, with any
    remapping such as oc22 -> oc20) and the inference defaults (`inference`).
    """

    def __init__(self, config: dict):
        super().__init__()
        from fairchem.core.models.uma.escn_moe import eSCNMDMoeBackbone

        self.config = config
        raw = eSCNMDMoeBackbone(**{**config["backbone"], **_INFERENCE_OVERRIDES})
        self.backbone = TimeFiLMBackbone(raw, inject_block_indices=list(config["film_blocks"]),
                                         inject_force=False)
        self.velocity_head = VelocityHead(sphere_channels=raw.sphere_channels, input_lmax=raw.lmax,
                                          depth=int(config["head_depth"]))
        self.eval()

    @classmethod
    def from_pretrained(cls, repo_id_or_path: str = DEFAULT_REPO, *, revision: str | None = None,
                        device: str | torch.device | None = None, token: str | bool | None = None,
                        cache_dir: str | None = None) -> SaddleFlow:
        """Build the model from a Hub repository (default `SciLM/SaddleFlow`) or a local directory
        holding `config.json` and `model.safetensors`."""
        from safetensors.torch import load_file

        path = Path(repo_id_or_path)
        if path.is_dir():
            cfg_file, w_file = path / CONFIG_NAME, path / WEIGHTS_NAME
        else:
            from huggingface_hub import hf_hub_download
            kw = {"revision": revision, "token": token, "cache_dir": cache_dir}
            cfg_file = hf_hub_download(repo_id_or_path, CONFIG_NAME, **kw)
            w_file = hf_hub_download(repo_id_or_path, WEIGHTS_NAME, **kw)
        model = cls(json.loads(Path(cfg_file).read_text()))
        missing, unexpected = model.load_state_dict(load_file(str(w_file)), strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected keys in {w_file}: {unexpected[:5]}")
        # The time-FiLM wrapper keeps a `film` alias of its last FiLM; nothing else may be missing.
        missing = [k for k in missing if not k.startswith("backbone.film.")]
        if missing:
            raise RuntimeError(f"missing keys in {w_file}: {missing[:5]}")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return model.to(device).eval()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _task(self, atoms: Atoms, task_name: str | None) -> tuple[str, str]:
        """Return (task, task the backbone is routed through); e.g. oc22 is routed through oc20."""
        task = task_name or atoms.info.get("task_name") or self.config["tasks"]["default"]
        if task not in self.config["tasks"]["map"]:
            raise ValueError(f"task {task!r}: SaddleFlow was trained on {sorted(self.config['tasks']['map'])} "
                             f"(omat for bulk materials, oc20/oc22 for catalyst surfaces)")
        return task, self.config["tasks"]["map"][task]

    def _record(self, atoms: Atoms, task_name: str | None) -> dict:
        fixed = np.zeros(len(atoms), dtype=bool)
        for c in atoms.constraints:
            if isinstance(c, FixAtoms):
                fixed[c.get_indices()] = True
        task, routed = self._task(atoms, task_name)
        return {"Z": torch.as_tensor(atoms.get_atomic_numbers(), dtype=torch.long),
                "cell": torch.as_tensor(np.asarray(atoms.get_cell()), dtype=torch.float32),
                "fixed": torch.as_tensor(fixed), "task_name": routed, "task": task,
                "charge": int(atoms.info.get("charge", 0)), "spin": int(atoms.info.get("spin", 0))}

    @torch.no_grad()
    def _integrate(self, xs: list[torch.Tensor], recs: list[dict], steps: int) -> list[torch.Tensor]:
        """`steps` forward-Euler steps from t = 0 to 1 for a batch of systems (one network call per step)."""
        dev, B = self.device, len(xs)
        sizes = [int(r["Z"].shape[0]) for r in recs]
        fixed = torch.cat([r["fixed"] for r in recs]).to(dev)
        for k in range(steps):
            t = torch.full((B,), k / steps, dtype=torch.float32, device=dev)
            data = [build_atomic_data(x, r["Z"], r["cell"], r["task_name"], r["charge"], r["spin"], r["fixed"])
                    for x, r in zip(xs, recs)]
            batch = data_list_collater(data, otf_graph=True).to(dev)
            h = self.backbone(batch, t, batch.batch)["node_embedding"]
            v = self.velocity_head(h, t, batch.batch)
            v = apply_output_projections(v, fixed, batch.batch, B).cpu().float()
            xs = [wrap_positions(x + dv / steps, r["cell"]) for x, dv, r in zip(xs, torch.split(v, sizes), recs)]
        return xs

    def propose(self, atoms: Atoms | Sequence[Atoms], n: int = 1, *, sigma: float | None = None,
                passes: int | None = None, steps: int | None = None, seed: int | None = None,
                task_name: str | None = None, batch_size: int = 16,
                return_passes: bool = False) -> list[Atoms] | list[list[Atoms]]:
        """Propose `n` saddle points for each input structure.

        Args:
            atoms: one structure or a list; periodic cells, FixAtoms constraints respected.
            n: noise draws per structure; different draws can lead to different saddles.
            sigma: Gaussian noise (A) added to the mobile atoms of the start. Defaults to the training
                value (0.3 A); larger values are refused because the model degrades sharply beyond it.
            passes, steps: flow passes and Euler steps per pass (default 5 x 4).
            seed: seed for the noise draws (reproducible proposals).
            task_name: UMA task, `omat` (bulk), `oc20` or `oc22` (surfaces); else `atoms.info['task_name']`,
                else `omat`.
            batch_size: systems per network call.
            return_passes: also return the endpoint after every pass (`info['passes']`, shape (passes+1, N, 3)).

        Returns:
            For one input structure, a list of `n` structures; for a list, one such list per structure.
            Each proposal keeps the input's cell, numbers and constraints and records the settings in `info`.
        """
        inf = self.config["inference"]
        sigma = float(inf["sigma"] if sigma is None else sigma)
        passes = int(inf["passes"] if passes is None else passes)
        steps = int(inf["steps"] if steps is None else steps)
        if sigma > inf["sigma"] + 1e-9:
            raise ValueError(f"sigma={sigma} exceeds the training noise {inf['sigma']} A; "
                             f"the model degrades sharply when run wider than it was trained")
        single = isinstance(atoms, Atoms)
        structures = [atoms] if single else list(atoms)
        gen = torch.Generator().manual_seed(seed) if seed is not None else None

        jobs = []                                            # (structure index, draw, record, start)
        for i, at in enumerate(structures):
            rec = self._record(at, task_name)
            pos = torch.as_tensor(at.get_positions(), dtype=torch.float32)
            mobile = (~rec["fixed"]).float().unsqueeze(1)
            for d in range(n):
                noise = torch.randn(pos.shape, generator=gen) if gen is not None else torch.randn(pos.shape)
                jobs.append((i, d, rec, wrap_positions(pos + sigma * noise * mobile, rec["cell"])))

        out = [[None] * n for _ in structures]
        with _no_tf32():
            for s0 in range(0, len(jobs), batch_size):
                chunk = jobs[s0:s0 + batch_size]
                xs, recs = [j[3].clone() for j in chunk], [j[2] for j in chunk]
                trace = [[x.numpy().copy()] for x in xs] if return_passes else None
                for _ in range(passes):
                    xs = self._integrate(xs, recs, steps)
                    if return_passes:
                        for tr, x in zip(trace, xs):
                            tr.append(x.numpy().copy())
                for j, (i, d, rec, _) in enumerate(chunk):
                    prop = structures[i].copy()
                    prop.calc = None
                    prop.set_positions(xs[j].numpy().astype(float), apply_constraint=False)
                    prop.info.update(saddleflow_draw=d, saddleflow_sigma=sigma, saddleflow_passes=passes,
                                     saddleflow_steps=steps, task_name=rec["task"])
                    if seed is not None:
                        prop.info["saddleflow_seed"] = seed
                    if return_passes:
                        prop.info["passes"] = np.stack(trace[j], 0)
                    out[i][d] = prop
        return out[0] if single else out
