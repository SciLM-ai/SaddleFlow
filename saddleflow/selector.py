"""Atom-set selector: proposes which atoms react, one atom at a time, from the minimum alone.

A reaction's set is the atoms moving more than `thresh` (A) from the minimum to the saddle, plus the most-moving one,
ordered by displacement, most-moving first. The network sees the minimum with the atoms chosen so far flagged (the per-atom
flag input of TimeFiLMBackbone, at t = 0) and gives a logit for every mobile atom not yet chosen and a stop logit; sampling
starts empty and draws the next atom or stop from the softmax until it stops. Training is by likelihood: a random prefix of
a set (length j uniform in 0..k) is shown and the target is atom j + 1, or stop at j = k; each example is weighted by
k + 1, so the expected gradient is that of the full log-likelihood. The sets it samples go to a flagged SaddleFlow model:

    sel = AtomSetSelector.from_checkpoint("runs/selector/checkpoint_final")
    sets = sel.sample(atoms, n=8, temperature=1.5, seed=0)             # 8 index lists
    guesses = SaddleFlow.from_pretrained(revision="fm2").propose([atoms] * 8, atoms_to_move=sets)
"""

from __future__ import annotations

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
from .flow.matching import build_atomic_data
from .models.time_filmed_backbone import TimeFiLMBackbone
from .utils import load_ema_weights, load_uma_backbone


class AtomHead(nn.Module):
    """Per-atom logit from the l = 0 channels of the node embedding."""

    def __init__(self, channels: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, emb: torch.Tensor) -> torch.Tensor:          # (N, num_sph, C) -> (N,)
        return self.mlp(emb[:, 0, :]).squeeze(-1)


class StopHead(nn.Module):
    """Per-system stop logit from the mean l = 0 embedding of its atoms."""

    def __init__(self, channels: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, emb: torch.Tensor, batch: torch.Tensor, nsys: int) -> torch.Tensor:    # -> (nsys,)
        s = torch.zeros(nsys, emb.shape[-1], device=emb.device, dtype=emb.dtype).index_add_(0, batch, emb[:, 0, :])
        cnt = torch.bincount(batch, minlength=nsys).clamp_min(1).to(emb.dtype)
        return self.mlp(s / cnt[:, None]).squeeze(-1)


def ordered_set(sample: dict, thresh: float) -> torch.Tensor:
    """The reaction's set (mobile atoms moving more than thresh from start to saddle, plus the most-moving one), most-moving
    first, then by decreasing displacement."""
    mobile = ~sample["fixed"]
    n = torch.where(mobile, (sample["saddle_un_pos"] - sample["start_pos"]).norm(dim=1),
                    torch.full_like(sample["start_pos"][:, 0], -1.0))
    m = mobile & (n > thresh); m[int(n.argmax())] = True
    idx = torch.where(m)[0]
    return idx[torch.argsort(n[idx], descending=True)]


def build_selector(backbone_name: str, film_blocks: Sequence[int], device: str):
    """UMA backbone (every block trainable) with time-FiLM and the flag input, plus the atom and stop heads."""
    raw = load_uma_backbone(backbone_name, device=device, freeze=True, eval_mode=True)
    for blk in raw.blocks:
        for p in blk.parameters():
            p.requires_grad_(True)
    backbone = TimeFiLMBackbone(raw, inject_block_indices=list(film_blocks), inject_force=False, inject_flag=True).to(device)
    return backbone, AtomHead(raw.sphere_channels).to(device), StopHead(raw.sphere_channels).to(device)


class SelectorLoss(nn.Module):
    """Cross-entropy of the next set atom (or stop) after a random prefix, weighted by the set size + 1."""

    def __init__(self, backbone: TimeFiLMBackbone, atom_head: AtomHead, stop_head: StopHead, thresh: float):
        super().__init__()
        self.backbone, self.atom_head, self.stop_head, self.thresh = backbone, atom_head, stop_head, thresh

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()                        # UMA's dropouts stay off (as FlowMatchingLoss)
        return self

    def forward(self, batch: list[dict]) -> dict:
        dev = next(self.atom_head.parameters()).device
        datas, flags, mob, tgt, w = [], [], [], [], []
        for s in batch:
            order = ordered_set(s, self.thresh); k = len(order)
            j = int(torch.randint(k + 1, (1,)))
            f = torch.zeros(len(s["fixed"]), dtype=torch.bool, device=s["fixed"].device); f[order[:j]] = True
            datas.append(build_atomic_data(wrap_positions(s["start_pos"], s["cell"]), s["Z"], s["cell"], s["task_name"],
                                           s["charge"], s["spin"], s["fixed"]))
            flags.append(f); mob.append(~s["fixed"]); tgt.append(int(order[j]) if j < k else -1); w.append(k + 1.0)
        b = data_list_collater(datas, otf_graph=True).to(dev)
        nsys = len(batch); F = torch.cat(flags).to(dev); M = torch.cat(mob).to(dev)
        emb = self.backbone(b, torch.zeros(nsys, device=dev), b.batch, flag=F.float())["node_embedding"]
        a = self.atom_head(emb).masked_fill(~M | F, float("-inf"))
        st = self.stop_head(emb, b.batch, nsys)
        ce = []
        for i in range(nsys):                       # softmax over system i's addable atoms + stop (not at an empty set)
            sl = b.batch == i
            si = st[i] if F[sl].any() else st.new_tensor(float("-inf"))
            target = tgt[i] if tgt[i] >= 0 else int(sl.sum())
            ce.append(nn.functional.cross_entropy(torch.cat([a[sl], si[None]])[None], torch.tensor([target], device=dev)))
        w = torch.tensor(w, device=dev)
        loss = (torch.stack(ce) * w).sum() / w.sum()
        # Multi-GPU: if every prefix on a rank is empty, stop is masked out and the stop head gets no gradient; DDP then
        # never all-reduces its bucket and every rank hangs. A zero-weighted term keeps it in the graph.
        return {"loss": loss + 0.0 * st.sum(), "mode": 0, "n_mobile": int(M.sum())}


class AtomSetSelector:
    """A trained selector (EMA weights of a train_selector.py checkpoint) that samples atom sets for ASE structures."""

    def __init__(self, backbone, atom_head, stop_head, task_map: dict | None = None):
        self.backbone, self.atom_head, self.stop_head = backbone.eval(), atom_head.eval(), stop_head.eval()
        self.task_map = dict(task_map or {})

    @classmethod
    def from_checkpoint(cls, ckpt_dir: str, device: str | None = None) -> AtomSetSelector:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ex = json.loads((Path(ckpt_dir).parent / "config.json").read_text())["extras"]
        blocks = [int(x) for x in str(ex.get("early_time_film_blocks", "0,1,2,3")).split(",")]
        backbone, atom_head, stop_head = build_selector(ex["backbone"], blocks, device)
        load_ema_weights(ckpt_dir, [backbone, atom_head, stop_head], device=device, use_ema=True)
        tm = ex.get("task_name_map") or ""
        return cls(backbone, atom_head, stop_head, dict(x.split("=") for x in tm.split(",") if x))

    @torch.no_grad()
    def sample(self, atoms: Atoms, n: int = 1, *, temperature: float = 1.0, max_atoms: int = 12, seed: int | None = None,
               task_name: str | None = None) -> list[list[int]]:
        """`n` atom sets for one structure (FixAtoms atoms are never chosen); each is a list of atom indices."""
        dev = next(self.atom_head.parameters()).device
        task = task_name or atoms.info.get("task_name") or "omat"
        fixed = np.zeros(len(atoms), bool)
        for c in atoms.constraints:
            if isinstance(c, FixAtoms):
                fixed[c.get_indices()] = True
        pos = torch.as_tensor(atoms.get_positions(), dtype=torch.float32)
        cell = torch.as_tensor(np.asarray(atoms.get_cell()), dtype=torch.float32)
        data = build_atomic_data(wrap_positions(pos, cell), torch.as_tensor(atoms.get_atomic_numbers()), cell,
                                 self.task_map.get(task, task), int(atoms.info.get("charge", 0)),
                                 int(atoms.info.get("spin", 0)), torch.as_tensor(fixed))
        b = data_list_collater([data] * n, otf_graph=True).to(dev)
        N = len(atoms); M = torch.as_tensor(~fixed, device=dev).repeat(n); F = torch.zeros_like(M)
        done = torch.zeros(n, dtype=torch.bool)
        gen = torch.Generator(device=dev).manual_seed(seed) if seed is not None else None
        for _ in range(max_atoms):
            emb = self.backbone(b, torch.zeros(n, device=dev), b.batch, flag=F.float())["node_embedding"]
            av = self.atom_head(emb).masked_fill(~M | F, float("-inf")); st = self.stop_head(emb, b.batch, n)
            for i in range(n):
                if done[i]:
                    continue
                sl = slice(i * N, (i + 1) * N)
                si = st[i] if F[sl].any() else st.new_tensor(float("-inf"))
                c = int(torch.multinomial(torch.softmax(torch.cat([av[sl], si[None]]) / temperature, 0), 1, generator=gen))
                if c == N:
                    done[i] = True
                else:
                    F[i * N + c] = True
            if done.all():
                break
        return [torch.where(F[i * N:(i + 1) * N])[0].tolist() for i in range(n)]
