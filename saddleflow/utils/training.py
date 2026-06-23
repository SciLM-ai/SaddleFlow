"""
HuggingFace-`accelerate`-based training loop for SaddleFlow.

Thin on purpose: AdamW + linear-warmup-then-cosine LR schedule + EMA + grad-clip
+ checkpointing. No Hydra, no Lightning, no TorchTNT — CLAUDE.md §"Training
infrastructure" explains why we skip fairchem's full stack.

This module is agnostic to what's inside the loss module. It just needs:
    loss_module(batch) -> {"loss": scalar tensor, "per_obj": (2,) long,
                           "per_obj_loss": (2,) float, "n_mobile": int}
and a dataset yielding sample dicts consumed by `FlowMatchingLoss`.

Multi-GPU / multi-node come free via `accelerate launch`:
    accelerate launch --multi_gpu --num_machines=N examples/li_on_carbon/train.py ...
"""

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def identity_collate(batch):
    """Pass-through collate — yields `list[dict]`. Needed because samples have
    heterogeneous N and default collation stacks tensors, which fails."""
    return list(batch)


@dataclass
class TrainingConfig:
    output_dir: str

    num_epochs: int = 100
    batch_size: int = 8
    num_workers: int = 2

    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    warmup_steps: int = 100
    min_lr_ratio: float = 0.01  # LR floor as fraction of `learning_rate`

    grad_clip_norm: float = 1.0
    ema_decay: float = 0.9999

    mixed_precision: str = "bf16"  # "bf16" | "fp16" | "no"
    seed: int = 42

    log_every: int = 10
    save_every_epochs: int = 10
    # If > 0, also checkpoint every N optimizer steps to
    # ``<output-dir>/checkpoint_step_NNNNNNNN``. Essential for multi-day runs
    # where a single epoch exceeds the SLURM time-limit — without intra-epoch
    # saves a kill loses all work.
    save_every_steps: int = 0
    # If > 0, run validation every N optimizer steps (in addition to the
    # per-epoch val controlled by ``val_every_epochs``). Useful for big-data
    # runs where one epoch is many thousand steps — gives a finer-grained val
    # curve. Each call is a full sweep over the val loader, so set this large
    # enough that the val cost is a small fraction of training time.
    val_every_steps: int = 0

    resume_from: str | None = None

    # --- Benchmark instrumentation (all no-ops at defaults; production unaffected) ---
    # If > 0, stop after this many optimizer steps regardless of num_epochs. Also
    # used as the scheduler's total-step horizon so warmup/cosine behave sanely.
    max_steps: int = 0
    # Steps to run before opening the timing window — excludes DDP init, cudnn
    # autotune, dataloader warm-up and the first allreduce from the throughput
    # measurement. Only consulted when bench_output is set.
    bench_warmup_steps: int = 0
    # If set, write a throughput JSON here when training stops AND skip the final
    # checkpoint + test eval (a benchmark run produces no model artifact).
    bench_output: str | None = None
    # If set, run torch.profiler over the timed window and write a step-breakdown
    # JSON here (GPU-kernel vs CPU/host time, dataload vs compute, top ops). Also
    # implies the bench short-circuit (no checkpoint / test eval).
    profile_output: str | None = None

    extras: dict = field(default_factory=dict)  # stashed for reproducibility in run-config


class EMA:
    """Exponential moving average of trainable parameters.

    Holds a list of shadow tensors parallel to `params`. Call `update()` after
    each `optimizer.step()`. For eval, use `swap_in(params)` to install EMA
    weights into the live model and `swap_out(params)` to restore.
    """

    def __init__(self, params, decay: float = 0.9999):
        self.decay = decay
        self._params_ref = [p for p in params if p.requires_grad]
        self.shadow = [p.detach().clone() for p in self._params_ref]
        self._backup: list[torch.Tensor] | None = None

    @torch.no_grad()
    def update(self) -> None:
        for s, p in zip(self.shadow, self._params_ref):
            s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def swap_in(self) -> None:
        assert self._backup is None, "EMA.swap_in called while already swapped in"
        self._backup = [p.detach().clone() for p in self._params_ref]
        for s, p in zip(self.shadow, self._params_ref):
            p.data.copy_(s.data)

    @torch.no_grad()
    def swap_out(self) -> None:
        assert self._backup is not None, "EMA.swap_out called without prior swap_in"
        for b, p in zip(self._backup, self._params_ref):
            p.data.copy_(b.data)
        self._backup = None

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": [s.cpu() for s in self.shadow]}

    def load_state_dict(self, sd: dict) -> None:
        self.decay = sd["decay"]
        for s, loaded in zip(self.shadow, sd["shadow"]):
            s.data.copy_(loaded.to(s.device))


def _lr_lambda(step: int, warmup_steps: int, total_steps: int, min_ratio: float) -> float:
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return max(min_ratio, cosine)


def train(
    loss_module: nn.Module,
    dataset: Dataset,
    config: TrainingConfig,
    val_dataset: Dataset | None = None,
    val_every_epochs: int = 10,
    test_dataset: Dataset | None = None,
    param_groups: list[dict] | None = None,
) -> dict:
    """Plain accelerate-based train loop.

    `param_groups`, if provided, is passed straight to `AdamW` instead of the
    default single-group `(trainable, lr=config.learning_rate)`. Use this for
    discriminative LR setups (e.g. v1 with unfrozen UMA `blocks[-1]`: low LR
    on the unfrozen backbone params, normal LR on the head). Each group is a
    dict like `{"params": [...], "lr": ..., "weight_decay": ...}`. The cosine
    + warmup schedule applies as a multiplicative factor on EVERY group's LR.
    EMA is built over the union of all params across groups.
    """
    # `accelerate` is imported lazily so EMA / TrainingConfig / identity_collate
    # can be used in environments where the dependency is not installed.
    from datetime import timedelta
    from accelerate import Accelerator, InitProcessGroupKwargs
    from accelerate.utils import set_seed

    set_seed(config.seed)
    # NCCL's default 10-min collective timeout is too tight on multi-node setups
    # where the *first* allreduce (inside accelerator.prepare's DDP weight
    # broadcast) lands after slow concurrent backbone load + dataset setup.
    # 30 min covers cold-cache UMA load across 64-512 GPUs on Perlmutter without
    # affecting steady-state training (per-step collectives finish in ms).
    ddp_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=30))
    handlers = [ddp_kwargs]
    # FIND_UNUSED_PARAMS=1 -> DDP find_unused_parameters=True. The default (False)
    # deadlocks the gradient reducer if any parameter receives no gradient on some
    # rank for some batch (a hang at a data-dependent step). Suspected cause of the
    # Vista 32-node ALLREDUCE hangs; opt-in so the validated default path is unchanged.
    if os.environ.get("FIND_UNUSED_PARAMS") == "1":
        from accelerate.utils import DistributedDataParallelKwargs
        handlers.append(DistributedDataParallelKwargs(find_unused_parameters=True))
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        kwargs_handlers=handlers,
    )
    out_dir = Path(config.output_dir)
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, default=str))

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=identity_collate,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=identity_collate,
            num_workers=config.num_workers,
            persistent_workers=config.num_workers > 0,
        )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=identity_collate,
            num_workers=config.num_workers,
            persistent_workers=config.num_workers > 0,
        )

    trainable = [p for p in loss_module.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("no parameters have requires_grad=True; nothing to train")
    if param_groups is None:
        optimizer = torch.optim.AdamW(
            trainable, lr=config.learning_rate, weight_decay=config.weight_decay,
        )
    else:
        # Sanity: every entry in param_groups['params'] must have requires_grad.
        # AdamW requires it; surface the violation here with a clearer message.
        for i, g in enumerate(param_groups):
            for p in g.get("params", []):
                if not p.requires_grad:
                    raise ValueError(
                        f"param_groups[{i}] contains a parameter with requires_grad=False"
                    )
        optimizer = torch.optim.AdamW(
            param_groups, lr=config.learning_rate, weight_decay=config.weight_decay,
        )
        if accelerator.is_main_process:
            for i, g in enumerate(param_groups):
                lr_i = g.get("lr", config.learning_rate)
                wd_i = g.get("weight_decay", config.weight_decay)
                n_i = sum(p.numel() for p in g.get("params", []))
                print(f"[train] param_group[{i}] '{g.get('name', '?')}': "
                      f"{n_i:,} params  lr={lr_i:.2e}  wd={wd_i:g}")
    total_steps = max(1, config.num_epochs * len(dataloader))
    if config.max_steps > 0:
        total_steps = config.max_steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: _lr_lambda(s, config.warmup_steps, total_steps, config.min_lr_ratio),
    )
    ema = EMA(trainable, decay=config.ema_decay)

    loss_module, optimizer, dataloader, scheduler = accelerator.prepare(
        loss_module, optimizer, dataloader, scheduler,
    )
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)
    if test_loader is not None:
        test_loader = accelerator.prepare(test_loader)

    start_epoch = 0
    global_step = 0
    if config.resume_from:
        accelerator.load_state(config.resume_from)
        ema_path = Path(config.resume_from) / "ema.pt"
        if ema_path.exists():
            ema.load_state_dict(torch.load(ema_path, map_location="cpu"))
        meta_path = Path(config.resume_from) / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            start_epoch = int(meta.get("epoch", 0))
            global_step = int(meta.get("global_step", 0))

    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in trainable)
        print(f"[train] trainable params: {n_params:,}")
        print(f"[train] total steps: {total_steps}  (epochs {config.num_epochs} × {len(dataloader)} batches)")
        print(f"[train] starting at epoch {start_epoch}, global step {global_step}")

    history: list[dict] = []
    t0 = time.time()

    # Stop when global_step reaches the planned total. Without this, a resume
    # from a mid-epoch checkpoint (e.g., step 8000 of a 30k-step run) would
    # iterate num_epochs full epochs from start_epoch and OVERSHOOT — adding
    # thousands of extra steps at the LR floor that just burn wall-clock for
    # near-zero training gain. With this cap, resume(8000)→30000 produces the
    # same model+step count as a continuous run.
    total_optim_steps = max(1, config.num_epochs * len(dataloader))
    if config.max_steps > 0:
        total_optim_steps = config.max_steps
    stop_now = False

    # Benchmark timing state (active for either throughput bench or profiling).
    profile_active = bool(config.profile_output)
    bench_active = bool(config.bench_output) or profile_active
    bench_step_times: list[float] = []   # full wall-clock per timed step (incl. data load)
    bench_window_t0: float | None = None
    prof = None                          # torch.profiler handle (profiling runs only)
    prof_dataload = 0.0                  # Σ host wait fetching batches, timed window
    prof_compute = 0.0                   # Σ fwd+bwd+opt+ema+gather, timed window

    for epoch in range(start_epoch, config.num_epochs):
        if stop_now:
            break
        loss_module.train()
        epoch_loss = 0.0
        epoch_n = 0
        _prev_t = None  # for per-step timing (reset each epoch; loses 1 sample/epoch boundary)
        for batch in dataloader:
            _body_top = time.perf_counter()  # after the dataloader yielded this batch
            # Open the timing window exactly when warmup ends, after syncing so
            # no warm-up GPU work bleeds into the measured wall-clock, and reset
            # the peak-memory counter so the reported footprint is steady-state.
            if bench_active and global_step == config.bench_warmup_steps:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                bench_window_t0 = time.perf_counter()
                if profile_active:
                    from torch.profiler import ProfilerActivity
                    from torch.profiler import profile as _torch_profile
                    _acts = [ProfilerActivity.CPU]
                    if torch.cuda.is_available():
                        _acts.append(ProfilerActivity.CUDA)
                    prof = _torch_profile(activities=_acts, record_shapes=False, with_stack=False)
                    prof.start()
            with accelerator.accumulate(loss_module):
                out = loss_module(batch)
                loss = out["loss"]
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, config.grad_clip_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                ema.update()

            loss_val = accelerator.gather(loss.detach()).mean().item()
            epoch_loss += loss_val
            epoch_n += 1

            # Per-step wall-clock between consecutive sync points (the gather above
            # forces a CUDA sync via .item()), so this includes data loading and
            # collectives — the honest end-to-end step cost. Only kept in the
            # timed window.
            if bench_active and global_step >= config.bench_warmup_steps:
                _now = time.perf_counter()
                if _prev_t is not None:
                    bench_step_times.append(_now - _prev_t)
                    if profile_active:
                        # Partition the inter-step interval: time waiting on the
                        # dataloader to yield (_body_top - prev end) vs time in the
                        # actual step (fwd+bwd+opt+ema+gather).
                        prof_dataload += _body_top - _prev_t
                        prof_compute += _now - _body_top
                _prev_t = _now

            if accelerator.is_main_process and global_step % config.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                mode = int(out.get("mode", -1))
                print(f"[train] ep {epoch:4d} step {global_step:7d} mode {mode} "
                      f"loss {loss_val:.4f}  lr {lr:.2e}  n_mobile {out['n_mobile']}")
            global_step += 1
            # Optional intra-epoch checkpoint. Only writes on main_process; the
            # `accelerator.save_state` call inside _save_checkpoint is itself
            # rank-aware (sharded under FSDP). Triggered AFTER the step bump so
            # `meta.json` records the post-step value.
            if (config.save_every_steps > 0
                    and global_step % config.save_every_steps == 0
                    and accelerator.is_main_process):
                _save_checkpoint(
                    accelerator, ema, out_dir,
                    f"checkpoint_step_{global_step:08d}",
                    epoch, global_step,
                )
            # Optional intra-epoch validation. Full val sweep — _evaluate
            # handles eval/train mode toggling and EMA swap. All ranks call it
            # (it does collective gather() internally). Logged to history with
            # global_step + epoch so the curve can be plotted directly.
            if (config.val_every_steps > 0
                    and global_step % config.val_every_steps == 0
                    and val_loader is not None):
                val_losses = _evaluate(loss_module, val_loader, ema, accelerator)
                if accelerator.is_main_process:
                    print(f"[val]   step {global_step:7d} live {val_losses['live']:.4f}  "
                          f"ema {val_losses['ema']:.4f}")
                    history.append({
                        "epoch": epoch, "global_step": global_step,
                        "mean_val_loss_live": val_losses["live"],
                        "mean_val_loss_ema": val_losses["ema"],
                        "mean_val_loss": val_losses["ema"],
                        "elapsed_sec": time.time() - t0,
                        "kind": "val_step",
                    })
            if global_step >= total_optim_steps:
                stop_now = True
                break

        mean_epoch_loss = epoch_loss / max(1, epoch_n)
        rec = {"epoch": epoch, "global_step": global_step,
               "mean_train_loss": mean_epoch_loss,
               "elapsed_sec": time.time() - t0}

        if val_loader is not None and (epoch + 1) % val_every_epochs == 0:
            val_losses = _evaluate(loss_module, val_loader, ema, accelerator)
            rec["mean_val_loss_live"] = val_losses["live"]
            rec["mean_val_loss_ema"] = val_losses["ema"]
            rec["mean_val_loss"] = val_losses["ema"]  # backward-compat key
            if accelerator.is_main_process:
                print(f"[val]   ep {epoch:4d} live {val_losses['live']:.4f}  "
                      f"ema {val_losses['ema']:.4f}")

        history.append(rec)

        if accelerator.is_main_process and (epoch + 1) % config.save_every_epochs == 0:
            _save_checkpoint(accelerator, ema, out_dir, f"checkpoint_epoch_{epoch+1:05d}",
                             epoch + 1, global_step)

    # Benchmark finalization: close the timing window, write throughput JSON, and
    # short-circuit before the (slow, irrelevant) final checkpoint + test eval.
    if bench_active:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        window_sec = (time.perf_counter() - bench_window_t0) if bench_window_t0 is not None else 0.0
        timed_steps = max(0, global_step - config.bench_warmup_steps)
        if prof is not None:
            prof.stop()
        if accelerator.is_main_process:
            if config.bench_output:
                _write_bench_result(config, accelerator, window_sec, timed_steps, bench_step_times)
            if config.profile_output and prof is not None:
                _write_profile_result(config, accelerator, window_sec, timed_steps,
                                      prof, prof_dataload, prof_compute)
        return {"history": history, "global_step": global_step, "test_loss": None}

    if accelerator.is_main_process:
        _save_checkpoint(accelerator, ema, out_dir, "checkpoint_final", config.num_epochs, global_step)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(f"[train] done. total time {(time.time() - t0) / 60:.2f} min")

    test_losses = None
    if test_loader is not None:
        test_losses = _evaluate(loss_module, test_loader, ema, accelerator)
        if accelerator.is_main_process:
            print(f"[test]  final  live {test_losses['live']:.4f}  "
                  f"ema {test_losses['ema']:.4f}  "
                  f"({len(test_loader)} batches × {config.batch_size})")
            (out_dir / "test_results.json").write_text(json.dumps({
                "test_loss_live": test_losses["live"],
                "test_loss_ema": test_losses["ema"],
                "test_loss": test_losses["ema"],  # backward-compat key
                "global_step": global_step,
                "epoch": config.num_epochs,
                "weights": "live+ema",
            }, indent=2))

    return {"history": history, "global_step": global_step,
            "test_loss": test_losses}


@torch.no_grad()
def _evaluate(loss_module, val_loader, ema: EMA, accelerator) -> dict:
    """Returns {"live": float, "ema": float}. Reporting both surfaces EMA-decay
    misconfiguration immediately — if decay is too high for the run length,
    `ema` tracks initial-weight loss while `live` reflects training progress."""
    loss_module.eval()
    try:
        total, n = 0.0, 0
        for batch in val_loader:
            out = loss_module(batch)
            total += accelerator.gather(out["loss"].detach()).mean().item()
            n += 1
        live = total / max(1, n)
        ema.swap_in()
        try:
            total, n = 0.0, 0
            for batch in val_loader:
                out = loss_module(batch)
                total += accelerator.gather(out["loss"].detach()).mean().item()
                n += 1
            ema_loss = total / max(1, n)
        finally:
            ema.swap_out()
        return {"live": live, "ema": ema_loss}
    finally:
        loss_module.train()


def _write_bench_result(config, accelerator, window_sec: float, timed_steps: int,
                        step_times: list[float]) -> None:
    """Write a single-config throughput record to ``config.bench_output``.

    Called once, on the main process only, after the timing window closes. The
    window-based ``samples_per_sec`` is the headline number (it includes data
    loading, the optimizer step and all DDP collectives — the honest end-to-end
    rate); the per-step median/p90 are a jitter diagnostic.
    """
    import socket
    import statistics

    world = accelerator.num_processes
    bs = config.batch_size
    global_batch = bs * world
    sps = (timed_steps * global_batch / window_sec) if window_sec > 0 else 0.0

    def _q(xs, q):
        if not xs:
            return None
        s = sorted(xs)
        i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
        return s[i]

    gpu_name = None
    peak_alloc = peak_resv = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_resv = torch.cuda.max_memory_reserved() / (1024 ** 3)

    result = {
        "hostname": socket.gethostname(),
        "gpu_name": gpu_name,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "num_nodes": int(os.environ.get("SLURM_NNODES", 0)) or None,
        "world_size": world,
        "gpus_per_node": (world // int(os.environ["SLURM_NNODES"]))
                          if os.environ.get("SLURM_NNODES") else None,
        "batch_size_per_gpu": bs,
        "global_batch": global_batch,
        "mixed_precision": config.mixed_precision,
        "warmup_steps": config.bench_warmup_steps,
        "timed_steps": timed_steps,
        "window_sec": window_sec,
        "steps_per_sec": (timed_steps / window_sec) if window_sec > 0 else 0.0,
        "samples_per_sec": sps,
        "samples_per_sec_per_gpu": sps / world if world else 0.0,
        "mean_ms_per_step": (window_sec / timed_steps * 1000.0) if timed_steps else None,
        "median_ms_per_step": (_q(step_times, 0.5) * 1000.0) if step_times else None,
        "p10_ms_per_step": (_q(step_times, 0.10) * 1000.0) if step_times else None,
        "p90_ms_per_step": (_q(step_times, 0.90) * 1000.0) if step_times else None,
        "peak_mem_alloc_gib": peak_alloc,
        "peak_mem_reserved_gib": peak_resv,
        "wall_clock_unix": time.time(),
    }
    out = Path(config.bench_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[bench] {world} GPU(s) bs={bs} (global {global_batch}): "
          f"{sps:,.1f} samples/s total  "
          f"({result['samples_per_sec_per_gpu']:,.1f}/GPU)  "
          f"mean {result['mean_ms_per_step']:.1f} ms/step  "
          f"peak {peak_alloc:.1f} GiB → {out}")


def _write_profile_result(config, accelerator, window_sec: float, timed_steps: int,
                          prof, dataload_s: float, compute_s: float) -> None:
    """Write a per-step breakdown from a torch.profiler run to config.profile_output.

    The headline is ``gpu_util_pct`` = GPU-kernel busy time ÷ wall-clock: if it's
    well under 100%, the step is CPU/host-bound (graph construction, dispatch,
    host↔device copies), which is the case for the Grace-CPU + NVLink-C2C argument.
    ``top_ops`` names where the time goes. Profiler overhead inflates absolute
    wall_ms vs a clean bench — read the RATIOS, not the absolutes.
    """
    import socket
    from torch.autograd import DeviceType

    n = max(1, timed_steps)
    ka = prof.key_averages()

    def _cuda(k):
        for attr in ("self_device_time_total", "self_cuda_time_total"):
            v = getattr(k, attr, None)
            if v:
                return float(v)
        return 0.0

    def _cpu(k):
        return float(getattr(k, "self_cpu_time_total", 0.0))

    def _is_device(k):
        # Actual on-GPU kernel events have device_type CUDA; CPU-side operators and
        # user annotations (which also carry an attributed device time for the
        # kernels they launch) are CPU — summing only device events avoids the
        # double-count that otherwise pushes "GPU busy" past wall-clock.
        return getattr(k, "device_type", None) == DeviceType.CUDA

    # True GPU busy time = Σ self device time over device(kernel) events only.
    gpu_busy_ms = sum(_cuda(k) for k in ka if _is_device(k)) / 1e3 / n
    tot_cpu_us = sum(_cpu(k) for k in ka if not _is_device(k))
    wall_ms = window_sec * 1e3 / n

    # CPU-side kernel-dispatch + sync overhead — the signature of a launch-bound
    # step (many tiny kernels). Named CUDA-runtime calls in the profiler.
    DISPATCH = {"cudaLaunchKernel", "cudaStreamSynchronize", "cudaMemcpyAsync",
                "cudaDeviceSynchronize", "cudaMemsetAsync", "Command Buffer Full"}
    cpu_dispatch_ms = sum(_cpu(k) for k in ka if k.key in DISPATCH) / 1e3 / n
    _launch = next((k for k in ka if k.key == "cudaLaunchKernel"), None)
    kernel_launches_per_step = (_launch.count / n) if _launch is not None else None

    top = sorted(ka, key=lambda k: _cpu(k) + _cuda(k), reverse=True)[:15]
    ops = [{
        "name": k.key,
        "cpu_ms_per_step": _cpu(k) / 1e3 / n,
        "cuda_ms_per_step": _cuda(k) / 1e3 / n,
        "calls_per_step": k.count / n,
    } for k in top]

    result = {
        "hostname": socket.gethostname(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "world_size": accelerator.num_processes,
        "batch_size_per_gpu": config.batch_size,
        "timed_steps": timed_steps,
        "wall_ms_per_step": wall_ms,
        # Σ device-kernel time across ALL CUDA streams. Can exceed wall_ms when
        # kernels overlap on concurrent streams — so it is NOT a utilization %;
        # it is total GPU kernel-work. The launch-bound signal is cpu_dispatch.
        "device_kernel_ms_per_step": gpu_busy_ms,
        "cpu_dispatch_ms_per_step": cpu_dispatch_ms,
        "cpu_dispatch_frac_pct": (100.0 * cpu_dispatch_ms / wall_ms) if wall_ms else None,
        "kernel_launches_per_step": kernel_launches_per_step,
        "cpu_self_ms_per_step": tot_cpu_us / 1e3 / n,
        "dataload_ms_per_step": dataload_s * 1e3 / n,
        "compute_ms_per_step": compute_s * 1e3 / n,
        "top_ops": ops,
        "note": "torch.profiler overhead inflates absolute wall_ms vs a clean "
                "bench; the launch-bound signal is cpu_dispatch_frac_pct (CPU time "
                "in kernel launch/sync ÷ wall) and kernel_launches_per_step.",
    }
    out = Path(config.profile_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    frac = result["cpu_dispatch_frac_pct"]
    print(f"[profile] wall {wall_ms:.0f} ms/step | device-kernel {gpu_busy_ms:.0f} ms "
          f"| CPU dispatch {cpu_dispatch_ms:.0f} ms ({frac:.0f}% of wall) over "
          f"{kernel_launches_per_step:,.0f} launches | dataload "
          f"{result['dataload_ms_per_step']:.0f} ms → {out}")


def _save_checkpoint(
    accelerator, ema: EMA, out_dir: Path, name: str,
    epoch: int, global_step: int,
) -> None:
    ckpt_dir = out_dir / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    accelerator.save_state(str(ckpt_dir))
    torch.save(ema.state_dict(), ckpt_dir / "ema.pt")
    (ckpt_dir / "meta.json").write_text(json.dumps({"epoch": epoch, "global_step": global_step}))
    print(f"[train] saved checkpoint → {ckpt_dir}")
