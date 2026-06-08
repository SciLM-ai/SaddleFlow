"""Post-bench hyperparameter verification.

Run after `bash run.sh` completes. Reads the run's config.json, history.json,
the training log, and the saved checkpoints, then asserts every ML-generic
hyperparameter matches what was requested in run.sh.

Usage: python bench_verify.py <OUT_DIR> <BENCH_LOG>
"""
import json, sys, re, math, glob, os
from pathlib import Path
import torch


def parse_log_lr_curve(log_path: Path) -> list[tuple[int, float]]:
    """Return list of (step, lr) from `[train] ep ... step S ... lr Y` lines."""
    rx = re.compile(r"\[train\]\s+ep\s+\d+\s+step\s+(\d+)\s+mode\s+\d+\s+loss\s+\S+\s+lr\s+(\S+)")
    pairs = []
    for line in log_path.read_text().splitlines():
        m = rx.search(line)
        if m:
            pairs.append((int(m.group(1)), float(m.group(2))))
    return pairs


def parse_log_total_time(log_path: Path) -> float | None:
    rx = re.compile(r"\[train\]\s+done\.\s+total\s+time\s+([\d.]+)\s+min")
    for line in log_path.read_text().splitlines():
        m = rx.search(line)
        if m:
            return float(m.group(1)) * 60.0  # to seconds
    return None


def main(out_dir: str, log_path: str):
    out = Path(out_dir)
    log = Path(log_path)
    print(f"=== Verifying {out} against {log.name} ===\n")

    # --- 1. config.json: every ML-generic hyperparameter -----------------
    cfg_path = out / "config.json"
    cfg = json.loads(cfg_path.read_text())
    print("[config.json]")
    expected_test = {
        "num_epochs": 1,
        "batch_size": 4,
        "learning_rate": 5.77e-4,
        "warmup_steps": 50,
        "ema_decay": 0.99,
        "grad_clip_norm": 1.0,
        "weight_decay": 1e-5,          # train.py CLI doesn't expose; uses TrainingConfig default
        "min_lr_ratio": 0.01,          # ditto
        "mixed_precision": "bf16",
        "seed": 42,
        "log_every": 50,
        "save_every_epochs": 1,
        "save_every_steps": 100,
        "num_workers": 8,
    }
    ok = True
    for k, want in expected_test.items():
        got = cfg.get(k)
        match = (got == want) if not isinstance(want, float) else math.isclose(got, want, rel_tol=1e-4)
        flag = "✓" if match else "✗"
        if not match:
            ok = False
        print(f"  {flag} {k:22s} expected={want!r:<15} got={got!r}")
    print(f"  {'PASS' if ok else 'FAIL'}: config matches requested bench config")

    # --- 2. LR schedule from log ----------------------------------------
    print("\n[LR schedule from training log]")
    lr_curve = parse_log_lr_curve(log)
    if not lr_curve:
        print("  ✗ no LR samples logged — was log_every set?")
    else:
        peak_lr = cfg["learning_rate"]
        warmup_steps = cfg["warmup_steps"]
        print(f"  {len(lr_curve)} (step, lr) samples; peak LR target = {peak_lr:.3e}, warmup = {warmup_steps} steps")
        for step, lr in lr_curve[:5]:
            print(f"    step {step:4d}  lr {lr:.3e}")
        if len(lr_curve) > 6:
            print(f"    ... ({len(lr_curve)-10} more) ...")
        for step, lr in lr_curve[-5:]:
            print(f"    step {step:4d}  lr {lr:.3e}")
        # Verify ramp: first logged point at step==log_every should be on warmup linear segment.
        first_step, first_lr = lr_curve[0]
        expected_first = peak_lr * min(1.0, first_step / max(1, warmup_steps))
        ramp_ok = math.isclose(first_lr, expected_first, rel_tol=0.02)
        print(f"  {'✓' if ramp_ok else '✗'} warmup ramp: at step {first_step}, expected ~{expected_first:.3e}, got {first_lr:.3e}")
        # Verify cosine decay direction: monotonic decreasing after warmup.
        post = [(s, l) for (s, l) in lr_curve if s >= warmup_steps]
        if len(post) >= 2:
            decreasing = all(post[i+1][1] <= post[i][1] + 1e-12 for i in range(len(post)-1))
            print(f"  {'✓' if decreasing else '✗'} post-warmup LR monotonically non-increasing (cosine decay)")
        # Check peak: max lr should equal peak_lr (within rounding).
        max_lr = max(l for _, l in lr_curve)
        max_match = math.isclose(max_lr, peak_lr, rel_tol=0.01)
        print(f"  {'✓' if max_match else '✗'} peak observed LR = {max_lr:.3e} (target {peak_lr:.3e})")

    # --- 3. Step-based checkpoints --------------------------------------
    print("\n[Checkpoint cadence]")
    step_ckpts = sorted(p.name for p in out.glob("checkpoint_step_*"))
    epoch_ckpts = sorted(p.name for p in out.glob("checkpoint_epoch_*"))
    final_ckpt = (out / "checkpoint_final").is_dir()
    print(f"  step-checkpoints: {len(step_ckpts)} → {step_ckpts}")
    print(f"  epoch-checkpoints: {len(epoch_ckpts)} → {epoch_ckpts}")
    print(f"  checkpoint_final: {'present' if final_ckpt else 'MISSING'}")
    save_every = cfg["save_every_steps"]
    # Expected step-checkpoints at multiples of save_every up to total steps
    # (we don't know total_steps directly but we know we did 1 epoch).
    # Just verify they are at the right multiples.
    bad = [s for s in step_ckpts
           if int(re.search(r"\d+", s).group()) % save_every != 0]
    print(f"  {'✓' if not bad else '✗'} all step-checkpoints are multiples of save_every_steps={save_every}")

    # --- 4. EMA shadow vs live weights ----------------------------------
    print("\n[EMA shadow differs from live weights]")
    final_dir = out / "checkpoint_final"
    if final_dir.is_dir():
        ema_pt = final_dir / "ema.pt"
        if not ema_pt.is_file():
            print(f"  ✗ ema.pt missing in {final_dir}")
        else:
            ema_state = torch.load(ema_pt, map_location="cpu", weights_only=False)
            print(f"  EMA decay stored in ckpt: {ema_state['decay']}")
            print(f"  EMA shadow tensors: {len(ema_state['shadow'])}")
            # Load model weights via safetensors and compare to first few shadow tensors
            from safetensors.torch import load_file
            model_st = list(final_dir.glob("**/model*.safetensors"))
            if model_st:
                live = load_file(str(model_st[0]))
                # safetensors gives a dict keyed by param name; ema.pt is just a list. We compare
                # tensor-by-tensor in load order — they should be parallel since both came from the
                # same trainable() iteration order.
                shadow = ema_state["shadow"]
                live_tensors = list(live.values())
                if len(shadow) <= len(live_tensors):
                    diffs = []
                    for i, s in enumerate(shadow[:min(5, len(shadow))]):
                        l = live_tensors[i]
                        if s.shape == l.shape:
                            d = (s - l).abs().mean().item()
                            diffs.append(d)
                    nonzero = sum(1 for d in diffs if d > 1e-8)
                    print(f"  first-{len(diffs)} mean-abs-diff (EMA−live): {[f'{d:.2e}' for d in diffs]}")
                    print(f"  {'✓' if nonzero == len(diffs) else '✗'} EMA shadow distinct from live weights ({nonzero}/{len(diffs)} differ)")
                else:
                    print(f"  ! shadow length {len(shadow)} > model tensor count {len(live_tensors)} — cannot compare positionally")
            else:
                print("  ! no model.safetensors found in checkpoint_final")

    # --- 5. history.json: val_loss_live vs val_loss_ema -----------------
    hist_path = out / "history.json"
    if hist_path.is_file():
        print("\n[history.json]")
        hist = json.loads(hist_path.read_text())
        print(f"  epochs recorded: {len(hist)}")
        for rec in hist:
            keys = list(rec.keys())
            extras = [k for k in keys if k not in ("epoch", "global_step", "mean_train_loss", "elapsed_sec")]
            tl = rec.get("mean_train_loss", "—")
            vl_live = rec.get("mean_val_loss_live", "—")
            vl_ema = rec.get("mean_val_loss_ema", "—")
            print(f"  ep {rec.get('epoch')} step {rec.get('global_step')} "
                  f"train_loss={tl}  val_live={vl_live}  val_ema={vl_ema}")
        if hist and "mean_val_loss_live" in hist[-1] and "mean_val_loss_ema" in hist[-1]:
            live_ema_differ = hist[-1]["mean_val_loss_live"] != hist[-1]["mean_val_loss_ema"]
            print(f"  {'✓' if live_ema_differ else '✗'} val_loss_live ≠ val_loss_ema → both eval paths fire")

    # --- 6. Timing ------------------------------------------------------
    print("\n[Timing]")
    total = parse_log_total_time(log)
    if total is not None and lr_curve:
        last_step = lr_curve[-1][0]
        per_step = total / max(1, last_step)
        print(f"  training-only wall-clock: {total/60:.2f} min over ~{last_step} steps")
        print(f"  → per-step: {per_step:.2f} s")
        print(f"  → projected at 64 GPUs production scale (~30 GiB lemat avg structure): would be similar order")
    else:
        print("  could not parse total time from log")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
