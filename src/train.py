"""The hand-written training loop, and the shared evaluation path.

One run_training() serves every hand-written arm; arms differ only
by the frozen factors under test (initialisation, precision,
torch.compile, gradient checkpointing, loader configuration,
channels_last). Evaluation lives here so the Lightning arm scores
its model through exactly the same code.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .common import RECIPE, RESULTS, RUNS, env_info, gpu_snapshot, seed_all, write_json
from .data import make_datasets, make_loader
from .model import build_model


@torch.no_grad()
def evaluate(model, loader, device, want_correct: bool = False):
    """Returns (accuracy, mean NLL). Optionally also the per-example
    correctness array in loader order (the test loader is never
    shuffled, so indices align across arms)."""
    model.eval()
    total, correct, nll_sum = 0, 0, 0.0
    flags: list[np.ndarray] = []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        nll_sum += F.cross_entropy(logits, y, reduction="sum").item()
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
        if want_correct:
            flags.append((pred == y).cpu().numpy())
    out = (correct / total, nll_sum / total)
    if want_correct:
        return out + (np.concatenate(flags),)
    return out


@dataclass
class TrainConfig:
    arm: str
    init: str = "scratch"  # scratch | pretrained
    precision: str = "fp32"  # fp32 | amp
    compiled: bool = False
    checkpointing: bool = False
    channels_last: bool = False
    epochs: int = 15
    seed: int = 42
    device: str = "cuda"
    workers: int = 0
    pin: bool = False
    persistent: bool = False
    batch_size: int = 128
    max_steps: int | None = None  # probe cap per epoch
    limit_train: int | None = None  # test-only subset
    run_id: str = ""


def run_training(cfg: TrainConfig) -> dict:
    seed_all(cfg.seed)
    device = torch.device(cfg.device)
    run_id = cfg.run_id or f"{cfg.arm.lower()}_s{cfg.seed}"
    train_ds, val_ds, test_ds = make_datasets()
    if cfg.limit_train is not None:
        from torch.utils.data import Subset

        train_ds = Subset(train_ds, list(range(cfg.limit_train)))
    pin = cfg.pin and device.type == "cuda"
    train_loader = make_loader(
        train_ds, cfg.batch_size, shuffle=True, workers=cfg.workers,
        pin=pin, persistent=cfg.persistent, seed=cfg.seed,
    )
    val_loader = make_loader(val_ds, cfg.batch_size, shuffle=False, workers=0)
    test_loader = make_loader(test_ds, cfg.batch_size, shuffle=False, workers=0)

    model = build_model(cfg.init == "pretrained", cfg.seed, cfg.checkpointing).to(device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(
        model.parameters(), lr=RECIPE["lr"], betas=RECIPE["betas"],
        weight_decay=RECIPE["weight_decay"],
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.precision == "amp" and device.type == "cuda")
    amp_on = cfg.precision == "amp" and device.type == "cuda"

    compile_time_s = None
    if cfg.compiled:
        model = torch.compile(model)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    snap_start = gpu_snapshot() if device.type == "cuda" else None

    run_t0 = time.perf_counter()
    epoch_rows = []
    tta_s = None
    step_time_total, images_total = 0.0, 0  # epochs 2..N accounting
    first_step_s = None
    for epoch in range(cfg.epochs):
        model.train()
        ep_t0 = time.perf_counter()
        step_time, images, loss_sum, seen = 0.0, 0, 0.0, 0
        for i, (x, y) in enumerate(train_loader):
            if cfg.max_steps is not None and i >= cfg.max_steps:
                break
            if device.type == "cuda":
                torch.cuda.synchronize()
            s0 = time.perf_counter()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if cfg.channels_last:
                x = x.to(memory_format=torch.channels_last)
            opt.zero_grad(set_to_none=True)
            if amp_on:
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = crit(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss = crit(model(x), y)
                loss.backward()
                opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - s0
            if first_step_s is None:
                first_step_s = dt
                if cfg.compiled:
                    compile_time_s = dt
            step_time += dt
            images += y.numel()
            loss_sum += loss.item() * y.numel()
            seen += y.numel()
        train_wall = time.perf_counter() - ep_t0
        sched.step()
        val_acc, val_nll = evaluate(model, val_loader, device)
        wall = time.perf_counter() - run_t0
        if tta_s is None and val_acc >= RECIPE["tta_threshold"]:
            tta_s = wall
        epoch_rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": loss_sum / max(seen, 1),
                "train_step_time_s": step_time,
                "train_wall_s": train_wall,
                "epoch_wall_s": time.perf_counter() - ep_t0,
                "throughput_img_s": images / train_wall if train_wall > 0 else 0.0,
                "val_acc": val_acc,
                "val_nll": val_nll,
                "lr": sched.get_last_lr()[0],
            }
        )
        if epoch >= 1:  # frozen throughput definition excludes epoch 1
            step_time_total += train_wall
            images_total += images

    test_acc, test_nll, correct = evaluate(model, test_loader, device, want_correct=True)
    peak_mem = None
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
    result = {
        "run_id": run_id,
        "arm": cfg.arm,
        "init": cfg.init,
        "precision": cfg.precision,
        "compiled": cfg.compiled,
        "checkpointing": cfg.checkpointing,
        "channels_last": cfg.channels_last,
        "epochs": cfg.epochs,
        "seed": cfg.seed,
        "batch_size": cfg.batch_size,
        "workers": cfg.workers,
        "env": env_info(device),
        "gpu_start": snap_start,
        "gpu_end": gpu_snapshot() if device.type == "cuda" else None,
        "epoch_rows": epoch_rows,
        "throughput_img_s": images_total / step_time_total if step_time_total > 0 else 0.0,
        "peak_mem_mib": peak_mem,
        "tta_s": tta_s,
        "test_acc": test_acc,
        "test_nll": test_nll,
        "first_step_s": first_step_s,
        "compile_step_s": compile_time_s,
        "wall_s": time.perf_counter() - run_t0,
        "command": f"python -m src.train --arm {cfg.arm} --init {cfg.init} "
        f"--precision {cfg.precision} --epochs {cfg.epochs} --seed {cfg.seed} "
        f"--device {cfg.device}",
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    write_json(RUNS / f"{run_id}.json", result)
    np.savez_compressed(RUNS / f"{run_id}_testcorrect.npz", correct=correct)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--init", default="scratch", choices=["scratch", "pretrained"])
    ap.add_argument("--precision", default="fp32", choices=["fp32", "amp"])
    ap.add_argument("--compiled", action="store_true")
    ap.add_argument("--checkpointing", action="store_true")
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--pin", action="store_true")
    ap.add_argument("--persistent", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--run-id", default="")
    args = ap.parse_args()
    cfg = TrainConfig(
        arm=args.arm, init=args.init, precision=args.precision,
        compiled=args.compiled, checkpointing=args.checkpointing,
        channels_last=args.channels_last, epochs=args.epochs, seed=args.seed,
        device=args.device, workers=args.workers, pin=args.pin,
        persistent=args.persistent, max_steps=args.max_steps, run_id=args.run_id,
    )
    res = run_training(cfg)
    print(json.dumps({k: res[k] for k in ("run_id", "throughput_img_s", "test_acc", "test_nll", "peak_mem_mib", "tta_s")}, indent=2))


if __name__ == "__main__":
    main()
