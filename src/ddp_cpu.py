"""CPU arms C1 (single process) and C2 (DDP, gloo, world size 2).

Reduced sub-protocol frozen in PREREG.md: 3 epochs, effective
batch 128, seed 42, same optimizer and schedule (T_max = 3),
loader workers 0, 2 torch threads per process. C2 uses
DistributedSampler and per-rank batch 64. This is a correctness
arm: it tests whether DDP converges to the same recipe, and it
makes no speed or scaling claim.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from .common import RECIPE, RESULTS, env_info, seed_all, write_json
from .data import make_datasets, make_loader
from .model import build_model
from .train import evaluate


def _train(rank: int, world: int, epochs: int, out: dict) -> None:
    torch.set_num_threads(2)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    seed_all(42)
    if world > 1:
        dist.init_process_group("gloo", rank=rank, world_size=world)
    device = torch.device("cpu")
    train_ds, val_ds, _ = make_datasets()
    per_rank_batch = RECIPE["batch_size"] // world
    sampler = None
    if world > 1:
        sampler = DistributedSampler(
            train_ds, num_replicas=world, rank=rank, shuffle=True, seed=42
        )
    loader = make_loader(
        train_ds, per_rank_batch, shuffle=sampler is None, workers=0,
        seed=42, sampler=sampler,
    )
    val_loader = make_loader(val_ds, RECIPE["batch_size"], shuffle=False, workers=0)
    model = build_model(False, 42)
    if world > 1:
        model = DDP(model)
    opt = torch.optim.AdamW(
        model.parameters(), lr=RECIPE["lr"], betas=RECIPE["betas"],
        weight_decay=RECIPE["weight_decay"],
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    rows = []
    t0 = time.perf_counter()
    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        ep_t0 = time.perf_counter()
        loss_sum, seen = 0.0, 0
        model.train()
        for x, y in loader:
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * y.numel()
            seen += y.numel()
        sched.step()
        mean_loss = loss_sum / max(seen, 1)
        if world > 1:
            t = torch.tensor([mean_loss * seen, float(seen)])
            dist.all_reduce(t)
            mean_loss = (t[0] / t[1]).item()
        row = {"epoch": epoch + 1, "train_loss": mean_loss,
               "epoch_wall_s": time.perf_counter() - ep_t0}
        if rank == 0:
            raw = model.module if world > 1 else model
            acc, nll = evaluate(raw, val_loader, device)
            row["val_acc"] = acc
            row["val_nll"] = nll
        rows.append(row)
    if rank == 0:
        out.update(
            {
                "arm": "C2" if world > 1 else "C1",
                "world_size": world,
                "epochs": epochs,
                "seed": 42,
                "per_rank_batch": per_rank_batch,
                "effective_batch": RECIPE["batch_size"],
                "env": env_info(device),
                "epoch_rows": rows,
                "wall_s": time.perf_counter() - t0,
                "command": f"python -m src.ddp_cpu --world {world}",
            }
        )
    if world > 1:
        dist.destroy_process_group()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=1, choices=[1, 2])
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()
    out: dict = {}
    if args.world == 1:
        _train(0, 1, args.epochs, out)
    else:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29517")
        import torch.multiprocessing as mp

        # spawn passes args positionally; wrap to collect rank-0 output
        # through a manager dict.
        manager = mp.Manager()
        shared = manager.dict()
        mp.spawn(_train, args=(args.world, args.epochs, shared), nprocs=args.world)
        out = dict(shared)
    name = "c2_ddp" if args.world > 1 else "c1_single"
    write_json(RESULTS / "cpu" / f"{name}.json", out)
    print(json.dumps(out.get("epoch_rows", []), indent=2))


if __name__ == "__main__":
    main()
