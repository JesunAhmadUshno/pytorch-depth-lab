"""A3: the PyTorch Lightning baseline arm.

Fairness contract (PREREG.md): the frozen recipe exactly. Same
model build, same optimizer and schedule, same loss, same loader
construction and generator seed, same evaluation code (the shared
evaluate() from src/train.py runs at each train-epoch end from a
callback, so Lightning's own validation loop is not used).
Throughput uses per-epoch train wall time from callbacks, the same
definition as the hand loop (fetch waits included, evaluation
excluded, epoch 1 excluded).
"""
from __future__ import annotations

import argparse
import json
import time

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback

from .common import RECIPE, RUNS, env_info, gpu_snapshot, seed_all, write_json
from .data import make_datasets, make_loader
from .model import build_model
from .train import evaluate


class LitResNet(L.LightningModule):
    def __init__(self, seed: int, epochs: int):
        super().__init__()
        self.model = build_model(False, seed)
        self.epochs = epochs

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self.model(x), y)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=RECIPE["lr"], betas=RECIPE["betas"],
            weight_decay=RECIPE["weight_decay"],
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        return {"optimizer": opt, "lr_scheduler": sched}


def effective_config() -> dict:
    """The Lightning arm's config, read off the objects it builds,
    for the config-diff audit."""
    m = LitResNet(0, RECIPE["epochs_full"])
    cfg = m.configure_optimizers()
    opt = cfg["optimizer"]
    g = opt.param_groups[0]
    return {
        "optimizer": type(opt).__name__,
        "lr": g["lr"],
        "betas": list(g["betas"]),
        "weight_decay": g["weight_decay"],
        "scheduler": "CosineAnnealingLR(T_max=epochs)",
        "loss": "CrossEntropyLoss",
        "label_smoothing": 0.0,
        "batch_size": RECIPE["batch_size"],
        "augmentation": RECIPE["augmentation"],
        "normalisation": "ImageNet mean/std",
        "cudnn_benchmark": False,
        "precision": "float32 (unless the arm varies it)",
        "sampler": "RandomSampler with per-epoch generator seeded by run seed + epoch",
        "data_order": "DataLoader passed explicitly, generator seeded",
        "lightning_trainer": "logger off, checkpointing off, progress bar off, "
        "sanity steps 0, validation loop unused (shared evaluate via callback)",
    }


class EpochProbe(Callback):
    """Records per-epoch train wall time, then runs the shared
    evaluator (after the timing mark, so eval is excluded)."""

    def __init__(self, val_loader, test_loader, device, threshold):
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.threshold = threshold
        self.rows: list[dict] = []
        self._t0 = None
        self.run_t0 = None
        self.tta_s = None

    def on_train_start(self, trainer, pl_module):
        self.run_t0 = time.perf_counter()

    def on_train_epoch_start(self, trainer, pl_module):
        self._t0 = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        train_wall = time.perf_counter() - self._t0
        acc, nll = evaluate(pl_module.model, self.val_loader, self.device)
        wall = time.perf_counter() - self.run_t0
        if self.tta_s is None and acc >= self.threshold:
            self.tta_s = wall
        self.rows.append(
            {
                "epoch": len(self.rows) + 1,
                "train_wall_s": train_wall,
                "throughput_img_s": 0.0,  # filled by run_lightning below
                "val_acc": acc,
                "val_nll": nll,
            }
        )


def run_lightning(seed: int, epochs: int, device: str) -> dict:
    seed_all(seed)
    dev = torch.device(device)
    train_ds, val_ds, test_ds = make_datasets()
    train_loader = make_loader(train_ds, RECIPE["batch_size"], shuffle=True, workers=0, seed=seed)
    val_loader = make_loader(val_ds, RECIPE["batch_size"], shuffle=False, workers=0)
    test_loader = make_loader(test_ds, RECIPE["batch_size"], shuffle=False, workers=0)
    n_train = len(train_ds)

    lit = LitResNet(seed, epochs)
    probe = EpochProbe(val_loader, test_loader, dev, RECIPE["tta_threshold"])
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    snap_start = gpu_snapshot() if dev.type == "cuda" else None
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if dev.type == "cuda" else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=[probe],
        deterministic=False,
    )
    t0 = time.perf_counter()
    trainer.fit(lit, train_dataloaders=train_loader)
    wall = time.perf_counter() - t0
    # Lightning restores the module to its pre-fit device (CPU)
    # when fit returns; move it back before the final evaluation.
    lit.model.to(dev)
    # Same throughput definition as the hand loop: epochs 2..N.
    img_total, time_total = 0, 0.0
    for i, row in enumerate(probe.rows):
        row["throughput_img_s"] = n_train / row["train_wall_s"]
        if i >= 1:
            img_total += n_train
            time_total += row["train_wall_s"]
    test_acc, test_nll, correct = evaluate(lit.model, test_loader, dev, want_correct=True)
    peak_mem = None
    if dev.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
    run_id = f"a3_lightning_s{seed}"
    result = {
        "run_id": run_id,
        "arm": "A3",
        "init": "scratch",
        "precision": "fp32",
        "compiled": False,
        "checkpointing": False,
        "channels_last": False,
        "epochs": epochs,
        "seed": seed,
        "batch_size": RECIPE["batch_size"],
        "workers": 0,
        "framework": f"lightning {L.__version__}",
        "env": env_info(dev),
        "gpu_start": snap_start,
        "gpu_end": gpu_snapshot() if dev.type == "cuda" else None,
        "epoch_rows": probe.rows,
        "throughput_img_s": img_total / time_total if time_total > 0 else 0.0,
        "peak_mem_mib": peak_mem,
        "tta_s": probe.tta_s,
        "test_acc": test_acc,
        "test_nll": test_nll,
        "first_step_s": None,
        "compile_step_s": None,
        "wall_s": wall,
        "command": f"python -m src.lightning_arm --seed {seed} --epochs {epochs} --device {device}",
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    write_json(RUNS / f"{run_id}.json", result)
    np.savez_compressed(RUNS / f"{run_id}_testcorrect.npz", correct=correct)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    res = run_lightning(args.seed, args.epochs, args.device)
    print(json.dumps({k: res[k] for k in ("run_id", "throughput_img_s", "test_acc", "test_nll")}, indent=2))


if __name__ == "__main__":
    main()
