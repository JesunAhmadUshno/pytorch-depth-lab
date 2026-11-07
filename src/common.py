"""Shared recipe, seeding, environment capture, and IO helpers.

Every arm reads the frozen recipe from here (PREREG.md). The
Lightning arm publishes its own effective config and
src/config_diff.py diffs the two.
"""
from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
RUNS = RESULTS / "runs"

# Frozen recipe (PREREG.md). Do not edit without a PREREG amendment.
RECIPE = {
    "model": "torchvision ResNet-18, fc replaced by Linear(512, 10)",
    "optimizer": "AdamW",
    "lr": 1e-3,
    "betas": (0.9, 0.999),
    "weight_decay": 1e-4,
    "scheduler": "CosineAnnealingLR",
    "loss": "CrossEntropyLoss (label_smoothing=0.0)",
    "batch_size": 128,
    "epochs_full": 15,
    "epochs_cpu": 3,
    "augmentation": "RandomHorizontalFlip(0.5), RandomVerticalFlip(0.5), train only",
    "normalisation_mean": (0.485, 0.456, 0.406),
    "normalisation_std": (0.229, 0.224, 0.225),
    "cudnn_benchmark": False,
    "split_seed": 20261107,
    "tta_threshold": 0.96,
}

IMAGENET_MEAN = RECIPE["normalisation_mean"]
IMAGENET_STD = RECIPE["normalisation_std"]


def recipe_dict() -> dict:
    """The hand-written arm's effective config, for the config diff."""
    return {
        "optimizer": "AdamW",
        "lr": RECIPE["lr"],
        "betas": list(RECIPE["betas"]),
        "weight_decay": RECIPE["weight_decay"],
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
    }


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def env_info(device: torch.device) -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cpu_count": os.cpu_count(),
        "device": str(device),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        import torchvision

        info["torchvision"] = torchvision.__version__
    except Exception:
        pass
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
    return info


def gpu_snapshot() -> dict | None:
    """nvidia-smi state (clocks, temperature, power), or None."""
    exe = "nvidia-smi"
    if platform.system() == "Windows":
        cand = r"C:\Windows\System32\nvidia-smi.exe"
        exe = cand if os.path.exists(cand) else "nvidia-smi"
    try:
        out = subprocess.run(
            [
                exe,
                "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,"
                "power.draw,driver_version,name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if out.returncode != 0:
            return None
        parts = [p.strip() for p in out.stdout.strip().split(",")]
        keys = ["sm_clock_mhz", "mem_clock_mhz", "temp_c", "power_w", "driver", "name"]
        return dict(zip(keys, parts))
    except Exception:
        return None


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return path


def fmt_ci(lo: float, hi: float) -> str:
    return f"[{lo:.4f}, {hi:.4f}]"
