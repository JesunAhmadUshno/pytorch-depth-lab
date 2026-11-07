"""A7 data-loading benchmark, A10 memory probe, compile probes.

A7: loader-only throughput for the frozen loader configurations
L0 to L3, plus a training step-time decomposition (data wait vs
compute) at L0 and L3.
A10: peak allocated memory over 60 steps at batch
{32, 64, 128, 256} with and without gradient checkpointing, and a
max-feasible-batch probe at 512.
Compile probes: torch.compile first-step time for default mode,
and a fullgraph=True attempt whose success or exact error is
recorded (pack stop rule S2 evidence).
"""
from __future__ import annotations

import argparse
import json
import time

import torch
from torch import nn

from .common import RESULTS, env_info, seed_all, write_json
from .data import make_datasets, make_loader
from .model import build_model

LOADER_CONFIGS = {
    "L0": {"workers": 0, "pin": False, "persistent": False},
    "L1": {"workers": 0, "pin": True, "persistent": False},
    "L2": {"workers": 2, "pin": True, "persistent": False},
    "L3": {"workers": 4, "pin": True, "persistent": True},
}


def loader_throughput(device: torch.device, epochs: int = 3) -> list[dict]:
    train_ds, _, _ = make_datasets()
    rows = []
    for name, cfg in LOADER_CONFIGS.items():
        loader = make_loader(
            train_ds, 128, shuffle=True, workers=cfg["workers"],
            pin=cfg["pin"] and device.type == "cuda",
            persistent=cfg["persistent"], seed=42,
        )
        images = 0
        t0 = time.perf_counter()
        for _ in range(epochs):
            for x, _y in loader:
                images += x.shape[0]
        dt = time.perf_counter() - t0
        rows.append({**cfg, "config": name, "images": images, "time_s": dt,
                     "loader_img_s": images / dt})
    return rows


def step_decomposition(device: torch.device, steps: int = 100) -> dict:
    """Per-step data-wait vs compute at L0, L3, and L0 with
    channels_last (the two candidate optimizations of the frozen
    A8 rule), batch 128."""
    train_ds, _, _ = make_datasets()
    out = {}
    for name, channels_last in (("L0", False), ("L3", False), ("L0CL", True)):
        cfg = LOADER_CONFIGS["L0" if name == "L0CL" else name]
        seed_all(42)
        model = build_model(False, 42).to(device)
        if channels_last:
            model = model.to(memory_format=torch.channels_last)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        loader = make_loader(
            train_ds, 128, shuffle=True, workers=cfg["workers"],
            pin=cfg["pin"] and device.type == "cuda",
            persistent=cfg["persistent"], seed=42,
        )
        data_wait, compute, n = 0.0, 0.0, 0
        it = iter(loader)
        t_fetch = time.perf_counter()
        for i in range(steps):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            fetch_s = time.perf_counter() - t_fetch
            if device.type == "cuda":
                torch.cuda.synchronize()
            c0 = time.perf_counter()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if channels_last:
                x = x.to(memory_format=torch.channels_last)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            compute += time.perf_counter() - c0
            # Skip the first fetch (loader spin-up) from data wait.
            if i > 0:
                data_wait += fetch_s
                n += 1
            t_fetch = time.perf_counter()
        total = data_wait + compute
        out[name] = {
            "steps": steps, "data_wait_s": data_wait, "compute_s": compute,
            "data_wait_share": data_wait / total if total > 0 else 0.0,
            "mean_step_ms": 1000 * total / max(n, 1),
        }
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def memory_probe(device: torch.device, steps: int = 60) -> dict:
    train_ds, _, _ = make_datasets()
    cells = []
    for ckpt in (False, True):
        for batch in (32, 64, 128, 256):
            seed_all(42)
            loader = make_loader(train_ds, batch, shuffle=True, workers=0, seed=42)
            model = build_model(False, 42, checkpointing=ckpt).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            crit = nn.CrossEntropyLoss()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            it = iter(loader)
            for _ in range(steps):
                x, y = next(it)
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = crit(model(x), y)
                loss.backward()
                opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            peak = (
                torch.cuda.max_memory_allocated() / (1024 * 1024)
                if device.type == "cuda" else None
            )
            cells.append({
                "checkpointing": ckpt, "batch": batch, "steps": steps,
                "peak_mem_mib": peak, "step_ms": 1000 * dt / steps,
            })
            del model, opt
            if device.type == "cuda":
                torch.cuda.empty_cache()
    # Max feasible batch probe at 512, 20 steps.
    feasible = {}
    for ckpt in (False, True):
        seed_all(42)
        loader = make_loader(train_ds, 512, shuffle=True, workers=0, seed=42)
        model = build_model(False, 42, checkpointing=ckpt).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        ok, err = True, None
        try:
            it = iter(loader)
            for _ in range(20):
                x, y = next(it)
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = crit(model(x), y)
                loss.backward()
                opt.step()
        except RuntimeError as exc:  # OOM is a result, not a crash
            ok, err = False, str(exc)[:300]
        feasible[f"ckpt_{ckpt}"] = {"batch": 512, "ok": ok, "error": err}
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {"cells": cells, "feasible_512": feasible}


def compile_probes(device: torch.device) -> dict:
    """Default-mode first-step cost and the fullgraph=True attempt."""
    train_ds, _, _ = make_datasets()
    loader = make_loader(train_ds, 128, shuffle=True, workers=0, seed=42)
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    out: dict = {"env": env_info(device)}
    for label, kwargs in (("default", {}), ("fullgraph", {"fullgraph": True})):
        seed_all(42)
        model = build_model(False, 42).to(device)
        crit = nn.CrossEntropyLoss()
        entry: dict = {"mode_kwargs": str(kwargs)}
        try:
            cmodel = torch.compile(model, **kwargs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss = crit(cmodel(x), y)
            loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            entry["first_step_s"] = time.perf_counter() - t0
            entry["status"] = "ok"
        except Exception as exc:  # failure modes are results (S2)
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        out[label] = entry
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--which", default="all", choices=["all", "loader", "memory", "compile"])
    args = ap.parse_args()
    device = torch.device(args.device)
    if args.which in ("all", "loader"):
        payload = {
            "env": env_info(device),
            "loader_throughput": loader_throughput(device),
            "step_decomposition": step_decomposition(device),
        }
        write_json(RESULTS / "benchmarks" / "a7_loader.json", payload)
        print(json.dumps(payload["step_decomposition"], indent=2))
    if args.which in ("all", "memory"):
        payload = {"env": env_info(device), **memory_probe(device)}
        write_json(RESULTS / "benchmarks" / "a10_memory.json", payload)
        print(json.dumps(payload["cells"], indent=2))
    if args.which in ("all", "compile"):
        payload = compile_probes(device)
        write_json(RESULTS / "benchmarks" / "compile_probe.json", payload)
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
