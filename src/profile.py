"""A8: torch.profiler study.

Profiles 30 training steps of a configuration and saves the top
operator table (share of self CUDA time, or self CPU time on a CPU
device) plus the chrome trace (results/profiles/, not committed).
The H4 bottleneck call itself combines this table with the A7
step-time decomposition; both artefacts are published.
"""
from __future__ import annotations

import argparse
import json

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile

from .common import RESULTS, env_info, seed_all, write_json
from .data import make_datasets, make_loader
from .model import build_model

PROFILES = RESULTS / "profiles"


def run_profile(device: torch.device, workers: int, pin: bool,
                channels_last: bool, steps: int = 30) -> dict:
    seed_all(42)
    train_ds, _, _ = make_datasets()
    loader = make_loader(
        train_ds, 128, shuffle=True, workers=workers,
        pin=pin and device.type == "cuda", persistent=workers > 0, seed=42,
    )
    model = build_model(False, 42).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    acts = [ProfilerActivity.CPU]
    if device.type == "cuda":
        acts.append(ProfilerActivity.CUDA)
    it = iter(loader)
    batches = []
    for _ in range(steps + 5):
        try:
            batches.append(next(it))
        except StopIteration:
            it = iter(loader)
            batches.append(next(it))
    # Warmup outside the trace.
    for x, y in batches[:5]:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if channels_last:
            x = x.to(memory_format=torch.channels_last)
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    PROFILES.mkdir(parents=True, exist_ok=True)
    tag = f"w{workers}_{'cl' if channels_last else 'std'}"
    with profile(activities=acts, record_shapes=False) as prof:
        for x, y in batches[5:]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if channels_last:
                x = x.to(memory_format=torch.channels_last)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            prof.step()
    prof.export_chrome_trace(str(PROFILES / f"trace_{tag}.json"))
    avgs = list(prof.key_averages())
    # torch 2.11 renamed the CUDA self-time attribute, and on
    # stacks where CUPTI fails to initialise (seen on BlackBox:
    # CUPTI_ERROR_INVALID_DEVICE) only CPU time exists. Pick the
    # strongest metric actually present and say which was used.
    key = "self_cpu_time_total"
    if avgs:
        for cand in ("self_device_time_total", "self_cuda_time_total"):
            if hasattr(avgs[0], cand) and sum(getattr(a, cand) for a in avgs) > 0:
                key = cand
                break
    total = sum(getattr(a, key) for a in avgs) or 1
    table = [
        {"op": a.key, "self_time_us": getattr(a, key),
         "share": getattr(a, key) / total, "count": a.count}
        for a in sorted(avgs, key=lambda a: -getattr(a, key))[:15]
    ]
    return {
        "env": env_info(device),
        "config": {"workers": workers, "pin": pin, "channels_last": channels_last,
                   "steps_profiled": steps, "batch": 128},
        "metric": key,
        "cuda_activity_captured": key != "self_cpu_time_total",
        "top_ops": table,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--pin", action="store_true")
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    payload = run_profile(
        torch.device(args.device), args.workers, args.pin, args.channels_last
    )
    write_json(RESULTS / "benchmarks" / args.out, payload)
    print(json.dumps(payload["top_ops"][:5], indent=2))


if __name__ == "__main__":
    main()
