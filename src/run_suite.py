"""Suite runner. Executes the frozen arm list on one machine.

GPU mode (BlackBox): data verify, config diff, the six full arms at
seeds 42 and 43 in randomised order (seed 20261107, written to
results/run_order.json before the first arm starts), then the
benchmarks in dependency order (A7 loader, A8 profiles chosen by
the frozen rule from the A7 decomposition, A10 memory, compile
probes, A9 FSDP2 check).
CPU mode (this VM): data verify, C1, C2.

Each step runs as a subprocess; a failed step is recorded in the
suite log and the suite continues, because a failed arm is a
result under the stop rules, not a reason to lose the rest.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

from .common import REPO_ROOT, RESULTS, env_info, gpu_snapshot, write_json
import torch


def run_step(log, label: str, cmd: list[str]) -> bool:
    log.write(f"\n=== {label}: {' '.join(cmd)}\n")
    log.flush()
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    log.write(proc.stdout[-4000:])
    log.write(proc.stderr[-4000:])
    ok = proc.returncode == 0
    log.write(f"--- {label} rc={proc.returncode} in {time.time() - t0:.0f}s\n")
    log.flush()
    print(f"{label}: rc={proc.returncode}", flush=True)
    return ok


def versions(machine: str) -> None:
    import torchvision

    payload = {
        "machine": machine,
        "env": env_info(torch.device("cuda" if machine == "gpu" else "cpu")),
        "torchvision": torchvision.__version__,
        "gpu_snapshot": gpu_snapshot() if machine == "gpu" else None,
    }
    try:
        import lightning

        payload["lightning"] = lightning.__version__
    except Exception:
        pass
    write_json(RESULTS / f"versions_{machine}.json", payload)


def gpu_suite(log) -> None:
    py = sys.executable
    run_step(log, "data-verify", [py, "-m", "src.data", "--verify"])
    run_step(log, "config-diff", [py, "-m", "src.config_diff"])
    arms: list[tuple[str, list[str]]] = []
    for seed in (42, 43):
        arms += [
            (f"A1_s{seed}", [py, "-m", "src.train", "--arm", "A1", "--seed", str(seed), "--device", "cuda"]),
            (f"A2_s{seed}", [py, "-m", "src.train", "--arm", "A2", "--init", "pretrained", "--seed", str(seed), "--device", "cuda"]),
            (f"A3_s{seed}", [py, "-m", "src.lightning_arm", "--seed", str(seed), "--device", "cuda"]),
            (f"A4_s{seed}", [py, "-m", "src.train", "--arm", "A4", "--precision", "amp", "--seed", str(seed), "--device", "cuda"]),
            (f"A5_s{seed}", [py, "-m", "src.train", "--arm", "A5", "--compiled", "--seed", str(seed), "--device", "cuda"]),
            (f"A6_s{seed}", [py, "-m", "src.train", "--arm", "A6", "--checkpointing", "--seed", str(seed), "--device", "cuda"]),
        ]
    random.Random(20261107).shuffle(arms)
    write_json(
        RESULTS / "run_order.json",
        {"seed": 20261107, "machine": "gpu", "order": [a[0] for a in arms]},
    )
    for label, cmd in arms:
        run_step(log, label, cmd)
    run_step(log, "A7-loader", [py, "-m", "src.benchmark", "--device", "cuda", "--which", "loader"])
    # A8: choose the optimized profile config by the frozen rule.
    opt_cmd = [py, "-m", "src.profile", "--device", "cuda", "--workers", "4",
               "--pin", "--out", "a8_profile_optimized.json"]
    try:
        a7 = json.loads((RESULTS / "benchmarks" / "a7_loader.json").read_text())
        share = a7["step_decomposition"]["L0"]["data_wait_share"]
        if share < 0.5:
            opt_cmd = [py, "-m", "src.profile", "--device", "cuda",
                       "--channels-last", "--out", "a8_profile_optimized.json"]
    except Exception:
        pass
    run_step(log, "A8-profile-baseline", [py, "-m", "src.profile", "--device", "cuda",
                                         "--out", "a8_profile_baseline.json"])
    run_step(log, "A8-profile-optimized", opt_cmd)
    run_step(log, "A10-memory", [py, "-m", "src.benchmark", "--device", "cuda", "--which", "memory"])
    run_step(log, "compile-probes", [py, "-m", "src.benchmark", "--device", "cuda", "--which", "compile"])
    run_step(log, "A9-fsdp", [py, "-m", "src.fsdp_check", "--device", "cuda"])
    versions("gpu")


def cpu_suite(log) -> None:
    py = sys.executable
    run_step(log, "data-verify", [py, "-m", "src.data", "--verify"])
    run_step(log, "config-diff", [py, "-m", "src.config_diff"])
    run_step(log, "C1-single", [py, "-m", "src.ddp_cpu", "--world", "1"])
    run_step(log, "C2-ddp", [py, "-m", "src.ddp_cpu", "--world", "2"])
    versions("cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", choices=["gpu", "cpu"], required=True)
    args = ap.parse_args()
    log_path = REPO_ROOT / f"suite_{args.machine}.log"
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n##### suite {args.machine} start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
        if args.machine == "gpu":
            gpu_suite(log)
        else:
            cpu_suite(log)
    print("SUITE DONE", args.machine)


if __name__ == "__main__":
    main()
