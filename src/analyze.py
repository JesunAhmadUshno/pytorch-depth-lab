"""Analysis under the frozen rules of PREREG.md.

Reads results/runs/*.json (+ test-correctness arrays),
results/benchmarks/*.json, and results/cpu/*.json; computes the
frozen statistics (paired bootstrap for accuracy differences,
epoch bootstrap for throughput ratios); emits per-hypothesis
verdicts into results/summary.json and the README table into
results/readme_table.md. Every number in RESULTS.md comes from
these files, recounted here from the raw run artefacts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .common import RESULTS, RUNS, write_json

BOOT_SEED = 7
N_BOOT = 10_000


def _load_runs() -> dict[str, dict]:
    runs = {}
    for p in sorted(RUNS.glob("*.json")):
        d = json.loads(p.read_text())
        runs[d["run_id"]] = d
    return runs


def _arm_runs(runs: dict, arm: str) -> list[dict]:
    return [r for r in runs.values() if r["arm"] == arm]


def _epoch_throughputs(run: dict) -> list[float]:
    return [row["throughput_img_s"] for row in run["epoch_rows"][1:]]


def ratio_ci(a_runs: list[dict], b_runs: list[dict]) -> dict:
    a = np.array([v for r in a_runs for v in _epoch_throughputs(r)])
    b = np.array([v for r in b_runs for v in _epoch_throughputs(r)])
    point = float(a.mean() / b.mean())
    rng = np.random.default_rng(BOOT_SEED)
    ratios = []
    for _ in range(N_BOOT):
        ra = rng.choice(a, size=len(a), replace=True).mean()
        rb = rng.choice(b, size=len(b), replace=True).mean()
        ratios.append(ra / rb)
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return {
        "point": point, "ci95": [float(lo), float(hi)],
        "a_mean": float(a.mean()), "b_mean": float(b.mean()),
        "a_epochs": int(len(a)), "b_epochs": int(len(b)),
    }


def paired_acc_ci(a_runs: list[dict], b_runs: list[dict]) -> dict:
    """Paired bootstrap of the test accuracy difference, matched by
    seed, averaged over seed pairs."""
    by_seed_a = {r["seed"]: r for r in a_runs}
    by_seed_b = {r["seed"]: r for r in b_runs}
    seeds = sorted(set(by_seed_a) & set(by_seed_b))
    rng = np.random.default_rng(BOOT_SEED)
    boot = []
    point_diffs = []
    for s in seeds:
        ca = np.load(RUNS / f"{by_seed_a[s]['run_id']}_testcorrect.npz")["correct"].astype(float)
        cb = np.load(RUNS / f"{by_seed_b[s]['run_id']}_testcorrect.npz")["correct"].astype(float)
        d = ca - cb
        point_diffs.append(float(d.mean()))
        idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
        boot.append(d[idx].mean(axis=1))
    samples = np.mean(np.stack(boot), axis=0) if boot else np.array([0.0])
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {
        "point": float(np.mean(point_diffs)) if point_diffs else 0.0,
        "ci95": [float(lo), float(hi)], "seeds": seeds,
    }


def _within_pp(ci: dict, pp: float = 0.01) -> bool:
    return ci["ci95"][0] >= -pp and ci["ci95"][1] <= pp


def analyze() -> dict:
    runs = _load_runs()
    bench_dir = RESULTS / "benchmarks"
    a7 = json.loads((bench_dir / "a7_loader.json").read_text())
    a10 = json.loads((bench_dir / "a10_memory.json").read_text())
    compile_probe = json.loads((bench_dir / "compile_probe.json").read_text())
    fsdp = json.loads((bench_dir / "fsdp_check.json").read_text())
    c1 = json.loads((RESULTS / "cpu" / "c1_single.json").read_text())
    c2 = json.loads((RESULTS / "cpu" / "c2_ddp.json").read_text())

    A1, A2, A3 = _arm_runs(runs, "A1"), _arm_runs(runs, "A2"), _arm_runs(runs, "A3")
    A4, A5, A6 = _arm_runs(runs, "A4"), _arm_runs(runs, "A5"), _arm_runs(runs, "A6")

    out: dict = {"arms": {}, "hypotheses": {}}
    for arm, rs in (("A1", A1), ("A2", A2), ("A3", A3), ("A4", A4), ("A5", A5), ("A6", A6)):
        out["arms"][arm] = [
            {
                "run_id": r["run_id"], "seed": r["seed"],
                "throughput_img_s": r["throughput_img_s"],
                "test_acc": r["test_acc"], "test_nll": r["test_nll"],
                "peak_mem_mib": r["peak_mem_mib"], "tta_s": r["tta_s"],
                "wall_s": r["wall_s"],
            }
            for r in rs
        ]

    # H1: hand vs Lightning
    r_h1 = ratio_ci(A1, A3)
    d_h1 = paired_acc_ci(A1, A3)
    out["hypotheses"]["H1"] = {
        "ratio_a1_over_a3": r_h1, "acc_diff_a1_minus_a3": d_h1,
        "verdict": "SUPPORTED" if (r_h1["point"] >= 0.90 and _within_pp(d_h1)) else "NOT SUPPORTED",
    }
    # H2: AMP
    r_h2 = ratio_ci(A4, A1)
    d_h2 = paired_acc_ci(A4, A1)
    mem_a1 = float(np.mean([r["peak_mem_mib"] for r in A1]))
    mem_a4 = float(np.mean([r["peak_mem_mib"] for r in A4]))
    out["hypotheses"]["H2"] = {
        "ratio_a4_over_a1": r_h2, "acc_diff_a4_minus_a1": d_h2,
        "peak_mem_a1_mib": mem_a1, "peak_mem_a4_mib": mem_a4,
        "peak_mem_saving_frac": 1 - mem_a4 / mem_a1,
        "verdict": "SUPPORTED" if (r_h2["point"] >= 1.30 and _within_pp(d_h2)) else "NOT SUPPORTED",
    }
    # H3: compile
    r_h3 = ratio_ci(A5, A1)
    d_h3 = paired_acc_ci(A5, A1)
    out["hypotheses"]["H3"] = {
        "ratio_a5_over_a1": r_h3, "acc_diff_a5_minus_a1": d_h3,
        "compile_step_s": [r.get("compile_step_s") for r in A5],
        "fullgraph_probe": compile_probe.get("fullgraph"),
        "verdict": "SUPPORTED" if (r_h3["point"] >= 1.10 and _within_pp(d_h3)) else "NOT SUPPORTED",
    }
    # H4 + H5: data loading
    decomp = a7["step_decomposition"]
    l0 = decomp["L0"]
    data_share_l0 = l0["data_wait_share"]
    compute_share_l0 = 1 - data_share_l0
    loaders = {r["config"]: r for r in a7["loader_throughput"]}
    best_name = max(loaders, key=lambda k: loaders[k]["loader_img_s"])
    best = loaders[best_name]
    loader_ratio = best["loader_img_s"] / loaders["L0"]["loader_img_s"]
    if data_share_l0 >= compute_share_l0:
        bottleneck = "data loading"
        opt_share = decomp["L3"]["data_wait_share"]
        opt_name = "loader configuration L3 (workers 4, pin memory, persistent workers)"
        h4_ok = opt_share < data_share_l0
        h4_before, h4_after = data_share_l0, opt_share
    else:
        bottleneck = "compute"
        opt_name = "channels_last memory format (L0CL)"
        h4_before = l0["compute_s"]
        h4_after = decomp["L0CL"]["compute_s"]
        h4_ok = h4_after < h4_before
    out["hypotheses"]["H4"] = {
        "bottleneck": bottleneck, "optimization": opt_name,
        "share_or_time_before": h4_before, "share_or_time_after": h4_after,
        "verdict": "SUPPORTED" if h4_ok else "NOT SUPPORTED",
    }
    not_bottleneck = data_share_l0 < 0.02
    out["hypotheses"]["H5"] = {
        "loader_throughput": loaders, "best_config": best_name,
        "best_over_l0_ratio": loader_ratio,
        "data_wait_share_l0": data_share_l0,
        "data_wait_share_best": decomp[best_name]["data_wait_share"] if best_name in decomp else None,
        "verdict": (
            "NOT A BOTTLENECK" if not_bottleneck
            else "SUPPORTED" if loader_ratio > 1.00 else "NOT SUPPORTED"
        ),
    }
    # H6: checkpointing
    cells = {(c["checkpointing"], c["batch"]): c for c in a10["cells"]}
    peak_no = cells[(False, 128)]["peak_mem_mib"]
    peak_ck = cells[(True, 128)]["peak_mem_mib"]
    reduction = 1 - peak_ck / peak_no
    r_h6 = ratio_ci(A6, A1)
    d_h6 = paired_acc_ci(A6, A1)
    out["hypotheses"]["H6"] = {
        "peak_mem_no_ckpt_mib": peak_no, "peak_mem_ckpt_mib": peak_ck,
        "reduction_frac": reduction,
        "throughput_ratio_a6_over_a1": r_h6,
        "ratio_band_prior": [0.65, 0.90],
        "acc_diff_a6_minus_a1": d_h6,
        "feasible_512": a10["feasible_512"],
        "verdict": "SUPPORTED" if (reduction >= 0.30 and _within_pp(d_h6)) else "NOT SUPPORTED",
    }
    # H7: CPU DDP
    losses1 = [r["train_loss"] for r in c1["epoch_rows"]]
    losses2 = [r["train_loss"] for r in c2["epoch_rows"]]
    loss_dev = float(np.mean([abs(a - b) for a, b in zip(losses1, losses2)]))
    acc_dev = abs(c1["epoch_rows"][-1]["val_acc"] - c2["epoch_rows"][-1]["val_acc"])
    et1 = float(np.mean([r["epoch_wall_s"] for r in c1["epoch_rows"]]))
    et2 = float(np.mean([r["epoch_wall_s"] for r in c2["epoch_rows"]]))
    out["hypotheses"]["H7"] = {
        "c1_losses": losses1, "c2_losses": losses2,
        "mean_abs_loss_diff": loss_dev, "final_val_acc_diff_pp": acc_dev * 100,
        "epoch_time_ratio_c2_over_c1": et2 / et1,
        "verdict": "SUPPORTED" if (loss_dev <= 0.05 and acc_dev <= 0.01) else "NOT SUPPORTED",
    }
    out["fsdp_check"] = {"status": fsdp.get("status"), "n_dtensor": fsdp.get("n_dtensor"),
                         "error": fsdp.get("error")}
    out["a2_descriptive"] = {
        "vs_a1_acc_diff": paired_acc_ci(A2, A1),
        "tta_a2_s": [r["tta_s"] for r in A2], "tta_a1_s": [r["tta_s"] for r in A1],
    }
    write_json(RESULTS / "summary.json", out)

    lines = [
        "| Arm | n (seeds) | Throughput img/s (mean) | Test acc | Test NLL | Peak mem MiB | TTA s |",
        "|---|---|---|---|---|---|---|",
    ]
    names = {
        "A1": "Hand FP32 scratch", "A2": "Hand FP32 fine-tuned",
        "A3": "Lightning FP32 scratch", "A4": "Hand AMP FP16 scratch",
        "A5": "Hand FP32 + torch.compile", "A6": "Hand FP32 + grad ckpt",
    }
    for arm in ("A1", "A2", "A3", "A4", "A5", "A6"):
        rs = out["arms"][arm]
        if not rs:
            continue
        tta = ", ".join(
            "not reached" if r["tta_s"] is None else f"{r['tta_s']:.0f}" for r in rs
        )
        lines.append(
            f"| {arm} {names[arm]} | {len(rs)} | "
            f"{np.mean([r['throughput_img_s'] for r in rs]):.1f} | "
            f"{np.mean([r['test_acc'] for r in rs]):.4f} | "
            f"{np.mean([r['test_nll'] for r in rs]):.4f} | "
            f"{np.mean([r['peak_mem_mib'] for r in rs]):.0f} | "
            f"{tta} |"
        )
    (RESULTS / "readme_table.md").write_text("\n".join(lines) + "\n")
    return out


if __name__ == "__main__":
    s = analyze()
    for h, v in s["hypotheses"].items():
        print(h, v["verdict"])
