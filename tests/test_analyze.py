import numpy as np

import src.analyze as az


def _run(run_id, seed, thr, acc=0.97):
    return {
        "run_id": run_id, "seed": seed, "arm": "AX",
        "epoch_rows": [{"throughput_img_s": 100.0}] + [
            {"throughput_img_s": thr} for _ in range(14)
        ],
        "throughput_img_s": thr, "test_acc": acc, "test_nll": 0.1,
        "peak_mem_mib": 1000.0, "tta_s": 10.0, "wall_s": 60.0,
    }


def test_ratio_ci_point():
    a = [_run("a1", 42, 200.0), _run("a2", 43, 200.0)]
    b = [_run("b1", 42, 100.0), _run("b2", 43, 100.0)]
    out = az.ratio_ci(a, b)
    assert abs(out["point"] - 2.0) < 1e-9
    assert out["ci95"][0] <= 2.0 <= out["ci95"][1]


def test_paired_acc_ci_identical(tmp_path, monkeypatch):
    monkeypatch.setattr(az, "RUNS", tmp_path)
    rng = np.random.default_rng(0)
    correct = rng.random(2700) < 0.97
    for rid in ("a1_s42", "b1_s42"):
        np.savez_compressed(tmp_path / f"{rid}_testcorrect.npz", correct=correct)
    a = [_run("a1_s42", 42, 100.0)]
    b = [_run("b1_s42", 42, 100.0)]
    out = az.paired_acc_ci(a, b)
    assert out["point"] == 0.0
    assert out["ci95"] == [0.0, 0.0]
