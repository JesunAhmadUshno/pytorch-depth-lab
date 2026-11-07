import pytest

from src.data import IMAGE_DIR

pytestmark = pytest.mark.skipif(
    not IMAGE_DIR.exists(), reason="EuroSAT not downloaded on this machine"
)


def test_hand_loop_smoke():
    from src.train import TrainConfig, run_training

    cfg = TrainConfig(
        arm="SMOKE", epochs=1, seed=42, device="cpu",
        max_steps=2, limit_train=256, run_id="smoke_test",
    )
    res = run_training(cfg)
    assert 0.0 <= res["test_acc"] <= 1.0
    assert res["epoch_rows"][0]["train_loss"] > 0
