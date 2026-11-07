import pytest
from PIL import Image

from src.data import build_split


@pytest.fixture()
def fake_eurosat(tmp_path):
    root = tmp_path / "EuroSAT_RGB"
    for c in range(10):
        d = root / f"Class{c}"
        d.mkdir(parents=True)
        for i in range(40):
            Image.new("RGB", (8, 8), (c * 20 % 255, i % 255, 0)).save(d / f"img_{i:03d}.jpg")
    return root


def test_split_sizes_and_stratification(fake_eurosat):
    s = build_split(fake_eurosat)
    assert len(s["split"]["train"]) == 320
    assert len(s["split"]["val"]) == 40
    assert len(s["split"]["test"]) == 40
    for cls, parts in s["per_class"].items():
        assert parts == {"train": 32, "val": 4, "test": 4}, cls


def test_split_disjoint_and_deterministic(fake_eurosat):
    a = build_split(fake_eurosat)["split"]
    b = build_split(fake_eurosat)["split"]
    assert a == b
    tr, va, te = set(a["train"]), set(a["val"]), set(a["test"])
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert len(tr | va | te) == 400
