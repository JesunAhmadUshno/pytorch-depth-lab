import torch

from src.model import EXPECTED_PARAMS, build_model


def test_param_count():
    m = build_model(False, 0)
    assert sum(p.numel() for p in m.parameters()) == EXPECTED_PARAMS == 11_181_642


def test_forward_shape():
    m = build_model(False, 0)
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 10)


def test_checkpointed_forward_shape():
    m = build_model(False, 0, checkpointing=True)
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 10)
