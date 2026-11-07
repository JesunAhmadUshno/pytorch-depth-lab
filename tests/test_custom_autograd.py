import torch
import torch.nn.functional as F
from torch.autograd import gradcheck

from src.custom_autograd import FusedCrossEntropy, fused_cross_entropy


def test_gradcheck_double():
    torch.manual_seed(0)
    logits = torch.randn(4, 6, dtype=torch.double, requires_grad=True)
    targets = torch.tensor([0, 2, 5, 1])
    assert gradcheck(FusedCrossEntropy.apply, (logits, targets), eps=1e-6, atol=1e-4)


def test_forward_matches_cross_entropy():
    torch.manual_seed(1)
    logits = torch.randn(8, 10)
    targets = torch.randint(0, 10, (8,))
    assert torch.allclose(
        fused_cross_entropy(logits, targets), F.cross_entropy(logits, targets), atol=1e-6
    )
