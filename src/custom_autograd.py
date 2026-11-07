"""A custom autograd function: fused log-softmax + NLL (cross
entropy) with an analytical backward pass.

Exists to evidence custom-autograd fluency at the framework level.
Verified against torch.autograd.gradcheck in double precision and
against F.cross_entropy in the test suite.
"""
from __future__ import annotations

import torch
from torch import nn


class FusedCrossEntropy(torch.autograd.Function):
    """forward: mean over the batch of -log_softmax(logits)[target].
    backward (analytical): d loss / d logits = (softmax - one_hot) / N.
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logp = torch.log_softmax(logits, dim=1)
        n = logits.shape[0]
        loss = -logp[torch.arange(n, device=logits.device), targets].mean()
        probs = logp.exp()
        ctx.save_for_backward(probs, targets)
        return loss

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        probs, targets = ctx.saved_tensors
        n = probs.shape[0]
        grad = probs.clone()
        grad[torch.arange(n, device=probs.device), targets] -= 1.0
        grad = grad / n * grad_out
        return grad, None


def fused_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return FusedCrossEntropy.apply(logits, targets)


class FusedCEModule(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return fused_cross_entropy(logits, targets)
