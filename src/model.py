"""ResNet-18 construction for EuroSAT (10 classes)."""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchvision.models import ResNet18_Weights, resnet18

# The verdict pack's 11,689,512 is ResNet-18 with the 1000-class
# ImageNet head (verified against torchvision). With the 10-class
# EuroSAT head the count is 11,181,642. See PREREG Amendment A1.
EXPECTED_PARAMS = 11_181_642


class CheckpointedResNet18(nn.Module):
    """ResNet-18 whose four layer blocks run under gradient
    checkpointing (use_reentrant=False)."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.base
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)
        for layer in (m.layer1, m.layer2, m.layer3, m.layer4):
            x = checkpoint(layer, x, use_reentrant=False)
        x = m.avgpool(x)
        x = torch.flatten(x, 1)
        return m.fc(x)


def build_model(pretrained: bool, seed: int, checkpointing: bool = False) -> nn.Module:
    torch.manual_seed(seed)
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    # Re-initialise the head under the run seed in both cases, so the
    # only difference between arms is the backbone initialisation.
    torch.manual_seed(seed + 1)
    model.fc = nn.Linear(model.fc.in_features, 10)
    n = sum(p.numel() for p in model.parameters())
    assert n == EXPECTED_PARAMS, f"parameter count {n} != {EXPECTED_PARAMS}"
    if checkpointing:
        return CheckpointedResNet18(model)
    return model
