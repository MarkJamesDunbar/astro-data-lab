"""A small, readable masked 1-D CNN (the model contract is in models/__init__.py).

Convolutions are good at local shapes: a transit is a dip with a depth, a
width and sloped edges, and it means the same wherever it occurs in the curve.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MaskedConvBlock(nn.Module):
    """Convolution -> GELU -> max-pool, keeping padded positions out of the result."""

    def __init__(self, in_channels: int, out_channels: int, pool: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=9, padding=4)
        self.pool = pool

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.gelu(self.conv(x * valid)) * valid
        # -inf makes max-pool ignore padding; ceil_mode keeps the final partial window.
        x = F.max_pool1d(x.masked_fill(~valid, -torch.inf), self.pool, ceil_mode=True)
        valid = F.max_pool1d(valid.float(), self.pool, ceil_mode=True) > 0
        return x.masked_fill(~valid, 0.0), valid


class LightCurveCNN(nn.Module):
    """Four conv blocks shrink a ~18,000-reading curve 256x (each output step then
    "sees" several hours, the length of a typical transit), then masked mean+max
    pooling summarises the whole curve and a small head predicts the class."""

    def __init__(self, num_classes: int, width: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        channels = [3, width, width * 2, width * 2, width * 4]
        self.blocks = nn.ModuleList(
            MaskedConvBlock(channels[i], channels[i + 1], pool=4) for i in range(4)
        )
        self.head = nn.Sequential(
            nn.Linear(channels[-1] * 2, width * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(width * 2, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x, valid = inputs[:, :3], inputs[:, 3:4] > 0
        for block in self.blocks:
            x, valid = block(x, valid)
        count = valid.sum(-1).clamp_min(1)
        mean = x.sum(-1) / count
        maximum = x.masked_fill(~valid, -torch.inf).amax(-1)
        return self.head(torch.cat((mean, maximum), dim=1))
