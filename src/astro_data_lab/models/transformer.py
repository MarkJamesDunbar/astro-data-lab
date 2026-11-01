"""A Transformer encoder that reads a whole light curve as a sequence of patches.

Attention over every reading (~18,000) would be far too expensive, so the curve is
cut into patches of `patch` readings (64 readings ~ 2 hours at 2-minute cadence),
each patch becomes one token, and the encoder relates tokens across the whole
curve, e.g. repeated transits days apart. The same encoder is the natural starting
point for self-supervised pre-training on unlabelled curves.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PatchTransformer(nn.Module):
    def __init__(self, num_classes: int, patch: int = 64, width: int = 64, depth: int = 4,
                 heads: int = 4, dropout: float = 0.1, max_patches: int = 2048) -> None:
        super().__init__()
        self.patch, self.max_patches = patch, max_patches
        self.embed = nn.Linear(3 * patch, width)
        self.position = nn.Parameter(torch.zeros(1, max_patches, width))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(width, heads, dim_feedforward=4 * width, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, _, length = inputs.shape
        count = -(-length // self.patch)  # ceil: the last patch is zero-padded
        if count > self.max_patches:
            raise ValueError(f"curve has {count} patches; raise max_patches above {self.max_patches}")
        inputs = F.pad(inputs, (0, count * self.patch - length))
        valid = inputs[:, 3:4] > 0
        features = (inputs[:, :3] * valid).reshape(batch, 3, count, self.patch)
        tokens = self.embed(features.permute(0, 2, 1, 3).reshape(batch, count, 3 * self.patch))
        tokens = tokens + self.position[:, :count]
        real = valid.reshape(batch, count, self.patch).any(-1)  # patch holds at least one reading
        encoded = self.norm(self.encoder(tokens, src_key_padding_mask=~real))
        pooled = (encoded * real.unsqueeze(-1)).sum(1) / real.sum(1, keepdim=True)
        return self.head(pooled)
