"""A Transformer encoder that reads a whole light curve as a sequence of patches.

Attention over every reading (~18,000) would be far too expensive, so the curve is
cut into patches of `patch` readings (64 readings ~ 2 hours at 2-minute cadence),
each patch becomes one token, and the encoder relates tokens across the whole
curve, e.g. repeated transits days apart.

The pieces are separate methods so that `foundation.MaskedPatchPretrainer` can
reuse them: `patchify` (curve -> patches), `encode` (patches -> contextual
tokens, optionally with some patches hidden) and `pool` (curve -> one vector).
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

    def patchify(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(batch, 4, readings) -> features (batch, patches, 3, patch), padding zeroed,
        and valid (batch, patches, patch): True where a reading is real."""
        batch, _, length = inputs.shape
        count = -(-length // self.patch)  # ceil: the last patch is zero-padded
        if count > self.max_patches:
            raise ValueError(f"curve has {count} patches; raise max_patches above {self.max_patches}")
        inputs = F.pad(inputs, (0, count * self.patch - length))
        valid = inputs[:, 3] > 0
        features = (inputs[:, :3] * valid.unsqueeze(1)).reshape(batch, 3, count, self.patch)
        return features.permute(0, 2, 1, 3), valid.reshape(batch, count, self.patch)

    def encode(self, features: torch.Tensor, valid: torch.Tensor, hidden: torch.Tensor | None = None,
               hidden_token: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Contextual tokens (batch, patches, width) and which patches hold any real reading.

        `hidden` (batch, patches) replaces those patches' content with `hidden_token`, so the
        encoder must infer them from the rest of the curve (used for pre-training).
        """
        batch, count = features.shape[:2]
        tokens = self.embed(features.reshape(batch, count, -1))
        if hidden is not None:
            tokens = torch.where(hidden.unsqueeze(-1), hidden_token.to(tokens.dtype).expand_as(tokens), tokens)
        real = valid.any(-1)
        encoded = self.encoder(tokens + self.position[:, :count], src_key_padding_mask=~real)
        return self.norm(encoded), real

    def pool(self, inputs: torch.Tensor) -> torch.Tensor:
        """One vector per curve (batch, width): the mean of its real patches' tokens."""
        encoded, real = self.encode(*self.patchify(inputs))
        return (encoded * real.unsqueeze(-1)).sum(1) / real.sum(1, keepdim=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(inputs))
