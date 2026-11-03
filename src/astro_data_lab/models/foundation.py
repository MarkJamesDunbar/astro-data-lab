"""A light-curve foundation model: pre-train a PatchTransformer on unlabelled curves.

Self-supervised objective (masked patch modelling, as in BERT/MAE/Astromer):
hide `mask_fraction` of each curve's ~2-hour patches and predict their readings
(raw and detrended flux) from the rest of the curve. To do this well the encoder
must learn what normal stellar behaviour, noise, gaps and recurring dips look like,
without any labels, so it can learn from all ~437,000 training-split curves
instead of the ~4,400 with a known transit.

`MaskedPatchPretrainer` IS a `PatchTransformer` plus two extra parts used only for
pre-training (`hidden_token`, `decoder`). Its encoder weights therefore have
exactly the names a `PatchTransformer` expects: fine-tune by building a
PatchTransformer with the same options and setting `init_from` to the
pre-training run's name in the experiment file.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from astro_data_lab.models.transformer import PatchTransformer


class MaskedPatchPretrainer(PatchTransformer):
    def __init__(self, num_classes: int = 1, mask_fraction: float = 0.4, **options) -> None:
        super().__init__(num_classes=1, **options)  # the classification head is not trained here
        if not 0 < mask_fraction < 1:
            raise ValueError("mask_fraction must be between 0 and 1")
        width = self.embed.out_features
        self.mask_fraction = mask_fraction
        self.hidden_token = nn.Parameter(torch.zeros(width))
        nn.init.normal_(self.hidden_token, std=0.02)
        self.decoder = nn.Linear(width, 2 * self.patch)

    def reconstruction_loss(self, inputs: torch.Tensor, generator: torch.Generator | None = None
                            ) -> tuple[torch.Tensor, torch.Tensor]:
        """(model loss, baseline loss) on the hidden readings, both Huber.

        The baseline predicts each hidden patch as the average of the nearest visible
        patches on either side: simple interpolation. Beating it means the model has
        learned structure, not just smoothness.
        """
        features, valid = self.patchify(inputs)
        batch, count = valid.shape[:2]
        real = valid.any(-1)
        draw = torch.rand(real.shape, device=inputs.device, generator=generator)
        hidden = (draw < self.mask_fraction) & real
        encoded, _ = self.encode(features, valid, hidden, self.hidden_token)
        predicted = self.decoder(encoded).reshape(batch, count, 2, self.patch)
        target = features[:, :, :2]                      # channels 0-1: raw %, detrended in noise units
        scored = (hidden.unsqueeze(-1) & valid).unsqueeze(2).expand_as(target)
        if not scored.any():
            zero = predicted.sum() * 0
            return zero, zero.detach()
        loss = F.smooth_l1_loss(predicted.float()[scored], target[scored])
        baseline = F.smooth_l1_loss(_neighbour_estimate(features, valid, hidden)[scored], target[scored])
        return loss, baseline.detach()


def _neighbour_estimate(features: torch.Tensor, valid: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    """Each patch predicted as the mean of the nearest visible patch before and after it."""
    batch, count, _, patch = features.shape
    counts = valid.sum(-1, keepdim=True).clamp_min(1)                              # (B, P, 1)
    means = (features[:, :, :2] * valid.unsqueeze(2)).sum(-1) / counts             # (B, P, 2)
    visible = valid.any(-1) & ~hidden
    index = torch.arange(count, device=features.device).expand(batch, count)
    before = torch.where(visible, index, -1).cummax(1).values
    after = torch.where(visible, index, count).flip(1).cummin(1).values.flip(1)

    def take(where: torch.Tensor) -> torch.Tensor:
        return means.gather(1, where.clamp(0, count - 1).unsqueeze(-1).expand(-1, -1, 2))

    use_before = (before >= 0).float().unsqueeze(-1)
    use_after = (after < count).float().unsqueeze(-1)
    estimate = (take(before) * use_before + take(after) * use_after) / (use_before + use_after).clamp_min(1)
    return estimate.unsqueeze(-1).expand(-1, -1, -1, patch)
