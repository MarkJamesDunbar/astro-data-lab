"""Step 4: models. Any architecture works if it follows one contract:

    model = Factory(num_classes=N, **options)          # returns an nn.Module
    logits = model(inputs)   # inputs (batch, 4, readings) -> logits (batch, N)

Two examples are included: `LightCurveCNN` (cnn.py) and `PatchTransformer`
(transformer.py). Put new models in this folder, export them below, and point
an experiment at them: factory = "astro_data_lab.models:YourModel".
"""
from __future__ import annotations

import importlib
from typing import Any

from torch import nn

from astro_data_lab.models.cnn import LightCurveCNN, MaskedConvBlock
from astro_data_lab.models.transformer import PatchTransformer

__all__ = ["LightCurveCNN", "MaskedConvBlock", "PatchTransformer", "build_model", "check_model"]


def build_model(factory: str, num_classes: int, options: dict[str, Any]) -> nn.Module:
    module_name, _, attribute = factory.partition(":")
    model = getattr(importlib.import_module(module_name), attribute)(num_classes=num_classes, **options)
    if not isinstance(model, nn.Module):
        raise TypeError(f"{factory} did not return a torch.nn.Module")
    return model


def check_model(model: nn.Module, num_classes: int) -> None:
    """Fail fast with a clear message if `model` breaks the contract above.

    Checks the output shape, that outputs are finite, and that padding does not
    change a prediction (a curve alone must score the same as when padded).
    """
    import torch

    torch.manual_seed(0)
    short, long = torch.randn(4, 1500), torch.randn(4, 4000)
    short[3], long[3] = 1, 1
    padded = torch.stack([torch.nn.functional.pad(short, (0, 2500)), long])
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            output = model(padded)
            alone = model(short.unsqueeze(0))
    finally:
        model.train(was_training)
    if output.shape != (2, num_classes):
        raise ValueError(f"expected logits of shape (2, {num_classes}), got {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise ValueError("model produced NaN/inf for ordinary inputs")
    if not torch.allclose(alone[0], output[0], atol=1e-4):
        raise ValueError("padding changed the prediction: mask padded readings using channel 3")
