"""Step 7: classify one complete new light curve with a trained checkpoint."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from astro_data_lab.config import Experiment
from astro_data_lab.data.lightcurves import INPUT_VERSION, clean_fits, model_input
from astro_data_lab.models import build_model


def load_trained(checkpoint: Path, device: torch.device) -> tuple[Experiment, nn.Module, int]:
    """(experiment, model in eval mode on `device`, round) from a saved checkpoint."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("input_version", 1) != INPUT_VERSION:
        raise ValueError(f"{checkpoint} was trained on an older version of the model inputs "
                         f"(v{saved.get('input_version', 1)}, current v{INPUT_VERSION}); retrain it.")
    experiment = Experiment(**saved["experiment"])
    model = build_model(experiment.factory, len(experiment.classes), experiment.options)
    model.load_state_dict(saved["model"])
    return experiment, model.to(device).eval(), int(saved["round"])


def predict(checkpoint: Path, fits_path: Path, device: str | None = None) -> dict:
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    experiment, model, round_number = load_trained(checkpoint, target)
    curve = clean_fits(fits_path)
    inputs = model_input(curve.time, curve.flux).unsqueeze(0).to(target)
    with torch.inference_mode():
        probabilities = model(inputs).softmax(-1)[0].cpu().tolist()
    best = max(range(len(probabilities)), key=probabilities.__getitem__)
    return {
        "tic_id": curve.tic_id,
        "sector": curve.sector,
        "readings_used": inputs.shape[-1],
        "prediction": experiment.classes[best],
        "scores": dict(zip(experiment.classes, probabilities)),
        "model": f"{experiment.name}, round {round_number}",
    }
