"""Settings: where things live (from .env) and what to train (from a TOML file)."""
from __future__ import annotations

import dataclasses
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GIB = 1024**3

TARGETS: dict[str, tuple[str, ...]] = {
    # Class names in model-output order. See data/labels.py for how curves are labelled.
    "transit": ("no_signal", "signal"),
}
PRETRAIN = "pretrain"  # self-supervised target: no labels, see models/foundation.py


@dataclass(frozen=True)
class Settings:
    """Machine-specific locations and limits, read from environment variables."""

    data_dir: Path
    cache_limit_gib: float
    free_space_floor_gib: float
    mlflow_uri: str
    mlflow_experiment: str

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            data_dir=Path(os.getenv("ASTRO_DATA_DIR", "/data/astro-data-lab")),
            cache_limit_gib=float(os.getenv("CURVE_CACHE_GIB", "50")),
            free_space_floor_gib=float(os.getenv("FREE_SPACE_FLOOR_GIB", "330")),
            mlflow_uri=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
            mlflow_experiment=os.getenv("MLFLOW_EXPERIMENT_NAME", "tess-classifiers"),
        )

    @property
    def curve_cache(self) -> Path:
        return self.data_dir / "curves"

    @property
    def checkpoints(self) -> Path:
        return self.data_dir / "checkpoints"


@dataclass(frozen=True)
class Experiment:
    """Everything that defines a model run. Change any of it -> use a new name."""

    name: str
    target: str = "transit"
    # [data]
    curves_per_round: int = 1024
    injected_fraction: float = 0.5   # share of "signal" examples that are no_signal curves + a synthetic transit
    rounds: int = 30
    validation_curves_per_class: int = 300
    download_workers: int = 8
    max_bad_fraction: float = 0.2
    # [training]
    passes_per_round: int = 1        # gradient passes over each downloaded round
    minibatch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    # [model]
    factory: str = "astro_data_lab.models:LightCurveCNN"
    options: dict[str, Any] = field(default_factory=dict)
    init_from: str = ""              # start from another run's best weights (run name or checkpoint path)

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.name):
            raise ValueError("name may contain only letters, digits, dot, underscore and hyphen")
        if self.target not in TARGETS and self.target != PRETRAIN:
            raise ValueError(f"target must be one of {', '.join([*TARGETS, PRETRAIN])}")
        if self.curves_per_round < len(self.classes) or self.minibatch_size < 1 or self.passes_per_round < 1:
            raise ValueError("curves_per_round, minibatch_size and passes_per_round are too small")
        if self.rounds < 0 or self.validation_curves_per_class < 1 or self.download_workers < 1:
            raise ValueError("rounds must be >= 0; validation and worker counts must be positive")
        if not 0 <= self.max_bad_fraction < 1 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("max_bad_fraction must be in [0, 1); learning_rate positive")
        if not 0 <= self.injected_fraction < 1:
            raise ValueError("injected_fraction must be in [0, 1): some real signals are always needed")
        if ":" not in self.factory:
            raise ValueError("factory must look like 'package.module:ClassOrFunction'")

    @property
    def classes(self) -> tuple[str, ...]:
        """Class names in model-output order; empty for self-supervised pre-training."""
        return TARGETS.get(self.target, ())

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def load(cls, path: Path, **overrides: Any) -> Experiment:
        """Read [run], [data], [training] and [model] sections; reject typos."""
        with path.open("rb") as stream:
            document = tomllib.load(stream)
        sections = {"run", "data", "training", "model"}
        if unknown := set(document) - sections:
            raise ValueError(f"Unknown config sections: {', '.join(sorted(unknown))}")
        values: dict[str, Any] = {}
        for section in sections:
            table = document.get(section, {})
            if not isinstance(table, dict):
                raise ValueError(f"[{section}] must be a table")
            values.update(table)
        defaults = {f.name: f for f in dataclasses.fields(cls)}
        if unknown := set(values) - set(defaults):
            raise ValueError(f"Unknown config settings: {', '.join(sorted(unknown))}")
        for key, value in values.items():
            expected = type(defaults[key].default) if defaults[key].default is not dataclasses.MISSING else str
            if key == "options":
                expected = dict
            if expected is float and type(value) is int:
                value = float(value)
                values[key] = value
            if type(value) is not expected:
                raise ValueError(f"{key} must be {expected.__name__}, got {type(value).__name__}")
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)
