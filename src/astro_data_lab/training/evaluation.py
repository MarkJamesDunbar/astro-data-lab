"""Step 6: measure whether the model is genuinely useful on stars it never saw.

Validation and test sets contain an equal number of curves per class (or as many
as exist), so always guessing one class scores a balanced accuracy of exactly
1 / number_of_classes. A model is "better than chance" only if the lower end of
the 95% bootstrap interval for balanced accuracy is above that line.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from astro_data_lab.config import Experiment, Settings
from astro_data_lab.data import labels
from astro_data_lab.data.lightcurves import Prepared, check_quality, collate, prepare_many
from astro_data_lab.database import connect, ensure_schema
from astro_data_lab.predict import load_trained


@dataclass(frozen=True)
class Report:
    classes: tuple[str, ...]
    loss: float
    accuracy: float
    balanced_accuracy: float
    balanced_accuracy_95: tuple[float, float]
    macro_f1: float
    recall: tuple[float, ...]
    precision: tuple[float, ...]
    support: tuple[int, ...]
    confusion: np.ndarray  # rows = true class, columns = predicted class

    @property
    def chance(self) -> float:
        return 1 / len(self.classes)

    @property
    def better_than_chance(self) -> bool:
        return self.balanced_accuracy_95[0] > self.chance

    def metrics(self, prefix: str) -> dict[str, float]:
        values = {f"{prefix}_loss": self.loss, f"{prefix}_accuracy": self.accuracy,
                  f"{prefix}_balanced_accuracy": self.balanced_accuracy,
                  f"{prefix}_balanced_accuracy_low": self.balanced_accuracy_95[0],
                  f"{prefix}_macro_f1": self.macro_f1}
        values.update({f"{prefix}_recall_{name}": r for name, r in zip(self.classes, self.recall)})
        return values

    def text(self) -> str:
        header = "true / predicted"
        width = max(len(header), *(len(name) for name in self.classes))
        lines = [
            f"Balanced accuracy {self.balanced_accuracy:.3f} "
            f"(95% CI {self.balanced_accuracy_95[0]:.3f}-{self.balanced_accuracy_95[1]:.3f}); "
            f"chance = {self.chance:.3f} -> "
            + ("BETTER THAN CHANCE" if self.better_than_chance else "NOT distinguishable from chance"),
            f"Macro F1 {self.macro_f1:.3f} | accuracy {self.accuracy:.3f} | loss {self.loss:.3f}",
            "",
            f"{header:>{width}}  " + "  ".join(f"{n[:10]:>10}" for n in self.classes)
            + "    recall  precision  curves",
        ]
        for i, name in enumerate(self.classes):
            row = "  ".join(f"{int(v):>10}" for v in self.confusion[i])
            lines.append(f"{name:>{width}}  {row}    {self.recall[i]:6.3f}  {self.precision[i]:9.3f}"
                         f"  {self.support[i]:6d}")
        return "\n".join(lines)


def _balanced_accuracy(confusion: np.ndarray) -> float:
    support = confusion.sum(1)
    present = support > 0
    return float(np.mean(np.diag(confusion)[present] / support[present]))


def score(targets: np.ndarray, probabilities: np.ndarray, classes: tuple[str, ...],
          bootstrap: int = 1000, seed: int = 0) -> Report:
    count = len(classes)
    predictions = probabilities.argmax(1)
    confusion = np.zeros((count, count), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    support = confusion.sum(1)
    predicted = confusion.sum(0)
    correct = np.diag(confusion)
    recall = np.divide(correct, support, out=np.zeros(count), where=support > 0)
    precision = np.divide(correct, predicted, out=np.zeros(count), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(count), where=(precision + recall) > 0)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(bootstrap):
        pick = rng.integers(0, len(targets), len(targets))
        resampled = np.zeros_like(confusion)
        np.add.at(resampled, (targets[pick], predictions[pick]), 1)
        samples.append(_balanced_accuracy(resampled))
    chosen = np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1)
    return Report(
        classes=classes,
        loss=float(-np.log(chosen).mean()),
        accuracy=float(correct.sum() / len(targets)),
        balanced_accuracy=_balanced_accuracy(confusion),
        balanced_accuracy_95=(float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))),
        macro_f1=float(f1[support > 0].mean()),
        recall=tuple(map(float, recall)),
        precision=tuple(map(float, precision)),
        support=tuple(map(int, support)),
        confusion=confusion,
    )


@torch.inference_mode()
def predict_probabilities(model: nn.Module, curves: list[torch.Tensor], device: torch.device,
                          minibatch_size: int = 64) -> np.ndarray:
    model.eval()
    # Sorting by length keeps padding (wasted work) small; results are re-ordered back.
    order = sorted(range(len(curves)), key=lambda i: curves[i].shape[-1])
    chunks = []
    for start in range(0, len(order), minibatch_size):
        batch = collate([curves[i] for i in order[start:start + minibatch_size]]).to(device)
        chunks.append(F.softmax(model(batch).float(), dim=1).cpu().numpy())
    output = np.concatenate(chunks)
    result = np.empty_like(output)
    result[order] = output
    return result


def held_out_set(experiment: Experiment, settings: Settings, split: str) -> Prepared:
    """A fixed, class-balanced sample of real curves from stars outside the training split.

    Signal candidates are over-sampled because some turn out not to contain a
    transit; the first `validation_curves_per_class` that do are kept, so the
    set is the same every time. No injected transits are used here.
    """
    per_class = experiment.validation_curves_per_class
    with connect() as conn:
        candidates = labels.fixed_sample(conn, experiment.target, split, per_class * 3)
    needs_check = [c for c in candidates if c.ephemerides]
    curves = needs_check + labels.balanced_subset([c for c in candidates if not c.ephemerides], per_class)
    print(f"Preparing {split} curves from stars never used for training...", flush=True)
    prepared = prepare_many(curves, settings, experiment.download_workers)
    check_quality(prepared, len(curves), experiment.max_bad_fraction, f"{split} set")
    keep = labels.balanced_subset(prepared.curves, per_class)
    chosen = {id(c) for c in keep}
    tensors = [t for c, t in zip(prepared.curves, prepared.tensors) if id(c) in chosen]
    return Prepared(keep, tensors, prepared.excluded, prepared.failed, prepared.no_transit)


def evaluate_curves(model: nn.Module, data: Prepared, classes: tuple[str, ...],
                    device: torch.device) -> Report:
    probabilities = predict_probabilities(model, data.tensors, device)
    return score(np.array([c.target for c in data.curves]), probabilities, classes)


def evaluate(name: str, settings: Settings, split: str = "test",
             checkpoint: Path | None = None) -> Report:
    """Score a run's best checkpoint on held-out stars and log the result to MLflow."""
    ensure_schema()
    with connect() as conn:
        row = conn.execute("SELECT best_checkpoint_path, mlflow_run_id FROM training_runs "
                           "WHERE name = %s", (name,)).fetchone()
    if row is None or not (checkpoint or row[0]):
        raise SystemExit(f"No trained checkpoint for run '{name}'")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment, model, round_number = load_trained(checkpoint or Path(row[0]), device)
    report = evaluate_curves(model, held_out_set(experiment, settings, split), experiment.classes, device)
    print(f"\n{name}, round {round_number} checkpoint, {split} stars:\n{report.text()}")
    if row[1]:
        mlflow.set_tracking_uri(settings.mlflow_uri)
        with mlflow.start_run(run_id=row[1]):
            mlflow.log_metrics(report.metrics(split))
    return report
