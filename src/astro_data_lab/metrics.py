"""Step 6: score predictions honestly.

    report(y_true, y_pred, class_names)  ->  Report  (print it for a summary)

Balanced accuracy is the mean of each class's recall. Guessing, or always
predicting one class, scores 1 / number_of_classes whatever the class mix, so it
is the number to compare models on. The 95% interval comes from resampling the
examples (bootstrap); two models whose intervals overlap heavily are not
clearly different.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Report:
    class_names: tuple[str, ...]
    balanced_accuracy: float
    ci95: tuple[float, float]
    accuracy: float
    recall: tuple[float, ...]     # per class: share of that class predicted correctly
    precision: tuple[float, ...]  # per class: share of predictions of that class that were right
    support: tuple[int, ...]      # per class: how many examples
    confusion: np.ndarray         # rows = true class, columns = predicted class

    @property
    def chance(self) -> float:
        return 1 / len(self.class_names)

    def __str__(self) -> str:
        width = max(16, *(len(n) for n in self.class_names))
        verdict = "better than chance" if self.ci95[0] > self.chance else "NOT clearly better than chance"
        lines = [f"balanced accuracy {self.balanced_accuracy:.3f} (95% CI {self.ci95[0]:.3f}-{self.ci95[1]:.3f}), "
                 f"chance {self.chance:.3f}: {verdict}",
                 f"accuracy {self.accuracy:.3f}", "",
                 f"{'true / predicted':>{width}}  " + "  ".join(f"{n[:10]:>10}" for n in self.class_names)
                 + "    recall  precision  examples"]
        for i, name in enumerate(self.class_names):
            row = "  ".join(f"{int(v):>10}" for v in self.confusion[i])
            lines.append(f"{name:>{width}}  {row}    {self.recall[i]:6.3f}  {self.precision[i]:9.3f}  {self.support[i]:8d}")
        return "\n".join(lines)


def _balanced_accuracy(confusion: np.ndarray) -> float:
    support = confusion.sum(1)
    present = support > 0
    return float(np.mean(np.diag(confusion)[present] / support[present]))


def report(y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str] | None = None,
           resamples: int = 1000, seed: int = 0) -> Report:
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    if y_true.shape != y_pred.shape or y_true.ndim != 1 or len(y_true) == 0:
        raise ValueError("y_true and y_pred must be non-empty 1-D sequences of the same length")
    count = int(max(y_true.max(), y_pred.max()) + 1) if class_names is None else len(class_names)
    names = tuple(class_names) if class_names is not None else tuple(str(i) for i in range(count))
    confusion = np.zeros((count, count), dtype=np.int64)
    np.add.at(confusion, (y_true, y_pred), 1)
    support, predicted, correct = confusion.sum(1), confusion.sum(0), np.diag(confusion)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(resamples):
        pick = rng.integers(0, len(y_true), len(y_true))
        resampled = np.zeros_like(confusion)
        np.add.at(resampled, (y_true[pick], y_pred[pick]), 1)
        samples.append(_balanced_accuracy(resampled))
    return Report(
        class_names=names,
        balanced_accuracy=_balanced_accuracy(confusion),
        ci95=(float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))),
        accuracy=float(correct.sum() / len(y_true)),
        recall=tuple(float(c / s) if s else 0.0 for c, s in zip(correct, support)),
        precision=tuple(float(c / p) if p else 0.0 for c, p in zip(correct, predicted)),
        support=tuple(int(s) for s in support),
        confusion=confusion,
    )
