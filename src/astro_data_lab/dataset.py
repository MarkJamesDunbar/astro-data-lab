"""Step 4: turn a table of labelled curves into PyTorch batches.

    LightCurveDataset(table, transform)  ->  dataset[i] = (transform(lightcurve), label)
    pad_collate(items)                   ->  (x, mask, y)

`table` is any DataFrame with a `data_uri` column and an integer `label` column,
e.g. built from `load_curves()` and `load_tois()`.

`transform` is your preprocessing: a function from a `LightCurve` to a tensor
(or numpy array) of shape (readings,) or (channels, readings).

Use with a normal DataLoader. Light curves download inside the DataLoader's worker
processes, so set `num_workers` to 8-16: downloads then run in parallel with
training. A curve that cannot be downloaded or read is skipped (and counted in
`dataset.skipped`), not fatal:

    loader = DataLoader(LightCurveDataset(table, transform), batch_size=32, shuffle=True,
                        num_workers=8, collate_fn=pad_collate)
    for x, mask, y in loader:
        ...
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from astro_data_lab.lightcurves import LightCurve, LightCurveUnavailable, read_lightcurve


class LightCurveDataset(Dataset):
    def __init__(self, table: pd.DataFrame, transform: Callable[[LightCurve], torch.Tensor | np.ndarray],
                 label_column: str = "label") -> None:
        missing = {"data_uri", label_column} - set(table.columns)
        if missing:
            raise ValueError(f"table needs columns {sorted(missing)}")
        self.data_uris = table["data_uri"].tolist()
        self.labels = table[label_column].astype(int).tolist()
        self.transform = transform
        self.skipped = 0  # per process: with num_workers > 0, each worker counts its own

    def __len__(self) -> int:
        return len(self.data_uris)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int] | None:
        try:
            lightcurve = read_lightcurve(self.data_uris[index])
        except LightCurveUnavailable:
            self.skipped += 1
            return None
        x = torch.as_tensor(self.transform(lightcurve), dtype=torch.float32)
        return x, self.labels[index]


def pad_collate(items: list[tuple[torch.Tensor, int] | None]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack variable-length examples into one batch.

    In:   list of (x, label); x has shape (readings,) or (channels, readings). None items are dropped.
    Out:  x     (batch, channels, longest)  zero-padded at the end
          mask  (batch, longest)            True where there is real data, False for padding
          y     (batch,)                    int64 labels
    """
    items = [item for item in items if item is not None]
    if not items:
        raise RuntimeError("every light curve in this batch was unavailable")
    tensors = [x if x.ndim == 2 else x.unsqueeze(0) for x, _ in items]
    longest = max(t.shape[-1] for t in tensors)
    x = torch.stack([F.pad(t, (0, longest - t.shape[-1])) for t in tensors])
    mask = torch.zeros(len(tensors), longest, dtype=torch.bool)
    for row, t in enumerate(tensors):
        mask[row, :t.shape[-1]] = True
    y = torch.tensor([label for _, label in items], dtype=torch.long)
    return x, mask, y
