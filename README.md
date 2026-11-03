# Astro Data Lab

Plain Python functions for training your own models on TESS light curves. The lab
handles the plumbing: what exists, downloading, caching and bounded disk use. You
write the parts that matter, in ordinary pandas and PyTorch: the labels, the
preprocessing, the model and the training loop.

Start with [`notebooks/getting_started.ipynb`](notebooks/getting_started.ipynb).
It runs every step on real data in about a minute.

## The steps

```python
from astro_data_lab import (sync, load_curves, load_tois, load_confirmed_planets,
                            read_lightcurve, LightCurveDataset, pad_collate, report)
```

| step | function | input | output |
|---|---|---|---|
| 1. sync | `sync()` | — | catalogues and light-curve manifest updated in PostgreSQL |
| 2. tables | `load_curves()` | — | DataFrame, one row per light curve: `data_uri, tic_id, sector, size_bytes, split` |
| | `load_tois()` | — | DataFrame, one row per TESS Object of Interest: `toi, tic_id, disposition, epoch_btjd, period_days, duration_hours, depth_ppm` |
| | `load_confirmed_planets()` | — | DataFrame: `planet_name, tic_id, hostname, system_planet_count` |
| **you** | labels | the tables above | DataFrame with `data_uri`, `label` (int) and `split` |
| 3. read | `read_lightcurve(data_uri)` | a `data_uri` | `LightCurve`: `time, flux, flux_err, sap_flux, quality, tic_id, sector` as raw numpy arrays |
| **you** | `transform(lightcurve)` | a `LightCurve` | tensor or array, shape `(readings,)` or `(channels, readings)` |
| 4. batch | `LightCurveDataset(table, transform)` | labelled table, your transform | PyTorch Dataset: `(transform(lightcurve), label)` |
| | `pad_collate` | list of examples | batch `(x, mask, y)`: `x (batch, channels, longest)`, `mask` True where real |
| **you** | model, training loop | batches | a trained `nn.Module` |
| 5. score | `report(y_true, y_pred, class_names)` | labels and predictions | balanced accuracy, 95% interval, confusion matrix (print it) |

A full training setup is a normal PyTorch DataLoader:

```python
loader = DataLoader(LightCurveDataset(table, transform), batch_size=32, shuffle=True,
                    num_workers=8, collate_fn=pad_collate)
for x, mask, y in loader:
    ...
```

## What each piece does, and doesn't

- **The data.** TESS SPOC light curves at 2-minute cadence: one star observed for one
  ~27-day sector, about 18,000 readings. `sync` has indexed sectors 1–28: 547,402 curves of
  251,324 stars.
- **`split`** is fixed per star from a hash of its TIC ID: 80% train, 10% validation and
  10% test. Every sector of a star gets the same split, so models are always tested on
  stars they never trained on.
- **`read_lightcurve`** returns exactly what is in the file. NaNs and readings with quality
  flags set are kept; clean them in your `transform`.
- **`epoch_btjd`** in `load_tois()` is on the same time scale as `LightCurve.time`
  (BTJD = BJD − 2457000), so you can work out whether a transit falls inside a curve.
- **Caching.** The first read downloads the file, from MAST's public S3 copy with the MAST
  API as fallback; later reads take about 1 ms. Each cached curve takes about 0.4 MB. When the cache passes
  `LIGHTCURVE_CACHE_GIB`, the least recently used curves are deleted automatically. A
  deleted curve is downloaded again if needed, so there is nothing to clean up.
- **Big runs.** Downloads happen inside DataLoader workers, so `num_workers=8–16` downloads
  in parallel while the GPU trains. One pass over any number of curves works with any
  cache size. Repeat passes over more curves than fit in the cache download them again,
  so size the cache to your working set.
- **Bad data.** A curve that cannot be downloaded or read raises `LightCurveUnavailable`.
  `LightCurveDataset` skips it (counting it in `dataset.skipped`) and `pad_collate` drops
  it, so training never stops for one bad file. Problems in your own `transform` are
  *not* hidden.
- **Scoring.** Balanced accuracy is the mean of per-class recall. Always guessing one
  class scores 1 / number of classes, whatever the class mix. Compare models on it, and
  trust a difference only when the 95% intervals don't heavily overlap.

## Setup

1. Python 3.13 and [uv](https://docs.astral.sh/uv/); then `uv sync`.
2. `cp .env.example .env` and fill it in: PostgreSQL connection, a password file, and
   data folders.
3. `uv run --env-file .env astro-data-lab sync --max-sectors 30` creates the tables and indexes
   sectors (metadata only). Rerun it now and then; finished sectors are skipped.
4. Optional: an MLflow server (`MLFLOW_TRACKING_URI`) to log your runs.

Other commands: `astro-data-lab cache` shows the cache size. Run the tests with
`uv run python -m unittest discover tests` (no database or network needed).

### Weekly updates

The installed user systemd timers run catalogue sync on Sundays at 03:00 and
manifest sync on Mondays at 03:00 (system local time):

```sh
uv run --env-file .env astro-data-lab sync --only catalogues
uv run --env-file .env astro-data-lab sync --only manifest --max-sectors 2 --refresh-one
```

Catalogue sync refreshes TOIs and confirmed planets. Manifest sync indexes up to
two new sectors and refreshes the oldest previously indexed sector to pick up
new or changed products. These jobs download metadata, not light-curve payloads.
The installed units live in `~/.config/systemd/user/`; they are not installed by
`uv sync`. Check schedules with `systemctl --user list-timers 'astro-*'` and logs
with `journalctl --user -u astro-catalogues.service -u astro-mast.service`.

## Layout

```
src/astro_data_lab/
├── __init__.py      the functions above, in one place
├── sync/            step 1: catalogues.py (NASA tables), manifest.py (MAST file list)
├── tables.py        step 2: load_curves, load_tois, load_confirmed_planets, split_of
├── lightcurves.py   step 3: read_lightcurve, LightCurve, the download cache
├── dataset.py       step 4: LightCurveDataset, pad_collate
├── metrics.py       step 5: report
├── database.py      PostgreSQL connection; applies schema.sql automatically
├── schema.sql       the whole database: catalogue and manifest tables
└── cli.py           `astro-data-lab sync` and `astro-data-lab cache`
notebooks/getting_started.ipynb   every step on real data
tests/test_lab.py                 fast checks
docs/                             GLOSSARY.md (astronomy terms), tmux.md (long runs over SSH)
```

The earlier framework (models, round-based trainer, foundation model, working paper)
is preserved in git history at commit `f864848`.
