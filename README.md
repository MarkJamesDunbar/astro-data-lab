# Astro Data Lab

Train PyTorch models that answer one question about a **complete TESS light curve**:
*does it contain a transit-like signal?* Then measure, honestly, whether they work.

- **One example = one complete light curve**: one star observed for one TESS
  sector, about 15,000–20,000 brightness readings over ~27 days. No cropping, no
  windows, no resampling.
- **Labels you can trust per curve.** "Signal" curves belong to TESS Objects of
  Interest, *and* a predicted transit is checked to fall inside that curve's
  readings. "No signal" curves belong to stars in no planet catalogue.
- **Unlimited extra positives.** Synthetic transits are injected into fresh
  no-signal curves, so models cannot simply memorise the ~4,000 real examples.
- **Disk stays bounded.** About 550,000 light curves (~1 TB) are streamed through a
  small cache; you never download the archive.
- **Evaluation is built in.** Scores on stars the model never saw, with a chance
  line and a confidence interval.
- **Architectures are a config change.** Write an `nn.Module`, point an experiment
  file at it, run one command. A CNN and a Transformer are included.

```
astro-data-lab sync                                     # 1. refresh catalogues and the list of light curves
astro-data-lab train experiments/transit_cnn.toml       # 2. train (resumable)
astro-data-lab evaluate experiments/transit_cnn.toml    # 3. final score on unseen test stars
astro-data-lab predict best.pt new_lc.fits              # 4. classify a new light curve
```

**New here?** Read the [Quick start](#1-quick-start), then work through
[`notebooks/walkthrough.ipynb`](notebooks/walkthrough.ipynb). It shows every step on
real data, including writing a model and training it with the platform.

---

## Contents

1. [Quick start](#1-quick-start)
2. [How it works](#2-how-it-works)
3. [Repository layout](#3-repository-layout)
4. [Setup in detail](#4-setup-in-detail)
5. [Commands](#5-commands)
6. [Experiment files](#6-experiment-files)
7. [Building a model](#7-building-a-model)
8. [Choosing an architecture, and foundation models](#8-choosing-an-architecture-and-foundation-models)
9. [Reading the results](#9-reading-the-results)
10. [Monitoring a run](#10-monitoring-a-run)
11. [Data, storage and bad data](#11-data-storage-and-bad-data)
12. [The database](#12-the-database)
13. [Extending the lab](#13-extending-the-lab)
14. [Troubleshooting](#14-troubleshooting)
15. [Known limitations](#15-known-limitations)
16. [Glossary](#16-glossary)

---

## 1. Quick start

This assumes PostgreSQL, MLflow and `.env` are already set up. Otherwise see
[Setup](#4-setup-in-detail).

```bash
uv sync                                                                    # install
uv run --env-file .env astro-data-lab sync                                 # labels + light-curve list
uv run --env-file .env astro-data-lab train experiments/transit_cnn.toml --rounds 1   # quick check
uv run --env-file .env astro-data-lab train experiments/transit_cnn.toml              # full run, resumes
uv run --env-file .env astro-data-lab status experiments/transit_cnn.toml
uv run --env-file .env astro-data-lab evaluate experiments/transit_cnn.toml
```

Every command needs `--env-file .env`, which supplies the database and storage
settings. `uv run astro-data-lab --help` lists all commands and needs no database.

To try your own architecture:

1. Prototype it in the walkthrough notebook. Sections 5 and 6 define a model and
   train it with the platform.
2. Save it as `src/astro_data_lab/models/your_model.py` and export it from
   `src/astro_data_lab/models/__init__.py`.
3. `cp experiments/transit_cnn.toml experiments/your-model.toml`, then change `name`
   and `factory`. Train and evaluate it as above.
4. Compare runs in MLflow.

---

## 2. How it works

### The pipeline

| Step | What happens | Code | Triggered by |
|---|---|---|---|
| 1a | Download the NASA confirmed-planet and TOI catalogues, including transit timing | `data/catalogues.py` | `sync` |
| 1b | Index which light-curve files exist at MAST (metadata only) | `data/manifest.py` | `sync` |
| 2 | Label each curve; split **stars** into train/validation/test | `data/labels.py` | automatic |
| 3 | Download, clean, confirm labels, inject synthetic transits, build tensors | `data/lightcurves.py` | automatic |
| 4 | The model | `models/` | your choice |
| 5 | Train in rounds, scoring validation stars after every round | `training/trainer.py` | `train` |
| 6 | Score the best checkpoint on untouched test stars | `training/evaluation.py` | `evaluate` |
| 7 | Classify one new FITS file | `predict.py` | `predict` |

### Labels

| id | class | rule | training curves |
|---|---|---|---|
| 0 | `no_signal` | the star is in neither planet catalogue | ~432,000 |
| 1 | `signal` | the star is a TOI (PC, APC, CP, KP or FP), and one of its predicted transits falls inside this curve's readings | ~4,350 |

Left out entirely:

- confirmed planets that TESS never flagged, often because they do not transit;
- false alarms (FA);
- TOIs without transit timing;
- TOI curves where no predicted transit falls inside the readings.

That last check uses each TOI's epoch, period and duration from the catalogue.
It needs the curve's timestamps, so it runs after download (`transit_readings` in
`data/lightcurves.py`). At least 10 readings must fall inside a predicted transit.
87% of candidate signal curves pass.

"Signal" includes false positives, such as eclipsing binaries and blended
neighbours. The question is *is there a transit-like event*, not *is it a planet*.
`no_signal` means **not catalogued**, not *proven quiet*: a small fraction of those
stars have undiscovered planets.

### Train / validation / test splits

Stars, not curves, are split 80/10/10 using a fixed hash of the TIC ID
(`split_of` in `data/labels.py`):

- Every sector of a star lands in the same split, so a model is never tested on a
  star it has already seen.
- The split is identical for every run, model and machine, so comparisons are fair.
- **Training uses only the train split.** Validation chooses the best checkpoint.
  Test is for the final number only, and should be used once.

### One training round

`curves_per_round` (default 1,024) is half `signal` and half `no_signal`:

- **no_signal (512):** fresh curves this run has never used. They are downloaded,
  used, then deleted, so disk use stays flat.
- **real signals (256 = 512 × (1 − `injected_fraction`)):** drawn from the ~4,350
  real signal curves. These were checked once at the start of the run and stay cached.
- **injected signals (256 = 512 × `injected_fraction`):** the *same* fresh no_signal
  curves as above, each with a synthetic transit added. The model sees one star with
  and without a transit, so it must learn the transit, not the star. No extra
  downloading is needed.

Then:

1. The next round downloads while this one trains.
2. `passes_per_round` passes of minibatch gradient descent: AdamW, cross-entropy,
   gradient clipping at 1.0. See `train_one_round` in `trainer.py`.
3. Score the fixed validation set: real curves only, 300 per class, from validation stars.
4. Save a checkpoint. If validation balanced accuracy improved, copy it to `best.pt`.
   Every curve used is recorded in PostgreSQL.
5. Delete the streamed files, log metrics to MLflow, and start the next round.

Stop with Ctrl+C or `kill <pid>` at any time. Rerunning the same command resumes
from the last saved round.

### Synthetic transits

`inject_transit` in `data/lightcurves.py` adds a periodic, trapezoid-shaped dip to a
cleaned curve:

- period: 0.5–15 days, log-uniform;
- duration: 1–8 hours, at most 15% of the period;
- epoch: centred on a real reading, so at least one transit is always visible;
- depth: chosen for a total signal-to-noise of 8–40, capped at 5%. TESS's own
  pipeline flags candidates from about 7.

Each curve's injection is seeded by its URI, so it is reproducible. Injected curves
are only ever used for training, never for validation or test.

### What the model receives

`model_input` in `data/lightcurves.py` turns one curve into a `(4, readings)` tensor:

| channel | meaning | how | clipped to |
|---|---|---|---|
| 0 | brightness change, % | `(flux / median − 1) × 100` | ±100 |
| 1 | **detrended** brightness in noise units | `detrend(...) / point_noise(...)` | ±50 |
| 2 | gap before each reading | `log(1 + gap / typical_gap)` | 0–10 |
| 3 | real reading (1) or padding (0) | | |

Cleaning keeps readings with `QUALITY == 0` and finite time and flux. It also sorts
by time and drops duplicate timestamps. There is no cropping or interpolation, and
real gaps stay gaps.

`detrend` divides by a running 1-day median, sampled every 6 hours. That removes
stellar variability slower than about a day while keeping hours-long dips. A second
pass leaves readings more than 3σ below the trend out of the median, so a transit
cannot drag the trend down and erase itself. Tests check that a 0.2% transit keeps
its full depth on a quiet star, and at least 70% on a star varying by 1%.

A minibatch pads every curve with zeros to its longest one. **Models must ignore
padding** using channel 3; `check_model` verifies this.

---

## 3. Repository layout

```
astro-data-lab/
├── README.md                     you are here
├── pyproject.toml                dependencies and the `astro-data-lab` command
├── .env.example                  template for machine settings (copy to .env)
├── experiments/
│   ├── transit_cnn.toml          CNN experiment
│   ├── transit_transformer.toml  same data and task, small Transformer
│   ├── foundation_pretrain.toml  foundation model, stage 1: self-supervised pre-training
│   ├── transit_foundation.toml   foundation model, stage 2: fine-tune on transits
│   └── transit_transformer_large.toml  control: the same large Transformer, no pre-training
├── notebooks/
│   └── walkthrough.ipynb         the whole platform on real data, step by step
├── src/astro_data_lab/
│   ├── cli.py                    the `astro-data-lab` command (sync/train/status/evaluate/predict)
│   ├── config.py                 Settings (from .env), Experiment (from TOML), class names
│   ├── predict.py                step 7: load a checkpoint, classify one FITS file
│   ├── data/
│   │   ├── catalogues.py         step 1a: NASA confirmed planets + TOIs (with transit timing)
│   │   ├── manifest.py           step 1b: which light-curve files exist at MAST
│   │   ├── labels.py             step 2: labels, star splits, sampling
│   │   └── lightcurves.py        step 3: download, clean, detrend, transit check, injection
│   ├── models/
│   │   ├── __init__.py           the model contract, build_model, check_model
│   │   ├── cnn.py                LightCurveCNN
│   │   ├── transformer.py        PatchTransformer
│   │   └── foundation.py         MaskedPatchPretrainer (pre-trains a PatchTransformer without labels)
│   ├── training/
│   │   ├── trainer.py            step 5: rounds, streaming, checkpoints, MLflow
│   │   └── evaluation.py         step 6: metrics, confusion matrix, confidence interval
│   └── database/
│       ├── __init__.py           connect() and ensure_schema()
│       └── schema.sql            the entire database (applied automatically)
├── tests/
│   └── test_lab.py               fast CPU-only tests, no database or network
└── docs/
    ├── paper/                    working paper on the results (paper.md), and make_figures.py to rebuild it
    ├── GLOSSARY.md               astronomy terms (TESS, TIC, TOI, SPOC, ...)
    └── tmux.md                   keeping long runs alive over SSH
```

Generated data lives outside the repository, under `ASTRO_DATA_DIR` (default
`/data/astro-data-lab`):

```
/data/astro-data-lab/
├── curves/                 cleaned curve cache (*.npz, ~0.3 MB each)
├── checkpoints/<run name>/ round_NNNNN.pt (latest) and best.pt
└── metadata/               manifest staging
```

---

## 4. Setup in detail

### Requirements

- Linux, Python 3.13, [uv](https://docs.astral.sh/uv/)
- PostgreSQL 14+ with an empty database, and a user that can create tables
- An MLflow tracking server
- An NVIDIA GPU with CUDA 12.4-compatible drivers. CPU training works, but slowly.
  `predict` and the tests do not need a GPU.
- A fast internet connection. Training downloads ~1 GB of FITS files per round, so
  bandwidth, not the GPU, sets the pace (see [Monitoring](#10-monitoring-a-run)).

### Install

```bash
git clone <this repo> && cd astro-data-lab
uv sync            # PyTorch (CUDA 12.4 build) plus dev tools for the notebook
```

### Configure `.env`

```bash
cp .env.example .env
```

| variable | meaning | default |
|---|---|---|
| `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER` | PostgreSQL connection | required |
| `ASTRO_DB_PASSWORD_FILE` | path to a file containing only the DB password | required |
| `ASTRO_DATA_DIR` | root for the cache and checkpoints | `/data/astro-data-lab` |
| `MAST_INVENTORY_DIR` | staging area for manifest sync | required for `sync` |
| `CATALOGUE_SNAPSHOT_DIR` | raw catalogue downloads are archived here | required for `sync` |
| `CURVE_CACHE_GIB` | refuse to grow the curve cache beyond this | `50` |
| `FREE_SPACE_FLOOR_GIB` | never download if free disk space would fall below this | `330` |
| `MLFLOW_TRACKING_URI` | MLflow server | `http://127.0.0.1:5000` |
| `MLFLOW_EXPERIMENT_NAME` | MLflow experiment that runs are logged to | `tess-classifiers` |

The password lives in a separate file, so `.env` holds no secrets. Never commit
`.env`; it is already in `.gitignore`.

### First sync

```bash
uv run --env-file .env astro-data-lab sync --max-sectors 30
```

This creates all tables, downloads both catalogues and indexes up to 30 sectors.
It downloads metadata only, about 20,000 curves per sector. Rerun it periodically:
completed sectors are skipped and new ones are added.

---

## 5. Commands

Run every command as `uv run --env-file .env astro-data-lab <command>`.

### `sync`: refresh labels and the light-curve list

```
astro-data-lab sync [--only catalogues|manifest] [--max-sectors N]
```

Downloads the latest NASA catalogues: confirmed planets, and TOIs with their
disposition, epoch, period, duration and depth. Each download is archived and every
change is logged. Then it indexes up to `N` new TESS sectors (default 2). It never
downloads light curves.

### `train`: train or resume an experiment

```
astro-data-lab train EXPERIMENT.toml [--rounds N] [--curves-per-round N]
```

- **First run:** creates the run, named by `name` in the file. It checks every real
  signal training curve for a transit; the first time, this downloads ~5,000 curves,
  which are cached afterwards. Then it trains until `rounds` total rounds are done
  (`rounds = 0` means until stopped).
- **Later runs with the same file:** **resume** from the last saved round.
- `--rounds N`: train **N more** rounds, then stop. `--rounds 1` is a good quick check.
- `--curves-per-round N` changes the experiment, so use it only with a fresh `name`.
- Editing the file after a run has started is refused with *"already exists with
  different settings"*. Change `name` for a new run.
- Only one `train` can run per machine at a time; a lock file enforces this.

Each round prints a line like this:

```
Round 1: 128 curves (32 injected), 12 steps, train loss 0.753 | validation balanced acc 0.562 (chance 0.500), macro F1 0.496 *best* | GPU 0s, waited 87s for data
```

### `status`: progress of a run

```
astro-data-lab status EXPERIMENT.toml
```

Shows rounds completed, examples trained, the best validation score, the best
checkpoint's path, and how many curves have been excluded as unusable.

### `evaluate`: the honest final score

```
astro-data-lab evaluate EXPERIMENT.toml [--split test|validation] [--checkpoint PATH]
```

Loads the run's `best.pt`, or `--checkpoint`, and scores it on the fixed test set:
`validation_curves_per_class` real curves per class from test stars. It prints the
confusion matrix and logs `test_*` metrics to the run's MLflow entry. Use
`--split test` once per model, when you have finished tuning.

### `predict`: classify one light curve

```
astro-data-lab predict CHECKPOINT.pt LIGHTCURVE_lc.fits [--device cpu|cuda]
```

Prints JSON with the TIC ID, sector, readings used, the prediction and both class
scores. This real output comes from the walkthrough's demo model:

```json
{
  "tic_id": 278862750,
  "sector": 1,
  "readings_used": 18277,
  "prediction": "no_signal",
  "scores": {"no_signal": 0.522, "signal": 0.478},
  "model": "notebook-dip-detector, round 1"
}
```

 The file can be any TESS SPOC
`*_lc.fits`, and is cleaned exactly like training data. Scores are softmax outputs,
**not calibrated probabilities** (see [Limitations](#15-known-limitations)).

---

## 6. Experiment files

An experiment file completely describes a run. The run's `name` identifies it in
PostgreSQL, MLflow and the checkpoint folder. **If you change anything, change the
name.**

```toml
[run]
name = "transit-cnn-v1"
target = "transit"

[data]
curves_per_round = 1024
injected_fraction = 0.5
rounds = 100
validation_curves_per_class = 300

[training]
passes_per_round = 3
minibatch_size = 32
learning_rate = 0.001

[model]
factory = "astro_data_lab.models:LightCurveCNN"

[model.options]
width = 32
```

Every setting, with its default:

| setting | default | meaning |
|---|---|---|
| `name` | required | letters, digits, `.`, `_`, `-` |
| `target` | `transit` | `transit` (classify), or `pretrain` (self-supervised, no labels; see [section 8](#8-choosing-an-architecture-and-foundation-models)) |
| `curves_per_round` | `1024` | curves per round: half signal, half no_signal |
| `injected_fraction` | `0.5` | share of signal examples that are synthetic, in [0, 1) |
| `rounds` | `30` | total rounds; `0` = until stopped |
| `validation_curves_per_class` | `300` | size of the validation set, and of the test set used by `evaluate` |
| `download_workers` | `8` | parallel downloads |
| `max_bad_fraction` | `0.2` | a round fails if more than this share of its curves are unreadable or unreachable |
| `passes_per_round` | `1` | gradient passes over each downloaded round |
| `minibatch_size` | `32` | curves per gradient step |
| `learning_rate` | `0.001` | AdamW learning rate |
| `weight_decay` | `0.0001` | AdamW weight decay |
| `seed` | `42` | controls sampling, shuffling and initialisation |
| `factory` | `astro_data_lab.models:LightCurveCNN` | `module:Callable` that builds the model |
| `options` (`[model.options]`) | `{}` | keyword arguments passed to the factory |
| `init_from` | `""` | start from another run's best weights: a run name or a checkpoint path. Every weight whose name and shape match is copied; the rest start fresh. Used only when a run starts, not on resume |

Unknown keys and wrong types, such as `rounds = "5"`, are rejected with a clear
message. The section names are for readability only: any setting may appear in any
of `[run]`, `[data]`, `[training]` and `[model]`.

---

## 7. Building a model

### The contract

```python
model = YourModel(num_classes=2, **options)    # must return a torch.nn.Module
logits = model(inputs)
# inputs: float32 (batch, 4, readings); readings is ~15k-20k and varies by batch
# logits: (batch, 2), raw scores before softmax
```

Rules:

1. **Accept any length.** Curves differ in length, and each batch is padded to its
   longest curve.
2. **Ignore padding.** Where `inputs[:, 3] == 0` the reading is padding, and padding
   must not change a curve's result. Multiply features by the mask, and pool with
   masked means or maxima.
3. **Output raw logits.** The trainer applies cross-entropy and evaluation applies
   softmax.
4. **Take `num_classes` as a keyword argument.** Everything else comes from
   `[model.options]`.

`check_model(model, num_classes)` in `astro_data_lab.models` tests the shape,
finite outputs and padding invariance in under a second. Run it first.

### The included models

**`LightCurveCNN`** (`models/cnn.py`), 147k parameters:

```
input (B, 4, L)
 └─ 4 × MaskedConvBlock: Conv1d(kernel 9) → GELU → MaxPool(4)   channels 3→32→64→64→128
      (padding is re-masked in every block; afterwards one step covers 256 readings, ~8.5 h)
 └─ masked mean + masked max over time → (B, 256)
 └─ Linear(256→64) → GELU → Dropout(0.2) → Linear(64→N)
```

Options: `width` (32), `dropout` (0.2).

**`PatchTransformer`** (`models/transformer.py`), 344k parameters:

```
input (B, 4, L)
 └─ cut into patches of 64 readings (~2 h), flatten → Linear → token (B, L/64, width)
 └─ + learned position embedding
 └─ Transformer encoder (pre-norm, GELU), all-padding patches masked out of attention
 └─ masked mean over patches → Linear → N
```

Options: `patch` (64), `width` (64), `depth` (4), `heads` (4), `dropout` (0.1),
`max_patches` (2048).

Both take under 70 ms per training step for 32 curves of 18,000 readings on a 6 GB
RTX 2060. Neither comes close to the GPU's memory.

**`MaskedPatchPretrainer`** (`models/foundation.py`): a `PatchTransformer` plus a
`hidden_token` and a small `decoder`, used only to pre-train (see section 8). Extra
option: `mask_fraction` (0.4). Its encoder weights carry the same names as a
`PatchTransformer`'s, so `init_from` transfers them directly.

### Adding a model to the package

The walkthrough notebook defines `DipDetector`, a ~2,000-parameter matched filter.
To make it a real experiment:

```python
# src/astro_data_lab/models/dip.py
import torch
from torch import nn


class DipDetector(nn.Module):
    """Slide learned dip templates along the detrended curve; report the strongest and average match."""

    def __init__(self, num_classes: int, templates: int = 16, length: int = 121):
        super().__init__()
        self.match = nn.Conv1d(1, templates, length, padding=length // 2)
        self.head = nn.Linear(2 * templates, num_classes)

    def forward(self, inputs):
        valid = inputs[:, 3:4] > 0
        response = torch.relu(self.match(inputs[:, 1:2] * valid))
        strongest = response.masked_fill(~valid, 0).amax(-1)
        average = (response * valid).sum(-1) / valid.sum(-1)
        return self.head(torch.cat((strongest, average), dim=1))
```

```python
# src/astro_data_lab/models/__init__.py  (add the import and the __all__ entry)
from astro_data_lab.models.dip import DipDetector
```

```toml
# experiments/dip_detector.toml
[run]
name = "dip-detector-v1"

[data]
rounds = 100

[training]
passes_per_round = 3
learning_rate = 0.003

[model]
factory = "astro_data_lab.models:DipDetector"

[model.options]
templates = 16
```

```bash
uv run --env-file .env astro-data-lab train experiments/dip_detector.toml --rounds 1
uv run --env-file .env astro-data-lab train experiments/dip_detector.toml
uv run --env-file .env astro-data-lab evaluate experiments/dip_detector.toml
```

`factory` can point at anything importable: a class or a function, in any
installed package. In a notebook, `"__main__:YourModel"` works while that kernel
runs; the walkthrough uses this.

### Building blocks for notebooks

Everything the command line does is a plain Python function, so a notebook can do the
same work. The walkthrough uses each of these; none of it is reimplemented there.

| task | function |
|---|---|
| settings from `.env`, an experiment in code | `config.Settings.from_env()`, `config.Experiment(...)` |
| how many usable curves per class and split | `data.labels.summary(conn)` |
| the same curves every time, per split | `data.labels.fixed_sample(conn, target, split, per_class)` |
| download, clean, check labels, build model input | `data.lightcurves.prepare_many(curves, settings, workers)` |
| a prepared curve's cleaned time and flux | `data.lightcurves.curve_arrays(curve, settings)` |
| predicted transit times inside a curve | `data.lightcurves.transit_centres(time, curve.ephemerides)` |
| detrending, model input, synthetic transits | `detrend`, `model_input`, `inject_transit` (same module) |
| download one raw FITS file | `data.lightcurves.download_fits(session, uri, size, path)` |
| check a model obeys the contract | `models.check_model(model, num_classes)` |
| train or resume a run | `training.trainer.train(experiment, settings)` |
| one round of gradient descent on your own data | `training.trainer.train_one_round(...)` |
| progress, per-round metrics, forget a run | `training.trainer.status`, `.history`, `.delete_run` |
| score on held-out stars | `training.evaluation.evaluate(name, settings, split)` |
| classify one FITS file | `predict.predict(checkpoint, fits_path)` |

All of these live under `astro_data_lab.`, e.g. `from astro_data_lab.data import labels`.

Open notebooks in VS Code with the project's `.venv` as the kernel, or run
`uv run --with jupyterlab jupyter lab`. They load `.env` themselves.

### Practical tips

- **Memory** grows with `minibatch_size × readings × channels`, so downsample early.
- **Learning rate:** if you see *"minibatches produced non-finite loss"*, lower it.
  Transformers usually want about half the CNN's rate.
- **Speed:** a round's GPU work takes about a second; downloading takes the rest.
  A bigger model costs little extra wall time, and `passes_per_round` makes more use
  of each download.
- **Fair comparisons:** runs with the same `validation_curves_per_class` use exactly
  the same validation and test curves.

---

## 8. Choosing an architecture, and foundation models

**What matters most is the data, not the architecture.** The first version of this
lab classified stars into four catalogue classes with a CNN. It reached a test
balanced accuracy of 0.389 against a chance level of 0.25. The labels were the
problem: 46% of "confirmed planet hosts" were never flagged by TESS as transiting,
and "confirmed" versus "candidate" is decided by follow-up observations, not by the
light curve. A better network cannot learn labels the data does not contain. The
current task fixes this in three ways: each "signal" curve is verified to contain a
transit, slow stellar variability is removed from the input, and synthetic
transits provide unlimited positives.

**CNN vs Transformer for this task:**

| | CNN (`LightCurveCNN`) | Transformer (`PatchTransformer`) |
|---|---|---|
| good at | local shapes such as a dip's depth, width and ingress | relating distant parts of the curve, such as repeated transits days apart |
| built-in assumption | a dip means the same wherever it occurs | almost none; it learns everything from data |
| data needs | modest | large, which injection helps with |
| cost here | very low | low (attention over ~300 patches) |

The CNN is a strong, cheap baseline. The Transformer should gain the most from more
data and from pre-training. Train both on the same data and compare their
validation scores; that is what the two experiment files are for.

**Other proven approaches** include phase folding (the AstroNet family, built for
Kepler and TESS candidate vetting). Those models take a known period, fold the
curve so every transit stacks into one clean dip, and classify that. They are very
good at *planet vs false positive* for an existing detection. They answer a
different question, because they need the period first. Classic searches such as
Box Least Squares find the period without any machine learning.

### The foundation model

The labelled task has about 4,400 real transit curves, but the training split holds
**438,704 curves** in total. A foundation model learns from all of them without
labels, then is fine-tuned on the labelled task. This lab does that in two stages.

**Stage 1: self-supervised pre-training** (`experiments/foundation_pretrain.toml`,
`target = "pretrain"`). Every round streams 2,048 training-split curves the run has
never used, of any kind, and deletes them afterwards. For each curve,
`MaskedPatchPretrainer` hides 40% of its ~2-hour patches and learns to predict their
readings (raw and detrended flux) from the rest. This is the same idea as BERT,
MAE and Astromer. Filling a hidden patch well needs an understanding of stellar
variability, noise, gaps and recurring dips, which is what transit detection needs
too. One pass over all 438,704 curves is ~830 GB of downloads, about 12 hours at
~10 curves/s. `rounds = 0` means until every training-split curve has been used once.

The model is 15M parameters (width 384, 8 layers, 8 heads), 44 times the small
Transformer. Each round runs 3 passes with fresh random masks.

Two numbers are logged per round, both on the fixed validation stars:

| metric | meaning |
|---|---|
| `validation_reconstruction_skill` | `1 − masked loss ÷ interpolation loss`. The interpolation baseline predicts each hidden patch as the average of its nearest visible neighbours. 0 = no better than that, 1 = perfect, negative = worse. `best.pt` is the round with the highest skill |
| `validation_probe_balanced_accuracy` | a logistic regression on the frozen encoder's curve embeddings, predicting transit vs no transit, scored by 5-fold cross-validation. It measures how much transit information the encoder learned **without ever being told about transits**. Round 0 logs the untrained encoder for comparison |

**Stage 2: fine-tuning** (`experiments/transit_foundation.toml`). An ordinary
`PatchTransformer` experiment with the same options and
`init_from = "foundation-v1"`, which copies every encoder weight from the
pre-training run's `best.pt`. Only the 2-class head starts from random weights.

**The control** (`experiments/transit_transformer_large.toml`) is identical except
for `init_from`. **Pre-training helped only if the fine-tuned run's test score beats
the control's.**

Run all three in order:

```bash
uv run --env-file .env astro-data-lab train experiments/foundation_pretrain.toml
uv run --env-file .env astro-data-lab train experiments/transit_foundation.toml
uv run --env-file .env astro-data-lab evaluate experiments/transit_foundation.toml
uv run --env-file .env astro-data-lab train experiments/transit_transformer_large.toml
uv run --env-file .env astro-data-lab evaluate experiments/transit_transformer_large.toml
```

Stopping pre-training early (Ctrl+C or `kill`) is fine: fine-tuning uses the best
checkpoint so far. `evaluate` refuses a pre-training run, because there is nothing
to classify yet; evaluate the fine-tuned run instead.

To pre-train a different encoder, subclass it the same way: add the extra
pre-training parts, and keep the shared weights named as in the classifier.

---

## 9. Reading the results

### Balanced accuracy and chance

Validation and test sets hold the same number of curves per class. So:

- Always predicting one class scores a **balanced accuracy of 0.5**.
- `evaluate` reports a 95% bootstrap confidence interval. It prints **BETTER THAN
  CHANCE** only when the interval's lower end is above 0.5; otherwise it prints
  *NOT distinguishable from chance*.
- **Recall per class** tells you what share of real transits the model catches
  (signal recall), and how often it raises a false alarm on quiet stars
  (1 − no_signal recall).

### Example

This is the real `evaluate` output for the walkthrough's demo: `DipDetector`, about
2,000 parameters, after one 128-curve round, on 80 test curves.

```
Balanced accuracy 0.688 (95% CI 0.612-0.761); chance = 0.500 -> BETTER THAN CHANCE
Macro F1 0.654 | accuracy 0.688 | loss 0.590

true / predicted   no_signal      signal    recall  precision  curves
       no_signal          40           0     1.000      0.615      40
          signal          25          15     0.375      1.000      40
```

Rows are the true class and columns the prediction. This tiny model never raised
a false alarm on a quiet star, but it caught only 38% of real transits: it is
cautious. With only 80 curves the interval is wide. The full experiments use 300
real curves per class.

### Validation vs test

The best checkpoint is chosen using validation stars, so its validation score is
slightly optimistic. Report the **test** score from `evaluate`, and stop tuning once
you have looked at it.

---

## 10. Monitoring a run

- **Terminal or log:** one line per round (see `train`). For background runs:

  ```bash
  nohup uv run --env-file .env astro-data-lab train experiments/transit_cnn.toml > train.log 2>&1 &
  tail -f train.log
  ```

  To survive SSH disconnects, use tmux; see [docs/tmux.md](docs/tmux.md).

- **MLflow:** experiment `MLFLOW_EXPERIMENT_NAME` (default **tess-classifiers**).
  Logged per round:

  | metric | meaning |
  |---|---|
  | `validation_balanced_accuracy` | **the main number**; compare runs on this |
  | `validation_balanced_accuracy_low` | lower end of its 95% interval |
  | `validation_recall_signal`, `validation_recall_no_signal` | transits caught; quiet stars correctly left alone |
  | `validation_loss`, `train_loss` | cross-entropy |
  | `curves_trained`, `injected_signals`, `optimizer_steps` | work done |
  | `excluded_curves`, `failed_downloads`, `signals_without_transit`, `skipped_minibatches` | data problems |
  | `gpu_seconds`, `waiting_for_data_seconds` | where the time goes |
  | `test_*` | written by `evaluate` |

  The parameters record the full experiment, so every run can be reproduced.

- **Where the time goes:** each round downloads about 512 fresh curves (~1 GB).
  At 50 MB/s that takes about 20 seconds; on a 2 MB/s home connection, about 8
  minutes. The GPU works for about a second per pass. If `waiting_for_data_seconds`
  dominates, raise `passes_per_round` or `download_workers`, or get a faster
  connection. A bigger GPU will not help.

- **GPU dashboards** sample far more slowly than these one-second bursts, and often
  show 0%. Use `nvidia-smi -l 1`, or a dashboard that samples every second.

---

## 11. Data, storage and bad data

### Downloads

Light curves come from MAST's public copy on AWS S3 (`stpubdata`); the MAST API is the
fallback. Each curve is about 1.9 MB, so **your internet speed sets the pace of
training**. Every round streams fresh `no_signal` curves: about 1 GB for
`curves_per_round = 1024`. At 30 MB/s that takes about 35 seconds. At 1 MB/s, as
on a weak Wi-Fi link, it takes about 15 minutes. `waiting_for_data_seconds` in
MLflow and `waited Ns for data` in the log show it. `passes_per_round` gets more
training from each download.

### Disk usage

- Cleaned curves are cached in `ASTRO_DATA_DIR/curves` as compressed `.npz` files
  (~0.3 MB each). Raw FITS files (~2 MB) are deleted as soon as they are cleaned.
- Real signal training curves (~5,000) and the validation and test curves stay
  cached: about 2–3 GB. Streamed no_signal curves are deleted after their round,
  so the cache does not grow as a run continues.
- Downloads stop if the cache would exceed `CURVE_CACHE_GIB`, or if free disk
  space would fall below `FREE_SPACE_FLOOR_GIB`.
- Checkpoints: one latest `round_NNNNN.pt` plus `best.pt` per run, ~2–4 MB each.
- The curve cache can be deleted whenever no training is running
  (`rm -r /data/astro-data-lab/curves`). It is rebuilt on demand.

### Bad data never stops a run

| problem | what happens |
|---|---|
| unreadable FITS, fewer than 100 good readings, non-positive median flux | curve skipped and recorded in `excluded_curves` with the reason; no run selects it again |
| a signal curve whose predicted transits all miss its readings | left out of training and evaluation; counted as `signals_without_transit` |
| download fails (3 attempts each from the public S3 copy, then MAST) | curve skipped for this round only |
| extreme values (flares, glitches) | input channels are clipped (see [What the model receives](#what-the-model-receives)) |
| a minibatch produces NaN or inf loss | that step is skipped; if more than 10% are skipped, the run stops, usually because the learning rate is too high |
| more than `max_bad_fraction` of a round is unreadable or unreachable | the run stops with a clear error, usually because MAST or the network is down; rerun to resume |

To inspect the exclusions:

```sql
SELECT reason, COUNT(*) FROM excluded_curves GROUP BY reason ORDER BY 2 DESC;
```

---

## 12. The database

`src/astro_data_lab/database/schema.sql` is the whole database. Every command
applies it with `CREATE ... IF NOT EXISTS`, so there are **no migration scripts**
and existing data is never touched.

| table | filled by | contents |
|---|---|---|
| `tic_objects` | sync | every star (TIC ID) seen in any source |
| `confirmed_planets` | sync | NASA confirmed planets → host TIC |
| `tess_objects_of_interest` | sync | TOIs: disposition (PC, APC, CP, KP, FP, FA), epoch, period, duration, depth |
| `catalogue_sync_runs`, `catalogue_change_events` | sync | history: when catalogues were fetched and what changed |
| `mast_products` | sync | the manifest: every light-curve file (URI, TIC, sector, size) |
| `ingestion_sync_runs` | sync | which sectors have been indexed |
| `excluded_curves` | training | unusable curves and why |
| `training_runs` | training | one row per experiment: config, progress, best checkpoint, MLflow ID |
| `training_examples` | training | every curve used in every round, with its label and whether a transit was injected |

Useful queries:

```sql
-- what has a run trained on?
SELECT round, COUNT(*) FILTER (WHERE target = 1 AND NOT injected) AS real_signals,
       COUNT(*) FILTER (WHERE injected) AS injected, COUNT(*) FILTER (WHERE target = 0) AS no_signal
FROM training_examples e JOIN training_runs r ON r.id = e.run_id
WHERE r.name = 'transit-cnn-v1' GROUP BY round ORDER BY round;

```

To forget a run completely (database records and checkpoints; MLflow keeps its history):

```python
from astro_data_lab.config import Settings
from astro_data_lab.training.trainer import delete_run
delete_run("my-broken-run", Settings.from_env())
```

---

## 13. Extending the lab

- **New model:** see [Building a model](#7-building-a-model). No other code changes
  are needed.
- **New target:** add its class names to `TARGETS` in `config.py`, and its labelling
  SQL to `_LABELLED` in `data/labels.py`. The SQL must yield `data_uri`,
  `size_bytes`, `ephemerides` and `target` (or `NULL` to leave a curve out). Splits,
  balancing, evaluation and prediction then follow automatically. Injection assumes
  class 0 is "nothing" and class 1 is "transit".
- **Different input features:** edit `model_input` in `data/lightcurves.py` and
  increase `INPUT_VERSION`. Old checkpoints are then refused instead of silently
  getting different inputs. The cache stores cleaned time and flux, not features,
  so it stays valid.
- **Tests:** `uv run python -m unittest discover tests` takes about 2 seconds on CPU,
  with no database or network. The tests cover config validation, bad data, input
  clipping, detrending, the transit check, injection, star splits, round
  composition, padding invariance for both models, and the metrics. One also checks
  that the CNN learns a synthetic transit. Add a test when you change any of these.

---

## 14. Troubleshooting

| message | cause and fix |
|---|---|
| `KeyError: 'ASTRO_DB_PASSWORD_FILE'` or another setting | you forgot `--env-file .env` |
| `Run '...' already exists with different settings` | the experiment file changed after the run started; change `name` or restore the file |
| `... trained on older model inputs` / `older version of the model inputs` | the checkpoint predates a change to `model_input`; train a new run |
| `Another training run is active on this machine` | only one `train` at a time; check with `ps aux \| grep astro-data-lab` |
| `No training curves for class(es) [...]` | run `sync` first |
| `x/y curves unusable or unreachable` | MAST or the network is down; wait and rerun, and it resumes |
| `free-space floor reached` / `curve cache exceeds` | free some disk space, or adjust `FREE_SPACE_FLOOR_GIB` / `CURVE_CACHE_GIB` |
| `minibatches produced non-finite loss` | lower `learning_rate` |
| `padding changed the prediction` (from `check_model`) | your model uses padded positions; mask with `inputs[:, 3:4]` |
| `MLflow experiment '...' is deleted` | it was deleted in the MLflow UI; restore it there, or set `MLFLOW_EXPERIMENT_NAME` |
| `Previous MLflow run is missing or deleted` | harmless; logging continues in a new MLflow run |
| rounds are slow and `waiting_for_data_seconds` is high | download bandwidth; see [Monitoring](#10-monitoring-a-run) |

---

## 15. Known limitations

- **"Signal" is not "planet".** It includes eclipsing binaries and other false
  positives, because the question is whether a transit-like event is present.
- **Some no_signal stars have undiscovered planets.** The labels are slightly noisy
  in that direction.
- **Transit timing is projected** from the catalogue epoch and period, sometimes
  across years. Small period errors can push predictions off. The check is
  conservative: such curves are left out rather than mislabelled.
- **Scores are not probabilities.** Softmax outputs are uncalibrated, and validation
  and test sets are balanced. In reality, transit signals are rare.
- **One sector per example.** Multiple sectors of a star are not stitched together.
- **Manifest coverage** is whatever `sync` has indexed (currently TESS sectors 1–28),
  not the whole TESS archive.
- **Download bandwidth sets the pace**, not the GPU.

---

## 16. Glossary

See [docs/GLOSSARY.md](docs/GLOSSARY.md) for TESS, TIC, TOI, SPOC, PDCSAP, sector,
transit and the catalogue disposition codes.
