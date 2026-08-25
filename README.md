# astro-data-lab

A hobbyist deep dive into stellar photometry - starting with TESS (and Kepler/K2)
light curves, working toward a homebrew Linux server that ingests survey data and
classifies stars with machine learning, around the clock.

## Roadmap

The full project plan including phases, tasks, and the reasoning behind each lives in
[`roadmap.html`](roadmap.html).

<!--
  Phase 0 status: <in progress / done>
  Currently on: <phase name>
-->

## Tech Stack


## Repository structure

```
tutorials/     Lightkurve / TESS-Kepler exploration notebooks
roadmap.html   Project roadmap and task tracker
```

## Getting started

I'm using [uv](https://docs.astral.sh/uv/). I recommend using it too.
There's no `pyproject.toml` or lockfile yet, but pinning real dependencies is on the
roadmap when i move past notebooks (Phase 7: Tracking & Repo Practices).

For now all you need is:

```sh
uv venv
uv pip install lightkurve jupyter
```

Then open the notebooks in `tutorials/` via Jupyter or VS Code.

## Tutorial Notebooks

Basic and Exploratory for now.

| Notebook | What it covers |
| --- | --- |
| `learn_the_tradecraft.ipynb` | First Lightkurve pass: download a target pixel file, extract a light curve, fold/bin it, periodograms, animations |
| `searching_downloading_data.ipynb` | In progress |

<!-- Add a row per new notebook as tutorials/ grows -->

## Commit conventions

This is a solo project, I'm committing straight to `main` for now - but keeping habits tidy so
this doesn't turn into one giant undifferentiated blob of history. I suck at this right now because I'm lazy in personal projects, so it's for my own discipline that it's explicitly outlined here:

- **One change per commit.** E.g. a notebook fix and a README update are two
  commits, not one, unless they're part of the same change.
- **[Conventional Commits](https://www.conventionalcommits.org/):**
  `<type>: <imperative subject>` - "feat: add imports cell", not
  "Added" or "Adding". Keep the subject under ~50 characters. Common types:
  - `feat` - new capability (a notebook cell, a script, a feature)
  - `fix` - correcting broken/incorrect behavior
  - `docs` - documentation only (README, comments)
  - `refactor` - restructuring with no behavior change
  - `chore` - tooling, `.gitignore`, dependencies, repo housekeeping
- **Explain *why* in the body, not *what*.** The diff already shows what changed;
  use the commit body for context a diff can't carry (why a fix was needed, what
  broke, what was tried and rejected).
- **Stage files by name** (`git add path/to/file`), not `git add -A` / `git add .`.
  One must skim `git status` before committing so nothing unintended sneaks in.
- **Never commit generated output, secrets, or environment files.** If something
  new needs excluding, add it to `.gitignore` before it gets staged, not after. (Duh.)
- **Re-read the diff before committing** (`git diff --staged`). Catching a stray
  debug print or leftover cell output here is cheaper than catching it later.
- **Small, frequent commits beat big, rare ones.** A commit you can describe in
  one sentence is usually the right size.


