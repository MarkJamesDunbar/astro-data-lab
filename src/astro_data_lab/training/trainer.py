"""Step 5: train in rounds, streaming curves through a bounded disk cache.

Every round:
  1. picks `curves_per_round` training curves, half per class. Real signals (a
     few thousand) are cached and re-sampled each round. The huge no_signal
     class streams fresh curves that are deleted after the round. Some of those
     same curves are reused as "signal" examples with a synthetic transit added
     (`injected_fraction`), so the model sees one star with and without a
     transit and must learn the transit itself, at no extra download cost;
  2. downloads and cleans them (the next round downloads while this one trains);
  3. runs `passes_per_round` passes of minibatch gradient descent over the round;
  4. scores the fixed validation set (stars the model never trains on);
  5. saves a checkpoint and records exactly which curves were used.

Bad curves are skipped and remembered in `excluded_curves`; a round only fails
if more than `max_bad_fraction` of its curves are bad (that signals an outage).
"""
from __future__ import annotations

import dataclasses
import fcntl
import json
import queue
import random
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import mlflow
import torch
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from astro_data_lab.config import Experiment, Settings
from astro_data_lab.data import labels
from astro_data_lab.data.labels import NO_SIGNAL, SIGNAL, Curve
from astro_data_lab.data.lightcurves import (
    INPUT_VERSION, Prepared, cache_path, check_quality, collate, prepare_many,
)
from astro_data_lab.database import connect, ensure_schema
from astro_data_lab.models import build_model
from astro_data_lab.training.evaluation import evaluate_curves, held_out_set

STREAM_CLASS_ABOVE = 20_000  # classes with more training curves than this are streamed
_DONE = object()


@dataclass
class Round:
    number: int
    data: Prepared
    streamed: list[Path]         # files to delete once the round is committed


class RoundSampler:
    """Which curves each round uses. Fixed for the run once training starts."""

    def __init__(self, experiment: Experiment, run_id: int, settings: Settings) -> None:
        self.experiment, self.run_id = experiment, run_id
        with connect() as conn:
            counts = labels.class_counts(conn, experiment.target, "train")
            missing = [name for i, name in enumerate(experiment.classes) if not counts.get(i)]
            if missing:
                raise RuntimeError(f"No training curves for class(es) {missing}. "
                                   "Run `astro-data-lab sync` first.")
            self.streamed = {i for i, n in counts.items() if n > STREAM_CLASS_ABOVE}
            self.pools = {i: labels.training_pool(conn, experiment.target, i)
                          for i in counts if i not in self.streamed}
        self.in_flight: set[str] = set()
        for label, pool in self.pools.items():
            if any(c.ephemerides for c in pool):  # keep only curves whose transit is really there
                print(f"Checking {len(pool):,} {experiment.classes[label]} training curves for a transit "
                      "inside the readings (downloads once, then cached)...", flush=True)
                vetted = prepare_many(pool, settings, experiment.download_workers, load=False)
                check_quality(vetted, len(pool), experiment.max_bad_fraction, "training pool")
                self.pools[label] = vetted.curves
                print(f"  {len(vetted.curves):,} contain a transit; {vetted.no_transit:,} do not "
                      "and are left out.", flush=True)
        names = experiment.classes
        print("Training curves per class: " + ", ".join(
            f"{names[i]}={len(self.pools[i]) if i in self.pools else counts[i]:,}"
            + (" (streamed)" if i in self.streamed else " (cached, re-sampled)")
            for i in sorted(counts)), flush=True)

    def _draw(self, label: int, count: int, rng: random.Random) -> list[Curve] | None:
        if count <= 0:
            return []
        if label in self.streamed:
            with connect() as conn:
                fresh = labels.unseen_training_curves(conn, self.experiment.target, label, self.run_id,
                                                      count, self.experiment.seed, self.in_flight)
            self.in_flight.update(c.data_uri for c in fresh)
            return fresh or None
        pool = self.pools[label]
        return rng.sample(pool, count) if len(pool) >= count else rng.choices(pool, k=count)

    def select(self, number: int) -> tuple[list[Curve], list[Curve]] | None:
        """Returns (all curves, streamed curves), or None when streamed data runs out."""
        experiment = self.experiment
        per_class = experiment.curves_per_round // len(experiment.classes)
        injected = round(per_class * experiment.injected_fraction)
        rng = random.Random(f"{experiment.seed}:{number}")
        chosen: list[Curve] = []
        streamed: list[Curve] = []
        for label in range(len(experiment.classes)):
            drawn = self._draw(label, per_class - (injected if label == SIGNAL else 0), rng)
            if drawn is None:
                return None
            chosen += drawn
            if label in self.streamed:
                streamed += drawn
            if label == NO_SIGNAL:  # the same curves, plus a synthetic transit, as signal examples
                chosen += [dataclasses.replace(c, target=SIGNAL, inject=True) for c in drawn[:injected]]
        rng.shuffle(chosen)
        return chosen, streamed


def _produce(plan: RoundSampler, first: int, last: int | None, settings: Settings,
             output: queue.Queue, stop: threading.Event) -> None:
    def put(item: object) -> None:
        while not stop.is_set():
            try:
                output.put(item, timeout=1)
                return
            except queue.Full:
                continue

    try:
        number = first
        while not stop.is_set() and (last is None or number <= last):
            selection = plan.select(number)
            if selection is None:
                print("No unseen streamed curves remain; finishing.", flush=True)
                break
            chosen, streamed = selection
            data = prepare_many(chosen, settings, plan.experiment.download_workers, stop)
            if stop.is_set():
                return
            check_quality(data, len(chosen), plan.experiment.max_bad_fraction, f"Round {number}")
            put(Round(number, data, [cache_path(settings.curve_cache, c.data_uri) for c in streamed]))
            number += 1
        put(_DONE)
    except BaseException as error:  # handed to the training loop, which re-raises it
        put(error)


def train_one_round(model: torch.nn.Module, optimizer: torch.optim.Optimizer, data: Prepared,
                 minibatch_size: int, device: torch.device, seed: int) -> tuple[float, int, int]:
    """One pass over the round in random minibatches. Returns mean loss, steps, skipped."""
    model.train()
    order = torch.randperm(len(data.curves), generator=torch.Generator().manual_seed(seed))
    total, steps, skipped, trained = 0.0, 0, 0, 0
    for start in range(0, len(order), minibatch_size):
        index = order[start:start + minibatch_size].tolist()
        inputs = collate([data.tensors[i] for i in index]).to(device, non_blocking=True)
        targets = torch.tensor([data.curves[i].target for i in index], device=device)
        loss = F.cross_entropy(model(inputs), targets)
        if not torch.isfinite(loss):
            skipped += 1
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * len(index)
        trained += len(index)
        steps += 1
    if skipped > max(1, steps // 10):
        raise RuntimeError(f"{skipped} minibatches produced non-finite loss; lower learning_rate")
    return total / max(1, trained), steps, skipped


def _save(path: Path, contents: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(contents, temporary)
    temporary.replace(path)


def _open_run(experiment: Experiment) -> tuple[int, dict]:
    with connect() as conn:
        conn.execute("INSERT INTO training_runs (name, config) VALUES (%s, %s) "
                     "ON CONFLICT (name) DO NOTHING", (experiment.name, json.dumps(experiment.to_dict())))
        row = conn.execute("SELECT id, config, rounds_completed, checkpoint_path, "
                           "best_validation_score, mlflow_run_id FROM training_runs WHERE name = %s",
                           (experiment.name,)).fetchone()
    if row[1] != experiment.to_dict():
        raise RuntimeError(f"Run '{experiment.name}' already exists with different settings. "
                           "Give the experiment a new name.")
    return row[0], {"rounds": row[2], "checkpoint": row[3], "best": row[4], "mlflow": row[5]}


def _mlflow_run(settings: Settings, experiment: Experiment, run_id: int, existing: str | None) -> str:
    mlflow.set_tracking_uri(settings.mlflow_uri)
    found = MlflowClient().get_experiment_by_name(settings.mlflow_experiment)
    if found is not None and found.lifecycle_stage == "deleted":
        raise RuntimeError(f"MLflow experiment '{settings.mlflow_experiment}' is deleted. Restore it in the "
                           "MLflow UI, or set MLFLOW_EXPERIMENT_NAME in .env to use another one.")
    mlflow.set_experiment(settings.mlflow_experiment)
    if existing:
        try:
            if MlflowClient().get_run(existing).info.lifecycle_stage == "active":
                return existing
        except MlflowException:
            pass
        print("Previous MLflow run is missing or deleted; logging to a new one.", flush=True)
    with mlflow.start_run(run_name=experiment.name) as run:
        params = {k: v for k, v in experiment.to_dict().items() if k != "options"}
        mlflow.log_params({**params, "options": json.dumps(experiment.options)})
    with connect() as conn:
        conn.execute("UPDATE training_runs SET mlflow_run_id = %s WHERE id = %s",
                     (run.info.run_id, run_id))
    return run.info.run_id


def train(experiment: Experiment, settings: Settings, rounds: int | None = None) -> None:
    """Train (or resume) `experiment`. Ctrl+C stops safely after the last saved round."""
    ensure_schema()
    run_dir = settings.checkpoints / experiment.name
    run_dir.mkdir(parents=True, exist_ok=True)
    with (settings.checkpoints / ".trainer.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another training run is active on this machine") from None
        _train_locked(experiment, settings, run_dir, rounds)


def _train_locked(experiment: Experiment, settings: Settings, run_dir: Path, rounds: int | None) -> None:
    run_id, state = _open_run(experiment)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no CUDA GPU found; training on CPU will be slow.", flush=True)
    torch.manual_seed(experiment.seed)
    model = build_model(experiment.factory, len(experiment.classes), experiment.options).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=experiment.learning_rate,
                                  weight_decay=experiment.weight_decay)
    if state["checkpoint"]:
        saved = torch.load(state["checkpoint"], map_location=device, weights_only=True)
        if saved.get("input_version", 1) != INPUT_VERSION:
            raise RuntimeError(f"Run '{experiment.name}' was trained on older model inputs; "
                               "start a new run with a different name.")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        print(f"Resuming '{experiment.name}' after round {state['rounds']}.", flush=True)
    for stale in run_dir.glob("round_*.pt"):  # rounds saved but never committed
        if str(stale) != state["checkpoint"]:
            stale.unlink()
    parameters = sum(p.numel() for p in model.parameters())
    print(f"Model {experiment.factory}: {parameters:,} parameters on {device}.", flush=True)

    validation = held_out_set(experiment, settings, "validation")
    plan = RoundSampler(experiment, run_id, settings)
    mlflow_id = _mlflow_run(settings, experiment, run_id, state["mlflow"])
    # `rounds` (from the command line) means "this many more"; otherwise train up to
    # experiment.rounds in total, or indefinitely when it is 0.
    last = state["rounds"] + rounds if rounds is not None else (experiment.rounds or None)
    rounds_queue: queue.Queue = queue.Queue(maxsize=1)
    stop = threading.Event()
    producer = threading.Thread(target=_produce, daemon=True, args=(
        plan, state["rounds"] + 1, last, settings, rounds_queue, stop))
    producer.start()
    best = state["best"]
    try:
        with mlflow.start_run(run_id=mlflow_id):
            while True:
                waited = time.monotonic()
                item = rounds_queue.get()
                waited = time.monotonic() - waited
                if item is _DONE:
                    break
                if isinstance(item, BaseException):
                    raise RuntimeError("Preparing training data failed") from item
                best = _finish_round(item, model, optimizer, experiment, validation, device,
                                     run_id, run_dir, best, waited)
    except KeyboardInterrupt:
        print("Stopped. Run the same command again to resume from the last saved round.", flush=True)
    finally:
        stop.set()
        producer.join(timeout=60)


def _finish_round(item: Round, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                  experiment: Experiment, validation: Prepared, device: torch.device,
                  run_id: int, run_dir: Path, best: float | None, waited: float) -> float | None:
    started = time.monotonic()
    steps = skipped = 0
    for repeat in range(experiment.passes_per_round):
        loss, done, bad = train_one_round(model, optimizer, item.data, experiment.minibatch_size, device,
                                          (experiment.seed * 100_003 + item.number) * 101 + repeat)
        steps, skipped = steps + done, skipped + bad
    report = evaluate_curves(model, validation, experiment.classes, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    gpu_seconds = time.monotonic() - started

    path = run_dir / f"round_{item.number:05d}.pt"
    _save(path, {"experiment": experiment.to_dict(), "round": item.number,
                 "input_version": INPUT_VERSION,
                 "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "validation": report.metrics("validation")})
    improved = best is None or report.balanced_accuracy > best
    if improved:
        best = report.balanced_accuracy
        shutil.copyfile(path, run_dir / "best.tmp")
        (run_dir / "best.tmp").replace(run_dir / "best.pt")
    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO training_examples (run_id, round, data_uri, target, injected) "
                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                [(run_id, item.number, c.data_uri, c.target, c.inject) for c in item.data.curves])
        conn.execute(
            """UPDATE training_runs SET rounds_completed = %s, checkpoint_path = %s,
                   best_checkpoint_path = %s, best_validation_score = %s, updated_at = NOW()
               WHERE id = %s""",
            (item.number, str(path), str(run_dir / "best.pt"), best, run_id))
    for old in run_dir.glob("round_*.pt"):
        if old != path:
            old.unlink()
    for streamed in item.streamed:
        streamed.unlink(missing_ok=True)

    injected = sum(c.inject for c in item.data.curves)
    mlflow.log_metrics({"train_loss": loss, "optimizer_steps": steps, "skipped_minibatches": skipped,
                        "curves_trained": len(item.data.curves), "injected_signals": injected,
                        "excluded_curves": item.data.excluded, "failed_downloads": item.data.failed,
                        "signals_without_transit": item.data.no_transit, "gpu_seconds": gpu_seconds,
                        "waiting_for_data_seconds": waited, **report.metrics("validation")},
                       step=item.number)
    print(f"Round {item.number}: {len(item.data.curves)} curves ({injected} injected), {steps} steps, "
          f"train loss {loss:.3f} | "
          f"validation balanced acc {report.balanced_accuracy:.3f} "
          f"(chance {report.chance:.3f}), macro F1 {report.macro_f1:.3f}"
          f"{' *best*' if improved else ''} | GPU {gpu_seconds:.0f}s, waited {waited:.0f}s for data",
          flush=True)
    return best


def status(name: str) -> None:
    ensure_schema()
    with connect() as conn:
        row = conn.execute("SELECT id, rounds_completed, best_validation_score, best_checkpoint_path, "
                           "updated_at, config->>'target' FROM training_runs WHERE name = %s",
                           (name,)).fetchone()
        if row is None:
            raise SystemExit(f"No run named '{name}'")
        used = conn.execute("SELECT COUNT(*), COUNT(DISTINCT data_uri) FROM training_examples "
                            "WHERE run_id = %s", (row[0],)).fetchone()
        excluded = conn.execute("SELECT COUNT(*) FROM excluded_curves").fetchone()[0]
    print(f"Run '{name}' (target: {row[5]}), last saved {row[4]:%Y-%m-%d %H:%M}")
    print(f"  rounds completed: {row[1]}")
    print(f"  examples trained: {used[0]:,} ({used[1]:,} distinct curves)")
    best = "none yet" if row[2] is None else f"{row[2]:.3f}"
    print(f"  best validation balanced accuracy: {best}")
    print(f"  best checkpoint: {row[3]}")
    print(f"  curves excluded as unusable (all runs): {excluded:,}")
