"""Regenerate every figure and number in docs/paper/paper.md from the trained runs.

    uv run --env-file .env python docs/paper/make_figures.py [--device cpu|cuda]

Writes docs/paper/figures/*.png and docs/paper/results.json. Runs that have not
finished yet are skipped and listed as pending. Uses only platform functions, so
the numbers are exactly what the pipeline produces.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from astro_data_lab.config import Experiment, Settings  # noqa: E402
from astro_data_lab.data import labels  # noqa: E402
from astro_data_lab.data.lightcurves import (  # noqa: E402
    curve_arrays, detrend, inject_transit, injection_seed, model_input, point_noise, transit_centres,
    transit_readings,
)
from astro_data_lab.database import connect  # noqa: E402
from astro_data_lab.predict import load_trained  # noqa: E402
from astro_data_lab.training.evaluation import held_out_set, predict_probabilities, score  # noqa: E402
from astro_data_lab.training.trainer import history  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
CLASSIFIERS = {  # run name -> label in figures
    "transit-cnn-v1": "CNN (147k)",
    "transit-transformer-v1": "Transformer (344k)",
    "transit-transformer-large-v1": "Transformer, scratch (15M)",
    "transit-foundation-v1": "Transformer, pre-trained (15M)",
}
COMPARISONS = (("transit-cnn-v1", "transit-transformer-v1"),
               ("transit-foundation-v1", "transit-transformer-large-v1"))  # (A, B): is A better than B?
SNR_GRID = (3, 5, 7, 10, 15, 20, 30, 50)
SNR_BINS = (0, 7, 12, 20, 50, np.inf)


def paired_difference(targets: np.ndarray, a: np.ndarray, b: np.ndarray, resamples: int = 2000,
                      seed: int = 0) -> dict:
    """Balanced accuracy of A minus B on the same curves, with a paired bootstrap 95% interval."""
    def balanced(pick: np.ndarray, predicted: np.ndarray) -> float:
        t, p = targets[pick], predicted[pick]
        return float(np.mean([np.mean(p[t == k] == k) for k in np.unique(t)]))

    everything = np.arange(len(targets))
    rng = np.random.default_rng(seed)
    draws = [balanced(pick, a) - balanced(pick, b)
             for pick in (rng.integers(0, len(targets), len(targets)) for _ in range(resamples))]
    return {"difference": balanced(everything, a) - balanced(everything, b),
            "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]}


def roc_auc(targets: np.ndarray, signal_score: np.ndarray) -> float:
    """Probability that a random signal curve scores above a random no_signal curve (ties count half)."""
    positive, negative = signal_score[targets == 1], signal_score[targets == 0]
    greater = (positive[:, None] > negative[None, :]).mean()
    ties = (positive[:, None] == negative[None, :]).mean()
    return float(greater + 0.5 * ties)


def best_checkpoint(name: str) -> Path | None:
    with connect() as conn:
        row = conn.execute("SELECT best_checkpoint_path, config->>'rounds', rounds_completed "
                           "FROM training_runs WHERE name = %s", (name,)).fetchone()
    if row is None or not row[0] or int(row[2]) < int(row[1] or 0):
        return None  # missing or still training
    return Path(row[0])


def star_tois(tic_ids: list[int]) -> dict[int, list[tuple]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT tic_id, disposition, transit_epoch_bjd - 2457000, COALESCE(period_days, 0), "
            "duration_hours / 24, depth_ppm FROM tess_objects_of_interest WHERE tic_id = ANY(%s)",
            (tic_ids,)).fetchall()
    tois: dict[int, list[tuple]] = {}
    for tic, disposition, epoch, period, duration, depth in rows:
        tois.setdefault(int(tic), []).append((disposition, epoch, period, duration, depth))
    return tois


def describe_signals(test, settings) -> list[dict]:
    """Disposition group and expected transit SNR of every real signal curve in the test set."""
    with connect() as conn:
        tic_of = dict(conn.execute("SELECT data_uri, tic_id FROM mast_products WHERE data_uri = ANY(%s)",
                                   ([c.data_uri for c in test.curves],)).fetchall())
    tois = star_tois(sorted({int(t) for t in tic_of.values()}))
    rows = []
    for index, curve in enumerate(test.curves):
        if curve.target != labels.SIGNAL:
            continue
        time, flux = curve_arrays(curve, settings)
        sigma = point_noise(detrend(time, flux))
        snr = 0.0
        found = tois.get(int(tic_of[curve.data_uri]), [])
        for _, epoch, period, duration, depth in found:
            if epoch is None or not duration or not depth:
                continue
            n = transit_readings(time, ((epoch, period, duration),))
            snr = max(snr, depth * 1e-6 / sigma * np.sqrt(n))
        dispositions = {d for d, *_ in found}
        group = ("confirmed/known planet" if dispositions & {"CP", "KP"} else
                 "planet candidate" if dispositions & {"PC", "APC"} else "false positive")
        rows.append({"index": index, "group": group, "snr": float(snr)})
    return rows


def injection_recovery(model, test, settings, device) -> dict[float, float]:
    """Share of quiet test curves called 'signal' after injecting a transit of fixed SNR."""
    quiet = [c for c in test.curves if c.target == labels.NO_SIGNAL]
    original = [model_input(*curve_arrays(c, settings)) for c in quiet]
    rates = {0.0: float((predict_probabilities(model, original, device).argmax(1) == labels.SIGNAL).mean())}
    for snr in SNR_GRID:
        tensors = []
        for c in quiet:
            time, flux = curve_arrays(c, settings)
            injected = inject_transit(time, flux, injection_seed(c.data_uri), snr=snr)
            if injected is not None:
                tensors.append(model_input(time, injected[0]))
        rates[float(snr)] = float((predict_probabilities(model, tensors, device).argmax(1) == labels.SIGNAL).mean())
    return rates


def figure_examples(test, settings) -> None:
    signal = next(c for c in test.curves if c.target == labels.SIGNAL and c.ephemerides
                  and len(transit_centres(curve_arrays(c, settings)[0], c.ephemerides)) >= 2)
    quiet = next(c for c in test.curves if c.target == labels.NO_SIGNAL)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5))
    t, f = curve_arrays(signal, settings)
    axes[0].plot(t, detrend(t, f) * 100, ",k")
    duration = signal.ephemerides[0][2]
    for centre in transit_centres(t, signal.ephemerides):
        axes[0].axvspan(centre - duration / 2, centre + duration / 2, color="tab:red", alpha=0.3)
    axes[0].set_title("(a) signal: a TESS Object of Interest; red = predicted transits", loc="left", fontsize=10)
    t, f = curve_arrays(quiet, settings)
    axes[1].plot(t, detrend(t, f) * 100, ",k")
    axes[1].set_title("(b) no_signal: a star in neither planet catalogue", loc="left", fontsize=10)
    injected, (epoch, period, duration) = inject_transit(t, f, injection_seed(quiet.data_uri), snr=15)
    axes[2].plot(t, detrend(t, injected) * 100, ",k")
    for centre in transit_centres(t, ((epoch, period, duration),)):
        axes[2].axvspan(centre - duration / 2, centre + duration / 2, color="tab:blue", alpha=0.3)
    axes[2].set_title(f"(c) the same star as (b) with a synthetic transit, SNR 15 "
                      f"(P = {period:.2f} d, duration {duration * 24:.1f} h)", loc="left", fontsize=10)
    for ax in axes:
        ax.set_ylabel("detrended flux (%)")
    axes[2].set_xlabel("time (BTJD days)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_examples.png", dpi=150)
    plt.close(fig)


def figure_learning_curves(settings, names: list[str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for name in names:
        try:
            frame = history(name, settings)
        except ValueError:
            continue
        axes[0].plot(frame.index, frame["train_loss"], label=CLASSIFIERS[name])
        axes[1].plot(frame.index, frame["validation_balanced_accuracy"].rolling(5, min_periods=1).mean(),
                     label=CLASSIFIERS[name])
    axes[0].set(title="(a) training loss (cross-entropy)", xlabel="round")
    axes[1].axhline(0.5, ls="--", c="grey", label="chance")
    axes[1].set(title="(b) validation balanced accuracy (5-round mean)", xlabel="round")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_learning_curves.png", dpi=150)
    plt.close(fig)


def figure_sensitivity(results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for name, result in results.items():
        rates = result["injection_recovery"]
        snrs = sorted(float(k) for k in rates if float(k) > 0)
        line, = axes[0].plot(snrs, [rates[str(s)] for s in snrs], marker="o", label=CLASSIFIERS[name])
        axes[0].axhline(rates["0.0"], ls=":", color=line.get_color(), lw=1)
        centres, recall = [], []
        for low, high in zip(SNR_BINS[:-1], SNR_BINS[1:]):
            hits = [s["correct"] for s in result["signals"] if low <= s["snr"] < high]
            if hits:
                centres.append(f"{low:g}-{high:g}" if np.isfinite(high) else f">{low:g}")
                recall.append(np.mean(hits))
        axes[1].plot(centres, recall, marker="o", label=CLASSIFIERS[name])
    axes[0].set(xscale="log", xlabel="injected transit SNR", ylabel="share flagged as signal", ylim=(0, 1.02),
                title="(a) injection-recovery (dotted: false alarms)")
    axes[0].set_xticks(SNR_GRID, [str(s) for s in SNR_GRID])
    axes[0].minorticks_off()
    axes[1].set(xlabel="expected SNR of the catalogued transit(s)", ylabel="recall", ylim=(0, 1.02),
                title="(b) real test signals by expected SNR")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_sensitivity.png", dpi=150)
    plt.close(fig)


def figure_pretraining(settings) -> dict | None:
    try:
        frame = history("foundation-v1", settings)
    except ValueError:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].plot(frame.index, frame["validation_reconstruction_skill"], marker=".")
    axes[0].axhline(0, ls="--", c="grey")
    axes[0].set(title="(a) reconstruction skill vs. interpolation", xlabel="round")
    axes[1].plot(frame.index, frame["validation_probe_balanced_accuracy"], marker=".")
    axes[1].axhline(0.5, ls="--", c="grey")
    axes[1].set(title="(b) linear probe: transit vs. no transit", xlabel="round", ylabel="balanced accuracy")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_pretraining.png", dpi=150)
    plt.close(fig)
    last = frame.dropna(subset=["validation_probe_balanced_accuracy"]).iloc[-1]
    return {"rounds_logged": int(frame.index.max()),
            "untrained": {"skill": float(frame.loc[0, "validation_reconstruction_skill"]),
                          "probe": float(frame.loc[0, "validation_probe_balanced_accuracy"])},
            "latest": {"round": int(last.name), "skill": float(last["validation_reconstruction_skill"]),
                       "probe": float(last["validation_probe_balanced_accuracy"]),
                       "probe_low": float(last["validation_probe_balanced_accuracy_low"])}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))  # leave CPU for a running trainer
    device = torch.device(args.device)
    settings = Settings.from_env()
    FIGURES.mkdir(parents=True, exist_ok=True)

    test = held_out_set(Experiment(name="paper", target="transit", validation_curves_per_class=300),
                        settings, "test")
    targets = np.array([c.target for c in test.curves])
    with connect() as conn:
        counts = labels.class_counts(conn, "transit", "train")
    prevalence = counts[labels.SIGNAL] / (counts[labels.SIGNAL] + counts[labels.NO_SIGNAL])
    signals = describe_signals(test, settings)
    results, pending, predictions = {}, [], {}
    for name in CLASSIFIERS:
        path = best_checkpoint(name)
        if path is None:
            pending.append(name)
            continue
        experiment, model, round_number = load_trained(path, device)
        probabilities = predict_probabilities(model, test.tensors, device)
        report = score(targets, probabilities, experiment.classes)
        predicted = probabilities.argmax(1)
        predictions[name] = predicted
        results[name] = {
            "best_round": round_number,
            "parameters": sum(p.numel() for p in model.parameters()),
            "balanced_accuracy": report.balanced_accuracy, "ci95": list(report.balanced_accuracy_95),
            "recall": dict(zip(experiment.classes, report.recall)),
            "precision": dict(zip(experiment.classes, report.precision)),
            "confusion": report.confusion.tolist(),
            "roc_auc": roc_auc(targets, probabilities[:, labels.SIGNAL]),
            # precision if signal curves were as rare as in the survey, from test recall and false-alarm rate
            "precision_at_survey_prevalence": float(
                report.recall[labels.SIGNAL] * prevalence
                / (report.recall[labels.SIGNAL] * prevalence + (1 - report.recall[labels.NO_SIGNAL]) * (1 - prevalence))),
            "signals": [{**s, "correct": bool(predicted[s["index"]] == labels.SIGNAL)} for s in signals],
            "injection_recovery": {str(k): v for k, v in injection_recovery(model, test, settings, device).items()},
        }
        groups: dict[str, list[bool]] = {}
        for s in results[name]["signals"]:
            groups.setdefault(s["group"], []).append(s["correct"])
        results[name]["recall_by_group"] = {g: [float(np.mean(v)), len(v)] for g, v in groups.items()}
        print(f"{name}: balanced accuracy {report.balanced_accuracy:.3f}", flush=True)

    comparisons = {f"{a} minus {b}": paired_difference(targets, predictions[a], predictions[b])
                   for a, b in COMPARISONS if a in predictions and b in predictions}
    for label, result in comparisons.items():
        print(f"{label}: {result['difference']:+.3f} (95% CI {result['ci95'][0]:+.3f} to {result['ci95'][1]:+.3f})")
    groups = {}
    for s in signals:
        groups.setdefault(s["group"], []).append(s["snr"])
    snr_by_group = {g: {"median_snr": float(np.median(v)), "curves": len(v)} for g, v in groups.items()}

    figure_examples(test, settings)
    figure_learning_curves(settings, list(results))
    figure_sensitivity(results)
    pretraining = figure_pretraining(settings)
    with connect() as conn:
        data = {"curves": conn.execute("SELECT COUNT(*), COUNT(DISTINCT tic_id), MIN(sector), MAX(sector) "
                                       "FROM mast_products").fetchone(),
                "summary": labels.summary(conn).to_dict()}
    (HERE / "results.json").write_text(json.dumps(
        {"test_set": {"curves": len(test.curves), "signal": int((targets == 1).sum()),
                      "no_signal": int((targets == 0).sum())},
         "data": data, "survey_signal_prevalence": prevalence, "classifiers": results, "comparisons": comparisons, "snr_by_group": snr_by_group,
         "pending": pending, "pretraining": pretraining},
        indent=1, default=str))
    print("pending (not finished yet):", pending or "none")


if __name__ == "__main__":
    main()
