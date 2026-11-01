"""Fast checks (CPU only, no database or network) for the parts that must not break."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from astropy.io import fits

from astro_data_lab.cli import main
from astro_data_lab.config import Experiment
from astro_data_lab.data.labels import Curve, balanced_subset, split_of
from astro_data_lab.data.lightcurves import (
    MIN_TRANSIT_READINGS, Prepared, UnusableLightCurve, check_quality, clean_fits, collate,
    detrend, inject_transit, model_input, point_noise, transit_readings,
)
from astro_data_lab.models import LightCurveCNN, PatchTransformer, build_model, check_model
from astro_data_lab.training.evaluation import score
from astro_data_lab.training.trainer import RoundSampler, train_one_round

ROOT = Path(__file__).resolve().parents[1]


def write_fits(path: Path, flux: np.ndarray, quality: int = 0) -> None:
    columns = [
        fits.Column(name="TIME", format="D", array=np.arange(len(flux)) / 720),
        fits.Column(name="PDCSAP_FLUX", format="D", array=flux),
        fits.Column(name="QUALITY", format="J", array=np.full(len(flux), quality)),
    ]
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns, name="LIGHTCURVE")]
                 ).writeto(path, overwrite=True)


def synthetic_curve(transit: bool, rng: np.random.Generator, length: int = 2000) -> torch.Tensor:
    time = np.arange(length) / 720
    flux = 1 + rng.normal(0, 1e-3, length)
    if transit:
        start = rng.integers(0, length - 60)
        flux[start:start + 60] -= 0.01
    return model_input(time, flux)


class LabTests(unittest.TestCase):
    def test_example_configs_load_build_and_typos_are_rejected(self):
        for path in sorted((ROOT / "experiments").glob("*.toml")):
            with self.subTest(path=path.name):
                experiment = Experiment.load(path)
                self.assertEqual(experiment.classes, ("no_signal", "signal"))
                check_model(build_model(experiment.factory, 2, experiment.options), 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.toml"
            for text in ('[run]\nname="x"\nlearning_rat=1', '[run]\nname="x"\nrounds="5"',
                         '[run]\nname="x"\ntarget="planets"', '[oops]\nname="x"',
                         '[run]\nname="x"\ninjected_fraction=1.0'):
                path.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValueError):
                    Experiment.load(path)

    def test_bad_curves_are_reported_as_unusable_not_crashes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "curve.fits"
            for flux, quality, reason in ((np.full(500, -5.0), 0, "median"),
                                          (np.ones(500), 1, "valid readings"),
                                          (np.ones(50), 0, "valid readings")):
                write_fits(path, flux, quality)
                with self.subTest(reason=reason), self.assertRaisesRegex(UnusableLightCurve, reason):
                    clean_fits(path)
            path.write_bytes(b"not a fits file")
            with self.assertRaisesRegex(UnusableLightCurve, "unreadable"):
                clean_fits(path)
            write_fits(path, 1000 + np.random.default_rng(0).normal(0, 1, 500))
            self.assertEqual(len(clean_fits(path).flux), 500)

    def test_extreme_values_are_clipped_to_safe_inputs(self):
        flux = np.ones(500)
        flux[10] = 1e9
        inputs = model_input(np.arange(500.0), flux)
        self.assertEqual(inputs.shape, (4, 500))
        self.assertTrue(torch.isfinite(inputs).all())
        self.assertLessEqual(inputs.abs().max().item(), 100)

    def test_too_many_bad_curves_stops_but_a_few_are_tolerated(self):
        good = Prepared([Curve("a", 1, 0)] * 9, [], excluded=1, failed=0)
        check_quality(good, 10, 0.2, "round")
        # signals whose transit misses the curve are a labelling outcome, not an outage
        check_quality(Prepared([Curve("a", 1, 0)] * 5, [], 0, 0, no_transit=5), 10, 0.2, "round")
        with self.assertRaisesRegex(RuntimeError, "max_bad_fraction"):
            check_quality(Prepared([Curve("a", 1, 0)] * 5, [], 5, 0), 10, 0.2, "round")

    def test_transit_check_counts_only_readings_inside_predicted_transits(self):
        time = np.arange(0, 27, 2 / 1440)                         # 27 days, 2-minute cadence
        hour, minute = 1 / 24, 1 / 1440                            # epochs fall between readings
        self.assertEqual(transit_readings(time, ((5 + minute, 0.0, 2 * hour),)), 60)   # one 2-h transit
        self.assertEqual(transit_readings(time, ((5 + minute, 10.0, 2 * hour),)), 180) # days 5, 15, 25
        self.assertEqual(transit_readings(time, ((-95 + minute, 10.0, 2 * hour),)), 180)  # old epoch
        self.assertEqual(transit_readings(time, ((40.0, 0.0, 2 * hour),)), 0)          # other sector
        gap = time[(time < 4.9) | (time > 5.1)]                    # transit fell in a data gap
        self.assertEqual(transit_readings(gap, ((5 + minute, 0.0, 2 * hour),)), 0)

    def test_detrending_removes_slow_variability_but_keeps_a_transit(self):
        time = np.arange(0, 27, 2 / 1440)
        rng = np.random.default_rng(0)
        flux = 1 + 0.01 * np.sin(2 * np.pi * time / 5) + rng.normal(0, 2e-4, len(time))
        dip = (time > 10) & (time < 10 + 3 / 24)                   # 3-hour, 0.2% transit
        flux[dip] -= 0.002
        detrended = detrend(time, flux)
        self.assertLess(np.std(detrended[~dip]), 1e-3)             # the 1% wave is gone
        self.assertLess(np.median(detrended[dip]), -0.0014)        # >= 70% of the transit depth kept
        self.assertLess(np.median(model_input(time, flux)[1, dip]), -5)  # many sigma in channel 1

    def test_injected_transits_are_visible_realistic_and_reproducible(self):
        time = np.arange(0, 27, 2 / 1440)
        flux = 1 + np.random.default_rng(1).normal(0, 1e-3, len(time))
        for seed in range(20):
            new_flux, (epoch, period, duration) = inject_transit(time, flux, seed)
            changed = new_flux < flux
            self.assertGreaterEqual(changed.sum(), MIN_TRANSIT_READINGS)
            self.assertGreaterEqual(transit_readings(time, ((epoch, period, duration),)), changed.sum())
            depth = (1 - new_flux / flux).max()
            snr = depth / point_noise(flux) * np.sqrt(changed.sum())
            self.assertTrue(0 < depth <= 0.05 and 4 < snr < 60, (depth, snr))
            self.assertTrue(np.array_equal(new_flux, inject_transit(time, flux, seed)[0]))
            self.assertTrue(np.array_equal(new_flux[~changed], flux[~changed]))

    def test_balanced_subset_keeps_order_and_caps_each_class(self):
        curves = [Curve(str(i), 1, t) for i, t in enumerate([1, 0, 1, 1, 0, 0, 0])]
        self.assertEqual([c.data_uri for c in balanced_subset(curves, 2)], ["0", "1", "2", "4"])

    def test_split_is_by_star_and_roughly_80_10_10(self):
        splits = [split_of(tic) for tic in range(20_000)]
        self.assertEqual(split_of(12345), split_of(12345))
        for name, share in (("train", 0.8), ("validation", 0.1), ("test", 0.1)):
            self.assertAlmostEqual(splits.count(name) / len(splits), share, delta=0.01)

    def test_model_checker_accepts_the_examples_and_rejects_padding_leaks(self):
        check_model(LightCurveCNN(num_classes=4), 4)
        check_model(PatchTransformer(num_classes=2), 2)

        class LeaksPadding(torch.nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.head = torch.nn.Linear(3, num_classes)

            def forward(self, inputs):
                return self.head(inputs[:, :3].mean(-1))  # averages padding zeros too

        with self.assertRaisesRegex(ValueError, "padding"):
            check_model(LeaksPadding(4), 4)

    def test_padding_never_changes_a_prediction(self):
        torch.manual_seed(0)
        model = LightCurveCNN(num_classes=4).eval()
        rng = np.random.default_rng(1)
        short, long = synthetic_curve(True, rng, 1237), synthetic_curve(False, rng, 3001)
        with torch.no_grad():
            alone = model(short.unsqueeze(0))
            padded = model(collate([short, long]))[:1]
        self.assertTrue(torch.allclose(alone, padded, atol=1e-5))

    def test_rounds_are_balanced_and_injected_signals_come_from_fresh_no_signal_curves(self):
        plan = RoundSampler.__new__(RoundSampler)
        plan.experiment = Experiment(name="t", curves_per_round=100, injected_fraction=0.4)
        plan.run_id, plan.streamed, plan.in_flight = 1, {0}, set()
        plan.pools = {1: [Curve(f"real-{n}", 1, 1, ((1.0, 2.0, 0.1),)) for n in range(30)]}
        fresh = iter(range(10_000))

        def unseen(_conn, _target, label, _run_id, count, _seed, _skip):
            return [Curve(f"fresh-{next(fresh)}", 1, label) for _ in range(count)]

        with mock.patch("astro_data_lab.training.trainer.connect"), \
                mock.patch("astro_data_lab.data.labels.unseen_training_curves", unseen):
            chosen, streamed = plan.select(1)
            second, _ = plan.select(2)
        signals = [c for c in chosen if c.target == 1]
        no_signals = {c.data_uri for c in chosen if c.target == 0}
        self.assertEqual(len(signals), 50)
        self.assertEqual(len(no_signals), 50)
        self.assertEqual(sum(c.inject for c in signals), 20)                    # 40% of 50
        # injected signals reuse this round's fresh no_signal curves: no extra downloads
        self.assertTrue({c.data_uri for c in signals if c.inject} <= no_signals)
        self.assertEqual({c.data_uri for c in streamed}, no_signals)            # deleted after the round
        self.assertFalse({c.data_uri for c in streamed} & {c.data_uri for c in second})  # never reused

    def test_example_model_learns_a_transit_from_noise(self):
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        targets = [i % 2 for i in range(256)]
        data = Prepared([Curve(str(i), 1, t) for i, t in enumerate(targets)],
                        [synthetic_curve(bool(t), rng) for t in targets], 0, 0)
        model = LightCurveCNN(num_classes=2, width=16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        losses = [train_one_round(model, optimizer, data, 32, torch.device("cpu"), seed)[0]
                  for seed in range(6)]
        held_out = [synthetic_curve(bool(i % 2), rng) for i in range(100)]
        with torch.no_grad():
            probabilities = model.eval()(collate(held_out)).softmax(-1).numpy()
        report = score(np.array([i % 2 for i in range(100)]), probabilities, ("none", "transit"))
        self.assertLess(losses[-1], losses[0])
        self.assertTrue(report.better_than_chance, report.text())

    def test_metrics_match_a_known_confusion_matrix(self):
        targets = np.array([0, 0, 0, 0, 1, 1])
        predictions = np.eye(2)[[0, 0, 0, 1, 1, 1]]
        report = score(targets, predictions, ("a", "b"))
        self.assertAlmostEqual(report.balanced_accuracy, (0.75 + 1.0) / 2)
        self.assertEqual(report.confusion.tolist(), [[3, 1], [0, 2]])
        self.assertFalse(score(targets, np.tile([1.0, 0.0], (6, 1)), ("a", "b")).better_than_chance)

    def test_downloads_try_the_public_s3_copy_before_mast(self):
        from astro_data_lab.data.lightcurves import DOWNLOAD_URL, download_urls

        uri = "mast:TESS/product/tess2019357164649-s0020-0000000367103210-0165-s_lc.fits"
        (s3, _), (mast, params) = download_urls(uri)
        self.assertEqual(s3, "https://stpubdata.s3.us-east-1.amazonaws.com/tess/public/tid/s0020/"
                             "0000/0003/6710/3210/tess2019357164649-s0020-0000000367103210-0165-s_lc.fits")
        self.assertEqual((mast, params), (DOWNLOAD_URL, {"uri": uri}))
        self.assertEqual(download_urls("mast:TESS/product/unexpected.fits"),
                         [(DOWNLOAD_URL, {"uri": "mast:TESS/product/unexpected.fits"})])

    def test_help_needs_no_database_or_gpu(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main([])
        for command in ("sync", "train", "evaluate", "predict"):
            self.assertIn(command, output.getvalue())


if __name__ == "__main__":
    unittest.main()
