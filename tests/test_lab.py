"""Fast checks (CPU only, no database or network): uv run python -m unittest discover tests"""
import contextlib
import io
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import requests
import torch
from astropy.io import fits
from torch.utils.data import DataLoader

from astro_data_lab import LightCurveDataset, LightCurveUnavailable, pad_collate, read_lightcurve, report, split_of
from astro_data_lab import lightcurves
from astro_data_lab.cli import main
from astro_data_lab.sync import sync

URI = "mast:TESS/product/tess2019357164649-s0020-0000000367103210-0165-s_lc.fits"


def fits_bytes(readings: int = 500, tic: int = 367103210, sector: int = 20) -> bytes:
    columns = [
        fits.Column(name="TIME", format="D", array=np.arange(readings) / 720.0),
        fits.Column(name="PDCSAP_FLUX", format="E", array=np.full(readings, 1000.0)),
        fits.Column(name="PDCSAP_FLUX_ERR", format="E", array=np.ones(readings)),
        fits.Column(name="SAP_FLUX", format="E", array=np.full(readings, 990.0)),
        fits.Column(name="QUALITY", format="J", array=np.zeros(readings, dtype=np.int32)),
    ]
    header = fits.Header({"TICID": tic, "SECTOR": sector})
    buffer = io.BytesIO()
    fits.HDUList([fits.PrimaryHDU(header=header),
                  fits.BinTableHDU.from_columns(columns, name="LIGHTCURVE")]).writeto(buffer)
    return buffer.getvalue()


class FakeResponse:
    def __init__(self, content: bytes | None):
        self.content = content

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def raise_for_status(self):
        if self.content is None:
            raise requests.HTTPError("503")

    def iter_content(self, _):
        yield self.content


class CachedDownloads(unittest.TestCase):
    """read_lightcurve against a temporary cache and a fake network."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        patches = [mock.patch.dict(os.environ, {"ASTRO_DATA_DIR": self.directory.name, "FREE_SPACE_FLOOR_GIB": "0"}),
                   mock.patch.object(lightcurves, "_session", None),
                   mock.patch.object(lightcurves.clock, "sleep")]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(self.directory.cleanup)

    def network(self, respond):
        session = mock.Mock()
        session.get.side_effect = lambda url, **_: FakeResponse(respond(url))
        return mock.patch.object(lightcurves.requests, "Session", return_value=session), session

    def test_returns_raw_arrays_and_caches(self):
        patch, session = self.network(lambda url: fits_bytes())
        with patch:
            lc = read_lightcurve(URI)
            again = read_lightcurve(URI)
        self.assertEqual((lc.tic_id, lc.sector, len(lc)), (367103210, 20, 500))
        self.assertEqual(lc.flux.dtype, np.float32)
        self.assertTrue(np.array_equal(lc.time, again.time))
        self.assertEqual(session.get.call_count, 1)  # second read came from the cache

    def test_falls_back_from_s3_to_mast(self):
        patch, session = self.network(lambda url: fits_bytes() if "mast.stsci.edu" in url else None)
        with patch:
            read_lightcurve(URI)
        urls = [call.args[0] for call in session.get.call_args_list]
        self.assertIn("stpubdata", urls[0])
        self.assertIn("mast.stsci.edu", urls[1])

    def test_unreadable_files_raise_lightcurve_unavailable(self):
        patch, _ = self.network(lambda url: b"not a fits file")
        with patch, self.assertRaises(LightCurveUnavailable):
            read_lightcurve(URI)

    def test_dataset_skips_unavailable_curves(self):
        import pandas as pd

        good = "mast:TESS/product/tess2019357164649-s0020-0000000000000001-0165-s_lc.fits"
        patch, _ = self.network(lambda url: fits_bytes(300) if "0000000000000001" in url else None)
        table = pd.DataFrame({"data_uri": [good, URI], "label": [1, 0]})
        dataset = LightCurveDataset(table, lambda lc: lc.flux / 1000 - 1)
        with patch:
            x, mask, y = next(iter(DataLoader(dataset, batch_size=2, collate_fn=pad_collate)))
        self.assertEqual(tuple(x.shape), (1, 1, 300))
        self.assertEqual(y.tolist(), [1])
        self.assertEqual(dataset.skipped, 1)

    def test_cache_deletes_least_recently_used_beyond_its_limit(self):
        folder = lightcurves.cache_dir()
        folder.mkdir(parents=True)
        for i in range(10):
            path = folder / f"{i}.npz"
            path.write_bytes(b"x" * 1000)
            os.utime(path, (i, i))  # file i was last used at time i
        with mock.patch.dict(os.environ, {"LIGHTCURVE_CACHE_GIB": str(5000 / lightcurves.GIB)}):
            lightcurves._evict_least_recently_used()
        self.assertEqual(sorted(p.stem for p in folder.glob("*.npz")), ["6", "7", "8", "9"])


class Functions(unittest.TestCase):
    def test_split_is_fixed_per_star_and_roughly_80_10_10(self):
        splits = [split_of(tic) for tic in range(20_000)]
        self.assertEqual(split_of(12345), split_of(12345))
        for name, share in (("train", 0.8), ("validation", 0.1), ("test", 0.1)):
            self.assertAlmostEqual(splits.count(name) / len(splits), share, delta=0.01)

    def test_s3_url_is_built_from_the_file_name(self):
        (s3, _), (mast, params) = lightcurves.download_urls(URI)
        self.assertEqual(s3, "https://stpubdata.s3.us-east-1.amazonaws.com/tess/public/tid/s0020/"
                             "0000/0003/6710/3210/tess2019357164649-s0020-0000000367103210-0165-s_lc.fits")
        self.assertEqual((mast, params), (lightcurves.MAST_URL, {"uri": URI}))

    def test_pad_collate_pads_and_masks(self):
        x, mask, y = pad_collate([(torch.ones(3), 0), None, (torch.ones(5), 1)])  # None = unavailable curve
        self.assertEqual(tuple(x.shape), (2, 1, 5))
        self.assertEqual(mask.sum(1).tolist(), [3, 5])
        self.assertEqual(x[0, 0, 3:].tolist(), [0.0, 0.0])
        self.assertEqual(y.dtype, torch.long)

    def test_report_matches_a_known_confusion_matrix(self):
        result = report([0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1], ["a", "b"])
        self.assertAlmostEqual(result.balanced_accuracy, (0.75 + 1.0) / 2)
        self.assertEqual(result.confusion.tolist(), [[3, 1], [0, 2]])
        self.assertIn("balanced accuracy", str(result))
        self.assertLess(report([0, 0, 1, 1], [0, 0, 0, 0]).ci95[0], 0.5 + 1e-9)  # always "0" is chance

    def test_help_needs_no_database(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            main([])
        self.assertIn("sync", output.getvalue())


class ScheduledSync(unittest.TestCase):
    def test_weekly_catalogue_command(self):
        with mock.patch("astro_data_lab.sync.sync") as run:
            main(["sync", "--only", "catalogues"])
        run.assert_called_once_with(catalogues=True, manifest=False, max_sectors=2, refresh_one=False)

    def test_weekly_manifest_command(self):
        with mock.patch("astro_data_lab.sync.sync") as run:
            main(["sync", "--only", "manifest", "--max-sectors", "2", "--refresh-one"])
        run.assert_called_once_with(catalogues=False, manifest=True, max_sectors=2, refresh_one=True)

    def test_refresh_requires_manifest(self):
        with mock.patch("astro_data_lab.sync.sync") as run, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main(["sync", "--only", "catalogues", "--refresh-one"])
        self.assertEqual(error.exception.code, 2)
        run.assert_not_called()

    def test_catalogue_sync_does_not_run_manifest(self):
        with mock.patch("astro_data_lab.sync.ensure_schema") as schema, \
             mock.patch("astro_data_lab.sync.catalogues.synchronise_catalogues") as catalogues, \
             mock.patch("astro_data_lab.sync.manifest.sync_incremental") as manifest:
            sync(catalogues=True, manifest=False)
        schema.assert_called_once_with()
        catalogues.assert_called_once_with()
        manifest.assert_not_called()

    def test_manifest_sync_forwards_refresh_under_lock(self):
        from pathlib import Path

        with mock.patch.dict(os.environ, {"MAST_INVENTORY_DIR": "/tmp/test-inventory"}), \
             mock.patch("astro_data_lab.sync.ensure_schema"), \
             mock.patch("astro_data_lab.sync.catalogues.synchronise_catalogues") as catalogues, \
             mock.patch("astro_data_lab.sync.manifest.inventory_lock") as lock, \
             mock.patch("astro_data_lab.sync.manifest.sync_incremental") as manifest:
            sync(catalogues=False, manifest=True, max_sectors=2, refresh_one=True)
        catalogues.assert_not_called()
        lock.assert_called_once_with(Path("/tmp/test-inventory"))
        lock.return_value.__enter__.assert_called_once()
        manifest.assert_called_once_with(Path("/tmp/test-inventory"), 1, 2, refresh_one=True)

    def test_refreshes_existing_sector_even_if_no_new_data(self):
        from pathlib import Path
        from astro_data_lab.sync.manifest import sync_incremental

        with mock.patch("astro_data_lab.sync.manifest.completed_sectors", return_value={1, 2}), \
             mock.patch("astro_data_lab.sync.manifest.query_sector_observations", return_value=[]) as query, \
             mock.patch("astro_data_lab.sync.manifest.oldest_completed_sector", return_value=1), \
             mock.patch("astro_data_lab.sync.manifest.sync_sector") as sector, \
             contextlib.redirect_stdout(io.StringIO()):
            sync_incremental(Path("/tmp/test-inventory"), 1, 2, refresh_one=True)
        query.assert_called_once_with(3)
        sector.assert_called_once_with(1, Path("/tmp/test-inventory"), 1, 1)


if __name__ == "__main__":
    unittest.main()
