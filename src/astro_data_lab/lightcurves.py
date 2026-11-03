"""Step 3: get one light curve as plain arrays.

    read_lightcurve(data_uri)  ->  LightCurve(time, flux, flux_err, sap_flux, quality, ...)

The arrays are exactly what is in the TESS file: nothing is removed, normalised or
filtered (NaNs and flagged readings included). Cleaning is to be trialled in a
`transform` (step 5), so we can tweak what is done.

Downloads are cached on disk under ASTRO_DATA_DIR/lightcurves, ~0.5 MB per curve.
When the cache grows past LIGHTCURVE_CACHE_GIB (default 50), the least recently
used curves are deleted automatically, so it never fills the disk. Nothing ever
needs cleaning up manually: anything deleted is simply downloaded again if needed.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import time as clock
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from astropy.io import fits

GIB = 1024**3
MAST_URL = "https://mast.stsci.edu/api/v0.1/Download/file"
# MAST serves TESS files from this public AWS bucket; going there directly is much faster.
S3_URL = "https://stpubdata.s3.us-east-1.amazonaws.com/tess/public/tid/s{sector}/{a}/{b}/{c}/{d}/{name}"
_S3_NAME = re.compile(r"tess\d+-s(\d{4})-(\d{16})-\d+-s_lc\.fits")
_session: requests.Session | None = None
_saved_since_check = 0


class LightCurveUnavailable(Exception):
    """The file could not be downloaded or read. `LightCurveDataset` skips these."""


@dataclass(frozen=True)
class LightCurve:
    data_uri: str
    tic_id: int
    sector: int
    time: np.ndarray      # float64, BTJD days (BJD - 2457000); NaN where TESS has no timestamp
    flux: np.ndarray      # float32, PDCSAP flux (systematics-corrected), electrons/s; may contain NaN
    flux_err: np.ndarray  # float32, uncertainty on `flux`
    sap_flux: np.ndarray  # float32, simple aperture flux (before systematics correction)
    quality: np.ndarray   # int32, TESS quality flags per reading; 0 = no known problem

    def __len__(self) -> int:
        return len(self.time)


def cache_dir() -> Path:
    return Path(os.getenv("ASTRO_DATA_DIR", "/data/astro-data-lab")) / "lightcurves"


def read_lightcurve(data_uri: str) -> LightCurve:
    """Return one light curve, downloading it first if it is not cached.

    Raises LightCurveUnavailable if it cannot be downloaded (after retries) or read.
    """
    path = cache_dir() / f"{hashlib.sha256(data_uri.encode()).hexdigest()[:24]}.npz"
    for _ in range(2):  # a second try covers a cached file deleted by another process mid-read
        if not path.exists():
            _download(data_uri, path)
        try:
            with np.load(path, allow_pickle=False) as saved:
                arrays = {key: saved[key] for key in saved.files}
            os.utime(path)  # mark as recently used, so the cache keeps it
            return LightCurve(data_uri=data_uri, tic_id=int(arrays.pop("tic_id")),
                              sector=int(arrays.pop("sector")), **arrays)
        except FileNotFoundError:
            continue
        except (OSError, ValueError, KeyError) as error:
            path.unlink(missing_ok=True)
            raise LightCurveUnavailable(f"{data_uri}: corrupt cache file ({error})") from error
    raise LightCurveUnavailable(f"{data_uri}: removed from the cache while reading")


def cache_info() -> dict:
    """How many curves are cached and how much space they use."""
    files = list(cache_dir().glob("*.npz"))
    return {"directory": str(cache_dir()), "curves": len(files),
            "size_gib": sum(f.stat().st_size for f in files) / GIB, "limit_gib": _cache_limit() / GIB}


def download_urls(data_uri: str) -> list[tuple[str, dict]]:
    """Where to fetch a light-curve FITS file from, fastest first."""
    name = data_uri.rsplit("/", 1)[-1]
    urls = [(MAST_URL, {"uri": data_uri})]
    if match := _S3_NAME.fullmatch(name):
        tic = match.group(2)
        urls.insert(0, (S3_URL.format(sector=match.group(1), a=tic[:4], b=tic[4:8], c=tic[8:12],
                                      d=tic[12:], name=name), {}))
    return urls


def _cache_limit() -> int:
    return int(float(os.getenv("LIGHTCURVE_CACHE_GIB", "50")) * GIB)


def _download(data_uri: str, path: Path) -> None:
    """Fetch the FITS file, keep only the arrays we need, and save them to `path`."""
    global _session, _saved_since_check
    path.parent.mkdir(parents=True, exist_ok=True)
    floor = float(os.getenv("FREE_SPACE_FLOOR_GIB", "330")) * GIB
    if shutil.disk_usage(path.parent).free < floor:
        raise LightCurveUnavailable(f"disk is below FREE_SPACE_FLOOR_GIB; not downloading {data_uri}")
    if _session is None:
        _session = requests.Session()  # one per process (DataLoader workers each get their own)
    raw = path.with_name(f"{path.stem}.{os.getpid()}.fits")
    sources = download_urls(data_uri)
    try:
        for attempt in range(3 * len(sources)):
            url, params = sources[attempt % len(sources)]
            try:
                with _session.get(url, params=params, stream=True, timeout=(20, 120)) as response:
                    response.raise_for_status()
                    with raw.open("wb") as stream:
                        for chunk in response.iter_content(1024 * 1024):
                            stream.write(chunk)
                arrays = _read_fits(raw)
                break
            except (requests.RequestException, OSError, KeyError, ValueError, TypeError) as error:
                if attempt == 3 * len(sources) - 1:
                    raise LightCurveUnavailable(f"{data_uri}: {type(error).__name__}: {error}") from error
                clock.sleep(attempt // len(sources) + 1)
        partial = path.with_name(f"{path.stem}.{os.getpid()}.part")
        with partial.open("wb") as stream:
            np.savez(stream, **arrays)
        partial.replace(path)  # atomic: other processes never see a half-written file
    finally:
        raw.unlink(missing_ok=True)
        path.with_name(f"{path.stem}.{os.getpid()}.part").unlink(missing_ok=True)
    _saved_since_check += 1
    if _saved_since_check >= 50:  # listing the cache is not free, so check now and then
        _saved_since_check = 0
        _evict_least_recently_used()


def _read_fits(path: Path) -> dict[str, np.ndarray]:
    with fits.open(path, memmap=False) as hdul:
        data, header = hdul["LIGHTCURVE"].data, hdul[0].header
        return {
            "time": np.asarray(data["TIME"], dtype=np.float64),
            "flux": np.asarray(data["PDCSAP_FLUX"], dtype=np.float32),
            "flux_err": np.asarray(data["PDCSAP_FLUX_ERR"], dtype=np.float32),
            "sap_flux": np.asarray(data["SAP_FLUX"], dtype=np.float32),
            "quality": np.asarray(data["QUALITY"], dtype=np.int32),
            "tic_id": np.int64(header["TICID"]),
            "sector": np.int64(header["SECTOR"]),
        }


def _evict_least_recently_used() -> None:
    """If the cache is over its limit, delete the least recently used curves down to 90% of it."""
    limit = _cache_limit()
    entries = []
    for entry in os.scandir(cache_dir()):
        if entry.name.endswith(".npz"):
            try:
                stat = entry.stat()
            except FileNotFoundError:
                continue
            entries.append((stat.st_mtime, stat.st_size, entry.path))
    total = sum(size for _, size, _ in entries)
    if total <= limit:
        return
    for _, size, file in sorted(entries):
        if total <= 0.9 * limit:
            break
        Path(file).unlink(missing_ok=True)
        total -= size
