"""Step 3: download a FITS light curve, clean it, and turn it into model input.

A whole curve is one example. Cleaning drops flagged/invalid readings only (no
cropping or interpolation); `model_input` adds a detrended view so slow stellar
variability does not hide hours-long transit dips. Curves that cannot be used
are reported with `UnusableLightCurve`, so callers skip them instead of failing.

Also here: `transit_readings` (does a known transit fall inside this curve?)
and `inject_transit` (add a realistic synthetic transit for extra training data).
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import torch
from astropy.io import fits

from astro_data_lab.config import GIB, Settings
from astro_data_lab.database import connect
from astro_data_lab.data.labels import Curve

DOWNLOAD_URL = "https://mast.stsci.edu/api/v0.1/Download/file"
# MAST serves TESS files from this public AWS bucket; going there directly skips a
# multi-second redirect per file. MAST is the fallback for anything not found there.
S3_URL = "https://stpubdata.s3.us-east-1.amazonaws.com/tess/public/tid/s{sector}/{a}/{b}/{c}/{d}/{name}"
_S3_NAME = re.compile(r"tess\d+-s(\d{4})-(\d{16})-\d+-s_lc\.fits")
MIN_CADENCES = 100
MIN_TRANSIT_READINGS = 10  # a "signal" curve needs this many readings inside a transit
INPUT_CHANNELS = 4  # relative flux %, detrended flux in noise units, log time gap, valid mask
INPUT_VERSION = 2   # bump whenever model_input changes; checkpoints record it
_session = threading.local()


class UnusableLightCurve(ValueError):
    """The file is fine, but its data cannot become a training example."""


class StorageFull(RuntimeError):
    """Writing would breach the configured cache limit or free-space floor."""


@dataclass(frozen=True)
class CleanLightCurve:
    time: np.ndarray  # BTJD days, strictly increasing
    flux: np.ndarray  # PDCSAP flux divided by its median (~1.0)
    tic_id: int | None
    sector: int | None


def clean_fits(path: Path) -> CleanLightCurve:
    """Keep QUALITY == 0, finite readings; normalise by the median flux."""
    try:
        with fits.open(path, memmap=False) as hdul:
            data = hdul["LIGHTCURVE"].data
            time_ = np.asarray(data["TIME"], dtype=np.float64)
            flux = np.asarray(data["PDCSAP_FLUX"], dtype=np.float64)
            quality = np.asarray(data["QUALITY"], dtype=np.int64)
            header = hdul[0].header
            tic_id, sector = header.get("TICID"), header.get("SECTOR")
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise UnusableLightCurve(f"unreadable light-curve FITS: {error}") from error
    keep = np.isfinite(time_) & np.isfinite(flux) & (quality == 0)
    time_, flux = time_[keep], flux[keep]
    if len(time_) < MIN_CADENCES:
        raise UnusableLightCurve(f"only {len(time_)} valid readings (need {MIN_CADENCES})")
    order = np.argsort(time_, kind="stable")
    time_, flux = time_[order], flux[order]
    unique = np.concatenate(([True], np.diff(time_) > 0))
    time_, flux = time_[unique], flux[unique]
    median = float(np.median(flux))
    if not median > 0:
        raise UnusableLightCurve(f"non-positive median flux ({median:.1f})")
    return CleanLightCurve(time_, flux / median,
                           int(tic_id) if tic_id is not None else None,
                           int(sector) if sector is not None else None)


def detrend(time_: np.ndarray, flux: np.ndarray, window_days: float = 1.0,
            step_days: float = 0.25) -> np.ndarray:
    """flux / (running 1-day median, sampled every 6 hours and interpolated) - 1.

    Removes slow stellar variability (spots, pulsations longer than a day) while
    keeping dips that last hours, such as transits. A second pass leaves readings
    more than 3 sigma below the first trend out of the median, so a transit
    cannot drag the trend down and erase part of itself.
    """
    nodes = np.arange(time_[0], time_[-1] + step_days, step_days)
    starts = np.searchsorted(time_, nodes - window_days / 2)
    ends = np.searchsorted(time_, nodes + window_days / 2)
    use = np.ones(len(flux), dtype=bool)
    for _ in range(2):
        centres, levels = [], []
        for start, end in zip(starts, ends):
            window = use[start:end]
            if window.sum() >= 20:
                centres.append(time_[start:end][window].mean())
                levels.append(np.median(flux[start:end][window]))
        if not levels:
            return flux / np.median(flux) - 1
        trend = np.maximum(np.interp(time_, centres, levels), 1e-3)
        residual = flux / trend - 1
        use = residual > -3 * point_noise(residual)
    return residual


def point_noise(values: np.ndarray) -> float:
    """Robust reading-to-reading scatter (ignores slow trends and single outliers)."""
    differences = np.diff(values)
    mad = np.median(np.abs(differences - np.median(differences))) if len(differences) else 0.0
    return max(1.4826 * float(mad) / np.sqrt(2), 1e-6)


def model_input(time_: np.ndarray, flux: np.ndarray) -> torch.Tensor:
    """Shape (4, cadences). Values are clipped so one wild curve cannot blow up training."""
    relative = flux - 1.0
    detrended = detrend(time_, flux)
    gaps = np.diff(time_, prepend=time_[0])
    typical_gap = float(np.median(gaps[1:])) if len(gaps) > 1 else 1.0
    features = np.stack((
        np.clip(relative * 100, -100, 100),                          # brightness change, %
        np.clip(detrended / point_noise(detrended), -50, 50),        # detrended, in noise units
        np.clip(np.log1p(gaps / typical_gap), 0, 10),                # data gaps
        np.ones_like(flux),                                          # 1 = real reading, 0 = padding
    ))
    return torch.from_numpy(features.astype(np.float32))


def transit_readings(time_: np.ndarray, ephemerides: tuple[tuple[float, float, float], ...]) -> int:
    """How many readings fall inside a predicted transit (epoch BTJD, period days, duration days).

    A period of 0 means only one transit is known (at the epoch).
    """
    inside = np.zeros(len(time_), dtype=bool)
    for epoch, period, duration in ephemerides:
        offset = time_ - epoch
        if period > 0:
            offset = (offset + period / 2) % period - period / 2
        inside |= np.abs(offset) < duration / 2
    return int(inside.sum())


def inject_transit(time_: np.ndarray, flux: np.ndarray, seed: int
                   ) -> tuple[np.ndarray, tuple[float, float, float]] | None:
    """Add a synthetic periodic transit that is detectable but not obvious.

    Period 0.5-15 days (log-uniform), duration 1-8 hours, trapezoid shape, and a
    depth chosen so the total signal-to-noise is 8-40 (TESS flags candidates
    from about 7). Returns the new flux and (epoch, period, duration), or None
    if no placement puts enough readings in transit.
    """
    rng = np.random.default_rng(seed)
    noise = point_noise(flux)
    for _ in range(20):
        period = float(np.exp(rng.uniform(np.log(0.5), np.log(15.0))))
        duration = min(rng.uniform(1.0, 8.0) / 24, 0.15 * period)
        epoch = float(rng.choice(time_))  # centred on a real reading, so at least one is visible
        offset = (time_ - epoch + period / 2) % period - period / 2
        shape = np.clip((1 - np.abs(offset) / (duration / 2)) / 0.3, 0, 1)  # flat bottom, sloped edges
        readings = int(np.count_nonzero(shape))
        if readings >= MIN_TRANSIT_READINGS:
            depth = min(rng.uniform(8, 40) * noise / np.sqrt(readings), 0.05)
            return flux * (1 - depth * shape), (epoch, period, duration)
    return None


def collate(curves: list[torch.Tensor]) -> torch.Tensor:
    """Zero-pad curves to the longest in the minibatch; channel 3 marks real readings."""
    length = max(curve.shape[-1] for curve in curves)
    return torch.stack([torch.nn.functional.pad(c, (0, length - c.shape[-1])) for c in curves])


def cache_path(cache_dir: Path, data_uri: str) -> Path:
    return cache_dir / f"{hashlib.sha256(data_uri.encode()).hexdigest()[:24]}.npz"


def load_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Cleaned (time, flux) from the cache."""
    with np.load(path, allow_pickle=False) as saved:
        return saved["time"], saved["flux"].astype(np.float64)


def injection_seed(data_uri: str) -> int:
    return int(hashlib.sha256(data_uri.encode()).hexdigest()[:8], 16)


def curve_tensor(curve: Curve, path: Path) -> torch.Tensor | None:
    """Model input for one cached curve, or None if its label cannot be confirmed.

    Real signals must have a predicted transit inside the readings; injected
    curves get their synthetic transit here.
    """
    time_, flux = load_arrays(path)
    if curve.inject:
        injected = inject_transit(time_, flux, injection_seed(curve.data_uri))
        if injected is None:
            return None
        flux = injected[0]
    elif curve.ephemerides and transit_readings(time_, curve.ephemerides) < MIN_TRANSIT_READINGS:
        return None
    return model_input(time_, flux)


def download_urls(data_uri: str) -> list[tuple[str, dict]]:
    """Where to fetch a SPOC light curve from, fastest first."""
    name = data_uri.rsplit("/", 1)[-1]
    urls = [(DOWNLOAD_URL, {"uri": data_uri})]
    if match := _S3_NAME.fullmatch(name):
        tic = match.group(2)
        urls.insert(0, (S3_URL.format(sector=match.group(1), a=tic[:4], b=tic[4:8], c=tic[8:12],
                                      d=tic[12:], name=name), {}))
    return urls


def directory_size(directory: Path) -> int:
    return sum(p.stat().st_size for p in directory.glob("*.npz")) if directory.exists() else 0


def prepare_curve(session: requests.Session, data_uri: str, size_bytes: int,
                  cache_dir: Path, free_floor_bytes: int) -> Path:
    """Return the cached cleaned curve, downloading and cleaning it if needed.

    Raises UnusableLightCurve for bad data, StorageFull for disk limits, and
    other exceptions for network problems (after three attempts per source).
    """
    target = cache_path(cache_dir, data_uri)
    if target.exists():
        return target
    cache_dir.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(cache_dir).free - 3 * size_bytes < free_floor_bytes:
        raise StorageFull("free-space floor reached")
    raw = target.with_suffix(f".{os.getpid()}.fits.part")
    try:
        sources = download_urls(data_uri)
        for attempt in range(3 * len(sources)):
            url, params = sources[attempt % len(sources)]
            try:
                with session.get(url, params=params, stream=True, timeout=(20, 120)) as response:
                    response.raise_for_status()
                    with raw.open("wb") as stream:
                        for chunk in response.iter_content(1024 * 1024):
                            stream.write(chunk)
                if raw.stat().st_size != size_bytes:
                    raise OSError(f"downloaded {raw.stat().st_size} bytes, expected {size_bytes}")
                break
            except (requests.RequestException, OSError):
                if attempt == 3 * len(sources) - 1:
                    raise
                time.sleep(attempt // len(sources) + 1)
        curve = clean_fits(raw)
        partial = target.with_suffix(f".{os.getpid()}.part")
        with partial.open("wb") as stream:
            np.savez_compressed(stream, time=curve.time, flux=curve.flux.astype(np.float32))
        partial.replace(target)
        return target
    finally:
        raw.unlink(missing_ok=True)
        target.with_suffix(f".{os.getpid()}.part").unlink(missing_ok=True)


@dataclass
class Prepared:
    curves: list[Curve]          # successfully prepared, in the order requested
    tensors: list[torch.Tensor]  # empty when prepared with load=False
    excluded: int                # unusable data, now in excluded_curves
    failed: int                  # e.g. download errors; may succeed another time
    no_transit: int = 0          # "signal" stars whose known transits miss this curve


def prepare_many(curves: list[Curve], settings: Settings, workers: int,
                 stop: threading.Event | None = None, load: bool = True) -> Prepared:
    """Download/clean what is not cached yet, confirm labels, and build model inputs.

    With load=False, curves are downloaded and checked but no tensors are kept
    (used to vet a whole training pool without holding it in memory).
    """
    if directory_size(settings.curve_cache) > settings.cache_limit_gib * GIB:
        raise StorageFull(f"curve cache exceeds CURVE_CACHE_GIB={settings.cache_limit_gib}")
    sizes = {c.data_uri: c.size_bytes for c in curves}
    floor = int(settings.free_space_floor_gib * GIB)

    def fetch(uri: str) -> tuple[str, str | None, str | None]:
        if stop is not None and stop.is_set():
            return uri, "failed", "cancelled"
        if not hasattr(_session, "value"):
            _session.value = requests.Session()
        try:
            prepare_curve(_session.value, uri, sizes[uri], settings.curve_cache, floor)
            return uri, None, None
        except UnusableLightCurve as error:
            return uri, "excluded", str(error)
        except StorageFull:
            raise
        except Exception as error:  # network/server problems: skip this curve for now
            return uri, "failed", f"{type(error).__name__}: {error}"

    with ThreadPoolExecutor(workers) as pool:
        outcomes = list(pool.map(fetch, sizes))
    problems = {uri: (kind, reason) for uri, kind, reason in outcomes if kind}
    excluded = [(uri, reason) for uri, (kind, reason) in problems.items() if kind == "excluded"]
    if excluded:
        with connect() as conn, conn.cursor() as cursor:
            cursor.executemany("INSERT INTO excluded_curves (data_uri, reason) VALUES (%s, %s) "
                               "ON CONFLICT DO NOTHING", excluded)
        print(f"  skipped {len(excluded)} unusable curves, e.g. {excluded[0][1]}", flush=True)
    failed = [reason for kind, reason in problems.values() if kind == "failed"]
    if failed and not (stop is not None and stop.is_set()):
        print(f"  {len(failed)} curves could not be downloaded, e.g. {failed[0]}", flush=True)
    kept, tensors, no_transit = [], [], 0
    for curve in curves:
        if curve.data_uri in problems:
            continue
        tensor = curve_tensor(curve, cache_path(settings.curve_cache, curve.data_uri))
        if tensor is None:
            no_transit += 1
            continue
        kept.append(curve)
        if load:
            tensors.append(tensor)
    return Prepared(kept, tensors, len(excluded), len(failed), no_transit)


def check_quality(data: Prepared, requested: int, limit: float, what: str) -> None:
    """Fail if too many curves were bad data or unreachable (signals an outage)."""
    bad = data.excluded + data.failed
    if requested and bad / requested > limit:
        raise RuntimeError(f"{what}: {bad}/{requested} curves unusable or unreachable "
                           f"(> max_bad_fraction={limit}); check network/MAST and retry")
