"""Step 2: load what the database knows, as pandas DataFrames.

    load_curves()             ->  one row per light-curve file
    load_tois()               ->  one row per TESS Object of Interest
    load_confirmed_planets()  ->  one row per confirmed planet
    split_of(tic_id)          ->  "train" | "validation" | "test"

Nothing here downloads light curves. You build your labels from these tables
(e.g. join curves to TOIs on `tic_id`), giving a DataFrame with at least
`data_uri` and `label` columns, which is what `LightCurveDataset` takes.
"""
from __future__ import annotations

import hashlib

import pandas as pd

from astro_data_lab.database import connect, ensure_schema

SPLITS = ("train",) * 8 + ("validation", "test")  # 80 / 10 / 10


def split_of(tic_id: int) -> str:
    """Fixed train/validation/test split for a star, from a hash of its TIC ID.

    Every sector of a star gets the same split, so a model is always tested on
    stars it never trained on. The answer never changes between runs.
    """
    return SPLITS[int(hashlib.md5(str(int(tic_id)).encode()).hexdigest()[:8], 16) % 10]


def _query(sql: str) -> pd.DataFrame:
    ensure_schema()
    with connect() as conn, conn.cursor() as cursor:
        cursor.execute(sql)
        return pd.DataFrame(cursor.fetchall(), columns=[d.name for d in cursor.description])


def load_curves() -> pd.DataFrame:
    """Every indexed light curve: one star observed for one TESS sector.

    Columns:
      data_uri    str   the file's ID; pass it to `read_lightcurve`
      tic_id      int   the star (TESS Input Catalog ID)
      sector      int   TESS observing sector (~27 days)
      size_bytes  int   size of the FITS file at MAST
      split       str   "train" / "validation" / "test", fixed per star (`split_of`)
    """
    curves = _query("SELECT data_uri, tic_id, sector, size_bytes FROM mast_products "
                    "WHERE tic_id IS NOT NULL ORDER BY data_uri")
    splits = {tic: split_of(tic) for tic in curves["tic_id"].unique()}
    curves["split"] = curves["tic_id"].map(splits)
    return curves


def load_tois() -> pd.DataFrame:
    """Every TESS Object of Interest: a signal TESS flagged around a star.

    Columns:
      toi             str    e.g. "1234.01" (one star can have several)
      tic_id          int    the star
      disposition     str    PC planet candidate, APC ambiguous candidate, CP confirmed
                             planet, KP known planet, FP false positive, FA false alarm
      epoch_btjd      float  centre of one transit, in BTJD (same time scale as the light curves)
      period_days     float  time between transits (NaN when only one transit is known)
      duration_hours  float  length of one transit
      depth_ppm       float  how much the star dims, in parts per million
    """
    return _query("""SELECT toi_id AS toi, tic_id, disposition,
                            transit_epoch_bjd - 2457000 AS epoch_btjd,
                            NULLIF(period_days, 0) AS period_days, duration_hours, depth_ppm
                     FROM tess_objects_of_interest WHERE tic_id IS NOT NULL ORDER BY toi_id""")


def load_confirmed_planets() -> pd.DataFrame:
    """Every confirmed planet (NASA Exoplanet Archive), however it was found.

    Columns: planet_name, tic_id, hostname, system_planet_count.
    Many were found by other telescopes or methods and never transit in TESS data.
    """
    return _query("SELECT planet_name, tic_id, hostname, system_planet_count FROM confirmed_planets "
                  "WHERE tic_id IS NOT NULL ORDER BY planet_name")
