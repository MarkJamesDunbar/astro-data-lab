"""Step 1: keep the database up to date with what exists.

    sync()   ->  None

Updates two things in PostgreSQL (nothing is downloaded except metadata):
  - the planet catalogues (`confirmed_planets`, `tess_objects_of_interest`);
  - the manifest of light-curve files at MAST (`mast_products`), adding up to
    `max_sectors` sectors that are not indexed yet.
Safe to run as often as you like; completed sectors are skipped unless
`refresh_one=True`, which also refreshes the oldest completed sector.
"""
from __future__ import annotations

import os
from pathlib import Path

from astro_data_lab.database import ensure_schema


def sync(catalogues: bool = True, manifest: bool = True, max_sectors: int = 2,
         refresh_one: bool = False) -> None:
    ensure_schema()
    if catalogues:
        from astro_data_lab.sync.catalogues import synchronise_catalogues

        synchronise_catalogues()
    if manifest:
        from astro_data_lab.sync.manifest import inventory_lock, sync_incremental

        directory = Path(os.environ["MAST_INVENTORY_DIR"])
        with inventory_lock(directory):
            sync_incremental(directory, 1, max_sectors, refresh_one=refresh_one)
