
"""Synchronise NASA exoplanet catalogues with PostgreSQL."""

import os
import re
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd
import psycopg
import requests


TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

CATALOGUES = {
    "confirmed_planets": (
        "SELECT pl_name, hostname, tic_id, sy_pnum "
        "FROM pscomppars"
    ),
    "tess_candidates": (
        "SELECT toi, tid, tfopwg_disp FROM toi"
    ),
}


def normalise_tic(value):
    """Return a numeric TIC ID, or None if unavailable."""
    match = re.fullmatch(
        r"(?:TIC\s*)?(\d+)(?:\.0+)?",
        str(value).strip(),
        flags=re.IGNORECASE,
    )
    return int(match.group(1)) if match else None


def fetch_catalogue(query):
    """Retrieve a complete catalogue from NASA's TAP service."""
    response = requests.get(
        TAP_URL,
        params={"query": query, "format": "csv"},
        timeout=180,
    )
    response.raise_for_status()

    data = pd.read_csv(
        BytesIO(response.content),
        dtype=str,
        keep_default_na=False,
    )

    if data.empty:
        raise ValueError("NASA returned an empty catalogue")

    return data, response.content


def synchronise_catalogues():
    """Download catalogue snapshots and upsert their records."""

    snapshot_dir = Path(os.environ["CATALOGUE_SNAPSHOT_DIR"])
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    catalogues = {}

    for name, query in CATALOGUES.items():
        print(f"Fetching {name}...", flush=True)

        data, content = fetch_catalogue(query)
        catalogues[name] = data

        destination = snapshot_dir / name
        destination.mkdir(parents=True, exist_ok=True)

        (destination / f"{timestamp}.csv").write_bytes(content)

        print(f"Retrieved {len(data):,} records", flush=True)

    confirmed = catalogues["confirmed_planets"]
    candidates = catalogues["tess_candidates"]

    planets = [
        (
            row.pl_name,
            normalise_tic(row.tic_id),
            row.hostname or None,
            int(row.sy_pnum) if row.sy_pnum else None,
        )
        for row in confirmed.itertuples(index=False)
        if row.pl_name
    ]

    tois = [
        (
            row.toi,
            normalise_tic(row.tid),
            row.tfopwg_disp or None,
        )
        for row in candidates.itertuples(index=False)
        if row.toi
    ]

    tic_ids = {
        tic
        for tic in (
            [row[1] for row in planets]
            + [row[1] for row in tois]
        )
        if tic is not None
    }

    password = Path(
        os.environ["ASTRO_DB_PASSWORD_FILE"]
    ).read_text().strip()

    with psycopg.connect(
        host=os.environ["PGHOST"],
        port=int(os.environ["PGPORT"]),
        dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=password,
    ) as connection:

        with connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO tic_objects (tic_id)
                VALUES (%s)
                ON CONFLICT DO NOTHING
                """,
                [(tic,) for tic in tic_ids],
            )

            cursor.executemany(
                """
                INSERT INTO confirmed_planets
                    (planet_name, tic_id, hostname, system_planet_count)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (planet_name)
                DO UPDATE SET
                    tic_id = EXCLUDED.tic_id,
                    hostname = EXCLUDED.hostname,
                    system_planet_count = EXCLUDED.system_planet_count,
                    last_seen_at = NOW()
                """,
                planets,
            )

            cursor.executemany(
                """
                INSERT INTO tess_objects_of_interest
                    (toi_id, tic_id, disposition)
                VALUES (%s, %s, %s)
                ON CONFLICT (toi_id)
                DO UPDATE SET
                    tic_id = EXCLUDED.tic_id,
                    disposition = EXCLUDED.disposition,
                    last_seen_at = NOW()
                """,
                tois,
            )

    print(f"Confirmed planets processed: {len(planets):,}")
    print(f"TOI records processed: {len(tois):,}")
    print("Catalogue synchronisation complete")


if __name__ == "__main__":
    synchronise_catalogues()

