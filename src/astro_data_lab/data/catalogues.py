"""Step 1a: download the NASA planet catalogues (confirmed planets and TOIs).

These provide the labels; each TOI's transit timing (epoch, period, duration)
lets us check whether a transit falls inside a particular light curve. Each download is snapshotted to
CATALOGUE_SNAPSHOT_DIR and every changed record is logged in
`catalogue_change_events`.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
import requests
from psycopg.types.json import Jsonb

from astro_data_lab.database import connect as connect_database


TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
CATALOGUES = {
    "confirmed_planets": {
        "query": "SELECT pl_name, hostname, tic_id, sy_pnum FROM pscomppars",
        "columns": {"pl_name", "hostname", "tic_id", "sy_pnum"},
    },
    "tess_candidates": {
        "query": "SELECT toi, tid, tfopwg_disp, pl_tranmid, pl_orbper, pl_trandurh, pl_trandep FROM toi",
        "columns": {"toi", "tid", "tfopwg_disp", "pl_tranmid", "pl_orbper", "pl_trandurh", "pl_trandep"},
    },
}


def normalise_tic(value: Any) -> int | None:
    """Parse NASA's 'TIC 12345' or numeric TIC identifiers safely."""
    if value is None:
        return None
    match = re.fullmatch(
        r"(?:TIC\s*)?(\d+)(?:\.0+)?",
        str(value).strip(),
        flags=re.IGNORECASE,
    )
    return int(match.group(1)) if match else None


def optional_text(value: Any) -> str | None:
    """Treat blank strings as missing values."""
    text = str(value).strip() if value is not None else ""
    return text or None


def optional_float(value: Any) -> float | None:
    text = optional_text(value)
    return float(text) if text else None


def fetch_catalogue(name: str) -> tuple[pd.DataFrame, bytes]:
    """Fetch and validate one complete NASA catalogue."""
    definition = CATALOGUES[name]
    response = requests.get(
        TAP_URL,
        params={"query": definition["query"], "format": "csv"},
        timeout=(15, 180),
    )
    response.raise_for_status()
    content = response.content
    data = pd.read_csv(BytesIO(content), dtype=str, keep_default_na=False)

    missing = definition["columns"] - set(data.columns)
    if missing:
        raise ValueError(f"{name}: NASA response missing columns {sorted(missing)}")
    if data.empty:
        raise ValueError(f"{name}: NASA returned zero records; refusing to sync")

    return data, content


def write_snapshot(name: str, content: bytes) -> Path:
    """Retain a timestamped source snapshot; never partially overwrite it."""
    base = Path(os.environ["CATALOGUE_SNAPSHOT_DIR"]) / name
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = base / f"{stamp}.csv"
    temporary = base / f"{stamp}.csv.part"
    temporary.write_bytes(content)
    temporary.replace(output)
    return output


def prepare_records(name: str, data: pd.DataFrame) -> dict[str, dict]:
    """Map each stable catalogue identifier to the fields we track."""
    records: dict[str, dict] = {}

    if name == "confirmed_planets":
        for row in data.itertuples(index=False):
            key = optional_text(row.pl_name)
            if not key:
                raise ValueError("Confirmed-planet record has no planet name")
            raw_count = optional_text(row.sy_pnum)
            record = {
                "tic_id": normalise_tic(row.tic_id),
                "hostname": optional_text(row.hostname),
                "system_planet_count": int(raw_count) if raw_count else None,
            }
            if key in records:
                raise ValueError(f"Duplicate confirmed-planet identifier: {key}")
            records[key] = record

    elif name == "tess_candidates":
        for row in data.itertuples(index=False):
            key = optional_text(row.toi)
            if not key:
                raise ValueError("TOI record has no identifier")
            record = {
                "tic_id": normalise_tic(row.tid),
                "disposition": optional_text(row.tfopwg_disp),
                "transit_epoch_bjd": optional_float(row.pl_tranmid),
                "period_days": optional_float(row.pl_orbper),
                "duration_hours": optional_float(row.pl_trandurh),
                "depth_ppm": optional_float(row.pl_trandep),
            }
            if key in records:
                raise ValueError(f"Duplicate TOI identifier: {key}")
            records[key] = record
    else:
        raise ValueError(f"Unknown catalogue: {name}")

    return records


def existing_records(cursor: psycopg.Cursor, name: str) -> dict[str, dict]:
    """Fetch existing values once so we don't issue a query for every row."""
    if name == "confirmed_planets":
        cursor.execute(
            "SELECT planet_name, tic_id, hostname, system_planet_count "
            "FROM confirmed_planets"
        )
        return {
            key: {
                "tic_id": tic_id,
                "hostname": hostname,
                "system_planet_count": count,
            }
            for key, tic_id, hostname, count in cursor.fetchall()
        }

    cursor.execute("SELECT toi_id, tic_id, disposition, transit_epoch_bjd, period_days, "
                   "duration_hours, depth_ppm FROM tess_objects_of_interest")
    return {
        key: {"tic_id": tic_id, "disposition": disposition, "transit_epoch_bjd": epoch,
              "period_days": period, "duration_hours": duration, "depth_ppm": depth}
        for key, tic_id, disposition, epoch, period, duration, depth in cursor.fetchall()
    }


def import_records(name: str, records: dict[str, dict]) -> tuple[int, int]:
    """Atomically upsert records, log differences and complete the sync run."""
    with connect_database() as connection:
        with connection.cursor() as cursor:
            old_records = existing_records(cursor, name)
            changes = []
            added = 0
            updated = 0

            for key, current in records.items():
                previous = old_records.get(key)
                if previous == current:
                    continue

                if previous is None:
                    change_type = "added"
                    added += 1
                else:
                    updated += 1
                    if name == "tess_candidates" and (
                        previous["disposition"] != current["disposition"]
                    ):
                        change_type = "disposition_changed"
                    elif name == "confirmed_planets" and (
                        previous["system_planet_count"] != current["system_planet_count"]
                    ):
                        change_type = "planet_count_changed"
                    else:
                        change_type = "updated"

                changes.append((name, key, change_type, previous, current))

            # New stars (from catalogues) must exist before adding FK references.
            tic_ids = sorted(
                {record["tic_id"] for record in records.values() if record["tic_id"] is not None}
            )
            cursor.executemany(
                "INSERT INTO tic_objects (tic_id) VALUES (%s) ON CONFLICT DO NOTHING",
                [(tic,) for tic in tic_ids],
            )

            if name == "confirmed_planets":
                cursor.executemany(
                    """
                    INSERT INTO confirmed_planets
                        (planet_name, tic_id, hostname, system_planet_count)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (planet_name) DO UPDATE SET
                        tic_id = EXCLUDED.tic_id,
                        hostname = EXCLUDED.hostname,
                        system_planet_count = EXCLUDED.system_planet_count,
                        last_seen_at = NOW()
                    """,
                    [
                        (key, rec["tic_id"], rec["hostname"], rec["system_planet_count"])
                        for key, rec in records.items()
                    ],
                )
            else:
                cursor.executemany(
                    """
                    INSERT INTO tess_objects_of_interest (toi_id, tic_id, disposition,
                        transit_epoch_bjd, period_days, duration_hours, depth_ppm)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (toi_id) DO UPDATE SET
                        tic_id = EXCLUDED.tic_id,
                        disposition = EXCLUDED.disposition,
                        transit_epoch_bjd = EXCLUDED.transit_epoch_bjd,
                        period_days = EXCLUDED.period_days,
                        duration_hours = EXCLUDED.duration_hours,
                        depth_ppm = EXCLUDED.depth_ppm,
                        last_seen_at = NOW()
                    """,
                    [(key, rec["tic_id"], rec["disposition"], rec["transit_epoch_bjd"],
                      rec["period_days"], rec["duration_hours"], rec["depth_ppm"])
                     for key, rec in records.items()],
                )

            cursor.execute(
                """
                INSERT INTO catalogue_sync_runs
                    (catalogue, status, started_at, finished_at,
                     records_seen, records_added, records_changed)
                VALUES (%s, 'completed', NOW(), NOW(), %s, %s, %s)
                RETURNING id
                """,
                (name, len(records), added, updated),
            )
            run_id = cursor.fetchone()[0]

            cursor.executemany(
                """
                INSERT INTO catalogue_change_events
                    (sync_run_id, catalogue, record_id, change_type,
                     previous_value, current_value)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                [
                    (run_id, catalogue, key, kind,
                     Jsonb(previous) if previous is not None else None, Jsonb(current))
                    for catalogue, key, kind, previous, current in changes
                ],
            )

    return added, updated


def record_failure(name: str, message: str) -> None:
    """Leave a visible failure record even when an import transaction rolls back."""
    try:
        with connect_database() as connection:
            connection.execute(
                """
                INSERT INTO catalogue_sync_runs
                    (catalogue, status, started_at, finished_at, error_message)
                VALUES (%s, 'failed', NOW(), NOW(), %s)
                """,
                (name, message[:4000]),
            )
    except Exception as error:
        print(f"WARNING: failed to record sync error in PostgreSQL: {error}", flush=True)


def synchronise_catalogues() -> None:
    """Fetch each catalogue, snapshot it, import it and record the differences."""
    for name in CATALOGUES:
        print(f"\nChecking {name}...", flush=True)
        try:
            data, content = fetch_catalogue(name)
            records = prepare_records(name, data)
            snapshot = write_snapshot(name, content)
            added, changed = import_records(name, records)
        except Exception as error:
            record_failure(name, str(error))
            print(f"FAILED {name}: {error}", flush=True)
            raise

        unchanged = len(records) - added - changed
        print(f"Snapshot: {snapshot}", flush=True)
        print(
            f"{name}: {len(records):,} seen | {added:,} new | "
            f"{changed:,} changed | {unchanged:,} unchanged",
            flush=True,
        )

    print("\nCatalogue synchronisation complete", flush=True)


if __name__ == "__main__":
    synchronise_catalogues()
