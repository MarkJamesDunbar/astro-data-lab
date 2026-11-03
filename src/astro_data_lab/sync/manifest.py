"""Index which TESS SPOC light-curve files exist at MAST (the manifest).

Records metadata only (URI, star, sector, size); never downloads FITS files.
Completed sectors are skipped, so repeated syncs only fetch new sectors.
"""

import fcntl
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd
from astroquery.mast import Observations

from astro_data_lab.database import connect as connect_postgres

TIC_PATTERN = re.compile(r"-s\d{4}-(\d+)-")
REQUIRED_COLUMNS = {"dataURI", "productFilename", "size"}


@dataclass(frozen=True)
class ImportSummary:
    products: int
    new: int
    changed: int
    unchanged: int
    not_seen: int
    size_bytes: int


def import_mast_inventory(
    path: Path,
    sector: int,
    *,
    run_id: int | None = None,
) -> ImportSummary:
    """Upsert one complete sector inventory in a single transaction.

    Missing products are *not* deleted: they may have been temporarily absent
    at MAST. We record the count for review instead.
    """
    inventory = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS - set(inventory.columns)
    if missing_columns:
        raise ValueError(f"Missing MAST columns: {sorted(missing_columns)}")
    if inventory.empty:
        raise ValueError(f"Sector {sector} inventory is empty; refusing import")
    if inventory[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("MAST inventory has missing identifiers, names or sizes")

    inventory = inventory.drop_duplicates(subset="dataURI").copy()
    tic_text = inventory["productFilename"].str.extract(TIC_PATTERN)[0]
    if tic_text.isna().any():
        raise ValueError("Some SPOC filenames have no recognisable TIC ID")
    inventory["tic_id"] = tic_text.astype("int64")

    sizes = pd.to_numeric(inventory["size"], errors="raise")
    if ((sizes < 0) | (sizes % 1 != 0)).any():
        raise ValueError("Invalid product sizes in inventory")
    inventory["size_bytes"] = sizes.astype("int64")

    stars = [(int(tic),) for tic in inventory["tic_id"].unique()]
    products = [
        (
            str(row.dataURI),
            int(row.tic_id),
            int(sector),
            str(row.productFilename),
            int(row.size_bytes),
        )
        for row in inventory.itertuples(index=False)
    ]

    # One transaction includes the scientific inventory and its completed audit.
    with connect_postgres() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """SELECT data_uri, tic_id, sector, filename, size_bytes
                   FROM mast_products WHERE sector = %s""",
                (sector,),
            )
            existing = {
                uri: (int(tic), int(sec), filename, int(size))
                for uri, tic, sec, filename, size in cursor.fetchall()
            }
            new = changed = unchanged = 0
            seen = set()
            for uri, tic, sec, filename, size in products:
                seen.add(uri)
                old = existing.get(uri)
                if old is None:
                    new += 1
                elif old != (tic, sec, filename, size):
                    changed += 1
                else:
                    unchanged += 1
            not_seen = len(existing.keys() - seen)

            cursor.executemany(
                """INSERT INTO tic_objects (tic_id) VALUES (%s)
                   ON CONFLICT (tic_id) DO NOTHING""",
                stars,
            )
            cursor.executemany(
                """INSERT INTO mast_products
                       (data_uri, tic_id, sector, filename, size_bytes)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (data_uri) DO UPDATE SET
                       tic_id = EXCLUDED.tic_id,
                       sector = EXCLUDED.sector,
                       filename = EXCLUDED.filename,
                       size_bytes = EXCLUDED.size_bytes,
                       last_seen_at = NOW()""",
                products,
            )

            # Check that all incoming product keys now exist in the right sector.
            cursor.execute(
                """SELECT COUNT(*) FROM mast_products
                   WHERE sector = %s AND data_uri = ANY(%s)""",
                (sector, list(seen)),
            )
            verified = cursor.fetchone()[0]
            if verified != len(products):
                raise RuntimeError(
                    f"PostgreSQL verification failed: {verified} / {len(products)}"
                )

            total_size_bytes = int(inventory["size_bytes"].sum())
            if run_id is not None:
                cursor.execute(
                    """UPDATE ingestion_sync_runs SET
                           status = 'completed', finished_at = NOW(),
                           products_found = %s, new_products = %s,
                           changed_products = %s, unchanged_products = %s,
                           not_seen_products = %s, total_size_bytes = %s,
                           error_message = NULL
                       WHERE id = %s AND status = 'running'""",
                    (
                        len(products), new, changed, unchanged,
                        not_seen, total_size_bytes, run_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("Sync run was not in a running state")

    return ImportSummary(
        products=len(products), new=new, changed=changed,
        unchanged=unchanged, not_seen=not_seen,
        size_bytes=total_size_bytes,
    )


BATCH_SIZE = 250
MAX_RETRIES = 3
SOURCE = "mast_spoc"


def duration(seconds: float) -> str:
    """Format elapsed time as hours:minutes:seconds."""
    return str(timedelta(seconds=int(seconds)))


def progress(done: int, total: int) -> str:
    """Build a fixed-width progress bar with its exact percentage."""
    fraction = done / total if total else 0.0
    blocks = min(20, int(fraction * 20))
    return f"[{'#' * blocks}{'.' * (20 - blocks)}] {fraction:6.1%}"


@contextmanager
def inventory_lock(directory: Path):
    """Prevent concurrent inventory jobs from modifying the same staging area."""
    directory.mkdir(parents=True, exist_ok=True)
    # Keep this file: deleting a flock lock file can break concurrency safety.
    with (directory / ".inventory.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another MAST inventory job is active") from exc
        yield


def retry(operation, label: str):
    """Retry recoverable request failures without masking the final error."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return operation()
        except Exception as error:
            if attempt == MAX_RETRIES:
                raise
            pause = 5 * attempt
            print(f"{label}: {error}; retry in {pause}s ({attempt}/{MAX_RETRIES})", flush=True)
            time.sleep(pause)


def completed_before(sector: int) -> bool:
    with connect_postgres() as conn:
        row = conn.execute(
            """SELECT EXISTS (
                SELECT 1 FROM ingestion_sync_runs
                WHERE source = %s AND scope = %s AND status = 'completed'
            )""",
            (SOURCE, f"sector:{sector}"),
        ).fetchone()
    return bool(row[0])


def completed_sectors() -> set[int]:
    """Read completed sector IDs once for efficient incremental scheduling."""
    with connect_postgres() as conn:
        rows = conn.execute(
            """SELECT DISTINCT scope FROM ingestion_sync_runs
               WHERE source = %s AND status = 'completed'
                 AND scope LIKE 'sector:%%'""",
            (SOURCE,),
        ).fetchall()
    completed = set()
    for (scope,) in rows:
        match = re.fullmatch(r"sector:(\d+)", scope)
        if match:
            completed.add(int(match.group(1)))
    return completed


def oldest_completed_sector() -> int | None:
    """Choose the sector with the oldest successful inventory for a refresh."""
    with connect_postgres() as conn:
        rows = conn.execute(
            """SELECT scope FROM ingestion_sync_runs
               WHERE source = %s AND status = 'completed'
                 AND scope LIKE 'sector:%%'
               GROUP BY scope
               ORDER BY MAX(finished_at) ASC NULLS FIRST
               LIMIT 1""",
            (SOURCE,),
        ).fetchall()
    if not rows:
        return None
    match = re.fullmatch(r"sector:(\d+)", rows[0][0])
    return int(match.group(1)) if match else None


def query_sector_observations(sector: int):
    """Request SPOC time-series observations for a single TESS sector."""
    return retry(
        lambda: Observations.query_criteria(
            obs_collection="TESS",
            provenance_name="SPOC",
            dataproduct_type="timeseries",
            sequence_number=sector,
        ),
        f"Sector {sector} observation query",
    )


def start_sync(sector: int) -> int:
    with connect_postgres() as conn:
        conn.execute(
            """UPDATE ingestion_sync_runs
               SET status = 'interrupted', finished_at = NOW(),
                   error_message = 'Previous process stopped unexpectedly'
               WHERE source = %s AND scope = %s AND status = 'running'""",
            (SOURCE, f"sector:{sector}"),
        )
        row = conn.execute(
            """INSERT INTO ingestion_sync_runs (source, scope, status)
               VALUES (%s, %s, 'running') RETURNING id""",
            (SOURCE, f"sector:{sector}"),
        ).fetchone()
    return int(row[0])


def finish_unsuccessful(run_id: int, status: str, error: BaseException):
    """Mark a failed/interrupted attempt; never overwrite a completed run."""
    with connect_postgres() as conn:
        conn.execute(
            """UPDATE ingestion_sync_runs
               SET status = %s, finished_at = NOW(), error_message = %s
               WHERE id = %s AND status = 'running'""",
            (status, str(error)[:2000], run_id),
        )


def discover_sector(sector: int, ordinal: int, total_sectors: int, observations=None) -> pd.DataFrame:
    """Collect SPOC LC product metadata, reusing an incremental preflight query."""
    sector_started = time.monotonic()
    print(f"\n[{ordinal}/{total_sectors}] Sector {sector:02d}: querying MAST", flush=True)
    if observations is None:
        observations = query_sector_observations(sector)
    total = len(observations)
    if total == 0:
        raise ValueError(f"Sector {sector}: no SPOC observations returned")

    batch_count = (total + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Sector {sector:02d}: {total:,} observations in {batch_count} batches", flush=True)
    batches = []

    for batch_index, start in enumerate(range(0, total, BATCH_SIZE), 1):
        stop = min(start + BATCH_SIZE, total)
        products = retry(
            lambda a=start, b=stop: Observations.get_product_list(
                observations[a:b]
            ).to_pandas(),
            f"Sector {sector} batch {batch_index}",
        )
        required = {
            "productType", "productSubGroupDescription", "productFilename",
            "dataURI", "size",
        }
        if not required.issubset(products.columns):
            raise ValueError(f"MAST response missing {sorted(required - set(products.columns))}")
        lightcurves = products.loc[
            products["productType"].eq("SCIENCE")
            & products["productSubGroupDescription"].eq("LC")
            & products["productFilename"].str.endswith("_lc.fits", na=False)
        ]
        batches.append(lightcurves)

        print(
            f"  Sector {sector:02d} {progress(stop, total)} | "
            f"{stop:,}/{total:,} obs | batch {batch_index}/{batch_count} | "
            f"elapsed {duration(time.monotonic() - sector_started)}",
            flush=True,
        )

    inventory = pd.concat(batches, ignore_index=True).drop_duplicates(subset="dataURI")
    if inventory.empty:
        raise ValueError(f"Sector {sector}: no SPOC LC products returned")
    return inventory


def sync_sector(sector: int, directory: Path, ordinal: int, total_sectors: int, observations=None):
    """Import one sector transactionally and remove only this run's temp files."""
    run_id = start_sync(sector)
    work_dir = directory / ".staging"
    work_dir.mkdir(parents=True, exist_ok=True)
    staged = work_dir / f"sector_{sector:03d}_run_{run_id}.csv"
    partial = work_dir / f"sector_{sector:03d}_run_{run_id}.tmp"
    started = time.monotonic()

    try:
        inventory = discover_sector(sector, ordinal, total_sectors, observations=observations)
        inventory.to_csv(partial, index=False)
        partial.replace(staged)
        print(f"  Sector {sector:02d}: importing {len(inventory):,} records", flush=True)
        summary = import_mast_inventory(staged, sector, run_id=run_id)
        print(
            f"  Sector {sector:02d} COMPLETE | new {summary.new:,} | "
            f"changed {summary.changed:,} | unchanged {summary.unchanged:,} | "
            f"{summary.size_bytes / 1024**3:.2f} GiB represented | "
            f"elapsed {duration(time.monotonic() - started)}",
            flush=True,
        )
        return summary
    except KeyboardInterrupt as error:
        finish_unsuccessful(run_id, "interrupted", error)
        print(f"\nSector {sector:02d}: cancelled; database imports remain consistent", flush=True)
        raise
    except Exception as error:
        finish_unsuccessful(run_id, "failed", error)
        print(f"\nSector {sector:02d} FAILED: {error}", flush=True)
        raise
    finally:
        # Never touch older sector_XXX_lc.csv snapshots in the parent directory.
        partial.unlink(missing_ok=True)
        staged.unlink(missing_ok=True)


def cleanup_stale(directory: Path):
    """Clean interrupted-run temporary files, not previously saved inventory CSVs."""
    candidates = list((directory / ".staging").glob("sector_*_run_*.csv"))
    candidates += list((directory / ".staging").glob("sector_*_run_*.tmp"))
    # Previous script used sector_NNN_lc.tmp in the inventory root.
    candidates += [
        p for p in directory.glob("sector_*_lc.tmp")
        if re.fullmatch(r"sector_\d{3}_lc\.tmp", p.name)
    ]
    for path in candidates:
        if path.is_file():
            path.unlink()
            print(f"Removed stale temporary file: {path}", flush=True)
    with connect_postgres() as conn:
        result = conn.execute(
            """UPDATE ingestion_sync_runs
               SET status = 'interrupted', finished_at = NOW(),
                   error_message = 'Cancelled or stopped before completion'
               WHERE source = %s AND status = 'running'""",
            (SOURCE,),
        )
        stale_runs = result.rowcount
    print(
        f"Cleanup complete: {len(candidates)} temporary files, "
        f"{stale_runs} orphaned run records updated. "
        "Existing sector CSVs and database products preserved.",
        flush=True,
    )


def sync_incremental(directory: Path, start_sector: int, max_sectors: int, refresh_one: bool = False):
    """Backfill next missing sectors, stopping when MAST has none available.

    Existing completed sectors are skipped. A sector that has no SPOC time-series
    observations yet is not marked as completed or failed: it is retried next run.
    Network or product errors are real failures, so they stop this invocation.
    """
    completed = completed_sectors()
    sector = start_sector
    ingested = 0

    while ingested < max_sectors:
        while sector in completed:
            sector += 1

        print(f"Checking next unindexed TESS sector: {sector}", flush=True)
        observations = query_sector_observations(sector)
        if len(observations) == 0:
            print(
                f"Sector {sector}: no SPOC time-series observations available yet. "
                "No import attempted; will check again next run.",
                flush=True,
            )
            break

        sync_sector(
            sector, directory, ingested + 1, max_sectors,
            observations=observations,
        )
        completed.add(sector)
        ingested += 1
        sector += 1

    if refresh_one:
        older_sector = oldest_completed_sector()
        if older_sector is not None:
            print(f"Refreshing oldest previously inventoried sector: {older_sector}", flush=True)
            sync_sector(older_sector, directory, 1, 1)

    print(f"Incremental MAST sync complete: {ingested} new sector(s) inventoried.", flush=True)
