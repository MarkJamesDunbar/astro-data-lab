"""Discover and synchronise TESS SPOC light-curve product metadata.

Only inventories MAST products; never downloads actual FITS light curves.
Runs are recorded in PostgreSQL, and completed sectors are skipped unless
--refresh is requested. Working CSVs are deleted after use or interruption.
"""

import argparse
import fcntl
import os
import re
import time
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import pandas as pd
from astroquery.mast import Observations

from astro_data_lab.ingestion.mast_inventory_import import (
    connect_postgres,
    import_mast_inventory,
)

BATCH_SIZE = 1000
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


def discover_sector(sector: int, ordinal: int, total_sectors: int) -> pd.DataFrame:
    """Collect all SPOC LC product metadata for one sector, with progress."""
    sector_started = time.monotonic()
    print(f"\n[{ordinal}/{total_sectors}] Sector {sector:02d}: querying MAST", flush=True)
    observations = retry(
        lambda: Observations.query_criteria(
            obs_collection="TESS",
            provenance_name="SPOC",
            dataproduct_type="timeseries",
            sequence_number=sector,
        ),
        f"Sector {sector} observation query",
    )
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


def sync_sector(sector: int, directory: Path, ordinal: int, total_sectors: int):
    """Import one sector transactionally and remove only this run's temp files."""
    run_id = start_sync(sector)
    work_dir = directory / ".staging"
    work_dir.mkdir(parents=True, exist_ok=True)
    staged = work_dir / f"sector_{sector:03d}_run_{run_id}.csv"
    partial = work_dir / f"sector_{sector:03d}_run_{run_id}.tmp"
    started = time.monotonic()

    try:
        inventory = discover_sector(sector, ordinal, total_sectors)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-sector", type=int, default=1)
    parser.add_argument("--end-sector", type=int, default=26)
    parser.add_argument("--sector", type=int, help="Scan one specific sector")
    parser.add_argument("--refresh", action="store_true", help="Rescan completed sectors")
    parser.add_argument("--cleanup-stale", action="store_true", help="Clean interrupted staging files only")
    args = parser.parse_args()

    first = last = args.sector if args.sector is not None else None
    if first is None:
        first, last = args.start_sector, args.end_sector
    if first < 1 or last < first:
        parser.error("Sector numbers must be positive and end >= start")

    directory = Path(os.environ["MAST_INVENTORY_DIR"])
    with inventory_lock(directory):
        if args.cleanup_stale:
            cleanup_stale(directory)
            return

        started = time.monotonic()
        done = skipped = succeeded = 0
        failures = []
        total = last - first + 1
        for ordinal, sector in enumerate(range(first, last + 1), 1):
            try:
                if not args.refresh and completed_before(sector):
                    skipped += 1
                    print(f"\n[{ordinal}/{total}] Sector {sector:02d}: SKIPPED (completed before)", flush=True)
                else:
                    sync_sector(sector, directory, ordinal, total)
                    succeeded += 1
            except KeyboardInterrupt:
                print(
                    f"\nCancelled. {done}/{total} sectors finished in "
                    f"{duration(time.monotonic() - started)}. "
                    "Rerun to resume from incomplete sectors.",
                    flush=True,
                )
                raise SystemExit(130)
            except Exception as error:
                failures.append(sector)
                print(f"Continuing after Sector {sector:02d}: {error}", flush=True)
            done += 1
            print(
                f"OVERALL {progress(done, total)} | {done}/{total} visited | "
                f"{succeeded} completed, {skipped} skipped, "
                f"{len(failures)} failed | elapsed {duration(time.monotonic() - started)}",
                flush=True,
            )

    if failures:
        raise SystemExit(f"Failed sectors {failures}. Rerun to retry those sectors.")
    print("\nAll requested sectors have been inventoried successfully.", flush=True)


if __name__ == "__main__":
    main()
