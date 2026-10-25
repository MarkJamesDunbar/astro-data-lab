"""Import an inventory of mission-produced TESS light curves into PostgreSQL.

This module imports *product metadata only*; it never downloads FITS files.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import psycopg

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


def connect_postgres() -> psycopg.Connection:
    """Open a database connection using the uv-loaded runtime environment."""
    password_file = Path(os.environ["ASTRO_DB_PASSWORD_FILE"])
    return psycopg.connect(
        host=os.environ["PGHOST"],
        port=int(os.environ["PGPORT"]),
        dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=password_file.read_text(encoding="utf-8").strip(),
        connect_timeout=15,
    )


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
