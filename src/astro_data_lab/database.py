"""PostgreSQL connection, and `schema.sql` (the whole database), applied automatically."""
from __future__ import annotations

import os
from importlib import resources
from pathlib import Path

import psycopg


def connect() -> psycopg.Connection:
    """Connect using PG* variables and a password file (keeps secrets out of .env)."""
    password = Path(os.environ["ASTRO_DB_PASSWORD_FILE"]).read_text(encoding="utf-8").strip()
    return psycopg.connect(
        host=os.environ["PGHOST"],
        port=int(os.environ["PGPORT"]),
        dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=password,
        connect_timeout=15,
    )


def ensure_schema() -> None:
    """Create any missing tables. Safe to call every time; existing data is untouched."""
    schema = resources.files("astro_data_lab").joinpath("schema.sql").read_text(encoding="utf-8")
    with connect() as conn:
        conn.execute("SELECT pg_advisory_xact_lock(hashtext('astro_data_lab_schema'))")
        conn.execute(schema)
