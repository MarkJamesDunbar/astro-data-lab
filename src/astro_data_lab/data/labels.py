"""Step 2: label every curve, split stars into train/validation/test, and sample curves.

Target "transit" asks one question per light curve: is there a transit-like signal?

- signal (1): the star is a TESS Object of Interest (planet candidate, confirmed
  or known planet, or a false positive such as an eclipsing binary), AND one of
  its predicted transits falls inside this curve's readings. That second check
  needs the curve's timestamps, so it happens after download
  (`lightcurves.transit_readings`): every "signal" curve really contains an event.
- no_signal (0): the star is in neither planet catalogue.
- left out: confirmed planets TESS never flagged (often non-transiting), false
  alarms (FA), and TOIs without transit timing.

Stars, not curves, are split 80/10/10: every sector of one star lands in the
same split, so the model is always tested on stars it never saw. The split is a
fixed hash of the TIC ID, identical for every run and architecture.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import psycopg

NO_SIGNAL, SIGNAL = 0, 1
SPLITS = ("train",) * 8 + ("validation", "test")  # 80 / 10 / 10 by star
SPLIT_SQL = ("(ARRAY['train','train','train','train','train','train','train','train',"
             "'validation','test'])[mod(('x' || substr(md5(p.tic_id::text), 1, 8))::bit(32)::bigint, 10) + 1]")

# One row per usable curve: (data_uri, size_bytes, ephemerides, target, split).
# Ephemerides are [epoch (BTJD = BJD - 2457000), period (days, 0 if unknown), duration (days)].
_LABELLED = {
    "transit": f"""
WITH signals AS (
    SELECT tic_id, array_agg(ARRAY[transit_epoch_bjd - 2457000,
                                   COALESCE(period_days, 0), duration_hours / 24]) AS ephemerides
    FROM tess_objects_of_interest
    WHERE tic_id IS NOT NULL AND disposition IN ('PC', 'APC', 'CP', 'KP', 'FP')
      AND transit_epoch_bjd IS NOT NULL AND duration_hours > 0
    GROUP BY tic_id
), catalogued AS (
    SELECT tic_id FROM confirmed_planets WHERE tic_id IS NOT NULL
    UNION SELECT tic_id FROM tess_objects_of_interest WHERE tic_id IS NOT NULL
), labelled AS (
    SELECT p.data_uri, p.size_bytes, s.ephemerides,
           CASE WHEN s.tic_id IS NOT NULL THEN {SIGNAL}
                WHEN NOT EXISTS (SELECT 1 FROM catalogued c WHERE c.tic_id = p.tic_id) THEN {NO_SIGNAL}
           END AS target,
           {SPLIT_SQL} AS split
    FROM mast_products p
    LEFT JOIN signals s ON s.tic_id = p.tic_id
    WHERE p.tic_id IS NOT NULL AND p.size_bytes > 0
      AND NOT EXISTS (SELECT 1 FROM excluded_curves x WHERE x.data_uri = p.data_uri)
)""",
}


@dataclass(frozen=True)
class Curve:
    data_uri: str
    size_bytes: int
    target: int
    ephemerides: tuple[tuple[float, float, float], ...] | None = None  # real signals only
    inject: bool = False  # a no_signal curve that will receive a synthetic transit


def split_of(tic_id: int) -> str:
    """Python twin of SPLIT_SQL (tested to agree)."""
    return SPLITS[int(hashlib.md5(str(tic_id).encode()).hexdigest()[:8], 16) % 10]


def _curves(rows: list[tuple]) -> list[Curve]:
    return [Curve(uri, int(size), int(target),
                  tuple(tuple(map(float, e)) for e in ephemerides) if ephemerides else None)
            for uri, size, ephemerides, target in rows]


def class_counts(conn: psycopg.Connection, target: str, split: str) -> dict[int, int]:
    """Curves per class, before the after-download transit check."""
    rows = conn.execute(
        _LABELLED[target] + "SELECT target, COUNT(*) FROM labelled "
        "WHERE split = %s AND target IS NOT NULL GROUP BY target", (split,),
    ).fetchall()
    return {int(t): int(n) for t, n in rows}


def fixed_sample(conn: psycopg.Connection, target: str, split: str, per_class: int) -> list[Curve]:
    """The same curves every time, in a fixed order: used for validation and test sets."""
    rows = conn.execute(
        _LABELLED[target] + """, ranked AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY target ORDER BY md5(data_uri)) AS n
            FROM labelled WHERE split = %s AND target IS NOT NULL)
        SELECT data_uri, size_bytes, ephemerides, target FROM ranked WHERE n <= %s
        ORDER BY target, n""",
        (split, per_class),
    ).fetchall()
    return _curves(rows)


def balanced_subset(curves: list[Curve], per_class: int) -> list[Curve]:
    """The first `per_class` curves of each class, keeping their order."""
    taken: dict[int, int] = {}
    kept = []
    for curve in curves:
        if taken.get(curve.target, 0) < per_class:
            taken[curve.target] = taken.get(curve.target, 0) + 1
            kept.append(curve)
    return kept


def training_pool(conn: psycopg.Connection, target: str, label: int) -> list[Curve]:
    """All training curves of one class (used for classes small enough to cache)."""
    rows = conn.execute(
        _LABELLED[target] + "SELECT data_uri, size_bytes, ephemerides, target FROM labelled "
        "WHERE split = 'train' AND target = %s ORDER BY data_uri", (label,),
    ).fetchall()
    return _curves(rows)


def unseen_training_curves(conn: psycopg.Connection, target: str, label: int, run_id: int,
                           count: int, seed: int, skip: set[str]) -> list[Curve]:
    """Fresh curves of a huge class that this run has never trained on."""
    rows = conn.execute(
        _LABELLED[target] + """SELECT data_uri, size_bytes, ephemerides, target FROM labelled l
        WHERE split = 'train' AND target = %s AND data_uri <> ALL(%s)
          AND NOT EXISTS (SELECT 1 FROM training_examples e
                          WHERE e.run_id = %s AND e.data_uri = l.data_uri)
        ORDER BY md5(data_uri || %s) LIMIT %s""",
        (label, list(skip), run_id, str(seed), count),
    ).fetchall()
    return _curves(rows)
