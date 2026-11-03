-- The complete database. Every statement is idempotent, so astro-data-lab
-- applies this file automatically; there are no migration scripts to run.

-- 1. Planet catalogues (astro-data-lab sync)
CREATE TABLE IF NOT EXISTS tic_objects (
    tic_id BIGINT PRIMARY KEY,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS confirmed_planets (
    planet_name TEXT PRIMARY KEY,
    tic_id BIGINT REFERENCES tic_objects(tic_id),
    hostname TEXT,
    system_planet_count INTEGER,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_confirmed_planets_tic ON confirmed_planets(tic_id);

CREATE TABLE IF NOT EXISTS tess_objects_of_interest (
    toi_id TEXT PRIMARY KEY,
    tic_id BIGINT REFERENCES tic_objects(tic_id),
    disposition TEXT,                    -- PC, APC, CP, KP, FP, FA
    transit_epoch_bjd DOUBLE PRECISION,  -- time of one transit centre
    period_days DOUBLE PRECISION,        -- NULL/0 when only one transit is known
    duration_hours DOUBLE PRECISION,
    depth_ppm DOUBLE PRECISION,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_toi_tic ON tess_objects_of_interest(tic_id);

-- Audit trail of what changed between catalogue downloads.
CREATE TABLE IF NOT EXISTS catalogue_sync_runs (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    catalogue TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    records_seen INTEGER NOT NULL DEFAULT 0,
    records_added INTEGER NOT NULL DEFAULT 0,
    records_changed INTEGER NOT NULL DEFAULT 0,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS catalogue_change_events (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    sync_run_id BIGINT NOT NULL REFERENCES catalogue_sync_runs(id),
    catalogue TEXT NOT NULL,
    record_id TEXT NOT NULL,
    change_type TEXT NOT NULL,
    previous_value JSONB,
    current_value JSONB,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_catalogue_events_detected ON catalogue_change_events(detected_at);
CREATE INDEX IF NOT EXISTS idx_catalogue_events_record ON catalogue_change_events(catalogue, record_id);

-- 2. Light-curve manifest: which FITS files exist at MAST (metadata only)
CREATE TABLE IF NOT EXISTS mast_products (
    data_uri TEXT PRIMARY KEY,
    tic_id BIGINT REFERENCES tic_objects(tic_id),
    sector INTEGER,
    filename TEXT NOT NULL,
    size_bytes BIGINT CHECK (size_bytes >= 0),
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_mast_products_tic ON mast_products(tic_id);

-- Which sectors have been indexed, so sync only fetches new ones.
CREATE TABLE IF NOT EXISTS ingestion_sync_runs (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source TEXT NOT NULL,
    scope TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed', 'interrupted')),
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    products_found INTEGER,
    new_products INTEGER,
    changed_products INTEGER,
    unchanged_products INTEGER,
    not_seen_products INTEGER,
    total_size_bytes BIGINT,
    error_message TEXT
);
CREATE INDEX IF NOT EXISTS idx_ingestion_sync_runs_lookup
    ON ingestion_sync_runs (source, scope, status, finished_at DESC);
