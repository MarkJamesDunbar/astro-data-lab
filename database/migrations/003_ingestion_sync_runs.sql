-- Record each attempt to refresh an external data inventory.
-- Designed to support MAST now and exoplanet catalogues later.
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
