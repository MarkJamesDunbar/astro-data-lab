
-- Record each catalogue synchronisation attempt.
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

-- Preserve meaningful changes between catalogue snapshots.
CREATE TABLE IF NOT EXISTS catalogue_change_events (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    sync_run_id BIGINT NOT NULL
        REFERENCES catalogue_sync_runs(id),
    catalogue TEXT NOT NULL,
    record_id TEXT NOT NULL,
    change_type TEXT NOT NULL,
    previous_value JSONB,
    current_value JSONB,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_catalogue_events_detected
    ON catalogue_change_events(detected_at);

CREATE INDEX IF NOT EXISTS idx_catalogue_events_record
    ON catalogue_change_events(catalogue, record_id);
