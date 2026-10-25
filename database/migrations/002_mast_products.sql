
CREATE TABLE IF NOT EXISTS mast_products (
    data_uri TEXT PRIMARY KEY,
    tic_id BIGINT REFERENCES tic_objects(tic_id),
    sector INTEGER,
    filename TEXT NOT NULL,
    size_bytes BIGINT CHECK (size_bytes >= 0),
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_mast_products_tic
ON mast_products(tic_id);
