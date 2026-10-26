
-- Confirmed planets from the NASA Exoplanet Archive.
CREATE TABLE IF NOT EXISTS confirmed_planets (
    planet_name TEXT PRIMARY KEY,
    tic_id BIGINT REFERENCES tic_objects(tic_id),
    hostname TEXT,
    system_planet_count INTEGER,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- TESS Objects of Interest, including candidate classifications.
CREATE TABLE IF NOT EXISTS tess_objects_of_interest (
    toi_id TEXT PRIMARY KEY,
    tic_id BIGINT REFERENCES tic_objects(tic_id),
    disposition TEXT,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_confirmed_planets_tic
    ON confirmed_planets(tic_id);

CREATE INDEX IF NOT EXISTS idx_toi_tic
    ON tess_objects_of_interest(tic_id);
