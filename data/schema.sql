-- schema.sql – TimescaleDB hypertable definitions for Pulsecast
-- Run once against the TimescaleDB instance to initialise the schema.

CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- -----------------------------------------------------------------------
-- demand: hourly pickup volume per route/zone
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS demand (
    route_id    INTEGER     NOT NULL,
    hour        TIMESTAMPTZ NOT NULL,
    volume      INTEGER     NOT NULL CHECK (volume >= 0),
    PRIMARY KEY (route_id, hour)
);

-- Partition by day, keeping the last 36 months of data online.
SELECT create_hypertable(
    'demand',
    'hour',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

ALTER TABLE demand SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'route_id'
);

SELECT add_compression_policy('demand', INTERVAL '7 days', if_not_exists => TRUE);

-- -----------------------------------------------------------------------
-- delay_index: MTA GTFS-Realtime congestion signal per zone/hour
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS delay_index (
    zone_id          INTEGER     NOT NULL,
    hour             TIMESTAMPTZ NOT NULL,
    delay_index      DOUBLE PRECISION NOT NULL,
    disruption_flag  BOOLEAN     NOT NULL DEFAULT FALSE,
    PRIMARY KEY (zone_id, hour)
);

SELECT create_hypertable(
    'delay_index',
    'hour',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

ALTER TABLE delay_index SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'zone_id'
);

SELECT add_compression_policy('delay_index', INTERVAL '7 days', if_not_exists => TRUE);

-- Useful index for serving-layer look-ups.
CREATE INDEX IF NOT EXISTS idx_delay_index_zone_hour
    ON delay_index (zone_id, hour DESC);
