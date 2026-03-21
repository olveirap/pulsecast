-- schema.sql – TimescaleDB hypertable definitions for Pulsecast
-- Run once against the TimescaleDB instance to initialise the schema.

CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- -----------------------------------------------------------------------
-- routes: Reference table mapping route_id to (origin, destination) pairs
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS routes (
    route_id          SERIAL PRIMARY KEY,
    origin_zone_id    INTEGER NOT NULL,
    destination_zone_id INTEGER NOT NULL,
    UNIQUE(origin_zone_id, destination_zone_id)
);

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
-- congestion: Bus position variance signal per zone/hour (training + RT)
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS congestion (
    zone_id          INTEGER          NOT NULL,
    hour             TIMESTAMPTZ      NOT NULL,
    travel_time_var  DOUBLE PRECISION NOT NULL,
    sample_count     INTEGER          NOT NULL,
    disruption_flag  BOOLEAN          NOT NULL DEFAULT FALSE,
    PRIMARY KEY (zone_id, hour)
);

SELECT create_hypertable(
    'congestion',
    'hour',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

ALTER TABLE congestion SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'zone_id'
);

SELECT add_compression_policy('congestion', INTERVAL '7 days', if_not_exists => TRUE);

-- Useful index for serving-layer look-ups.
CREATE INDEX IF NOT EXISTS idx_congestion_zone_hour
    ON congestion (zone_id, hour DESC);

-- -----------------------------------------------------------------------
-- subway_delay: MTA GTFS-Realtime subway delays (RT only)
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS subway_delay (
    zone_id     INTEGER          NOT NULL,
    hour        TIMESTAMPTZ      NOT NULL,
    feed_id     VARCHAR(32)      NOT NULL,
    mean_delay  DOUBLE PRECISION NOT NULL,
    trip_count  INTEGER          NOT NULL,
    PRIMARY KEY (zone_id, hour, feed_id)
);

SELECT create_hypertable(
    'subway_delay',
    'hour',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

ALTER TABLE subway_delay SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'zone_id, feed_id'
);

SELECT add_compression_policy('subway_delay', INTERVAL '7 days', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_subway_delay_zone_hour
    ON subway_delay (zone_id, hour DESC);
