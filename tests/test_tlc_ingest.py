"""
tests/test_tlc_ingest.py – Unit and integration-style tests for tlc.py.

All network and database calls are mocked so that the tests run offline
without any TLC download or TimescaleDB instance.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from pulsecast.data.ingest.tlc import (
    aggregate_hourly,
    ingest,
    write_to_db,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HOUR = datetime(2024, 3, 7, 14, 0, 0)


def _make_hourly_df(rows: list[tuple[int, int, datetime, int, float]]) -> pl.DataFrame:
    """Build a small aggregated DataFrame with the schema produced by aggregate_hourly."""
    return pl.DataFrame(
        {
            "PULocationID": [r[0] for r in rows],
            "DOLocationID": [r[1] for r in rows],
            "hour": [r[2] for r in rows],
            "pickup_count": [r[3] for r in rows],
            "avg_duration": [r[4] for r in rows],
        },
        schema={
            "PULocationID": pl.Int64,
            "DOLocationID": pl.Int64,
            "hour": pl.Datetime("us"),
            "pickup_count": pl.UInt32,
            "avg_duration": pl.Float64,
        },
    )


@pytest.fixture()
def mock_db_conn():
    """Return a MagicMock psycopg2 connection that supports context-manager usage."""
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_cur = mock_conn.cursor.return_value
    mock_cur.__enter__ = MagicMock(return_value=mock_cur)
    mock_cur.__exit__ = MagicMock(return_value=False)
    # Return mock data for the routes table fetch
    mock_cur.fetchall.return_value = [
        (999, 1, 2),
        (1000, 10, 20),
        (1001, 1, 1),
        (42, 42, 42),
    ]
    return mock_conn


# ---------------------------------------------------------------------------
# aggregate_hourly
# ---------------------------------------------------------------------------


def test_aggregate_hourly_counts_per_zone_and_hour():
    raw = pl.DataFrame(
        {
            "pickup_datetime": [
                datetime(2024, 3, 7, 14, 5),
                datetime(2024, 3, 7, 14, 45),
                datetime(2024, 3, 7, 15, 10),
                datetime(2024, 3, 7, 14, 20),
            ],
            "dropoff_datetime": [
                datetime(2024, 3, 7, 14, 15), # 10 min
                datetime(2024, 3, 7, 15, 5),  # 20 min
                datetime(2024, 3, 7, 15, 20), # 10 min
                datetime(2024, 3, 7, 14, 30), # 10 min
            ],
            "PULocationID": [1, 1, 1, 2],
            "DOLocationID": [2, 2, 2, 3],
        },
        schema={"pickup_datetime": pl.Datetime("us"), "dropoff_datetime": pl.Datetime("us"), "PULocationID": pl.Int64, "DOLocationID": pl.Int64},
    )
    result = aggregate_hourly(raw)
    assert result.shape[0] == 3
    row = result.filter(
        (pl.col("PULocationID") == 1) & (pl.col("DOLocationID") == 2) & (pl.col("hour") == datetime(2024, 3, 7, 14, 0, 0))
    )
    assert row["pickup_count"].item() == 2
    # mean of 10 min (600s) and 20 min (1200s) = 900s
    assert row["avg_duration"].item() == pytest.approx(900.0)


def test_aggregate_hourly_output_columns():
    raw = pl.DataFrame(
        {
            "pickup_datetime": [datetime(2024, 1, 1, 8, 0)],
            "dropoff_datetime": [datetime(2024, 1, 1, 8, 10)],
            "PULocationID": [10],
            "DOLocationID": [20],
        },
        schema={"pickup_datetime": pl.Datetime("us"), "dropoff_datetime": pl.Datetime("us"), "PULocationID": pl.Int64, "DOLocationID": pl.Int64},
    )
    result = aggregate_hourly(raw)
    assert set(result.columns) == {"PULocationID", "DOLocationID", "hour", "pickup_count", "avg_duration"}


# ---------------------------------------------------------------------------
# write_to_db
# ---------------------------------------------------------------------------


def test_write_to_db_returns_zero_for_empty_df():
    empty = pl.DataFrame(
        schema={"PULocationID": pl.Int64, "DOLocationID": pl.Int64, "hour": pl.Datetime("us"), "pickup_count": pl.UInt32, "avg_duration": pl.Float64}
    )
    result = write_to_db(empty, "postgresql://fake/fake")
    assert result == 0


def test_write_to_db_upserts_rows(mock_db_conn):
    df = _make_hourly_df([(1, 2, _HOUR, 10, 300.0), (10, 20, _HOUR, 5, 400.0)])

    with (
        patch("pulsecast.data.ingest.tlc.psycopg2.connect", return_value=mock_db_conn),
        patch("pulsecast.data.ingest.tlc.execute_values") as mock_ev,
    ):
        count = write_to_db(df, "postgresql://fake/fake")

    assert count == 2
    mock_ev.assert_called_once()
    params_list = mock_ev.call_args[0][2]
    assert len(params_list) == 2
    assert params_list[0][3] == 300.0


def test_write_to_db_uses_zone_id_as_route_id(mock_db_conn):
    """write_to_db should map PULocationID and DOLocationID to route_id."""
    df = _make_hourly_df([(1, 2, _HOUR, 3, 300.0)])

    with (
        patch("pulsecast.data.ingest.tlc.psycopg2.connect", return_value=mock_db_conn),
        patch("pulsecast.data.ingest.tlc.execute_values") as mock_ev,
    ):
        write_to_db(df, "postgresql://fake/fake")

    mock_ev.assert_called_once()
    params_list = mock_ev.call_args[0][2]
    assert params_list[0][0] == 999  # (1, 2) maps to 999 based on mock


def test_write_to_db_preserves_zone_id(mock_db_conn):
    """write_to_db should preserve mapped routes correctly."""
    df = _make_hourly_df([(42, 42, _HOUR, 7, 300.0)])

    with (
        patch("pulsecast.data.ingest.tlc.psycopg2.connect", return_value=mock_db_conn),
        patch("pulsecast.data.ingest.tlc.execute_values") as mock_ev,
    ):
        write_to_db(df, "postgresql://fake/fake")

    mock_ev.assert_called_once()
    params_list = mock_ev.call_args[0][2]
    assert params_list[0][0] == 42


def test_write_to_db_hour_is_utc_aware(mock_db_conn):
    """Hours passed to the DB must carry explicit UTC tzinfo."""
    from datetime import UTC

    df = _make_hourly_df([(1, 2, _HOUR, 5, 300.0)])

    with (
        patch("pulsecast.data.ingest.tlc.psycopg2.connect", return_value=mock_db_conn),
        patch("pulsecast.data.ingest.tlc.execute_values") as mock_ev,
    ):
        write_to_db(df, "postgresql://fake/fake")

    params_list = mock_ev.call_args[0][2]
    assert params_list[0][1].tzinfo is UTC


def test_write_to_db_always_closes_connection():
    """The DB connection must be closed even when an exception is raised."""
    df = _make_hourly_df([(1, 2, _HOUR, 1, 300.0)])

    mock_conn = MagicMock()
    mock_conn.cursor.side_effect = RuntimeError("boom")
    
    with (
        patch("pulsecast.data.ingest.tlc.psycopg2.connect", return_value=mock_conn),
        pytest.raises(RuntimeError),
    ):
        write_to_db(df, "postgresql://fake/fake")

    mock_conn.close.assert_called_once()


# ---------------------------------------------------------------------------
# ingest (integration-style with all external calls mocked)
# ---------------------------------------------------------------------------


def _make_parquet_bytes() -> bytes:
    """Build a minimal in-memory Parquet file with the yellow-taxi schema."""
    import io

    df = pl.DataFrame(
        {
            "tpep_pickup_datetime": [datetime(2024, 3, 7, 14, 5), datetime(2024, 3, 7, 14, 30)],
            "tpep_dropoff_datetime": [datetime(2024, 3, 7, 14, 15), datetime(2024, 3, 7, 14, 40)],
            "PULocationID": [1, 2],
            "DOLocationID": [2, 3],
            "trip_distance": [1.2, 3.4],
            "fare_amount": [8.5, 12.0],
        },
        schema={
            "tpep_pickup_datetime": pl.Datetime("us"),
            "tpep_dropoff_datetime": pl.Datetime("us"),
            "PULocationID": pl.Int64,
            "DOLocationID": pl.Int64,
            "trip_distance": pl.Float64,
            "fare_amount": pl.Float64,
        },
    )
    buf = io.BytesIO()
    df.write_parquet(buf)
    return buf.getvalue()


def test_ingest_returns_nonempty_dataframe(tmp_path: Path):
    """ingest() must return a non-empty DataFrame when data is available."""
    parquet_bytes = _make_parquet_bytes()

    def _fake_download(color, year, month, dest_dir):
        p = dest_dir / f"{color}_tripdata_{year}-{month:02d}.parquet"
        p.write_bytes(parquet_bytes)
        return p

    with (
        patch("pulsecast.data.ingest.tlc.download_parquet", side_effect=_fake_download),
        patch("pulsecast.data.ingest.tlc._month_range", return_value=[(2024, 3)]),
    ):
        # Pass dsn=None to avoid a real DB connection if TIMESCALE_DSN is set in the environment.
        result = ingest(dest_dir=tmp_path, months=1, colors=("yellow",), dsn=None)

    assert result.shape[0] > 0
    assert set(result.columns) == {"PULocationID", "DOLocationID", "hour", "pickup_count", "avg_duration"}


def test_ingest_calls_write_to_db_when_dsn_provided(tmp_path: Path):
    """When dsn is provided ingest() must call write_to_db with the combined DataFrame."""
    parquet_bytes = _make_parquet_bytes()

    def _fake_download(color, year, month, dest_dir):
        p = dest_dir / f"{color}_tripdata_{year}-{month:02d}.parquet"
        p.write_bytes(parquet_bytes)
        return p

    with (
        patch("pulsecast.data.ingest.tlc.download_parquet", side_effect=_fake_download),
        patch("pulsecast.data.ingest.tlc._month_range", return_value=[(2024, 3)]),
        patch("pulsecast.data.ingest.tlc.write_to_db", return_value=2) as mock_write,
    ):
        ingest(dest_dir=tmp_path, months=1, colors=("yellow",), dsn="postgresql://x/y")

    mock_write.assert_called_once()
    written_df = mock_write.call_args[0][0]
    assert written_df.shape[0] > 0


def test_ingest_does_not_call_write_to_db_without_dsn(tmp_path: Path):
    """When dsn is None write_to_db must never be called."""
    parquet_bytes = _make_parquet_bytes()

    def _fake_download(color, year, month, dest_dir):
        p = dest_dir / f"{color}_tripdata_{year}-{month:02d}.parquet"
        p.write_bytes(parquet_bytes)
        return p

    with (
        patch("pulsecast.data.ingest.tlc.download_parquet", side_effect=_fake_download),
        patch("pulsecast.data.ingest.tlc._month_range", return_value=[(2024, 3)]),
        patch("pulsecast.data.ingest.tlc.write_to_db") as mock_write,
    ):
        ingest(dest_dir=tmp_path, months=1, colors=("yellow",), dsn=None)

    mock_write.assert_not_called()


def test_ingest_row_count_greater_than_zero(tmp_path: Path):
    """Integration guard: the returned DataFrame must have at least one row."""
    parquet_bytes = _make_parquet_bytes()

    def _fake_download(color, year, month, dest_dir):
        p = dest_dir / f"{color}_tripdata_{year}-{month:02d}.parquet"
        p.write_bytes(parquet_bytes)
        return p

    with (
        patch("pulsecast.data.ingest.tlc.download_parquet", side_effect=_fake_download),
        patch("pulsecast.data.ingest.tlc._month_range", return_value=[(2024, 3)]),
        patch("pulsecast.data.ingest.tlc.write_to_db", return_value=2) as mock_write,
    ):
        result = ingest(
            dest_dir=tmp_path,
            months=1,
            colors=("yellow",),
            dsn="postgresql://fake/fake",
        )

    assert result.shape[0] > 0
    mock_write.assert_called_once()
