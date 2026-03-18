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

from data.ingest.tlc import (
    _ZONE_TO_ROUTE,
    aggregate_hourly,
    ingest,
    write_to_db,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HOUR = datetime(2024, 3, 7, 14, 0, 0)


def _make_hourly_df(rows: list[tuple[int, datetime, int]]) -> pl.DataFrame:
    """Build a small aggregated DataFrame with the schema produced by aggregate_hourly."""
    return pl.DataFrame(
        {
            "PULocationID": [r[0] for r in rows],
            "hour": [r[1] for r in rows],
            "pickup_count": [r[2] for r in rows],
        },
        schema={
            "PULocationID": pl.Int64,
            "hour": pl.Datetime("us"),
            "pickup_count": pl.UInt32,
        },
    )


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
            "PULocationID": [1, 1, 1, 2],
        },
        schema={"pickup_datetime": pl.Datetime("us"), "PULocationID": pl.Int64},
    )
    result = aggregate_hourly(raw)
    # Zone 1, hour 14 → 2 trips; zone 1, hour 15 → 1 trip; zone 2, hour 14 → 1 trip
    assert result.shape[0] == 3
    row = result.filter(
        (pl.col("PULocationID") == 1) & (pl.col("hour") == datetime(2024, 3, 7, 14, 0, 0))
    )
    assert row["pickup_count"].item() == 2


def test_aggregate_hourly_output_columns():
    raw = pl.DataFrame(
        {
            "pickup_datetime": [datetime(2024, 1, 1, 8, 0)],
            "PULocationID": [10],
        },
        schema={"pickup_datetime": pl.Datetime("us"), "PULocationID": pl.Int64},
    )
    result = aggregate_hourly(raw)
    assert set(result.columns) == {"PULocationID", "hour", "pickup_count"}


# ---------------------------------------------------------------------------
# write_to_db
# ---------------------------------------------------------------------------


def test_write_to_db_returns_zero_for_empty_df():
    empty = pl.DataFrame(
        schema={"PULocationID": pl.Int64, "hour": pl.Datetime("us"), "pickup_count": pl.UInt32}
    )
    result = write_to_db(empty, "postgresql://fake/fake")
    assert result == 0


def test_write_to_db_upserts_rows():
    df = _make_hourly_df([(1, _HOUR, 10), (2, _HOUR, 5)])

    mock_cur = MagicMock()
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    with patch("data.ingest.tlc.psycopg2.connect", return_value=mock_conn):
        count = write_to_db(df, "postgresql://fake/fake")

    assert count == 2
    mock_cur.executemany.assert_called_once()
    # The second argument to executemany is the list of params; it must have 2 tuples.
    params_list = mock_cur.executemany.call_args[0][1]
    assert len(params_list) == 2


def test_write_to_db_uses_zone_to_route_mapping(monkeypatch):
    """When _ZONE_TO_ROUTE has an entry the mapped route_id must be used."""
    monkeypatch.setitem(_ZONE_TO_ROUTE, 1, 999)

    df = _make_hourly_df([(1, _HOUR, 3)])

    mock_cur = MagicMock()
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    with patch("data.ingest.tlc.psycopg2.connect", return_value=mock_conn):
        write_to_db(df, "postgresql://fake/fake")

    mock_cur.executemany.assert_called_once()
    params_list = mock_cur.executemany.call_args[0][1]
    assert params_list[0][0] == 999  # route_id should be remapped


def test_write_to_db_identity_fallback_when_no_mapping():
    """When PULocationID has no entry in _ZONE_TO_ROUTE it is used as-is."""
    df = _make_hourly_df([(42, _HOUR, 7)])

    mock_cur = MagicMock()
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    with patch("data.ingest.tlc.psycopg2.connect", return_value=mock_conn):
        write_to_db(df, "postgresql://fake/fake")

    mock_cur.executemany.assert_called_once()
    params_list = mock_cur.executemany.call_args[0][1]
    assert params_list[0][0] == 42  # identity fallback


def test_write_to_db_always_closes_connection():
    """The DB connection must be closed even when an exception is raised."""
    df = _make_hourly_df([(1, _HOUR, 1)])

    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(side_effect=RuntimeError("boom"))
    mock_conn.__exit__ = MagicMock(return_value=False)

    with (
        patch("data.ingest.tlc.psycopg2.connect", return_value=mock_conn),
        pytest.raises(RuntimeError),
    ):
        write_to_db(df, "postgresql://fake/fake")

    mock_conn.close.assert_called_once()
# ingest (integration-style with all external calls mocked)
# ---------------------------------------------------------------------------


def _make_parquet_bytes() -> bytes:
    """Build a minimal in-memory Parquet file with the yellow-taxi schema."""
    import io

    df = pl.DataFrame(
        {
            "tpep_pickup_datetime": [datetime(2024, 3, 7, 14, 5), datetime(2024, 3, 7, 14, 30)],
            "PULocationID": [1, 2],
            "trip_distance": [1.2, 3.4],
            "fare_amount": [8.5, 12.0],
        },
        schema={
            "tpep_pickup_datetime": pl.Datetime("us"),
            "PULocationID": pl.Int64,
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

    with patch("data.ingest.tlc.download_parquet", side_effect=_fake_download):
        result = ingest(dest_dir=tmp_path, months=1, colors=("yellow",))

    assert result.shape[0] > 0
    assert set(result.columns) == {"PULocationID", "hour", "pickup_count"}


def test_ingest_calls_write_to_db_when_dsn_provided(tmp_path: Path):
    """When dsn is provided ingest() must call write_to_db with the combined DataFrame."""
    parquet_bytes = _make_parquet_bytes()

    def _fake_download(color, year, month, dest_dir):
        p = dest_dir / f"{color}_tripdata_{year}-{month:02d}.parquet"
        p.write_bytes(parquet_bytes)
        return p

    with (
        patch("data.ingest.tlc.download_parquet", side_effect=_fake_download),
        patch("data.ingest.tlc.write_to_db", return_value=2) as mock_write,
    ):
        ingest(dest_dir=tmp_path, months=1, colors=("yellow",), dsn="postgresql://x/y")

    mock_write.assert_called_once()
    written_df = mock_write.call_args[0][0]
    assert written_df.shape[0] > 0


def test_ingest_does_not_call_write_to_db_without_dsn(tmp_path: Path):
    """When dsn is None (default) write_to_db must never be called."""
    parquet_bytes = _make_parquet_bytes()

    def _fake_download(color, year, month, dest_dir):
        p = dest_dir / f"{color}_tripdata_{year}-{month:02d}.parquet"
        p.write_bytes(parquet_bytes)
        return p

    with (
        patch("data.ingest.tlc.download_parquet", side_effect=_fake_download),
        patch("data.ingest.tlc.write_to_db") as mock_write,
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

    mock_cur = MagicMock()
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    with (
        patch("data.ingest.tlc.download_parquet", side_effect=_fake_download),
        patch("data.ingest.tlc.psycopg2.connect", return_value=mock_conn),
    ):
        result = ingest(
            dest_dir=tmp_path,
            months=1,
            colors=("yellow",),
            dsn="postgresql://fake/fake",
        )

    assert result.shape[0] > 0
    assert mock_cur.executemany.call_count > 0
