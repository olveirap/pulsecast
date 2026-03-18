"""
tests/test_gtfs_rt_backfill.py – Unit tests for the GTFS-RT backfill module.

All S3 and database calls are mocked so that the tests run offline without any
AWS credentials or TimescaleDB instance.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------
from data.ingest.gtfs_rt_backfill import (
    _date_range,
    _parse_args,
    _s3_key,
    backfill,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_feed_bytes(timestamp: int, delays: list[tuple[str, int]]) -> bytes:
    """Build a minimal GTFS-RT FeedMessage protobuf byte string.

    Args:
        timestamp: Unix timestamp to embed in the feed header.
        delays:    List of (stop_id, delay_seconds) tuples.

    Returns:
        Serialised protobuf bytes.
    """
    from google.transit import gtfs_realtime_pb2

    feed = gtfs_realtime_pb2.FeedMessage()
    feed.header.gtfs_realtime_version = "2.0"
    feed.header.timestamp = timestamp

    for i, (stop_id, delay) in enumerate(delays):
        entity = feed.entity.add()
        entity.id = str(i)
        tu = entity.trip_update
        tu.trip.trip_id = f"trip-{i}"
        stu = tu.stop_time_update.add()
        stu.stop_id = stop_id
        stu.arrival.delay = delay

    return feed.SerializeToString()


# ---------------------------------------------------------------------------
# _date_range
# ---------------------------------------------------------------------------


def test_date_range_single_day():
    d = date(2024, 1, 15)
    assert _date_range(d, d) == [d]


def test_date_range_multiple_days():
    start = date(2024, 1, 1)
    end = date(2024, 1, 3)
    result = _date_range(start, end)
    assert result == [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]


def test_date_range_empty_when_start_after_end():
    assert _date_range(date(2024, 1, 5), date(2024, 1, 1)) == []


# ---------------------------------------------------------------------------
# _s3_key
# ---------------------------------------------------------------------------


def test_s3_key_no_prefix():
    key = _s3_key(date(2024, 3, 7), 14)
    assert key == "2024/03/07/14/gtfs_rt.pb"


def test_s3_key_with_prefix(monkeypatch):
    import data.ingest.gtfs_rt_backfill as mod

    monkeypatch.setattr(mod, "_PREFIX", "feeds/mta")
    key = mod._s3_key(date(2024, 3, 7), 9)
    assert key == "feeds/mta/2024/03/07/09/gtfs_rt.pb"


def test_s3_key_zero_padded():
    key = _s3_key(date(2024, 1, 5), 3)
    assert "01/05/03" in key


# ---------------------------------------------------------------------------
# _fetch_archive
# ---------------------------------------------------------------------------


def test_fetch_archive_returns_feed_on_success():
    from data.ingest.gtfs_rt_backfill import _fetch_archive

    ts = int(datetime(2024, 3, 7, 14, 0, 0, tzinfo=UTC).timestamp())
    payload = _make_feed_bytes(ts, [("101", 30)])

    s3_mock = MagicMock()
    s3_mock.get_object.return_value = {"Body": MagicMock(read=MagicMock(return_value=payload))}

    feed = _fetch_archive(s3_mock, "2024/03/07/14/gtfs_rt.pb")
    assert feed is not None
    assert feed.header.timestamp == ts


def test_fetch_archive_returns_none_on_missing_key():
    from botocore.exceptions import ClientError

    from data.ingest.gtfs_rt_backfill import _fetch_archive

    s3_mock = MagicMock()
    error_response = {"Error": {"Code": "NoSuchKey", "Message": "Not found"}}
    s3_mock.get_object.side_effect = ClientError(error_response, "GetObject")

    feed = _fetch_archive(s3_mock, "2024/03/07/14/gtfs_rt.pb")
    assert feed is None


def test_fetch_archive_injects_timestamp_when_zero():
    from data.ingest.gtfs_rt_backfill import _fetch_archive

    # Build a feed with timestamp=0
    payload = _make_feed_bytes(0, [("101", 10)])

    s3_mock = MagicMock()
    s3_mock.get_object.return_value = {"Body": MagicMock(read=MagicMock(return_value=payload))}

    feed = _fetch_archive(s3_mock, "2024/03/07/14/gtfs_rt.pb")
    assert feed is not None
    expected_ts = int(datetime(2024, 3, 7, 14, 0, 0, tzinfo=UTC).timestamp())
    assert feed.header.timestamp == expected_ts


def test_fetch_archive_returns_none_when_timestamp_unparseable():
    from data.ingest.gtfs_rt_backfill import _fetch_archive

    # Feed with timestamp=0 and a key that has no parseable date components.
    payload = _make_feed_bytes(0, [("101", 10)])

    s3_mock = MagicMock()
    s3_mock.get_object.return_value = {"Body": MagicMock(read=MagicMock(return_value=payload))}

    # Key with too few path segments – cannot extract YYYY/MM/DD/HH.
    feed = _fetch_archive(s3_mock, "gtfs_rt.pb")
    assert feed is None


# ---------------------------------------------------------------------------
# backfill (integration-style with all external calls mocked)
# ---------------------------------------------------------------------------


def test_backfill_dry_run_does_not_upsert():
    """In dry-run mode, _upsert_rows must never be called."""
    ts = int(datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC).timestamp())
    payload = _make_feed_bytes(ts, [("101", 60), ("201", 120)])

    s3_mock = MagicMock()
    s3_mock.get_object.return_value = {"Body": MagicMock(read=MagicMock(return_value=payload))}

    mock_session = MagicMock()
    mock_session.client.return_value = s3_mock

    with (
        patch("data.ingest.gtfs_rt_backfill.boto3.Session", return_value=mock_session),
        patch("data.ingest.gtfs_rt_backfill._upsert_rows") as mock_upsert,
    ):
        total = backfill(date(2024, 1, 1), date(2024, 1, 1), dry_run=True)

    mock_upsert.assert_not_called()
    # 24 hours × at least 1 row per call where data is available
    assert total > 0


def test_backfill_calls_upsert_when_not_dry_run():
    """Without dry-run, _upsert_rows must be called for each non-empty result."""
    ts = int(datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC).timestamp())
    payload = _make_feed_bytes(ts, [("101", 30)])

    s3_mock = MagicMock()
    s3_mock.get_object.return_value = {"Body": MagicMock(read=MagicMock(return_value=payload))}

    mock_session = MagicMock()
    mock_session.client.return_value = s3_mock

    with (
        patch("data.ingest.gtfs_rt_backfill.boto3.Session", return_value=mock_session),
        patch("data.ingest.gtfs_rt_backfill._upsert_rows") as mock_upsert,
    ):
        total = backfill(date(2024, 1, 1), date(2024, 1, 1), dry_run=False)

    assert mock_upsert.call_count > 0
    assert total > 0


def test_backfill_skips_missing_objects():
    """Missing S3 keys (NoSuchKey) should be silently skipped."""
    from botocore.exceptions import ClientError

    s3_mock = MagicMock()
    error_response = {"Error": {"Code": "NoSuchKey", "Message": "Not found"}}
    s3_mock.get_object.side_effect = ClientError(error_response, "GetObject")

    mock_session = MagicMock()
    mock_session.client.return_value = s3_mock

    with (
        patch("data.ingest.gtfs_rt_backfill.boto3.Session", return_value=mock_session),
        patch("data.ingest.gtfs_rt_backfill._upsert_rows") as mock_upsert,
    ):
        total = backfill(date(2024, 1, 1), date(2024, 1, 1), dry_run=False)

    mock_upsert.assert_not_called()
    assert total == 0


# ---------------------------------------------------------------------------
# _parse_args
# ---------------------------------------------------------------------------


def test_parse_args_defaults():
    args = _parse_args(["--start", "2023-01-01", "--end", "2023-12-31"])
    assert args.start == date(2023, 1, 1)
    assert args.end == date(2023, 12, 31)
    assert args.profile is None
    assert args.dry_run is False


def test_parse_args_default_end_is_date(monkeypatch):
    """--end default must be a date object, not a string, so comparisons work."""
    args = _parse_args(["--start", "2023-01-01"])
    assert isinstance(args.end, date)


def test_parse_args_explicit_end():
    args = _parse_args(["--start", "2023-01-01", "--end", "2023-06-30"])
    assert args.end == date(2023, 6, 30)


def test_parse_args_dry_run_flag():
    args = _parse_args(["--start", "2023-01-01", "--dry-run"])
    assert args.dry_run is True


def test_parse_args_profile():
    args = _parse_args(["--start", "2023-01-01", "--profile", "my-profile"])
    assert args.profile == "my-profile"
