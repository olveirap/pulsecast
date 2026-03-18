from __future__ import annotations

from datetime import UTC, datetime

import pytest
from google.transit import gtfs_realtime_pb2

from pulsecast.data.ingest import gtfs_rt


def _make_feed(stop_id: str, delay: int, timestamp: int) -> gtfs_realtime_pb2.FeedMessage:
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.header.gtfs_realtime_version = "2.0"
    feed.header.timestamp = timestamp

    entity = feed.entity.add()
    entity.id = "1"
    trip_update = entity.trip_update
    trip_update.trip.trip_id = "trip-1"
    update = trip_update.stop_time_update.add()
    update.stop_id = stop_id
    update.arrival.delay = delay
    return feed


def test_load_stop_to_zone_reads_csv(tmp_path):
    csv_path = tmp_path / "stop_to_zone.csv"
    csv_path.write_text("stop_id,zone_id\n101N,10\n101S,11\n", encoding="utf-8")

    mapping = gtfs_rt._load_stop_to_zone(csv_path)

    assert mapping == {"101N": 10, "101S": 11}


def test_load_stop_to_zone_missing_file_returns_empty(tmp_path):
    mapping = gtfs_rt._load_stop_to_zone(tmp_path / "does_not_exist.csv")
    assert mapping == {}


def test_load_stop_to_zone_invalid_header_raises(tmp_path):
    csv_path = tmp_path / "stop_to_zone.csv"
    csv_path.write_text("stop,zone\n101N,10\n", encoding="utf-8")

    with pytest.raises(ValueError):
        gtfs_rt._load_stop_to_zone(csv_path)


def test_compute_delay_index_uses_loaded_mapping(monkeypatch):
    ts = int(datetime(2024, 3, 7, 12, 3, 0, tzinfo=UTC).timestamp())
    feed = _make_feed("101N", 90, ts)

    monkeypatch.setattr(gtfs_rt, "_STOP_TO_ZONE", {"101N": 7})
    rows = gtfs_rt._compute_delay_index(feed)

    assert len(rows) == 1
    assert rows[0]["zone_id"] == 7
    assert rows[0]["delay_index"] == 90


def test_compute_delay_index_drops_unmapped_stop(monkeypatch):
    ts = int(datetime(2024, 3, 7, 12, 3, 0, tzinfo=UTC).timestamp())
    feed = _make_feed("UNKNOWN", 90, ts)

    monkeypatch.setattr(gtfs_rt, "_STOP_TO_ZONE", {})
    rows = gtfs_rt._compute_delay_index(feed)

    assert rows == []
