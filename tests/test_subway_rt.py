from datetime import UTC, datetime

import pulsecast.data.ingest.subway_rt as subway_rt
from pulsecast.data.ingest.subway_rt import process_delays


def test_process_delays():
    # Inject a mock zone map
    subway_rt._ZONE_MAP["101N"] = 220
    subway_rt._ZONE_MAP["102S"] = 221
    
    delays = [
        {"stop_id": "101N", "delay": 60.0},
        {"stop_id": "101N", "delay": 120.0},
        {"stop_id": "102S", "delay": 30.0},
        {"stop_id": "UNKNOWN", "delay": 500.0}, # Should be ignored because it's not in map
    ]
    
    current_hour = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
    
    agg = process_delays(1, delays, current_hour)
    
    # Assert two zones mapped
    assert len(agg) == 2
    
    z220 = agg[agg["zone_id"] == 220].iloc[0]
    assert z220["mean_delay"] == 90.0 # (60 + 120) / 2
    assert z220["trip_count"] == 2
    assert z220["feed_id"] == "1"
    
    z221 = agg[agg["zone_id"] == 221].iloc[0]
    assert z221["mean_delay"] == 30.0
    assert z221["trip_count"] == 1
    assert z221["feed_id"] == "1"
