
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from pulsecast.data.ingest.bus_positions import compute_variance


def test_compute_variance():
    # Mock bus positions DataFrame
    df = pd.DataFrame({
        "timestamp": [
            "2024-01-01T10:15:00Z",
            "2024-01-01T10:45:00Z",
            "2024-01-01T10:30:00Z",
            "2024-01-01T11:15:00Z"
        ],
        "latitude": [40.7128, 40.7130, 40.7129, 40.7500],
        "longitude": [-74.0060, -74.0062, -74.0061, -73.9800],
        "segment_travel_time": [120, 130, 125, 200]
    })

    # Mock TLC zones GeoDataFrame
    # Create a polygon that encompasses the first three points
    zone_1_poly = Point(-74.0060, 40.7128).buffer(0.01)
    # Create a polygon for the fourth point
    zone_2_poly = Point(-73.9800, 40.7500).buffer(0.01)

    zones = gpd.GeoDataFrame({
        "zone_id": [1, 2],
        "geometry": [zone_1_poly, zone_2_poly]
    }, crs="EPSG:4326")

    agg = compute_variance(df, zones)
    
    # 2 groups should be formed: 
    # (zone 1, hour 10) - 3 records -> var = 25.0, count = 3
    # (zone 2, hour 11) - 1 record -> var = 0.0, count = 1
    
    assert len(agg) == 2
    
    z1 = agg[agg["zone_id"] == 1].iloc[0]
    assert z1["sample_count"] == 3
    assert np.isclose(z1["travel_time_var"], 25.0)  # var([120, 125, 130]) = 25.0
    
    z2 = agg[agg["zone_id"] == 2].iloc[0]
    assert z2["sample_count"] == 1
    assert z2["travel_time_var"] == 0.0
