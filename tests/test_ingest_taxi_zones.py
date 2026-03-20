import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest
import requests
from shapely.geometry import Polygon

# We'll import the function from the module once we've applied the fix,
# but for the "replicate" part, we can define a testable version here
# that matches the logic we WANT to have.

def load_taxi_zones_logic(url: str) -> gpd.GeoDataFrame:
    """The fixed logic we want to implement."""
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    
    # Use mkstemp to get a path that we can close before geopandas opens it
    fd, path = tempfile.mkstemp(suffix=".zip")
    tmp_path = Path(path)
    try:
        with os.fdopen(fd, 'wb') as tmp:
            tmp.write(resp.content)
        
        # Read the local file
        zones = gpd.read_file(tmp_path)
        
        if "LocationID" not in zones.columns:
            if "objectid" in zones.columns:
                zones = zones.rename(columns={"objectid": "zone_id"})
        else:
            zones = zones.rename(columns={"LocationID": "zone_id"})
        
        return zones[["zone_id", "geometry"]]
    finally:
        if tmp_path.exists():
            os.remove(tmp_path)

@patch("requests.get")
@patch("geopandas.read_file")
def test_load_taxi_zones_mocked(mock_read_file, mock_get):
    """Verify that the loading logic works with mocks."""
    # Mock network response
    mock_resp = MagicMock()
    mock_resp.content = b"fake zip content"
    mock_get.return_value = mock_resp
    
    # Mock geopandas reading a file and returning a GDF
    mock_gdf = gpd.GeoDataFrame({
        "LocationID": [1, 2],
        "geometry": [Polygon([(0, 0), (1, 0), (1, 1)]), Polygon([(1, 1), (2, 1), (2, 2)])]
    })
    mock_read_file.return_value = mock_gdf
    
    url = "https://example.com/taxi_zones.zip"
    result = load_taxi_zones_logic(url)
    
    # Assertions
    assert len(result) == 2
    assert "zone_id" in result.columns
    assert list(result["zone_id"]) == [1, 2]
    mock_get.assert_called_once_with(url, timeout=120)
    # Ensure it called read_file with a local path
    args, kwargs = mock_read_file.call_args
    assert isinstance(args[0], Path)
    assert str(args[0]).endswith(".zip")

if __name__ == "__main__":
    pytest.main([__file__])
