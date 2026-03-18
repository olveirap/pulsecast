"""Build stop_id -> TLC taxi zone mapping from public GTFS + TLC spatial data.

This script:
1) Downloads MTA GTFS static feed (stops.txt).
2) Downloads TLC taxi zone polygons.
3) Spatially joins stop points to polygons.
4) Writes pulsecast/data/stop_to_zone.csv.

Usage:
    python scripts/build_stop_zone_map.py

Optional arguments:
    --gtfs-static-url URL
    --taxi-zones-url URL
    --output PATH
    --workdir PATH
"""

from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

DEFAULT_GTFS_STATIC_URL = "https://rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip"
DEFAULT_TAXI_ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
DEFAULT_OUTPUT = Path("pulsecast/data/stop_to_zone.csv")


def _download(url: str, destination: Path) -> None:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    destination.write_bytes(response.content)


def _load_stops(gtfs_zip_path: Path) -> gpd.GeoDataFrame:
    with zipfile.ZipFile(gtfs_zip_path) as zf:
        with zf.open("stops.txt") as f:
            stops = pd.read_csv(f, usecols=["stop_id", "stop_lat", "stop_lon"])
    stops = stops.dropna(subset=["stop_id", "stop_lat", "stop_lon"])
    stops_gdf = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]),
        crs="EPSG:4326",
    )
    return stops_gdf


def _load_taxi_zones(taxi_zip_path: Path) -> gpd.GeoDataFrame:
    with zipfile.ZipFile(taxi_zip_path) as zf:
        shapefiles = [name for name in zf.namelist() if name.lower().endswith(".shp")]
        if not shapefiles:
            raise ValueError("Taxi zones zip has no .shp file")
        shp_name = shapefiles[0]
        extract_dir = taxi_zip_path.parent / "taxi_zones"
        zf.extractall(extract_dir)
    zones = gpd.read_file(extract_dir / shp_name)
    if "LocationID" not in zones.columns:
        raise ValueError("Taxi zones dataset is missing required LocationID column")
    if zones.crs is None:
        zones = zones.set_crs("EPSG:4326")
    else:
        zones = zones.to_crs("EPSG:4326")
    return zones[["LocationID", "geometry"]].rename(columns={"LocationID": "zone_id"})


def _join_stops_to_zones(stops: gpd.GeoDataFrame, zones: gpd.GeoDataFrame) -> pd.DataFrame:
    joined = gpd.sjoin(stops, zones, how="left", predicate="within")

    missing = joined["zone_id"].isna()
    if missing.any():
        nearest = gpd.sjoin_nearest(
            stops[missing],
            zones,
            how="left",
            max_distance=0.01,
            distance_col="distance",
        )
        joined.loc[missing, "zone_id"] = nearest["zone_id"].to_numpy()

    result = joined[["stop_id", "zone_id"]].dropna(subset=["zone_id"]).copy()
    result["zone_id"] = result["zone_id"].astype(int)
    result = result.drop_duplicates(subset=["stop_id"]).sort_values("stop_id")
    return result


def build_mapping(
    gtfs_static_url: str,
    taxi_zones_url: str,
    output_path: Path,
    workdir: Path | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if workdir is None:
        tmp_dir_ctx = tempfile.TemporaryDirectory()
        tmp_root = Path(tmp_dir_ctx.__enter__())
    else:
        workdir.mkdir(parents=True, exist_ok=True)
        tmp_root = workdir
        tmp_dir_ctx = None

    try:
        gtfs_zip = tmp_root / "gtfs_static.zip"
        taxi_zip = tmp_root / "taxi_zones.zip"

        _download(gtfs_static_url, gtfs_zip)
        _download(taxi_zones_url, taxi_zip)

        stops = _load_stops(gtfs_zip)
        zones = _load_taxi_zones(taxi_zip)
        mapping = _join_stops_to_zones(stops, zones)

        mapping.to_csv(output_path, index=False)
        return output_path
    finally:
        if tmp_dir_ctx is not None:
            tmp_dir_ctx.__exit__(None, None, None)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build stop_id -> TLC zone mapping from GTFS static and taxi zones data."
    )
    parser.add_argument("--gtfs-static-url", default=DEFAULT_GTFS_STATIC_URL)
    parser.add_argument("--taxi-zones-url", default=DEFAULT_TAXI_ZONES_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workdir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = build_mapping(
        gtfs_static_url=args.gtfs_static_url,
        taxi_zones_url=args.taxi_zones_url,
        output_path=args.output,
        workdir=args.workdir,
    )
    print(f"Wrote stop-to-zone mapping to {output}")


if __name__ == "__main__":
    main()
