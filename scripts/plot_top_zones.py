import os
import logging
import psycopg2
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_top_zones_map():
    """
    Generates a spatial distribution plot (choropleth map) of the top 10
    highest-volume NYC TLC taxi zones based on historical demand data.

    Returns:
        matplotlib.figure.Figure: The generated figure, or None if an error occurred.
    """
    # Load environment variables
    load_dotenv()
    dsn = os.getenv("TIMESCALE_DSN", "postgresql://pulsecast:pulsecast@localhost:5432/pulsecast")
    
    # Path to the shapefile
    # Resolving relative to the script location assuming it's run from the project root
    project_root = Path(__file__).resolve().parent.parent
    shapefile_path = project_root / "data" / "raw" / "taxi_zones" / "taxi_zones" / "taxi_zones.shp"

    if not shapefile_path.exists():
        logger.error(f"Shapefile not found at: {shapefile_path}")
        return None

    logger.info("Connecting to the database...")
    try:
        with psycopg2.connect(dsn) as conn:
            query = """
                SELECT route_id as "LocationID", sum(volume) as total_volume
                FROM demand
                GROUP BY route_id
                ORDER BY total_volume DESC
                LIMIT 10;
            """
            logger.info("Executing query to fetch top 10 zones...")
            with conn.cursor() as cur:
                cur.execute(query)
                columns = [desc[0] for desc in cur.description]
                data = cur.fetchall()
            df = pd.DataFrame(data, columns=columns)
    except Exception as e:
        logger.error(f"Failed to query database: {e}")
        return None
    
    if df.empty:
        logger.warning("No data returned from the database.")
        return None

    # LocationID might be int or float depending on pandas, ensure integer
    df['LocationID'] = df['LocationID'].astype(int)

    logger.info(f"Loading shapefile from {shapefile_path}...")
    try:
        gdf = gpd.read_file(shapefile_path)
    except Exception as e:
        logger.error(f"Failed to read shapefile: {e}")
        return None

    # The TLC shapefile usually has LocationID as a float/int
    gdf['LocationID'] = gdf['LocationID'].astype(int)

    logger.info("Merging data and shapefile...")
    # Merge the data
    merged = gdf.merge(df, on='LocationID', how='left')
    
    # Fill missing values with 0 so non-top-10 zones show up as well (or keep them missing to color them differently)
    merged['total_volume'] = merged['total_volume'].fillna(0)

    logger.info("Generating plot...")
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    merged.plot(
        column='total_volume',
        cmap='OrRd',
        linewidth=0.8,
        ax=ax,
        edgecolor='0.8',
        legend=True,
        legend_kwds={'label': "Total Pickup Volume"}
    )
    
    ax.set_title("Top 10 High-Volume TLC Taxi Zones", fontsize=16)
    ax.axis('off')
    
    return fig

if __name__ == "__main__":
    fig = generate_top_zones_map()
    if fig:
        output_file = "top_10_zones_map.png"
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved successfully to {output_file}")
    else:
        logger.error("Plot generation failed.")