import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()
dsn = os.getenv("TIMESCALE_DSN", "postgresql://pulsecast:pulsecast@localhost:5432/pulsecast")

def check_monthly_demand():
    try:
        with psycopg2.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT date_trunc('month', hour) as month, count(*) as count
                    FROM demand
                    WHERE hour >= '2024-01-01' AND hour < '2026-04-01'
                    GROUP BY 1
                    ORDER BY 1
                """)
                rows = cur.fetchall()
                print("Monthly Demand Counts:")
                for month, count in rows:
                    print(f"  {month}: {count}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_monthly_demand()
