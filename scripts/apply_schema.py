import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()
dsn = os.getenv("TIMESCALE_DSN", "postgresql://pulsecast:pulsecast@localhost:5432/pulsecast")

def apply_schema():
    try:
        with psycopg2.connect(dsn) as conn:
            with conn.cursor() as cur:
                with open("pulsecast/data/schema.sql", "r") as f:
                    schema_sql = f.read()
                cur.execute(schema_sql)
                print("Schema applied successfully.")
    except Exception as e:
        print(f"Error applying schema: {e}")

if __name__ == "__main__":
    apply_schema()
