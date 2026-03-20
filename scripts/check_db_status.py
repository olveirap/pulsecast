import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()
dsn = os.getenv("TIMESCALE_DSN", "postgresql://pulsecast:pulsecast@localhost:5432/pulsecast")

def check_db():
    try:
        with psycopg2.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """)
                tables = [t[0] for t in cur.fetchall()]
                print(f"Tables in database: {tables}")
                
                for table in ["demand", "congestion", "delay_index"]:
                    if table in tables:
                        cur.execute(f"SELECT count(*) FROM {table}")
                        count = cur.fetchone()[0]
                        print(f"Table '{table}' row count: {count}")
                        if count > 0:
                            cur.execute(f"SELECT MIN(hour), MAX(hour) FROM {table}")
                            min_h, max_h = cur.fetchone()
                            print(f"  Range: {min_h} to {max_h}")
                    else:
                        print(f"Table '{table}' does NOT exist.")
                        
    except Exception as e:
        print(f"Error connecting to DB: {e}")

if __name__ == "__main__":
    check_db()
