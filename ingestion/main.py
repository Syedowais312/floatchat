import os
import xarray as xr
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
from sentence_transformers import SentenceTransformer
import math

DB_CONFIG = {
    "host": "localhost",
    "database": "floatchatai",
    "user": "postgres",
    "password": "Owais@786"
}

DATA_DIR = "./data"


model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-d embeddings

def get_embedding(text):
    return model.encode(text).tolist()  # store as FLOAT8[]


def ingest_nc_file(file_path, conn):
    ds = xr.open_dataset(file_path)

    for prof in range(ds.sizes["N_PROF"]):
        juld = pd.to_datetime(ds['JULD'][prof].values).to_pydatetime()
        lat = float(ds['LATITUDE'][prof].values)
        lon = float(ds['LONGITUDE'][prof].values)

        # Create profile embedding
        desc = f"Ocean profile at lat {lat}, lon {lon}, date {juld.strftime('%Y-%m-%d')}"
        emb = get_embedding(desc)

        # Insert profile
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO profiles (N_PROF, JULD, LATITUDE, LONGITUDE, embedding)
                VALUES (%s, %s, %s, %s, %s) RETURNING id
            """, (prof, juld, lat, lon, emb))
            profile_id = cur.fetchone()[0]

        # Insert depth-level rows
        level_rows = [
            (profile_id, int(level), 
             float(ds['PRES'][prof, level].values),
             float(ds['TEMP'][prof, level].values),
             float(ds['PSAL'][prof, level].values),juld)
            for level in range(ds.sizes["N_LEVELS"])
        ]
        with conn.cursor() as cur:
            execute_batch(cur, """
                INSERT INTO profile_levels (profile_id, N_LEVELS, PRES, TEMP, PSAL,juld)
                VALUES (%s, %s, %s, %s, %s,%s)
            """, level_rows, page_size=500)

        conn.commit()
        print(f"Inserted profile {prof} with {len(level_rows)} levels")

def process_all_files():
    conn = psycopg2.connect(**DB_CONFIG)
    for file in os.listdir(DATA_DIR):
        if file.endswith(".nc"):
            ingest_nc_file(os.path.join(DATA_DIR, file), conn)
    conn.close()

if __name__ == "__main__":
    process_all_files()
