from pathlib import Path
import polars as pl
from sqlalchemy import create_engine

CLEAN = Path("data/cleaned")

DB_USER = "karanjangra"
DB_NAME = "nfl_analytics"
DB_HOST = "localhost"
DB_PORT = "5432"

engine = create_engine(f"postgresql+psycopg2://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

files_to_tables = {
    "games.parquet": "games",
    "team_games_base.parquet": "team_games_base",
    "team_games_with_stats.parquet": "team_games_with_stats",
    "team_games_with_rolling.parquet": "team_games_with_rolling",
    "model_features.parquet": "model_features",
}

for filename, table_name in files_to_tables.items():
    file_path = CLEAN / filename
    print(f"Loading {filename} into {table_name}...")
    
    df = pl.read_parquet(file_path).to_pandas()
    df.to_sql(table_name, engine, if_exists="append", index=False)
    
    print(f"Loaded {len(df)} rows into {table_name}")

print("All tables loaded successfully.")