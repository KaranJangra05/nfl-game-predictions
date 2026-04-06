from pathlib import Path
import os

print("Current directory:", os.getcwd())

DATA_RAW = Path("data/raw")
DATA_RAW.mkdir(parents=True, exist_ok=True)
print("Saving to folder:", DATA_RAW.resolve())

SEASONS = [2024, 2025]

import nflreadpy as nfl

print("Loading schedules...")
schedules = nfl.load_schedules(SEASONS)
print("Schedules loaded:", schedules.shape)

print("Loading pbp...")
pbp = nfl.load_pbp(SEASONS)
print("PBP loaded:", pbp.shape)

schedules.write_parquet(DATA_RAW / "schedules_2024_2025.parquet")
pbp.write_parquet(DATA_RAW / "pbp_2024_2025.parquet")

print("Done.")