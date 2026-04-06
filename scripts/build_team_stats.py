from pathlib import Path
import polars as pl

RAW = Path("data/raw")
CLEAN = Path("data/cleaned")
CLEAN.mkdir(parents=True, exist_ok=True)

pbp = pl.read_parquet(RAW / "pbp_2024_2025.parquet")
team_games = pl.read_parquet(CLEAN / "team_games_base.parquet")

print("Original PBP shape:", pbp.shape)

pbp_clean = (
    pbp
    .filter(pl.col("season_type") == "REG")
    .filter(pl.col("posteam").is_not_null())
    .filter(pl.col("play_type").is_in(["run", "pass"]))
    .with_columns([
        pl.col("epa").cast(pl.Float64),
        pl.col("interception").fill_null(0).cast(pl.Int64),
        pl.col("fumble_lost").fill_null(0).cast(pl.Int64),
    ])
)

print("Filtered PBP shape:", pbp_clean.shape)

team_game_stats = (
    pbp_clean
    .group_by(["game_id", "season", "week", "posteam"])
    .agg([
        pl.len().alias("offensive_plays"),
        pl.col("epa").mean().alias("off_epa_per_play"),
        (pl.col("play_type") == "pass").sum().alias("pass_plays"),
        (pl.col("play_type") == "run").sum().alias("rush_plays"),
        pl.col("interception").sum().alias("interceptions_thrown"),
        pl.col("fumble_lost").sum().alias("fumbles_lost"),
    ])
    .rename({"posteam": "team"})
    .with_columns([
        (pl.col("pass_plays") / pl.col("offensive_plays")).alias("pass_rate"),
        (pl.col("interceptions_thrown") + pl.col("fumbles_lost")).alias("turnovers"),
    ])
)

print("Team-game stats shape:", team_game_stats.shape)
print(team_game_stats.head(10))

model_base = team_games.join(
    team_game_stats,
    on=["game_id", "season", "week", "team"],
    how="left"
)

model_base.write_parquet(CLEAN / "team_games_with_stats.parquet")
model_base.write_csv(CLEAN / "team_games_with_stats.csv")

print("Merged model base shape:", model_base.shape)
print(model_base.head(10))