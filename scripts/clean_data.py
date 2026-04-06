from pathlib import Path
import polars as pl

RAW = Path("data/raw")
CLEAN = Path("data/cleaned")
CLEAN.mkdir(parents=True, exist_ok=True)

schedules = pl.read_parquet(RAW / "schedules_2024_2025.parquet")

games = (
    schedules
    .filter(pl.col("game_type") == "REG")
    .select([
        "game_id",
        "season",
        "week",
        "gameday",
        "away_team",
        "away_score",
        "home_team",
        "home_score",
        "away_rest",
        "home_rest"
    ])
    .unique()
)

games.write_parquet(CLEAN / "games.parquet")
games.write_csv(CLEAN / "games.csv")

home = (
    games
    .select([
        "game_id",
        "season",
        "week",
        "gameday",
        pl.col("home_team").alias("team"),
        pl.col("away_team").alias("opponent"),
        pl.col("home_score").alias("points_for"),
        pl.col("away_score").alias("points_against"),
        pl.col("home_rest").alias("rest_days")
    ])
    .with_columns(pl.lit(1).alias("is_home"))
)

away = (
    games
    .select([
        "game_id",
        "season",
        "week",
        "gameday",
        pl.col("away_team").alias("team"),
        pl.col("home_team").alias("opponent"),
        pl.col("away_score").alias("points_for"),
        pl.col("home_score").alias("points_against"),
        pl.col("away_rest").alias("rest_days")
    ])
    .with_columns(pl.lit(0).alias("is_home"))
)

team_games = (
    pl.concat([home, away])
    .with_columns(
        (pl.col("points_for") > pl.col("points_against")).cast(pl.Int8).alias("win")
    )
    .sort(["season", "week", "game_id", "team"])
)

team_games.write_parquet(CLEAN / "team_games_base.parquet")
team_games.write_csv(CLEAN / "team_games_base.csv")

print("Games shape:", games.shape)
print("Team-games shape:", team_games.shape)
print(team_games.head(10))