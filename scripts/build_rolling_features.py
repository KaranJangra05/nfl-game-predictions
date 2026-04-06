from pathlib import Path
import polars as pl

CLEAN = Path("data/cleaned")

df = pl.read_parquet(CLEAN / "team_games_with_stats.parquet")

df = (
    df.with_columns(
        pl.col("gameday").str.to_date(strict=False)
    )
    .sort(["team", "gameday", "game_id"])
)

df = df.with_columns([
    (pl.col("points_for") - pl.col("points_against")).alias("point_diff")
])

df = df.with_columns([
    pl.col("points_for").shift(1).over("team").alias("prev_points_for"),
    pl.col("points_against").shift(1).over("team").alias("prev_points_against"),
    pl.col("win").shift(1).over("team").alias("prev_win"),
    pl.col("off_epa_per_play").shift(1).over("team").alias("prev_off_epa_per_play"),
    pl.col("turnovers").shift(1).over("team").alias("prev_turnovers"),
    pl.col("point_diff").shift(1).over("team").alias("prev_point_diff"),
])

df = df.with_columns([
    pl.col("points_for").shift(1).rolling_mean(window_size=3).over("team").alias("rolling_points_for_3"),
    pl.col("points_against").shift(1).rolling_mean(window_size=3).over("team").alias("rolling_points_against_3"),
    pl.col("win").shift(1).rolling_mean(window_size=3).over("team").alias("rolling_win_pct_3"),
    pl.col("off_epa_per_play").shift(1).rolling_mean(window_size=3).over("team").alias("rolling_off_epa_3"),
    pl.col("turnovers").shift(1).rolling_mean(window_size=3).over("team").alias("rolling_turnovers_3"),
    pl.col("point_diff").shift(1).rolling_mean(window_size=3).over("team").alias("rolling_point_diff_3"),
])

df.write_parquet(CLEAN / "team_games_with_rolling.parquet")
df.write_csv(CLEAN / "team_games_with_rolling.csv")

print("Rolling-feature shape:", df.shape)
print(
    df.select([
        "game_id", "team", "week", "gameday", "win",
        "rolling_win_pct_3", "rolling_off_epa_3",
        "rolling_turnovers_3", "rolling_point_diff_3"
    ]).head(12)
)