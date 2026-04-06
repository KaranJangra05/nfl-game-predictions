from pathlib import Path
import polars as pl

CLEAN = Path("data/cleaned")

df = pl.read_parquet(CLEAN / "team_games_with_rolling.parquet")

opp = df.select([
    "game_id",
    pl.col("team").alias("opponent"),
    pl.col("rolling_points_for_3").alias("opp_rolling_points_for_3"),
    pl.col("rolling_points_against_3").alias("opp_rolling_points_against_3"),
    pl.col("rolling_win_pct_3").alias("opp_rolling_win_pct_3"),
    pl.col("rolling_off_epa_3").alias("opp_rolling_off_epa_3"),
    pl.col("rolling_turnovers_3").alias("opp_rolling_turnovers_3"),
    pl.col("rolling_point_diff_3").alias("opp_rolling_point_diff_3"),
])

df = df.join(opp, on=["game_id", "opponent"], how="left")

df = df.with_columns([
    (pl.col("rolling_off_epa_3") - pl.col("opp_rolling_off_epa_3")).alias("epa_diff_3"),
    (pl.col("rolling_win_pct_3") - pl.col("opp_rolling_win_pct_3")).alias("win_pct_diff_3"),
    (pl.col("rolling_turnovers_3") - pl.col("opp_rolling_turnovers_3")).alias("turnover_diff_3"),
    (pl.col("rolling_point_diff_3") - pl.col("opp_rolling_point_diff_3")).alias("point_diff_gap_3"),
])

df.write_parquet(CLEAN / "model_features.parquet")
df.write_csv(CLEAN / "model_features.csv")

print("Model-features shape:", df.shape)
print(
    df.select([
        "game_id", "team", "opponent", "week", "win",
        "rolling_point_diff_3", "opp_rolling_point_diff_3", "point_diff_gap_3",
        "rolling_win_pct_3", "opp_rolling_win_pct_3", "win_pct_diff_3"
    ]).head(12)
)