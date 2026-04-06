from pathlib import Path
import polars as pl
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CLEAN = Path("data/cleaned")
OUT = Path("outputs/model_results")
OUT.mkdir(parents=True, exist_ok=True)

df = pl.read_parquet(CLEAN / "model_features.parquet")

all_feature_cols = [
    "is_home",
    "rest_days",
    "rolling_points_for_3",
    "rolling_points_against_3",
    "rolling_win_pct_3",
    "rolling_off_epa_3",
    "rolling_turnovers_3",
    "rolling_point_diff_3",
    "opp_rolling_points_for_3",
    "opp_rolling_points_against_3",
    "opp_rolling_win_pct_3",
    "opp_rolling_off_epa_3",
    "opp_rolling_turnovers_3",
    "opp_rolling_point_diff_3",
    "epa_diff_3",
    "win_pct_diff_3",
    "turnover_diff_3",
    "point_diff_gap_3",
]

keep_cols = ["season", "week", "team", "opponent", "game_id", "win"] + all_feature_cols
df = df.select(keep_cols)

required_rolling_cols = [
    "rolling_points_for_3",
    "rolling_points_against_3",
    "rolling_win_pct_3",
    "rolling_off_epa_3",
    "rolling_turnovers_3",
    "rolling_point_diff_3",
    "opp_rolling_points_for_3",
    "opp_rolling_points_against_3",
    "opp_rolling_win_pct_3",
    "opp_rolling_off_epa_3",
    "opp_rolling_turnovers_3",
    "opp_rolling_point_diff_3",
]
df = df.drop_nulls(subset=required_rolling_cols)

pdf = df.to_pandas()

train = pdf[pdf["season"] == 2024].copy()
test = pdf[pdf["season"] == 2025].copy()

y_train = train["win"]
y_test = test["win"]

feature_sets = {
    "win_pct_only": ["win_pct_diff_3"],
    "win_pct_pointdiff_home": ["win_pct_diff_3", "point_diff_gap_3", "is_home"],
    "epa_home_winpct": ["win_pct_diff_3", "epa_diff_3", "is_home"],
    "small_set": ["win_pct_diff_3", "epa_diff_3", "point_diff_gap_3", "is_home", "rest_days"],
    "full_model": all_feature_cols,
}

results = []
saved_outputs = {}

for model_name, feature_cols in feature_sets.items():
    X_train = train[feature_cols]
    X_test = test[feature_cols]

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    pred_probs = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, preds)

    results.append({
        "model_name": model_name,
        "n_features": len(feature_cols),
        "accuracy": accuracy
    })

    saved_outputs[model_name] = {
        "feature_cols": feature_cols,
        "pipeline": pipeline,
        "preds": preds,
        "pred_probs": pred_probs
    }

results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
results_df.to_csv(OUT / "model_comparison.csv", index=False)

print("\nModel comparison:")
print(results_df)

home_baseline_acc = (test["is_home"] == test["win"]).mean()
winpct_baseline_preds = (test["win_pct_diff_3"] > 0).astype(int)
winpct_baseline_acc = (winpct_baseline_preds == y_test).mean()

print("\nBenchmarks:")
print("Home-team baseline accuracy:", round(home_baseline_acc, 4))
print("Win pct diff baseline accuracy:", round(winpct_baseline_acc, 4))

best_model_name = results_df.iloc[0]["model_name"]
best_info = saved_outputs[best_model_name]
best_pipeline = best_info["pipeline"]
best_feature_cols = best_info["feature_cols"]
best_preds = best_info["preds"]
best_pred_probs = best_info["pred_probs"]

best_accuracy = accuracy_score(y_test, best_preds)
best_report = classification_report(y_test, best_preds)
best_cm = confusion_matrix(y_test, best_preds)

print(f"\nBest model: {best_model_name}")
print("Best model accuracy:", round(best_accuracy, 4))
print("\nClassification report:\n", best_report)
print("\nConfusion matrix:\n", best_cm)

test_results = test[["season", "week", "game_id", "team", "opponent", "win"]].copy()
test_results["pred_win"] = best_preds
test_results["pred_prob"] = best_pred_probs
test_results["model_name"] = best_model_name
test_results.to_csv(OUT / "best_model_test_predictions.csv", index=False)

coef_df = pd.DataFrame({
    "feature": best_feature_cols,
    "coefficient": best_pipeline.named_steps["model"].coef_[0]
}).sort_values("coefficient", ascending=False)

coef_df.to_csv(OUT / "best_model_coefficients.csv", index=False)

print("\nTop positive coefficients:")
print(coef_df.head(10))

print("\nTop negative coefficients:")
print(coef_df.tail(10))