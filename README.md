# NFL Game Outcome Prediction Pipeline by Karan Jangra

## Overview
This project builds an end-to-end analytics pipeline using public NFL data from nflverse for the 2024 and 2025 regular seasons. The goal is to predict NFL game outcomes using pre-game team performance indicators.

Link to Video Demo: https://youtu.be/8V798OWAyCg

## Question
Which pre-game indicators are the strongest predictors of NFL game outcomes?

## Task
**Prediction**

## Tools
- Python
- PostgreSQL
- Tableau
- Notebook

## Data
Source: nflverse / nflreadpy

Raw data used:
- schedules/results
- play-by-play data

## Pipeline
1. Acquire raw NFL data  
2. Clean and reshape schedule data into team-game rows  
3. Aggregate play-by-play into team-level game stats  
4. Create rolling pre-game features  
5. Create opponent matchup features  
6. Train and compare logistic regression models  
7. Store cleaned data in PostgreSQL  
8. Report results with visuals and dashboard

## Main Outputs
Cleaned data:
- `games.parquet`
- `team_games_base.parquet`
- `team_games_with_stats.parquet`
- `team_games_with_rolling.parquet`
- `model_features.parquet`

Model outputs:
- `model_comparison.csv`
- `best_model_test_predictions.csv`
- `best_model_coefficients.csv`

## Best Result
Best model: `win_pct_only`  
Best test accuracy: **63.24%**

Benchmarks:
- Home-team baseline: **53.83%**
- Win percentage differential baseline: **63.24%**

## Main Finding
Recent 3-game win percentage differential was the strongest predictor among the features tested. More complex feature sets did not improve test accuracy.

## Files
- `data_dictionary.csv` for variable definitions
- `outputs/figures/` for EDA visuals
- `outputs/tables/` for summary tables
- `outputs/model_results/` for model results

## Run Order
```bash
python scripts/acquire_data.py
python scripts/clean_data.py
python scripts/build_team_stats.py
python scripts/build_rolling_features.py
python scripts/build_model_features.py
python scripts/train_model.py
psql -d nfl_analytics -f sql/create_tables.sql
python scripts/load_to_postgres.py
psql -d nfl_analytics -f sql/queries.sql

