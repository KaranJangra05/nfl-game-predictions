-- home vs away win rate
SELECT
    is_home,
    ROUND(AVG(win::numeric), 4) AS win_rate,
    COUNT(*) AS games
FROM model_features
GROUP BY is_home
ORDER BY is_home;

-- average recent win pct differential for wins vs losses
SELECT
    win,
    ROUND(AVG(win_pct_diff_3::numeric), 4) AS avg_win_pct_diff_3,
    COUNT(*) AS games
FROM model_features
WHERE win_pct_diff_3 IS NOT NULL
GROUP BY win
ORDER BY win;

-- average EPA differential for wins vs losses
SELECT
    win,
    ROUND(AVG(epa_diff_3::numeric), 4) AS avg_epa_diff_3,
    COUNT(*) AS games
FROM model_features
WHERE epa_diff_3 IS NOT NULL
GROUP BY win
ORDER BY win;

-- team-level average win rate
SELECT
    team,
    ROUND(AVG(win::numeric), 4) AS avg_win_rate,
    COUNT(*) AS team_games
FROM model_features
GROUP BY team
ORDER BY avg_win_rate DESC, team_games DESC;

-- games where recent-performance signal was strongest
SELECT
    season,
    week,
    game_id,
    team,
    opponent,
    win,
    ROUND(win_pct_diff_3::numeric, 4) AS win_pct_diff_3
FROM model_features
WHERE win_pct_diff_3 IS NOT NULL
ORDER BY ABS(win_pct_diff_3) DESC
LIMIT 20;