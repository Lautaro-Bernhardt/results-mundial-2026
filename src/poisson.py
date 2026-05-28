"""
Dixon-Coles bivariate Poisson model for international football.

Functions:
    compute_team_strengths  - Exponentially weighted attack/defense estimates
    match_proba             - Bivariate Poisson + DC correction
    calibrate_rho           - Optimise rho on recent non-friendly matches
"""

import math
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize_scalar
import pandas as pd


def compute_team_strengths(
    results_df: pd.DataFrame,
    teams_list=None,
    half_life_days: float = 548,
    min_weight: float = 5.0,
) -> tuple:
    """
    Compute exponentially-weighted attack and defense strengths for each team.

    Parameters
    ----------
    results_df   : DataFrame with columns date, home_team, away_team,
                   home_score, away_score
    teams_list   : list of team names to compute strengths for. If None,
                   uses all teams in the data.
    half_life_days : half-life for recency weighting (default 548 ≈ 1.5 years)
    min_weight   : minimum cumulative weight needed for full confidence.
                   Teams with less weight are shrunk towards league average.

    Returns
    -------
    (strengths_dict, global_avg_goals)
        strengths_dict maps team -> {"attack": float, "defense": float,
                                      "n_matches": int}
        global_avg_goals is the weighted average goals per game
    """
    df = results_df.dropna(subset=["home_score", "away_score"]).copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Reference date: most recent match in dataset
    latest_date = pd.to_datetime(df["date"].max())
    decay_rate = math.log(2) / half_life_days

    # Add weight column
    dates = pd.to_datetime(df["date"])
    days_ago = (latest_date - dates).dt.days.clip(lower=0).values
    df["_w"] = np.exp(-decay_rate * days_ago)

    # Accumulate weighted goals for each team (as home and away)
    from collections import defaultdict
    scored_w = defaultdict(float)
    conceded_w = defaultdict(float)
    games_w = defaultdict(float)
    n_matches = defaultdict(int)

    for _, row in df.iterrows():
        w = row["_w"]
        home = row["home_team"]
        away = row["away_team"]
        hs = float(row["home_score"])
        as_ = float(row["away_score"])

        scored_w[home] += w * hs
        conceded_w[home] += w * as_
        games_w[home] += w
        n_matches[home] += 1

        scored_w[away] += w * as_
        conceded_w[away] += w * hs
        games_w[away] += w
        n_matches[away] += 1

    # Global weighted average goals per game
    total_w_goals = sum(scored_w.values())
    total_w_games = sum(games_w.values())
    global_avg = total_w_goals / total_w_games if total_w_games > 0 else 1.3

    # Build strength dict
    all_teams = set(scored_w.keys())
    if teams_list:
        all_teams = all_teams | set(teams_list)

    strengths = {}
    for team in all_teams:
        w_sum = games_w.get(team, 0.0)
        if w_sum > 0:
            avg_scored = scored_w[team] / w_sum
            avg_conceded = conceded_w[team] / w_sum
            attack = avg_scored / global_avg if global_avg > 0 else 1.0
            defense = avg_conceded / global_avg if global_avg > 0 else 1.0
        else:
            attack = 1.0
            defense = 1.0

        # Shrink towards 1.0 for teams with insufficient data
        shrinkage = min(1.0, w_sum / min_weight)
        if shrinkage < 1.0:
            attack = shrinkage * attack + (1.0 - shrinkage) * 1.0
            defense = shrinkage * defense + (1.0 - shrinkage) * 1.0

        strengths[team] = {
            "attack": attack,
            "defense": defense,
            "n_matches": n_matches.get(team, 0),
        }

    return strengths, global_avg


def match_proba(
    attack_a: float,
    defense_b: float,
    attack_b: float,
    defense_a: float,
    global_avg: float,
    rho: float = -0.08,
    max_goals: int = 10,
) -> tuple:
    """
    Compute P(home win), P(draw), P(away win) using Dixon-Coles bivariate Poisson.

    Parameters
    ----------
    attack_a   : attack strength of team A (home)
    defense_b  : defensive weakness of team B (higher = leakier)
    attack_b   : attack strength of team B (away)
    defense_a  : defensive weakness of team A
    global_avg : global average goals per game
    rho        : Dixon-Coles correlation parameter (typically -0.1 to 0)
    max_goals  : maximum goals per team to consider

    Returns
    -------
    (p_home_win, p_draw, p_away_win)
    """
    lambda_a = max(0.1, attack_a * defense_b * global_avg)
    lambda_b = max(0.1, attack_b * defense_a * global_avg)

    goals = np.arange(max_goals + 1)
    pmf_a = poisson.pmf(goals, lambda_a)  # shape: (max_goals+1,)
    pmf_b = poisson.pmf(goals, lambda_b)

    # Build joint probability matrix
    joint = np.outer(pmf_a, pmf_b)  # joint[i, j] = P(A scores i, B scores j)

    # Dixon-Coles correction for low scores (i, j <= 1)
    def _dc(i, j):
        if i == 0 and j == 0:
            return 1.0 - lambda_a * lambda_b * rho
        elif i == 1 and j == 0:
            return 1.0 + lambda_b * rho
        elif i == 0 and j == 1:
            return 1.0 + lambda_a * rho
        elif i == 1 and j == 1:
            return 1.0 - rho
        return 1.0

    for i in range(min(2, max_goals + 1)):
        for j in range(min(2, max_goals + 1)):
            joint[i, j] *= _dc(i, j)

    # Normalise
    total = joint.sum()
    if total > 0:
        joint /= total

    # Extract match outcome probabilities
    # Home win: i > j (A scores more)
    # Draw: i == j
    # Away win: j > i
    n = max_goals + 1
    idx_i, idx_j = np.meshgrid(goals, goals, indexing="ij")
    p_home_win = float(joint[idx_i > idx_j].sum())
    p_draw = float(joint[idx_i == idx_j].sum())
    p_away_win = float(joint[idx_i < idx_j].sum())

    # Safety clamp
    total_out = p_home_win + p_draw + p_away_win
    if total_out > 0:
        p_home_win /= total_out
        p_draw /= total_out
        p_away_win /= total_out
    else:
        p_home_win, p_draw, p_away_win = 1 / 3, 1 / 3, 1 / 3

    return p_home_win, p_draw, p_away_win


def calibrate_rho(
    results_df: pd.DataFrame,
    teams_list=None,
    rho_range: tuple = (-0.15, 0.05),
) -> float:
    """
    Find rho that minimises Brier score on recent non-friendly matches (last 5 years).

    Parameters
    ----------
    results_df : full historical results DataFrame
    teams_list : optional list of teams (unused here, kept for API consistency)
    rho_range  : (min_rho, max_rho) bounds for search

    Returns
    -------
    Optimal rho value (float).
    """
    df = results_df.dropna(subset=["home_score", "away_score"]).copy()
    df = df[~df["tournament"].str.contains("Friendly", case=False, na=False)]

    # Last 5 years
    latest = pd.to_datetime(df["date"].max())
    cutoff = latest - pd.Timedelta(days=5 * 365)
    recent = df[pd.to_datetime(df["date"]) >= cutoff].copy()

    if len(recent) < 50:
        # Fallback: use all non-friendly if not enough recent data
        recent = df.copy()

    # Compute strengths on the full dataset for calibration
    strengths, global_avg = compute_team_strengths(results_df, teams_list)

    def brier_for_rho(rho_val):
        total_brier = 0.0
        count = 0
        for _, row in recent.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            hs = int(row["home_score"])
            as_ = int(row["away_score"])

            sh = strengths.get(home, {"attack": 1.0, "defense": 1.0})
            sa = strengths.get(away, {"attack": 1.0, "defense": 1.0})

            try:
                p_h, p_d, p_a = match_proba(
                    sh["attack"], sa["defense"],
                    sa["attack"], sh["defense"],
                    global_avg, rho=rho_val
                )
            except Exception:
                continue

            if hs > as_:
                actual = (1.0, 0.0, 0.0)
            elif hs == as_:
                actual = (0.0, 1.0, 0.0)
            else:
                actual = (0.0, 0.0, 1.0)

            total_brier += sum((p - a) ** 2 for p, a in zip([p_h, p_d, p_a], actual))
            count += 1

        return total_brier / count if count > 0 else 999.0

    result = minimize_scalar(
        brier_for_rho,
        bounds=rho_range,
        method="bounded",
        options={"xatol": 1e-4},
    )
    return float(result.x)
