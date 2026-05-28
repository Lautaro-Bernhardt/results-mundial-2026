"""ELO rating engine for international football teams."""
import math
import pandas as pd
from collections import defaultdict

# K-factors by tournament importance
K_FACTORS = {
    "FIFA World Cup": 60,
    "Confederations Cup": 45,
    "Copa América": 45,
    "UEFA Euro": 45,
    "Africa Cup of Nations": 40,
    "Asian Cup": 40,
    "Gold Cup": 35,
    "Copa América Centenario": 45,
    "FIFA World Cup qualification": 30,
    "UEFA Euro qualification": 25,
    "AFC Asian Cup qualification": 20,
    "CAF Africa Cup of Nations qualification": 20,
    "CONCACAF Nations League": 25,
    "UEFA Nations League": 25,
    "Friendly": 10,
}

DEFAULT_K = 15
HOME_ADVANTAGE = 100  # added to home team's ELO for expected calc
START_ELO = 1500


def _k_factor(tournament: str) -> float:
    for key, k in K_FACTORS.items():
        if key in tournament:
            return k
    return DEFAULT_K


def _gd_multiplier(gd: int) -> float:
    """Goal-difference multiplier (from ELO world ratings system)."""
    if gd <= 1:
        return 1.0
    elif gd == 2:
        return 1.5
    elif gd == 3:
        return 1.75
    else:
        return 1.75 + 0.05 * (gd - 3)


def calculate_elo(results_df: pd.DataFrame) -> dict:
    """
    Calculate ELO ratings for all teams from historical results.
    Returns dict of team_name -> elo_rating.
    """
    ratings: dict = defaultdict(lambda: START_ELO)
    results_df = results_df.sort_values("date").reset_index(drop=True)

    for _, row in results_df.iterrows():
        if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
            continue
        home = row["home_team"]
        away = row["away_team"]
        hs = int(row["home_score"])
        as_ = int(row["away_score"])
        neutral = bool(row["neutral"])
        tournament = str(row["tournament"])

        k = _k_factor(tournament)
        ha = 0 if neutral else HOME_ADVANTAGE

        ra = ratings[home] + ha
        rb = ratings[away]
        ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        eb = 1.0 - ea

        if hs > as_:
            sa, sb = 1.0, 0.0
        elif hs < as_:
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5

        gdm = _gd_multiplier(abs(hs - as_))
        ratings[home] += k * gdm * (sa - ea)
        ratings[away] += k * gdm * (sb - eb)

    return dict(ratings)


def win_probabilities(elo_a: float, elo_b: float, neutral: bool = True) -> tuple:
    """
    Return (p_win_a, p_draw, p_win_b) using a logistic ELO model.
    Draw probability peaks when teams are equal and decreases with rating gap.
    """
    ha = 0 if neutral else HOME_ADVANTAGE
    diff = (elo_a + ha) - elo_b
    p_a_no_draw = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    # Draw model: higher draw rate when teams are similar
    draw_base = 0.285
    draw_prob = draw_base * math.exp(-abs(diff) / 600.0)
    draw_prob = max(0.05, min(0.38, draw_prob))

    p_win_a = p_a_no_draw * (1.0 - draw_prob)
    p_win_b = (1.0 - p_a_no_draw) * (1.0 - draw_prob)

    return p_win_a, draw_prob, p_win_b


def load_and_calculate(results_path: str, name_map: dict | None = None) -> dict:
    """Load results CSV and compute ELO ratings, applying optional name mapping."""
    df = pd.read_csv(results_path, parse_dates=["date"])
    if name_map:
        df["home_team"] = df["home_team"].replace(name_map)
        df["away_team"] = df["away_team"].replace(name_map)
    return calculate_elo(df)


def get_recent_form(
    results_df: pd.DataFrame, team: str, n_matches: int = 20
) -> dict:
    """Return win rate and average goal diff for a team's last N matches."""
    mask = (results_df["home_team"] == team) | (results_df["away_team"] == team)
    team_df = results_df[mask].dropna(subset=["home_score", "away_score"])
    team_df = team_df.sort_values("date").tail(n_matches)

    if team_df.empty:
        return {"win_rate": 0.5, "draw_rate": 0.2, "avg_gd": 0.0, "n": 0}

    wins, draws, gd_total = 0, 0, 0
    for _, row in team_df.iterrows():
        if row["home_team"] == team:
            gd = int(row["home_score"]) - int(row["away_score"])
        else:
            gd = int(row["away_score"]) - int(row["home_score"])
        gd_total += gd
        if gd > 0:
            wins += 1
        elif gd == 0:
            draws += 1

    n = len(team_df)
    return {
        "win_rate": wins / n,
        "draw_rate": draws / n,
        "avg_gd": gd_total / n,
        "n": n,
    }


def get_wc_performance(results_df: pd.DataFrame, team: str) -> dict:
    """Return World Cup-specific performance metrics with recency weighting."""
    mask = (
        ((results_df["home_team"] == team) | (results_df["away_team"] == team))
        & (results_df["tournament"].str.contains("FIFA World Cup", na=False))
        & (~results_df["tournament"].str.contains("qualif", case=False, na=False))
    )
    wc_df = results_df[mask].dropna(subset=["home_score", "away_score"]).sort_values("date")

    if wc_df.empty:
        return {"wc_win_rate": 0.33, "wc_avg_gd": 0.0, "wc_matches": 0}

    wins, draws, gd_total = 0, 0, 0
    weights_sum = 0.0
    weighted_wins = 0.0
    weighted_gd = 0.0

    for _, row in wc_df.iterrows():
        year = row["date"].year
        weight = max(0.1, 1.0 - (2026 - year) * 0.08)
        if row["home_team"] == team:
            gd = int(row["home_score"]) - int(row["away_score"])
        else:
            gd = int(row["away_score"]) - int(row["home_score"])
        result = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
        weighted_wins += weight * result
        weighted_gd += weight * gd
        weights_sum += weight
        if gd > 0:
            wins += 1
        elif gd == 0:
            draws += 1
        gd_total += gd

    n = len(wc_df)
    return {
        "wc_win_rate": weighted_wins / weights_sum if weights_sum > 0 else 0.33,
        "wc_avg_gd": weighted_gd / weights_sum if weights_sum > 0 else 0.0,
        "wc_matches": n,
    }
