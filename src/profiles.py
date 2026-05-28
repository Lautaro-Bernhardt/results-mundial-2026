"""
Probability profiles for World Cup 2026 match predictions.

Each profile estimates P(team_a wins), P(draw), P(team_b wins) for a given match.

Profiles:
  - fifa_rank    : Based on FIFA ranking points
  - elo          : ELO calculated from full historical record
  - form         : Recent form (last 20 matches)
  - tournament   : World Cup-specific historical performance
  - host_boost   : ELO + home/region advantage for USA, Canada, Mexico
  - combined     : Weighted blend of all profiles (uses backtest-optimized weights
                   from data/optimal_weights.json if available)
"""
import json
import math
import os
import pandas as pd
import numpy as np

from elo import win_probabilities, get_recent_form, get_wc_performance

OPTIMAL_WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "optimal_weights.json"
)

# Historical name mapping: canonical team name -> name used in martj42 dataset
HISTORICAL_NAME_MAP = {
    "Korea Republic":          "South Korea",
    "Czechia":                 "Czech Republic",
    "Congo DR":                "DR Congo",
    "Cabo Verde":              "Cape Verde",
    "Bosnia and Herzegovina":  "Bosnia-Herzegovina",
    "Iran":                    "IR Iran",
    "Curacao":                 "Curacao",
    "Turkey":                  "Turkey",
    "Ivory Coast":             "Ivory Coast",
}

# REVERSE mapping (historical -> canonical)
CANONICAL_NAME_MAP = {v: k for k, v in HISTORICAL_NAME_MAP.items()}


def to_hist(name: str) -> str:
    return HISTORICAL_NAME_MAP.get(name, name)


def fifa_rank_to_pseudo_elo(fifa_pts: float) -> float:
    """Convert FIFA points directly to a comparable pseudo-ELO scale."""
    return 400.0 + fifa_pts * 0.8


def _clamp(p_win, p_draw, p_lose):
    """Ensure probabilities are positive and sum to 1."""
    p_win = max(0.02, p_win)
    p_draw = max(0.02, p_draw)
    p_lose = max(0.02, p_lose)
    total = p_win + p_draw + p_lose
    return p_win / total, p_draw / total, p_lose / total


class BaseProfile:
    name: str = "base"

    def match_proba(self, team_a: str, team_b: str, neutral: bool = True) -> tuple:
        raise NotImplementedError


class FIFARankProfile(BaseProfile):
    """Uses April-2026 FIFA ranking points to estimate probabilities."""
    name = "fifa_rank"

    def __init__(self, teams_df: pd.DataFrame):
        self.pts = dict(zip(teams_df["team"], teams_df["fifa_pts"]))

    def match_proba(self, team_a: str, team_b: str, neutral: bool = True) -> tuple:
        elo_a = fifa_rank_to_pseudo_elo(self.pts.get(team_a, 1000))
        elo_b = fifa_rank_to_pseudo_elo(self.pts.get(team_b, 1000))
        return _clamp(*win_probabilities(elo_a, elo_b, neutral))


class EloProfile(BaseProfile):
    """Uses ELO ratings calculated from full historical record."""
    name = "elo"

    def __init__(self, elo_ratings: dict, teams_df: pd.DataFrame):
        self.ratings = {}
        for _, row in teams_df.iterrows():
            hist_name = to_hist(row["team"])
            elo = elo_ratings.get(row["team"], elo_ratings.get(hist_name, 1500))
            self.ratings[row["team"]] = elo

    def match_proba(self, team_a: str, team_b: str, neutral: bool = True) -> tuple:
        elo_a = self.ratings.get(team_a, 1500)
        elo_b = self.ratings.get(team_b, 1500)
        return _clamp(*win_probabilities(elo_a, elo_b, neutral))


class FormProfile(BaseProfile):
    """Uses recent form (last 20 matches) as probability adjustment."""
    name = "form"

    def __init__(self, results_df: pd.DataFrame, teams_df: pd.DataFrame,
                 elo_ratings: dict, n_matches: int = 20):
        self.base_elo = {}
        self.form_adj = {}
        for _, row in teams_df.iterrows():
            hist_name = to_hist(row["team"])
            elo = elo_ratings.get(row["team"], elo_ratings.get(hist_name, 1500))
            self.base_elo[row["team"]] = elo

            # Build form adjustment: map canonical name to historical for lookup
            team_hist = to_hist(row["team"])
            # Try canonical first, then historical
            form_canon = get_recent_form(results_df, row["team"], n_matches)
            form_hist = get_recent_form(results_df, team_hist, n_matches)
            form = form_canon if form_canon["n"] >= form_hist["n"] else form_hist

            # Convert recent win_rate and avg_gd into an ELO adjustment
            baseline_win_rate = 0.50
            wr_adj = (form["win_rate"] - baseline_win_rate) * 150
            gd_adj = form["avg_gd"] * 20
            self.form_adj[row["team"]] = wr_adj + gd_adj

    def _effective_elo(self, team: str) -> float:
        return self.base_elo.get(team, 1500) + self.form_adj.get(team, 0)

    def match_proba(self, team_a: str, team_b: str, neutral: bool = True) -> tuple:
        elo_a = self._effective_elo(team_a)
        elo_b = self._effective_elo(team_b)
        return _clamp(*win_probabilities(elo_a, elo_b, neutral))


class TournamentProfile(BaseProfile):
    """Uses World Cup-specific historical performance."""
    name = "tournament"

    def __init__(self, results_df: pd.DataFrame, teams_df: pd.DataFrame,
                 elo_ratings: dict):
        self.elo_adj = {}
        for _, row in teams_df.iterrows():
            hist_name = to_hist(row["team"])
            elo = elo_ratings.get(row["team"], elo_ratings.get(hist_name, 1500))

            wc_canon = get_wc_performance(results_df, row["team"])
            wc_hist = get_wc_performance(results_df, hist_name)
            wc = wc_canon if wc_canon["wc_matches"] >= wc_hist["wc_matches"] else wc_hist

            baseline = 0.42  # historical average WC win rate for qualified teams
            if wc["wc_matches"] >= 3:
                wr_adj = (wc["wc_win_rate"] - baseline) * 200
                gd_adj = wc["wc_avg_gd"] * 25
                adj = wr_adj + gd_adj
            else:
                adj = (elo - 1500) * 0.1  # teams with no WC history: use ELO

            self.elo_adj[row["team"]] = elo + adj

    def match_proba(self, team_a: str, team_b: str, neutral: bool = True) -> tuple:
        elo_a = self.elo_adj.get(team_a, 1500)
        elo_b = self.elo_adj.get(team_b, 1500)
        return _clamp(*win_probabilities(elo_a, elo_b, neutral))


class HostBoostProfile(BaseProfile):
    """ELO profile with continental/host advantage for USA, Canada, Mexico."""
    name = "host_boost"

    HOST_BOOST = 75         # direct hosts
    CONCACAF_BOOST = 25     # other CONCACAF teams benefit from proximity

    def __init__(self, elo_ratings: dict, teams_df: pd.DataFrame):
        self.ratings = {}
        for _, row in teams_df.iterrows():
            hist_name = to_hist(row["team"])
            elo = elo_ratings.get(row["team"], elo_ratings.get(hist_name, 1500))
            if row["is_host"]:
                elo += self.HOST_BOOST
            elif row["confederation"] == "CONCACAF":
                elo += self.CONCACAF_BOOST
            self.ratings[row["team"]] = elo

    def match_proba(self, team_a: str, team_b: str, neutral: bool = True) -> tuple:
        elo_a = self.ratings.get(team_a, 1500)
        elo_b = self.ratings.get(team_b, 1500)
        return _clamp(*win_probabilities(elo_a, elo_b, neutral))


class CombinedProfile(BaseProfile):
    """
    Weighted blend of all profiles.

    If data/optimal_weights.json exists (generated by backtest.py), uses
    backtest-optimized weights for the elo/form/tournament trio and keeps
    fifa_rank and host_boost at fixed fractions.  Falls back to hard-coded
    defaults otherwise.
    """
    name = "combined"

    # Default weights (used when no optimized weights are available)
    DEFAULT_WEIGHTS = {
        "elo":        0.30,
        "fifa_rank":  0.20,
        "form":       0.25,
        "tournament": 0.15,
        "host_boost": 0.10,
    }

    # Fixed share reserved for non-backtestable profiles
    _FIFA_RANK_SHARE = 0.15
    _HOST_BOOST_SHARE = 0.08

    def __init__(self, profiles: dict, weights_path: str = OPTIMAL_WEIGHTS_PATH):
        self.profiles = profiles
        self.weights = dict(self.DEFAULT_WEIGHTS)
        self.weights_source = "default"

        if weights_path and os.path.exists(weights_path):
            try:
                with open(weights_path) as f:
                    opt = json.load(f)
                w_e = opt.get("elo", 0.40)
                w_f = opt.get("form", 0.35)
                w_t = opt.get("tournament", 0.25)
                total_bt = w_e + w_f + w_t
                if total_bt > 0:
                    remaining = 1.0 - self._FIFA_RANK_SHARE - self._HOST_BOOST_SHARE
                    self.weights["elo"]        = (w_e / total_bt) * remaining
                    self.weights["form"]       = (w_f / total_bt) * remaining
                    self.weights["tournament"] = (w_t / total_bt) * remaining
                    self.weights["fifa_rank"]  = self._FIFA_RANK_SHARE
                    self.weights["host_boost"] = self._HOST_BOOST_SHARE
                    self.weights_source = "optimized"
            except Exception:
                pass  # silently fall back to defaults

    def match_proba(self, team_a: str, team_b: str, neutral: bool = True) -> tuple:
        p_a, p_d, p_b = 0.0, 0.0, 0.0
        total_w = 0.0
        for name, weight in self.weights.items():
            if name in self.profiles:
                pa, pd_, pb = self.profiles[name].match_proba(team_a, team_b, neutral)
                p_a += weight * pa
                p_d += weight * pd_
                p_b += weight * pb
                total_w += weight
        if total_w > 0:
            p_a, p_d, p_b = p_a / total_w, p_d / total_w, p_b / total_w
        return _clamp(p_a, p_d, p_b)


def build_all_profiles(teams_df: pd.DataFrame, results_df: pd.DataFrame,
                       elo_ratings: dict) -> dict:
    """Build and return all profiles keyed by name."""
    fifa = FIFARankProfile(teams_df)
    elo_p = EloProfile(elo_ratings, teams_df)
    form_p = FormProfile(results_df, teams_df, elo_ratings)
    tournament_p = TournamentProfile(results_df, teams_df, elo_ratings)
    host_p = HostBoostProfile(elo_ratings, teams_df)

    base_profiles = {
        "fifa_rank":  fifa,
        "elo":        elo_p,
        "form":       form_p,
        "tournament": tournament_p,
        "host_boost": host_p,
    }
    combined = CombinedProfile(base_profiles)
    return {**base_profiles, "combined": combined}
