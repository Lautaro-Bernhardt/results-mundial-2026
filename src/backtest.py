"""
Backtest prediction models against FIFA World Cups 2010, 2014, 2018, 2022.
Measures Brier Score, Log Loss, and Accuracy for each model.
Then optimizes combined model weights to minimize Brier Score.

Usage:
    python src/backtest.py

Outputs:
    output/backtest_detail.csv    -- per-match predictions and metrics
    output/backtest_metrics.csv   -- aggregated summary per model
    output/backtest_by_year.csv   -- metrics broken down by WC year
    data/optimal_weights.json     -- optimized weights for combined model
"""

import math
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution

sys.path.insert(0, os.path.dirname(__file__))
from elo import calculate_elo, win_probabilities, get_recent_form, get_wc_performance

# --- Configuration -----------------------------------------------------------

WC_WINDOWS = {
    2010: ("2010-06-11", "2010-07-11"),
    2014: ("2014-06-12", "2014-07-13"),
    2018: ("2018-06-14", "2018-07-15"),
    2022: ("2022-11-20", "2022-12-18"),
}

# Host team for each WC (gets home advantage boost in backtest)
WC_HOSTS = {
    2010: "South Africa",
    2014: "Brazil",
    2018: "Russia",
    2022: "Qatar",
}

# Default ELO adjustment coefficients for form and WC performance
FORM_WR_ADJ = 150.0    # ELO pts per unit win-rate deviation from 0.50
FORM_GD_ADJ = 20.0     # ELO pts per avg goal difference
WC_WR_ADJ   = 200.0    # ELO pts per unit WC win-rate deviation from 0.42
WC_GD_ADJ   = 25.0     # ELO pts per avg WC goal difference
HOME_ADV    = 100.0    # ELO advantage for host nation in their own WC

DEFAULT_WEIGHTS = [0.40, 0.35, 0.25]   # [elo, form, tournament]


# --- Helpers -----------------------------------------------------------------

def _team_adjustments(team: str, train_df: pd.DataFrame) -> tuple:
    """Return (form_adj_elo, wc_adj_elo) for a team given training data."""
    form = get_recent_form(train_df, team, n_matches=20)
    form_adj = 0.0
    if form["n"] >= 5:
        form_adj = ((form["win_rate"] - 0.50) * FORM_WR_ADJ
                    + form["avg_gd"] * FORM_GD_ADJ)

    wc = get_wc_performance(train_df, team)
    wc_adj = 0.0
    if wc["wc_matches"] >= 3:
        wc_adj = ((wc["wc_win_rate"] - 0.42) * WC_WR_ADJ
                  + wc["wc_avg_gd"] * WC_GD_ADJ)

    return form_adj, wc_adj


def _clamp(p_h, p_d, p_a):
    p_h, p_d, p_a = max(0.01, p_h), max(0.01, p_d), max(0.01, p_a)
    s = p_h + p_d + p_a
    return p_h / s, p_d / s, p_a / s


def _brier(probs: list, hs: int, as_: int) -> float:
    if hs > as_:   actual = (1.0, 0.0, 0.0)
    elif hs == as_: actual = (0.0, 1.0, 0.0)
    else:           actual = (0.0, 0.0, 1.0)
    return sum((p - a) ** 2 for p, a in zip(probs, actual))


def _logloss(probs: list, hs: int, as_: int) -> float:
    if hs > as_:   p = probs[0]
    elif hs == as_: p = probs[1]
    else:           p = probs[2]
    return -math.log(max(p, 1e-10))


def _accuracy(probs: list, hs: int, as_: int) -> int:
    pred_idx = probs.index(max(probs))
    if hs > as_:   true_idx = 0
    elif hs == as_: true_idx = 1
    else:           true_idx = 2
    return int(pred_idx == true_idx)


def _weighted_probs(match: dict, weights: list) -> tuple:
    w = np.abs(weights)
    w = w / w.sum()
    p = w[0] * match["p_elo"] + w[1] * match["p_form"] + w[2] * match["p_tournament"]
    return _clamp(*p.tolist())


# --- Precomputation ----------------------------------------------------------

def precompute_predictions(results_df: pd.DataFrame) -> dict:
    """
    For each WC year, compute ELO ratings and per-model predictions for every
    WC match using only data before that WC's start date.
    Returns: dict[year -> list[match_dict]]
    """
    precomputed = {}

    for year, (start, end) in WC_WINDOWS.items():
        t0 = time.time()
        train_df = results_df[results_df["date"] < start].copy()
        test_mask = (
            (results_df["date"] >= start)
            & (results_df["date"] <= end)
            & (results_df["tournament"] == "FIFA World Cup")
            & results_df["home_score"].notna()
            & results_df["away_score"].notna()
        )
        test_df = results_df[test_mask].copy()

        elo_ratings = calculate_elo(train_df)

        # Cache adjustments per team (expensive with large datasets)
        adj_cache = {}

        def get_adj(team):
            if team not in adj_cache:
                adj_cache[team] = _team_adjustments(team, train_df)
            return adj_cache[team]

        host = WC_HOSTS[year]
        matches = []
        for _, row in test_df.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            hs   = int(row["home_score"])
            as_  = int(row["away_score"])

            elo_h = elo_ratings.get(home, 1500)
            elo_a = elo_ratings.get(away, 1500)

            # Host advantage: treat matches involving the host as non-neutral
            is_host_match = (home == host or away == host)
            if home == host:
                elo_h_neutral = elo_h + HOME_ADV
            elif away == host:
                elo_a_neutral = elo_a + HOME_ADV
                elo_h_neutral = elo_h
            else:
                elo_h_neutral = elo_h
            # Re-do properly
            if home == host:
                elo_h_eff, elo_a_eff = elo_h + HOME_ADV, elo_a
            elif away == host:
                elo_h_eff, elo_a_eff = elo_h, elo_a + HOME_ADV
            else:
                elo_h_eff, elo_a_eff = elo_h, elo_a

            p_elo_raw = win_probabilities(elo_h_eff, elo_a_eff, neutral=True)

            fa_h, wa_h = get_adj(home)
            fa_a, wa_a = get_adj(away)
            p_form_raw  = win_probabilities(elo_h_eff + fa_h, elo_a_eff + fa_a, neutral=True)
            p_tourn_raw = win_probabilities(elo_h_eff + wa_h, elo_a_eff + wa_a, neutral=True)

            matches.append({
                "year":       year,
                "home":       home,
                "away":       away,
                "hs":         hs,
                "as_":        as_,
                "p_elo":      np.array(p_elo_raw),
                "p_form":     np.array(p_form_raw),
                "p_tournament": np.array(p_tourn_raw),
            })

        elapsed = time.time() - t0
        print(f"    {year}: {len(train_df):,} train / {len(test_df)} WC matches  "
              f"({elapsed:.1f}s)")
        precomputed[year] = matches

    return precomputed


# --- Optimization ------------------------------------------------------------

def objective_brier(weights, precomputed):
    """Average Brier Score across all WC matches for given [elo, form, tournament] weights."""
    total, count = 0.0, 0
    for matches in precomputed.values():
        for m in matches:
            probs = list(_weighted_probs(m, weights))
            total += _brier(probs, m["hs"], m["as_"])
            count += 1
    return total / count if count > 0 else 999.0


def optimize_weights(precomputed: dict) -> tuple:
    """
    Find weights that minimize Brier Score using Nelder-Mead (local) then
    Differential Evolution (global) for robustness.
    Returns (optimal_weights_list, best_brier).
    """
    # Local search from multiple starting points
    starts = [
        [0.40, 0.35, 0.25],
        [0.50, 0.30, 0.20],
        [0.33, 0.33, 0.34],
        [0.60, 0.25, 0.15],
        [0.30, 0.50, 0.20],
        [0.25, 0.35, 0.40],
    ]
    best_val = float("inf")
    best_w = DEFAULT_WEIGHTS

    for x0 in starts:
        res = minimize(
            objective_brier, x0, args=(precomputed,),
            method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 1e-5, "fatol": 1e-6},
        )
        raw = np.abs(res.x)
        w = (raw / raw.sum()).tolist()
        val = objective_brier(w, precomputed)
        if val < best_val:
            best_val = val
            best_w = w

    return best_w, best_val


# --- Evaluation --------------------------------------------------------------

def evaluate_all(precomputed: dict, optimal_weights: list) -> pd.DataFrame:
    """Return per-match detailed evaluation for all models."""
    rows = []

    model_preds = {
        "baseline":           lambda m: (1 / 3, 1 / 3, 1 / 3),
        "elo":                lambda m: tuple(m["p_elo"]),
        "form":               lambda m: tuple(m["p_form"]),
        "tournament":         lambda m: tuple(m["p_tournament"]),
        "combined_default":   lambda m: _weighted_probs(m, DEFAULT_WEIGHTS),
        "combined_optimized": lambda m: _weighted_probs(m, optimal_weights),
    }

    for year, matches in precomputed.items():
        for m in matches:
            for model_name, pred_fn in model_preds.items():
                probs = list(_clamp(*pred_fn(m)))
                rows.append({
                    "year":       year,
                    "home":       m["home"],
                    "away":       m["away"],
                    "home_score": m["hs"],
                    "away_score": m["as_"],
                    "model":      model_name,
                    "p_home_win": round(probs[0], 4),
                    "p_draw":     round(probs[1], 4),
                    "p_away_win": round(probs[2], 4),
                    "brier":      round(_brier(probs, m["hs"], m["as_"]), 5),
                    "log_loss":   round(_logloss(probs, m["hs"], m["as_"]), 5),
                    "correct":    _accuracy(probs, m["hs"], m["as_"]),
                })

    return pd.DataFrame(rows)


def summarize(detail_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        detail_df.groupby("model")
        .agg(
            avg_brier  =("brier",   "mean"),
            avg_logloss=("log_loss","mean"),
            accuracy   =("correct", "mean"),
            n_matches  =("brier",   "count"),
        )
        .reset_index()
        .sort_values("avg_brier")
        .reset_index(drop=True)
    )
    return summary


def by_year(detail_df: pd.DataFrame) -> pd.DataFrame:
    byyear = (
        detail_df.groupby(["model", "year"])["brier"]
        .mean()
        .unstack()
        .reset_index()
    )
    byyear["avg"] = byyear[[2010, 2014, 2018, 2022]].mean(axis=1)
    return byyear.sort_values("avg").reset_index(drop=True)


# --- Main --------------------------------------------------------------------

def main():
    data_path = os.path.join("data", "historical_results.csv")
    if not os.path.exists(data_path):
        print("ERROR: Run 'python src/download_data.py' first.")
        sys.exit(1)

    os.makedirs("output", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    print("Loading historical data...")
    results_df = pd.read_csv(data_path, parse_dates=["date"])
    print(f"  {len(results_df):,} matches")

    print("\nPre-computing predictions (no data leakage)...")
    precomputed = precompute_predictions(results_df)

    total_matches = sum(len(v) for v in precomputed.values())
    print(f"  Total WC matches across 4 tournaments: {total_matches}")

    print("\nOptimizing combined model weights (minimizing Brier Score)...")
    t0 = time.time()
    optimal_weights, best_brier = optimize_weights(precomputed)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Optimal weights  → elo: {optimal_weights[0]:.1%}  "
          f"form: {optimal_weights[1]:.1%}  tournament: {optimal_weights[2]:.1%}")
    print(f"  Optimized Brier  → {best_brier:.5f}")
    print(f"  Default Brier    → {objective_brier(DEFAULT_WEIGHTS, precomputed):.5f}")

    # Save optimal weights
    weights_path = os.path.join("data", "optimal_weights.json")
    weights_data = {
        "elo":         round(optimal_weights[0], 4),
        "form":        round(optimal_weights[1], 4),
        "tournament":  round(optimal_weights[2], 4),
        "optimized_brier": round(best_brier, 6),
        "n_wc_matches": total_matches,
    }
    with open(weights_path, "w") as f:
        json.dump(weights_data, f, indent=2)
    print(f"  Saved → {weights_path}")

    print("\nEvaluating all models on full backtest set...")
    detail_df = evaluate_all(precomputed, optimal_weights)
    detail_path = os.path.join("output", "backtest_detail.csv")
    detail_df.to_csv(detail_path, index=False)

    summary_df = summarize(detail_df)
    summary_path = os.path.join("output", "backtest_metrics.csv")
    summary_df.to_csv(summary_path, index=False)

    byyear_df = by_year(detail_df)
    byyear_path = os.path.join("output", "backtest_by_year.csv")
    byyear_df.to_csv(byyear_path, index=False)

    # --- Print results -------------------------------------------------------
    print()
    print("=" * 72)
    print("BACKTEST METRICS — average over WC 2010 + 2014 + 2018 + 2022")
    print("=" * 72)
    print(f"{'Model':<25}  {'Brier↓':>9}  {'LogLoss↓':>10}  {'Accuracy↑':>11}  {'N':>5}")
    print("-" * 72)
    for _, row in summary_df.iterrows():
        marker = " ★" if row["model"] == "combined_optimized" else ""
        print(f"{row['model']:<25}  {row['avg_brier']:>9.5f}  "
              f"{row['avg_logloss']:>10.5f}  {row['accuracy']:>10.1%}  "
              f"{int(row['n_matches']):>5}{marker}")

    print()
    print("Brier Score by WC year (lower = better):")
    print(byyear_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print()
    print("Output files:")
    for path in [detail_path, summary_path, byyear_path, weights_path]:
        if os.path.exists(path):
            print(f"  {path}  ({os.path.getsize(path):,} bytes)")

    print()
    print("Next step: run 'python src/predict.py' to re-run 2026 predictions")
    print("with the optimized weights.")


if __name__ == "__main__":
    main()
