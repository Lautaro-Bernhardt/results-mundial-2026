"""
Main prediction pipeline for World Cup 2026.

Usage:
    python src/predict.py [--simulations N] [--seed SEED]

Outputs (in output/):
    elo_ratings.csv          -- ELO rating for all 48 teams
    match_predictions.csv    -- Group stage P(win/draw/loss) per model
    simulation_results.csv   -- Per-team stage probabilities per model
    champion_probabilities.csv -- Champion probabilities summary across models
"""
import argparse
import os
import sys

import pandas as pd
import numpy as np

# Add src/ to path when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from profiles import build_all_profiles, HISTORICAL_NAME_MAP, CANONICAL_NAME_MAP
from simulate import GROUPS, run_simulation


GROUPS_MATCHDAY = [
    # (group, matchday, home_idx, away_idx)
    ("A", 1, 0, 1), ("A", 1, 2, 3),
    ("A", 2, 0, 2), ("A", 2, 1, 3),
    ("A", 3, 0, 3), ("A", 3, 1, 2),
    ("B", 1, 0, 1), ("B", 1, 2, 3),
    ("B", 2, 0, 2), ("B", 2, 1, 3),
    ("B", 3, 0, 3), ("B", 3, 1, 2),
    ("C", 1, 0, 1), ("C", 1, 2, 3),
    ("C", 2, 0, 2), ("C", 2, 1, 3),
    ("C", 3, 0, 3), ("C", 3, 1, 2),
    ("D", 1, 0, 1), ("D", 1, 2, 3),
    ("D", 2, 0, 2), ("D", 2, 1, 3),
    ("D", 3, 0, 3), ("D", 3, 1, 2),
    ("E", 1, 0, 1), ("E", 1, 2, 3),
    ("E", 2, 0, 2), ("E", 2, 1, 3),
    ("E", 3, 0, 3), ("E", 3, 1, 2),
    ("F", 1, 0, 1), ("F", 1, 2, 3),
    ("F", 2, 0, 2), ("F", 2, 1, 3),
    ("F", 3, 0, 3), ("F", 3, 1, 2),
    ("G", 1, 0, 1), ("G", 1, 2, 3),
    ("G", 2, 0, 2), ("G", 2, 1, 3),
    ("G", 3, 0, 3), ("G", 3, 1, 2),
    ("H", 1, 0, 1), ("H", 1, 2, 3),
    ("H", 2, 0, 2), ("H", 2, 1, 3),
    ("H", 3, 0, 3), ("H", 3, 1, 2),
    ("I", 1, 0, 1), ("I", 1, 2, 3),
    ("I", 2, 0, 2), ("I", 2, 1, 3),
    ("I", 3, 0, 3), ("I", 3, 1, 2),
    ("J", 1, 0, 1), ("J", 1, 2, 3),
    ("J", 2, 0, 2), ("J", 2, 1, 3),
    ("J", 3, 0, 3), ("J", 3, 1, 2),
    ("K", 1, 0, 1), ("K", 1, 2, 3),
    ("K", 2, 0, 2), ("K", 2, 1, 3),
    ("K", 3, 0, 3), ("K", 3, 1, 2),
    ("L", 1, 0, 1), ("L", 1, 2, 3),
    ("L", 2, 0, 2), ("L", 2, 1, 3),
    ("L", 3, 0, 3), ("L", 3, 1, 2),
]


def load_data(data_dir="data"):
    teams_df = pd.read_csv(os.path.join(data_dir, "teams.csv"))
    results_path = os.path.join(data_dir, "historical_results.csv")
    if not os.path.exists(results_path):
        print("historical_results.csv not found. Run: python src/download_data.py")
        sys.exit(1)
    results_df = pd.read_csv(results_path, parse_dates=["date"])
    # Apply canonical name mapping so ELO dict uses same names as teams_df
    results_df["home_team"] = results_df["home_team"].replace(CANONICAL_NAME_MAP)
    results_df["away_team"] = results_df["away_team"].replace(CANONICAL_NAME_MAP)
    return teams_df, results_df


def compute_elo(results_df, teams_df):
    from elo import calculate_elo
    elo_raw = calculate_elo(results_df)

    rows = []
    for _, row in teams_df.iterrows():
        hist = HISTORICAL_NAME_MAP.get(row["team"], row["team"])
        elo = elo_raw.get(row["team"], elo_raw.get(hist, 1500))
        rows.append({"team": row["team"], "elo": round(elo, 1),
                     "group": row["group"], "confederation": row["confederation"],
                     "fifa_rank": row["fifa_rank"], "fifa_pts": row["fifa_pts"]})

    elo_df = pd.DataFrame(rows).sort_values("elo", ascending=False).reset_index(drop=True)
    elo_df["elo_rank"] = elo_df.index + 1
    return elo_df, elo_raw


def generate_match_predictions(profiles, groups=GROUPS):
    rows = []
    for grp, md, hi, ai in GROUPS_MATCHDAY:
        teams = groups[grp]
        home = teams[hi]
        away = teams[ai]
        for pname, profile in profiles.items():
            pa, pd_, pb = profile.match_proba(home, away, neutral=True)
            rows.append({
                "group": grp,
                "matchday": md,
                "home_team": home,
                "away_team": away,
                "model": pname,
                "p_home_win": round(pa, 4),
                "p_draw": round(pd_, 4),
                "p_away_win": round(pb, 4),
            })
    return pd.DataFrame(rows)


def generate_simulation_results(profiles, n_sim):
    all_rows = []
    for pname, profile in profiles.items():
        print(f"  Simulating {n_sim:,} tournaments with profile '{pname}' ...")
        sim_df = run_simulation(profile, n=n_sim)
        sim_df["model"] = pname
        all_rows.append(sim_df)
    return pd.concat(all_rows, ignore_index=True)


def generate_champion_summary(sim_df):
    pivot = sim_df.pivot_table(
        index="team", columns="model", values="p_champion"
    ).reset_index()
    # Add average column
    model_cols = [c for c in pivot.columns if c != "team"]
    pivot["avg_champion_prob"] = pivot[model_cols].mean(axis=1)
    return pivot.sort_values("avg_champion_prob", ascending=False).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    teams_df, results_df = load_data(args.data_dir)
    print(f"  Teams: {len(teams_df)} | Historical matches: {len(results_df):,}")

    print("Computing ELO ratings...")
    elo_df, elo_raw = compute_elo(results_df, teams_df)
    elo_out = os.path.join(args.output_dir, "elo_ratings.csv")
    elo_df.to_csv(elo_out, index=False)
    print(f"  Saved {elo_out}")

    print("Building probability profiles...")
    profiles = build_all_profiles(teams_df, results_df, elo_raw)

    print("Generating match predictions (group stage)...")
    match_df = generate_match_predictions(profiles)
    match_out = os.path.join(args.output_dir, "match_predictions.csv")
    match_df.to_csv(match_out, index=False)
    print(f"  Saved {match_out}  ({len(match_df)} rows)")

    print(f"Running Monte Carlo simulations (N={args.simulations:,})...")
    sim_df = generate_simulation_results(profiles, args.simulations)
    sim_out = os.path.join(args.output_dir, "simulation_results.csv")
    sim_df.to_csv(sim_out, index=False)
    print(f"  Saved {sim_out}")

    print("Generating champion probability summary...")
    champ_df = generate_champion_summary(sim_df)
    champ_out = os.path.join(args.output_dir, "champion_probabilities.csv")
    champ_df.to_csv(champ_out, index=False)
    print(f"  Saved {champ_out}")

    print("\n=== TOP 10 CHAMPION PROBABILITIES (combined model) ===")
    top = (
        sim_df[sim_df["model"] == "combined"]
        .sort_values("p_champion", ascending=False)
        .head(10)[["team", "p_champion", "p_final", "p_sf", "p_qf"]]
    )
    print(top.to_string(index=False))

    print("\nDone! Output files:")
    for f in ["elo_ratings.csv", "match_predictions.csv",
              "simulation_results.csv", "champion_probabilities.csv"]:
        path = os.path.join(args.output_dir, f)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  {path} ({size:,} bytes)")


if __name__ == "__main__":
    main()
