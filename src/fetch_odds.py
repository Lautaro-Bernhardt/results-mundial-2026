"""
Fetch current 2026 World Cup match odds from The Odds API.
Free tier: 500 requests/month. Sign up at https://the-odds-api.com/

Usage:
    export ODDS_API_KEY=your_key_here
    python src/fetch_odds.py

Or:
    python src/fetch_odds.py --api-key YOUR_KEY

Saves output to: data/match_odds_2026.csv
Columns: sport_key, home_team, away_team, bookmaker, market,
         p_home_win, p_draw, p_away_win, last_update
"""

import argparse, os, sys, json
import requests
import pandas as pd

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "soccer_fifa_world_cup"

def american_to_decimal(american: int) -> float:
    if american > 0:
        return american / 100 + 1
    else:
        return 100 / abs(american) + 1

def odds_to_proba(home_dec, draw_dec, away_dec):
    """Convert 3-way decimal odds to normalized probabilities."""
    p_h = 1 / home_dec
    p_d = 1 / draw_dec
    p_a = 1 / away_dec
    total = p_h + p_d + p_a
    return p_h / total, p_d / total, p_a / total

def fetch_events(api_key: str):
    url = f"{BASE_URL}/sports/{SPORT_KEY}/odds/"
    params = {
        "apiKey": api_key,
        "regions": "eu",
        "markets": "h2h",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    r = requests.get(url, params=params, timeout=30)
    if r.status_code == 401:
        print("ERROR: Invalid API key")
        sys.exit(1)
    r.raise_for_status()
    remaining = r.headers.get("x-requests-remaining", "?")
    print(f"  API requests remaining: {remaining}")
    return r.json()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.environ.get("ODDS_API_KEY", ""))
    parser.add_argument("--output", default="data/match_odds_2026.csv")
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: Provide API key via --api-key or ODDS_API_KEY env var")
        print("Get a free key (500 req/month) at https://the-odds-api.com/")
        sys.exit(1)

    print(f"Fetching WC 2026 match odds from The Odds API...")
    events = fetch_events(args.api_key)
    print(f"  Found {len(events)} events")

    rows = []
    for event in events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        commence = event.get("commence_time", "")
        for bookie in event.get("bookmakers", []):
            bname = bookie.get("key", "")
            for market in bookie.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                p_h, p_d, p_a = odds_to_proba(
                    outcomes.get(home, 10),
                    outcomes.get("Draw", 5),
                    outcomes.get(away, 10),
                )
                rows.append({
                    "commence_time": commence,
                    "home_team": home,
                    "away_team": away,
                    "bookmaker": bname,
                    "p_home_win": round(p_h, 4),
                    "p_draw": round(p_d, 4),
                    "p_away_win": round(p_a, 4),
                    "odds_home": outcomes.get(home, ""),
                    "odds_draw": outcomes.get("Draw", ""),
                    "odds_away": outcomes.get(away, ""),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        print("WARNING: No match odds found. The market may not be open yet.")
    else:
        df.to_csv(args.output, index=False)
        print(f"  Saved {len(df)} rows → {args.output}")
        print()
        avg = df.groupby(["home_team", "away_team"])[["p_home_win", "p_draw", "p_away_win"]].mean()
        print("Average odds (consensus across bookmakers):")
        print(avg.round(3).to_string())

if __name__ == "__main__":
    main()
