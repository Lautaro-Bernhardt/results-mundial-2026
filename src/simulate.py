"""
Monte Carlo simulation of the 2026 FIFA World Cup.

Format:
  - 12 groups of 4 (72 group stage matches)
  - Top 2 + 8 best 3rd-place finishers = 32 advance
  - Round of 32 -> R16 -> QF -> SF -> Final (+ 3rd place)
"""
import random
import numpy as np
import pandas as pd
from collections import defaultdict

GROUPS = {
    "A": ["Mexico", "South Africa", "Korea Republic", "Czechia"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cabo Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Uzbekistan", "Colombia", "Congo DR"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# Round of 32 bracket: (group_winner_group, opponent_source)
# Based on the standard 2026 WC bracket structure
# Format: [(w_group, r_group), ...] — winner of group vs runner-up of group
# Third-place teams fill slots based on their group letters
R32_BRACKET = [
    # Zone 1
    ("A", "B"),  # W_A vs R_B
    ("B", "A"),  # W_B vs R_A
    # Zone 2
    ("C", "D"),  # W_C vs R_D
    ("D", "C"),  # W_D vs R_C
    # Zone 3
    ("E", "F"),  # W_E vs R_F
    ("F", "E"),  # W_F vs R_E
    # Zone 4
    ("G", "H"),  # W_G vs R_H
    ("H", "G"),  # W_H vs R_G
    # Zone 5
    ("I", "J"),  # W_I vs R_J
    ("J", "I"),  # W_J vs R_I
    # Zone 6
    ("K", "L"),  # W_K vs R_L
    ("L", "K"),  # W_L vs R_K
    # Slots for 8 best 3rd-place teams (filled dynamically)
    # Zone 7 (3rd-place slots)
    ("3rd_1", None),
    ("3rd_2", None),
    ("3rd_3", None),
    ("3rd_4", None),
]


def _simulate_match(p_a, p_draw, p_b) -> tuple:
    """Return (goals_a, goals_b) approximately consistent with probabilities."""
    r = random.random()
    if r < p_a:
        winner = "a"
        ga = random.choices([1, 2, 3, 4], weights=[30, 40, 20, 10])[0]
        gb = random.choices([0, 1], weights=[60, 40])[0]
        gb = min(gb, ga - 1)
    elif r < p_a + p_draw:
        winner = "draw"
        ga = gb = random.choices([0, 1, 2], weights=[25, 50, 25])[0]
    else:
        winner = "b"
        gb = random.choices([1, 2, 3, 4], weights=[30, 40, 20, 10])[0]
        ga = random.choices([0, 1], weights=[60, 40])[0]
        ga = min(ga, gb - 1)
    return ga, gb


def _simulate_knockout(p_a, p_draw, p_b) -> str:
    """Simulate a knockout match. In case of draw, decide via penalty (50/50)."""
    r = random.random()
    if r < p_a:
        return "a"
    elif r < p_a + p_draw:
        # Extra time / penalties
        return "a" if random.random() < 0.5 else "b"
    else:
        return "b"


def _group_matchups(teams):
    """Return all 6 (home, away) pairs for a group of 4 teams."""
    pairs = []
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            pairs.append((teams[i], teams[j]))
    return pairs


def _simulate_group(teams, profile):
    """Simulate group stage, return sorted standing [(team, pts, gd, gf), ...]."""
    pts = defaultdict(int)
    gd = defaultdict(int)
    gf = defaultdict(int)

    for t1, t2 in _group_matchups(teams):
        p_a, p_draw, p_b = profile.match_proba(t1, t2, neutral=True)
        ga, gb = _simulate_match(p_a, p_draw, p_b)

        gf[t1] += ga
        gf[t2] += gb
        gd[t1] += ga - gb
        gd[t2] += gb - ga

        if ga > gb:
            pts[t1] += 3
        elif ga < gb:
            pts[t2] += 3
        else:
            pts[t1] += 1
            pts[t2] += 1

    standing = sorted(
        teams,
        key=lambda t: (pts[t], gd[t], gf[t], random.random()),
        reverse=True,
    )
    return standing, pts, gd, gf


def _best_third_places(third_teams_data):
    """Select 8 best 3rd-place finishers from 12 groups."""
    ranked = sorted(
        third_teams_data,
        key=lambda x: (x["pts"], x["gd"], x["gf"], random.random()),
        reverse=True,
    )
    return [x["team"] for x in ranked[:8]]


def simulate_once(profile, groups=None):
    """
    Run one full tournament simulation.
    Returns dict: team -> stage reached (0=group, 1=r32, 2=r16, 3=qf, 4=sf, 5=final, 6=champion)
    """
    if groups is None:
        groups = GROUPS

    results = {t: 0 for grp in groups.values() for t in grp}

    # --- Group stage ---
    group_standings = {}
    third_data = []

    for grp, teams in groups.items():
        standing, pts, gd, gf = _simulate_group(teams, profile)
        group_standings[grp] = standing
        third_data.append({
            "team": standing[2],
            "group": grp,
            "pts": pts[standing[2]],
            "gd": gd[standing[2]],
            "gf": gf[standing[2]],
        })

    best_thirds = _best_third_places(third_data)

    # All 32 qualified teams
    r32_teams = set()
    winners = {}
    runners = {}
    for grp, standing in group_standings.items():
        winners[grp] = standing[0]
        runners[grp] = standing[1]
        results[standing[0]] = 1
        results[standing[1]] = 1

    for t in best_thirds:
        results[t] = 1
        r32_teams.add(t)

    # Build R32 bracket
    r32_bracket = []
    for w_grp, r_grp in R32_BRACKET[:12]:
        r32_bracket.append((winners[w_grp], runners[r_grp]))

    # Assign 8 best 3rd-place teams to remaining slots
    # Pair them against the group winners from their "zone"
    zone_winners_for_3rd = ["A", "B", "C", "D", "E", "F", "G", "H"]
    for i, third_team in enumerate(best_thirds):
        opp_grp = zone_winners_for_3rd[i % len(zone_winners_for_3rd)]
        # find a winner not yet assigned to a third-place slot
        r32_bracket.append((third_team, winners.get(opp_grp, third_team)))

    # Deduplicate and trim to 16 unique matchups
    seen = set()
    clean_bracket = []
    for a, b in r32_bracket:
        if a != b and (a, b) not in seen and (b, a) not in seen:
            seen.add((a, b))
            clean_bracket.append((a, b))
        if len(clean_bracket) == 16:
            break

    # Pad if needed
    all_r32 = list({winners[g] for g in GROUPS} | {runners[g] for g in GROUPS} | set(best_thirds))
    random.shuffle(all_r32)
    paired = set(t for pair in clean_bracket for t in pair)
    unpaired = [t for t in all_r32 if t not in paired]
    while len(clean_bracket) < 16 and len(unpaired) >= 2:
        clean_bracket.append((unpaired.pop(), unpaired.pop()))

    # --- Knockout stages ---
    # Stage values: 0=group out, 1=r32, 2=r16, 3=qf, 4=sf, 5=final, 6=champion
    # current_round starts as 16 R32 matchups
    current_round = clean_bracket
    stage_num = 2  # winning R32 grants stage 2 (= reached R16)

    for stage_label in ["r32", "r16", "qf", "sf", "final"]:
        next_round = []
        for a, b in current_round:
            if a not in results or b not in results:
                continue
            p_a, p_draw, p_b = profile.match_proba(a, b, neutral=True)
            winner = a if _simulate_knockout(p_a, p_draw, p_b) == "a" else b
            next_round.append(winner)
            if stage_label == "final":
                results[winner] = 6  # champion
                # loser stays at stage 5 (finalist, assigned in "sf" round)
            else:
                results[winner] = max(results[winner], stage_num)

        if stage_label != "final":
            random.shuffle(next_round)
            current_round = [(next_round[i], next_round[i + 1])
                             for i in range(0, len(next_round) - 1, 2)]
            stage_num += 1

    return results


def run_simulation(profile, n: int = 10_000, groups=None) -> pd.DataFrame:
    """
    Run N simulations and return a DataFrame with per-team stage probabilities.

    Columns: team, p_group_exit, p_r32, p_r16, p_qf, p_sf, p_final, p_champion
    """
    if groups is None:
        groups = GROUPS

    stage_counts = defaultdict(lambda: defaultdict(int))
    all_teams = [t for grp in groups.values() for t in grp]

    for _ in range(n):
        res = simulate_once(profile, groups)
        for team, stage in res.items():
            stage_counts[team][stage] += 1

    rows = []
    for team in all_teams:
        c = stage_counts[team]
        total = n
        rows.append({
            "team": team,
            "p_group_exit": c[0] / total,
            "p_r32": c[1] / total,
            "p_r16": c[2] / total,
            "p_qf": c[3] / total,
            "p_sf": c[4] / total,
            "p_final": c[5] / total,
            "p_champion": c[6] / total,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("p_champion", ascending=False).reset_index(drop=True)
    return df
