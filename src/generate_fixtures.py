"""Generate the complete 2026 World Cup fixture CSV."""
import csv
import os
from datetime import date, timedelta

GROUPS = {
    "A": ["Mexico", "South Africa", "Korea Republic", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
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

# Known confirmed venues (MD1 matches from official sources)
CONFIRMED_VENUES = {
    ("Mexico", "South Africa"):           ("Estadio Azteca", "Mexico City", "Mexico"),
    ("Korea Republic", "Czechia"):        ("Estadio Akron", "Guadalajara", "Mexico"),
    ("Canada", "Bosnia and Herzegovina"): ("BMO Field", "Toronto", "Canada"),
    ("Qatar", "Switzerland"):             ("Levi's Stadium", "Santa Clara", "United States"),
    ("Canada", "Switzerland"):            ("BMO Field", "Toronto", "Canada"),
    ("United States", "Paraguay"):        ("SoFi Stadium", "Inglewood", "United States"),
    ("Australia", "Turkey"):              ("BC Place", "Vancouver", "Canada"),
}

# Approximate venue pools by host country
VENUE_POOLS = {
    "Mexico": [
        ("Estadio Azteca", "Mexico City"), ("Estadio Akron", "Guadalajara"),
        ("Estadio BBVA", "Monterrey"),
    ],
    "United States": [
        ("AT&T Stadium", "Dallas"), ("SoFi Stadium", "Inglewood"),
        ("MetLife Stadium", "East Rutherford"), ("Levi's Stadium", "Santa Clara"),
        ("Arrowhead Stadium", "Kansas City"), ("Hard Rock Stadium", "Miami"),
        ("Gillette Stadium", "Boston"), ("Lincoln Financial Field", "Philadelphia"),
        ("Seattle Center", "Seattle"),
    ],
    "Canada": [
        ("BMO Field", "Toronto"), ("BC Place", "Vancouver"),
    ],
}

GROUP_HOST = {
    "A": "Mexico", "B": "United States", "C": "United States",
    "D": "United States", "E": "United States", "F": "United States",
    "G": "United States", "H": "Mexico", "I": "United States",
    "J": "United States", "K": "United States", "L": "Canada",
}

# MD1 start dates per group (approx from confirmed + pattern)
MD1_DATES = {
    "A": date(2026, 6, 11), "B": date(2026, 6, 12), "C": date(2026, 6, 12),
    "D": date(2026, 6, 12), "E": date(2026, 6, 13), "F": date(2026, 6, 13),
    "G": date(2026, 6, 14), "H": date(2026, 6, 14), "I": date(2026, 6, 15),
    "J": date(2026, 6, 15), "K": date(2026, 6, 16), "L": date(2026, 6, 17),
}

MD2_OFFSET = 7   # days after MD1
MD3_OFFSET = 14  # days after MD1 (simultaneous final round)


def get_venue(t1, t2, group):
    if (t1, t2) in CONFIRMED_VENUES:
        s, c, ctry = CONFIRMED_VENUES[(t1, t2)]
        return s, c, ctry
    host = GROUP_HOST[group]
    venues = VENUE_POOLS[host]
    idx = (hash(t1 + t2) % len(venues))
    s, c = venues[idx]
    return s, c, host


def generate_group_fixtures():
    rows = []
    match_id = 1

    # Matchday pairings for each group: [(t1_idx, t2_idx), ...]
    # MD1: (0,1),(2,3) | MD2: (0,2),(1,3) | MD3: (0,3),(1,2)
    matchday_pairs = [
        (1, [(0, 1), (2, 3)]),
        (2, [(0, 2), (1, 3)]),
        (3, [(0, 3), (1, 2)]),
    ]

    for grp, teams in GROUPS.items():
        md1_date = MD1_DATES[grp]
        for md, pairs in matchday_pairs:
            if md == 1:
                base_date = md1_date
            elif md == 2:
                base_date = md1_date + timedelta(days=MD2_OFFSET)
            else:
                base_date = md1_date + timedelta(days=MD3_OFFSET)

            for offset, (i, j) in enumerate(pairs):
                t1, t2 = teams[i], teams[j]
                match_date = base_date + timedelta(days=offset)
                stadium, city, country = get_venue(t1, t2, grp)
                rows.append({
                    "match_id": match_id,
                    "stage": "group",
                    "group": grp,
                    "matchday": md,
                    "date": match_date.isoformat(),
                    "home_team": t1,
                    "away_team": t2,
                    "stadium": stadium,
                    "city": city,
                    "country": country,
                })
                match_id += 1

    return rows


def generate_knockout_placeholders():
    """Placeholder rows for knockout rounds with TBD teams."""
    rows = []
    match_id = 73
    stages = [
        ("round_of_32", 16, date(2026, 6, 29)),
        ("round_of_16", 8, date(2026, 7, 4)),
        ("quarterfinal", 4, date(2026, 7, 9)),
        ("semifinal", 2, date(2026, 7, 14)),
        ("third_place", 1, date(2026, 7, 18)),
        ("final", 1, date(2026, 7, 19)),
    ]
    for stage, count, start_date in stages:
        for i in range(count):
            rows.append({
                "match_id": match_id,
                "stage": stage,
                "group": "",
                "matchday": "",
                "date": (start_date + timedelta(days=i)).isoformat(),
                "home_team": "TBD",
                "away_team": "TBD",
                "stadium": "TBD",
                "city": "TBD",
                "country": "TBD",
            })
            match_id += 1
    return rows


def main():
    os.makedirs("data", exist_ok=True)
    rows = generate_group_fixtures() + generate_knockout_placeholders()
    fieldnames = ["match_id", "stage", "group", "matchday", "date",
                  "home_team", "away_team", "stadium", "city", "country"]
    with open("data/fixtures.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(rows)} fixtures -> data/fixtures.csv")


if __name__ == "__main__":
    main()
