#!/usr/bin/env python3
"""Predict winners for the remaining IPL games.

Uses ELO ratings (computed from the full match history), current-season form,
and home-venue advantage. The trained match_winner classifier is skipped
because its held-out AUC is ~0.50 with honest features (no score leakage).

Usage:
    python predict_remaining.py                       # use fixtures.csv
    python predict_remaining.py --fixtures FILE       # custom fixtures CSV
    python predict_remaining.py --from-match 34       # only predict match_number >= 34
    python predict_remaining.py --out predictions.csv # custom output path

fixtures.csv schema (one row per match):
    match_number,date,team1,team2,venue[,winner]

If `winner` is present, the match is treated as played and used to update
ELO before predicting later fixtures (so you can feed in results for matches
that postdate the Cricsheet dump).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))


# ── Team metadata ──────────────────────────────────────────────────────────

HOME_VENUES = {
    "Mumbai Indians": "Wankhede",
    "Chennai Super Kings": "Chepauk",
    "Royal Challengers Bangalore": "Chinnaswamy",
    "Royal Challengers Bengaluru": "Chinnaswamy",
    "Kolkata Knight Riders": "Eden Gardens",
    "Delhi Capitals": "Arun Jaitley",
    "Delhi Daredevils": "Arun Jaitley",
    "Sunrisers Hyderabad": "Rajiv Gandhi",
    "Rajasthan Royals": "Sawai Mansingh",
    "Punjab Kings": "Bindra Stadium",
    "Kings XI Punjab": "Bindra Stadium",
    "Gujarat Titans": "Narendra Modi",
    "Lucknow Super Giants": "Ekana",
}

# Franchise continuity — keep ELO when a team is renamed.
TEAM_ALIASES = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Kings XI Punjab": "Punjab Kings",
    "Delhi Daredevils": "Delhi Capitals",
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}

CURRENT_TEAMS = [
    "Chennai Super Kings",
    "Delhi Capitals",
    "Gujarat Titans",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad",
]


def canonical(team: str) -> str:
    return TEAM_ALIASES.get(team, team)


# ── ELO ────────────────────────────────────────────────────────────────────

ELO_K = 20
ELO_INIT = 1500


def elo_expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update_elo(elo: dict, team_a: str, team_b: str, a_won: float, k: float = ELO_K) -> None:
    ea = elo.get(team_a, ELO_INIT)
    eb = elo.get(team_b, ELO_INIT)
    expected_a = elo_expected(ea, eb)
    elo[team_a] = ea + k * (a_won - expected_a)
    elo[team_b] = eb + k * ((1 - a_won) - (1 - expected_a))


def _record_result(elo: dict, season_record: dict, in_latest_season: bool,
                   t1: str, t2: str, a_won: float) -> None:
    update_elo(elo, t1, t2, a_won)
    if in_latest_season:
        for team, won in ((t1, a_won), (t2, 1 - a_won)):
            rec = season_record.setdefault(team, {"w": 0, "l": 0, "recent": []})
            if won == 1:
                rec["w"] += 1
            else:
                rec["l"] += 1
            rec["recent"].append(int(won))
            rec["recent"] = rec["recent"][-5:]


def build_team_state(matches: pd.DataFrame) -> tuple[dict, dict, set, str]:
    """Walk historical matches chronologically, returning:
      - elo: current ELO per team
      - season_record: latest-season win/loss per team
      - seen_keys: set of (season, match_number) pairs already folded into ELO
      - latest_season: the season label used for form features
    """
    m = matches.copy()
    m["date"] = pd.to_datetime(m["date"].astype(str).str[:10], errors="coerce")
    m = m.dropna(subset=["date", "team1", "team2", "winner"])
    m = m.sort_values(["date", "match_id"]).reset_index(drop=True)

    latest_season = str(m["season"].astype(str).max())

    elo: dict[str, float] = {}
    season_record: dict[str, dict] = {}
    seen_keys: set[tuple[str, int]] = set()

    for _, row in m.iterrows():
        t1 = canonical(str(row["team1"]))
        t2 = canonical(str(row["team2"]))
        winner = canonical(str(row["winner"])) if pd.notna(row["winner"]) else None
        if winner is None or winner == "nan":
            continue
        a_won = 1.0 if winner == t1 else 0.0
        season = str(row["season"])
        _record_result(elo, season_record, season.startswith(latest_season), t1, t2, a_won)

        mn = row.get("match_number")
        if pd.notna(mn):
            seen_keys.add((season[:4], int(mn)))

    return elo, season_record, seen_keys, latest_season


# ── Prediction ─────────────────────────────────────────────────────────────

HOME_ELO_BONUS = 40   # ~5-6% win-prob bump at equal rating
FORM_WEIGHT = 0.15    # blend season form into pure-ELO probability


def is_home(team: str, venue: str) -> bool:
    kw = HOME_VENUES.get(team, "")
    return bool(kw) and kw.lower() in (venue or "").lower()


def season_win_rate(record: dict | None) -> float | None:
    if not record:
        return None
    games = record["w"] + record["l"]
    if games < 2:
        return None
    return record["w"] / games


def predict_match(
    team1: str,
    team2: str,
    venue: str,
    elo: dict,
    season_record: dict,
) -> dict:
    t1 = canonical(team1)
    t2 = canonical(team2)
    e1 = elo.get(t1, ELO_INIT)
    e2 = elo.get(t2, ELO_INIT)

    adj_e1 = e1 + (HOME_ELO_BONUS if is_home(t1, venue) else 0)
    adj_e2 = e2 + (HOME_ELO_BONUS if is_home(t2, venue) else 0)

    p1_elo = elo_expected(adj_e1, adj_e2)

    wr1 = season_win_rate(season_record.get(t1))
    wr2 = season_win_rate(season_record.get(t2))
    if wr1 is not None and wr2 is not None:
        # Map a win-rate gap to a probability shift (tanh keeps it bounded).
        gap = wr1 - wr2
        form_prob = 0.5 + 0.5 * math.tanh(1.5 * gap)
        p1 = (1 - FORM_WEIGHT) * p1_elo + FORM_WEIGHT * form_prob
    else:
        p1 = p1_elo

    winner = t1 if p1 >= 0.5 else t2
    return {
        "team1_elo": round(e1, 1),
        "team2_elo": round(e2, 1),
        "team1_home": int(is_home(t1, venue)),
        "team2_home": int(is_home(t2, venue)),
        "team1_season_wr": round(wr1, 3) if wr1 is not None else None,
        "team2_season_wr": round(wr2, 3) if wr2 is not None else None,
        "team1_win_prob": round(p1, 3),
        "team2_win_prob": round(1 - p1, 3),
        "predicted_winner": winner,
        "confidence": round(max(p1, 1 - p1), 3),
    }


# ── Fixtures I/O ───────────────────────────────────────────────────────────

def write_fixtures_template(path: Path) -> None:
    sample = pd.DataFrame([
        {"match_number": 34, "date": "2026-04-24", "team1": "Team A",
         "team2": "Team B", "venue": "Stadium, City", "winner": ""},
    ])
    sample.to_csv(path, index=False)


def load_fixtures(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"match_number", "date", "team1", "team2", "venue"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"fixtures file missing columns: {sorted(missing)}")
    if "winner" not in df.columns:
        df["winner"] = ""
    df["winner"] = df["winner"].fillna("").astype(str).str.strip()
    return df.sort_values("match_number").reset_index(drop=True)


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixtures", default=str(PROJECT_DIR / "fixtures.csv"))
    ap.add_argument("--from-match", type=int, default=None,
                    help="Only predict matches with match_number >= this value")
    ap.add_argument("--out", default=str(PROJECT_DIR / "predictions.csv"))
    args = ap.parse_args()

    matches_path = PROJECT_DIR / "data" / "processed" / "matches.parquet"
    if not matches_path.exists():
        print(f"Missing processed data at {matches_path}. Run pipeline.py first.")
        return 1
    matches = pd.read_parquet(matches_path)
    print(f"Loaded {len(matches)} historical matches (seasons {matches['season'].min()}–{matches['season'].max()}).")

    elo, season_record, seen_keys, latest_season = build_team_state(matches)
    print(f"Using season '{latest_season}' for current-form features.")
    print(f"Folded {len(seen_keys)} (season, match_number) results from processed data.")
    print("\nCurrent ELO ratings (current franchises):")
    for team in sorted(CURRENT_TEAMS, key=lambda t: -elo.get(t, ELO_INIT)):
        rec = season_record.get(team, {"w": 0, "l": 0})
        print(f"  {team:<32} {elo.get(team, ELO_INIT):7.1f}   "
              f"{latest_season} record: {rec['w']}-{rec['l']}")

    fixtures_path = Path(args.fixtures)
    if not fixtures_path.exists():
        write_fixtures_template(fixtures_path)
        print(f"\nNo fixtures file found. Wrote a template to {fixtures_path}.")
        print("Fill it in with match_number, date, team1, team2, venue rows "
              "(add 'winner' for already-played matches to update ELO), then re-run.")
        return 0

    fixtures = load_fixtures(fixtures_path)
    print(f"\nLoaded {len(fixtures)} fixtures from {fixtures_path}.")

    fixture_season_year = latest_season[:4]
    predictions = []
    folded_from_fixtures = 0

    for _, row in fixtures.iterrows():
        mn = int(row["match_number"])
        t1 = str(row["team1"]).strip()
        t2 = str(row["team2"]).strip()
        venue = str(row["venue"]).strip()
        known_winner = row["winner"].strip()

        # Already in processed data — skip (ELO already applied).
        if (fixture_season_year, mn) in seen_keys:
            continue

        # Known winner from fixtures (played but not in Cricsheet yet).
        if known_winner:
            w = canonical(known_winner)
            a_won = 1.0 if w == canonical(t1) else 0.0
            _record_result(elo, season_record, True, canonical(t1), canonical(t2), a_won)
            folded_from_fixtures += 1
            continue

        if args.from_match is not None and mn < args.from_match:
            continue

        pred = predict_match(t1, t2, venue, elo, season_record)
        predictions.append({
            "match_number": mn,
            "date": row["date"],
            "team1": t1,
            "team2": t2,
            "venue": venue,
            **pred,
        })

    if folded_from_fixtures:
        print(f"Folded {folded_from_fixtures} additional results from fixtures.csv.")

    if not predictions:
        print("\nNothing to predict — all fixtures had known winners or were below --from-match.")
        return 0

    out_df = pd.DataFrame(predictions)
    out_path = Path(args.out)
    out_df.to_csv(out_path, index=False)

    print(f"\nPredictions written to {out_path}.\n")
    print(f"{'#':>3}  {'Date':<10}  {'Matchup':<60}  {'Pick':<30}  Conf")
    print("-" * 120)
    for p in predictions:
        matchup = f"{p['team1']} vs {p['team2']}"
        print(f"{p['match_number']:>3}  {p['date']:<10}  {matchup:<60}  "
              f"{p['predicted_winner']:<30}  {p['confidence']:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
