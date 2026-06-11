"""Feature engineering for IPL betting model.

Builds player-level and match-level features from ball-by-ball data.
"""

import numpy as np
import pandas as pd
from pathlib import Path

FEATURES_DIR = Path(__file__).parent


def _ensure_columns(df, required, context=""):
    """Check required columns exist, return list of missing ones."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  Warning ({context}): missing columns {missing}")
    return missing


def build_innings_summary(deliveries):
    """Aggregate ball-by-ball data into per-innings player summaries."""
    df = deliveries.copy()

    # Standardize column names (Cricsheet CSV2 format)
    col_map = {
        "match_id": "match_id",
        "season": "season",
        "innings": "innings",
        "ball": "ball",
        "batting_team": "batting_team",
        "bowling_team": "bowling_team",
        "striker": "batter",
        "batter": "batter",
        "non_striker": "non_striker",
        "bowler": "bowler",
        "runs_off_bat": "runs_off_bat",
        "extras": "extras",
        "wides": "wides",
        "noballs": "noballs",
        "byes": "byes",
        "legbyes": "legbyes",
        "penalty": "penalty",
        "wicket_type": "wicket_type",
        "player_dismissed": "player_dismissed",
        "venue": "venue",
        "start_date": "start_date",
    }

    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Ensure numeric columns
    for col in ["runs_off_bat", "extras", "wides", "noballs"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # --- Batting summary per innings ---
    bat_agg = df.groupby(["match_id", "innings", "batter", "batting_team"]).agg(
        runs=("runs_off_bat", "sum"),
        balls_faced=("runs_off_bat", "count"),
        fours=("runs_off_bat", lambda x: (x == 4).sum()),
        sixes=("runs_off_bat", lambda x: (x == 6).sum()),
    ).reset_index()

    # Check if batter was dismissed
    if "player_dismissed" in df.columns:
        dismissals = df[df["player_dismissed"].notna()].groupby(
            ["match_id", "innings", "player_dismissed"]
        ).size().reset_index(name="times_dismissed")
        dismissals = dismissals.rename(columns={"player_dismissed": "batter"})
        bat_agg = bat_agg.merge(dismissals, on=["match_id", "innings", "batter"], how="left")
        bat_agg["times_dismissed"] = bat_agg["times_dismissed"].fillna(0).astype(int)
        bat_agg["not_out"] = (bat_agg["times_dismissed"] == 0).astype(int)
    else:
        bat_agg["times_dismissed"] = 0
        bat_agg["not_out"] = 1

    bat_agg["strike_rate"] = np.where(
        bat_agg["balls_faced"] > 0,
        bat_agg["runs"] / bat_agg["balls_faced"] * 100,
        0,
    )

    # --- Bowling summary per innings ---
    # Exclude wides/noballs from balls bowled count
    valid_balls = df.copy()
    if "wides" in valid_balls.columns:
        valid_balls["is_valid_ball"] = (valid_balls["wides"] == 0)
        if "noballs" in valid_balls.columns:
            valid_balls["is_valid_ball"] &= (valid_balls["noballs"] == 0)
    else:
        valid_balls["is_valid_ball"] = True

    bowl_agg = df.groupby(["match_id", "innings", "bowler", "bowling_team"]).agg(
        runs_conceded=("runs_off_bat", "sum"),
        total_extras=("extras", "sum"),
        deliveries=("ball", "count"),
    ).reset_index()

    # Count wickets (exclude run outs)
    if "wicket_type" in df.columns:
        wicket_df = df[
            df["wicket_type"].notna() &
            ~df["wicket_type"].str.contains("run out", case=False, na=False) &
            ~df["wicket_type"].str.contains("retired", case=False, na=False)
        ]
        wickets = wicket_df.groupby(["match_id", "innings", "bowler"]).size().reset_index(name="wickets")
        bowl_agg = bowl_agg.merge(wickets, on=["match_id", "innings", "bowler"], how="left")
    else:
        bowl_agg["wickets"] = 0

    bowl_agg["wickets"] = bowl_agg["wickets"].fillna(0).astype(int)
    bowl_agg["economy"] = np.where(
        bowl_agg["deliveries"] > 0,
        (bowl_agg["runs_conceded"] + bowl_agg["total_extras"]) / (bowl_agg["deliveries"] / 6),
        0,
    )

    # --- Team innings totals ---
    team_totals = df.groupby(["match_id", "innings", "batting_team"]).agg(
        team_runs=("runs_off_bat", "sum"),
        team_extras=("extras", "sum"),
        total_balls=("ball", "count"),
    ).reset_index()
    team_totals["team_total"] = team_totals["team_runs"] + team_totals["team_extras"]

    # Add match metadata
    meta_cols = ["match_id"]
    for col in ["venue", "start_date", "season"]:
        if col in df.columns:
            meta_cols.append(col)

    if len(meta_cols) > 1:
        match_meta = df[meta_cols].drop_duplicates(subset=["match_id"])
        bat_agg = bat_agg.merge(match_meta, on="match_id", how="left")
        bowl_agg = bowl_agg.merge(match_meta, on="match_id", how="left")
        team_totals = team_totals.merge(match_meta, on="match_id", how="left")

    return bat_agg, bowl_agg, team_totals


def _rolling_stats(group, col, windows=(5, 10), min_periods=1):
    """Compute rolling mean and exponentially weighted mean."""
    features = {}
    vals = group[col].values.astype(float)
    s = pd.Series(vals)

    for w in windows:
        features[f"{col}_roll_{w}"] = s.shift(1).rolling(w, min_periods=min_periods).mean().values
    features[f"{col}_ewm"] = s.shift(1).ewm(span=10, min_periods=min_periods).mean().values

    return features


def build_batting_features(bat_agg):
    """Build player batting features with rolling stats."""
    df = bat_agg.sort_values(["batter", "start_date", "match_id"]).copy() if "start_date" in bat_agg.columns \
        else bat_agg.sort_values(["batter", "match_id"]).copy()

    all_features = []

    for batter, group in df.groupby("batter"):
        g = group.copy()
        idx = g.index

        # Rolling batting average and strike rate
        for col in ["runs", "strike_rate", "balls_faced"]:
            if col in g.columns:
                feats = _rolling_stats(g, col)
                for k, v in feats.items():
                    g[k] = v

        # Career stats up to this point (expanding)
        g["career_runs_mean"] = g["runs"].shift(1).expanding().mean()
        g["career_sr_mean"] = g["strike_rate"].shift(1).expanding().mean()
        g["career_innings"] = range(len(g))

        # Recent form indicator
        g["last_5_avg"] = g["runs"].shift(1).rolling(5, min_periods=1).mean()

        # Consistency (std dev of last 10 scores)
        g["runs_std_10"] = g["runs"].shift(1).rolling(10, min_periods=3).std()

        # Career hit rates — what % of past innings did this player exceed each threshold?
        # Uses exponentially weighted mean (span=30) so recent matches count more
        # than matches from years ago. This prevents stale career data from
        # dominating when a player's form has declined.
        # Also last-5 rate catches very recent form changes.
        for threshold in [10, 15, 20, 25, 30, 40, 50, 70, 100]:
            exceeded = (g["runs"] >= threshold).astype(float)
            g[f"career_rate_{threshold}_runs"] = exceeded.shift(1).ewm(span=30, min_periods=1).mean()
            g[f"recent_rate_{threshold}_runs"] = exceeded.shift(1).rolling(10, min_periods=3).mean()
            g[f"last5_rate_{threshold}_runs"] = exceeded.shift(1).rolling(5, min_periods=2).mean()

        # Targets
        for threshold in [10, 15, 20, 25, 30, 40, 50, 70, 100]:
            g[f"target_{threshold}_runs"] = (g["runs"] >= threshold).astype(int)

        all_features.append(g)

    return pd.concat(all_features, ignore_index=True) if all_features else pd.DataFrame()


def build_bowling_features(bowl_agg):
    """Build player bowling features with rolling stats."""
    df = bowl_agg.sort_values(["bowler", "start_date", "match_id"]).copy() if "start_date" in bowl_agg.columns \
        else bowl_agg.sort_values(["bowler", "match_id"]).copy()

    all_features = []

    for bowler, group in df.groupby("bowler"):
        g = group.copy()

        # Rolling economy and wicket stats
        for col in ["wickets", "economy", "runs_conceded"]:
            if col in g.columns:
                feats = _rolling_stats(g, col)
                for k, v in feats.items():
                    g[k] = v

        g["career_wickets_mean"] = g["wickets"].shift(1).expanding().mean()
        g["career_economy_mean"] = g["economy"].shift(1).expanding().mean()
        g["career_bowl_innings"] = range(len(g))

        # Career hit rates for wicket thresholds (EWM so recent matches weigh more)
        # Also add a last-5 rate to catch very recent form changes
        for threshold in [1, 2, 3, 4, 5]:
            exceeded = (g["wickets"] >= threshold).astype(float)
            g[f"career_rate_{threshold}_wickets"] = exceeded.shift(1).ewm(span=30, min_periods=1).mean()
            g[f"recent_rate_{threshold}_wickets"] = exceeded.shift(1).rolling(10, min_periods=3).mean()
            g[f"last5_rate_{threshold}_wickets"] = exceeded.shift(1).rolling(5, min_periods=2).mean()

        # Targets
        for threshold in [1, 2, 3, 4, 5]:
            g[f"target_{threshold}_wickets"] = (g["wickets"] >= threshold).astype(int)

        all_features.append(g)

    return pd.concat(all_features, ignore_index=True) if all_features else pd.DataFrame()


def build_venue_features(bat_agg, bowl_agg):
    """Build venue-specific expanding averages for batters and bowlers.

    Uses shift(1) + expanding mean to avoid data leakage — each row only
    sees the player's historical performance at this venue, not the current match.
    """
    venue_bat = pd.DataFrame()
    venue_bowl = pd.DataFrame()

    if "venue" in bat_agg.columns:
        sort_col = "start_date" if "start_date" in bat_agg.columns else "match_id"
        sorted_bat = bat_agg.sort_values([sort_col])

        bat_parts = []
        min_venue_innings = 3  # Need 3+ prior innings for venue stats to be meaningful
        for (batter, venue), group in sorted_bat.groupby(["batter", "venue"]):
            g = group.copy()
            g["venue_bat_innings"] = range(len(g))
            g["venue_bat_avg"] = g["runs"].shift(1).expanding().mean()
            g["venue_bat_sr"] = g["strike_rate"].shift(1).expanding().mean()
            # Mask venue stats to NaN when insufficient history
            g.loc[g["venue_bat_innings"] < min_venue_innings, "venue_bat_avg"] = float("nan")
            g.loc[g["venue_bat_innings"] < min_venue_innings, "venue_bat_sr"] = float("nan")
            bat_parts.append(g[["match_id", "innings", "batter", "venue",
                                "venue_bat_avg", "venue_bat_sr", "venue_bat_innings"]])
        if bat_parts:
            venue_bat = pd.concat(bat_parts, ignore_index=True)

    if "venue" in bowl_agg.columns:
        sort_col = "start_date" if "start_date" in bowl_agg.columns else "match_id"
        sorted_bowl = bowl_agg.sort_values([sort_col])

        bowl_parts = []
        min_venue_innings = 3
        for (bowler, venue), group in sorted_bowl.groupby(["bowler", "venue"]):
            g = group.copy()
            g["venue_bowl_innings"] = range(len(g))
            g["venue_bowl_econ"] = g["economy"].shift(1).expanding().mean()
            g["venue_bowl_wickets_avg"] = g["wickets"].shift(1).expanding().mean()
            g.loc[g["venue_bowl_innings"] < min_venue_innings, "venue_bowl_econ"] = float("nan")
            g.loc[g["venue_bowl_innings"] < min_venue_innings, "venue_bowl_wickets_avg"] = float("nan")
            bowl_parts.append(g[["match_id", "innings", "bowler", "venue",
                                  "venue_bowl_econ", "venue_bowl_wickets_avg", "venue_bowl_innings"]])
        if bowl_parts:
            venue_bowl = pd.concat(bowl_parts, ignore_index=True)

    return venue_bat, venue_bowl


def build_team_features(team_totals):
    """Build team-level features: recent form, venue stats."""
    df = team_totals.sort_values(["batting_team", "start_date", "match_id"]).copy() \
        if "start_date" in team_totals.columns else team_totals.sort_values(["batting_team", "match_id"]).copy()

    all_features = []

    for team, group in df.groupby("batting_team"):
        g = group.copy()

        # Rolling team totals
        for col in ["team_total"]:
            feats = _rolling_stats(g, col, windows=(3, 5))
            for k, v in feats.items():
                g[k] = v

        g["team_avg_total"] = g["team_total"].shift(1).expanding().mean()
        all_features.append(g)

    result = pd.concat(all_features, ignore_index=True) if all_features else pd.DataFrame()

    # First innings total targets
    first_innings = result[result["innings"] == 1].copy()
    for line in [140, 150, 160, 170, 180]:
        first_innings[f"target_over_{line}"] = (first_innings["team_total"] >= line).astype(int)

    return result, first_innings


def _compute_elo(results_df, k=20, initial=1500):
    """Compute ELO ratings for all teams across matches.

    Returns DataFrame with match_id, team1_elo, team2_elo (before the match).
    """
    elo = {}
    records = []

    for _, row in results_df.iterrows():
        t1 = row["team1"]
        t2 = row["team2"]
        if pd.isna(t1) or pd.isna(t2):
            continue

        e1 = elo.get(t1, initial)
        e2 = elo.get(t2, initial)

        records.append({
            "match_id": row["match_id"],
            "team1_elo": e1,
            "team2_elo": e2,
            "elo_diff": e1 - e2,
        })

        # Update ELO based on result
        expected1 = 1.0 / (1.0 + 10 ** ((e2 - e1) / 400))
        actual1 = row.get("team1_won", 0.5)
        elo[t1] = e1 + k * (actual1 - expected1)
        elo[t2] = e2 + k * ((1 - actual1) - (1 - expected1))

    return pd.DataFrame(records)


def build_match_winner_features(deliveries, matches=None):
    """Build match-winner prediction features with ELO, h2h, home advantage, team form."""
    df = deliveries.copy()

    # Get teams per match
    teams_per_match = df.groupby("match_id").agg(
        team1=("batting_team", "first"),
        team2=("bowling_team", "first"),
    ).reset_index()

    # Get first innings total
    first_inn = df[df["innings"] == 1].groupby("match_id").agg(
        first_inn_total=("runs_off_bat", "sum"),
        first_inn_extras=("extras", "sum"),
    ).reset_index()
    first_inn["first_inn_score"] = first_inn["first_inn_total"] + first_inn["first_inn_extras"]

    # Get second innings total
    second_inn = df[df["innings"] == 2].groupby("match_id").agg(
        second_inn_total=("runs_off_bat", "sum"),
        second_inn_extras=("extras", "sum"),
    ).reset_index()
    second_inn["second_inn_score"] = second_inn["second_inn_total"] + second_inn["second_inn_extras"]

    match_results = teams_per_match.merge(first_inn, on="match_id", how="left") \
                                    .merge(second_inn, on="match_id", how="left")

    # Determine winner
    match_results["team1_won"] = (
        match_results["first_inn_score"] > match_results["second_inn_score"].fillna(0)
    ).astype(int)

    # Add metadata
    meta_cols = ["match_id"]
    for col in ["venue", "start_date", "season"]:
        if col in df.columns:
            meta_cols.append(col)
    if len(meta_cols) > 1:
        meta = df[meta_cols].drop_duplicates(subset=["match_id"])
        match_results = match_results.merge(meta, on="match_id", how="left")

    # Sort by date for time-series features
    if "start_date" in match_results.columns:
        match_results = match_results.sort_values("start_date").reset_index(drop=True)

    # ── ELO ratings ──
    elo_df = _compute_elo(match_results)
    if not elo_df.empty:
        match_results = match_results.merge(elo_df, on="match_id", how="left")

    # ── Team form: rolling win rate (last 5 matches per team) ──
    # Build per-team match history with win/loss
    team_records = []
    for _, row in match_results.iterrows():
        mid = row["match_id"]
        t1_won = row.get("team1_won", 0)
        team_records.append({"match_id": mid, "team": row["team1"], "won": t1_won, "start_date": row.get("start_date")})
        team_records.append({"match_id": mid, "team": row["team2"], "won": 1 - t1_won, "start_date": row.get("start_date")})

    team_df = pd.DataFrame(team_records)
    if "start_date" in team_df.columns:
        team_df = team_df.sort_values("start_date")

    # Rolling win rate per team
    team_form = {}
    for team, group in team_df.groupby("team"):
        g = group.copy()
        g["win_rate_5"] = g["won"].shift(1).rolling(5, min_periods=1).mean()
        g["win_rate_10"] = g["won"].shift(1).rolling(10, min_periods=3).mean()
        g["career_win_rate"] = g["won"].shift(1).expanding().mean()
        team_form[team] = g.set_index("match_id")[["win_rate_5", "win_rate_10", "career_win_rate"]]

    # Merge team1 and team2 form
    t1_form_list = []
    t2_form_list = []
    for _, row in match_results.iterrows():
        mid = row["match_id"]
        t1 = row["team1"]
        t2 = row["team2"]
        t1f = team_form.get(t1, pd.DataFrame()).loc[mid] if t1 in team_form and mid in team_form[t1].index else {}
        t2f = team_form.get(t2, pd.DataFrame()).loc[mid] if t2 in team_form and mid in team_form[t2].index else {}
        t1_form_list.append({
            "match_id": mid,
            "team1_win_rate_5": t1f.get("win_rate_5"),
            "team1_win_rate_10": t1f.get("win_rate_10"),
            "team1_career_win_rate": t1f.get("career_win_rate"),
        })
        t2_form_list.append({
            "match_id": mid,
            "team2_win_rate_5": t2f.get("win_rate_5"),
            "team2_win_rate_10": t2f.get("win_rate_10"),
            "team2_career_win_rate": t2f.get("career_win_rate"),
        })

    match_results = match_results.merge(pd.DataFrame(t1_form_list), on="match_id", how="left")
    match_results = match_results.merge(pd.DataFrame(t2_form_list), on="match_id", how="left")

    # ── Head-to-head record at this venue ──
    if "venue" in match_results.columns:
        h2h_records = []
        h2h_cache = {}  # (team1, team2, venue) -> [wins]
        for _, row in match_results.iterrows():
            t1, t2, venue = row["team1"], row["team2"], row["venue"]
            key = (min(t1, t2), max(t1, t2), venue)

            past = h2h_cache.get(key, [])
            if past:
                # From team1's perspective
                t1_wins = sum(1 for w in past if w == t1)
                h2h_records.append({
                    "match_id": row["match_id"],
                    "h2h_venue_matches": len(past),
                    "h2h_venue_win_rate": t1_wins / len(past),
                })
            else:
                h2h_records.append({
                    "match_id": row["match_id"],
                    "h2h_venue_matches": 0,
                    "h2h_venue_win_rate": float("nan"),
                })

            # Record winner
            winner = t1 if row.get("team1_won", 0) == 1 else t2
            h2h_cache.setdefault(key, []).append(winner)

        match_results = match_results.merge(pd.DataFrame(h2h_records), on="match_id", how="left")

    # ── Home advantage ──
    # Map teams to their primary home venues
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

    if "venue" in match_results.columns:
        def _is_home(team, venue):
            if pd.isna(team) or pd.isna(venue):
                return 0
            home_keyword = HOME_VENUES.get(team, "")
            return 1 if home_keyword and home_keyword.lower() in venue.lower() else 0

        match_results["team1_is_home"] = match_results.apply(
            lambda r: _is_home(r["team1"], r["venue"]), axis=1
        )
        match_results["team2_is_home"] = match_results.apply(
            lambda r: _is_home(r["team2"], r["venue"]), axis=1
        )

    return match_results


def build_powerplay_features(deliveries):
    """Build powerplay (overs 1-6) and death (overs 16-20) specific stats.

    Returns bat_pp (batter powerplay stats) and bowl_pp (bowler phase stats)
    keyed on match_id, innings, player.
    """
    df = deliveries.copy()
    if "striker" in df.columns and "batter" not in df.columns:
        df = df.rename(columns={"striker": "batter"})

    # Parse over number from ball column (e.g., 0.1 -> over 1, 5.3 -> over 6, 19.2 -> over 20)
    df["over"] = df["ball"].astype(float).apply(lambda x: int(x) + 1)
    df["runs_off_bat"] = pd.to_numeric(df["runs_off_bat"], errors="coerce").fillna(0)

    df["is_powerplay"] = (df["over"] <= 6).astype(int)
    df["is_death"] = (df["over"] >= 17).astype(int)

    # ── Batter powerplay stats ──
    pp_bat = df[df["is_powerplay"] == 1].groupby(["match_id", "innings", "batter"]).agg(
        pp_runs=("runs_off_bat", "sum"),
        pp_balls=("runs_off_bat", "count"),
    ).reset_index()
    pp_bat["pp_sr"] = np.where(pp_bat["pp_balls"] > 0, pp_bat["pp_runs"] / pp_bat["pp_balls"] * 100, 0)

    # Rolling powerplay stats per batter
    if "start_date" not in pp_bat.columns:
        # Merge start_date from deliveries
        dates = df[["match_id", "start_date"]].drop_duplicates() if "start_date" in df.columns else None
        if dates is not None:
            pp_bat = pp_bat.merge(dates, on="match_id", how="left")

    bat_pp_features = []
    if "start_date" in pp_bat.columns:
        pp_bat = pp_bat.sort_values("start_date")
    for batter, group in pp_bat.groupby("batter"):
        g = group.copy()
        g["pp_runs_roll_5"] = g["pp_runs"].shift(1).rolling(5, min_periods=1).mean()
        g["pp_sr_roll_5"] = g["pp_sr"].shift(1).rolling(5, min_periods=1).mean()
        bat_pp_features.append(g[["match_id", "innings", "batter", "pp_runs_roll_5", "pp_sr_roll_5"]])

    bat_pp = pd.concat(bat_pp_features, ignore_index=True) if bat_pp_features else pd.DataFrame()

    # ── Bowler death overs stats ──
    death_bowl = df[df["is_death"] == 1].groupby(["match_id", "innings", "bowler"]).agg(
        death_runs=("runs_off_bat", "sum"),
        death_balls=("runs_off_bat", "count"),
    ).reset_index()
    death_bowl["death_econ"] = np.where(
        death_bowl["death_balls"] > 0,
        death_bowl["death_runs"] / (death_bowl["death_balls"] / 6),
        0,
    )

    if "start_date" not in death_bowl.columns:
        dates = df[["match_id", "start_date"]].drop_duplicates() if "start_date" in df.columns else None
        if dates is not None:
            death_bowl = death_bowl.merge(dates, on="match_id", how="left")

    bowl_death_features = []
    if "start_date" in death_bowl.columns:
        death_bowl = death_bowl.sort_values("start_date")
    for bowler, group in death_bowl.groupby("bowler"):
        g = group.copy()
        g["death_econ_roll_5"] = g["death_econ"].shift(1).rolling(5, min_periods=1).mean()
        bowl_death_features.append(g[["match_id", "innings", "bowler", "death_econ_roll_5"]])

    bowl_death = pd.concat(bowl_death_features, ignore_index=True) if bowl_death_features else pd.DataFrame()

    # ── Bowler powerplay stats ──
    pp_bowl = df[df["is_powerplay"] == 1].groupby(["match_id", "innings", "bowler"]).agg(
        pp_bowl_runs=("runs_off_bat", "sum"),
        pp_bowl_balls=("runs_off_bat", "count"),
    ).reset_index()
    pp_bowl["pp_bowl_econ"] = np.where(
        pp_bowl["pp_bowl_balls"] > 0,
        pp_bowl["pp_bowl_runs"] / (pp_bowl["pp_bowl_balls"] / 6),
        0,
    )

    if "start_date" not in pp_bowl.columns and "start_date" in df.columns:
        dates = df[["match_id", "start_date"]].drop_duplicates()
        pp_bowl = pp_bowl.merge(dates, on="match_id", how="left")

    bowl_pp_features = []
    if "start_date" in pp_bowl.columns:
        pp_bowl = pp_bowl.sort_values("start_date")
    for bowler, group in pp_bowl.groupby("bowler"):
        g = group.copy()
        g["pp_bowl_econ_roll_5"] = g["pp_bowl_econ"].shift(1).rolling(5, min_periods=1).mean()
        bowl_pp_features.append(g[["match_id", "innings", "bowler", "pp_bowl_econ_roll_5"]])

    bowl_pp = pd.concat(bowl_pp_features, ignore_index=True) if bowl_pp_features else pd.DataFrame()

    # Merge bowler PP and death stats
    if not bowl_pp.empty and not bowl_death.empty:
        bowl_phases = bowl_pp.merge(bowl_death, on=["match_id", "innings", "bowler"], how="outer")
    elif not bowl_pp.empty:
        bowl_phases = bowl_pp
    elif not bowl_death.empty:
        bowl_phases = bowl_death
    else:
        bowl_phases = pd.DataFrame()

    return bat_pp, bowl_phases


def build_conditions_features(deliveries, team_totals):
    """Build pitch/conditions features: day/night and venue recent scoring trends."""
    conditions = pd.DataFrame()

    if "start_date" not in deliveries.columns:
        return conditions

    # ── Venue recent scoring trend (all teams, not player-specific) ──
    if "venue" in team_totals.columns and "start_date" in team_totals.columns:
        first_inn = team_totals[team_totals["innings"] == 1].copy()
        first_inn = first_inn.sort_values("start_date")

        venue_parts = []
        for venue, group in first_inn.groupby("venue"):
            g = group.copy()
            g["venue_recent_avg_score"] = g["team_total"].shift(1).rolling(10, min_periods=1).mean()
            g["venue_overall_avg_score"] = g["team_total"].shift(1).expanding().mean()
            venue_parts.append(g[["match_id", "venue", "venue_recent_avg_score", "venue_overall_avg_score"]])

        if venue_parts:
            conditions = pd.concat(venue_parts, ignore_index=True)

    return conditions


def build_matchup_features(deliveries):
    """Build batter vs bowler type matchup stats.

    Groups bowlers into types based on their economy/style patterns.
    """
    df = deliveries.copy()
    # Normalize column name — raw data uses 'striker', summaries use 'batter'
    if "striker" in df.columns and "batter" not in df.columns:
        df = df.rename(columns={"striker": "batter"})
    if "bowler" not in df.columns or "batter" not in df.columns:
        return pd.DataFrame()

    # Direct batter vs bowler matchup stats
    matchups = df.groupby(["batter", "bowler"]).agg(
        matchup_runs=("runs_off_bat", "sum"),
        matchup_balls=("runs_off_bat", "count"),
    ).reset_index()

    if "wicket_type" in df.columns:
        wicket_df = df[df["wicket_type"].notna() & ~df["wicket_type"].str.contains("run out", case=False, na=False)]
        matchup_wkts = wicket_df.groupby(["batter", "bowler"]).size().reset_index(name="matchup_dismissals")
        # Use player_dismissed to correctly attribute
        if "player_dismissed" in df.columns:
            matchup_wkts = wicket_df.groupby(["player_dismissed", "bowler"]).size().reset_index(name="matchup_dismissals")
            matchup_wkts = matchup_wkts.rename(columns={"player_dismissed": "batter"})
        matchups = matchups.merge(matchup_wkts, on=["batter", "bowler"], how="left")
    matchups["matchup_dismissals"] = matchups.get("matchup_dismissals", 0)
    if "matchup_dismissals" in matchups.columns:
        matchups["matchup_dismissals"] = matchups["matchup_dismissals"].fillna(0)

    matchups["matchup_sr"] = np.where(
        matchups["matchup_balls"] > 0,
        matchups["matchup_runs"] / matchups["matchup_balls"] * 100,
        0,
    )

    return matchups


def build_batting_position(deliveries):
    """Estimate batting position from ball-by-ball data.

    Position is determined by order of first appearance in an innings.
    """
    df = deliveries.copy()
    # Normalize column name — raw data uses 'striker', summaries use 'batter'
    if "striker" in df.columns and "batter" not in df.columns:
        df = df.rename(columns={"striker": "batter"})
    if "batter" not in df.columns:
        return pd.DataFrame()

    # Get first ball faced by each batter per innings
    first_ball = df.groupby(["match_id", "innings", "batter"])["ball"].min().reset_index()
    first_ball = first_ball.sort_values(["match_id", "innings", "ball"])

    # Assign position based on order
    first_ball["batting_position"] = first_ball.groupby(["match_id", "innings"]).cumcount() + 1

    return first_ball[["match_id", "innings", "batter", "batting_position"]]


def build_opponent_strength(bat_agg, bowl_agg):
    """Build opponent bowling/batting strength features.

    For each match+innings, compute the average quality of the opposition.
    Key insight: in innings 1, batting_team=A and bowling_team=B.
    We want batters in team A to see team B's bowling quality.
    So we aggregate by bowling_team, keep that column name, and merge onto
    batting features where bat_features.batting_team != opp.bowling_team
    but they share the same match_id and innings.

    Simpler approach: aggregate by (match_id, innings, bowling_team), then
    merge directly — since in a given innings, the batting_team rows and
    the bowling_team column already refer to opposite teams.
    """
    # --- Opposition bowling strength (for batting predictions) ---
    bowl_career = bowl_agg.groupby("bowler").agg(
        bowler_career_econ=("economy", "mean"),
        bowler_career_wr=("wickets", "mean"),
    ).reset_index()

    match_bowlers = bowl_agg[["match_id", "innings", "bowler", "bowling_team"]].drop_duplicates()
    match_bowlers = match_bowlers.merge(bowl_career, on="bowler", how="left")

    # Aggregate bowling quality per match+innings (this is the bowling team's quality)
    opp_bowling = match_bowlers.groupby(["match_id", "innings"]).agg(
        opp_bowl_econ=("bowler_career_econ", "mean"),
        opp_bowl_wr=("bowler_career_wr", "mean"),
    ).reset_index()
    # Merge on match_id + innings: batting features already have the correct innings,
    # and in that innings, the opposition IS the bowling team.

    # --- Opposition batting strength (for bowling predictions) ---
    bat_career = bat_agg.groupby("batter").agg(
        batter_career_avg=("runs", "mean"),
        batter_career_sr=("strike_rate", "mean"),
    ).reset_index()

    match_batters = bat_agg[["match_id", "innings", "batter", "batting_team"]].drop_duplicates()
    match_batters = match_batters.merge(bat_career, on="batter", how="left")

    # Aggregate batting quality per match+innings (this is the batting team's quality)
    opp_batting = match_batters.groupby(["match_id", "innings"]).agg(
        opp_bat_avg=("batter_career_avg", "mean"),
        opp_bat_sr=("batter_career_sr", "mean"),
    ).reset_index()
    # Merge on match_id + innings: bowling features in that innings face these batters.

    return opp_bowling, opp_batting


def build_venue_scores(team_totals):
    """Build venue average first-innings score for team total predictions."""
    if "venue" not in team_totals.columns:
        return pd.DataFrame()

    first_inn = team_totals[team_totals["innings"] == 1].copy()
    if first_inn.empty:
        return pd.DataFrame()

    # Sort by date to compute expanding mean per venue (no leakage)
    if "start_date" in first_inn.columns:
        first_inn = first_inn.sort_values("start_date")

    venue_avgs = []
    for venue, group in first_inn.groupby("venue"):
        g = group.copy()
        g["venue_avg_first_inn"] = g["team_total"].shift(1).expanding().mean()
        g["venue_avg_first_inn_last5"] = g["team_total"].shift(1).rolling(5, min_periods=1).mean()
        g["venue_matches_played"] = range(len(g))
        venue_avgs.append(g[["match_id", "venue", "venue_avg_first_inn", "venue_avg_first_inn_last5", "venue_matches_played"]])

    return pd.concat(venue_avgs, ignore_index=True) if venue_avgs else pd.DataFrame()


def build_toss_features(matches):
    """Extract toss result and decision from match info.

    Args:
        matches: DataFrame from load_match_info() with toss_winner, toss_decision, etc.
    """
    if matches is None or matches.empty:
        return pd.DataFrame(columns=["match_id"])

    cols = ["match_id"]
    for col in ["toss_winner", "toss_decision", "team1", "team2"]:
        if col in matches.columns:
            cols.append(col)

    toss = matches[cols].copy()

    if "toss_decision" in toss.columns:
        toss["toss_elected_bat"] = (toss["toss_decision"] == "bat").astype(int)
        toss["toss_elected_field"] = (toss["toss_decision"] == "field").astype(int)

    # For each batting team in a match, did they win the toss?
    # This gets merged per-team later
    return toss


def build_phase_features(matches):
    """Build phase-of-tournament features from match info.

    IPL has ~60 league stage matches then ~4 playoff matches per season.
    Match numbers > 56 (varies by season) are typically playoffs.
    """
    if matches is None or matches.empty:
        return pd.DataFrame(columns=["match_id"])

    if "match_number" not in matches.columns:
        return pd.DataFrame(columns=["match_id"])

    phase = matches[["match_id", "match_number"]].copy()
    phase["match_number"] = pd.to_numeric(phase["match_number"], errors="coerce")

    # Playoff threshold: matches after 56 in a season are playoffs
    # (IPL has had 56-60 league matches depending on year)
    phase["is_playoff"] = (phase["match_number"] > 56).astype(int)
    phase["is_early_season"] = (phase["match_number"] <= 20).astype(int)
    phase["is_mid_season"] = ((phase["match_number"] > 20) & (phase["match_number"] <= 40)).astype(int)

    return phase[["match_id", "match_number", "is_playoff", "is_early_season", "is_mid_season"]]


def build_sample_weights(df):
    """Build sample weights that give more importance to recent seasons.

    Returns a weight column: recent seasons get higher weight.
    Seasons 2020+ get 2x, 2023+ get 3x weight relative to older seasons.
    """
    season = None
    if "season" in df.columns:
        season = pd.to_numeric(df["season"], errors="coerce")
    elif "start_date" in df.columns:
        season = pd.to_datetime(df["start_date"], errors="coerce").dt.year

    if season is None:
        return np.ones(len(df))

    weights = np.ones(len(df))
    weights[season >= 2018] = 1.5
    weights[season >= 2020] = 2.0
    weights[season >= 2023] = 3.0
    return weights


def build_all_features(deliveries, matches=None):
    """Master function: build all features from raw deliveries and match info.

    Args:
        deliveries: ball-by-ball DataFrame
        matches: match info DataFrame (from load_match_info), optional

    Returns dict of DataFrames ready for modelling.
    """
    print("\n=== Building Features ===")

    print("Building innings summaries...")
    bat_agg, bowl_agg, team_totals = build_innings_summary(deliveries)

    print("Building batting features...")
    bat_features = build_batting_features(bat_agg)

    print("Building bowling features...")
    bowl_features = build_bowling_features(bowl_agg)

    # Add innings context
    if not bat_features.empty and "innings" in bat_features.columns:
        bat_features["is_second_innings"] = (bat_features["innings"] == 2).astype(int)
    if not bowl_features.empty and "innings" in bowl_features.columns:
        bowl_features["is_second_innings"] = (bowl_features["innings"] == 2).astype(int)

    print("Building venue features...")
    venue_bat, venue_bowl = build_venue_features(bat_agg, bowl_agg)

    print("Building team features...")
    team_features, first_inn_features = build_team_features(team_totals)

    print("Building match winner features...")
    match_results = build_match_winner_features(deliveries, matches)

    print("Building matchup features...")
    matchups = build_matchup_features(deliveries)

    print("Building batting position features...")
    bat_positions = build_batting_position(deliveries)

    print("Building opponent strength features...")
    opp_bowling, opp_batting = build_opponent_strength(bat_agg, bowl_agg)

    print("Building venue score features...")
    venue_scores = build_venue_scores(team_totals)

    print("Building powerplay/death features...")
    bat_pp, bowl_phases = build_powerplay_features(deliveries)

    print("Building conditions features...")
    conditions = build_conditions_features(deliveries, team_totals)

    print("Building toss features...")
    toss = build_toss_features(matches)

    print("Building phase-of-tournament features...")
    phase = build_phase_features(matches)

    print("Building sample weights...")
    # Add weights to each feature set
    for fdf_name, fdf in [("bat", bat_features), ("bowl", bowl_features)]:
        if fdf is not None and not fdf.empty:
            w = build_sample_weights(fdf)
            fdf["sample_weight"] = w

    # Merge batting position into batting features
    if not bat_positions.empty and not bat_features.empty:
        bat_features = bat_features.merge(
            bat_positions, on=["match_id", "innings", "batter"], how="left"
        )

    # Merge venue stats into player features (now keyed on match_id + innings + player)
    if not venue_bat.empty and not bat_features.empty:
        bat_features = bat_features.merge(
            venue_bat, on=["match_id", "innings", "batter", "venue"], how="left", suffixes=("", "_venue")
        )

    if not venue_bowl.empty and not bowl_features.empty:
        bowl_features = bowl_features.merge(
            venue_bowl, on=["match_id", "innings", "bowler", "venue"], how="left", suffixes=("", "_venue")
        )

    # Merge opponent bowling strength into batting features (same match+innings)
    if not opp_bowling.empty and not bat_features.empty:
        bat_features = bat_features.merge(
            opp_bowling, on=["match_id", "innings"], how="left"
        )

    # Merge opponent batting strength into bowling features (same match+innings)
    if not opp_batting.empty and not bowl_features.empty:
        bowl_features = bowl_features.merge(
            opp_batting, on=["match_id", "innings"], how="left"
        )

    # Merge venue scores into first innings features
    if not venue_scores.empty and not first_inn_features.empty:
        first_inn_features = first_inn_features.merge(
            venue_scores, on=["match_id", "venue"], how="left"
        )

    # Merge powerplay stats into batting features
    if not bat_pp.empty and not bat_features.empty:
        bat_features = bat_features.merge(
            bat_pp, on=["match_id", "innings", "batter"], how="left"
        )

    # Merge bowler phase stats into bowling features
    if not bowl_phases.empty and not bowl_features.empty:
        bowl_features = bowl_features.merge(
            bowl_phases, on=["match_id", "innings", "bowler"], how="left"
        )

    # Merge conditions (venue recent scoring) into all relevant feature sets
    if not conditions.empty:
        if not bat_features.empty and "venue" in bat_features.columns:
            bat_features = bat_features.merge(conditions, on=["match_id", "venue"], how="left")
        if not bowl_features.empty and "venue" in bowl_features.columns:
            bowl_features = bowl_features.merge(conditions, on=["match_id", "venue"], how="left")
        if not first_inn_features.empty and "venue" in first_inn_features.columns:
            first_inn_features = first_inn_features.merge(conditions, on=["match_id", "venue"], how="left")

    # Merge average matchup stats into batting features (batter's avg performance vs this bowling team)
    if not matchups.empty and not bat_features.empty:
        # Aggregate matchup stats: for each batter, avg SR and dismissal rate across all bowlers they've faced
        batter_matchup_agg = matchups.groupby("batter").agg(
            avg_matchup_sr=("matchup_sr", "mean"),
            avg_matchup_dismissal_rate=("matchup_dismissals", "mean"),
            total_matchup_balls=("matchup_balls", "sum"),
        ).reset_index()
        bat_features = bat_features.merge(batter_matchup_agg, on="batter", how="left")

    # Merge toss features into batting and bowling features
    if not toss.empty:
        # For batters: did the batting team win the toss?
        if "toss_winner" in toss.columns and not bat_features.empty:
            toss_slim = toss[["match_id", "toss_elected_bat", "toss_elected_field"]].drop_duplicates(subset=["match_id"])
            bat_features = bat_features.merge(toss_slim, on="match_id", how="left")
            bowl_features = bowl_features.merge(toss_slim, on="match_id", how="left")
            first_inn_features = first_inn_features.merge(toss_slim, on="match_id", how="left")
            match_results = match_results.merge(toss_slim, on="match_id", how="left")

    # Merge phase-of-tournament features
    if not phase.empty:
        phase_slim = phase[["match_id", "match_number", "is_playoff", "is_early_season", "is_mid_season"]]
        if not bat_features.empty:
            bat_features = bat_features.merge(phase_slim, on="match_id", how="left")
        if not bowl_features.empty:
            bowl_features = bowl_features.merge(phase_slim, on="match_id", how="left")
        if not first_inn_features.empty:
            first_inn_features = first_inn_features.merge(phase_slim, on="match_id", how="left")
        if not match_results.empty:
            match_results = match_results.merge(phase_slim, on="match_id", how="left")

    # Save all feature sets
    output_dir = FEATURES_DIR / "processed"
    output_dir.mkdir(exist_ok=True)

    feature_sets = {
        "batting": bat_features,
        "bowling": bowl_features,
        "team": team_features,
        "first_innings": first_inn_features,
        "match_results": match_results,
        "matchups": matchups,
    }

    for name, fdf in feature_sets.items():
        if isinstance(fdf, pd.DataFrame) and not fdf.empty:
            path = output_dir / f"{name}_features.parquet"
            fdf.to_parquet(path, index=False)
            print(f"  Saved {name} features: {fdf.shape} -> {path}")

    print("=== Feature engineering complete ===\n")
    return feature_sets


if __name__ == "__main__":
    from data import get_processed_data
    deliveries, matches = get_processed_data()
    feature_sets = build_all_features(deliveries)
