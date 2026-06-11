#!/usr/bin/env python3
"""End-to-end IPL betting model pipeline.

Usage:
    python pipeline.py                  # Full pipeline
    python pipeline.py --skip-download  # Skip data download
    python pipeline.py --skip-train     # Use existing models
    python pipeline.py --odds odds.csv  # Custom odds file
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from data.download import download_cricsheet, get_processed_data, parse_cricsheet_data
from data.odds_scraper import fetch_odds as fetch_live_odds
from features.engineering import build_all_features
from models.train import (
    train_all_models, load_models, MARKETS,
    BATTING_FEATURE_COLS, BOWLING_FEATURE_COLS, TEAM_FEATURE_COLS, MATCH_FEATURE_COLS,
    _calibrate_prob, load_batting_regression,
)
from models.calibration import plot_calibration_curves, plot_feature_importance
from output import (
    load_odds, map_market_to_model, find_value_bets,
    suggest_multis, save_output, build_historical_correlations,
)


def load_playing_xi():
    """Load today's playing XI from JSON file."""
    xi_path = PROJECT_DIR / "data" / "playing_xi.json"
    if xi_path.exists():
        import json
        with open(xi_path) as f:
            xi = json.load(f)
        # Remove instructions key
        xi.pop("_instructions", None)
        return xi
    return {}


def compute_xi_opponent_strength(feature_sets, playing_xi, player_name, market_type):
    """Compute opponent strength from the actual playing XI.

    For a batter: average bowling quality of opposing team's bowlers in the XI.
    For a bowler: average batting quality of opposing team's batters in the XI.

    Returns dict of override features, or empty dict.
    """
    if not playing_xi:
        return {}

    # Find which team this player is on
    player_team = None
    opponent_team = None
    for team, players in playing_xi.items():
        # Fuzzy match by last name
        last_name = player_name.split()[-1] if " " in player_name else player_name
        for p in players:
            if last_name.lower() in p.lower():
                player_team = team
                break
        if player_team:
            break

    if not player_team:
        return {}

    # Find opponent team
    for team in playing_xi:
        if team != player_team:
            opponent_team = team
            break

    if not opponent_team:
        return {}

    opp_players = playing_xi[opponent_team]
    overrides = {}

    if "runs" in market_type:
        # Batter facing opponent bowlers — compute their avg quality
        bowl_df = feature_sets.get("bowling")
        if bowl_df is not None and not bowl_df.empty:
            opp_stats = []
            for opp_player in opp_players:
                opp_last = opp_player.split()[-1] if " " in opp_player else opp_player
                match = bowl_df[bowl_df["bowler"].str.contains(opp_last, case=False, na=False)]
                if not match.empty:
                    latest = match.sort_values("start_date").iloc[-1] if "start_date" in match.columns else match.iloc[-1]
                    econ = latest.get("career_economy_mean")
                    wr = latest.get("career_wickets_mean")
                    if pd.notna(econ):
                        opp_stats.append({"econ": econ, "wr": wr if pd.notna(wr) else 0})

            if opp_stats:
                overrides["opp_bowl_econ"] = np.mean([s["econ"] for s in opp_stats])
                overrides["opp_bowl_wr"] = np.mean([s["wr"] for s in opp_stats])

    elif "wickets" in market_type:
        # Bowler facing opponent batters
        bat_df = feature_sets.get("batting")
        if bat_df is not None and not bat_df.empty:
            opp_stats = []
            for opp_player in opp_players:
                opp_last = opp_player.split()[-1] if " " in opp_player else opp_player
                match = bat_df[bat_df["batter"].str.contains(opp_last, case=False, na=False)]
                if not match.empty:
                    latest = match.sort_values("start_date").iloc[-1] if "start_date" in match.columns else match.iloc[-1]
                    avg = latest.get("career_runs_mean")
                    sr = latest.get("career_sr_mean")
                    if pd.notna(avg):
                        opp_stats.append({"avg": avg, "sr": sr if pd.notna(sr) else 0})

            if opp_stats:
                overrides["opp_bat_avg"] = np.mean([s["avg"] for s in opp_stats])
                overrides["opp_bat_sr"] = np.mean([s["sr"] for s in opp_stats])

    # ── XI-specific matchup stats ──
    matchups = feature_sets.get("matchups")
    if matchups is not None and not matchups.empty and "runs" in market_type:
        # For this batter, compute avg SR and dismissal rate vs the opposing bowlers
        last_name = player_name.split()[-1]
        batter_matchups = matchups[matchups["batter"].str.contains(last_name, case=False, na=False)]

        if not batter_matchups.empty:
            opp_bowler_matchups = []
            for opp_player in opp_players:
                opp_last = opp_player.split()[-1] if " " in opp_player else opp_player
                m = batter_matchups[batter_matchups["bowler"].str.contains(opp_last, case=False, na=False)]
                if not m.empty:
                    opp_bowler_matchups.append(m.iloc[0])

            if opp_bowler_matchups:
                matchup_df = pd.DataFrame(opp_bowler_matchups)
                total_balls = matchup_df["matchup_balls"].sum()
                # Only override matchup stats if sufficient sample (100+ balls)
                # Small samples are noisier than career averages
                if total_balls >= 100:
                    overrides["avg_matchup_sr"] = matchup_df["matchup_sr"].mean()
                    if "matchup_dismissals" in matchup_df.columns:
                        overrides["avg_matchup_dismissal_rate"] = matchup_df["matchup_dismissals"].mean()
                    overrides["total_matchup_balls"] = total_balls

    return overrides


# Common name mismatches between Sportsbet and Cricsheet
PLAYER_ALIASES = {
    "V Chakravarthy": "CV Varun",
    "Varun Chakravarthy": "CV Varun",
    "Suryakumar Yadav": "SA Yadav",
    "Prasidh Krishna": "M Prasidh Krishna",
    "P Krishna": "M Prasidh Krishna",
    "Mukesh Kumar": "Mukesh Kumar",
    "FH Allen": "FH Allen",
}


def build_prediction_features(feature_sets, player_name, market_type):
    """Build feature vector for a specific player and market.

    Uses the most recent data available for the player.
    Returns dict of feature_name -> value, or None if player not found.
    """
    # Check alias map first
    player_name = PLAYER_ALIASES.get(player_name, player_name)
    if "runs" in market_type:
        data_key = "batting"
        feature_cols = BATTING_FEATURE_COLS
        name_col = "batter"
    elif "wickets" in market_type:
        data_key = "bowling"
        feature_cols = BOWLING_FEATURE_COLS
        name_col = "bowler"
    elif "over" in market_type:
        data_key = "first_innings"
        feature_cols = TEAM_FEATURE_COLS
        name_col = None
    elif market_type == "match_winner":
        data_key = "match_results"
        feature_cols = MATCH_FEATURE_COLS
        name_col = None
    else:
        return None

    if data_key not in feature_sets:
        return None

    df = feature_sets[data_key]

    if name_col:
        # Find player's most recent record
        player_data = df[df[name_col].str.contains(player_name, case=False, na=False)]
        if player_data.empty:
            # Try fuzzy match: last name + first initial
            # Cricsheet uses "V Kohli" format, input may be "Virat Kohli"
            parts = player_name.split()
            if len(parts) >= 2:
                last_name = parts[-1]
                first_initial = parts[0][0]
                # Match last name
                candidates = df[df[name_col].str.contains(last_name, case=False, na=False)]
                # Narrow by first initial if multiple matches
                if len(candidates[name_col].unique()) > 1:
                    candidates = candidates[
                        candidates[name_col].str.startswith(first_initial.upper(), na=False)
                    ]
                player_data = candidates
            else:
                player_data = df[df[name_col].str.contains(player_name, case=False, na=False)]

        if player_data.empty:
            return None

        # Use most recent record
        if "start_date" in player_data.columns:
            latest = player_data.sort_values("start_date").iloc[-1]
        else:
            latest = player_data.iloc[-1]
    elif data_key == "match_results" and player_name:
        # Match winner: compute CURRENT state features for this team vs its opponent
        # (identified from playing_xi). This avoids using stale features from the
        # team's last historical match.

        # Get all matches involving this team
        team_matches_all = df[
            df["team1"].str.contains(player_name, case=False, na=False) |
            df["team2"].str.contains(player_name, case=False, na=False)
        ].sort_values("start_date")

        if team_matches_all.empty:
            return None

        # Also grab the opponent team's matches to compute their current state
        # For now: use the team's latest row as base, but compute form freshly
        latest = team_matches_all.iloc[-1]
        is_team1 = bool(
            pd.notna(latest.get("team1")) and
            player_name.lower() in str(latest["team1"]).lower()
        )

        # Recompute form using team's last matches (from this team's perspective as "home")
        team_wins = []
        for _, m in team_matches_all.iterrows():
            if player_name.lower() in str(m.get("team1", "")).lower():
                team_wins.append(m.get("team1_won", 0))
            else:
                team_wins.append(1 - m.get("team1_won", 0))

        team_wins_series = pd.Series(team_wins)
        win_rate_5 = team_wins_series.tail(5).mean() if len(team_wins_series) >= 1 else None
        win_rate_10 = team_wins_series.tail(10).mean() if len(team_wins_series) >= 3 else None
        career_win_rate = team_wins_series.mean()

        # Grab the latest ELO from this team's most recent match
        # (whether they were team1 or team2, we need their own ELO)
        if player_name.lower() in str(latest.get("team1", "")).lower():
            team_elo = latest.get("team1_elo")
        else:
            team_elo = latest.get("team2_elo")

        features = {
            "_is_team1": is_team1,
            # Form features — recomputed freshly
            "team1_win_rate_5": win_rate_5 if is_team1 else latest.get("team2_win_rate_5"),
            "team1_win_rate_10": win_rate_10 if is_team1 else latest.get("team2_win_rate_10"),
            "team1_career_win_rate": career_win_rate if is_team1 else latest.get("team2_career_win_rate"),
            "team2_win_rate_5": latest.get("team2_win_rate_5") if is_team1 else win_rate_5,
            "team2_win_rate_10": latest.get("team2_win_rate_10") if is_team1 else win_rate_10,
            "team2_career_win_rate": latest.get("team2_career_win_rate") if is_team1 else career_win_rate,
            # ELO — use fresh ELO (from this team's last match)
            "team1_elo": team_elo if is_team1 else latest.get("team2_elo"),
            "team2_elo": latest.get("team2_elo") if is_team1 else team_elo,
            # Other features from the latest record (these are OK to use)
            "h2h_venue_matches": latest.get("h2h_venue_matches", 0),
            "h2h_venue_win_rate": latest.get("h2h_venue_win_rate"),
            "team1_is_home": latest.get("team1_is_home", 0),
            "team2_is_home": latest.get("team2_is_home", 0),
            "toss_elected_bat": latest.get("toss_elected_bat", 0),
            "toss_elected_field": latest.get("toss_elected_field", 0),
            "match_number": latest.get("match_number", 0),
            "is_playoff": latest.get("is_playoff", 0),
        }
        features["elo_diff"] = (features.get("team1_elo") or 1500) - (features.get("team2_elo") or 1500)

        return features
    else:
        # Team totals - use most recent
        if "start_date" in df.columns:
            latest = df.sort_values("start_date").iloc[-1]
        else:
            latest = df.iloc[-1]

    # Extract features — include model features plus any career/recent rate columns
    # (used for blending in post-processing)
    available_cols = [c for c in feature_cols if c in df.columns]
    extra_rate_cols = [c for c in df.columns if c.startswith(("career_rate_", "recent_rate_", "last5_rate_")) and c not in available_cols]
    all_cols = available_cols + extra_rate_cols

    features = {}
    for col in all_cols:
        val = latest.get(col)
        if val is not None and not (isinstance(val, float) and pd.isna(val)):
            features[col] = float(val) if not isinstance(val, (int, float)) else val
        # Leave missing features absent — they'll become NaN at prediction time

    # Include the last match date (metadata — for activity filtering)
    if "start_date" in df.columns:
        last_date = latest.get("start_date")
        if last_date is not None:
            features["_last_match_date"] = str(last_date)

    return features


def run_pipeline(skip_download=False, skip_train=False, odds_path=None, force_download=False):
    """Run the full pipeline."""
    print("=" * 60)
    print("IPL BETTING MODEL PIPELINE")
    print("=" * 60)

    # ── Step 1: Download/update data ──
    if not skip_download:
        print("\n[1/5] Downloading IPL data from Cricsheet...")
        try:
            download_cricsheet(force=force_download)
        except Exception as e:
            print(f"Download failed: {e}")
            print("Checking for existing data...")

    # ── Step 2: Load and process data ──
    print("\n[2/5] Loading processed data...")
    try:
        deliveries, matches = get_processed_data()
        print(f"Loaded {len(deliveries)} deliveries")
    except FileNotFoundError:
        print("No data found. Running download first...")
        download_cricsheet(force=True)
        deliveries, matches = parse_cricsheet_data()

    # ── Step 3: Build features ──
    print("\n[3/5] Building features...")
    if matches is not None:
        print(f"Loaded {len(matches)} match info records")
    feature_sets = build_all_features(deliveries, matches)

    # ── Step 4: Train models ──
    if not skip_train:
        print("\n[4/5] Training models...")
        results = train_all_models(feature_sets)

        # Generate calibration plots
        models_dict = {name: model for name, (model, _) in results.items()}
        if models_dict:
            print("\nGenerating calibration plots...")
            plot_calibration_curves(models_dict, feature_sets, MARKETS)
            plot_feature_importance(models_dict, MARKETS, feature_sets)
    else:
        print("\n[4/5] Loading existing models...")
        models_dict = load_models()
        if not models_dict:
            print("No trained models found. Running training...")
            results = train_all_models(feature_sets)
            models_dict = {name: model for name, (model, _) in results.items()}

    # ── Step 5: Compare with bookmaker odds ──
    print("\n[5/5] Comparing with bookmaker odds...")

    # Load playing XI first — needed for filtering match_winner odds
    playing_xi = load_playing_xi()
    if playing_xi:
        print(f"  Loaded playing XI for {len(playing_xi)} teams: {', '.join(playing_xi.keys())}")

    # Always load manual odds first (odds_input.csv)
    manual_odds = load_odds(odds_path)

    # Try live odds from API as supplement (not replacement)
    if not odds_path:
        live_odds = fetch_live_odds(save=False)
        if not live_odds.empty:
            print(f"  Fetched {len(live_odds)} live markets from The Odds API")

            # Append match_winner odds to odds_input.csv — only for today's match
            match_winner_odds = live_odds[live_odds["market"] == "match_winner"]
            if not match_winner_odds.empty and playing_xi:
                # Filter to only teams in the playing XI
                xi_teams = set(playing_xi.keys())
                match_winner_odds = match_winner_odds[
                    match_winner_odds["player"].isin(xi_teams)
                ]

            if not match_winner_odds.empty:
                manual_no_mw = manual_odds[manual_odds["market"] != "match_winner"] if not manual_odds.empty else pd.DataFrame()
                updated_csv = pd.concat([manual_no_mw[["player", "market", "odds"]], match_winner_odds[["player", "market", "odds"]]], ignore_index=True)
                csv_path = odds_path or (PROJECT_DIR / "odds_input.csv")
                updated_csv[["player", "market", "odds"]].to_csv(csv_path, index=False)
                print(f"  Added {len(match_winner_odds)} match_winner odds to {csv_path}")

            # Also filter live odds used for predictions to today's match teams only
            if playing_xi:
                xi_teams = set(playing_xi.keys())
                live_odds = live_odds[
                    (live_odds["market"] != "match_winner") |
                    (live_odds["player"].isin(xi_teams))
                ]

            # Reload manual odds (CSV was just updated with filtered match_winner)
            manual_odds = load_odds(odds_path)

            # Combine: manual odds take precedence over live for same player/market
            odds_df = pd.concat([live_odds, manual_odds], ignore_index=True)
            odds_df = odds_df.drop_duplicates(subset=["player", "market"], keep="last")
            odds_df["implied_prob"] = 1.0 / odds_df["odds"]
        else:
            odds_df = manual_odds
    else:
        odds_df = manual_odds

    # Filter out match_winner — model's pre-match AUC is 0.50 (no signal after
    # removing first_inn_score which was a leakage feature). Re-enable when
    # we add proper team strength features (squad quality, injuries, venue-specific form).
    if not odds_df.empty:
        n_before = len(odds_df)
        odds_df = odds_df[odds_df["market"] != "match_winner"].reset_index(drop=True)
        n_removed = n_before - len(odds_df)
        if n_removed > 0:
            print(f"  Excluded {n_removed} match_winner markets (model has no pre-match signal)")

    if odds_df.empty:
        print("No odds data. Create odds_input.csv or set ODDS_API_KEY env var.")
        return

    # playing_xi already loaded above

    # Load batting regression model
    bat_reg_model, bat_reg_features = load_batting_regression()
    if bat_reg_model is not None:
        print("  Loaded batting regression model")

    # Generate predictions for each market in the odds file
    model_predictions = {}
    for _, row in odds_df.iterrows():
        player = row["player"]
        market = row["market"]
        model_name = map_market_to_model(market)

        if model_name not in models_dict:
            print(f"  No model for market: {market} (mapped to {model_name})")
            continue

        model = models_dict[model_name]
        features = build_prediction_features(feature_sets, player, market)

        if features is None:
            print(f"  No data for player: {player}")
            continue

        # Skip players with insufficient IPL history — predictions are unreliable
        # when career/recent features are NaN (model produces default values that
        # don't reflect the player's actual ability).
        MIN_INNINGS = 5
        if "runs" in market:
            innings = features.get("career_innings", 0)
        elif "wickets" in market:
            innings = features.get("career_bowl_innings", 0)
        else:
            innings = MIN_INNINGS  # pass-through for team markets

        if innings is None or pd.isna(innings) or innings < MIN_INNINGS:
            print(f"  Skipping {player}/{market}: only {innings} prior innings (need {MIN_INNINGS}+)")
            continue

        # Activity filter: skip players whose last match was over 1 year ago.
        # Stale players (missed entire IPL seasons) have unreliable predictions
        # because their career stats reflect a different era of their career.
        MAX_INACTIVE_DAYS = 365
        last_date_str = features.get("_last_match_date")
        if last_date_str:
            try:
                last_date = pd.to_datetime(last_date_str)
                days_inactive = (pd.Timestamp.now() - last_date).days
                if days_inactive > MAX_INACTIVE_DAYS:
                    print(f"  Skipping {player}/{market}: inactive for {days_inactive} days (threshold {MAX_INACTIVE_DAYS})")
                    continue
            except Exception:
                pass

        # Override opponent strength with XI-specific values if available
        xi_overrides = compute_xi_opponent_strength(feature_sets, playing_xi, player, market)
        if xi_overrides:
            features.update(xi_overrides)

        try:
            # ── BATTING: use regression model ──
            if "runs" in market and bat_reg_model is not None:
                threshold = int(market.replace("runs_", ""))
                try:
                    reg_features = bat_reg_features
                    X_reg = pd.DataFrame([{col: features.get(col, float("nan")) for col in reg_features}])
                    for col in X_reg.columns:
                        X_reg[col] = pd.to_numeric(X_reg[col], errors="coerce")

                    # Regression model: P(runs >= threshold) from predicted distribution
                    reg_prob = float(bat_reg_model.predict_threshold_prob(X_reg, threshold)[0])

                    # Blend with career/recent hit rate (regression model is better
                    # calibrated than binary classifier, so give it more weight)
                    t = market.replace("runs_", "")
                    career_rate = features.get(f"career_rate_{t}_runs")
                    recent_rate = features.get(f"recent_rate_{t}_runs")

                    last5_rate = features.get(f"last5_rate_{t}_runs")

                    if career_rate is not None and not pd.isna(career_rate):
                        # Effective recent: blend last 10 and last 5
                        effective_recent = recent_rate
                        if (recent_rate is not None and not pd.isna(recent_rate) and
                            last5_rate is not None and not pd.isna(last5_rate)):
                            effective_recent = 0.6 * recent_rate + 0.4 * last5_rate
                        elif last5_rate is not None and not pd.isna(last5_rate):
                            effective_recent = last5_rate

                        w_model, w_career, w_recent = 0.45, 0.25, 0.30
                        if effective_recent is not None and not pd.isna(effective_recent):
                            if career_rate > 0.1 and effective_recent < career_rate * 0.5:
                                w_model, w_career, w_recent = 0.20, 0.10, 0.70
                            prob = w_model * reg_prob + w_career * career_rate + w_recent * effective_recent
                        else:
                            prob = (w_model + w_recent) * reg_prob + w_career * career_rate
                    else:
                        prob = reg_prob

                    prob = float(np.clip(prob, 0.02, 0.98))
                    model_predictions[(player, market)] = prob
                    continue
                except Exception:
                    pass  # Fall through to binary classifier

            # ── NON-BATTING or fallback: use binary classifier ──
            try:
                trained_features = model.get_booster().feature_names
            except Exception:
                config = MARKETS.get(model_name, {})
                trained_features = config.get("features", [])

            if not trained_features:
                continue

            X = pd.DataFrame([{col: features.get(col, float("nan")) for col in trained_features if not col.startswith("_")}])
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")

            raw_prob = model.predict_proba(X)[0, 1]
            prob = _calibrate_prob(model, raw_prob)

            # For match_winner: model predicts P(team1 wins).
            if market == "match_winner" and not features.get("_is_team1", True):
                prob = 1.0 - prob

            # Blend model + career + recent + last-5 rates for wickets.
            if "wickets" in market:
                t = market.replace("wickets_", "")
                threshold = int(t)
                career_rate = features.get(f"career_rate_{t}_wickets")
                recent_rate = features.get(f"recent_rate_{t}_wickets")
                last5_rate = features.get(f"last5_rate_{t}_wickets")

                if career_rate is not None and not pd.isna(career_rate):
                    # Effective "recent" blend: 60% last-10, 40% last-5
                    effective_recent = recent_rate
                    if (recent_rate is not None and not pd.isna(recent_rate) and
                        last5_rate is not None and not pd.isna(last5_rate)):
                        effective_recent = 0.6 * recent_rate + 0.4 * last5_rate
                    elif last5_rate is not None and not pd.isna(last5_rate):
                        effective_recent = last5_rate

                    if threshold >= 3:
                        w_model, w_career, w_recent = 0.25, 0.30, 0.45
                    else:
                        w_model, w_career, w_recent = 0.45, 0.25, 0.30

                    if effective_recent is not None and not pd.isna(effective_recent):
                        # Form-break detection: if recent form is much worse
                        # than career, the player has declined — trust recent more.
                        if career_rate > 0.05 and effective_recent < career_rate * 0.5:
                            w_model, w_career, w_recent = 0.15, 0.10, 0.75

                        prob = w_model * prob + w_career * career_rate + w_recent * effective_recent
                    else:
                        prob = (w_model + w_recent) * prob + w_career * career_rate

            prob = float(np.clip(prob, 0.02, 0.98))
            model_predictions[(player, market)] = prob
        except Exception as e:
            print(f"  Prediction failed for {player}/{market}: {e}")

    # Find value bets
    value_bets = find_value_bets(odds_df, model_predictions)

    # Build historical correlation matrix for multi construction
    corr_matrix = build_historical_correlations(feature_sets)
    multis = suggest_multis(value_bets, corr_matrix=corr_matrix)

    # Output results
    save_output(value_bets, multis)

    print("\nPipeline complete!")
    return value_bets, multis


def main():
    parser = argparse.ArgumentParser(description="IPL Betting Model Pipeline")
    parser.add_argument("--skip-download", action="store_true", help="Skip data download")
    parser.add_argument("--skip-train", action="store_true", help="Use existing trained models")
    parser.add_argument("--odds", type=str, default=None, help="Path to odds CSV file")
    parser.add_argument("--force-download", action="store_true", help="Force re-download data")
    args = parser.parse_args()

    run_pipeline(
        skip_download=args.skip_download,
        skip_train=args.skip_train,
        odds_path=args.odds,
        force_download=args.force_download,
    )


if __name__ == "__main__":
    main()
