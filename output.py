"""Odds comparison, value bet detection, and multi leg suggestions."""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
ODDS_PATH = PROJECT_DIR / "odds_input.csv"


def load_odds(path=None):
    """Load today's odds from CSV.

    Expected CSV format:
        player,market,odds
        Virat Kohli,runs_15,1.55
        Jasprit Bumrah,wickets_1,1.80
        Team Total,over_160,1.90
        Mumbai Indians,match_winner,2.10

    Odds are decimal format.
    """
    path = path or ODDS_PATH
    if not Path(path).exists():
        print(f"No odds file found at {path}")
        print("Create odds_input.csv with columns: player, market, odds")
        return pd.DataFrame(columns=["player", "market", "odds"])

    # Find the header row (skip any preamble lines like "CSV Data")
    skip_rows = 0
    with open(path) as f:
        for i, line in enumerate(f):
            if line.strip().startswith("player,market,odds"):
                skip_rows = i
                break

    df = pd.read_csv(path, skiprows=skip_rows)
    required = ["player", "market", "odds"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"odds_input.csv missing columns: {missing}")

    df["odds"] = pd.to_numeric(df["odds"], errors="coerce")
    df = df.dropna(subset=["odds"])
    df["implied_prob"] = 1.0 / df["odds"]

    return df


def map_market_to_model(market_str):
    """Map odds CSV market string to model market name.

    Mapping:
        runs_15 -> player_runs_15
        runs_30 -> player_runs_30
        runs_50 -> player_runs_50
        wickets_1 -> player_wickets_1
        wickets_2 -> player_wickets_2
        over_160 -> first_innings_over_160
        match_winner -> match_winner
    """
    # Auto-map runs_N -> player_runs_N, wickets_N -> player_wickets_N
    if market_str.startswith("runs_"):
        return f"player_{market_str}"
    if market_str.startswith("wickets_"):
        return f"player_{market_str}"
    if market_str.startswith("over_"):
        return f"first_innings_{market_str}"
    return market_str


def kelly_fraction(model_prob, decimal_odds, fraction=0.10, max_kelly=0.05):
    """Calculate fractional Kelly criterion.

    Full Kelly % = (bp - q) / b
    where b = decimal_odds - 1, p = model_prob, q = 1 - p

    We use fractional Kelly (default 1/4 Kelly) to be conservative — full Kelly
    is too aggressive for sports betting where model uncertainty is high.

    Args:
        fraction: Kelly divisor (0.25 = quarter Kelly, recommended for sports)
        max_kelly: hard cap per bet (default 5% of bankroll)
    """
    b = decimal_odds - 1
    p = model_prob
    q = 1 - p

    if b <= 0 or p <= 0:
        return 0.0

    full_kelly = (b * p - q) / b
    fractional = full_kelly * fraction
    return max(0.0, min(fractional, max_kelly))


def find_value_bets(odds_df, model_predictions, edge_threshold=0.03):
    """Compare model probabilities against implied probabilities.

    Args:
        odds_df: DataFrame with columns [player, market, odds, implied_prob]
        model_predictions: dict of {(player, market): model_probability}
        edge_threshold: minimum edge to flag (default 5%)

    Returns:
        DataFrame of value bets sorted by edge
    """
    results = []

    for _, row in odds_df.iterrows():
        player = row["player"]
        market = row["market"]
        odds = row["odds"]
        implied = row["implied_prob"]

        model_key = (player, market)
        if model_key not in model_predictions:
            continue

        model_prob = model_predictions[model_key]
        edge = model_prob - implied

        if edge > edge_threshold:
            fair_odds = 1.0 / model_prob if model_prob > 0 else float("inf")
            kf = kelly_fraction(model_prob, odds)

            results.append({
                "player": player,
                "market": market,
                "model_prob": round(model_prob, 4),
                "implied_prob": round(implied, 4),
                "edge": round(edge, 4),
                "edge_pct": f"{edge * 100:.1f}%",
                "book_odds": odds,
                "fair_odds": round(fair_odds, 2),
                "kelly_fraction": round(kf, 4),
                "kelly_pct": f"{kf * 100:.2f}%",
            })

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("edge", ascending=False).reset_index(drop=True)
    return result_df


def build_historical_correlations(feature_sets=None):
    """Compute historical co-occurrence rates between outcome types.

    Returns a dict of ((player_a, market_a), (player_b, market_b)) -> correlation.
    For efficiency, computes at the category level (e.g., runs_30 with wickets_1)
    rather than per-player-pair.
    """
    if feature_sets is None:
        return {}

    corr_cache = {}

    # Load batting and bowling target columns
    batting = feature_sets.get("batting")
    bowling = feature_sets.get("bowling")

    if batting is None or bowling is None:
        return corr_cache

    # Build per-match outcome indicators
    bat_targets = [c for c in batting.columns if c.startswith("target_") and c.endswith("_runs")]
    bowl_targets = [c for c in bowling.columns if c.startswith("target_") and c.endswith("_wickets")]

    # Aggregate to match level: did ANY batter hit this target in this match?
    # And cross-reference with bowling targets in the same match
    bat_by_match = batting.groupby("match_id")[bat_targets].max()
    bowl_by_match = bowling.groupby("match_id")[bowl_targets].max()

    combined = bat_by_match.join(bowl_by_match, how="inner")
    if combined.empty:
        return corr_cache

    # Compute pairwise correlations between all target columns
    # Suppress divide-by-zero warnings for zero-variance columns
    all_targets = bat_targets + bowl_targets
    for i, col_a in enumerate(all_targets):
        for col_b in all_targets[i + 1:]:
            if col_a == col_b:
                continue
            # Phi coefficient (correlation between binary variables)
            if col_a in combined.columns and col_b in combined.columns:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    corr_val = combined[col_a].corr(combined[col_b])
                if not np.isnan(corr_val):
                    corr_cache[(col_a, col_b)] = corr_val
                    corr_cache[(col_b, col_a)] = corr_val

    # Same-player correlations: if same batter, higher run thresholds are nested
    # (30+ implies 15+), so correlation should be very high
    for i, t_a in enumerate(bat_targets):
        for t_b in bat_targets[i + 1:]:
            # Same player, nested thresholds: mark as highly correlated
            corr_cache[("same_player", t_a, t_b)] = 0.9

    for i, t_a in enumerate(bowl_targets):
        for t_b in bowl_targets[i + 1:]:
            corr_cache[("same_player", t_a, t_b)] = 0.9

    return corr_cache


def _market_to_target(market_str):
    """Convert market string to target column name for correlation lookup."""
    if market_str.startswith("runs_"):
        threshold = market_str.replace("runs_", "")
        return f"target_{threshold}_runs"
    if market_str.startswith("wickets_"):
        threshold = market_str.replace("wickets_", "")
        return f"target_{threshold}_wickets"
    return market_str


def suggest_multis(value_bets_df, max_legs=4, max_correlation=0.3, corr_matrix=None):
    """Suggest multi/parlay combinations from value bets.

    Uses historical correlation data when available, falls back to heuristics.
    Prefers low-correlation legs for better multi value.
    """
    if value_bets_df.empty or len(value_bets_df) < 2:
        return []

    bets = value_bets_df.to_dict("records")

    def correlation_score(leg_a, leg_b):
        """Compute correlation between two legs using historical data + heuristics."""
        # Same player = always high correlation
        if leg_a["player"] == leg_b["player"]:
            return 0.85

        target_a = _market_to_target(leg_a["market"])
        target_b = _market_to_target(leg_b["market"])

        # Try historical correlation
        if corr_matrix:
            hist_corr = corr_matrix.get((target_a, target_b))
            if hist_corr is not None:
                return abs(hist_corr)

        # Fallback heuristics
        score = 0.0
        market_a = leg_a["market"]
        market_b = leg_b["market"]

        # Same market type (e.g., both runs_30)
        if market_a == market_b:
            score += 0.15

        # Both batting or both bowling (within same match, some correlation)
        both_runs = "runs" in market_a and "runs" in market_b
        both_wickets = "wickets" in market_a and "wickets" in market_b
        if both_runs:
            score += 0.08
        if both_wickets:
            score += 0.08

        # Cross-type (runs + wickets) = low correlation
        if ("runs" in market_a and "wickets" in market_b) or \
           ("wickets" in market_a and "runs" in market_b):
            score += 0.02

        return score

    # ── Build multis: try all combinations of 2-4 legs, score by edge / correlation ──
    from itertools import combinations

    candidates = []
    max_bets = min(len(bets), 15)  # Limit combinatorics
    top_bets = bets[:max_bets]

    for n_legs in range(2, max_legs + 1):
        for combo in combinations(range(max_bets), n_legs):
            legs = [top_bets[i] for i in combo]

            # Check all pairwise correlations
            max_corr = 0.0
            total_corr = 0.0
            n_pairs = 0
            for i in range(len(legs)):
                for j in range(i + 1, len(legs)):
                    corr = correlation_score(legs[i], legs[j])
                    max_corr = max(max_corr, corr)
                    total_corr += corr
                    n_pairs += 1

            # Skip if any pair is too correlated
            if max_corr > max_correlation:
                continue

            avg_corr = total_corr / max(n_pairs, 1)

            combined_odds = 1.0
            combined_prob = 1.0
            total_edge = 0.0
            for leg in legs:
                combined_odds *= leg["book_odds"]
                combined_prob *= leg["model_prob"]
                total_edge += leg["edge"]

            implied_combined = 1.0 / combined_odds
            multi_edge = combined_prob - implied_combined

            # Score: favour high edge, low correlation, reasonable number of legs
            score = multi_edge * (1 - avg_corr * 0.5)

            candidates.append({
                "legs": legs,
                "n_legs": n_legs,
                "combined_odds": round(combined_odds, 2),
                "combined_model_prob": round(combined_prob, 4),
                "combined_implied_prob": round(implied_combined, 4),
                "combined_edge": round(multi_edge, 4),
                "avg_correlation": round(avg_corr, 3),
                "max_correlation": round(max_corr, 3),
                "players": list({leg["player"] for leg in legs}),
                "score": score,
            })

    # Sort by score, deduplicate similar multis
    candidates.sort(key=lambda x: x["score"], reverse=True)

    # Keep top diverse multis (don't return 5 variants of the same 3 legs)
    selected = []
    seen_player_sets = []
    for multi in candidates:
        player_set = frozenset(multi["players"])
        market_set = frozenset(leg["market"] for leg in multi["legs"])
        sig = (player_set, market_set)

        # Skip if too similar to an already selected multi
        is_duplicate = False
        for prev_sig in seen_player_sets:
            overlap = len(sig[0] & prev_sig[0]) / max(len(sig[0] | prev_sig[0]), 1)
            if overlap > 0.7:
                is_duplicate = True
                break

        if not is_duplicate:
            selected.append(multi)
            seen_player_sets.append(sig)

        if len(selected) >= 5:
            break

    return selected


def format_output(value_bets_df, multis):
    """Format value bets and multi suggestions for display."""
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("IPL VALUE BETS - TODAY'S OPPORTUNITIES")
    output_lines.append("=" * 80)

    if value_bets_df.empty:
        output_lines.append("\nNo value bets found (model prob > implied prob + 5%).")
    else:
        output_lines.append(f"\nFound {len(value_bets_df)} value bet(s):\n")
        display_cols = [
            "player", "market", "model_prob", "implied_prob",
            "edge_pct", "book_odds", "fair_odds", "kelly_pct",
        ]
        available = [c for c in display_cols if c in value_bets_df.columns]
        output_lines.append(value_bets_df[available].to_string(index=False))

    if multis:
        output_lines.append("\n" + "=" * 80)
        output_lines.append("SUGGESTED MULTIS (Low-Correlation Legs)")
        output_lines.append("=" * 80)

        for i, multi in enumerate(multis[:5], 1):
            output_lines.append(f"\nMulti #{i} ({multi['n_legs']} legs) — "
                                f"Combined odds: {multi['combined_odds']:.2f}")
            output_lines.append(f"  Model prob: {multi['combined_model_prob']:.4f} | "
                                f"Implied: {multi['combined_implied_prob']:.4f} | "
                                f"Edge: {multi['combined_edge']:.4f}")
            for leg in multi["legs"]:
                output_lines.append(f"  • {leg['player']} — {leg['market']} "
                                    f"@ {leg['book_odds']} (edge: {leg['edge_pct']})")
    else:
        output_lines.append("\nNo multi suggestions (need 2+ value bets with low correlation).")

    output_lines.append("\n" + "=" * 80)
    return "\n".join(output_lines)


def save_output(value_bets_df, multis, path=None):
    """Save value bets to CSV and print formatted output."""
    path = path or PROJECT_DIR / "value_bets.csv"

    output_text = format_output(value_bets_df, multis)
    print(output_text)

    if not value_bets_df.empty:
        value_bets_df.to_csv(path, index=False)
        print(f"\nValue bets saved to {path}")

    # Save full output text
    text_path = PROJECT_DIR / "value_bets_report.txt"
    with open(text_path, "w") as f:
        f.write(output_text)
    print(f"Report saved to {text_path}")

    return output_text
