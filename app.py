"""Streamlit dashboard for IPL betting model.

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / "models"
FEATURES_DIR = PROJECT_DIR / "features" / "processed"
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"


st.set_page_config(
    page_title="IPL Betting Model",
    page_icon="🏏",
    layout="wide",
)


@st.cache_data
def load_metrics():
    path = MODELS_DIR / "metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_value_bets():
    path = PROJECT_DIR / "value_bets.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_report():
    path = PROJECT_DIR / "value_bets_report.txt"
    if path.exists():
        with open(path) as f:
            return f.read()
    return ""


def main():
    st.title("🏏 IPL Betting Model Dashboard")
    st.markdown("ML-powered value bet identification for IPL prop markets")

    # Sidebar
    st.sidebar.header("Configuration")
    st.sidebar.markdown("---")

    # Odds input
    st.sidebar.subheader("📝 Odds Input")
    odds_path = PROJECT_DIR / "odds_input.csv"
    if odds_path.exists():
        odds_df = pd.read_csv(odds_path)
        st.sidebar.success(f"Loaded {len(odds_df)} markets from odds_input.csv")
    else:
        st.sidebar.warning("No odds_input.csv found")
        odds_df = pd.DataFrame()

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Run pipeline:** `python pipeline.py`\n\n"
        "**Update odds:** Edit `odds_input.csv`"
    )

    # Main content - tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💰 Value Bets", "📊 Model Performance", "📈 Calibration",
        "🏏 Odds Input", "📋 Full Report"
    ])

    # ── Tab 1: Value Bets ──
    with tab1:
        st.header("Today's Value Bets")

        value_bets = load_value_bets()
        if value_bets.empty:
            st.info(
                "No value bets generated yet. Run `python pipeline.py` first."
            )
        else:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Value Bets Found", len(value_bets))
            with col2:
                avg_edge = value_bets["edge"].mean() if "edge" in value_bets.columns else 0
                st.metric("Avg Edge", f"{avg_edge * 100:.1f}%")
            with col3:
                max_edge = value_bets["edge"].max() if "edge" in value_bets.columns else 0
                st.metric("Max Edge", f"{max_edge * 100:.1f}%")
            with col4:
                total_kelly = value_bets["kelly_fraction"].sum() if "kelly_fraction" in value_bets.columns else 0
                st.metric("Total Kelly", f"{total_kelly * 100:.2f}%")

            st.markdown("---")

            # Highlight value bets
            st.subheader("Ranked Value Bets")

            display_cols = [
                "player", "market", "model_prob", "implied_prob",
                "edge_pct", "book_odds", "fair_odds", "kelly_pct",
            ]
            available = [c for c in display_cols if c in value_bets.columns]

            st.dataframe(
                value_bets[available],
                use_container_width=True,
                hide_index=True,
            )

            # Multi suggestions
            st.markdown("---")
            st.subheader("Suggested Multis")

            report = load_report()
            if "SUGGESTED MULTIS" in report:
                multi_section = report.split("SUGGESTED MULTIS")[1]
                multi_section = multi_section.split("=" * 80)[0] if "=" * 80 in multi_section else multi_section
                st.text(multi_section.strip())
            else:
                st.info("No multi suggestions available.")

    # ── Tab 2: Model Performance ──
    with tab2:
        st.header("Model Performance Metrics")

        metrics = load_metrics()
        if not metrics:
            st.info("No metrics available. Run `python pipeline.py` first.")
        else:
            metrics_data = []
            for market, m in metrics.items():
                metrics_data.append({
                    "Market": market,
                    "Brier Score": f"{m.get('brier_score', 'N/A'):.4f}" if isinstance(m.get('brier_score'), (int, float)) else "N/A",
                    "ROC AUC": f"{m.get('roc_auc', 'N/A'):.4f}" if isinstance(m.get('roc_auc'), (int, float)) else "N/A",
                    "Log Loss": f"{m.get('log_loss', 'N/A'):.4f}" if isinstance(m.get('log_loss'), (int, float)) else "N/A",
                    "Train Size": m.get("train_size", "N/A"),
                    "Val Size": m.get("val_size", "N/A"),
                    "Val Pos Rate": f"{m.get('val_pos_rate', 0):.3f}" if isinstance(m.get('val_pos_rate'), (int, float)) else "N/A",
                })

            st.dataframe(
                pd.DataFrame(metrics_data),
                use_container_width=True,
                hide_index=True,
            )

            # Performance interpretation
            st.markdown("---")
            st.markdown("""
            **Interpreting metrics:**
            - **Brier Score**: Lower is better (0 = perfect). < 0.20 is reasonable for sports.
            - **ROC AUC**: Higher is better (1.0 = perfect). > 0.60 shows predictive signal.
            - **Val Pos Rate**: Base rate — model must beat this to add value.
            """)

    # ── Tab 3: Calibration Curves ──
    with tab3:
        st.header("Calibration Curves")

        cal_path = NOTEBOOKS_DIR / "calibration_curves.png"
        if cal_path.exists():
            st.image(str(cal_path), use_container_width=True)
            st.markdown(
                "Points close to the diagonal indicate well-calibrated probabilities. "
                "Above the line = model underestimates, below = overestimates."
            )
        else:
            st.info("No calibration plots yet. Run `python pipeline.py` first.")

        # Feature importance
        st.markdown("---")
        st.subheader("Feature Importance")

        importance_files = sorted(NOTEBOOKS_DIR.glob("importance_*.png"))
        if importance_files:
            cols = st.columns(2)
            for i, img_path in enumerate(importance_files):
                with cols[i % 2]:
                    market_name = img_path.stem.replace("importance_", "")
                    st.markdown(f"**{market_name}**")
                    st.image(str(img_path), use_container_width=True)
        else:
            st.info("No feature importance plots yet.")

    # ── Tab 4: Odds Input ──
    with tab4:
        st.header("Current Odds Input")

        if not odds_df.empty:
            st.dataframe(odds_df, use_container_width=True, hide_index=True)

            # Quick add form
            st.markdown("---")
            st.subheader("Add New Odds")
            with st.form("add_odds"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    new_player = st.text_input("Player Name")
                with col2:
                    new_market = st.selectbox("Market", [
                        "runs_15", "runs_30", "runs_50",
                        "wickets_1", "wickets_2",
                        "over_160", "match_winner",
                    ])
                with col3:
                    new_odds = st.number_input("Decimal Odds", min_value=1.01, value=2.00, step=0.05)

                submitted = st.form_submit_button("Add")
                if submitted and new_player:
                    new_row = pd.DataFrame([{
                        "player": new_player,
                        "market": new_market,
                        "odds": new_odds,
                    }])
                    updated = pd.concat([odds_df, new_row], ignore_index=True)
                    updated.to_csv(odds_path, index=False)
                    st.success(f"Added {new_player} {new_market} @ {new_odds}")
                    st.rerun()
        else:
            st.warning("No odds_input.csv found. Create one with columns: player, market, odds")

    # ── Tab 5: Full Report ──
    with tab5:
        st.header("Full Report")

        report = load_report()
        if report:
            st.text(report)
        else:
            st.info("No report generated yet. Run `python pipeline.py` first.")


if __name__ == "__main__":
    main()
