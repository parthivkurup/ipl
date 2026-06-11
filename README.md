# IPL Betting Model

A machine-learning pipeline for predicting IPL outcomes and surfacing **value bets**
against bookmaker odds. It models match winners, first-innings totals, and per-player
runs/wickets thresholds, then compares model probabilities to market odds to flag
positive-expected-value wagers.

> ⚠️ **For research and educational use only.** This is a modelling project, not
> betting advice. Gamble responsibly and within the law in your jurisdiction.

## What it does

- **Match winner** — ELO-based prediction (history + current-season form + home venue).
- **First-innings total** — probability of the first innings going over 160.
- **Player markets** — calibrated classifiers for a batter exceeding various run
  thresholds (10–100) and a bowler taking 1–5+ wickets.
- **Value detection** — converts model probabilities to fair odds, compares against the
  bookmaker line, and reports value bets and multi-leg suggestions.

## Setup

Requires Python 3.10+.

```sh
pip install -r requirements.txt
```

The large raw `data/` directory (Cricsheet match dumps) is **not** tracked in this repo
— `pipeline.py` downloads it on first run.

## Usage

### Full pipeline

```sh
python pipeline.py                  # download data, train, predict, find value bets
python pipeline.py --skip-download  # reuse existing data/
python pipeline.py --skip-train     # reuse existing models/
python pipeline.py --odds odds.csv  # use a custom odds file
```

### Odds input

Provide today's odds one of two ways:

1. Create `odds_input.csv`:
   ```csv
   player,market,odds
   Virat Kohli,runs_15,1.55
   Jasprit Bumrah,wickets_1,1.80
   Team Total,over_160,1.90
   Mumbai Indians,match_winner,2.10
   ```
2. Or set an Odds API key in a `.env` file (kept out of git):
   ```
   ODDS_API_KEY=your_key_here
   ```

Value bets are written to `value_bets.csv` / `value_bets_report.txt`.

### Predict remaining fixtures

```sh
python predict_remaining.py                  # uses fixtures.csv
python predict_remaining.py --from-match 34  # only matches >= 34
python predict_remaining.py --out preds.csv
```

`fixtures.csv` schema (one row per match): `match_number,date,team1,team2,venue[,winner]`.
If `winner` is present, the match is treated as played and used to update ELO before
predicting later fixtures.

### Dashboard

```sh
streamlit run app.py
```

An interactive view of model metrics, feature importances, and predictions.

## Layout

| Path | Purpose |
|------|---------|
| `pipeline.py` | End-to-end orchestration (download → features → train → predict → value bets) |
| `predict_remaining.py` | ELO-based winner predictions for upcoming fixtures |
| `output.py` | Odds comparison, value-bet detection, multi-leg suggestions |
| `app.py` | Streamlit dashboard |
| `features/` | Feature engineering + processed feature tables (`.parquet`) |
| `models/` | Trained models (`.joblib`), calibration, training code, `metrics.json` |
| `notebooks/` | Feature-importance and calibration plots |
| `data/` | Raw Cricsheet data (gitignored; downloaded on first run) |

## Notes

The trained `match_winner` classifier is intentionally bypassed in
`predict_remaining.py`: with honest, non-leaking features its held-out AUC is ~0.50, so
ELO is used instead. Player and totals markets use proper probability calibration.
