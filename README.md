# AusSuperPredictor

A Python desktop application that predicts the next-day direction of the ASX 200 index using a Random Forest classifier. Built with **tkinter** following the **MVVM** architectural pattern.

## Features

- **Daily ASX 200 data** sourced from the Investing.com API with accurate daily closes
- **ASX 200 futures** fetched from Investing.com (pair 8824) for the `futures_premium` feature — measures overnight divergence between futures and the index
- **10 global market indicators** fetched via yfinance and Investing.com (S&P 500 futures, VIX, ASX VIX, gold, copper, oil, BHP.AX, AUD/USD, AU 10Y yield, US 10Y yield)
- **Bond yield features** — yield level, daily change (diff), and AU–US yield spread
- **Configurable OHLC field** — Investing.com sources support `price_field` to select open/high/low/close per ticker (e.g. `last_openRaw` for S&P futures to avoid look-ahead bias)
- **42 engineered features** including lagged returns, technical indicators (RSI, MACD, EMA), volatility metrics, commodity returns, futures premium, bond yield spread, and VIX spread
- **Random Forest classifier** with configurable hyperparameters and 80/20 time-series split
- **Calibration analysis** — ECE/MCE metrics computed on the test set after each training run
- **Confidence-based decisions** — predictions are classified as POSITIVE_EXPECTED, NEGATIVE_EXPECTED, or NEUTRAL with confidence levels (VERY_LOW → VERY_HIGH)
- **Live intraday price** — fetches the current ASX 200 price via yfinance `^AXJO` during market hours
- **Prediction history** — every prediction is saved to CSV with all feature inputs, model version, and market regime
- **Retrospective back-fill** — actual results are automatically matched to past predictions with descriptive labels (CORRECT_UP, WRONG_DOWN, etc.)
- **Hypothetical returns** — tracks what you would have earned following each signal
- **Market regime classification** — labels each prediction as TRENDING_UP, TRENDING_DOWN, VOLATILE, or RANGE_BOUND
- **Model version tracking** — records which model version made each prediction
- **Performance dashboard** — dedicated UI tab with accuracy breakdowns by confidence, day-of-week, market regime, and model version
- **Drift detection** — warns when recent accuracy drops significantly below the long-term average
- **Historical performance log** — daily snapshots of aggregate metrics saved to track accuracy evolution
- **Email notifications** — CLI predictions can send an HTML email with results via Gmail SMTP (credentials stored in `.env`, not in config)
- **Configurable data folder** — all data files stored in a configurable directory (default: `~/Library/Application Support/AusSuperPredictor/data`)
- **Config-driven market sources** — market indicators defined in `config.json` with support for both yfinance and Investing.com sources
- **Auto-run** — optional scheduled prediction at configurable Sydney time
- **CLI mode** — headless `--predict` and `--train` flags for cron/launchd automation

## Screenshots

The application has two main tabs:

- **Log** — real-time output of data updates, training, and predictions
- **Performance** — dashboard with accuracy stats, hypothetical returns, regime/version breakdowns, threshold analysis, and recent prediction history

## Project Structure

```
AusSuperPredictor/
├── AusSuperPredictor.py          # Entry point (v2.1.0)
├── config.json                   # Configuration (market sources, model params, email, schedule)
├── .env                          # Email password (gitignored)
├── requirements.txt              # Dependencies
├── models/
│   ├── data_manager.py           # Data fetching, storage, history, performance analysis
│   └── model_manager.py          # Feature engineering, training, prediction, calibration
├── viewmodels/
│   └── main_viewmodel.py         # Business logic, threading, state management
├── views/
│   ├── main_window.py            # Main application window with tabbed notebook
│   ├── log_panel.py              # Colour-coded scrolling log output
│   ├── performance_panel.py      # Performance analytics dashboard
│   └── file_viewer.py            # Popup viewer for CSV/model files
├── utils/
│   ├── config_manager.py         # JSON config loader
│   ├── email_sender.py           # SMTP email sender with .env support
│   ├── queue_handler.py          # Thread-safe message queue
│   └── time_utils.py             # Sydney timezone utilities
└── data/                         # Generated at runtime
    ├── australian_super_daily.csv # ASX 200 daily price data
    ├── asx200history.csv          # Prediction history with outcomes
    ├── performance_log.csv        # Daily performance snapshots
    ├── model.pkl                  # Trained Random Forest model
    ├── features.pkl               # Feature column names
    └── feature_names.txt          # Human-readable feature list
```

## Architecture

The app follows the **Model-View-ViewModel (MVVM)** pattern:

| Layer | Responsibility |
|-------|---------------|
| **Models** | Data fetching (Investing.com API, yfinance), CSV storage, ML training/prediction, calibration, performance analysis |
| **ViewModels** | Threading, state management, business logic orchestration |
| **Views** | tkinter UI — settings panel, log output, performance dashboard |
| **Utils** | Configuration, thread-safe queuing, Sydney time, email notifications |

## Requirements

- **Python 3.10+** (developed on 3.13.1)
- macOS / Linux / Windows

### Key Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Random Forest classifier |
| `yfinance` | Market indicator data & live ASX 200 price |
| `requests` | Investing.com API calls |
| `joblib` | Model serialisation |
| `tkinter` | GUI (included with Python) |

## Installation

```bash
# Clone the repository
git clone https://github.com/mpinnuck/AusSuperPredictor.git
cd AusSuperPredictor

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install pandas numpy scikit-learn yfinance requests joblib
```

## Usage

### GUI Mode

```bash
python AusSuperPredictor.py
```

### CLI Mode

```bash
# Run prediction (update data → predict → email → save)
python AusSuperPredictor.py --predict

# Run training (update data → train model)
python AusSuperPredictor.py --train
```

### Workflow

1. **Update Data** — fetches the latest ASX 200 daily closes and market indicators
2. **Train Model** — engineers 32 features and trains a Random Forest classifier with calibration analysis
3. **Run Prediction** — predicts next trading day direction with confidence level and decision signal

Each prediction automatically:
- Back-fills actual results for previous predictions
- Logs performance statistics and drift warnings
- Saves the prediction with all feature inputs, market regime, and model version
- Records a daily performance snapshot
- Sends an email notification (CLI mode, if configured)

### Email Setup

1. Create a `.env` file in the project root:
   ```
   ASP_EMAIL_PASSWORD=your_gmail_app_password
   ```
2. Configure the email section in `config.json` (no password here):
   ```json
   "email": {
       "enabled": true,
       "smtp_server": "smtp.gmail.com",
       "smtp_port": 587,
       "username": "you@gmail.com",
       "from": "you@gmail.com",
       "to": "recipient@example.com"
   }
   ```
3. The `.env` file is gitignored — your credentials stay local.

### Configuration

Edit `config.json` to adjust:

```json
{
    "data_folder": "~/Library/Application Support/AusSuperPredictor/data",
    "data": {
        "local_csv_path": "australian_super_daily.csv",
        "start_date": "01/07/2008",
        "end_date_offset_days": 1,
        "fund_option": "Australian Shares"
    },
    "model": {
        "n_estimators": 100,
        "max_depth": 7,
        "min_samples_split": 10,
        "min_samples_leaf": 15,
        "random_state": 42
    },
    "market_sources": [
        {"name": "asx_futures", "ticker": "8824", "source": "investing", "shift": false},
        {"name": "sp500_futures", "ticker": "ES=F", "shift": false},
        {"name": "vix", "ticker": "^VIX", "shift": true}
    ]
}
```

Market sources support two providers:
- `"source": "investing"` — Investing.com historical API (use pair ID as ticker)
- Default (omitted) — yfinance

The `shift` flag controls time-zone alignment: `true` shifts the series forward one day for US-session data so it aligns with ASX trading dates.

#### `price_field` — OHLC selection for Investing.com sources

Investing.com sources can specify which daily OHLC value to use:

| `price_field` value | Description |
|---|---|
| `"last_closeRaw"` | Daily close (default if omitted) |
| `"last_openRaw"` | Daily open |
| `"last_maxRaw"` | Daily high |
| `"last_minRaw"` | Daily low |

Example — S&P 500 futures uses the **open** to avoid look-ahead bias (the futures close occurs after the ASX has already closed):

```json
{
    "name": "sp500_futures",
    "ticker": "1175153",
    "source": "investing",
    "shift": false,
    "category": "futures",
    "price_field": "last_openRaw"
}
```

This field is used for both historical data fetching and live prediction quotes.

#### Market source categories

The `category` field controls how features are engineered from each source:

| Category | Features created |
|---|---|
| `futures` | `_return` (pct_change), `futures_premium` (vs ASX close) |
| `volatility` | `_change` (pct_change), `_level`, cross-source `vix_spread` |
| `bond_yield` | `_change` (diff), `_level`, cross-source `yield_spread` |
| `commodity` | `_return` (pct_change) |
| `currency` | `_return` (pct_change) |

## Data Sources

| Source | Data | Method |
|--------|------|--------|
| [Investing.com](https://www.investing.com/indices/s-p-asx-200-historical-data) | ASX 200 daily OHLC | REST API (pair ID 171) |
| [Investing.com](https://www.investing.com/indices/s-p-asx-200-futures) | ASX 200 Futures | REST API (pair ID 8824) |
| [Investing.com](https://www.investing.com/indices/us-spx-500-futures) | S&P 500 Futures | REST API (pair ID 1175153) |
| [Yahoo Finance](https://finance.yahoo.com/) | VIX, ASX VIX, gold, copper, oil, BHP.AX, AUD/USD | yfinance package |
| [Investing.com](https://www.investing.com/rates-bonds/) | AU/US 10-year bond yields | REST API (bond pair IDs) |
| [Yahoo Finance](https://finance.yahoo.com/) | Live ASX 200 intraday price | yfinance `^AXJO` fast_info |

## Confidence Levels

| Level | P(up) range |
|-------|------------|
| VERY_HIGH | ≥ 90% or ≤ 10% |
| HIGH | 80–89% or 11–20% |
| MODERATE | 70–79% or 21–30% |
| LOW | 60–69% or 31–40% |
| VERY_LOW | 41–59% |

## Performance Tracking

The app maintains three levels of historical data:

1. **`asx200history.csv`** — every individual prediction with 50+ columns: metadata, all feature inputs, outcome, result label, hypothetical return, market regime, model version
2. **`performance_log.csv`** — daily aggregate snapshots: overall accuracy, rolling accuracy, best threshold, drift status
3. **Performance tab** — live dashboard computed from history on each refresh

## Disclaimer

This project is for **educational and research purposes only**. It is not financial advice. Past model performance does not guarantee future results. Always do your own research before making investment decisions.

## License

MIT
