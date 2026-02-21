# AusSuperPredictor

A Python desktop application that predicts the next-day direction of the ASX 200 index using a Random Forest classifier. Built with **tkinter** following the **MVVM** architectural pattern.

## Features

- **Daily ASX 200 data** sourced from the Investing.com API with accurate daily closes
- **8 global market indicators** fetched via yfinance (S&P 500 futures, VIX, gold, copper, oil, BHP.AX, AUD/USD, ASX futures)
- **31 engineered features** including lagged returns, technical indicators (RSI, MACD, EMA), volatility metrics, and commodity returns
- **Random Forest classifier** with configurable hyperparameters and 80/20 time-series split
- **Calibration analysis** — ECE/MCE metrics computed on the test set after each training run
- **Confidence-based decisions** — predictions are classified as POSITIVE_EXPECTED, NEGATIVE_EXPECTED, or NEUTRAL with confidence levels (VERY_LOW → VERY_HIGH)
- **Live price fetch** — grabs the current ASX 200 price during market hours for intraday predictions
- **Prediction history** — every prediction is saved to CSV with all feature inputs, model version, and market regime
- **Retrospective back-fill** — actual results are automatically matched to past predictions with descriptive labels (CORRECT_UP, WRONG_DOWN, etc.)
- **Hypothetical returns** — tracks what you would have earned following each signal
- **Market regime classification** — labels each prediction as TRENDING_UP, TRENDING_DOWN, VOLATILE, or RANGE_BOUND
- **Model version tracking** — records which model version made each prediction
- **Performance dashboard** — dedicated UI tab with accuracy breakdowns by confidence, day-of-week, market regime, and model version
- **Drift detection** — warns when recent accuracy drops significantly below the long-term average
- **Historical performance log** — daily snapshots of aggregate metrics saved to track accuracy evolution
- **Auto-run** — optional scheduled prediction at 15:30 Sydney time

## Screenshots

The application has two main tabs:

- **Log** — real-time output of data updates, training, and predictions
- **Performance** — dashboard with accuracy stats, hypothetical returns, regime/version breakdowns, threshold analysis, and recent prediction history

## Project Structure

```
AusSuperPredictor/
├── AusSuperPredictor.py          # Entry point (v1.5.0)
├── config.json                   # Configuration
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
| **Utils** | Configuration, thread-safe queuing, Sydney time calculations |

## Requirements

- **Python 3.10+** (developed on 3.13.1)
- macOS / Linux / Windows

### Key Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Random Forest classifier |
| `yfinance` | Market indicator data |
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

```bash
python AusSuperPredictor.py
```

### Workflow

1. **Update Data** — fetches the latest ASX 200 daily closes and market indicators
2. **Train Model** — engineers 31 features and trains a Random Forest classifier with calibration analysis
3. **Run Prediction** — predicts next trading day direction with confidence level and decision signal

Each prediction automatically:
- Back-fills actual results for previous predictions
- Logs performance statistics and drift warnings
- Saves the prediction with all feature inputs, market regime, and model version
- Records a daily performance snapshot

### Configuration

Edit `config.json` to adjust:

```json
{
    "data": {
        "local_csv_path": "data/australian_super_daily.csv",
        "start_date": "01/07/2008",
        "end_date_offset_days": 1,
        "fund_option": "Australian Shares"
    },
    "model": {
        "save_path": "data/model.pkl",
        "features_save_path": "data/features.pkl",
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "random_state": 42
    }
}
```

## Data Sources

| Source | Data | Method |
|--------|------|--------|
| [Investing.com](https://www.investing.com/indices/s-p-asx-200-historical-data) | ASX 200 daily OHLC | REST API (pair ID 171) |
| [Yahoo Finance](https://finance.yahoo.com/) | S&P 500 futures, VIX, gold, copper, oil, BHP.AX, AUD/USD, ASX futures | yfinance package |

## Performance Tracking

The app maintains three levels of historical data:

1. **`asx200history.csv`** — every individual prediction with 40+ columns: metadata, all feature inputs, outcome, result label, hypothetical return, market regime, model version
2. **`performance_log.csv`** — daily aggregate snapshots: overall accuracy, rolling accuracy, best threshold, drift status
3. **Performance tab** — live dashboard computed from history on each refresh

## Disclaimer

This project is for **educational and research purposes only**. It is not financial advice. Past model performance does not guarantee future results. Always do your own research before making investment decisions.

## License

MIT
