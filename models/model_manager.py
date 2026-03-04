"""
Model Manager - single responsibility for ML model operations
Now saves models to the data folder
"""
import pandas as pd
import numpy as np
import json as _json
import time
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from utils.app_config import AppConfig


def get_max_rolling_lag_window(config: AppConfig) -> int:
    """
    Determine the largest rolling or lag window needed for feature engineering.
    Includes lagged returns and config-driven technical indicators.
    """
    # Lagged returns (hardcoded in engineer_features)
    lagged_lags = [0, 1, 2, 3, 5]
    max_lag = max(lagged_lags) if lagged_lags else 0

    # Technical indicators
    indicators = config.technical_indicators
    if not indicators:
        from utils.app_config import TechnicalIndicator
        indicators = [
            TechnicalIndicator(type='macd', fast=12, slow=26, signal=9),
            TechnicalIndicator(type='rsi', period=14),
        ]
    max_window = max_lag
    for ind in indicators:
        if ind.type == 'macd':
            max_window = max(max_window, ind.fast or 12, ind.slow or 26, ind.signal or 9)
        elif ind.type == 'rsi':
            max_window = max(max_window, ind.period or 14)
    return max_window

class ModelManager:
    """Manages all ML model operations"""
    
    def __init__(self, config: AppConfig, log_queue=None):
        self.config = config
        self.log_queue = log_queue
        self.model = None
        self.feature_columns = None
        self.model_path = config.model.save_path
        self.features_path = config.model.features_save_path
        
        # Get params from typed config
        self.n_estimators = config.model.n_estimators
        self.max_depth = config.model.max_depth
        self.min_child_weight = config.model.min_child_weight
        self.learning_rate = config.model.learning_rate
        self.subsample = config.model.subsample
        self.colsample_bytree = config.model.colsample_bytree
        self.random_state = config.model.random_state
        self.training_snapshot_path = os.path.join(
            os.path.dirname(self.model_path), 'last_training.json'
        )
        
        # Ensure data directory exists
        self._ensure_data_directory()
    
    def _log(self, message: str, level: str = 'info'):
        """Log message to UI log panel or fallback to print"""
        if self.log_queue:
            self.log_queue.put(message, level)
        else:
            print(message)
    
    def _ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        data_dirs = [
            os.path.dirname(self.model_path),
            os.path.dirname(self.features_path)
        ]
        for data_dir in data_dirs:
            if data_dir and not os.path.exists(data_dir):
                os.makedirs(data_dir, exist_ok=True)
                self._log(f"Created directory: {data_dir}")
    
    def engineer_features(
        self,
        df: pd.DataFrame,
        for_prediction: bool = False,
        live_overrides: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """Create predictive features from raw data with validation.

        Args:
            for_prediction: If True, keeps the last row (no target needed).
            live_overrides: ``{col_name: value}`` of pre-computed live
                return/change values to restore after ``pct_change()``
                overwrites them.  Built by the caller (DataManager) from
                live market quotes.
        """
        if df.empty:
            self._log("⚠ Cannot engineer features: DataFrame is empty", 'warning')
            return pd.DataFrame()
        
        # Check required columns
        required_cols = ['daily_return', 'price']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            self._log(f"⚠ Missing required columns: {missing}", 'warning')
            return pd.DataFrame()
        
        # Check minimum data points (need enough for lags and indicators)
        if len(df) < 50:
            self._log(f"⚠ Insufficient data: {len(df)} records (need at least 50)", 'warning')
            return pd.DataFrame()
        
        df = df.copy()
        _live_overrides = live_overrides or {}
        
        # Target: 1 if next day positive, else 0
        df['target'] = (df['daily_return'].shift(-1) > 0).astype(int)
        
        # Remove last row only for training (no target available)
        if not for_prediction:
            df = df.iloc[:-1]
        
        # Lagged returns (lag 0 = same-day return, critical for prediction)
        for lag in [0, 1, 2, 3, 5]:
            df[f'return_lag_{lag}'] = df['daily_return'].shift(lag)

        # For live prediction, explicitly set return_lag_0 to the live
        # ASX200 % change so it reflects the current intraday move rather
        # than a possibly-stale or ffill-overwritten value.
        if for_prediction and _live_overrides.get('daily_return') is not None:
            live_ret = _live_overrides['daily_return']
            df.at[df.index[-1], 'return_lag_0'] = live_ret
            self._log(
                f"✓ return_lag_0 set to live ASX200 change: {live_ret * 100:+.2f}%",
                'success',
            )

        # ── Consecutive streak counters ───────────────────────────────
        # Positive streak: how many consecutive positive-return days end
        # at each row.  Reversal probability shifts after 6-7 days.
        positive = (df['daily_return'] > 0).astype(int).values
        pos_streak = np.zeros(len(positive), dtype=int)
        count = 0
        for i in range(len(positive)):
            if positive[i] == 1:
                count += 1
            else:
                count = 0
            pos_streak[i] = count
        df['positive_streak'] = pos_streak

        # ── Config-driven market-source features ──────────────────────
        vol_sources = []  # track volatility sources for cross-feature
        for src in self.config.market_sources:
            name = src.name
            cat = src.category or 'commodity'
            if name not in df.columns:
                continue

            if cat == 'futures':
                df[f'{name}_return'] = df[name].pct_change()

            elif cat == 'volatility':
                df[f'{name}_change'] = df[name].pct_change()
                df[f'{name}_level'] = df[name]
                vol_sources.append(name)

            elif cat == 'bond_yield':
                df[f'{name}_change'] = df[name].diff()
                df[f'{name}_level'] = df[name]

            elif cat in ('commodity', 'currency'):
                df[f'{name}_return'] = df[name].pct_change()

        # Restore pre-injected live returns that pct_change() just overwrote
        if _live_overrides:
            live_idx = df.index[-1]
            for col, val in _live_overrides.items():
                if col in df.columns:
                    df.at[live_idx, col] = val

        # Cross-source: yield spread (first two bond yield sources)
        bond_sources = [s.name for s in self.config.market_sources
                        if s.category == 'bond_yield' and s.name in df.columns]
        if len(bond_sources) >= 2:
            df['yield_spread'] = df[bond_sources[0]] - df[bond_sources[1]]

        # Cross-source: spread between first two volatility sources
        if len(vol_sources) >= 2 and all(v in df.columns for v in vol_sources[:2]):
            df['vix_spread'] = df[vol_sources[1]] - df[vol_sources[0]]
        
        # ── Config-driven technical indicators ────────────────────
        indicators = self.config.technical_indicators
        if not indicators:
            from utils.app_config import TechnicalIndicator
            indicators = [
                TechnicalIndicator(type='macd', fast=12, slow=26, signal=9),
                TechnicalIndicator(type='rsi', period=14),
            ]
        for ind in indicators:
            if ind.type == 'macd':
                fast = ind.fast or 12
                slow = ind.slow or 26
                sig = ind.signal or 9
                df[f'ema{fast}'] = df['price'].ewm(span=fast).mean()
                df[f'ema{slow}'] = df['price'].ewm(span=slow).mean()
                df['macd'] = df[f'ema{fast}'] - df[f'ema{slow}']
                df['macd_signal'] = df['macd'].ewm(span=sig).mean()
                df['macd_histogram'] = df['macd'] - df['macd_signal']
            elif ind.type == 'rsi':
                period = ind.period or 14
                delta = df['price'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            else:
                self._log(f"⚠ Unknown indicator type: {ind.type}", 'warning')
        
        # When predicting, forward-fill computed features so the live row
        # inherits the latest market returns instead of NaN
        if for_prediction:
            last_row = df.iloc[[-1]].copy()
            df = df.ffill()
        
        # Drop NaN rows from feature engineering
        initial_rows = len(df)
        df.dropna(inplace=True)
        rows_dropped = initial_rows - len(df)
        min_nan_drop_log_rows = get_max_rolling_lag_window(self.config)
        if rows_dropped >= min_nan_drop_log_rows:
            self._log(f"⚠ Dropped {rows_dropped} rows with NaN values (warmup for rolling/lagged features)", 'warning')
        
        # Ensure the live row survives for prediction — if dropna removed it,
        # re-append the forward-filled version so predict() has something to use.
        if for_prediction and (df.empty or df.index[-1] != last_row.index[0]):
            last_filled = last_row.ffill(axis=0)
            nan_cols = last_filled.columns[last_filled.isna().any()].tolist()
            if nan_cols:
                # _level columns represent absolute prices/indices — 0 is
                # nonsensical.  Forward-fill from the last known value in
                # the training data instead.  _return/_change columns are
                # safe to zero (meaning "no movement").
                level_nans = [c for c in nan_cols if c.endswith('_level')]
                other_nans = [c for c in nan_cols if not c.endswith('_level')]
                if level_nans and not df.empty:
                    for col in level_nans:
                        if col in df.columns:
                            last_filled[col] = df[col].iloc[-1]
                    still_nan = [c for c in level_nans
                                 if last_filled[c].isna().any()]
                    if still_nan:
                        self._log(f"⚠ Level columns still NaN (zeroed): {still_nan}", 'warning')
                        last_filled[still_nan] = 0
                if other_nans:
                    self._log(f"⚠ Live row NaN → 0 for return/change cols: {other_nans}", 'warning')
                    last_filled[other_nans] = 0
            # Restore return_lag_0 from live override so re-appended row
            # reflects the actual intraday ASX200 change, not a filled value.
            if _live_overrides.get('daily_return') is not None:
                last_filled['return_lag_0'] = _live_overrides['daily_return']
            df = pd.concat([df, last_filled])
        
        if df.empty:
            self._log("⚠ No valid data after feature engineering", 'warning')
            return pd.DataFrame()
        
        self._log(f"✓ Feature engineering complete: {len(df)} rows, {len(df.columns)} columns", 'success')
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns (exclude target, identifiers, and raw market source columns)"""
        raw_source_names = {s.name for s in self.config.market_sources}
        exclude_cols = {'target', 'daily_return', 'price'} | raw_source_names
        return [col for col in df.columns if col not in exclude_cols]
    
    def save_feature_names(self, feature_names: List[str]) -> None:
        """Save feature names separately for better model viewing"""
        base_dir = os.path.dirname(self.features_path)
        feature_names_path = os.path.join(base_dir, "feature_names.txt")
        try:
            with open(feature_names_path, 'w') as f:
                for i, name in enumerate(feature_names, 1):
                    f.write(f"{i:3d}. {name}\n")
            self._log(f"✓ Feature names saved to {feature_names_path}", 'success')
        except Exception as e:
            self._log(f"⚠ Failed to save feature names: {e}", 'warning')
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train XGBoost model on the provided DataFrame"""
        result = {
            "success": True,
            "train_accuracy": 0,
            "test_accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
            "feature_importance": None,
            "message": ""
        }
        
        try:
            self.feature_columns = self.get_feature_columns(df)
            X = df[self.feature_columns]
            y = df['target']
            
            # Save feature names for viewer
            self.save_feature_names(self.feature_columns)
            
            # Check if we have enough data
            if len(X) < 100:
                result["success"] = False
                result["message"] = f"Insufficient data: {len(X)} rows (need at least 100)"
                return result
            
            # Train/test split (80/20 time-series)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Initialize XGBoost
            self.model = XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                min_child_weight=self.min_child_weight,
                random_state=self.random_state,
                eval_metric='logloss',
                verbosity=0,
            )
            self._log("── Model Parameters ──", 'info')
            self._log(f"  n_estimators:         {self.n_estimators}", 'info')
            self._log(f"  max_depth:            {self.max_depth}", 'info')
            self._log(f"  learning_rate:        {self.learning_rate}", 'info')
            self._log(f"  subsample:            {self.subsample}", 'info')
            self._log(f"  colsample_bytree:     {self.colsample_bytree}", 'info')
            self._log(f"  min_child_weight:     {self.min_child_weight}", 'info')
            self._log(f"  random_state:         {self.random_state}", 'info')
            self._log(
                f"Fitting XGBoost ("
                f"{len(X_train)} rows, {len(self.feature_columns)} features)...",
                'progress',
            )
            t0 = time.time()
            self.model.fit(X_train, y_train)
            fit_secs = time.time() - t0
            self._log(f"✓ Model fitted in {fit_secs:.1f}s ({self.n_estimators} trees)", 'success')
            
            # Predictions
            y_train_pred = self.model.predict(X_train)
            y_test_pred = self.model.predict(X_test)
            
            # Calculate metrics
            result["train_accuracy"] = accuracy_score(y_train, y_train_pred)
            result["test_accuracy"] = accuracy_score(y_test, y_test_pred)
            
            # Only calculate additional metrics if we have both classes in test set
            if len(np.unique(y_test)) > 1:
                result["precision"] = precision_score(y_test, y_test_pred, average='binary', zero_division=0)
                result["recall"] = recall_score(y_test, y_test_pred, average='binary', zero_division=0)
                result["f1_score"] = f1_score(y_test, y_test_pred, average='binary', zero_division=0)
            
            # Feature importance
            importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            result["feature_importance"] = importance.head(10).to_dict('records')

            # Log all features ranked by importance
            self._log("── Feature Importance ──────────────────────", 'info')
            for rank, row in enumerate(importance.itertuples(), 1):
                bar = '█' * int(row.importance * 100)
                self._log(f"  {rank:2d}. {row.feature:<28s} {row.importance:.4f}  {bar}", 'info')
            self._log("────────────────────────────────────────────", 'info')
            
            # Ensure directory exists before saving
            self._ensure_data_directory()
            
            # Save model and features
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.feature_columns, self.features_path)
            result["message"] = f"Model saved to {self.model_path}"
            
            self._log(f"✓ Model training complete. Test accuracy: {result['test_accuracy']:.3f}", 'success')
            
            # Compute calibration on the test set
            try:
                y_test_prob = self.model.predict_proba(X_test)[:, 1]
                calib = self._calibration_from_arrays(y_test_prob, y_test.values)
                result['calibration'] = calib
                self._log(f"✓ Calibration computed (ECE={calib['expected_calibration_error']:.4f})", 'success')
            except Exception as cal_err:
                self._log(f"⚠ Calibration computation failed: {cal_err}", 'warning')
            
            # Save training snapshot for future comparison
            self._save_training_snapshot(result)
            
        except Exception as e:
            result["success"] = False
            result["message"] = str(e)
            self._log(f"✗ Model training failed: {e}", 'error')
        
        return result
    
    def train_with_cv(self, df: pd.DataFrame, cv_folds: int = 5) -> Dict[str, Any]:
        """Train with time-series cross-validation for more robust evaluation.

        Uses ``TimeSeriesSplit`` instead of KFold to preserve temporal
        ordering and avoid data leakage (future data in training folds).
        """
        result = self.train(df)
        
        if result["success"] and self.model is not None:
            try:
                X = df[self.feature_columns]
                y = df['target']
                
                # Time-series aware CV — respects temporal ordering
                tscv = TimeSeriesSplit(n_splits=cv_folds)
                cv_scores = cross_val_score(self.model, X, y, cv=tscv)
                result["cv_mean"] = cv_scores.mean()
                result["cv_std"] = cv_scores.std()
                result["cv_scores"] = cv_scores.tolist()
                
                self._log(f"✓ Time-series CV complete. Mean: {result['cv_mean']:.3f} (±{result['cv_std']:.3f})", 'success')
                
            except Exception as e:
                result["cv_error"] = str(e)
                self._log(f"⚠ Cross-validation failed: {e}", 'warning')
        
        return result

    # ── Multi-seed ablation experiment ────────────────────────────────

    def run_multi_seed_ablation(
        self,
        df: pd.DataFrame,
        seeds: range = range(10),
        ablate_cols: Optional[List[str]] = None,
        progress_cb=None,
        verdict_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Train with and without *ablate_cols* across multiple random seeds.

        Compares accuracy, ECE and MCE distributions to determine whether
        a feature set's calibration benefit is robust or an artifact of a
        single train/test split.

        Args:
            df: Feature-engineered DataFrame (with ``target`` column).
            seeds: Range of ``random_state`` values to evaluate.
            ablate_cols: Columns to drop in the "without" variant.
                         **Required** — caller must supply a non-empty list.
            progress_cb: Optional ``fn(current, total)`` called after each
                         seed pair completes (for progress bars).
            verdict_threshold: Effect-size multiplier against the pooled
                         standard deviation to declare a winner (default
                         0.5 — i.e. delta must exceed 0.5× the larger
                         std to be considered meaningful).

        Returns:
            Dict with ``full_features`` and ``model_features`` summary
            stats, per-seed detail, verdicts, and metadata.
        """
        ablate_cols = ablate_cols or []
        if not ablate_cols:
            self._log("⚠ No columns selected for ablation", 'warning')
            return {}
        missing = [c for c in ablate_cols if c not in df.columns]
        if missing:
            self._log(f"⚠ Ablation columns not in data: {missing}", 'warning')
            return {}

        feature_cols_full = self.get_feature_columns(df)
        feature_cols_model = [c for c in feature_cols_full if c not in ablate_cols]

        split_idx = int(len(df) * 0.8)
        X_full = df[feature_cols_full]
        X_model = df[feature_cols_model]
        y = df['target']

        X_train_f, X_test_f = X_full.iloc[:split_idx], X_full.iloc[split_idx:]
        X_train_m, X_test_m = X_model.iloc[:split_idx], X_model.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        rows = []
        seed_list = list(seeds)
        total_seeds = len(seed_list)
        self._log(
            f"── Multi-seed ablation: {total_seeds} seeds, "
            f"ablating {ablate_cols} ──",
            'info',
        )

        for i, seed in enumerate(seed_list):
            for label, X_tr, X_te in [
                ('full', X_train_f, X_test_f),
                ('model', X_train_m, X_test_m),
            ]:
                clf = XGBClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    subsample=self.subsample,
                    colsample_bytree=self.colsample_bytree,
                    min_child_weight=self.min_child_weight,
                    random_state=seed,
                    eval_metric='logloss',
                    verbosity=0,
                )
                clf.fit(X_tr, y_train)
                y_prob = clf.predict_proba(X_te)[:, 1]
                y_pred = clf.predict(X_te)
                acc = float(accuracy_score(y_test, y_pred))
                bin_edges = np.linspace(0, 1, 11)
                ece = self._compute_ece(y_prob, y_test.values, bin_edges)
                mce = self._compute_mce(y_prob, y_test.values, bin_edges)
                rows.append({
                    'seed': seed, 'variant': label,
                    'accuracy': acc, 'ece': ece, 'mce': mce,
                })
            if progress_cb:
                progress_cb(i + 1, total_seeds)

        results = pd.DataFrame(rows)
        full_df = results[results['variant'] == 'full']
        model_df = results[results['variant'] == 'model']

        def _stats(vals):
            return {
                'mean': float(vals.mean()),
                'std': float(vals.std()),
            }

        full_stats = {
            f'{m}_mean': _stats(full_df[m])['mean'] for m in ('accuracy', 'ece', 'mce')
        }
        full_stats.update({
            f'{m}_std': _stats(full_df[m])['std'] for m in ('accuracy', 'ece', 'mce')
        })
        model_stats = {
            f'{m}_mean': _stats(model_df[m])['mean'] for m in ('accuracy', 'ece', 'mce')
        }
        model_stats.update({
            f'{m}_std': _stats(model_df[m])['std'] for m in ('accuracy', 'ece', 'mce')
        })

        # Per-metric verdicts (KEEP / REMOVE / INCONCLUSIVE)
        # For accuracy: higher is better → full > model means KEEP
        # For ECE/MCE: lower is better → full < model means KEEP
        verdicts = {}
        for metric in ('accuracy', 'ece', 'mce'):
            f_mean = full_stats[f'{metric}_mean']
            m_mean = model_stats[f'{metric}_mean']
            pooled_std = max(full_stats[f'{metric}_std'],
                            model_stats[f'{metric}_std'], 1e-9)
            threshold = verdict_threshold * pooled_std
            delta = f_mean - m_mean
            # "full wins" depends on direction
            if metric == 'accuracy':
                full_wins = delta > threshold         # higher is better
            else:
                full_wins = delta < -threshold        # lower is better
            model_wins = not full_wins and abs(delta) > threshold
            if full_wins:
                verdicts[metric] = 'KEEP'
            elif model_wins:
                verdicts[metric] = 'REMOVE'
            else:
                verdicts[metric] = 'INCONCLUSIVE'

        keep_count = sum(1 for v in verdicts.values() if v == 'KEEP')
        remove_count = sum(1 for v in verdicts.values() if v == 'REMOVE')
        if keep_count >= 2:
            overall = 'KEEP'
        elif remove_count >= 2:
            overall = 'REMOVE'
        else:
            overall = 'INCONCLUSIVE'

        # Log results
        self._log("── Ablation Results ────────────────────────", 'info')
        self._log(f"  Seeds: {seed_list}", 'info')
        self._log(f"  Ablated: {ablate_cols}", 'info')
        self._log("", 'info')
        hdr = f"  {'Metric':<10s} {'Full':>14s} {'Model':>14s} {'Delta':>10s}  {'Verdict':>13s}"
        self._log(hdr, 'info')
        self._log("  " + "─" * 65, 'info')
        for metric in ('accuracy', 'ece', 'mce'):
            fm, fs = full_stats[f'{metric}_mean'], full_stats[f'{metric}_std']
            mm_, ms = model_stats[f'{metric}_mean'], model_stats[f'{metric}_std']
            delta = fm - mm_
            self._log(
                f"  {metric:<10s} "
                f"{fm:.4f}±{fs:.4f} "
                f"{mm_:.4f}±{ms:.4f} "
                f"{delta:+.4f}  "
                f"[{verdicts[metric]}]",
                'info',
            )
        self._log("────────────────────────────────────────────", 'info')

        # Per-seed detail
        self._log("", 'info')
        self._log(f"  {'Seed':>4s}  {'Acc(f)':>7s} {'Acc(m)':>7s}  "
                  f"{'ECE(f)':>7s} {'ECE(m)':>7s}  "
                  f"{'MCE(f)':>7s} {'MCE(m)':>7s}", 'info')
        for seed in seed_list:
            f_row = full_df[full_df['seed'] == seed].iloc[0]
            m_row = model_df[model_df['seed'] == seed].iloc[0]
            self._log(
                f"  {seed:4d}  {f_row['accuracy']:.4f}  {m_row['accuracy']:.4f}  "
                f"{f_row['ece']:.4f}  {m_row['ece']:.4f}  "
                f"{f_row['mce']:.4f}  {m_row['mce']:.4f}",
                'info',
            )

        self._log(f"\n  Overall: [{overall}]", 'info')

        output = {
            'full_features': full_stats,
            'model_features': model_stats,
            'verdicts': verdicts,
            'overall': overall,
            'ablated_cols': ablate_cols,
            'seeds': seed_list,
            'per_seed': rows,
            'ran_at': datetime.now().isoformat(),
        }

        # Persist result
        self._save_ablation_result(output)

        return output

    def _save_ablation_result(self, result: Dict[str, Any]) -> None:
        """Persist ablation result to ``ablation_result.json``."""
        path = os.path.join(os.path.dirname(self.model_path), 'ablation_result.json')
        try:
            with open(path, 'w') as f:
                _json.dump(result, f, indent=2, default=str)
        except Exception as e:
            self._log(f"⚠ Could not save ablation result: {e}", 'warning')

    def load_ablation_result(self) -> Optional[Dict[str, Any]]:
        """Load the last ablation result, or None if unavailable."""
        path = os.path.join(os.path.dirname(self.model_path), 'ablation_result.json')
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                return _json.load(f)
        except Exception:
            return None

    # ── Training Snapshot ─────────────────────────────────────────────

    def _save_training_snapshot(self, result: Dict[str, Any]) -> None:
        """Persist key training metrics so the next run can show a comparison."""
        snapshot = {
            'trained_at': datetime.now().isoformat(),
            'train_accuracy': result.get('train_accuracy'),
            'test_accuracy': result.get('test_accuracy'),
            'precision': result.get('precision'),
            'recall': result.get('recall'),
            'f1_score': result.get('f1_score'),
            'feature_importance': result.get('feature_importance'),  # list of dicts
        }
        cal = result.get('calibration')
        if cal:
            snapshot['calibration'] = {
                'ece': cal['expected_calibration_error'],
                'mce': cal['max_calibration_error'],
                'table': cal['calibration_table'],
            }
        try:
            with open(self.training_snapshot_path, 'w') as f:
                _json.dump(snapshot, f, indent=2, default=str)
        except Exception as e:
            self._log(f"⚠ Could not save training snapshot: {e}", 'warning')

    def load_training_snapshot(self) -> Optional[Dict[str, Any]]:
        """Load the previous training snapshot, or None if unavailable."""
        if not os.path.exists(self.training_snapshot_path):
            return None
        try:
            with open(self.training_snapshot_path) as f:
                return _json.load(f)
        except Exception:
            return None
    
    def load_model(self) -> bool:
        """Load pre-trained model if exists"""
        if os.path.exists(self.model_path) and os.path.exists(self.features_path):
            try:
                self.model = joblib.load(self.model_path)
                self.feature_columns = joblib.load(self.features_path)
                self._log(f"✓ Model loaded from {self.model_path}", 'success')
                return True
            except Exception as e:
                self._log(f"✗ Failed to load model: {e}", 'error')
                return False
        return False
    
    def save_model_with_version(self, version: str = None) -> str:
        """Save model with version tag"""
        if self.model is None:
            self._log("⚠ No model to save", 'warning')
            return ""
        
        if version is None:
            version = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save with version in filename
        versioned_model_path = self.model_path.replace('.pkl', f'_{version}.pkl')
        versioned_features_path = self.features_path.replace('.pkl', f'_{version}.pkl')
        
        try:
            joblib.dump(self.model, versioned_model_path)
            joblib.dump(self.feature_columns, versioned_features_path)
            self._log(f"✓ Model saved with version: {version}", 'success')
            return version
        except Exception as e:
            self._log(f"✗ Failed to save versioned model: {e}", 'error')
            return ""
    
    def predict(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Predict probability for the latest row.
        Returns dict with probability, feature values, and importances.
        """
        if self.model is None or self.feature_columns is None:
            if not self.load_model():
                self._log("⚠ No trained model available. Please train first.", 'warning')
                return None
        
        try:
            # Check if all required features are present
            missing_features = [col for col in self.feature_columns if col not in df.columns]
            if missing_features:
                self._log(f"⚠ Missing features for prediction: {missing_features}", 'warning')
                return None
            
            latest = df.iloc[[-1]][self.feature_columns]
            
            # Check for NaN values
            if latest.isna().any().any():
                nan_cols = latest.columns[latest.isna().any()].tolist()
                self._log(f"⚠ Latest data contains NaN values in: {nan_cols}", 'warning')
                return None
            
            prob = self.model.predict_proba(latest)[0][1]
            
            # Build feature importance / value breakdown
            importances = pd.Series(
                self.model.feature_importances_, index=self.feature_columns
            ).sort_values(ascending=False)
            
            feature_values = latest.iloc[0]
            
            feature_details = []
            for feat in importances.index:
                feature_details.append({
                    'name': feat,
                    'value': feature_values[feat],
                    'importance': importances[feat],
                })
            
            return {
                'probability': prob,
                'feature_details': feature_details,
            }
            
        except Exception as e:
            self._log(f"✗ Prediction failed: {e}", 'error')
            return None
    
    # ── Calibration & Decision ─────────────────────────────────────

    def evaluate_calibration(self, df: pd.DataFrame, bins: int = 10) -> Dict:
        """Evaluate calibration of probability predictions on a labelled DataFrame.

        Args:
            df: DataFrame with features (must include 'target' column)
            bins: Number of equal-width probability bins (default 10)

        Returns:
            Dictionary with calibration table and overall metrics
        """
        if self.model is None and not self.load_model():
            return {}

        X = df[self.feature_columns]
        y_true = df['target'].values
        y_prob = self.model.predict_proba(X)[:, 1]
        return self._calibration_from_arrays(y_prob, y_true, bins)

    def _calibration_from_arrays(
        self, y_prob: np.ndarray, y_true: np.ndarray, bins: int = 10
    ) -> Dict:
        """Build a calibration table from probability and truth arrays."""
        bin_edges = np.linspace(0, 1, bins + 1)
        bin_indices = np.digitize(y_prob, bin_edges) - 1

        calibration_table = []
        for i in range(bins):
            mask = bin_indices == i
            count = int(np.sum(mask))
            if count == 0:
                continue
            pred_mean = float(np.mean(y_prob[mask]))
            actual_freq = float(np.mean(y_true[mask]))
            calibration_table.append({
                'bin': f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}",
                'predicted_prob': round(pred_mean, 4),
                'actual_freq': round(actual_freq, 4),
                'count': count,
            })

        return {
            'calibration_table': calibration_table,
            'expected_calibration_error': self._compute_ece(y_prob, y_true, bin_edges),
            'max_calibration_error': self._compute_mce(y_prob, y_true, bin_edges),
        }

    @staticmethod
    def _compute_ece(y_prob, y_true, bin_edges) -> float:
        """Expected Calibration Error."""
        bin_indices = np.digitize(y_prob, bin_edges) - 1
        n = len(y_prob)
        ece = 0.0
        for i in range(len(bin_edges) - 1):
            mask = bin_indices == i
            count = np.sum(mask)
            if count == 0:
                continue
            ece += (count / n) * abs(float(np.mean(y_prob[mask])) - float(np.mean(y_true[mask])))
        return float(ece)

    @staticmethod
    def _compute_mce(y_prob, y_true, bin_edges) -> float:
        """Maximum Calibration Error."""
        bin_indices = np.digitize(y_prob, bin_edges) - 1
        mce = 0.0
        for i in range(len(bin_edges) - 1):
            mask = bin_indices == i
            if np.sum(mask) == 0:
                continue
            mce = max(mce, abs(float(np.mean(y_prob[mask])) - float(np.mean(y_true[mask]))))
        return float(mce)

    def get_decision(
        self, df: pd.DataFrame, threshold: float = 0.6
    ) -> Dict[str, Any]:
        """Return a decision recommendation based on prediction confidence.

        Args:
            df: DataFrame with latest engineered features
            threshold: Minimum probability to act (default 0.6)

        Returns:
            Dictionary with decision, probability, confidence level,
            and full feature details from predict().
        """
        result = self.predict(df)
        if result is None:
            return {'decision': 'NO_PREDICTION', 'probability': None}

        prob = result['probability']

        # Confidence level
        if prob >= 0.9 or prob <= 0.1:
            level = 'VERY_HIGH'
        elif prob >= 0.8 or prob <= 0.2:
            level = 'HIGH'
        elif prob >= 0.7 or prob <= 0.3:
            level = 'MODERATE'
        elif prob >= 0.6 or prob <= 0.4:
            level = 'LOW'
        else:
            level = 'VERY_LOW'

        # Direction decision
        if prob >= threshold:
            decision = 'POSITIVE_EXPECTED'
        elif prob <= 1 - threshold:
            decision = 'NEGATIVE_EXPECTED'
        else:
            decision = 'NEUTRAL'

        return {
            'decision': decision,
            'probability': prob,
            'confidence_level': level,
            'threshold_used': threshold,
            'feature_details': result['feature_details'],
        }

    def get_feature_importance_data(self, top_n: int = 20) -> Dict:
        """Get feature importance data formatted for visualization"""
        if self.model is None:
            if not self.load_model():
                return {}
        
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        return {
            'features': importance['feature'].tolist(),
            'importance': importance['importance'].tolist()
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        info = {
            "model_loaded": self.model is not None,
            "feature_count": len(self.feature_columns) if self.feature_columns else 0,
            "model_path": self.model_path,
            "features_path": self.features_path,
            "config": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "random_state": self.random_state
            }
        }
        
        # Add model parameters if model is loaded
        if self.model is not None:
            info["model_params"] = {
                "n_classes": len(self.model.classes_),
                "n_features": self.model.n_features_in_,
                "tree_count": len(self.model.estimators_)
            }
            
            # Add feature importance summary
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                info["feature_importance"] = {
                    "mean": float(np.mean(importances)),
                    "std": float(np.std(importances)),
                    "max": float(np.max(importances)),
                    "min": float(np.min(importances))
                }
        
        # Add file info if model exists
        if os.path.exists(self.model_path):
            info["model_file_size"] = os.path.getsize(self.model_path)
            info["model_modified"] = datetime.fromtimestamp(
                os.path.getmtime(self.model_path)
            ).strftime('%Y-%m-%d %H:%M:%S')
        
        return info
