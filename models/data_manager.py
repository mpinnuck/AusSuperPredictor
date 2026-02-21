"""
Data Manager - single responsibility for data operations
Now saves data to the data folder
"""
import os
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta, date
from typing import Optional, Dict, Any
from utils.time_utils import SydneyTimeUtils

class DataManager:
    """Manages all data operations - AustralianSuper and market data"""
    
    def __init__(self, config: Dict[str, Any], log_queue=None):
        self.config = config
        self.local_csv_path = config['data']['local_csv_path']
        self.start_date = datetime.strptime(config['data']['start_date'], '%d/%m/%Y')
        self.time_utils = SydneyTimeUtils()
        self.log_queue = log_queue
        
        # Ensure data directory exists
        self._ensure_data_directory()
    
    def _ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        data_dir = os.path.dirname(self.local_csv_path)
        if data_dir and not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
            self._log(f"Created data directory: {data_dir}", 'info')
    
    def get_last_stored_date(self) -> Optional[date]:
        """Return the latest date in local CSV, or None if file missing"""
        if os.path.exists(self.local_csv_path):
            # First read to check column structure
            df_check = pd.read_csv(self.local_csv_path, nrows=1)
            
            # Check if 'date' column exists, otherwise use first column as date index
            if 'date' in df_check.columns:
                df = pd.read_csv(self.local_csv_path, parse_dates=['date'], index_col='date')
            else:
                # Use first column (index 0) as date index
                df = pd.read_csv(self.local_csv_path, parse_dates=[0], index_col=0)
                
            if not df.empty:
                return df.index.max().date()
        return None
    
    def fetch_australiansuper_data(self, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Fetch AustralianSuper CSV from the API for a given date range
        Returns DataFrame with columns 'daily_return' and 'price'
        API returns: Rate Date, High Growth, Balanced, Socially Aware, etc.
        Values are percentages (e.g. 0.4556 means 0.4556%)
        """
        # TEMPORARY: Skip AustralianSuper API, go straight to ASX200 fallback
        self._log("Skipping AustralianSuper API (temporary), using ASX200 fallback", 'info')
        return self._get_asx200_proxy(start_date, end_date)
    
    def _get_asx200_proxy(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Fallback using ASX200 daily data from Investing.com (accurate close prices)."""
        self._log("Fetching daily ASX200 data from Investing.com...", 'info')
        self._log(f"Date range: {start_date} to {end_date}", 'info')
        
        try:
            import requests as req
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'domain-id': 'au',
            }
            
            start_str = start_date.strftime('%Y-%m-%d') if isinstance(start_date, date) else str(start_date)
            end_str = end_date.strftime('%Y-%m-%d') if isinstance(end_date, date) else str(end_date)
            
            url = (
                f"https://api.investing.com/api/financialdata/historical/171"
                f"?start-date={start_str}&end-date={end_str}"
                f"&time-frame=Daily&add-missing-rows=false"
            )
            
            response = req.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            records = response.json().get('data', [])
            
            if not records:
                self._log("⚠ No ASX200 data returned from Investing.com", 'warning')
                return pd.DataFrame()
            
            # Parse records into Series — records are newest-first
            dates = []
            closes = []
            for rec in records:
                dates.append(pd.Timestamp(rec['rowDateTimestamp']))
                closes.append(float(rec['last_closeRaw']))
            
            daily_closes = pd.Series(closes, index=pd.DatetimeIndex(dates).normalize().tz_localize(None))
            daily_closes = daily_closes[~daily_closes.index.duplicated(keep='first')]
            daily_closes.sort_index(inplace=True)
            
            self._log(f"ASX200 data: {len(daily_closes)} trading days", 'info')
            returns = daily_closes.pct_change().dropna()
            
            if returns.empty:
                self._log("⚠ No valid returns calculated", 'warning')
                return pd.DataFrame()
            
            fund_data = pd.DataFrame({
                'daily_return': returns.values,
                'price': daily_closes.loc[returns.index].values
            }, index=returns.index)
            
            fund_data.index.name = 'date'
            self._log(f"✓ Successfully fetched {len(fund_data)} records from Investing.com ASX200", 'success')
            return fund_data
            
        except Exception as e:
            self._log(f"✗ Error fetching Investing.com ASX200 data: {e}", 'error')
            self._log("Falling back to yfinance daily data...", 'warning')
            return self._get_asx200_yfinance_fallback(start_date, end_date)
    
    def _get_asx200_yfinance_fallback(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Last resort fallback using yfinance daily data."""
        try:
            asx_daily = yf.download('^AXJO', start=start_date, end=end_date, progress=False)
            
            if asx_daily.empty:
                self._log("⚠ No ASX200 data available", 'warning')
                return pd.DataFrame()
            
            if isinstance(asx_daily.columns, pd.MultiIndex):
                asx_daily.columns = asx_daily.columns.get_level_values(0)
            
            daily_closes = asx_daily['Close'].copy()
            daily_closes = pd.Series(daily_closes.values.flatten().astype(float),
                                     index=pd.to_datetime(asx_daily.index).normalize())
            daily_closes = daily_closes[~daily_closes.index.duplicated(keep='last')]
            daily_closes.sort_index(inplace=True)
            
            returns = daily_closes.pct_change().dropna()
            if returns.empty:
                return pd.DataFrame()
            
            fund_data = pd.DataFrame({
                'daily_return': returns.values.flatten(),
                'price': daily_closes.loc[returns.index].values.flatten()
            }, index=returns.index)
            
            fund_data.index.name = 'date'
            self._log(f"✓ Fetched {len(fund_data)} records from yfinance ASX200 (fallback)", 'success')
            return fund_data
        except Exception as e:
            self._log(f"✗ Error fetching yfinance ASX200 data: {e}", 'error')
            return pd.DataFrame()
    
    def update_local_data(self) -> Dict[str, Any]:
        """
        Fetch new data since last stored date and append to local CSV
        Returns status dictionary with counts and messages
        """
        result = {"success": True, "records_added": 0, "message": ""}
        last_date = self.get_last_stored_date()
        end_date = datetime.now() - timedelta(days=self.config['data']['end_date_offset_days'])
        end_date = end_date.date()
        
        try:
            if last_date is None:
                start = self.start_date.date()
                self._log(f"Initial data fetch from {start} to {end_date}", 'info')
                new_data = self.fetch_australiansuper_data(start, end_date)
                if not new_data.empty:
                    # Ensure directory exists before saving
                    self._ensure_data_directory()
                    # Ensure index is properly named before saving
                    new_data.index.name = 'date'
                    new_data.to_csv(self.local_csv_path)
                    result["records_added"] = len(new_data)
                    result["message"] = f"Saved {len(new_data)} records to {self.local_csv_path.split('/')[-1]}"
                else:
                    result["message"] = "No data retrieved"
            else:
                start = last_date + timedelta(days=1)
                if start <= end_date:
                    # Fetch from last_date (not start) so pct_change has a reference price
                    # for calculating returns on the first new day
                    new_data = self.fetch_australiansuper_data(last_date, end_date)
                    if not new_data.empty:
                        # Trim to only new data (after last_date)
                        new_data = new_data[new_data.index > pd.Timestamp(last_date)]
                    if not new_data.empty:
                        # Ensure directory exists before saving
                        self._ensure_data_directory()
                        
                        # Read existing data with flexible column handling
                        df_check = pd.read_csv(self.local_csv_path, nrows=1)
                        if 'date' in df_check.columns:
                            existing = pd.read_csv(self.local_csv_path, parse_dates=['date'], index_col='date')
                        else:
                            existing = pd.read_csv(self.local_csv_path, parse_dates=[0], index_col=0)
                            
                        combined = pd.concat([existing, new_data])
                        combined = combined[~combined.index.duplicated(keep='last')]
                        combined.sort_index(inplace=True)
                        # Ensure index is properly named before saving
                        combined.index.name = 'date'
                        combined.to_csv(self.local_csv_path)
                        result["records_added"] = len(new_data)
                        result["message"] = f"Appended {len(new_data)} new records"
                    else:
                        result["message"] = "No new data available"
                else:
                    result["message"] = "Local data is already up to date"
        except Exception as e:
            result["success"] = False
            result["message"] = str(e)
        
        return result
    
    def load_local_data(self) -> pd.DataFrame:
        """Load the full local dataset"""
        if os.path.exists(self.local_csv_path):
            # First read to check column structure
            df_check = pd.read_csv(self.local_csv_path, nrows=1)
            
            # Check if 'date' column exists, otherwise use first column as date index
            if 'date' in df_check.columns:
                return pd.read_csv(self.local_csv_path, parse_dates=['date'], index_col='date')
            else:
                # Use first column (index 0) as date index
                return pd.read_csv(self.local_csv_path, parse_dates=[0], index_col=0)
        return pd.DataFrame()
    
    def get_market_data(self, end_date: date) -> pd.DataFrame:
        """Fetch market data (futures, commodities, etc.) up to end_date"""
        if isinstance(end_date, date):
            end_date = datetime.combine(end_date, datetime.min.time())
        
        market_data = {}
        
        # ASX200 futures
        try:
            asx_fut = yf.download('AP', start=self.start_date, end=end_date, progress=False)
            if not asx_fut.empty and 'Close' in asx_fut.columns:
                market_data['asx_futures'] = asx_fut['Close']
        except:
            pass
        
        # SP500 futures
        try:
            sp_fut = yf.download('ES=F', start=self.start_date, end=end_date, progress=False)
            if not sp_fut.empty and 'Close' in sp_fut.columns:
                market_data['sp500_futures'] = sp_fut['Close']
        except:
            pass
        
        # VIX
        try:
            vix = yf.download('^VIX', start=self.start_date, end=end_date, progress=False)
            if not vix.empty and 'Close' in vix.columns:
                market_data['vix'] = vix['Close']
        except:
            pass
        
        # Commodities
        commodities = {}
        try:
            gold = yf.download('GC=F', start=self.start_date, end=end_date, progress=False)
            if not gold.empty and 'Close' in gold.columns:
                commodities['gold'] = gold['Close']
        except:
            pass
        try:
            copper = yf.download('HG=F', start=self.start_date, end=end_date, progress=False)
            if not copper.empty and 'Close' in copper.columns:
                commodities['copper'] = copper['Close']
        except:
            pass
        try:
            oil = yf.download('CL=F', start=self.start_date, end=end_date, progress=False)
            if not oil.empty and 'Close' in oil.columns:
                commodities['oil'] = oil['Close']
        except:
            pass
        try:
            bhp = yf.download('BHP.AX', start=self.start_date, end=end_date, progress=False)
            if not bhp.empty and 'Close' in bhp.columns:
                commodities['iron_ore_proxy'] = bhp['Close']
        except:
            pass
        
        for name, series in commodities.items():
            if hasattr(series, 'values') and len(series) > 0:
                market_data[name] = series
            else:
                self._log(f"⚠ Skipping {name} - no valid data", 'warning')
        
        # AUD/USD
        try:
            audusd = yf.download('AUDUSD=X', start=self.start_date, end=end_date, progress=False)
            if not audusd.empty and 'Close' in audusd.columns:
                market_data['audusd'] = audusd['Close']
        except Exception as e:
            self._log(f"⚠ Failed to fetch AUDUSD: {e}", 'warning')
        
        # Combine all market data
        if market_data:
            try:
                # Filter out empty or invalid series before creating DataFrame
                valid_data = {}
                for name, series in market_data.items():
                    try:
                        # Basic validation - check if it looks like a pandas Series
                        if not hasattr(series, 'values'):
                            self._log(f"⚠ Skipping {name} - not a series object", 'warning')
                            continue
                            
                        if not hasattr(series, 'index'):
                            self._log(f"⚠ Skipping {name} - no index", 'warning')
                            continue
                            
                        # Check length
                        series_len = len(series)
                        if series_len == 0:
                            self._log(f"⚠ Skipping {name} - empty series", 'warning')
                            continue
                        
                        # Check for all NaN values (if applicable)
                        has_valid_data = True
                        if hasattr(series, 'isna'):
                            try:
                                all_nan = series.isna().all()
                                if all_nan:
                                    has_valid_data = False
                                    self._log(f"⚠ Skipping {name} - all NaN values", 'warning')
                                    continue
                            except:
                                # If isna() check fails, assume it has valid data
                                pass
                        
                        if has_valid_data:
                            valid_data[name] = series
                            self._log(f"✓ Added {name}: {series_len} records", 'info')
                            
                    except Exception as e:
                        self._log(f"⚠ Error validating {name}: {e}", 'warning')
                
                if valid_data:
                    try:
                        # Create DataFrame by explicitly converting each series to proper pandas Series
                        self._log("Converting market data to proper pandas format...", 'info')
                        clean_data = {}
                        common_index = None
                        
                        for name, series in valid_data.items():
                            try:
                                # Convert to proper pandas Series with datetime index
                                if hasattr(series, 'values') and hasattr(series, 'index'):
                                    # Extract values and flatten them to 1D
                                    values = series.values.flatten()  # This fixes the 2D array issue
                                    index = pd.to_datetime(series.index)
                                    clean_series = pd.Series(values, index=index, name=name)
                                    clean_data[name] = clean_series
                                    
                                    # Track a common index for alignment
                                    if common_index is None:
                                        common_index = index
                                    
                                    self._log(f"✓ Converted {name}: {len(clean_series)} records", 'info')
                                else:
                                    self._log(f"⚠ Skipping {name}: invalid format", 'warning')
                            except Exception as e:
                                self._log(f"⚠ Failed to convert {name}: {e}", 'warning')
                        
                        if clean_data and common_index is not None:
                            # Create DataFrame from clean Series objects
                            df = pd.DataFrame(clean_data)
                            self._log(f"✓ Market data loaded: {len(df)} records, {len(clean_data)} indicators", 'success')
                            return df
                        else:
                            self._log("⚠ No valid clean market data", 'warning')
                            
                    except Exception as df_error:
                        self._log(f"✗ DataFrame creation failed: {df_error}", 'error')
                else:
                    self._log("⚠ No valid market data available", 'warning')
                    
            except Exception as e:
                self._log(f"✗ Error creating market data DataFrame: {e}", 'error')
                
        return pd.DataFrame()
    
    def prepare_combined_data(self) -> pd.DataFrame:
        """
        Load local AustralianSuper data and merge with market data
        Returns a single DataFrame with all features
        """
        fund_data = self.load_local_data()
        if fund_data.empty:
            return pd.DataFrame()
        
        end_date = fund_data.index.max().date()
        # yfinance 'end' is exclusive, so add 1 day to include the last date
        market_end = end_date + timedelta(days=1)
        market_data = self.get_market_data(market_end)
        
        if market_data.empty:
            return fund_data
        
        combined = fund_data.join(market_data, how='left')
        combined = combined.ffill()
        combined.dropna(inplace=True)
        return combined
    
    def fetch_live_asx200(self) -> Optional[pd.DataFrame]:
        """Fetch today's live ASX200 data from Investing.com.
        Returns a single-row DataFrame with daily_return and price,
        or None if today's data is not available or already in CSV.
        """
        try:
            import requests as req
            
            fund_data = self.load_local_data()
            if fund_data.empty:
                return None
            
            last_csv_date = fund_data.index.max().date()
            today = date.today()
            
            # Only fetch if today is newer than what's in the CSV
            if last_csv_date >= today:
                return None
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'domain-id': 'au',
            }
            
            url = (
                f"https://api.investing.com/api/financialdata/historical/171"
                f"?start-date={today.strftime('%Y-%m-%d')}&end-date={today.strftime('%Y-%m-%d')}"
                f"&time-frame=Daily&add-missing-rows=false"
            )
            
            response = req.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            records = response.json().get('data', [])
            
            if not records:
                self._log("No live ASX200 data available for today", 'info')
                return None
            
            # Use the first (most recent) record
            rec = records[0]
            live_price = float(rec['last_closeRaw'])
            prev_price = float(fund_data['price'].iloc[-1])
            live_return = (live_price - prev_price) / prev_price
            
            live_date = pd.Timestamp(rec['rowDateTimestamp']).normalize().tz_localize(None)
            
            live_row = pd.DataFrame({
                'daily_return': [live_return],
                'price': [live_price]
            }, index=pd.DatetimeIndex([live_date], name='date'))
            
            self._log(f"✓ Live ASX200: {live_price:.2f} ({live_return*100:+.2f}%) as of {live_date.date()}", 'success')
            return live_row
            
        except Exception as e:
            self._log(f"⚠ Could not fetch live ASX200 data: {e}", 'warning')
            return None
    
    def prepare_combined_data_for_prediction(self) -> pd.DataFrame:
        """
        Like prepare_combined_data but appends today's live ASX200 data
        so the prediction uses the most current information.
        """
        fund_data = self.load_local_data()
        if fund_data.empty:
            return pd.DataFrame()
        
        # Append live data for today if available
        live_row = self.fetch_live_asx200()
        if live_row is not None:
            fund_data = pd.concat([fund_data, live_row])
            fund_data = fund_data[~fund_data.index.duplicated(keep='last')]
            fund_data.sort_index(inplace=True)
        
        end_date = fund_data.index.max().date()
        market_end = end_date + timedelta(days=1)
        market_data = self.get_market_data(market_end)
        
        if market_data.empty:
            return fund_data
        
        combined = fund_data.join(market_data, how='left')
        
        if live_row is not None and len(combined) > 1:
            # Identify market columns that are NaN in the live row (no market data yet)
            market_cols = [c for c in market_data.columns if c in combined.columns]
            live_market_nans = combined.iloc[-1][market_cols].isna()
            
            # Forward-fill everything (needed for historical rows)
            combined = combined.ffill()
            
            # Restore NaN for the live row's market columns that had no data.
            # This ensures pct_change in engineer_features computes NaN (not 0)
            # for the live row. engineer_features will then ffill the computed
            # features so the live row inherits the previous day's market returns.
            for col in market_cols:
                if live_market_nans.get(col, False):
                    combined.iloc[-1, combined.columns.get_loc(col)] = np.nan
        else:
            combined = combined.ffill()
        
        combined.dropna(subset=['daily_return', 'price'], inplace=True)
        return combined
    
    # ── Market Regime Classification ──────────────────────────────────

    def classify_market_regime(self, lookback: int = 20) -> str:
        """Classify the current market regime from recent price data.

        Returns one of: 'TRENDING_UP', 'TRENDING_DOWN', 'VOLATILE', 'RANGE_BOUND'

        Logic:
          - Compute rolling std of daily returns (volatility)
          - Compute rolling mean of daily returns (trend)
          - High vol (> 1.5× long-term avg) → VOLATILE
          - Strong positive trend → TRENDING_UP
          - Strong negative trend → TRENDING_DOWN
          - Otherwise → RANGE_BOUND
        """
        fund = self.load_local_data()
        if fund is None or len(fund) < lookback * 2:
            return 'UNKNOWN'

        returns = fund['daily_return'].dropna()
        recent = returns.tail(lookback)
        long_term_vol = returns.std()
        recent_vol = recent.std()
        recent_mean = recent.mean()

        # Thresholds
        vol_ratio = recent_vol / long_term_vol if long_term_vol > 0 else 1.0
        trend_threshold = long_term_vol * 0.3  # ~30% of 1-std as trend signal

        if vol_ratio > 1.5:
            return 'VOLATILE'
        elif recent_mean > trend_threshold:
            return 'TRENDING_UP'
        elif recent_mean < -trend_threshold:
            return 'TRENDING_DOWN'
        else:
            return 'RANGE_BOUND'

    # ── Model Version Tracking ───────────────────────────────────────

    def get_model_version(self) -> str:
        """Return a version identifier for the current trained model.

        Format: 'YYYYMMDD_HHMMSS' based on the model file's modification time.
        Returns 'unknown' if the model file doesn't exist.
        """
        model_path = self.config.get('model', {}).get('save_path', 'data/model.pkl')
        if os.path.exists(model_path):
            mtime = os.path.getmtime(model_path)
            return datetime.fromtimestamp(mtime).strftime('%Y%m%d_%H%M%S')
        return 'unknown'

    # ── Prediction History ───────────────────────────────────────────
    HISTORY_PATH = 'data/asx200history.csv'

    def update_prediction_history(self) -> int:
        """Back-fill actual close prices and success flags for past predictions.

        For every row in the history CSV where ``actual_close`` is NaN,
        look up the real ASX200 close for that ``prediction_date`` from the
        Investing.com data already stored in the local CSV.  If the date
        is today and we have a live price, use that too.

        Returns the number of rows updated.
        """
        if not os.path.exists(self.HISTORY_PATH):
            return 0

        hist = pd.read_csv(self.HISTORY_PATH)
        if hist.empty or 'actual_close' not in hist.columns:
            return 0

        pending = hist['actual_close'].isna()
        if not pending.any():
            # Even if no new back-fills needed, patch legacy rows
            # that are missing the newer columns
            updated_legacy = self._backfill_legacy_columns(hist)
            return updated_legacy

        # Load prices we already know (CSV + live)
        fund = self.load_local_data()
        live = self.fetch_live_asx200()
        if live is not None:
            fund = pd.concat([fund, live])
            fund = fund[~fund.index.duplicated(keep='last')]

        price_lookup = fund['price']           # Series  date→price
        return_lookup = fund['daily_return']    # Series  date→return

        updated = 0
        for idx in hist.index[pending]:
            pred_date = pd.Timestamp(hist.at[idx, 'prediction_date'])
            if pred_date in price_lookup.index:
                actual_close = price_lookup[pred_date]
                actual_return = return_lookup[pred_date]
                predicted_up = hist.at[idx, 'predicted_up']

                # Success = direction matches
                actual_up = 1 if actual_return > 0 else 0
                success = 1 if int(predicted_up) == actual_up else 0

                hist.at[idx, 'actual_close'] = actual_close
                hist.at[idx, 'actual_return'] = actual_return
                hist.at[idx, 'success'] = success

                # Descriptive result label
                if int(predicted_up) == 1 and actual_up == 1:
                    hist.at[idx, 'result_label'] = 'CORRECT_UP'
                elif int(predicted_up) == 0 and actual_up == 0:
                    hist.at[idx, 'result_label'] = 'CORRECT_DOWN'
                elif int(predicted_up) == 1 and actual_up == 0:
                    hist.at[idx, 'result_label'] = 'WRONG_UP'
                else:
                    hist.at[idx, 'result_label'] = 'WRONG_DOWN'

                # Hypothetical return: gain if followed the signal
                # If predicted UP → you bought → actual_return
                # If predicted DOWN → you shorted → -actual_return
                # If NEUTRAL → you did nothing → 0
                signal_str = str(hist.at[idx, 'signal']) if 'signal' in hist.columns else ''
                if 'POSITIVE' in signal_str:
                    hist.at[idx, 'hypothetical_return'] = actual_return
                elif 'NEGATIVE' in signal_str:
                    hist.at[idx, 'hypothetical_return'] = -actual_return
                else:
                    hist.at[idx, 'hypothetical_return'] = 0.0

                updated += 1

        if updated:
            hist.to_csv(self.HISTORY_PATH, index=False, float_format='%.6f')
            self._log(f"✓ Updated {updated} prediction(s) in history with actual results", 'success')

        # Also patch any legacy rows missing the newer columns
        self._backfill_legacy_columns(hist)

        return updated

    def _backfill_legacy_columns(self, hist: pd.DataFrame) -> int:
        """Patch older history rows that lack result_label / hypothetical_return.

        Returns number of rows patched.
        """
        patched = 0
        completed = hist.dropna(subset=['success'])

        for idx in completed.index:
            needs_patch = False

            # result_label
            if 'result_label' not in hist.columns or pd.isna(hist.at[idx, 'result_label']):
                needs_patch = True
                predicted_up = int(hist.at[idx, 'predicted_up'])
                actual_return = hist.at[idx, 'actual_return']
                actual_up = 1 if actual_return > 0 else 0
                if predicted_up == 1 and actual_up == 1:
                    hist.at[idx, 'result_label'] = 'CORRECT_UP'
                elif predicted_up == 0 and actual_up == 0:
                    hist.at[idx, 'result_label'] = 'CORRECT_DOWN'
                elif predicted_up == 1 and actual_up == 0:
                    hist.at[idx, 'result_label'] = 'WRONG_UP'
                else:
                    hist.at[idx, 'result_label'] = 'WRONG_DOWN'

            # hypothetical_return
            if 'hypothetical_return' not in hist.columns or pd.isna(hist.at[idx, 'hypothetical_return']):
                needs_patch = True
                actual_return = hist.at[idx, 'actual_return']
                signal_str = str(hist.at[idx, 'signal']) if 'signal' in hist.columns else ''
                if 'POSITIVE' in signal_str:
                    hist.at[idx, 'hypothetical_return'] = actual_return
                elif 'NEGATIVE' in signal_str:
                    hist.at[idx, 'hypothetical_return'] = -actual_return
                else:
                    hist.at[idx, 'hypothetical_return'] = 0.0

            if needs_patch:
                patched += 1

        if patched:
            hist.to_csv(self.HISTORY_PATH, index=False, float_format='%.6f')
            self._log(f"✓ Patched {patched} legacy row(s) with result_label/hypothetical_return", 'info')

        return patched

    def save_prediction_to_history(
        self,
        prediction_date: str,
        base_date: str,
        base_price: float,
        probability: float,
        predicted_up: int,
        signal: str,
        confidence_level: str,
        feature_details: list,
        model_version: str = 'unknown',
        market_regime: str = 'UNKNOWN',
    ) -> None:
        """Append one prediction row to the history CSV.

        ``feature_details`` is the list of dicts returned by
        ``ModelManager.predict()`` — each with *name*, *value*, *importance*.
        All feature values are stored as individual columns so you can
        inspect exactly what the model saw.
        """
        row = {
            'prediction_date': prediction_date,
            'run_date': date.today().isoformat(),
            'base_date': base_date,
            'base_price': base_price,
            'probability': probability,
            'predicted_up': predicted_up,
            'signal': signal,
            'confidence_level': confidence_level,
            'model_version': model_version,
            'market_regime': market_regime,
        }

        # Add every feature value as its own column
        for fd in feature_details:
            row[f"feat_{fd['name']}"] = fd['value']

        row['actual_close'] = None
        row['actual_return'] = None
        row['success'] = None
        row['result_label'] = None
        row['hypothetical_return'] = None

        new_row = pd.DataFrame([row])

        if os.path.exists(self.HISTORY_PATH):
            hist = pd.read_csv(self.HISTORY_PATH)
            # Don't duplicate — overwrite if same prediction_date exists
            hist = hist[hist['prediction_date'] != prediction_date]
            hist = pd.concat([hist, new_row], ignore_index=True)
        else:
            hist = new_row

        hist.to_csv(self.HISTORY_PATH, index=False, float_format='%.6f')
        self._log(f"✓ Saved prediction for {prediction_date} to history", 'success')

    # ── Performance Analysis ──────────────────────────────────────────

    def get_prediction_performance(self, min_predictions: int = 5) -> Optional[Dict[str, Any]]:
        """Return a summary of prediction performance from the history CSV.

        Returns None if there aren't enough completed predictions.
        """
        if not os.path.exists(self.HISTORY_PATH):
            return None

        hist = pd.read_csv(self.HISTORY_PATH)
        completed = hist.dropna(subset=['success'])
        if len(completed) < min_predictions:
            return None

        total = len(completed)
        correct = int(completed['success'].sum())
        accuracy = correct / total

        # Day-of-week breakdown
        completed = completed.copy()
        completed['dow'] = pd.to_datetime(completed['prediction_date']).dt.day_name()
        dow_stats = (
            completed.groupby('dow')['success']
            .agg(['mean', 'count'])
            .rename(columns={'mean': 'accuracy', 'count': 'n'})
        )
        # Order Mon-Fri
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        dow_stats = dow_stats.reindex([d for d in day_order if d in dow_stats.index])

        # Confidence-level breakdown
        conf_stats = None
        if 'confidence_level' in completed.columns:
            conf_stats = (
                completed.groupby('confidence_level')['success']
                .agg(['mean', 'count'])
                .rename(columns={'mean': 'accuracy', 'count': 'n'})
            )
            level_order = ['VERY_LOW', 'LOW', 'MODERATE', 'HIGH', 'VERY_HIGH']
            conf_stats = conf_stats.reindex([l for l in level_order if l in conf_stats.index])

        # Rolling accuracy (last 10 / 20 / all)
        recent_10 = completed.tail(10)['success'].mean() if total >= 10 else None
        recent_20 = completed.tail(20)['success'].mean() if total >= 20 else None

        # Hypothetical returns
        hyp = {}
        if 'hypothetical_return' in completed.columns:
            h = completed['hypothetical_return'].dropna()
            if len(h) > 0:
                hyp['total_return'] = float(h.sum())
                hyp['mean_return'] = float(h.mean())
                hyp['cumulative_pct'] = float(((1 + h).prod() - 1))
                hyp['win_rate'] = float((h > 0).mean())
                hyp['n_trades'] = int((h != 0).sum())
                hyp['avg_win'] = float(h[h > 0].mean()) if (h > 0).any() else 0.0
                hyp['avg_loss'] = float(h[h < 0].mean()) if (h < 0).any() else 0.0

        # By market regime
        regime_stats = {}
        if 'market_regime' in completed.columns:
            rs = (
                completed.groupby('market_regime')['success']
                .agg(['mean', 'count'])
                .rename(columns={'mean': 'accuracy', 'count': 'n'})
            )
            regime_stats = rs.to_dict('index') if not rs.empty else {}

        # By model version
        version_stats = {}
        if 'model_version' in completed.columns:
            vs = (
                completed.groupby('model_version')['success']
                .agg(['mean', 'count'])
                .rename(columns={'mean': 'accuracy', 'count': 'n'})
            )
            version_stats = vs.to_dict('index') if not vs.empty else {}

        return {
            'total': total,
            'correct': correct,
            'accuracy': accuracy,
            'recent_10': recent_10,
            'recent_20': recent_20,
            'by_day': dow_stats.to_dict('index') if not dow_stats.empty else {},
            'by_confidence': conf_stats.to_dict('index') if conf_stats is not None and not conf_stats.empty else {},
            'hypothetical': hyp,
            'by_regime': regime_stats,
            'by_model_version': version_stats,
        }

    def analyze_thresholds(self, step: float = 0.05) -> Optional[pd.DataFrame]:
        """Find optimal probability threshold from history.

        Returns a DataFrame with threshold / accuracy / trade-count,
        or None if not enough data.
        """
        if not os.path.exists(self.HISTORY_PATH):
            return None

        hist = pd.read_csv(self.HISTORY_PATH).dropna(subset=['success'])
        if len(hist) < 10:
            return None

        thresholds = np.arange(0.50, 0.95, step)
        rows = []
        for t in thresholds:
            # Positive-direction trades above threshold
            pos = hist[hist['probability'] >= t]
            # Negative-direction trades below 1-threshold
            neg = hist[hist['probability'] <= 1 - t]
            trades = pd.concat([pos, neg])
            if len(trades) >= 5:
                rows.append({
                    'threshold': round(float(t), 2),
                    'accuracy': trades['success'].mean(),
                    'trades': len(trades),
                    'correct': int(trades['success'].sum()),
                })

        if not rows:
            return None
        return pd.DataFrame(rows).sort_values('accuracy', ascending=False)

    PERF_LOG_PATH = 'data/performance_log.csv'

    def save_performance_snapshot(self) -> None:
        """Append a daily summary row to performance_log.csv.

        Records the aggregated accuracy metrics at the time of the call
        so you can track how model performance evolves over time.
        Skips writing if a row for today already exists.
        """
        perf = self.get_prediction_performance(min_predictions=1)
        if perf is None:
            return

        today = date.today().isoformat()

        # Don't duplicate today's snapshot
        if os.path.exists(self.PERF_LOG_PATH):
            existing = pd.read_csv(self.PERF_LOG_PATH)
            if today in existing['snapshot_date'].values:
                return

        # Best threshold from analyze_thresholds
        best_threshold = None
        best_threshold_acc = None
        try:
            th = self.analyze_thresholds()
            if th is not None and not th.empty:
                best = th.iloc[0]
                best_threshold = best['threshold']
                best_threshold_acc = best['accuracy']
        except Exception:
            pass

        drift = self.detect_model_drift()

        row = {
            'snapshot_date': today,
            'total_predictions': perf['total'],
            'correct': perf['correct'],
            'accuracy': round(perf['accuracy'], 4),
            'recent_10': round(perf['recent_10'], 4) if perf.get('recent_10') is not None else None,
            'recent_20': round(perf['recent_20'], 4) if perf.get('recent_20') is not None else None,
            'best_threshold': best_threshold,
            'best_threshold_acc': best_threshold_acc,
            'drift_detected': drift,
        }

        new_row = pd.DataFrame([row])
        if os.path.exists(self.PERF_LOG_PATH):
            log = pd.read_csv(self.PERF_LOG_PATH)
            log = pd.concat([log, new_row], ignore_index=True)
        else:
            log = new_row

        log.to_csv(self.PERF_LOG_PATH, index=False)
        self._log(f"✓ Performance snapshot saved for {today}", 'info')

    def get_performance_log(self) -> Optional[pd.DataFrame]:
        """Return the full performance_log.csv as a DataFrame, or None."""
        if not os.path.exists(self.PERF_LOG_PATH):
            return None
        log = pd.read_csv(self.PERF_LOG_PATH)
        return log if not log.empty else None

    def detect_model_drift(self, window: int = 20, drop_pct: float = 0.10) -> bool:
        """Return True if recent accuracy is significantly worse than overall.

        Args:
            window: number of most-recent completed predictions to check
            drop_pct: accuracy drop that triggers the warning (default 10%)
        """
        if not os.path.exists(self.HISTORY_PATH):
            return False

        hist = pd.read_csv(self.HISTORY_PATH).dropna(subset=['success'])
        if len(hist) < window:
            return False

        overall = hist['success'].mean()
        recent = hist.tail(window)['success'].mean()

        if recent < overall - drop_pct:
            self._log(
                f"⚠ Model drift detected! Recent {window} accuracy: "
                f"{recent:.1%} vs overall: {overall:.1%}",
                'warning',
            )
            return True
        return False

    def _log(self, message: str, level: str = 'info'):
        """Log a message to the queue if available, otherwise print to console"""
        if self.log_queue:
            self.log_queue.put(message, level)
        else:
            print(message)
