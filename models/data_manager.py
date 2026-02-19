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
    
    def _log(self, message: str, level: str = 'info'):
        """Log a message to the queue if available, otherwise print to console"""
        if self.log_queue:
            self.log_queue.put(message, level)
        else:
            print(message)
