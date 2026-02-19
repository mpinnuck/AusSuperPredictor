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
        """
        start_str = start_date.strftime('%d/%m/%Y')
        end_str = end_date.strftime('%d/%m/%Y')
        
        base_url = "https://www.australiansuper.com//api/graphs/dailyrates/download/"
        params = {
            'start': start_str,
            'end': end_str,
            'cumulative': 'False',
            'superType': 'super',
            'truncateDecimalPlaces': 'True',
            'outputFilename': f'Daily Rates {start_str.replace("/", " ")} - {end_str.replace("/", " ")}.csv'
        }
        url = base_url + "?" + "&".join([f"{k}={v}" for k, v in params.items()])
        
        try:
            # Add headers to request CSV directly without dialog
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/csv,application/csv,text/plain,*/*',
                'Accept-Encoding': 'gzip, deflate, br',
                'Accept-Language': 'en-US,en;q=0.9',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            }
            
            self._log("Fetching AustralianSuper data from API...", 'info')
            response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
            response.raise_for_status()
            
            # Check if response is actually CSV data
            content_type = response.headers.get('content-type', '').lower()
            if 'text/csv' in content_type or 'application/csv' in content_type or response.text.startswith('Date,'):
                data = response.text
                df = pd.read_csv(pd.StringIO(data))
                
                if len(df.columns) >= 2:
                    df.rename(columns={df.columns[0]: 'date', df.columns[1]: 'daily_return'}, inplace=True)
                    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')
                    df.set_index('date', inplace=True)
                    df.sort_index(inplace=True)
                    df['daily_return'] = pd.to_numeric(df['daily_return'], errors='coerce')
                    df.dropna(subset=['daily_return'], inplace=True)
                    df['price'] = (1 + df['daily_return']).cumprod() * 100
                    self._log("✓ Successfully fetched AustralianSuper data", 'success')
                    return df
                        
            self._log("⚠ Response is not CSV data (likely HTML dialog). Using ASX200 as fallback.", 'warning')
            return self._get_asx200_proxy(start_date, end_date)
                    
        except Exception as e:
            self._log(f"✗ Failed to fetch AustralianSuper data: {e}", 'error')
            self._log("Using ASX200 as fallback data source", 'info')
            return self._get_asx200_proxy(start_date, end_date)
    
    def _get_asx200_proxy(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Fallback using ASX200 index"""
        self._log("Fetching ASX200 data as proxy for AustralianSuper...", 'info')
        self._log(f"Date range: {start_date} to {end_date}", 'info')
        
        try:
            asx = yf.download('^AXJO', start=start_date, end=end_date, progress=False)
            
            if asx.empty:
                self._log("⚠ No ASX200 data available for the specified date range", 'warning')
                return pd.DataFrame()
            
            self._log(f"Raw ASX200 data: {len(asx)} records", 'info')
            returns = asx['Close'].pct_change().dropna()
            
            if returns.empty:
                self._log("⚠ No valid returns calculated from ASX200 data", 'warning')
                return pd.DataFrame()
            
            fund_data = pd.DataFrame({
                'daily_return': returns.values.flatten(),
                'price': asx['Close'].loc[returns.index].values.flatten()
            }, index=returns.index)
            
            # Ensure index is named 'date' for consistency with AustralianSuper data
            fund_data.index.name = 'date'
            self._log(f"✓ Successfully fetched {len(fund_data)} records from ASX200", 'success')
            return fund_data
            
        except Exception as e:
            self._log(f"✗ Error fetching ASX200 data: {e}", 'error')
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
                # Limit initial fetch to last 5 years to avoid issues with very old dates
                start = max(self.start_date.date(), (datetime.now() - timedelta(days=5*365)).date())
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
                    new_data = self.fetch_australiansuper_data(start, end_date)
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
        market_data = self.get_market_data(end_date)
        
        if market_data.empty:
            return fund_data
        
        combined = fund_data.join(market_data, how='left')
        combined = combined.ffill()
        combined.dropna(inplace=True)
        return combined
    
    def _log(self, message: str, level: str = 'info'):
        """Log a message to the queue if available, otherwise print to console"""
        if self.log_queue:
            self.log_queue.put(message, level)
        else:
            print(message)
