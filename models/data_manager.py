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
    
    # Default market sources (used when config omits the key)
    _DEFAULT_MARKET_SOURCES = [
        {'name': 'asx_futures',    'ticker': '8824',     'source': 'investing', 'shift': False, 'category': 'futures', 'live_source': 'investing', 'live_ticker': '8824'},
        {'name': 'sp500_futures',  'ticker': 'ES=F',      'shift': False, 'category': 'futures', 'live_source': 'investing', 'live_ticker': '1175153', 'price_field': 'last_openRaw'},
        {'name': 'vix',            'ticker': '^VIX',      'shift': True,  'category': 'volatility'},
        {'name': 'asx_vix',        'ticker': '^AXVI',     'shift': True,  'category': 'volatility'},
        {'name': 'gold',           'ticker': 'GC=F',      'shift': True,  'category': 'commodity'},
        {'name': 'copper',         'ticker': 'HG=F',      'shift': True,  'category': 'commodity'},
        {'name': 'oil',            'ticker': 'CL=F',      'shift': True,  'category': 'commodity'},
        {'name': 'iron_ore_proxy', 'ticker': 'BHP.AX',    'shift': False, 'category': 'commodity'},
        {'name': 'audusd',         'ticker': 'AUDUSD=X',  'shift': False, 'category': 'currency'},
    ]

    def __init__(self, config: Dict[str, Any], log_queue=None):
        self.config = config
        self.local_csv_path = config['data']['local_csv_path']
        self.start_date = datetime.strptime(config['data']['start_date'], '%d/%m/%Y')
        self.time_utils = SydneyTimeUtils()
        self.log_queue = log_queue

        # Market data source table – read from config, fall back to defaults
        self.MARKET_SOURCES = config.get('market_sources',
                                         self._DEFAULT_MARKET_SOURCES)
        self._SHIFT_COLS = tuple(s['name'] for s in self.MARKET_SOURCES
                                 if s['shift'])

        # Derive history / performance log paths from data_folder
        data_folder = config.get('data_folder', 'data')
        self.HISTORY_PATH = os.path.join(data_folder, 'asx200history.csv')
        self.PERF_LOG_PATH = os.path.join(data_folder, 'performance_log.csv')

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

    # ── Investing.com helper ─────────────────────────────────────────

    def _fetch_investing_series(
        self, pair_id: int, start_date: date, end_date: date,
        price_field: str = 'last_closeRaw',
    ) -> Optional[pd.Series]:
        """Fetch daily prices from the Investing.com historical API.

        *price_field* selects which OHLC field to extract, e.g.
        ``'last_closeRaw'`` (default), ``'last_openRaw'``,
        ``'last_maxRaw'``, ``'last_minRaw'``.

        Returns a datetime-indexed Series, or None on failure.
        """
        import requests as req

        headers = {
            'User-Agent': ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                           'AppleWebKit/537.36'),
            'domain-id': 'au',
        }
        start_str = start_date.strftime('%Y-%m-%d') if isinstance(start_date, date) else str(start_date)
        end_str = end_date.strftime('%Y-%m-%d') if isinstance(end_date, date) else str(end_date)
        url = (
            f"https://api.investing.com/api/financialdata/historical/{pair_id}"
            f"?start-date={start_str}&end-date={end_str}"
            f"&time-frame=Daily&add-missing-rows=false"
        )

        try:
            resp = req.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            records = resp.json().get('data', [])
            if not records:
                return None
            dates = [pd.Timestamp(r['rowDateTimestamp']) for r in records]
            values = [float(r[price_field]) for r in records]
            series = pd.Series(
                values,
                index=pd.DatetimeIndex(dates).normalize().tz_localize(None),
            )
            series = series[~series.index.duplicated(keep='first')]
            series.sort_index(inplace=True)
            return series
        except Exception as e:
            self._log(f"⚠ Investing.com fetch failed (pair {pair_id}): {e}", 'warning')
            return None

    def get_market_data(self, end_date: date) -> pd.DataFrame:
        """Fetch market data for all sources in MARKET_SOURCES."""
        if isinstance(end_date, date):
            end_date = datetime.combine(end_date, datetime.min.time())

        # ── 1. Download each ticker ──────────────────────────────────
        raw: Dict[str, pd.Series] = {}
        for src in self.MARKET_SOURCES:
            name, ticker = src['name'], src['ticker']
            source = src.get('source', 'yfinance')
            try:
                if source == 'investing':
                    close = self._fetch_investing_series(
                        pair_id=int(ticker),
                        start_date=self.start_date.date()
                            if isinstance(self.start_date, datetime)
                            else self.start_date,
                        end_date=end_date.date()
                            if isinstance(end_date, datetime)
                            else end_date,
                        price_field=src.get('price_field', 'last_closeRaw'),
                    )
                    if close is not None and not close.empty:
                        raw[name] = close
                    else:
                        self._log(f"⚠ {name} (investing:{ticker}) returned empty data", 'warning')
                else:
                    data = yf.download(ticker, start=self.start_date,
                                       end=end_date, progress=False)
                    # Handle MultiIndex columns from yfinance
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    if not data.empty and 'Close' in data.columns:
                        close = data['Close']
                        # yfinance may return MultiIndex → squeeze to 1-D Series
                        if isinstance(close, pd.DataFrame):
                            close = close.squeeze(axis=1)
                        raw[name] = close
                    else:
                        self._log(f"⚠ {name} ({ticker}) returned empty data", 'warning')
            except Exception as e:
                self._log(f"⚠ Failed to fetch {name} ({ticker}): {e}", 'warning')

        if not raw:
            return pd.DataFrame()

        # ── 2. Validate & convert to clean 1-D Series ────────────────
        clean_data: Dict[str, pd.Series] = {}
        for name, series in raw.items():
            try:
                if len(series) == 0:
                    self._log(f"⚠ Skipping {name} - empty series", 'warning')
                    continue
                if hasattr(series, 'isna') and series.isna().all():
                    self._log(f"⚠ Skipping {name} - all NaN values", 'warning')
                    continue
                values = series.values.flatten()
                index = pd.to_datetime(series.index)
                clean_data[name] = pd.Series(values, index=index, name=name)
                self._log(f"✓ {name}: {len(values)} records", 'info')
            except Exception as e:
                self._log(f"⚠ Failed to convert {name}: {e}", 'warning')

        if not clean_data:
            self._log("⚠ No valid market data available", 'warning')
            return pd.DataFrame()

        # ── 3. Assemble DataFrame ────────────────────────────────────
        try:
            df = pd.DataFrame(clean_data)
            self._log(f"✓ Market data loaded: {len(df)} records, "
                       f"{len(clean_data)} indicators", 'success')
            return df
        except Exception as e:
            self._log(f"✗ DataFrame creation failed: {e}", 'error')
            return pd.DataFrame()

    def _ffill_with_staleness_check(
        self, df: pd.DataFrame, max_stale_days: int = 5,
    ) -> pd.DataFrame:
        """Forward-fill NaN values and warn about stale columns.

        For each market-source column, count the maximum consecutive
        NaN gap that ffill has to bridge.  If any column exceeds
        *max_stale_days*, log a warning so the user knows the model
        is using potentially outdated data.
        """
        market_cols = [s['name'] for s in self.MARKET_SOURCES
                       if s['name'] in df.columns]
        stale: dict = {}  # col → max consecutive NaN gap
        for col in market_cols:
            is_nan = df[col].isna()
            if not is_nan.any():
                continue
            # Run-length of consecutive NaNs
            groups = (is_nan != is_nan.shift()).cumsum()
            max_gap = is_nan.groupby(groups).sum().max()
            if max_gap > max_stale_days:
                stale[col] = int(max_gap)

        filled = df.ffill()

        if stale:
            parts = [f"{col} ({days}d)" for col, days in
                     sorted(stale.items(), key=lambda x: -x[1])]
            self._log(
                f"⚠ Stale data forward-filled (>{max_stale_days}d gap): "
                + ", ".join(parts),
                'warning',
            )
        return filled

    def _time_align_market(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Shift non-Australian close prices back one day to remove
        look-ahead bias.  At 15:30 Sydney on day D only the D-1 close
        for US/European markets is available."""
        df = market_data.copy()
        for col in self._SHIFT_COLS:
            if col in df.columns:
                df[col] = df[col].shift(1)
        return df

    def prepare_combined_data(self) -> pd.DataFrame:
        """
        Load local AustralianSuper data and merge with market data.
        Non-Australian closes are shifted by one day so that training
        only uses information available at the 15:30 Sydney decision time.
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
        
        # Time-align: shift non-Australian closes to avoid look-ahead bias
        market_data = self._time_align_market(market_data)

        combined = fund_data.join(market_data, how='left')
        combined = self._ffill_with_staleness_check(combined)
        combined.dropna(inplace=True)
        return combined
    
    def fetch_live_asx200(self) -> Optional[pd.DataFrame]:
        """Fetch today's live/intraday ASX200 price via yfinance ^AXJO.

        Returns a single-row DataFrame with daily_return and price,
        or None if today's data is not available or already in CSV.
        """
        try:
            import yfinance as yf

            fund_data = self.load_local_data()
            if fund_data.empty:
                return None

            last_csv_date = fund_data.index.max().date()
            today = date.today()

            # Only fetch if today is newer than what's in the CSV
            if last_csv_date >= today:
                return None

            ticker = yf.Ticker("^AXJO")
            info = ticker.fast_info

            live_price = float(info.last_price)
            prev_close = float(info.previous_close)
            if live_price <= 0 or prev_close <= 0:
                self._log("No live ASX200 data available", 'info')
                return None

            # Return relative to last price in our CSV for consistency
            prev_price = float(fund_data['price'].iloc[-1])
            live_return = (live_price - prev_price) / prev_price

            live_date = pd.Timestamp(today)

            live_row = pd.DataFrame({
                'daily_return': [live_return],
                'price': [live_price]
            }, index=pd.DatetimeIndex([live_date], name='date'))

            intraday_chg = (live_price - prev_close) / prev_close * 100
            self._log(
                f"✓ Live ASX200: {live_price:,.2f} ({intraday_chg:+.2f}%) as of {today}",
                'success'
            )
            return live_row

        except Exception as e:
            self._log(f"⚠ Could not fetch live ASX200 data: {e}", 'warning')
            return None

    def fetch_live_market_quotes(self) -> Dict[str, Dict[str, float]]:
        """
        Fetch live prices and daily percentage changes for non-shifted sources.

        At prediction time (15:45 Sydney) we need current prices for
        instruments that trade live during Australian hours:
        - futures (sp500_futures, asx_futures via yfinance or Investing.com)
        - non-shifted commodities (iron_ore_proxy / BHP.AX)
        - currency (audusd)

        Shifted sources (vix, gold, copper, oil) use yesterday's daily
        close which is already in the historical data — no live fetch needed.

        Returns a dict: {source_name: {'price': price, 'pct': pct_change}}
        where pct_change is the daily percentage change (as a decimal, e.g., 0.0005 for +0.05%).
        """
        import requests as req
        import yfinance as yf

        quotes = {}
        for src in self.MARKET_SOURCES:
            if src['shift']:
                continue
            name = src['name']
            live_source = src.get('live_source', src.get('source', 'yfinance'))
            live_ticker = src.get('live_ticker', src['ticker'])

            if live_source == 'investing':
                try:
                    today = date.today()
                    # Fetch a small range (last 5 days) to get today's data and previous close
                    start = today - timedelta(days=5)
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                        'domain-id': 'au',
                    }
                    url = (
                        f"https://api.investing.com/api/financialdata/historical/{live_ticker}"
                        f"?start-date={start.strftime('%Y-%m-%d')}"
                        f"&end-date={today.strftime('%Y-%m-%d')}"
                        f"&time-frame=Daily&add-missing-rows=false"
                    )
                    resp = req.get(url, headers=headers, timeout=30)
                    resp.raise_for_status()
                    data = resp.json().get('data', [])
                    if not data:
                        self._log(f"⚠ No live data for {name}", 'warning')
                        continue

                    # Most recent record is first in list
                    latest = data[0]
                    pf = src.get('price_field', 'last_closeRaw')
                    live_price = float(latest[pf])
                    # Use the provided percentage change directly (convert from percent to decimal)
                    live_pct = float(latest['change_precentRaw']) / 100.0
                    quotes[name] = {'price': live_price, 'pct': live_pct}
                    self._log(f"✓ Live {name}: {live_price:,.2f} ({live_pct*100:+.2f}%)", 'success')
                except Exception as e:
                    self._log(f"⚠ Live fetch failed for {name}: {e}", 'warning')
                continue

            # yfinance sources
            try:
                t = yf.Ticker(live_ticker)
                info = t.fast_info
                price = float(info.last_price)
                prev = float(info.previous_close)
                if price > 0 and prev > 0:
                    pct = (price - prev) / prev
                    quotes[name] = {'price': price, 'pct': pct}
                    self._log(f"✓ Live {name}: {price:,.4f} ({pct*100:+.2f}%)", 'success')
            except Exception as e:
                self._log(f"⚠ Live fetch failed for {name}: {e}", 'warning')

        return quotes
    
    def prepare_combined_data_for_prediction(self) -> pd.DataFrame:
        """
        Prepare data for prediction with live market prices and returns.

        Pipeline:
        1. Load local AustralianSuper CSV
        2. Append today's live ASX200 row (price + daily_return)
        3. Fetch historical market data and time-align (shift) as in training
        4. Join market data to fund data
        5. Fetch live quotes for non-shifted sources and inject into the
        live row:
            - Set the raw price column
            - Also set the corresponding `_return` column using the live
            percentage change (so it exactly matches what the ticker reports)
        6. Forward-fill remaining gaps (historical rows + any sources that
        failed to return a live quote)
        7. Drop rows missing essential columns (daily_return, price)

        This ensures that the `_return` features for the live row are
        consistent with the live feed's percentage change, eliminating
        mismatches caused by differing previous closes between data sources.
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

        # Time-align: shift non-Australian closes to avoid look-ahead bias
        market_data = self._time_align_market(market_data)

        combined = fund_data.join(market_data, how='left')

        # ── Inject live market prices and returns into the live row ─────
        if live_row is not None and len(combined) > 1:
            live_quotes = self.fetch_live_market_quotes()
            for col, data in live_quotes.items():
                if col in combined.columns:
                    # Set the raw price
                    combined.iloc[-1, combined.columns.get_loc(col)] = data['price']
                    # Set the return feature if it exists
                    return_col = f"{col}_return"
                    if return_col in combined.columns and data['pct'] is not None:
                        combined.iloc[-1, combined.columns.get_loc(return_col)] = data['pct']

        # Forward-fill remaining gaps (with staleness warnings)
        combined = self._ffill_with_staleness_check(combined)

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

        # Force string columns to object dtype so .at assignments don't fail
        for col in ('result_label', 'signal', 'confidence_level',
                     'market_regime', 'model_version'):
            if col in hist.columns:
                hist[col] = hist[col].astype(object)

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

        # Force string columns to object dtype so .at assignments don't fail
        for col in ('result_label', 'signal', 'confidence_level',
                     'market_regime', 'model_version'):
            if col in hist.columns:
                hist[col] = hist[col].astype(object)

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

        row['actual_close'] = np.nan
        row['actual_return'] = np.nan
        row['success'] = np.nan
        row['result_label'] = ''
        row['hypothetical_return'] = np.nan

        new_row = pd.DataFrame([row])
        # Ensure string columns stay as object dtype so concat doesn't fail
        for col in ('result_label', 'signal', 'confidence_level',
                     'market_regime', 'model_version'):
            if col in new_row.columns:
                new_row[col] = new_row[col].astype(object)

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

    # PERF_LOG_PATH is set in __init__ from config['data_folder']

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

    def detect_model_drift(
        self,
        window: int = 20,
        drop_pct: float = 0.10,
        min_total: int = 40,
        relative_drop: float = 0.20,
    ) -> bool:
        """Return True if recent accuracy is significantly worse than overall.

        Uses both an absolute and a relative drop check so the signal
        adapts to the baseline accuracy level.  Also requires a minimum
        number of total completed predictions before reporting drift,
        since at low counts a 10-pt absolute drop can easily be noise.

        Args:
            window: number of most-recent completed predictions to check
            drop_pct: absolute accuracy drop that triggers the warning
                      (default 0.10 → 10 percentage points)
            min_total: minimum total completed predictions before drift
                       detection is trusted (default 40, i.e. 2× window)
            relative_drop: proportional accuracy drop that triggers the
                           warning (default 0.20 → 20% relative decline,
                           e.g. 55% → 44%)
        """
        if not os.path.exists(self.HISTORY_PATH):
            return False

        hist = pd.read_csv(self.HISTORY_PATH).dropna(subset=['success'])
        if len(hist) < max(window, min_total):
            return False

        overall = hist['success'].mean()
        recent = hist.tail(window)['success'].mean()

        abs_drop = overall - recent
        rel_drop = abs_drop / overall if overall > 0 else 0.0

        if abs_drop >= drop_pct or rel_drop >= relative_drop:
            self._log(
                f"⚠ Model drift detected! Recent {window} accuracy: "
                f"{recent:.1%} vs overall: {overall:.1%} "
                f"(abs drop {abs_drop:.1%}, rel drop {rel_drop:.1%})",
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
