"""
Yahoo Finance historical market data provider.
Fetches daily OHLCV candles for replay/backtesting.

Note: Daily candles are used as the default because Yahoo Finance has limitations
on intraday data availability and reliability. For historical backtesting, daily
candles provide sufficient granularity and better data quality.
"""

from typing import List, Dict, Optional
from datetime import datetime, timedelta, timezone
import yfinance as yf
import pandas as pd
from utils import ensure_utc_datetime


def fetch_yahoo_candles(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1d"
) -> List[Dict]:
    """
    Fetch historical daily candles from Yahoo Finance.
    
    Args:
        symbol: Trading symbol (e.g., "AAPL", "SPY")
        start_date: Start date in "YYYY-MM-DD" format (inclusive)
        end_date: End date in "YYYY-MM-DD" format (inclusive - yfinance end is exclusive, so we add 1 day)
        interval: Data interval (default: "1d" for daily candles)
                  Note: Intraday intervals (e.g., "5m", "1h") are not supported
                  due to Yahoo Finance limitations. Use "1d" for reliable backtesting.
    
    Returns:
        List of candle dicts with keys: timestamp, open, high, low, close, volume
    
    Raises:
        ValueError: If no data is returned, validation fails, or intraday interval is requested
    """
    # Validate interval: Only daily candles are supported
    if interval != "1d":
        raise ValueError(
            f"Intraday intervals are not supported due to Yahoo Finance limitations. "
            f"Requested interval: {interval}. Use '1d' for daily candles instead."
        )
    
    # Parse dates
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid date format. Use YYYY-MM-DD. Error: {e}")
    
    # CRITICAL: yfinance end_date is EXCLUSIVE, so add 1 day to make it inclusive
    # This ensures we get data for the end_date itself
    end_dt_inclusive = end_dt + timedelta(days=1)
    
    # Log fetch attempt
    print(f"[YAHOO FINANCE] Fetching DAILY candles for {symbol}")
    print(f"[YAHOO FINANCE]   Start date: {start_date} (inclusive)")
    print(f"[YAHOO FINANCE]   End date: {end_date} (inclusive, yfinance end={end_dt_inclusive.date()})")
    
    # Fetch data from Yahoo Finance
    ticker = yf.Ticker(symbol)
    
    # Request daily candles, unadjusted
    df = ticker.history(
        start=start_dt,
        end=end_dt_inclusive,  # yfinance end is exclusive, so we add 1 day
        interval="1d",  # Daily candles (default and only supported interval)
        auto_adjust=False,
        prepost=False
    )
    
    # Log raw DataFrame info
    print(f"[YAHOO FINANCE]   Raw DataFrame rows returned: {len(df)}")
    
    # Check if data is empty
    if df.empty:
        error_msg = f"No DAILY data returned from Yahoo Finance for {symbol} between {start_date} and {end_date}"
        print(f"[YAHOO FINANCE] ERROR: {error_msg}")
        raise ValueError(error_msg)
    
    # Normalize to canonical format
    candles = normalize_yahoo_candles(df, symbol)
    
    # Log normalized candle count
    print(f"[YAHOO FINANCE]   Valid candles after normalization: {len(candles)}")
    
    # Validate candles
    validate_candles(candles, symbol)
    
    print(f"[YAHOO FINANCE] SUCCESS: Fetched {len(candles)} daily candles for {symbol}")
    
    return candles


def normalize_yahoo_candles(df: pd.DataFrame, symbol: str) -> List[Dict]:
    """
    Normalize Yahoo Finance DataFrame to canonical candle format.
    
    Args:
        df: Yahoo Finance DataFrame with OHLCV columns
        symbol: Trading symbol (for error messages)
    
    Returns:
        List of normalized candle dicts
    """
    candles = []
    
    for idx, row in df.iterrows():
        # FIX 1: CANONICAL TIME HANDLING
        # Convert index (DatetimeIndex) to UTC timezone-aware datetime
        if isinstance(idx, pd.Timestamp):
            # Yahoo Finance returns timestamps in market timezone (usually EST/EDT)
            # Convert to UTC timezone-aware datetime
            if idx.tz is None:
                # yfinance typically returns timezone-naive timestamps
                # For daily data, we treat as UTC (yfinance may return UTC-naive)
                # This ensures canonical UTC representation
                timestamp = idx.replace(tzinfo=timezone.utc)
            else:
                # Convert to UTC if timezone-aware
                timestamp = idx.tz_convert(timezone.utc)
        else:
            raise ValueError(f"Unexpected index type: {type(idx)}")
        
        # FIX 1: CANONICAL TIME HANDLING - Validate and ensure UTC timezone
        timestamp = ensure_utc_datetime(timestamp, f"Yahoo Finance candle for {symbol}")
        
        # Extract OHLCV values
        open_price = float(row['Open'])
        high_price = float(row['High'])
        low_price = float(row['Low'])
        close_price = float(row['Close'])
        volume = int(row['Volume'])
        
        # Skip rows with missing values (NaN)
        if pd.isna(open_price) or pd.isna(high_price) or pd.isna(low_price) or pd.isna(close_price) or pd.isna(volume):
            continue
        
        # Create normalized candle
        candle = {
            "timestamp": timestamp,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume
        }
        
        candles.append(candle)
    
    return candles


def validate_candles(candles: List[Dict], symbol: str):
    """
    Validate candles meet requirements:
    - Strictly ordered by timestamp ascending
    - No missing OHLCV values (already filtered in normalize)
    - Valid OHLC relationships (low <= open/close <= high)
    
    Note: For daily candles, we don't enforce exact spacing because markets have
    weekends and holidays, so gaps between trading days are expected.
    
    Args:
        candles: List of candle dicts
        symbol: Trading symbol (for error messages)
    
    Raises:
        ValueError: If validation fails
    """
    if not candles:
        raise ValueError(f"No valid candles after normalization for {symbol}")
    
    # Check ordering
    for i in range(1, len(candles)):
        prev_timestamp = candles[i-1]["timestamp"]
        curr_timestamp = candles[i]["timestamp"]
        
        # FIX 1: CANONICAL TIME HANDLING - Ensure timestamps are UTC
        ensure_utc_datetime(prev_timestamp, f"Previous candle timestamp for {symbol}")
        ensure_utc_datetime(curr_timestamp, f"Current candle timestamp for {symbol}")
        
        if curr_timestamp <= prev_timestamp:
            raise ValueError(
                f"Candles not strictly ordered for {symbol}. "
                f"Index {i-1}: {prev_timestamp}, Index {i}: {curr_timestamp}"
            )
    
    # Note: For daily candles, we don't enforce exact spacing because:
    # - Markets have weekends (no trading Saturday/Sunday)
    # - Markets have holidays (no trading on certain days)
    # - We only ensure they're ordered and valid
    
    # Validate OHLCV values are valid numbers
    for i, candle in enumerate(candles):
        if not all(isinstance(candle[key], (int, float)) for key in ['open', 'high', 'low', 'close']):
            raise ValueError(f"Invalid OHLC values in candle {i} for {symbol}")
        if not isinstance(candle['volume'], int):
            raise ValueError(f"Invalid volume in candle {i} for {symbol}")
        
        # Validate OHLC relationships
        if not (candle['low'] <= candle['open'] <= candle['high']):
            raise ValueError(f"Invalid OHLC relationship in candle {i} for {symbol}: low={candle['low']}, open={candle['open']}, high={candle['high']}")
        if not (candle['low'] <= candle['close'] <= candle['high']):
            raise ValueError(f"Invalid OHLC relationship in candle {i} for {symbol}: low={candle['low']}, close={candle['close']}, high={candle['high']}")


def convert_to_replay_format(candles: List[Dict]) -> List[Dict]:
    """
    Convert normalized candles to ReplayEngine input format.
    
    ReplayEngine expects candles with:
    - time: Unix timestamp (int)
    - open, high, low, close: float
    - volume: int
    
    Args:
        candles: List of normalized candles with timestamp (datetime)
    
    Returns:
        List of candles in ReplayEngine format
    """
    replay_candles = []
    
    for candle in candles:
        # Convert timezone-aware datetime to Unix timestamp
        timestamp_dt = candle["timestamp"]
        if isinstance(timestamp_dt, datetime):
            # Convert to Unix timestamp (seconds)
            unix_timestamp = int(timestamp_dt.timestamp())
        else:
            raise ValueError(f"Unexpected timestamp type: {type(timestamp_dt)}")
        
        replay_candle = {
            "time": unix_timestamp,
            "open": candle["open"],
            "high": candle["high"],
            "low": candle["low"],
            "close": candle["close"],
            "volume": candle["volume"]
        }
        
        replay_candles.append(replay_candle)
    
    return replay_candles
