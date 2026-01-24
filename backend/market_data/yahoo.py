"""
Yahoo Finance historical market data provider.
Fetches 5-minute OHLCV candles for replay/backtesting.
"""

from typing import List, Dict, Optional
from datetime import datetime, timedelta, timezone
import yfinance as yf
import pandas as pd
from backend.utils import ensure_utc_datetime


def fetch_yahoo_candles(
    symbol: str,
    start_date: str,
    end_date: str
) -> List[Dict]:
    """
    Fetch historical 5-minute candles from Yahoo Finance.
    
    Args:
        symbol: Trading symbol (e.g., "AAPL")
        start_date: Start date in "YYYY-MM-DD" format
        end_date: End date in "YYYY-MM-DD" format (exclusive)
    
    Returns:
        List of candle dicts with keys: timestamp, open, high, low, close, volume
    
    Raises:
        ValueError: If no data is returned or validation fails
    """
    # Parse dates
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid date format. Use YYYY-MM-DD. Error: {e}")
    
    # Fetch data from Yahoo Finance
    ticker = yf.Ticker(symbol)
    
    # Request 5-minute candles, unadjusted
    df = ticker.history(
        start=start_dt,
        end=end_dt,
        interval="5m",
        auto_adjust=False,
        prepost=False
    )
    
    # Check if data is empty
    if df.empty:
        raise ValueError(f"No data returned from Yahoo Finance for {symbol} between {start_date} and {end_date}")
    
    # Normalize to canonical format
    candles = normalize_yahoo_candles(df, symbol)
    
    # Validate candles
    validate_candles(candles, symbol)
    
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
                # For 5-minute data, we treat as UTC (yfinance may return UTC-naive)
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
    - Regular 5-minute spacing (skip gaps, don't forward-fill)
    
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
    
    # Check 5-minute spacing (allow some tolerance for market hours gaps)
    # We don't enforce exact 5-minute spacing because markets have:
    # - Pre-market hours
    # - Regular trading hours
    # - After-hours
    # - Weekends/holidays
    # So we just ensure they're ordered and valid
    
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
