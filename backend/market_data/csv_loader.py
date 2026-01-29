"""
CSV-based historical market data provider.
Loads OHLCV candles from CSV files for deterministic backtesting.
"""

from typing import List, Dict
from datetime import datetime, timezone
import csv
import os
from pathlib import Path
from utils import ensure_utc_datetime, calculate_candle_timestamps


def load_csv_candles(csv_path: str, symbol: str = None) -> List[Dict]:
    """
    Load historical candles from a CSV file.
    
    Expected CSV format:
    - Header row with columns: date, open, high, low, close, volume
    - Date column can be: YYYY-MM-DD or ISO format
    - All other columns are floats (except volume which is int)
    
    Args:
        csv_path: Path to CSV file (relative to backend/ or absolute)
        symbol: Trading symbol (for error messages)
    
    Returns:
        List of candle dicts with keys: timestamp, open, high, low, close, volume
        All timestamps are UTC timezone-aware datetime objects
    
    Raises:
        FileNotFoundError: If CSV file does not exist
        ValueError: If CSV format is invalid or validation fails
    """
    # Resolve CSV path
    # Try relative to backend directory first, then absolute
    backend_dir = Path(__file__).parent.parent
    csv_file = backend_dir / csv_path
    
    if not csv_file.exists():
        # Try absolute path
        csv_file = Path(csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(
                f"CSV file not found: {csv_path} "
                f"(tried: {backend_dir / csv_path} and {csv_path})"
            )
    
    symbol_display = symbol if symbol else "unknown"
    print(f"[CSV LOADER] Loading candles from: {csv_file}")
    print(f"[CSV LOADER]   Symbol: {symbol_display}")
    
    candles = []
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            # Try to detect delimiter
            sample = f.read(1024)
            f.seek(0)
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            
            reader = csv.DictReader(f, delimiter=delimiter)
            
            # Validate required columns
            required_columns = {'date', 'open', 'high', 'low', 'close', 'volume'}
            if not required_columns.issubset(reader.fieldnames):
                missing = required_columns - set(reader.fieldnames)
                raise ValueError(
                    f"CSV missing required columns: {missing}. "
                    f"Found columns: {reader.fieldnames}"
                )
            
            row_count = 0
            for row in reader:
                row_count += 1
                
                try:
                    # Parse date
                    date_str = row['date'].strip()
                    # Try multiple date formats
                    date_parsed = None
                    for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y/%m/%d', '%m/%d/%Y']:
                        try:
                            date_parsed = datetime.strptime(date_str, fmt)
                            break
                        except ValueError:
                            continue
                    
                    if date_parsed is None:
                        raise ValueError(f"Could not parse date: {date_str}")
                    
                    # Convert to UTC timezone-aware datetime
                    # For daily data: set time to 00:00:00 UTC
                    # For intraday data: preserve the time component
                    is_daily = date_parsed.hour == 0 and date_parsed.minute == 0 and date_parsed.second == 0
                    if is_daily:
                        # Daily data - normalize to midnight UTC
                        timestamp = date_parsed.replace(
                            hour=0, minute=0, second=0, microsecond=0,
                            tzinfo=timezone.utc
                        )
                    else:
                        # Intraday data - preserve time component, assume UTC
                        timestamp = date_parsed.replace(tzinfo=timezone.utc)
                    timestamp = ensure_utc_datetime(timestamp, f"CSV row {row_count} for {symbol_display}")
                    
                    # Calculate open_time and close_time
                    open_time, close_time = calculate_candle_timestamps(timestamp, is_daily=is_daily)
                    
                    # Parse OHLCV values
                    open_price = float(row['open'])
                    high_price = float(row['high'])
                    low_price = float(row['low'])
                    close_price = float(row['close'])
                    volume = int(float(row['volume']))  # Handle float volume strings
                    
                    # Skip rows with missing/invalid values (NaN check)
                    import math
                    if any(math.isnan(val) for val in [open_price, high_price, low_price, close_price]) or math.isnan(float(volume)):
                        print(f"[CSV LOADER]   Skipping row {row_count}: missing values")
                        continue
                    
                    # Validate OHLC relationships
                    if not (low_price <= open_price <= high_price):
                        raise ValueError(
                            f"Invalid OHLC in row {row_count}: "
                            f"low={low_price}, open={open_price}, high={high_price}"
                        )
                    if not (low_price <= close_price <= high_price):
                        raise ValueError(
                            f"Invalid OHLC in row {row_count}: "
                            f"low={low_price}, close={close_price}, high={high_price}"
                        )
                    
                    # Create normalized candle with explicit open_time and close_time
                    candle = {
                        "timestamp": timestamp,  # Keep for backward compatibility
                        "open_time": open_time,
                        "close_time": close_time,
                        "open": open_price,
                        "high": high_price,
                        "low": low_price,
                        "close": close_price,
                        "volume": volume
                    }
                    
                    candles.append(candle)
                    
                except (ValueError, KeyError) as e:
                    print(f"[CSV LOADER]   Warning: Skipping row {row_count}: {e}")
                    continue
        
        print(f"[CSV LOADER]   Loaded {len(candles)} valid candles from {row_count} rows")
        
    except FileNotFoundError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to read CSV file {csv_path}: {str(e)}")
    
    # Validate candles
    if not candles:
        raise ValueError(f"No valid candles found in CSV file: {csv_path}")
    
    # Validate minimum candle count (>= 100 for meaningful replay)
    MIN_CANDLES_REQUIRED = 100
    if len(candles) < MIN_CANDLES_REQUIRED:
        raise ValueError(
            f"Insufficient candles in CSV file: {csv_path}. "
            f"Got {len(candles)} candles, minimum {MIN_CANDLES_REQUIRED} required."
        )
    
    # Sort candles by timestamp (ascending) - ensure chronological order
    candles.sort(key=lambda x: x["timestamp"])
    
    # Validate ordering (ascending timestamps)
    for i in range(1, len(candles)):
        prev_timestamp = candles[i-1]["timestamp"]
        curr_timestamp = candles[i]["timestamp"]
        
        ensure_utc_datetime(prev_timestamp, f"Previous candle timestamp for {symbol_display}")
        ensure_utc_datetime(curr_timestamp, f"Current candle timestamp for {symbol_display}")
        
        if curr_timestamp <= prev_timestamp:
            raise ValueError(
                f"Candles not strictly ordered in CSV. "
                f"Row {i}: {prev_timestamp} >= {curr_timestamp}"
            )
    
    print(f"[CSV LOADER] SUCCESS: Loaded {len(candles)} candles for {symbol_display}")
    print(f"[CSV LOADER]   Date range: {candles[0]['timestamp'].date()} to {candles[-1]['timestamp'].date()}")
    
    return candles


def convert_to_replay_format(candles: List[Dict]) -> List[Dict]:
    """
    Convert normalized candles to ReplayEngine input format.
    
    ReplayEngine expects candles with:
    - time: Unix timestamp (int) - kept for backward compatibility, maps to close_time
    - open_time: Unix timestamp (int) - candle open time
    - close_time: Unix timestamp (int) - candle close time
    - open, high, low, close: float
    - volume: int
    
    Args:
        candles: List of normalized candles with open_time/close_time (datetime)
    
    Returns:
        List of candles in ReplayEngine format
    """
    replay_candles = []
    
    for candle in candles:
        # Convert open_time and close_time to Unix timestamps
        open_time_dt = candle.get("open_time")
        close_time_dt = candle.get("close_time")
        
        # Fallback to timestamp for backward compatibility
        if open_time_dt is None or close_time_dt is None:
            timestamp_dt = candle.get("timestamp")
            if isinstance(timestamp_dt, datetime):
                is_daily = timestamp_dt.hour == 0 and timestamp_dt.minute == 0
                open_time_dt, close_time_dt = calculate_candle_timestamps(timestamp_dt, is_daily=is_daily)
            else:
                raise ValueError(f"Missing open_time/close_time and invalid timestamp: {type(timestamp_dt)}")
        
        if isinstance(open_time_dt, datetime) and isinstance(close_time_dt, datetime):
            open_time_unix = int(open_time_dt.timestamp())
            close_time_unix = int(close_time_dt.timestamp())
        else:
            raise ValueError(f"Invalid timestamp types: open_time={type(open_time_dt)}, close_time={type(close_time_dt)}")
        
        replay_candle = {
            "time": close_time_unix,  # Backward compatibility: time maps to close_time
            "open_time": open_time_unix,
            "close_time": close_time_unix,
            "open": candle["open"],
            "high": candle["high"],
            "low": candle["low"],
            "close": candle["close"],
            "volume": candle["volume"]
        }
        
        replay_candles.append(replay_candle)
    
    return replay_candles
