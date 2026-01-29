"""
Stooq data loader for daily and intraday candles.
Searches the existing directory structure to locate data files.
"""

from typing import List, Dict, Optional
from datetime import datetime, timezone
from pathlib import Path
import csv
import os
from utils import ensure_utc_datetime


def find_data_file(symbol: str, data_type: str = "daily") -> Optional[Path]:
    """
    Find a data file for the given symbol by searching ONLY the specified directory.
    
    Directory structure:
    - Daily: data d/daily/us/{exchange}/{asset_class}/{subdir}/{symbol}*.txt
    - Intraday: data 5/5 min/us/{exchange}/{asset_class}/{subdir}/{symbol}*.txt
    
    IMPORTANT: This function ONLY searches in the specified base directory:
    - For daily: ONLY searches under data d/daily/us/ (does NOT search raw_data/, data 5/, etc.)
    - For intraday: ONLY searches under data 5/5 min/us/ (does NOT search raw_data/, data d/, etc.)
    
    Args:
        symbol: Trading symbol (e.g., "SPY", "AAPL")
        data_type: "daily" or "intraday"
    
    Returns:
        Path to the data file, or None if not found in the specified directory
    """
    # Determine base directory path
    # CRITICAL: Daily candles ONLY searched under data d/daily/us/
    # CRITICAL: Intraday candles ONLY searched under data 5/5 min/us/
    # Does NOT search: raw_data/, backend/data/, or any other directories
    if data_type == "daily":
        base_path = "data d/daily/us"
    elif data_type == "intraday":
        base_path = "data 5/5 min/us"
    else:
        raise ValueError(f"Invalid data_type: {data_type}. Must be 'daily' or 'intraday'")
    
    # Try multiple base directory locations
    # __file__ is backend/market_data/stooq_loader.py
    # Resolve to absolute path first, then go up: backend/market_data/ -> backend/ -> project root
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    base_dir_candidates = [
        project_root / base_path,  # Relative to project root (preferred)
        Path(base_path),  # Relative to current working directory (fallback)
    ]
    
    base_dir = None
    for candidate in base_dir_candidates:
        if candidate.exists() and candidate.is_dir():
            base_dir = candidate
            break
    
    if base_dir is None:
        # Clear error: searched only in the specified directory
        if data_type == "daily":
            return None  # Error message handled in load_daily_candles()
        else:
            return None  # Error message handled in load_intraday_candles()
    
    # Search pattern: {symbol}*.txt or {symbol}*.csv (case-insensitive)
    # File format is typically: {SYMBOL}.us.txt (e.g., "AAAU.US.TXT" or "aaau.us.txt")
    symbol_upper = symbol.upper()
    symbol_lower = symbol.lower()
    
    # Search all exchanges
    # Structure: data d/daily/us/{exchange}/{subdir}/{file}.txt
    # OR: data d/daily/us/{exchange}/{asset_class}/{subdir}/{file}.txt
    # Collect all matches across ALL exchanges first, then prioritize exact matches
    exact_matches = []
    prefix_matches = []
    
    for exchange_dir in base_dir.iterdir():
        if not exchange_dir.is_dir() or exchange_dir.name.startswith('.'):
            continue
        
        # Check if subdirectories are directly under exchange (structure 1)
        # or if there's an asset_class level (structure 2)
        subdirs_to_search = []
        
        for item in exchange_dir.iterdir():
            if not item.is_dir() or item.name.startswith('.'):
                continue
            
            # Check if this is a subdir (numeric name like "1", "2") or asset_class
            # If it's numeric, it's a subdir. Otherwise, it might be an asset_class.
            if item.name.isdigit():
                # Direct subdir structure
                subdirs_to_search.append(item)
            else:
                # Asset class structure - search inside
                for subdir in item.iterdir():
                    if subdir.is_dir() and not subdir.name.startswith('.'):
                        subdirs_to_search.append(subdir)
        
        # Search all subdirectories for files matching symbol
        for subdir in subdirs_to_search:
            # Search for files matching symbol
            for file_path in subdir.iterdir():
                if not file_path.is_file() or file_path.name.startswith('.'):
                    continue
                
                # Check if it's a data file (.txt or .csv)
                if file_path.suffix.lower() not in ['.txt', '.csv']:
                    continue
                
                # Check if file matches symbol (case-insensitive)
                # Format: {SYMBOL}.us.txt (e.g., "aaau.us.txt" or "AAAU.US.TXT")
                file_name_upper = file_path.stem.upper()  # Filename without extension (e.g., "AAAU.US")
                
                # Extract the symbol part (before first dot)
                symbol_part_upper = file_name_upper.split('.')[0] if '.' in file_name_upper else file_name_upper
                
                # Prioritize exact matches to avoid false matches (e.g., "SPY" matching "DSPY" or "SPYU")
                if symbol_part_upper == symbol_upper:
                    exact_matches.append(file_path)
                elif (symbol_part_upper.startswith(symbol_upper) and 
                      len(symbol_part_upper) > len(symbol_upper)):
                    # Store prefix match as fallback only if file symbol is longer
                    # (e.g., "SPYQ" for "SPY", but NOT "SPYU" for "SPY" since they're different symbols)
                    # This prevents false matches like "SPY" matching "SPYU" or "DSPY"
                    prefix_matches.append(file_path)
    
    # Return exact match if found (preferred), otherwise return first prefix match
    if exact_matches:
        return exact_matches[0]  # Return first exact match found
    elif prefix_matches:
        return prefix_matches[0]  # Fallback to prefix match if no exact match
    
    return None


def load_stooq_file(file_path: Path, symbol: str = None) -> List[Dict]:
    """
    Load candles from a Stooq format file (.txt).
    
    Stooq format:
    <TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>
    SPY.US,D,20230103,000000,380.50,381.20,379.80,380.90,50000000,0
    
    Args:
        file_path: Path to Stooq .txt file
        symbol: Trading symbol (for error messages)
    
    Returns:
        List of candle dicts with keys: timestamp, open, high, low, close, volume
    """
    symbol_display = symbol if symbol else "unknown"
    print(f"[STOOQ LOADER] Loading candles from: {file_path}")
    print(f"[STOOQ LOADER]   Symbol: {symbol_display}")
    
    candles = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Stooq format columns
            row_count = 0
            for row in reader:
                row_count += 1
                
                try:
                    # Parse date (YYYYMMDD format)
                    date_str = row.get('<DATE>', '').strip()
                    if not date_str:
                        continue
                    
                    # Parse date: YYYYMMDD
                    date_parsed = datetime.strptime(date_str, '%Y%m%d')
                    
                    # Parse time (HHMMSS format, usually 000000 for daily)
                    time_str = row.get('<TIME>', '000000').strip()
                    if time_str and time_str != '000000':
                        # Parse time: HHMMSS
                        hour = int(time_str[0:2])
                        minute = int(time_str[2:4])
                        second = int(time_str[4:6])
                        date_parsed = date_parsed.replace(hour=hour, minute=minute, second=second)
                    
                    # Convert to UTC timezone-aware datetime
                    if date_parsed.hour == 0 and date_parsed.minute == 0 and date_parsed.second == 0:
                        # Daily data - normalize to midnight UTC
                        timestamp = date_parsed.replace(
                            hour=0, minute=0, second=0, microsecond=0,
                            tzinfo=timezone.utc
                        )
                    else:
                        # Intraday data - preserve time component, assume UTC
                        timestamp = date_parsed.replace(tzinfo=timezone.utc)
                    timestamp = ensure_utc_datetime(timestamp, f"Stooq row {row_count} for {symbol_display}")
                    
                    # Parse OHLCV values
                    open_price = float(row.get('<OPEN>', 0))
                    high_price = float(row.get('<HIGH>', 0))
                    low_price = float(row.get('<LOW>', 0))
                    close_price = float(row.get('<CLOSE>', 0))
                    volume = int(float(row.get('<VOL>', 0)))
                    
                    # Skip rows with missing/invalid values
                    import math
                    if any(math.isnan(val) for val in [open_price, high_price, low_price, close_price]) or math.isnan(float(volume)):
                        print(f"[STOOQ LOADER]   Skipping row {row_count}: missing values")
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
                    
                except (ValueError, KeyError) as e:
                    print(f"[STOOQ LOADER]   Warning: Skipping row {row_count}: {e}")
                    continue
        
        print(f"[STOOQ LOADER]   Loaded {len(candles)} valid candles from {row_count} rows")
        
    except FileNotFoundError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to read Stooq file {file_path}: {str(e)}")
    
    # Validate candles
    if not candles:
        raise ValueError(f"No valid candles found in Stooq file: {file_path}")
    
    # Note: No minimum candle requirement for Stooq files
    # This allows loading small test datasets without validation errors
    
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
                f"Candles not strictly ordered in Stooq file. "
                f"Row {i}: {prev_timestamp} >= {curr_timestamp}"
            )
    
    print(f"[STOOQ LOADER] SUCCESS: Loaded {len(candles)} candles for {symbol_display}")
    if candles:
        print(f"[STOOQ LOADER]   Date range: {candles[0]['timestamp'].date()} to {candles[-1]['timestamp'].date()}")
    
    return candles


def load_daily_candles(symbol: str) -> List[Dict]:
    """
    Load daily candles for a symbol by searching ONLY under data d/daily/us/.
    
    Searches ONLY: data d/daily/us/{exchange}/{asset_class}/{subdir}/{symbol}*.txt
    Does NOT search: raw_data/, data 5/, or any other directories
    
    Args:
        symbol: Trading symbol (e.g., "SPY", "AAPL")
    
    Returns:
        List of candle dicts with keys: timestamp, open, high, low, close, volume
    
    Raises:
        FileNotFoundError: If no daily data file found under data d/daily/us/
        ValueError: If data format is invalid
    """
    # Find the data file (ONLY searches under data d/daily/us/)
    file_path = find_data_file(symbol, data_type="daily")
    
    if file_path is None:
        raise FileNotFoundError(
            f"Daily data not found under data d/daily/us/ for symbol {symbol}. "
            f"Please ensure the file exists in: data d/daily/us/{{exchange}}/{{asset_class}}/{{subdir}}/{symbol}*.txt"
        )
    
    # Debug log: show which file is being used
    print(f"[STOOQ LOADER] Using DAILY file: {file_path}")
    
    # Load the file (handle both Stooq .txt and CSV formats)
    if file_path.suffix.lower() == '.txt':
        candles = load_stooq_file(file_path, symbol)
    else:
        # Use existing CSV loader
        from market_data.csv_loader import load_csv_candles
        candles = load_csv_candles(str(file_path), symbol)
    
    return candles


def load_intraday_candles(symbol: str, interval: str = "5min") -> List[Dict]:
    """
    Load intraday candles for a symbol by searching the directory structure.
    
    Searches: data 5/5 min/us/{exchange}/{asset_class}/{subdir}/{symbol}*.txt
    
    Args:
        symbol: Trading symbol (e.g., "SPY", "AAPL")
        interval: Data interval (currently only "5min" supported)
    
    Returns:
        List of candle dicts with keys: timestamp, open, high, low, close, volume
    
    Raises:
        FileNotFoundError: If no data file found for the symbol
        ValueError: If data format is invalid or interval not supported
    """
    if interval != "5min":
        raise ValueError(f"Only '5min' interval is currently supported. Got: {interval}")
    
    # Find the data file
    file_path = find_data_file(symbol, data_type="intraday")
    
    if file_path is None:
        raise FileNotFoundError(
            f"No intraday data file found for symbol {symbol} (interval={interval}). "
            f"Searched in: data 5/5 min/us/"
        )
    
    # Load the file (handle both Stooq .txt and CSV formats)
    if file_path.suffix.lower() == '.txt':
        candles = load_stooq_file(file_path, symbol)
    else:
        # Use existing CSV loader
        from market_data.csv_loader import load_csv_candles
        candles = load_csv_candles(str(file_path), symbol)
    
    return candles
