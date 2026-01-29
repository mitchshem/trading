"""
Utility functions for time handling and validation.
Enforces canonical UTC timezone-aware datetime representation.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, Union
import math


def ensure_utc_datetime(dt: datetime, context: str = "") -> datetime:
    """
    Ensure a datetime is UTC timezone-aware.
    
    FIX 1: CANONICAL TIME HANDLING
    Fail fast if datetime is naive or not UTC.
    
    Args:
        dt: Datetime object to validate
        context: Context string for error messages
    
    Returns:
        UTC timezone-aware datetime
    
    Raises:
        ValueError: If datetime is naive or not UTC
    """
    if dt is None:
        raise ValueError(f"None datetime provided in context: {context}")
    
    if dt.tzinfo is None:
        raise ValueError(
            f"Naive datetime detected in context: {context}. "
            f"All timestamps must be timezone-aware UTC. Got: {dt}"
        )
    
    if dt.tzinfo != timezone.utc:
        # Convert to UTC if not already
        dt = dt.astimezone(timezone.utc)
    
    return dt


def unix_to_utc_datetime(unix_timestamp: int) -> datetime:
    """
    Convert Unix timestamp (seconds) to UTC timezone-aware datetime.
    
    FIX 1: CANONICAL TIME HANDLING
    Conversion happens at API boundary (candles use Unix timestamps for transport).
    
    Args:
        unix_timestamp: Unix timestamp in seconds
    
    Returns:
        UTC timezone-aware datetime
    """
    return datetime.fromtimestamp(unix_timestamp, tz=timezone.utc)


def utc_datetime_to_unix(dt: datetime) -> int:
    """
    Convert UTC timezone-aware datetime to Unix timestamp (seconds).
    
    FIX 1: CANONICAL TIME HANDLING
    Conversion happens at API boundary only.
    
    Args:
        dt: UTC timezone-aware datetime
    
    Returns:
        Unix timestamp in seconds
    """
    ensure_utc_datetime(dt, "utc_datetime_to_unix")
    return int(dt.timestamp())


def fmt(value: Union[int, float, None, str], decimals: int = 2) -> str:
    """
    Safely format a numeric value for display.
    
    Prevents crashes when values are None, strings, or missing.
    
    Args:
        value: Numeric value to format (int, float, None, or str)
        decimals: Number of decimal places (default: 2)
    
    Returns:
        Formatted string or "N/A" if value cannot be formatted
    """
    if value is None:
        return "N/A"
    
    if isinstance(value, str):
        return value
    
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return "N/A"
        return f"{value:.{decimals}f}"
    
    return "N/A"


def fmt_pct(value: Union[int, float, None, str], decimals: int = 2) -> str:
    """
    Safely format a percentage value for display.
    
    Prevents crashes when values are None, strings, or missing.
    
    Args:
        value: Numeric value to format as percentage (int, float, None, or str)
        decimals: Number of decimal places (default: 2)
    
    Returns:
        Formatted percentage string or "N/A" if value cannot be formatted
    """
    if value is None:
        return "N/A"
    
    if isinstance(value, str):
        return value
    
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return "N/A"
        return f"{value:.{decimals}f}%"
    
    return "N/A"


def fmt_currency(value: Union[int, float, None, str], decimals: int = 2) -> str:
    """
    Safely format a currency value for display.
    
    Prevents crashes when values are None, strings, or missing.
    
    Args:
        value: Numeric value to format as currency (int, float, None, or str)
        decimals: Number of decimal places (default: 2)
    
    Returns:
        Formatted currency string with $ prefix or "N/A" if value cannot be formatted
    """
    if value is None:
        return "N/A"
    
    if isinstance(value, str):
        return value
    
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return "N/A"
        return f"${value:,.{decimals}f}"
    
    return "N/A"


def calculate_candle_timestamps(base_timestamp: datetime, is_daily: bool = True) -> tuple[datetime, datetime]:
    """
    Calculate open_time and close_time for a candle.
    
    For daily candles:
    - open_time: Market open (09:30 ET = 13:30 UTC)
    - close_time: Market close (16:00 ET = 20:00 UTC)
    
    For intraday candles:
    - open_time: Base timestamp (candle start)
    - close_time: Base timestamp + interval duration
    
    Args:
        base_timestamp: Base timestamp (date for daily, start time for intraday)
        is_daily: True for daily candles, False for intraday
    
    Returns:
        Tuple of (open_time, close_time) as UTC timezone-aware datetimes
    """
    base_timestamp = ensure_utc_datetime(base_timestamp, "calculate_candle_timestamps")
    
    if is_daily:
        # Daily candle: use market hours
        # Market open: 09:30 ET = 13:30 UTC
        # Market close: 16:00 ET = 20:00 UTC
        open_time = base_timestamp.replace(hour=13, minute=30, second=0, microsecond=0)
        close_time = base_timestamp.replace(hour=20, minute=0, second=0, microsecond=0)
    else:
        # Intraday candle: assume 5-minute interval
        # open_time = base timestamp
        # close_time = base timestamp + 5 minutes
        open_time = base_timestamp.replace(second=0, microsecond=0)
        close_time = open_time + timedelta(minutes=5)
    
    return (open_time, close_time)
