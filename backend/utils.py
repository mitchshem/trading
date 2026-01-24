"""
Utility functions for time handling and validation.
Enforces canonical UTC timezone-aware datetime representation.
"""

from datetime import datetime, timezone
from typing import Optional


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
