"""
Consolidation detection module.
Detects periods of price consolidation (low volatility, sideways movement).
"""

from typing import List, Dict, Optional
from indicators import atr


def detect_consolidation(
    candles: List[Dict],
    window_size: int = 20,
    range_pct: float = 0.02,
    atr_period: int = 14
) -> Optional[Dict]:
    """
    Detect if the last N candles represent a consolidation period.
    
    Consolidation conditions:
    - High-low range over last N candles <= range_pct * average price
    - ATR(14) flat or declining (current <= previous)
    
    Args:
        candles: List of candle dicts with keys: time, open, high, low, close, volume
        window_size: Number of candles to analyze (default: 20)
        range_pct: Maximum range as percentage of average price (default: 0.02 = 2%)
        atr_period: Period for ATR calculation (default: 14)
    
    Returns:
        Dict with keys:
        - "is_consolidation": bool
        - "consolidation_range": float (high - low over window)
        - "average_price": float
        - "range_pct": float (actual range as % of average)
        - "highest_high": float (highest high in consolidation window)
        - "lowest_low": float (lowest low in consolidation window)
        - "atr_current": float (current ATR value)
        - "atr_previous": float (previous ATR value)
        None if insufficient data
    """
    if len(candles) < max(window_size, atr_period + 1):
        return None
    
    # Get last window_size candles
    window_candles = candles[-window_size:]
    
    # Calculate price range over consolidation window
    highs = [c["high"] for c in window_candles]
    lows = [c["low"] for c in window_candles]
    closes = [c["close"] for c in window_candles]
    
    highest_high = max(highs)
    lowest_low = min(lows)
    consolidation_range = highest_high - lowest_low
    
    # Calculate average price (using closes)
    average_price = sum(closes) / len(closes)
    
    # Check range condition
    range_pct_actual = (consolidation_range / average_price) if average_price > 0 else float('inf')
    range_condition = range_pct_actual <= range_pct
    
    # Calculate ATR
    all_highs = [c["high"] for c in candles]
    all_lows = [c["low"] for c in candles]
    all_closes = [c["close"] for c in candles]
    
    atr_values = atr(all_highs, all_lows, all_closes, atr_period)
    
    if len(atr_values) < 2:
        return None
    
    current_atr = atr_values[-1]
    previous_atr = atr_values[-2]
    
    if current_atr is None or previous_atr is None:
        return None
    
    # Check ATR condition (flat or declining)
    atr_condition = current_atr <= previous_atr
    
    # Consolidation detected if both conditions met
    is_consolidation = range_condition and atr_condition
    
    return {
        "is_consolidation": is_consolidation,
        "consolidation_range": consolidation_range,
        "average_price": average_price,
        "range_pct": range_pct_actual,
        "highest_high": highest_high,
        "lowest_low": lowest_low,
        "atr_current": current_atr,
        "atr_previous": previous_atr,
        "window_size": window_size
    }
