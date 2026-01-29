"""
Market regime classification module.
Classifies each candle into TREND_UP, TREND_DOWN, or CHOP using simple, explainable rules.
"""

from typing import List, Dict, Optional
from indicators import ema


def classify_regime(
    candles: List[Dict],
    current_index: int,
    ma_period: int = 200
) -> Optional[str]:
    """
    Classify market regime for a candle using simple, explainable rules.
    
    Rules:
    - TREND_UP: Price > MA(200) AND MA(200) is rising (current > previous)
    - TREND_DOWN: Price < MA(200) AND MA(200) is falling (current < previous)
    - CHOP: Otherwise (sideways/choppy market)
    
    Args:
        candles: List of candle dicts with keys: time, open, high, low, close, volume
        current_index: Index of the current candle (0-based)
        ma_period: Period for the moving average (default: 200)
    
    Returns:
        "TREND_UP", "TREND_DOWN", "CHOP", or None if insufficient data
    """
    if current_index < ma_period:
        # Need at least ma_period candles to compute MA
        return None
    
    # Extract closes up to and including current candle
    closes = [c["close"] for c in candles[:current_index + 1]]
    
    # Calculate EMA(200) - using EMA for consistency with existing indicators
    ma_values = ema(closes, ma_period)
    
    # Get current and previous MA values
    current_ma = ma_values[current_index]
    if current_ma is None:
        return None
    
    # Need previous MA value to determine slope
    if current_index < ma_period:
        return None
    
    prev_ma = ma_values[current_index - 1]
    if prev_ma is None:
        return None
    
    # Get current price
    current_price = candles[current_index]["close"]
    
    # Classify regime
    price_above_ma = current_price > current_ma
    ma_rising = current_ma > prev_ma
    ma_falling = current_ma < prev_ma
    
    if price_above_ma and ma_rising:
        return "TREND_UP"
    elif not price_above_ma and ma_falling:
        return "TREND_DOWN"
    else:
        # CHOP: price above MA but MA falling, or price below MA but MA rising
        return "CHOP"


def get_regime_at_timestamp(
    candles: List[Dict],
    timestamp,
    ma_period: int = 200
) -> Optional[str]:
    """
    Get regime classification for a specific timestamp.
    
    Args:
        candles: List of candle dicts ordered by time
        timestamp: Unix timestamp or datetime to find regime for
        ma_period: Period for the moving average (default: 200)
    
    Returns:
        "TREND_UP", "TREND_DOWN", "CHOP", or None if not found
    """
    # Convert timestamp to comparable format
    if hasattr(timestamp, 'timestamp'):
        # datetime object
        target_ts = int(timestamp.timestamp())
    else:
        # Assume Unix timestamp
        target_ts = int(timestamp)
    
    # Find candle index matching the timestamp
    # Candles in replay format use Unix timestamps (integers)
    for i, candle in enumerate(candles):
        candle_time = candle.get("time")
        if hasattr(candle_time, 'timestamp'):
            # datetime object
            candle_ts = int(candle_time.timestamp())
        elif isinstance(candle_time, (int, float)):
            # Already a Unix timestamp
            candle_ts = int(candle_time)
        else:
            continue  # Skip invalid candle time
        
        if candle_ts == target_ts:
            return classify_regime(candles, i, ma_period)
        elif candle_ts > target_ts:
            # Past the target timestamp, use previous candle
            if i > 0:
                return classify_regime(candles, i - 1, ma_period)
            break
    
    # If exact match not found, find closest candle
    closest_index = None
    min_diff = float('inf')
    
    for i, candle in enumerate(candles):
        candle_time = candle.get("time")
        if hasattr(candle_time, 'timestamp'):
            # datetime object
            candle_ts = int(candle_time.timestamp())
        elif isinstance(candle_time, (int, float)):
            # Already a Unix timestamp
            candle_ts = int(candle_time)
        else:
            continue  # Skip invalid candle time
        
        diff = abs(candle_ts - target_ts)
        if diff < min_diff:
            min_diff = diff
            closest_index = i
    
    if closest_index is not None:
        return classify_regime(candles, closest_index, ma_period)
    
    return None
