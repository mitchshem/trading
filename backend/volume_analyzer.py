"""
Volume analysis module.
Detects volume spikes and calculates volume-based indicators.
"""

from typing import List, Dict, Optional
import statistics


def detect_volume_spike(
    candles: List[Dict],
    volume_period: int = 20,
    volume_multiplier: float = 1.5
) -> Optional[Dict]:
    """
    Detect if current candle has a volume spike.
    
    Volume spike condition:
    - current_volume >= volume_multiplier * rolling_avg_volume
    
    Args:
        candles: List of candle dicts with keys: time, open, high, low, close, volume
        volume_period: Period for rolling average (default: 20)
        volume_multiplier: Multiplier threshold (default: 1.5)
    
    Returns:
        Dict with keys:
        - "is_spike": bool
        - "current_volume": int
        - "average_volume": float
        - "volume_ratio": float (current / average)
        None if insufficient data
    """
    if len(candles) < volume_period + 1:
        return None
    
    current_candle = candles[-1]
    current_volume = current_candle.get("volume", 0)
    
    # Get previous N candles for rolling average (exclude current)
    previous_candles = candles[-(volume_period + 1):-1]
    volumes = [c.get("volume", 0) for c in previous_candles]
    
    if not volumes:
        return None
    
    average_volume = statistics.mean(volumes)
    
    if average_volume <= 0:
        return None
    
    volume_ratio = current_volume / average_volume
    is_spike = volume_ratio >= volume_multiplier
    
    return {
        "is_spike": is_spike,
        "current_volume": current_volume,
        "average_volume": average_volume,
        "volume_ratio": volume_ratio,
        "volume_multiplier": volume_multiplier
    }


def get_volume_percentile(
    candle: Dict,
    candles: List[Dict],
    volume_period: int = 20
) -> Optional[float]:
    """
    Calculate what percentile the current candle's volume is relative to recent history.
    
    Args:
        candle: Current candle dict
        candles: List of historical candles
        volume_period: Period to look back (default: 20)
    
    Returns:
        Percentile (0-100) or None if insufficient data
    """
    if len(candles) < volume_period + 1:
        return None
    
    current_volume = candle.get("volume", 0)
    previous_candles = candles[-(volume_period + 1):-1]
    volumes = [c.get("volume", 0) for c in previous_candles]
    
    if not volumes:
        return None
    
    # Count how many previous volumes are less than current
    volumes_below = sum(1 for v in volumes if v < current_volume)
    percentile = (volumes_below / len(volumes)) * 100.0
    
    return percentile
