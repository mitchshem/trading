"""
Technical indicators module.
Pure Python implementations (no TA-Lib dependency).
"""

from typing import List, Optional


def ema(prices: List[float], period: int) -> List[float]:
    """
    Calculate Exponential Moving Average (EMA).
    
    Args:
        prices: List of price values (typically closes)
        period: EMA period (e.g., 20, 50)
    
    Returns:
        List of EMA values (same length as prices, with None for insufficient data)
    """
    if len(prices) < period:
        return [None] * len(prices)
    
    # Calculate multiplier
    multiplier = 2.0 / (period + 1)
    
    # Start with SMA for first value
    ema_values = [None] * (period - 1)
    sma = sum(prices[:period]) / period
    ema_values.append(sma)
    
    # Calculate EMA for remaining values
    for i in range(period, len(prices)):
        ema_value = (prices[i] - ema_values[i - 1]) * multiplier + ema_values[i - 1]
        ema_values.append(ema_value)
    
    return ema_values


def rsi(prices: List[float], period: int = 14) -> List[Optional[float]]:
    """
    Calculate Relative Strength Index (RSI).

    Uses the smoothed (Wilder) method: initial average is SMA of first
    ``period`` gains/losses, then exponentially smoothed with alpha = 1/period.

    Args:
        prices: List of price values (typically closes).
        period: RSI period (default 14).

    Returns:
        List of RSI values (same length as prices, None where insufficient data).
    """
    n = len(prices)
    if n < period + 1:
        return [None] * n

    results: List[Optional[float]] = [None] * n

    # Calculate price changes
    gains = []
    losses = []
    for i in range(1, n):
        delta = prices[i] - prices[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))

    # Initial average gain / loss (SMA over first `period` changes)
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    if avg_loss == 0:
        results[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        results[period] = 100.0 - 100.0 / (1.0 + rs)

    # Smoothed RSI for remaining values
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            results[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            results[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    return results


def atr(highs: List[float], lows: List[float], closes: List[float], period: int) -> List[float]:
    """
    Calculate Average True Range (ATR).
    
    Args:
        highs: List of high prices
        lows: List of low prices
        closes: List of close prices
        period: ATR period (typically 14)
    
    Returns:
        List of ATR values (same length as prices, with None for insufficient data)
    """
    if len(highs) != len(lows) or len(lows) != len(closes):
        raise ValueError("highs, lows, and closes must have the same length")
    
    if len(highs) < period + 1:  # Need at least period+1 candles for ATR
        return [None] * len(highs)
    
    # Calculate True Range (TR) for each candle
    tr_values = []
    for i in range(len(highs)):
        if i == 0:
            # First candle: TR = High - Low
            tr = highs[i] - lows[i]
        else:
            # TR = max(High - Low, abs(High - PrevClose), abs(Low - PrevClose))
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1])
            )
        tr_values.append(tr)
    
    # Calculate ATR using EMA of TR
    atr_values = [None] * period
    # Initial ATR is SMA of first period TR values
    initial_atr = sum(tr_values[1:period + 1]) / period  # Skip first TR (no prev close)
    atr_values.append(initial_atr)
    
    # Calculate remaining ATR values using EMA
    multiplier = 2.0 / (period + 1)
    for i in range(period + 1, len(tr_values)):
        atr_value = (tr_values[i] - atr_values[i - 1]) * multiplier + atr_values[i - 1]
        atr_values.append(atr_value)
    
    return atr_values
