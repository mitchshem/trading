"""
Strategy module for generating trading signals.
Deterministic, rule-based strategies only.
"""

from typing import List, Dict, Optional, Literal
from indicators import ema, atr


SignalType = Literal["BUY", "EXIT", "HOLD"]


class PositionState:
    """Tracks current position state for a symbol."""
    def __init__(self):
        self.has_position: bool = False
        self.entry_price: Optional[float] = None
        self.entry_time: Optional[int] = None


def ema_trend_v1(
    candles: List[Dict],
    ema20_values: List[Optional[float]],
    ema50_values: List[Optional[float]],
    atr14_values: List[Optional[float]],
    position_state: PositionState
) -> Dict[str, any]:
    """
    EMA Trend Following Strategy v1.
    
    Rules:
    - ENTRY (BUY): EMA(20) crosses above EMA(50) AND close > EMA(50) AND no position
    - EXIT: close < EMA(50) OR price <= entry_price - (2 * ATR)
    
    Args:
        candles: List of candle dicts with keys: time, open, high, low, close, volume
        ema20_values: EMA(20) values (aligned with candles)
        ema50_values: EMA(50) values (aligned with candles)
        atr14_values: ATR(14) values (aligned with candles)
        position_state: Current position state for this symbol
    
    Returns:
        Dict with keys: signal ("BUY" | "EXIT" | "HOLD"), reason (str)
    """
    if len(candles) < 2:
        return {"signal": "HOLD", "reason": "Insufficient data"}
    
    # Get current (last) candle and previous candle
    current_candle = candles[-1]
    prev_candle = candles[-2]
    current_close = current_candle["close"]
    current_time = current_candle["time"]
    
    # Get indicator values for current and previous candles
    current_idx = len(candles) - 1
    prev_idx = len(candles) - 2
    
    # Check if we have enough data for indicators
    if (ema20_values[current_idx] is None or 
        ema50_values[current_idx] is None or 
        ema20_values[prev_idx] is None or 
        ema50_values[prev_idx] is None):
        return {"signal": "HOLD", "reason": "Indicators not ready"}
    
    current_ema20 = ema20_values[current_idx]
    current_ema50 = ema50_values[current_idx]
    prev_ema20 = ema20_values[prev_idx]
    prev_ema50 = ema50_values[prev_idx]
    
    # EXIT conditions (check first if we have a position)
    if position_state.has_position:
        # Exit condition 1: Close below EMA(50)
        if current_close < current_ema50:
            return {
                "signal": "EXIT",
                "reason": f"Close {current_close:.2f} below EMA(50) {current_ema50:.2f}"
            }
        
        # Exit condition 2: Stop loss (price <= entry_price - 2*ATR)
        if atr14_values[current_idx] is not None:
            stop_loss_price = position_state.entry_price - (2 * atr14_values[current_idx])
            if current_close <= stop_loss_price:
                return {
                    "signal": "EXIT",
                    "reason": f"Stop loss triggered: {current_close:.2f} <= {stop_loss_price:.2f}"
                }
    
    # ENTRY conditions (only if no position)
    if not position_state.has_position:
        # Check for EMA crossover: EMA(20) crosses above EMA(50)
        ema_cross_above = (prev_ema20 <= prev_ema50 and current_ema20 > current_ema50)
        
        # Check if close is above EMA(50)
        close_above_ema50 = current_close > current_ema50
        
        if ema_cross_above and close_above_ema50:
            return {
                "signal": "BUY",
                "reason": f"EMA(20) crossed above EMA(50), close {current_close:.2f} > EMA(50) {current_ema50:.2f}"
            }
    
    return {"signal": "HOLD", "reason": "No signal conditions met"}
