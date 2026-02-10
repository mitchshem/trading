"""
Dual Momentum Strategy v1 — Absolute + Relative momentum filter.

Thesis: Hold the asset only when both absolute momentum (positive return
over lookback period) and relative price strength are positive. This is a
systematic, rules-based approach to being in the market only when the trend
supports it.

Since this is single-asset for now (no benchmark comparison), we simplify:
  - Absolute momentum: close > close[lookback] (positive N-day return)
  - Trend confirmation: close > EMA(lookback) (price above long-term average)
  - Exit when either condition fails

Rules:
  ENTRY (BUY):
    - close > close[lookback]  (positive absolute momentum)
    - close > EMA(lookback)    (above long-term moving average)
    - No open position
  EXIT:
    - close < close[lookback]  OR  close < EMA(lookback)
  HOLD:
    - Neither condition met

Stop-loss and risk management handled externally by PaperBroker.
"""

from typing import Dict, List, Optional
from indicators import ema, atr
from strategy import PositionState


def dual_momentum_v1(
    candles: List[Dict],
    position_state: PositionState,
    params: Optional[Dict] = None,
) -> Dict:
    """
    Evaluate dual momentum strategy on current candle history.

    Args:
        candles: Candle history up to and including the current candle.
        position_state: Current position state for this symbol.
        params: Strategy parameters (optional overrides).

    Returns:
        Dict with keys: signal, reason, stop_distance, current_atr.
    """
    params = params or {}
    lookback = params.get("lookback", 126)  # ~6 months of trading days
    stop_atr_mult = params.get("stop_loss_atr_multiplier", 2.0)

    min_candles = lookback + 1
    if len(candles) < min_candles:
        return {"signal": "HOLD", "reason": "Insufficient data", "stop_distance": None, "current_atr": None}

    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]

    current_close = closes[-1]
    lookback_close = closes[-(lookback + 1)]

    ema_values = ema(closes, lookback)
    atr_values = atr(highs, lows, closes, 14)

    current_ema = ema_values[-1]
    current_atr = atr_values[-1] if atr_values and atr_values[-1] is not None else None

    if current_ema is None:
        return {"signal": "HOLD", "reason": "EMA not ready", "stop_distance": None, "current_atr": current_atr}

    stop_distance = (stop_atr_mult * current_atr) if current_atr else None

    abs_momentum = current_close > lookback_close
    trend_above = current_close > current_ema
    ret_pct = ((current_close - lookback_close) / lookback_close) * 100

    # --- EXIT ---
    if position_state.has_position:
        if not abs_momentum or not trend_above:
            reasons = []
            if not abs_momentum:
                reasons.append(f"{lookback}-day return negative ({ret_pct:.1f}%)")
            if not trend_above:
                reasons.append(f"close {current_close:.2f} < EMA({lookback}) {current_ema:.2f}")
            return {
                "signal": "EXIT",
                "reason": f"Momentum lost: {'; '.join(reasons)}",
                "stop_distance": stop_distance,
                "current_atr": current_atr,
            }
        return {
            "signal": "HOLD",
            "reason": f"Momentum intact: {lookback}-day return {ret_pct:.1f}%, above EMA({lookback})",
            "stop_distance": stop_distance,
            "current_atr": current_atr,
        }

    # --- ENTRY ---
    if abs_momentum and trend_above:
        return {
            "signal": "BUY",
            "reason": f"Dual momentum positive: {lookback}-day return {ret_pct:.1f}%, close > EMA({lookback})",
            "stop_distance": stop_distance,
            "current_atr": current_atr,
        }

    return {
        "signal": "HOLD",
        "reason": f"Momentum not confirmed: {lookback}-day return {ret_pct:.1f}%, {'above' if trend_above else 'below'} EMA({lookback})",
        "stop_distance": stop_distance,
        "current_atr": current_atr,
    }
