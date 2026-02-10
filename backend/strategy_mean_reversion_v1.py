"""
Mean Reversion Strategy v1 — RSI-based with optional regime gate.

Thesis: In an uptrend, buy when RSI signals oversold conditions and exit
when RSI normalizes. This is a fundamentally different approach from
trend-following, providing diversification of strategy thesis.

Rules:
  ENTRY (BUY):
    - RSI(period) < rsi_oversold (default 30)
    - [optional] Current regime is TREND_UP (price > EMA(200), EMA(200) rising)
    - No open position
  EXIT:
    - RSI(period) > rsi_exit (default 50)  — price has "reverted to mean"
  HOLD:
    - Neither entry nor exit conditions met

Stop-loss and risk management are handled externally by PaperBroker.
"""

from typing import Dict, List, Optional
from indicators import rsi, atr, ema
from strategy import PositionState
from utils import fmt


def mean_reversion_v1(
    candles: List[Dict],
    position_state: PositionState,
    params: Optional[Dict] = None,
) -> Dict:
    """
    Evaluate mean reversion strategy on current candle history.

    Args:
        candles: Candle history up to and including the current candle.
        position_state: Current position state for this symbol.
        params: Strategy parameters (optional overrides).

    Returns:
        Dict with keys: signal ("BUY"|"EXIT"|"HOLD"), reason (str),
        stop_distance (float|None), current_atr (float|None).
    """
    params = params or {}
    rsi_period = params.get("rsi_period", 14)
    rsi_oversold = params.get("rsi_oversold", 30)
    rsi_exit = params.get("rsi_exit", 50)
    stop_atr_mult = params.get("stop_loss_atr_multiplier", 2.0)
    require_trend_up = params.get("require_trend_up", True)

    min_candles = max(rsi_period + 1, 50)  # need enough for EMA(50) and RSI
    if len(candles) < min_candles:
        return {"signal": "HOLD", "reason": "Insufficient data", "stop_distance": None, "current_atr": None}

    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]

    rsi_values = rsi(closes, rsi_period)
    atr_values = atr(highs, lows, closes, 14)

    current_rsi = rsi_values[-1]
    current_atr = atr_values[-1] if atr_values and atr_values[-1] is not None else None

    if current_rsi is None:
        return {"signal": "HOLD", "reason": "RSI not ready", "stop_distance": None, "current_atr": current_atr}

    stop_distance = (stop_atr_mult * current_atr) if current_atr else None

    # --- EXIT ---
    if position_state.has_position:
        if current_rsi > rsi_exit:
            return {
                "signal": "EXIT",
                "reason": f"RSI {current_rsi:.1f} > exit threshold {rsi_exit}",
                "stop_distance": stop_distance,
                "current_atr": current_atr,
            }
        return {
            "signal": "HOLD",
            "reason": f"RSI {current_rsi:.1f} — waiting for exit at {rsi_exit}",
            "stop_distance": stop_distance,
            "current_atr": current_atr,
        }

    # --- ENTRY ---
    if current_rsi < rsi_oversold:
        # Optional regime gate
        if require_trend_up:
            from regime_classifier import classify_regime
            regime = classify_regime(candles, len(candles) - 1)
            if regime is not None and regime != "TREND_UP":
                return {
                    "signal": "HOLD",
                    "reason": f"RSI {current_rsi:.1f} oversold but regime is {regime}, not TREND_UP",
                    "stop_distance": stop_distance,
                    "current_atr": current_atr,
                }

        return {
            "signal": "BUY",
            "reason": f"RSI {current_rsi:.1f} < oversold threshold {rsi_oversold}",
            "stop_distance": stop_distance,
            "current_atr": current_atr,
        }

    return {
        "signal": "HOLD",
        "reason": f"RSI {current_rsi:.1f} — no signal",
        "stop_distance": stop_distance,
        "current_atr": current_atr,
    }
