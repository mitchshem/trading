"""
Momentum Breakout with Volume Confirmation Strategy.

Strategy 1: Exploits consolidation breakouts with volume confirmation.
Targets continuation after consolidation with asymmetric profit-taking.

Core Logic:
- Detects consolidation periods (low volatility, sideways movement)
- Waits for breakout above consolidation high with volume confirmation
- Enters on next candle open
- Asymmetric exits: Target 1 (2.0x ATR, 50%), Target 2 (3.5x ATR, 50%)
- Stop-loss: 1.0x ATR (tight)
- Trailing stop: Moves to entry + 1.0x ATR after Target 1
- Time stop: Exit if Target 1 not hit within 10 candles

Regime Gate:
- Only TREND_UP (price > EMA(200) and EMA(200) rising)
"""

from typing import List, Dict, Optional, Literal
from indicators import ema, atr
from regime_classifier import classify_regime
from strategy import PositionState, SignalType
from consolidation_detector import detect_consolidation
from volume_analyzer import detect_volume_spike
from utils import fmt


class BreakoutStrategyState:
    """
    Extended position state for breakout strategy.
    
    Tracks:
    - Entry ATR (for profit target calculations)
    - Entry candle index (for accurate time tracking)
    - Target hit status (for trailing stop logic)
    """
    def __init__(self):
        self.has_position: bool = False
        self.entry_price: Optional[float] = None
        self.entry_time: Optional[int] = None
        self.entry_atr: Optional[float] = None  # ATR at entry (for exit calculations)
        self.entry_index: Optional[int] = None  # Candle index at entry (for time tracking)
        self.target1_hit: bool = False  # Track if Target 1 has been hit
        self.target2_hit: bool = False  # Track if Target 2 has been hit
    
    def reset(self):
        """Reset state for new position."""
        self.has_position = False
        self.entry_price = None
        self.entry_time = None
        self.entry_atr = None
        self.entry_index = None
        self.target1_hit = False
        self.target2_hit = False
    
    def update_from_position(self, position_state: PositionState, entry_index: int):
        """Update state from PositionState after entry."""
        self.has_position = position_state.has_position
        self.entry_price = position_state.entry_price
        self.entry_time = position_state.entry_time
        self.entry_index = entry_index


def is_close_in_top_quarter(candle: Dict) -> bool:
    """
    Check if candle close is in top 25% of candle range.
    
    Args:
        candle: Candle dict with open, high, low, close
    
    Returns:
        True if close is in top 25% of range
    """
    candle_open = candle["open"]
    candle_high = candle["high"]
    candle_low = candle["low"]
    candle_close = candle["close"]
    
    candle_range = candle_high - candle_low
    if candle_range <= 0:
        return False
    
    # Top 25% means close is in upper quarter of range
    top_quarter_threshold = candle_low + (0.75 * candle_range)
    
    return candle_close >= top_quarter_threshold


def momentum_breakout_v1(
    candles: List[Dict],
    ema200_values: List[Optional[float]],
    atr14_values: List[Optional[float]],
    position_state: PositionState,
    strategy_state: Optional[BreakoutStrategyState] = None,
    consolidation_window: int = 20,
    consolidation_range_pct: float = 0.02,
    volume_multiplier: float = 1.5,
    volume_period: int = 20,
    atr_period: int = 14
) -> Dict[str, any]:
    """
    Momentum Breakout with Volume Confirmation Strategy v1.
    
    Entry Logic:
    1. Detect consolidation (low volatility, sideways movement)
    2. Wait for breakout above consolidation high
    3. Confirm with volume spike (>= 1.5x average)
    4. Confirm close in top 25% of candle range
    5. Regime gate: Only TREND_UP
    6. Generate BUY signal on candle close
    7. Entry executes on next candle open
    
    Exit Logic (Asymmetric):
    - Target 1: +2.0x ATR → exit 50% of position
    - Target 2: +3.5x ATR → exit remaining 50%
    - Stop-loss: -1.0x ATR
    - After Target 1: move stop to entry + 1.0x ATR (trailing stop)
    - Time stop: exit fully if Target 1 not hit within 10 candles
    
    Args:
        candles: List of candle dicts with keys: time, open, high, low, close, volume
        ema200_values: EMA(200) values aligned with candles (for regime check)
        atr14_values: ATR(14) values aligned with candles
        position_state: Current position state for this symbol
        strategy_state: Extended state for breakout strategy (tracks targets hit)
        consolidation_window: Number of candles for consolidation detection (default: 20)
        consolidation_range_pct: Max range as % of average price (default: 0.02 = 2%)
        volume_multiplier: Volume spike multiplier (default: 1.5)
        volume_period: Period for volume average (default: 20)
        atr_period: Period for ATR calculation (default: 14)
    
    Returns:
        Dict with keys:
        - signal: "BUY" | "EXIT" | "HOLD"
        - reason: String explanation
        - stop_distance: Optional[float] - stop distance for entry (ATR multiple)
        - entry_atr: Optional[float] - ATR at entry (to be stored in strategy_state)
    """
    if len(candles) < max(consolidation_window, atr_period + 1, volume_period + 1, 200):
        return {"signal": "HOLD", "reason": "Insufficient data"}
    
    current_index = len(candles) - 1
    current_candle = candles[current_index]
    current_close = current_candle["close"]
    current_time = current_candle.get("close_time", current_candle.get("time"))
    
    current_atr = atr14_values[current_index]
    if current_atr is None or current_atr <= 0:
        return {"signal": "HOLD", "reason": "ATR not ready"}
    
    # Initialize strategy state if not provided
    if strategy_state is None:
        strategy_state = BreakoutStrategyState()
    
    # EXIT LOGIC (check first if we have a position)
    if position_state.has_position and strategy_state.entry_atr is not None:
        entry_price = position_state.entry_price
        entry_atr = strategy_state.entry_atr
        
        # Calculate profit targets (in price terms)
        target1_price = entry_price + (2.0 * entry_atr)  # Target 1: +2.0x ATR
        target2_price = entry_price + (3.5 * entry_atr)  # Target 2: +3.5x ATR
        
        # Calculate stop-loss (in price terms)
        stop_loss_price = entry_price - (1.0 * entry_atr)  # Stop: -1.0x ATR
        
        # After Target 1 hit, move stop to entry + 1.0x ATR (trailing stop)
        if strategy_state.target1_hit:
            stop_loss_price = entry_price + (1.0 * entry_atr)  # Trailing stop
        
        # Check stop-loss
        if current_close <= stop_loss_price:
            strategy_state.reset()  # Reset state on exit
            return {
                "signal": "EXIT",
                "reason": f"Stop-loss hit: {fmt(current_close)} <= {fmt(stop_loss_price)} (entry: {fmt(entry_price)}, ATR: {fmt(entry_atr)})"
            }
        
        # Check Target 2 (exit fully - note: PaperBroker doesn't support partial exits,
        # so we exit fully at Target 2, which captures the remaining 50% profit)
        if current_close >= target2_price:
            if not strategy_state.target2_hit:
                strategy_state.target2_hit = True
            strategy_state.reset()  # Reset state on exit
            return {
                "signal": "EXIT",
                "reason": f"Target 2 hit: {fmt(current_close)} >= {fmt(target2_price)} (+3.5x ATR)"
            }
        
        # Check Target 1 (exit fully - note: PaperBroker doesn't support partial exits,
        # so we exit fully at Target 1, which captures 50% of intended profit)
        # In a system with partial exits, we'd exit 50% here and continue to Target 2
        if current_close >= target1_price:
            if not strategy_state.target1_hit:
                strategy_state.target1_hit = True
                # Exit fully at Target 1 (since partial exits not supported)
                strategy_state.reset()  # Reset state on exit
                return {
                    "signal": "EXIT",
                    "reason": f"Target 1 hit: {fmt(current_close)} >= {fmt(target1_price)} (+2.0x ATR)"
                }
        
        # Time-based exit: If Target 1 not hit within 10 candles, exit fully
        if strategy_state.entry_index is not None:
            candles_since_entry = current_index - strategy_state.entry_index
            if candles_since_entry >= 10 and not strategy_state.target1_hit:
                strategy_state.reset()  # Reset state on exit
                return {
                    "signal": "EXIT",
                    "reason": f"Time stop: Target 1 not hit within 10 candles (candles since entry: {candles_since_entry})"
                }
    
    # ENTRY LOGIC (only if no position)
    if not position_state.has_position:
        # Regime gate: Only TREND_UP
        current_regime = classify_regime(candles, current_index, ma_period=200)
        if current_regime is None:
            return {"signal": "HOLD", "reason": "Regime not classified (insufficient data for EMA(200))"}
        
        if current_regime != "TREND_UP":
            return {"signal": "HOLD", "reason": f"Regime gate: {current_regime} not allowed (only TREND_UP)"}
        
        # Additional regime check: Price > EMA(200) and EMA(200) rising
        current_ema200 = ema200_values[current_index]
        if current_ema200 is None:
            return {"signal": "HOLD", "reason": "EMA(200) not ready"}
        
        if current_index < 200:
            return {"signal": "HOLD", "reason": "Insufficient data for EMA(200)"}
        
        prev_ema200 = ema200_values[current_index - 1]
        if prev_ema200 is None:
            return {"signal": "HOLD", "reason": "Previous EMA(200) not ready"}
        
        price_above_ema200 = current_close > current_ema200
        ema200_rising = current_ema200 > prev_ema200
        
        if not (price_above_ema200 and ema200_rising):
            return {"signal": "HOLD", "reason": f"Regime check failed: price {fmt(current_close)} vs EMA(200) {fmt(current_ema200)}, rising={ema200_rising}"}
        
        # Detect consolidation
        consolidation_info = detect_consolidation(
            candles=candles,
            window_size=consolidation_window,
            range_pct=consolidation_range_pct,
            atr_period=atr_period
        )
        
        if consolidation_info is None:
            return {"signal": "HOLD", "reason": "Consolidation detection failed (insufficient data)"}
        
        if not consolidation_info["is_consolidation"]:
            return {"signal": "HOLD", "reason": f"No consolidation detected (range: {fmt(consolidation_info['range_pct']*100)}%, ATR: {consolidation_info['atr_current']} vs {consolidation_info['atr_previous']})"}
        
        consolidation_high = consolidation_info["highest_high"]
        
        # Check breakout: Close > highest high of consolidation window
        breakout_condition = current_close > consolidation_high
        
        if not breakout_condition:
            return {"signal": "HOLD", "reason": f"No breakout: close {fmt(current_close)} <= consolidation high {fmt(consolidation_high)}"}
        
        # Detect volume spike
        volume_info = detect_volume_spike(
            candles=candles,
            volume_period=volume_period,
            volume_multiplier=volume_multiplier
        )
        
        if volume_info is None:
            return {"signal": "HOLD", "reason": "Volume analysis failed (insufficient data)"}
        
        if not volume_info["is_spike"]:
            return {"signal": "HOLD", "reason": f"No volume spike: {fmt(volume_info['current_volume'])} vs {fmt(volume_info['average_volume'])} (ratio: {fmt(volume_info['volume_ratio'])})"}
        
        # Check if close is in top 25% of candle range
        if not is_close_in_top_quarter(current_candle):
            return {"signal": "HOLD", "reason": "Close not in top 25% of candle range"}
        
        # All conditions met - generate BUY signal
        strategy_state.entry_atr = current_atr
        strategy_state.entry_index = current_index
        
        return {
            "signal": "BUY",
            "reason": f"Breakout above consolidation high {fmt(consolidation_high)} with volume spike ({fmt(volume_info['volume_ratio'])}x avg) and close in top quarter",
            "stop_distance": 1.0 * current_atr,  # Stop: 1.0x ATR
            "entry_atr": current_atr  # Store ATR for exit calculations
        }
    
    return {"signal": "HOLD", "reason": "No signal conditions met"}


def momentum_breakout_wrapper(
    candles: List[Dict],
    ema200_values: List[Optional[float]],
    atr14_values: List[Optional[float]],
    position_state: PositionState,
    strategy_state: BreakoutStrategyState,
    current_index: int,
    entry_atr_from_signal: Optional[float] = None,
    consolidation_window: int = 20,
    consolidation_range_pct: float = 0.02,
    volume_multiplier: float = 1.5,
    volume_period: int = 20
) -> Dict[str, any]:
    """
    Wrapper function for momentum breakout strategy.
    Integrates with ReplayEngine by managing strategy state.
    
    Args:
        candles: List of candle dicts
        ema200_values: EMA(200) values aligned with candles
        atr14_values: ATR(14) values aligned with candles
        position_state: Current position state
        strategy_state: Strategy-specific state (persists across candles)
        current_index: Current candle index
        entry_atr_from_signal: ATR value from BUY signal (set after entry executes)
        consolidation_window: Consolidation window size
        consolidation_range_pct: Max range as % of average price
        volume_multiplier: Volume spike multiplier
        volume_period: Period for volume average
    
    Returns:
        Strategy result dict with signal, reason, etc.
    """
    current_candle = candles[current_index]
    current_timestamp = current_candle.get("close_time", current_candle.get("time"))
    
    # Update strategy state from position_state
    if position_state.has_position:
        strategy_state.has_position = True
        # Check if this is a new position (entry_price changed or not set)
        if (strategy_state.entry_price is None or 
            strategy_state.entry_price != position_state.entry_price):
            # New position entered - update entry info
            strategy_state.entry_price = position_state.entry_price
            strategy_state.entry_time = position_state.entry_time
            strategy_state.entry_index = current_index  # Entry executed on this candle
            # Set entry ATR if provided (from previous BUY signal)
            if entry_atr_from_signal is not None:
                strategy_state.entry_atr = entry_atr_from_signal
    else:
        # No position - check if we just exited
        if strategy_state.has_position:
            # Position was just closed - reset state
            strategy_state.reset()
    
    # Call strategy
    result = momentum_breakout_v1(
        candles=candles,
        ema200_values=ema200_values,
        atr14_values=atr14_values,
        position_state=position_state,
        strategy_state=strategy_state,
        consolidation_window=consolidation_window,
        consolidation_range_pct=consolidation_range_pct,
        volume_multiplier=volume_multiplier,
        volume_period=volume_period
    )
    
    return result
