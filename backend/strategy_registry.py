"""
Strategy registry for the trading system.
Provides a common interface for all strategies and allows the replay engine,
parameter sensitivity harness, and walk-forward harness to work with any
registered strategy by name.
"""

from typing import Dict, List, Optional, Callable, Any
from strategy import PositionState


class StrategyConfig:
    """Configuration for a registered strategy."""

    def __init__(
        self,
        name: str,
        evaluate_fn: Callable,
        default_params: Dict[str, Any],
        param_ranges: Optional[Dict[str, list]] = None,
        description: str = "",
    ):
        """
        Args:
            name: Unique strategy identifier (e.g. "ema_trend_v1")
            evaluate_fn: The strategy evaluation function
            default_params: Default parameter values for the strategy
            param_ranges: Parameter ranges for sensitivity testing
            description: Human-readable description
        """
        self.name = name
        self.evaluate_fn = evaluate_fn
        self.default_params = default_params
        self.param_ranges = param_ranges or {}
        self.description = description


# Global registry
_registry: Dict[str, StrategyConfig] = {}


def register_strategy(config: StrategyConfig):
    """Register a strategy in the global registry."""
    _registry[config.name] = config


def get_strategy(name: str) -> StrategyConfig:
    """Get a registered strategy by name. Raises KeyError if not found."""
    if name not in _registry:
        available = ", ".join(_registry.keys()) if _registry else "(none)"
        raise KeyError(f"Strategy '{name}' not registered. Available: {available}")
    return _registry[name]


def list_strategies() -> List[str]:
    """Return names of all registered strategies."""
    return list(_registry.keys())


# ---------------------------------------------------------------------------
# Register built-in strategies
# ---------------------------------------------------------------------------

def _register_ema_trend_v1():
    """Register ema_trend_v1."""
    from strategy import ema_trend_v1
    from indicators import ema, atr

    def evaluate(candles, position_state, params=None):
        params = params or {}
        ema_fast = params.get("ema_fast", 20)
        ema_slow = params.get("ema_slow", 50)

        closes = [c["close"] for c in candles]
        ema_fast_vals = ema(closes, ema_fast)
        ema_slow_vals = ema(closes, ema_slow)

        result = ema_trend_v1(candles, ema_fast_vals, ema_slow_vals, position_state)

        # Attach ATR for position sizing (stop_distance = atr_mult * ATR)
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        atr_vals = atr(highs, lows, closes, 14)
        current_atr = atr_vals[-1] if atr_vals and atr_vals[-1] is not None else None
        atr_mult = params.get("stop_loss_atr_multiplier", 2.0)
        result["stop_distance"] = (atr_mult * current_atr) if current_atr else None
        result["current_atr"] = current_atr

        return result

    register_strategy(StrategyConfig(
        name="ema_trend_v1",
        evaluate_fn=evaluate,
        default_params={
            "ema_fast": 20,
            "ema_slow": 50,
            "stop_loss_atr_multiplier": 2.0,
        },
        param_ranges={
            "ema_fast": [15, 20, 25],
            "ema_slow": [40, 50, 60],
            "stop_loss_atr_multiplier": [1.5, 2.0, 2.5],
        },
        description="EMA trend following: buy on EMA crossover, exit on close < slow EMA",
    ))


def _register_momentum_breakout_v1():
    """Register momentum_breakout_v1."""
    from breakout_strategy import momentum_breakout_v1, BreakoutStrategyState
    from indicators import ema, atr

    # Persistent strategy state across calls (tracks targets hit, etc.)
    _breakout_state = BreakoutStrategyState()

    def evaluate(candles, position_state, params=None):
        nonlocal _breakout_state
        params = params or {}

        # Sync position state into breakout state
        _breakout_state.has_position = position_state.has_position
        _breakout_state.entry_price = position_state.entry_price
        _breakout_state.entry_time = position_state.entry_time

        # Compute indicators
        closes = [c["close"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        ema200_values = ema(closes, 200)
        atr14_values = atr(highs, lows, closes, 14)

        result = momentum_breakout_v1(
            candles=candles,
            ema200_values=ema200_values,
            atr14_values=atr14_values,
            position_state=position_state,
            strategy_state=_breakout_state,
            consolidation_window=params.get("consolidation_window", 20),
            consolidation_range_pct=params.get("consolidation_range_pct", 0.02),
            volume_multiplier=params.get("volume_multiplier", 1.5),
        )

        # Attach stop_distance for PaperBroker position sizing
        current_atr = atr14_values[-1] if atr14_values and atr14_values[-1] is not None else None
        atr_mult = params.get("stop_loss_atr_multiplier", 1.0)
        if result.get("stop_distance") is None and current_atr:
            result["stop_distance"] = atr_mult * current_atr
        result["current_atr"] = current_atr

        return result

    register_strategy(StrategyConfig(
        name="momentum_breakout_v1",
        evaluate_fn=evaluate,
        default_params={
            "consolidation_window": 20,
            "consolidation_range_pct": 0.02,
            "volume_multiplier": 1.5,
            "stop_loss_atr_multiplier": 1.0,
            "target1_atr_multiplier": 2.0,
            "target2_atr_multiplier": 3.5,
        },
        param_ranges={
            "consolidation_window": [15, 20, 25],
            "consolidation_range_pct": [0.015, 0.02, 0.025],
            "volume_multiplier": [1.3, 1.5, 2.0],
            "stop_loss_atr_multiplier": [0.75, 1.0, 1.5],
        },
        description="Consolidation breakout with volume confirmation and asymmetric exits",
    ))


def _register_mean_reversion_v1():
    """Register mean_reversion_v1 (RSI-based)."""
    try:
        from strategy_mean_reversion_v1 import mean_reversion_v1
    except ImportError:
        return  # Strategy module not yet created

    def evaluate(candles, position_state, params=None):
        params = params or {}
        return mean_reversion_v1(candles, position_state, params)

    register_strategy(StrategyConfig(
        name="mean_reversion_v1",
        evaluate_fn=evaluate,
        default_params={
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_exit": 50,
            "stop_loss_atr_multiplier": 2.0,
            "require_trend_up": True,
        },
        param_ranges={
            "rsi_period": [10, 14, 20],
            "rsi_oversold": [25, 30, 35],
            "rsi_exit": [45, 50, 55],
            "stop_loss_atr_multiplier": [1.5, 2.0, 2.5],
        },
        description="Mean reversion: buy on RSI oversold in uptrend, exit when RSI normalizes",
    ))


def _register_dual_momentum_v1():
    """Register dual_momentum_v1."""
    try:
        from strategy_dual_momentum_v1 import dual_momentum_v1
    except ImportError:
        return

    def evaluate(candles, position_state, params=None):
        params = params or {}
        return dual_momentum_v1(candles, position_state, params)

    register_strategy(StrategyConfig(
        name="dual_momentum_v1",
        evaluate_fn=evaluate,
        default_params={
            "lookback": 126,
            "stop_loss_atr_multiplier": 2.0,
        },
        param_ranges={
            "lookback": [63, 126, 252],
            "stop_loss_atr_multiplier": [1.5, 2.0, 3.0],
        },
        description="Dual momentum: hold only when absolute and relative momentum are positive",
    ))


# Auto-register on import
_register_ema_trend_v1()
_register_momentum_breakout_v1()
_register_mean_reversion_v1()
_register_dual_momentum_v1()
