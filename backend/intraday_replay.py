"""
Intraday replay engine for 5-minute candle execution.
Executes minimal intraday trades gated by daily regime.

Trading Rules:
- Entry: daily_regime == TREND_UP AND current close > previous candle high
- Exit: current close < previous candle low
- Position constraints: One position max, fixed size ($10,000)
"""

from typing import List, Dict, Optional
from datetime import datetime, timezone
import uuid

from database import Signal, Trade, EquityCurve
from regime_classifier import get_regime_at_timestamp
from utils import ensure_utc_datetime, unix_to_utc_datetime, fmt, fmt_pct, fmt_currency
from intraday_risk_manager import IntradayRiskManager
from indicators import atr


def get_session_for_timestamp(timestamp) -> Optional[str]:
    """
    Determine which trading session a timestamp belongs to.
    
    Sessions (UTC times):
    - market_open: 13:30-14:30 UTC (09:30-10:30 ET)
    - midday: 14:30-18:30 UTC (10:30-14:30 ET)
    - power_hour: 18:30-20:00 UTC (14:30-16:00 ET)
    
    Args:
        timestamp: Unix timestamp (int/float) or datetime object
    
    Returns:
        Session name ("market_open", "midday", "power_hour") or None if outside trading hours
    """
    # Convert to datetime if needed
    if isinstance(timestamp, (int, float)):
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    elif isinstance(timestamp, datetime):
        dt = timestamp
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    else:
        return None
    
    hour = dt.hour
    minute = dt.minute
    time_minutes = hour * 60 + minute
    
    # Market Open: 13:30-14:30 UTC (09:30-10:30 ET)
    if 13 * 60 + 30 <= time_minutes < 14 * 60 + 30:
        return "market_open"
    # Power Hour: 18:30-20:00 UTC (14:30-16:00 ET)
    elif 18 * 60 + 30 <= time_minutes < 20 * 60:
        return "power_hour"
    # Midday: 14:30-18:30 UTC (10:30-14:30 ET)
    elif 14 * 60 + 30 <= time_minutes < 18 * 60 + 30:
        return "midday"
    
    return None  # Outside trading hours


def align_intraday_with_daily(
    intraday_candle: Dict,
    daily_candles: List[Dict]
) -> Optional[str]:
    """
    Determine the daily regime for an intraday candle.
    
    Maps an intraday (5-minute) candle to its corresponding daily candle
    and returns the daily regime classification.
    
    Args:
        intraday_candle: Intraday candle dict with time (Unix timestamp)
        daily_candles: List of daily candles (for regime classification)
    
    Returns:
        Daily regime string ("TREND_UP", "TREND_DOWN", "CHOP") or None
    """
    # Convert intraday candle timestamp to datetime
    intraday_time = intraday_candle.get("time")
    if isinstance(intraday_time, (int, float)):
        intraday_dt = datetime.fromtimestamp(intraday_time, tz=timezone.utc)
    elif isinstance(intraday_time, datetime):
        intraday_dt = intraday_time
    else:
        return None
    
    # Find the daily candle that contains this intraday timestamp
    # For daily candles, we use the date (ignoring time component)
    intraday_date = intraday_dt.date()
    
    # Find matching daily candle by date
    for daily_candle in daily_candles:
        daily_time = daily_candle.get("time")
        if isinstance(daily_time, (int, float)):
            daily_dt = datetime.fromtimestamp(daily_time, tz=timezone.utc)
        elif isinstance(daily_time, datetime):
            daily_dt = daily_time
        else:
            continue
        
        # Match by date (daily candles are normalized to midnight UTC)
        if daily_dt.date() == intraday_date:
            # Found matching daily candle, get its regime
            regime = get_regime_at_timestamp(daily_candles, daily_time)
            return regime or "CHOP"  # Default to CHOP if can't classify
    
    return None  # No matching daily candle found


def evaluate_intraday_entry(
    intraday_candle: Dict,
    previous_candle: Optional[Dict],
    daily_regime: Optional[str],
    has_position: bool,
    allowed_sessions: Optional[List[str]] = None,
    entry_variant: Optional[str] = None,
    diagnostic_mode: bool = False,
    bypass_daily_regime_gate: bool = False,
    bypass_session_gate: bool = False
) -> Dict:
    """
    Evaluate intraday entry signal using configurable entry variant.
    
    Entry conditions:
    - daily_regime == TREND_UP
    - Entry variant check (see entry_variant parameter)
    - no existing position
    - current session is in allowed_sessions (if specified)
    
    Entry variants:
    - "break_prev_high" (default): current close > previous candle high
    - "close_above_prev_close": current close > previous candle close
    - "close_above_prev_low": current close > previous candle low
    - "break_prev_high_atr": current close > previous high AND (close - prev_high) >= 0.25 * ATR(14)
    
    Args:
        intraday_candle: Current intraday candle dict with time, open, high, low, close, volume
        previous_candle: Previous intraday candle (for momentum check)
        daily_regime: Daily regime string ("TREND_UP", "TREND_DOWN", "CHOP") or None
        has_position: True if position is already open
        allowed_sessions: Optional list of allowed sessions ("market_open", "midday", "power_hour")
                         If None, all sessions are allowed
        entry_variant: Entry variant ("break_prev_high" or "close_above_prev_close")
                       If None or unknown, defaults to "break_prev_high"
    
    Returns:
        Dict with:
        - signal: "BUY" or "NO_TRADE"
        - reason: Explanation
        - daily_regime: The daily regime for this intraday candle
    """
    # Gate: Only trade in TREND_UP regime (bypassed in diagnostic mode or if bypass_daily_regime_gate is True)
    if not diagnostic_mode and not bypass_daily_regime_gate and daily_regime != "TREND_UP":
        return {
            "signal": "NO_TRADE",
            "reason": f"Daily regime is {daily_regime}, not TREND_UP",
            "daily_regime": daily_regime
        }
    
    # Gate: No position already open
    if has_position:
        return {
            "signal": "NO_TRADE",
            "reason": "Position already open (one position max)",
            "daily_regime": daily_regime
        }
    
    # Gate: Session gating (if specified, bypassed in diagnostic mode or if bypass_session_gate is True)
    if not diagnostic_mode and not bypass_session_gate and allowed_sessions:
        candle_time = intraday_candle.get("time")
        current_session = get_session_for_timestamp(candle_time)
        
        if current_session is None:
            return {
                "signal": "NO_TRADE",
                "reason": "Outside trading hours",
                "daily_regime": daily_regime
            }
        
        if current_session not in allowed_sessions:
            return {
                "signal": "NO_TRADE",
                "reason": f"Session {current_session} not in allowed sessions {allowed_sessions}",
                "daily_regime": daily_regime
            }
    
    # Gate: Need previous candle for momentum check
    if previous_candle is None:
        return {
            "signal": "NO_TRADE",
            "reason": "No previous candle for momentum check",
            "daily_regime": daily_regime
        }
    
    # Normalize entry_variant (default to "break_prev_high" if None or unknown)
    variant = entry_variant if entry_variant in ["break_prev_high", "close_above_prev_close", "close_above_prev_low", "break_prev_high_atr"] else "break_prev_high"
    
    # Entry rule: Branch based on entry_variant
    current_close = intraday_candle["close"]
    candle_time = intraday_candle.get("time")
    prev_close = previous_candle["close"] if previous_candle else None
    prev_high = previous_candle["high"] if previous_candle else None
    prev_low = previous_candle["low"] if previous_candle else None
    
    # Debug log at top of function
    current_atr = intraday_candle.get("atr")
    print(f"[DEBUG] evaluate_intraday_entry called: variant={variant}, close={fmt(current_close)}, prev_close={fmt(prev_close)}, prev_high={fmt(prev_high)}, prev_low={fmt(prev_low)}, atr={fmt(current_atr)}")
    
    if variant == "break_prev_high":
        # Default variant: current close > previous candle high
        previous_high = previous_candle["high"]
        
        if current_close > previous_high:
            # Convert timestamp for logging
            if isinstance(candle_time, (int, float)):
                timestamp_dt = datetime.fromtimestamp(candle_time, tz=timezone.utc)
            else:
                timestamp_dt = candle_time
            print(f"[ENTRY HIT] variant={variant} at {timestamp_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            print(f"[INTRADAY ENTRY] variant={variant} triggered at {timestamp_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            
            return {
                "signal": "BUY",
                "reason": f"Intraday momentum: close {fmt(current_close)} > prev high {fmt(previous_high)} (daily_regime=TREND_UP)",
                "daily_regime": daily_regime
            }
        
        return {
            "signal": "NO_TRADE",
            "reason": f"Intraday momentum not met: close {fmt(current_close)} <= prev high {fmt(previous_high)}",
            "daily_regime": daily_regime
        }
    
    elif variant == "close_above_prev_close":
        # New variant: current close > previous candle close
        previous_close = previous_candle["close"]
        
        if current_close > previous_close:
            # Convert timestamp for logging
            if isinstance(candle_time, (int, float)):
                timestamp_dt = datetime.fromtimestamp(candle_time, tz=timezone.utc)
            else:
                timestamp_dt = candle_time
            print(f"[ENTRY HIT] variant={variant} at {timestamp_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            print(f"[INTRADAY ENTRY] variant={variant} triggered at {timestamp_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            
            return {
                "signal": "BUY",
                "reason": f"Intraday momentum: close {fmt(current_close)} > prev close {fmt(previous_close)} (daily_regime=TREND_UP)",
                "daily_regime": daily_regime
            }
        
        return {
            "signal": "NO_TRADE",
            "reason": f"Intraday momentum not met: close {fmt(current_close)} <= prev close {fmt(previous_close)}",
            "daily_regime": daily_regime
        }
    
    elif variant == "close_above_prev_low":
        # Temporary variant: current close > previous candle low
        previous_low = previous_candle["low"]
        
        if current_close > previous_low:
            # Convert timestamp for logging
            if isinstance(candle_time, (int, float)):
                timestamp_dt = datetime.fromtimestamp(candle_time, tz=timezone.utc)
            else:
                timestamp_dt = candle_time
            print(f"[ENTRY HIT] variant={variant} at {timestamp_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            print(f"[INTRADAY ENTRY] variant={variant} triggered at {timestamp_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            
            return {
                "signal": "BUY",
                "reason": f"Intraday momentum: close {fmt(current_close)} > prev low {fmt(previous_low)} (daily_regime=TREND_UP)",
                "daily_regime": daily_regime
            }
        
        return {
            "signal": "NO_TRADE",
            "reason": f"Intraday momentum not met: close {fmt(current_close)} <= prev low {fmt(previous_low)}",
            "daily_regime": daily_regime
        }
    
    elif variant == "break_prev_high_atr":
        # Volatility-aware variant: current close > previous high AND breakout >= 0.25 * ATR
        previous_high = previous_candle["high"]
        current_atr = intraday_candle.get("atr")
        
        # Check basic breakout condition
        if current_close <= previous_high:
            return {
                "signal": "NO_TRADE",
                "reason": f"Breakout not met: close {fmt(current_close)} <= prev high {fmt(previous_high)}",
                "daily_regime": daily_regime
            }
        
        # Check ATR filter if ATR is available
        if current_atr is not None and current_atr > 0:
            breakout_size = current_close - previous_high
            min_breakout = 0.25 * current_atr
            
            if breakout_size < min_breakout:
                return {
                    "signal": "NO_TRADE",
                    "reason": f"Breakout too weak: {fmt(breakout_size)} < 0.25*ATR ({fmt(min_breakout)})",
                    "daily_regime": daily_regime
                }
            
            # Both conditions met
            if isinstance(candle_time, (int, float)):
                timestamp_dt = datetime.fromtimestamp(candle_time, tz=timezone.utc)
            else:
                timestamp_dt = candle_time
            print(f"[ENTRY HIT] variant={variant} at {timestamp_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            print(f"[INTRADAY ENTRY] variant={variant} triggered at {timestamp_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            
            return {
                "signal": "BUY",
                "reason": f"Volatility-filtered breakout: close {fmt(current_close)} > prev high {fmt(previous_high)}, breakout {fmt(breakout_size)} >= 0.25*ATR ({fmt(min_breakout)})",
                "daily_regime": daily_regime
            }
        else:
            # ATR not available - fall back to basic breakout
            if isinstance(candle_time, (int, float)):
                timestamp_dt = datetime.fromtimestamp(candle_time, tz=timezone.utc)
            else:
                timestamp_dt = candle_time
            print(f"[ENTRY HIT] variant={variant} (ATR unavailable, using basic breakout) at {timestamp_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            
            return {
                "signal": "BUY",
                "reason": f"Breakout (ATR unavailable): close {fmt(current_close)} > prev high {fmt(previous_high)}",
                "daily_regime": daily_regime
            }
    
    else:
        # Fallback (should not reach here due to normalization above)
        return {
            "signal": "NO_TRADE",
            "reason": f"Unknown entry variant: {entry_variant}, defaulting to break_prev_high",
            "daily_regime": daily_regime
        }


def evaluate_intraday_exit(
    intraday_candle: Dict,
    previous_candle: Optional[Dict],
    entry_price: float,
    exit_mode: str = "momentum_reversal",
    candles_since_entry: int = 0,
    max_favorable_price: Optional[float] = None,
    exit_params: Optional[Dict] = None
) -> Dict:
    """
    Evaluate intraday exit signal using configurable exit modes.
    
    Exit modes:
    - "momentum_reversal" (default): Exit when close < previous candle low
    - "time_based": Exit after N candles (default N=6, i.e. 30 minutes)
    - "mfe_trailing": Exit when price retraces X% of MFE (default X=50%)
    
    Args:
        intraday_candle: Current intraday candle dict with time, open, high, low, close, volume
        previous_candle: Previous intraday candle (for momentum check)
        entry_price: Entry price of the position
        exit_mode: Exit mode ("momentum_reversal", "time_based", "mfe_trailing")
        candles_since_entry: Number of candles since entry (for time_based mode)
        max_favorable_price: Maximum favorable price reached (for mfe_trailing mode)
        exit_params: Optional dict with exit parameters (e.g., {"time_candles": 6, "mfe_retrace_pct": 50})
    
    Returns:
        Dict with:
        - signal: "EXIT" or "HOLD"
        - reason: Explanation
    """
    if exit_params is None:
        exit_params = {}
    
    current_close = intraday_candle["close"]
    
    # Exit mode: momentum_reversal (default)
    if exit_mode == "momentum_reversal":
        # Gate: Need previous candle for momentum check
        if previous_candle is None:
            return {
                "signal": "HOLD",
                "reason": "No previous candle for exit check"
            }
        
        # Exit rule: current close < previous candle low
        previous_low = previous_candle["low"]
        
        if current_close < previous_low:
            return {
                "signal": "EXIT",
                "reason": f"Intraday momentum reversal: close {fmt(current_close)} < prev low {fmt(previous_low)}"
            }
        
        return {
            "signal": "HOLD",
            "reason": f"Intraday momentum intact: close {fmt(current_close)} >= prev low {fmt(previous_low)}"
        }
    
    # Exit mode: time_based
    elif exit_mode == "time_based":
        time_candles = exit_params.get("time_candles", 6)  # Default: 6 candles (30 minutes)
        
        if candles_since_entry >= time_candles:
            return {
                "signal": "EXIT",
                "reason": f"Time-based exit: {candles_since_entry} candles elapsed (limit: {time_candles})"
            }
        
        return {
            "signal": "HOLD",
            "reason": f"Time-based hold: {candles_since_entry}/{time_candles} candles elapsed"
        }
    
    # Exit mode: mfe_trailing
    elif exit_mode == "mfe_trailing":
        if max_favorable_price is None:
            # First candle after entry - initialize MFE
            return {
                "signal": "HOLD",
                "reason": "MFE trailing: Initializing max favorable price"
            }
        
        mfe_retrace_pct = exit_params.get("mfe_retrace_pct", 50.0)  # Default: 50% retracement
        
        # Calculate MFE (max favorable excursion from entry)
        mfe = max_favorable_price - entry_price
        
        if mfe <= 0:
            # No favorable move yet
            return {
                "signal": "HOLD",
                "reason": f"MFE trailing: No favorable move yet (current: {fmt(current_close)}, entry: {fmt(entry_price)})"
            }
        
        # Calculate current price relative to entry
        current_move = current_close - entry_price
        
        # Calculate how much of MFE we've given up
        mfe_given_up = mfe - current_move
        mfe_given_up_pct = (mfe_given_up / mfe * 100) if mfe > 0 else 0.0
        
        # Exit if we've given up X% of MFE
        if mfe_given_up_pct >= mfe_retrace_pct:
            return {
                "signal": "EXIT",
                "reason": f"MFE trailing exit: Retraced {fmt_pct(mfe_given_up_pct, 1)} of MFE (limit: {fmt(mfe_retrace_pct)}%, MFE: {fmt(mfe)})"
            }
        
        return {
            "signal": "HOLD",
            "reason": f"MFE trailing hold: Retraced {fmt_pct(mfe_given_up_pct, 1)} of MFE (limit: {fmt(mfe_retrace_pct)}%)"
        }
    
    else:
        # Unknown exit mode - default to HOLD
        return {
            "signal": "HOLD",
            "reason": f"Unknown exit mode: {exit_mode}, defaulting to HOLD"
        }


class IntradayReplayEngine:
    """
    Intraday replay engine for processing 5-minute candles.
    Executes minimal intraday trades gated by daily regime.
    """
    
    def __init__(
        self,
        initial_equity: float = 100000.0,
        risk_per_trade_pct: float = 0.25,
        max_daily_loss_pct: float = 1.0,
        max_concurrent_positions: int = 1
    ):
        self.initial_equity = initial_equity
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_concurrent_positions = max_concurrent_positions
        
        self.replay_id: Optional[str] = None
        self.symbol: Optional[str] = None
        self.intraday_candles: List[Dict] = []
        self.daily_candles: List[Dict] = []
        self.current_candle_index = 0
        self.total_candles = 0
        self.status = "idle"  # idle, running, completed, error
        self.error_message: Optional[str] = None
        self.source: Optional[str] = None
        
        # Position tracking (one position max)
        self.has_position = False
        self.entry_price: Optional[float] = None
        self.entry_time: Optional[int] = None
        self.entry_shares: Optional[int] = None
        self.candles_since_entry: int = 0  # For time_based exit mode
        self.max_favorable_price: Optional[float] = None  # For mfe_trailing exit mode
        
        # Exit configuration
        self.exit_mode: str = "momentum_reversal"  # Default exit mode
        self.exit_params: Dict = {}  # Exit mode parameters
        
        # Entry configuration
        self.entry_variant: str = "break_prev_high"  # Default entry variant
        
        # Session gating
        self.allowed_sessions: Optional[List[str]] = None  # Optional session gate for entries
        
        # Diagnostic mode (bypasses entry gates for testing)
        self.diagnostic_mode: bool = False
        
        # Individual gate bypass flags
        self.bypass_daily_regime_gate: bool = False
        self.bypass_session_gate: bool = False
        
        # Risk manager (initialized in start_intraday_replay)
        self.risk_manager: Optional[IntradayRiskManager] = None
        
        # Trade execution
        self.trades: List[Dict] = []  # List of executed trades
    
    def reset(self):
        """Reset all state for a new replay."""
        self.intraday_candles = []
        self.daily_candles = []
        self.current_candle_index = 0
        self.total_candles = 0
        self.status = "idle"
        self.error_message = None
        self.source = None
        self.replay_id = None
        self.symbol = None
        
        # Reset position tracking
        self.has_position = False
        self.entry_price = None
        self.entry_time = None
        self.entry_shares = None
        self.candles_since_entry = 0
        self.max_favorable_price = None
        
        # Reset exit configuration
        self.exit_mode = "momentum_reversal"
        self.exit_params = {}
        
        # Reset entry configuration
        self.entry_variant = "break_prev_high"
        
        # Reset session gating
        self.allowed_sessions = None
        
        # Reset diagnostic mode and gate bypass flags
        self.diagnostic_mode = False
        self.bypass_daily_regime_gate = False
        self.bypass_session_gate = False
        
        # Reset risk manager and trades
        self.risk_manager = None
        self.trades = []
    
    def start_intraday_replay(
        self,
        symbol: str,
        intraday_candles: List[Dict],
        daily_candles: List[Dict],
        replay_id: Optional[str] = None,
        source: Optional[str] = None,
        exit_mode: str = "momentum_reversal",
        exit_params: Optional[Dict] = None,
        allowed_sessions: Optional[List[str]] = None,
        entry_variant: Optional[str] = None,
        diagnostic_mode: bool = False,
        bypass_daily_regime_gate: bool = False,
        bypass_session_gate: bool = False,
        risk_per_trade_pct: Optional[float] = None,
        max_daily_loss_pct: Optional[float] = None,
        max_concurrent_positions: Optional[int] = None
    ) -> str:
        """
        Start a new intraday replay.
        
        Args:
            symbol: Trading symbol
            intraday_candles: Ordered list of 5-minute candle dicts
            daily_candles: Ordered list of daily candle dicts (for regime classification)
            replay_id: Optional replay ID (generated if not provided)
            source: Data source (e.g., "csv")
        
        Returns:
            replay_id: Unique identifier for this replay
        """
        if self.status == "running":
            raise ValueError("Intraday replay already in progress")
        
        # Validate candles
        if not intraday_candles or len(intraday_candles) < 10:
            raise ValueError("Need at least 10 intraday candles for replay")
        
        if not daily_candles or len(daily_candles) < 200:
            raise ValueError("Need at least 200 daily candles for regime classification (EMA(200))")
        
        # Validate candle ordering
        for i in range(1, len(intraday_candles)):
            if intraday_candles[i]["time"] <= intraday_candles[i-1]["time"]:
                raise ValueError(f"Intraday candles must be ordered by time. Candle {i} has timestamp {intraday_candles[i]['time']} <= previous {intraday_candles[i-1]['time']}")
        
        for i in range(1, len(daily_candles)):
            if daily_candles[i]["time"] <= daily_candles[i-1]["time"]:
                raise ValueError(f"Daily candles must be ordered by time. Candle {i} has timestamp {daily_candles[i]['time']} <= previous {daily_candles[i-1]['time']}")
        
        # Generate replay_id if not provided
        if replay_id is None:
            replay_id = str(uuid.uuid4())
        
        # Reset state
        self.reset()
        
        # Initialize
        self.replay_id = replay_id
        self.symbol = symbol
        self.intraday_candles = intraday_candles.copy()
        self.daily_candles = daily_candles.copy()
        self.total_candles = len(intraday_candles)
        self.status = "running"
        self.source = source or "csv_intraday"
        
        # Set exit mode and parameters
        self.exit_mode = exit_mode
        self.exit_params = exit_params or {}
        
        # Set entry variant (default to "break_prev_high" if None or unknown)
        if entry_variant and entry_variant in ["break_prev_high", "close_above_prev_close", "close_above_prev_low", "break_prev_high_atr"]:
            self.entry_variant = entry_variant
        else:
            self.entry_variant = "break_prev_high"
        
        # Set session gating
        self.allowed_sessions = allowed_sessions
        
        # Set diagnostic mode and gate bypass flags
        self.diagnostic_mode = diagnostic_mode
        self.bypass_daily_regime_gate = bypass_daily_regime_gate
        self.bypass_session_gate = bypass_session_gate
        
        # Use provided risk parameters or defaults
        risk_pct = risk_per_trade_pct if risk_per_trade_pct is not None else self.risk_per_trade_pct
        max_loss_pct = max_daily_loss_pct if max_daily_loss_pct is not None else self.max_daily_loss_pct
        max_positions = max_concurrent_positions if max_concurrent_positions is not None else self.max_concurrent_positions
        
        # Initialize risk manager
        self.risk_manager = IntradayRiskManager(
            initial_equity=self.initial_equity,
            risk_per_trade_pct=risk_pct,
            max_daily_loss_pct=max_loss_pct,
            max_concurrent_positions=max_positions
        )
        
        print(f"[INTRADAY REPLAY] Started replay {replay_id}")
        print(f"[INTRADAY REPLAY]   Symbol: {symbol}")
        print(f"[INTRADAY REPLAY]   Intraday candles: {len(intraday_candles)}")
        print(f"[INTRADAY REPLAY]   Daily candles: {len(daily_candles)}")
        print(f"[INTRADAY REPLAY]   Initial equity: ${self.initial_equity:,.2f}")
        print(f"[INTRADAY REPLAY]   Risk per trade: {fmt_pct(risk_pct)}")
        print(f"[INTRADAY REPLAY]   Max daily loss: {fmt_pct(max_loss_pct)}")
        print(f"[INTRADAY REPLAY]   Max concurrent positions: {max_positions}")
        print(f"[INTRADAY REPLAY]   Exit mode: {exit_mode}")
        if exit_params:
            print(f"[INTRADAY REPLAY]   Exit params: {exit_params}")
        print(f"[INTRADAY REPLAY]   Entry variant: {self.entry_variant}")
        if allowed_sessions:
            print(f"[INTRADAY REPLAY]   Allowed sessions: {allowed_sessions}")
        if self.diagnostic_mode:
            print(f"[INTRADAY REPLAY]   [DIAGNOSTIC MODE ENABLED] Entry gates bypassed")
        else:
            print(f"[INTRADAY REPLAY]   Allowed sessions: All (no session gate)")
        
        return replay_id
    
    def process_intraday_candle(self, candle: Dict, db_session, previous_candle: Optional[Dict] = None) -> Optional[Dict]:
        """
        Process a single intraday candle and execute trades if conditions are met.
        
        Args:
            candle: Intraday candle dict with keys: time, open, high, low, close, volume
            db_session: Database session for storing trades
            previous_candle: Previous intraday candle (for momentum checks)
        
        Returns:
            Dict with processing info, or None
        """
        # Determine daily regime for this intraday candle
        daily_regime = align_intraday_with_daily(candle, self.daily_candles)
        
        # Log intraday candle with daily regime
        candle_time = candle.get("time")
        if isinstance(candle_time, (int, float)):
            candle_dt = datetime.fromtimestamp(candle_time, tz=timezone.utc)
        else:
            candle_dt = candle_time
        
        print(f"[INTRADAY] Processing candle: {candle_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC, "
              f"close={fmt(candle.get('close'))}, daily_regime={daily_regime}, has_position={self.has_position}")
        
        # Check for exit signal first (if position is open)
        if self.has_position:
            # Increment candles since entry (for time_based mode) - do this first
            self.candles_since_entry += 1
            
            # Update MFE for trailing stop mode
            if self.exit_mode == "mfe_trailing":
                if self.max_favorable_price is None:
                    self.max_favorable_price = candle["high"]
                else:
                    self.max_favorable_price = max(self.max_favorable_price, candle["high"])
            
            # Evaluate exit based on configured mode
            exit_result = evaluate_intraday_exit(
                candle,
                previous_candle,
                self.entry_price,
                exit_mode=self.exit_mode,
                candles_since_entry=self.candles_since_entry,
                max_favorable_price=self.max_favorable_price,
                exit_params=self.exit_params
            )
            
            if exit_result["signal"] == "EXIT":
                # Execute exit through risk manager
                exit_price = candle["close"]
                exit_time = candle_time
                
                exit_result_dict = self.risk_manager.exit_position(
                    symbol=self.symbol,
                    exit_price=exit_price,
                    timestamp=exit_time,
                    reason=exit_result["reason"]
                )
                
                if exit_result_dict:
                    pnl = exit_result_dict["pnl"]
                    fill_exit_price = exit_result_dict["exit_price"]
                    
                    # Update existing trade record in database
                    from utils import unix_to_utc_datetime
                    exit_time_dt = unix_to_utc_datetime(exit_time)
                    
                    open_trade = db_session.query(Trade).filter(
                        Trade.symbol == self.symbol,
                        Trade.exit_time.is_(None),
                        Trade.replay_id == self.replay_id
                    ).first()
                    
                    if open_trade:
                        open_trade.exit_time = exit_time_dt
                        open_trade.exit_price = fill_exit_price
                        open_trade.pnl = pnl
                        open_trade.reason = exit_result["reason"]
                        db_session.commit()
                    
                    # Log exit
                    current_equity = self.risk_manager.get_current_equity()
                    print(f"[INTRADAY TRADE] EXIT: {self.symbol} @ {fmt(fill_exit_price)}, "
                          f"P&L={fmt_currency(pnl)}, equity={fmt_currency(current_equity)}, reason: {exit_result['reason']}")
                    
                    # Record trade
                    self.trades.append({
                        "entry_time": self.entry_time,
                        "entry_price": exit_result_dict["entry_price"],
                        "exit_time": exit_time,
                        "exit_price": fill_exit_price,
                        "shares": exit_result_dict["shares"],
                        "pnl": pnl,
                        "reason": exit_result["reason"]
                    })
                    
                    # Reset position
                    self.has_position = False
                    self.entry_price = None
                    self.entry_time = None
                    self.entry_shares = None
                    self.candles_since_entry = 0
                    self.max_favorable_price = None
                
                return {
                    "intraday_timestamp": candle_time,
                    "price": candle["close"],
                    "daily_regime": daily_regime,
                    "exit_signal": "EXIT",
                    "exit_reason": exit_result["reason"],
                    "pnl": pnl
                }
        
        # Check for entry signal (if no position)
        if not self.has_position:
            # Debug log before calling entry evaluator
            candle_time = candle.get("time")
            if isinstance(candle_time, (int, float)):
                timestamp_dt = datetime.fromtimestamp(candle_time, tz=timezone.utc)
            else:
                timestamp_dt = candle_time
            print(f"[DEBUG] Calling entry evaluator with entry_variant={self.entry_variant} at {timestamp_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            
            entry_result = evaluate_intraday_entry(
                candle, previous_candle, daily_regime, self.has_position, self.allowed_sessions, 
                self.entry_variant, self.diagnostic_mode, self.bypass_daily_regime_gate, self.bypass_session_gate
            )
            
            if entry_result["signal"] == "BUY":
                # Calculate stop distance using risk manager
                entry_price = candle["close"]
                stop_distance = self.risk_manager.calculate_stop_distance(
                    entry_price=entry_price,
                    previous_candle=previous_candle
                )
                
                # Calculate position size based on risk
                shares = self.risk_manager.calculate_position_size(
                    entry_price=entry_price,
                    stop_distance=stop_distance
                )
                
                if shares > 0:
                    # Check if we can enter position (risk checks)
                    if isinstance(candle_time, (int, float)):
                        entry_dt = datetime.fromtimestamp(candle_time, tz=timezone.utc)
                    else:
                        entry_dt = candle_time
                    current_date = entry_dt.date()
                    
                    can_enter, reason_msg = self.risk_manager.can_enter_position(self.symbol, current_date)
                    
                    if can_enter:
                        # Execute entry through risk manager
                        position_info = self.risk_manager.enter_position(
                            symbol=self.symbol,
                            entry_price=entry_price,
                            shares=shares,
                            stop_distance=stop_distance,
                            timestamp=candle_time,
                            reason=entry_result["reason"]
                        )
                        
                        if position_info:
                            fill_entry_price = position_info["entry_price"]
                            actual_shares = position_info["shares"]
                            
                            # Store trade entry in database
                            from utils import unix_to_utc_datetime
                            entry_time_dt = unix_to_utc_datetime(candle_time)
                            
                            trade = Trade(
                                symbol=self.symbol,
                                entry_time=entry_time_dt,
                                entry_price=fill_entry_price,
                                exit_time=None,  # Position is open
                                exit_price=None,
                                shares=actual_shares,
                                pnl=None,  # Unrealized P&L
                                reason=entry_result["reason"],
                                replay_id=self.replay_id
                            )
                            db_session.add(trade)
                            db_session.commit()
                            
                            # Log entry
                            current_equity = self.risk_manager.get_current_equity()
                            risk_amount = current_equity * (self.risk_per_trade_pct / 100.0)
                            print(f"[INTRADAY TRADE] ENTRY: {self.symbol} @ {fmt(fill_entry_price)}, "
                                  f"shares={actual_shares}, risk={fmt_currency(risk_amount)} ({fmt_pct(self.risk_per_trade_pct)}), "
                                  f"equity={fmt_currency(current_equity)}, reason: {entry_result['reason']}")
                            
                            # Update position state
                            self.has_position = True
                            self.entry_price = fill_entry_price
                            self.entry_time = candle_time
                            self.entry_shares = actual_shares
                            self.candles_since_entry = 0  # Reset counter
                            self.max_favorable_price = None  # Reset MFE tracking
                            
                            return {
                                "intraday_timestamp": candle_time,
                                "price": candle["close"],
                                "daily_regime": daily_regime,
                                "entry_signal": "BUY",
                                "entry_reason": entry_result["reason"],
                                "shares": actual_shares,
                                "entry_price": fill_entry_price,
                                "risk_amount": risk_amount
                            }
                    else:
                        # Entry blocked by risk manager
                        print(f"[INTRADAY TRADE] ENTRY BLOCKED: {self.symbol} - {reason_msg}")
        
        return {
            "intraday_timestamp": candle_time,
            "price": candle["close"],
            "daily_regime": daily_regime,
            "signal": "HOLD"
        }
    
    def run(self, db_session) -> Dict:
        """
        Run the complete intraday replay, processing all 5-minute candles sequentially.
        
        Args:
            db_session: Database session for storing trades
        
        Returns:
            Dict with replay results: status, replay_id, total_candles, trades, etc.
        """
        if self.status != "running":
            raise ValueError(f"Cannot run intraday replay in status: {self.status}")
        
        try:
            # Initialize final_equity at the start to ensure it's always defined
            final_equity = self.initial_equity  # Default value, will be updated later
            
            processed_count = 0
            previous_candle = None
            
            # Compute ATR(14) on intraday candles and attach to each candle
            if len(self.intraday_candles) >= 15:  # Need at least 15 candles for ATR(14)
                highs = [c["high"] for c in self.intraday_candles]
                lows = [c["low"] for c in self.intraday_candles]
                closes = [c["close"] for c in self.intraday_candles]
                atr_values = atr(highs, lows, closes, period=14)
                
                # Attach ATR to each candle
                for i, atr_val in enumerate(atr_values):
                    self.intraday_candles[i]["atr"] = atr_val
            else:
                # Not enough candles for ATR - set to None
                for candle in self.intraday_candles:
                    candle["atr"] = None
            
            # Process intraday candles one by one
            for i, candle in enumerate(self.intraday_candles):
                self.current_candle_index = i + 1
                self.process_intraday_candle(candle, db_session, previous_candle)
                previous_candle = candle  # Update for next iteration
                processed_count += 1
            
            # Close any remaining open position at the end
            if self.has_position:
                # Exit at final candle close through risk manager
                final_candle = self.intraday_candles[-1]
                exit_price = final_candle["close"]
                exit_time = final_candle["time"]
                
                exit_result_dict = self.risk_manager.exit_position(
                    symbol=self.symbol,
                    exit_price=exit_price,
                    timestamp=exit_time,
                    reason="End of replay"
                )
                
                if exit_result_dict:
                    pnl = exit_result_dict["pnl"]
                    fill_exit_price = exit_result_dict["exit_price"]
                    
                    # Update trade in database
                    from utils import unix_to_utc_datetime
                    exit_time_dt = unix_to_utc_datetime(exit_time)
                    
                    from database import Trade
                    open_trade = db_session.query(Trade).filter(
                        Trade.symbol == self.symbol,
                        Trade.exit_time.is_(None),
                        Trade.replay_id == self.replay_id
                    ).first()
                    
                    if open_trade:
                        open_trade.exit_time = exit_time_dt
                        open_trade.exit_price = fill_exit_price
                        open_trade.pnl = pnl
                        open_trade.reason = "End of replay"
                        db_session.commit()
                    
                    current_equity = self.risk_manager.get_current_equity()
                    print(f"[INTRADAY TRADE] EXIT (end of replay): {self.symbol} @ {fmt(fill_exit_price)}, "
                          f"P&L={fmt_currency(pnl)}, equity={fmt_currency(current_equity)}")
                    
                    self.trades.append({
                        "entry_time": self.entry_time,
                        "entry_price": exit_result_dict["entry_price"],
                        "exit_time": exit_time,
                        "exit_price": fill_exit_price,
                        "shares": exit_result_dict["shares"],
                        "pnl": pnl,
                        "reason": "End of replay"
                    })
            
            self.status = "completed"
            
            # Count closed trades
            closed_trades = [t for t in self.trades if "exit_time" in t and t["exit_time"] is not None]
            
            # Get risk metrics
            max_drawdown = self.risk_manager.get_max_drawdown() if self.risk_manager else {"max_drawdown_pct": 0.0, "max_drawdown_absolute": 0.0}
            equity_curve_data = self.risk_manager.get_equity_curve() if self.risk_manager else []
            
            # Add initial equity point if equity curve is empty or doesn't start with initial equity
            if not equity_curve_data or (equity_curve_data and equity_curve_data[0]["equity"] != self.initial_equity):
                # Add initial equity at first candle timestamp
                if self.intraday_candles:
                    first_timestamp = self.intraday_candles[0]["time"]
                    equity_curve_data.insert(0, {
                        "timestamp": first_timestamp,
                        "equity": self.initial_equity
                    })
            
            # Store equity curve in database
            if equity_curve_data:
                from database import EquityCurve
                from utils import unix_to_utc_datetime
                
                for point in equity_curve_data:
                    timestamp_dt = unix_to_utc_datetime(point["timestamp"])
                    equity_point = EquityCurve(
                        timestamp=timestamp_dt,
                        equity=point["equity"],
                        replay_id=self.replay_id
                    )
                    db_session.add(equity_point)
                db_session.commit()
            
            print(f"[INTRADAY REPLAY] Completed: {processed_count} intraday candles processed")
            print(f"[INTRADAY REPLAY]   Trades executed: {len(closed_trades)}")
            
            # Ensure final_equity is always defined before return
            final_equity = self.risk_manager.current_equity if self.risk_manager else self.initial_equity
            
            print(f"[INTRADAY REPLAY]   Final equity: {fmt_currency(final_equity)}")
            print(f"[INTRADAY REPLAY]   Max drawdown: {fmt_pct(max_drawdown.get('max_drawdown_pct'))}")
            
            return {
                "status": "completed",
                "replay_id": self.replay_id,
                "symbol": self.symbol,
                "total_intraday_candles": self.total_candles,
                "total_daily_candles": len(self.daily_candles),
                "candles_processed": processed_count,
                "trades_executed": len(closed_trades),
                "final_equity": final_equity,
                "max_drawdown_pct": max_drawdown["max_drawdown_pct"],
                "max_drawdown_absolute": max_drawdown["max_drawdown_absolute"]
            }
        
        except Exception as e:
            self.status = "error"
            self.error_message = str(e)
            raise
    
    def get_status(self) -> Dict:
        """Get current intraday replay status."""
        return {
            "status": self.status,
            "replay_id": self.replay_id,
            "symbol": self.symbol,
            "current_candle": self.current_candle_index,
            "total_candles": self.total_candles,
            "progress_pct": (self.current_candle_index / self.total_candles * 100) if self.total_candles > 0 else 0,
            "error_message": self.error_message
        }
