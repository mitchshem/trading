"""
Historical replay (backtest) engine for trading system.
Replays historical candles through the full pipeline: indicators → signals → broker → trades → metrics.
"""

from typing import List, Dict, Optional
from datetime import datetime, timezone
import uuid

from database import Signal, Trade, EquityCurve
from indicators import ema, atr
from strategy import ema_trend_v1, PositionState
from paper_broker import PaperBroker
from regime_classifier import classify_regime
from utils import ensure_utc_datetime, unix_to_utc_datetime


class ReplayEngine:
    """
    Replay engine that processes historical candles through the trading pipeline.
    Uses the same components as live trading for identical behavior.
    """
    
    def __init__(self, initial_equity: float = 100000.0, commission_per_share: float = 0.0, commission_per_trade: float = 0.0, slippage: float = 0.0002):
        self.initial_equity = initial_equity
        self.commission_per_share = commission_per_share
        self.commission_per_trade = commission_per_trade
        self.slippage = slippage
        self.replay_id: Optional[str] = None
        self.symbol: Optional[str] = None
        self.broker: Optional[PaperBroker] = None
        self.position_state: Optional[PositionState] = None
        self.candle_history: List[Dict] = []
        self.current_candle_index = 0
        self.total_candles = 0
        self.status = "idle"  # idle, running, completed, error
        self.error_message: Optional[str] = None
        self.source: Optional[str] = None  # Data source (e.g., "yahoo_finance")
        self.allowed_entry_regimes: Optional[List[str]] = None  # Optional regime gate for entries
    
    def reset(self):
        """Reset all state for a new replay."""
        self.broker = PaperBroker(
            initial_equity=self.initial_equity,
            commission_per_share=self.commission_per_share,
            commission_per_trade=self.commission_per_trade,
            slippage=self.slippage
        )
        self.position_state = PositionState()
        self.candle_history = []
        self.current_candle_index = 0
        self.total_candles = 0
        self.status = "idle"
        self.error_message = None
        self.source = None
        self.allowed_entry_regimes = None
    
    def start_replay(
        self,
        symbol: str,
        candles: List[Dict],
        replay_id: Optional[str] = None,
        source: Optional[str] = None,
        allowed_entry_regimes: Optional[List[str]] = None
    ) -> str:
        """
        Start a new replay.
        
        Args:
            symbol: Trading symbol
            candles: Ordered list of candle dicts with keys: time, open, high, low, close, volume
            replay_id: Optional replay ID (generated if not provided)
        
        Returns:
            replay_id: Unique identifier for this replay
        """
        if self.status == "running":
            raise ValueError("Replay already in progress")
        
        # Validate candles
        if not candles or len(candles) < 50:
            raise ValueError("Need at least 50 candles for replay (EMA(50) requires 50 candles)")
        
        # Validate candle ordering
        for i in range(1, len(candles)):
            if candles[i]["time"] <= candles[i-1]["time"]:
                raise ValueError(f"Candles must be ordered by time. Candle {i} has timestamp {candles[i]['time']} <= previous {candles[i-1]['time']}")
        
        # Generate replay_id if not provided
        if replay_id is None:
            replay_id = str(uuid.uuid4())
        
        # Reset state
        self.reset()
        
        # Initialize
        self.replay_id = replay_id
        self.symbol = symbol
        self.candle_history = candles.copy()
        self.total_candles = len(candles)
        self.status = "running"
        self.source = source or "unknown"  # Log data source
        self.allowed_entry_regimes = allowed_entry_regimes  # Optional regime gate
        
        # Log source and regime gate for tracking
        if source:
            regime_gate_info = f", regime_gate: {allowed_entry_regimes}" if allowed_entry_regimes else ""
            print(f"Replay {replay_id} started with source: {source}, symbol: {symbol}, candles: {len(candles)}{regime_gate_info}")
        
        return replay_id
    
    def process_candle(self, db_session, candle: Dict) -> Optional[Dict]:
        """
        Process a single candle through the trading pipeline.
        This mirrors the logic in evaluate_strategy_on_candle_close().
        
        Execution Model:
        1. Process pending orders from previous candle at current candle OPEN
        2. Evaluate strategy on current candle CLOSE
        3. Create pending orders for signals (BUY/EXIT)
        4. Check stop-losses (create pending EXIT orders)
        5. Check kill-switch (create pending EXIT orders)
        
        Key guarantee: Signals NEVER execute on the same candle.
        All orders execute at the NEXT candle's OPEN price.
        
        Args:
            db_session: Database session
            candle: Candle dict with keys: time, open, high, low, close, volume
        
        Returns:
            Dict with signal info if generated, None otherwise
            (Trades execute on next candle open, not returned here)
        """
        # Process pending orders from previous candle at current candle open
        # Use open_time for execution timestamp
        open_time = candle.get("open_time", candle["time"])  # Fallback for backward compatibility
        executed_trades = self.broker.process_pending_orders(
            current_open_price=candle["open"],
            timestamp=open_time
        )
        
        # Update trade records for executed orders
        for trade in executed_trades:
            if trade["action"] == "BUY":
                # FIX 1: CANONICAL TIME HANDLING
                entry_time = unix_to_utc_datetime(trade["timestamp"])
                ensure_utc_datetime(entry_time, f"replay BUY entry time for {trade['symbol']}")
                
                # FIX 2: EXPLICIT REPLAY ISOLATION
                if self.replay_id is None:
                    raise ValueError("Replay ID must be set for replay trades")
                trade_record = Trade(
                    symbol=trade["symbol"],
                    entry_time=entry_time,
                    entry_price=trade["entry_price"],
                    shares=trade["shares"],
                    exit_time=None,
                    exit_price=None,
                    pnl=None,
                    reason=None,
                    replay_id=self.replay_id
                )
                db_session.add(trade_record)
                db_session.commit()
                
                # Update position state if this was the current symbol
                if trade["symbol"] == self.symbol:
                    self.position_state.has_position = True
                    self.position_state.entry_price = trade["entry_price"]
                    self.position_state.entry_time = trade["timestamp"]
            
            elif trade["action"] == "EXIT":
                # Find open trade and update with exit
                open_trade = db_session.query(Trade).filter(
                    Trade.symbol == trade["symbol"],
                    Trade.exit_time.is_(None),
                    Trade.replay_id == self.replay_id
                ).first()
                
                if open_trade:
                    # FIX 1: CANONICAL TIME HANDLING
                    exit_time = unix_to_utc_datetime(trade["timestamp"])
                    ensure_utc_datetime(exit_time, f"replay EXIT exit time for {trade['symbol']}")
                    open_trade.exit_time = exit_time
                    open_trade.exit_price = trade["exit_price"]
                    open_trade.pnl = trade["pnl"]
                    open_trade.reason = trade["reason"]
                    db_session.commit()
                
                # Update position state if this was the current symbol
                if trade["symbol"] == self.symbol:
                    self.position_state.has_position = False
                    self.position_state.entry_price = None
                    self.position_state.entry_time = None
        
        # Note: candle_history is already populated from start_replay()
        # We process candles sequentially, using the accumulated history up to current index
        # The current candle is at index current_candle_index - 1
        
        # Get the slice of history up to and including current candle
        # This ensures we only use past data (no lookahead)
        current_history = self.candle_history[:self.current_candle_index]
        
        # Need at least 50 candles for EMA(50)
        if len(current_history) < 50:
            return None
        
        # Keep only last 500 candles for memory efficiency (for indicator calculation)
        if len(current_history) > 500:
            current_history = current_history[-500:]
        
        # Extract price arrays from current history (no lookahead)
        closes = [c["close"] for c in current_history]
        highs = [c["high"] for c in current_history]
        lows = [c["low"] for c in current_history]
        
        # Calculate indicators
        ema20_values = ema(closes, 20)
        ema50_values = ema(closes, 50)
        atr14_values = atr(highs, lows, closes, 14)
        
        # Get current ATR for position sizing
        current_atr = atr14_values[-1] if atr14_values and atr14_values[-1] is not None else None
        
        # Evaluate strategy using current history
        result = ema_trend_v1(
            candles=current_history,
            ema20_values=ema20_values,
            ema50_values=ema50_values,
            position_state=self.position_state
        )
        
        # Store signal if not HOLD
        signal_stored = False
        if result["signal"] != "HOLD":
            # FIX 1: CANONICAL TIME HANDLING
            # Use close_time for signal timestamp (signals generated on candle close)
            close_time = candle.get("close_time", candle["time"])  # Fallback for backward compatibility
            signal_time = unix_to_utc_datetime(close_time)
            ensure_utc_datetime(signal_time, f"replay signal timestamp for {self.symbol}")
            
            # FIX 2: EXPLICIT REPLAY ISOLATION
            # Replay signals: replay_id must match this replay (not None, not other replay)
            existing_signal = db_session.query(Signal).filter(
                Signal.symbol == self.symbol,
                Signal.timestamp == signal_time,
                Signal.signal == result["signal"],
                Signal.replay_id == self.replay_id  # Only this replay's signals
            ).first()
            
            if not existing_signal:
                # FIX 2: EXPLICIT REPLAY ISOLATION
                # Replay signals: replay_id is UUID (not None)
                # This ensures replay data never mixes with live trading data
                if self.replay_id is None:
                    raise ValueError("Replay ID must be set for replay signals")
                signal = Signal(
                    timestamp=signal_time,
                    symbol=self.symbol,
                    signal=result["signal"],
                    price=candle["close"],
                    reason=result["reason"],
                    replay_id=self.replay_id  # UUID for replays, None for live trading
                )
                db_session.add(signal)
                db_session.commit()
                signal_stored = True
        
        # Check stop-losses on every candle close (before processing signals)
        # Stop-losses create pending orders that execute on next candle open
        current_prices_all = {}
        for sym in self.broker.positions.keys():
            if sym == self.symbol and len(current_history) > 0:
                current_prices_all[sym] = current_history[-1]["close"]
        # Add current symbol's price (use current candle)
        current_prices_all[self.symbol] = candle["close"]
        
        # Use close_time for stop-loss checks (checked on candle close)
        close_time = candle.get("close_time", candle["time"])  # Fallback for backward compatibility
        stop_loss_orders = self.broker.check_stop_losses(current_prices_all, close_time)
        
        # Track which symbols have pending stop-loss exits to prevent double processing
        stop_loss_symbols = set()
        for exit_order in stop_loss_orders:
            stop_loss_symbols.add(exit_order["symbol"])
            # Note: Orders will execute on next candle open, so we don't update trade records here
        
        # Route signal through paper broker (creates pending orders)
        kill_switch_triggered = False
        
        # REGIME GATE: Check if BUY signal is allowed based on current regime
        buy_allowed = True
        if result["signal"] == "BUY" and self.allowed_entry_regimes:
            # Classify current regime (need at least 200 candles for EMA(200))
            current_regime = classify_regime(self.candle_history, self.current_candle_index - 1)
            if current_regime is None:
                # Not enough candles to classify regime yet - allow trade (default behavior)
                buy_allowed = True
            elif current_regime not in self.allowed_entry_regimes:
                buy_allowed = False
                # Log regime gate blocking
                print(f"[REGIME GATE] BUY signal blocked: current regime {current_regime} not in allowed {self.allowed_entry_regimes}")
        
        if result["signal"] == "BUY" and current_atr is not None and self.symbol not in stop_loss_symbols and buy_allowed:
            # Calculate stop distance (2 * ATR)
            stop_distance = 2 * current_atr
            
            # Create pending BUY order (executes on next candle open)
            # Signal timestamp uses close_time (signal generated on close)
            close_time = candle.get("close_time", candle["time"])  # Fallback for backward compatibility
            pending_order = self.broker.execute_buy(
                symbol=self.symbol,
                signal_price=candle["close"],
                stop_distance=stop_distance,
                timestamp=close_time
            )
            
            # Note: Trade record will be created when order executes on next candle open
        
        elif result["signal"] == "EXIT" and self.symbol not in stop_loss_symbols:
            # Create pending EXIT order (executes on next candle open)
            # Signal timestamp uses close_time (signal generated on close)
            close_time = candle.get("close_time", candle["time"])  # Fallback for backward compatibility
            pending_order = self.broker.execute_exit(
                symbol=self.symbol,
                signal_price=candle["close"],
                timestamp=close_time,
                reason=result["reason"]
            )
            
            # Note: Trade record will be updated when order executes on next candle open
        
        # Ensure equity is updated before risk checks
        self.broker.update_equity(current_prices_all)
        
        # Check and enforce risk controls on every candle close
        # Kill switch creates pending EXIT orders that execute on next candle open
        # Use close_time for kill switch checks (checked on candle close)
        close_time = candle.get("close_time", candle["time"])  # Fallback for backward compatibility
        kill_switch_orders = self.broker.check_and_enforce_risk_controls(current_prices_all, close_time)
        
        # Note: Orders will execute on next candle open, so we don't update trade records here
        
        if kill_switch_orders:
            kill_switch_triggered = True
        
        # Update equity curve
        # FIX 1: CANONICAL TIME HANDLING
        # Use close_time for equity curve (equity calculated on candle close)
        close_time = candle.get("close_time", candle["time"])  # Fallback for backward compatibility
        equity_timestamp = unix_to_utc_datetime(close_time)
        ensure_utc_datetime(equity_timestamp, f"replay equity curve timestamp for {self.symbol}")
        
        # FIX 2: EXPLICIT REPLAY ISOLATION
        # Replay equity curve: replay_id is UUID (not None)
        # This ensures replay data never mixes with live trading data
        if self.replay_id is None:
            raise ValueError("Replay ID must be set for replay equity curve")
        equity_point = EquityCurve(
            timestamp=equity_timestamp,
            equity=self.broker.equity,
            replay_id=self.replay_id  # UUID for replays, None for live trading
        )
        db_session.add(equity_point)
        db_session.commit()
        
        # Return result with signal info (trades execute on next candle open)
        if signal_stored or kill_switch_triggered:
            return {
                "signal": result if signal_stored else None,
                "trade": None,  # Trades execute on next candle open
                "kill_switch": kill_switch_triggered
            }
        
        return None
    
    def run(self, db_session) -> Dict:
        """
        Run the complete replay, processing all candles sequentially.
        
        Args:
            db_session: Database session
        
        Returns:
            Dict with replay results: status, replay_id, total_candles, etc.
        """
        if self.status != "running":
            raise ValueError(f"Cannot run replay in status: {self.status}")
        
        try:
            # Process candles one by one, building history as we go
            for i, candle in enumerate(self.candle_history):
                self.current_candle_index = i + 1
                
                # Process candle (it will use the accumulated history up to this point)
                self.process_candle(db_session, candle)
            
            self.status = "completed"
            
            return {
                "status": "completed",
                "replay_id": self.replay_id,
                "symbol": self.symbol,
                "total_candles": self.total_candles,
                "final_equity": self.broker.equity
            }
        
        except Exception as e:
            self.status = "error"
            self.error_message = str(e)
            raise
    
    def get_status(self) -> Dict:
        """Get current replay status."""
        return {
            "status": self.status,
            "replay_id": self.replay_id,
            "symbol": self.symbol,
            "current_candle": self.current_candle_index,
            "total_candles": self.total_candles,
            "progress_pct": (self.current_candle_index / self.total_candles * 100) if self.total_candles > 0 else 0,
            "error_message": self.error_message
        }
